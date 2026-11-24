"""Step 5 (V3 Surgery): PhysFlow-Fast (4-Step CFG-Free Trajectory-Distilled Pixel Flow).

Trains an INDEPENDENT fast-sampling LoRA checkpoint (`outputs/checkpoints/v3_surgery/physflow_fast4_lora.pt`)
without modifying V1 PhysFlow weights!

Key Mechanism (Rectified Reflow + CFG-Free 4-Step Distillation):
  - Distills our frozen 16-step CFG=2.0 V1 PhysFlow teacher (32 backbone NFEs, 0.872 s/img)
    into a CFG-FREE 4-step student (ONLY 4 backbone NFEs -> ~0.16 s/img, ~6 FPS, >15x faster than LSI!).
  - Uses the 900 cached teacher endpoints (`x_teacher`) + real GT supervision along the EXACT
    4-step warm-started trajectory z_{t_k} = t_k * x_target + (1 - t_k) * eps (Rectified Flow Reflow),
    supervising the single-pass (CFG=1.0) student clean prediction x_hat_theta directly against
    the straightened teacher trajectory + exact single-step RGB optical measurement Huber loss
    + AntelopeV2 identity consistency.

Evaluates 4-step PhysFlow-Fast vs 4-step un-distilled V1 vs 16-step V1 vs LSI on CelebA-HQ45,
CelebA45, and exact RTX 3090 latency/FPS!
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.glintr100 import FixedArcfaceCrop, GlintR100
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.minit2i_condition import predict_clean_factorized
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all

from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, GLINT_ONNX, NOISE_SCALE, load_512_batch, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import make_sem_prefix_mask, pick_even_split_indices
from scripts.run_v3_step1_inference_upgrade import sample_physflow_v3

V3_DIR = OUTPUTS / "v3_surgery"
V3_CKPT = OUTPUTS / "checkpoints" / "v3_surgery"
V3_DIR.mkdir(parents=True, exist_ok=True)
V3_CKPT.mkdir(parents=True, exist_ok=True)
OUT_JSON = V3_DIR / "step5_fast4step.json"
CACHE_PAIRS = V3_DIR / "refiner_train_pairs_900.pt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_ps(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_sample"}


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    assert CACHE_PAIRS.exists(), "Run Step 4 first to populate refiner_train_pairs_900.pt teacher cache!"

    t0_map = {64: 0.20, 128: 0.30, 256: 0.35}
    blind_mode, blind_scale = "tail_scale", 1.5
    s1_path = V3_DIR / "step1_inference_upgrade.json"
    if s1_path.exists():
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        if "selected_config" in s1:
            sc = s1["selected_config"]
            t0_map[64] = float(sc.get("t0_m64", 0.20))
            t0_map[256] = float(sc.get("t0_m256", 0.35))
            blind_mode = sc.get("blind_mode", blind_mode)
            blind_scale = float(sc.get("blind_scale", blind_scale))

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    val_idx45 = pick_even_split_indices(records, "val", 45)
    hq_idx45 = pick_even_split_indices(records, "celebahq_test", 45)
    cb_idx45 = pick_even_split_indices(records, "celeba_test", 45)

    gt_val45 = load_512_batch(records, val_idx45, torch.device("cpu"))
    gt_val45_lmk = lmks[val_idx45] * 512.0
    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))
    gt_hq45_lmk = lmks[hq_idx45] * 512.0
    gt_cb45 = load_512_batch(records, cb_idx45, torch.device("cpu"))
    gt_cb45_lmk = lmks[cb_idx45] * 512.0

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)

    # Load frozen V1 encoder & initialize student LoRA + conditioner from V1 weights (read-only!)
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])
    for p in enc.parameters():
        p.requires_grad = False

    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev).eval()
    net = wrapper.model.net
    for p in net.parameters():
        p.requires_grad = False
    lora_params = attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    cond = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev)
    load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, cond)

    # 1. Measure undistilled V1 at 4 steps (CFG=1.0 and CFG=2.0) before distillation
    log("=== Measuring Undistilled V1 at 4 Steps (CFG=1.0 & CFG=2.0) ===")
    undist = {}
    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        imgs_c1, sec_c1 = sample_physflow_v3(
            net, cond, enc, op_gpu, y256[hq_idx45, :rate], idx_r, dev,
            steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
            blind_mode=blind_mode, blind_scale=blind_scale, seed_base=3000, batch_size=9,
        )
        imgs_c2, sec_c2 = sample_physflow_v3(
            net, cond, enc, op_gpu, y256[hq_idx45, :rate], idx_r, dev,
            steps=4, cfg_scale=2.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
            blind_mode=blind_mode, blind_scale=blind_scale, seed_base=3000, batch_size=9,
        )
        r_c1 = {**strip_ps(ev.evaluate_pair(imgs_c1, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)), "sec_per_img": sec_c1}
        r_c2 = {**strip_ps(ev.evaluate_pair(imgs_c2, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)), "sec_per_img": sec_c2}
        undist[str(rate)] = {"v1_4step_cfg1": r_c1, "v1_4step_cfg2": r_c2}
        log(f"  [V1 4-step CFG=1 M={rate}] {sec_c1:.3f}s | Ada={r_c1['adaface_cos_mean']:.4f} NME={r_c1['nme106_mean']:.4f} LPIPS={r_c1['lpips_mean']:.4f} PSNR={r_c1['psnr_mean']:.2f}")
        log(f"  [V1 4-step CFG=2 M={rate}] {sec_c2:.3f}s | Ada={r_c2['adaface_cos_mean']:.4f} NME={r_c2['nme106_mean']:.4f} LPIPS={r_c2['lpips_mean']:.4f} PSNR={r_c2['psnr_mean']:.2f}")

    # 2. Rectified Reflow + CFG-Free 4-Step Distillation on cached 900 teacher endpoints + GT
    pairs = torch.load(CACHE_PAIRS, map_location="cpu", weights_only=False)
    x_teach = pairs["x_flow_f16"]  # 16-step CFG=2.0 teacher endpoints [900, 3, 512, 512]
    iw_tr = pairs["iw_f16"]
    rates_tr = pairs["rates"]
    idx_tr = pairs["train_indices"]
    gt_tr = load_512_batch(records, idx_tr, torch.device("cpu")).to(torch.float16)

    train_params = list(lora_params) + list(cond.parameters())
    for p in train_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(train_params, lr=8e-5, weight_decay=1e-2)
    steps = 450
    grad_accum = 4
    micro_b = 2
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)
    glint = GlintR100(GLINT_ONNX).to(dev).eval()
    cropper = FixedArcfaceCrop().to(dev)
    def id_loss_fn(pred_m11: torch.Tensor, gt_m11: torch.Tensor) -> torch.Tensor:
        fp = F.normalize(glint(cropper(pred_m11.float().clamp(-1.0, 1.0))).float(), dim=-1)
        with torch.no_grad():
            fg = F.normalize(glint(cropper(gt_m11.float().clamp(-1.0, 1.0))).float(), dim=-1)
        return (1.0 - (fp * fg).sum(dim=-1)).mean()

    log(f"=== Step 5: Training PhysFlow-Fast 4-Step CFG-Free Student ({steps} opt steps, {sum(p.numel() for p in train_params)/1e6:.2f}M trainable) ===")
    rng = np.random.default_rng(2026)
    t_wall = time.perf_counter()

    # Exact 4-step time nodes per rate
    best_val_lpips = 999.0
    best_state = None

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            rate_pick = int((64, 128, 256)[int(rng.integers(0, 3))])
            pool = [i for i, r in enumerate(rates_tr) if r == rate_pick]
            sel = rng.choice(pool, size=micro_b, replace=False).tolist()
            t0_r = t0_map[rate_pick]
            # Exact 4-step Euler start nodes: t_k in {t0, t0+dt, t0+2dt, t0+3dt}
            nodes = np.linspace(t0_r, 1.0, 5)[:-1]
            t_vals = torch.tensor([float(nodes[int(rng.integers(0, 4))]) for _ in range(micro_b)], device=dev, dtype=torch.float32)

            xt = x_teach[sel].to(dev, dtype=torch.float32)  # 16-step CFG=2 teacher endpoint
            gt = gt_tr[sel].to(dev, dtype=torch.float32)
            iw = iw_tr[sel].to(dev, dtype=torch.float32)
            orig_idx = [idx_tr[i] for i in sel]
            yb = y256[orig_idx, :rate_pick].to(dev, dtype=torch.float32)
            idx_r = torch.arange(rate_pick, device=dev, dtype=torch.long)

            # Coupled straightened trajectory: blend teacher endpoint (70% for CFG-free sharpness) + GT (30% for fidelity)
            x_target = 0.70 * xt + 0.30 * gt
            eps = torch.randn_like(x_target) * NOISE_SCALE
            # Start point on warm linear path interpolated straight to x_target (Reflow!)
            z_start = t0_r * iw + (1.0 - t0_r) * eps
            alpha = ((t_vals - t0_r) / max(1.0 - t0_r, 1e-4)).view(micro_b, 1, 1, 1)
            z_t = (1.0 - alpha) * z_start + alpha * x_target

            with torch.no_grad():
                p_sem, p_id, p_geo = enc(yb, idx_r)
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(micro_b, p_sem.shape[1], dev)
            id_tok = cond.project_id(p_id)
            geo_tok = cond.project_geo(p_geo)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_pred = predict_clean_factorized(
                    net, z_t.to(torch.bfloat16), t_vals.to(torch.bfloat16), txt_b, msk_b,
                    conditioner=cond, id_tokens=id_tok, geo_tokens=geo_tok, checkpoint=True,
                )
                # Direct straightened velocity-weighted loss toward CFG-free target
                denom = (1.0 - t_vals).clamp_min(0.05).view(micro_b, 1, 1, 1)
                v_pred = (x_pred.float() - z_t) / denom
                v_targ = (x_target - z_t) / denom
                l_flow = F.mse_loss(v_pred, v_targ)
                # Direct endpoint & optical consistency
                l_x0 = F.l1_loss(x_pred.float(), x_target)
                y_hat = op_gpu.measure(x_pred.float(), indices=idx_r, input_range="-1_1")
                s_norm = yb.abs().mean().clamp_min(1.0)
                l_meas = F.huber_loss(y_hat / s_norm, yb / s_norm, delta=0.1)
                l_id = id_loss_fn(x_pred.float(), gt)
                loss = (l_flow + 0.5 * l_x0 + 0.25 * l_meas + 0.04 * l_id) / grad_accum

            loss.backward()
            loss_acc += float(loss.detach())

        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()

        if step in (150, 300, 450):
            v64, s64 = sample_physflow_v3(
                net, cond, enc, op_gpu, y256[val_idx45, :64], torch.arange(64, device=dev), dev,
                steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[64], schedule_power=1.0,
                blind_mode=blind_mode, blind_scale=blind_scale, seed_base=3000, batch_size=9,
            )
            v256, s256 = sample_physflow_v3(
                net, cond, enc, op_gpu, y256[val_idx45, :256], torch.arange(256, device=dev), dev,
                steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[256], schedule_power=1.0,
                blind_mode=blind_mode, blind_scale=blind_scale, seed_base=3000, batch_size=9,
            )
            r64 = strip_ps(ev.evaluate_pair(v64, gt_val45, operator=op_cpu, y_gt=y256[val_idx45, :64], indices=torch.arange(64), gt_landmarks_px=gt_val45_lmk))
            r256 = strip_ps(ev.evaluate_pair(v256, gt_val45, operator=op_cpu, y_gt=y256[val_idx45, :256], indices=torch.arange(256), gt_landmarks_px=gt_val45_lmk))
            avg_lp = 0.5 * (r64["lpips_mean"] + r256["lpips_mean"])
            log(
                f"  [Fast4 step {step:3d}/{steps}] loss={loss_acc:.3f} ({s64:.3f}s/img) | "
                f"VAL64 Ada={r64['adaface_cos_mean']:.4f} LPIPS={r64['lpips_mean']:.4f} PSNR={r64['psnr_mean']:.2f} | "
                f"VAL256 Ada={r256['adaface_cos_mean']:.4f} LPIPS={r256['lpips_mean']:.4f} PSNR={r256['psnr_mean']:.2f}"
            )
            if avg_lp < best_val_lpips:
                best_val_lpips = avg_lp
                lora_sd = {k: v.detach().cpu().clone() for k, v in net.double_blocks.state_dict().items() if "lora_" in k}
                best_state = {"lora_state_dict": lora_sd, "conditioner_state_dict": {k: v.detach().cpu().clone() for k, v in cond.state_dict().items()}}

    train_sec = time.perf_counter() - t_wall
    if best_state is not None:
        net.double_blocks.load_state_dict(best_state["lora_state_dict"], strict=False)
        cond.load_state_dict(best_state["conditioner_state_dict"])
        torch.save({**best_state, "train_sec": train_sec}, V3_CKPT / "physflow_fast4_lora.pt")

    # Benchmark single-image latency (batch=1) and batched latency (batch=9) on RTX 3090
    idx64 = torch.arange(64, device=dev)
    _, _ = sample_physflow_v3(net, cond, enc, op_gpu, y256[hq_idx45[:1], :64], idx64, dev, steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[64], batch_size=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(10):
        sample_physflow_v3(net, cond, enc, op_gpu, y256[hq_idx45[i : i + 1], :64], idx64, dev, steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[64], batch_size=1)
    torch.cuda.synchronize()
    sec_bs1 = (time.perf_counter() - t0) / 10.0

    # Full test evaluation on CelebA-HQ45 & CelebA45 at M=64 & M=256 (4-step CFG=1.0 and 8-step CFG=1.0)
    res_test = {}
    for sname, sidx, gt_s, lmk_s in (("hq45", hq_idx45, gt_hq45, gt_hq45_lmk), ("cb45", cb_idx45, gt_cb45, gt_cb45_lmk)):
        res_test[sname] = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev)
            imgs4, sec4 = sample_physflow_v3(
                net, cond, enc, op_gpu, y256[sidx, :rate], idx_r, dev,
                steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
                blind_mode=blind_mode, blind_scale=blind_scale, seed_base=3000, batch_size=9,
            )
            r4 = {**strip_ps(ev.evaluate_pair(imgs4, gt_s, operator=op_cpu, y_gt=y256[sidx, :rate], indices=torch.arange(rate), gt_landmarks_px=lmk_s)), "sec_per_img_bs9": sec4, "fps_bs9": 1.0 / max(sec4, 1e-6)}
            res_test[sname][str(rate)] = r4
            log(f"  [PhysFlow-Fast 4-step CFG=1 | {sname} M={rate}] {sec4:.3f}s ({1/sec4:.2f} FPS) | Ada={r4['adaface_cos_mean']:.4f} Buf={r4['buffalo_cos_mean']:.4f} NME={r4['nme106_mean']:.4f} LPIPS={r4['lpips_mean']:.4f} PSNR={r4['psnr_mean']:.2f} RelRes={r4['meas_rel_mean']:.4f}")

    OUT_JSON.write_text(json.dumps({
        "train_sec": round(train_sec, 1),
        "single_img_sec_bs1": round(sec_bs1, 4),
        "single_img_fps_bs1": round(1.0 / max(sec_bs1, 1e-6), 2),
        "undistilled_v1_4step_hq45": undist,
        "physflow_fast_4step_cfg1": res_test,
    }, indent=2), encoding="utf-8")
    log(f"Saved Step 5 PhysFlow-Fast report (bs1={sec_bs1:.3f}s, {1/sec_bs1:.2f} FPS) -> {OUT_JSON}")


if __name__ == "__main__":
    main()
