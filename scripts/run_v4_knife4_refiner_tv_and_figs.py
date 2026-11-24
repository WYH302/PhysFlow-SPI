"""Knife 4 (V4 Top-Tier): Artifact-Free TV-Regularized 2.27M Pixel Refiner & V4 Fig. 1 / Fig. 6 Tiles.

Streamlined low-VRAM (< 1.6 GB) high-SM-efficiency version:
  1. Uses existing 900 cached flow pairs (840 train / 60 held-out val, ZERO extra training images!)
     to train LateStagePixelHFRefiner (base=48, 2.27M params) with explicit residual Total Variation
     + 2nd-order curvature regularization (0.35 * residual_tv_loss(delta)) using bs=2 x grad_accum=2
     (effective bs=4 at HALF VRAM -> zero WDDM paging).
  2. Generates hq45 endpoints (M=64 & M=256) at bs=3 (16-step) & bs=5 (Fast-4) and caches them.
  3. Evaluates Base Flow vs V3 Refiner vs V4 TV-Refiner vs Fast4+TV-Refiner on hq45 (CPU evaluator
     for zero GPU contention), saves side-by-side Visual QA strips and exact Fig. 1 & Fig. 6 tiles.

100% read-only wrt V1 and V3 checkpoints. Saves to:
  - outputs/checkpoints/v4_toptier/pixel_hf_refiner_tv.pt
  - outputs/v4_toptier/knife4_refiner_tv.json
  - outputs/v4_toptier/visual_qa_m64_tv.png & visual_qa_m256_tv.png
  - outputs/v4_toptier/fig1_fig6_v4_tiles.pt
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

import lpips
import numpy as np
import torch
import torch.nn.functional as F

from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.glintr100 import FixedArcfaceCrop, GlintR100
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all
from src.vis import save_image_grid

from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, GLINT_ONNX, load_512_batch, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import pick_even_split_indices
from scripts.run_v3_step1_inference_upgrade import sample_physflow_v3
from scripts.run_v3_step4_pixel_refiner import CACHE_PAIRS, LateStagePixelHFRefiner, sobel_grad
from scripts.run_v3_step6_final_full479 import apply_refiner_physics

V3_CKPT = OUTPUTS / "checkpoints" / "v3_surgery"
V4_DIR = OUTPUTS / "v4_toptier"
V4_CKPT = OUTPUTS / "checkpoints" / "v4_toptier"
V4_DIR.mkdir(parents=True, exist_ok=True)
V4_CKPT.mkdir(parents=True, exist_ok=True)
OUT_JSON = V4_DIR / "knife4_refiner_tv.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_ps(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_sample"}


def residual_tv_loss(delta: torch.Tensor) -> torch.Tensor:
    """Isotropic TV + 2nd-order curvature penalty on predicted residual delta to kill vertical striations."""
    dh = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    dw = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    d2w = (delta[:, :, :, 2:] - 2.0 * delta[:, :, :, 1:-1] + delta[:, :, :, :-2]).abs().mean()
    d2h = (delta[:, :, 2:, :] - 2.0 * delta[:, :, 1:-1, :] + delta[:, :, :-2, :]).abs().mean()
    return dh + dw + 0.5 * (d2w + d2h)


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    t0_map = {64: 0.25, 256: 0.35}
    blind_scale = 8.0

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    hq_idx45 = pick_even_split_indices(records, "celebahq_test", 45)
    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))
    gt_hq45_lmk = lmks[hq_idx45] * 512.0

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()

    # ------------------------------------------------------------------
    # Step A: Train TV-Regularized 2.27M Pixel Refiner on cached 900 pairs (840 train / 60 val)
    # ------------------------------------------------------------------
    assert CACHE_PAIRS.exists(), f"Missing {CACHE_PAIRS}"
    pairs = torch.load(CACHE_PAIRS, map_location="cpu", weights_only=False)
    x_flow_all = pairs["x_flow_f16"]
    iw_all = pairs["iw_f16"]
    rates_all = pairs["rates"]
    idx_all = pairs["train_indices"]
    gt_all_pairs = load_512_batch(records, idx_all, torch.device("cpu")).to(torch.float16)

    # Stratified 840 train / 60 val split (20 val per rate in {64, 128, 256})
    tr_pools, val_pools = {}, {}
    for r in (64, 128, 256):
        ids_r = [i for i, rr in enumerate(rates_all) if rr == r]
        val_pools[r] = ids_r[:20]
        tr_pools[r] = ids_r[20:]

    refiner = LateStagePixelHFRefiner(base=48).to(dev)
    vgg_lpips = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    for p in vgg_lpips.parameters():
        p.requires_grad = False
    glint = GlintR100(GLINT_ONNX).to(dev).eval()
    cropper = FixedArcfaceCrop().to(dev)

    def id_loss_fn(pred_m11: torch.Tensor, gt_m11: torch.Tensor) -> torch.Tensor:
        fp = F.normalize(glint(cropper(pred_m11.float().clamp(-1.0, 1.0))).float(), dim=-1)
        with torch.no_grad():
            fg = F.normalize(glint(cropper(gt_m11.float().clamp(-1.0, 1.0))).float(), dim=-1)
        return (1.0 - (fp * fg).sum(dim=-1)).mean()

    opt = torch.optim.AdamW(refiner.parameters(), lr=2.5e-4, weight_decay=1e-2)
    steps = 650
    batch_size = 2
    grad_accum = 2
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    @torch.no_grad()
    def eval_val60() -> dict:
        refiner.eval()
        res = {}
        for r in (64, 256):
            vp = val_pools[r]
            lps, ids, psnrs, tvs = [], [], [], []
            for s in range(0, len(vp), 4):
                sel = vp[s : s + 4]
                xb = x_flow_all[sel].to(dev, dtype=torch.float32)
                iwb = iw_all[sel].to(dev, dtype=torch.float32)
                gtb = gt_all_pairs[sel].to(dev, dtype=torch.float32)
                rn = torch.full((len(sel),), math.log2(r) / 8.0, device=dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pr = refiner(xb, iwb, rn).float()
                lps.append(vgg_lpips(pr, gtb).mean().item())
                ids.append(1.0 - id_loss_fn(pr, gtb).item())
                mse = ((pr - gtb) * 0.5).square().mean(dim=(1, 2, 3)).clamp_min(1e-8)
                psnrs.append((-10.0 * torch.log10(mse)).mean().item())
                tvs.append(residual_tv_loss(pr - xb).item())
            res[r] = {"vgg_lpips": float(np.mean(lps)), "glint_cos": float(np.mean(ids)), "psnr": float(np.mean(psnrs)), "tv": float(np.mean(tvs))}
        return res

    base_val = {}
    with torch.no_grad():
        for r in (64, 256):
            vp = val_pools[r]
            lps, ids, psnrs = [], [], []
            for s in range(0, len(vp), 4):
                sel = vp[s : s + 4]
                xb = x_flow_all[sel].to(dev, dtype=torch.float32)
                gtb = gt_all_pairs[sel].to(dev, dtype=torch.float32)
                lps.append(vgg_lpips(xb, gtb).mean().item())
                ids.append(1.0 - id_loss_fn(xb, gtb).item())
                mse = ((xb - gtb) * 0.5).square().mean(dim=(1, 2, 3)).clamp_min(1e-8)
                psnrs.append((-10.0 * torch.log10(mse)).mean().item())
            base_val[r] = {"vgg_lpips": float(np.mean(lps)), "glint_cos": float(np.mean(ids)), "psnr": float(np.mean(psnrs))}

    log(f"=== Knife 4A: Training TV-Regularized Refiner ({steps} steps, bs=2x2 < 1.5 GB VRAM) ===")
    log(f"  Val60 Base: M=64 VGG-LP={base_val[64]['vgg_lpips']:.4f} PSNR={base_val[64]['psnr']:.2f} | M=256 VGG-LP={base_val[256]['vgg_lpips']:.4f} PSNR={base_val[256]['psnr']:.2f}")

    best_score = -1e9
    best_sd = None
    rng = np.random.default_rng(108)
    t_start = time.perf_counter()

    for step in range(1, steps + 1):
        refiner.train()
        opt.zero_grad(set_to_none=True)
        for _acc in range(grad_accum):
            rate_pick = int((64, 128, 256)[int(rng.integers(0, 3))])
            sel = rng.choice(tr_pools[rate_pick], size=batch_size, replace=False).tolist()
            xb = x_flow_all[sel].to(dev, dtype=torch.float32)
            iwb = iw_all[sel].to(dev, dtype=torch.float32)
            gtb = gt_all_pairs[sel].to(dev, dtype=torch.float32)
            orig_idx = [idx_all[i] for i in sel]
            yb = y256[orig_idx, :rate_pick].to(dev, dtype=torch.float32)
            idx_r = torch.arange(rate_pick, device=dev, dtype=torch.long)

            flipped = bool(rng.random() < 0.5)
            if flipped:
                xb = torch.flip(xb, dims=[-1])
                iwb = torch.flip(iwb, dims=[-1])
                gtb = torch.flip(gtb, dims=[-1])

            rn = torch.full((batch_size,), math.log2(rate_pick) / 8.0, device=dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = refiner(xb, iwb, rn)
                delta = pred - xb
                l_l1 = F.l1_loss(pred, gtb)
                l_lp = vgg_lpips(pred, gtb).mean()
                l_grad = F.l1_loss(sobel_grad(pred), sobel_grad(gtb))
                l_tv = residual_tv_loss(delta.float())
                l_id = id_loss_fn(pred.float(), gtb.float())
                if not flipped:
                    y_hat = op_gpu.measure(pred.float(), indices=idx_r, input_range="-1_1")
                    s_norm = yb.abs().mean().clamp_min(1.0)
                    l_meas = F.huber_loss(y_hat / s_norm, yb / s_norm, delta=0.1)
                else:
                    l_meas = pred.new_zeros(())
                loss = (0.90 * l_l1 + 0.36 * l_lp + 0.14 * l_grad + 0.35 * l_tv + 0.12 * l_id + 0.25 * l_meas) / grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt.step()
        sched.step()

        if step in (200, 400, 550, 650):
            vr = eval_val60()
            sc = (
                (base_val[256]["vgg_lpips"] - vr[256]["vgg_lpips"]) * 3.0
                + (base_val[64]["vgg_lpips"] - vr[64]["vgg_lpips"]) * 2.0
                + (vr[256]["glint_cos"] - base_val[256]["glint_cos"]) * 2.5
                + (vr[64]["glint_cos"] - base_val[64]["glint_cos"]) * 2.5
                + (vr[256]["psnr"] - base_val[256]["psnr"]) * 0.15
            )
            log(
                f"  [Step {step:3d}/{steps}] M64 LP={vr[64]['vgg_lpips']:.4f} ID={vr[64]['glint_cos']:.4f} PSNR={vr[64]['psnr']:.2f} tv={vr[64]['tv']:.4f} | "
                f"M256 LP={vr[256]['vgg_lpips']:.4f} ID={vr[256]['glint_cos']:.4f} PSNR={vr[256]['psnr']:.2f} tv={vr[256]['tv']:.4f} | sc={sc:+.4f}"
            )
            if sc > best_score:
                best_score = sc
                best_sd = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}

    train_sec = time.perf_counter() - t_start
    if best_sd is not None:
        refiner.load_state_dict(best_sd)
    torch.save({"state_dict": refiner.state_dict(), "train_sec": train_sec, "best_val_score": best_score}, V4_CKPT / "pixel_hf_refiner_tv.pt")
    log(f"  Saved V4 TV-Refiner -> pixel_hf_refiner_tv.pt ({train_sec:.1f}s)")

    # Free training-only losses and pair caches cleanly!
    vgg_lpips.to("cpu"); glint.to("cpu")
    del vgg_lpips, glint, cropper, opt, sched, pairs, x_flow_all, iw_all, gt_all_pairs
    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step B: Generate ONLY hq45 flow endpoints (bs=3 for 16-step, bs=5 for Fast-4) & cache!
    # ------------------------------------------------------------------
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])

    hq_cache_path = V4_DIR / "knife4_hq45_flow_cache.pt"
    if hq_cache_path.exists():
        log(f"=== Knife 4B: Loading cached hq45 flow endpoints from {hq_cache_path.name} ===")
        fc = torch.load(hq_cache_path, map_location="cpu", weights_only=False)
        hq_flow16, hq_fast4 = fc["hq_flow16"], fc["hq_fast4"]
    else:
        log("=== Knife 4B: Generating hq45 endpoints (M=64 & M=256, 16-step bs=3 & Fast-4 bs=5) ===")
        wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
        wrapper = wrapper.to(dev).eval()
        net = wrapper.model.net
        attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
        net.double_blocks.to(dev)
        cond = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev).eval()
        load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, cond)

        hq_flow16 = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            xf, _ = sample_physflow_v3(
                net, cond, enc, op_gpu, y256[hq_idx45, :rate], idx_r, dev,
                steps=16, cfg_scale=2.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
                blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=3,
            )
            hq_flow16[rate] = xf
            log(f"  Generated hq45 M={rate} (16-step)")

        fast_ckpt = torch.load(V3_CKPT / "physflow_fast4_lora.pt", map_location="cpu", weights_only=False)
        net.double_blocks.load_state_dict(fast_ckpt["lora_state_dict"], strict=False)
        cond.load_state_dict(fast_ckpt["conditioner_state_dict"])
        hq_fast4 = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            xf4, _ = sample_physflow_v3(
                net, cond, enc, op_gpu, y256[hq_idx45, :rate], idx_r, dev,
                steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
                blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=5,
            )
            hq_fast4[rate] = xf4
            log(f"  Generated hq45 M={rate} (Fast-4)")

        wrapper.to("cpu"); cond.to("cpu")
        del wrapper, cond, net
        gc.collect(); torch.cuda.empty_cache()
        torch.save({"hq_flow16": hq_flow16, "hq_fast4": hq_fast4}, hq_cache_path)

    # ------------------------------------------------------------------
    # Step C: Apply V3 Refiner vs V4 TV-Refiner + CPU CrossModelEvaluator + Visual QA strips!
    # ------------------------------------------------------------------
    log("=== Knife 4C: Side-by-side V3 Refiner vs V4 TV-Refiner on hq45 & Visual QA ===")
    ref_v3 = LateStagePixelHFRefiner(base=48).to(dev).eval()
    ref_v3.load_state_dict(torch.load(V3_CKPT / "pixel_hf_refiner.pt", map_location="cpu", weights_only=False)["state_dict"])
    refiner.eval()

    v4_hq_imgs, v3_hq_imgs, f4_hq_imgs = {}, {}, {}
    for rate in (64, 256):
        yb = y256[hq_idx45, :rate]
        v4_hq_imgs[rate] = apply_refiner_physics(refiner, enc, op_gpu, hq_flow16[rate], yb, rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=3)
        v3_hq_imgs[rate] = apply_refiner_physics(ref_v3, enc, op_gpu, hq_flow16[rate], yb, rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=3)
        f4_hq_imgs[rate] = apply_refiner_physics(refiner, enc, op_gpu, hq_fast4[rate], yb, rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=3)

    enc.to("cpu"); refiner.to("cpu"); ref_v3.to("cpu")
    del enc, refiner, ref_v3
    gc.collect(); torch.cuda.empty_cache()

    ev = CrossModelEvaluator(device="cpu", use_lpips=True, use_buffalo=True)
    out = {"train_sec": round(train_sec, 1), "best_val_score": round(best_score, 4), "hq45_v3_ref": {}, "hq45_v4_tvref": {}, "hq45_fast4_tvref": {}}
    for rate in (64, 256):
        yb = y256[hq_idx45, :rate]
        for key, imgs in (("hq45_v3_ref", v3_hq_imgs[rate]), ("hq45_v4_tvref", v4_hq_imgs[rate]), ("hq45_fast4_tvref", f4_hq_imgs[rate])):
            r = strip_ps(ev.evaluate_pair(imgs, gt_hq45, operator=op_cpu, y_gt=yb, indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk))
            out[key][str(rate)] = r
            log(f"  [{key:17s} M={rate:3d}] Ada={r['adaface_cos_mean']:.4f} Buf={r['buffalo_cos_mean']:.4f} NME={r['nme106_mean']:.4f} LPIPS={r['lpips_mean']:.4f} PSNR={r['psnr_mean']:.2f} RelRes={r['meas_rel_mean']:.4f}")

    # Save 4-face x 5-col Visual QA (GT | Flow | V3 Old Ref | V4 TV-Ref | Fast4+TV-Ref)
    for rate in (64, 256):
        g_imgs, g_lbls = [], []
        for i in range(4):
            for t, lbl in (
                (gt_hq45[i], "GT"),
                (hq_flow16[rate][i], f"Flow M={rate}"),
                (v3_hq_imgs[rate][i], f"V3 Ref M={rate}"),
                (v4_hq_imgs[rate][i], f"V4 TV-Ref M={rate}"),
                (f4_hq_imgs[rate][i], f"Fast4+TV M={rate}"),
            ):
                g_imgs.append(t)
                g_lbls.append(lbl)
        save_image_grid(torch.stack(g_imgs), V4_DIR / f"visual_qa_m{rate}_tv.png", nrow=5, titles=g_lbls)

    torch.save({
        "hq45_v4_m64": v4_hq_imgs[64],
        "hq45_v4_m256": v4_hq_imgs[256],
        "hq45_fast4_m64": f4_hq_imgs[64],
        "hq45_fast4_m256": f4_hq_imgs[256],
    }, V4_DIR / "fig1_fig6_v4_tiles.pt")

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"KNIFE 4 COMPLETE -> {OUT_JSON}")


if __name__ == "__main__":
    main()
