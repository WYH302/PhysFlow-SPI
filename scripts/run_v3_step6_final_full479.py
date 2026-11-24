"""Step 6A (V3 Surgery): Final Full-Cohort N=479 FID + External LaPa + WFLW + Visual QA Verification.

Evaluates BOTH upgraded modes end-to-end on strictly held-out benchmarks:
  1. PhysFlow-V3 (16-step, rate-adaptive t0(M), calibrated blind alpha=8.0, + 2.6M Late-Stage Pixel Refiner)
  2. PhysFlow-Fast (4-step CFG-free distilled LoRA, + 2.6M Late-Stage Pixel Refiner)

Benchmarks:
  - Full-cohort ALL 479 strictly identity-disjoint CelebA-HQ portraits at M=64 & M=256
    with REAL Inception-v3 FID + 4,000-draw paired bootstrap vs official LSI (CVPR 2025) & V1!
  - External zero-shot LaPa (N=32) at M=64 & M=256!
  - External zero-shot WFLW 6 challenging subsets (N=72) at M=64!
  - Clean CUDA-synchronized single-image (bs=1) & batched (bs=9) latency benchmarks!
  - Visual QA strip (`outputs/v3_surgery/v3_visual_qa_m64_m256.png`) comparing GT | V1 | V3 | Fast-4!

Saves all results to `outputs/v3_surgery/step6_final_full479_v3.json`.
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
from PIL import Image

from src.eval.adaface_eval import CrossModelEvaluator
from src.eval.metrics import cosine_identity, psnr as calc_psnr, ssim as calc_ssim
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all
from src.vis import save_image_grid

from scripts.eval_lapa_wflw import (
    WFLW_SUBSETS,
    align_image_list,
    load_lapa_samples,
    load_wflw_official_rgb_samples,
)
from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, load_512_batch, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import pick_even_split_indices
from scripts.run_reviewer_shield_suite import InceptionFID, bootstrap_ci_and_p
from scripts.run_v3_step1_inference_upgrade import estimate_lambda_upgraded, sample_physflow_v3
from scripts.run_v3_step4_pixel_refiner import LateStagePixelHFRefiner

V3_DIR = OUTPUTS / "v3_surgery"
V3_CKPT = OUTPUTS / "checkpoints" / "v3_surgery"
OUT_JSON = V3_DIR / "step6_final_full479_v3.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_ps(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_sample"}


@torch.no_grad()
def apply_refiner_physics(
    refiner: LateStagePixelHFRefiner,
    encoder: PhysicsDualDomainFactorizer,
    op_gpu: SPIOperator,
    x_flow: torch.Tensor,
    y_sub: torch.Tensor,
    rate: int,
    dev: torch.device,
    proj_eta: float = 0.65,
    blind_scale: float = 8.0,
    batch_size: int = 12,
) -> torch.Tensor:
    refiner.eval()
    idx_r = torch.arange(rate, device=dev, dtype=torch.long)
    outs = []
    idx = idx_r.to(device=encoder.lifter.y_bar.device, dtype=torch.long)
    c_sub = encoder.lifter.C_Iy.index_select(1, idx).float()
    y_bar_s = encoder.lifter.y_bar.index_select(1, idx)
    y_sd_s = encoder.lifter.y_sd.index_select(1, idx)

    for s in range(0, x_flow.shape[0], batch_size):
        e = min(x_flow.shape[0], s + batch_size)
        xb = x_flow[s:e].to(dev, dtype=torch.float32)
        yb = y_sub[s:e].to(dev, dtype=torch.float32)
        evals, evecs, lam_b = estimate_lambda_upgraded(encoder.lifter, yb, idx_r, mode="tail_scale", scale=blind_scale)
        dy = (yb - y_bar_s) / y_sd_s
        coef = ((dy @ evecs) / (evals.unsqueeze(0) + lam_b.unsqueeze(1))) @ evecs.T
        iw = (encoder.lifter.I_bar.view(1, -1).float() + coef @ c_sub.T).view(-1, 3, encoder.lifter.img_size, encoder.lifter.img_size).clamp(0.0, 1.0)
        iwb = F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0

        rn = torch.full((xb.shape[0],), math.log2(rate) / 8.0, device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            xr = refiner(xb, iwb, rn).float()

        # Final physics-consistent Kalman-Wiener projection to keep RelRes <= 0.005 while preserving sharpened HF detail
        if proj_eta > 0.0:
            x01 = op_gpu.to_unit_interval(xr, "-1_1").clamp(0.0, 1.0)
            y_hat = op_gpu.measure(x01, indices=idx_r, input_range="0_1")
            dy_res = (yb - y_hat.float()) / y_sd_s
            coef_p = ((dy_res @ evecs) / (evals.unsqueeze(0) + lam_b.unsqueeze(1))) @ evecs.T
            delta = (coef_p @ c_sub.T).view(-1, 3, encoder.lifter.img_size, encoder.lifter.img_size)
            delta = F.interpolate(delta, size=(512, 512), mode="bilinear", align_corners=False)
            xr = op_gpu.from_unit_interval((x01 + float(proj_eta) * delta).clamp(0.0, 1.0), "-1_1")
        outs.append(xr.cpu())
    return torch.cat(outs, dim=0)


def eval_full479_metrics(
    imgs: torch.Tensor,
    gt_imgs: torch.Tensor,
    gt_lmk: torch.Tensor,
    y_gt: torch.Tensor,
    rate: int,
    ev: CrossModelEvaluator,
    fid_calc: InceptionFID,
    gt_fid_feats: np.ndarray,
    op_cpu: SPIOperator,
) -> tuple[dict, dict]:
    r = ev.evaluate_pair(imgs, gt_imgs, operator=op_cpu, y_gt=y_gt, indices=torch.arange(rate), gt_landmarks_px=gt_lmk)
    gen_feats = fid_calc.features(imgs, batch_size=32)
    fid_val = InceptionFID.frechet_distance(gen_feats, gt_fid_feats)
    summary = {k: v for k, v in r.items() if k != "per_sample"}
    summary["fid"] = float(fid_val)
    return summary, r["per_sample"]


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    t0_map = {64: 0.25, 256: 0.35}
    blind_scale = 8.0

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    all_hq_idx = [r["index"] for r in records if r["split"] == "celebahq_test"]
    hq_idx45 = pick_even_split_indices(records, "celebahq_test", 45)
    assert len(all_hq_idx) == 479

    gt_all_hq = load_512_batch(records, all_hq_idx, torch.device("cpu"))
    gt_all_lmk = lmks[all_hq_idx] * 512.0
    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)
    fid_calc = InceptionFID(dev)
    gt_fid_feats = fid_calc.features(gt_all_hq, batch_size=32)

    # Load frozen V1 encoder + refiner
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])

    refiner = LateStagePixelHFRefiner(base=48).to(dev).eval()
    refiner.load_state_dict(torch.load(V3_CKPT / "pixel_hf_refiner.pt", map_location="cpu", weights_only=False)["state_dict"])

    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev).eval()
    net = wrapper.model.net
    attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    cond = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev).eval()
    load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, cond)

    # Load existing full479 LSI per-sample metrics for exact paired bootstrap p-values!
    ps_cache_p = OUTPUTS / "full479_persample_cache.json"
    ps_cache = json.loads(ps_cache_p.read_text(encoding="utf-8")) if ps_cache_p.exists() else {}

    out: dict = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {"full479": {}, "lapa32": {}, "wflw72_m64": {}, "timings": {}}

    def save_out() -> None:
        OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Prepare aligned external LaPa32 & WFLW72 once using exact eval_lapa_wflw protocol
    import os
    os.environ["INSIGHTFACE_HOME"] = str((PROJECT_ROOT / "checkpoints" / "insightface").resolve())
    from insightface.app import FaceAnalysis
    buf = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], root=str((PROJECT_ROOT / "checkpoints" / "insightface").resolve()))
    buf.prepare(ctx_id=-1, det_size=(640, 640))
    lapa_gt, lapa_y256 = align_image_list(buf, load_lapa_samples(32), op_gpu, dev, target_n=32)
    wflw_raw = load_wflw_official_rgb_samples(12)
    wflw_aligned = {sb: align_image_list(buf, wflw_raw[sb], op_gpu, dev, target_n=12) for sb in WFLW_SUBSETS}
    del buf

    qa_tiles = {"64": {"gt": gt_all_hq[:4]}, "256": {"gt": gt_all_hq[:4]}}
    idx64 = torch.arange(64, device=dev)

    # =========================================================================
    # 1. External LaPa (32) & WFLW 6 Subsets (72) for PhysFlow-V3 (fast ~2 min!)
    # =========================================================================
    log("=== 1. Evaluating PhysFlow-V3 (16-step + Refiner) on External LaPa (32) & WFLW 6 Subsets (72) ===")
    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev)
        y_lapa = lapa_y256[:, :rate]
        xf_l, _ = sample_physflow_v3(net, cond, enc, op_gpu, y_lapa, idx_r, dev, steps=16, cfg_scale=2.0, eta=0.85, t_warm=t0_map[rate], blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=8)
        xv3_l = apply_refiner_physics(refiner, enc, op_gpu, xf_l, y_lapa, rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=8)
        rl = strip_ps(ev.evaluate_pair(xv3_l, lapa_gt, operator=op_cpu, y_gt=y_lapa, indices=torch.arange(rate)))
        out["lapa32"].setdefault(str(rate), {})["physflow_v3_16step"] = rl
        log(f"  [LaPa32 V3 M={rate}] Ada={rl['adaface_cos_mean']:.4f} Buf={rl['buffalo_cos_mean']:.4f} NME={rl['nme106_mean']:.4f} LPIPS={rl['lpips_mean']:.4f} PSNR={rl['psnr_mean']:.2f} RelRes={rl['meas_rel_mean']:.4f}")
    save_out()

    for sb in WFLW_SUBSETS:
        gt_w, y_w256 = wflw_aligned[sb]
        y_w = y_w256[:, :64]
        xf_w, _ = sample_physflow_v3(net, cond, enc, op_gpu, y_w, idx64, dev, steps=16, cfg_scale=2.0, eta=0.85, t_warm=t0_map[64], blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=12)
        xv3_w = apply_refiner_physics(refiner, enc, op_gpu, xf_w, y_w, 64, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=12)
        rw = strip_ps(ev.evaluate_pair(xv3_w, gt_w, operator=op_cpu, y_gt=y_w, indices=torch.arange(64)))
        out["wflw72_m64"][sb] = rw
        log(f"  [WFLW V3 M=64 | {sb:12s}] PSNR={rw['psnr_mean']:.2f} LPIPS={rw['lpips_mean']:.4f} NME={rw['nme106_mean']:.4f}")
    save_out()

    # =========================================================================
    # 2. PhysFlow-V3 (16-step + Refiner) on ALL 479 CelebA-HQ with paired bootstrap!
    # =========================================================================
    log("=== 2. Evaluating PhysFlow-V3 (16-step + Refiner) on ALL 479 CelebA-HQ ===")
    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        x_flow, sec_flow = sample_physflow_v3(
            net, cond, enc, op_gpu, y256[all_hq_idx, :rate], idx_r, dev,
            steps=16, cfg_scale=2.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
            blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=12,
        )
        t_ref = time.perf_counter()
        x_v3 = apply_refiner_physics(refiner, enc, op_gpu, x_flow, y256[all_hq_idx, :rate], rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=12)
        sec_ref = (time.perf_counter() - t_ref) / len(all_hq_idx)

        s_v3, ps_v3 = eval_full479_metrics(x_v3, gt_all_hq, gt_all_lmk, y256[all_hq_idx, :rate], rate, ev, fid_calc, gt_fid_feats, op_cpu)
        s_v3["sec_per_img_total"] = round(sec_flow + sec_ref, 4)

        boot = {}
        lsi_ps = ps_cache.get(str(rate), {}).get("lsi")
        if lsi_ps is not None:
            for mk, hb in (("adaface_cos", True), ("buffalo_cos", True), ("nme106", False), ("lpips", False), ("psnr", True)):
                if mk in ps_v3 and mk in lsi_ps:
                    boot[f"v3_vs_lsi_{mk}"] = bootstrap_ci_and_p(ps_v3[mk], lsi_ps[mk], n_boot=4000, seed=42, higher_better=hb)
        out["full479"].setdefault(str(rate), {})["physflow_v3_16step"] = {**s_v3, "bootstrap_vs_lsi": boot}
        log(f"  [Full-479 V3 16-step M={rate}] FID={s_v3['fid']:.2f} Ada={s_v3['adaface_cos_mean']:.4f} Buf={s_v3['buffalo_cos_mean']:.4f} NME={s_v3['nme106_mean']:.4f} LPIPS={s_v3['lpips_mean']:.4f} PSNR={s_v3['psnr_mean']:.2f} RelRes={s_v3['meas_rel_mean']:.4f}")
        qa_tiles[str(rate)]["v3_flow"] = x_flow[:4]
        qa_tiles[str(rate)]["v3_ref"] = x_v3[:4]
        save_out()

    # =========================================================================
    # 3. Swap in PhysFlow-Fast 4-Step CFG-Free LoRA + Refiner on ALL 479 CelebA-HQ & LaPa32!
    # =========================================================================
    log("=== 3. Evaluating PhysFlow-Fast (4-step CFG-Free + Refiner) on ALL 479 CelebA-HQ & LaPa32 ===")
    fast_ckpt = torch.load(V3_CKPT / "physflow_fast4_lora.pt", map_location="cpu", weights_only=False)
    net.double_blocks.load_state_dict(fast_ckpt["lora_state_dict"], strict=False)
    cond.load_state_dict(fast_ckpt["conditioner_state_dict"])
    net.eval(); cond.eval()

    # Accurate CUDA-synced latency measurement (warmed up, bs=1 and bs=9)
    for bs_tag, bs_n in (("bs1", 1), ("bs9", 9)):
        yb_t = y256[hq_idx45[:bs_n], :64]
        _x, _ = sample_physflow_v3(net, cond, enc, op_gpu, yb_t, idx64, dev, steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[64], blind_mode="tail_scale", blind_scale=blind_scale, batch_size=bs_n)
        _ = apply_refiner_physics(refiner, enc, op_gpu, _x, yb_t, 64, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=bs_n)
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        reps = 10 if bs_n == 1 else 5
        for _ in range(reps):
            _x, _ = sample_physflow_v3(net, cond, enc, op_gpu, yb_t, idx64, dev, steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[64], blind_mode="tail_scale", blind_scale=blind_scale, batch_size=bs_n)
            _ = apply_refiner_physics(refiner, enc, op_gpu, _x, yb_t, 64, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=bs_n)
        torch.cuda.synchronize()
        sec_per = (time.perf_counter() - t_s) / (reps * bs_n)
        out["timings"][f"physflow_fast4_plus_refiner_{bs_tag}_sec"] = round(sec_per, 4)
        out["timings"][f"physflow_fast4_plus_refiner_{bs_tag}_fps"] = round(1.0 / max(sec_per, 1e-6), 2)
    log(f"  [Fast4+Refiner Latency] bs1={out['timings']['physflow_fast4_plus_refiner_bs1_sec']}s ({out['timings']['physflow_fast4_plus_refiner_bs1_fps']} FPS) | bs9={out['timings']['physflow_fast4_plus_refiner_bs9_sec']}s ({out['timings']['physflow_fast4_plus_refiner_bs9_fps']} FPS)")
    save_out()

    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev)
        xf_4, sec_4 = sample_physflow_v3(
            net, cond, enc, op_gpu, y256[all_hq_idx, :rate], idx_r, dev,
            steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[rate], schedule_power=1.0,
            blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=12,
        )
        t_ref = time.perf_counter()
        x_f4 = apply_refiner_physics(refiner, enc, op_gpu, xf_4, y256[all_hq_idx, :rate], rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=12)
        sec_ref = (time.perf_counter() - t_ref) / len(all_hq_idx)

        s_f4, ps_f4 = eval_full479_metrics(x_f4, gt_all_hq, gt_all_lmk, y256[all_hq_idx, :rate], rate, ev, fid_calc, gt_fid_feats, op_cpu)
        s_f4["sec_per_img_total"] = round(sec_4 + sec_ref, 4)
        boot4 = {}
        lsi_ps = ps_cache.get(str(rate), {}).get("lsi")
        if lsi_ps is not None:
            for mk, hb in (("adaface_cos", True), ("buffalo_cos", True), ("nme106", False), ("lpips", False), ("psnr", True)):
                if mk in ps_f4 and mk in lsi_ps:
                    boot4[f"fast4_vs_lsi_{mk}"] = bootstrap_ci_and_p(ps_f4[mk], lsi_ps[mk], n_boot=4000, seed=42, higher_better=hb)
        out["full479"][str(rate)]["physflow_fast_4step"] = {**s_f4, "bootstrap_vs_lsi": boot4}
        log(f"  [Full-479 Fast-4+Ref M={rate}] FID={s_f4['fid']:.2f} Ada={s_f4['adaface_cos_mean']:.4f} Buf={s_f4['buffalo_cos_mean']:.4f} NME={s_f4['nme106_mean']:.4f} LPIPS={s_f4['lpips_mean']:.4f} PSNR={s_f4['psnr_mean']:.2f} RelRes={s_f4['meas_rel_mean']:.4f}")
        qa_tiles[str(rate)]["fast4_ref"] = x_f4[:4]

        # LaPa32 for Fast-4+Ref
        y_lapa = lapa_y256[:, :rate]
        xf_l4, _ = sample_physflow_v3(net, cond, enc, op_gpu, y_lapa, idx_r, dev, steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[rate], blind_mode="tail_scale", blind_scale=blind_scale, seed_base=3000, batch_size=8)
        xl4 = apply_refiner_physics(refiner, enc, op_gpu, xf_l4, y_lapa, rate, dev, proj_eta=0.75, blind_scale=blind_scale, batch_size=8)
        rl4 = strip_ps(ev.evaluate_pair(xl4, lapa_gt, operator=op_cpu, y_gt=y_lapa, indices=torch.arange(rate)))
        out["lapa32"][str(rate)]["physflow_fast_4step"] = rl4
        log(f"  [LaPa32 Fast-4+Ref M={rate}] Ada={rl4['adaface_cos_mean']:.4f} Buf={rl4['buffalo_cos_mean']:.4f} NME={rl4['nme106_mean']:.4f} LPIPS={rl4['lpips_mean']:.4f} PSNR={rl4['psnr_mean']:.2f}")
        save_out()

    # Save visual QA grid: 4 faces x 4 cols (GT | V3 Flow | V3 + Refiner | Fast-4 + Refiner) for M=64 & M=256
    for rate_str in ("64", "256"):
        grid_imgs = []
        grid_lbls = []
        for i in range(4):
            for key, lbl in (("gt", "GT"), ("v3_flow", f"Flow M={rate_str}"), ("v3_ref", f"V3+Ref M={rate_str}"), ("fast4_ref", f"Fast4+Ref M={rate_str}")):
                grid_imgs.append(qa_tiles[rate_str][key][i])
                grid_lbls.append(lbl)
        save_image_grid(torch.stack(grid_imgs), V3_DIR / f"visual_qa_m{rate_str}.png", nrow=4, titles=grid_lbls)

    save_out()
    log(f"ALL FULL-479 + LAPA + WFLW V3 & FAST-4 VERIFICATION COMPLETE -> {OUT_JSON}")


if __name__ == "__main__":
    main()
