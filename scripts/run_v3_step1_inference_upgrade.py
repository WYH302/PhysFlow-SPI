"""Step 1 (V3 Surgery): Zero-Retrain Inference Upgrades (100% Read-Only wrt V1 Checkpoints).

Benchmarks on RTX 3090 without touching V1 checkpoints:
  Part 1A: Rate-Adaptive Warm-Start t0(M) & Non-Uniform End-Dense Time Grid (p in {1.0, 1.35, 1.65})
           on validation/diagnostic CelebA-HQ45 at M=64 and M=256 (16 steps, 0 extra NFE).
  Part 1B: Blind Spectral Noise Estimator Upgrade on 6-axis noise (CPU seed=777 exact parity):
           Baseline scale=1.5 vs Calibrated scale in {4.0, 6.0, 8.0, 12.0} vs Pure Spectral GCV
           vs Hybrid Tail-SURE.
  Part 1C: Once winning (t0(M), schedule_p, blind_mode) passes strict independent validation gate,
           runs full-cohort N=479 CelebA-HQ (with Inception-v3 FID & paired bootstrap vs LSI/V1),
           CelebA-45, LaPa-32, WFLW-6-subsets, and 6-axis noise parity, saving all results to
           `outputs/v3_surgery/step1_inference_upgrade.json`.
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
from src.eval.metrics import cosine_identity, psnr as calc_psnr, ssim as calc_ssim
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.minit2i_condition import predict_clean_factorized
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator, add_sensor_noise
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all

from scripts.eval_lapa_wflw import (
    WFLW_SUBSETS,
    align_image_list,
    load_lapa_samples,
    load_wflw_official_rgb_samples,
)
from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, NOISE_SCALE, load_512_batch, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import make_sem_prefix_mask, pick_even_split_indices
from scripts.run_reviewer_shield_suite import InceptionFID, NOISE_MODES, bootstrap_ci_and_p

V3_DIR = OUTPUTS / "v3_surgery"
V3_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = V3_DIR / "step1_inference_upgrade.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_time_grid(t0: float, steps: int, power: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create monotonic time grid on [t0, 1.0]. power=1.0 is uniform; power>1.0 densifies near t=1.0."""
    u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    if abs(power - 1.0) < 1e-6:
        t = t0 + (1.0 - t0) * u
    else:
        # s(u) = 1 - (1 - u)^p pushes more points toward u=1 (t=1) for fine detail synthesis
        s = 1.0 - torch.pow(1.0 - u, float(power))
        t = t0 + (1.0 - t0) * s
    t[0] = float(t0)
    t[-1] = 1.0
    return t.to(dtype=dtype)


@torch.no_grad()
def estimate_lambda_upgraded(
    lifter,
    y_sub: torch.Tensor,
    indices: torch.Tensor,
    mode: str = "tail_scale",
    base_reg: float = 4e-4,
    scale: float = 1.5,
    max_reg: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare blind spectral noise regularization strategies with zero noise metadata."""
    idx = indices.to(device=lifter.y_bar.device, dtype=torch.long)
    dy = (y_sub.float() - lifter.y_bar.index_select(1, idx)) / lifter.y_sd.index_select(1, idx)
    s_mat = lifter._subset_cov(idx).float()
    evals, evecs = torch.linalg.eigh(s_mat)
    evals = evals.clamp_min(0.0)
    m = int(idx.numel())
    k = max(4, m // 4)
    dy_eig = dy @ evecs  # [B, M]
    dy_sq = dy_eig.square()

    if mode == "tail_scale":
        excess = (dy_sq[:, :k] - evals[:k].unsqueeze(0)).mean(dim=-1).clamp_min(0.0)
        lam_b = (float(base_reg) + float(scale) * excess).clamp(float(base_reg), float(max_reg))
        return evals, evecs, lam_b

    if mode == "gcv":
        # Spectral Generalized Cross-Validation (Golub-Heath-Wahba 1979): zero magic constants
        grid = torch.logspace(math.log10(base_reg), math.log10(max_reg), 48, device=dy.device, dtype=torch.float32)  # [G]
        # Shrinkage residual weight: r_j(lam) = lam / (eval_j + lam)
        # evals: [1, 1, M], grid: [1, G, 1]
        ev = evals.view(1, 1, m)
        lm = grid.view(1, -1, 1)
        r = lm / (ev + lm).clamp_min(1e-12)  # [1, G, M]
        num = (r.square() * dy_sq.unsqueeze(1)).mean(dim=-1)  # [B, G]
        den = r.mean(dim=-1).square().clamp_min(1e-8)  # [1, G]
        gcv = num / den  # [B, G]
        best_idx = gcv.argmin(dim=-1)  # [B]
        lam_b = grid[best_idx]
        return evals, evecs, lam_b

    if mode == "tail_power":
        # Sublinear amplification boosts mid-SNR additive noise while capping cleanly at max_reg
        excess = (dy_sq[:, :k] - evals[:k].unsqueeze(0)).mean(dim=-1).clamp_min(0.0)
        lam_b = (float(base_reg) + float(scale) * torch.pow(excess + 1e-12, 0.75)).clamp(float(base_reg), float(max_reg))
        lam_b = torch.where(excess <= 1e-6, torch.full_like(lam_b, float(base_reg)), lam_b)
        return evals, evecs, lam_b

    if mode == "tail_gcv_hybrid":
        # Unbiased tail variance gates clean vs noisy; GCV refines moderate/high noise
        excess = (dy_sq[:, :k] - evals[:k].unsqueeze(0)).mean(dim=-1).clamp_min(0.0)
        lam_tail = (float(base_reg) + float(scale) * excess).clamp(float(base_reg), float(max_reg))
        grid = torch.logspace(math.log10(base_reg), math.log10(max_reg), 48, device=dy.device, dtype=torch.float32)
        ev = evals.view(1, 1, m)
        lm = grid.view(1, -1, 1)
        r = lm / (ev + lm).clamp_min(1e-12)
        gcv = (r.square() * dy_sq.unsqueeze(1)).mean(dim=-1) / r.mean(dim=-1).square().clamp_min(1e-8)
        lam_gcv = grid[gcv.argmin(dim=-1)]
        # Only engage GCV when excess tail variance detects real noise above clean floor
        lam_b = torch.where(excess > 3e-4, torch.maximum(lam_tail, lam_gcv), lam_tail)
        return evals, evecs, lam_b

    raise ValueError(f"Unknown blind mode: {mode}")


@torch.no_grad()
def sample_physflow_v3(
    net,
    conditioner: FactorizedJointConditioner,
    encoder: PhysicsDualDomainFactorizer,
    op: SPIOperator,
    y_meas: torch.Tensor,
    indices_m: torch.Tensor,
    dev: torch.device,
    steps: int = 16,
    cfg_scale: float = 2.0,
    eta: float = 0.85,
    t_warm: float = 0.35,
    schedule_power: float = 1.0,
    blind_mode: str = "tail_scale",
    blind_scale: float = 1.5,
    base_reg: float = 4e-4,
    max_reg: float = 0.25,
    seed_base: int = 3000,
    batch_size: int = 9,
) -> tuple[torch.Tensor, float]:
    net.eval()
    conditioner.eval()
    encoder.eval()
    n = int(y_meas.shape[0])
    outs = []
    t0_wall = time.perf_counter()

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        b = e - s
        yb = y_meas[s:e].to(dev)
        p_sem, p_id, p_geo = encoder(yb, indices_m)
        txt_b = p_sem.to(dtype=torch.bfloat16)
        msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
        id_tok = conditioner.project_id(p_id)
        geo_tok = conditioner.project_geo(p_geo)

        # Compute blind spectral damping once per batch
        evals, evecs, lam_b = estimate_lambda_upgraded(
            encoder.lifter, yb, indices_m, mode=blind_mode, base_reg=base_reg, scale=blind_scale, max_reg=max_reg
        )
        idx = indices_m.to(device=encoder.lifter.y_bar.device, dtype=torch.long)
        c_sub = encoder.lifter.C_Iy.index_select(1, idx).float()
        y_bar_s = encoder.lifter.y_bar.index_select(1, idx)
        y_sd_s = encoder.lifter.y_sd.index_select(1, idx)

        if t_warm > 0.0:
            dy = (yb.float() - y_bar_s) / y_sd_s
            coef = ((dy @ evecs) / (evals.unsqueeze(0) + lam_b.unsqueeze(1))) @ evecs.T
            iw = (encoder.lifter.I_bar.view(1, -1).float() + coef @ c_sub.T).view(-1, 3, encoder.lifter.img_size, encoder.lifter.img_size).clamp(0.0, 1.0)
            x_init = F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
        else:
            x_init = None

        gen = torch.Generator(device=dev).manual_seed(seed_base + s)
        dtype = txt_b.dtype
        eps0 = torch.randn(b, 3, 512, 512, device=dev, dtype=dtype, generator=gen) * NOISE_SCALE
        t0 = float(t_warm) if x_init is not None and t_warm > 0.0 else 0.0
        x = (t0 * x_init.to(device=dev, dtype=dtype) + (1.0 - t0) * eps0) if t0 > 0.0 else eps0
        timesteps = make_time_grid(t0, steps, schedule_power, dev, dtype)
        null = torch.zeros_like(msk_b)

        for idx_step in range(steps):
            t_cur = timesteps[idx_step].expand(b)
            t_next = timesteps[idx_step + 1]
            x0_c = predict_clean_factorized(
                net, x, t_cur, txt_b, msk_b,
                conditioner=conditioner, id_tokens=id_tok, geo_tokens=geo_tok, checkpoint=False,
            )
            if cfg_scale != 1.0:
                x0_u = predict_clean_factorized(net, x, t_cur, txt_b, null, conditioner=None, checkpoint=False)
                x0 = x0_u + (x0_c - x0_u) * cfg_scale
            else:
                x0 = x0_c

            ts = float(timesteps[idx_step])
            if ts < 0.93 and eta != 0.0:
                w_t = float(eta) if ts < 0.80 else float(eta) * 0.45
                with torch.amp.autocast(device_type=dev.type, enabled=False):
                    x01 = op.to_unit_interval(x0.float(), "-1_1").clamp(0.0, 1.0)
                    y_hat = op.measure(x01, indices=idx, input_range="0_1")
                    dy_res = (yb.float() - y_hat.float()) / y_sd_s
                    coef_p = ((dy_res @ evecs) / (evals.unsqueeze(0) + lam_b.unsqueeze(1))) @ evecs.T
                    delta = (coef_p @ c_sub.T).view(-1, 3, encoder.lifter.img_size, encoder.lifter.img_size)
                    delta = F.interpolate(delta, size=(512, 512), mode="bilinear", align_corners=False)
                    x0 = op.from_unit_interval((x01 + w_t * delta).clamp(0.0, 1.0), "-1_1").to(dtype=x0.dtype)

            step = (t_next - timesteps[idx_step]).to(dtype=x.dtype)
            v = (x0 - x) / (1.0 - t_cur)[:, None, None, None].clamp_min(1e-3)
            x = x + step * v

        outs.append(x.clamp(-1.0, 1.0).float().cpu())

    sec = (time.perf_counter() - t0_wall) / max(n, 1)
    return torch.cat(outs, dim=0), sec


def strip_ps(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_sample"}


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    payload: dict = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    hq_idx45 = pick_even_split_indices(records, "celebahq_test", 45)
    cb_idx45 = pick_even_split_indices(records, "celeba_test", 45)
    val_idx45 = pick_even_split_indices(records, "val", 45)
    all_hq_idx = [r["index"] for r in records if r["split"] == "celebahq_test"]

    gt_val45 = load_512_batch(records, val_idx45, torch.device("cpu"))
    gt_val45_lmk = lmks[val_idx45] * 512.0
    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))
    gt_hq45_lmk = lmks[hq_idx45] * 512.0
    gt_cb45 = load_512_batch(records, cb_idx45, torch.device("cpu"))
    gt_cb45_lmk = lmks[cb_idx45] * 512.0

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)

    # Load frozen V1 PhysFlow (100% read-only)
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])

    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev).eval()
    net = wrapper.model.net
    attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    cond = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev).eval()
    load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, cond)

    # =========================================================================
    # PART 1A: Validation Sweep of Rate-Adaptive t0(M) & Non-Uniform Time Grid
    # (Done strictly on CelebA `val` split to prevent test leakage!)
    # =========================================================================
    if "part1a_val_schedule_sweep" not in payload:
        log("=== PART 1A: Val Sweep of Rate-Adaptive t0(M) & End-Dense Time Grid ===")
        sw = {"m64": {}, "m256": {}}
        for rate_key, rate, candidates in (
            ("m64", 64, [
                ("t0.35_p1.00_v1", 0.35, 1.00),
                ("t0.25_p1.00",    0.25, 1.00),
                ("t0.20_p1.00",    0.20, 1.00),
                ("t0.25_p1.35",    0.25, 1.35),
                ("t0.20_p1.35",    0.20, 1.35),
                ("t0.20_p1.60",    0.20, 1.60),
                ("t0.35_p1.35",    0.35, 1.35),
            ]),
            ("m256", 256, [
                ("t0.35_p1.00_v1", 0.35, 1.00),
                ("t0.30_p1.00",    0.30, 1.00),
                ("t0.25_p1.00",    0.25, 1.00),
                ("t0.35_p1.35",    0.35, 1.35),
                ("t0.30_p1.35",    0.30, 1.35),
                ("t0.25_p1.35",    0.25, 1.35),
                ("t0.40_p1.35",    0.40, 1.35),
            ]),
        ):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            for tag, t0, pw in candidates:
                imgs, sec = sample_physflow_v3(
                    net, cond, enc, op_gpu, y256[val_idx45, :rate], idx_r, dev,
                    steps=16, cfg_scale=2.0, eta=0.85, t_warm=t0, schedule_power=pw,
                    blind_mode="tail_scale", blind_scale=1.5, seed_base=3000, batch_size=9,
                )
                r = ev.evaluate_pair(imgs, gt_val45, operator=op_cpu, y_gt=y256[val_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_val45_lmk)
                sw[rate_key][tag] = {**strip_ps(r), "t0": t0, "power": pw, "sec_per_img": sec}
                log(f"  [VAL {rate_key} | {tag:15s}] Ada={r['adaface_cos_mean']:.4f} Buf={r['buffalo_cos_mean']:.4f} NME={r['nme106_mean']:.4f} LPIPS={r['lpips_mean']:.4f} PSNR={r['psnr_mean']:.2f} RelRes={r['meas_rel_mean']:.4f}")
        payload["part1a_val_schedule_sweep"] = sw
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # =========================================================================
    # PART 1B: Blind Spectral Noise Estimator Upgrade on Validation + HQ45 (CPU seed=777)
    # =========================================================================
    if "part1b_blind_noise_upgrade" not in payload:
        log("=== PART 1B: Blind Spectral Noise Estimator Upgrade (6-Axis CPU seed=777) ===")
        idx64 = torch.arange(64, device=dev, dtype=torch.long)
        strategies = [
            ("v1_tail_1.5",     "tail_scale",      1.5),
            ("tail_4.0",        "tail_scale",      4.0),
            ("tail_6.0",        "tail_scale",      6.0),
            ("tail_8.0",        "tail_scale",      8.0),
            ("pure_gcv",        "gcv",             1.0),
            ("hybrid_tail_gcv", "tail_gcv_hybrid", 4.0),
        ]
        # Evaluate on CelebA-val45 first for fair selection, then verify exact HQ45 6-axis parity
        res_val = {}
        for sname, bmode, bscale in strategies:
            res_val[sname] = {}
            psnrs, lpips_l, nmes = [], [], []
            for nm in NOISE_MODES:
                if nm == "clean":
                    y_in, idx_use = y256[val_idx45, :64], idx64
                else:
                    gen_n = torch.Generator().manual_seed(777)
                    y_in, km = add_sensor_noise(y256[val_idx45, :64], mode=nm, dropout_prob=0.20, generator=gen_n)
                    if km is not None:
                        y_in, idx_use = y_in[:, km], idx64[km.to(dev)]
                    else:
                        idx_use = idx64
                _, _, lam_b = estimate_lambda_upgraded(enc.lifter, y_in.to(dev), idx_use, mode=bmode, scale=bscale)
                imgs, _ = sample_physflow_v3(
                    net, cond, enc, op_gpu, y_in, idx_use, dev,
                    steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35, schedule_power=1.0,
                    blind_mode=bmode, blind_scale=bscale, seed_base=3000, batch_size=9,
                )
                r = ev.evaluate_pair(imgs, gt_val45, operator=op_cpu, y_gt=y256[val_idx45, :64], indices=torch.arange(64), gt_landmarks_px=gt_val45_lmk)
                res_val[sname][nm] = {**strip_ps(r), "lam_mean": float(lam_b.mean())}
                psnrs.append(r["psnr_mean"]); lpips_l.append(r["lpips_mean"]); nmes.append(r["nme106_mean"])
            log(f"  [VAL Noise {sname:16s}] Avg6 PSNR={np.mean(psnrs):.2f} LPIPS={np.mean(lpips_l):.4f} NME={np.mean(nmes):.4f} | Clean={psnrs[0]:.2f} N1={psnrs[1]:.2f} N2={psnrs[2]:.2f} N3={psnrs[3]:.2f} N3b={psnrs[4]:.2f} N4={psnrs[5]:.2f}")
        payload["part1b_blind_noise_upgrade"] = {"val45": res_val}
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log("Part 1A & 1B validation sweeps completed cleanly! Inspect results before locking Full-479.")


if __name__ == "__main__":
    main()
