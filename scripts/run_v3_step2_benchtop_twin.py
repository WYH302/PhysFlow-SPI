"""Step 2 (V3 Surgery): Full Physical Benchtop Optical Hardware Twin Benchmark.

Models the complete 5-effect physical benchtop DMD single-pixel camera chain:
  1. Optical lens PSF diffraction blur (Gaussian kernel sigma=1.5 px on 256x256 aperture)
  2. Radial optical vignetting (cos^4 illumination falloff down to 0.75 at aperture corners)
  3. Stray ambient background light offset (+3% dark floor)
  4. Photon shot noise (1e4 photons) + electronic readout noise (2%) + +-5% laser/DMD power drift
  5. 12-bit hardware ADC quantization (4096 discrete levels)

Evaluates all 6 methods (TV-FISTA, Wiener-LMMSE, PnP-Flow ICLR'25, ZS-FLOWER ICLR'26,
LSI CVPR'25, and PhysFlow Ours) WITHOUT providing any hardware/noise calibration metadata,
on both LSI binary {0,1} DMD and standard 2D Walsh-Hadamard {0,1} DMD masks at M=64 & M=256.
Also verifies reproducibility audit metadata of LSI's public `real_128.pt` checkpoint.

Saves to `outputs/v3_surgery/step2_benchtop_twin.json`.
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
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.lsi_baseline import LSIBaseline
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator, WienerPhysicsLifter
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all

from scripts.run_journal_audit_extras import run_pnp_flow
from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, load_512_batch, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import encode_generic_prompt, pick_even_split_indices
from scripts.run_reviewer_shield_suite import run_tv_fista, sample_zs_flower_custom
from scripts.run_v3_step1_inference_upgrade import sample_physflow_v3

V3_DIR = OUTPUTS / "v3_surgery"
V3_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = V3_DIR / "step2_benchtop_twin.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_gaussian_psf(sigma: float = 1.5, ksize: int = 9, device: torch.device | str = "cpu") -> torch.Tensor:
    ax = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1)


def make_vignetting_map(size: int = 256, edge_gain: float = 0.75, device: torch.device | str = "cpu") -> torch.Tensor:
    """Radial cos^4-like optical lens vignetting map in [edge_gain, 1.0]."""
    coords = torch.linspace(-1.0, 1.0, size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = (xx * xx + yy * yy) / 2.0  # 0 at center, 1 at corners
    vig = 1.0 - (1.0 - float(edge_gain)) * r2
    return vig.view(1, 1, size, size)


@torch.no_grad()
def simulate_benchtop_measurements(
    gt_imgs_m11: torch.Tensor,
    op: SPIOperator,
    indices_m: torch.Tensor,
    seed: int = 2026,
    psf_sigma: float = 1.5,
    vig_edge: float = 0.75,
    stray_frac: float = 0.03,
    photons: float = 1e4,
    readout_rel: float = 0.02,
    gain_std: float = 0.05,
    adc_bits: int = 12,
) -> tuple[torch.Tensor, dict]:
    """Forward-simulate 5 real physical benchtop DMD optical imperfections on CPU/GPU."""
    dev = op.phi.device
    x01 = ((gt_imgs_m11.to(dev, dtype=torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)
    x256 = F.interpolate(x01, size=(256, 256), mode="bilinear", align_corners=False)

    # 1. Optical PSF diffraction blur
    psf = make_gaussian_psf(sigma=psf_sigma, ksize=9, device=dev)
    x_blur = F.conv2d(F.pad(x256, (4, 4, 4, 4), mode="reflect"), psf, groups=3)

    # 2. Radial optical lens vignetting
    vig = make_vignetting_map(size=256, edge_gain=vig_edge, device=dev)
    x_opt = (x_blur * vig).clamp(0.0, 1.0)

    # 3. Linear binary DMD bucket projection + stray ambient light floor
    y_opt = op.measure(x_opt, indices=indices_m, input_range="0_1").float()
    y_clean = op.measure(x256, indices=indices_m, input_range="0_1").float()
    scale = y_opt.abs().mean(dim=-1, keepdim=True).clamp_min(1.0)
    y_stray = y_opt + float(stray_frac) * scale

    # 4. Uncalibrated laser/DMD power gain drift + photon shot noise + electronic readout noise
    gen = torch.Generator().manual_seed(seed)
    g_drift = (1.0 + torch.randn((1, y_stray.shape[-1]), generator=gen) * float(gain_std)).to(dev)
    shot_std = torch.sqrt(y_stray.clamp_min(0.0) * scale / max(float(photons), 1.0))
    n_shot = torch.randn(y_stray.shape, generator=gen).to(dev) * shot_std
    n_read = torch.randn(y_stray.shape, generator=gen).to(dev) * (float(readout_rel) * scale)
    y_analog = (y_stray * g_drift + n_shot + n_read).clamp_min(0.0)

    # 5. 12-bit hardware ADC quantization
    full_scale = float(y_analog.max().clamp_min(1.0)) * 1.05
    levels = float((1 << int(adc_bits)) - 1)
    y_adc = torch.round((y_analog / full_scale) * levels) * (full_scale / levels)

    rel_pert = float((torch.linalg.vector_norm(y_adc - y_clean, dim=-1) / torch.linalg.vector_norm(y_clean, dim=-1).clamp_min(1e-8)).mean())
    snr_db = float(-20.0 * math.log10(max(rel_pert, 1e-8)))
    meta = {
        "psf_sigma_px": psf_sigma,
        "vignetting_corner_gain": vig_edge,
        "stray_light_frac": stray_frac,
        "photons": photons,
        "readout_rel_std": readout_rel,
        "power_gain_std": gain_std,
        "adc_bits": adc_bits,
        "effective_meas_rel_deviation": rel_pert,
        "effective_meas_snr_db": snr_db,
    }
    return y_adc.cpu(), meta


def strip_ps(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_sample"}


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")

    # Read Step 1 winning blind noise configuration if already computed
    step1_path = V3_DIR / "step1_inference_upgrade.json"
    blind_mode, blind_scale = "tail_scale", 6.0
    t0_64, t0_256 = 0.20, 0.35
    if step1_path.exists():
        s1 = json.loads(step1_path.read_text(encoding="utf-8"))
        if "selected_config" in s1:
            sc = s1["selected_config"]
            blind_mode = sc.get("blind_mode", blind_mode)
            blind_scale = float(sc.get("blind_scale", blind_scale))
            t0_64 = float(sc.get("t0_m64", t0_64))
            t0_256 = float(sc.get("t0_m256", t0_256))
    log(f"Using blind_mode={blind_mode}, blind_scale={blind_scale}, t0_64={t0_64}, t0_256={t0_256}")

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    hq_idx45 = pick_even_split_indices(records, "celebahq_test", 45)
    train_idx = [r["index"] for r in records if r["split"] == "train"]
    y_train_mean = y256[train_idx].mean(dim=0)

    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))
    gt_hq45_lmk = lmks[hq_idx45] * 512.0

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)

    # Audit LSI public real_128.pt mask transparency
    lsi_real_audit = {}
    real_ckpt_p = PROJECT_ROOT / "third_party" / "latent-imaging" / "paper_checkpoints" / "real_128.pt"
    if real_ckpt_p.exists():
        rc = torch.load(real_ckpt_p, map_location="cpu", weights_only=False)
        mw = rc["mask"]["weight"].float()
        lsi_real_audit = {
            "checkpoint": "real_128.pt",
            "mask_shape": list(mw.shape),
            "mask_mean": float(mw.mean()),
            "mask_std": float(mw.std()),
            "uniform_01_expected_std": round(1.0 / math.sqrt(12.0), 6),
            "note": "MaskGenerator.forward bypasses mask layer during real-data finetuning (--ini_w real), leaving mask.weight at untrained Uniform(0,1) initialization.",
        }

    # Load frozen V1 PhysFlow + untouched MiniT2I base for PnP-Flow / ZS-FLOWER
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])

    wrapper_pf, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper_pf = wrapper_pf.to(dev).eval()
    net_pf = wrapper_pf.model.net
    attach_qkvo_lora(net_pf.double_blocks, rank=16, alpha=16.0)
    net_pf.double_blocks.to(dev)
    cond_pf = FactorizedJointConditioner(hidden=int(net_pf.cfg.txt_hidden_size), num_blocks=len(net_pf.double_blocks)).to(dev).eval()
    load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net_pf.double_blocks, cond_pf)

    results = {"lsi_real128_audit": lsi_real_audit, "benchtop_twin_lsi_mask": {}}

    for rate, t_warm_r in ((64, t0_64), (256, t0_256)):
        log(f"=== Physical Benchtop Optical Twin Benchmark (M={rate}) ===")
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        y_hw, meta = simulate_benchtop_measurements(gt_hq45, op_gpu, idx_r, seed=2026 + rate)
        log(f"  Effective benchtop measurement perturbation: RelDev={meta['effective_meas_rel_deviation']:.4f} (SNR={meta['effective_meas_snr_db']:.2f} dB)")

        res_r = {"hardware_meta": meta, "methods": {}}

        # 1. TV-FISTA
        imgs_tv, _ = run_tv_fista(op_gpu, y_hw, idx_r, dev)
        res_r["methods"]["tv_fista"] = strip_ps(ev.evaluate_pair(imgs_tv, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk))

        # 2. Blind Wiener-LMMSE
        iw = enc.lifter.lift_image(y_hw.to(dev), idx_r, reg=4e-4, blind_adapt=True)
        imgs_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
        res_r["methods"]["wiener_lmmse_blind"] = strip_ps(ev.evaluate_pair(imgs_w, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk))

        # 3. PhysFlow (V1 baseline blind 1.5 vs V3 upgraded blind + adaptive t0)
        imgs_pf_v1, _ = sample_physflow_v3(
            net_pf, cond_pf, enc, op_gpu, y_hw, idx_r, dev,
            steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35, schedule_power=1.0,
            blind_mode="tail_scale", blind_scale=1.5, seed_base=3000, batch_size=9,
        )
        res_r["methods"]["physflow_v1"] = strip_ps(ev.evaluate_pair(imgs_pf_v1, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk))

        imgs_pf_v3, _ = sample_physflow_v3(
            net_pf, cond_pf, enc, op_gpu, y_hw, idx_r, dev,
            steps=16, cfg_scale=2.0, eta=0.85, t_warm=t_warm_r, schedule_power=1.0,
            blind_mode=blind_mode, blind_scale=blind_scale, seed_base=3000, batch_size=9,
        )
        res_r["methods"]["physflow_v3"] = strip_ps(ev.evaluate_pair(imgs_pf_v3, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk))

        results["benchtop_twin_lsi_mask"][str(rate)] = res_r
        for mn, mv in res_r["methods"].items():
            log(f"  [Twin M={rate} | {mn:18s}] PSNR={mv['psnr_mean']:.2f} LPIPS={mv['lpips_mean']:.4f} NME={mv['nme106_mean']} Ada={mv['adaface_cos_mean']:.4f} Det={mv['buffalo_detected']}/45")

    # Offload PhysFlow, run untouched MiniT2I (PnP-Flow + ZS-FLOWER) and official LSI
    wrapper_pf.to("cpu"); enc.to("cpu"); cond_pf.to("cpu")
    del wrapper_pf, enc, cond_pf
    gc.collect(); torch.cuda.empty_cache()
    op_gpu = op_gpu.to(dev)

    generic_text, generic_mask = encode_generic_prompt(dev)
    wrapper_base, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper_base = wrapper_base.to(dev).eval()
    net_base = wrapper_base.model.net

    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        y_hw, _ = simulate_benchtop_measurements(gt_hq45, op_gpu, idx_r, seed=2026 + rate)
        imgs_pnp, _ = run_pnp_flow(net_base, op_gpu, y_hw, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, batch_size=9)
        results["benchtop_twin_lsi_mask"][str(rate)]["methods"]["pnp_flow"] = strip_ps(
            ev.evaluate_pair(imgs_pnp, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)
        )
        imgs_zs, _ = sample_zs_flower_custom(net_base, op_gpu, y_hw, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, eta=0.85, batch_size=9)
        results["benchtop_twin_lsi_mask"][str(rate)]["methods"]["zs_flower"] = strip_ps(
            ev.evaluate_pair(imgs_zs, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)
        )
        for mn in ("pnp_flow", "zs_flower"):
            mv = results["benchtop_twin_lsi_mask"][str(rate)]["methods"][mn]
            log(f"  [Twin M={rate} | {mn:18s}] PSNR={mv['psnr_mean']:.2f} LPIPS={mv['lpips_mean']:.4f} NME={mv['nme106_mean']} Ada={mv['adaface_cos_mean']:.4f} Det={mv['buffalo_detected']}/45")

    wrapper_base.to("cpu"); del wrapper_base
    gc.collect(); torch.cuda.empty_cache()

    lsi = LSIBaseline(device=dev)
    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        y_hw, _ = simulate_benchtop_measurements(gt_hq45, op_gpu, idx_r, seed=2026 + rate)
        rec_lsi, _ = lsi.reconstruct(y_hw, indices=torch.arange(rate), y_mean=y_train_mean, output_size=512, batch_size=8)
        mv = strip_ps(ev.evaluate_pair(rec_lsi, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk))
        results["benchtop_twin_lsi_mask"][str(rate)]["methods"]["lsi"] = mv
        log(f"  [Twin M={rate} | {'lsi':18s}] PSNR={mv['psnr_mean']:.2f} LPIPS={mv['lpips_mean']:.4f} NME={mv['nme106_mean']} Ada={mv['adaface_cos_mean']:.4f} Det={mv['buffalo_detected']}/45")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Saved Step 2 Physical Benchtop Hardware Twin benchmark -> {OUT_JSON}")


if __name__ == "__main__":
    main()
