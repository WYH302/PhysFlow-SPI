"""Reviewer-Shield GPU Expansion Suite on RTX 3090 (100% Read-Only wrt V1 Checkpoints).

Executes 5 high-ROI SCI/JVCI reviewer-shield experiments and saves results incrementally
to `outputs/reviewer_shield_results.json` after each module:

  Module A: Direct Ablation of Blind Spectral Noise Estimator (Eqs. 13-14) across all
            6 noise regimes (N0-N4) at M=64 on CelebA-HQ (N=45):
            Fixed small reg (4e-4) vs Fixed large reg (0.05) vs Blind Adaptive (Ours),
            plus exact per-regime estimated lambda_adapt telemetry.
  Module B: Hyperparameter Sensitivity (t_0 warm-start & w_cfg guidance scale) +
            5-Seed Stochastic Uncertainty & Null-Space Posterior Contraction (M=64 vs M=256).
  Module C: Baseline Parity Completion across Tables 2, 3, 5, and 6:
            Classical TV-FISTA (Beck & Teboulle 2009), Wiener-LMMSE, and PnP-Flow (ICLR 2025)
            on CelebA-HQ (5 rates for TV-FISTA), CelebA (45), LaPa (32), 6-Axis Noise (45x6),
            and WFLW 6 Subsets (72).
  Module D: Hardware-Agnostic Optical DMD Mask Generalization (Zero Generator Retraining):
            1) Sequency-Ordered 2D Walsh-Hadamard {0,1} DMD mask (M=64 & M=256)
            2) Random Bernoulli {0,1} binary DMD mask (M=64 & M=256)
            Compares TV-FISTA, Wiener-LMMSE, PnP-Flow (ICLR'25), ZS-FLOWER (ICLR'26), and
            PhysFlow (re-fitted Wiener moments + 2-min Stage-1A encoder + frozen Stage-2 flow).
  Module E: Full-Scale Cohort Benchmark on ALL 479 Identity-Disjoint CelebA-HQ Portraits
            (10.6x scale-up over N=45) at M=64 & M=256 across 6 methods (TV-FISTA, Wiener-LMMSE,
            PnP-Flow, ZS-FLOWER, LSI CVPR'25, PhysFlow) with real Inception-v3 FID,
            95% Bootstrap CIs, and paired bootstrap p-values.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import torchvision.models as tv_models

from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer
from src.solvers.flower import flower_sample
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.lsi_baseline import LSIBaseline
from src.models.minit2i_condition import predict_clean_factorized
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import RATES, SPIOperator, WienerPhysicsLifter, add_sensor_noise
from src.paths import DATA, OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all

from scripts.eval_lapa_wflw import (
    WFLW_SUBSETS,
    align_image_list,
    load_lapa_samples,
    load_wflw_official_rgb_samples,
)
from scripts.run_journal_audit_extras import run_pnp_flow
from scripts.run_phase1a_real import train_encoder
from scripts.run_phase1b_oracle_real import (
    CACHE_DIR,
    CKPT_DIR,
    NOISE_SCALE,
    load_512_batch,
    load_generator_ckpt,
)
from scripts.run_phase2_e2e_physflow import (
    encode_generic_prompt,
    make_sem_prefix_mask,
    pick_even_split_indices,
)

OUT_JSON = OUTPUTS / "reviewer_shield_results.json"
NOISE_MODES = ("clean", "gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_payload(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Exact Inception-v3 FID (2048-D pool3)
# ---------------------------------------------------------------------------
class InceptionFID:
    def __init__(self, dev: torch.device):
        self.dev = dev
        m = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        m.eval().to(dev)
        self.net = m
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, imgs_m11: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        feats = []
        for s in range(0, int(imgs_m11.shape[0]), batch_size):
            x = imgs_m11[s : s + batch_size].to(self.dev, dtype=torch.float32)
            x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            f = self.net(x)
            feats.append(f.cpu().numpy().astype(np.float64))
        return np.concatenate(feats, axis=0)

    @staticmethod
    def frechet_distance(f_gen: np.ndarray, f_ref: np.ndarray) -> float:
        mu1, mu2 = f_gen.mean(axis=0), f_ref.mean(axis=0)
        sig1 = np.cov(f_gen, rowvar=False)
        sig2 = np.cov(f_ref, rowvar=False)
        diff = mu1 - mu2
        eps = 1e-6 * np.eye(sig1.shape[0], dtype=np.float64)
        covmean = scipy.linalg.sqrtm((sig1 + eps) @ (sig2 + eps))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))
        return max(0.0, fid)


# ---------------------------------------------------------------------------
# 2. Classical Compressive Sensing Baseline: GPU-Batched TV-FISTA (Beck & Teboulle 2009)
# ---------------------------------------------------------------------------
def _tv_prox_chambolle(y: torch.Tensor, weight: float, n_iter: int = 15) -> torch.Tensor:
    """Isotropic 2D TV proximal operator: argmin_x 0.5 ||x - y||_2^2 + weight * TV(x)."""
    px = torch.zeros_like(y)
    py = torch.zeros_like(y)
    tau = 0.248
    for _ in range(n_iter):
        # div(p)
        div_x = px.clone()
        div_x[..., 1:] = px[..., 1:] - px[..., :-1]
        div_x[..., -1] = -px[..., -2]
        div_y = py.clone()
        div_y[..., 1:, :] = py[..., 1:, :] - py[..., :-1, :]
        div_y[..., -1, :] = -py[..., -2, :]
        u = y + weight * (div_x + div_y)
        # grad(u)
        gx = torch.zeros_like(u)
        gx[..., :-1] = u[..., 1:] - u[..., :-1]
        gy = torch.zeros_like(u)
        gy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
        norm = torch.sqrt(gx * gx + gy * gy).clamp_min(1e-8)
        denom = 1.0 + (tau / max(weight, 1e-8)) * norm
        px = (px + (tau / max(weight, 1e-8)) * gx) / denom
        py = (py + (tau / max(weight, 1e-8)) * gy) / denom
    div_x = px.clone()
    div_x[..., 1:] = px[..., 1:] - px[..., :-1]
    div_x[..., -1] = -px[..., -2]
    div_y = py.clone()
    div_y[..., 1:, :] = py[..., 1:, :] - py[..., :-1, :]
    div_y[..., -1, :] = -py[..., -2, :]
    return (y + weight * (div_x + div_y)).clamp(0.0, 1.0)


@torch.no_grad()
def run_tv_fista(
    op: SPIOperator,
    y_meas: torch.Tensor,
    indices_m: torch.Tensor,
    dev: torch.device,
    iters: int = 100,
    tv_weight: float = 0.008,
    batch_size: int = 32,
) -> tuple[torch.Tensor, float]:
    """Solve min_{x in [0,1]} 0.5 ||Phi_S (3x) - y_S||_2^2 + tau TV(x) via FISTA on GPU."""
    phi = op._phi_rows(indices_m).to(dev, dtype=torch.float32)  # (M, N)
    # Exact spectral norm squared of A = 3 * Phi_S via small M x M Gram matrix
    gram = phi @ phi.T
    smax_sq = float(torch.linalg.eigvalsh(gram).max().clamp_min(1.0))
    L = 9.0 * smax_sq
    step = 1.0 / L
    # Scale TV weight naturally relative to measurement gradient energy
    lam_tv = float(tv_weight)
    res = op.resolution

    n = int(y_meas.shape[0])
    outs = []
    t0 = time.perf_counter()
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        yb = y_meas[s:e].to(dev, dtype=torch.float32)
        b = e - s
        # Warm initialize at neutral gray 0.5
        x = torch.full((b, 1, res, res), 0.5, device=dev, dtype=torch.float32)
        z = x.clone()
        tk = 1.0
        for _ in range(iters):
            z_flat = (3.0 * z).flatten(1)  # sum-of-RGB
            resid = z_flat @ phi.T - yb  # (b, M)
            grad = (3.0 * (resid @ phi)).view(b, 1, res, res)
            x_next = _tv_prox_chambolle((z - step * grad).clamp(-0.5, 1.5), weight=lam_tv, n_iter=12)
            tk_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            z = x_next + ((tk - 1.0) / tk_next) * (x_next - x)
            x = x_next
            tk = tk_next
        rgb = F.interpolate(x.repeat(1, 3, 1, 1), size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
        outs.append(rgb.cpu())
    sec = (time.perf_counter() - t0) / max(n, 1)
    return torch.cat(outs, dim=0), sec


# ---------------------------------------------------------------------------
# 3. Flexible PhysFlow Sampler (toggle blind_adapt, base_reg, t_warm, cfg, seed)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_physflow_custom(
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
    base_reg: float = 4e-4,
    blind_adapt: bool = True,
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

        if t_warm > 0.0:
            iw = encoder.lifter.lift_image(yb, indices_m, reg=base_reg, blind_adapt=blind_adapt)
            x_init = F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
        else:
            x_init = None

        gen = torch.Generator(device=dev).manual_seed(seed_base + s)

        def _pred(x, t, text_in, mask_in, _id=id_tok, _geo=geo_tok):
            is_uncond = float(mask_in.detach().float().sum()) == 0.0
            return predict_clean_factorized(
                net,
                x,
                t,
                text_in,
                mask_in,
                conditioner=None if is_uncond else conditioner,
                id_tokens=None if is_uncond else _id,
                geo_tokens=None if is_uncond else _geo,
                checkpoint=False,
            )

        def _proj(x0, t_cur, eta_t, _y=yb, _idx=indices_m):
            ts = float(t_cur[0])
            if ts >= 0.93:
                return x0
            w_t = eta_t if ts < 0.80 else eta_t * 0.45
            return encoder.lifter.wiener_proximal(
                x0, _y, op, _idx, eta=w_t, base_reg=base_reg, blind_adapt=blind_adapt, input_range="-1_1"
            )

        imgs = flower_sample(
            _pred,
            txt_b,
            msk_b,
            steps=steps,
            cfg_scale=cfg_scale,
            eta=eta,
            image_size=512,
            noise_scale=NOISE_SCALE,
            project=_proj,
            generator=gen,
            stochastic=False,
            eta_power=0.0,
            x_init=x_init,
            t_start=t_warm,
        )
        outs.append(imgs.float().cpu())
    sec = (time.perf_counter() - t0_wall) / max(n, 1)
    return torch.cat(outs, dim=0), sec


@torch.no_grad()
def sample_zs_flower_custom(
    net_base,
    op: SPIOperator,
    y_meas: torch.Tensor,
    indices_m: torch.Tensor,
    generic_text: torch.Tensor,
    generic_mask: torch.Tensor,
    dev: torch.device,
    steps: int = 16,
    cfg_scale: float = 2.0,
    eta: float = 0.85,
    batch_size: int = 9,
) -> tuple[torch.Tensor, float]:
    net_base.eval()
    n = int(y_meas.shape[0])
    outs = []
    t0_wall = time.perf_counter()
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        b = e - s
        yb = y_meas[s:e].to(dev)
        txt_b = generic_text.expand(b, -1, -1).to(dev, dtype=torch.bfloat16)
        msk_b = generic_mask.expand(b, -1).to(dev, dtype=torch.bfloat16)
        gen = torch.Generator(device=dev).manual_seed(3000 + s)

        def _pred(x, t, text_in, mask_in):
            return predict_clean_factorized(net_base, x, t, text_in, mask_in, conditioner=None, checkpoint=False)

        def _proj(x0, t_cur, eta_t, _y=yb, _idx=indices_m):
            if float(t_cur[0]) >= 0.90:
                return x0
            return op.proximal(x0.float(), _y, indices=_idx, lam=1e-2, eta=min(eta_t, 0.65), input_range="-1_1").to(dtype=x0.dtype)

        imgs = flower_sample(
            _pred,
            txt_b,
            msk_b,
            steps=steps,
            cfg_scale=cfg_scale,
            eta=eta,
            image_size=512,
            noise_scale=NOISE_SCALE,
            project=_proj,
            generator=gen,
            stochastic=False,
            eta_power=0.0,
            x_init=None,
            t_start=0.0,
        )
        outs.append(imgs.float().cpu())
    sec = (time.perf_counter() - t0_wall) / max(n, 1)
    return torch.cat(outs, dim=0), sec


# ---------------------------------------------------------------------------
# 4. Hardware DMD Mask Builders (Sequency-Ordered 2D Hadamard {0,1} & Bernoulli {0,1})
# ---------------------------------------------------------------------------
def build_hadamard_2d_dmd(n_patterns: int = 256, resolution: int = 256) -> torch.Tensor:
    """Sequency-ordered (Cake-Cutting) 2D Walsh-Hadamard shifted to physical DMD {0,1}."""
    k = 16  # 16 x 16 = 256 patterns
    H = scipy.linalg.hadamard(k).astype(np.float32)
    # Sort rows by sequency (number of zero-crossings / sign flips)
    sequencies = np.sum(H[:, :-1] != H[:, 1:], axis=1)
    order = np.argsort(sequencies)
    H_seq = H[order]

    # Order 2D basis (u, v) by low-to-high total sequency (u + v, max(u, v), u)
    pairs = sorted([(u, v) for u in range(k) for v in range(k)], key=lambda uv: (uv[0] + uv[1], max(uv[0], uv[1]), uv[0]))
    scale = resolution // k
    patterns = []
    for u, v in pairs[:n_patterns]:
        pat_16 = np.outer(H_seq[u], H_seq[v])  # in {-1, +1}
        pat_01 = (pat_16 > 0.0).astype(np.float32)  # exact {0, 1} DMD
        pat_res = np.repeat(np.repeat(pat_01, scale, axis=0), scale, axis=1)
        patterns.append(pat_res)
    return torch.from_numpy(np.stack(patterns, axis=0))  # (M, 256, 256)


def build_bernoulli_dmd(n_patterns: int = 256, resolution: int = 256, seed: int = 2026) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n_patterns, resolution, resolution), generator=g).float()


def bootstrap_ci_and_p(a: list[float], b: list[float], n_boot: int = 4000, seed: int = 42, higher_better: bool = True) -> dict:
    xa = np.asarray([v for v in a if v is not None and np.isfinite(v)], dtype=np.float64)
    xb = np.asarray([v for v in b if v is not None and np.isfinite(v)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = min(len(a), len(b))
    pairs = [(a[i], b[i]) for i in range(m) if a[i] is not None and b[i] is not None and np.isfinite(a[i]) and np.isfinite(b[i])]
    if len(pairs) < 5:
        return {"n_paired": len(pairs), "diff_mean": None, "ci95": None, "p_one_sided": None}
    pa = np.asarray([p[0] for p in pairs], dtype=np.float64)
    pb = np.asarray([p[1] for p in pairs], dtype=np.float64)
    diff = (pa - pb) if higher_better else (pb - pa)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    p_val = float(np.mean(means <= 0.0))
    return {
        "n_paired": len(diff),
        "diff_mean": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_one_sided": p_val,
    }


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
    all_hq_idx = [r["index"] for r in records if r["split"] == "celebahq_test"]
    train_idx = [r["index"] for r in records if r["split"] == "train"]
    y_train_mean = y256[train_idx].mean(dim=0)

    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))
    gt_hq45_lmk = lmks[hq_idx45] * 512.0
    gt_cb45 = load_512_batch(records, cb_idx45, torch.device("cpu"))
    gt_cb45_lmk = lmks[cb_idx45] * 512.0

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)

    # Load V1 PhysFlow (100% READ-ONLY!)
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
    # MODULE A: Direct Ablation of Blind Spectral Noise Estimator (Eqs. 13-14)
    # =========================================================================
    if "module_a_blind_spectral_ablation" not in payload:
        log("=== MODULE A: Blind Spectral Noise Estimator Ablation (Eqs. 13-14) across N0-N4 ===")
        idx64 = torch.arange(64, device=dev, dtype=torch.long)
        res_a: dict = {"lambda_telemetry": {}, "variants": {"fixed_small_4e4": {}, "fixed_large_5e2": {}, "blind_adaptive_ours": {}}}

        for nm in NOISE_MODES:
            if nm == "clean":
                y_in = y256[hq_idx45, :64]
                idx_use = idx64
            else:
                gen_n = torch.Generator().manual_seed(777)
                y_in, km = add_sensor_noise(y256[hq_idx45, :64], mode=nm, dropout_prob=0.20, generator=gen_n)
                if km is not None:
                    y_in = y_in[:, km]
                    idx_use = idx64[km.to(dev)]
                else:
                    idx_use = idx64

            # Measure exact estimated lambda_adapt telemetry from Eqs. (13)-(14)
            with torch.no_grad():
                _, _, lam_b = enc.lifter.estimate_spectral_reg(y_in.to(dev), idx_use, base_reg=4e-4)
                lam_np = lam_b.cpu().numpy()
                res_a["lambda_telemetry"][nm] = {
                    "mean": float(lam_np.mean()),
                    "std": float(lam_np.std()),
                    "median": float(np.median(lam_np)),
                    "min": float(lam_np.min()),
                    "max": float(lam_np.max()),
                }

            for vname, badapt, breg in (
                ("fixed_small_4e4", False, 4e-4),
                ("fixed_large_5e2", False, 5e-2),
                ("blind_adaptive_ours", True, 4e-4),
            ):
                imgs_v, _ = sample_physflow_custom(
                    net, cond, enc, op_gpu, y_in, idx_use, dev,
                    steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35,
                    base_reg=breg, blind_adapt=badapt, seed_base=3000, batch_size=9,
                )
                r = ev.evaluate_pair(
                    imgs_v, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :64],
                    indices=torch.arange(64), gt_landmarks_px=gt_hq45_lmk,
                )
                res_a["variants"][vname][nm] = {k: v for k, v in r.items() if k != "per_sample"}
            log(
                f"  [Noise={nm:16s}] lam_adapt={res_a['lambda_telemetry'][nm]['mean']:.4f} | "
                f"Fixed(4e-4) PSNR={res_a['variants']['fixed_small_4e4'][nm]['psnr_mean']:.2f} LPIPS={res_a['variants']['fixed_small_4e4'][nm]['lpips_mean']:.4f} NME={res_a['variants']['fixed_small_4e4'][nm]['nme106_mean']:.4f} | "
                f"Fixed(5e-2) PSNR={res_a['variants']['fixed_large_5e2'][nm]['psnr_mean']:.2f} LPIPS={res_a['variants']['fixed_large_5e2'][nm]['lpips_mean']:.4f} | "
                f"Ours(Blind) PSNR={res_a['variants']['blind_adaptive_ours'][nm]['psnr_mean']:.2f} LPIPS={res_a['variants']['blind_adaptive_ours'][nm]['lpips_mean']:.4f} NME={res_a['variants']['blind_adaptive_ours'][nm]['nme106_mean']:.4f}"
            )
        payload["module_a_blind_spectral_ablation"] = res_a
        save_payload(payload)
        log("Saved Module A -> reviewer_shield_results.json")

    # =========================================================================
    # MODULE B: Hyperparameter Sensitivity (t_0, w_cfg) + 5-Seed Uncertainty
    # =========================================================================
    if "module_b_sensitivity_and_uncertainty" not in payload:
        log("=== MODULE B: Hyperparameter Sensitivity (t_0, w_cfg) & 5-Seed Stochastic Uncertainty ===")
        idx64 = torch.arange(64, device=dev, dtype=torch.long)
        idx256 = torch.arange(256, device=dev, dtype=torch.long)
        res_b: dict = {"t_warm_sweep_m64": {}, "cfg_sweep_m64": {}, "multi_seed_5": {}}

        for tw in (0.0, 0.20, 0.35, 0.50, 0.65):
            imgs_t, _ = sample_physflow_custom(
                net, cond, enc, op_gpu, y256[hq_idx45, :64], idx64, dev,
                steps=16, cfg_scale=2.0, eta=0.85, t_warm=tw, batch_size=9,
            )
            r = ev.evaluate_pair(imgs_t, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :64], indices=torch.arange(64), gt_landmarks_px=gt_hq45_lmk)
            res_b["t_warm_sweep_m64"][f"{tw:.2f}"] = {k: v for k, v in r.items() if k != "per_sample"}
            log(f"  [t_warm={tw:.2f}] Ada={r['adaface_cos_mean']:.4f} NME={r['nme106_mean']:.4f} LPIPS={r['lpips_mean']:.4f} PSNR={r['psnr_mean']:.2f} Res={r['meas_rel_mean']:.4f}")

        for cg in (1.0, 1.5, 2.0, 2.5, 3.0):
            imgs_c, _ = sample_physflow_custom(
                net, cond, enc, op_gpu, y256[hq_idx45, :64], idx64, dev,
                steps=16, cfg_scale=cg, eta=0.85, t_warm=0.35, batch_size=9,
            )
            r = ev.evaluate_pair(imgs_c, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :64], indices=torch.arange(64), gt_landmarks_px=gt_hq45_lmk)
            res_b["cfg_sweep_m64"][f"{cg:.1f}"] = {k: v for k, v in r.items() if k != "per_sample"}
            log(f"  [w_cfg={cg:.1f}]  Ada={r['adaface_cos_mean']:.4f} NME={r['nme106_mean']:.4f} LPIPS={r['lpips_mean']:.4f} PSNR={r['psnr_mean']:.2f} Res={r['meas_rel_mean']:.4f}")

        # 5-Seed Stochastic Uncertainty & Null-Space Posterior Contraction (M=64 vs M=256)
        seeds = (3000, 4000, 5000, 6000, 7000)
        for rate, idx_r in ((64, idx64), (256, idx256)):
            per_seed = []
            stack_01 = []
            for sb in seeds:
                imgs_s, _ = sample_physflow_custom(
                    net, cond, enc, op_gpu, y256[hq_idx45, :rate], idx_r, dev,
                    steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35, seed_base=sb, batch_size=9,
                )
                r = ev.evaluate_pair(imgs_s, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)
                per_seed.append({k: v for k, v in r.items() if k != "per_sample"})
                # Downsample to 256 for fast posterior pixel std calculation
                stack_01.append(F.interpolate(((imgs_s + 1.0) * 0.5).clamp(0.0, 1.0), size=(256, 256), mode="bilinear", align_corners=False))
            st = torch.stack(stack_01, dim=0)  # (5, 45, 3, 256, 256)
            post_std_pixel = float(st.std(dim=0, unbiased=True).mean())
            summary = {}
            for k in ("adaface_cos_mean", "buffalo_cos_mean", "nme106_mean", "lpips_mean", "psnr_mean", "ssim_mean", "meas_rel_mean"):
                vals = [d[k] for d in per_seed if d.get(k) is not None]
                summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
            summary["posterior_pixel_std_01"] = post_std_pixel
            res_b["multi_seed_5"][str(rate)] = {"seeds": list(seeds), "summary": summary, "per_seed": per_seed}
            log(
                f"  [5-Seed M={rate:3d}] Ada={summary['adaface_cos_mean']['mean']:.4f}±{summary['adaface_cos_mean']['std']:.4f} | "
                f"NME={summary['nme106_mean']['mean']:.4f}±{summary['nme106_mean']['std']:.4f} | "
                f"PSNR={summary['psnr_mean']['mean']:.2f}±{summary['psnr_mean']['std']:.2f} | "
                f"LPIPS={summary['lpips_mean']['mean']:.4f}±{summary['lpips_mean']['std']:.4f} | "
                f"PosteriorPixelStd={post_std_pixel:.4f}"
            )
        payload["module_b_sensitivity_and_uncertainty"] = res_b
        save_payload(payload)
        log("Saved Module B -> reviewer_shield_results.json")

    # =========================================================================
    # MODULE D: Hardware-Agnostic Optical DMD Masks (Hadamard & Bernoulli {0,1})
    # =========================================================================
    if "module_d_hardware_masks" not in payload:
        log("=== MODULE D: Hardware-Agnostic DMD Mask Generalization (Hadamard & Bernoulli {0,1}) ===")
        imgs256_u8 = torch.load(CACHE_DIR / "images256_u8.pt", map_location="cpu", weights_only=True)
        sem_h = torch.load(CACHE_DIR / "sem_hidden_bf16.pt", map_location="cpu", weights_only=True)
        sem_m = torch.load(CACHE_DIR / "sem_mask_u8.pt", map_location="cpu", weights_only=True)
        ident_all = torch.load(CACHE_DIR / "identity_antelope.pt", map_location="cpu", weights_only=True).float()
        geo_all = lmks.view(-1, 212)
        sem_mean_tr = sem_h[train_idx].float().mean(dim=0)
        geo_mean_tr = geo_all[train_idx].float().mean(dim=0)
        val_idx = [r["index"] for r in records if r["split"] == "val"]
        generic_text, generic_mask = encode_generic_prompt(dev, max_len=64)

        # Also need untouched frozen MiniT2I for PnP-Flow / ZS-FLOWER on these hardware masks
        wrapper_base, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
        wrapper_base = wrapper_base.to(dev).eval()
        net_base = wrapper_base.model.net
        wrapper_base.requires_grad_(False)

        res_d: dict = {}
        for mask_name, phi_mat in (
            ("hadamard_dmd_01", build_hadamard_2d_dmd(256, 256)),
            ("bernoulli_dmd_01", build_bernoulli_dmd(256, 256, seed=2026)),
        ):
            t_m0 = time.perf_counter()
            op_m_gpu = SPIOperator(phi_mat).to(dev)
            op_m_cpu = SPIOperator(phi_mat)
            # Measure all 6,479 images through this hardware mask in ~0.5s
            y_list = []
            with torch.no_grad():
                for s in range(0, len(imgs256_u8), 256):
                    xb = imgs256_u8[s : s + 256].to(dev, dtype=torch.float32) / 255.0
                    y_list.append(op_m_gpu.measure(xb, indices=None, input_range="0_1").cpu())
            y_all_m = torch.cat(y_list, dim=0)

            # Exact condition numbers (Gram Phi Phi^T & measurement covariance Sigma_y)
            gram_m = (op_m_cpu.phi @ op_m_cpu.phi.T).double()
            ev_g = torch.linalg.eigvalsh(gram_m)
            kappa_gram = float(ev_g.max() / ev_g.min().clamp_min(1e-12))

            lifter_m = WienerPhysicsLifter(max_measurements=256, img_size=128).to(dev)
            lifter_m.fit(imgs256_u8[train_idx], y_all_m[train_idx])
            enc_m = PhysicsDualDomainFactorizer(
                op_m_gpu, lifter=lifter_m, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6
            ).to(dev)
            enc_m.set_target_means(sem_mean_tr, geo_mean_tr)
            ev_s = torch.linalg.eigvalsh(enc_m.lifter.Sigma_y.double())
            kappa_cov = float(ev_s.max() / ev_s.min().clamp_min(1e-12))
            log(f"  [{mask_name}] kappa(Phi Phi^T)={kappa_gram:.2e}, kappa(Sigma_y)={kappa_cov:.2e}")

            # Train Stage-1A dual-domain factorizer ONLY (~2 min, ZERO generator retraining!)
            tr_info = train_encoder(
                f"mask_{mask_name}", enc_m, y_all_m, sem_h, sem_m, ident_all, geo_all,
                train_idx, val_idx, dev, steps=700, batch_size=32, lr_head=5e-4, lr_backbone=6e-5, use_geo_inv=True, seed=42,
            )
            enc_m.eval()

            mask_res: dict = {
                "kappa_gram": kappa_gram,
                "kappa_cov": kappa_cov,
                "encoder_adapt_sec": float(tr_info["seconds"]),
                "generator_retrained": False,
                "methods": {"tv_fista": {}, "wiener_lmmse": {}, "pnp_flow": {}, "zs_flower": {}, "physflow_zero_gen_retrain": {}},
            }
            for rate in (64, 256):
                idx_r = torch.arange(rate, device=dev, dtype=torch.long)
                yb_hq = y_all_m[hq_idx45, :rate]
                # 1) TV-FISTA
                im_tv, _ = run_tv_fista(op_m_gpu, yb_hq, idx_r, dev, iters=100, tv_weight=0.008)
                # 2) Wiener-LMMSE
                with torch.no_grad():
                    iw = enc_m.lifter.lift_image(yb_hq.to(dev), idx_r, reg=4e-4, blind_adapt=True)
                    im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
                # 3) PnP-Flow (ICLR 2025)
                im_pnp, _ = run_pnp_flow(net_base, op_m_gpu, yb_hq, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, batch_size=9)
                # 4) ZS-FLOWER (ICLR 2026)
                im_flw, _ = sample_zs_flower_custom(net_base, op_m_gpu, yb_hq, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, eta=0.85, batch_size=9)
                # 5) PhysFlow (untouched Stage-2 generator!)
                im_pf, _ = sample_physflow_custom(net, cond, enc_m, op_m_gpu, yb_hq, idx_r, dev, steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35, batch_size=9)

                for mkey, ims in (
                    ("tv_fista", im_tv),
                    ("wiener_lmmse", im_w),
                    ("pnp_flow", im_pnp),
                    ("zs_flower", im_flw),
                    ("physflow_zero_gen_retrain", im_pf),
                ):
                    r = ev.evaluate_pair(ims, gt_hq45, operator=op_m_cpu, y_gt=yb_hq, indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)
                    mask_res["methods"][mkey][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}
                pf64 = mask_res["methods"]["physflow_zero_gen_retrain"][str(rate)]
                flw64 = mask_res["methods"]["zs_flower"][str(rate)]
                pnp64 = mask_res["methods"]["pnp_flow"][str(rate)]
                tv64 = mask_res["methods"]["tv_fista"][str(rate)]
                log(
                    f"  [{mask_name} M={rate:3d}] "
                    f"TV-FISTA PSNR={tv64['psnr_mean']:.2f} LPIPS={tv64['lpips_mean']:.4f} | "
                    f"PnP-Flow Ada={pnp64['adaface_cos_mean']:.4f} NME={pnp64['nme106_mean']:.4f} PSNR={pnp64['psnr_mean']:.2f} | "
                    f"ZS-FLW Ada={flw64['adaface_cos_mean']:.4f} PSNR={flw64['psnr_mean']:.2f} | "
                    f"PhysFlow Ada={pf64['adaface_cos_mean']:.4f} NME={pf64['nme106_mean']:.4f} LPIPS={pf64['lpips_mean']:.4f} PSNR={pf64['psnr_mean']:.2f} Res={pf64['meas_rel_mean']:.4f}"
                )
            mask_res["total_sec"] = time.perf_counter() - t_m0
            res_d[mask_name] = mask_res
            del enc_m
            gc.collect()
            torch.cuda.empty_cache()

        del imgs256_u8, sem_h, sem_m, ident_all
        wrapper_base.to("cpu")
        del wrapper_base
        gc.collect()
        torch.cuda.empty_cache()

        payload["module_d_hardware_masks"] = res_d
        save_payload(payload)
        log("Saved Module D -> reviewer_shield_results.json")

    # Free V1 PhysFlow temporarily while running untouched base baselines in Module C & E, or keep VRAM clean
    # =========================================================================
    # MODULE C: Baseline Parity Completion (TV-FISTA, Wiener-LMMSE, PnP-Flow)
    # =========================================================================
    if "module_c_baseline_parity" not in payload:
        log("=== MODULE C: Baseline Parity Completion (TV-FISTA, Wiener-LMMSE, PnP-Flow) ===")
        # Move PhysFlow to CPU briefly so untouched frozen MiniT2I has full VRAM
        wrapper.to("cpu"); cond.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

        generic_text, generic_mask = encode_generic_prompt(dev, max_len=64)
        wrapper_base, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
        wrapper_base = wrapper_base.to(dev).eval()
        net_base = wrapper_base.model.net
        wrapper_base.requires_grad_(False)

        res_c: dict = {"celebahq45_tv_fista": {}, "celeba45": {}, "lapa32": {}, "noise_m64": {}, "wflw_6subsets_m64": {}}

        # C1: TV-FISTA across all 5 rates on CelebA-HQ (N=45)
        for rate in RATES:
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            im_tv, sec_tv = run_tv_fista(op_gpu, y256[hq_idx45, :rate], idx_r, dev, iters=100, tv_weight=0.008)
            r = ev.evaluate_pair(im_tv, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq45_lmk)
            res_c["celebahq45_tv_fista"][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}
            res_c["celebahq45_tv_fista"][str(rate)]["sec_per_img"] = sec_tv
        log(
            f"  [TV-FISTA CelebA-HQ] M=64: Ada={res_c['celebahq45_tv_fista']['64']['adaface_cos_mean']:.4f} "
            f"NME={res_c['celebahq45_tv_fista']['64']['nme106_mean'] or -1:.4f} Det={res_c['celebahq45_tv_fista']['64']['buffalo_detected']}/45 "
            f"LPIPS={res_c['celebahq45_tv_fista']['64']['lpips_mean']:.4f} PSNR={res_c['celebahq45_tv_fista']['64']['psnr_mean']:.2f} "
            f"({res_c['celebahq45_tv_fista']['64']['sec_per_img']:.3f}s/img)"
        )

        # C2: CelebA (N=45) M=64 & M=256 for TV-FISTA, Wiener-LMMSE, PnP-Flow
        for mkey in ("tv_fista", "wiener_lmmse", "pnp_flow"):
            res_c["celeba45"][mkey] = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            yb = y256[cb_idx45, :rate]
            im_tv, _ = run_tv_fista(op_gpu, yb, idx_r, dev)
            with torch.no_grad():
                iw = enc.lifter.lift_image(yb.to(dev), idx_r, reg=4e-4, blind_adapt=True)
                im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
            im_pnp, _ = run_pnp_flow(net_base, op_gpu, yb, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, batch_size=9)
            for mkey, ims in (("tv_fista", im_tv), ("wiener_lmmse", im_w), ("pnp_flow", im_pnp)):
                r = ev.evaluate_pair(ims, gt_cb45, operator=op_cpu, y_gt=yb, indices=torch.arange(rate), gt_landmarks_px=gt_cb45_lmk)
                res_c["celeba45"][mkey][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}
        log(f"  [CelebA45 PnP-Flow] M=64 Ada={res_c['celeba45']['pnp_flow']['64']['adaface_cos_mean']:.4f} PSNR={res_c['celeba45']['pnp_flow']['64']['psnr_mean']:.2f} | M=256 Ada={res_c['celeba45']['pnp_flow']['256']['adaface_cos_mean']:.4f} PSNR={res_c['celeba45']['pnp_flow']['256']['psnr_mean']:.2f}")

        # C3: 6-Axis Sensor Perturbation Noise (M=64, N=45) for TV-FISTA, Wiener-LMMSE, PnP-Flow
        idx64 = torch.arange(64, device=dev, dtype=torch.long)
        for mkey in ("tv_fista", "wiener_lmmse", "pnp_flow"):
            res_c["noise_m64"][mkey] = {}
        for nm in NOISE_MODES:
            if nm == "clean":
                y_in = y256[hq_idx45, :64]
                idx_use = idx64
            else:
                gen_n = torch.Generator().manual_seed(777)
                y_in, km = add_sensor_noise(y256[hq_idx45, :64], mode=nm, dropout_prob=0.20, generator=gen_n)
                if km is not None:
                    y_in = y_in[:, km]
                    idx_use = idx64[km.to(dev)]
                else:
                    idx_use = idx64
            im_tv, _ = run_tv_fista(op_gpu, y_in, idx_use, dev)
            with torch.no_grad():
                iw = enc.lifter.lift_image(y_in.to(dev), idx_use, reg=4e-4, blind_adapt=True)
                im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
            im_pnp, _ = run_pnp_flow(net_base, op_gpu, y_in, idx_use, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, batch_size=9)
            for mkey, ims in (("tv_fista", im_tv), ("wiener_lmmse", im_w), ("pnp_flow", im_pnp)):
                r = ev.evaluate_pair(ims, gt_hq45, operator=op_cpu, y_gt=y256[hq_idx45, :64], indices=torch.arange(64), gt_landmarks_px=gt_hq45_lmk)
                res_c["noise_m64"][mkey][nm] = {k: v for k, v in r.items() if k != "per_sample"}
        log(f"  [Noise N4 Combined] Wiener PSNR={res_c['noise_m64']['wiener_lmmse']['combined']['psnr_mean']:.2f} | PnP-Flow PSNR={res_c['noise_m64']['pnp_flow']['combined']['psnr_mean']:.2f} Det={res_c['noise_m64']['pnp_flow']['combined']['buffalo_detected']}/45")

        # C4: External LaPa (N=32, M=64 & 256) and External WFLW 6 Subsets (N=72, M=64)
        os.environ["INSIGHTFACE_HOME"] = str((PROJECT_ROOT / "checkpoints" / "insightface").resolve())
        from insightface.app import FaceAnalysis

        buf_align = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], root=str((PROJECT_ROOT / "checkpoints" / "insightface").resolve()))
        buf_align.prepare(ctx_id=-1, det_size=(640, 640))
        lapa_gt, lapa_y256 = align_image_list(buf_align, load_lapa_samples(32), op_gpu, dev, target_n=32)
        wflw_raw = load_wflw_official_rgb_samples(12)
        wflw_aligned = {sub: align_image_list(buf_align, wflw_raw[sub], op_gpu, dev, target_n=12) for sub in WFLW_SUBSETS}
        del buf_align
        gc.collect()

        for mkey in ("tv_fista", "wiener_lmmse", "pnp_flow"):
            res_c["lapa32"][mkey] = {}
            res_c["wflw_6subsets_m64"][mkey] = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            yb = lapa_y256[:, :rate]
            im_tv, _ = run_tv_fista(op_gpu, yb, idx_r, dev)
            with torch.no_grad():
                iw = enc.lifter.lift_image(yb.to(dev), idx_r, reg=4e-4, blind_adapt=True)
                im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
            im_pnp, _ = run_pnp_flow(net_base, op_gpu, yb, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, batch_size=8)
            for mkey, ims in (("tv_fista", im_tv), ("wiener_lmmse", im_w), ("pnp_flow", im_pnp)):
                r = ev.evaluate_pair(ims, lapa_gt, operator=op_cpu, y_gt=yb, indices=torch.arange(rate))
                res_c["lapa32"][mkey][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}
        log(f"  [LaPa32 PnP-Flow] M=64 Ada={res_c['lapa32']['pnp_flow']['64']['adaface_cos_mean']:.4f} PSNR={res_c['lapa32']['pnp_flow']['64']['psnr_mean']:.2f} | M=256 Ada={res_c['lapa32']['pnp_flow']['256']['adaface_cos_mean']:.4f} PSNR={res_c['lapa32']['pnp_flow']['256']['psnr_mean']:.2f}")

        for sub in WFLW_SUBSETS:
            gt_s, y_s = wflw_aligned[sub]
            yb = y_s[:, :64]
            im_tv, _ = run_tv_fista(op_gpu, yb, idx64, dev)
            with torch.no_grad():
                iw = enc.lifter.lift_image(yb.to(dev), idx64, reg=4e-4, blind_adapt=True)
                im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
            im_pnp, _ = run_pnp_flow(net_base, op_gpu, yb, idx64, generic_text, generic_mask, dev, steps=12, cfg_scale=2.0, batch_size=6)
            for mkey, ims in (("tv_fista", im_tv), ("wiener_lmmse", im_w), ("pnp_flow", im_pnp)):
                r = ev.evaluate_pair(ims, gt_s, operator=op_cpu, y_gt=yb, indices=torch.arange(64))
                res_c["wflw_6subsets_m64"][mkey][sub] = {k: v for k, v in r.items() if k != "per_sample"}
        log("  [WFLW 6-Subsets] Completed TV-FISTA, Wiener-LMMSE, and PnP-Flow")

        wrapper_base.to("cpu")
        del wrapper_base
        gc.collect()
        torch.cuda.empty_cache()

        payload["module_c_baseline_parity"] = res_c
        save_payload(payload)
        log("Saved Module C -> reviewer_shield_results.json")

    # =========================================================================
    # MODULE E: Full-Scale Cohort Benchmark (ALL 479 Identity-Disjoint CelebA-HQ + FID)
    # =========================================================================
    if "module_e_full479_fid" not in payload:
        log(f"=== MODULE E: Full-Scale Benchmark on ALL {len(all_hq_idx)} Identity-Disjoint CelebA-HQ Portraits + Real FID ===")
        gt_all = load_512_batch(records, all_hq_idx, torch.device("cpu"))
        gt_all_lmk = lmks[all_hq_idx] * 512.0
        fid_calc = InceptionFID(dev)
        f_ref = fid_calc.features(gt_all, batch_size=32)
        log(f"  Extracted reference Inception-v3 pool3 features: {f_ref.shape}")

        res_e: dict = {"n_portraits": len(all_hq_idx), "rates": {"64": {}, "256": {}}, "paired_bootstrap_vs_physflow": {"64": {}, "256": {}}}
        per_sample_store: dict[str, dict[str, dict]] = {"64": {}, "256": {}}

        def _eval_and_record(rate_int: int, mname: str, imgs_cpu: torch.Tensor, sec_img: float) -> None:
            rkey = str(rate_int)
            f_gen = fid_calc.features(imgs_cpu, batch_size=32)
            fid_val = fid_calc.frechet_distance(f_gen, f_ref)
            r = ev.evaluate_pair(
                imgs_cpu, gt_all, operator=op_cpu, y_gt=y256[all_hq_idx, :rate_int],
                indices=torch.arange(rate_int), gt_landmarks_px=gt_all_lmk,
            )
            ps = r.pop("per_sample")
            per_sample_store[rkey][mname] = ps
            r["fid_inceptionv3"] = fid_val
            r["sec_per_img"] = sec_img
            res_e["rates"][rkey][mname] = r
            log(
                f"  [Full479 M={rate_int:3d} | {mname:14s}] FID={fid_val:6.2f} | "
                f"Ada={r['adaface_cos_mean']:.4f} Buf={r['buffalo_cos_mean'] or 0:.4f} "
                f"NME={r['nme106_mean'] or -1:.4f} Det={r['buffalo_detected']}/{len(all_hq_idx)} | "
                f"LPIPS={r['lpips_mean']:.4f} PSNR={r['psnr_mean']:.2f} SSIM={r['ssim_mean']:.4f} Res={r['meas_rel_mean']:.4f}"
            )

        # 1) TV-FISTA & Wiener-LMMSE (ultra-fast)
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            yb = y256[all_hq_idx, :rate]
            im_tv, sec_tv = run_tv_fista(op_gpu, yb, idx_r, dev, iters=100, tv_weight=0.008, batch_size=32)
            _eval_and_record(rate, "tv_fista", im_tv, sec_tv)
            del im_tv

            t1 = time.perf_counter()
            with torch.no_grad():
                iw = enc.lifter.lift_image(yb.to(dev), idx_r, reg=4e-4, blind_adapt=True)
                im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
            sec_w = (time.perf_counter() - t1) / len(all_hq_idx)
            _eval_and_record(rate, "wiener_lmmse", im_w, sec_w)
            del im_w
            gc.collect()

        # 2) PhysFlow (Ours, 16-step) on GPU
        wrapper.to(dev); cond.to(dev)
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            im_pf, sec_pf = sample_physflow_custom(
                net, cond, enc, op_gpu, y256[all_hq_idx, :rate], idx_r, dev,
                steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35, batch_size=12,
            )
            _eval_and_record(rate, "physflow", im_pf, sec_pf)
            del im_pf
            gc.collect()
        wrapper.to("cpu"); cond.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

        # 3) Untouched frozen MiniT2I -> ZS-FLOWER (ICLR 2026) & PnP-Flow (ICLR 2025)
        generic_text, generic_mask = encode_generic_prompt(dev, max_len=64)
        wrapper_base, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
        wrapper_base = wrapper_base.to(dev).eval()
        net_base = wrapper_base.model.net
        wrapper_base.requires_grad_(False)
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            yb = y256[all_hq_idx, :rate]
            im_flw, sec_flw = sample_zs_flower_custom(net_base, op_gpu, yb, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, eta=0.85, batch_size=12)
            _eval_and_record(rate, "zs_flower", im_flw, sec_flw)
            del im_flw

            im_pnp, sec_pnp = run_pnp_flow(net_base, op_gpu, yb, idx_r, generic_text, generic_mask, dev, steps=16, cfg_scale=2.0, batch_size=12)
            _eval_and_record(rate, "pnp_flow", im_pnp, sec_pnp)
            del im_pnp
            gc.collect()
        wrapper_base.to("cpu")
        del wrapper_base
        gc.collect(); torch.cuda.empty_cache()

        # 4) Official CVPR 2025 LSI (StyleGAN-XL) on GPU
        lsi = LSIBaseline(device=dev)
        for rate in (64, 256):
            t_l0 = time.perf_counter()
            im_lsi, _ = lsi.reconstruct(y256[all_hq_idx, :rate], indices=torch.arange(rate), y_mean=y_train_mean, output_size=512, batch_size=16)
            sec_lsi = (time.perf_counter() - t_l0) / len(all_hq_idx)
            _eval_and_record(rate, "lsi", im_lsi, sec_lsi)
            del im_lsi
            gc.collect()
        del lsi
        gc.collect(); torch.cuda.empty_cache()

        # 5) Paired Bootstrap 95% CIs and p-values (PhysFlow vs every baseline on N=479!)
        for rkey in ("64", "256"):
            pf_ps = per_sample_store[rkey]["physflow"]
            for base_m in ("tv_fista", "wiener_lmmse", "pnp_flow", "zs_flower", "lsi"):
                b_ps = per_sample_store[rkey][base_m]
                res_e["paired_bootstrap_vs_physflow"][rkey][base_m] = {
                    "adaface": bootstrap_ci_and_p(pf_ps["adaface_cos"], b_ps["adaface_cos"], higher_better=True),
                    "nme106": bootstrap_ci_and_p(pf_ps["nme106"], b_ps["nme106"], higher_better=False),
                    "lpips": bootstrap_ci_and_p(pf_ps["lpips"], b_ps["lpips"], higher_better=False),
                    "psnr": bootstrap_ci_and_p(pf_ps["psnr"], b_ps["psnr"], higher_better=True),
                }

        payload["module_e_full479_fid"] = res_e
        save_payload(payload)
        log("Saved Module E -> reviewer_shield_results.json")

    log(f"ALL 5 REVIEWER-SHIELD MODULES COMPLETE -> {OUT_JSON}")


if __name__ == "__main__":
    main()
