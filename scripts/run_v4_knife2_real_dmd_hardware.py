"""Knife 2 (V4 Top-Tier): Real Physical Optical Benchtop DMD Camera Validation (Zero Retraining).

Validates PhysFlow's Blind Spectral Noise-Adaptive Kalman-Wiener proximal solver and frozen
generic MiniT2I-B/16 pixel-space flow prior directly on FIVE genuine physical optical single-pixel
camera datasets acquired on European laboratory DMD benches (CREATIS / INSA-Lyon):
  1-4. SPIHIM 64x64 Vialux DMD + White LED + Avantes spectrometer (514.7-750.3 nm, 8,192 patterns):
       - zoom_x12_usaf_group5 (USAF 1951 Group 5 optical resolution target)
       - zoom_x12_starsector (Siemens Star radial resolution target)
       - zoom_x12_tomato_slice_2 (Microscopic tomato plant cells, x12 mag)
       - zoom_x2_tomato_slice_2 (Biological tomato tissue, x2 mag)
  5.   ONE-PIX 2026 32x32x3 RGB ColorChecker (Vialux DMD + OceanInsight FLMS21318 spectrometer,
       350-800 nm, 2,048 split Walsh-Hadamard patterns).

Three genuine physical hardware protocols:
  (A) Broadband high-SNR differential DMD (520-700 nm) at SR = 6.25% (M=256), 12.5% (M=512), 25% (M=1024).
  (B) Narrowband low-photon-flux DMD (600 +/- 1 nm, authentic spectrometer shot+readout noise) at SR = 12.5%
      to test Proposition 2 blind spectral noise estimator lambda_adapt without any noise calibration.
  (C) Single-exposure unipolar {0,1} binary DMD (positive patterns ONLY, half acquisition time) at SR = 12.5%.

Compares 6 methods with ZERO retraining:
  Pinv | TV-FISTA | PnP-Flow (ICLR'25) | ZS-FLOWER (ICLR'26) | Blind Spectral Wiener (Ours) | PhysFlow-ZS (Ours)

Peak VRAM < 3.5 GB. Saves outputs/v4_toptier/knife2_real_dmd.json + QA strip + figure tiles.
"""
from __future__ import annotations

import ast
import gc
import json
import math
import re
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
from scipy.linalg import hadamard

from src.eval.metrics import psnr as calc_psnr, ssim as calc_ssim
from src.models.minit2i_condition import pad_context_to_256, predict_clean
from src.models.minit2i_loader import load_b16_transformer
from src.paths import DATA, OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all
from scripts.run_phase1b_oracle_real import NOISE_SCALE

BENCH_DIR = DATA / "raw" / "real_dmd_benchtop"
V4_DIR = OUTPUTS / "v4_toptier"
V4_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = V4_DIR / "knife2_real_dmd.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def walsh_1d(n: int) -> np.ndarray:
    H = hadamard(n).astype(np.float64)
    zc = np.sum(np.abs(np.diff(H, axis=1)) > 0, axis=1)
    return H[np.argsort(zc)]


def build_stationary_cov(size: int, corr_len: float = 5.0) -> torch.Tensor:
    """Closed-form 2D stationary Ornstein-Uhlenbeck (Matern-1/2) natural-image spatial covariance."""
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pts = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (N, 2)
    dists = torch.cdist(pts, pts, p=2.0)
    return torch.exp(-dists / float(corr_len))


def blind_spectral_wiener_solve(
    phi: torch.Tensor,
    y: torch.Tensor,
    sigma_nat: torch.Tensor,
    x_mean: float = 0.5,
    base_lam: float = 1e-4,
    max_lam: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """Closed-form DC-Invariant Spectral GCV Kalman-Wiener Proximal Lifting (Zero Noise Calibration).

    Normalizes prior measurement covariance by its median eigenvalue so rank-1 DC spikes in
    unipolar {0,1} binary DMD matrices cannot skew AC regularization, then selects the optimal
    spectral damping lambda_adapt via exact closed-form Generalized Cross-Validation (Golub-Heath-Wahba).
    """
    M, N = phi.shape
    c_iy = sigma_nat @ phi.T  # [N, M]
    sy = phi @ c_iy           # [M, M]
    sy = 0.5 * (sy + sy.T)
    evals_raw, evecs = torch.linalg.eigh(sy)
    tr_scale = float(evals_raw.median().clamp_min(1e-8).item())
    evals = (evals_raw / tr_scale).clamp_min(1e-7)

    y0 = phi.sum(dim=1, keepdim=True).T * float(x_mean)  # [1, M]
    dy = y - y0  # [C, M]
    proj = dy @ evecs  # [C, M]
    proj_sq = proj.square()

    # Evaluate Spectral GCV over AC spectrum (exclude rank-1 DC spike if unipolar {0,1} DMD)
    is_unip = float(evals[-1] / evals[-2].clamp_min(1e-6)) > 15.0
    sl = slice(0, M - 1) if is_unip else slice(0, M)
    ev_g = evals[sl].view(1, 1, -1)
    ps_g = proj_sq[:, sl].unsqueeze(1)
    grid = torch.logspace(math.log10(base_lam), math.log10(max_lam), 64, device=phi.device)
    lm = grid.view(1, -1, 1)
    r = lm / (ev_g + lm)
    gcv = (r.square() * ps_g).mean(dim=-1) / r.mean(dim=-1).square().clamp_min(1e-10)
    lam_adapt = grid[gcv.argmin(dim=-1)]  # [C]

    denom = tr_scale * (evals.unsqueeze(0) + lam_adapt.unsqueeze(1))  # [C, M]
    coef = (proj / denom) @ evecs.T  # [C, M]
    x_w = float(x_mean) + coef @ c_iy.T  # [C, N]
    return x_w, {
        "lam_adapt_mean": float(lam_adapt.mean().item()),
        "tr_scale": tr_scale,
    }


def kalman_wiener_prox_step(
    x_cur: torch.Tensor,
    phi: torch.Tensor,
    y: torch.Tensor,
    c_iy: torch.Tensor,
    evals: torch.Tensor,
    evecs: torch.Tensor,
    tr_scale: float,
    lam_adapt: torch.Tensor,
    eta: float = 0.85,
) -> torch.Tensor:
    """Exact Proposition 1 Kalman-Wiener proximal data-consistency update on [C, N] vector."""
    y_hat = x_cur @ phi.T  # [C, M]
    res = y - y_hat
    proj = res @ evecs
    denom = tr_scale * (evals.unsqueeze(0) + lam_adapt.unsqueeze(1))
    delta = ((proj / denom) @ evecs.T) @ c_iy.T
    return x_cur + float(eta) * delta


def tv_fista_solve(phi: torch.Tensor, y: torch.Tensor, img_size: int, lam_tv: float = 0.012, n_iters: int = 120) -> torch.Tensor:
    """Classical Compressed Sensing 2D Isotropic TV-FISTA solver for [C, M] -> [C, img_size, img_size]."""
    C, M = y.shape
    # Lipschitz constant L = ||Phi||_2^2 via power iteration
    v = torch.randn(phi.shape[1], 1, device=phi.device)
    for _ in range(15):
        v = phi.T @ (phi @ v)
        v = v / v.norm().clamp_min(1e-8)
    L = float(((phi @ v).norm() ** 2).item()) + 1e-6
    step = 1.0 / L

    def chambolle_tv_prox(u_img: torch.Tensor, weight: float, inner: int = 12) -> torch.Tensor:
        px = torch.zeros_like(u_img)
        py = torch.zeros_like(u_img)
        tau = 0.24
        for _ in range(inner):
            div = torch.zeros_like(u_img)
            div[:, :, :-1] += px[:, :, :-1]
            div[:, :, 1:] -= px[:, :, :-1]
            div[:, :-1, :] += py[:, :-1, :]
            div[:, 1:, :] -= py[:, :-1, :]
            z = div - u_img / max(weight, 1e-8)
            gx = torch.zeros_like(u_img)
            gy = torch.zeros_like(u_img)
            gx[:, :, :-1] = z[:, :, 1:] - z[:, :, :-1]
            gy[:, :-1, :] = z[:, 1:, :] - z[:, :-1, :]
            norm = torch.sqrt(gx * gx + gy * gy).clamp_min(1.0)
            px = (px + tau * gx) / (1.0 + tau * norm)
            py = (py + tau * gy) / (1.0 + tau * norm)
        div = torch.zeros_like(u_img)
        div[:, :, :-1] += px[:, :, :-1]
        div[:, :, 1:] -= px[:, :, :-1]
        div[:, :-1, :] += py[:, :-1, :]
        div[:, 1:, :] -= py[:, :-1, :]
        return u_img - weight * div

    # Initialize with damped backprojection
    x = torch.full((C, img_size, img_size), 0.5, device=phi.device)
    z = x.clone()
    t_k = 1.0
    for _ in range(n_iters):
        zf = z.view(C, -1)
        grad = ((zf @ phi.T) - y) @ phi  # [C, N]
        u = z - step * grad.view(C, img_size, img_size)
        x_next = chambolle_tv_prox(u, lam_tv * step * L * 0.15, inner=10).clamp(0.0, 1.0)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
        z = x_next + ((t_k - 1.0) / t_next) * (x_next - x)
        x = x_next
        t_k = t_next
    return x


def encode_neutral_optical_prompt(dev: torch.device, prompt: str = "a sharp high-contrast optical image") -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a neutral non-face prompt with FLAN-T5-Large and immediately free T5 VRAM."""
    from transformers import AutoTokenizer, T5EncoderModel

    flan = PROJECT_ROOT / "checkpoints" / "flan-t5-large"
    tok = AutoTokenizer.from_pretrained(flan, local_files_only=True)
    enc = T5EncoderModel.from_pretrained(flan, torch_dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    inp = tok([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    with torch.inference_mode():
        h = enc(input_ids=inp.input_ids.to(dev), attention_mask=inp.attention_mask.to(dev)).last_hidden_state
    m = inp.attention_mask.to(dtype=torch.uint8)
    h, m = pad_context_to_256(h.to(dtype=torch.bfloat16), m, prompt_length=256)
    h, m = h.cpu(), m.cpu()
    enc.to("cpu"); del enc, tok; gc.collect(); torch.cuda.empty_cache()
    return h.to(dev), m.to(dev)


@torch.no_grad()
def predict_uncond_clean(net, x_t: torch.Tensor, t_val: float, txt_emb: torch.Tensor, txt_mask: torch.Tensor) -> torch.Tensor:
    """Predict clean x_1 estimate using frozen generic MiniT2I-B/16 x-prediction flow backbone (NO face LoRA!)."""
    b = x_t.shape[0]
    t_tensor = torch.full((b,), float(t_val), device=x_t.device, dtype=torch.float32)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        x1_hat = predict_clean(
            net,
            x_t.to(torch.bfloat16),
            t_tensor,
            txt_emb.expand(b, -1, -1),
            txt_mask.expand(b, -1),
            extra=None,
            checkpoint=False,
        )
    return x1_hat.float().clamp(-1.0, 1.0)


@torch.no_grad()
def run_frozen_flow_methods(
    net,
    txt_emb: torch.Tensor,
    txt_mask: torch.Tensor,
    phi: torch.Tensor,
    y: torch.Tensor,
    sigma_nat: torch.Tensor,
    img_size: int,
    x_w_01: torch.Tensor,
    wiener_stats: dict,
    dev: torch.device,
) -> dict[str, torch.Tensor]:
    """Run PnP-Flow (ICLR'25), ZS-FLOWER (ICLR'26), and PhysFlow-ZS (Ours) using frozen MiniT2I-B/16.

    All three share the IDENTICAL frozen generic MiniT2I-B/16 pixel-space flow prior (ZERO retraining!).
    Args:
        y: [C, M] where C=1 (grayscale, replicated to 3ch for flow net) or C=3 (RGB).
    """
    C = y.shape[0]
    M, N = phi.shape
    c_iy = sigma_nat @ phi.T
    sy = phi @ c_iy
    sy = 0.5 * (sy + sy.T)
    tr_scale = wiener_stats["tr_scale"]
    evals, evecs = torch.linalg.eigh(sy / tr_scale)
    evals = evals.clamp_min(1e-7)
    lam_vec = torch.full((C,), wiener_stats["lam_adapt_mean"], device=dev)

    # Precompute pseudo-inverse for ZS-FLOWER and Lipschitz step for PnP-Flow
    phi_pinv = torch.linalg.pinv(phi, rcond=1e-4)  # [N, M]
    # Operator norm^2
    v_p = torch.randn(N, 1, device=dev)
    for _ in range(12):
        v_p = phi.T @ (phi @ v_p)
        v_p = v_p / v_p.norm().clamp_min(1e-8)
    L_op = float(((phi @ v_p).norm() ** 2).item()) + 1e-6

    def to_512_m11(x_c_hw: torch.Tensor) -> torch.Tensor:
        if x_c_hw.shape[0] == 1:
            x3 = x_c_hw.repeat(3, 1, 1).unsqueeze(0)
        else:
            x3 = x_c_hw.unsqueeze(0)
        return F.interpolate(x3, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0

    def from_512_m11(x512: torch.Tensor) -> torch.Tensor:
        x01 = ((x512.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        xs = F.interpolate(x01, size=(img_size, img_size), mode="area").squeeze(0)  # [3, H, W]
        if C == 1:
            return xs.mean(dim=0, keepdim=True)  # [1, H, W]
        return xs

    gen = torch.Generator(device=dev)
    gen.manual_seed(42)
    z0 = torch.randn((1, 3, 512, 512), device=dev, generator=gen) * NOISE_SCALE

    # 1. PnP-Flow (ICLR 2025): naive l2 gradient step + flow denoiser from t=0 -> 1 (12 steps)
    steps = 12
    x = z0.clone()
    ts = torch.linspace(0.0, 1.0, steps + 1, device=dev)
    for idx_s in range(steps):
        t_v = float(ts[idx_s].item())
        t_next = float(ts[idx_s + 1].item())
        x1_hat = predict_uncond_clean(net, x, t_v, txt_emb, txt_mask)
        xs = from_512_m11(x1_hat).view(C, N)
        grad = ((xs @ phi.T) - y) @ phi / L_op
        xs_upd = (xs - 0.85 * grad).clamp(0.0, 1.0).view(C, img_size, img_size)
        x1_dc = to_512_m11(xs_upd)
        v = (x1_dc - x) / max(1.0 - t_v, 1e-3)
        x = x + (t_next - t_v) * v
    pnp_out = from_512_m11(x).clamp(0.0, 1.0)

    # 2. ZS-FLOWER (ICLR 2026): unregularized exact range-nullspace projection from t=0 -> 1 (12 steps)
    x = z0.clone()
    for idx_s in range(steps):
        t_v = float(ts[idx_s].item())
        t_next = float(ts[idx_s + 1].item())
        x1_hat = predict_uncond_clean(net, x, t_v, txt_emb, txt_mask)
        xs = from_512_m11(x1_hat).view(C, N)
        res = y - (xs @ phi.T)
        xs_dc = (xs + res @ phi_pinv.T).clamp(0.0, 1.0).view(C, img_size, img_size)
        x1_dc = to_512_m11(xs_dc)
        v = (x1_dc - x) / max(1.0 - t_v, 1e-3)
        x = x + (t_next - t_v) * v
    flower_out = from_512_m11(x).clamp(0.0, 1.0)

    # 3. PhysFlow-ZS (Ours): Blind Spectral Kalman-Wiener warm-start (t0=0.60) + Wiener-metric proximal updates!
    t_warm = 0.60
    x_w_512 = to_512_m11(x_w_01)
    x = (1.0 - t_warm) * z0 + t_warm * x_w_512
    ts_p = torch.linspace(t_warm, 1.0, steps + 1, device=dev)
    for idx_s in range(steps):
        t_v = float(ts_p[idx_s].item())
        t_next = float(ts_p[idx_s + 1].item())
        x1_hat = predict_uncond_clean(net, x, t_v, txt_emb, txt_mask)
        xs = from_512_m11(x1_hat).view(C, N)
        xs_dc = kalman_wiener_prox_step(xs, phi, y, c_iy, evals, evecs, tr_scale, lam_vec, eta=0.90).clamp(0.0, 1.0)
        x1_dc = to_512_m11(xs_dc.view(C, img_size, img_size))
        v = (x1_dc - x) / max(1.0 - t_v, 1e-3)
        x = x + (t_next - t_v) * v
    # Final blind Kalman-Wiener proximal polish
    xs_fin = from_512_m11(x).view(C, N)
    xs_fin = kalman_wiener_prox_step(xs_fin, phi, y, c_iy, evals, evecs, tr_scale, lam_vec, eta=0.95).clamp(0.0, 1.0)
    phys_out = xs_fin.view(C, img_size, img_size)

    return {"pnp_flow": pnp_out, "zs_flower": flower_out, "physflow_zs": phys_out}


def compute_metrics(pred_chw: torch.Tensor, gt_chw: torch.Tensor, phi: torch.Tensor, y: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, and physical measurement relative residual on [C, H, W] in [0, 1]."""
    p_m11 = pred_chw.unsqueeze(0) * 2.0 - 1.0
    g_m11 = gt_chw.unsqueeze(0) * 2.0 - 1.0
    if p_m11.shape[1] == 1:
        p_m11 = p_m11.repeat(1, 3, 1, 1)
        g_m11 = g_m11.repeat(1, 3, 1, 1)
    ps = float(calc_psnr(p_m11.cpu(), g_m11.cpu()).item())
    ss = float(calc_ssim(p_m11.cpu(), g_m11.cpu()).item())
    y_hat = pred_chw.view(pred_chw.shape[0], -1) @ phi.T
    rel_res = float(((y_hat - y).norm() / y.norm().clamp_min(1e-8)).item())
    return {"psnr": round(ps, 2), "ssim": round(ss, 4), "rel_res": round(rel_res, 4)}


def load_spihim_scene(scene_name: str, wl_range: tuple[float, float] = (520.0, 700.0)) -> dict:
    """Load real physical CREATIS SPIHIM 64x64 DMD scene in exact experimental acquisition order."""
    sdir = BENCH_DIR / "creatis_spihim" / scene_name
    meta = json.loads(next(sdir.glob("*.json")).read_text(encoding="utf-8"))
    pats = np.array(ast.literal_eval(meta[3]["patterns"]), dtype=int)
    wls = np.array(ast.literal_eval(meta[3]["wavelengths"]), dtype=float)
    raw = np.load(next(sdir.glob("*.npz")))["spectral_data"]  # (8192, 2048)

    # Full-band high-SNR optical ground-truth reference (all 4,096 physical Walsh-Hadamard modes)
    m_gt = (wls >= 520.0) & (wls <= 700.0)
    y_pos_gt = raw[0::2, m_gt].mean(axis=1)
    y_neg_gt = raw[1::2, m_gt].mean(axis=1)
    y_diff_gt = y_pos_gt - y_neg_gt
    modes = pats[0::2] // 2  # Acquisition order!

    W1 = walsh_1d(64)
    H2 = np.kron(W1, W1)  # (4096, 4096) in {-1, +1}
    H_acq = H2[modes]     # (4096, 4096) ordered by actual DMD acquisition sequence!

    x_raw = (H_acq.T @ y_diff_gt) / 4096.0  # Exact inverse of full 4096 real modes
    lo, hi = np.percentile(x_raw, [0.5, 99.5])
    scale = max(float(hi - lo), 1e-8)
    x_gt_01 = np.clip((x_raw - lo) / scale, 0.0, 1.0).reshape(64, 64)

    # Target spectral band measurements (broadband high-SNR OR narrowband 600+/-1nm low-SNR)
    m_band = (wls >= wl_range[0]) & (wls <= wl_range[1])
    y_pos_b = raw[0::2, m_band].mean(axis=1)
    y_neg_b = raw[1::2, m_band].mean(axis=1)
    # Match band gain to broadband DC energy (so scale calibration is purely 1 scalar exposure factor!)
    dc_ratio = float(y_diff_gt[0] / max(float((y_pos_b - y_neg_b)[0]), 1e-8))
    y_pos_cal = y_pos_b * dc_ratio
    y_neg_cal = y_neg_b * dc_ratio
    # Transform measurements linearly to normalized [0, 1] scene units: y_norm = (y - lo * sum(phi)) / scale
    # For differential H_acq: sum(row 0) = 4096, sum(row k>0) = 0
    row_sum_diff = H_acq.sum(axis=1)
    y_diff_norm = ((y_pos_cal - y_neg_cal) - lo * row_sum_diff) / scale
    # For unipolar binary DMD H_pos = (1 + H_acq) / 2 in {0, 1}:
    H_pos = 0.5 * (1.0 + H_acq)
    # Remove uniform dark-count floor estimated from mean(y_pos + y_neg) vs H_pos(1)*x_raw
    dark_offset = float(np.median(0.5 * (y_pos_cal + y_neg_cal) - 0.5 * y_diff_gt[0]))
    row_sum_pos = H_pos.sum(axis=1)
    y_pos_norm = ((y_pos_cal - dark_offset) - lo * row_sum_pos) / scale

    return {
        "name": scene_name,
        "x_gt_01": x_gt_01.astype(np.float32),
        "H_diff": H_acq.astype(np.float32),
        "y_diff": y_diff_norm.astype(np.float32),
        "H_pos": H_pos.astype(np.float32),
        "y_pos": y_pos_norm.astype(np.float32),
        "integration_ms": float(meta[2].get("integration_time_ms", 0.0)),
        "n_wl_bins": int(m_band.sum()),
    }


def load_onepix_rgb_scene() -> dict:
    """Load real physical CREATIS 2026 ONE-PIX 32x32x3 RGB ColorChecker DMD+Spectrometer dataset."""
    odir = BENCH_DIR / "onepix_2026"
    spectra = np.load(odir / "spectra.npy")          # (2048, 1289)
    pats = np.load(odir / "patterns_order.npy")      # ['Hpos(0,0)', 'Hneg(0,0)', ...]
    wls = np.load(odir / "wavelengths.npy")          # (1289,)

    # Visible RGB wavelength bands
    r_m = (wls >= 600.0) & (wls <= 680.0)
    g_m = (wls >= 515.0) & (wls <= 575.0)
    b_m = (wls >= 435.0) & (wls <= 495.0)
    rgb_spec = np.stack([spectra[:, r_m].mean(1), spectra[:, g_m].mean(1), spectra[:, b_m].mean(1)], axis=0)  # [3, 2048]

    W1 = walsh_1d(32)
    pat_re = re.compile(r"H(pos|neg)\((\d+),(\d+)\)")
    pos_map, neg_map = {}, {}
    order_uv = []
    for i, s in enumerate(pats):
        m = pat_re.match(str(s))
        assert m, f"Unexpected pattern label: {s}"
        sign, u, v = m.group(1), int(m.group(2)), int(m.group(3))
        if sign == "pos":
            pos_map[(u, v)] = rgb_spec[:, i]
            order_uv.append((u, v))
        else:
            neg_map[(u, v)] = rgb_spec[:, i]

    # Sort modes by sequency u + v (standard low-to-high spatial frequency acquisition order!)
    order_seq = sorted(order_uv, key=lambda uv: (uv[0] + uv[1], max(uv[0], uv[1]), uv[0]))
    H_list, y_list = [], []
    for u, v in order_seq:
        H_list.append(np.outer(W1[u], W1[v]).reshape(-1))
        y_list.append(pos_map[(u, v)] - neg_map[(u, v)])
    H_seq = np.stack(H_list, axis=0).astype(np.float64)    # [1024, 1024]
    y_rgb = np.stack(y_list, axis=1).astype(np.float64)    # [3, 1024]

    # Full 1,024-mode optical ground-truth RGB reference
    x_raw = (y_rgb @ H_seq) / 1024.0  # [3, 1024]
    x_gt = np.zeros((3, 32, 32), dtype=np.float32)
    y_norm = np.zeros_like(y_rgb, dtype=np.float32)
    row_sum = H_seq.sum(axis=1)
    for c in range(3):
        lo, hi = np.percentile(x_raw[c], [1.0, 99.0])
        sc = max(float(hi - lo), 1e-8)
        x_gt[c] = np.clip((x_raw[c] - lo) / sc, 0.0, 1.0).reshape(32, 32)
        y_norm[c] = ((y_rgb[c] - lo * row_sum) / sc).astype(np.float32)

    return {
        "name": "onepix_2026_colorchecker_rgb",
        "x_gt_01": x_gt,
        "H_diff": H_seq.astype(np.float32),
        "y_diff": y_norm,
    }


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    log("=== Knife 2: Loading Frozen Generic MiniT2I-B/16 Flow Prior (ZERO Face LoRA, ZERO Retraining) ===")
    txt_emb, txt_mask = encode_neutral_optical_prompt(dev, prompt="a sharp high-contrast optical image")
    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev).eval()
    net = wrapper.model.net

    sigma_64 = build_stationary_cov(64, corr_len=2.0).to(dev)
    sigma_32 = build_stationary_cov(32, corr_len=1.8).to(dev)

    scenes = ("zoom_x12_usaf_group5", "zoom_x12_starsector", "zoom_x12_tomato_slice_2", "zoom_x2_tomato_slice_2")
    out = {
        "broadband_diff": {},
        "narrowband_low_snr": {},
        "ultranarrow_very_low_snr": {},
        "unipolar_binary_dmd": {},
        "onepix_rgb": {},
    }
    saved_tiles = {}

    # ------------------------------------------------------------------
    # Protocol A: Broadband High-SNR Differential DMD at SR = 6.25%, 12.5%, 25.0%
    # ------------------------------------------------------------------
    log("=== Knife 2A: CREATIS SPIHIM Broadband High-SNR (520-700 nm) on 4 Real Optical Scenes ===")
    for rate in (256, 512, 1024):
        sr_pct = round(100.0 * rate / 4096.0, 2)
        out["broadband_diff"][str(rate)] = {"sr_pct": sr_pct, "per_scene": {}, "mean": {}}
        agg = {m: [] for m in ("pinv", "tv_fista", "pnp_flow", "zs_flower", "blind_wiener", "physflow_zs")}
        lams = []
        for sname in scenes:
            sc = load_spihim_scene(sname, wl_range=(520.0, 700.0))
            gt = torch.from_numpy(sc["x_gt_01"]).unsqueeze(0).to(dev)  # [1, 64, 64]
            phi = torch.from_numpy(sc["H_diff"][:rate]).to(dev)        # [M, 4096]
            y = torch.from_numpy(sc["y_diff"][:rate]).unsqueeze(0).to(dev)  # [1, M]

            # 1. Pinv (Inverse Hadamard)
            x_pinv = (y @ phi / 4096.0).clamp(0.0, 1.0).view(1, 64, 64)
            # 2. TV-FISTA
            x_tv = tv_fista_solve(phi, y, 64, lam_tv=0.010, n_iters=120)
            # 3. Blind Spectral Wiener (Ours)
            x_w_vec, w_stats = blind_spectral_wiener_solve(phi, y, sigma_64, x_mean=0.5)
            x_w = x_w_vec.clamp(0.0, 1.0).view(1, 64, 64)
            lams.append(w_stats["lam_adapt_mean"])
            # 4-6. Frozen MiniT2I-B/16 flow methods (PnP-Flow, ZS-FLOWER, PhysFlow-ZS)
            flow_res = run_frozen_flow_methods(net, txt_emb, txt_mask, phi, y, sigma_64, 64, x_w, w_stats, dev)

            recs = {
                "pinv": x_pinv,
                "tv_fista": x_tv,
                "pnp_flow": flow_res["pnp_flow"],
                "zs_flower": flow_res["zs_flower"],
                "blind_wiener": x_w,
                "physflow_zs": flow_res["physflow_zs"],
            }
            sc_dict = {"lambda_adapt": round(w_stats["lam_adapt_mean"], 5)}
            for m, img_m in recs.items():
                met = compute_metrics(img_m, gt, phi, y)
                sc_dict[m] = met
                agg[m].append(met)
            out["broadband_diff"][str(rate)]["per_scene"][sname] = sc_dict
            if rate == 512:
                saved_tiles[sname] = {k: v.cpu() for k, v in [("gt", gt)] + list(recs.items())}
            log(
                f"  [Broadband M={rate:4d} ({sr_pct:5.2f}%) {sname:24s}] "
                f"Pinv={sc_dict['pinv']['psnr']:5.2f} | TV={sc_dict['tv_fista']['psnr']:5.2f} | "
                f"PnP={sc_dict['pnp_flow']['psnr']:5.2f} | FLW={sc_dict['zs_flower']['psnr']:5.2f} | "
                f"Wiener={sc_dict['blind_wiener']['psnr']:5.2f} | PhysFlow={sc_dict['physflow_zs']['psnr']:5.2f} "
                f"(lam={w_stats['lam_adapt_mean']:.4f})"
            )
        for m in agg:
            out["broadband_diff"][str(rate)]["mean"][m] = {
                "psnr": round(float(np.mean([d["psnr"] for d in agg[m]])), 2),
                "ssim": round(float(np.mean([d["ssim"] for d in agg[m]])), 4),
                "rel_res": round(float(np.mean([d["rel_res"] for d in agg[m]])), 4),
            }
        out["broadband_diff"][str(rate)]["mean_lambda_adapt"] = round(float(np.mean(lams)), 5)
        mm = out["broadband_diff"][str(rate)]["mean"]
        log(
            f"  >> MEAN M={rate} ({sr_pct}%): Pinv={mm['pinv']['psnr']:.2f} | TV={mm['tv_fista']['psnr']:.2f} | "
            f"PnP={mm['pnp_flow']['psnr']:.2f} | FLW={mm['zs_flower']['psnr']:.2f} | "
            f"Wiener={mm['blind_wiener']['psnr']:.2f} | PhysFlow={mm['physflow_zs']['psnr']:.2f} (lam={np.mean(lams):.4f})"
        )

    # ------------------------------------------------------------------
    # Protocol B: Narrowband & Ultra-Narrowband Low-Photon-Flux Real Shot Noise at SR = 12.5% (M=512)
    # ------------------------------------------------------------------
    for p_key, wl_r, p_title in (
        ("narrowband_low_snr", (599.0, 601.0), "Low-SNR 600+/-1nm (16 bins)"),
        ("ultranarrow_very_low_snr", (600.0, 600.3), "Very-Low-SNR 600.0nm (2 bins)"),
    ):
        log(f"=== Knife 2B: {p_title} Real Spectrometer Shot Noise, M=512 ===")
        rate = 512
        agg_nb = {m: [] for m in ("pinv", "tv_fista", "pnp_flow", "zs_flower", "blind_wiener", "physflow_zs")}
        lams_nb = []
        out[p_key] = {"wl_nm": list(wl_r), "rate": rate, "per_scene": {}, "mean": {}}
        for sname in scenes:
            sc = load_spihim_scene(sname, wl_range=wl_r)
            gt = torch.from_numpy(sc["x_gt_01"]).unsqueeze(0).to(dev)
            phi = torch.from_numpy(sc["H_diff"][:rate]).to(dev)
            y = torch.from_numpy(sc["y_diff"][:rate]).unsqueeze(0).to(dev)
            x_pinv = (y @ phi / 4096.0).clamp(0.0, 1.0).view(1, 64, 64)
            x_tv = tv_fista_solve(phi, y, 64, lam_tv=0.015, n_iters=120)
            x_w_vec, w_stats = blind_spectral_wiener_solve(phi, y, sigma_64, x_mean=0.5)
            x_w = x_w_vec.clamp(0.0, 1.0).view(1, 64, 64)
            lams_nb.append(w_stats["lam_adapt_mean"])
            flow_res = run_frozen_flow_methods(net, txt_emb, txt_mask, phi, y, sigma_64, 64, x_w, w_stats, dev)
            recs = {"pinv": x_pinv, "tv_fista": x_tv, "pnp_flow": flow_res["pnp_flow"], "zs_flower": flow_res["zs_flower"], "blind_wiener": x_w, "physflow_zs": flow_res["physflow_zs"]}
            sc_dict = {"lambda_adapt": round(w_stats["lam_adapt_mean"], 5), "n_wl_bins": sc["n_wl_bins"]}
            for m, img_m in recs.items():
                met = compute_metrics(img_m, gt, phi, y)
                sc_dict[m] = met
                agg_nb[m].append(met)
            out[p_key]["per_scene"][sname] = sc_dict
            log(
                f"  [{p_title:29s} {sname:24s}] Pinv={sc_dict['pinv']['psnr']:5.2f} | "
                f"FLW={sc_dict['zs_flower']['psnr']:5.2f} | Wiener={sc_dict['blind_wiener']['psnr']:5.2f} | "
                f"PhysFlow={sc_dict['physflow_zs']['psnr']:5.2f} (lam={w_stats['lam_adapt_mean']:.4f})"
            )
        for m in agg_nb:
            out[p_key]["mean"][m] = {
                "psnr": round(float(np.mean([d["psnr"] for d in agg_nb[m]])), 2),
                "ssim": round(float(np.mean([d["ssim"] for d in agg_nb[m]])), 4),
                "rel_res": round(float(np.mean([d["rel_res"] for d in agg_nb[m]])), 4),
            }
        out[p_key]["mean_lambda_adapt"] = round(float(np.mean(lams_nb)), 5)

    # ------------------------------------------------------------------
    # Protocol C: Single-Exposure Unipolar {0,1} Binary DMD (H_pos ONLY, 2x faster hardware capture!)
    # ------------------------------------------------------------------
    log("=== Knife 2C: Single-Exposure Unipolar {0,1} Binary DMD (H_pos ONLY, M=512) ===")
    agg_u = {m: [] for m in ("pinv", "tv_fista", "pnp_flow", "zs_flower", "blind_wiener", "physflow_zs")}
    out["unipolar_binary_dmd"] = {"rate": 512, "per_scene": {}, "mean": {}}
    for sname in scenes:
        sc = load_spihim_scene(sname, wl_range=(520.0, 700.0))
        gt = torch.from_numpy(sc["x_gt_01"]).unsqueeze(0).to(dev)
        phi = torch.from_numpy(sc["H_pos"][:512]).to(dev)  # True unipolar {0, 1} binary DMD matrix!
        y = torch.from_numpy(sc["y_pos"][:512]).unsqueeze(0).to(dev)
        phi_pinv = torch.linalg.pinv(phi, rcond=1e-4)
        x_pinv = (y @ phi_pinv.T).clamp(0.0, 1.0).view(1, 64, 64)
        x_tv = tv_fista_solve(phi, y, 64, lam_tv=0.012, n_iters=120)
        x_w_vec, w_stats = blind_spectral_wiener_solve(phi, y, sigma_64, x_mean=0.5)
        x_w = x_w_vec.clamp(0.0, 1.0).view(1, 64, 64)
        flow_res = run_frozen_flow_methods(net, txt_emb, txt_mask, phi, y, sigma_64, 64, x_w, w_stats, dev)
        recs = {"pinv": x_pinv, "tv_fista": x_tv, "pnp_flow": flow_res["pnp_flow"], "zs_flower": flow_res["zs_flower"], "blind_wiener": x_w, "physflow_zs": flow_res["physflow_zs"]}
        sc_dict = {"lambda_adapt": round(w_stats["lam_adapt_mean"], 5)}
        for m, img_m in recs.items():
            met = compute_metrics(img_m, gt, phi, y)
            sc_dict[m] = met
            agg_u[m].append(met)
        out["unipolar_binary_dmd"]["per_scene"][sname] = sc_dict
        log(
            f"  [Unipolar {{0,1}} M=512 {sname:24s}] Pinv={sc_dict['pinv']['psnr']:5.2f} | "
            f"TV={sc_dict['tv_fista']['psnr']:5.2f} | FLW={sc_dict['zs_flower']['psnr']:5.2f} | "
            f"Wiener={sc_dict['blind_wiener']['psnr']:5.2f} | PhysFlow={sc_dict['physflow_zs']['psnr']:5.2f}"
        )
    for m in agg_u:
        out["unipolar_binary_dmd"]["mean"][m] = {
            "psnr": round(float(np.mean([d["psnr"] for d in agg_u[m]])), 2),
            "ssim": round(float(np.mean([d["ssim"] for d in agg_u[m]])), 4),
            "rel_res": round(float(np.mean([d["rel_res"] for d in agg_u[m]])), 4),
        }

    # ------------------------------------------------------------------
    # Protocol D: CREATIS 2026 ONE-PIX 32x32x3 RGB ColorChecker Real Optical Bench
    # ------------------------------------------------------------------
    log("=== Knife 2D: CREATIS 2026 ONE-PIX 32x32x3 RGB ColorChecker ===")
    sc_rgb = load_onepix_rgb_scene()
    gt_rgb = torch.from_numpy(sc_rgb["x_gt_01"]).to(dev)  # [3, 32, 32]
    for rate in (128, 256):
        sr_pct = round(100.0 * rate / 1024.0, 2)
        phi = torch.from_numpy(sc_rgb["H_diff"][:rate]).to(dev)  # [M, 1024]
        y = torch.from_numpy(sc_rgb["y_diff"][:, :rate]).to(dev)  # [3, M]
        x_pinv = (y @ phi / 1024.0).clamp(0.0, 1.0).view(3, 32, 32)
        x_tv = tv_fista_solve(phi, y, 32, lam_tv=0.010, n_iters=120)
        x_w_vec, w_stats = blind_spectral_wiener_solve(phi, y, sigma_32, x_mean=0.5)
        x_w = x_w_vec.clamp(0.0, 1.0).view(3, 32, 32)
        flow_res = run_frozen_flow_methods(net, txt_emb, txt_mask, phi, y, sigma_32, 32, x_w, w_stats, dev)
        recs = {"pinv": x_pinv, "tv_fista": x_tv, "pnp_flow": flow_res["pnp_flow"], "zs_flower": flow_res["zs_flower"], "blind_wiener": x_w, "physflow_zs": flow_res["physflow_zs"]}
        rdict = {"sr_pct": sr_pct, "lambda_adapt": round(w_stats["lam_adapt_mean"], 5)}
        for m, img_m in recs.items():
            rdict[m] = compute_metrics(img_m, gt_rgb, phi, y)
        out["onepix_rgb"][str(rate)] = rdict
        if rate == 128:
            saved_tiles["onepix_rgb_128"] = {k: v.cpu() for k, v in [("gt", gt_rgb)] + list(recs.items())}
        log(
            f"  [ONE-PIX RGB M={rate:3d} ({sr_pct:5.2f}%)] Pinv={rdict['pinv']['psnr']:5.2f} | "
            f"TV={rdict['tv_fista']['psnr']:5.2f} | PnP={rdict['pnp_flow']['psnr']:5.2f} | "
            f"FLW={rdict['zs_flower']['psnr']:5.2f} | Wiener={rdict['blind_wiener']['psnr']:5.2f} | "
            f"PhysFlow={rdict['physflow_zs']['psnr']:5.2f}"
        )

    # Save visual QA mosaic (5 rows x 7 cols: GT | Pinv | TV-FISTA | PnP-Flow | ZS-FLOWER | Blind Wiener | PhysFlow-ZS)
    col_order = ["gt", "pinv", "tv_fista", "pnp_flow", "zs_flower", "blind_wiener", "physflow_zs"]
    row_keys = ["zoom_x12_usaf_group5", "zoom_x12_starsector", "zoom_x12_tomato_slice_2", "zoom_x2_tomato_slice_2", "onepix_rgb_128"]
    S = 192
    canvas = Image.new("RGB", (S * len(col_order), S * len(row_keys)))
    for r, rk in enumerate(row_keys):
        for c, ck in enumerate(col_order):
            t = saved_tiles[rk][ck].float().clamp(0.0, 1.0)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
            im = Image.fromarray(arr).resize((S, S), Image.NEAREST)
            canvas.paste(im, (c * S, r * S))
    canvas.save(V4_DIR / "knife2_real_dmd_qa.png")
    torch.save(saved_tiles, V4_DIR / "knife2_real_dmd_tiles.pt")
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"KNIFE 2 COMPLETE -> {OUT_JSON}")


if __name__ == "__main__":
    main()
