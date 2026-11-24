"""Pure-PyTorch optical physics primitives for V4:
  1. DC-invariant Spectral GCV Kalman-Wiener proximal lifting (real physical DMD hardware).
  2. Exact algebraic linear operator pullback via autograd VJP through F.grid_sample (unaligned FOV).
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def build_stationary_cov(size: int, corr_len: float = 2.0) -> torch.Tensor:
    """Closed-form 2D stationary Ornstein-Uhlenbeck spatial covariance prior."""
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pts = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
    dists = torch.cdist(pts, pts, p=2.0)
    return torch.exp(-dists / float(corr_len))


def blind_spectral_wiener_solve(
    phi: torch.Tensor,
    y: torch.Tensor,
    sigma_nat: torch.Tensor,
    x_mean: float = 0.5,
    base_lam: float = 1e-4,
    max_lam: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Closed-form DC-invariant Spectral GCV Kalman-Wiener proximal lifting."""
    m, _n = phi.shape
    c_iy = sigma_nat @ phi.T
    sy = phi @ c_iy
    sy = 0.5 * (sy + sy.T)
    evals_raw, evecs = torch.linalg.eigh(sy)
    tr_scale = float(evals_raw.median().clamp_min(1e-8).item())
    evals = (evals_raw / tr_scale).clamp_min(1e-7)

    y0 = phi.sum(dim=1, keepdim=True).T * float(x_mean)
    dy = y - y0
    proj = dy @ evecs
    proj_sq = proj.square()

    is_unip = float(evals[-1] / evals[-2].clamp_min(1e-6)) > 15.0
    sl = slice(0, m - 1) if is_unip else slice(0, m)
    ev_g = evals[sl].view(1, 1, -1)
    ps_g = proj_sq[:, sl].unsqueeze(1)
    grid = torch.logspace(math.log10(base_lam), math.log10(max_lam), 64, device=phi.device)
    lm = grid.view(1, -1, 1)
    r = lm / (ev_g + lm)
    gcv = (r.square() * ps_g).mean(dim=-1) / r.mean(dim=-1).square().clamp_min(1e-10)
    lam_adapt = grid[gcv.argmin(dim=-1)]

    denom = tr_scale * (evals.unsqueeze(0) + lam_adapt.unsqueeze(1))
    coef = (proj / denom) @ evecs.T
    x_w = float(x_mean) + coef @ c_iy.T
    return x_w, {"lam_adapt_mean": float(lam_adapt.mean().item()), "tr_scale": tr_scale}


def pose_to_inv_affine(pose: torch.Tensor) -> torch.Tensor:
    """Convert 4-DoF pose [B, 4] = (tx, ty, log_s, theta_rad) to inverse affine grid matrix."""
    tx, ty, inv_s, th = pose[:, 0], pose[:, 1], torch.exp(-pose[:, 2]), pose[:, 3]
    c, s = torch.cos(th), torch.sin(th)
    a00, a01, a10, a11 = inv_s * c, inv_s * s, -inv_s * s, inv_s * c
    b0 = -(a00 * tx + a01 * ty)
    b1 = -(a10 * tx + a11 * ty)
    return torch.stack([torch.stack([a00, a01, b0], dim=-1), torch.stack([a10, a11, b1], dim=-1)], dim=1)


def warp_canon_to_scene(img_canon: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Border-clamped similarity warp T_p(I_canon) without black-corner artifacts."""
    theta = pose_to_inv_affine(pose).to(dtype=img_canon.dtype, device=img_canon.device)
    grid = F.affine_grid(theta, size=img_canon.shape, align_corners=False)
    return F.grid_sample(img_canon, grid, mode="bilinear", padding_mode="border", align_corners=False)


def compute_pulled_back_phi_64(phi_64: torch.Tensor, pose_1: torch.Tensor) -> torch.Tensor:
    """Exact algebraic adjoint Phi_tilde(p) = W_p^T Phi_S via vectorized autograd VJP."""
    m = phi_64.shape[0]
    with torch.enable_grad():
        x_dummy = torch.zeros((m, 3, 64, 64), device=phi_64.device, dtype=torch.float32, requires_grad=True)
        theta = pose_to_inv_affine(pose_1.detach()).expand(m, -1, -1)
        grid = F.affine_grid(theta, size=x_dummy.shape, align_corners=False)
        warped = F.grid_sample(x_dummy, grid, mode="bilinear", padding_mode="border", align_corners=False)
        loss = (warped * phi_64).sum()
        (grad_x,) = torch.autograd.grad(loss, x_dummy, create_graph=False)
    return grad_x.detach().reshape(m, -1)
