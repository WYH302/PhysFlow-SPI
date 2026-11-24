"""Unit tests for V4 exact linear operator pullback and DC-invariant Spectral GCV Kalman-Wiener."""
from __future__ import annotations

import torch

from src.optical.v4_primitives import (
    blind_spectral_wiener_solve,
    build_stationary_cov,
    compute_pulled_back_phi_64,
    warp_canon_to_scene,
)


def test_exact_linear_operator_pullback_identity() -> None:
    """Verify <Phi_tilde(p), I_canon> == <Phi, T_p(I_canon)> to machine precision (< 1e-5)."""
    torch.manual_seed(42)
    phi_64 = torch.randn(32, 3, 64, 64)
    pose = torch.tensor([[0.11, -0.08, 0.06, -0.14]])
    img_canon = torch.rand(1, 3, 64, 64)

    img_scene = warp_canon_to_scene(img_canon, pose)
    y_fwd = (phi_64 * img_scene).sum(dim=(1, 2, 3))

    phi_tilde = compute_pulled_back_phi_64(phi_64, pose)
    y_pullback = phi_tilde @ img_canon.reshape(-1)

    rel_err = float(((y_fwd - y_pullback).norm() / y_fwd.norm().clamp_min(1e-8)).item())
    assert rel_err < 1e-5, f"Operator pullback error {rel_err:.2e} exceeded 1e-5"


def test_dc_invariant_spectral_gcv_unipolar_dmd() -> None:
    """Verify DC-invariant Spectral GCV Kalman-Wiener handles rank-1 DC spikes on unipolar {0,1} DMD."""
    torch.manual_seed(7)
    size = 16
    n = size * size
    m = 64
    # Create unipolar {0,1} binary measurement matrix with explicit all-ones DC row 0
    phi = torch.randint(0, 2, (m, n), dtype=torch.float32)
    phi[0] = 1.0
    sigma_nat = build_stationary_cov(size, corr_len=2.0)

    # Smooth synthetic target image in [0, 1]
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    x_true = (0.35 + 0.30 * torch.sin(3.14 * xx) * torch.cos(3.14 * yy)).reshape(1, n)
    y_clean = x_true @ phi.T
    y_noisy = y_clean + 0.8 * torch.randn_like(y_clean)

    x_clean, st_clean = blind_spectral_wiener_solve(phi, y_clean, sigma_nat, x_mean=0.5)
    x_noisy, st_noisy = blind_spectral_wiener_solve(phi, y_noisy, sigma_nat, x_mean=0.5)

    assert torch.isfinite(x_clean).all() and torch.isfinite(x_noisy).all()
    assert st_noisy["lam_adapt_mean"] >= st_clean["lam_adapt_mean"]
    mse_clean = float((x_clean.clamp(0.0, 1.0) - x_true).square().mean().item())
    assert mse_clean < 0.02, f"Unexpectedly high unipolar MSE: {mse_clean:.4f}"
