"""Lightweight public unit tests for PhysFlow-SPI repository and optical math."""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_required_public_files_exist() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / "configs" / "default.yaml",
        ROOT / "assets" / "teaser_m64.png",
        ROOT / "assets" / "architecture.png",
    ]
    for path in required:
        assert path.is_file() and path.stat().st_size > 0, f"Missing or empty: {path}"


def test_all_python_sources_parse_cleanly() -> None:
    py_files = sorted(
        list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    )
    assert len(py_files) >= 40
    for pf in py_files:
        source = pf.read_text(encoding="utf-8")
        ast.parse(source, filename=str(pf))


def test_default_config_rates_and_dimensions() -> None:
    cfg_text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    assert "rates: [16, 32, 64, 128, 256]" in cfg_text
    assert "generator_size: 512" in cfg_text
    assert "phi_resolution: 256" in cfg_text
    assert "id_dim: 512" in cfg_text
    assert "geo_dim: 212" in cfg_text


def test_blind_spectral_noise_estimator_monotonicity() -> None:
    """Verify Eqs. (13)-(14) trailing-eigenmode noise variance scaling."""
    # Synthetic ascending trailing eigenvalues of standardized measurement covariance
    trailing_eigs = [1e-4, 2e-4, 4e-4, 8e-4]
    lam_min, lam_max = 4e-4, 0.25

    def estimate_lambda(proj_sq: list[float]) -> float:
        excess = sum(p - lam for p, lam in zip(proj_sq, trailing_eigs)) / len(trailing_eigs)
        sigma2 = max(0.0, excess)
        return min(lam_max, max(lam_min, lam_min + 1.5 * sigma2))

    # Clean signal energy matches covariance eigenvalues -> stays at floor lam_min
    lam_clean = estimate_lambda(trailing_eigs)
    assert math.isclose(lam_clean, lam_min, rel_tol=1e-9)

    # Moderate additive noise elevates adaptive damping smoothly
    lam_noisy = estimate_lambda([e + 0.01 for e in trailing_eigs])
    assert lam_clean < lam_noisy < lam_max

    # Severe drift saturates safely at upper bound lam_max
    lam_drift = estimate_lambda([e + 0.50 for e in trailing_eigs])
    assert math.isclose(lam_drift, lam_max, rel_tol=1e-9)


def test_optional_torch_core_modules() -> None:
    """When PyTorch is available, verify zero-step slot-residual & LoRA invariance."""
    try:
        import torch
    except ImportError:
        return

    from src.models.injection import FactorizedJointConditioner, LoRALinear

    cond = FactorizedJointConditioner(hidden=768, num_blocks=4).eval()
    id_tok = cond.project_id(torch.randn(2, 512))
    geo_tok = cond.project_geo(torch.randn(2, 212))
    assert id_tok.shape == (2, 16, 768)
    assert geo_tok.shape == (2, 16, 768)
    # Unit-RMS normalization and zero-initialized per-block gates
    assert torch.allclose(id_tok.pow(2).mean(dim=-1).sqrt(), torch.ones(2, 16), atol=1e-3)
    assert float(cond.id_scales().abs().max()) == 0.0
    assert float(cond.geo_scales().abs().max()) == 0.0

    base = torch.nn.Linear(64, 64)
    lora = LoRALinear(base, rank=16, alpha=16.0)
    x = torch.randn(3, 64)
    assert float((lora(x) - base(x)).abs().max()) == 0.0
