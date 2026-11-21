from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


RATES = (16, 32, 64, 128, 256)


class SPIOperator(nn.Module):
    """Frozen single-pixel operator matching LSI's sum-of-RGB measurement.

    Official masking.py measures a [0, 1] image as
    y = (R + G + B) @ W, with W binary of shape (H*W, M) after quantization.
    This module stores Phi as (M, H*W) so y = gray @ Phi.T.
    A 512 generator image is differentiably downsampled to Phi's resolution.
    """

    def __init__(
        self,
        phi: torch.Tensor,
        resolution: int | None = None,
        calib_a: torch.Tensor | None = None,
        calib_b: torch.Tensor | None = None,
    ):
        super().__init__()
        if phi.ndim == 3:
            resolution = int(phi.shape[-1])
            phi = phi.flatten(1)
        if phi.ndim != 2:
            raise ValueError(f"Phi must be (M, N) or (M, H, W), got {tuple(phi.shape)}")
        measurements, pixels = int(phi.shape[0]), int(phi.shape[1])
        if resolution is None:
            side = int(math.isqrt(pixels))
            if side * side != pixels:
                raise ValueError(f"Phi pixel count {pixels} is not a square")
            resolution = side
        if resolution * resolution != pixels:
            raise ValueError(f"resolution {resolution} does not match N={pixels}")
        self.resolution = int(resolution)
        self.n_measurements = measurements
        self.register_buffer("phi", phi.float().contiguous())
        a = torch.ones(measurements) if calib_a is None else calib_a.float().reshape(-1)
        b = torch.zeros(measurements) if calib_b is None else calib_b.float().reshape(-1)
        if a.numel() != measurements or b.numel() != measurements:
            raise ValueError("calibration vectors must have one entry per measurement")
        self.register_buffer("calib_a", a.contiguous())
        self.register_buffer("calib_b", b.contiguous())

    def sample_indices(self, rate: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if rate > self.n_measurements:
            raise ValueError(f"rate {rate} exceeds Phi rows {self.n_measurements}")
        return torch.randperm(self.n_measurements, generator=generator, device=self.phi.device)[:rate]

    def to_unit_interval(self, image: torch.Tensor, input_range: str) -> torch.Tensor:
        if input_range == "-1_1":
            return image * 0.5 + 0.5
        if input_range == "0_1":
            return image
        raise ValueError(input_range)

    def from_unit_interval(self, image: torch.Tensor, input_range: str) -> torch.Tensor:
        if input_range == "-1_1":
            return image * 2.0 - 1.0
        if input_range == "0_1":
            return image
        raise ValueError(input_range)

    def gray(self, image: torch.Tensor, input_range: str = "-1_1") -> torch.Tensor:
        """Differentiable downsample and RGB sum. Returns (B, H*W) in the Phi frame."""
        x = self.to_unit_interval(image, input_range)
        if x.shape[-2] != self.resolution or x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            gray = x[:, 0]
        elif x.shape[1] == 3:
            gray = x[:, 0] + x[:, 1] + x[:, 2]
        else:
            raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")
        return gray.flatten(1)

    def _phi_rows(self, indices: torch.Tensor | None) -> torch.Tensor:
        if indices is None:
            return self.phi
        return self.phi.index_select(0, indices.to(self.phi.device))

    def calibrate(self, y: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
        if indices is None:
            return self.calib_a * y + self.calib_b
        a = self.calib_a.index_select(0, indices.to(self.calib_a.device))
        b = self.calib_b.index_select(0, indices.to(self.calib_b.device))
        return a * y + b

    def measure(
        self,
        image: torch.Tensor,
        indices: torch.Tensor | None = None,
        input_range: str = "-1_1",
    ) -> torch.Tensor:
        gray = self.gray(image, input_range=input_range)
        phi = self._phi_rows(indices)
        y = gray @ phi.transpose(0, 1).to(dtype=gray.dtype)
        return self.calibrate(y, indices)

    def backproject(
        self,
        y: torch.Tensor,
        indices: torch.Tensor | None = None,
        lam: float = 1e-2,
    ) -> torch.Tensor:
        """Tikhonov image I_bp = Phi^T (Phi Phi^T + lam I)^{-1} y.

        y is the calibrated measurement. The inverse calibration is applied
        before the spatial adjoint so I_bp lives in the sum-of-RGB domain.
        """
        with torch.amp.autocast(device_type=self.phi.device.type, enabled=False):
            phi = self._phi_rows(indices).float()
            if indices is None:
                a, b = self.calib_a.float(), self.calib_b.float()
            else:
                idx = indices.to(self.calib_a.device)
                a = self.calib_a.index_select(0, idx).float()
                b = self.calib_b.index_select(0, idx).float()
            y_raw = (y.float() - b) / a.clamp_min(1e-8)
            gram = phi @ phi.transpose(0, 1)
            diag_scale = gram.diagonal().mean().clamp_min(1.0)
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.float32)
            coef = torch.linalg.solve(gram + float(lam) * diag_scale * eye, y_raw.transpose(0, 1)).transpose(0, 1)
            image = coef @ phi
            return image.view(y.shape[0], 1, self.resolution, self.resolution)

    def proximal(
        self,
        image: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor | None = None,
        lam: float = 1e-2,
        eta: float = 1.0,
        input_range: str = "-1_1",
    ) -> torch.Tensor:
        """One closed-form measurement projection of a clean RGB prediction."""
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            x01 = self.to_unit_interval(image.float(), input_range)
            low = x01
            if x01.shape[-1] != self.resolution or x01.shape[-2] != self.resolution:
                low = F.interpolate(x01, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
            channel_scale = float(low.shape[1])
            y_hat = self.measure(image.float(), indices=indices, input_range=input_range)
            resid = y_hat.float() - y.float()
            phi = self._phi_rows(indices).float()
            if indices is None:
                a = self.calib_a.float()
            else:
                a = self.calib_a.index_select(0, indices.to(self.calib_a.device)).float()
            resid_raw = resid / a.clamp_min(1e-8)
            gram = channel_scale * (phi @ phi.transpose(0, 1))
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.float32)
            coef = torch.linalg.solve(gram + float(lam) * eye, resid_raw.transpose(0, 1)).transpose(0, 1)
            corr = (coef @ phi).view(image.shape[0], 1, self.resolution, self.resolution)
            if corr.shape[-2:] != x01.shape[-2:]:
                corr = F.interpolate(corr, size=x01.shape[-2:], mode="bilinear", align_corners=False)
            updated = (x01 - float(eta) * corr).clamp(0.0, 1.0)
            return self.from_unit_interval(updated, input_range).to(dtype=image.dtype)

    @classmethod
    def from_simulation_256(cls, path: str | Path | None = None, bits: int = 1) -> "SPIOperator":
        """Load simulation_256.pt and store the official bits=1 mask.

        The raw checkpoint tensor is float, not {0, 1}. ``binarize_lsi_weight``
        applies masking.py's clamp-and-round quantizer. Construction asserts
        that a [0, 1] RGB measurement matches sum-of-RGB then mask multiply.
        """
        from src.optical.phi_loader import SIMULATION_256, assert_official_phi, read_checkpoint_weight

        ckpt = Path(path) if path is not None else SIMULATION_256
        raw, key = read_checkpoint_weight(ckpt)
        report = assert_official_phi(raw, bits=bits)
        phi = report.pop("phi")
        operator = cls(phi)
        report["path"] = str(ckpt)
        report["key"] = key
        operator.lsi_match = report
        return operator

    def residual(
        self,
        image: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor | None = None,
        input_range: str = "-1_1",
    ) -> torch.Tensor:
        y_hat = self.measure(image, indices=indices, input_range=input_range)
        return torch.linalg.vector_norm(y_hat - y, dim=-1) / math.sqrt(y.shape[-1])

    def relative_residual(
        self,
        image: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor | None = None,
        input_range: str = "-1_1",
    ) -> torch.Tensor:
        y_hat = self.measure(image, indices=indices, input_range=input_range)
        num = torch.linalg.vector_norm(y_hat.float() - y.float(), dim=-1)
        den = torch.linalg.vector_norm(y.float(), dim=-1).clamp_min(1e-8)
        return num / den


def add_sensor_noise(
    y: torch.Tensor,
    mode: str = "clean",
    gaussian_rel_std: float = 0.02,
    poisson_photons: float = 1e4,
    gain_std: float = 0.05,
    bias_rel_std: float = 0.02,
    dropout_prob: float = 0.20,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Simulate computational-imaging sensor perturbations on calibrated measurements.

    Returns (y_noisy, keep_mask) where keep_mask is None unless channels are dropped.
    Modes: clean (N0), gaussian, poisson_gaussian (N1), gain_bias (N2),
           dropout (N3), combined (N4 unseen combination).
    """
    y_f = y.float()
    scale = y_f.abs().mean(dim=-1, keepdim=True).clamp_min(1.0)
    if mode in ("clean", "N0"):
        return y_f.to(dtype=y.dtype), None

    def _randn(shape):
        return torch.randn(shape, device=y.device, dtype=torch.float32, generator=generator)

    out = y_f.clone()
    keep_mask = None
    if mode in ("gaussian",):
        out = out + _randn(out.shape) * (gaussian_rel_std * scale)
    elif mode in ("poisson_gaussian", "N1"):
        # Shot noise proportional to sqrt(max(y, 0)) + Gaussian readout noise
        y_pos = out.clamp_min(0.0)
        shot_std = torch.sqrt(y_pos * scale / max(poisson_photons, 1.0))
        out = out + _randn(out.shape) * shot_std + _randn(out.shape) * (gaussian_rel_std * scale)
    elif mode in ("gain_bias", "N2"):
        gain = 1.0 + _randn((1, out.shape[-1])) * gain_std
        bias = _randn((1, out.shape[-1])) * (bias_rel_std * scale.mean())
        out = out * gain + bias
    elif mode in ("dropout", "N3"):
        u = torch.rand((out.shape[-1],), device=y.device, generator=generator)
        keep = u >= dropout_prob
        if int(keep.sum()) < max(4, out.shape[-1] // 4):
            keep[: max(4, out.shape[-1] // 4)] = True
        keep_mask = keep
    elif mode in ("combined", "N4"):
        y_pos = out.clamp_min(0.0)
        shot_std = torch.sqrt(y_pos * scale / max(poisson_photons, 1.0))
        gain = 1.0 + _randn((1, out.shape[-1])) * gain_std
        bias = _randn((1, out.shape[-1])) * (bias_rel_std * scale.mean())
        out = out * gain + bias + _randn(out.shape) * shot_std + _randn(out.shape) * (gaussian_rel_std * scale)
        u = torch.rand((out.shape[-1],), device=y.device, generator=generator)
        keep = u >= dropout_prob
        if int(keep.sum()) < max(4, out.shape[-1] // 4):
            keep[: max(4, out.shape[-1] // 4)] = True
        keep_mask = keep
    else:
        raise ValueError(f"unknown sensor noise mode: {mode}")
    return out.to(dtype=y.dtype), keep_mask


def random_binary_phi(measurements: int, resolution: int, generator: torch.Generator | None = None) -> torch.Tensor:
    bits = torch.randint(0, 2, (measurements, resolution, resolution), generator=generator)
    return bits.float()


class WienerPhysicsLifter(nn.Module):
    """Zero-parameter closed-form Wiener (LMMSE) image lifting + subset-adaptive ZCA whitening.

    Addresses the severe ill-conditioning (kappa ~ 2.4e8) of non-negative binary masks Phi in {0,1}^{MxN}:
      1. Closed-form Wiener RGB face lifting for ANY subset S:
         I_wiener(y_S) = I_bar + C_{I, y_S} (Sigma_{y_S} + reg * I)^{-1} (y_S - y_bar_S)
      2. Subset-adaptive ZCA whitening for 1D sensor tokens:
         u_S = (Sigma_{y_S} + reg_w * I)^{-1/2} ((y_S - y_bar_S) / y_sd_S)
    """

    def __init__(self, max_measurements: int = 256, img_size: int = 128):
        super().__init__()
        self.max_measurements = int(max_measurements)
        self.img_size = int(img_size)
        pix = 3 * self.img_size * self.img_size
        self.register_buffer("I_bar", torch.full((1, 3, self.img_size, self.img_size), 0.5))
        self.register_buffer("y_bar", torch.zeros(1, self.max_measurements))
        self.register_buffer("y_sd", torch.ones(1, self.max_measurements))
        self.register_buffer("C_Iy", torch.zeros(pix, self.max_measurements))
        self.register_buffer("Sigma_y", torch.eye(self.max_measurements))
        self._fitted = False

    @torch.no_grad()
    def fit(self, images_u8_or_01: torch.Tensor, y_full: torch.Tensor) -> None:
        """Estimate empirical first/second moments once on the training set (CPU/GPU)."""
        dev = self.I_bar.device
        n = int(y_full.shape[0])
        imgs = images_u8_or_01.float()
        if float(imgs.max()) > 1.5:
            imgs = imgs / 255.0
        if imgs.shape[-1] != self.img_size or imgs.shape[-2] != self.img_size:
            imgs = F.interpolate(imgs, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        i_bar = imgs.mean(dim=0, keepdim=True)
        d_i = (imgs - i_bar).view(n, -1)

        y_f = y_full.float()
        y_bar = y_f.mean(dim=0, keepdim=True)
        d_y = y_f - y_bar
        y_sd = d_y.std(dim=0, keepdim=True).clamp_min(1e-4)
        d_yn = d_y / y_sd

        c_iy = (d_i.T @ d_yn) / float(n)
        sigma_y = (d_yn.T @ d_yn) / float(n)
        # Ensure exact symmetry
        sigma_y = 0.5 * (sigma_y + sigma_y.T)

        self.I_bar.copy_(i_bar.to(dev))
        self.y_bar.copy_(y_bar.to(dev))
        self.y_sd.copy_(y_sd.to(dev))
        self.C_Iy.copy_(c_iy.to(dev))
        self.Sigma_y.copy_(sigma_y.to(dev))
        self._fitted = True

    def _subset_cov(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(device=self.Sigma_y.device, dtype=torch.long)
        return self.Sigma_y.index_select(0, idx).index_select(1, idx)

    def estimate_spectral_reg(
        self,
        y_sub: torch.Tensor,
        indices: torch.Tensor,
        base_reg: float = 4e-4,
        scale: float = 1.5,
        max_reg: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blind per-sample spectral noise-variance estimator from trailing eigenmodes of Sigma_{y_S}.

        Returns (evals [M], evecs [M, M], lam_b [B]) in float32.
        """
        idx = indices.to(device=self.y_bar.device, dtype=torch.long)
        dy = (y_sub.float() - self.y_bar.index_select(1, idx)) / self.y_sd.index_select(1, idx)
        s_mat = self._subset_cov(idx).float()
        evals, evecs = torch.linalg.eigh(s_mat)
        evals = evals.clamp_min(0.0)
        k = max(4, idx.numel() // 4)
        proj_tail = dy @ evecs[:, :k]
        excess = (proj_tail.square() - evals[:k].unsqueeze(0)).mean(dim=-1).clamp_min(0.0)
        lam_b = (float(base_reg) + float(scale) * excess).clamp(float(base_reg), float(max_reg))
        return evals, evecs, lam_b

    def lift_image(
        self,
        y_sub: torch.Tensor,
        indices: torch.Tensor,
        reg: float = 2e-4,
        blind_adapt: bool = False,
    ) -> torch.Tensor:
        """Closed-form Wiener RGB image in [0, 1] of shape [B, 3, img_size, img_size]."""
        with torch.amp.autocast(device_type=y_sub.device.type, enabled=False):
            idx = indices.to(device=self.y_bar.device, dtype=torch.long)
            dy = (y_sub.float() - self.y_bar.index_select(1, idx)) / self.y_sd.index_select(1, idx)
            c_sub = self.C_Iy.index_select(1, idx).float()
            if blind_adapt:
                evals, evecs, lam_b = self.estimate_spectral_reg(y_sub, idx, base_reg=max(reg, 4e-4))
                dy_eig = dy @ evecs
                coef = (dy_eig / (evals.unsqueeze(0) + lam_b.unsqueeze(1))) @ evecs.T
            else:
                s_mat = self._subset_cov(idx).float()
                eye = torch.eye(idx.numel(), device=s_mat.device, dtype=torch.float32)
                coef = torch.linalg.solve(s_mat + float(reg) * eye, dy.T).T
            rec = self.I_bar.view(1, -1).float() + coef @ c_sub.T
            return rec.view(-1, 3, self.img_size, self.img_size).clamp(0.0, 1.0)

    def wiener_proximal(
        self,
        image: torch.Tensor,
        y_sub: torch.Tensor,
        operator: SPIOperator,
        indices: torch.Tensor,
        eta: float = 0.85,
        base_reg: float = 4e-4,
        blind_adapt: bool = True,
        input_range: str = "-1_1",
    ) -> torch.Tensor:
        """Closed-form Kalman-Wiener pixel-space proximal update onto Phi_S x = y_S.

        Uses aligned-face cross-covariance C_{I, y_S} instead of raw binary Phi_S^T so
        updates are RGB-calibrated, speckle-free, and blind noise-adaptive via tail eigenmodes.
        """
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            idx = indices.to(device=self.y_bar.device, dtype=torch.long)
            x01 = operator.to_unit_interval(image.float(), input_range).clamp(0.0, 1.0)
            y_hat = operator.measure(x01, indices=idx, input_range="0_1")
            dy_res = (y_sub.float() - y_hat.float()) / self.y_sd.index_select(1, idx)
            c_sub = self.C_Iy.index_select(1, idx).float()
            if blind_adapt:
                evals, evecs, lam_b = self.estimate_spectral_reg(y_sub, idx, base_reg=base_reg)
                coef = ((dy_res @ evecs) / (evals.unsqueeze(0) + lam_b.unsqueeze(1))) @ evecs.T
            else:
                s_mat = self._subset_cov(idx).float()
                eye = torch.eye(idx.numel(), device=s_mat.device, dtype=torch.float32)
                coef = torch.linalg.solve(s_mat + float(base_reg) * eye, dy_res.T).T
            delta = (coef @ c_sub.T).view(-1, 3, self.img_size, self.img_size)
            if delta.shape[-2:] != x01.shape[-2:]:
                delta = F.interpolate(delta, size=x01.shape[-2:], mode="bilinear", align_corners=False)
            updated = (x01 + float(eta) * delta).clamp(0.0, 1.0)
            return operator.from_unit_interval(updated, input_range).to(dtype=image.dtype)

    def whiten_1d(self, y_sub: torch.Tensor, indices: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
        """Subset-adaptive ZCA whitening u_S = (Sigma_S + reg*I)^{-1/2} dy_S, shape [B, M]."""
        with torch.amp.autocast(device_type=y_sub.device.type, enabled=False):
            idx = indices.to(device=self.y_bar.device, dtype=torch.long)
            dy = (y_sub.float() - self.y_bar.index_select(1, idx)) / self.y_sd.index_select(1, idx)
            s_mat = self._subset_cov(idx).float()
            eigvals, eigvecs = torch.linalg.eigh(s_mat)
            inv_sqrt = (eigvals.clamp_min(0.0) + float(reg)).rsqrt()
            # ZCA whitening preserves sensor coordinate alignment while making eigenvalues O(1)
            w_zca = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T
            return (dy @ w_zca).clamp(-10.0, 10.0)
