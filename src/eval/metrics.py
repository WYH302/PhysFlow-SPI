from __future__ import annotations

import torch
import torch.nn.functional as F


def to_unit(image: torch.Tensor) -> torch.Tensor:
    if float(image.detach().min()) < -1e-4:
        return image * 0.5 + 0.5
    return image


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    a = to_unit(pred).float()
    b = to_unit(target).float()
    mse = (a - b).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(window: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - window // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel2d = kernel[:, None] * kernel[None, :]
    return kernel2d.expand(channels, 1, window, window).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    a = to_unit(pred).float()
    b = to_unit(target).float()
    channels = a.shape[1]
    window = _gaussian_window(11, 1.5, channels, a.device, a.dtype)
    pad = 5
    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a2 = mu_a.pow(2)
    mu_b2 = mu_b.pow(2)
    mu_ab = mu_a * mu_b
    sigma_a = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a2
    sigma_b = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab
    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a + sigma_b + c2))
    return value.mean(dim=(1, 2, 3))


def nme(pred: torch.Tensor, target: torch.Tensor, left: int = 0, right: int = 1) -> torch.Tensor:
    """Inter-ocular NME. Default indices are the two eyes of a 5-point set."""
    iod = torch.linalg.vector_norm(target[:, left, :] - target[:, right, :], dim=-1).clamp_min(1e-6)
    err = torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1)
    return err / iod


def cosine_identity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    a = F.normalize(pred.float(), dim=-1)
    b = F.normalize(target.float(), dim=-1)
    return (a * b).sum(dim=-1)


def measurement_rmse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((y_hat - y).pow(2).mean(dim=-1))
