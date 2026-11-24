"""Lightweight zero-initialized post-flow pixel high-frequency refiner (2.27M params)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LateStagePixelHFRefiner(nn.Module):
    """2.27M-param zero-initialized three-scale residual U-Net refiner (23.89 ms at 512px)."""

    def __init__(self, base: int = 48):
        super().__init__()
        # Input: x_flow (3) + (x_flow - I_wiener_up) (3) + log2(M)/8 map (1) = 7 channels
        self.enc0 = nn.Sequential(nn.Conv2d(7, base, 3, padding=1), ResBlock(base))
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), ResBlock(base * 2))
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), ResBlock(base * 4), ResBlock(base * 4))
        self.up1 = nn.Sequential(nn.Conv2d(base * 6, base * 2, 3, padding=1), ResBlock(base * 2))
        self.up0 = nn.Sequential(nn.Conv2d(base * 3, base, 3, padding=1), ResBlock(base))
        self.head = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))
        # Strict zero initialization -> exact identity at step 0
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_flow_m11: torch.Tensor, iw_m11: torch.Tensor, rate_norm: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x_flow_m11.shape
        rmap = rate_norm.view(b, 1, 1, 1).expand(b, 1, h, w).to(dtype=x_flow_m11.dtype, device=x_flow_m11.device)
        inp = torch.cat([x_flow_m11, x_flow_m11 - iw_m11, rmap], dim=1)
        f0 = self.enc0(inp)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        u1 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, f1], dim=1))
        u0 = F.interpolate(u1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, f0], dim=1))
        delta = self.head(u0)
        return (x_flow_m11 + delta).clamp(-1.0, 1.0)
