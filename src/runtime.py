from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def make_optimizer(params, lr: float, weight_decay: float = 0.0):
    params = [p for p in params if p.requires_grad]
    try:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay), "adamw8bit"
    except Exception as exc:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay), f"adamw ({type(exc).__name__}: {exc})"
