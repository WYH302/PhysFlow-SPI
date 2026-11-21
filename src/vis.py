from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    x = image.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if float(x.min()) < -1e-4:
        x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    array = (x.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    return array


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, titles: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _to_uint8(images)
    n, h, w, _ = frames.shape
    ncol = min(nrow, n)
    nrow_grid = int(np.ceil(n / ncol))
    label_h = 18 if titles else 0
    canvas = Image.new("RGB", (ncol * w, nrow_grid * (h + label_h)), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        r, c = divmod(index, ncol)
        tile = Image.fromarray(frame)
        y = r * (h + label_h)
        canvas.paste(tile, (c * w, y))
        if titles and index < len(titles):
            draw.text((c * w + 2, y + h + 2), titles[index][:28], fill=(230, 230, 230))
    canvas.save(path)
    return path


def save_gray_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> Path:
    x = images.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    norm = (x - lo) / (hi - lo).clamp_min(1e-8)
    rgb = norm.repeat(1, 3, 1, 1) if norm.shape[1] == 1 else norm
    return save_image_grid(rgb * 2 - 1, path, nrow=nrow)
