from __future__ import annotations

import torch


@torch.no_grad()
def flower_sample(
    predict_x0,
    text: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 8,
    cfg_scale: float = 1.0,
    eta: float = 1.0,
    image_size: int = 512,
    noise_scale: float = 2.0,
    project=None,
    clamp_min: float = 0.001,
    generator: torch.Generator | None = None,
    stochastic: bool = False,
    eta_power: float = 0.0,
    x_init: torch.Tensor | None = None,
    t_start: float = 0.0,
) -> torch.Tensor:
    """Pixel-space FLOWER against an x-prediction model.

    The clean image is the network output. Velocity is the official conversion
    v = (x0 - x) / clamp(1 - t), then an Euler step follows the hub sampler.
    When `x_init` ([-1, 1] RGB, e.g., closed-form Wiener lift) and `t_start > 0`
    are provided, initializes strictly on the official linear flow path:
        z_{t_0} = t_0 * x_init + (1 - t_0) * (noise_scale * eps).
    `project(x0, t, eta_t)` is the closed-form measurement proximal.
    """
    batch = text.shape[0]
    device = text.device
    dtype = text.dtype
    eps0 = torch.randn(
        batch,
        3,
        image_size,
        image_size,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * noise_scale
    t0 = float(t_start)
    if x_init is not None and t0 > 0.0:
        x = (t0 * x_init.to(device=device, dtype=dtype) + (1.0 - t0) * eps0)
    else:
        t0 = 0.0
        x = eps0
    timesteps = torch.linspace(t0, 1.0, steps + 1, device=device, dtype=dtype)
    null = torch.zeros_like(mask)
    for index in range(steps):
        t_cur = timesteps[index].expand(batch)
        t_next = timesteps[index + 1]
        x0 = predict_x0(x, t_cur, text, mask)
        if cfg_scale != 1.0:
            x0_uncond = predict_x0(x, t_cur, text, null)
            x0 = x0_uncond + (x0 - x0_uncond) * cfg_scale
        if project is not None and eta != 0.0:
            t_scalar = float(timesteps[index])
            eta_t = float(eta) * (max(t_scalar, 0.1) ** float(eta_power)) if eta_power > 0.0 else float(eta)
            x0 = project(x0, t_cur, eta_t)
        if stochastic:
            if index == steps - 1:
                x = x0
            else:
                eps = torch.randn(x.shape, device=device, dtype=dtype, generator=generator) * noise_scale
                tn = t_next.to(dtype=x.dtype)
                x = tn * x0 + (1.0 - tn) * eps
        else:
            step = (t_next - timesteps[index]).to(dtype=x.dtype)
            v = (x0 - x) / (1.0 - t_cur)[:, None, None, None].clamp_min(clamp_min)
            x = x + step * v
    return x.clamp(-1.0, 1.0)
