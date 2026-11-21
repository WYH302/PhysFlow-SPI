from __future__ import annotations

import torch
import torch.nn.functional as F


def huber_measurement_loss(
    x_pred: torch.Tensor,
    y: torch.Tensor,
    operator,
    indices: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    t_min: float = 0.7,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber penalty on Phi(x_pred). x_pred is the clean-image head, not a velocity.

    When t is provided, samples with t < t_min contribute nothing. Callers that
    train only near the clean end can pass t and keep a single code path.
    The loss does not unroll an ODE.
    """
    y_hat = operator.measure(x_pred, indices=indices, input_range="-1_1")
    per = F.huber_loss(y_hat, y, reduction="none", delta=delta).mean(dim=-1)
    if t is None:
        return per.mean()
    weight = (t.reshape(-1) >= t_min).to(per.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (per * weight).sum() / denom
