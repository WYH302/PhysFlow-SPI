"""Short surrogate runs used when the official MiniT2I weights are not in memory.

These functions train the injection and measurement-loss code. They are not
MiniT2I-B/16 results. Callers must label the numbers that way.
"""

from __future__ import annotations

import torch

from src.models.surrogate import TinyXPred, flow_sample_xt, official_v_loss
from src.runtime import make_optimizer
from src.solvers.losses import huber_measurement_loss


MODES = (
    "discrete_text",
    "continuous_semantic",
    "zero_shot_flower",
    "plus_id",
    "plus_geometry",
    "physical_loop",
)


def _context(batch, text_dim: int) -> torch.Tensor:
    color = batch["color"]
    # Discrete text does not contain the identity color. A constant token does.
    return color.new_zeros(color.shape[0], 4, text_dim)


def train_mode(
    mode: str,
    batch: dict,
    operator,
    indices: torch.Tensor,
    steps: int,
    lr: float = 1e-3,
    image_size: int = 32,
    text_dim: int = 16,
    measurement_weight: float = 0.1,
    t_min: float = 0.0,
    t_low: float = 0.15,
    t_high: float = 0.95,
) -> dict:
    torch.manual_seed(0)
    device = batch["image"].device
    model = TinyXPred(image_size=image_size, patch=8 if image_size % 8 == 0 else 4, dim=32, depth=1, text_dim=text_dim).to(device)
    gate = [model.condition.gate]
    others = [param for param in model.trainable_parameters() if param is not model.condition.gate]
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(
            [{"params": gate, "lr": lr * 20.0}, {"params": others, "lr": lr}],
            weight_decay=0.0,
        )
        opt_name = "adamw8bit"
    except Exception as exc:
        opt = torch.optim.AdamW(
            [{"params": gate, "lr": lr * 20.0}, {"params": others, "lr": lr}],
            weight_decay=0.0,
        )
        opt_name = f"adamw ({type(exc).__name__}: {exc})"
    use_sem = mode in {"continuous_semantic", "plus_id", "plus_geometry", "physical_loop"}
    use_id = mode in {"plus_id", "plus_geometry", "physical_loop"}
    use_geo = mode in {"plus_geometry", "physical_loop"}
    physical = mode == "physical_loop"
    weight = measurement_weight
    history = []
    images = batch["image"]
    n = images.shape[0]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pick = torch.randint(0, n, (min(8, n),), device=device)
        image = images[pick]
        low = 0.55 if physical else t_low
        high = 0.95 if physical else t_high
        t = torch.empty(image.shape[0], device=device).uniform_(low, high)
        x_t, _ = flow_sample_xt(image, t)
        context = _context(batch, text_dim)[pick]
        mask = torch.ones(context.shape[:2], device=device)
        x0 = model(
            x_t,
            t,
            context,
            mask,
            sem=batch["sem"][pick] if use_sem else None,
            identity=batch["identity"][pick] if use_id else None,
            geo=batch["geo"][pick] if use_geo else None,
            use_sem=use_sem,
            use_id=use_id,
            use_geo=use_geo,
        )
        loss = official_v_loss(x0, x_t, image, t)
        y = operator.measure(image, indices=indices).detach()
        meas = huber_measurement_loss(x0, y, operator, indices=indices, t=t, t_min=t_min if physical else 0.0)
        if physical:
            loss = loss + weight * meas
        loss.backward()
        opt.step()
        history.append({"step": step, "loss": float(loss.detach()), "measurement": float(meas.detach()), "weight": weight})
    eval_out = evaluate_mode(model, mode, batch, operator, indices, text_dim)
    eval_out["optimizer"] = opt_name
    eval_out["history_last"] = history[-1] if history else None
    eval_out["history_first"] = history[0] if history else None
    eval_out["gate_grad_abs"] = float(model.condition.gate.detach().abs())
    eval_out["model"] = model
    return eval_out


@torch.no_grad()
def evaluate_mode(model, mode, batch, operator, indices, text_dim: int) -> dict:
    device = batch["image"].device
    use_sem = mode in {"continuous_semantic", "plus_id", "plus_geometry", "physical_loop"}
    use_id = mode in {"plus_id", "plus_geometry", "physical_loop"}
    use_geo = mode in {"plus_geometry", "physical_loop"}
    image = batch["image"][:8]
    context = _context(batch, text_dim)[:8]
    if mode == "zero_shot_flower":
        context = torch.zeros_like(context)
    mask = torch.ones(context.shape[:2], device=device)
    t = torch.zeros(image.shape[0], device=device)
    x_t = torch.randn_like(image) * 2
    x0 = model(
        x_t,
        t,
        context,
        mask,
        sem=batch["sem"][:8] if use_sem else None,
        identity=batch["identity"][:8] if use_id else None,
        geo=batch["geo"][:8] if use_geo else None,
        use_sem=use_sem,
        use_id=use_id,
        use_geo=use_geo,
    )
    if mode in {"zero_shot_flower", "physical_loop"}:
        y = operator.measure(image, indices=indices)
        x0 = operator.proximal(x0, y, indices=indices, eta=1.0)
    pred_color = (x0.mean(dim=(2, 3)) * 0.5 + 0.5).clamp(0, 1)
    target = (image.mean(dim=(2, 3)) * 0.5 + 0.5).clamp(0, 1)
    color_l2 = float((pred_color - target).pow(2).mean())
    color_cos = float(torch.nn.functional.cosine_similarity(pred_color, target, dim=-1).mean())
    y_hat = operator.measure(x0, indices=indices)
    y = operator.measure(image, indices=indices)
    residual = float(torch.sqrt((y_hat - y).pow(2).mean()))
    return {
        "mode": mode,
        "color_l2": color_l2,
        "color_cosine": color_cos,
        "measurement_rmse": residual,
        "image_std": float(x0.std()),
        "x0": x0.detach().cpu(),
        "official_minit2i": False,
    }
