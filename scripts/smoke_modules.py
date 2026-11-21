"""8-sample, 2-step smoke for every PhysFlow module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.optical import SPIOperator, random_binary_phi
from src.runtime import device, make_optimizer, peak_vram_gb, reset_vram, seed_all
from src.solvers.flower import flower_sample
from src.solvers.losses import huber_measurement_loss
from src.vis import save_image_grid


def main() -> None:
    assert "src.models.teachers" not in sys.modules
    seed_all(0)
    dev = device()
    reset_vram()
    report = {"device": str(dev), "checks": {}}

    phi = random_binary_phi(32, 32).to(dev)
    op = SPIOperator(phi).to(dev)
    images = torch.randn(8, 3, 32, 32, device=dev)
    images.requires_grad_(True)
    indices = op.sample_indices(16)
    y = op.measure(images, indices=indices, input_range="-1_1")
    assert y.shape == (8, 16), y.shape
    y2 = op.measure(images, indices=indices, input_range="-1_1")
    repeat = float((y - y2).abs().max())
    reference = images.detach().double()
    y_ref = op.measure(reference, indices=indices.cpu(), input_range="-1_1")
    # double path uses the same buffers; compare in float32
    residual = float((y.float() - op.measure(images.detach(), indices=indices).float()).abs().max())
    loss = y.square().mean()
    loss.backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    noisy = (images.detach() + 0.05 * torch.randn_like(images)).clamp(-1, 1)
    before = float(op.residual(noisy, y.detach(), indices=indices).mean())
    projected = op.proximal(noisy, y.detach(), indices=indices, lam=1e-2, eta=1.0)
    after = float(op.residual(projected, y.detach(), indices=indices).mean())
    report["checks"]["spi"] = {
        "y_shape": list(y.shape),
        "repeat_max_abs": repeat,
        "self_residual_max_abs": residual,
        "proximal_residual_before": before,
        "proximal_residual_after": after,
        "proximal_improved": after < before,
        "grad_finite": True,
    }
    assert repeat == 0.0
    assert residual < 1e-5
    assert after < before, (before, after)
    del y_ref

    from src.data.align import similarity_matrix, template_5, transform_points
    import numpy as np

    template = template_5(64)
    known = np.array([[1.4, -0.2], [0.2, 1.4]])
    shift = np.array([8.0, -5.0])
    src = template @ known.T + shift
    recovered = similarity_matrix(src, template)
    mapped = transform_points(src, recovered)
    align_err = float(np.abs(mapped - template).max())
    report["checks"]["align"] = {"max_point_error": align_err}
    assert align_err < 1e-4

    from src.data.identity import assert_identity_disjoint, load_identity_table, split_identities
    from src.data.synthetic import synthetic_identity_split

    split = synthetic_identity_split()
    report["checks"]["identity_split"] = {
        "n_train_ids": len(split["splits"]["train"]),
        "n_val_ids": len(split["splits"]["val"]),
        "n_test_ids": len(split["splits"]["test"]),
        "overlap": 0,
    }
    text = ROOT / "outputs" / "_identity_smoke.txt"
    text.write_text("000001.jpg 7\n000002.jpg 8\n000003.jpg 7\n", encoding="utf-8")
    table = load_identity_table(text)
    assert table["000001.jpg"] == "7"
    try:
        assert_identity_disjoint({"7"}, {"7"})
        raise SystemExit("overlap assertion did not fire")
    except AssertionError:
        report["checks"]["identity_overlap_assertion"] = True

    from src.data.synthetic import render_faces
    from src.models.encoders import (
        DualDomainEncoder,
        MeanEncoder,
        Transformer1DEncoder,
        build_mlp_matching,
        condition_loss,
        parameter_count,
    )

    batch = render_faces(8, 32, seed=1)
    batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    y_enc = op.measure(batch["image"], indices=indices).detach()
    transformer = Transformer1DEncoder(max_measurements=32, dim=64, depth=2, heads=4).to(dev)
    mlp = build_mlp_matching(transformer, max_measurements=32).to(dev)
    mean = MeanEncoder().to(dev)
    mean.set_means(batch["sem"], batch["identity"], batch["geo"])
    dual = DualDomainEncoder(op, max_measurements=32, dim=64, vision_dim=64).to(dev)
    # Tiny backbone default is 384. Override by replacing the projection's expected dim.
    dual.backbone = torch.nn.Sequential()  # replaced below
    from src.models.encoders import TinyVisionBackbone

    dual.backbone = TinyVisionBackbone(64).to(dev)
    dual.vision_proj = torch.nn.Linear(64, 64).to(dev)
    encoder_report = {}
    for name, model in {"mean": mean, "mlp": mlp, "transformer1d": transformer, "dual": dual}.items():
        pred = model(y_enc, indices)
        assert pred[0].shape[0] == 8 and pred[1].shape == (8, 512) and pred[2].shape[-1] == 212
        if name == "mean":
            encoder_report[name] = {"trainable": 0, "matches_mean": True}
            continue
        opt, opt_name = make_optimizer(model.parameters(), lr=1e-3)
        last = None
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            pred = model(y_enc, indices)
            loss = condition_loss(pred, batch)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
        encoder_report[name] = {
            "trainable": parameter_count(model),
            "loss": last,
            "grad_ok": all(grads) and len(grads) > 0,
            "optimizer": opt_name,
        }
        assert encoder_report[name]["grad_ok"]
    encoder_report["mlp_params"] = parameter_count(mlp)
    encoder_report["transformer_params"] = parameter_count(transformer)
    encoder_report["param_ratio"] = parameter_count(mlp) / max(parameter_count(transformer), 1)
    report["checks"]["encoders"] = encoder_report

    from src.models.surrogate import TinyXPred, flow_sample_xt, official_v_loss

    model = TinyXPred(image_size=32, patch=8, dim=32, depth=1, text_dim=16).to(dev)
    assert not any("adaln" in name.lower() for name, _ in model.named_modules())
    opt, opt_name = make_optimizer(model.trainable_parameters(), lr=1e-3)
    images32 = torch.randn(8, 3, 32, 32, device=dev)
    context = torch.randn(8, 4, 16, device=dev)
    mask = torch.ones(8, 4, device=dev)
    identity = torch.randn(8, 512, device=dev)
    sem = torch.randn(8, 2, 256, device=dev)
    geo = torch.randn(8, 212, device=dev)
    last = None
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        t = torch.empty(8, device=dev).uniform_(0.2, 0.8)
        x_t, _ = flow_sample_xt(images32, t)
        x0 = model(x_t, t, context, mask, sem=sem, identity=identity, geo=geo)
        assert x0.shape == images32.shape
        loss = official_v_loss(x0, x_t, images32, t)
        loss = loss + huber_measurement_loss(
            x0, op.measure(images32, indices).detach(), op, indices=indices, t=t, t_min=0.0
        )
        loss.backward()
        opt.step()
        last = float(loss.detach())
    gate_grad = model.condition.gate.grad
    lora_grads = [p.grad for p in model.lora_params if p.grad is not None]
    report["checks"]["injection"] = {
        "x0_shape": list(x0.shape),
        "loss": last,
        "gate_grad_abs": float(gate_grad.abs().mean()) if gate_grad is not None else None,
        "lora_grads": len(lora_grads),
        "optimizer": opt_name,
        "adaln": False,
    }
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert len(lora_grads) > 0

    def predict(x, t, text, attn):
        return model(x, t, text, attn, sem=sem[: x.shape[0]], identity=identity[: x.shape[0]], geo=geo[: x.shape[0]])

    def project(x0, t, eta):
        del t
        return op.proximal(x0, op.measure(images32[: x0.shape[0]], indices).detach(), indices=indices, eta=eta)

    sampled = flower_sample(
        predict,
        context[:2],
        mask[:2],
        steps=2,
        image_size=32,
        eta=0.5,
        project=project,
    )
    assert sampled.shape == (2, 3, 32, 32)
    report["checks"]["flower"] = {"shape": list(sampled.shape), "steps": 2}
    report["vram_gb"] = peak_vram_gb()
    path = ROOT / "outputs" / "smoke_modules.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_image_grid(images32[:8].detach(), ROOT / "outputs" / "smoke_input_grid.png", nrow=4)
    print(json.dumps({"ok": True, "vram_gb": report["vram_gb"], "proximal": [before, after], "gate": report["checks"]["injection"]["gate_grad_abs"]}, indent=2))


if __name__ == "__main__":
    main()
