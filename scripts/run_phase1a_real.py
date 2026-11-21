"""Phase 1A on real aligned faces: Measurement-to-Condition Factorization (Loop B).

Trains and evaluates the complete 6-model scientific ladder on 5,000 real train / 1,479 disjoint test faces:
  1. MeanEncoder (training mean baseline)
  2. WhitenedMLPEncoder (use_whiten=False: demonstrates kappa = 2.38e8 raw binary ill-conditioning)
  3. WhitenedMLPEncoder (use_whiten=True: strong subset-adaptive ZCA-whitened 1D MLP baseline)
  4. WhitenedTransformer1DEncoder (strong ZCA-whitened 1D Sensor Transformer baseline)
  5. PhysicsDualDomainFactorizer (no L_geo_inv: Closed-Form Wiener Physics Lifting -> DINOv2-Small
     257 spatial tokens + ZCA-whitened 1D fusion)
  6. PhysicsDualDomainFactorizer (FULL OURS: + Measurement-Domain Canonical Geometry Invariance)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.models.encoders import (
    MeanEncoder,
    PhysicsDualDomainFactorizer,
    WhitenedMLPEncoder,
    WhitenedTransformer1DEncoder,
    factorized_condition_loss,
    parameter_count,
)
from src.optical.spi_operator import RATES, SPIOperator, WienerPhysicsLifter, add_sensor_noise
from src.paths import DATA, OUTPUTS
from src.runtime import peak_vram_gb, reset_vram, seed_all
from src.vis import save_image_grid

CACHE_DIR = DATA / "cache" / "physflow_real"
CKPT_DIR = OUTPUTS / "checkpoints"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device(min_free_gb: float = 4.0) -> torch.device:
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b / (1024**3) >= min_free_gb:
            return torch.device("cuda")
    return torch.device("cpu")


def landmark_nme106(pred_212: torch.Tensor, gt_212: torch.Tensor) -> torch.Tensor:
    p = pred_212.float().view(-1, 106, 2)
    g = gt_212.float().view(-1, 106, 2)
    left_eye = g[:, 33:43, :].mean(dim=1)
    right_eye = g[:, 87:97, :].mean(dim=1)
    iod = torch.linalg.vector_norm(left_eye - right_eye, dim=-1).clamp_min(1e-6)
    err = torch.linalg.vector_norm(p - g, dim=-1).mean(dim=-1)
    return err / iod


def evaluate_encoder(
    model: torch.nn.Module,
    y256: torch.Tensor,
    sem: torch.Tensor,
    sem_mask: torch.Tensor,
    ident: torch.Tensor,
    geo: torch.Tensor,
    indices_eval: list[int],
    dev: torch.device,
    rate: int,
    noise_mode: str = "clean",
    batch_size: int = 64,
) -> dict:
    model.eval()
    idx_tensor = torch.arange(rate, device=dev, dtype=torch.long)
    id_cos_all, nme_all, geo_l1_all, sem_cos_all = [], [], [], []
    gen = torch.Generator(device=dev).manual_seed(42 + rate)
    use_amp = dev.type == "cuda"
    with torch.no_grad():
        for s in range(0, len(indices_eval), batch_size):
            pick = indices_eval[s : s + batch_size]
            y_b = y256[pick, :rate].to(dev)
            y_in, keep = add_sensor_noise(y_b, mode=noise_mode, generator=gen)
            if keep is not None:
                y_in = y_in[:, keep]
                idx_use = idx_tensor[keep]
            else:
                idx_use = idx_tensor
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                p_sem, p_id, p_geo = model(y_in, idx_use)
            gt_id = ident[pick].to(dev)
            gt_geo = geo[pick].to(dev)
            gt_sem = sem[pick].to(dev, dtype=torch.float32)
            gt_m = sem_mask[pick].to(dev, dtype=torch.float32)

            id_cos_all.append(F.cosine_similarity(F.normalize(p_id.float(), dim=-1), F.normalize(gt_id.float(), dim=-1), dim=-1).cpu())
            nme_all.append(landmark_nme106(p_geo, gt_geo).cpu())
            geo_l1_all.append((p_geo.float() - gt_geo.float()).abs().mean(dim=-1).cpu())
            sc = F.cosine_similarity(p_sem.float(), gt_sem, dim=-1)
            sem_cos_all.append(((sc * gt_m).sum(dim=-1) / gt_m.sum(dim=-1).clamp_min(1.0)).cpu())

    return {
        "id_cosine": float(torch.cat(id_cos_all).mean()),
        "nme106": float(torch.cat(nme_all).mean()),
        "geo_l1": float(torch.cat(geo_l1_all).mean()),
        "sem_cosine": float(torch.cat(sem_cos_all).mean()),
    }


def train_encoder(
    name: str,
    model: torch.nn.Module,
    y256: torch.Tensor,
    sem: torch.Tensor,
    sem_mask: torch.Tensor,
    ident: torch.Tensor,
    geo: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    dev: torch.device,
    steps: int,
    batch_size: int,
    lr_head: float,
    lr_backbone: float = 5e-5,
    use_geo_inv: bool = True,
    seed: int = 42,
) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or steps <= 0:
        return {"steps": 0, "train_loss_first": None, "train_loss_last": None}

    seed_all(seed)
    if hasattr(model, "backbone_parameters") and hasattr(model, "head_parameters"):
        param_groups = [
            {"params": model.head_parameters(), "lr": lr_head, "weight_decay": 1e-3},
            {"params": model.backbone_parameters(), "lr": lr_backbone, "weight_decay": 1e-2},
        ]
    else:
        param_groups = [{"params": params, "lr": lr_head, "weight_decay": 1e-3}]

    opt = torch.optim.AdamW(param_groups)
    warmup = min(80, max(10, steps // 15))

    def _lr_scale(step_i: int) -> float:
        if step_i < warmup:
            return float(step_i + 1) / float(warmup)
        prog = float(step_i - warmup) / float(max(1, steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_scale)
    rng = np.random.default_rng(seed)
    t_gen = torch.Generator(device=dev).manual_seed(seed)
    noise_modes = ("gaussian", "poisson_gaussian", "gain_bias", "dropout")
    n_train = len(train_idx)
    train_arr = np.asarray(train_idx, dtype=np.int64)
    use_amp = dev.type == "cuda"

    first_loss = None
    last_loss = None
    t0 = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        pick = train_arr[rng.integers(0, n_train, size=batch_size)].tolist()
        rate = int(RATES[int(rng.integers(0, len(RATES)))])
        idx_clean = torch.arange(rate, device=dev, dtype=torch.long)
        y_clean = y256[pick, :rate].to(dev)

        # Light sensor jitter on 25% of clean steps
        if rng.random() < 0.25:
            y_in, keep = add_sensor_noise(y_clean, mode="gaussian", gaussian_rel_std=0.005, generator=t_gen)
            idx_in = idx_clean[keep] if keep is not None else idx_clean
            y_in = y_in[:, keep] if keep is not None else y_in
        else:
            y_in, idx_in = y_clean, idx_clean

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(y_in, idx_in)
            pred_noisy = None
            if use_geo_inv:
                nm = noise_modes[int(rng.integers(0, len(noise_modes)))]
                y_n, km = add_sensor_noise(y_clean, mode=nm, dropout_prob=0.20, generator=t_gen)
                if km is not None:
                    y_n = y_n[:, km]
                    idx_n = idx_clean[km]
                else:
                    idx_n = idx_clean
                pred_noisy = model(y_n, idx_n)

            gt_sem = sem[pick].to(dev, dtype=torch.float32)
            gt_m = sem_mask[pick].to(dev, dtype=torch.float32)
            gt_id = ident[pick].to(dev, dtype=torch.float32)
            gt_geo = geo[pick].to(dev, dtype=torch.float32)

            loss, parts = factorized_condition_loss(
                pred, gt_sem, gt_m, gt_id, gt_geo, pred_noisy=pred_noisy, geo_inv_weight=0.5 if use_geo_inv else 0.0
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if first_loss is None:
            first_loss = parts["total"]
        last_loss = parts["total"]
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            v64 = evaluate_encoder(model, y256, sem, sem_mask, ident, geo, val_idx[:200], dev, rate=64)
            v256 = evaluate_encoder(model, y256, sem, sem_mask, ident, geo, val_idx[:200], dev, rate=256)
            model.train()
            log(
                f"  [{name}] step {step:4d}/{steps} loss={parts['total']:.3f} "
                f"(id={parts['l_id']:.3f} geo={parts['l_geo']:.5f}) "
                f"| val@64 id={v64['id_cosine']:.4f} nme={v64['nme106']:.4f} "
                f"| val@256 id={v256['id_cosine']:.4f} nme={v256['nme106']:.4f}"
            )
    return {
        "steps": steps,
        "seconds": time.perf_counter() - t0,
        "train_loss_first": first_loss,
        "train_loss_last": last_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = pick_device(min_free_gb=4.0)
    log(f"Phase 1A Real (Wiener Physics Lifting + ZCA Whitening) running on device={dev}")
    reset_vram()

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    sem = torch.load(CACHE_DIR / "sem_hidden_bf16.pt", map_location="cpu", weights_only=True)
    sem_mask = torch.load(CACHE_DIR / "sem_mask_u8.pt", map_location="cpu", weights_only=True)
    ident = torch.load(CACHE_DIR / "identity_antelope.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    geo = lmks.reshape(lmks.shape[0], 212)
    imgs256 = torch.load(CACHE_DIR / "images256_u8.pt", map_location="cpu", weights_only=True)

    train_idx = [r["index"] for r in manifest["records"] if r["split"] == "train"]
    val_idx = [r["index"] for r in manifest["records"] if r["split"] == "val"]
    test_idx = [r["index"] for r in manifest["records"] if r["split"] in ("celeba_test", "celebahq_test")]
    log(f"Loaded real cache: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} (CelebA + CelebA-HQ disjoint)")

    op = SPIOperator.from_simulation_256().to(dev)
    lifter = WienerPhysicsLifter(max_measurements=256, img_size=128).to(dev)
    log("Fitting zero-parameter closed-form WienerPhysicsLifter on 5,000 training faces ...")
    lifter.fit(imgs256[train_idx], y256[train_idx])

    sem_dim = int(sem.shape[-1])
    sem_len = int(sem.shape[1])
    sem_mean = sem[train_idx].float().mean(dim=0)
    geo_mean = geo[train_idx].float().mean(dim=0)

    mean_enc = MeanEncoder(sem_tokens=sem_len, sem_dim=sem_dim, id_dim=512, geo_dim=212).to(dev)
    mean_enc.set_means(sem[train_idx].float(), ident[train_idx], geo[train_idx])

    mlp_raw = WhitenedMLPEncoder(
        lifter, max_measurements=256, hidden=512, depth=3, sem_tokens=sem_len, sem_dim=sem_dim, use_whiten=False
    ).to(dev)
    mlp_raw.set_target_means(sem_mean, geo_mean)

    mlp_white = WhitenedMLPEncoder(
        lifter, max_measurements=256, hidden=512, depth=3, sem_tokens=sem_len, sem_dim=sem_dim, use_whiten=True
    ).to(dev)
    mlp_white.set_target_means(sem_mean, geo_mean)

    trans1d_white = WhitenedTransformer1DEncoder(
        lifter, max_measurements=256, dim=256, depth=4, heads=4, sem_tokens=sem_len, sem_dim=sem_dim
    ).to(dev)
    trans1d_white.set_target_means(sem_mean, geo_mean)

    dual_no_inv = PhysicsDualDomainFactorizer(
        op, lifter=lifter, max_measurements=256, dim=384, sem_tokens=sem_len, sem_dim=sem_dim, lam=2e-4, unfreeze_last_n=6
    ).to(dev)
    dual_no_inv.set_target_means(sem_mean, geo_mean)

    dual_full = PhysicsDualDomainFactorizer(
        op, lifter=lifter, max_measurements=256, dim=384, sem_tokens=sem_len, sem_dim=sem_dim, lam=2e-4, unfreeze_last_n=6
    ).to(dev)
    dual_full.set_target_means(sem_mean, geo_mean)

    models_cfg = [
        ("mean", mean_enc, 0, 1e-3, 0.0, False),
        ("mlp_raw", mlp_raw, args.steps, 5e-4, 0.0, False),
        ("mlp_whitened", mlp_white, args.steps, 5e-4, 0.0, False),
        ("transformer1d_whitened", trans1d_white, args.steps, 5e-4, 0.0, False),
        ("dual_domain_no_inv", dual_no_inv, args.steps, 5e-4, 6e-5, False),
        ("dual_domain_full", dual_full, args.steps, 5e-4, 6e-5, True),
    ]

    results = {}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, model, n_steps, lr_h, lr_b, use_inv in models_cfg:
        log(f"=== Training {name} (trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}) ===")
        tinfo = train_encoder(
            name, model, y256, sem, sem_mask, ident, geo, train_idx, val_idx, dev,
            steps=n_steps, batch_size=args.batch_size, lr_head=lr_h, lr_backbone=lr_b, use_geo_inv=use_inv, seed=args.seed,
        )
        per_rate = {str(r): evaluate_encoder(model, y256, sem, sem_mask, ident, geo, test_idx, dev, rate=r) for r in RATES}
        per_noise = {
            nm: evaluate_encoder(model, y256, sem, sem_mask, ident, geo, test_idx, dev, rate=64, noise_mode=nm)
            for nm in ("clean", "gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined")
        }
        results[name] = {
            "parameters": parameter_count(model),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train": tinfo,
            "rates": per_rate,
            "noise_at_64": per_noise,
        }
        if name == "dual_domain_full":
            torch.save(
                {"state_dict": model.state_dict(), "config": {"sem_len": sem_len, "sem_dim": sem_dim, "lam": 2e-4}},
                CKPT_DIR / "dual_encoder_real.pt",
            )

    d = results["dual_domain_full"]["rates"]
    m_nme64 = results["mean"]["rates"]["64"]["nme106"]
    d_nme64 = d["64"]["nme106"]
    nme_drop_64 = (m_nme64 - d_nme64) / max(m_nme64, 1e-8)
    baselines = ("mean", "mlp_raw", "mlp_whitened", "transformer1d_whitened")
    gate = {
        "real_dataset": True,
        "n_train": len(train_idx),
        "n_test_identity_disjoint": len(test_idx),
        "id_128_beats_64": d["128"]["id_cosine"] > d["64"]["id_cosine"],
        "id_256_beats_128": d["256"]["id_cosine"] > d["128"]["id_cosine"],
        "dual_beats_all_baselines_id64": d["64"]["id_cosine"] > max(results[k]["rates"]["64"]["id_cosine"] for k in baselines),
        "dual_beats_all_baselines_id128": d["128"]["id_cosine"] > max(results[k]["rates"]["128"]["id_cosine"] for k in baselines),
        "dual_beats_all_baselines_nme64": d["64"]["nme106"] < min(results[k]["rates"]["64"]["nme106"] for k in baselines),
        "nme106_rel_drop_vs_mean_64": nme_drop_64,
        "geo_drop_ge_10pct": nme_drop_64 >= 0.10,
        "inv_helps_combined_noise_nme": results["dual_domain_full"]["noise_at_64"]["combined"]["nme106"]
        < results["dual_domain_no_inv"]["noise_at_64"]["combined"]["nme106"],
        "peak_vram_gb": peak_vram_gb(),
    }
    gate["phase1a_pass"] = bool(
        gate["id_128_beats_64"]
        and gate["dual_beats_all_baselines_id64"]
        and gate["dual_beats_all_baselines_id128"]
        and gate["geo_drop_ge_10pct"]
    )

    md = [
        "# Phase 1A Real Measurement-to-Condition Factorization (Loop B — Wiener Physics Lifting + ZCA Whitening)",
        "",
        f"Train: `{len(train_idx)}` real faces (FFHQ + CelebA train). Test: `{len(test_idx)}` strictly identity-disjoint faces (CelebA + CelebA-HQ).",
        f"Operator: Official binarized LSI `simulation_256.pt` (condition number `kappa = 2.38e8` before whitening).",
        "",
        "## Clean Measurement Rates (ID Cosine ↑ / 106-pt NME ↓)",
        "",
        "| Encoder | Trainable Params | ID@16 | ID@32 | ID@64 | ID@128 | ID@256 | NME@64 | NME@128 | NME@256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("mean", "mlp_raw", "mlp_whitened", "transformer1d_whitened", "dual_domain_no_inv", "dual_domain_full"):
        r = results[name]
        rt = r["rates"]
        md.append(
            f"| {name} | {r['trainable_parameters']:,} | "
            f"{rt['16']['id_cosine']:.4f} | {rt['32']['id_cosine']:.4f} | {rt['64']['id_cosine']:.4f} | "
            f"{rt['128']['id_cosine']:.4f} | {rt['256']['id_cosine']:.4f} | "
            f"{rt['64']['nme106']:.4f} | {rt['128']['nme106']:.4f} | {rt['256']['nme106']:.4f} |"
        )
    md += [
        "",
        "## Sensor Perturbation Robustness at M=64 (NME106 ↓ / ID Cosine ↑)",
        "",
        "| Encoder | Clean | Gaussian | Poisson+Gauss | Gain+Bias | 20% Dropout | Combined (N4) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("mlp_whitened", "transformer1d_whitened", "dual_domain_no_inv", "dual_domain_full"):
        ns = results[name]["noise_at_64"]
        md.append(
            f"| {name} | {ns['clean']['nme106']:.4f} / {ns['clean']['id_cosine']:.3f} | "
            f"{ns['gaussian']['nme106']:.4f} / {ns['gaussian']['id_cosine']:.3f} | "
            f"{ns['poisson_gaussian']['nme106']:.4f} / {ns['poisson_gaussian']['id_cosine']:.3f} | "
            f"{ns['gain_bias']['nme106']:.4f} / {ns['gain_bias']['id_cosine']:.3f} | "
            f"{ns['dropout']['nme106']:.4f} / {ns['dropout']['id_cosine']:.3f} | "
            f"{ns['combined']['nme106']:.4f} / {ns['combined']['id_cosine']:.3f} |"
        )
    md += ["", "## Gate Verdict", "```json", json.dumps(gate, indent=2), "```"]
    (OUTPUTS / "phase1a_real_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUTPUTS / "phase1a_real_results.json").write_text(json.dumps({"results": results, "gate": gate}, indent=2), encoding="utf-8")
    log("PHASE 1A REAL COMPLETE!")
    log(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
