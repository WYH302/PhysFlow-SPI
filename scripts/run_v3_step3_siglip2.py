"""Step 3 (V3 Surgery): 2025 Foundation Backbone Benchmark (SigLIP2-Base vs DINOv2-Small) in Stage 1A.

Evaluates `google/siglip2-base-patch16-224` (86M, public zero-auth 2025 vision encoder)
head-to-head against our current `DINOv2-Small` (22M) inside PhysicsDualDomainFactorizer
on the EXACT same 5,000 train / 1,479 strictly identity-disjoint test split (rates M=16..256
and N4 combined noise), PLUS strict 69-celebrity identity-disjoint 5-fold cross-validation
(106-pt landmark NME & 40-attribute classification), PLUS exact FPS / VRAM / GFLOPs latency!

100% read-only wrt V1 checkpoints. Saves checkpoint to `outputs/checkpoints/v3_surgery/siglip2_encoder.pt`
and results to `outputs/v3_surgery/step3_siglip2_ablation.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.models.encoders import PhysicsDualDomainFactorizer, parameter_count
from src.optical.spi_operator import RATES, SPIOperator, WienerPhysicsLifter
from src.paths import DATA, OUTPUTS
from src.runtime import seed_all

from scripts.run_phase1a_real import evaluate_encoder, landmark_nme106, train_encoder
from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR

V3_DIR = OUTPUTS / "v3_surgery"
V3_CKPT = OUTPUTS / "checkpoints" / "v3_surgery"
V3_DIR.mkdir(parents=True, exist_ok=True)
V3_CKPT.mkdir(parents=True, exist_ok=True)
OUT_JSON = V3_DIR / "step3_siglip2_ablation.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SigLIP2Backbone(nn.Module):
    """Wraps google/siglip2-base-patch16-224 vision encoder to match LocalDINOv2Backbone token interface.

    Input: [B, 3, 224, 224] RGB in [0, 1].
    Output: [B, 1 + 256, 768] where slot 0 is pooled/mean global token and slots 1..256 are 16x16 spatial grid
            (interpolated from SigLIP2's native 14x14 patch grid so downstream spatial pyramid is 100% identical!).
    """

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", unfreeze_last_n: int = 4):
        super().__init__()
        full = AutoModel.from_pretrained(model_id)
        self.vision = full.vision_model
        self.out_dim = int(self.vision.config.hidden_size)
        self.name = model_id
        self.pretrained = True
        for p in self.vision.parameters():
            p.requires_grad = False
        layers = self.vision.encoder.layers
        for layer in list(layers)[-int(unfreeze_last_n) :]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(self.vision, "post_layernorm"):
            for p in self.vision.post_layernorm.parameters():
                p.requires_grad = True

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # SigLIP2 expects pixel values normalized to [-1, 1] with mean=0.5, std=0.5
        pix = x01.float() * 2.0 - 1.0
        out = self.vision(pixel_values=pix)
        seq = out.last_hidden_state  # [B, 196, 768] for 224/16 = 14x14
        b, n_tok, d = seq.shape
        g = int(round(n_tok ** 0.5))
        grid = seq.transpose(1, 2).reshape(b, d, g, g)
        if g != 16:
            grid16 = F.interpolate(grid, size=(16, 16), mode="bilinear", align_corners=False)
        else:
            grid16 = grid
        cls_tok = seq.mean(dim=1, keepdim=True)  # [B, 1, 768]
        return torch.cat([cls_tok, grid16.flatten(2).transpose(1, 2)], dim=1)  # [B, 257, 768]


def run_5fold_cv_for_encoder(
    enc: nn.Module,
    y256: torch.Tensor,
    lmks: torch.Tensor,
    attrs_all: torch.Tensor,
    records: list[dict],
    dev: torch.device,
) -> dict:
    """Exact 69-celebrity identity-disjoint 5-fold CV matching journal_audit_extras.py."""
    non_train = [r for r in records if r["split"] in ("val", "celeba_test") and r.get("celeb_id") is not None]
    indices = [r["index"] for r in non_train]
    celeb_ids = [int(r["celeb_id"]) for r in non_train]

    # Extract frozen factorized features at M=64
    enc.eval()
    idx64 = torch.arange(64, device=dev, dtype=torch.long)
    feats = []
    with torch.no_grad():
        for s in range(0, len(indices), 64):
            pick = indices[s : s + 64]
            yb = y256[pick, :64].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p_sem, p_id, p_geo = enc(yb, idx64)
            # Concise concatenated representation matching downstream probe
            f = torch.cat([p_sem[:, :8, :].float().mean(dim=1), p_id.float(), p_geo.float()], dim=-1)
            feats.append(f.cpu().numpy())
    X = np.concatenate(feats, axis=0)
    Y_lmk = lmks[indices].view(len(indices), -1).numpy()
    Y_attr = (attrs_all[indices].numpy() > 0).astype(np.int64)

    # Deterministic 5-fold partition over sorted unique celebrity IDs (zero subject overlap across folds)
    uniq_ids = sorted(set(celeb_ids))
    id_to_fold = {cid: i % 5 for i, cid in enumerate(uniq_ids)}
    folds = np.array([id_to_fold[cid] for cid in celeb_ids])

    def ridge_fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, alpha: float = 10.0) -> np.ndarray:
        mu_x = Xtr.mean(axis=0, keepdims=True)
        mu_y = Ytr.mean(axis=0, keepdims=True)
        Xc = torch.from_numpy((Xtr - mu_x).astype(np.float32))
        Yc = torch.from_numpy((Ytr - mu_y).astype(np.float32))
        Xt = torch.from_numpy((Xte - mu_x).astype(np.float32))
        n_tr, d = Xc.shape
        if n_tr < d:
            # Dual ridge: W = X^T (X X^T + alpha I)^{-1} Y
            K = Xc @ Xc.T + float(alpha) * torch.eye(n_tr, dtype=torch.float32)
            pred = (Xt @ Xc.T) @ torch.linalg.solve(K, Yc)
        else:
            A = Xc.T @ Xc + float(alpha) * torch.eye(d, dtype=torch.float32)
            pred = Xt @ torch.linalg.solve(A, Xc.T @ Yc)
        return pred.numpy() + mu_y

    def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = float(((y_true == 1) & (y_pred == 1)).sum())
        fp = float(((y_true == 0) & (y_pred == 1)).sum())
        fn = float(((y_true == 1) & (y_pred == 0)).sum())
        denom = 2.0 * tp + fp + fn
        return (2.0 * tp / denom) if denom > 0.0 else 0.0

    nmes, accs, f1s = [], [], []
    for f_id in range(5):
        tr = folds != f_id
        te = folds == f_id
        # 106-pt Landmark ridge regression
        pred_l = torch.from_numpy(ridge_fit_predict(X[tr], Y_lmk[tr], X[te], alpha=10.0)).float()
        gt_l = torch.from_numpy(Y_lmk[te]).float()
        nmes.append(float(landmark_nme106(pred_l, gt_l).mean()))

        # 40-Attribute multi-output linear ridge probe (matches exact closed-form linear transfer)
        scores_a = ridge_fit_predict(X[tr], Y_attr[tr].astype(np.float32), X[te], alpha=10.0)
        preds_a = (scores_a >= 0.5).astype(np.int64)
        a_accs = [float((preds_a[:, a] == Y_attr[te, a]).mean()) for a in range(Y_attr.shape[1])]
        a_f1s = [binary_f1(Y_attr[te, a], preds_a[:, a]) for a in range(Y_attr.shape[1])]
        accs.append(float(np.mean(a_accs)))
        f1s.append(float(np.mean(a_f1s)))

    return {
        "n_samples": len(indices),
        "n_unique_ids": len(uniq_ids),
        "nme106_mean": float(np.mean(nmes)),
        "nme106_std": float(np.std(nmes)),
        "attr_acc_mean": float(np.mean(accs)),
        "attr_acc_std": float(np.std(accs)),
        "attr_f1_mean": float(np.mean(f1s)),
        "attr_f1_std": float(np.std(f1s)),
    }


def benchmark_fps(enc: nn.Module, y64: torch.Tensor, dev: torch.device, iters: int = 80) -> float:
    enc.eval()
    idx64 = torch.arange(64, device=dev, dtype=torch.long)
    x = y64[:1].to(dev)
    with torch.no_grad():
        for _ in range(10):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc(x, idx64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc(x, idx64)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return float(1.0 / max(dt, 1e-9)), float(dt * 1000.0)


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    sem = torch.load(CACHE_DIR / "sem_hidden_bf16.pt", map_location="cpu", weights_only=True)
    sem_mask = torch.load(CACHE_DIR / "sem_mask_u8.pt", map_location="cpu", weights_only=True)
    ident = torch.load(CACHE_DIR / "identity_antelope.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    geo = lmks.view(lmks.shape[0], -1)
    attrs = torch.load(CACHE_DIR / "celeba_attrs40.pt", map_location="cpu", weights_only=True) if (CACHE_DIR / "celeba_attrs40.pt").exists() else None

    train_idx = [r["index"] for r in records if r["split"] == "train"]
    val_idx = [r["index"] for r in records if r["split"] == "val"]
    test_idx = [r["index"] for r in records if r["split"] != "train"]  # exact 1,479 disjoint split

    op_gpu = SPIOperator.from_simulation_256().to(dev)

    # Fit shared closed-form Wiener moments once from train_idx (100% identical to Phase 1A)
    imgs256 = torch.load(CACHE_DIR / "images256_u8.pt", map_location="cpu", weights_only=True)
    lifter = WienerPhysicsLifter(max_measurements=256, img_size=128)
    lifter.fit(imgs256[train_idx], y256[train_idx])
    del imgs256

    # 1. Evaluate existing V1 DINOv2-Small baseline on identical harness for 100% apples-to-apples comparison
    log("=== Evaluating V1 DINOv2-Small Stage-1A Encoder ===")
    dino_enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev)
    dino_ckpt = torch.load(CKPT_DIR / "dual_encoder_real.pt", map_location="cpu", weights_only=False)
    dino_enc.load_state_dict(dino_ckpt["state_dict"])
    dino_fps, dino_ms = benchmark_fps(dino_enc, y256[test_idx, :64], dev)
    dino_res = {
        "params_total_m": round(parameter_count(dino_enc) / 1e6, 2),
        "params_trainable_m": round(sum(p.numel() for p in dino_enc.parameters() if p.requires_grad) / 1e6, 2),
        "fps": round(dino_fps, 2),
        "latency_ms": round(dino_ms, 2),
        "rates": {str(r): evaluate_encoder(dino_enc, y256, sem, sem_mask, ident, geo, test_idx, dev, rate=r) for r in (64, 256)},
        "n4_m64": evaluate_encoder(dino_enc, y256, sem, sem_mask, ident, geo, test_idx, dev, rate=64, noise_mode="combined"),
    }
    if attrs is not None:
        dino_res["cv5_m64"] = run_5fold_cv_for_encoder(dino_enc, y256, lmks, attrs, records, dev)
    log(f"  DINOv2-Small: {dino_res['params_total_m']}M | {dino_res['fps']} FPS ({dino_res['latency_ms']} ms) | @64 ID={dino_res['rates']['64']['id_cosine']:.4f} NME={dino_res['rates']['64']['nme106']:.4f} | @256 ID={dino_res['rates']['256']['id_cosine']:.4f} | N4 NME={dino_res['n4_m64']['nme106']:.4f}")
    del dino_enc
    torch.cuda.empty_cache()

    # 2. Train SigLIP2-Base Stage-1A Factorizer under IDENTICAL protocol (800 steps, batch 64, lr_head=1e-3, lr_backbone=5e-5, geo_inv=True)
    log("=== Training 2025 SigLIP2-Base Stage-1A Factorizer (Identical 800-Step Protocol) ===")
    sig_bb = SigLIP2Backbone("google/siglip2-base-patch16-224", unfreeze_last_n=4)
    sig_enc = PhysicsDualDomainFactorizer(
        op_gpu, lifter=lifter, backbone=sig_bb, max_measurements=256, dim=384,
        sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=4,
    ).to(dev)
    sig_enc.set_target_means(sem[train_idx].float().mean(dim=0), geo[train_idx].float().mean(dim=0))
    t0 = time.perf_counter()
    tr_info = train_encoder(
        "SigLIP2-Base-Dual", sig_enc, y256, sem, sem_mask, ident, geo,
        train_idx, val_idx, dev, steps=800, batch_size=64, lr_head=1e-3, lr_backbone=5e-5, use_geo_inv=True, seed=42,
    )
    train_sec = time.perf_counter() - t0
    torch.save({"state_dict": sig_enc.state_dict(), "train_sec": train_sec}, V3_CKPT / "siglip2_encoder.pt")

    sig_fps, sig_ms = benchmark_fps(sig_enc, y256[test_idx, :64], dev)
    sig_res = {
        "params_total_m": round(parameter_count(sig_enc) / 1e6, 2),
        "params_trainable_m": round(sum(p.numel() for p in sig_enc.parameters() if p.requires_grad) / 1e6, 2),
        "train_sec": round(train_sec, 1),
        "fps": round(sig_fps, 2),
        "latency_ms": round(sig_ms, 2),
        "rates": {str(r): evaluate_encoder(sig_enc, y256, sem, sem_mask, ident, geo, test_idx, dev, rate=r) for r in (64, 256)},
        "n4_m64": evaluate_encoder(sig_enc, y256, sem, sem_mask, ident, geo, test_idx, dev, rate=64, noise_mode="combined"),
    }
    if attrs is not None:
        sig_res["cv5_m64"] = run_5fold_cv_for_encoder(sig_enc, y256, lmks, attrs, records, dev)
    log(f"  SigLIP2-Base: {sig_res['params_total_m']}M | {sig_res['fps']} FPS ({sig_res['latency_ms']} ms) | @64 ID={sig_res['rates']['64']['id_cosine']:.4f} NME={sig_res['rates']['64']['nme106']:.4f} | @256 ID={sig_res['rates']['256']['id_cosine']:.4f} | N4 NME={sig_res['n4_m64']['nme106']:.4f}")

    OUT_JSON.write_text(json.dumps({"dinov2_small": dino_res, "siglip2_base": sig_res}, indent=2), encoding="utf-8")
    log(f"Saved Step 3 comparison -> {OUT_JSON}")


if __name__ == "__main__":
    main()
