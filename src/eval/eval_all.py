from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval.metrics import cosine_identity, measurement_rmse, nme, psnr, ssim
from src.paths import OUTPUTS


def evaluate_batch(pred: dict, target: dict, y_hat: torch.Tensor | None = None) -> dict:
    """One evaluation entry point. Identity teachers are chosen by the caller.

    `pred` / `target` may contain image, landmarks, id_adaface, id_buffalo, id_antelope.
    Missing keys are omitted. This function does not load a teacher.
    """
    report = {"n": int(pred["image"].shape[0])}
    report["psnr"] = float(psnr(pred["image"], target["image"]).mean())
    report["ssim"] = float(ssim(pred["image"], target["image"]).mean())
    if "landmarks" in pred and "landmarks" in target:
        k = pred["landmarks"].shape[1]
        left, right = (0, 1) if k <= 5 else (66, 75)
        if k > right:
            report["nme"] = float(nme(pred["landmarks"], target["landmarks"], left=left, right=right).mean())
    for key in ("id_adaface", "id_buffalo", "id_antelope"):
        if key in pred and key in target:
            report[key] = float(cosine_identity(pred[key], target[key]).mean())
    if y_hat is not None and "y" in target:
        report["measurement_rmse"] = float(measurement_rmse(y_hat, target["y"]).mean())
    report["lpips"] = None
    report["fid"] = None
    report["kid"] = None
    return report


def write_eval_json(report: dict, name: str = "eval_all.json") -> Path:
    path = OUTPUTS / name
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
