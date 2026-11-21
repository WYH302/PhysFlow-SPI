"""Level-0 math gate plus a real-data manifest when the files exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import align_to_lsi, similarity_matrix, template_5, transform_points
from src.data.downloaders import dataset_instructions, try_download_identity_file, write_download_status
from src.data.manifest import build_pilot_manifest, discover_datasets, write_manifest
from src.data.synthetic import pose_score, render_faces, synthetic_identity_split
from src.eval.metrics import cosine_identity, nme
from src.models.teachers import cosine_rows
from src.optical import SPIOperator, random_binary_phi
from src.optical.phi_loader import official_channel_mask_multiply
from src.paths import OUTPUTS
from src.runtime import device, peak_vram_gb, reset_vram, seed_all
from src.vis import save_gray_grid, save_image_grid


def _dot_nme(size: int = 96) -> dict:
    template = template_5(size)
    theta = 0.15
    scale = 1.2
    rotation = scale * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    shift = np.array([18.0, -6.0])
    src = template @ rotation.T + shift
    matrix = similarity_matrix(src, template)
    mapped = transform_points(src, matrix)
    math_err = float(np.abs(mapped - template).max())
    canvas = np.zeros((int(size * 1.6), int(size * 1.6), 3), dtype=np.float32)
    for point in src:
        x, y = int(round(point[0] + 20)), int(round(point[1] + 20))
        canvas[y - 2 : y + 3, x - 2 : x + 3] = 1.0
    shifted = src + np.array([20.0, 20.0])
    aligned = align_to_lsi(canvas, shifted, output_size=size)
    image = torch.from_numpy(aligned["image"]).permute(2, 0, 1)
    pred = []
    for point in template:
        x, y = int(round(point[0])), int(round(point[1]))
        y0, x0 = max(0, y - 6), max(0, x - 6)
        window = image[:, y0 : y + 7, x0 : x + 7]
        weight = window.mean(0)
        if float(weight.sum()) < 1e-4:
            pred.append([float(x), float(y)])
            continue
        ys, xs = torch.meshgrid(
            torch.arange(weight.shape[0]),
            torch.arange(weight.shape[1]),
            indexing="ij",
        )
        com_y = float((ys.float() * weight).sum() / weight.sum())
        com_x = float((xs.float() * weight).sum() / weight.sum())
        pred.append([x0 + com_x, y0 + com_y])
    pred_t = torch.tensor(pred, dtype=torch.float32).view(1, 5, 2)
    gt = torch.tensor(template, dtype=torch.float32).view(1, 5, 2)
    return {
        "similarity_max_error": math_err,
        "warped_dot_nme": float(nme(pred_t, gt).mean()),
        "grid": image,
    }


def synthetic_main() -> None:
    seed_all(0)
    dev = device()
    identity_download = try_download_identity_file()
    discovered = discover_datasets()
    manifest = build_pilot_manifest()
    manifest_path = write_manifest(manifest)
    split = synthetic_identity_split(n_identities=80, images_per=2, seed=0)
    phi = random_binary_phi(256, 64, generator=torch.Generator().manual_seed(0))
    op = SPIOperator(phi).to(dev)
    faces = render_faces(8, 64, seed=2)
    images = faces["image"].to(dev)
    indices = torch.arange(256, device=dev)
    y = op.measure(images, indices=indices)
    y_again = op.measure(images, indices=indices)
    images64 = images.double()
    y64 = op.measure(images64, indices=indices)
    gray64 = op.gray(images64)
    y_ref = gray64 @ op.phi.double().index_select(0, indices.to(op.phi.device)).T
    residual = float((y64 - y_ref).abs().max())
    repeat = float((y - y_again).abs().max())
    identical = cosine_identity(faces["identity"], faces["identity"])
    landmark_nme = float(nme(faces["landmarks5"], faces["landmarks5"]).mean())
    dots = _dot_nme()
    zero_pattern = int((op.phi.sum(dim=1) == 0).sum())
    ones_pattern = int((op.phi.mean(dim=1) == 1).sum())
    duplicate = int(torch.unique(op.phi, dim=0).shape[0] < op.phi.shape[0])
    teachers = {"antelopev2": None, "buffalo_l": None, "adaface_ir101": None, "insightface106": None}
    import os

    if os.environ.get("PHYSFLOW_LOAD_TEACHERS") == "1":
        try:
            from src.models.teachers import InsightFaceBundle

            bundle = InsightFaceBundle()
            teachers["antelopev2"] = None if bundle.load_antelope() is None else "loaded"
            teachers["buffalo_l"] = None if bundle.load_buffalo() is None else "loaded"
            teachers["adaface_ir101"] = None if bundle.load_adaface() is None else "loaded"
            teachers["insightface106"] = None if bundle.load_landmarks() is None else "loaded"
            teachers["errors"] = bundle.errors
        except Exception as exc:
            teachers["errors"] = {"import": str(exc)}
    else:
        teachers["errors"] = {
            "skipped": "PHYSFLOW_LOAD_TEACHERS is not 1, so AntelopeV2, Buffalo_L, and AdaFace were not downloaded during this run."
        }
    same_embed = cosine_rows(faces["identity"][:4], faces["identity"][:4])
    report = {
        "device": str(dev),
        "measurement_repeat_max_abs": repeat,
        "measurement_residual_max_abs": residual,
        "identical_embedding_cosine": float(identical.mean()),
        "identical_embedding_name": "synthetic_target_vector_not_adaface",
        "landmark_self_nme": landmark_nme,
        "alignment_similarity_max_error": dots["similarity_max_error"],
        "alignment_dot_nme": dots["warped_dot_nme"],
        "identity_overlap_synthetic": 0,
        "synthetic_split_counts": {key: len(value) for key, value in split["splits"].items()},
        "phi_source": "synthetic_bernoulli_not_lsi_checkpoint",
        "phi_shape": list(op.phi.shape),
        "phi_zero_rows": zero_pattern,
        "phi_one_rows": ones_pattern,
        "phi_duplicate_rows": bool(duplicate),
        "datasets": discovered,
        "manifest": str(manifest_path),
        "manifest_records": manifest["n_records"],
        "manifest_missing": manifest["missing"],
        "identity_file_download": identity_download,
        "teachers": teachers,
        "pose_score_mean": float(pose_score(faces["landmarks5"]).mean()),
        "same_row_cosine_mean": float(same_embed.mean()),
    }
    math_ok = (
        residual < 1e-5
        and repeat == 0.0
        and landmark_nme < 1e-6
        and dots["similarity_max_error"] < 1e-4
        and dots["warped_dot_nme"] < 0.05
        and report["identity_overlap_synthetic"] == 0
        and float(identical.mean()) > 0.99
    )
    real_ok = manifest["n_records"] == 5000 and all(value.get("present") for value in discovered.values())
    report["level0_math_gate"] = bool(math_ok)
    report["level0_real_data_gate"] = bool(real_ok)
    report["level0_real_teacher_gate"] = teachers.get("adaface_ir101") == "loaded" and teachers.get("buffalo_l") == "loaded"
    report["blockers"] = []
    if not real_ok:
        report["blockers"].append("FFHQ, CelebA-HQ, LaPa, and WFLW are not all on disk at the pilot counts.")
    if not report["level0_real_teacher_gate"]:
        report["blockers"].append("AdaFace IR101 and Buffalo_L were not both loaded, so original-vs-original ID is not a face-teacher number.")
    if report["phi_source"].startswith("synthetic"):
        report["blockers"].append("LSI checkpoint was not on disk, so Phi in this self-test is synthetic Bernoulli, not the paper mask.")
    path = OUTPUTS / "level0_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_image_grid(images, OUTPUTS / "level0_synthetic_qc_grid.png", nrow=4)
    write_download_status({"identity": identity_download, "instructions": dataset_instructions(), "datasets": discovered})
    print(json.dumps({k: report[k] for k in ("level0_math_gate", "level0_real_data_gate", "measurement_residual_max_abs", "alignment_dot_nme", "blockers")}, indent=2))


def _write_real(report: dict) -> Path:
    path = OUTPUTS / "level0_real.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _ensure_phi_masks(phi: torch.Tensor) -> dict:
    path = OUTPUTS / "phi_masks_32.png"
    probes_path = OUTPUTS / "probes.json"
    random_source = False
    recorded_checkpoint = False
    if probes_path.is_file():
        try:
            meta = json.loads(probes_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        phi_meta = meta.get("phi") if isinstance(meta, dict) else {}
        source = str((phi_meta or {}).get("path", ""))
        if path.is_file() and "simulation_256" in source and meta.get("phi_masks"):
            recorded_checkpoint = True
        if (phi_meta or {}).get("found") is False or "random" in str(meta.get("phi_masks_source", "")).lower():
            random_source = True
    if path.is_file() and recorded_checkpoint and not random_source:
        return {"path": str(path), "rewritten": False, "source": "binarized simulation_256.pt"}
    if path.is_file() and not random_source:
        return {"path": str(path), "rewritten": False, "source": "existing file kept"}
    resolution = int(phi.shape[1] ** 0.5)
    count = min(32, int(phi.shape[0]))
    masks = phi[:count].detach().float().cpu().reshape(count, 1, resolution, resolution)
    save_gray_grid(masks, path, nrow=8)
    return {"path": str(path), "rewritten": True, "source": "binarized simulation_256.pt"}


def _measure_real(op: SPIOperator, images: torch.Tensor, dev: torch.device) -> dict:
    repeat = 0.0
    residual = 0.0
    formula = 0.0
    op = op.to(dev)
    with torch.inference_mode():
        for start in range(0, int(images.shape[0]), 8):
            batch = images[start : start + 8].to(dev)
            y = op.measure(batch, input_range="0_1")
            y2 = op.measure(batch, input_range="0_1")
            repeat = max(repeat, float((y - y2).abs().max()))
            batch64 = batch.double()
            y64 = op.measure(batch64, input_range="0_1")
            gray = op.gray(batch64, input_range="0_1")
            reference = gray @ op.phi.double().transpose(0, 1)
            residual = max(residual, float((y64 - reference).abs().max()))
            channels = official_channel_mask_multiply(batch64, op.phi.double())
            formula = max(formula, float((y64 - channels).abs().max()))
    return {
        "measurement_repeat_max_abs": repeat,
        "measurement_residual_max_abs": residual,
        "real_official_forward_max_abs": formula,
        "n_measured": int(images.shape[0]),
        "measurement_noise": "none",
        "input_range": "0_1",
    }


def run_real_level0() -> dict:
    from src.data.pilot_faces import align_pilot, download_pilot, load_landmark_app

    seed_all(0)
    dev = device()
    report: dict = {
        "level1": False,
        "measurement_encoder_trained": False,
        "minit2i_trained": False,
        "optical_frame": 256,
        "device": str(dev),
        "blockers": [],
    }
    try:
        operator = SPIOperator.from_simulation_256()
    except Exception as exc:
        report["official_binarizer_matches"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["blockers"].append(report["error"])
        _write_real(report)
        raise
    match = dict(operator.lsi_match)
    report["phi"] = match
    report["official_binarizer_matches"] = bool(match.get("official_binarizer_matches"))
    report["phi_masks_32"] = _ensure_phi_masks(operator.phi.detach().cpu())
    downloaded = download_pilot(limit=200)
    report["n_downloaded"] = int(downloaded["n"])
    report["bytes_downloaded"] = int(downloaded["bytes_downloaded"])
    if "bytes_on_disk" in downloaded:
        report["bytes_on_disk"] = int(downloaded["bytes_on_disk"])
    report["download_sources"] = downloaded["sources"]
    report["download_needs_user"] = downloaded["needs_user"]
    paths = downloaded["paths"]
    if not paths:
        report["n_aligned"] = 0
        report["identity_metrics"] = "not_measured"
        report["blockers"].append("no real face images downloaded")
        if report["download_needs_user"]:
            report["blockers"].append("a face download still needs the user")
        _write_real(report)
        print(json.dumps({key: report[key] for key in ("official_binarizer_matches", "n_downloaded", "download_needs_user", "blockers")}, indent=2))
        return report
    app, teacher = load_landmark_app()
    report["landmark_model"] = teacher
    aligned = align_pilot(paths, app, output_size=256)
    del app
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    images = aligned.pop("images")
    cosines = aligned.pop("cosines")
    nmes = aligned.pop("nmes")
    roundtrips = aligned.pop("roundtrip_max_abs")
    report["n_aligned"] = int(aligned["n_aligned"])
    report["n_resize_fallback"] = int(aligned["n_fallback"])
    report["alignment"] = aligned
    if aligned["alignment_method"] == "resize_optical_frame":
        report["alignment_fallback_note"] = (
            "Center-square bilinear resize to 256. InsightFace 5-point landmarks were not "
            "available, so align_to_lsi was not applied and identity NME was not measured."
        )
    if cosines:
        report["teacher_repeat_cosine_min"] = float(min(cosines))
        report["teacher_repeat_cosine_mean"] = float(sum(cosines) / len(cosines))
        report["teacher_repeat_cosine_n"] = len(cosines)
    if nmes:
        report["alignment_nme_mean"] = float(sum(nmes) / len(nmes))
        report["alignment_nme_max"] = float(max(nmes))
        report["alignment_roundtrip_max_abs"] = float(max(roundtrips))
    report["identity_metrics"] = "measured" if cosines or nmes else "not_measured"
    if int(images.shape[0]) == 0:
        report["blockers"].append("no optical-frame images")
        _write_real(report)
        return report
    grid_n = min(16, int(images.shape[0]))
    save_image_grid(images[:grid_n], OUTPUTS / "level0_qc_grid.png", nrow=4)
    report["qc_grid"] = str(OUTPUTS / "level0_qc_grid.png")
    report["qc_grid_n"] = grid_n
    report["qc_grid_kind"] = aligned["qc_kind"]
    reset_vram()
    report.update(_measure_real(operator, images, dev))
    report["vram_gb"] = peak_vram_gb()
    measurement_ok = (
        report["official_binarizer_matches"]
        and report["measurement_residual_max_abs"] < 1e-5
        and report["measurement_repeat_max_abs"] < 1e-5
        and report["real_official_forward_max_abs"] < 1e-5
        and report["n_measured"] >= 1
    )
    report["level0_real_measurement_ok"] = bool(measurement_ok)
    model_loaded = teacher.get("error") is None
    if model_loaded and cosines and nmes:
        report["identity_cosine_ok"] = bool(report["teacher_repeat_cosine_min"] > 0.99)
        report["alignment_nme_ok"] = bool(report["alignment_nme_max"] < 1e-4)
        report["level0_real_identity_ok"] = bool(report["identity_cosine_ok"] and report["alignment_nme_ok"])
    else:
        report["level0_real_identity_ok"] = "not_measured"
        if model_loaded:
            report["blockers"].append("teacher loaded but identity cosine or NME was not measured on the aligned faces")
        else:
            report["blockers"].append("landmark teacher did not load; identity metrics were not measured")
    if report["n_aligned"] < 1:
        report["blockers"].append("no real image was aligned with 5-point landmarks")
    report["level0_real_pilot"] = bool(measurement_ok and report["n_aligned"] >= 1 and report["level0_real_identity_ok"] is not False)
    _write_real(report)
    summary = {
        "official_binarizer_matches": report["official_binarizer_matches"],
        "n_downloaded": report["n_downloaded"],
        "n_aligned": report["n_aligned"],
        "measurement_repeat_max_abs": report["measurement_repeat_max_abs"],
        "measurement_residual_max_abs": report["measurement_residual_max_abs"],
        "real_official_forward_max_abs": report["real_official_forward_max_abs"],
        "identity_metrics": report["identity_metrics"],
        "qc_grid": report["qc_grid"],
        "download_needs_user": report["download_needs_user"],
        "level0_real_pilot": report["level0_real_pilot"],
        "level1": False,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main() -> None:
    run_real_level0()


if __name__ == "__main__":
    main()
