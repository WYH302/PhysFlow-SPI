"""External Benchmark Evaluation on Official LaPa (106-pt) and Official Uncorrupted WFLW/WIDER-FACE (6 Challenging Subsets).

Every image passes through the shared 5-point similarity `align_to_lsi(rgb, kps5, output_size=512)`
before computing y = Phi(I):
  - Official LaPa Test: aligned strictly via the annotated target face's official 106 GT landmarks
    ([104, 105, 54, 84, 90]) to avoid edge-bystander contamination; evaluated across all 5 rates.
  - Official WFLW 6 Challenging Subsets (pose, expression, illumination, makeup, occlusion, blur):
    loaded strictly from the original full-resolution uncorrupted RGB WIDER FACE JPEGs matched to
    the official WFLW test split flags (zero corrupted binary PNGs!), evaluated across all 5 rates.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from src.data.align import align_to_lsi, detect_face_robust
from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.lsi_baseline import LSIBaseline
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import RATES, SPIOperator
from src.paths import DATA, OUTPUTS, ROOT as PROJECT_ROOT
from src.vis import save_image_grid

from scripts.run_phase1b_oracle_real import CKPT_DIR, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import encode_generic_prompt, sample_mode

WFLW_SUBSETS = ("pose", "expression", "illumination", "makeup", "occlusion", "blur")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_lapa_samples(max_n: int = 32) -> list[dict]:
    root = DATA / "raw" / "lapa" / "extracted"
    if not root.exists():
        return []
    test_img_dirs = [p for p in root.rglob("images") if "test" in str(p).lower()]
    if not test_img_dirs:
        test_img_dirs = list(root.rglob("images"))
    if not test_img_dirs:
        return []
    img_dir = test_img_dirs[0]
    lmk_dir = img_dir.parent / "landmarks"
    samples = []
    for p in sorted(img_dir.glob("*.*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lmk_path = lmk_dir / (p.stem + ".txt")
        if not lmk_path.is_file():
            continue
        lines = [ln.strip() for ln in lmk_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        pts = []
        for ln in lines:
            toks = ln.split()
            if len(toks) == 2:
                try:
                    pts.append([float(toks[0]), float(toks[1])])
                except ValueError:
                    continue
        if len(pts) != 106:
            continue
        pts_arr = np.asarray(pts, dtype=np.float64)
        # Verify the labeled target face's 5 key points sit cleanly inside the image canvas
        w, h = Image.open(p).size
        kps5 = np.stack([pts_arr[104], pts_arr[105], pts_arr[54], pts_arr[84], pts_arr[90]], axis=0)
        iod = float(np.linalg.norm(kps5[0] - kps5[1]))
        if iod < 24.0:
            continue
        if (kps5[:, 0].min() < 2.0) or (kps5[:, 0].max() > w - 2.0) or (kps5[:, 1].min() < 2.0) or (kps5[:, 1].max() > h - 2.0):
            continue
        samples.append({"path": p, "kps5": kps5, "dataset": "lapa"})
        if len(samples) >= max_n * 2:
            break
    return samples


def load_wflw_official_rgb_samples(max_per_subset: int = 12) -> dict[str, list[dict]]:
    """Load genuine full-res RGB WIDER FACE JPEGs matched strictly to official WFLW test subset flags."""
    csv_path = DATA / "raw" / "wflw" / "extracted" / "test_data" / "labels.csv"
    wider_zip = DATA / "raw" / "wider_face" / "data" / "WIDER_val.zip"
    extract_dir = DATA / "raw" / "wider_face" / "wflw_matched_rgb"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file() or not wider_zip.is_file():
        raise FileNotFoundError(f"Missing WFLW labels ({csv_path}) or WIDER_val.zip ({wider_zip})")

    with zipfile.ZipFile(wider_zip) as zf:
        val_map = {Path(n).stem: n for n in zf.namelist() if n.endswith(".jpg")}
        by_subset: dict[str, list[dict]] = {s: [] for s in WFLW_SUBSETS}
        used_per_subset: dict[str, set[str]] = {s: set() for s in WFLW_SUBSETS}
        # Prefer unique images across subsets first so qualitative figure shows 6 totally distinct faces
        global_used: set[str] = set()
        candidates_by_sub: dict[str, list[tuple[str, str]]] = {s: [] for s in WFLW_SUBSETS}

        for ln in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            toks = ln.strip().split(",")
            if len(toks) < 206:
                continue
            fn = Path(toks[205]).stem
            m = re.match(r"^\d+_(.+)_\d+$", fn)
            base = m.group(1) if m else fn
            if base not in val_map:
                continue
            flags = [int(float(x)) for x in toks[196:202]]
            for idx_f, name_s in enumerate(WFLW_SUBSETS):
                if flags[idx_f] == 1 and base not in used_per_subset[name_s]:
                    used_per_subset[name_s].add(base)
                    candidates_by_sub[name_s].append((base, val_map[base]))

        # Two passes: prefer globally disjoint images first, then allow multi-attribute fallback if needed
        for sub_name in WFLW_SUBSETS:
            ordered = [x for x in candidates_by_sub[sub_name] if x[0] not in global_used] + [
                x for x in candidates_by_sub[sub_name] if x[0] in global_used
            ]
            for base, zmember in ordered:
                if len(by_subset[sub_name]) >= max_per_subset * 3:
                    break
                out_jpg = extract_dir / f"{base}.jpg"
                if not out_jpg.is_file():
                    out_jpg.write_bytes(zf.read(zmember))
                by_subset[sub_name].append({"path": out_jpg, "base": base, "dataset": "wflw"})
                global_used.add(base)
    return by_subset


def align_image_list(
    buf,
    raw_list: list[dict],
    op: SPIOperator,
    dev: torch.device,
    target_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned_imgs = []
    for s in raw_list:
        rgb = np.asarray(Image.open(s["path"]).convert("RGB"))
        # Sanity assertion: never allow corrupted binary {0,1} images!
        if int(rgb.max()) < 30 or len(np.unique(rgb)) < 16:
            continue
        if "kps5" in s:
            kps5 = s["kps5"]
        else:
            det = detect_face_robust(buf, rgb, prefer_center=False)
            if det is None:
                continue
            kps5 = det["kps5"]
            # Require a clearly resolved face (inter-ocular >= 20 px) in the high-res WIDER photo
            if float(np.linalg.norm(kps5[0] - kps5[1])) < 20.0:
                continue
        al = align_to_lsi(rgb, kps5, output_size=512)
        arr_u8 = (al["image"] * 255.0).round().clip(0, 255).astype(np.uint8)
        # Confirm Buffalo_L cleanly detects the centered aligned GT face so NME is 100% well-defined
        chk = detect_face_robust(buf, arr_u8, prefer_center=True)
        if chk is None or chk["lmk106"] is None:
            continue
        t_img = torch.from_numpy(al["image"].transpose(2, 0, 1).copy()).float().mul(2.0).sub(1.0)
        aligned_imgs.append(t_img)
        if len(aligned_imgs) >= target_n:
            break
    if not aligned_imgs:
        raise RuntimeError("No valid faces aligned!")
    gt_t = torch.stack(aligned_imgs, dim=0)
    with torch.no_grad():
        y256 = op.measure(gt_t.to(dev), indices=None, input_range="-1_1").cpu()
    return gt_t, y256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lapa-n", type=int, default=32)
    parser.add_argument("--wflw-n", type=int, default=12)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--wflw-steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.85)
    parser.add_argument("--t-warm", type=float, default=0.35)
    args = parser.parse_args()

    dev = torch.device("cuda") if (torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / (1024**3) >= 5.0) else torch.device("cpu")
    log(f"External LaPa & Official WFLW/WIDER-RGB Evaluation starting on {dev}")

    os.environ["INSIGHTFACE_HOME"] = str((PROJECT_ROOT / "checkpoints" / "insightface").resolve())
    from insightface.app import FaceAnalysis

    buf = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], root=str((PROJECT_ROOT / "checkpoints" / "insightface").resolve()))
    buf.prepare(ctx_id=-1, det_size=(640, 640))

    op = SPIOperator.from_simulation_256().to(dev)
    gen_txt, gen_msk = encode_generic_prompt(dev, max_len=64)

    lapa_raw = load_lapa_samples(args.lapa_n)
    lapa_gt, lapa_y256 = align_image_list(buf, lapa_raw, op, dev, target_n=args.lapa_n)
    log(f"Aligned clean LaPa test faces: {len(lapa_gt)}")

    wflw_raw = load_wflw_official_rgb_samples(args.wflw_n)
    wflw_aligned: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    gt_qc_tiles, gt_qc_titles = [], []
    for sub in WFLW_SUBSETS:
        gt_s, y_s = align_image_list(buf, wflw_raw[sub], op, dev, target_n=args.wflw_n)
        wflw_aligned[sub] = (gt_s, y_s)
        log(f"Aligned clean RGB WFLW[{sub:12s}]: {len(gt_s)}")
        for k in range(min(2, len(gt_s))):
            gt_qc_tiles.append(gt_s[k])
            gt_qc_titles.append(f"WFLW:{sub}")
    save_image_grid(torch.stack(gt_qc_tiles), OUTPUTS / "external_wflw_gt_qc.png", nrow=6, titles=gt_qc_titles)
    log("Saved WFLW GT visual QC grid -> outputs/external_wflw_gt_qc.png")

    del buf
    gc.collect()

    # Stage 1: Untouched frozen MiniT2I-B/16 -> Zero-Shot FLOWER on LaPa + WFLW
    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
    wrapper = wrapper.to(dev)
    net = wrapper.model.net
    wrapper.requires_grad_(False)

    log("=== Stage 1: Zero-Shot FLOWER on Untouched Frozen MiniT2I-B/16 ===")
    lapa_imgs: dict[str, dict[int, torch.Tensor]] = {"zeroshot_flower": {}}
    for rate in RATES:
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        lapa_imgs["zeroshot_flower"][rate], _ = sample_mode(
            "zeroshot_flower", net, None, None, op, lapa_y256[:, :rate], idx_r,
            gen_txt.expand(len(lapa_gt), -1, -1), gen_msk.expand(len(lapa_gt), -1),
            gen_txt, gen_msk, dev, steps=args.steps, cfg_scale=args.cfg, eta=args.eta, t_warm=0.0, batch_size=8,
        )
    wflw_imgs: dict[str, dict[str, dict[int, torch.Tensor]]] = {sub: {"zeroshot_flower": {}} for sub in WFLW_SUBSETS}
    for sub in WFLW_SUBSETS:
        gt_s, y_s = wflw_aligned[sub]
        for rate in RATES:
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            wflw_imgs[sub]["zeroshot_flower"][rate], _ = sample_mode(
                "zeroshot_flower", net, None, None, op, y_s[:, :rate], idx_r,
                gen_txt.expand(len(gt_s), -1, -1), gen_msk.expand(len(gt_s), -1),
                gen_txt, gen_msk, dev, steps=args.wflw_steps, cfg_scale=args.cfg, eta=args.eta, t_warm=0.0, batch_size=12,
            )
    log("  Zero-Shot FLOWER on LaPa & WFLW complete.")

    # Stage 2: PhysFlow E2E Modes on LaPa (all 5 rates) & WFLW (6 subsets x all 5 rates)
    log("=== Stage 2: PhysFlow Modes on LaPa & WFLW ===")
    attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    conditioner = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev)
    encoder = PhysicsDualDomainFactorizer(op, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev)
    load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, conditioner)
    encoder.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])

    for mode in ("b0_continuous", "b1_id", "b2_geo", "b3_physflow"):
        lapa_imgs[mode] = {}
        for rate in RATES:
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            lapa_imgs[mode][rate], _ = sample_mode(
                mode, net, conditioner, encoder, op, lapa_y256[:, :rate], idx_r,
                gen_txt.expand(len(lapa_gt), -1, -1), gen_msk.expand(len(lapa_gt), -1),
                gen_txt, gen_msk, dev, steps=args.steps, cfg_scale=args.cfg, eta=args.eta, t_warm=args.t_warm, batch_size=8,
            )
        log(f"  LaPa {mode:14s} all 5 rates complete.")

    for sub in WFLW_SUBSETS:
        gt_s, y_s = wflw_aligned[sub]
        for mode in ("b0_continuous", "b3_physflow"):
            wflw_imgs[sub][mode] = {}
            for rate in RATES:
                idx_r = torch.arange(rate, device=dev, dtype=torch.long)
                wflw_imgs[sub][mode][rate], _ = sample_mode(
                    mode, net, conditioner, encoder, op, y_s[:, :rate], idx_r,
                    gen_txt.expand(len(gt_s), -1, -1), gen_msk.expand(len(gt_s), -1),
                    gen_txt, gen_msk, dev, steps=args.wflw_steps, cfg_scale=args.cfg, eta=args.eta, t_warm=args.t_warm, batch_size=12,
                )
        log(f"  WFLW[{sub:12s}] all 5 rates complete.")

    wrapper.to("cpu")
    encoder.to("cpu")
    conditioner.to("cpu")
    del wrapper, encoder, conditioner
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 3: Official LSI on LaPa (all 5 rates) & WFLW (M=64, M=256)
    log("=== Stage 3: Official LSI Baseline on LaPa & WFLW ===")
    lsi_dev = dev if (dev.type == "cuda" and torch.cuda.mem_get_info()[0] / (1024**3) >= 4.0) else torch.device("cpu")
    lsi = LSIBaseline(device=lsi_dev)
    y_train_mean = torch.load(DATA / "cache" / "physflow_real" / "y256.pt", map_location="cpu", weights_only=True).float().mean(dim=0)
    lapa_imgs["lsi"] = {}
    for rate in RATES:
        lapa_imgs["lsi"][rate], _ = lsi.reconstruct(lapa_y256[:, :rate], indices=torch.arange(rate), y_mean=y_train_mean, output_size=512, batch_size=8)
    for sub in WFLW_SUBSETS:
        _, y_s = wflw_aligned[sub]
        wflw_imgs[sub]["lsi"] = {}
        for rate in (64, 256):
            wflw_imgs[sub]["lsi"][rate], _ = lsi.reconstruct(y_s[:, :rate], indices=torch.arange(rate), y_mean=y_train_mean, output_size=512, batch_size=8)
    del lsi
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 4: Independent Cross-Model Evaluation (center-locked Buffalo_L + AdaFace IR-101 + LPIPS + PSNR + RelRes)
    log("=== Stage 4: Evaluating LaPa & WFLW with Independent Evaluators ===")
    evaluator = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)
    op_cpu = SPIOperator.from_simulation_256()

    out_report: dict = {"lapa": {"present": True, "n": len(lapa_gt), "rates": {}}, "wflw_6subsets": {}}
    for mode in ("b0_continuous", "zeroshot_flower", "b1_id", "b2_geo", "lsi", "b3_physflow"):
        out_report["lapa"]["rates"][mode] = {}
        for rate in RATES:
            r = evaluator.evaluate_pair(
                lapa_imgs[mode][rate], lapa_gt, operator=op_cpu, y_gt=lapa_y256[:, :rate], indices=torch.arange(rate), gt_landmarks_px=None
            )
            out_report["lapa"]["rates"][mode][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}

    tiles, titles = [], []
    for i in range(min(8, len(lapa_gt))):
        tiles.extend([lapa_gt[i], lapa_imgs["zeroshot_flower"][64][i], lapa_imgs["lsi"][64][i], lapa_imgs["b3_physflow"][64][i]])
        titles.extend(["LaPa GT", "ZS-FLOWER", "LSI@64", "PhysFlow@64"])
    save_image_grid(torch.stack(tiles), OUTPUTS / "external_lapa_m64.png", nrow=4, titles=titles)

    qual_tiles, qual_titles = [], []
    for sub in WFLW_SUBSETS:
        gt_s, y_s = wflw_aligned[sub]
        out_report["wflw_6subsets"][sub] = {"n": len(gt_s), "rates": {}}
        for mode in ("b0_continuous", "zeroshot_flower", "b3_physflow"):
            out_report["wflw_6subsets"][sub]["rates"][mode] = {}
            for rate in RATES:
                r = evaluator.evaluate_pair(
                    wflw_imgs[sub][mode][rate], gt_s, operator=op_cpu, y_gt=y_s[:, :rate], indices=torch.arange(rate), gt_landmarks_px=None
                )
                out_report["wflw_6subsets"][sub]["rates"][mode][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}
        out_report["wflw_6subsets"][sub]["rates"]["lsi"] = {}
        for rate in (64, 256):
            r = evaluator.evaluate_pair(
                wflw_imgs[sub]["lsi"][rate], gt_s, operator=op_cpu, y_gt=y_s[:, :rate], indices=torch.arange(rate), gt_landmarks_px=None
            )
            out_report["wflw_6subsets"][sub]["rates"]["lsi"][str(rate)] = {k: v for k, v in r.items() if k != "per_sample"}

        for i in range(min(2, len(gt_s))):
            qual_tiles.extend([gt_s[i], wflw_imgs[sub]["zeroshot_flower"][64][i], wflw_imgs[sub]["lsi"][64][i], wflw_imgs[sub]["b3_physflow"][64][i]])
            qual_titles.extend([f"{sub}:GT", "ZS-FLOWER", "LSI@64", "PhysFlow@64"])

    if qual_tiles:
        save_image_grid(torch.stack(qual_tiles), OUTPUTS / "external_wflw_6subsets_m64.png", nrow=8, titles=qual_titles)

    (OUTPUTS / "phase3_external_lapa_wflw.json").write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    log("EXTERNAL CLEAN LAPA & WFLW 6-SUBSET EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
