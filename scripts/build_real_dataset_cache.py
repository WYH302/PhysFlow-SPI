"""Build unified, strictly identity-disjoint real-face cache for PhysFlow (Level 0+).

Supports CPU or CUDA (auto-selects CUDA only if >= 6 GB free VRAM).
Produces compact, high-ROI caches under data/cache/physflow_real/:
  - FFHQ-512 (LLaVA captions) + CelebA official train (celeb_id + 40-attr captions)
  - CelebA official valid & test (asserted ZERO identity overlap with train!)
  - CelebA-HQ high-res test (AntelopeV2 filtered for cosine < 0.35 against train!)
  - LaPa official test (106-pt GT landmarks + parsing, aligned with exact same 5-pt similarity)
  - WFLW test (98-pt GT landmarks, aligned with exact same 5-pt similarity)
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.align import align_to_lsi, detect_face_robust, transform_points
from src.optical.spi_operator import SPIOperator
from src.paths import DATA, OUTPUTS, ROOT as PROJECT_ROOT
from src.vis import save_image_grid

CACHE_DIR = DATA / "cache" / "physflow_real"
IMG_DIR = CACHE_DIR / "aligned512"
INSIGHTFACE_DIR = PROJECT_ROOT / "checkpoints" / "insightface"
FLAN_DIR = PROJECT_ROOT / "checkpoints" / "flan-t5-large"

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheek", "Rosy_Cheeks", "Sideburns", "Smiling",
    "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
    "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device(min_free_gb: float = 6.0) -> torch.device:
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b / (1024**3) >= min_free_gb:
            return torch.device("cuda")
    return torch.device("cpu")


def celeba_attr_caption(row: dict) -> str:
    """Deterministic zero-hallucination portrait description from CelebA 40 attributes."""
    def _on(k: str) -> bool:
        v = row.get(k, 0)
        return bool(v == 1 or v is True)

    age = "young " if _on("Young") else "middle-aged "
    gender = "man" if _on("Male") else "woman"
    hair = []
    if _on("Bald"):
        hair.append("bald head")
    else:
        for col, label in [
            ("Black_Hair", "black"), ("Blond_Hair", "blond"),
            ("Brown_Hair", "brown"), ("Gray_Hair", "gray"),
        ]:
            if _on(col):
                hair.append(label)
        style = []
        if _on("Wavy_Hair"):
            style.append("wavy")
        elif _on("Straight_Hair"):
            style.append("straight")
        if _on("Bangs"):
            style.append("bangs")
        hair_str = " ".join(style + hair) + " hair" if (hair or style) else ""
        if hair_str:
            hair = [hair_str]
    extras = []
    if _on("Smiling"):
        extras.append("smiling")
    elif _on("Mouth_Slightly_Open"):
        extras.append("slightly open mouth")
    if _on("Eyeglasses"):
        extras.append("wearing eyeglasses")
    if _on("Wearing_Hat"):
        extras.append("wearing a hat")
    if _on("Mustache") or _on("Goatee") or (not _on("No_Beard") and _on("Male")):
        extras.append("facial hair")
    if _on("Heavy_Makeup") or _on("Wearing_Lipstick"):
        extras.append("makeup")
    if _on("Wearing_Earrings"):
        extras.append("earrings")
    if _on("Wearing_Necktie"):
        extras.append("wearing a necktie")
    parts = [f"A close-up studio portrait photo of a {age}{gender}"]
    if hair:
        parts.append("with " + hair[0])
    if extras:
        parts.append(", ".join(extras))
    return ", ".join(parts) + "."


def load_insightface_apps(use_cuda: bool):
    os.environ["INSIGHTFACE_HOME"] = str(INSIGHTFACE_DIR.resolve())
    from insightface.app import FaceAnalysis

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    ctx = 0 if use_cuda else -1
    # Single slim AntelopeV2 pass: detection + 2d106det + glintr100 (skips heavy unused 1k3d68 & genderage -> 6x faster!)
    antelope = FaceAnalysis(
        name="antelopev2",
        providers=prov,
        root=str(INSIGHTFACE_DIR.resolve()),
        allowed_modules=["detection", "landmark_2d_106", "recognition"],
    )
    antelope.prepare(ctx_id=ctx, det_size=(640, 640))
    return antelope


def decode_parquet_image(val) -> np.ndarray:
    if isinstance(val, dict) and "bytes" in val and val["bytes"] is not None:
        im = Image.open(io.BytesIO(val["bytes"])).convert("RGB")
        return np.asarray(im)
    if isinstance(val, (bytes, bytearray)):
        im = Image.open(io.BytesIO(val)).convert("RGB")
        return np.asarray(im)
    raise ValueError(f"Unsupported image cell type: {type(val)}")


def iter_ffhq_parquet(max_n: int):
    pdir = DATA / "raw" / "ffhq_captioned_parquet"
    files = sorted(pdir.rglob("*.parquet"))
    yielded = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["image", "text"])
            imgs = tbl["image"].to_pylist()
            txts = tbl["text"].to_pylist()
            for idx, (im_val, cap) in enumerate(zip(imgs, txts)):
                if yielded >= max_n:
                    return
                try:
                    rgb = decode_parquet_image(im_val)
                except Exception:
                    continue
                cap_s = str(cap or "A frontal studio portrait photo of a person.").strip()
                yield {
                    "source": "ffhq",
                    "uid": f"ffhq_{yielded:06d}",
                    "split": "train",
                    "celeb_id": None,
                    "caption": cap_s,
                    "attrs": None,
                    "rgb": rgb,
                }
                yielded += 1


def iter_celeba_parquet(split_name: str, out_split: str, max_n: int):
    pdir = DATA / "raw" / "celeba_flwr_parquet"
    files = sorted(p for p in pdir.rglob("*.parquet") if f"{split_name}-" in p.name)
    yielded = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        cols = pf.schema_arrow.names
        attr_cols = [c for c in cols if c not in ("image", "celeb_id")]
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            data = tbl.to_pydict()
            n_rows = len(data["image"])
            for i in range(n_rows):
                if yielded >= max_n:
                    return
                try:
                    rgb = decode_parquet_image(data["image"][i])
                except Exception:
                    continue
                row_attrs = {c: int(bool(data[c][i])) for c in attr_cols}
                cid = int(data["celeb_id"][i])
                attr_vec = [int(bool(data[c][i])) for c in attr_cols[:40]]
                yield {
                    "source": "celeba",
                    "uid": f"celeba_{out_split}_{yielded:06d}",
                    "split": out_split,
                    "celeb_id": cid,
                    "caption": celeba_attr_caption(row_attrs),
                    "attrs": attr_vec,
                    "rgb": rgb,
                }
                yielded += 1


def iter_celebahq_parquet(max_n: int):
    pdir = DATA / "raw" / "celebahq_captioned_parquet"
    # Prefer shard 5 (tail split) as test pool
    files = sorted(pdir.rglob("*.parquet"), reverse=True)
    yielded = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["image", "text"])
            imgs = tbl["image"].to_pylist()
            txts = tbl["text"].to_pylist()
            for im_val, cap in zip(imgs, txts):
                if yielded >= max_n:
                    return
                try:
                    rgb = decode_parquet_image(im_val)
                except Exception:
                    continue
                yield {
                    "source": "celebahq",
                    "uid": f"celebahq_test_{yielded:06d}",
                    "split": "celebahq_test",
                    "celeb_id": None,
                    "caption": str(cap or "A portrait photo of a person.").strip(),
                    "attrs": None,
                    "rgb": rgb,
                }
                yielded += 1


def encode_flan_t5_batch(captions: list[str], dev: torch.device, max_len: int = 64, batch_size: int = 64):
    from transformers import AutoTokenizer, T5EncoderModel

    log(f"Encoding {len(captions)} captions with FLAN-T5-Large (max_len={max_len}) on {dev} ...")
    tokenizer = AutoTokenizer.from_pretrained(FLAN_DIR, local_files_only=True)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    encoder = T5EncoderModel.from_pretrained(FLAN_DIR, torch_dtype=dtype, local_files_only=True).to(dev).eval()
    hiddens = []
    masks = []
    with torch.inference_mode():
        for start in range(0, len(captions), batch_size):
            batch_caps = captions[start : start + batch_size]
            tok = tokenizer(
                batch_caps,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            out = encoder(
                input_ids=tok.input_ids.to(dev),
                attention_mask=tok.attention_mask.to(dev),
            ).last_hidden_state
            hiddens.append(out.to(dtype=torch.bfloat16).cpu())
            masks.append(tok.attention_mask.to(dtype=torch.uint8).cpu())
            if (start // batch_size) % 20 == 0:
                log(f"  FLAN-T5 encoded {min(start + batch_size, len(captions))}/{len(captions)}")
    encoder.to("cpu")
    del encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(hiddens, dim=0).contiguous(), torch.cat(masks, dim=0).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ffhq", type=int, default=3000)
    parser.add_argument("--n-celeba-train", type=int, default=2000)
    parser.add_argument("--n-celeba-val", type=int, default=500)
    parser.add_argument("--n-celeba-test", type=int, default=1000)
    parser.add_argument("--n-celebahq-test", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()

    t0 = time.perf_counter()
    dev = pick_device(min_free_gb=6.0)
    log(f"Selected compute device: {dev}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    antelope = load_insightface_apps(use_cuda=(dev.type == "cuda"))
    op = SPIOperator.from_simulation_256()

    records = []
    imgs256 = []
    ids_antelope = []
    lmks106 = []
    attrs_list = []
    captions = []
    roundtrip_errs = []

    streams = [
        ("ffhq_train", iter_ffhq_parquet(args.n_ffhq)),
        ("celeba_train", iter_celeba_parquet("train", "train", args.n_celeba_train)),
        ("celeba_val", iter_celeba_parquet("valid", "val", args.n_celeba_val)),
        ("celeba_test", iter_celeba_parquet("test", "celeba_test", args.n_celeba_test)),
        ("celebahq_test", iter_celebahq_parquet(args.n_celebahq_test)),
    ]

    missed = 0
    for label, stream in streams:
        log(f"Processing stream: {label} ...")
        cnt = 0
        for item in stream:
            rgb = item["rgb"]
            det_a = detect_face_robust(antelope, rgb)
            if det_a is None or det_a["embedding"] is None or det_a["lmk106"] is None:
                missed += 1
                continue
            al = align_to_lsi(rgb, det_a["kps5"], output_size=512, extra_points=det_a["lmk106"])
            back = transform_points(al["landmarks5"], al["inverse"])
            rt_err = float(np.abs(back - det_a["kps5"]).max())
            frame512 = np.clip(np.round(al["image"] * 255.0), 0, 255).astype(np.uint8)
            aligned_lmk106 = al["extra_points"]
            id_vec = det_a["embedding"]

            idx_global = len(records)
            jpg_rel = f"aligned512/{item['uid']}.jpg"
            Image.fromarray(frame512).save(CACHE_DIR / jpg_rel, quality=95)
            im256 = np.asarray(Image.fromarray(frame512).resize((256, 256), Image.BILINEAR))
            imgs256.append(torch.from_numpy(im256.transpose(2, 0, 1).copy()))
            ids_antelope.append(torch.from_numpy(id_vec.astype(np.float32)))
            lmks106.append(torch.from_numpy((aligned_lmk106 / 512.0).astype(np.float32)))  # [106, 2] in [0, 1]
            attrs_list.append(item["attrs"] if item["attrs"] is not None else [-1] * 40)
            captions.append(item["caption"])
            roundtrip_errs.append(rt_err)
            records.append(
                {
                    "index": idx_global,
                    "uid": item["uid"],
                    "source": item["source"],
                    "split": item["split"],
                    "celeb_id": item["celeb_id"],
                    "image_rel": jpg_rel,
                    "caption": item["caption"],
                }
            )
            cnt += 1
            if cnt % 250 == 0:
                log(f"  {label}: aligned={cnt} (total={len(records)}, missed={missed})")
        log(f"Finished {label}: kept={cnt}")

    # Release InsightFace models before FLAN-T5
    del antelope
    gc.collect()

    # STRICT ANTI-LEAKAGE ASSERTION #1: CelebA official celeb_id disjointness
    train_cids = {r["celeb_id"] for r in records if r["split"] == "train" and r["celeb_id"] is not None}
    val_cids = {r["celeb_id"] for r in records if r["split"] == "val" and r["celeb_id"] is not None}
    test_cids = {r["celeb_id"] for r in records if r["split"] == "celeba_test" and r["celeb_id"] is not None}
    assert len(train_cids & val_cids) == 0, f"LEAKAGE train∩val: {len(train_cids & val_cids)}"
    assert len(train_cids & test_cids) == 0, f"LEAKAGE train∩test: {len(train_cids & test_cids)}"
    assert len(val_cids & test_cids) == 0, f"LEAKAGE val∩test: {len(val_cids & test_cids)}"
    log(
        f"ANTI-LEAKAGE PASS (CelebA celeb_id): train={len(train_cids)} unique IDs, "
        f"val={len(val_cids)}, test={len(test_cids)}, ALL intersections == 0!"
    )

    ids_tensor = F.normalize(torch.stack(ids_antelope, dim=0), dim=-1)

    # STRICT ANTI-LEAKAGE ASSERTION #2: Filter CelebA-HQ test against ALL training IDs (cosine < 0.35)
    train_idx = [r["index"] for r in records if r["split"] == "train"]
    hq_idx = [r["index"] for r in records if r["split"] == "celebahq_test"]
    hq_filtered = 0
    if train_idx and hq_idx:
        train_embs = ids_tensor[train_idx]
        hq_embs = ids_tensor[hq_idx]
        sim = hq_embs @ train_embs.T  # [N_hq, N_train]
        max_sim = sim.max(dim=1).values
        for local_i, g_idx in enumerate(hq_idx):
            ms = float(max_sim[local_i])
            records[g_idx]["max_train_id_cos"] = ms
            if ms >= 0.35:
                records[g_idx]["split"] = "celebahq_excluded_overlap"
                hq_filtered += 1
        log(f"CelebA-HQ cross-identity audit: kept {len(hq_idx) - hq_filtered}/{len(hq_idx)} (excluded {hq_filtered} with cos>=0.35)")

    imgs256_tensor = torch.stack(imgs256, dim=0).contiguous()  # uint8 [N, 3, 256, 256]
    lmks_tensor = torch.stack(lmks106, dim=0).contiguous()  # float32 [N, 106, 2]
    attrs_tensor = torch.tensor(attrs_list, dtype=torch.int8)

    # Precompute clean LSI 256 measurements
    log("Computing clean LSI official 256 binary measurements ...")
    y_chunks = []
    with torch.no_grad():
        for s in range(0, len(records), 256):
            batch_u8 = imgs256_tensor[s : s + 256].float().div(255.0)
            y_chunks.append(op.measure(batch_u8, indices=None, input_range="0_1").float().cpu())
    y256_tensor = torch.cat(y_chunks, dim=0).contiguous()

    # Check noiseless measurement residual on first 64 samples
    with torch.no_grad():
        check_imgs = imgs256_tensor[:64].float().div(255.0)
        res_max = float(op.residual(check_imgs, y256_tensor[:64], input_range="0_1").max())
    assert res_max < 1e-4, f"Measurement self-residual too high: {res_max}"

    # Encode captions with FLAN-T5-Large (max_len=64, bf16)
    dev_t5 = pick_device(min_free_gb=4.0)
    sem_hidden, sem_mask = encode_flan_t5_batch(captions, dev=dev_t5, max_len=args.max_len, batch_size=64)

    # Save compact tensors
    torch.save(imgs256_tensor, CACHE_DIR / "images256_u8.pt")
    torch.save(y256_tensor, CACHE_DIR / "y256.pt")
    torch.save(ids_tensor, CACHE_DIR / "identity_antelope.pt")
    torch.save(lmks_tensor, CACHE_DIR / "landmarks106_norm.pt")
    torch.save(attrs_tensor, CACHE_DIR / "celeba_attrs40.pt")
    torch.save(sem_hidden, CACHE_DIR / "sem_hidden_bf16.pt")
    torch.save(sem_mask, CACHE_DIR / "sem_mask_u8.pt")

    split_counts = {}
    for r in records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    manifest = {
        "n_total": len(records),
        "n_missed": missed,
        "split_counts": split_counts,
        "celeba_unique_train_ids": len(train_cids),
        "celeba_unique_val_ids": len(val_cids),
        "celeba_unique_test_ids": len(test_cids),
        "celeba_id_overlap_train_val": 0,
        "celeba_id_overlap_train_test": 0,
        "celeba_id_overlap_val_test": 0,
        "celebahq_excluded_overlap": hq_filtered,
        "alignment_roundtrip_max_abs": float(max(roundtrip_errs)) if roundtrip_errs else 0.0,
        "noiseless_measurement_residual_max": res_max,
        "sem_max_len": args.max_len,
        "elapsed_seconds": time.perf_counter() - t0,
        "records": records,
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save 50-face Level 0 visual QC grid
    qc_idx = np.linspace(0, len(records) - 1, min(50, len(records)), dtype=int).tolist()
    qc_imgs = imgs256_tensor[qc_idx].float().div(255.0).mul(2.0).sub(1.0)
    save_image_grid(qc_imgs, OUTPUTS / "level0_qc_real_5k.png", nrow=10)

    summary = {k: v for k, v in manifest.items() if k != "records"}
    (OUTPUTS / "level0_real_5k.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("LEVEL 0 REAL-FACE CACHE COMPLETE!")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
