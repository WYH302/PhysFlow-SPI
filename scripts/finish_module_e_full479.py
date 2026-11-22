"""Fast Incremental Completion of Module E (ALL 479 Identity-Disjoint CelebA-HQ + Real Inception-v3 FID).

Pre-caches GT Buffalo_L embeddings and GT AdaFace/Inception-v3 features ONCE (2x faster),
runs LSI (M=64, 256), Wiener-LMMSE (M=64, 256), and PhysFlow (M=64, 256) with per-method
incremental saves into `outputs/reviewer_shield_results.json`, integrates verified full-479
metrics for TV-FISTA, ZS-FLOWER, and PnP-Flow from the completed run, and computes exact
N=479 paired bootstrap 95% CIs and p-values (PhysFlow vs LSI and PhysFlow vs Wiener-LMMSE).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.align import detect_face_robust
from src.eval.adaface_eval import CrossModelEvaluator
from src.eval.metrics import cosine_identity, psnr, ssim
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.lsi_baseline import LSIBaseline
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all

from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, load_512_batch, load_generator_ckpt
from scripts.run_reviewer_shield_suite import (
    OUT_JSON,
    InceptionFID,
    bootstrap_ci_and_p,
    sample_physflow_custom,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fast_evaluate_pair(
    ev: CrossModelEvaluator,
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_ada_embs: torch.Tensor,
    gt_buf_dets: list[dict | None],
    gt_landmarks_px: torch.Tensor,
    operator: SPIOperator,
    y_gt: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int = 16,
) -> dict:
    """Exact bit-for-bit equivalent of CrossModelEvaluator.evaluate_pair with pre-cached GT features."""
    n = int(pred.shape[0])
    pred_f = pred.detach().float().cpu()
    gt_f = gt.detach().float().cpu()

    psnr_vals = psnr(pred_f, gt_f).tolist()
    ssim_vals = ssim(pred_f, gt_f).tolist()

    # AdaFace cosine (pre-cached GT embeddings!) + full-res 512px LPIPS (100% identical to CrossModelEvaluator)
    ada_cos = []
    lpips_vals = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            pb = pred_f[s:e].to(ev.device)
            gb = gt_f[s:e].to(ev.device)
            ep = ev.adaface.embed_aligned(pb)
            ada_cos.extend(cosine_identity(ep, gt_ada_embs[s:e].to(ev.device)).cpu().tolist())
            lpips_vals.extend(ev.lpips_fn(pb.clamp(-1.0, 1.0), gb.clamp(-1.0, 1.0)).view(-1).cpu().tolist())

    # Buffalo_L + 106-pt NME (only runs Buffalo_L on pred; GT already cached!)
    buf_cos: list[float | None] = []
    nme106_vals: list[float | None] = []
    det_pred = 0
    for i in range(n):
        p_det = detect_face_robust(ev.buffalo, ev._to_rgb_u8(pred_f[i]), prefer_center=True)
        g_det = gt_buf_dets[i]
        if p_det is not None:
            det_pred += 1
        if p_det is not None and g_det is not None and p_det["embedding"] is not None and g_det["embedding"] is not None:
            buf_cos.append(float(np.dot(p_det["embedding"], g_det["embedding"])))
        else:
            buf_cos.append(None)
        if p_det is not None and p_det["lmk106"] is not None:
            ref_lmk = gt_landmarks_px[i].detach().float().cpu()
            pred_lmk = torch.from_numpy(p_det["lmk106"]).float()
            left_eye = ref_lmk[33:43].mean(dim=0)
            right_eye = ref_lmk[87:97].mean(dim=0)
            iod = float(torch.linalg.vector_norm(left_eye - right_eye).clamp_min(1e-6))
            err = float(torch.linalg.vector_norm(pred_lmk - ref_lmk, dim=-1).mean())
            nme106_vals.append(err / iod)
        else:
            nme106_vals.append(None)

    # Physical measurement residual
    op_dev = operator.phi.device
    rel_list, rmse_list = [], []
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        pb = pred_f[s:e].to(op_dev)
        yb = y_gt[s:e].to(op_dev)
        idx_b = indices.to(op_dev)
        rel_list.extend(operator.relative_residual(pb, yb, indices=idx_b, input_range="-1_1").cpu().tolist())
        rmse_list.extend(operator.residual(pb, yb, indices=idx_b, input_range="-1_1").cpu().tolist())

    valid_buf = [v for v in buf_cos if v is not None]
    valid_nme = [v for v in nme106_vals if v is not None]
    return {
        "n": n,
        "image_size": int(pred_f.shape[-1]),
        "adaface_cos_mean": float(np.mean(ada_cos)),
        "adaface_cos_std": float(np.std(ada_cos)),
        "buffalo_cos_mean": float(np.mean(valid_buf)) if valid_buf else None,
        "buffalo_cos_std": float(np.std(valid_buf)) if valid_buf else None,
        "buffalo_detected": det_pred,
        "nme106_mean": float(np.mean(valid_nme)) if valid_nme else None,
        "nme106_std": float(np.std(valid_nme)) if valid_nme else None,
        "nme106_n": len(valid_nme),
        "psnr_mean": float(np.mean(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "lpips_mean": float(np.mean(lpips_vals)),
        "meas_rel_mean": float(np.mean(rel_list)),
        "meas_rmse_mean": float(np.mean(rmse_list)),
        "per_sample": {
            "adaface_cos": ada_cos,
            "buffalo_cos": buf_cos,
            "nme106": nme106_vals,
            "psnr": psnr_vals,
            "ssim": ssim_vals,
            "lpips": lpips_vals,
            "meas_rel": rel_list,
        },
    }


def main() -> None:
    seed_all(42)
    dev = torch.device("cuda")
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    ps_cache_path = OUTPUTS / "full479_persample_cache.json"
    per_sample_store: dict = json.loads(ps_cache_path.read_text(encoding="utf-8")) if ps_cache_path.exists() else {"64": {}, "256": {}}

    res_e = payload.setdefault(
        "module_e_full479_fid",
        {
            "n_portraits": 479,
            "rates": {"64": {}, "256": {}},
            "paired_bootstrap_vs_physflow": {"64": {}, "256": {}},
        },
    )

    # Record verified task-1174 exact full-479 metrics for TV-FISTA, ZS-FLOWER, and PnP-Flow
    verified_prior = {
        "64": {
            "tv_fista": {"n": 479, "fid_inceptionv3": 373.91, "adaface_cos_mean": 0.0137, "buffalo_cos_mean": None, "nme106_mean": None, "buffalo_detected": 0, "lpips_mean": 0.8561, "psnr_mean": 13.74, "ssim_mean": 0.5488, "meas_rel_mean": 0.0088},
            "zs_flower": {"n": 479, "fid_inceptionv3": 149.67, "adaface_cos_mean": 0.0037, "buffalo_cos_mean": 0.0080, "nme106_mean": 0.4121, "buffalo_detected": 479, "lpips_mean": 0.6606, "psnr_mean": 11.75, "ssim_mean": 0.4240, "meas_rel_mean": 0.0144},
            "pnp_flow": {"n": 479, "fid_inceptionv3": 146.94, "adaface_cos_mean": 0.0051, "buffalo_cos_mean": 0.0082, "nme106_mean": 0.4090, "buffalo_detected": 476, "lpips_mean": 0.6899, "psnr_mean": 12.60, "ssim_mean": 0.4823, "meas_rel_mean": 0.0127},
        },
        "256": {
            "tv_fista": {"n": 479, "fid_inceptionv3": 349.62, "adaface_cos_mean": 0.0236, "buffalo_cos_mean": None, "nme106_mean": None, "buffalo_detected": 0, "lpips_mean": 0.8471, "psnr_mean": 14.08, "ssim_mean": 0.5560, "meas_rel_mean": 0.0100},
            "zs_flower": {"n": 479, "fid_inceptionv3": 181.94, "adaface_cos_mean": 0.0060, "buffalo_cos_mean": 0.0124, "nme106_mean": 0.3975, "buffalo_detected": 479, "lpips_mean": 0.6623, "psnr_mean": 14.02, "ssim_mean": 0.3980, "meas_rel_mean": 0.0101},
            "pnp_flow": {"n": 479, "fid_inceptionv3": 147.65, "adaface_cos_mean": 0.0212, "buffalo_cos_mean": 0.0255, "nme106_mean": 0.3349, "buffalo_detected": 478, "lpips_mean": 0.6475, "psnr_mean": 15.86, "ssim_mean": 0.5527, "meas_rel_mean": 0.0101},
        },
    }
    for rkey, mdict in verified_prior.items():
        for mname, mvals in mdict.items():
            res_e["rates"][rkey].setdefault(mname, mvals)

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    all_hq_idx = [r["index"] for r in records if r["split"] == "celebahq_test"]
    train_idx = [r["index"] for r in records if r["split"] == "train"]
    y_train_mean = y256[train_idx].mean(dim=0)

    gt_all = load_512_batch(records, all_hq_idx, torch.device("cpu"))
    gt_all_lmk = lmks[all_hq_idx] * 512.0

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)
    fid_calc = InceptionFID(dev)

    log(f"Pre-caching GT Inception-v3, AdaFace, and Buffalo_L features ONCE for {len(all_hq_idx)} CelebA-HQ portraits...")
    f_ref = fid_calc.features(gt_all, batch_size=32)
    gt_ada_list = []
    with torch.no_grad():
        for s in range(0, len(all_hq_idx), 16):
            gt_ada_list.append(ev.adaface.embed_aligned(gt_all[s : s + 16].float()).cpu())
    gt_ada_embs = torch.cat(gt_ada_list, dim=0)
    gt_buf_dets = [detect_face_robust(ev.buffalo, ev._to_rgb_u8(gt_all[i].float()), prefer_center=True) for i in range(len(all_hq_idx))]
    log("GT feature cache ready!")

    def _eval_save(rate_int: int, mname: str, imgs_cpu: torch.Tensor, sec_img: float) -> None:
        rkey = str(rate_int)
        f_gen = fid_calc.features(imgs_cpu, batch_size=32)
        fid_val = fid_calc.frechet_distance(f_gen, f_ref)
        r = fast_evaluate_pair(
            ev, imgs_cpu, gt_all, gt_ada_embs, gt_buf_dets, gt_all_lmk,
            op_cpu, y256[all_hq_idx, :rate_int], torch.arange(rate_int), batch_size=16,
        )
        ps = r.pop("per_sample")
        per_sample_store[rkey][mname] = ps
        r["fid_inceptionv3"] = fid_val
        r["sec_per_img"] = sec_img
        res_e["rates"][rkey][mname] = r
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ps_cache_path.write_text(json.dumps(per_sample_store), encoding="utf-8")
        log(
            f"  [Full479 M={rate_int:3d} | {mname:12s}] FID={fid_val:6.2f} | "
            f"Ada={r['adaface_cos_mean']:.4f} Buf={r['buffalo_cos_mean'] or 0:.4f} "
            f"NME={r['nme106_mean'] or -1:.4f} Det={r['buffalo_detected']}/479 | "
            f"LPIPS={r['lpips_mean']:.4f} PSNR={r['psnr_mean']:.2f} SSIM={r['ssim_mean']:.4f} Res={r['meas_rel_mean']:.4f}"
        )

    # 1) Official CVPR 2025 LSI (StyleGAN-XL) on ALL 479 portraits (M=64 & M=256)
    if ("lsi" not in per_sample_store["64"]) or ("lsi" not in per_sample_store["256"]):
        log("=== Running Official CVPR 2025 LSI on ALL 479 CelebA-HQ Portraits ===")
        lsi = LSIBaseline(device=dev)
        for rate in (64, 256):
            if "lsi" in per_sample_store[str(rate)]:
                continue
            t0 = time.perf_counter()
            im_lsi, _ = lsi.reconstruct(y256[all_hq_idx, :rate], indices=torch.arange(rate), y_mean=y_train_mean, output_size=512, batch_size=16)
            sec_lsi = (time.perf_counter() - t0) / len(all_hq_idx)
            _eval_save(rate, "lsi", im_lsi, sec_lsi)
            del im_lsi
            gc.collect()
        del lsi
        gc.collect()
        torch.cuda.empty_cache()

    # 2) Wiener-LMMSE on ALL 479 portraits (M=64 & M=256) — ultra fast!
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])
    for rate in (64, 256):
        if "wiener_lmmse" in per_sample_store[str(rate)]:
            continue
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        yb = y256[all_hq_idx, :rate]
        t0 = time.perf_counter()
        with torch.no_grad():
            iw = enc.lifter.lift_image(yb.to(dev), idx_r, reg=4e-4, blind_adapt=True)
            im_w = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()
        sec_w = (time.perf_counter() - t0) / len(all_hq_idx)
        _eval_save(rate, "wiener_lmmse", im_w, sec_w)
        del im_w
        gc.collect()

    # 3) PhysFlow (Ours, 16-step) on ALL 479 portraits (M=64 & M=256) — gives exact paired bootstrap vs LSI & Wiener!
    if ("physflow" not in per_sample_store["64"]) or ("physflow" not in per_sample_store["256"]):
        log("=== Running PhysFlow (Ours, 16-step) on ALL 479 CelebA-HQ Portraits ===")
        wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
        wrapper = wrapper.to(dev).eval()
        net = wrapper.model.net
        attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
        net.double_blocks.to(dev)
        cond = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev).eval()
        load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, cond)
        for rate in (64, 256):
            if "physflow" in per_sample_store[str(rate)]:
                continue
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            im_pf, sec_pf = sample_physflow_custom(
                net, cond, enc, op_gpu, y256[all_hq_idx, :rate], idx_r, dev,
                steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.35, batch_size=12,
            )
            _eval_save(rate, "physflow", im_pf, sec_pf)
            del im_pf
            gc.collect()
        wrapper.to("cpu"); cond.to("cpu")
        del wrapper, cond
        gc.collect()
        torch.cuda.empty_cache()

    # 4) Exact N=479 Paired Bootstrap 95% CIs and p-values (PhysFlow vs LSI & Wiener-LMMSE)
    for rkey in ("64", "256"):
        pf_ps = per_sample_store[rkey]["physflow"]
        for base_m in ("wiener_lmmse", "lsi"):
            b_ps = per_sample_store[rkey][base_m]
            res_e["paired_bootstrap_vs_physflow"][rkey][base_m] = {
                "adaface": bootstrap_ci_and_p(pf_ps["adaface_cos"], b_ps["adaface_cos"], higher_better=True),
                "buffalo": bootstrap_ci_and_p(pf_ps["buffalo_cos"], b_ps["buffalo_cos"], higher_better=True),
                "nme106": bootstrap_ci_and_p(pf_ps["nme106"], b_ps["nme106"], higher_better=False),
                "lpips": bootstrap_ci_and_p(pf_ps["lpips"], b_ps["lpips"], higher_better=False),
                "psnr": bootstrap_ci_and_p(pf_ps["psnr"], b_ps["psnr"], higher_better=True),
            }
            bt = res_e["paired_bootstrap_vs_physflow"][rkey][base_m]
            log(
                f"  [Bootstrap N=479 M={rkey:>3s} PhysFlow vs {base_m:12s}] "
                f"dAda={bt['adaface']['diff_mean']:+.4f} (p={bt['adaface']['p_one_sided']:.4f}) | "
                f"dNME={bt['nme106']['diff_mean']:+.4f} (p={bt['nme106']['p_one_sided']:.4f}) | "
                f"dLPIPS={bt['lpips']['diff_mean']:+.4f} (p={bt['lpips']['p_one_sided']:.4f}) | "
                f"dPSNR={bt['psnr']['diff_mean']:+.2f} (p={bt['psnr']['p_one_sided']:.4f})"
            )

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"MODULE E (ALL 479 + REAL FID + PAIRED BOOTSTRAP) 100% COMPLETE -> {OUT_JSON}")


if __name__ == "__main__":
    main()
