"""Step 6B (V3 Surgery): Fast (~35s) Parity Checks & Visual QA Strips.

1. Visual QA grids (visual_qa_m64.png, visual_qa_m256.png): 4 faces x 4 cols (GT | Flow | V3+Ref | Fast4+Ref).
2. Exact 1,479-split (celeba_test + celebahq_test) evaluation of SigLIP2-Base vs DINOv2-Small for Table 4(a) parity.
3. Exact hq45 & cb45 evaluation of PhysFlow-V3 (16-step + Refiner, proj_eta=0.75) & PhysFlow-Fast (4-step + Refiner, proj_eta=0.75)
   for Table 3 (CelebA45), Table 4(c) (hq45), and Fig. 5(c) (Pareto frontier), plus warmed-up CUDA latencies (bs=1, bs=9).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.minit2i_loader import load_b16_transformer
from src.optical.spi_operator import SPIOperator, WienerPhysicsLifter
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import seed_all
from src.vis import save_image_grid

from scripts.run_phase1a_real import evaluate_encoder
from scripts.run_phase1b_oracle_real import CACHE_DIR, CKPT_DIR, load_512_batch, load_generator_ckpt
from scripts.run_phase2_e2e_physflow import pick_even_split_indices
from scripts.run_v3_step1_inference_upgrade import sample_physflow_v3
from scripts.run_v3_step3_siglip2 import SigLIP2Backbone
from scripts.run_v3_step4_pixel_refiner import LateStagePixelHFRefiner
from scripts.run_v3_step6_final_full479 import apply_refiner_physics

V3_DIR = OUTPUTS / "v3_surgery"
V3_CKPT = OUTPUTS / "checkpoints" / "v3_surgery"
OUT_JSON = V3_DIR / "step6b_parity_and_qa.json"


def strip_ps(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_sample"}


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
    geo = lmks.reshape(lmks.shape[0], 212)

    train_idx = [r["index"] for r in records if r["split"] == "train"]
    test1479_idx = [r["index"] for r in records if r["split"] in ("celeba_test", "celebahq_test")]
    assert len(test1479_idx) == 1479
    hq_idx45 = pick_even_split_indices(records, "celebahq_test", 45)
    cb_idx45 = pick_even_split_indices(records, "celeba_test", 45)
    gt_hq45 = load_512_batch(records, hq_idx45, torch.device("cpu"))
    gt_cb45 = load_512_batch(records, cb_idx45, torch.device("cpu"))

    op_gpu = SPIOperator.from_simulation_256().to(dev)
    op_cpu = SPIOperator.from_simulation_256()
    out: dict = {}

    # 1. Exact 1,479-split SigLIP2 vs DINOv2 check (Table 4a parity)
    imgs256 = torch.load(CACHE_DIR / "images256_u8.pt", map_location="cpu", weights_only=True)
    lifter = WienerPhysicsLifter(max_measurements=256, img_size=128)
    lifter.fit(imgs256[train_idx], y256[train_idx])
    del imgs256

    dino_1a = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    dino_1a.load_state_dict(torch.load(CKPT_DIR / "dual_encoder_real.pt", map_location="cpu", weights_only=False)["state_dict"])
    out["stage1a_exact1479"] = {
        "dinov2_small": {
            "64": evaluate_encoder(dino_1a, y256, sem, sem_mask, ident, geo, test1479_idx, dev, rate=64),
            "256": evaluate_encoder(dino_1a, y256, sem, sem_mask, ident, geo, test1479_idx, dev, rate=256),
            "n4_64": evaluate_encoder(dino_1a, y256, sem, sem_mask, ident, geo, test1479_idx, dev, rate=64, noise_mode="combined"),
        }
    }
    del dino_1a
    torch.cuda.empty_cache()

    sig_bb = SigLIP2Backbone("google/siglip2-base-patch16-224", unfreeze_last_n=4)
    sig_1a = PhysicsDualDomainFactorizer(op_gpu, lifter=lifter, backbone=sig_bb, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=4).to(dev).eval()
    sig_1a.load_state_dict(torch.load(V3_CKPT / "siglip2_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])
    out["stage1a_exact1479"]["siglip2_base"] = {
        "64": evaluate_encoder(sig_1a, y256, sem, sem_mask, ident, geo, test1479_idx, dev, rate=64),
        "256": evaluate_encoder(sig_1a, y256, sem, sem_mask, ident, geo, test1479_idx, dev, rate=256),
        "n4_64": evaluate_encoder(sig_1a, y256, sem, sem_mask, ident, geo, test1479_idx, dev, rate=64, noise_mode="combined"),
    }
    del sig_1a, sig_bb
    torch.cuda.empty_cache()
    print("[Exact 1479] DINOv2:", out["stage1a_exact1479"]["dinov2_small"]["64"]["id_cosine"], out["stage1a_exact1479"]["dinov2_small"]["256"]["id_cosine"], out["stage1a_exact1479"]["dinov2_small"]["64"]["nme106"], out["stage1a_exact1479"]["dinov2_small"]["n4_64"]["nme106"], flush=True)
    print("[Exact 1479] SigLIP2:", out["stage1a_exact1479"]["siglip2_base"]["64"]["id_cosine"], out["stage1a_exact1479"]["siglip2_base"]["256"]["id_cosine"], out["stage1a_exact1479"]["siglip2_base"]["64"]["nme106"], out["stage1a_exact1479"]["siglip2_base"]["n4_64"]["nme106"], flush=True)

    # 2. Load E2E encoder, refiner, generator, evaluator
    ev = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)
    enc = PhysicsDualDomainFactorizer(op_gpu, max_measurements=256, dim=384, sem_tokens=64, sem_dim=1024, lam=2e-4, unfreeze_last_n=6).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT_DIR / "physflow_e2e_encoder.pt", map_location="cpu", weights_only=False)["state_dict"])
    refiner = LateStagePixelHFRefiner(base=48).to(dev).eval()
    refiner.load_state_dict(torch.load(V3_CKPT / "pixel_hf_refiner.pt", map_location="cpu", weights_only=False)["state_dict"])

    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev).eval()
    net = wrapper.model.net
    attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    cond = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev).eval()
    load_generator_ckpt(CKPT_DIR / "physflow_e2e_full.pt", net.double_blocks, cond)

    t0_map = {64: 0.25, 256: 0.35}
    qa_tiles = {"64": {"gt": gt_hq45[:4]}, "256": {"gt": gt_hq45[:4]}}
    out["v3_16step_proj075"] = {"hq45": {}, "cb45": {}}
    out["v3_16step_flow_only"] = {"hq45": {}, "cb45": {}}
    out["fast4_proj075"] = {"hq45": {}, "cb45": {}}
    out["timings_v3_16step"] = {}

    # Measure clean warmed-up CUDA latencies for 16-step + Refiner (bs=1 and bs=9)
    idx64 = torch.arange(64, device=dev)
    for bs_tag, bs_n in (("bs1", 1), ("bs9", 9)):
        yb = y256[hq_idx45[:bs_n], :64]
        _x, _ = sample_physflow_v3(net, cond, enc, op_gpu, yb, idx64, dev, steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.25, blind_mode="tail_scale", blind_scale=8.0, batch_size=bs_n)
        _ = apply_refiner_physics(refiner, enc, op_gpu, _x, yb, 64, dev, proj_eta=0.75, blind_scale=8.0, batch_size=bs_n)
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        reps = 4 if bs_n == 1 else 2
        for _ in range(reps):
            _x, _ = sample_physflow_v3(net, cond, enc, op_gpu, yb, idx64, dev, steps=16, cfg_scale=2.0, eta=0.85, t_warm=0.25, blind_mode="tail_scale", blind_scale=8.0, batch_size=bs_n)
            _ = apply_refiner_physics(refiner, enc, op_gpu, _x, yb, 64, dev, proj_eta=0.75, blind_scale=8.0, batch_size=bs_n)
        torch.cuda.synchronize()
        sec_per = (time.perf_counter() - t_s) / (reps * bs_n)
        out["timings_v3_16step"][f"{bs_tag}_sec"] = round(sec_per, 4)
        out["timings_v3_16step"][f"{bs_tag}_fps"] = round(1.0 / max(sec_per, 1e-6), 2)
    print("[V3 16-step+Refiner Latency]", out["timings_v3_16step"], flush=True)

    for split_name, idxs, gt_imgs in (("hq45", hq_idx45, gt_hq45), ("cb45", cb_idx45, gt_cb45)):
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev)
            yb = y256[idxs, :rate]
            xf, _ = sample_physflow_v3(net, cond, enc, op_gpu, yb, idx_r, dev, steps=16, cfg_scale=2.0, eta=0.85, t_warm=t0_map[rate], blind_mode="tail_scale", blind_scale=8.0, seed_base=3000, batch_size=9)
            xr = apply_refiner_physics(refiner, enc, op_gpu, xf, yb, rate, dev, proj_eta=0.75, blind_scale=8.0, batch_size=9)
            rf = strip_ps(ev.evaluate_pair(xf, gt_imgs, operator=op_cpu, y_gt=yb, indices=torch.arange(rate), gt_landmarks_px=lmks[idxs] * 512.0))
            rr = strip_ps(ev.evaluate_pair(xr, gt_imgs, operator=op_cpu, y_gt=yb, indices=torch.arange(rate), gt_landmarks_px=lmks[idxs] * 512.0))
            out["v3_16step_flow_only"][split_name][str(rate)] = rf
            out["v3_16step_proj075"][split_name][str(rate)] = rr
            if split_name == "hq45":
                qa_tiles[str(rate)]["v3_flow"] = xf[:4]
                qa_tiles[str(rate)]["v3_ref"] = xr[:4]
            print(f"[V3+Ref075 {split_name} M={rate}] Ada={rr['adaface_cos_mean']:.4f} Buf={rr['buffalo_cos_mean']:.4f} NME={rr['nme106_mean']:.4f} LPIPS={rr['lpips_mean']:.4f} PSNR={rr['psnr_mean']:.2f} RelRes={rr['meas_rel_mean']:.4f}", flush=True)

    # Swap to Fast-4 LoRA
    fast_ckpt = torch.load(V3_CKPT / "physflow_fast4_lora.pt", map_location="cpu", weights_only=False)
    net.double_blocks.load_state_dict(fast_ckpt["lora_state_dict"], strict=False)
    cond.load_state_dict(fast_ckpt["conditioner_state_dict"])
    net.eval(); cond.eval()

    for split_name, idxs, gt_imgs in (("hq45", hq_idx45, gt_hq45), ("cb45", cb_idx45, gt_cb45)):
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev)
            yb = y256[idxs, :rate]
            xf4, _ = sample_physflow_v3(net, cond, enc, op_gpu, yb, idx_r, dev, steps=4, cfg_scale=1.0, eta=0.85, t_warm=t0_map[rate], blind_mode="tail_scale", blind_scale=8.0, seed_base=3000, batch_size=9)
            xr4 = apply_refiner_physics(refiner, enc, op_gpu, xf4, yb, rate, dev, proj_eta=0.75, blind_scale=8.0, batch_size=9)
            rr4 = strip_ps(ev.evaluate_pair(xr4, gt_imgs, operator=op_cpu, y_gt=yb, indices=torch.arange(rate), gt_landmarks_px=lmks[idxs] * 512.0))
            out["fast4_proj075"][split_name][str(rate)] = rr4
            if split_name == "hq45":
                qa_tiles[str(rate)]["fast4_ref"] = xr4[:4]
            print(f"[Fast4+Ref075 {split_name} M={rate}] Ada={rr4['adaface_cos_mean']:.4f} Buf={rr4['buffalo_cos_mean']:.4f} NME={rr4['nme106_mean']:.4f} LPIPS={rr4['lpips_mean']:.4f} PSNR={rr4['psnr_mean']:.2f} RelRes={rr4['meas_rel_mean']:.4f}", flush=True)

    # Save visual QA grids (4 faces x 4 cols: GT | Flow | V3+Ref | Fast4+Ref)
    for rate_str in ("64", "256"):
        grid_imgs, grid_lbls = [], []
        for i in range(4):
            for key, lbl in (("gt", "GT"), ("v3_flow", f"Flow M={rate_str}"), ("v3_ref", f"V3+Ref M={rate_str}"), ("fast4_ref", f"Fast4+Ref M={rate_str}")):
                grid_imgs.append(qa_tiles[rate_str][key][i])
                grid_lbls.append(lbl)
        save_image_grid(torch.stack(grid_imgs), V3_DIR / f"visual_qa_m{rate_str}.png", nrow=4, titles=grid_lbls)

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved Step 6B parity & QA -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
