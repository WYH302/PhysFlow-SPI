"""Phase 1B Real Oracle Verification (Loop A) on 5,000+ real faces.

Uses our RMSNorm-safe native-256 per-block residual FactorizedJointConditioner
(zero AdaLN, exact step-0 identity, rank-16 QKVO LoRA) on MiniT2I-B/16.
Evaluates O0 (Sem), O1 (Sem+ID), O2 (Sem+ID+Geo), and O2+PredCond (from Phase 1A)
on strictly identity-disjoint test faces using INDEPENDENT cross-model evaluators:
  - AdaFace IR-101 (WebFace12M)
  - InsightFace Buffalo_L
  - 106-point Landmark NME
  - LPIPS / PSNR / SSIM
"""

from __future__ import annotations

import argparse
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
from PIL import Image

from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer
from src.models.glintr100 import FixedArcfaceCrop, GlintR100
from src.models.injection import QKVO_NAMES, FactorizedJointConditioner, attach_qkvo_lora
from src.models.minit2i_condition import predict_clean_factorized
from src.models.minit2i_loader import load_b16_transformer
from src.models.surrogate import official_v_loss
from src.optical.spi_operator import SPIOperator
from src.paths import DATA, OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import peak_vram_gb, reset_vram, seed_all
from src.solvers.flower import flower_sample
from src.vis import save_image_grid

CACHE_DIR = DATA / "cache" / "physflow_real"
CKPT_DIR = OUTPUTS / "checkpoints"
GLINT_ONNX = PROJECT_ROOT / "checkpoints" / "insightface" / "models" / "antelopev2" / "glintr100.onnx"
NOISE_SCALE = 2.0
VRAM_STOP_GB = 21.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_unique_identity_indices(records: list[dict], split: str, max_n: int) -> list[int]:
    """Select at most 1 image per unique celeb_id (so sorted CelebA shards don't repeat 1 person 20x)."""
    seen_cids = set()
    chosen = []
    for r in records:
        if r["split"] != split:
            continue
        cid = r.get("celeb_id")
        if cid is not None:
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
        chosen.append(r["index"])
        if len(chosen) >= max_n:
            break
    return chosen


def load_512_batch(records: list[dict], indices: list[int], dev: torch.device) -> torch.Tensor:
    imgs = []
    for idx in indices:
        p = CACHE_DIR / records[idx]["image_rel"]
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        t = torch.from_numpy(arr.transpose(2, 0, 1)).div_(255.0).mul_(2.0).sub_(1.0)
        imgs.append(t)
    return torch.stack(imgs, dim=0).to(dev, non_blocking=True)


def sample_t_batch(batch_size: int, generator: torch.Generator, dev: torch.device) -> torch.Tensor:
    z = torch.randn(batch_size, device=dev, generator=generator) * 0.8 - 0.8
    return torch.sigmoid(z).clamp(1e-4, 1.0 - 1e-4)


def save_generator_ckpt(path: Path, blocks, conditioner: FactorizedJointConditioner | None, meta: dict) -> None:
    lora_state = {}
    for i, blk in enumerate(blocks):
        for name in QKVO_NAMES:
            layer = getattr(blk, name)
            lora_state[f"{i}.{name}.A"] = layer.lora_A.detach().float().cpu()
            lora_state[f"{i}.{name}.B"] = layer.lora_B.detach().float().cpu()
    payload = {
        "lora": lora_state,
        "conditioner": conditioner.state_dict() if conditioner is not None else None,
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_generator_ckpt(path: Path, blocks, conditioner: FactorizedJointConditioner | None = None) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    lora_state = blob["lora"]
    for i, blk in enumerate(blocks):
        for name in QKVO_NAMES:
            layer = getattr(blk, name)
            layer.lora_A.data.copy_(lora_state[f"{i}.{name}.A"].to(device=layer.lora_A.device, dtype=layer.lora_A.dtype))
            layer.lora_B.data.copy_(lora_state[f"{i}.{name}.B"].to(device=layer.lora_B.device, dtype=layer.lora_B.dtype))
    if conditioner is not None and blob.get("conditioner") is not None:
        conditioner.load_state_dict(blob["conditioner"], strict=False)
    return blob.get("meta", {})


def train_stage(
    stage_name: str,
    net,
    lora_params: list[torch.nn.Parameter],
    conditioner: FactorizedJointConditioner | None,
    use_id: bool,
    use_geo: bool,
    records: list[dict],
    sem_hidden: torch.Tensor,
    sem_mask: torch.Tensor,
    ident: torch.Tensor,
    geo: torch.Tensor,
    train_idx: list[int],
    dev: torch.device,
    opt_steps: int,
    micro_batch: int,
    grad_accum: int,
    lr_lora: float,
    lr_proj: float,
    lr_gate: float,
    id_embedder: GlintR100 | None = None,
    cropper: FixedArcfaceCrop | None = None,
    id_loss_weight: float = 0.08,
    seed: int = 42,
) -> dict:
    param_groups = [{"params": lora_params, "lr": lr_lora, "weight_decay": 1e-4}]
    if conditioner is not None:
        if use_id:
            for p in conditioner.id_net.parameters():
                p.requires_grad_(True)
            for p in conditioner.id_norm.parameters():
                p.requires_grad_(True)
            conditioner.id_pos.requires_grad_(True)
            conditioner.id_gates.requires_grad_(True)
            param_groups.append(
                {
                    "params": list(conditioner.id_net.parameters()) + list(conditioner.id_norm.parameters()) + [conditioner.id_pos],
                    "lr": lr_proj,
                    "weight_decay": 1e-4,
                }
            )
            param_groups.append({"params": [conditioner.id_gates], "lr": lr_gate, "weight_decay": 0.0})
        else:
            for p in list(conditioner.id_net.parameters()) + list(conditioner.id_norm.parameters()) + [conditioner.id_pos, conditioner.id_gates]:
                p.requires_grad_(False)
        if use_geo:
            for p in conditioner.geo_net.parameters():
                p.requires_grad_(True)
            for p in conditioner.geo_norm.parameters():
                p.requires_grad_(True)
            conditioner.geo_pos.requires_grad_(True)
            conditioner.geo_gates.requires_grad_(True)
            param_groups.append(
                {
                    "params": list(conditioner.geo_net.parameters()) + list(conditioner.geo_norm.parameters()) + [conditioner.geo_pos],
                    "lr": lr_proj,
                    "weight_decay": 1e-4,
                }
            )
            param_groups.append({"params": [conditioner.geo_gates], "lr": lr_gate, "weight_decay": 0.0})
        else:
            for p in list(conditioner.geo_net.parameters()) + list(conditioner.geo_norm.parameters()) + [conditioner.geo_pos, conditioner.geo_gates]:
                p.requires_grad_(False)

    opt = torch.optim.AdamW(param_groups)
    warmup = min(50, max(10, opt_steps // 10))

    def _lr_lambda(step_i: int) -> float:
        if step_i < warmup:
            return float(step_i + 1) / float(warmup)
        progress = float(step_i - warmup) / float(max(1, opt_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    rng = np.random.default_rng(seed)
    t_gen = torch.Generator(device=dev).manual_seed(seed)
    train_arr = np.asarray(train_idx, dtype=np.int64)
    n_train = len(train_arr)

    net.train()
    if conditioner is not None:
        conditioner.train()

    t0 = time.perf_counter()
    history = []
    all_trainable = [p for g in param_groups for p in g["params"] if p.requires_grad]

    for step in range(1, opt_steps + 1):
        opt.zero_grad(set_to_none=True)
        acc_flow = 0.0
        acc_id = 0.0
        id_active_cnt = 0

        for _ in range(grad_accum):
            pick = train_arr[rng.integers(0, n_train, size=micro_batch)].tolist()
            target = load_512_batch(records, pick, dev)
            t = sample_t_batch(micro_batch, t_gen, dev)
            noise = torch.randn(target.shape, device=dev, dtype=torch.float32, generator=t_gen) * NOISE_SCALE
            x_t = target * t.view(-1, 1, 1, 1) + noise * (1.0 - t.view(-1, 1, 1, 1))
            x_in = x_t.detach().to(dtype=torch.bfloat16)

            txt_b = sem_hidden[pick].to(dev, dtype=torch.bfloat16)
            msk_b = sem_mask[pick].to(dev, dtype=torch.bfloat16)
            id_b = ident[pick].to(dev, dtype=torch.float32) if use_id else None
            geo_b = geo[pick].to(dev, dtype=torch.float32) if use_geo else None

            # CFG condition dropout: 10% full drop, 15% text-only drop; during O2 also 10% ID-drop to force C_geo usage!
            u_drop = rng.random()
            if u_drop < 0.10:
                msk_b = torch.zeros_like(msk_b)
                id_b = None
                geo_b = None
            elif (use_id or use_geo) and u_drop < 0.25:
                msk_b = torch.zeros_like(msk_b)
            elif use_id and use_geo and u_drop < 0.35:
                id_b = None

            pred = predict_clean_factorized(
                net,
                x_in,
                t,
                txt_b,
                msk_b,
                conditioner=conditioner if (id_b is not None or geo_b is not None) else None,
                identity=id_b,
                geo=geo_b,
                checkpoint=True,
            )
            flow_loss = official_v_loss(pred.float(), x_in.detach().float(), target, t)
            loss = flow_loss

            # Differentiable GlintR100 identity loss on clean prediction for t in [0.20, 0.70]
            if use_id and id_b is not None and id_embedder is not None and cropper is not None:
                in_win = (t >= 0.20) & (t <= 0.70)
                if bool(in_win.any()):
                    sub_pred = pred[in_win].float().clamp(-1.5, 1.5)
                    sub_gt = id_b[in_win]
                    sub_t = t[in_win]
                    feat = F.normalize(id_embedder(cropper(sub_pred)).float(), dim=-1)
                    per_id = 1.0 - (feat * F.normalize(sub_gt, dim=-1)).sum(dim=-1)
                    id_loss = (per_id * sub_t).mean()
                    loss = loss + id_loss_weight * id_loss
                    acc_id += float(per_id.detach().mean())
                    id_active_cnt += 1

            (loss / float(grad_accum)).backward()
            acc_flow += float(flow_loss.detach())

        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        opt.step()
        sched.step()

        vram_now = torch.cuda.max_memory_allocated() / (1024**3)
        if vram_now > VRAM_STOP_GB:
            raise RuntimeError(f"Peak VRAM {vram_now:.2f} GB exceeded cap {VRAM_STOP_GB} GB")

        if step == 1 or step % max(1, opt_steps // 10) == 0 or step == opt_steps:
            gsum = conditioner.gate_summary() if conditioner is not None else {}
            row = {
                "step": step,
                "flow_loss": acc_flow / float(grad_accum),
                "id_loss": (acc_id / id_active_cnt) if id_active_cnt > 0 else None,
                "gates": gsum,
                "vram_gb": vram_now,
                "elapsed_s": time.perf_counter() - t0,
            }
            history.append(row)
            id_str = f" id_loss={row['id_loss']:.4f}" if row["id_loss"] is not None else ""
            g_str = (
                f" id_g={gsum['id_tanh_abs_mean']:.3f} geo_g={gsum['geo_tanh_abs_mean']:.3f}"
                if gsum
                else ""
            )
            log(f"  [{stage_name}] step {step:4d}/{opt_steps} flow={row['flow_loss']:.4f}{id_str}{g_str} vram={vram_now:.2f}GB")

    return {
        "stage": stage_name,
        "opt_steps": opt_steps,
        "effective_batch": micro_batch * grad_accum,
        "seconds": time.perf_counter() - t0,
        "peak_vram_gb": peak_vram_gb(),
        "gates": conditioner.gate_summary() if conditioner is not None else None,
        "history": history,
    }


@torch.no_grad()
def sample_eval_set(
    net,
    conditioner: FactorizedJointConditioner | None,
    sem_hidden: torch.Tensor,
    sem_mask: torch.Tensor,
    ident: torch.Tensor | None,
    geo: torch.Tensor | None,
    eval_idx: list[int],
    dev: torch.device,
    steps: int = 32,
    cfg_scale: float = 2.5,
    batch_size: int = 4,
) -> torch.Tensor:
    net.eval()
    if conditioner is not None:
        conditioner.eval()
    outs = []
    for s in range(0, len(eval_idx), batch_size):
        pick = eval_idx[s : s + batch_size]
        b = len(pick)
        txt_b = sem_hidden[pick].to(dev, dtype=torch.bfloat16)
        msk_b = sem_mask[pick].to(dev, dtype=torch.bfloat16)
        id_tok = conditioner.project_id(ident[pick].to(dev)) if (conditioner is not None and ident is not None) else None
        geo_tok = conditioner.project_geo(geo[pick].to(dev)) if (conditioner is not None and geo is not None) else None

        gen = torch.Generator(device=dev).manual_seed(2026 + s)

        def _pred(x, t, text_in, mask_in, _id=id_tok, _geo=geo_tok):
            is_uncond = float(mask_in.detach().float().sum()) == 0.0
            return predict_clean_factorized(
                net,
                x,
                t,
                text_in,
                mask_in,
                conditioner=None if is_uncond else conditioner,
                id_tokens=None if is_uncond else _id,
                geo_tokens=None if is_uncond else _geo,
                checkpoint=False,
            )

        imgs = flower_sample(
            _pred,
            txt_b,
            msk_b,
            steps=steps,
            cfg_scale=cfg_scale,
            eta=0.0,
            image_size=512,
            noise_scale=NOISE_SCALE,
            generator=gen,
        )
        outs.append(imgs.float().cpu())
    return torch.cat(outs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o0-steps", type=int, default=350)
    parser.add_argument("--o1-steps", type=int, default=750)
    parser.add_argument("--o2-steps", type=int, default=450)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=48)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = torch.device("cuda")
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    log(f"Phase 1B Oracle Real starting on CUDA (free VRAM={free_gb:.2f} GB)")
    if free_gb < 3.0:
        raise RuntimeError(f"Need >= 3.0 GB free VRAM to run safely, only {free_gb:.2f} GB free!")

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    sem_hidden = torch.load(CACHE_DIR / "sem_hidden_bf16.pt", map_location="cpu", weights_only=True)
    sem_mask = torch.load(CACHE_DIR / "sem_mask_u8.pt", map_location="cpu", weights_only=True)
    ident = torch.load(CACHE_DIR / "identity_antelope.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    geo = lmks.reshape(lmks.shape[0], 212)
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()

    train_idx = [r["index"] for r in records if r["split"] == "train"]
    test_idx = pick_unique_identity_indices(records, "celeba_test", args.eval_n)
    if len(test_idx) < args.eval_n:
        # Supplement with strictly identity-disjoint CelebA-HQ test faces (cos < 0.35 vs all train)
        hq_extra = pick_unique_identity_indices(records, "celebahq_test", args.eval_n - len(test_idx))
        test_idx = test_idx + hq_extra
    log(f"Train={len(train_idx)}, Identity-Disjoint Unique-Person Eval={len(test_idx)}")

    gt_eval_imgs = load_512_batch(records, test_idx, torch.device("cpu"))
    gt_eval_lmks_px = lmks[test_idx] * 512.0

    # Load frozen GlintR100 training identity supervisor
    glint = GlintR100(GLINT_ONNX).to(dev).eval()
    for p in glint.parameters():
        p.requires_grad_(False)
    cropper = FixedArcfaceCrop().to(dev)

    # Load MiniT2I-B/16 + rank-16 QKVO LoRA + RMSNorm-safe FactorizedJointConditioner
    reset_vram()
    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev)
    net = wrapper.model.net
    wrapper.requires_grad_(False)
    seed_all(args.seed)
    lora_params = attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    for p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)
    conditioner = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev)

    # Stage O0: Oracle Semantic Only
    log("=== Training O0 (Oracle Semantic Only) ===")
    o0_train = train_stage(
        "O0", net, lora_params, None, False, False, records, sem_hidden, sem_mask, ident, geo,
        train_idx, dev, args.o0_steps, args.micro_batch, args.grad_accum,
        lr_lora=1e-4, lr_proj=5e-4, lr_gate=1e-2, seed=args.seed,
    )
    save_generator_ckpt(CKPT_DIR / "oracle_real_o0.pt", net.double_blocks, None, o0_train)
    o0_imgs = sample_eval_set(net, None, sem_hidden, sem_mask, None, None, test_idx, dev, steps=args.sample_steps, cfg_scale=args.cfg)

    # Stage O1: Oracle Semantic + Identity (initialized from O0!)
    log("=== Training O1 (Oracle Semantic + Identity) ===")
    o1_train = train_stage(
        "O1", net, lora_params, conditioner, True, False, records, sem_hidden, sem_mask, ident, geo,
        train_idx, dev, args.o1_steps, args.micro_batch, args.grad_accum,
        lr_lora=1e-4, lr_proj=5e-4, lr_gate=1.5e-2, id_embedder=glint, cropper=cropper, id_loss_weight=0.08, seed=args.seed + 1,
    )
    save_generator_ckpt(CKPT_DIR / "oracle_real_o1.pt", net.double_blocks, conditioner, o1_train)
    o1_imgs = sample_eval_set(net, conditioner, sem_hidden, sem_mask, ident, None, test_idx, dev, steps=args.sample_steps, cfg_scale=args.cfg)

    # Stage O2: Oracle Semantic + Identity + Canonical 106-pt Geometry (initialized from O1!)
    log("=== Training O2 (Oracle Semantic + Identity + Geometry) ===")
    o2_train = train_stage(
        "O2", net, lora_params, conditioner, True, True, records, sem_hidden, sem_mask, ident, geo,
        train_idx, dev, args.o2_steps, args.micro_batch, args.grad_accum,
        lr_lora=8e-5, lr_proj=5e-4, lr_gate=1.5e-2, id_embedder=glint, cropper=cropper, id_loss_weight=0.08, seed=args.seed + 2,
    )
    save_generator_ckpt(CKPT_DIR / "oracle_real_o2.pt", net.double_blocks, conditioner, o2_train)
    o2_imgs = sample_eval_set(net, conditioner, sem_hidden, sem_mask, ident, geo, test_idx, dev, steps=args.sample_steps, cfg_scale=args.cfg)

    # Zero-finetune predicted-condition check (feed Phase 1A DualDomainEncoder M=64 predictions into O2!)
    pred_imgs = None
    enc_ckpt = CKPT_DIR / "dual_encoder_real.pt"
    if enc_ckpt.is_file():
        log("=== Sampling O2 + Predicted Conditions (Phase 1A DualDomainEncoder @ M=64, zero finetune) ===")
        op = SPIOperator.from_simulation_256().to(dev)
        enc = PhysicsDualDomainFactorizer(op, max_measurements=256, dim=384, sem_tokens=int(sem_hidden.shape[1]), sem_dim=int(sem_hidden.shape[-1]), lam=2e-4, unfreeze_last_n=6).to(dev)
        enc.load_state_dict(torch.load(enc_ckpt, map_location="cpu", weights_only=False)["state_dict"])
        enc.eval()
        with torch.no_grad():
            idx64 = torch.arange(64, device=dev, dtype=torch.long)
            y64 = y256[test_idx, :64].to(dev)
            p_sem, p_id, p_geo = enc(y64, idx64)
        # Put predicted conditions into temporary full-size buffers indexed by 0..len(test_idx)-1
        local_idx = list(range(len(test_idx)))
        pred_mask = torch.ones((len(test_idx), int(sem_hidden.shape[1])), dtype=torch.uint8)
        pred_imgs = sample_eval_set(
            net, conditioner, p_sem.to(torch.bfloat16).cpu(), pred_mask, p_id.cpu(), p_geo.cpu(),
            local_idx, dev, steps=args.sample_steps, cfg_scale=args.cfg,
        )
        del enc, op
        torch.cuda.empty_cache()

    # Release generator before independent cross-model evaluation
    wrapper.to("cpu")
    glint.to("cpu")
    del wrapper, glint
    gc.collect()
    torch.cuda.empty_cache()

    log("=== Independent Cross-Model Evaluation (AdaFace IR-101 + Buffalo_L + 106-pt NME + LPIPS) ===")
    evaluator = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)
    eval_results = {
        "O0": evaluator.evaluate_pair(o0_imgs, gt_eval_imgs, gt_landmarks_px=gt_eval_lmks_px),
        "O1": evaluator.evaluate_pair(o1_imgs, gt_eval_imgs, gt_landmarks_px=gt_eval_lmks_px),
        "O2": evaluator.evaluate_pair(o2_imgs, gt_eval_imgs, gt_landmarks_px=gt_eval_lmks_px),
    }
    if pred_imgs is not None:
        eval_results["O2_PredCond_M64"] = evaluator.evaluate_pair(pred_imgs, gt_eval_imgs, gt_landmarks_px=gt_eval_lmks_px)

    # Gate checks (relative gains on independent cross-model metrics!)
    o0_ada = eval_results["O0"]["adaface_cos_mean"]
    o1_ada = eval_results["O1"]["adaface_cos_mean"]
    o0_buf = eval_results["O0"]["buffalo_cos_mean"]
    o1_buf = eval_results["O1"]["buffalo_cos_mean"]
    o1_nme = eval_results["O1"]["nme106_mean"]
    o2_nme = eval_results["O2"]["nme106_mean"]
    o0_lp = eval_results["O0"]["lpips_mean"]
    o2_lp = eval_results["O2"]["lpips_mean"]

    ada_rel = (o1_ada - o0_ada) / max(abs(o0_ada), 1e-4)
    buf_rel = (o1_buf - o0_buf) / max(abs(o0_buf), 1e-4) if (o0_buf is not None and o1_buf is not None) else None
    nme_drop = (o1_nme - o2_nme) / max(o1_nme, 1e-6) if (o1_nme is not None and o2_nme is not None) else None
    lpips_ratio = (o2_lp / max(o0_lp, 1e-6)) if (o0_lp is not None and o2_lp is not None) else None

    gate = {
        "cross_model_independent_eval": True,
        "train_id_teacher": "AntelopeV2 (Glint360K R100)",
        "eval_id_models": ["AdaFace IR-101 (WebFace12M)", "InsightFace Buffalo_L"],
        "n_eval_identity_disjoint": len(test_idx),
        "o0_adaface": o0_ada,
        "o1_adaface": o1_ada,
        "o2_adaface": eval_results["O2"]["adaface_cos_mean"],
        "o1_vs_o0_adaface_rel_gain": ada_rel,
        "o1_vs_o0_buffalo_rel_gain": buf_rel,
        "o2_vs_o1_nme106_rel_drop": nme_drop,
        "o2_vs_o0_lpips_ratio": lpips_ratio,
        "id_gate_pass_ge_10pct": bool(ada_rel >= 0.10 and (o1_ada - o0_ada) >= 0.05),
        "geo_gate_pass_ge_8pct": bool(nme_drop is not None and nme_drop >= 0.08),
    }

    # Save 16-face visual comparison grid
    n_vis = min(16, len(test_idx))
    tiles, titles = [], []
    cols = 5 if pred_imgs is not None else 4
    for i in range(n_vis):
        tiles.extend([gt_eval_imgs[i], o0_imgs[i], o1_imgs[i], o2_imgs[i]])
        titles.extend(["GT", "O0 (Sem)", "O1 (+ID)", "O2 (+Geo)"])
        if pred_imgs is not None:
            tiles.append(pred_imgs[i])
            titles.append("O2+Pred@64")
    save_image_grid(torch.stack(tiles), OUTPUTS / "phase1b_oracle_real_grid.png", nrow=cols, titles=titles)

    # Clean JSON (strip large per-sample lists from top summary if needed, keep compact)
    compact_eval = {k: {mk: mv for mk, mv in v.items() if mk != "per_sample"} for k, v in eval_results.items()}
    report = {
        "gate": gate,
        "eval": compact_eval,
        "train": {"O0": o0_train, "O1": o1_train, "O2": o2_train},
    }
    (OUTPUTS / "phase1b_oracle_real_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Phase 1B Real Oracle Results (Loop A — Cross-Model Evaluated)",
        "",
        f"Train: `{len(train_idx)}` real faces | Test: `{len(test_idx)}` strictly identity-disjoint CelebA test faces.",
        f"Sampling: `{args.sample_steps}` Euler steps, `CFG={args.cfg}`, `512x512` pixel-space MiniT2I-B/16.",
        "",
        "| Stage | AdaFace IR-101 ↑ | Buffalo_L ↑ | 106-pt NME ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | Det |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, v in compact_eval.items():
        md.append(
            f"| {k} | {v['adaface_cos_mean']:.4f} | "
            f"{v['buffalo_cos_mean']:.4f} | {v['nme106_mean']:.4f} | "
            f"{v['lpips_mean']:.4f} | {v['psnr_mean']:.2f} | {v['ssim_mean']:.4f} | {v['buffalo_detected']}/{v['n']} |"
        )
    md += ["", "## Gate Summary", "```json", json.dumps(gate, indent=2), "```"]
    (OUTPUTS / "phase1b_oracle_real_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("PHASE 1B ORACLE REAL COMPLETE!")
    log(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
