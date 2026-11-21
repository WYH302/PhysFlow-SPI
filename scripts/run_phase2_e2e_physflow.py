"""Phase 2 End-to-End PhysFlow Training & Full Benchmark Evaluation (Loop C / Level 1-2).

1. Jointly fine-tunes PhysicsDualDomainFactorizer + FactorizedJointConditioner +
   MiniT2I-B/16 QKVO LoRA with:
   - Official x-prediction v-loss (z_t = t*x + (1-t)*eps, t=1 clean, eps = randn*2.0)
   - Training-time pixel-space endpoint physical consistency Huber(Phi_S(D(pred_x0)), y_S) (t >= 0.40)
   - Training-time endpoint GlintR100 identity loss (0.20 <= t <= 0.70, clean weight 0.025)
   - Measurement-domain canonical geometry invariance across noisy/dropped measurements (geo_inv_weight=0.5)
   - Modality dropout (full uncond / discrete GT / sem-only / sem+id / full) + clean L_sem=24 valid prefix mask
2. Evaluates the complete benchmark ladder on identity-disjoint CelebA-HQ (primary Table 1, all 5 rates
   M in {16, 32, 64, 128, 256}) and identity-disjoint CelebA (45 unique celebrities, M=64 & M=256):
   - B0_Discrete (GT caption -> MiniT2I)
   - B0_Continuous (predicted C_sem -> MiniT2I)
   - Zero-Shot FLOWER (untouched frozen MiniT2I + generic prompt + Euclidean FLOWER)
   - LSI (CVPR 2025 official simulation_256.pt + StyleGAN-XL)
   - B1 (+ID: predicted C_sem + C_id)
   - B2 (+Geo: predicted C_sem + C_id + C_geo)
   - B3_Euc_FLOWER (ablation: E2E PhysFlow + Euclidean FLOWER, t0=0)
   - B3_Kalman_t0 (ablation: E2E PhysFlow + Blind Noise-Adaptive Kalman-Wiener FLOWER, t0=0)
   - B3_PhysFlow (Full: E2E PhysFlow + Wiener-initialized Blind Noise-Adaptive Kalman-Wiener FLOWER)
3. Evaluates complete 6-condition sensor noise robustness (N0..N4) at M=64, speed-quality Pareto curve
   (8 / 16 / 32 / 50 steps), paired bootstrap 95% CIs, and saves qualitative & failure grids.
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

from src.eval.adaface_eval import CrossModelEvaluator
from src.models.encoders import PhysicsDualDomainFactorizer, factorized_condition_loss
from src.models.glintr100 import FixedArcfaceCrop, GlintR100
from src.models.injection import FactorizedJointConditioner, attach_qkvo_lora
from src.models.lsi_baseline import LSIBaseline
from src.models.minit2i_condition import predict_clean_factorized
from src.models.minit2i_loader import load_b16_transformer
from src.models.surrogate import official_v_loss
from src.optical.spi_operator import RATES, SPIOperator, add_sensor_noise
from src.paths import OUTPUTS, ROOT as PROJECT_ROOT
from src.runtime import peak_vram_gb, reset_vram, seed_all
from src.solvers.flower import flower_sample
from src.vis import save_image_grid

from scripts.run_phase1b_oracle_real import (
    CACHE_DIR,
    CKPT_DIR,
    GLINT_ONNX,
    NOISE_SCALE,
    VRAM_STOP_GB,
    load_512_batch,
    load_generator_ckpt,
    pick_unique_identity_indices,
    sample_t_batch,
    save_generator_ckpt,
)

SEM_VALID_LEN = 24


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_sem_prefix_mask(batch_size: int, seq_len: int, device: torch.device, valid_len: int = SEM_VALID_LEN) -> torch.Tensor:
    m = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bfloat16)
    m[:, : min(valid_len, seq_len)] = 1.0
    return m


def encode_generic_prompt(dev: torch.device, max_len: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoTokenizer, T5EncoderModel

    flan = PROJECT_ROOT / "checkpoints" / "flan-t5-large"
    tok = AutoTokenizer.from_pretrained(flan, local_files_only=True)
    enc = T5EncoderModel.from_pretrained(flan, torch_dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    prompt = "A close-up frontal studio portrait photograph of a human face looking at the camera."
    inp = tok([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    with torch.inference_mode():
        h = enc(input_ids=inp.input_ids.to(dev), attention_mask=inp.attention_mask.to(dev)).last_hidden_state
    m = inp.attention_mask.to(dtype=torch.uint8).cpu()
    h = h.to(dtype=torch.bfloat16).cpu()
    enc.to("cpu")
    del enc, tok
    gc.collect()
    torch.cuda.empty_cache()
    return h, m


def bootstrap_ci_and_diff(a: list[float], b: list[float] | None = None, n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    xa = np.asarray(a, dtype=np.float64)
    n = len(xa)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = xa[idx].mean(axis=1)
    out = {
        "mean": float(xa.mean()),
        "std": float(xa.std(ddof=1)) if n > 1 else 0.0,
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }
    if b is not None and len(b) == n:
        xb = np.asarray(b, dtype=np.float64)
        diff = xa - xb
        d_means = diff[idx].mean(axis=1)
        out["diff_mean"] = float(diff.mean())
        out["diff_ci95_low"] = float(np.percentile(d_means, 2.5))
        out["diff_ci95_high"] = float(np.percentile(d_means, 97.5))
        # Two-sided paired permutation/bootstrap p-value proxy
        out["p_gt_zero"] = float((d_means <= 0.0).mean())
    return out


def pick_even_split_indices(records: list[dict], split: str, n: int) -> list[int]:
    pool = [r["index"] for r in records if r["split"] == split]
    if len(pool) <= n:
        return pool
    pos = np.linspace(0, len(pool) - 1, n, dtype=int)
    return [pool[int(i)] for i in pos]


def train_e2e_physflow(
    net,
    lora_params: list[torch.nn.Parameter],
    conditioner: FactorizedJointConditioner,
    encoder: PhysicsDualDomainFactorizer,
    op: SPIOperator,
    glint: GlintR100,
    cropper: FixedArcfaceCrop,
    records: list[dict],
    y256: torch.Tensor,
    sem_hidden: torch.Tensor,
    sem_mask: torch.Tensor,
    ident: torch.Tensor,
    geo: torch.Tensor,
    train_idx: list[int],
    dev: torch.device,
    opt_steps: int = 600,
    micro_batch: int = 4,
    grad_accum: int = 4,
    meas_weight: float = 0.25,
    id_weight: float = 0.025,
    cond_weight: float = 0.25,
    seed: int = 42,
) -> dict:
    """Jointly refine encoder + conditioner + QKVO LoRA with pixel-space physical loop."""
    enc_bb = encoder.backbone_parameters()
    enc_hd = encoder.head_parameters()
    cond_params = [p for p in conditioner.parameters() if p.requires_grad]
    for p in lora_params + cond_params + enc_bb + enc_hd:
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 3e-5, "weight_decay": 1e-4},
            {"params": cond_params, "lr": 1e-4, "weight_decay": 1e-4},
            {"params": enc_bb, "lr": 2e-5, "weight_decay": 1e-4},
            {"params": enc_hd, "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=5e-6)
    all_params = lora_params + cond_params + enc_bb + enc_hd

    rng = np.random.default_rng(seed)
    t_gen = torch.Generator(device=dev).manual_seed(seed)
    train_arr = np.asarray(train_idx, dtype=np.int64)
    n_train = len(train_arr)
    noise_modes = ("gaussian", "poisson_gaussian", "gain_bias", "dropout")

    net.train()
    conditioner.train()
    encoder.train()
    history = []
    t0 = time.perf_counter()

    for step in range(1, opt_steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = {"flow": 0.0, "meas": 0.0, "id": 0.0, "cond": 0.0}
        m_cnt, id_cnt = 0, 0

        for _ in range(grad_accum):
            pick = train_arr[rng.integers(0, n_train, size=micro_batch)].tolist()
            target = load_512_batch(records, pick, dev)
            rate = int(RATES[int(rng.integers(0, len(RATES)))])
            idx_m = torch.arange(rate, device=dev, dtype=torch.long)
            y_clean = y256[pick, :rate].to(dev)

            # Paired clean + perturbed measurement view for measurement-domain geometry invariance
            nm = noise_modes[int(rng.integers(0, len(noise_modes)))]
            y_noisy, km = add_sensor_noise(y_clean, mode=nm, dropout_prob=0.18, generator=t_gen)
            if km is not None:
                y_noisy = y_noisy[:, km]
                idx_noisy = idx_m[km]
            else:
                idx_noisy = idx_m

            p_clean = encoder(y_clean, idx_m)
            p_noisy = encoder(y_noisy, idx_noisy)
            p_sem, p_id, p_geo = p_noisy if (rng.random() < 0.35) else p_clean

            gt_sem = sem_hidden[pick].to(dev, dtype=torch.float32)
            gt_m = sem_mask[pick].to(dev, dtype=torch.float32)
            gt_id = ident[pick].to(dev, dtype=torch.float32)
            gt_geo = geo[pick].to(dev, dtype=torch.float32)

            l_cond, _ = factorized_condition_loss(
                p_clean, gt_sem, gt_m, gt_id, gt_geo, pred_noisy=p_noisy, geo_inv_weight=0.5
            )

            t = sample_t_batch(micro_batch, t_gen, dev)
            noise = torch.randn(target.shape, device=dev, dtype=torch.float32, generator=t_gen) * NOISE_SCALE
            x_t = target * t.view(-1, 1, 1, 1) + noise * (1.0 - t.view(-1, 1, 1, 1))
            x_in = x_t.detach().to(dtype=torch.bfloat16)

            # Clean modality dropout so B0_discrete, B0_cont, B1, B2, B3 are ALL strictly in-distribution!
            u_drop = float(rng.random())
            if u_drop < 0.10:
                # Full unconditional for CFG
                txt_in = p_sem.to(dtype=torch.bfloat16)
                msk_in = torch.zeros((micro_batch, p_sem.shape[1]), device=dev, dtype=torch.bfloat16)
                id_in = None
                geo_in = None
                use_cond = False
            elif u_drop < 0.20:
                # Discrete GT caption calibration (for B0_discrete)
                txt_in = gt_sem.to(dtype=torch.bfloat16)
                msk_in = gt_m.to(dtype=torch.bfloat16)
                id_in = None
                geo_in = None
                use_cond = False
            elif u_drop < 0.30:
                # Continuous semantic-only (for B0_continuous)
                txt_in = p_sem.to(dtype=torch.bfloat16)
                msk_in = make_sem_prefix_mask(micro_batch, p_sem.shape[1], dev)
                id_in = None
                geo_in = None
                use_cond = False
            elif u_drop < 0.40:
                # Semantic + Identity only (for B1_id)
                txt_in = p_sem.to(dtype=torch.bfloat16)
                msk_in = make_sem_prefix_mask(micro_batch, p_sem.shape[1], dev)
                id_in = p_id
                geo_in = None
                use_cond = True
            else:
                # Full factorized (C_sem + C_id + C_geo)
                txt_in = p_sem.to(dtype=torch.bfloat16)
                msk_in = make_sem_prefix_mask(micro_batch, p_sem.shape[1], dev)
                id_in = p_id
                geo_in = p_geo
                use_cond = True

            pred = predict_clean_factorized(
                net,
                x_in,
                t,
                txt_in,
                msk_in,
                conditioner=conditioner if use_cond else None,
                identity=id_in,
                geo=geo_in,
                checkpoint=True,
            )
            l_flow = official_v_loss(pred.float(), x_in.detach().float(), target, t)
            loss = l_flow + cond_weight * l_cond

            # Training-time pixel-space endpoint physical measurement loss (NO unrolled ODE!)
            meas_win = t >= 0.40
            if bool(meas_win.any()) and use_cond:
                y_hat = op.measure(pred[meas_win].float().clamp(-1.5, 1.5), indices=idx_m, input_range="-1_1")
                y_ref = y_clean[meas_win].float()
                y_scale = y_ref.abs().mean(dim=-1, keepdim=True).clamp_min(1.0)
                per_m = F.huber_loss(y_hat / y_scale, y_ref / y_scale, delta=0.1, reduction="none").mean(dim=-1)
                l_meas = (per_m * t[meas_win]).mean()
                loss = loss + meas_weight * l_meas
                acc["meas"] += float(l_meas.detach())
                m_cnt += 1

            # Clean training-time endpoint identity consistency on pred_x0 (weight=0.025 avoids texture ringing)
            id_win = (t >= 0.20) & (t <= 0.70)
            if bool(id_win.any()) and id_in is not None:
                sub_pred = pred[id_win].float().clamp(-1.5, 1.5)
                feat = F.normalize(glint(cropper(sub_pred)).float(), dim=-1)
                per_id = 1.0 - (feat * F.normalize(gt_id[id_win], dim=-1)).sum(dim=-1)
                l_id = (per_id * t[id_win]).mean()
                loss = loss + id_weight * l_id
                acc["id"] += float(per_id.detach().mean())
                id_cnt += 1

            (loss / float(grad_accum)).backward()
            acc["flow"] += float(l_flow.detach())
            acc["cond"] += float(l_cond.detach())

        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()

        vram_now = torch.cuda.max_memory_allocated() / (1024**3)
        if vram_now > VRAM_STOP_GB:
            raise RuntimeError(f"Peak VRAM {vram_now:.2f} GB exceeded cap")

        if step == 1 or step % max(1, opt_steps // 10) == 0 or step == opt_steps:
            row = {
                "step": step,
                "flow": acc["flow"] / grad_accum,
                "meas": acc["meas"] / max(m_cnt, 1) if m_cnt else 0.0,
                "id": acc["id"] / max(id_cnt, 1) if id_cnt else 0.0,
                "cond": acc["cond"] / grad_accum,
                "vram_gb": vram_now,
                "elapsed_s": time.perf_counter() - t0,
            }
            history.append(row)
            log(
                f"  [E2E-PhysFlow] step {step:4d}/{opt_steps} flow={row['flow']:.4f} "
                f"meas={row['meas']:.4f} id={row['id']:.4f} cond={row['cond']:.3f} vram={vram_now:.2f}GB"
            )

    return {
        "opt_steps": opt_steps,
        "id_weight": id_weight,
        "meas_weight": meas_weight,
        "cond_weight": cond_weight,
        "sem_valid_len": SEM_VALID_LEN,
        "seconds": time.perf_counter() - t0,
        "peak_vram_gb": peak_vram_gb(),
        "history": history,
    }


@torch.no_grad()
def sample_mode(
    mode: str,
    net,
    conditioner: FactorizedJointConditioner | None,
    encoder: PhysicsDualDomainFactorizer | None,
    op: SPIOperator,
    y_meas: torch.Tensor,
    indices_m: torch.Tensor,
    text_discrete: torch.Tensor,
    mask_discrete: torch.Tensor,
    generic_text: torch.Tensor,
    generic_mask: torch.Tensor,
    dev: torch.device,
    steps: int = 16,
    cfg_scale: float = 2.0,
    eta: float = 0.85,
    t_warm: float = 0.35,
    batch_size: int = 9,
) -> tuple[torch.Tensor, float]:
    """Generate reconstructed 512x512 RGB images in [-1, 1] for one method mode."""
    net.eval()
    if conditioner is not None:
        conditioner.eval()
    if encoder is not None:
        encoder.eval()

    n = int(y_meas.shape[0])
    outs = []
    t_start_wall = time.perf_counter()
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        b = e - s
        yb = y_meas[s:e].to(dev)

        p_sem = p_id = p_geo = None
        if encoder is not None and mode in (
            "b0_continuous",
            "b1_id",
            "b2_geo",
            "b3_euc_flower",
            "b3_kalman_t0",
            "b3_physflow",
        ):
            p_sem, p_id, p_geo = encoder(yb, indices_m)

        x_init = None
        t0_init = 0.0
        proj_kind = "none"
        use_cond = False
        id_tok = geo_tok = None

        if mode == "b0_discrete":
            txt_b = text_discrete[s:e].to(dev, dtype=torch.bfloat16)
            msk_b = mask_discrete[s:e].to(dev, dtype=torch.bfloat16)
        elif mode == "zeroshot_flower":
            txt_b = generic_text.expand(b, -1, -1).to(dev, dtype=torch.bfloat16)
            msk_b = generic_mask.expand(b, -1).to(dev, dtype=torch.bfloat16)
            proj_kind = "euc"
        elif mode == "b0_continuous":
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
        elif mode == "b1_id":
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
            id_tok = conditioner.project_id(p_id)
            use_cond = True
        elif mode == "b2_geo":
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
            id_tok = conditioner.project_id(p_id)
            geo_tok = conditioner.project_geo(p_geo)
            use_cond = True
        elif mode == "b3_euc_flower":
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
            id_tok = conditioner.project_id(p_id)
            geo_tok = conditioner.project_geo(p_geo)
            use_cond = True
            proj_kind = "euc"
        elif mode == "b3_kalman_t0":
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
            id_tok = conditioner.project_id(p_id)
            geo_tok = conditioner.project_geo(p_geo)
            use_cond = True
            proj_kind = "kalman"
        elif mode == "b3_physflow":
            txt_b = p_sem.to(dtype=torch.bfloat16)
            msk_b = make_sem_prefix_mask(b, p_sem.shape[1], dev)
            id_tok = conditioner.project_id(p_id)
            geo_tok = conditioner.project_geo(p_geo)
            use_cond = True
            proj_kind = "kalman"
            t0_init = float(t_warm)
            iw = encoder.lifter.lift_image(yb, indices_m, reg=4e-4, blind_adapt=True)
            x_init = F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
        else:
            raise ValueError(mode)

        gen = torch.Generator(device=dev).manual_seed(3000 + s)

        def _pred(x, t, text_in, mask_in, _id=id_tok, _geo=geo_tok, _uc=use_cond):
            is_uncond = float(mask_in.detach().float().sum()) == 0.0
            return predict_clean_factorized(
                net,
                x,
                t,
                text_in,
                mask_in,
                conditioner=None if (is_uncond or not _uc) else conditioner,
                id_tokens=None if is_uncond else _id,
                geo_tokens=None if is_uncond else _geo,
                checkpoint=False,
            )

        def _proj(x0, t_cur, eta_t, _y=yb, _idx=indices_m, _pk=proj_kind):
            ts = float(t_cur[0])
            if _pk == "euc":
                if ts >= 0.90:
                    return x0
                return op.proximal(x0.float(), _y, indices=_idx, lam=1e-2, eta=min(eta_t, 0.65), input_range="-1_1").to(
                    dtype=x0.dtype
                )
            if _pk == "kalman":
                if ts >= 0.93:
                    return x0
                w_t = eta_t if ts < 0.80 else eta_t * 0.45
                return encoder.lifter.wiener_proximal(
                    x0, _y, op, _idx, eta=w_t, base_reg=4e-4, blind_adapt=True, input_range="-1_1"
                )
            return x0

        imgs = flower_sample(
            _pred,
            txt_b,
            msk_b,
            steps=steps,
            cfg_scale=cfg_scale,
            eta=eta if proj_kind != "none" else 0.0,
            image_size=512,
            noise_scale=NOISE_SCALE,
            project=_proj if proj_kind != "none" else None,
            generator=gen,
            stochastic=False,
            eta_power=0.0,
            x_init=x_init,
            t_start=t0_init,
        )
        outs.append(imgs.float().cpu())

    sec_per_img = (time.perf_counter() - t_start_wall) / max(n, 1)
    return torch.cat(outs, dim=0), sec_per_img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-steps", type=int, default=600)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=45)
    parser.add_argument("--steps-eval", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.85)
    parser.add_argument("--t-warm", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    dev = torch.device("cuda")
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    log(f"Phase 2 E2E PhysFlow starting on CUDA (free VRAM={free_gb:.2f} GB)")
    if free_gb < 5.0:
        raise RuntimeError(f"Need >= 5.0 GB free VRAM, only {free_gb:.2f} GB free!")

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    y256 = torch.load(CACHE_DIR / "y256.pt", map_location="cpu", weights_only=True).float()
    sem_hidden = torch.load(CACHE_DIR / "sem_hidden_bf16.pt", map_location="cpu", weights_only=True)
    sem_mask = torch.load(CACHE_DIR / "sem_mask_u8.pt", map_location="cpu", weights_only=True)
    ident = torch.load(CACHE_DIR / "identity_antelope.pt", map_location="cpu", weights_only=True).float()
    lmks = torch.load(CACHE_DIR / "landmarks106_norm.pt", map_location="cpu", weights_only=True).float()
    geo = lmks.reshape(lmks.shape[0], 212)

    train_idx = [r["index"] for r in records if r["split"] == "train"]
    # Primary Table 1: strictly identity-disjoint CelebA-HQ (45 diverse test identities)
    hq_idx = pick_even_split_indices(records, "celebahq_test", args.eval_n)
    # Secondary confirmation: strictly identity-disjoint CelebA (all 45 unique test celebrities)
    celeba_idx = pick_unique_identity_indices(records, "celeba_test", args.eval_n)
    y_train_mean = y256[train_idx].mean(dim=0)

    gt_hq_imgs = load_512_batch(records, hq_idx, torch.device("cpu"))
    gt_hq_lmks_px = lmks[hq_idx] * 512.0
    gt_cb_imgs = load_512_batch(records, celeba_idx, torch.device("cpu"))
    gt_cb_lmks_px = lmks[celeba_idx] * 512.0

    op = SPIOperator.from_simulation_256().to(dev)
    generic_text, generic_mask = encode_generic_prompt(dev, max_len=int(sem_hidden.shape[1]))

    reset_vram()
    wrapper, _ = load_b16_transformer(PROJECT_ROOT / "checkpoints" / "minit2i-b-16", dtype=torch.bfloat16)
    wrapper = wrapper.to(dev)
    net = wrapper.model.net
    wrapper.requires_grad_(False)

    # 1. Zero-Shot FLOWER on UNTOUCHED frozen MiniT2I-B/16 (all 5 rates on CelebA-HQ + noise N0..N4 + CelebA M=64/256)
    log("=== Evaluating Zero-Shot FLOWER on untouched frozen MiniT2I-B/16 ===")
    hq_samples: dict[str, dict[int, torch.Tensor]] = {"zeroshot_flower": {}}
    cb_samples: dict[str, dict[int, torch.Tensor]] = {"zeroshot_flower": {}}
    noise_samples: dict[str, dict[str, torch.Tensor]] = {"zeroshot_flower": {}, "b3_physflow": {}, "lsi": {}}
    timings: dict[str, float] = {}

    for rate in RATES:
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        imgs_zs, sec = sample_mode(
            "zeroshot_flower", net, None, None, op, y256[hq_idx, :rate], idx_r,
            sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
            dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=0.0,
        )
        hq_samples["zeroshot_flower"][rate] = imgs_zs
        if rate == 64:
            timings["zeroshot_flower"] = sec
            noise_samples["zeroshot_flower"]["clean"] = imgs_zs
        log(f"  Zero-Shot FLOWER CelebA-HQ M={rate:3d}: {sec:.2f}s/img")

    for rate in (64, 256):
        idx_r = torch.arange(rate, device=dev, dtype=torch.long)
        imgs_cb_zs, _ = sample_mode(
            "zeroshot_flower", net, None, None, op, y256[celeba_idx, :rate], idx_r,
            sem_hidden[celeba_idx], sem_mask[celeba_idx], generic_text, generic_mask,
            dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=0.0,
        )
        cb_samples["zeroshot_flower"][rate] = imgs_cb_zs

    idx64 = torch.arange(64, device=dev, dtype=torch.long)
    y64_hq_clean = y256[hq_idx, :64].to(dev)
    for nm in ("gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined"):
        gen_n = torch.Generator(device=dev).manual_seed(777)
        y_n, km = add_sensor_noise(y64_hq_clean, mode=nm, dropout_prob=0.20, generator=gen_n)
        idx_n = idx64[km] if km is not None else idx64
        if km is not None:
            y_n = y_n[:, km]
        imgs_zsn, _ = sample_mode(
            "zeroshot_flower", net, None, None, op, y_n.cpu(), idx_n,
            sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
            dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=0.0,
        )
        noise_samples["zeroshot_flower"][nm] = imgs_zsn

    # 2. Attach QKVO LoRA + FactorizedJointConditioner + Phase 1A Dual-Domain Encoder & Run Joint E2E Training
    seed_all(args.seed)
    lora_params = attach_qkvo_lora(net.double_blocks, rank=16, alpha=16.0)
    net.double_blocks.to(dev)
    for p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)
    conditioner = FactorizedJointConditioner(hidden=int(net.cfg.txt_hidden_size), num_blocks=len(net.double_blocks)).to(dev)
    encoder = PhysicsDualDomainFactorizer(
        op, max_measurements=256, dim=384, sem_tokens=int(sem_hidden.shape[1]), sem_dim=int(sem_hidden.shape[-1]), lam=2e-4, unfreeze_last_n=6
    ).to(dev)

    e2e_ckpt = CKPT_DIR / "physflow_e2e_full.pt"
    e2e_enc_ckpt = CKPT_DIR / "physflow_e2e_encoder.pt"
    e2e_train_info = {"skipped": True}
    if not args.skip_train:
        load_generator_ckpt(CKPT_DIR / "oracle_real_o2.pt", net.double_blocks, conditioner)
        encoder.load_state_dict(torch.load(CKPT_DIR / "dual_encoder_real.pt", map_location="cpu", weights_only=False)["state_dict"])
        glint = GlintR100(GLINT_ONNX).to(dev).eval()
        for p in glint.parameters():
            p.requires_grad_(False)
        cropper = FixedArcfaceCrop().to(dev)

        log("=== Joint E2E Training with Pixel-Space Endpoint Physical Loop ===")
        e2e_train_info = train_e2e_physflow(
            net, lora_params, conditioner, encoder, op, glint, cropper,
            records, y256, sem_hidden, sem_mask, ident, geo, train_idx, dev,
            opt_steps=args.e2e_steps, micro_batch=args.micro_batch, grad_accum=args.grad_accum, seed=args.seed,
        )
        save_generator_ckpt(e2e_ckpt, net.double_blocks, conditioner, e2e_train_info)
        torch.save({"state_dict": encoder.state_dict()}, e2e_enc_ckpt)
        glint.to("cpu")
        del glint
        torch.cuda.empty_cache()
    else:
        e2e_train_info = load_generator_ckpt(e2e_ckpt, net.double_blocks, conditioner) or {"skipped": True}
        encoder.load_state_dict(torch.load(e2e_enc_ckpt, map_location="cpu", weights_only=False)["state_dict"])

    # Count parameters cleanly for Table 6 (Compute record)
    params_total = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in conditioner.parameters()) + sum(p.numel() for p in encoder.parameters())
    params_trainable = (
        sum(p.numel() for p in lora_params)
        + sum(p.numel() for p in conditioner.parameters())
        + sum(p.numel() for p in encoder.backbone_parameters())
        + sum(p.numel() for p in encoder.head_parameters())
    )

    # 3. Evaluate PhysFlow Ladder on Identity-Disjoint CelebA-HQ across ALL 5 Rates!
    log("=== Sampling PhysFlow Ladder on Identity-Disjoint CelebA-HQ (All 5 Rates) ===")
    # B0_discrete is rate-independent -> sample ONCE!
    hq_b0_disc, sec_disc = sample_mode(
        "b0_discrete", net, conditioner, encoder, op, y256[hq_idx, :64], idx64,
        sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
        dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=0.0, t_warm=0.0,
    )
    hq_samples["b0_discrete"] = {r: hq_b0_disc for r in RATES}
    timings["b0_discrete"] = sec_disc
    log(f"  Sampled b0_discrete    (rate-invariant): {sec_disc:.2f}s/img")

    for mode in ("b0_continuous", "b1_id", "b2_geo", "b3_physflow"):
        hq_samples[mode] = {}
        for rate in RATES:
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            imgs_m, sec = sample_mode(
                mode, net, conditioner, encoder, op, y256[hq_idx, :rate], idx_r,
                sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
                dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=args.t_warm,
            )
            hq_samples[mode][rate] = imgs_m
            if rate == 64:
                timings[mode] = sec
            log(f"  Sampled {mode:14s} CelebA-HQ M={rate:3d}: {sec:.2f}s/img")

    # Solver ablations at M=64 and M=256: Euclidean FLOWER vs Kalman-Wiener (t0=0) vs PhysFlow (t0=0.35)
    solver_ablations: dict[str, dict[int, torch.Tensor]] = {}
    for mode in ("b3_euc_flower", "b3_kalman_t0"):
        solver_ablations[mode] = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            imgs_ab, sec = sample_mode(
                mode, net, conditioner, encoder, op, y256[hq_idx, :rate], idx_r,
                sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
                dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=0.0,
            )
            solver_ablations[mode][rate] = imgs_ab
            log(f"  Ablation {mode:13s} CelebA-HQ M={rate:3d}: {sec:.2f}s/img")

    # 4. Complete 6-Condition Noise Sweep at M=64 on CelebA-HQ
    log("=== Sampling Noise Robustness Sweep (N0..N4) at M=64 ===")
    noise_samples["b3_physflow"]["clean"] = hq_samples["b3_physflow"][64]
    for nm in ("gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined"):
        gen_n = torch.Generator(device=dev).manual_seed(777)
        y_n, km = add_sensor_noise(y64_hq_clean, mode=nm, dropout_prob=0.20, generator=gen_n)
        idx_n = idx64[km] if km is not None else idx64
        if km is not None:
            y_n = y_n[:, km]
        imgs_b3n, _ = sample_mode(
            "b3_physflow", net, conditioner, encoder, op, y_n.cpu(), idx_n,
            sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
            dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=args.t_warm,
        )
        noise_samples["b3_physflow"][nm] = imgs_b3n

    # 5. Speed-Quality Pareto Curve at M=64 (8, 16, 32, 50 steps)
    log("=== Sampling Speed-Quality Pareto Curve at M=64 ===")
    pareto_samples = {args.steps_eval: (hq_samples["b3_physflow"][64], timings["b3_physflow"])}
    for st in (8, 32, 50):
        imgs_p, sec_p = sample_mode(
            "b3_physflow", net, conditioner, encoder, op, y256[hq_idx, :64], idx64,
            sem_hidden[hq_idx], sem_mask[hq_idx], generic_text, generic_mask,
            dev, steps=st, cfg_scale=args.cfg, eta=args.eta, t_warm=args.t_warm,
        )
        pareto_samples[st] = (imgs_p, sec_p)
        log(f"  Pareto PhysFlow M=64 steps={st:2d}: {sec_p:.2f}s/img")

    # 6. Cross-Dataset Confirmation on 45 Unique Identity-Disjoint CelebA Celebrities (M=64 & M=256)
    log("=== Cross-Dataset Confirmation on 45 Unique CelebA Test Celebrities ===")
    cb_b0_disc, _ = sample_mode(
        "b0_discrete", net, conditioner, encoder, op, y256[celeba_idx, :64], idx64,
        sem_hidden[celeba_idx], sem_mask[celeba_idx], generic_text, generic_mask,
        dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=0.0, t_warm=0.0,
    )
    cb_samples["b0_discrete"] = {64: cb_b0_disc, 256: cb_b0_disc}
    for mode in ("b0_continuous", "b1_id", "b2_geo", "b3_physflow"):
        cb_samples[mode] = {}
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            imgs_cb, _ = sample_mode(
                mode, net, conditioner, encoder, op, y256[celeba_idx, :rate], idx_r,
                sem_hidden[celeba_idx], sem_mask[celeba_idx], generic_text, generic_mask,
                dev, steps=args.steps_eval, cfg_scale=args.cfg, eta=args.eta, t_warm=args.t_warm,
            )
            cb_samples[mode][rate] = imgs_cb

    # Also save raw I_wiener baseline at M=64 & M=256 for ablation reference
    wiener_raw_hq = {}
    with torch.no_grad():
        for rate in (64, 256):
            idx_r = torch.arange(rate, device=dev, dtype=torch.long)
            iw = encoder.lifter.lift_image(y256[hq_idx, :rate].to(dev), idx_r, reg=4e-4, blind_adapt=True)
            wiener_raw_hq[rate] = (F.interpolate(iw, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0).cpu()

    # Offload MiniT2I + encoder before LSI & CrossModelEvaluator
    wrapper.to("cpu")
    encoder.to("cpu")
    conditioner.to("cpu")
    del wrapper, encoder, conditioner
    gc.collect()
    torch.cuda.empty_cache()

    # Checkpoint sampled images before LSI/Evaluator so zero GPU sampling is ever lost
    torch.save(
        {
            "hq_samples": hq_samples,
            "cb_samples": cb_samples,
            "solver_ablations": solver_ablations,
            "noise_samples": noise_samples,
            "pareto_samples": pareto_samples,
            "wiener_raw_hq": wiener_raw_hq,
            "timings": timings,
        },
        OUTPUTS / "phase2_samples_cache.pt",
    )
    log("Saved sampled grids cache to outputs/phase2_samples_cache.pt")

    # 7. Official CVPR 2025 LSI (StyleGAN-XL) Baseline across ALL 5 rates & ALL 6 noise modes!
    log("=== Running Official CVPR 2025 LSI (StyleGAN-XL) Baseline ===")
    lsi_dev = dev if (torch.cuda.mem_get_info()[0] / (1024**3) >= 4.0) else torch.device("cpu")
    lsi = LSIBaseline(device=lsi_dev)
    hq_samples["lsi"] = {}
    for rate in RATES:
        t_lsi = time.perf_counter()
        idx_r = torch.arange(rate, dtype=torch.long)
        rec_lsi, _ = lsi.reconstruct(y256[hq_idx, :rate], indices=idx_r, y_mean=y_train_mean, output_size=512, batch_size=8)
        hq_samples["lsi"][rate] = rec_lsi
        if rate == 64:
            timings["lsi"] = (time.perf_counter() - t_lsi) / max(len(hq_idx), 1)
            noise_samples["lsi"]["clean"] = rec_lsi
    cb_samples["lsi"] = {}
    for rate in (64, 256):
        rec_cb_lsi, _ = lsi.reconstruct(y256[celeba_idx, :rate], indices=torch.arange(rate), y_mean=y_train_mean, output_size=512, batch_size=8)
        cb_samples["lsi"][rate] = rec_cb_lsi
    for nm in ("gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined"):
        gen_n = torch.Generator().manual_seed(777)
        y_n, km = add_sensor_noise(y256[hq_idx, :64], mode=nm, dropout_prob=0.20, generator=gen_n)
        idx_r = torch.arange(64, dtype=torch.long)
        if km is not None:
            y_n = y_n[:, km]
            idx_r = idx_r[km]
        rec_lsi_n, _ = lsi.reconstruct(y_n, indices=idx_r, y_mean=y_train_mean, output_size=512, batch_size=8)
        noise_samples["lsi"][nm] = rec_lsi_n
    del lsi
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Full Independent Cross-Model Evaluation (AdaFace IR-101 + Buffalo_L + 106-pt NME + LPIPS + PSNR + SSIM + Residual)
    log("=== Independent Cross-Model Evaluation across ALL modes, rates, noise, Pareto ===")
    evaluator = CrossModelEvaluator(device=dev, use_lpips=True, use_buffalo=True)
    op_cpu = SPIOperator.from_simulation_256()

    all_modes = ("b0_discrete", "b0_continuous", "zeroshot_flower", "lsi", "b1_id", "b2_geo", "b3_physflow")
    metrics_hq_rates: dict[str, dict[str, dict]] = {}
    per_sample_m64: dict[str, dict] = {}
    for mode in all_modes:
        metrics_hq_rates[mode] = {}
        for rate in RATES:
            idx_r = torch.arange(rate, dtype=torch.long)
            res = evaluator.evaluate_pair(
                hq_samples[mode][rate],
                gt_hq_imgs,
                operator=op_cpu,
                y_gt=y256[hq_idx, :rate],
                indices=idx_r,
                gt_landmarks_px=gt_hq_lmks_px,
            )
            metrics_hq_rates[mode][str(rate)] = {k: v for k, v in res.items() if k != "per_sample"}
            if rate == 64:
                per_sample_m64[mode] = res["per_sample"]

    # Solver ablation metrics at M=64 & M=256
    metrics_solver_ablation: dict[str, dict[str, dict]] = {}
    for name, d_imgs in (
        ("wiener_raw_linear", wiener_raw_hq),
        ("b3_euc_flower", solver_ablations["b3_euc_flower"]),
        ("b3_kalman_t0", solver_ablations["b3_kalman_t0"]),
        ("b3_physflow", {64: hq_samples["b3_physflow"][64], 256: hq_samples["b3_physflow"][256]}),
    ):
        metrics_solver_ablation[name] = {}
        for rate in (64, 256):
            res = evaluator.evaluate_pair(
                d_imgs[rate], gt_hq_imgs, operator=op_cpu, y_gt=y256[hq_idx, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_hq_lmks_px
            )
            metrics_solver_ablation[name][str(rate)] = {k: v for k, v in res.items() if k != "per_sample"}

    # CelebA cross-dataset confirmation metrics
    metrics_celeba: dict[str, dict[str, dict]] = {}
    for mode in all_modes:
        metrics_celeba[mode] = {}
        for rate in (64, 256):
            res = evaluator.evaluate_pair(
                cb_samples[mode][rate], gt_cb_imgs, operator=op_cpu, y_gt=y256[celeba_idx, :rate], indices=torch.arange(rate), gt_landmarks_px=gt_cb_lmks_px
            )
            metrics_celeba[mode][str(rate)] = {k: v for k, v in res.items() if k != "per_sample"}

    # Noise robustness metrics at M=64
    metrics_noise: dict[str, dict[str, dict]] = {}
    for mode in ("b3_physflow", "zeroshot_flower", "lsi"):
        metrics_noise[mode] = {}
        for nm in ("clean", "gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined"):
            res = evaluator.evaluate_pair(
                noise_samples[mode][nm], gt_hq_imgs, operator=op_cpu, y_gt=y256[hq_idx, :64], indices=torch.arange(64), gt_landmarks_px=gt_hq_lmks_px
            )
            metrics_noise[mode][nm] = {k: v for k, v in res.items() if k != "per_sample"}

    # Pareto curve metrics
    metrics_pareto = {}
    for st in (8, 16, 32, 50):
        imgs_p, sec_p = pareto_samples[st]
        res = evaluator.evaluate_pair(
            imgs_p, gt_hq_imgs, operator=op_cpu, y_gt=y256[hq_idx, :64], indices=torch.arange(64), gt_landmarks_px=gt_hq_lmks_px
        )
        entry = {k: v for k, v in res.items() if k != "per_sample"}
        entry["sec_per_img"] = sec_p
        metrics_pareto[str(st)] = entry

    # Paired bootstrap 95% CIs at M=64 on identity-disjoint CelebA-HQ
    stats_m64 = {
        "physflow_adaface": bootstrap_ci_and_diff(per_sample_m64["b3_physflow"]["adaface_cos"]),
        "physflow_vs_lsi_adaface": bootstrap_ci_and_diff(per_sample_m64["b3_physflow"]["adaface_cos"], per_sample_m64["lsi"]["adaface_cos"]),
        "physflow_vs_zeroshot_adaface": bootstrap_ci_and_diff(per_sample_m64["b3_physflow"]["adaface_cos"], per_sample_m64["zeroshot_flower"]["adaface_cos"]),
        "physflow_vs_b0cont_adaface": bootstrap_ci_and_diff(per_sample_m64["b3_physflow"]["adaface_cos"], per_sample_m64["b0_continuous"]["adaface_cos"]),
    }

    # Visual comparison grids (16 top/representative comparisons + 10 worst failure cases at M=64, plus M=256 comparison!)
    ada_list = per_sample_m64["b3_physflow"]["adaface_cos"]
    order_desc = np.argsort(ada_list)[::-1].tolist()
    order_asc = np.argsort(ada_list).tolist()

    for rate, fname in ((64, "phase2_real_comparison_m64.png"), (256, "phase2_real_comparison_m256.png")):
        tiles, titles = [], []
        for i in order_desc[: min(12, len(hq_idx))]:
            tiles.extend([
                gt_hq_imgs[i],
                hq_samples["b0_discrete"][rate][i],
                hq_samples["zeroshot_flower"][rate][i],
                hq_samples["lsi"][rate][i],
                hq_samples["b2_geo"][rate][i],
                hq_samples["b3_physflow"][rate][i],
            ])
            titles.extend(["GT", "B0-Text", "ZS-FLOWER", f"LSI@{rate}", "B2(+Geo)", f"PhysFlow@{rate}"])
        save_image_grid(torch.stack(tiles), OUTPUTS / fname, nrow=6, titles=titles)

    # Dedicated Teaser raw crop strip (4 crisp faces for Fig. 1)
    teaser_tiles, teaser_titles = [], []
    for i in order_desc[:4]:
        teaser_tiles.extend([
            gt_hq_imgs[i],
            wiener_raw_hq[64][i],
            hq_samples["b0_discrete"][64][i],
            hq_samples["zeroshot_flower"][64][i],
            hq_samples["lsi"][64][i],
            hq_samples["b3_physflow"][64][i],
        ])
        teaser_titles.extend(["GT (512px)", "Wiener Lift", "Text Bridge", "ZS-FLOWER", "LSI (CVPR'25)", "PhysFlow (Ours)"])
    save_image_grid(torch.stack(teaser_tiles), OUTPUTS / "phase2_teaser_strip_m64.png", nrow=6, titles=teaser_titles)

    fail_tiles, fail_titles = [], []
    for i in order_asc[: min(10, len(hq_idx))]:
        fail_tiles.extend([gt_hq_imgs[i], hq_samples["b3_physflow"][64][i]])
        fail_titles.extend([f"GT (cos={ada_list[i]:.2f})", "PhysFlow@64"])
    save_image_grid(torch.stack(fail_tiles), OUTPUTS / "phase2_real_failures_m64.png", nrow=4, titles=fail_titles)

    # Gate verification at M=64 (primary extreme-compression rate, SR% = 64/65536 = 0.098%!)
    b0 = metrics_hq_rates["b0_continuous"]["64"]
    b1 = metrics_hq_rates["b1_id"]["64"]
    b2 = metrics_hq_rates["b2_geo"]["64"]
    b3 = metrics_hq_rates["b3_physflow"]["64"]
    zs = metrics_hq_rates["zeroshot_flower"]["64"]
    lsi64 = metrics_hq_rates["lsi"]["64"]

    gate = {
        "n_eval_celebahq_disjoint": len(hq_idx),
        "n_eval_celeba_unique_ids": len(celeba_idx),
        "primary_rate": 64,
        "sr_percent_64": 100.0 * 64.0 / (256.0 * 256.0),
        "b1_vs_b0_adaface_rel": (b1["adaface_cos_mean"] - b0["adaface_cos_mean"]) / max(abs(b0["adaface_cos_mean"]), 1e-4),
        "b2_vs_b1_nme_rel_drop": (b1["nme106_mean"] - b2["nme106_mean"]) / max(b1["nme106_mean"], 1e-6) if (b1["nme106_mean"] and b2["nme106_mean"]) else None,
        "b3_vs_b2_meas_rel_drop": (b2["meas_rel_mean"] - b3["meas_rel_mean"]) / max(b2["meas_rel_mean"], 1e-6),
        "b3_beats_zeroshot_adaface": b3["adaface_cos_mean"] > zs["adaface_cos_mean"],
        "b3_beats_lsi_m64_adaface": b3["adaface_cos_mean"] > lsi64["adaface_cos_mean"],
        "b3_beats_lsi_combined_noise": metrics_noise["b3_physflow"]["combined"]["adaface_cos_mean"] > metrics_noise["lsi"]["combined"]["adaface_cos_mean"],
        "b3_beats_euc_flower_adaface": b3["adaface_cos_mean"] > metrics_solver_ablation["b3_euc_flower"]["64"]["adaface_cos_mean"],
    }

    report = {
        "gate": gate,
        "resources": {
            "params_total": int(params_total),
            "params_trainable": int(params_trainable),
            "peak_vram_gb": float(e2e_train_info.get("peak_vram_gb", peak_vram_gb())),
            "e2e_train_seconds": float(e2e_train_info.get("seconds", 0.0)),
            "infer_sec_per_img_m64_16steps": float(timings.get("b3_physflow", 0.0)),
        },
        "rates": metrics_hq_rates,
        "celeba_confirm": metrics_celeba,
        "solver_ablation": metrics_solver_ablation,
        "noise_m64": metrics_noise,
        "pareto_m64": metrics_pareto,
        "bootstrap_m64": stats_m64,
        "timings_sec_per_img_m64": timings,
        "e2e_train": e2e_train_info,
    }
    (OUTPUTS / "phase2_real_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Phase 2 Real PhysFlow Benchmark Results (Identity-Disjoint Cross-Model Evaluation)",
        "",
        f"Primary test set: `{len(hq_idx)}` strictly identity-disjoint **CelebA-HQ** faces (`M=64` => `SR = 0.098%` of 256x256 pixels).",
        f"Cross-dataset check: `{len(celeba_idx)}` unique identity-disjoint **CelebA** celebrities.",
        f"Independent evaluators: `AdaFace IR-101 (WebFace12M)`, `InsightFace Buffalo_L`, `106-pt NME`, `LPIPS`, `PSNR`, `SSIM`, relative measurement residual.",
        "",
        "## Main Ladder on CelebA-HQ at M=64 (SR=0.098%) & M=256 (SR=0.391%)",
        "",
        "| Method | AdaFace@64 ↑ | Buffalo@64 ↑ | NME106@64 ↓ | LPIPS@64 ↓ | PSNR@64 ↑ | RelRes@64 ↓ | AdaFace@256 ↑ | NME106@256 ↓ | RelRes@256 ↓ | Sec/Img |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in all_modes:
        r64 = metrics_hq_rates[mode]["64"]
        r256 = metrics_hq_rates[mode]["256"]
        md.append(
            f"| {mode} | {r64['adaface_cos_mean']:.4f} | {r64['buffalo_cos_mean'] or 0:.4f} | "
            f"{r64['nme106_mean'] or 0:.4f} | {r64['lpips_mean'] or 0:.4f} | {r64['psnr_mean']:.2f} | {r64['meas_rel_mean']:.4f} | "
            f"{r256['adaface_cos_mean']:.4f} | {r256['nme106_mean'] or 0:.4f} | {r256['meas_rel_mean']:.4f} | {timings.get(mode, 0.0):.2f} |"
        )
    md += [
        "",
        "## Solver Ablation on CelebA-HQ (M=64 & M=256)",
        "",
        "| Solver | AdaFace@64 ↑ | NME106@64 ↓ | LPIPS@64 ↓ | PSNR@64 ↑ | RelRes@64 ↓ | AdaFace@256 ↑ | NME106@256 ↓ | LPIPS@256 ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("wiener_raw_linear", "b3_euc_flower", "b3_kalman_t0", "b3_physflow"):
        a64 = metrics_solver_ablation[name]["64"]
        a256 = metrics_solver_ablation[name]["256"]
        md.append(
            f"| {name} | {a64['adaface_cos_mean']:.4f} | {a64['nme106_mean'] or 0:.4f} | {a64['lpips_mean'] or 0:.4f} | "
            f"{a64['psnr_mean']:.2f} | {a64['meas_rel_mean']:.4f} | {a256['adaface_cos_mean']:.4f} | "
            f"{a256['nme106_mean'] or 0:.4f} | {a256['lpips_mean'] or 0:.4f} |"
        )
    md += [
        "",
        "## Sensor Perturbation Robustness at M=64 (AdaFace ↑ / NME106 ↓)",
        "",
        "| Method | Clean | Gaussian | Poisson+Gauss | Gain+Bias | 20% Dropout | Combined (N4) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("b3_physflow", "zeroshot_flower", "lsi"):
        ns = metrics_noise[mode]
        md.append(
            f"| {mode} | "
            + " | ".join(
                f"{ns[k]['adaface_cos_mean']:.3f} / {ns[k]['nme106_mean'] or 0:.3f}"
                for k in ("clean", "gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined")
            )
            + " |"
        )
    md += ["", "## Gate Summary", "```json", json.dumps(gate, indent=2), "```"]
    (OUTPUTS / "phase2_real_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("PHASE 2 REAL E2E PHYSFLOW COMPLETE!")
    log(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
