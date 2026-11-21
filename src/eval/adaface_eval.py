"""Cross-model evaluation suite: AdaFace IR-101, InsightFace Buffalo_L, LPIPS, PSNR, SSIM, NME.

Training uses InsightFace AntelopeV2 (Glint360K R100).
Evaluation reports AdaFace IR-101 (WebFace12M) and Buffalo_L independently so no
reported identity score comes exclusively from the training loss network.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.align import align_to_lsi, detect_face_robust
from src.eval.metrics import cosine_identity, nme, psnr, ssim, to_unit
from src.models.glintr100 import FixedArcfaceCrop
from src.paths import ROOT

ADAFACE_DIR = ROOT / "checkpoints" / "adaface_ir101_webface12m"
INSIGHTFACE_DIR = ROOT / "checkpoints" / "insightface"


class AdaFaceIR101(torch.nn.Module):
    """Local AdaFace IR-101 WebFace12M evaluator on 112x112 RGB [-1, 1] crops."""

    def __init__(self, checkpoint_dir: Path | str = ADAFACE_DIR):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(f"AdaFace directory missing: {self.checkpoint_dir}")
        self.cropper = FixedArcfaceCrop()
        self.backbone = self._load_backbone(self.checkpoint_dir)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _load_backbone(path: Path):
        import importlib
        import yaml
        from omegaconf import OmegaConf

        cwd = os.getcwd()
        path_str = str(path)
        saved_modules = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
        for k in list(saved_modules.keys()):
            sys.modules.pop(k, None)
        sys.path.insert(0, path_str)
        os.chdir(path_str)
        try:
            mod = importlib.import_module("models")
            conf = OmegaConf.create(yaml.safe_load((path / "pretrained_model" / "model.yaml").read_text()))
            net = mod.get_model(conf)
            net.load_state_dict_from_path(str(path / "pretrained_model" / "model.pt"))
            return net
        finally:
            os.chdir(cwd)
            if path_str in sys.path:
                sys.path.remove(path_str)
            for k in [k for k in list(sys.modules.keys()) if k == "models" or k.startswith("models.")]:
                sys.modules.pop(k, None)
            sys.modules.update(saved_modules)

    @torch.no_grad()
    def embed_aligned(self, images_m1_1: torch.Tensor) -> torch.Tensor:
        """Embed aligned [-1, 1] RGB images (any square size) -> L2-normalized [B, 512]."""
        dev = next(self.backbone.parameters()).device
        x = images_m1_1.to(device=dev, dtype=torch.float32)
        if x.min() >= -1e-4 and x.max() <= 1.0 + 1e-4:
            x = x * 2.0 - 1.0
        if x.shape[-1] != 512 or x.shape[-2] != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        crops = self.cropper(x).clamp(-1.0, 1.0)
        out = self.backbone(crops)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out.float(), dim=-1)


class CrossModelEvaluator:
    """Unified publication-grade evaluator (AdaFace IR-101 + Buffalo_L + 106-pt NME + LPIPS)."""

    def __init__(self, device: torch.device | str = "cpu", use_lpips: bool = True, use_buffalo: bool = True):
        self.device = torch.device(device)
        self.adaface = AdaFaceIR101().to(self.device).eval()
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()
                for p in self.lpips_fn.parameters():
                    p.requires_grad_(False)
            except Exception:
                self.lpips_fn = None
        self.buffalo = None
        if use_buffalo:
            self.buffalo = self._load_buffalo()

    def _load_buffalo(self):
        os.environ["INSIGHTFACE_HOME"] = str(INSIGHTFACE_DIR.resolve())
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx = 0 if self.device.type == "cuda" else -1
        for prov, cid in [(providers, ctx), (["CPUExecutionProvider"], -1)]:
            try:
                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=prov,
                    root=str(INSIGHTFACE_DIR.resolve()),
                    allowed_modules=["detection", "landmark_2d_106", "recognition"],
                )
                app.prepare(ctx_id=cid, det_size=(640, 640))
                return app
            except Exception:
                continue
        return None

    def _to_rgb_u8(self, tensor_chw: torch.Tensor) -> np.ndarray:
        img = to_unit(tensor_chw.detach().float().cpu()).clamp(0.0, 1.0)
        return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    @torch.no_grad()
    def evaluate_pair(
        self,
        pred_m1_1: torch.Tensor,
        gt_m1_1: torch.Tensor,
        operator=None,
        y_gt: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        gt_landmarks_px: torch.Tensor | None = None,
        batch_size: int = 16,
    ) -> dict:
        """Compute full per-sample and mean metrics on [N, 3, H, W] tensors in [-1, 1]."""
        n = int(pred_m1_1.shape[0])
        h = int(pred_m1_1.shape[-1])
        pred_f = pred_m1_1.detach().float().cpu()
        gt_f = gt_m1_1.detach().float().cpu()

        psnr_vals = psnr(pred_f, gt_f).tolist()
        ssim_vals = ssim(pred_f, gt_f).tolist()

        # AdaFace IR-101 (cross-model #1)
        ada_cos = []
        lpips_vals = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            p_b = pred_f[start:end].to(self.device)
            g_b = gt_f[start:end].to(self.device)
            ep = self.adaface.embed_aligned(p_b)
            eg = self.adaface.embed_aligned(g_b)
            ada_cos.extend(cosine_identity(ep, eg).cpu().tolist())
            if self.lpips_fn is not None:
                # LPIPS expects [-1, 1]
                lp = self.lpips_fn(p_b.clamp(-1, 1), g_b.clamp(-1, 1)).view(-1).cpu().tolist()
                lpips_vals.extend(lp)

        # Buffalo_L identity (cross-model #2) & 106-pt NME
        buf_cos: list[float | None] = []
        nme106_vals: list[float | None] = []
        det_pred = 0
        if self.buffalo is not None:
            for i in range(n):
                p_det = detect_face_robust(self.buffalo, self._to_rgb_u8(pred_f[i]), prefer_center=True)
                g_det = detect_face_robust(self.buffalo, self._to_rgb_u8(gt_f[i]), prefer_center=True)
                if p_det is not None:
                    det_pred += 1
                if p_det is not None and g_det is not None and p_det["embedding"] is not None and g_det["embedding"] is not None:
                    buf_cos.append(float(np.dot(p_det["embedding"], g_det["embedding"])))
                else:
                    buf_cos.append(None)
                if p_det is not None and p_det["lmk106"] is not None:
                    ref_lmk = None
                    if gt_landmarks_px is not None:
                        ref_lmk = gt_landmarks_px[i].detach().float().cpu()
                    elif g_det is not None and g_det["lmk106"] is not None:
                        ref_lmk = torch.from_numpy(g_det["lmk106"]).float()
                    if ref_lmk is not None and ref_lmk.shape[0] == 106:
                        pred_lmk = torch.from_numpy(p_det["lmk106"]).float()
                        # InsightFace 2d106det eye clusters: left eye [33:43], right eye [87:97]
                        left_eye = ref_lmk[33:43].mean(dim=0)
                        right_eye = ref_lmk[87:97].mean(dim=0)
                        iod = float(torch.linalg.vector_norm(left_eye - right_eye).clamp_min(1e-6))
                        err = float(torch.linalg.vector_norm(pred_lmk - ref_lmk, dim=-1).mean())
                        nme106_vals.append(err / iod)
                    else:
                        nme106_vals.append(None)
                else:
                    nme106_vals.append(None)

        # Physical measurement residual
        meas_rel = None
        meas_rmse = None
        if operator is not None and y_gt is not None:
            op_dev = operator.phi.device
            rel_list = []
            rmse_list = []
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                p_b = pred_f[start:end].to(op_dev)
                y_b = y_gt[start:end].to(op_dev)
                idx_b = indices.to(op_dev) if indices is not None else None
                rel_list.extend(operator.relative_residual(p_b, y_b, indices=idx_b, input_range="-1_1").cpu().tolist())
                rmse_list.extend(operator.residual(p_b, y_b, indices=idx_b, input_range="-1_1").cpu().tolist())
            meas_rel = rel_list
            meas_rmse = rmse_list

        valid_buf = [v for v in buf_cos if v is not None]
        valid_nme = [v for v in nme106_vals if v is not None]

        return {
            "n": n,
            "image_size": h,
            "adaface_cos_mean": float(np.mean(ada_cos)) if ada_cos else None,
            "adaface_cos_std": float(np.std(ada_cos)) if ada_cos else None,
            "buffalo_cos_mean": float(np.mean(valid_buf)) if valid_buf else None,
            "buffalo_cos_std": float(np.std(valid_buf)) if valid_buf else None,
            "buffalo_detected": det_pred,
            "nme106_mean": float(np.mean(valid_nme)) if valid_nme else None,
            "nme106_std": float(np.std(valid_nme)) if valid_nme else None,
            "nme106_n": len(valid_nme),
            "psnr_mean": float(np.mean(psnr_vals)),
            "ssim_mean": float(np.mean(ssim_vals)),
            "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
            "meas_rel_mean": float(np.mean(meas_rel)) if meas_rel else None,
            "meas_rmse_mean": float(np.mean(meas_rmse)) if meas_rmse else None,
            "per_sample": {
                "adaface_cos": ada_cos,
                "buffalo_cos": buf_cos,
                "nme106": nme106_vals,
                "psnr": psnr_vals,
                "ssim": ssim_vals,
                "lpips": lpips_vals,
                "meas_rel": meas_rel,
            },
        }
