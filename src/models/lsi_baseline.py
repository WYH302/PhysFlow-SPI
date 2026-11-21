"""Official CVPR 2025 Latent Space Imaging (LSI) baseline wrapper (100% pure PyTorch).

Loads official `simulation_256.pt` (and/or `real_128.pt`) `latent_camera` and
`latent_avg` alongside `pretrained_models/ffhq256.pkl` (StyleGAN-XL) without
importing unused StyleGAN2 C++/Ninja ops or `encoder_styleganxl.pt`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.paths import ROOT

LSI_ROOT = ROOT / "third_party" / "latent-imaging"
LSI_SIM_256 = LSI_ROOT / "paper_checkpoints" / "simulation_256.pt"
LSI_REAL_128 = LSI_ROOT / "paper_checkpoints" / "real_128.pt"
LSI_STYLEGAN_XL = LSI_ROOT / "pretrained_models" / "ffhq256.pkl"


def _build_mapping2stylegan(opts, gMLPVision, Attention):
    class StyleBlock(nn.Module):
        def __init__(self, in_c, out_c, spatial):
            super().__init__()
            self.middle_features = opts.b_multiply
            self.residual = opts.b_residual
            num_pools = int(np.log2(spatial))
            self.norm = nn.LayerNorm(in_c)
            self.initial = nn.Sequential(nn.Linear(in_c, self.middle_features), nn.SiLU())
            modules = []
            for _ in range(num_pools - 1):
                if opts.b_attention_features != 0:
                    modules += [
                        nn.LayerNorm(self.middle_features),
                        Attention(self.middle_features, self.middle_features, opts.b_attention_features),
                        nn.SiLU(),
                        nn.Dropout(opts.dropout),
                    ]
                else:
                    modules += [
                        nn.LayerNorm(self.middle_features),
                        nn.Linear(self.middle_features, self.middle_features),
                        nn.SiLU(),
                        nn.Dropout(opts.dropout),
                    ]
            self.convs = nn.Sequential(*modules)
            self.linear = nn.Linear(self.middle_features, out_c)

        def forward(self, inp):
            inp = self.initial(self.norm(inp))
            x = self.convs(inp) + inp if self.residual else self.convs(inp)
            return self.linear(x)

    class Mapping2StyleGAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.coarse_ind = 3
            self.middle_ind = 7
            self.number_layers = opts.styles
            out_dim = opts.encoding_size_out
            in_dim = opts.encoding_size
            self.styles = nn.ModuleList()
            for i in range(self.number_layers):
                sp = opts.b_coarse_n if i < self.coarse_ind else (opts.b_mid_n if i < self.middle_ind else opts.b_fine_n)
                self.styles.append(StyleBlock(in_dim, out_dim, sp))
            self.proj_input_0_glp = gMLPVision(
                image_size=256, patch_size=3, num_classes=1000, dim=out_dim, depth=1, attn_dim=None, prob_survival=1.0, ff_mult=2
            )
            self.proj_input_0_linear = nn.Linear(out_dim * 3, in_dim)
            self.proj_input_1_glp = gMLPVision(
                image_size=256, patch_size=4, num_classes=1000, dim=out_dim, depth=1, attn_dim=None, prob_survival=1.0, ff_mult=2
            )
            self.proj_input_1_linear = nn.Linear(out_dim * 4, in_dim)
            self.out = gMLPVision(
                image_size=256, patch_size=self.number_layers, num_classes=1000, dim=out_dim, depth=opts.b_out_depth, attn_dim=None, prob_survival=1.0, ff_mult=opts.b_out_ff
            )

        def forward(self, x):
            latents = [self.styles[j](x) for j in range(self.coarse_ind)]
            x2 = self.proj_input_0_glp(torch.cat([l.unsqueeze(1) for l in latents], dim=1))
            b, p, c = x2.shape
            x2 = self.proj_input_0_linear(x2.view(b, p * c)) + x
            for j in range(self.coarse_ind, self.middle_ind):
                latents.append(self.styles[j](x2))
            x3 = self.proj_input_1_glp(torch.cat([latents[j].unsqueeze(1) for j in range(self.coarse_ind, self.middle_ind)], dim=1))
            b, p, c = x3.shape
            x3 = self.proj_input_1_linear(x3.view(b, p * c)) + x
            for j in range(self.middle_ind, self.number_layers):
                latents.append(self.styles[j](x3))
            return self.out(torch.stack(latents, dim=1))

    return Mapping2StyleGAN()


class LSIBaseline:
    """Self-contained inference runner for official LSI (Souza et al., CVPR 2025)."""

    def __init__(
        self,
        ckpt_path: Path | str = LSI_SIM_256,
        stylegan_pkl: Path | str = LSI_STYLEGAN_XL,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.ckpt_path = Path(ckpt_path).resolve()
        self.stylegan_pkl = Path(stylegan_pkl).resolve()
        self.encoding_size = 256 if "256" in self.ckpt_path.name else 128
        self.styles = 18
        self._load()

    def _load(self) -> None:
        import importlib.util

        lsi_str = str(LSI_ROOT.resolve())
        xl_str = str((LSI_ROOT / "models" / "styleganxl").resolve())
        saved_mods = {
            k: v
            for k, v in sys.modules.items()
            if k in ("dnnlib", "torch_utils", "legacy")
            or k.startswith(("dnnlib.", "torch_utils."))
        }
        for k in list(saved_mods.keys()):
            sys.modules.pop(k, None)

        cwd = os.getcwd()
        if xl_str not in sys.path:
            sys.path.insert(0, xl_str)
        if lsi_str not in sys.path:
            sys.path.insert(0, lsi_str)
        os.chdir(lsi_str)
        try:
            spec = importlib.util.spec_from_file_location("lsi_g_mlp", LSI_ROOT / "models" / "g_mlp.py")
            gmlp_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gmlp_mod)

            import timm.layers
            import timm.models._efficientnet_blocks as _eff_blocks
            import pkgutil
            sys.modules.setdefault("timm.models.efficientnet_blocks", _eff_blocks)
            for _imp, mod_name, _ispkg in pkgutil.iter_modules(timm.layers.__path__):
                full_new = f"timm.layers.{mod_name}"
                full_old = f"timm.models.layers.{mod_name}"
                if full_old not in sys.modules:
                    try:
                        sys.modules[full_old] = __import__(full_new, fromlist=["*"])
                    except Exception:
                        pass

            import models.styleganxl.dnnlib as dnnlib
            import models.styleganxl.legacy as legacy
            import models.styleganxl.torch_utils.ops.bias_act as _sg_bias_act
            import models.styleganxl.torch_utils.ops.filtered_lrelu as _sg_filtered_lrelu
            import models.styleganxl.torch_utils.ops.upfirdn2d as _sg_upfirdn2d

            # Force pure-PyTorch reference ops on CUDA so Windows never requires MSVC/NVCC JIT compilation.
            _sg_bias_act._init = lambda: False
            _sg_filtered_lrelu._init = lambda: False
            _sg_upfirdn2d._init = lambda: False
            if "torch_utils.ops.bias_act" in sys.modules:
                sys.modules["torch_utils.ops.bias_act"]._init = lambda: False
            if "torch_utils.ops.filtered_lrelu" in sys.modules:
                sys.modules["torch_utils.ops.filtered_lrelu"]._init = lambda: False
            if "torch_utils.ops.upfirdn2d" in sys.modules:
                sys.modules["torch_utils.ops.upfirdn2d"]._init = lambda: False

            ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
            opts = SimpleNamespace(
                styles=self.styles,
                encoding_size=self.encoding_size,
                encoding_size_out=512,
                b_coarse_n=16,
                b_mid_n=32,
                b_fine_n=64,
                b_multiply=512,
                b_attention_features=128,
                b_residual=True,
                b_out_ff=2,
                b_out_depth=1,
                dropout=0.0,
            )
            self.latent_camera = _build_mapping2stylegan(opts, gmlp_mod.gMLPVision, gmlp_mod.Attention)
            self.latent_camera.load_state_dict(ckpt["latent_camera"], strict=True)
            self.latent_camera.to(self.device).eval()

            self.latent_avg = ckpt["latent_avg"].float().to(self.device)
            with dnnlib.util.open_url(str(self.stylegan_pkl)) as f:
                self.G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).to(self.device)

            self.cls_head = None
            if ckpt.get("cls") is not None:
                self.cls_head = nn.Linear(512 * self.styles, 40)
                self.cls_head.load_state_dict(ckpt["cls"], strict=True)
                self.cls_head.to(self.device).eval()
        finally:
            os.chdir(cwd)
            if xl_str in sys.path:
                sys.path.remove(xl_str)
            # Keep torch_utils/dnnlib loaded because G_ema forward references them!

    @torch.no_grad()
    def encode_latent(
        self,
        y: torch.Tensor,
        indices: torch.Tensor | None = None,
        y_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = y.shape[0]
        if indices is None and y.shape[1] == self.encoding_size:
            y_full = y.float().to(self.device)
        else:
            if y_mean is not None:
                y_full = y_mean.float().to(self.device).view(1, self.encoding_size).expand(b, -1).clone()
            else:
                y_full = torch.zeros((b, self.encoding_size), device=self.device, dtype=torch.float32)
            idx = (
                indices.to(device=self.device, dtype=torch.long)
                if indices is not None
                else torch.arange(y.shape[1], device=self.device, dtype=torch.long)
            )
            y_full[:, idx] = y.float().to(self.device)
        l_cam = self.latent_camera(y_full)
        avg = self.latent_avg.view(1, 1, -1).expand(b, self.styles, -1)
        return avg + l_cam

    @torch.no_grad()
    def reconstruct(
        self,
        y: torch.Tensor,
        indices: torch.Tensor | None = None,
        y_mean: torch.Tensor | None = None,
        output_size: int = 512,
        batch_size: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        imgs_out = []
        lats_out = []
        force_fp32 = self.device.type == "cpu"
        for s in range(0, y.shape[0], batch_size):
            yb = y[s : s + batch_size]
            w = self.encode_latent(yb, indices=indices, y_mean=y_mean)
            img = self.G.synthesis(w, noise_mode="const", force_fp32=force_fp32)
            img = img.float().clamp(-1.0, 1.0)
            if img.shape[-1] != output_size or img.shape[-2] != output_size:
                img = F.interpolate(img, size=(output_size, output_size), mode="bilinear", align_corners=False)
            imgs_out.append(img.cpu())
            lats_out.append(w.cpu())
        return torch.cat(imgs_out, dim=0), torch.cat(lats_out, dim=0)
