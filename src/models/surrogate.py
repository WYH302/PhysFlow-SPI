from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.injection import JointConditionTokens, JointTokenState, attach_qkvo_lora, wrap_joint_attention


class TinyJointBlock(nn.Module):
    """Small joint-attention block with the same Q/K/V/O names as MiniT2I.

    `vec` is accepted and unused. There is no AdaLN.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.txt_qkv = nn.Linear(dim, dim * 3)
        self.img_attn_proj = nn.Linear(dim, dim)
        self.txt_attn_proj = nn.Linear(dim, dim)
        self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def _attend(self, qkv_img, qkv_txt):
        b, li, _ = qkv_img.shape
        lt = qkv_txt.shape[1]
        h, d = self.heads, self.head_dim

        def split(qkv, length):
            q, k, v = qkv.view(b, length, 3, h, d).unbind(2)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        qi, ki, vi = split(qkv_img, li)
        qt, kt, vt = split(qkv_txt, lt)
        q = torch.cat([qt, qi], dim=2)
        k = torch.cat([kt, ki], dim=2)
        v = torch.cat([vt, vi], dim=2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * (d**-0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, lt + li, h * d)
        return out[:, :lt], out[:, lt:]

    def forward(self, x, txt, vec=None):
        del vec
        txt_attn, img_attn = self._attend(self.img_qkv(x), self.txt_qkv(txt))
        x = x + self.img_attn_proj(img_attn)
        txt = txt + self.txt_attn_proj(txt_attn)
        x = x + self.img_mlp(x)
        txt = txt + self.txt_mlp(txt)
        return x, txt


class TinyXPred(nn.Module):
    """x-prediction network with the official call signature (x_t, t, context, mask).

    t is not injected with AdaLN. The official MM-JiT computes a timestep vector
    and does not apply it inside the blocks; this surrogate matches that fact.
    """

    def __init__(self, image_size: int = 64, patch: int = 8, dim: int = 64, depth: int = 2, text_dim: int = 32):
        super().__init__()
        if image_size % patch != 0:
            raise ValueError("image_size must be divisible by patch")
        self.image_size = image_size
        self.patch = patch
        self.dim = dim
        self.grid = image_size // patch
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.txt_embed = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TinyJointBlock(dim) for _ in range(depth)])
        self.final = nn.Linear(dim, patch * patch * 3)
        self.condition = JointConditionTokens(hidden=dim, sem_dim=256, id_dim=512, geo_dim=212)
        self.state = JointTokenState()
        self.lora_params = attach_qkvo_lora(self.blocks, rank=16, alpha=16.0)
        for block in self.blocks:
            for name, param in block.named_parameters():
                if "mlp" in name:
                    param.requires_grad_(True)
        wrap_joint_attention(self.blocks, self.state)

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def unpatch(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        p = self.patch
        g = int(n**0.5)
        x = tokens.view(b, g, g, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, 3, g * p, g * p)

    def forward(self, x_t, t, context, attn_mask, sem=None, identity=None, geo=None, use_sem=True, use_id=True, use_geo=True):
        del t
        mask = attn_mask.to(dtype=torch.bool)
        context = context.masked_fill(~mask.unsqueeze(-1), 0.0)
        img = self.patch_embed(x_t).flatten(2).transpose(1, 2)
        txt = self.txt_embed(context)
        self.state.tokens = self.condition(sem, identity, geo, use_sem=use_sem, use_id=use_id, use_geo=use_geo)
        for block in self.blocks:
            img, txt = block(img, txt, None)
        self.state.tokens = None
        return self.unpatch(self.final(img))


def official_v_loss(x0: torch.Tensor, x_t: torch.Tensor, images: torch.Tensor, t: torch.Tensor, clamp: float = 0.05) -> torch.Tensor:
    """v-loss from mini_t2i/diffusion.py. The network itself returns x0."""
    scale = (1.0 - t)[:, None, None, None].clamp_min(clamp)
    target = (images - x_t) / scale
    pred = (x0 - x_t) / scale
    return F.mse_loss(pred, target)


def flow_sample_xt(images: torch.Tensor, t: torch.Tensor, noise_scale: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(images) * noise_scale
    x_t = images * t[:, None, None, None] + noise * (1.0 - t[:, None, None, None])
    return x_t, noise
