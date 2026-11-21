from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen base linear plus rank-r update. B starts at 0 so the layer is the base."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        if base.bias is not None:
            self.bias = base.bias
        else:
            self.register_parameter("bias", None)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        a = self.lora_A.to(dtype=x.dtype)
        b = self.lora_B.to(dtype=x.dtype)
        update = (x @ a.transpose(0, 1)) @ b.transpose(0, 1)
        return base + update * self.scale


QKVO_NAMES = ("img_qkv", "txt_qkv", "img_attn_proj", "txt_attn_proj")


def attach_qkvo_lora(blocks: nn.ModuleList, rank: int = 16, alpha: float = 16.0) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for block in blocks:
        for name in QKVO_NAMES:
            layer = getattr(block, name)
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"{name} is {type(layer).__name__}, expected nn.Linear")
            wrapped = LoRALinear(layer, rank=rank, alpha=alpha)
            setattr(block, name, wrapped)
            trainable.extend([wrapped.lora_A, wrapped.lora_B])
        trainable_ids = {id(param) for param in trainable}
        for param in block.parameters():
            if id(param) not in trainable_ids:
                param.requires_grad_(False)
    return trainable


class JointConditionTokens(nn.Module):
    """Project C_sem / C_id / C_geo into joint-attention tokens.

    The projection is randomly initialized and multiplied by softplus(gate).
    softplus(-4) is about 0.018, so the tokens start near 0 and the scale
    cannot become negative. A negative scale let the flow loss cancel the
    identity token. The projection is not zero, so the gate still receives a
    gradient at the start.
    """

    def __init__(
        self,
        hidden: int,
        sem_dim: int = 256,
        id_dim: int = 512,
        geo_dim: int = 212,
        n_id: int = 4,
        n_geo: int = 4,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_id = n_id
        self.n_geo = n_geo
        self.sem_proj = nn.Linear(sem_dim, hidden)
        self.id_proj = nn.Linear(id_dim, hidden * n_id)
        self.geo_proj = nn.Linear(geo_dim, hidden * n_geo)
        self.gate = nn.Parameter(torch.tensor([-4.0]))

    def forward(
        self,
        sem: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        geo: torch.Tensor | None = None,
        use_sem: bool = True,
        use_id: bool = True,
        use_geo: bool = True,
    ) -> torch.Tensor | None:
        parts = []
        batch = None
        if use_sem and sem is not None:
            if sem.ndim == 2:
                sem = sem.unsqueeze(1)
            parts.append(self.sem_proj(sem))
            batch = sem.shape[0]
        if use_id and identity is not None:
            parts.append(self.id_proj(identity).view(identity.shape[0], self.n_id, self.hidden))
            batch = identity.shape[0]
        if use_geo and geo is not None:
            parts.append(self.geo_proj(geo).view(geo.shape[0], self.n_geo, self.hidden))
            batch = geo.shape[0]
        if not parts or batch is None:
            return None
        tokens = torch.cat(parts, dim=1)
        return self.token_scale() * tokens

    def token_scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.gate).clamp(max=2.0)


class JointTokenState:
    def __init__(self):
        self.tokens: torch.Tensor | None = None


def wrap_joint_attention(blocks: nn.ModuleList, state: JointTokenState) -> None:
    """Append condition tokens inside every joint-attention block. vec is ignored."""
    for block in blocks:
        original = block.forward

        def forward(x, txt, vec=None, _original=original, _state=state):
            extra = _state.tokens
            if extra is None:
                return _original(x, txt, vec)
            extra = extra.to(device=txt.device, dtype=txt.dtype)
            if extra.shape[0] != txt.shape[0]:
                raise RuntimeError(f"condition batch {extra.shape[0]} != text batch {txt.shape[0]}")
            merged = torch.cat([txt, extra], dim=1)
            x_out, txt_out = _original(x, merged, vec)
            length = txt.shape[1]
            _state.tokens = txt_out[:, length:]
            return x_out, txt_out[:, :length]

        block.forward = forward


def enable_block_checkpoint(blocks: nn.ModuleList) -> None:
    for block in blocks:
        current = block.forward

        def forward(*args, _current=current, **kwargs):
            if not torch.is_grad_enabled():
                return _current(*args, **kwargs)
            return torch.utils.checkpoint.checkpoint(_current, *args, use_reentrant=False, **kwargs)

        block.forward = forward


class FactorizedJointConditioner(nn.Module):
    """RMSNorm-safe native-256 per-block residual joint-attention conditioner.

    Why this fixes the v1-v3 Oracle failure:
    1. Native 256-token text stream: injects 16 ID tokens into unused padding slots
       [224:240] and 16 canonical-geometry tokens into [240:256], keeping MiniT2I's
       pretrained 1D RoPE length at exactly 256 (captions use slots [0:64]).
    2. RMSNorm-safe zero initialization: pretrained mask_token slots have RMS ~23.1
       after txt_preamble_blocks. Adding `ref_scale * tanh(gamma_l) * unit_rms(tokens)`
       onto those non-zero slots is bit-for-bit identity at gamma_l = 0, and smoothly
       perturbs `txt_norm1` (RMSNorm) linearly instead of blowing up tiny tokens.
    3. Per-block injection across all 17 DoubleStreamDiTBlocks prevents depth dilution
       while bounding every gate strictly in [-1, 1] via tanh (zero AdaLN).
    """

    def __init__(
        self,
        hidden: int = 768,
        num_blocks: int = 17,
        id_dim: int = 512,
        geo_dim: int = 212,
        n_id: int = 16,
        n_geo: int = 16,
        ref_scale: float = 20.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.n_id = n_id
        self.n_geo = n_geo
        self.ref_scale = float(ref_scale)
        self.id_net = nn.Sequential(
            nn.Linear(id_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_id * hidden),
        )
        self.id_norm = nn.LayerNorm(hidden)
        self.id_pos = nn.Parameter(torch.randn(1, n_id, hidden) * 0.02)

        self.geo_net = nn.Sequential(
            nn.Linear(geo_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, n_geo * hidden),
        )
        self.geo_norm = nn.LayerNorm(hidden)
        self.geo_pos = nn.Parameter(torch.randn(1, n_geo, hidden) * 0.02)

        # Zero-initialized per-block residual gates: exact pretrained identity at step 0.
        self.id_gates = nn.Parameter(torch.zeros(num_blocks))
        self.geo_gates = nn.Parameter(torch.zeros(num_blocks))

    @staticmethod
    def _unit_rms(tokens: torch.Tensor) -> torch.Tensor:
        return tokens * torch.rsqrt(tokens.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def project_id(self, identity: torch.Tensor | None) -> torch.Tensor | None:
        if identity is None:
            return None
        b = identity.shape[0]
        tok = self.id_net(identity.float()).view(b, self.n_id, self.hidden)
        tok = self.id_norm(tok) + self.id_pos
        return self._unit_rms(tok)

    def project_geo(self, geo: torch.Tensor | None) -> torch.Tensor | None:
        if geo is None:
            return None
        b = geo.shape[0]
        tok = self.geo_net(geo.float()).view(b, self.n_geo, self.hidden)
        tok = self.geo_norm(tok) + self.geo_pos
        return self._unit_rms(tok)

    def id_scales(self) -> torch.Tensor:
        return self.ref_scale * torch.tanh(self.id_gates)

    def geo_scales(self) -> torch.Tensor:
        return self.ref_scale * torch.tanh(self.geo_gates)

    def gate_summary(self) -> dict:
        id_s = torch.tanh(self.id_gates).detach().float().cpu()
        geo_s = torch.tanh(self.geo_gates).detach().float().cpu()
        return {
            "id_tanh_abs_mean": float(id_s.abs().mean()),
            "id_tanh_abs_max": float(id_s.abs().max()),
            "geo_tanh_abs_mean": float(geo_s.abs().mean()),
            "geo_tanh_abs_max": float(geo_s.abs().max()),
        }
