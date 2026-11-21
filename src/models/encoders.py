from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FactorHeads(nn.Module):
    def __init__(self, dim: int, sem_tokens: int = 8, sem_dim: int = 256, id_dim: int = 512, geo_dim: int = 212):
        super().__init__()
        self.sem_tokens = sem_tokens
        self.sem_dim = sem_dim
        self.id_dim = id_dim
        self.geo_dim = geo_dim
        self.norm = nn.LayerNorm(dim)
        self.sem = nn.Linear(dim, sem_tokens * sem_dim)
        self.ident = nn.Linear(dim, id_dim)
        self.geo = nn.Linear(dim, geo_dim)

    def forward(self, pooled: torch.Tensor):
        h = self.norm(pooled)
        sem = self.sem(h).view(h.shape[0], self.sem_tokens, self.sem_dim)
        return sem, self.ident(h), self.geo(h)


class MeanEncoder(nn.Module):
    """Predicts the training-set mean and ignores y. Buffers, not a learned bias."""

    def __init__(self, sem_tokens: int = 8, sem_dim: int = 256, id_dim: int = 512, geo_dim: int = 212):
        super().__init__()
        self.register_buffer("sem_mean", torch.zeros(sem_tokens, sem_dim))
        self.register_buffer("id_mean", torch.zeros(id_dim))
        self.register_buffer("geo_mean", torch.zeros(geo_dim))
        self._fitted = False

    def set_means(self, sem: torch.Tensor, identity: torch.Tensor, geo: torch.Tensor) -> None:
        self.sem_mean.copy_(sem.detach().mean(dim=0))
        self.id_mean.copy_(identity.detach().mean(dim=0))
        self.geo_mean.copy_(geo.detach().mean(dim=0))
        self._fitted = True

    def forward(self, y: torch.Tensor, indices: torch.Tensor | None = None):
        del indices
        b = y.shape[0]
        return self.sem_mean.expand(b, -1, -1), self.id_mean.expand(b, -1), self.geo_mean.expand(b, -1)


class SensorTokenizer(nn.Module):
    def __init__(self, max_measurements: int, dim: int):
        super().__init__()
        self.val = nn.Linear(1, dim)
        self.idx = nn.Embedding(max_measurements, dim)

    def forward(self, y: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.val(y.unsqueeze(-1)) + self.idx(indices).unsqueeze(0)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        max_measurements: int = 256,
        hidden: int = 256,
        depth: int = 3,
        sem_tokens: int = 8,
        sem_dim: int = 256,
        id_dim: int = 512,
        geo_dim: int = 212,
    ):
        super().__init__()
        self.max_measurements = max_measurements
        self.register_buffer("y_mean", torch.zeros(max_measurements))
        self.register_buffer("y_std", torch.ones(max_measurements))
        in_dim = max_measurements * 2
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            dim = hidden
        self.net = nn.Sequential(*layers)
        self.heads = FactorHeads(hidden, sem_tokens, sem_dim, id_dim, geo_dim)

    def set_y_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.y_mean.copy_(mean.float().reshape(-1))
        self.y_std.copy_(std.float().reshape(-1).clamp_min(1e-4))

    def forward(self, y: torch.Tensor, indices: torch.Tensor):
        b, m = y.shape
        idx_1d = indices.to(device=y.device, dtype=torch.long)
        y_z = (y.float() - self.y_mean.index_select(0, idx_1d)) / self.y_std.index_select(0, idx_1d)
        packed = y_z.new_zeros(b, self.max_measurements)
        mask = y_z.new_zeros(b, self.max_measurements)
        idx = idx_1d.view(1, -1).expand(b, -1)
        packed.scatter_(1, idx, y_z)
        mask.scatter_(1, idx, 1.0)
        h = self.net(torch.cat([packed, mask], dim=-1))
        return self.heads(h)


class Transformer1DEncoder(nn.Module):
    def __init__(
        self,
        max_measurements: int = 256,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        sem_tokens: int = 8,
        sem_dim: int = 256,
        id_dim: int = 512,
        geo_dim: int = 212,
    ):
        super().__init__()
        self.register_buffer("y_mean", torch.zeros(max_measurements))
        self.register_buffer("y_std", torch.ones(max_measurements))
        self.tokens = SensorTokenizer(max_measurements, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.heads = FactorHeads(dim, sem_tokens, sem_dim, id_dim, geo_dim)

    def set_y_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.y_mean.copy_(mean.float().reshape(-1))
        self.y_std.copy_(std.float().reshape(-1).clamp_min(1e-4))

    def forward(self, y: torch.Tensor, indices: torch.Tensor):
        idx_1d = indices.to(device=y.device, dtype=torch.long)
        y_z = (y.float() - self.y_mean.index_select(0, idx_1d)) / self.y_std.index_select(0, idx_1d)
        tok = self.tokens(y_z, idx_1d)
        h = self.encoder(tok).mean(dim=1)
        return self.heads(h)


class TinyVisionBackbone(nn.Module):
    """Stand-in when DINOv2 weights are not on disk. Output tokens are 384-d."""

    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pretrained = False
        self.name = "tiny_conv_backbone"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return feat.flatten(2).transpose(1, 2)


def try_load_dinov2(device: torch.device | None = None):
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    except Exception as exc:
        return TinyVisionBackbone(), f"dinov2 unavailable: {exc}"
    model.eval()
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) >= 2:
        for param in model.parameters():
            param.requires_grad = False
        for block in list(blocks)[-2:]:
            for param in block.parameters():
                param.requires_grad = True
    model.pretrained = True
    model.name = "dinov2_vits14"
    if device is not None:
        model.to(device)
    return model, "loaded dinov2_vits14, last 2 blocks trainable"


class CrossFuse(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, nq, d = query.shape
        h = self.heads
        dh = d // h
        q = self.q(query).view(b, nq, h, dh).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        nk = context.shape[1]
        k = k.view(b, nk, h, dh).transpose(1, 2)
        v = v.view(b, nk, h, dh).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, nq, d)
        return query + self.proj(out)


class DualDomainEncoder(nn.Module):
    """Tikhonov backprojection through a vision backbone, fused with 1D sensor tokens."""

    def __init__(
        self,
        operator,
        backbone: nn.Module | None = None,
        max_measurements: int = 256,
        dim: int = 128,
        vision_dim: int = 384,
        sem_tokens: int = 8,
        sem_dim: int = 256,
        id_dim: int = 512,
        geo_dim: int = 212,
        lam: float = 1e-2,
    ):
        super().__init__()
        self.operator = operator
        self.lam = lam
        self.backbone = backbone if backbone is not None else TinyVisionBackbone(vision_dim)
        self.vision_proj = nn.Linear(vision_dim, dim)
        self.tokens = SensorTokenizer(max_measurements, dim)
        self.fuse = CrossFuse(dim)
        self.heads = FactorHeads(dim, sem_tokens, sem_dim, id_dim, geo_dim)

    def backproject(self, y: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.operator.backproject(y, indices=indices, lam=self.lam)

    def forward(self, y: torch.Tensor, indices: torch.Tensor):
        bp = self.backproject(y, indices)
        rgb = bp.repeat(1, 3, 1, 1)
        scale = rgb.detach().abs().mean().clamp_min(1e-4)
        rgb = (rgb / scale).clamp(-4, 4)
        view = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        visual = self.backbone(view)
        if visual.ndim == 2:
            visual = visual.unsqueeze(1)
        visual = self.vision_proj(visual)
        sensor = self.tokens(y, indices)
        fused = self.fuse(visual, sensor).mean(dim=1)
        return self.heads(fused)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_mlp_matching(reference: nn.Module, **kwargs) -> MLPEncoder:
    target = parameter_count(reference)
    best = None
    best_gap = None
    for hidden in (64, 96, 128, 160, 192, 256, 320, 384, 512, 768):
        for depth in (2, 3, 4):
            candidate = MLPEncoder(hidden=hidden, depth=depth, **kwargs)
            gap = abs(parameter_count(candidate) - target)
            if best_gap is None or gap < best_gap:
                best = candidate
                best_gap = gap
    assert best is not None
    return best


def condition_loss(pred, batch) -> torch.Tensor:
    sem, identity, geo = pred
    return (
        F.mse_loss(sem, batch["sem"])
        + F.mse_loss(identity, batch["identity"])
        + F.mse_loss(geo, batch["geo"])
    )


class LocalDINOv2Backbone(nn.Module):
    """Local HuggingFace DINOv2-Small emitting all 257 spatial tokens [B, 257, 384]."""

    def __init__(self, checkpoint_dir: str | None = None, unfreeze_last_n: int = 6):
        super().__init__()
        from pathlib import Path
        from transformers import AutoModel

        if checkpoint_dir is None:
            checkpoint_dir = str(Path(__file__).resolve().parents[2] / "checkpoints" / "dinov2-small")
        self.model = AutoModel.from_pretrained(checkpoint_dir, local_files_only=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if unfreeze_last_n > 0:
            layers = self.model.encoder.layer
            for layer in layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for p in self.model.layernorm.parameters():
                p.requires_grad_(True)
        self.out_dim = int(self.model.config.hidden_size)
        self.pretrained = True
        self.name = "local_dinov2_small"
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_01: torch.Tensor) -> torch.Tensor:
        """Expect clean [0, 1] RGB images (e.g., Wiener lifted faces) -> [B, 257, 384]."""
        pixel_values = (x_01.clamp(0.0, 1.0) - self.im_mean.to(x_01.dtype)) / self.im_std.to(x_01.dtype)
        return self.model(pixel_values=pixel_values).last_hidden_state


class CompactFactorHeads(nn.Module):
    """Residual-over-mean factorized heads with low-rank semantic bottleneck (~4M params, no 33M swamp)."""

    def __init__(self, in_dim: int, sem_tokens: int = 64, sem_dim: int = 1024, id_dim: int = 512, geo_dim: int = 212):
        super().__init__()
        self.sem_tokens = int(sem_tokens)
        self.sem_dim = int(sem_dim)
        self.register_buffer("geo_mean", torch.zeros(1, geo_dim))
        self.register_buffer("sem_mean", torch.zeros(1, sem_tokens, sem_dim))

        self.id_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, id_dim),
        )
        self.geo_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, geo_dim),
        )
        # Low-rank semantic factorization: in_dim -> 512 -> (sem_tokens, 64) -> (sem_tokens, sem_dim)
        self.sem_mid = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, sem_tokens * 64),
        )
        self.sem_up = nn.Linear(64, sem_dim)

    def set_target_means(self, sem_mean: torch.Tensor, geo_mean: torch.Tensor) -> None:
        self.sem_mean.copy_(sem_mean.float().view(1, self.sem_tokens, self.sem_dim))
        self.geo_mean.copy_(geo_mean.float().view(1, -1))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = feat.shape[0]
        c_id = F.normalize(self.id_head(feat).float(), dim=-1)
        c_geo = self.geo_mean.to(feat.device) + self.geo_head(feat).float()
        s_low = self.sem_mid(feat).view(b, self.sem_tokens, 64)
        c_sem = self.sem_mean.to(feat.device) + self.sem_up(s_low).float()
        return c_sem, c_id, c_geo


class WhitenedMLPEncoder(nn.Module):
    """Strong MLP baseline on subset-adaptive ZCA-whitened measurements (or raw Z-scored if whiten=False)."""

    def __init__(
        self,
        lifter,
        max_measurements: int = 256,
        hidden: int = 512,
        depth: int = 3,
        sem_tokens: int = 64,
        sem_dim: int = 1024,
        id_dim: int = 512,
        geo_dim: int = 212,
        use_whiten: bool = True,
    ):
        super().__init__()
        self.lifter = lifter
        self.max_measurements = int(max_measurements)
        self.use_whiten = bool(use_whiten)
        layers: list[nn.Module] = []
        d = max_measurements * 2
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)]
            d = hidden
        self.net = nn.Sequential(*layers)
        self.heads = CompactFactorHeads(hidden, sem_tokens, sem_dim, id_dim, geo_dim)

    def set_target_means(self, sem_mean: torch.Tensor, geo_mean: torch.Tensor) -> None:
        self.heads.set_target_means(sem_mean, geo_mean)

    def forward(self, y: torch.Tensor, indices: torch.Tensor):
        b = y.shape[0]
        idx = indices.to(device=y.device, dtype=torch.long)
        if self.use_whiten:
            u = self.lifter.whiten_1d(y, idx, reg=1e-3)
        else:
            u = (y.float() - self.lifter.y_bar.index_select(1, idx)) / self.lifter.y_sd.index_select(1, idx)
        packed = u.new_zeros(b, self.max_measurements)
        mask = u.new_zeros(b, self.max_measurements)
        idx_b = idx.view(1, -1).expand(b, -1)
        packed.scatter_(1, idx_b, u)
        mask.scatter_(1, idx_b, 1.0)
        h = self.net(torch.cat([packed, mask], dim=-1))
        return self.heads(h)


class WhitenedTransformer1DEncoder(nn.Module):
    """Strong 1D Sensor Transformer on ZCA-whitened measurements with attention pooling."""

    def __init__(
        self,
        lifter,
        max_measurements: int = 256,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        sem_tokens: int = 64,
        sem_dim: int = 1024,
        id_dim: int = 512,
        geo_dim: int = 212,
    ):
        super().__init__()
        self.lifter = lifter
        self.val_proj = nn.Linear(1, dim)
        self.idx_emb = nn.Embedding(max_measurements, dim)
        self.rate_emb = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pool_q = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.heads = CompactFactorHeads(dim * 4, sem_tokens, sem_dim, id_dim, geo_dim)

    def set_target_means(self, sem_mean: torch.Tensor, geo_mean: torch.Tensor) -> None:
        self.heads.set_target_means(sem_mean, geo_mean)

    def forward(self, y: torch.Tensor, indices: torch.Tensor):
        b, m = y.shape
        idx = indices.to(device=y.device, dtype=torch.long)
        u = self.lifter.whiten_1d(y, idx, reg=1e-3)
        rate_f = self.rate_emb(torch.full((b, 1, 1), math.log2(max(m, 1)) / 8.0, device=y.device))
        tok = self.val_proj(u.unsqueeze(-1)) + self.idx_emb(idx).unsqueeze(0) + rate_f
        h = self.encoder(tok)
        pooled, _ = self.pool_attn(self.pool_q.expand(b, -1, -1), h, h, need_weights=False)
        return self.heads(pooled.reshape(b, -1))


class PhysicsDualDomainFactorizer(nn.Module):
    """PhysFlow Hero Encoder: Closed-Form Wiener Physics Lifting -> Local DINOv2-Small
    (257 spatial tokens) fused with Subset-Adaptive ZCA-Whitened 1D Sensor Stream!

    Zero extra parameters for lifting/whitening; works seamlessly for arbitrary subsets S
    (M in {16, 32, 64, 128, 256} and arbitrary channel dropout).
    """

    def __init__(
        self,
        operator,
        lifter=None,
        backbone: nn.Module | None = None,
        max_measurements: int = 256,
        dim: int = 384,
        sem_tokens: int = 64,
        sem_dim: int = 1024,
        id_dim: int = 512,
        geo_dim: int = 212,
        lam: float = 2e-4,
        unfreeze_last_n: int = 6,
    ):
        super().__init__()
        from src.optical.spi_operator import WienerPhysicsLifter

        self.operator = operator
        self.lifter = lifter if lifter is not None else WienerPhysicsLifter(max_measurements=max_measurements, img_size=128)
        self.lam = float(lam)
        self.max_measurements = int(max_measurements)
        self.sem_tokens = int(sem_tokens)
        self.sem_dim = int(sem_dim)
        self.id_dim = int(id_dim)
        self.geo_dim = int(geo_dim)

        self.backbone = backbone if backbone is not None else LocalDINOv2Backbone(unfreeze_last_n=unfreeze_last_n)
        vision_dim = int(getattr(self.backbone, "out_dim", 384))

        # 2D spatial pyramid pooling on DINOv2 16x16 patch tokens (CLS + 2x2 quadrants -> 5*384 -> dim)
        self.vis_compress = nn.Sequential(
            nn.LayerNorm(vision_dim * 5),
            nn.Linear(vision_dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
        # 1D whitened sensor branch (scattered whitened vector + mask -> direct residual MLP)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(max_measurements * 2 + 1, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        # Cross-domain gated fusion between 2D Wiener-DINOv2 and 1D Whitened Sensor features
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Separate specialized projections for ID (prefers global+face-center DINOv2 + sensor),
        # Geo (prefers 4x4 spatial DINOv2 patches + sensor), and Sem (prefers CLS + sensor)
        self.geo_spatial = nn.Sequential(
            nn.LayerNorm(vision_dim * 16),
            nn.Linear(vision_dim * 16, dim),
            nn.GELU(),
        )
        self.register_buffer("geo_mean", torch.zeros(1, geo_dim))
        self.register_buffer("sem_mean", torch.zeros(1, sem_tokens, sem_dim))

        self.id_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, id_dim),
        )
        self.geo_head = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, 512),
            nn.GELU(),
            nn.Linear(512, geo_dim),
        )
        self.sem_mid = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, sem_tokens * 64),
        )
        self.sem_up = nn.Linear(64, sem_dim)

    def set_target_means(self, sem_mean: torch.Tensor, geo_mean: torch.Tensor) -> None:
        self.sem_mean.copy_(sem_mean.float().view(1, self.sem_tokens, self.sem_dim))
        self.geo_mean.copy_(geo_mean.float().view(1, -1))

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        bb_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids]

    def encode_features(self, y: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (C_sem [B, L, D_t5], C_id [B, 512], C_geo [B, 212])."""
        b, m = y.shape
        idx = indices.to(device=y.device, dtype=torch.long)

        # Path A: Closed-form Wiener Physics Lifting -> 2D RGB face -> DINOv2-Small 257 tokens
        i_wiener = self.lifter.lift_image(y, idx, reg=self.lam)  # [B, 3, 128, 128] in [0, 1]
        view = F.interpolate(i_wiener, size=(224, 224), mode="bilinear", align_corners=False)
        vis_tok = self.backbone(view)  # [B, 257, 384]
        cls_tok = vis_tok[:, 0]
        grid = vis_tok[:, 1:].transpose(1, 2).reshape(b, -1, 16, 16)
        q2 = F.adaptive_avg_pool2d(grid, (2, 2)).flatten(1)
        q4 = F.adaptive_avg_pool2d(grid, (4, 4)).flatten(1)
        h_vis = self.vis_compress(torch.cat([cls_tok, q2], dim=-1))
        h_vis_spatial = self.geo_spatial(q4)

        # Path B: Subset-Adaptive ZCA-Whitened 1D Sensor Representation
        u = self.lifter.whiten_1d(y, idx, reg=1e-3)
        packed = u.new_zeros(b, self.max_measurements)
        mask = u.new_zeros(b, self.max_measurements)
        idx_b = idx.view(1, -1).expand(b, -1)
        packed.scatter_(1, idx_b, u)
        mask.scatter_(1, idx_b, 1.0)
        rate_s = torch.full((b, 1), math.log2(max(m, 1)) / 8.0, device=y.device, dtype=u.dtype)
        h_sen = self.sensor_mlp(torch.cat([packed, mask, rate_s], dim=-1))

        # Dual-Domain Fusion
        h_fused = self.fusion(torch.cat([h_vis, h_sen], dim=-1))

        c_id = F.normalize(self.id_head(h_fused).float(), dim=-1)
        c_geo = self.geo_mean.to(y.device) + self.geo_head(torch.cat([h_fused, h_vis_spatial], dim=-1)).float()
        s_low = self.sem_mid(h_fused).view(b, self.sem_tokens, 64)
        c_sem = self.sem_mean.to(y.device) + self.sem_up(s_low).float()
        return c_sem, c_id, c_geo

    def forward(self, y: torch.Tensor, indices: torch.Tensor):
        return self.encode_features(y, indices)


def factorized_condition_loss(
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gt_sem: torch.Tensor,
    gt_sem_mask: torch.Tensor,
    gt_id: torch.Tensor,
    gt_geo: torch.Tensor,
    pred_noisy: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    geo_inv_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Well-balanced Factorized Stage-1 loss (no semantic swamp, crisp ID + Geo + Invariance)."""
    p_sem, p_id, p_geo = pred
    l_id = (1.0 - F.cosine_similarity(F.normalize(p_id.float(), dim=-1), F.normalize(gt_id.float(), dim=-1), dim=-1)).mean()
    l_geo = F.smooth_l1_loss(p_geo.float(), gt_geo.float(), beta=0.01)

    m = gt_sem_mask.float().unsqueeze(-1)
    denom = m.sum().clamp_min(1.0)
    sem_diff = F.smooth_l1_loss(p_sem.float(), gt_sem.float(), reduction="none", beta=0.5)
    l_sem_l1 = (sem_diff * m).sum() / (denom * p_sem.shape[-1])
    sem_cos = 1.0 - F.cosine_similarity(p_sem.float(), gt_sem.float(), dim=-1)
    l_sem_cos = (sem_cos * gt_sem_mask.float()).sum() / denom
    l_sem = l_sem_l1 + 0.5 * l_sem_cos

    l_inv = p_geo.new_zeros(())
    if pred_noisy is not None:
        _, _, p_geo_noisy = pred_noisy
        l_inv = F.smooth_l1_loss(p_geo_noisy.float(), p_geo.detach().float(), beta=0.01) + F.smooth_l1_loss(
            p_geo_noisy.float(), gt_geo.float(), beta=0.01
        )

    # Balanced scales: l_id ~ 0.7 (weight 2.0), l_geo ~ 0.01 (weight 40.0 -> 0.4), l_sem ~ 0.4 (weight 0.25 -> 0.1)
    total = 2.0 * l_id + 40.0 * l_geo + 0.25 * l_sem + float(geo_inv_weight) * 40.0 * l_inv
    return total, {
        "total": float(total.detach()),
        "l_id": float(l_id.detach()),
        "l_geo": float(l_geo.detach()),
        "l_sem": float(l_sem.detach()),
        "l_inv": float(l_inv.detach()),
    }
