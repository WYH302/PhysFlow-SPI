"""x-prediction forward for MiniT2I with optional joint-attention tokens.

The official block computes a timestep vector and does not apply it. This path
keeps that behavior. Identity tokens are concatenated onto the text stream
inside each joint block, then stripped before the final layer. Gradient
checkpointing threads those tokens as tensors so the backward recompute sees
the same tokens as the forward.
"""

from __future__ import annotations

import torch

from src.models.minit2i_loader import import_official_modules


def _plain(block):
    def fn(x, txt, vec):
        return block(x, txt, vec)

    return fn


def _with_tokens(block):
    def fn(x, txt, vec, extra):
        merged = torch.cat([txt, extra.to(dtype=txt.dtype)], dim=1)
        x_out, txt_out = block(x, merged, vec)
        length = txt.shape[1]
        return x_out, txt_out[:, :length].contiguous(), txt_out[:, length:].contiguous()

    return fn


def _run(fn, args, checkpoint: bool):
    if checkpoint and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def predict_clean(
    net,
    img: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    attn_mask: torch.Tensor,
    extra: torch.Tensor | None = None,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Official MMJiT clean-image head, plus optional identity tokens."""
    mmdit, _ = import_official_modules()
    attn_mask = attn_mask.to(device=context.device)
    context = torch.where(
        attn_mask[:, :, None] > 0.5,
        context,
        net.mask_token.to(dtype=context.dtype),
    )
    x = net.img_embedder(img)
    pos = mmdit.get_2d_sincos_pos_embed(net.cfg.hidden_size, net.latent_img_size, x.device, x.dtype)
    x = x + pos[None]
    t_vec = net.t_embedder(t)
    txt = net.txt_embedder(context.to(dtype=net.txt_embedder.weight.dtype))
    pooled = context.mean(dim=1)
    vec = t_vec + net.pooled_embedder(pooled.to(dtype=net.pooled_embedder.weight.dtype))
    for block in net.txt_preamble_blocks:
        txt = block(txt)
    if extra is None:
        for block in net.double_blocks:
            x, txt = _run(_plain(block), (x, txt, vec), checkpoint)
    else:
        if extra.shape[0] != txt.shape[0]:
            raise RuntimeError(f"identity batch {extra.shape[0]} != text batch {txt.shape[0]}")
        for block in net.double_blocks:
            x, txt, extra = _run(_with_tokens(block), (x, txt, vec, extra), checkpoint)
    combined = torch.cat([txt, x], dim=1)
    out = net.final_layer(combined, vec)
    image = out[:, txt.shape[1] :, :]
    return net.unpatchify(image)


def pad_context_to_256(
    context: torch.Tensor,
    attn_mask: torch.Tensor,
    prompt_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad compact max_len=64 FLAN-T5 caches to MiniT2I's native 256 RoPE length.

    MMJiT replaces every position with mask==0 by net.mask_token, so zero-padding
    on the fly is mathematically identical to encoding at max_length=256 while
    cutting disk usage by 75%.
    """
    length = int(context.shape[1])
    if length == prompt_length:
        return context, attn_mask
    if length > prompt_length:
        return context[:, :prompt_length], attn_mask[:, :prompt_length]
    pad = prompt_length - length
    ctx_pad = context.new_zeros(context.shape[0], pad, context.shape[2])
    mask_pad = attn_mask.new_zeros(attn_mask.shape[0], pad)
    return torch.cat([context, ctx_pad], dim=1), torch.cat([attn_mask, mask_pad], dim=1)


def _with_slot_injection(block, n_id: int, n_geo: int):
    def fn(x, txt, vec, id_tok, geo_tok, id_scale, geo_scale):
        total = int(txt.shape[1])
        id_start = total - (n_id + n_geo)
        geo_start = total - n_geo
        prefix = txt[:, :id_start]
        slot_id = txt[:, id_start:geo_start]
        slot_geo = txt[:, geo_start:]
        if id_tok is not None:
            slot_id = slot_id + id_scale.to(dtype=txt.dtype) * id_tok.to(dtype=txt.dtype)
        if geo_tok is not None:
            slot_geo = slot_geo + geo_scale.to(dtype=txt.dtype) * geo_tok.to(dtype=txt.dtype)
        merged = torch.cat([prefix, slot_id, slot_geo], dim=1)
        return block(x, merged, vec)

    return fn


def predict_clean_factorized(
    net,
    img: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    attn_mask: torch.Tensor,
    conditioner=None,
    identity: torch.Tensor | None = None,
    geo: torch.Tensor | None = None,
    id_tokens: torch.Tensor | None = None,
    geo_tokens: torch.Tensor | None = None,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Official MMJiT clean-image head with RMSNorm-safe native-256 slot conditioning."""
    mmdit, _ = import_official_modules()
    context, attn_mask = pad_context_to_256(context, attn_mask, prompt_length=int(net.cfg.prompt_length))
    attn_mask = attn_mask.to(device=context.device)
    context = torch.where(
        attn_mask[:, :, None] > 0.5,
        context,
        net.mask_token.to(dtype=context.dtype),
    )
    x = net.img_embedder(img)
    pos = mmdit.get_2d_sincos_pos_embed(net.cfg.hidden_size, net.latent_img_size, x.device, x.dtype)
    x = x + pos[None]
    t_vec = net.t_embedder(t)
    txt = net.txt_embedder(context.to(dtype=net.txt_embedder.weight.dtype))
    pooled = context.mean(dim=1)
    vec = t_vec + net.pooled_embedder(pooled.to(dtype=net.pooled_embedder.weight.dtype))
    for block in net.txt_preamble_blocks:
        txt = block(txt)

    if conditioner is not None:
        if id_tokens is None and identity is not None:
            id_tokens = conditioner.project_id(identity)
        if geo_tokens is None and geo is not None:
            geo_tokens = conditioner.project_geo(geo)

    if conditioner is None or (id_tokens is None and geo_tokens is None):
        for block in net.double_blocks:
            x, txt = _run(_plain(block), (x, txt, vec), checkpoint)
    else:
        id_scales = conditioner.id_scales()
        geo_scales = conditioner.geo_scales()
        n_id = int(conditioner.n_id)
        n_geo = int(conditioner.n_geo)
        # Ensure non-None dummy tensor only when needed for torch.utils.checkpoint compatibility
        for idx, block in enumerate(net.double_blocks):
            fn = _with_slot_injection(block, n_id=n_id, n_geo=n_geo)
            s_id = id_scales[idx]
            s_geo = geo_scales[idx]
            if checkpoint and torch.is_grad_enabled():
                # Checkpoint only accepts tensors cleanly; handle optional id/geo explicitly
                if id_tokens is not None and geo_tokens is not None:
                    x, txt = torch.utils.checkpoint.checkpoint(
                        fn, x, txt, vec, id_tokens, geo_tokens, s_id, s_geo, use_reentrant=False
                    )
                elif id_tokens is not None:
                    def _id_only(x_, txt_, vec_, id_, sid_, _fn=fn, _sgeo=s_geo):
                        return _fn(x_, txt_, vec_, id_, None, sid_, _sgeo)
                    x, txt = torch.utils.checkpoint.checkpoint(
                        _id_only, x, txt, vec, id_tokens, s_id, use_reentrant=False
                    )
                else:
                    def _geo_only(x_, txt_, vec_, geo_, sgeo_, _fn=fn, _sid=s_id):
                        return _fn(x_, txt_, vec_, None, geo_, _sid, sgeo_)
                    x, txt = torch.utils.checkpoint.checkpoint(
                        _geo_only, x, txt, vec, geo_tokens, s_geo, use_reentrant=False
                    )
            else:
                x, txt = fn(x, txt, vec, id_tokens, geo_tokens, s_id, s_geo)

    combined = torch.cat([txt, x], dim=1)
    out = net.final_layer(combined, vec)
    image = out[:, txt.shape[1] :, :]
    return net.unpatchify(image)
