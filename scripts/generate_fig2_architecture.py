"""Generate publication-grade 100% vector IEEE architecture diagram (Fig. 2) for PhysFlow.

Embeds real CelebA-HQ ground truth, real zero-parameter Wiener-LMMSE lift, and real
PhysFlow 512x512 reconstruction alongside crisp vector modules, exact LaTeX math,
and the unified colorblind-safe Okabe-Ito/Morandi palette used across Figs. 1-8.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIG_DIR = ROOT / "outputs" / "figures"

# Canvas geometry (13.6 x 6.25 in -> data-unit aspect ratio for exact 1:1 square images)
FIG_W, FIG_H = 13.6, 6.25
X_MAX, Y_MAX = 100.0, 46.8
# 1 data-unit in X = FIG_W/X_MAX inches; 1 data-unit in Y = FIG_H/Y_MAX inches
# To draw a physical square of height h_y (data units), width w_x = h_y * (FIG_H/Y_MAX) / (FIG_W/X_MAX)
SQUARE_K = (FIG_H / Y_MAX) / (FIG_W / X_MAX)

# Unified Okabe-Ito / Morandi IEEE Palette (exact match to Figs. 1, 3-8)
C = {
    "navy": "#1B4F72",
    "navy_fill": "#EDF4F8",
    "navy_mid": "#D6E6F2",
    "steel": "#2E86C1",
    "amber": "#D35400",
    "amber_fill": "#FDF4EC",
    "amber_mid": "#F8DFC8",
    "teal": "#148F77",
    "teal_fill": "#ECF7F4",
    "teal_mid": "#CFECE4",
    "crimson": "#B03A2E",
    "slate": "#566573",
    "slate_fill": "#F5F7F9",
    "slate_mid": "#E5E8EC",
    "white": "#FFFFFF",
    "black": "#000000",
    "line": "#222222",
}


def setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "stix",
        "text.color": C["black"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def load_real_showcase_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real Ground Truth, Wiener-LMMSE lift, and PhysFlow M=256 reconstruction."""
    strip = np.array(Image.open(OUTPUTS / "phase2_teaser_strip_m64.png").convert("RGB"))
    _, W, _ = strip.shape
    pad = 2
    s = (W - (6 + 1) * pad) // 6
    gt = strip[pad:pad + s, pad:pad + s]
    wiener = strip[pad:pad + s, pad + (s + pad):pad + (s + pad) + s]

    v3_m256 = torch.load(
        OUTPUTS / "v4_toptier" / "fig1_fig6_v3_tiles.pt",
        map_location="cpu",
        weights_only=False,
    )["fig6a_v3_m256"]
    t = v3_m256[10].detach().cpu().float().clamp(0.0, 1.0)
    phys = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return gt, wiener, phys


def rbox(ax, x, y, w, h, fc, ec, lw=1.0, r=0.55, ls="-", zorder=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color="#222222", lw=1.05, ls="-", rad=0.0, ms=10.0, zorder=5):
    """Razor-sharp IEEE vector arrow (proportionally scaled arrowheads)."""
    conn = f"arc3,rad={rad}" if abs(rad) > 1e-6 else "arc3,rad=0.0"
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>,head_length=0.45,head_width=0.24",
        connectionstyle=conn,
        color=color, linewidth=lw, linestyle=ls,
        mutation_scale=ms, zorder=zorder, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)
    return a


def badge(ax, x, y, kind="train", zorder=8):
    """Crisp geometric status marker: solid amber dot = trainable, slate square = frozen."""
    if kind == "train":
        ax.add_patch(Circle((x, y), 0.36, facecolor=C["amber"], edgecolor=C["black"], linewidth=0.45, zorder=zorder))
    else:
        ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64, facecolor=C["slate"], edgecolor=C["black"], linewidth=0.45, zorder=zorder))


def draw_dmd_icon(ax, x, y, h_y, zorder=6):
    """Draw a miniature 1:1 square stacked binary {0,1} DMD mask icon."""
    w_x = h_y * SQUARE_K
    ax.add_patch(Rectangle((x + 0.28, y + 0.28), w_x, h_y, facecolor="#DCE2E8", edgecolor=C["slate"], linewidth=0.55, zorder=zorder))
    ax.add_patch(Rectangle((x, y), w_x, h_y, facecolor=C["white"], edgecolor=C["black"], linewidth=0.65, zorder=zorder + 1))
    n = 6
    cw, ch = w_x / n, h_y / n
    rng = np.random.default_rng(7)
    pat = rng.integers(0, 2, size=(n, n))
    for r in range(n):
        for c in range(n):
            if pat[r, c]:
                ax.add_patch(Rectangle((x + c * cw, y + r * ch), cw, ch, facecolor=C["navy"], edgecolor="none", zorder=zorder + 2))


def draw_mini_bars(ax, x, y, w, h, heights, colors, zorder=6):
    """Draw clean miniature spectral/waveform bar chart."""
    n = len(heights)
    bw = w / (n * 1.35)
    step = w / n
    ax.plot([x - 0.12, x + w + 0.1], [y, y], color=C["black"], linewidth=0.55, zorder=zorder + 2)
    for i, (hh, col) in enumerate(zip(heights, colors)):
        ax.add_patch(Rectangle((x + i * step + 0.06, y), bw, hh * h, facecolor=col, edgecolor="none", zorder=zorder + 1))


def embed_square_image(ax, img_arr, cx, cy, h_y, border_color, border_lw=1.0, zorder=6):
    """Embed a 1:1 square image at exact canvas coordinates (DPI-independent)."""
    w_x = h_y * SQUARE_K
    x0, x1 = cx - w_x * 0.5, cx + w_x * 0.5
    y0, y1 = cy - h_y * 0.5, cy + h_y * 0.5
    ax.imshow(img_arr, extent=[x0, x1, y0, y1], aspect="auto", interpolation="bilinear", zorder=zorder)
    ax.add_patch(Rectangle((x0, y0), w_x, h_y, facecolor="none", edgecolor=border_color, linewidth=border_lw, zorder=zorder + 1))


def main() -> None:
    setup_rc()
    gt_img, wiener_img, phys_img = load_real_showcase_images()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(0.0, Y_MAX)
    ax.axis("off")

    # =========================================================================
    # TOP HEADER & UNIFIED LEGEND BAR (Y = 43.8 .. 46.2)
    # =========================================================================
    ax.text(0.8, 45.0, "PhysFlow", fontsize=13.0, fontweight="bold", va="center", ha="left", color=C["black"])
    ax.text(11.6, 44.95, "Pixel-Space Optical Flow Architecture", fontsize=9.0, va="center", ha="left", color="#333333")

    leg_items = [
        (39.0, C["navy_fill"], C["navy"], "Pillar 1: SD-DDF"),
        (52.6, C["amber_fill"], C["amber"], "Pillar 2: LP-Flow"),
        (66.0, C["teal_fill"], C["teal"], "Pillar 3: BCM-Solver"),
    ]
    for lx, fc, ec, lbl in leg_items:
        rbox(ax, lx, 44.2, 1.9, 1.45, fc, ec, lw=1.0, r=0.35, zorder=4)
        ax.text(lx + 2.4, 44.95, lbl, fontsize=7.6, va="center", ha="left", color=C["black"])

    badge(ax, 81.6, 44.95, "train")
    ax.text(82.4, 44.95, "Trainable", fontsize=7.6, va="center", ha="left", color=C["black"])
    badge(ax, 90.4, 44.95, "frozen")
    ax.text(91.2, 44.95, "Frozen / Fixed", fontsize=7.6, va="center", ha="left", color=C["black"])

    ax.plot([0.8, 99.2], [43.35, 43.35], color="#CCCCCC", linewidth=0.7)

    # =========================================================================
    # STAGE 0: OPTICAL SENSING (X = 0.8 .. 15.2, Y = 20.6 .. 42.4)
    # =========================================================================
    rbox(ax, 0.8, 20.6, 14.4, 21.8, C["slate_fill"], C["slate"], lw=1.1, r=0.7, zorder=1)
    ax.text(8.0, 41.15, "Optical Sensing", fontsize=8.6, fontweight="bold", ha="center", va="center", color=C["black"])

    # Target Face sub-box
    rbox(ax, 1.5, 33.8, 13.0, 5.9, C["white"], C["slate"], lw=0.75, r=0.45, zorder=2)
    embed_square_image(ax, gt_img, 4.3, 36.75, h_y=4.4, border_color=C["slate"], border_lw=0.85)
    ax.text(10.5, 38.1, "Target Face", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(10.5, 36.5, r"$\mathbf{x} \in \mathcal{X}$", fontsize=7.5, ha="center", va="center")
    ax.text(10.5, 35.0, r"$512{\times}512$ RGB", fontsize=6.6, ha="center", va="center")

    arrow(ax, 8.0, 33.8, 8.0, 32.3, color=C["line"], lw=1.0)

    # Binary DMD Mask sub-box
    rbox(ax, 1.5, 27.1, 13.0, 5.2, C["white"], C["slate"], lw=0.75, r=0.45, zorder=2)
    draw_dmd_icon(ax, 2.35, 28.1, h_y=3.0)
    badge(ax, 13.8, 31.6, "frozen")
    ax.text(10.0, 30.8, "Binary DMD", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(10.0, 29.3, r"$\mathbf{\Phi}_{\mathcal{S}} \in \{0,1\}^{M\times N}$", fontsize=6.7, ha="center", va="center")
    ax.text(10.0, 28.0, r"$\mathbf{y}_{\mathcal{S}} = \mathbf{H}_{\mathcal{S}}\mathbf{x} + \mathbf{n}$", fontsize=6.4, ha="center", va="center")

    arrow(ax, 8.0, 27.1, 8.0, 25.6, color=C["line"], lw=1.0)

    # Bucket Readings sub-box
    rbox(ax, 1.5, 21.4, 13.0, 4.2, C["white"], C["slate"], lw=0.75, r=0.45, zorder=2)
    wave_h = [0.92, 0.58, 0.78, 0.42, 0.85, 0.34, 0.62, 0.48, 0.72, 0.30, 0.55, 0.25]
    draw_mini_bars(ax, 2.2, 22.15, 4.3, 2.4, wave_h, [C["navy"]] * len(wave_h))
    ax.text(10.5, 24.45, "Bucket Signal", fontsize=7.2, fontweight="bold", ha="center", va="center")
    ax.text(10.5, 23.15, r"$\mathbf{y}_{\mathcal{S}} \in \mathcal{Y}_{\mathcal{S}}$", fontsize=7.2, ha="center", va="center")
    ax.text(10.5, 22.0, r"$M{\in}\{16..256\}$", fontsize=6.4, ha="center", va="center")

    # =========================================================================
    # PILLAR 1: SD-DDF (X = 16.9 .. 54.6, Y = 20.6 .. 42.4)
    # =========================================================================
    rbox(ax, 16.9, 20.6, 37.7, 21.8, C["navy_fill"], C["navy"], lw=1.35, r=0.7, zorder=1)
    ax.text(35.75, 41.15, "1. SD-DDF: Spectral-Decoupled Dual-Domain Factorizer", fontsize=8.4, fontweight="bold", ha="center", va="center", color=C["black"])

    # Upper branch: Wiener Lift -> DINOv2-Small
    rbox(ax, 17.7, 33.4, 9.7, 6.2, C["white"], C["navy"], lw=0.85, r=0.45, zorder=2)
    embed_square_image(ax, wiener_img, 20.4, 36.5, h_y=4.4, border_color=C["navy"], border_lw=0.85)
    badge(ax, 26.75, 38.9, "frozen")
    ax.text(25.0, 37.75, "Wiener", fontsize=7.1, fontweight="bold", ha="center", va="center")
    ax.text(25.0, 36.4, "Lift", fontsize=7.1, fontweight="bold", ha="center", va="center")
    ax.text(25.0, 34.8, r"$\mathbf{x}_{\mathrm{wiener}}$", fontsize=7.2, ha="center", va="center")

    arrow(ax, 27.4, 36.5, 28.8, 36.5, color=C["navy"], lw=1.05)

    rbox(ax, 28.8, 33.4, 8.9, 6.2, C["white"], C["navy"], lw=0.85, r=0.45, zorder=2)
    badge(ax, 37.0, 38.9, "train")
    ax.text(33.15, 37.65, "DINOv2-Small", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(33.15, 35.95, "Spatial Encoder", fontsize=6.8, ha="center", va="center")
    ax.text(33.15, 34.45, "(Last-6 Blocks FT)", fontsize=6.3, ha="center", va="center", color="#333333")

    # Lower branch: Subset ZCA -> Residual Sensor MLP
    rbox(ax, 17.7, 26.1, 9.7, 6.1, C["white"], C["navy"], lw=0.85, r=0.45, zorder=2)
    badge(ax, 26.75, 31.5, "frozen")
    draw_mini_bars(ax, 18.4, 26.85, 3.1, 2.2, [0.95, 0.55, 0.30, 0.16, 0.10], [C["slate"]] * 5)
    ax.text(22.2, 27.95, r"$\rightarrow$", fontsize=8.0, ha="center", va="center", fontweight="bold")
    draw_mini_bars(ax, 22.9, 26.85, 3.1, 2.2, [0.72, 0.72, 0.72, 0.72, 0.72], [C["navy"]] * 5)
    ax.text(22.4, 30.85, "Subset ZCA", fontsize=7.2, fontweight="bold", ha="center", va="center")
    ax.text(22.4, 29.7, r"$\mathbf{u}_{\mathcal{S}} = \mathbf{\Sigma}_{y_{\mathcal{S}}}^{-1/2}\tilde{\mathbf{y}}_{\mathcal{S}}$", fontsize=6.1, ha="center", va="center")

    arrow(ax, 27.4, 29.15, 28.8, 29.15, color=C["navy"], lw=1.05)

    rbox(ax, 28.8, 26.1, 8.9, 6.1, C["white"], C["navy"], lw=0.85, r=0.45, zorder=2)
    badge(ax, 37.0, 31.5, "train")
    ax.text(33.15, 30.25, "Sensor MLP", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(33.15, 28.65, "Residual 1D Net", fontsize=6.8, ha="center", va="center")
    ax.text(33.15, 27.15, "(Eq. High-Freq)", fontsize=6.3, ha="center", va="center", color="#333333")

    # Gated Fusion vertical block
    rbox(ax, 39.2, 26.1, 5.7, 13.5, C["navy_mid"], C["navy"], lw=1.05, r=0.5, zorder=2)
    badge(ax, 44.2, 38.9, "train")
    ax.text(42.05, 36.2, "Gated\nFusion", fontsize=7.3, fontweight="bold", ha="center", va="center", linespacing=1.15)
    ax.add_patch(Circle((42.05, 32.0), 1.0, facecolor=C["white"], edgecolor=C["navy"], linewidth=0.9, zorder=4))
    ax.text(42.05, 31.98, r"$\otimes$", fontsize=10.0, ha="center", va="center", zorder=5)
    ax.text(42.05, 28.35, "Rate-\nAdaptive", fontsize=6.6, ha="center", va="center", linespacing=1.1)

    arrow(ax, 37.7, 36.5, 39.2, 36.5, color=C["navy"], lw=1.05)
    arrow(ax, 37.7, 29.15, 39.2, 29.15, color=C["navy"], lw=1.05)

    # Output Factors block (3 stacked pills)
    rbox(ax, 46.2, 26.1, 7.6, 13.5, C["white"], C["navy"], lw=0.9, r=0.5, zorder=2)
    ax.text(50.0, 38.65, r"Factors $\mathcal{Z}$", fontsize=7.2, fontweight="bold", ha="center", va="center")
    fac_specs = [
        (35.3, C["navy_fill"], C["navy"], r"$\mathbf{C}_{\mathrm{sem}}$ (Sem.)"),
        (31.2, C["amber_fill"], C["amber"], r"$\mathbf{c}_{\mathrm{id}}$ (ID)"),
        (27.1, C["teal_fill"], C["teal"], r"$\mathbf{c}_{\mathrm{geo}}$ (106-pt)"),
    ]
    for fy, ffc, fec, flbl in fac_specs:
        rbox(ax, 46.65, fy, 6.7, 2.7, ffc, fec, lw=0.75, r=0.35, zorder=3)
        ax.text(50.0, fy + 1.35, flbl, fontsize=6.6, fontweight="bold", ha="center", va="center", zorder=4)
        arrow(ax, 44.9, fy + 1.35, 46.2, fy + 1.35, color=C["navy"], lw=0.9, ms=8.5)

    # Bottom Recon-Free Analytics bar inside Pillar 1
    rbox(ax, 17.7, 21.4, 36.1, 3.3, C["white"], C["navy"], lw=0.8, r=0.4, ls="--", zorder=2)
    ax.text(35.75, 23.05, "Recon-Free Analytics (55.9 FPS, 5.55 GFLOPs): 106-pt Landmarks | 40 Attr.",
            fontsize=6.05, fontweight="bold", ha="center", va="center")
    arrow(ax, 50.0, 26.1, 50.0, 24.7, color=C["navy"], lw=0.95, ms=8.5)

    # Routing from Optical Sensing y_S -> SD-DDF
    ax.plot([14.5, 16.05, 16.05], [23.5, 23.5, 36.5], color=C["line"], lw=1.05, zorder=4)
    arrow(ax, 16.05, 36.5, 17.7, 36.5, color=C["line"], lw=1.05)
    arrow(ax, 16.05, 29.15, 17.7, 29.15, color=C["line"], lw=1.05)

    # =========================================================================
    # PILLAR 2: LP-FLOW (X = 56.5 .. 86.2, Y = 20.6 .. 42.4)
    # =========================================================================
    rbox(ax, 56.5, 20.6, 29.7, 21.8, C["amber_fill"], C["amber"], lw=1.35, r=0.7, zorder=1)
    ax.text(71.35, 41.15, "2. LP-Flow: Pixel-Space Flow Backbone", fontsize=8.4, fontweight="bold", ha="center", va="center", color=C["black"])

    # Frozen MiniT2I-B/16 outer shell
    rbox(ax, 57.4, 21.4, 27.9, 18.3, C["white"], C["slate"], lw=0.95, r=0.55, zorder=2)
    badge(ax, 84.4, 38.9, "frozen")
    ax.text(70.8, 38.7, "Frozen MiniT2I-B/16  (VAE-Free Pixel Space)", fontsize=7.2, fontweight="bold", ha="center", va="center")

    # RMSNorm-Safe Trailing Slot Injection inner box
    rbox(ax, 58.3, 29.5, 26.1, 7.9, C["amber_fill"], C["amber"], lw=0.95, r=0.45, zorder=3)
    badge(ax, 83.75, 36.75, "train")
    ax.text(70.2, 36.35, "RMSNorm-Safe Trailing Slot Injection (Zero Gates)", fontsize=6.35, fontweight="bold", ha="center", va="center")

    slot_specs = [
        (59.0, 8.4, C["navy_fill"], C["navy"], "[0 : 224]", r"$\mathbf{C}_{\mathrm{sem}}$"),
        (67.85, 7.9, C["amber_mid"], C["amber"], "[224 : 240]", r"$+g_{\mathrm{id}}\,\mathbf{c}_{\mathrm{id}}$"),
        (76.2, 7.5, C["teal_mid"], C["teal"], "[240 : 256]", r"$+g_{\mathrm{geo}}\,\mathbf{c}_{\mathrm{geo}}$"),
    ]
    for sx, sw, sfc, sec, s_idx, s_eq in slot_specs:
        rbox(ax, sx, 30.5, sw, 4.3, sfc, sec, lw=0.8, r=0.3, zorder=4)
        ax.text(sx + sw / 2, 33.45, s_idx, fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(sx + sw / 2, 31.65, s_eq, fontsize=6.7, ha="center", va="center", zorder=5)

    # Attention LoRA inner box
    rbox(ax, 58.3, 24.8, 26.1, 3.6, C["slate_fill"], C["slate"], lw=0.8, r=0.4, zorder=3)
    badge(ax, 83.6, 27.65, "train")
    ax.text(70.85, 26.6, r"Rank-16 Attention LoRA ($\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O}$, $2.51\mathrm{M}$ Params)", fontsize=6.45, fontweight="bold", ha="center", va="center")

    # Bottom velocity / clean RGB equation strip
    ax.text(71.35, 22.8, r"$\mathbf{v}_\theta(\mathbf{z}_t, t, \mathcal{Z}) \;\longrightarrow\; \hat{\mathbf{x}}_\theta = \mathbf{z}_t + (1-t)\mathbf{v}_\theta$",
            fontsize=7.2, ha="center", va="center", fontweight="bold")

    # Factor bundle arrow from SD-DDF -> LP-Flow
    arrow(ax, 53.8, 32.6, 58.3, 32.6, color=C["amber"], lw=1.35, ms=11.0)
    ax.text(55.55, 34.15, r"$\mathcal{Z}$", fontsize=8.2, fontweight="bold", ha="center", va="center")

    # =========================================================================
    # FINAL OUTPUT PANEL (X = 87.8 .. 99.2, Y = 20.6 .. 42.4)
    # =========================================================================
    rbox(ax, 87.8, 20.6, 11.4, 21.8, C["white"], C["navy"], lw=1.6, r=0.7, zorder=1)
    ax.text(93.5, 41.15, "Reconstructed", fontsize=8.4, fontweight="bold", ha="center", va="center", color=C["black"])
    embed_square_image(ax, phys_img, 93.5, 33.4, h_y=11.0, border_color=C["navy"], border_lw=1.3)
    ax.text(93.5, 26.25, r"$512{\times}512$ RGB", fontsize=7.7, fontweight="bold", ha="center", va="center")
    rbox(ax, 88.5, 21.4, 10.0, 3.6, C["navy_fill"], C["navy"], lw=0.75, r=0.35, zorder=3)
    ax.text(93.5, 23.9, "Fast-4: 0.20 s", fontsize=6.7, fontweight="bold", ha="center", va="center")
    ax.text(93.5, 22.4, "Full-16: 0.90 s", fontsize=6.7, ha="center", va="center")

    # =========================================================================
    # BOTTOM-LEFT: TRAINING-ONLY SUPERVISION (X = 0.8 .. 37.8, Y = 0.8 .. 18.2)
    # =========================================================================
    rbox(ax, 0.8, 0.8, 37.0, 17.4, C["slate_fill"], C["slate"], lw=1.1, ls="--", r=0.7, zorder=1)
    ax.text(19.3, 16.95, "Training-Only Supervision  (Zero Inference Overhead)", fontsize=8.3, fontweight="bold", ha="center", va="center")

    # Left: Factor & Invariance Loss
    rbox(ax, 1.8, 1.8, 16.0, 13.7, C["white"], C["navy"], lw=0.85, r=0.5, zorder=2)
    rbox(ax, 2.5, 10.3, 14.6, 4.3, C["slate_fill"], C["slate"], lw=0.7, r=0.35, zorder=3)
    badge(ax, 16.4, 13.9, "frozen")
    ax.text(9.5, 13.1, "Frozen Teacher Targets", fontsize=7.0, fontweight="bold", ha="center", va="center")
    ax.text(9.8, 11.4, "FLAN-T5 | AntelopeV2 | 106-pt", fontsize=6.05, ha="center", va="center")

    arrow(ax, 9.8, 10.3, 9.8, 8.75, color=C["navy"], lw=1.0)

    rbox(ax, 2.5, 2.6, 14.6, 6.1, C["navy_fill"], C["navy"], lw=0.8, r=0.35, zorder=3)
    ax.text(9.8, 7.35, "Factor + Rate Invariance", fontsize=7.1, fontweight="bold", ha="center", va="center")
    ax.text(9.8, 5.5, r"$\mathcal{L}_{\mathrm{cond}}(\mathcal{Z}, \mathcal{Z}^{*}) + \mathcal{L}_{\mathrm{inv}}$", fontsize=7.1, ha="center", va="center")
    ax.text(9.8, 3.65, r"Variable-rate $M \in [16, 256]$", fontsize=6.4, ha="center", va="center", color="#333333")

    # Right: Direct RGB Optical Measurement Loss
    rbox(ax, 19.0, 1.8, 17.8, 13.7, C["amber_fill"], C["amber"], lw=0.95, r=0.5, zorder=2)
    ax.text(27.9, 14.2, "Single-Step RGB Optical Loss", fontsize=7.4, fontweight="bold", ha="center", va="center")
    ax.text(27.9, 12.6, r"Active on flow times $t \geq 0.40$", fontsize=6.8, ha="center", va="center")

    rbox(ax, 20.0, 5.9, 15.8, 5.6, C["white"], C["amber"], lw=0.75, r=0.35, zorder=3)
    ax.text(24.0, 9.5, r"$\hat{\mathbf{x}}_\theta \;\to\; \mathbf{H}_{\mathcal{S}}\hat{\mathbf{x}}_\theta$", fontsize=6.9, ha="center", va="center")
    ax.text(24.0, 7.2, r"$\mathbf{r}_{\mathcal{S}} = \mathbf{H}_{\mathcal{S}}\hat{\mathbf{x}}_\theta - \mathbf{y}_{\mathcal{S}}$", fontsize=6.5, ha="center", va="center")
    hx = np.linspace(-1.4, 1.4, 60)
    hy = np.where(np.abs(hx) <= 0.65, 0.5 * hx**2, 0.65 * (np.abs(hx) - 0.325))
    hy_norm = hy / hy.max()
    ax.plot(32.1 + hx * 2.0, 6.75 + hy_norm * 3.3, color=C["amber"], lw=1.6, zorder=5)
    ax.plot([29.1, 35.1], [6.75, 6.75], color=C["black"], lw=0.55, zorder=4)
    ax.text(32.1, 10.65, "Huber", fontsize=6.3, fontweight="bold", ha="center", va="center")

    ax.text(27.9, 3.7, r"$\mathcal{L}_{\mathrm{meas}} = \mathbb{E}[t\,\mathrm{Huber}(\mathbf{H}_{\mathcal{S}}\hat{\mathbf{x}}_\theta - \mathbf{y}_{\mathcal{S}})]$",
            fontsize=6.6, ha="center", va="center", fontweight="bold")

    # =========================================================================
    # BOTTOM-RIGHT: PILLAR 3 — BCM-SOLVER (X = 39.4 .. 99.2, Y = 0.8 .. 18.2)
    # =========================================================================
    rbox(ax, 39.4, 0.8, 59.8, 17.4, C["teal_fill"], C["teal"], lw=1.35, r=0.7, zorder=1)
    ax.text(40.4, 17.0, "3. BCM-Solver: Blind Covariance-Manifold Inference", fontsize=7.65, fontweight="bold", ha="left", va="center")

    # Stage A1 (Top-Left): Blind Spectral Damping
    rbox(ax, 40.4, 9.2, 14.3, 6.4, C["white"], C["teal"], lw=0.9, r=0.45, zorder=2)
    ax.text(47.55, 14.55, "Blind Spectral Damping", fontsize=7.1, fontweight="bold", ha="center", va="center")
    eig_h = [0.95, 0.72, 0.52, 0.38, 0.28, 0.22, 0.17, 0.13, 0.12, 0.11, 0.11]
    eig_c = [C["navy"]] * 7 + [C["amber"]] * 4
    draw_mini_bars(ax, 41.3, 10.0, 5.2, 3.2, eig_h, eig_c)
    ax.text(50.7, 12.6, r"$\lambda_{\mathrm{adapt}}$", fontsize=8.1, fontweight="bold", ha="center", va="center")
    ax.text(50.7, 11.15, r"$\alpha = 8.0$", fontsize=6.8, ha="center", va="center")
    ax.text(50.7, 9.95, "(Prop. 2)", fontsize=6.2, ha="center", va="center", color="#333333")

    # Stage A2 (Bottom-Left): MD-FaceLoc Unaligned Pullback (dashed optional path)
    rbox(ax, 40.4, 1.9, 14.3, 6.2, C["white"], C["navy"], lw=0.85, ls="--", r=0.45, zorder=2)
    badge(ax, 54.0, 7.42, "train")
    ax.text(46.7, 6.95, "MD-FaceLoc (0.053 ms)", fontsize=6.65, fontweight="bold", ha="center", va="center")
    ax.text(47.55, 5.15, r"Unaligned $\mathbf{u}_{\mathcal{S}} \to \hat{\mathbf{p}}$", fontsize=6.7, ha="center", va="center")
    ax.text(47.55, 3.25, r"$\tilde{\mathbf{\Phi}}_{\mathcal{S}}(\hat{\mathbf{p}}) = \mathbf{\Phi}_{\mathcal{S}}\mathbf{W}_{\hat{\mathbf{p}}}$", fontsize=6.45, fontweight="bold", ha="center", va="center")

    # Stage B1 (Top-Mid): Kalman-Wiener Proximal (receives x_hat_theta straight down + lambda_adapt straight right!)
    rbox(ax, 56.6, 9.2, 16.0, 6.4, C["teal_mid"], C["teal"], lw=1.05, r=0.45, zorder=2)
    ax.text(64.6, 14.45, "Kalman-Wiener Proximal", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(64.6, 12.45, r"$\mathbf{\Sigma}_x$-Metric MAP Update (Prop. 1)", fontsize=6.65, ha="center", va="center")
    ax.text(64.6, 10.45, r"Tapered Schedule ($t \leq 0.93$)", fontsize=6.7, fontweight="bold", ha="center", va="center")

    # Stage B2 (Bottom-Mid): Rate-Adaptive Warm-Start (initializes ODE loop!)
    rbox(ax, 56.6, 1.9, 16.0, 6.2, C["white"], C["teal"], lw=0.9, r=0.45, zorder=2)
    ax.text(64.6, 6.95, "Rate-Adaptive Warm-Start", fontsize=7.2, fontweight="bold", ha="center", va="center")
    ax.text(64.6, 5.05, r"$\mathbf{z}_{t_0} = t_0\,\mathcal{U}(\mathbf{x}_{\mathrm{wiener}}) + (1-t_0)\epsilon$", fontsize=6.75, ha="center", va="center")
    ax.text(64.6, 3.15, r"$t_0(M) \in \{0.25, 0.30, 0.35\}$", fontsize=6.6, fontweight="bold", ha="center", va="center")

    # Arrows Stage A -> Stage B (zero crossings!)
    arrow(ax, 54.7, 12.4, 56.6, 12.4, color=C["teal"], lw=1.05)
    arrow(ax, 54.7, 5.0, 56.6, 5.0, color=C["teal"], lw=1.05)
    # Blind damping & pullback feed cleanly between A1/A2 and B1/B2
    ax.plot([55.65, 55.65], [5.0, 11.1], color=C["navy"], lw=0.85, ls="--", zorder=4)
    arrow(ax, 55.65, 11.1, 56.6, 11.1, color=C["navy"], lw=0.85, ls="--", ms=8.5)

    # Stage C: Flow ODE Integration (16-step / 4-step Fast)
    rbox(ax, 74.5, 2.8, 11.4, 12.2, C["white"], C["teal"], lw=0.95, r=0.5, zorder=2)
    ax.text(80.2, 13.7, "Flow ODE Loop", fontsize=7.4, fontweight="bold", ha="center", va="center")
    rbox(ax, 75.3, 8.7, 9.8, 3.6, C["navy_fill"], C["navy"], lw=0.75, r=0.35, zorder=3)
    ax.text(80.2, 11.15, "Full: 16 Steps", fontsize=6.9, fontweight="bold", ha="center", va="center")
    ax.text(80.2, 9.65, "Uniform Euler Grid", fontsize=6.15, ha="center", va="center")
    rbox(ax, 75.3, 4.1, 9.8, 3.6, C["amber_fill"], C["amber"], lw=0.75, r=0.35, zorder=3)
    ax.text(80.2, 6.55, "Fast: 4 Steps", fontsize=6.9, fontweight="bold", ha="center", va="center")
    ax.text(80.2, 5.05, r"CFG-Free ($13.1\times$)", fontsize=6.2, ha="center", va="center")

    arrow(ax, 72.6, 12.4, 74.5, 11.2, color=C["teal"], lw=1.05)
    arrow(ax, 72.6, 5.0, 74.5, 6.2, color=C["teal"], lw=1.05)

    # Stage D: Post-Flow Pixel Refiner R_phi
    rbox(ax, 87.8, 2.8, 10.4, 12.2, C["amber_fill"], C["amber"], lw=1.05, r=0.5, zorder=2)
    badge(ax, 97.45, 14.25, "train")
    ax.text(92.9, 13.1, "Pixel Refiner", fontsize=7.4, fontweight="bold", ha="center", va="center")
    ax.text(92.9, 11.2, r"$R_\phi$ ($2.27\mathrm{M}$)", fontsize=7.4, fontweight="bold", ha="center", va="center")
    ax.text(92.9, 9.2, "Res-UNet +", fontsize=6.6, ha="center", va="center")
    ax.text(92.9, 7.8, r"$0.75$ Final Proj.", fontsize=6.6, ha="center", va="center")
    rbox(ax, 88.6, 3.8, 8.8, 2.6, C["white"], C["amber"], lw=0.7, r=0.3, zorder=3)
    ax.text(93.0, 5.1, r"$+23.9\;\mathrm{ms}$", fontsize=6.8, fontweight="bold", ha="center", va="center")

    arrow(ax, 85.9, 8.8, 87.8, 8.8, color=C["teal"], lw=1.15)

    # =========================================================================
    # INTER-PILLAR CLEAN BUS CONNECTIONS (ZERO CROSSINGS, ZERO TEXT COLLISIONS!)
    # =========================================================================
    # 1. Optical Sensing y_S -> BCM-Solver routed cleanly down the X=38.6 vertical gutter
    #    to feed BOTH Blind Spectral Damping (top) AND MD-FaceLoc (bottom) with zero title overlap!
    ax.plot([8.0, 8.0, 38.6, 38.6], [20.6, 19.35, 19.35, 5.0], color=C["line"], lw=1.05, zorder=4)
    arrow(ax, 38.6, 12.4, 40.4, 12.4, color=C["line"], lw=1.05, ms=9.5)
    arrow(ax, 38.6, 5.0, 40.4, 5.0, color=C["line"], lw=1.05, ms=9.5)
    ax.text(23.5, 20.0, r"Measurement $\mathbf{y}_{\mathcal{S}}$ & sensing matrix $\mathbf{H}_{\mathcal{S}}$", fontsize=6.6, ha="center", va="center", color="#222222")

    # 2. LP-Flow <-> BCM-Solver iterative loop (completely clear of left-aligned BCM-Solver title!)
    # Downward: clean estimate x_hat_theta from LP-Flow -> Kalman-Wiener Proximal (at X=71.2, generous gap from title!)
    arrow(ax, 71.2, 20.6, 71.2, 15.6, color=C["amber"], lw=1.2, ms=11.0)
    ax.text(70.4, 19.4, r"$\hat{\mathbf{x}}_\theta$", fontsize=8.0, fontweight="bold", ha="right", va="center")

    # Upward: updated state (z_t, t) from Flow ODE Loop -> LP-Flow (at X=80.2)
    arrow(ax, 80.2, 15.0, 80.2, 20.6, color=C["teal"], lw=1.2, ls="--", ms=11.0)
    ax.text(81.0, 19.4, r"$\mathbf{z}_t, t$", fontsize=8.0, fontweight="bold", ha="left", va="center")

    # 3. Post-Flow Pixel Refiner R_phi -> Reconstructed 512x512 RGB Output (straight up!)
    arrow(ax, 93.5, 15.0, 93.5, 20.6, color=C["navy"], lw=1.45, ms=12.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "fig2_architecture.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved crisp vector architecture diagram: {out_pdf}")


if __name__ == "__main__":
    main()
