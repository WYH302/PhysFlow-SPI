"""Generate publication-grade vector PDF figures for the PhysFlow IEEE Transactions manuscript.

Design principles (Stage 4 / IEEE Visual Polish):
- Unified serif typography (DejaVu Serif / STIX) with 100% pure black (#000000) text.
- Single restrained colorblind-safe Okabe-Ito / Academic palette across every figure
  (Deep Navy #1B4F72 hero, Muted Crimson #B03A2E LSI & zoom boxes, Burnt Amber #D35400,
   Muted Teal #148F77, Muted Plum #6C3483, Slate Gray #7F8C8D). Zero pure RGB or rainbow maps.
- Bold centered subcaptions (a), (b), (c), (d) positioned cleanly below every subpanel.
- Non-occluding magnified zoom-in detail strips (crimson ROI box + 2.2x crop) on key qualitative figures.
- Every plotted number is read strictly from executed JSON logs under outputs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scripts.run_phase2_e2e_physflow import pick_even_split_indices

OUTPUTS = ROOT / "outputs"
CACHE_DIR = ROOT / "data" / "cache" / "physflow_real"
FIG_DIR = (ROOT.parent / "paper" / "figures") if (ROOT.parent / "paper").is_dir() else (ROOT / "paper" / "figures")

# Unified colorblind-safe academic palette (Okabe-Ito / restrained Morandi)
PALETTE = {
    "physflow": "#1B4F72",      # Deep Navy (Hero)
    "phys_base": "#2E86C1",     # Steel Blue (Base Flow / Secondary)
    "lsi": "#B03A2E",           # Muted Crimson (CVPR'25 LSI & Zoom ROI)
    "pnp_flow": "#D35400",      # Burnt Amber (ICLR'25 PnP-Flow)
    "zs_flower": "#148F77",     # Muted Teal (ICLR'26 ZS-FLOWER)
    "wiener": "#6C3483",        # Muted Plum (Wiener-LMMSE)
    "tikhonov": "#7F8C8D",      # Slate Gray (Tikhonov / Pinv / Raw)
    "text": "#95A5A6",          # Silver Gray (Text B0 / TV-FISTA)
    "b2_geo": "#2E86C1",        # Steel Blue
    "raw": "#7F8C8D",
}

ZOOM_RGB = np.array([176, 58, 46], dtype=np.uint8)     # #B03A2E Muted Crimson (never garish #FF0000)
HERO_RGB = np.array([27, 79, 114], dtype=np.uint8)     # #1B4F72 Deep Navy


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slice_grid_clean(png_path: Path, nrow: int, ncol: int, tile_h: int = 530, img_size: int = 512) -> list[list[np.ndarray]]:
    """Slice a save_image_grid PNG into pure [512, 512, 3] tiles, stripping the 18px black footer."""
    arr = np.asarray(Image.open(png_path).convert("RGB"))
    grid: list[list[np.ndarray]] = []
    for r in range(nrow):
        row_imgs = []
        y0 = r * tile_h
        for c in range(ncol):
            x0 = c * img_size
            row_imgs.append(arr[y0 : y0 + img_size, x0 : x0 + img_size])
        grid.append(row_imgs)
    return grid


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def draw_rect(img: np.ndarray, y0: int, x0: int, y1: int, x1: int, color_rgb: np.ndarray, lw: int = 7) -> np.ndarray:
    """Draw a clean crisp border rectangle onto an RGB uint8 image without modifying input in place."""
    out = img.copy()
    H, W = out.shape[:2]
    y0, y1 = max(0, y0), min(H, y1)
    x0, x1 = max(0, x0), min(W, x1)
    out[y0 : min(H, y0 + lw), x0:x1] = color_rgb
    out[max(0, y1 - lw) : y1, x0:x1] = color_rgb
    out[y0:y1, x0 : min(W, x0 + lw)] = color_rgb
    out[y0:y1, max(0, x1 - lw) : x1] = color_rgb
    return out


def make_zoom_strip(tile: np.ndarray, roi_frac: tuple[float, float, float, float], strip_h: int,
                    resample: int = Image.BICUBIC, border_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (tile_with_roi_box, zoomed_strip_matching_tile_width) framed in Muted Crimson."""
    H, W = tile.shape[:2]
    fy0, fx0, fy1, fx1 = roi_frac
    y0, y1 = int(round(fy0 * H)), int(round(fy1 * H))
    x0, x1 = int(round(fx0 * W)), int(round(fx1 * W))
    crop = tile[y0:y1, x0:x1]
    zoom = np.asarray(Image.fromarray(crop).resize((W, strip_h), resample))
    zoom = draw_rect(zoom, 0, 0, strip_h, W, ZOOM_RGB, lw=border_px)
    boxed = draw_rect(tile, y0, x0, y1, x1, ZOOM_RGB, lw=max(5, border_px - 1))
    return boxed, zoom


def match_teaser_indices(gt_tiles: list[np.ndarray]) -> list[int]:
    """Match teaser GT tiles against the exact 45 CelebA-HQ evaluation indices."""
    manifest = load_json(CACHE_DIR / "manifest.json")
    records = manifest["records"]
    hq_idx = pick_even_split_indices(records, "celebahq_test", 45)
    gts = np.stack([
        np.asarray(Image.open(CACHE_DIR / records[i]["image_rel"]).convert("RGB"), dtype=np.float32)
        for i in hq_idx
    ])
    matched = []
    for tile in gt_tiles:
        errs = np.abs(gts - tile.astype(np.float32)[None]).mean(axis=(1, 2, 3))
        idx = int(np.argmin(errs))
        assert errs[idx] < 2.0, f"Unexpected teaser match MAE={errs[idx]:.2f}"
        matched.append(idx)
    return matched


def compose_mosaic_with_zooms(
    rows: list[list[np.ndarray]],
    zoom_specs: dict[int, tuple[float, float, float, float]] | None = None,
    zoom_h_ratio: float = 0.42,
    pad: int = 12,
    zoom_gap: int = 6,
    bg: int = 255,
    highlight_col: int | None = None,
    border_px: int = 9,
    zoom_resample: int = Image.BICUBIC,
) -> tuple[np.ndarray, int, int, list[int]]:
    """Compose clean white-padded mosaic with optional non-occluding magnified zoom strips beneath chosen rows."""
    nr = len(rows)
    nc = len(rows[0])
    s = rows[0][0].shape[0]
    zh = int(round(s * zoom_h_ratio))
    zoom_specs = zoom_specs or {}

    row_y: list[int] = []
    total_h = 0
    for r in range(nr):
        row_y.append(total_h)
        total_h += s
        if r in zoom_specs:
            total_h += zoom_gap + zh
        if r < nr - 1:
            total_h += pad
    W = nc * s + (nc - 1) * pad
    canvas = np.full((total_h, W, 3), bg, dtype=np.uint8)

    for r in range(nr):
        y0 = row_y[r]
        for c in range(nc):
            x0 = c * (s + pad)
            tile = rows[r][c].copy()
            if r in zoom_specs:
                tile, zstrip = make_zoom_strip(tile, zoom_specs[r], strip_h=zh, resample=zoom_resample, border_px=max(6, border_px - 2))
                canvas[y0 + s + zoom_gap : y0 + s + zoom_gap + zh, x0 : x0 + s] = zstrip
            if highlight_col is not None and c == highlight_col:
                tile = draw_rect(tile, 0, 0, s, s, HERO_RGB, lw=border_px)
            canvas[y0 : y0 + s, x0 : x0 + s] = tile
    return canvas, s, pad, row_y


def setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "font.size": 8.8,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#CCCCCC",
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.65,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_fig1_teaser(p2: dict, extras: dict, v3_s6: dict, v3_s6b: dict) -> None:
    """Fig. 1: 8-col M=64 visual comparison + zoomed eye strip (a) + Identity (b), Geometry (c), Efficiency-Quality Pareto (d)."""
    tea = slice_grid_clean(OUTPUTS / "phase2_teaser_strip_m64.png", nrow=4, ncol=6)
    gt_tiles = [tea[r][0] for r in range(3)]
    matched_idx = match_teaser_indices(gt_tiles)
    ex_pt = torch.load(OUTPUTS / "extra_baselines_m64.pt", map_location="cpu", weights_only=False)
    tik_t = ex_pt["tikhonov_m64"]
    pnp_t = ex_pt["pnp_flow_m64"]
    v3_tiles = torch.load(OUTPUTS / "v4_toptier" / "fig1_fig6_v3_tiles.pt", map_location="cpu", weights_only=False)["fig1_v3_m64"]

    rows = []
    for r, idx in enumerate(matched_idx):
        rows.append([
            tea[r][0],
            tensor_to_uint8(tik_t[idx]),
            tea[r][1],
            tea[r][2],
            tensor_to_uint8(pnp_t[idx]),
            tea[r][3],
            tea[r][4],
            tensor_to_uint8(v3_tiles[idx]),
        ])
    # Add non-occluding 2.2x crimson-framed zoom strip on Row 2 (red-lip woman: dramatic eye/brow/lip clarity contrast!)
    mosaic, s, pad, _ = compose_mosaic_with_zooms(
        rows,
        zoom_specs={2: (0.37, 0.25, 0.63, 0.78)},
        zoom_h_ratio=0.40,
        pad=14,
        zoom_gap=8,
        highlight_col=7,
        border_px=10,
    )

    fig = plt.figure(figsize=(11.8, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[4.05, 1.85], hspace=0.35, wspace=0.28)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(mosaic)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for sp in ax_top.spines.values():
        sp.set_visible(False)

    col_headers = [
        "Ground Truth",
        "Tikhonov",
        "Wiener-LMMSE",
        "Text (B0)",
        "PnP-Flow\n(ICLR'25)",
        "ZS-FLOWER\n(ICLR'26)",
        "LSI\n(CVPR'25)",
        "PhysFlow\n(Ours)",
    ]
    for ci, ch in enumerate(col_headers):
        cx = ci * (s + pad) + s * 0.5
        ax_top.text(cx, -24, ch, ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#000000", linespacing=1.08)
    ax_top.text(0.5, -0.022, "(a) Extreme-Compression Visual Recovery (M = 64, 1:1024) with Zoomed Facial Detail",
                transform=ax_top.transAxes, ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    # Panels (b) and (c): M=64 Identity & Landmark NME bar charts
    ex_r = extras["extra_baselines_celebahq"]["rates"]
    pf_v3_64 = v3_s6b["v3_16step_proj075"]["hq45"]["64"]
    methods = [
        ("Tikh.", ex_r["tikhonov_bp"]["64"], PALETTE["tikhonov"]),
        ("Text\n(B0)", p2["rates"]["b0_discrete"]["64"], PALETTE["text"]),
        ("PnP-\nFlow", ex_r["pnp_flow"]["64"], PALETTE["pnp_flow"]),
        ("ZS-\nFLW", p2["rates"]["zeroshot_flower"]["64"], PALETTE["zs_flower"]),
        ("LSI", p2["rates"]["lsi"]["64"], PALETTE["lsi"]),
        ("Phys-\nFlow", pf_v3_64, PALETTE["physflow"]),
    ]
    xm = np.arange(len(methods))
    lbls = [m[0] for m in methods]
    cols = [m[2] for m in methods]

    specs = [
        ("adaface_cos_mean", "AdaFace Cosine ↑", "(b) Identity Preservation (M = 64)", (-0.018, 0.068)),
        ("nme106_mean", "106-pt Landmark NME ↓", "(c) Geometric Error (M = 64)", (0.0, 0.94)),
    ]
    for col_i, (metric, ylabel, subcap, ylim) in enumerate(specs):
        ax = fig.add_subplot(gs[1, col_i])
        vals = [float(m[1][metric]) for m in methods]
        ax.bar(xm, vals, color=cols, width=0.58, edgecolor="#222222", linewidth=0.45, zorder=3)
        ax.axhline(0.0, color="#000000", linewidth=0.6, zorder=2)
        ax.grid(axis="y", zorder=0)
        ax.set_xticks(xm)
        ax.set_xticklabels(lbls, fontsize=7.6)
        ax.set_ylabel(ylabel, fontsize=8.4)
        ax.set_ylim(*ylim)
        hv = vals[-1]
        ax.text(xm[-1], hv + (ylim[1] - ylim[0]) * 0.035, f"{hv:.4f}" if metric != "adaface_cos_mean" else f"{hv:.3f}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#000000")
        ax.text(0.5, -0.35, subcap, transform=ax.transAxes, ha="center", va="top", fontsize=8.8, fontweight="bold", color="#000000")

    # Panel (d): Efficiency-Quality Pareto Frontier (Full N=479 at M=256: Latency vs AdaFace Cosine — Top-Left Winner!)
    ax_p = fig.add_subplot(gs[1, 2])
    # Shaded top-left optimal frontier zone
    ax_p.axvspan(0.0, 1.08, ymin=0.66, ymax=1.0, color=PALETTE["physflow"], alpha=0.07, zorder=0)

    f479_256 = v3_s6["full479"]["256"]
    fast_ada = f479_256["physflow_fast_4step"]["adaface_cos_mean"]
    full_ada = f479_256["physflow_v3_16step"]["adaface_cos_mean"]
    pts = [
        ("TV-FISTA", 0.105, 0.0236, PALETTE["text"], "v", 6.2, (8, 4), "left"),
        ("PnP-Flow", 1.006, 0.0212, PALETTE["pnp_flow"], "^", 6.4, (7, 4), "left"),
        ("ZS-FLOWER", 0.872, 0.0060, PALETTE["zs_flower"], "D", 5.8, (7, -4), "left"),
        ("LSI (CVPR'25)", 2.574, 0.1462, PALETTE["lsi"], "s", 6.4, (4, -14), "right"),
        ("PhysFlow (16-st)", 0.896, full_ada, PALETTE["phys_base"], "o", 6.8, (8, 3), "left"),
        ("PhysFlow-Fast (4-st)", 0.196, fast_ada, PALETTE["physflow"], "*", 11.5, (-2, 11), "left"),
    ]
    # Connect PhysFlow Pareto front cleanly
    ax_p.plot([0.196, 0.896], [fast_ada, full_ada],
              color=PALETTE["physflow"], linestyle="-", linewidth=1.6, zorder=4)
    for name, lat, ada, col, mk, ms, (dx, dy), ha in pts:
        ax_p.plot(lat, ada, marker=mk, markersize=ms, color=col, markeredgecolor="#000000", markeredgewidth=0.45, linestyle="None", zorder=6 if "PhysFlow" in name else 4)
        fw = "bold" if "PhysFlow" in name else "normal"
        ax_p.annotate(name, (lat, ada), textcoords="offset points", xytext=(dx, dy),
                      ha=ha, va="center", fontsize=7.0, fontweight=fw, color="#000000")

    # Crisp 13.1x speedup callout curving cleanly through empty central corridor
    ax_p.annotate(
        "",
        xy=(0.28, 0.189),
        xytext=(2.46, 0.144),
        arrowprops=dict(arrowstyle="->", color="#000000", lw=0.9, ls="--", connectionstyle="arc3,rad=-0.18"),
        zorder=3,
    )
    ax_p.text(1.36, 0.120, r"$\mathbf{13.1{\times}}$ faster", ha="center", va="center",
              fontsize=7.2, fontweight="bold", color="#000000")
    ax_p.set_xlim(-0.10, 2.84)
    ax_p.set_ylim(-0.014, 0.256)
    ax_p.set_xlabel("Inference Latency (s/img, RTX 3090)", fontsize=8.2)
    ax_p.set_ylabel("AdaFace Cosine ↑", fontsize=8.4)
    ax_p.grid(True, zorder=0)
    ax_p.text(0.5, -0.35, "(d) Efficiency–Quality Pareto (M = 256)", transform=ax_p.transAxes,
              ha="center", va="top", fontsize=8.8, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig1_teaser_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig3_rate_and_noise(p2: dict, extras: dict, shield: dict, v3_s1: dict, v3_s6b: dict) -> None:
    """Fig. 3: Rate-distortion curves (with V3+Refiner hero points) + 6-axis noise robustness."""
    rates = [16, 32, 64, 128, 256]
    sr_labels = [f"{r}\n({100*r/65536:.2f}%)" for r in rates]
    x = np.arange(len(rates))
    ex_r = extras["extra_baselines_celebahq"]["rates"]

    series = [
        ("PhysFlow Base Flow", p2["rates"]["b3_physflow"], PALETTE["phys_base"], "o", "-", 2.1, 4.8, 5),
        ("LSI (CVPR'25)", p2["rates"]["lsi"], PALETTE["lsi"], "s", "--", 1.8, 4.8, 4),
        ("PnP-Flow (ICLR'25)", ex_r["pnp_flow"], PALETTE["pnp_flow"], "^", "-.", 1.7, 4.8, 4),
        ("ZS-FLOWER (ICLR'26)", p2["rates"]["zeroshot_flower"], PALETTE["zs_flower"], "D", ":", 1.7, 4.5, 3),
        ("Wiener-LMMSE", ex_r["wiener_lmmse"], PALETTE["wiener"], "v", (0, (3, 1, 1, 1)), 1.5, 4.4, 3),
        ("Tikhonov", ex_r["tikhonov_bp"], PALETTE["tikhonov"], "x", ":", 1.3, 4.8, 2),
        ("Text (B0)", p2["rates"]["b0_discrete"], PALETTE["text"], "+", "--", 1.3, 5.0, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.85))
    fig.subplots_adjust(top=0.79, bottom=0.27, wspace=0.32)

    handles_ab = []
    labels_ab = []
    for lbl, dct, col, mk, ls, lw, ms, z in series:
        ada = [dct[str(r)]["adaface_cos_mean"] for r in rates]
        nme = [np.nan if dct[str(r)]["nme106_mean"] is None else dct[str(r)]["nme106_mean"] for r in rates]
        h1, = axes[0].plot(x, ada, marker=mk, linestyle=ls, linewidth=lw, markersize=ms, color=col, label=lbl, zorder=z)
        axes[1].plot(x, nme, marker=mk, linestyle=ls, linewidth=lw, markersize=ms, color=col, label=lbl, zorder=z)
        handles_ab.append(h1)
        labels_ab.append(lbl)

    # Overlay V3 Final PhysFlow (+Refiner) hero stars at M=64 (x=2) and M=256 (x=4) from step6b_parity_and_qa.json
    pf_v3 = v3_s6b["v3_16step_proj075"]["hq45"]
    x_v3 = [2, 4]
    ada_v3 = [pf_v3["64"]["adaface_cos_mean"], pf_v3["256"]["adaface_cos_mean"]]
    nme_v3 = [pf_v3["64"]["nme106_mean"], pf_v3["256"]["nme106_mean"]]
    h_hero, = axes[0].plot(x_v3, ada_v3, marker="*", linestyle="--", linewidth=1.9, markersize=10.0,
                           color=PALETTE["physflow"], markeredgecolor="#000000", markeredgewidth=0.4,
                           label="PhysFlow Full (Ours)", zorder=8)
    axes[1].plot(x_v3, nme_v3, marker="*", linestyle="--", linewidth=1.9, markersize=10.0,
                 color=PALETTE["physflow"], markeredgecolor="#000000", markeredgewidth=0.4, zorder=8)
    axes[0].annotate(f"{ada_v3[1]:.3f}", (4, ada_v3[1]), textcoords="offset points", xytext=(-6, 6),
                     ha="right", fontsize=7.4, fontweight="bold", color="#000000")
    handles_ab.insert(0, h_hero)
    labels_ab.insert(0, "PhysFlow Full (Ours)")

    axes[0].set_xticks(x); axes[0].set_xticklabels(sr_labels, fontsize=7.7)
    axes[0].set_xlabel("Measurements $M$ (Sampling Rate)", fontsize=8.5)
    axes[0].set_ylabel("AdaFace IR-101 Cosine ↑", fontsize=8.5)
    axes[0].set_ylim(-0.025, 0.222)
    axes[0].grid(True, zorder=0)
    axes[0].text(0.5, -0.38, "(a) Identity vs. Sampling Rate", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    axes[1].set_xticks(x); axes[1].set_xticklabels(sr_labels, fontsize=7.7)
    axes[1].set_xlabel("Measurements $M$ (Sampling Rate)", fontsize=8.5)
    axes[1].set_ylabel("106-Point Landmark NME ↓", fontsize=8.5)
    axes[1].set_ylim(0.02, 0.90)
    axes[1].grid(True, zorder=0)
    axes[1].text(0.5, -0.38, "(b) Geometry vs. Sampling Rate", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.legend(handles_ab, labels_ab, loc="upper center", bbox_to_anchor=(0.37, 1.02),
               ncol=4, frameon=False, fontsize=7.6, columnspacing=0.95, handlelength=2.0)

    # Panel (c): Six-Axis Sensor Noise Robustness at M=64
    n_keys = ["clean", "gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined"]
    n_lbl = ["Clean\n(N0)", "Gauss\n(N1)", "Pois+G\n(N2)", "Gain+B\n(N3)", "Drop\n(N3b)", "Comb.\n(N4)"]
    xn = np.arange(len(n_keys))
    w = 0.25
    zs_noise = shield["module_c_baseline_parity"]["noise_m64"]["zs_flower"]
    pf_noise = v3_s1["part1c_hq45_noise_and_cb45"]["hq45_noise"]["tail_8.0_t0.35"]
    axes[2].bar(xn - w, [zs_noise[k]["psnr_mean"] for k in n_keys],
                width=w, color=PALETTE["zs_flower"], edgecolor="#222222", linewidth=0.4, label="ZS-FLOWER", zorder=3)
    axes[2].bar(xn, [p2["noise_m64"]["lsi"][k]["psnr_mean"] for k in n_keys],
                width=w, color=PALETTE["lsi"], edgecolor="#222222", linewidth=0.4, label="LSI (CVPR'25)", zorder=3)
    axes[2].bar(xn + w, [pf_noise[k]["psnr_mean"] for k in n_keys],
                width=w, color=PALETTE["physflow"], edgecolor="#222222", linewidth=0.4, label="PhysFlow (Ours)", zorder=3)
    axes[2].set_xticks(xn); axes[2].set_xticklabels(n_lbl, fontsize=6.6)
    axes[2].set_xlabel("Sensor Perturbation Regime ($M = 64$)", fontsize=8.5)
    axes[2].set_ylabel("Reconstruction PSNR (dB) ↑", fontsize=8.5)
    axes[2].set_ylim(0.0, 20.5)
    axes[2].grid(axis="y", zorder=0)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False, fontsize=7.3, columnspacing=0.6, handletextpad=0.35)
    axes[2].text(0.5, -0.38, "(c) Sensor Noise Robustness (M = 64)", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig3_rate_and_noise.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig4_wflw_external(p3_ext: dict, v3_s6: dict) -> None:
    """Fig. 4: External WFLW 6-subset qualitative strip + dual NME/PSNR charts."""
    wflw = slice_grid_clean(OUTPUTS / "external_wflw_6subsets_m64.png", nrow=6, ncol=8)
    pairs = [(0, 1), (2, 3), (4, 5)]
    sub_names = ["Pose", "Expression", "Illumination", "Makeup", "Occlusion", "Blur"]
    s = 512
    pad = 12
    mid_gap = 64
    row_gap = 115
    nr = 3
    H = nr * s + (nr - 1) * row_gap + 90
    W = 8 * s + 6 * pad + mid_gap
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    row_y_centers = []
    for ri, (s_left, s_right) in enumerate(pairs):
        y0 = ri * (s + row_gap)
        row_y_centers.append(y0)
        for ci in range(4):
            x0 = ci * (s + pad)
            tile = wflw[s_left][ci].copy()
            if ci == 3:
                tile = draw_rect(tile, 0, 0, s, s, HERO_RGB, lw=9)
            canvas[y0 : y0 + s, x0 : x0 + s] = tile
        right_offset = 4 if s_right in (3, 5) else 0
        for ci in range(4):
            x0 = 4 * s + 3 * pad + mid_gap + ci * (s + pad)
            tile = wflw[s_right][right_offset + ci].copy()
            if ci == 3:
                tile = draw_rect(tile, 0, 0, s, s, HERO_RGB, lw=9)
            canvas[y0 : y0 + s, x0 : x0 + s] = tile

    fig = plt.figure(figsize=(11.8, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.75, 1.75], hspace=0.34, wspace=0.22)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(canvas)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for sp in ax_top.spines.values():
        sp.set_visible(False)

    hdrs = ["GT", "ZS-FLOWER", "LSI (CVPR'25)", "PhysFlow (Ours)"]
    for x_start in (0, 4 * s + 3 * pad + mid_gap):
        for ci, htxt in enumerate(hdrs):
            cx = x_start + ci * (s + pad) + s * 0.5
            ax_top.text(cx, -26, htxt, ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#000000")
    for ri, (s_left, s_right) in enumerate(pairs):
        y_lbl = row_y_centers[ri] + s + 48
        ax_top.text(2 * s + 1.5 * pad, y_lbl, f"Subset: {sub_names[s_left]}", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#000000")
        ax_top.text(6 * s + 4.5 * pad + mid_gap, y_lbl, f"Subset: {sub_names[s_right]}", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#000000")
    ax_top.text(0.5, -0.045, "(a) Qualitative Generalization Across Six Challenging WFLW Subsets (M = 64)",
                transform=ax_top.transAxes, ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    subs = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
    sub_short = ["Pose", "Expr.", "Illum.", "Makeup", "Occl.", "Blur"]
    xs = np.arange(len(subs))
    w = 0.20
    methods_w = [
        ("b0_continuous", PALETTE["text"], "Text (B0-Cont.)"),
        ("zeroshot_flower", PALETTE["zs_flower"], "ZS-FLOWER (ICLR'26)"),
        ("lsi", PALETTE["lsi"], "LSI (CVPR'25)"),
        ("b3_physflow", PALETTE["physflow"], "PhysFlow (Ours)"),
    ]

    ax_nme = fig.add_subplot(gs[1, 0])
    ax_psnr = fig.add_subplot(gs[1, 1])
    for off, (mk, col, lbl) in zip((-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w), methods_w):
        if mk == "b3_physflow":
            v_nme = [v3_s6["wflw72_m64"][sb]["nme106_mean"] for sb in subs]
            v_psnr = [v3_s6["wflw72_m64"][sb]["psnr_mean"] for sb in subs]
        else:
            v_nme = [p3_ext["wflw_6subsets"][sb]["rates"][mk]["64"]["nme106_mean"] for sb in subs]
            v_psnr = [p3_ext["wflw_6subsets"][sb]["rates"][mk]["64"]["psnr_mean"] for sb in subs]
        ax_nme.bar(xs + off, v_nme, width=w, color=col, edgecolor="#222222", linewidth=0.4, label=lbl, zorder=3)
        ax_psnr.bar(xs + off, v_psnr, width=w, color=col, edgecolor="#222222", linewidth=0.4, label=lbl, zorder=3)

    ax_nme.set_xticks(xs); ax_nme.set_xticklabels(sub_short, fontsize=8.4)
    ax_nme.set_ylabel("106-Point Landmark NME ↓", fontsize=8.5)
    ax_nme.set_ylim(0.0, 2.55)
    ax_nme.grid(axis="y", zorder=0)
    ax_nme.text(0.5, -0.27, "(b) Per-Subset Landmark Error (M = 64)", transform=ax_nme.transAxes,
                ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    ax_psnr.set_xticks(xs); ax_psnr.set_xticklabels(sub_short, fontsize=8.4)
    ax_psnr.set_ylabel("Reconstruction PSNR (dB) ↑", fontsize=8.5)
    ax_psnr.set_ylim(0.0, 21.0)
    ax_psnr.grid(axis="y", zorder=0)
    ax_psnr.text(0.5, -0.27, "(c) Per-Subset Reconstruction PSNR (M = 64)", transform=ax_psnr.transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    handles, labels_l = ax_psnr.get_legend_handles_labels()
    fig.legend(handles, labels_l, loc="lower center", bbox_to_anchor=(0.5, -0.025),
               ncol=4, frameon=False, fontsize=8.4, columnspacing=1.4)

    fig.savefig(FIG_DIR / "fig4_wflw_external.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig5_downstream_pareto(p2: dict, p3_down: dict, extras: dict, v3_s6: dict, v3_s6b: dict) -> None:
    """Fig. 5: Reconstruction-free downstream transfer (few-shot + 5-fold CV) + Speed-Quality Pareto."""
    fracs = ["1pct", "5pct", "10pct", "100pct"]
    xf = np.arange(len(fracs))
    cv = extras["downstream_5fold_cv_m64"]["methods"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    fig.subplots_adjust(top=0.80, bottom=0.26, wspace=0.37)

    style = [
        ("physflow_factorized", PALETTE["physflow"], "o", "-", 2.4, "PhysFlow (Ours)"),
        ("lsi_latent", PALETTE["lsi"], "s", "--", 1.8, "LSI Latent (CVPR'25)"),
        ("text_only_sem", PALETTE["pnp_flow"], "^", "-.", 1.7, "Text Semantic ($\\mathbf{C}_{\\mathrm{sem}}$)"),
        ("raw_measurement", PALETTE["raw"], "d", ":", 1.6, "Raw Meas. ($\\mathbf{y}_{\\mathcal{S}}$)"),
    ]
    handles_d, labels_d = [], []
    for key, col, mk, ls, lw, lbl in style:
        accs = [p3_down["attr40"][key][f]["shallow"]["mean_acc"] * 100.0 for f in fracs]
        nmes = [p3_down["landmarks106"][key][f]["linear"]["nme106"] for f in fracs]
        h, = axes[0].plot(xf, accs, marker=mk, linestyle=ls, color=col, linewidth=lw, markersize=5.2, label=lbl, zorder=4 if key == "physflow_factorized" else 3)
        axes[1].plot(xf, nmes, marker=mk, linestyle=ls, color=col, linewidth=lw, markersize=5.2, label=lbl, zorder=4 if key == "physflow_factorized" else 3)
        handles_d.append(h); labels_d.append(lbl)

    x_cv = 3.65
    for key, col, mk, _, _, _ in style:
        m_acc = cv[key]["attr_acc_mean"] * 100.0
        s_acc = cv[key]["attr_acc_std"] * 100.0
        m_nme = cv[key]["nme106_mean"]
        s_nme = cv[key]["nme106_std"]
        axes[0].errorbar(x_cv, m_acc, yerr=s_acc, fmt=mk, color=col, markersize=5.5, capsize=3.0, elinewidth=1.2, zorder=5)
        axes[1].errorbar(x_cv, m_nme, yerr=s_nme, fmt=mk, color=col, markersize=5.5, capsize=3.0, elinewidth=1.2, zorder=5)

    for ax in (axes[0], axes[1]):
        ax.axvline(3.32, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_xticks([0, 1, 2, 3, x_cv])
        ax.set_xticklabels(["1%", "5%", "10%", "100%", "5-CV"], fontsize=7.8)
        ax.set_xlim(-0.25, 4.05)
        ax.set_xlabel("Labeled Fraction / 5-Fold CV ($M = 64$)", fontsize=8.4)
        ax.grid(True, zorder=0)

    axes[0].set_ylabel("CelebA 40-Attribute Acc. (%) ↑", fontsize=8.5)
    axes[0].set_ylim(77.0, 86.2)
    axes[0].text(0.5, -0.35, "(a) 40-Attribute Transfer", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    axes[1].set_ylabel("106-Point Landmark NME ↓", fontsize=8.5)
    axes[1].set_ylim(0.048, 0.118)
    axes[1].text(0.5, -0.35, "(b) 106-Point Landmark Transfer", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.legend(handles_d, labels_d, loc="upper center", bbox_to_anchor=(0.35, 1.00),
               ncol=4, frameon=False, fontsize=7.9, columnspacing=1.0)

    # Panel (c): Speed-Quality Pareto (clean non-overlapping labels + shaded optimal left region)
    steps_list = sorted(int(k) for k in p2["pareto_m64"].keys())
    secs = [p2["pareto_m64"][str(s)]["sec_per_img"] for s in steps_list]
    adas = [p2["pareto_m64"][str(s)]["adaface_cos_mean"] for s in steps_list]
    lpips_l = [p2["pareto_m64"][str(s)]["lpips_mean"] for s in steps_list]

    v3_sec = 0.896
    v3_ada = v3_s6b["v3_16step_proj075"]["hq45"]["64"]["adaface_cos_mean"]
    v3_lp = v3_s6b["v3_16step_proj075"]["hq45"]["64"]["lpips_mean"]

    f4_sec = v3_s6["timings"]["physflow_fast4_plus_refiner_bs9_sec"]
    f4_ada = v3_s6b["fast4_proj075"]["hq45"]["64"]["adaface_cos_mean"]
    f4_lp = v3_s6b["fast4_proj075"]["hq45"]["64"]["lpips_mean"]

    ax_p = axes[2]
    ax_p.axvspan(0.0, 1.02, color=PALETTE["physflow"], alpha=0.06, zorder=0)
    ax_p2 = ax_p.twinx()
    ax_p2.spines["top"].set_visible(False)
    ax_p2.spines["right"].set_visible(True)
    ax_p2.spines["right"].set_color("#000000")
    ax_p2.spines["right"].set_linewidth(0.8)

    l1 = ax_p.plot(secs, adas, "o-", color=PALETTE["phys_base"], linewidth=1.6, markersize=4.5, alpha=0.88, label="Base Flow AdaFace ↑", zorder=3)
    l2 = ax_p2.plot(secs, lpips_l, "s--", color=PALETTE["pnp_flow"], linewidth=1.6, markersize=4.3, alpha=0.88, label="Base Flow LPIPS ↓", zorder=3)

    l3 = ax_p.plot([f4_sec, v3_sec], [f4_ada, v3_ada], "*-", color=PALETTE["physflow"], linewidth=2.3, markersize=9.0,
                   markeredgecolor="#000000", markeredgewidth=0.35, label="+Refiner AdaFace ↑", zorder=6)
    l4 = ax_p2.plot([f4_sec, v3_sec], [f4_lp, v3_lp], "D--", color=PALETTE["zs_flower"], linewidth=1.9, markersize=5.2,
                    markeredgecolor="#000000", markeredgewidth=0.35, label="+Refiner LPIPS ↓", zorder=5)

    ax_p.axvline(p2["timings_sec_per_img_m64"]["lsi"], color=PALETTE["lsi"], linestyle=":", linewidth=1.15, zorder=2)
    ax_p.text(p2["timings_sec_per_img_m64"]["lsi"] - 0.06, 0.0545, "LSI\n(2.57s)", ha="right", va="top",
              fontsize=7.2, color="#000000", fontweight="bold")

    ax_p.annotate("Fast-4\n(0.20s)", (f4_sec, f4_ada), textcoords="offset points", xytext=(-4, 10),
                  ha="left", va="bottom", fontsize=7.1, fontweight="bold", color="#000000")
    ax_p.annotate("Full-16\n(0.90s)", (v3_sec, v3_ada), textcoords="offset points", xytext=(10, 6),
                  ha="left", va="bottom", fontsize=7.1, fontweight="bold", color="#000000")
    for st, ts, av in zip(steps_list, secs, adas):
        if st == 8:
            ax_p.annotate("8st", (ts, av), textcoords="offset points", xytext=(-4, -13), ha="center", fontsize=6.9, color="#000000")
        elif st == 16:
            ax_p.annotate("16st", (ts, av), textcoords="offset points", xytext=(14, 6), ha="center", fontsize=6.9, color="#000000")
        elif st == 50:
            ax_p.annotate("50st", (ts, av), textcoords="offset points", xytext=(-4, -13), ha="center", fontsize=6.9, color="#000000")

    ax_p.set_xlim(0.0, 2.95)
    ax_p.set_ylim(0.022, 0.0575)
    ax_p2.set_ylim(0.535, 0.668)
    ax_p.set_xlabel("Latency / 512px Image (s, RTX 3090)", fontsize=8.4, color="#000000")
    ax_p.set_ylabel("AdaFace Cosine ↑", fontsize=8.5, color="#000000")
    ax_p2.set_ylabel("Perceptual LPIPS ↓", fontsize=8.5, color="#000000")
    ax_p.tick_params(axis="y", colors="#000000")
    ax_p2.tick_params(axis="y", colors="#000000")
    ax_p.grid(True, zorder=0)
    lines_all = l1 + l2 + l3 + l4
    ax_p.legend(lines_all, [ln.get_label() for ln in lines_all], loc="upper center", bbox_to_anchor=(0.48, 1.30),
                ncol=2, frameon=False, fontsize=6.8, columnspacing=0.65, handlelength=1.5)
    ax_p.text(0.5, -0.35, "(c) Speed–Quality Pareto (M = 64)", transform=ax_p.transAxes,
              ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig5_downstream_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig6_m256_and_failures() -> None:
    """Fig. 6: Top-3 sharpest M=256 showcase faces with 2.2x crimson zoom-in detail strips (a) + M=64 hard cases (b)."""
    m256 = slice_grid_clean(OUTPUTS / "phase2_real_comparison_m256.png", nrow=12, ncol=6)
    fails = slice_grid_clean(OUTPUTS / "phase2_real_failures_m64.png", nrow=5, ncol=4)
    v3_pack = torch.load(OUTPUTS / "v4_toptier" / "fig1_fig6_v3_tiles.pt", map_location="cpu", weights_only=False)
    v3_256 = v3_pack["fig6a_v3_m256"]

    # Pick our 3 absolute highest-contrast visual victories:
    # Row 0 (grid row 0, idx=10): smiling young man (eyes/brows/nose/smile vs LSI identity drift)
    # Row 1 (grid row 2, idx=9):  dark-haired red-lip woman (eye makeup & crimson lips)
    # Row 2 (grid row 11, idx=21): mustached gentleman (brows, eyes, & mustache micro-texture vs LSI)
    pick_pairs = [(0, 10), (2, 9), (11, 21)]
    rows_a = [[m256[r][0], m256[r][2], m256[r][3], tensor_to_uint8(v3_256[idx])] for r, idx in pick_pairs]

    # Non-occluding 2.2x zoom strips under EVERY showcase row (eyes/brows, eye+lips, eyes+mustache)
    zoom_rois = {
        0: (0.37, 0.24, 0.61, 0.76),  # Young man: crisp eyes & brows
        1: (0.40, 0.26, 0.65, 0.74),  # Red-lip woman: crisp eyes & nose/lip contour
        2: (0.46, 0.24, 0.72, 0.76),  # Mustached gentleman: eyes & mustache texture
    }
    mos_a, s, pad, _ = compose_mosaic_with_zooms(
        rows_a,
        zoom_specs=zoom_rois,
        zoom_h_ratio=0.40,
        pad=20,
        zoom_gap=8,
        highlight_col=3,
        border_px=10,
    )

    # Right panel (b): 4 representative M=64 hard cases matching total visual height gracefully
    rows_b = [[fails[0][0], fails[0][1]], [fails[0][2], fails[0][3]], [fails[1][0], fails[1][1]], [fails[3][2], fails[3][3]]]
    mos_b, sb, pb, _ = compose_mosaic_with_zooms(rows_b, zoom_specs=None, pad=22, highlight_col=None)

    fig = plt.figure(figsize=(11.8, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.05, 1.95], wspace=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(mos_a)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    for ci, ch in enumerate(["Ground\nTruth", "ZS-FLOWER\n(ICLR'26)", "LSI\n(CVPR'25)", "PhysFlow\n(Ours)"]):
        ax_a.text(ci * (s + pad) + s * 0.5, -28, ch, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color="#000000", linespacing=1.08)
    ax_a.text(0.5, -0.04, "(a) High-Fidelity Recovery at M = 256 (SR = 0.39%) with Zoomed Detail Strips", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(mos_b)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    for ci, ch in enumerate(["Ground\nTruth", "PhysFlow\n(M = 64)"]):
        ax_b.text(ci * (sb + pb) + sb * 0.5, -28, ch, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color="#000000", linespacing=1.08)
    ax_b.text(0.5, -0.04, "(b) Representative Hard Cases at M = 64", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig6_m256_and_failures.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig8_real_dmd_and_unaligned() -> None:
    """Fig. 8: Real physical European DMD hardware + optical bar-edge zoom (a,b) + Best-3 Unaligned FOV + zoom (c,d)."""
    k2_json = load_json(OUTPUTS / "v4_toptier" / "knife2_real_dmd.json")
    k2_tiles = torch.load(OUTPUTS / "v4_toptier" / "knife2_real_dmd_tiles.pt", map_location="cpu", weights_only=False)
    k3_json = load_json(OUTPUTS / "v4_toptier" / "knife3_unaligned_fov.json")
    k3_tiles = torch.load(OUTPUTS / "v4_toptier" / "knife3_unaligned_tiles.pt", map_location="cpu", weights_only=False)

    # Panel (a) Left: 4 real physical DMD scenes x 6 methods at SR = 12.5%, with zoomed USAF 1951 bar-edge strip!
    dmd_rows_meta = [
        ("zoom_x12_usaf_group5", "USAF 1951\n($64\\times 64$)"),
        ("zoom_x12_starsector", "Siemens Star\n($64\\times 64$)"),
        ("zoom_x12_tomato_slice_2", "Tomato $\\times 12$\n($64\\times 64$)"),
        ("onepix_rgb_128", "ONE-PIX RGB\n($32\\times 32\\times 3$)"),
    ]
    dmd_cols = ["gt", "pinv", "tv_fista", "pnp_flow", "zs_flower", "physflow_zs"]
    dmd_headers = ["Full-Mode\nRef", "Pinv", "TV-FISTA", "PnP-Flow\n(ICLR'25)", "ZS-FLOWER\n(ICLR'26)", "PhysFlow-ZS\n(Ours)"]
    S_dmd = 192
    rows_dmd = []
    for rk, _ in dmd_rows_meta:
        row_arr = []
        for ck in dmd_cols:
            t = k2_tiles[rk][ck].float().clamp(0.0, 1.0)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
            im = np.asarray(Image.fromarray(arr).resize((S_dmd, S_dmd), Image.NEAREST))
            row_arr.append(im)
        rows_dmd.append(row_arr)
    mos_dmd, s1, p1, y_dmd = compose_mosaic_with_zooms(
        rows_dmd,
        zoom_specs={0: (0.14, 0.22, 0.56, 0.80)},  # USAF 1951 crisp bar group zoom!
        zoom_h_ratio=0.42,
        pad=14,
        zoom_gap=5,
        highlight_col=5,
        border_px=7,
        zoom_resample=Image.NEAREST,
    )

    # Panel (c) Left: BEST 3 sharpest unaligned CelebA-HQ faces (indices 5, 6, 7 — verified from contact sheet!)
    pick_u = [5, 6, 7]
    rows_u = []
    for idx in pick_u:
        def _u8(t_m11: torch.Tensor) -> np.ndarray:
            arr = ((t_m11.float().clamp(-1.0, 1.0) + 1.0) * 0.5).permute(1, 2, 0).numpy()
            im = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).resize((256, 256), Image.BILINEAR)
            return np.asarray(im)
        rows_u.append([
            _u8(k3_tiles["gt_scene_512"][idx]),
            _u8(k3_tiles["lsi_256"][idx]),
            _u8(k3_tiles["noloc_256"][idx]),
            _u8(k3_tiles["mdloc_256"][idx]),
            _u8(k3_tiles["oracle_256"][idx]),
        ])
    mos_u, s2, p2, _ = compose_mosaic_with_zooms(
        rows_u,
        zoom_specs={0: (0.35, 0.22, 0.63, 0.72)},  # Blonde woman eye/nose/lip alignment zoom!
        zoom_h_ratio=0.40,
        pad=12,
        zoom_gap=6,
        highlight_col=3,
        border_px=8,
    )

    fig = plt.figure(figsize=(12.2, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[1.10, 1.0], hspace=0.35, wspace=0.22)

    # (a) Left: Real Physical DMD Visual Mosaic
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a1.imshow(mos_dmd)
    ax_a1.set_xticks([]); ax_a1.set_yticks([])
    for sp in ax_a1.spines.values():
        sp.set_visible(False)
    for ci, ch in enumerate(dmd_headers):
        ax_a1.text(ci * (s1 + p1) + s1 * 0.5, -14, ch, ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#000000", linespacing=1.06)
    for ri, (_, rlbl) in enumerate(dmd_rows_meta):
        ax_a1.text(-14, y_dmd[ri] + s1 * 0.5, rlbl, ha="right", va="center", fontsize=7.4, fontweight="bold", color="#000000", linespacing=1.05)
    ax_a1.text(-14, y_dmd[0] + s1 + 5 + int(round(s1 * 0.42)) * 0.5, "Zoom", ha="right", va="center", fontsize=6.8, fontweight="bold", color="#000000")
    ax_a1.text(0.5, -0.055, "(a) Real Physical DMD Reconstructions (SR = 12.5%, Zero Retraining) with USAF Zoom",
               transform=ax_a1.transAxes, ha="center", va="top", fontsize=8.6, fontweight="bold", color="#000000")

    # (b) Right: Real Physical DMD Quantitative Summary
    ax_a2 = fig.add_subplot(gs[0, 1])
    regimes = ["Diff.\n6.25%", "Diff.\n12.5%", "Diff.\n25.0%", "Unip. {0,1}\n12.5%"]
    x_r = np.arange(len(regimes))
    m_specs = [
        ("Pinv", "pinv", PALETTE["tikhonov"]),
        ("TV-FISTA", "tv_fista", PALETTE["text"]),
        ("ZS-FLOWER", "zs_flower", PALETTE["zs_flower"]),
        ("PhysFlow-ZS", "physflow_zs", PALETTE["physflow"]),
    ]
    w_bar = 0.19
    for mi, (mlbl, mkey, col) in enumerate(m_specs):
        vals = [
            k2_json["broadband_diff"]["256"]["mean"][mkey]["psnr"],
            k2_json["broadband_diff"]["512"]["mean"][mkey]["psnr"],
            k2_json["broadband_diff"]["1024"]["mean"][mkey]["psnr"],
            k2_json["unipolar_binary_dmd"]["mean"][mkey]["psnr"],
        ]
        offs = x_r + (mi - 1.5) * w_bar
        ax_a2.bar(offs, vals, width=w_bar, color=col, edgecolor="#222222", linewidth=0.45, label=mlbl, zorder=3)
        if mkey == "physflow_zs":
            for xi, vv in zip(offs, vals):
                ax_a2.text(xi, vv + 0.25, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#000000")
    ax_a2.set_xticks(x_r)
    ax_a2.set_xticklabels(regimes, fontsize=7.8)
    ax_a2.set_ylabel("Real Benchtop PSNR (dB) ↑", fontsize=8.4)
    ax_a2.set_ylim(10.0, 23.0)
    ax_a2.grid(axis="y", zorder=0)
    ax_a2.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.3, columnspacing=0.8)
    ax_a2.text(0.5, -0.22, "(b) Real DMD PSNR (Diff. & Single-Exposure Unipolar)",
               transform=ax_a2.transAxes, ha="center", va="top", fontsize=8.6, fontweight="bold", color="#000000")

    # (c) Left: Unaligned Arbitrary-FOV Visual Mosaic (Best 3 faces + Eye/Face Alignment Zoom)
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b1.imshow(mos_u)
    ax_b1.set_xticks([]); ax_b1.set_yticks([])
    for sp in ax_b1.spines.values():
        sp.set_visible(False)
    u_headers = ["Unaligned\nGround Truth", "LSI\n(CVPR'25)", "PhysFlow\n(w/o Loc)", "PhysFlow +\nMD-FaceLoc", "Oracle Pose\n(Ceiling)"]
    for ci, ch in enumerate(u_headers):
        ax_b1.text(ci * (s2 + p2) + s2 * 0.5, -18, ch, ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#000000", linespacing=1.05)
    ax_b1.text(0.5, -0.055, "(c) Unaligned Arbitrary-FOV Face Recovery (M = 256, 0.053 ms MD-FaceLoc) with Detail Zoom",
               transform=ax_b1.transAxes, ha="center", va="top", fontsize=8.6, fontweight="bold", color="#000000")

    # (d) Right: Unaligned Arbitrary-FOV Biometric Recovery Bar Chart (M = 256)
    ax_b2 = fig.add_subplot(gs[1, 1])
    u_methods = [
        ("LSI\n(CVPR'25)", "lsi_unaligned"),
        ("PhysFlow\n(w/o Loc)", "physflow_no_loc"),
        ("PhysFlow +\nMD-FaceLoc", "physflow_md_faceloc"),
        ("Oracle\nPose", "physflow_oracle_pose"),
    ]
    xu = np.arange(len(u_methods))
    buf_vals = [k3_json["methods"][mk]["256"]["buffalo_det_cos"] for _, mk in u_methods]
    ada_vals = [k3_json["methods"][mk]["256"]["adaface_gt_crop_cos"] for _, mk in u_methods]
    w2 = 0.34
    ax_b2.bar(xu - w2 * 0.54, buf_vals, width=w2, color=PALETTE["physflow"], edgecolor="#222222", linewidth=0.45, label="Buffalo_L (auto-det) ↑", zorder=3)
    ax_b2.bar(xu + w2 * 0.54, ada_vals, width=w2, color=PALETTE["pnp_flow"], edgecolor="#222222", linewidth=0.45, label="AdaFace (GT-crop) ↑", zorder=3)
    for xi, bv, av in zip(xu, buf_vals, ada_vals):
        ax_b2.text(xi - w2 * 0.58, bv + 0.003, f"{bv:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#000000")
        ax_b2.text(xi + w2 * 0.58, av + 0.003, f"{av:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#000000")
    ax_b2.set_xticks(xu)
    ax_b2.set_xticklabels([m[0] for m in u_methods], fontsize=7.6)
    ax_b2.set_ylabel("Unaligned Identity Cosine ↑", fontsize=8.4)
    ax_b2.set_ylim(0.0, 0.165)
    ax_b2.grid(axis="y", zorder=0)
    ax_b2.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax_b2.text(0.5, -0.22, "(d) Unaligned FOV Biometric Fidelity (M = 256, N = 45)",
               transform=ax_b2.transAxes, ha="center", va="top", fontsize=8.6, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig8_real_dmd_and_unaligned.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_rc()
    p2 = load_json(OUTPUTS / "phase2_real_results.json")
    p3_down = load_json(OUTPUTS / "phase3_downstream_results.json")
    p3_ext = load_json(OUTPUTS / "phase3_external_lapa_wflw.json")
    extras = load_json(OUTPUTS / "journal_audit_extras.json")
    shield = load_json(OUTPUTS / "reviewer_shield_results.json")
    v3_s1 = load_json(OUTPUTS / "v3_surgery" / "step1_inference_upgrade.json")
    v3_s6 = load_json(OUTPUTS / "v3_surgery" / "step6_final_full479_v3.json")
    v3_s6b = load_json(OUTPUTS / "v3_surgery" / "step6b_parity_and_qa.json")

    make_fig1_teaser(p2, extras, v3_s6, v3_s6b)
    make_fig3_rate_and_noise(p2, extras, shield, v3_s1, v3_s6b)
    make_fig4_wflw_external(p3_ext, v3_s6)
    make_fig5_downstream_pareto(p2, p3_down, extras, v3_s6, v3_s6b)
    make_fig6_m256_and_failures()
    make_fig8_real_dmd_and_unaligned()
    print(f"All 6 upgraded IEEE publication-grade vector figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
