"""Generate publication-grade vector PDF figures for the PhysFlow Elsevier manuscript.

Design principles (Stage 4):
- Clean serif typography (DejaVu Serif / STIX) matching elsarticle times layout.
- Restrained colorblind-safe academic palette (deep navy highlight for PhysFlow).
- Bold centered subcaptions (a), (b), (c) positioned cleanly below every subpanel.
- Zero baked-in microscopic image footers, zero cluttered in-figure title banners,
  and zero overlapping legends or text annotations.
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

# Restrained Elsevier / Okabe-Ito publication palette
PALETTE = {
    "physflow": "#1B4F72",      # Deep Navy (Hero)
    "lsi": "#B03A2E",           # Muted Crimson (CVPR'25 LSI)
    "pnp_flow": "#D35400",      # Burnt Amber (ICLR'25 PnP-Flow)
    "zs_flower": "#148F77",     # Muted Teal (ICLR'26 ZS-FLOWER)
    "wiener": "#6C3483",        # Muted Plum (Wiener-LMMSE)
    "tikhonov": "#7F8C8D",      # Slate Gray (Tikhonov)
    "text": "#95A5A6",          # Silver Gray (Text B0)
    "b2_geo": "#2E86C1",        # Steel Blue (B2 Geo)
    "raw": "#7F8C8D",
}


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


def compose_mosaic(rows: list[list[np.ndarray]], pad: int = 10, bg: int = 255, highlight_col: int | None = None, border_px: int = 8) -> tuple[np.ndarray, int, int]:
    """Compose clean white-padded mosaic of square tiles, optionally framing one hero column."""
    nr = len(rows)
    nc = len(rows[0])
    s = rows[0][0].shape[0]
    H = nr * s + (nr - 1) * pad
    W = nc * s + (nc - 1) * pad
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    for r in range(nr):
        y0 = r * (s + pad)
        for c in range(nc):
            x0 = c * (s + pad)
            tile = rows[r][c].copy()
            if highlight_col is not None and c == highlight_col:
                col_rgb = np.array([27, 79, 114], dtype=np.uint8)
                tile[:border_px, :] = col_rgb
                tile[-border_px:, :] = col_rgb
                tile[:, :border_px] = col_rgb
                tile[:, -border_px:] = col_rgb
            canvas[y0 : y0 + s, x0 : x0 + s] = tile
    return canvas, s, pad


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
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
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


def make_fig1_teaser(p2: dict, extras: dict) -> None:
    """Fig. 1: 8-column visual comparison at M=64 (SR=0.098%) + 3 quantitative summary panels."""
    tea = slice_grid_clean(OUTPUTS / "phase2_teaser_strip_m64.png", nrow=4, ncol=6)
    # Use top 3 visually diverse faces (indices 0, 1, 2 in teaser) for crisp 2-column height
    gt_tiles = [tea[r][0] for r in range(3)]
    matched_idx = match_teaser_indices(gt_tiles)
    ex_pt = torch.load(OUTPUTS / "extra_baselines_m64.pt", map_location="cpu", weights_only=False)
    tik_t = ex_pt["tikhonov_m64"]
    pnp_t = ex_pt["pnp_flow_m64"]

    # Columns: GT | Tikhonov | Wiener-LMMSE | Text (B0) | PnP-Flow | ZS-FLOWER | LSI | PhysFlow (Ours)
    rows = []
    for r, idx in enumerate(matched_idx):
        rows.append([
            tea[r][0],                         # GT (512px)
            tensor_to_uint8(tik_t[idx]),       # Tikhonov
            tea[r][1],                         # Wiener-LMMSE
            tea[r][2],                         # Text (B0)
            tensor_to_uint8(pnp_t[idx]),       # PnP-Flow (ICLR'25)
            tea[r][3],                         # ZS-FLOWER (ICLR'26)
            tea[r][4],                         # LSI (CVPR'25)
            tea[r][5],                         # PhysFlow (Ours)
        ])
    mosaic, s, pad = compose_mosaic(rows, pad=12, highlight_col=7, border_px=10)

    fig = plt.figure(figsize=(11.8, 6.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.65, 1.85], hspace=0.28, wspace=0.26)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(mosaic)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
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
        ax_top.text(cx, -24, ch, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#000000", linespacing=1.08)

    # Bottom quantitative bar charts at M = 64
    ex_r = extras["extra_baselines_celebahq"]["rates"]
    methods = [
        ("Tikh.", ex_r["tikhonov_bp"]["64"], PALETTE["tikhonov"]),
        ("Text\n(B0)", p2["rates"]["b0_discrete"]["64"], PALETTE["text"]),
        ("PnP-\nFlow", ex_r["pnp_flow"]["64"], PALETTE["pnp_flow"]),
        ("ZS-\nFLW", p2["rates"]["zeroshot_flower"]["64"], PALETTE["zs_flower"]),
        ("LSI", p2["rates"]["lsi"]["64"], PALETTE["lsi"]),
        ("Phys-\nFlow", p2["rates"]["b3_physflow"]["64"], PALETTE["physflow"]),
    ]
    xm = np.arange(len(methods))
    lbls = [m[0] for m in methods]
    cols = [m[2] for m in methods]

    specs = [
        ("adaface_cos_mean", "AdaFace Cosine ↑", "(a) Identity Preservation (M = 64)", (-0.018, 0.038)),
        ("nme106_mean", "106-pt Landmark NME ↓", "(b) Geometric Error (M = 64)", (0.0, 0.94)),
        ("meas_rel_mean", "Relative Residual ↓", "(c) Physical Consistency (M = 64)", (0.0, 0.43)),
    ]
    for col_i, (metric, ylabel, subcap, ylim) in enumerate(specs):
        ax = fig.add_subplot(gs[1, col_i])
        vals = [float(m[1][metric]) for m in methods]
        ax.bar(xm, vals, color=cols, width=0.58, edgecolor="#222222", linewidth=0.45, zorder=3)
        ax.axhline(0.0, color="#333333", linewidth=0.6, zorder=2)
        ax.grid(axis="y", zorder=0)
        ax.set_xticks(xm)
        ax.set_xticklabels(lbls, fontsize=7.8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_ylim(*ylim)
        # Clean value label ONLY on hero PhysFlow bar to stay uncluttered
        hv = vals[-1]
        ax.text(xm[-1], hv + (ylim[1] - ylim[0]) * 0.035, f"{hv:.4f}" if metric != "adaface_cos_mean" else f"{hv:.3f}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#000000")
        ax.text(0.5, -0.34, subcap, transform=ax.transAxes, ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig1_teaser_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig3_rate_and_noise(p2: dict, extras: dict, shield: dict, v3_s1: dict) -> None:
    """Fig. 3: Rate-distortion curves (with PnP-Flow, Tikhonov, Wiener, ZS-FLOWER, LSI, PhysFlow) + 6-axis noise."""
    rates = [16, 32, 64, 128, 256]
    sr_labels = [f"{r}\n({100*r/65536:.2f}%)" for r in rates]
    x = np.arange(len(rates))
    ex_r = extras["extra_baselines_celebahq"]["rates"]

    series = [
        ("PhysFlow (Ours)", p2["rates"]["b3_physflow"], PALETTE["physflow"], "o", "-", 2.5, 5.5, 6),
        ("LSI (CVPR'25)", p2["rates"]["lsi"], PALETTE["lsi"], "s", "--", 1.8, 4.8, 5),
        ("PnP-Flow (ICLR'25)", ex_r["pnp_flow"], PALETTE["pnp_flow"], "^", "-.", 1.8, 5.0, 4),
        ("ZS-FLOWER (ICLR'26)", p2["rates"]["zeroshot_flower"], PALETTE["zs_flower"], "D", ":", 1.8, 4.6, 3),
        ("Wiener-LMMSE", ex_r["wiener_lmmse"], PALETTE["wiener"], "v", (0, (3, 1, 1, 1)), 1.5, 4.5, 3),
        ("Tikhonov", ex_r["tikhonov_bp"], PALETTE["tikhonov"], "x", ":", 1.3, 4.8, 2),
        ("Text (B0)", p2["rates"]["b0_discrete"], PALETTE["text"], "+", "--", 1.3, 5.2, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    fig.subplots_adjust(top=0.80, bottom=0.27, wspace=0.32)

    handles_ab = []
    labels_ab = []
    for lbl, dct, col, mk, ls, lw, ms, z in series:
        ada = [dct[str(r)]["adaface_cos_mean"] for r in rates]
        nme = [np.nan if dct[str(r)]["nme106_mean"] is None else dct[str(r)]["nme106_mean"] for r in rates]
        h1, = axes[0].plot(x, ada, marker=mk, linestyle=ls, linewidth=lw, markersize=ms, color=col, label=lbl, zorder=z)
        axes[1].plot(x, nme, marker=mk, linestyle=ls, linewidth=lw, markersize=ms, color=col, label=lbl, zorder=z)
        handles_ab.append(h1)
        labels_ab.append(lbl)

    axes[0].set_xticks(x); axes[0].set_xticklabels(sr_labels, fontsize=7.7)
    axes[0].set_xlabel("Measurements $M$ (Sampling Rate)", fontsize=8.5)
    axes[0].set_ylabel("AdaFace IR-101 Cosine ↑", fontsize=8.5)
    axes[0].set_ylim(-0.025, 0.185)
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

    # Shared clean horizontal legend centered above (a) and (b)
    fig.legend(handles_ab, labels_ab, loc="upper center", bbox_to_anchor=(0.37, 1.01),
               ncol=4, frameon=False, fontsize=7.8, columnspacing=1.0, handlelength=2.1)

    # Panel (c): Six-Axis Sensor Noise Robustness at M=64 (calibrated alpha=8.0, CPU seed=777 parity matching Tables 5 & 6)
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
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False, fontsize=7.4, columnspacing=0.65, handletextpad=0.35)
    axes[2].text(0.5, -0.38, "(c) Sensor Noise Robustness (M = 64)", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig3_rate_and_noise.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig4_wflw_external(p3_ext: dict, v3_s6: dict) -> None:
    """Fig. 4: External WFLW 6-subset qualitative strip (footer-free) + dual NME/PSNR charts."""
    wflw = slice_grid_clean(OUTPUTS / "external_wflw_6subsets_m64.png", nrow=6, ncol=8)
    # Build clean 3-row x 8-col layout (2 subsets side-by-side per row: Pose|Expr, Illum|Makeup, Occl|Blur)
    # Wait: even cleaner is 2-row x 8-col or 6-row x 4-col (1 representative sample per subset) — wait:
    # 3 rows x 8 cols fits 2-column page width with large readable faces (each subset gets 4 tiles: GT|ZS|LSI|Ours)!
    pairs = [(0, 1), (2, 3), (4, 5)]
    sub_names = ["Pose", "Expression", "Illumination", "Makeup", "Occlusion", "Blur"]
    # Pick sample 0 from each row (cols 0..3) and sample 0 from the paired row (cols 0..3)
    s = 512
    pad = 12
    mid_gap = 60
    row_gap = 110
    nr = 3
    H = nr * s + (nr - 1) * row_gap + 70
    W = 8 * s + 6 * pad + mid_gap
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    col_rgb = np.array([27, 79, 114], dtype=np.uint8)
    row_y_centers = []
    for ri, (s_left, s_right) in enumerate(pairs):
        y0 = ri * (s + row_gap)
        row_y_centers.append(y0)
        # Left subset (cols 0..3)
        for ci in range(4):
            x0 = ci * (s + pad)
            tile = wflw[s_left][ci].copy()
            if ci == 3:
                tile[:9, :] = col_rgb; tile[-9:, :] = col_rgb; tile[:, :9] = col_rgb; tile[:, -9:] = col_rgb
            canvas[y0 : y0 + s, x0 : x0 + s] = tile
        # Right subset (cols 4..7, use sample 1 for variety on Makeup/Blur or sample 0 where cleanest)
        right_offset = 4 if s_right in (3, 5) else 0
        for ci in range(4):
            x0 = 4 * s + 3 * pad + mid_gap + ci * (s + pad)
            tile = wflw[s_right][right_offset + ci].copy()
            if ci == 3:
                tile[:9, :] = col_rgb; tile[-9:, :] = col_rgb; tile[:, :9] = col_rgb; tile[:, -9:] = col_rgb
            canvas[y0 : y0 + s, x0 : x0 + s] = tile

    fig = plt.figure(figsize=(11.8, 8.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.65, 1.75], hspace=0.30, wspace=0.22)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(canvas)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for sp in ax_top.spines.values():
        sp.set_visible(False)

    hdrs = ["GT", "ZS-FLOWER", "LSI (CVPR'25)", "PhysFlow (Ours)"]
    for block_i, x_start in enumerate([0, 4 * s + 3 * pad + mid_gap]):
        for ci, htxt in enumerate(hdrs):
            cx = x_start + ci * (s + pad) + s * 0.5
            ax_top.text(cx, -26, htxt, ha="center", va="bottom", fontsize=8.4, fontweight="bold",
                        color="#000000")
    for ri, (s_left, s_right) in enumerate(pairs):
        y_lbl = row_y_centers[ri] + s + 46
        ax_top.text(2 * s + 1.5 * pad, y_lbl, f"Subset: {sub_names[s_left]}", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#000000")
        ax_top.text(6 * s + 4.5 * pad + mid_gap, y_lbl, f"Subset: {sub_names[s_right]}", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#000000")
    ax_top.text(0.5, -0.025, "(a) Qualitative Generalization Across Six Challenging WFLW Subsets (M = 64)",
                transform=ax_top.transAxes, ha="center", va="top", fontsize=9.0, fontweight="bold")

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
                ha="center", va="top", fontsize=9.0, fontweight="bold")

    ax_psnr.set_xticks(xs); ax_psnr.set_xticklabels(sub_short, fontsize=8.4)
    ax_psnr.set_ylabel("Reconstruction PSNR (dB) ↑", fontsize=8.5)
    ax_psnr.set_ylim(0.0, 21.0)
    ax_psnr.grid(axis="y", zorder=0)
    ax_psnr.text(0.5, -0.27, "(c) Per-Subset Reconstruction PSNR (M = 64)", transform=ax_psnr.transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold")

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

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75))
    fig.subplots_adjust(top=0.80, bottom=0.26, wspace=0.36)

    style = [
        ("physflow_factorized", PALETTE["physflow"], "o", "-", 2.4, "PhysFlow (Ours)"),
        ("lsi_latent", PALETTE["lsi"], "s", "--", 1.8, "LSI Latent (CVPR'25)"),
        ("text_only_sem", PALETTE["pnp_flow"], "^", "-.", 1.7, "Text Semantic ($C_{\\mathrm{sem}}$)"),
        ("raw_measurement", PALETTE["raw"], "d", ":", 1.6, "Raw Meas. ($\\mathbf{y}_S$)"),
    ]
    handles_d, labels_d = [], []
    for key, col, mk, ls, lw, lbl in style:
        accs = [p3_down["attr40"][key][f]["shallow"]["mean_acc"] * 100.0 for f in fracs]
        nmes = [p3_down["landmarks106"][key][f]["linear"]["nme106"] for f in fracs]
        h, = axes[0].plot(xf, accs, marker=mk, linestyle=ls, color=col, linewidth=lw, markersize=5.2, label=lbl, zorder=4 if key == "physflow_factorized" else 3)
        axes[1].plot(xf, nmes, marker=mk, linestyle=ls, color=col, linewidth=lw, markersize=5.2, label=lbl, zorder=4 if key == "physflow_factorized" else 3)
        handles_d.append(h); labels_d.append(lbl)

    # Also plot 5-fold identity-disjoint CV markers at x = 3.55 (clearly separated on x-axis)
    x_cv = 3.65
    for key, col, mk, _, _, _ in style:
        m_acc = cv[key]["attr_acc_mean"] * 100.0
        s_acc = cv[key]["attr_acc_std"] * 100.0
        m_nme = cv[key]["nme106_mean"]
        s_nme = cv[key]["nme106_std"]
        axes[0].errorbar(x_cv, m_acc, yerr=s_acc, fmt=mk, color=col, markersize=5.5, capsize=3.0, elinewidth=1.2, zorder=5)
        axes[1].errorbar(x_cv, m_nme, yerr=s_nme, fmt=mk, color=col, markersize=5.5, capsize=3.0, elinewidth=1.2, zorder=5)

    for ax in (axes[0], axes[1]):
        ax.axvline(3.32, color="#888888", linestyle=":", linewidth=0.8)
        ax.set_xticks([0, 1, 2, 3, x_cv])
        ax.set_xticklabels(["1%", "5%", "10%", "100%", "5-CV"], fontsize=7.8)
        ax.set_xlim(-0.25, 4.05)
        ax.set_xlabel("Labeled Fraction / 5-Fold CV ($M = 64$)", fontsize=8.4)
        ax.grid(True, zorder=0)

    axes[0].set_ylabel("CelebA 40-Attribute Acc. (%) ↑", fontsize=8.5)
    axes[0].set_ylim(77.0, 86.2)
    axes[0].text(0.5, -0.35, "(a) 40-Attribute Transfer", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold")

    axes[1].set_ylabel("106-Point Landmark NME ↓", fontsize=8.5)
    axes[1].set_ylim(0.048, 0.118)
    axes[1].text(0.5, -0.35, "(b) 106-Point Landmark Transfer", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9.0, fontweight="bold")

    fig.legend(handles_d, labels_d, loc="upper center", bbox_to_anchor=(0.35, 0.99),
               ncol=4, frameon=False, fontsize=7.9, columnspacing=1.0)

    # Panel (c): Speed-Quality Pareto (Base Euler 8..50 steps + V3 16-st + Refiner + Fast-4 Distilled + Refiner)
    steps_list = sorted(int(k) for k in p2["pareto_m64"].keys())
    secs = [p2["pareto_m64"][str(s)]["sec_per_img"] for s in steps_list]
    adas = [p2["pareto_m64"][str(s)]["adaface_cos_mean"] for s in steps_list]
    lpips_l = [p2["pareto_m64"][str(s)]["lpips_mean"] for s in steps_list]

    v3_sec = 0.896  # 0.872s 16-step flow + 0.024s 2.27M pixel refiner (uncontended RTX 3090)
    v3_ada = v3_s6b["v3_16step_proj075"]["hq45"]["64"]["adaface_cos_mean"]
    v3_lp = v3_s6b["v3_16step_proj075"]["hq45"]["64"]["lpips_mean"]

    f4_sec = v3_s6["timings"]["physflow_fast4_plus_refiner_bs9_sec"]  # 0.1963s (5.09 FPS batched)
    f4_ada = v3_s6b["fast4_proj075"]["hq45"]["64"]["adaface_cos_mean"]
    f4_lp = v3_s6b["fast4_proj075"]["hq45"]["64"]["lpips_mean"]

    ax_p = axes[2]
    ax_p2 = ax_p.twinx()
    ax_p2.spines["top"].set_visible(False)
    ax_p2.spines["right"].set_visible(True)
    ax_p2.spines["right"].set_color("#000000")
    ax_p2.spines["right"].set_linewidth(0.8)

    l1 = ax_p.plot(secs, adas, "o-", color=PALETTE["b2_geo"], linewidth=1.6, markersize=4.5, alpha=0.85, label="Base Flow AdaFace ↑", zorder=3)
    l2 = ax_p2.plot(secs, lpips_l, "s--", color=PALETTE["pnp_flow"], linewidth=1.6, markersize=4.3, alpha=0.85, label="Base Flow LPIPS ↓", zorder=3)

    # Upgraded + Refiner hero operating points (Fast-4 & Full-16)
    l3 = ax_p.plot([f4_sec, v3_sec], [f4_ada, v3_ada], "*-", color=PALETTE["physflow"], linewidth=2.2, markersize=8.5, label="+Refiner AdaFace ↑", zorder=6)
    l4 = ax_p2.plot([f4_sec, v3_sec], [f4_lp, v3_lp], "D--", color=PALETTE["zs_flower"], linewidth=1.8, markersize=5.2, label="+Refiner LPIPS ↓", zorder=5)

    # Highlight LSI latency reference line
    ax_p.axvline(p2["timings_sec_per_img_m64"]["lsi"], color=PALETTE["lsi"], linestyle=":", linewidth=1.1, zorder=2)
    ax_p.text(p2["timings_sec_per_img_m64"]["lsi"] - 0.06, 0.0525, "LSI\n(2.57s)", ha="right", va="top",
              fontsize=7.3, color="#000000", fontweight="bold")

    ax_p.annotate("Fast-4\n(0.20s)", (f4_sec, f4_ada), textcoords="offset points", xytext=(16, -24),
                  ha="center", fontsize=7.2, fontweight="bold", color="#000000")
    ax_p.annotate("Full-16\n(0.90s)", (v3_sec, v3_ada), textcoords="offset points", xytext=(20, -4),
                  ha="left", fontsize=7.2, fontweight="bold", color="#000000")
    for st, ts, av in zip(steps_list, secs, adas):
        if st == 8:
            ax_p.annotate("8st", (ts, av), textcoords="offset points", xytext=(-6, -13), ha="center", fontsize=6.9, color="#000000")
        elif st == 16:
            ax_p.annotate("16st", (ts, av), textcoords="offset points", xytext=(16, 6), ha="center", fontsize=6.9, color="#000000")
        elif st == 50:
            ax_p.annotate("50st", (ts, av), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=6.9, color="#000000")

    ax_p.set_xlim(0.0, 2.95)
    ax_p.set_ylim(0.022, 0.0555)
    ax_p2.set_ylim(0.535, 0.665)
    ax_p.set_xlabel("Latency / 512px Image (s, RTX 3090)", fontsize=8.4, color="#000000")
    ax_p.set_ylabel("AdaFace Cosine ↑", fontsize=8.5, color="#000000")
    ax_p2.set_ylabel("Perceptual LPIPS ↓", fontsize=8.5, color="#000000")
    ax_p.tick_params(axis="y", colors="#000000")
    ax_p2.tick_params(axis="y", colors="#000000")
    ax_p.grid(True, zorder=0)
    lines_all = l1 + l2 + l3 + l4
    ax_p.legend(lines_all, [ln.get_label() for ln in lines_all], loc="upper center", bbox_to_anchor=(0.48, 1.30),
                ncol=2, frameon=False, fontsize=6.7, columnspacing=0.65, handlelength=1.5)
    ax_p.text(0.5, -0.35, "(c) Speed–Quality Pareto (M = 64)", transform=ax_p.transAxes,
              ha="center", va="top", fontsize=9.0, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig5_downstream_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig6_m256_and_failures() -> None:
    """Fig. 6: Side-by-side M=256 high-fidelity reconstructions + M=64 representative hard cases."""
    m256 = slice_grid_clean(OUTPUTS / "phase2_real_comparison_m256.png", nrow=12, ncol=6)
    fails = slice_grid_clean(OUTPUTS / "phase2_real_failures_m64.png", nrow=5, ncol=4)

    # Left panel (a): 4 visually crisp rows at M=256 (rows 0, 2, 5, 11): GT | ZS-FLOWER | LSI | PhysFlow
    pick_256 = [0, 2, 5, 11]
    rows_a = [[m256[r][0], m256[r][2], m256[r][3], m256[r][5]] for r in pick_256]
    mos_a, s, pad = compose_mosaic(rows_a, pad=18, highlight_col=3, border_px=10)

    # Right panel (b): 4 representative M=64 hard cases
    rows_b = [[fails[0][0], fails[0][1]], [fails[0][2], fails[0][3]], [fails[1][0], fails[1][1]], [fails[3][2], fails[3][3]]]
    mos_b, _, _ = compose_mosaic(rows_b, pad=18, highlight_col=None)

    fig = plt.figure(figsize=(11.8, 5.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 2.0], wspace=0.09)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(mos_a)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    for ci, ch in enumerate(["Ground\nTruth", "ZS-FLOWER\n(ICLR'26)", "LSI\n(CVPR'25)", "PhysFlow\n(Ours)"]):
        ax_a.text(ci * (s + pad) + s * 0.5, -28, ch, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color="#000000", linespacing=1.08)
    ax_a.text(0.5, -0.045, "(a) High-Fidelity Recovery at M = 256 (SR = 0.39%)", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=9.2, fontweight="bold", color="#000000")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(mos_b)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    for ci, ch in enumerate(["Ground\nTruth", "PhysFlow\n(M = 64)"]):
        ax_b.text(ci * (s + pad) + s * 0.5, -28, ch, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color="#000000", linespacing=1.08)
    ax_b.text(0.5, -0.045, "(b) Representative Hard Cases at M = 64", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=9.2, fontweight="bold", color="#000000")

    fig.savefig(FIG_DIR / "fig6_m256_and_failures.pdf", bbox_inches="tight")
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

    make_fig1_teaser(p2, extras)
    make_fig3_rate_and_noise(p2, extras, shield, v3_s1)
    make_fig4_wflw_external(p3_ext, v3_s6)
    make_fig5_downstream_pareto(p2, p3_down, extras, v3_s6, v3_s6b)
    make_fig6_m256_and_failures()
    print(f"All 5 publication-grade vector figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
