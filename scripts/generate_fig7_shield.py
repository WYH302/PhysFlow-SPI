"""Generate Figure 7 (Publication-Grade 3-Panel Vector PDF):
  (a) Blind Spectral Noise Estimator (Eqs. 13-14) Ablation + Auto-Scaled lambda_adapt across N0-N4
  (b) Warm-Start t0 & CFG Guidance w_cfg Sensitivity Frontier (M=64)
  (c) Hardware-Agnostic Optical DMD Mask Generalization (LSI, 2D Walsh-Hadamard {0,1}, Bernoulli {0,1})
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "outputs" / "figures" / "fig7_shield_ablations.pdf"
PREVIEW_PNG = ROOT / "outputs" / "figures" / "fig7_shield_ablations.png"


def main() -> None:
    d = json.loads((ROOT / "outputs" / "reviewer_shield_results.json").read_text(encoding="utf-8"))
    v3_s1 = json.loads((ROOT / "outputs" / "v3_surgery" / "step1_inference_upgrade.json").read_text(encoding="utf-8"))
    v3_s6 = json.loads((ROOT / "outputs" / "v3_surgery" / "step6_final_full479_v3.json").read_text(encoding="utf-8"))
    ma = d["module_a_blind_spectral_ablation"]
    mb = d["module_b_sensitivity_and_uncertainty"]
    md = d["module_d_hardware_masks"]
    me = d["module_e_full479_fid"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "font.size": 8.2,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Exact unified Okabe-Ito / Academic palette matching all figures
    C_PHYS = "#1B4F72"   # Deep Navy (PhysFlow Ours)
    C_RED = "#B03A2E"    # Muted Crimson (Fixed small / LSI)
    C_GREEN = "#148F77"  # Muted Teal (Fixed large / ZS-FLOWER)
    C_ORG = "#D35400"    # Burnt Amber (lambda_adapt / PnP-Flow)
    C_PURP = "#6C3483"   # Muted Plum (Wiener-LMMSE / GCV)
    C_GRAY = "#7F8C8D"   # Slate Gray (TV-FISTA)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.30), gridspec_kw={"width_ratios": [1.22, 1.0, 1.15], "wspace": 0.44})

    # ------------------------------------------------------------------
    # (a) Blind Spectral Estimator Ablation (incl. Pure GCV & Calibrated alpha=8.0) + lambda_adapt Telemetry
    # ------------------------------------------------------------------
    ax = axes[0]
    modes = ["clean", "gaussian", "poisson_gaussian", "gain_bias", "dropout", "combined"]
    xlabels = ["Clean\n(N0)", "Gauss\n(N1)", "Pois+G\n(N2)", "Gain+\nBias", "Drop\n(N3b)", "Comb.\n(N4)"]
    x = np.arange(len(modes))
    w = 0.195

    p_fix_s = [ma["variants"]["fixed_small_4e4"][m]["psnr_mean"] for m in modes]
    p_fix_l = [ma["variants"]["fixed_large_5e2"][m]["psnr_mean"] for m in modes]
    p_gcv = [v3_s1["part1c_hq45_noise_and_cb45"]["hq45_noise"]["pure_gcv_t0.35"][m]["psnr_mean"] for m in modes]
    p_blind = [v3_s1["part1c_hq45_noise_and_cb45"]["hq45_noise"]["tail_8.0_t0.35"][m]["psnr_mean"] for m in modes]
    lams = [v3_s1["part1c_hq45_noise_and_cb45"]["hq45_noise"]["tail_8.0_t0.35"][m]["lam_mean"] for m in modes]

    ax.bar(x - 1.5 * w, p_fix_s, width=w, color=C_RED, edgecolor="white", linewidth=0.35, label=r"Fixed $\lambda=4{\times}10^{-4}$")
    ax.bar(x - 0.5 * w, p_gcv, width=w, color=C_PURP, edgecolor="white", linewidth=0.35, label="Pure Spectral GCV")
    ax.bar(x + 0.5 * w, p_fix_l, width=w, color=C_GREEN, edgecolor="white", linewidth=0.35, label=r"Fixed $\lambda=0.05$")
    ax.bar(x + 1.5 * w, p_blind, width=w, color=C_PHYS, edgecolor="white", linewidth=0.35, label="Calib. " + r"$\hat{\lambda}_{\mathrm{adapt}}$ (Ours)")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Reconstruction PSNR (dB) ↑", color="#000000")
    ax.set_ylim(4.5, 20.8)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, lams, color=C_ORG, marker="D", markersize=4.4, linewidth=1.55, linestyle="-", label=r"Est. $\hat{\lambda}_{\mathrm{adapt}}$ (right, log)")
    ax2.set_yscale("log")
    ax2.set_ylim(1.5e-4, 1.2)
    ax2.set_ylabel(r"Estimated $\hat{\lambda}_{\mathrm{adapt}}$ (log scale)", color="#000000")
    ax2.tick_params(axis="y", colors="#000000")
    ax2.spines["top"].set_visible(False)

    # Annotate 595x automatic jump on Gain+Bias
    ax2.annotate(
        r"$595{\times}$",
        xy=(3, lams[3]),
        xytext=(2.10, 0.52),
        fontsize=7.2,
        fontweight="bold",
        color="#000000",
        arrowprops=dict(arrowstyle="->", color="#000000", lw=0.8),
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.48, 1.30), ncol=2, frameon=False, fontsize=6.6, columnspacing=0.65, handlelength=1.2)
    ax.text(0.5, -0.32, "(a) Blind Spectral Estimator Ablation (M = 64)", transform=ax.transAxes, ha="center", va="top", fontweight="bold", fontsize=8.4, color="#000000")

    # ------------------------------------------------------------------
    # (b) Warm-Start t0 Sensitivity (PSNR / NME / LPIPS Sweet Spot)
    # ------------------------------------------------------------------
    ax = axes[1]
    t0_keys = ["0.00", "0.20", "0.35", "0.50", "0.65"]
    t0_vals = [float(k) for k in t0_keys]
    t0_psnr = [mb["t_warm_sweep_m64"][k]["psnr_mean"] for k in t0_keys]
    t0_lpips = [mb["t_warm_sweep_m64"][k]["lpips_mean"] for k in t0_keys]
    t0_nme = [mb["t_warm_sweep_m64"][k]["nme106_mean"] for k in t0_keys]

    ax.axvspan(0.20, 0.38, color=C_PHYS, alpha=0.09, label="Operating sweet spot")
    ln1 = ax.plot(t0_vals, t0_psnr, color=C_PHYS, marker="o", markersize=4.8, linewidth=1.7, label="PSNR (left, dB) ↑")
    ax.set_xlabel(r"Warm-Start Time $t_0$ ($M = 64$)", color="#000000")
    ax.set_ylabel("PSNR (dB) ↑", color="#000000")
    ax.tick_params(axis="y", colors="#000000")
    ax.set_xticks(t0_vals)
    ax.set_ylim(15.9, 17.65)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
    ax.spines["top"].set_visible(False)

    ax_r = ax.twinx()
    ln2 = ax_r.plot(t0_vals, t0_lpips, color=C_ORG, marker="s", markersize=4.4, linewidth=1.55, linestyle="--", label="LPIPS (right) ↓")
    ln3 = ax_r.plot(t0_vals, [v * 6.0 for v in t0_nme], color=C_GREEN, marker="^", markersize=4.6, linewidth=1.45, linestyle=":", label=r"$6{\times}$NME106 (right) ↓")
    ax_r.set_ylabel(r"LPIPS / $6{\times}$NME106 ↓", color="#000000")
    ax_r.tick_params(axis="y", colors="#000000")
    ax_r.set_ylim(0.47, 0.72)
    ax_r.spines["top"].set_visible(False)

    ax.axvline(0.25, color=C_PHYS, linestyle="-.", linewidth=0.9, alpha=0.75)
    ax.text(0.265, 16.03, r"$t_0(64)=0.25$", fontsize=6.9, color="#000000", fontweight="bold")

    hl1, lb1 = ax.get_legend_handles_labels()
    hl2, lb2 = ax_r.get_legend_handles_labels()
    ax.legend(hl1 + hl2, lb1 + lb2, loc="upper center", bbox_to_anchor=(0.50, 1.24), ncol=2, frameon=False, columnspacing=0.6, handlelength=1.3)
    ax.text(0.5, -0.32, "(b) Warm-Start Sensitivity Trade-off (M = 64)", transform=ax.transAxes, ha="center", va="top", fontweight="bold", fontsize=8.4, color="#000000")

    # ------------------------------------------------------------------
    # (c) Hardware-Agnostic Optical DMD Masks (M = 256, Zero Gen. Retrain)
    # ------------------------------------------------------------------
    ax = axes[2]
    mask_labels = ["LSI Binary\n(Learned)", "2D Hadamard\n{0,1} DMD", "Bernoulli\n{0,1} DMD"]
    xm = np.arange(len(mask_labels))
    wm = 0.165

    tv_p = [me["rates"]["256"]["tv_fista"]["psnr_mean"], md["hadamard_dmd_01"]["methods"]["tv_fista"]["256"]["psnr_mean"], md["bernoulli_dmd_01"]["methods"]["tv_fista"]["256"]["psnr_mean"]]
    zs_p = [me["rates"]["256"]["zs_flower"]["psnr_mean"], md["hadamard_dmd_01"]["methods"]["zs_flower"]["256"]["psnr_mean"], md["bernoulli_dmd_01"]["methods"]["zs_flower"]["256"]["psnr_mean"]]
    pn_p = [me["rates"]["256"]["pnp_flow"]["psnr_mean"], md["hadamard_dmd_01"]["methods"]["pnp_flow"]["256"]["psnr_mean"], md["bernoulli_dmd_01"]["methods"]["pnp_flow"]["256"]["psnr_mean"]]
    wn_p = [me["rates"]["256"]["wiener_lmmse"]["psnr_mean"], md["hadamard_dmd_01"]["methods"]["wiener_lmmse"]["256"]["psnr_mean"], md["bernoulli_dmd_01"]["methods"]["wiener_lmmse"]["256"]["psnr_mean"]]
    pf_p = [v3_s6["full479"]["256"]["physflow_v3_16step"]["psnr_mean"], md["hadamard_dmd_01"]["methods"]["physflow_zero_gen_retrain"]["256"]["psnr_mean"], md["bernoulli_dmd_01"]["methods"]["physflow_zero_gen_retrain"]["256"]["psnr_mean"]]

    ax.bar(xm - 2 * wm, tv_p, width=wm, color=C_GRAY, edgecolor="white", linewidth=0.35, label="TV-FISTA")
    ax.bar(xm - 1 * wm, zs_p, width=wm, color=C_GREEN, edgecolor="white", linewidth=0.35, label="ZS-FLOWER")
    ax.bar(xm + 0 * wm, pn_p, width=wm, color=C_ORG, edgecolor="white", linewidth=0.35, label="PnP-Flow")
    ax.bar(xm + 1 * wm, wn_p, width=wm, color=C_PURP, edgecolor="white", linewidth=0.35, label="Wiener")
    b_pf = ax.bar(xm + 2 * wm, pf_p, width=wm, color=C_PHYS, edgecolor="white", linewidth=0.35, label="PhysFlow (Ours)")

    for rect, val in zip(b_pf, pf_p):
        ax.text(rect.get_x() + rect.get_width() / 2.0, val + 0.25, f"{val:.1f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#000000")

    ax.set_xticks(xm)
    ax.set_xticklabels(mask_labels)
    ax.set_ylabel("Reconstruction PSNR (dB) ↑", color="#000000")
    ax.set_ylim(8.0, 22.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, 1.24), ncol=3, frameon=False, columnspacing=0.65, handlelength=1.15)
    ax.text(0.5, -0.32, "(c) Hardware-Agnostic DMD Masks (M = 256)", transform=ax.transAxes, ha="center", va="top", fontweight="bold", fontsize=8.4, color="#000000")

    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.055, right=0.985)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PREVIEW_PNG, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Saved:", OUT_FIG)


if __name__ == "__main__":
    main()
