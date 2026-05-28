"""SMEFT-headline figures: regenerate the four diagnostic plots against
the analytic SMEFT oracle (Patch 1 headline of physics_patches.md).

Reads:
  experiments/intention-vs-deepsets/output_smeft/results.npz
  experiments/intention-vs-deepsets/output_smeft/summary.json
  experiments/intention-vs-deepsets/output_smeft/intention_learned.pt

Writes:
  paper/figures/intention_vs_deepsets_inside_outside_smeft.png
  paper/figures/intention_vs_deepsets_curves_smeft.png
  paper/figures/intention_vs_deepsets_scaling_smeft.png
  paper/figures/intention_psi_basis_smeft.png
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from intention_learned import IntentionFMLearned
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle

NPZ = HERE / "output_smeft" / "results.npz"
SUMMARY = HERE / "output_smeft" / "summary.json"
MODEL = HERE / "output_smeft" / "intention_learned.pt"

PLOTS = HERE.parent.parent / "paper" / "figures"
PLOTS.mkdir(parents=True, exist_ok=True)

DPI = 110
FIGSIZE = (12, 8)
SUFFIX = "_smeft"


def pick_scenarios(c: np.ndarray) -> list[int]:
    abs_c = np.abs(c)
    i_sm = int(np.argmin(abs_c.max(axis=1)))
    i_eg = int(np.argmax(abs_c[:, 2]))
    if i_eg == i_sm:
        i_eg = int(np.argsort(abs_c[:, 2])[-2])
    inter = -c[:, 0] * c[:, 2]
    i_int = int(np.argmax(inter))
    if i_int in (i_sm, i_eg):
        for j in np.argsort(inter)[::-1]:
            if int(j) not in (i_sm, i_eg):
                i_int = int(j); break
    return [i_sm, i_eg, i_int]


def plot_curves(npz):
    c = npz["test_in_c"]
    M_ctx = npz["test_in_M_ctx"]; Y_ctx = npz["test_in_Y_ctx"]
    M_query = npz["test_in_M_query"]; Y_query = npz["test_in_Y_query"]
    idxs = pick_scenarios(c)
    titles = ["SM-like", "energy-growth (clq3)", "interference"]
    m_fine = np.linspace(0.3, 2.3, 200)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for col, (idx, title) in enumerate(zip(idxs, titles)):
        ax = axes[col]
        c_i = c[idx]
        # Ground truth on fine grid
        gt = oracle.truth(np.tile(c_i, (len(m_fine), 1)), m_fine)
        ax.plot(m_fine, gt, "k-", lw=1.5, label="ground truth")
        ax.scatter(M_ctx[idx], Y_ctx[idx], c="grey", s=18, marker="o",
                   alpha=0.7, label="context")
        ax.scatter(M_query[idx], npz["yp_int_in"][idx], c="C1", s=28, marker="s",
                   label="Intention learned", alpha=0.9)
        ax.scatter(M_query[idx], npz["yp_fixed_in"][idx], c="C0", s=22, marker="D",
                   label="Intention fixed", alpha=0.7)
        ax.scatter(M_query[idx], npz["yp_ds_in"][idx], c="C2", s=22, marker="^",
                   label="DeepSets", alpha=0.7)
        ax.scatter(M_query[idx], npz["yp_reg_in"][idx], c="C3", s=22, marker="v",
                   label="cheat regressor", alpha=0.7)
        ax.set_xlabel("$m_{\\ell\\ell}$ [TeV]")
        if col == 0:
            ax.set_ylabel("$\\mu(\\mathbf{c}, m_{\\ell\\ell})$")
        ax.set_title(f"{title}\nc=({c_i[0]:+.2f}, {c_i[1]:+.2f}, {c_i[2]:+.2f}, {c_i[3]:+.2f})",
                     fontsize=9)
        ax.set_yscale("symlog", linthresh=1.5)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Held-out predictions on the analytic-SMEFT oracle (Drell-Yan, LO, CT18NNLO-equivalent)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS / f"intention_vs_deepsets_curves{SUFFIX}.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print(f"wrote {out}")


def plot_scaling(summary):
    int_hist = summary["IntentionFM_Learned"]["scaling_history"]
    ds_hist = summary["DeepSets_FM"]["scaling_history"]
    int_steps, int_r2 = zip(*int_hist)
    ds_steps, ds_r2 = zip(*ds_hist)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(int_steps, int_r2, "C1s-", lw=1.6, ms=8, label="Intention (learned $\\psi_\\theta$)")
    ax.semilogx(ds_steps, ds_r2, "C2^-", lw=1.6, ms=8, label="DeepSets baseline")
    r2_fixed = summary["IntentionFM_Fixed"]["in"]["r2_median"]
    r2_ceil = summary["IntentionFM_Regressor_cheat"]["in"]["r2_median"]
    ax.axhline(r2_fixed, color="C0", linestyle="--", alpha=0.7,
               label=f"Intention fixed (no training): {r2_fixed:+.4f}")
    ax.axhline(r2_ceil, color="C3", linestyle=":", alpha=0.7,
               label=f"Cheat regressor: {r2_ceil:+.4f}")
    ax.set_xlabel("Meta-steps (Adam updates)")
    ax.set_ylabel("Median held-out $R^2$ (in-distribution)")
    ax.set_title("Sample efficiency on the analytic-SMEFT oracle")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")
    ax.set_ylim(-1.5, 1.05)
    fig.tight_layout()
    out = PLOTS / f"intention_vs_deepsets_scaling{SUFFIX}.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print(f"wrote {out}")


def plot_inside_outside(summary):
    names = ["Intention fixed", "Intention learned", "DeepSets", "Cheat regressor"]
    keys = ["IntentionFM_Fixed", "IntentionFM_Learned",
            "DeepSets_FM", "IntentionFM_Regressor_cheat"]
    med_in = [summary[k]["in"]["r2_median"] for k in keys]
    p5_in = [summary[k]["in"]["r2_p5"] for k in keys]
    med_out = [summary[k]["out"]["r2_median"] for k in keys]
    p5_out = [summary[k]["out"]["r2_p5"] for k in keys]
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    x = np.arange(len(names)); w = 0.35
    for ax, med, p5, region in [
        (ax_in, med_in, p5_in, "inside $|\\mathbf{c}|_\\infty \\leq 0.7$"),
        (ax_out, med_out, p5_out, "magnitude extrapolation: $0.7 < |\\mathbf{c}|_\\infty \\leq 1.0$"),
    ]:
        ax.bar(x - w/2, med, w, color="C0", label="median $R^2$")
        ax.bar(x + w/2, p5, w, color="C3", alpha=0.85, label="5th percentile $R^2$")
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(region); ax.grid(True, axis="y", alpha=0.3)
        # Clip extreme negative p5s to keep the chart readable, annotate the real number
        ax.set_ylim(-2.0, 1.1)
        for xi, vi in zip(x - w/2, med):
            ax.text(xi, vi + 0.03 if vi >= 0 else vi - 0.12, f"{vi:+.3f}",
                    fontsize=8, ha="center")
        for xi, vi in zip(x + w/2, p5):
            txt = f"{vi:+.3f}"
            disp = max(vi, -1.9)
            ax.text(xi, disp + 0.03 if disp >= 0 else disp - 0.12, txt,
                    fontsize=8, ha="center")
    ax_in.set_ylabel("$R^2$ on held-out scenarios")
    ax_in.legend(loc="lower left")
    fig.suptitle("Architecture comparison on the analytic-SMEFT oracle",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS / f"intention_vs_deepsets_inside_outside{SUFFIX}.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print(f"wrote {out}")


def plot_psi_basis():
    int_model = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    int_model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    int_model.eval()
    m_fine = np.linspace(0.2, 2.5, 400).astype(np.float32)
    with torch.no_grad():
        psi_learned = int_model.psi(torch.from_numpy(m_fine)).numpy()
    log_m = np.log(m_fine / 1.0)
    psi_fixed = np.stack([log_m ** k for k in range(5)], axis=1)
    def normed(x):
        s = np.max(np.abs(x), axis=0, keepdims=True)
        return x / np.where(s < 1e-9, 1.0, s)
    pl = normed(psi_learned); pf = normed(psi_fixed)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
    for col in range(pl.shape[1]):
        axes[0].plot(m_fine, pl[:, col], lw=1.2, alpha=0.8)
    axes[0].set_title(f"Learned $\\psi_\\theta(m)$ ({pl.shape[1]} columns, SMEFT training)")
    axes[0].set_xlabel("$m_{\\ell\\ell}$ [TeV]")
    axes[0].set_ylabel("normalised feature value")
    axes[0].grid(alpha=0.3)
    for col in range(pf.shape[1]):
        axes[1].plot(m_fine, pf[:, col], lw=1.4, alpha=0.85,
                     label=f"$\\log^{{{col}}}(m/M_{{\\rm ref}})$")
    axes[1].set_title(f"Fixed polynomial basis ({pf.shape[1]} columns)")
    axes[1].set_xlabel("$m_{\\ell\\ell}$ [TeV]")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Learned vs fixed feature map (analytic-SMEFT training)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS / f"intention_psi_basis{SUFFIX}.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print(f"wrote {out}")


def main():
    npz = np.load(NPZ)
    with open(SUMMARY) as f:
        summary = json.load(f)
    plot_inside_outside(summary)
    plot_scaling(summary)
    plot_curves(npz)
    plot_psi_basis()


if __name__ == "__main__":
    main()
