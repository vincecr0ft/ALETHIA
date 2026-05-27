"""Make the four diagnostic plots for the Intention-vs-DeepSets FM comparison.

Inputs:
  /tmp/fm_compare/results.npz       (raw predictions per held-out scenario)
  /tmp/fm_compare/intention_learned.pt
  /tmp/fm_compare/summary.json

Outputs (1200x800 PNGs):
  /home/vince/ALETHIA/docs/research/plots/intention_vs_deepsets_curves.png
  /home/vince/ALETHIA/docs/research/plots/intention_vs_deepsets_scaling.png
  /home/vince/ALETHIA/docs/research/plots/intention_vs_deepsets_inside_outside.png
  /home/vince/ALETHIA/docs/research/plots/intention_psi_basis.png
"""
from __future__ import annotations
import json
import sys
sys.path.insert(0, "/tmp/fm_compare")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from intention_learned import IntentionFMLearned, IntentionFMFixed

PLOTS = "/home/vince/ALETHIA/docs/research/plots"
DPI = 100
FIGSIZE = (12, 8)


def pick_scenarios(c: np.ndarray) -> list[int]:
    """Pick three illustrative held-out scenarios.

    1. SM-like: minimum max|c|.
    2. Energy-growth: largest |clq3| (clq3 = index 2).
    3. Interference: c with opposite signs across operators (median max|c|).
    """
    abs_c = np.abs(c)
    # SM-like: smallest L_inf
    i_sm = int(np.argmin(abs_c.max(axis=1)))
    # Energy growth: largest |clq3|
    i_eg = int(np.argmax(abs_c[:, 2]))
    if i_eg == i_sm:
        i_eg = int(np.argsort(abs_c[:, 2])[-2])
    # Interference: largest opposite-sign cHq3-clq3 magnitude product
    inter = -c[:, 0] * c[:, 2]   # large if opposite signs
    i_int = int(np.argmax(inter))
    if i_int in (i_sm, i_eg):
        # fall back to a different one
        for j in np.argsort(inter)[::-1]:
            if int(j) not in (i_sm, i_eg):
                i_int = int(j); break
    return [i_sm, i_eg, i_int]


def plot_curves(npz, summary):
    """Three-panel: ground truth + each architecture's predictions for three
    held-out scenarios (SM-like, energy-growth, interference)."""
    c = npz["test_in_c"]
    M_ctx = npz["test_in_M_ctx"]
    Y_ctx = npz["test_in_Y_ctx"]
    M_query = npz["test_in_M_query"]
    Y_query = npz["test_in_Y_query"]
    yp_fixed = npz["yp_fixed_in"]
    yp_int = npz["yp_int_in"]
    yp_ds = npz["yp_ds_in"]
    yp_reg = npz["yp_reg_in"]

    idxs = pick_scenarios(c)
    titles = ["SM-like", "energy-growth (clq3)", "interference"]

    m_fine = np.linspace(0.3, 2.3, 200)
    # ground truth on m_fine for each chosen scenario
    sys.path.insert(0, "/home/vince/ALETHIA")
    from modules.surrogate.ground_truth import DummyAnalyticOracle
    o = DummyAnalyticOracle(seed=0, noise_frac=0.0)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for col, (idx, title) in enumerate(zip(idxs, titles)):
        ax = axes[col]
        c_i = c[idx]
        c_fine = np.tile(c_i, (len(m_fine), 1))
        y_fine = o.truth(c_fine, m_fine)
        ax.plot(m_fine, y_fine - 1, "k-", lw=1.6, label="truth: mu(m)-1")
        # sort queries by m to get a clean curve
        order = np.argsort(M_query[idx])
        m_q = M_query[idx][order]
        ax.plot(m_q, yp_fixed[idx][order] - 1, "C0o-", ms=3, lw=0.8, alpha=0.7, label="Fixed psi")
        ax.plot(m_q, yp_int[idx][order] - 1, "C1s-", ms=3, lw=0.8, alpha=0.7, label="Learned psi")
        ax.plot(m_q, yp_ds[idx][order] - 1, "C2^-", ms=3, lw=0.8, alpha=0.7, label="DeepSets-FM")
        ax.plot(m_q, yp_reg[idx][order] - 1, "C3d-", ms=3, lw=0.8, alpha=0.5, label="Regressor (cheat)")
        ax.plot(M_ctx[idx], Y_ctx[idx] - 1, "kx", ms=7, mew=1.5, label="context")
        ax.set_xlabel(r"$m_{\ell\ell}$  [TeV]")
        ax.set_ylabel(r"$\mu(m) - 1$")
        c_str = ", ".join([f"{name}={c_i[k]:+.2f}" for k, name
                           in enumerate(("cHq3", "cHq1", "clq3", "clq1"))])
        ax.set_title(f"{title}\n{c_str}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Held-out scenarios: prediction curves vs ground truth", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{PLOTS}/intention_vs_deepsets_curves.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def plot_scaling(summary):
    """Median held-out R^2 vs meta-steps for Intention_Learned and DeepSets."""
    int_hist = summary["IntentionFM_Learned"]["scaling_history"]
    ds_hist = summary["DeepSets_FM"]["scaling_history"]
    int_steps, int_r2 = zip(*int_hist)
    ds_steps, ds_r2 = zip(*ds_hist)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(int_steps, int_r2, "C1s-", lw=1.6, ms=8, label="IntentionFM_Learned")
    ax.semilogx(ds_steps, ds_r2, "C2^-", lw=1.6, ms=8, label="DeepSets-FM")
    # Fixed and ceiling as horizontal lines.
    r2_fixed = summary["IntentionFM_Fixed"]["in"]["r2_median"]
    r2_ceil = summary["IntentionFM_Regressor_cheat"]["in"]["r2_median"]
    ax.axhline(r2_fixed, color="C0", linestyle="--", alpha=0.7,
               label=f"Fixed psi (no training): {r2_fixed:+.3f}")
    ax.axhline(r2_ceil, color="C3", linestyle=":", alpha=0.7,
               label=f"Regressor cheat: {r2_ceil:+.3f}")
    ax.set_xlabel("Meta-steps (Adam updates on scenario batches)")
    ax.set_ylabel("Median held-out R$^2$ (inside-box)")
    ax.set_title("FM scaling: held-out median R$^2$ vs training meta-steps")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")
    ax.set_ylim(-1.5, 1.05)
    fig.tight_layout()
    out = f"{PLOTS}/intention_vs_deepsets_scaling.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def plot_inside_outside(summary):
    """Bar chart of median and p5 R^2 across four architectures, two regions."""
    names = ["Fixed psi", "Learned psi", "DeepSets-FM", "Regressor (cheat)"]
    keys = ["IntentionFM_Fixed", "IntentionFM_Learned",
            "DeepSets_FM", "IntentionFM_Regressor_cheat"]
    med_in = [summary[k]["in"]["r2_median"] for k in keys]
    p5_in = [summary[k]["in"]["r2_p5"] for k in keys]
    med_out = [summary[k]["out"]["r2_median"] for k in keys]
    p5_out = [summary[k]["out"]["r2_p5"] for k in keys]

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    x = np.arange(len(names))
    w = 0.35
    for ax, med, p5, region in [
        (ax_in, med_in, p5_in, "inside |c| <= 0.7"),
        (ax_out, med_out, p5_out, "outside 0.7 < max|c| <= 1.0"),
    ]:
        ax.bar(x - w/2, med, w, color="C0", label="median R$^2$")
        ax.bar(x + w/2, p5, w, color="C3", alpha=0.85, label="5th percentile R$^2$")
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(region)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(-2.0, 1.1)
        for xi, vi in zip(x - w/2, med):
            ax.text(xi, vi + 0.03 if vi >= 0 else vi - 0.1, f"{vi:+.3f}",
                    fontsize=7, ha="center")
        for xi, vi in zip(x + w/2, p5):
            ax.text(xi, vi + 0.03 if vi >= 0 else vi - 0.1, f"{vi:+.3f}",
                    fontsize=7, ha="center")
    ax_in.set_ylabel("R$^2$ on held-out scenarios")
    ax_in.legend(loc="lower left")
    fig.suptitle("Architecture comparison: held-out R$^2$ inside vs outside training c-box",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{PLOTS}/intention_vs_deepsets_inside_outside.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def plot_psi_basis():
    """Visualise the learned psi_theta basis vs the fixed polynomial basis."""
    int_model = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    int_model.load_state_dict(torch.load("/tmp/fm_compare/intention_learned.pt"))
    int_model.eval()

    m_fine = np.linspace(0.2, 2.5, 400)
    with torch.no_grad():
        psi_learned = int_model.psi(torch.from_numpy(m_fine.astype(np.float32))).numpy()
    # Fixed polynomial basis: 5 columns of log(m/M_REF)^k
    log_m = np.log(m_fine / 1.0)
    psi_fixed = np.stack([log_m ** k for k in range(5)], axis=1)

    # Normalise each column to unit max-abs for visual comparability.
    def normed(x):
        s = np.max(np.abs(x), axis=0, keepdims=True)
        s = np.where(s < 1e-9, 1.0, s)
        return x / s

    pl = normed(psi_learned)
    pf = normed(psi_fixed)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.95, pl.shape[1]))
    for k in range(pl.shape[1]):
        axL.plot(m_fine, pl[:, k], color=cmap[k], lw=1.3, alpha=0.9)
    axL.set_title(f"Learned $\\psi_\\theta$ basis ({pl.shape[1]} columns)")
    axL.set_xlabel(r"$m_{\ell\ell}$  [TeV]")
    axL.set_ylabel("normalised basis function")
    axL.grid(True, alpha=0.3)
    axL.axhline(0, color="k", lw=0.4)

    cmap2 = plt.cm.plasma(np.linspace(0, 0.85, pf.shape[1]))
    for k in range(pf.shape[1]):
        axR.plot(m_fine, pf[:, k], color=cmap2[k], lw=1.5,
                 label=f"$\\log^{{{k}}}(m/M_{{\\rm ref}})$")
    axR.set_title("Fixed polynomial basis (5 columns)")
    axR.set_xlabel(r"$m_{\ell\ell}$  [TeV]")
    axR.grid(True, alpha=0.3)
    axR.axhline(0, color="k", lw=0.4)
    axR.legend(loc="lower right", fontsize=8)
    fig.suptitle("Learned vs fixed $\\psi$ basis functions over $m \\in [0.2, 2.5]$ TeV",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{PLOTS}/intention_psi_basis.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    npz = np.load("/tmp/fm_compare/results.npz")
    with open("/tmp/fm_compare/summary.json") as f:
        summary = json.load(f)
    plot_curves(npz, summary)
    plot_scaling(summary)
    plot_inside_outside(summary)
    plot_psi_basis()


if __name__ == "__main__":
    main()
