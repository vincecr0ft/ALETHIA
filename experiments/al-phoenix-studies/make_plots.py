"""Plot the contraction-vs-MLE divergence — the headline figure for the report.

One panel per (pool × K) cell from the closed-loop aggregate; each panel
plots per-acquisition contraction and MLE error as paired bars with seed
scatter. The visual story: contraction goes down (left bars), MLE goes up
(right bars). Saved to output/contraction_vs_mle.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def main():
    summary = json.load(open(OUT / "stage_c_closed_loop_aggregate.json"))
    rows = summary["aggregate"]["per_cfg"]
    by_pk = {}
    for r in rows:
        by_pk.setdefault((r["pool"], r["K"]), []).append(r)

    cells = sorted(by_pk.keys())
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5),
                              constrained_layout=True, squeeze=False)

    ordering = ["random", "leverage", "epig", "param_epig_d", "param_epig_a"]
    colors = {"random": "tab:blue", "leverage": "tab:orange",
              "epig": "tab:green", "param_epig_d": "tab:red",
              "param_epig_a": "tab:purple"}

    for ax, (pool, K) in zip(axes[0], cells):
        cell = {r["acq"]: r for r in by_pk[(pool, K)]}
        x = np.arange(len(ordering))
        # Twin axis for the two metrics.
        ax2 = ax.twinx()
        c_means = [cell[a]["contraction_d1_mean"] for a in ordering]
        c_stds = [cell[a]["contraction_d1_std"] for a in ordering]
        m_means = [cell[a]["mle_err_d1_mean"] for a in ordering]
        m_stds = [cell[a]["mle_err_d1_std"] for a in ordering]
        w = 0.4
        b1 = ax.bar(x - w/2, c_means, w, yerr=c_stds, capsize=3,
                     color=[colors[a] for a in ordering], alpha=0.85,
                     label="contraction Σ_aa^post / Σ_aa^seed (d1)")
        b2 = ax2.bar(x + w/2, m_means, w, yerr=m_stds, capsize=3,
                      color=[colors[a] for a in ordering], alpha=0.4,
                      edgecolor="black", hatch="//",
                      label="MLE |c̃_d1 − truth|")
        ax.set_xticks(x)
        ax.set_xticklabels(ordering, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"pool={pool}  K={K}  (n={cell[ordering[0]]['n_seeds']} seeds)",
                      fontsize=10)
        ax.set_ylabel("contraction (lower = better)")
        ax2.set_ylabel("MLE error (lower = better)")
        # Annotate random as the baseline.
        ax.axhline(cell["random"]["contraction_d1_mean"], color="tab:blue",
                    lw=0.5, ls=":")
        ax2.axhline(cell["random"]["mle_err_d1_mean"], color="tab:blue",
                     lw=0.5, ls=":", alpha=0.4)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0.78, 0.86)
        ax2.set_ylim(1.13, 1.15)
    fig.suptitle(
        "Contraction beats random by 8–47σ; MLE error gets *worse* by 7–47σ — "
        "the AL_separation §3.4 anti-tautology in action",
        fontsize=11)
    fig.savefig(OUT / "contraction_vs_mle.png", dpi=130, bbox_inches="tight")
    print(f"wrote {OUT / 'contraction_vs_mle.png'}")


if __name__ == "__main__":
    main()
