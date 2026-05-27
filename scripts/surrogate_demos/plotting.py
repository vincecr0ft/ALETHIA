r"""
Plot helpers used by demo scripts. Kept out of the package proper because
they're matplotlib-flavoured and the library should stay numpy-only.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from modules.surrogate import N_WC, WC_NAMES


def drift_heatmap(
    ax,
    model,
    *,
    c0_grid: np.ndarray,
    m_grid: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
):
    """log10 leverage heatmap over (c_0, m), with other Wilson coords at 0.

    Returns (image, leverage_2d) so callers can share color scales.
    """
    cv, mv = np.meshgrid(c0_grid, m_grid)
    C = np.zeros((cv.size, N_WC))
    C[:, 0] = cv.flatten()
    M = mv.flatten()
    lev = model.leverage(C, M).reshape(cv.shape)
    im = ax.pcolormesh(c0_grid, m_grid, np.log10(np.maximum(lev, 1e-6)),
                       shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"{WC_NAMES[0]} (others = 0)")
    ax.set_ylabel(r"$m_{\ell\ell}$ / TeV")
    return im, lev


def overlay_picks_by_round(ax, acquired_history, *, cmap_name: str = "cool"):
    """Scatter the (c_0, m) projection of picks across rounds, coloured by round."""
    n_rounds  = len(acquired_history)
    cmap      = plt.get_cmap(cmap_name, n_rounds)
    for r, C_a, M_a in acquired_history:
        ax.scatter(C_a[:, 0], M_a, color=cmap(r), s=24,
                   edgecolors="black", linewidths=0.4, zorder=3)
    return cmap


def coverage_bars(
    ax,
    cc,
    raw_coverage_1s: np.ndarray,
    conformal_coverage_1s: np.ndarray,
    *,
    target: float = 0.683,
):
    """Side-by-side bars for raw vs conformal per-stratum coverage."""
    n = cc.n_strata
    x = np.arange(n)
    w = 0.36
    ax.bar(x - w / 2, raw_coverage_1s, w, color="C7", label="raw (1σ-equiv)")
    ax.bar(x + w / 2, conformal_coverage_1s, w, color="C3", label="conformal (1σ-equiv)")
    ax.axhline(target, color="k", ls="--", lw=0.7, label=f"target {target}")
    ax.set_xticks(x)
    ax.set_xlabel("leverage stratum (low → high)")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="lower left")


def learning_curves(
    ax,
    *,
    x_calls: np.ndarray,
    curves: dict[str, tuple[np.ndarray, str]],   # label -> (curve_matrix, colour)
    ylabel: str = "mean leverage on hard eval set",
    title: str = "",
    yscale: str = "log",
):
    """Mean ± std bands across seeds. ``curves`` rows are seeds, cols are rounds."""
    for label, (arr, colour) in curves.items():
        mean = arr.mean(axis=0)
        sd   = arr.std(axis=0)
        ax.plot(x_calls, mean, "o-", color=colour, label=label)
        ax.fill_between(x_calls, np.maximum(mean - sd, 1e-9), mean + sd,
                        color=colour, alpha=0.18)
    ax.set_xlabel("oracle calls (cumulative)")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
