"""Four headline plots for the full-chain run.

Reads experiments/full-chain-run/output/trajectory.npz and summary.json,
emits PNGs to docs/research/plots/full-chain/.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PLOTS = HERE.parent.parent / "docs" / "research" / "plots" / "full-chain"
PLOTS.mkdir(parents=True, exist_ok=True)

traj = np.load(OUT / "trajectory.npz", allow_pickle=True)
with open(OUT / "summary.json") as f:
    summary = json.load(f)

H_T = traj["H_T"]
kappa = traj["kappa"]
cov_68 = traj["cov_68_by_region"]
cov_95 = traj["cov_95_by_region"]
context_size = traj["context_size"]
oracle_calls = traj["oracle_calls"]
drift_flags = traj["drift_flags"]
action = traj["action"]
cusum_S = traj["cusum_S"]
pvalue_min = traj["cal_pvalue_min"]
proj_ratio = traj["proj_ratio"]
rmse_band = traj["rmse_band_trace"]
M_band = traj["M_band_eval"]
Y_truth = traj["Y_band_truth"]
mu_before = traj["mu_before"]
mu_after = traj["mu_after"]
n = len(H_T)
cycles = np.arange(n)


# ---------- Plot 1: chain trajectory (4 panels) ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

ax = axes[0, 0]
ax.plot(cycles, H_T, "C0-", lw=1.5)
ax.set_xlabel("Cycle")
ax.set_ylabel(r"target-set entropy $H_T$")
ax.set_title(r"$H_T = \frac{1}{2}\sum_i \log(2\pi e \sigma^2 \mathrm{lev}_{T_i})$")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.semilogy(cycles, kappa, "C3-", lw=1.5)
ax.axhline(summary["kappa_threshold"], color="gray", ls="--", lw=0.8,
           label=f"threshold = {summary['kappa_threshold']:.0e}")
ax.set_xlabel("Cycle")
ax.set_ylabel(r"$\kappa(A)$")
ax.set_title("design-matrix condition number")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, which="both")

ax = axes[1, 0]
cov_per_cycle = np.array([cov_68[i] for i in range(n)])
for r_i in range(cov_per_cycle.shape[1]):
    ax.plot(cycles, cov_per_cycle[:, r_i], lw=1.0, alpha=0.7,
            label=f"region {r_i}")
ax.axhline(0.683, color="black", ls="--", lw=1.0, label="nominal 0.683")
ax.fill_between([0, n - 1], 0.65, 0.72, color="gray", alpha=0.15,
                label=r"$\pm 2$pp band")
ax.set_xlabel("Cycle")
ax.set_ylabel(r"empirical $\mathrm{cov}_{68\%}$")
ax.set_title("per-region 68% coverage (windowed)")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(cycles, oracle_calls, "C2-", lw=1.5, label="oracle calls (cum)")
ax.set_xlabel("Cycle")
ax.set_ylabel("oracle calls (cumulative)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")
ax2 = ax.twinx()
ax2.plot(cycles, context_size, "C1-", lw=1.5, label="context size")
ax2.set_ylabel("context size", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")
ax.set_title("oracle budget vs context growth")
ax.grid(alpha=0.3)

fig.suptitle("ALETHIA full-chain run: trajectories\n"
             f"target $c_{{lq}}^{{(3)}}=0.8$ (in withheld band [0.6, 1.0]), "
             f"{summary['cycles_done']} cycles, "
             f"{summary['oracle_calls']} oracle calls",
             fontsize=12)
fig.savefig(PLOTS / "chain_trajectory.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved", PLOTS / "chain_trajectory.png")


# ---------- Plot 2: drift events timeline ----------
fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True,
                         sharex=True)

ax = axes[0]
ax.plot(cycles, cusum_S, "C0-", lw=1.2, label="DAS-CUSUM $S_t$")
ax.axhline(summary["cusum_h"], color="gray", ls="--", lw=0.8,
           label=f"threshold h={summary['cusum_h']}")
ax.set_ylabel("CUSUM statistic")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
ax.set_title("Accuracy drift: DAS-CUSUM on standardised residual median")

ax = axes[1]
ax.semilogy(cycles, np.maximum(pvalue_min, 1e-10), "C3-", lw=1.2,
            label="min per-region p-value")
ax.axhline(summary["bh_alpha"], color="gray", ls="--", lw=0.8,
           label=fr"$\alpha={summary['bh_alpha']}$")
ax.set_ylabel("BH min p-value")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, which="both")
ax.set_title("Calibration drift: BH-corrected per-region binomial coverage")

ax = axes[2]
# Mark each cycle's action with a coloured stripe.
unique_actions = list(set(action))
colors = {"noop": "white", "watch": "lightblue", "cooled": "moccasin",
          "recal": "lightgreen", "recal_then_check": "lightgreen",
          "local_retrain": "orange", "global_retrain": "red",
          "budget_exhausted": "gray"}
seen = set()
for i, a in enumerate(action):
    col = colors.get(a, "lightgray")
    if col != "white":
        lbl = a if a not in seen else None
        ax.axvspan(i - 0.5, i + 0.5, color=col, alpha=0.7,
                   label=lbl)
        seen.add(a)
ax.plot(cycles, context_size, "k-", lw=1.5, label="context size")
ax.set_xlabel("Cycle")
ax.set_ylabel("context size")
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.grid(alpha=0.3)
ax.set_title("Aggregator actions over time (stripes) and context growth")

fig.suptitle("ALETHIA full-chain run: drift events and actions",
             fontsize=12)
fig.savefig(PLOTS / "chain_drift_events.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved", PLOTS / "chain_drift_events.png")


# ---------- Plot 3: before vs after FM prediction in the band ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                         sharey=True)
for ax, mu_pred, label in zip(axes, [mu_before, mu_after],
                              ["before loop", "after loop"]):
    ax.plot(M_band, Y_truth, "k-", lw=2.0, label=r"truth $\mu(m)$ at $c_{lq}^{(3)}=0.8$")
    ax.plot(M_band, mu_pred, "C3-", lw=1.5, label="FM prediction")
    rmse = float(np.sqrt(np.mean((mu_pred - Y_truth) ** 2)))
    ax.set_title(f"{label}: RMSE = {rmse:.2f}")
    ax.set_xlabel(r"$m_{\ell\ell}$  [TeV]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
axes[0].set_ylabel(r"$\mu = \sigma_{\mathrm{total}} / \sigma_{\mathrm{SM}}$")

fig.suptitle("FM prediction in the withheld band: BEFORE vs AFTER the loop\n"
             f"engineered drift in $|c_{{lq}}^{{(3)}}| \\in [0.6, 1.0]$; "
             f"target c = {summary['engineered_target_c']}",
             fontsize=12)
fig.savefig(PLOTS / "chain_before_after.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved", PLOTS / "chain_before_after.png")


# ---------- Plot 4: RMSE recovery trajectory ----------
fig, ax = plt.subplots(1, 1, figsize=(11, 5.5), constrained_layout=True)
ax.semilogy(cycles, rmse_band, "C0-", lw=1.5, label="FM RMSE on target band")
# Mark cycles where local_retrain fired.
for i, a in enumerate(action):
    if a == "local_retrain":
        ax.axvline(i, color="orange", alpha=0.3, lw=0.8)
# A dummy patch for the legend (avoid add_patch on a Patch — use handles).
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color="C0", lw=1.5, label="FM RMSE on target band"),
           mpatches.Patch(color="orange", alpha=0.3,
                          label="local_retrain (oracle call)")]
ax.legend(handles=handles, fontsize=9, loc="upper right")
ax.set_xlabel("Cycle")
ax.set_ylabel(r"RMSE of $\hat\mu(m)$ vs truth on the band  (log scale)")
ax.set_title("Convergence: FM RMSE in the withheld band over the loop\n"
             "vertical lines mark EPIG-driven oracle invocations")
ax.grid(alpha=0.3, which="both")
fig.savefig(PLOTS / "chain_rmse_recovery.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved", PLOTS / "chain_rmse_recovery.png")

print(f"\nAll plots written to {PLOTS}")
print(json.dumps(summary, indent=2))
