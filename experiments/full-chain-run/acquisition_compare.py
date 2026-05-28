"""Patch 3 figure: compare EPIG, random, and leverage acquisition curves.

Reads:
  experiments/full-chain-run/output/trajectory.npz          (EPIG)
  experiments/full-chain-run/output_random/trajectory.npz   (uniform random)
  experiments/full-chain-run/output_leverage/trajectory.npz (leverage-greedy)

Writes:
  paper/figures/chain_acquisition_compare.png
  experiments/full-chain-run/output/acquisition_compare.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent / "paper" / "figures"
PAPER.mkdir(exist_ok=True)

LABELS = {
    "epig":     ("output",          "C0", "EPIG (closed-form info-gain)"),
    "random":   ("output_random",   "C1", "uniform random in [0.3, 2.3] TeV"),
    "leverage": ("output_leverage", "C2", "leverage-greedy (D-optimal)"),
}


def load(dir_name: str) -> dict:
    p = HERE / dir_name / "trajectory.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "rmse": d["rmse_band_trace"][:],
        "ctx":  d["context_size"][:],
        "oracle_calls": d["oracle_calls"][:],
        "H_T": d["H_T"][:],
    }


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    out_summary = {}
    for name, (dir_name, color, label) in LABELS.items():
        data = load(dir_name)
        if data is None:
            print(f"  [skip] {name}: no trajectory at {dir_name}")
            continue
        rmse = data["rmse"]
        cycles = np.arange(len(rmse))
        axes[0].semilogy(cycles, rmse, "-", color=color, lw=1.6, label=label)
        axes[1].plot(cycles, data["H_T"], "-", color=color, lw=1.6, label=label)
        rmse_final = float(np.median(rmse[-50:])) if len(rmse) >= 50 else float(rmse[-1])
        cycle_to_threshold = int(np.argmax(rmse < 10.0)) if (rmse < 10.0).any() else len(rmse)
        out_summary[name] = {
            "rmse_final_median": rmse_final,
            "rmse_initial": float(rmse[0]),
            "recovery_factor": float(rmse[0] / max(rmse_final, 1e-12)),
            "cycles_to_rmse_lt_10": cycle_to_threshold,
        }
        print(f"  {name:>10s}: RMSE_initial={rmse[0]:.1f}  RMSE_final={rmse_final:.2f}  "
              f"recovery={out_summary[name]['recovery_factor']:.1f}x  "
              f"cycles<10={cycle_to_threshold}")

    axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("Band RMSE (log)")
    axes[0].set_title("Recovery trajectory by acquisition function")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=9)
    axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("Target-set entropy $H_T$")
    axes[1].set_title("Predictive entropy by acquisition function")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.suptitle("Acquisition-function comparison on the engineered drift band", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = PAPER / "chain_acquisition_compare.png"
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"\nwrote {out_png}")

    out_json = HERE / "output" / "acquisition_compare.json"
    with open(out_json, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
