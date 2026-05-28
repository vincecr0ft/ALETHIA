"""Bootstrap confidence intervals on the held-out comparison.

Loads the per-scenario predictions/truths from one or more results.npz
files (the polynomial-toy and analytic-SMEFT headline runs), computes
per-scenario R^2 and MSE for every architecture, then attaches 95%
bootstrap CIs by resampling the 50 scenarios with replacement.

Usage:
    uv run python experiments/intention-vs-deepsets/bootstrap_metrics.py

Outputs:
    experiments/intention-vs-deepsets/output_smeft/bootstrap_metrics.csv
    experiments/intention-vs-deepsets/output_smeft/bootstrap_metrics.json
    (and the same in /tmp/fm_compare/ for the polynomial-toy run if present)
"""
from __future__ import annotations
import csv
import json
import os
from typing import Optional

import numpy as np


N_BOOT = 1000
SEED = 0
CI_LO = 2.5
CI_HI = 97.5
ARCHS = [
    ("IntentionFM_Fixed",   "yp_fixed_in",   "yp_fixed_out"),
    ("IntentionFM_Learned", "yp_int_in",     "yp_int_out"),
    ("DeepSets_FM",         "yp_ds_in",      "yp_ds_out"),
    ("CheatRegressor",      "yp_reg_in",     "yp_reg_out"),
]


def per_scenario_r2(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """y_pred, y_true: (S, Q). Returns (S,)."""
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def per_scenario_mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.mean((y_pred - y_true) ** 2, axis=1)


def bootstrap_stats(metric_per_scenario: np.ndarray,
                    n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap 95% CIs on median and 5th-percentile of the metric across
    scenarios. Returns a dict with the central value and CI for each."""
    rng = np.random.default_rng(seed)
    S = len(metric_per_scenario)
    medians = np.empty(n_boot)
    p5s = np.empty(n_boot)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, S, size=S)
        sample = metric_per_scenario[idx]
        medians[b] = float(np.median(sample))
        p5s[b] = float(np.percentile(sample, 5))
        means[b] = float(np.mean(sample))
    return {
        "median":       float(np.median(metric_per_scenario)),
        "median_lo":    float(np.percentile(medians, CI_LO)),
        "median_hi":    float(np.percentile(medians, CI_HI)),
        "p5":           float(np.percentile(metric_per_scenario, 5)),
        "p5_lo":        float(np.percentile(p5s, CI_LO)),
        "p5_hi":        float(np.percentile(p5s, CI_HI)),
        "mean":         float(np.mean(metric_per_scenario)),
        "mean_lo":      float(np.percentile(means, CI_LO)),
        "mean_hi":      float(np.percentile(means, CI_HI)),
        "n_scenarios":  int(S),
    }


def process_run(npz_path: str, out_dir: str, label: str) -> dict:
    if not os.path.exists(npz_path):
        return {"_missing": npz_path}
    print(f"\n# {label}  ({npz_path})")
    d = np.load(npz_path)
    Y_in  = d["test_in_Y_query"]    # (S_in, Q)
    Y_out = d["test_out_Y_query"]   # (S_out, Q)
    print(f"  in : S={Y_in.shape[0]}  Q={Y_in.shape[1]}")
    print(f"  out: S={Y_out.shape[0]} Q={Y_out.shape[1]}")
    table = {"label": label, "results": {}}
    rows = []
    for name, key_in, key_out in ARCHS:
        if key_in not in d.files:
            print(f"  [skip] {name}: missing key {key_in}")
            continue
        r2_in  = per_scenario_r2(d[key_in],  Y_in)
        r2_out = per_scenario_r2(d[key_out], Y_out)
        mse_in  = per_scenario_mse(d[key_in],  Y_in)
        mse_out = per_scenario_mse(d[key_out], Y_out)
        stats = {
            "r2_in":   bootstrap_stats(r2_in),
            "r2_out":  bootstrap_stats(r2_out),
            "mse_in":  bootstrap_stats(mse_in),
            "mse_out": bootstrap_stats(mse_out),
        }
        table["results"][name] = stats
        # Pretty-print:
        for pool, st in (("in", stats["r2_in"]), ("out", stats["r2_out"])):
            print(f"  {name:24s} {pool:3s}  "
                  f"R2 median = {st['median']:+.4f} [{st['median_lo']:+.4f}, {st['median_hi']:+.4f}]   "
                  f"p5 = {st['p5']:+.4f} [{st['p5_lo']:+.4f}, {st['p5_hi']:+.4f}]")
        for pool in ("in", "out"):
            st_r2  = stats[f"r2_{pool}"]
            st_mse = stats[f"mse_{pool}"]
            rows.append({
                "arch": name, "pool": pool,
                "r2_median":  f"{st_r2['median']:.4f}",
                "r2_median_ci": f"[{st_r2['median_lo']:.4f}, {st_r2['median_hi']:.4f}]",
                "r2_p5":       f"{st_r2['p5']:.4f}",
                "r2_p5_ci":    f"[{st_r2['p5_lo']:.4f}, {st_r2['p5_hi']:.4f}]",
                "mse_mean":    f"{st_mse['mean']:.4e}",
                "mse_mean_ci": f"[{st_mse['mean_lo']:.4e}, {st_mse['mean_hi']:.4e}]",
            })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bootstrap_metrics.csv")
    json_path = os.path.join(out_dir, "bootstrap_metrics.json")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    return table


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # Analytic-SMEFT headline (new).
    process_run(
        npz_path=os.path.join(here, "output_smeft", "results.npz"),
        out_dir=os.path.join(here, "output_smeft"),
        label="Analytic-SMEFT headline",
    )
    # Polynomial-toy ablation (existing /tmp run, kept for Appendix C).
    process_run(
        npz_path="/tmp/fm_compare/results.npz",
        out_dir="/tmp/fm_compare",
        label="Polynomial-toy ablation",
    )


if __name__ == "__main__":
    main()
