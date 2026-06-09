"""Scaling-study plots: does JEPA close the gap to Intention as we scale
data / params / training time / architecture?

Reads:  output_jepa_scaling/summary.json
Writes: docs/research/plots/jepa_scaling_*.png

Five panels:
  1. jepa_scaling_overview.png   — all configurations laid out side-by-side,
                                   JEPA vs Intention at matched scale.
  2. jepa_scaling_data.png       — R² vs N_train at fixed model+steps.
  3. jepa_scaling_params.png     — R² vs n_params at fixed data+steps.
  4. jepa_scaling_time.png       — R² vs n_meta_steps at fixed data+model.
  5. jepa_scaling_collapse.png   — min_std(embedding) vs scale, JEPA only.
                                   Shows whether collapse improves with size.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUM_PATH = HERE / "output_jepa_scaling" / "summary.json"
PLOT_DIR = HERE.parent.parent / "docs" / "research" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

with open(SUM_PATH) as f:
    SUMMARY = json.load(f)

# ---- helpers --------------------------------------------------------------
def pair(prefix):
    """Return (jepa_runs, intention_runs) matching a name prefix like 'B_data'."""
    js = sorted([k for k in SUMMARY if k.startswith(prefix) and "_jepa" in k])
    ints = sorted([k for k in SUMMARY if k.startswith(prefix) and "_int" in k])
    return js, ints


def get(key, *path):
    d = SUMMARY[key]
    for p in path:
        d = d[p]
    return d


# ---- 1. overview bar grid -------------------------------------------------
def plot_overview():
    groups = [
        ("A baseline",     ["A_baseline_jepa", "A_baseline_int"]),
        ("B +data 2k",     ["B_data_2k_jepa", "B_data_2k_int"]),
        ("B +data 10k",    ["B_data_10k_jepa", "B_data_10k_int"]),
        ("C +params 50k",  ["C_params_50k_jepa_tx", "C_params_50k_int"]),
        ("C +params 200k", ["C_params_200k_jepa_tx", "C_params_200k_int"]),
        ("D +time 6k",     ["D_time_6k_jepa", "D_time_6k_int"]),
        ("D +time 20k",    ["D_time_20k_jepa", "D_time_20k_int"]),
        ("E scaled all 2k",["E_scale_all_jepa_tx", "E_scale_all_int"]),
        ("E scaled all 10k",["E_scale_all_jepa_tx_big", "E_scale_all_int_big"]),
        ("F pretrain+probe",["F_pretrain_probe_jepa_tx"]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for split_i, split in enumerate(["in", "out"]):
        ax = axes[split_i]
        x = []; vals_j = []; vals_i = []; ticks = []
        for i, (label, keys) in enumerate(groups):
            ticks.append(label)
            jkey = next((k for k in keys if "_jepa" in k), None)
            ikey = next((k for k in keys if "_int" in k), None)
            # If F pretrain-probe, use the probe variant for the bar
            if jkey and "pretrain_probe" in jkey:
                med = get(jkey, "in_probe" if split == "in" else "out_probe",
                          "r2_median")
            else:
                med = get(jkey, split, "r2_median") if jkey else np.nan
            imed = get(ikey, split, "r2_median") if ikey else np.nan
            x.append(i); vals_j.append(med); vals_i.append(imed)
        x = np.array(x)
        w = 0.38
        ax.bar(x - w/2, vals_j, width=w, color="#3c8e3c",
               edgecolor="black", linewidth=0.4, label="JEPA-FM")
        ax.bar(x + w/2, vals_i, width=w, color="#2c5d8a",
               edgecolor="black", linewidth=0.4, label="Intention_Learned")
        for xi, v in zip(x, vals_j):
            if not np.isnan(v):
                ax.text(xi - w/2, max(v, 0) + 0.02, f"{v:+.2f}",
                        ha="center", fontsize=7)
        for xi, v in zip(x, vals_i):
            if not np.isnan(v):
                ax.text(xi + w/2, max(v, 0) + 0.02, f"{v:+.2f}",
                        ha="center", fontsize=7)
        ax.set_ylim(-0.4, 1.1)
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_ylabel(f"{split}-box median R²")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(ticks, rotation=18, ha="right",
                                              fontsize=8)
        if split_i == 0:
            ax.set_title("JEPA-FM vs IntentionFM_Learned across scaling axes "
                         "(matched scenarios, identical seeds)", fontsize=11)
            ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_scaling_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---- 2-4. one-axis curves --------------------------------------------------
def _axis_plot(axis_keys, x_extractor, x_label, title, filename,
               x_log=True):
    """Generic R²-vs-axis plotter. axis_keys: list of (jepa_key, int_key) pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for split_i, split in enumerate(["in", "out"]):
        ax = axes[split_i]
        for kind, color, label in [("jepa", "#3c8e3c", "JEPA-FM"),
                                    ("int",  "#2c5d8a", "Intention_Learned")]:
            xs, ys, ps = [], [], []
            for jepa_k, int_k in axis_keys:
                key = jepa_k if kind == "jepa" else int_k
                xs.append(x_extractor(SUMMARY[key]))
                ys.append(get(key, split, "r2_median"))
                ps.append(get(key, split, "r2_p5"))
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]; ps = np.array(ps)[order]
            ax.plot(xs, ys, "o-", color=color, label=f"{label} median",
                    markersize=6, linewidth=1.5)
            ax.plot(xs, ps, "s--", color=color, label=f"{label} p5",
                    markersize=4, linewidth=1, alpha=0.7)
        ax.set_xscale("log" if x_log else "linear")
        ax.set_xlabel(x_label)
        ax.set_ylabel("R²")
        ax.set_title(f"{split}-box")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_ylim(-2.5, 1.05)
        if split_i == 1:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / filename
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_data_curve():
    pairs = [
        ("A_baseline_jepa",  "A_baseline_int"),
        ("B_data_2k_jepa",   "B_data_2k_int"),
        ("B_data_10k_jepa",  "B_data_10k_int"),
    ]
    _axis_plot(pairs,
               x_extractor=lambda r: r["config"]["n_train"],
               x_label="training scenarios",
               title="JEPA vs Intention — DATA scaling (fixed: 6k params, scaled steps)",
               filename="jepa_scaling_data.png", x_log=True)


def plot_params_curve():
    pairs = [
        ("A_baseline_jepa",       "A_baseline_int"),
        ("C_params_50k_jepa_tx",  "C_params_50k_int"),
        ("C_params_200k_jepa_tx", "C_params_200k_int"),
    ]
    _axis_plot(pairs,
               x_extractor=lambda r: r["n_params"],
               x_label="trainable parameters",
               title="JEPA vs Intention — PARAM scaling (fixed: 200 scenarios, 1500 steps)",
               filename="jepa_scaling_params.png", x_log=True)


def plot_time_curve():
    pairs = [
        ("A_baseline_jepa",  "A_baseline_int"),
        ("D_time_6k_jepa",   "D_time_6k_int"),
        ("D_time_20k_jepa",  "D_time_20k_int"),
    ]
    _axis_plot(pairs,
               x_extractor=lambda r: r["config"]["n_steps"],
               x_label="meta-training steps",
               title="JEPA vs Intention — TIME scaling (fixed: 6k params, 200 scenarios)",
               filename="jepa_scaling_time.png", x_log=True)


def plot_collapse():
    """Embedding collapse diagnostic across all JEPA runs."""
    jepa_keys = [k for k in SUMMARY if "_jepa" in k]
    rows = []
    for k in jepa_keys:
        r = SUMMARY[k]
        if "collapse_in" not in r:
            continue
        rows.append((k, r["n_params"], r["collapse_in"]["min_std"],
                     r["collapse_in"]["mean_std"],
                     r["collapse_in"].get("rank99", 0)))
    rows.sort(key=lambda x: x[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [r[0].replace("_jepa", "").replace("_tx", "") for r in rows]
    params = np.array([r[1] for r in rows])
    min_stds = np.array([r[2] for r in rows])
    mean_stds = np.array([r[3] for r in rows])
    ranks = np.array([r[4] for r in rows])
    ax = axes[0]
    ax.bar(np.arange(len(rows)), min_stds, color="#9c2c2c",
           edgecolor="black", linewidth=0.4, label="min std (per-dim)")
    ax.bar(np.arange(len(rows)), mean_stds, color="#3c8e3c", alpha=0.5,
           edgecolor="black", linewidth=0.4, label="mean std (per-dim)")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8,
               label="I-JEPA target std=1")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("embedding std")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title("Per-dim std of EMA-target embeddings", fontsize=10)
    ax = axes[1]
    ax.bar(np.arange(len(rows)), ranks, color="#2c5d8a",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("effective rank (singular values > 0.01)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title("Effective rank of embedding matrix", fontsize=10)
    fig.suptitle("Embedding-collapse diagnostics across JEPA scaling axes",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_scaling_collapse.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_overview()
    plot_data_curve()
    plot_params_curve()
    plot_time_curve()
    plot_collapse()
