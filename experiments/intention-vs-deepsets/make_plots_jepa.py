"""Five diagnostic plots for the JEPA-FM addition to the intention-vs-deepsets comparison.

Reads:  output_jepa/results.npz, output_jepa/summary.json
Writes: docs/research/plots/jepa_fm_*.png (relative to the repo root)

The five panels:
  1. jepa_fm_bars.png           — in-box and out-box median + 5th-%-ile R² bar
                                   chart, all five architectures.
  2. jepa_fm_curves.png         — three held-out scenarios, JEPA-FM vs the
                                   reference architectures, on the same panels
                                   as the existing intention_vs_deepsets_curves
                                   layout.
  3. jepa_fm_ablations.png      — ablation table (jepa-only, aux-only, no-vicreg,
                                   no-ema, strong-aux) vs headline JEPA-FM.
  4. jepa_fm_embedding_pca.png  — PCA of E_ema(M_q, Y_q) on held-out queries,
                                   coloured by max|c_i|. Shows whether the
                                   JEPA target encoder organises the embedding
                                   space by Wilson-coefficient scale.
  5. jepa_fm_distance_vs_r2.png — per-scenario ‖z_pred − z_target‖ vs
                                   per-scenario R². Tests whether the JEPA
                                   pretext metric is predictive of downstream
                                   task quality.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output_jepa"
PLOT_DIR = HERE.parent.parent / "docs" / "research" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA = np.load(OUT_DIR / "results.npz", allow_pickle=True)
with open(OUT_DIR / "summary.json") as f:
    SUMMARY = json.load(f)


ARCH_ORDER = [
    ("IntentionFM_Fixed",            "fixed"),
    ("IntentionFM_Learned",          "int"),
    ("IntentionFM_Regressor_cheat",  "reg"),
    ("DeepSets_FM",                  "ds"),
    ("JEPA_FM",                      "jepa"),
]
ARCH_LABEL = {
    "IntentionFM_Fixed":           "Intention\n(fixed ψ)",
    "IntentionFM_Learned":         "Intention\n(learned ψ)",
    "IntentionFM_Regressor_cheat": "Regressor\n(c on f.p.)",
    "DeepSets_FM":                 "DeepSets\nFM",
    "JEPA_FM":                     "JEPA-FM\n(this work)",
}
ARCH_COLOR = {
    "IntentionFM_Fixed":           "#6a8caf",
    "IntentionFM_Learned":         "#2c5d8a",
    "IntentionFM_Regressor_cheat": "#b85450",
    "DeepSets_FM":                 "#d49a3c",
    "JEPA_FM":                     "#3c8e3c",
}


# ---- 1. Headline bar chart ------------------------------------------------
def plot_bars():
    keys = [k for k, _ in ARCH_ORDER]
    metrics = [
        ("in_median",  "in",  "r2_median"),
        ("in_p5",      "in",  "r2_p5"),
        ("out_median", "out", "r2_median"),
        ("out_p5",     "out", "r2_p5"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, (title, split, metric) in zip(axes, metrics):
        vals = [SUMMARY[k][split][metric] for k in keys]
        colors = [ARCH_COLOR[k] for k in keys]
        x = np.arange(len(keys))
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[k] for k in keys], fontsize=8)
        ax.set_title(title.replace("_", " "), fontsize=11)
        # clip the negative axis: DeepSets / JEPA p5 go very negative
        ax.set_ylim(-2.0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for xi, v in zip(x, vals):
            label = f"{v:+.3f}" if v > -1.99 else "<-2.0"
            ax.text(xi, max(v + 0.04, -1.95), label, ha="center", fontsize=7)
    axes[0].set_ylabel("R²", fontsize=10)
    fig.suptitle("JEPA-FM held-out R² vs Intention/DeepSets/Regressor (matched seed, K=12, Q=32)",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_fm_bars.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---- 2. Curves on three held-out scenarios -------------------------------
def _pick_three(c_arr: np.ndarray) -> list[int]:
    """Pick (smallest max|c|, median, largest) — matches intention-vs-deepsets pattern."""
    maxabs = np.max(np.abs(c_arr), axis=1)
    order = np.argsort(maxabs)
    n = len(order)
    return [int(order[1]), int(order[n // 2]), int(order[-2])]


def plot_curves():
    M_q = DATA["test_in_M_query"]
    Y_q = DATA["test_in_Y_query"]
    M_ctx = DATA["test_in_M_ctx"]
    Y_ctx = DATA["test_in_Y_ctx"]
    c = DATA["test_in_c"]
    picks = _pick_three(c)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    pred_keys = {
        "IntentionFM_Fixed":   "yp_fixed_in",
        "IntentionFM_Learned": "yp_int_in",
        "DeepSets_FM":         "yp_ds_in",
        "JEPA_FM":             "yp_jepa_in",
    }
    for ax, idx in zip(axes, picks):
        order = np.argsort(M_q[idx])
        ax.plot(M_q[idx][order], Y_q[idx][order],
                color="black", linewidth=1.4, label="ground truth")
        ax.scatter(M_ctx[idx], Y_ctx[idx], marker="x", color="black",
                   s=30, label="context")
        for arch, key in pred_keys.items():
            yp = DATA[key][idx]
            ax.plot(M_q[idx][order], yp[order],
                    color=ARCH_COLOR[arch], linewidth=1.2, alpha=0.85,
                    label=ARCH_LABEL[arch].replace("\n", " "))
        c_str = ", ".join(f"{v:+.2f}" for v in c[idx])
        ax.set_title(f"scenario {idx}  c=[{c_str}]", fontsize=9)
        ax.set_xlabel("m_ll (TeV)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Y")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Three held-out in-box scenarios — JEPA-FM tracks the slope but cannot lock onto curvature",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_fm_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---- 3. Ablation table figure --------------------------------------------
def plot_ablations():
    headline = ("headline", SUMMARY["JEPA_FM"]["in"]["r2_median"],
                SUMMARY["JEPA_FM"]["out"]["r2_median"],
                SUMMARY["JEPA_FM"]["collapse"]["min_std"])
    rows = [headline]
    for name, blk in SUMMARY["ablations"].items():
        rows.append((name, blk["in"]["r2_median"], blk["out"]["r2_median"],
                     blk["collapse"]["min_std"]))
    rows.append(("DeepSets-FM (ref)",
                 SUMMARY["DeepSets_FM"]["in"]["r2_median"],
                 SUMMARY["DeepSets_FM"]["out"]["r2_median"],
                 None))
    rows.append(("Intention_Learned (ref)",
                 SUMMARY["IntentionFM_Learned"]["in"]["r2_median"],
                 SUMMARY["IntentionFM_Learned"]["out"]["r2_median"],
                 None))
    names = [r[0] for r in rows]
    in_v = np.array([r[1] for r in rows])
    out_v = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.4
    bars_in = ax.bar(x - w/2, in_v, width=w, color="#3c8e3c", label="in-box median R²",
                     edgecolor="black", linewidth=0.4)
    bars_out = ax.bar(x + w/2, out_v, width=w, color="#7eb37e",
                      label="out-box median R²", edgecolor="black", linewidth=0.4)
    ax.axhline(SUMMARY["DeepSets_FM"]["in"]["r2_median"], linestyle="--",
               color="#d49a3c", linewidth=1, label="DeepSets in")
    ax.axhline(SUMMARY["IntentionFM_Learned"]["in"]["r2_median"], linestyle="--",
               color="#2c5d8a", linewidth=1, label="Intention in")
    for xi, name, mins in zip(x, names, [r[3] for r in rows]):
        if mins is not None:
            ax.text(xi, -0.08, f"min_std={mins:.2f}", ha="center", fontsize=7,
                    color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median R² on held-out scenarios")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("JEPA-FM ablations: aux MSE head is load-bearing; JEPA loss alone ≈ DeepSets",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_fm_ablations.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---- 4. PCA of EMA target embeddings --------------------------------------
def plot_embedding_pca():
    z_in = DATA["z_tgt_in"]          # (S_in, Q, d_emb)
    c_in = DATA["test_in_c"]         # (S_in, 4)
    z_out = DATA["z_tgt_out"]        # (S_out, Q, d_emb)
    c_out = DATA["test_out_c"]
    z_all = np.concatenate([z_in.mean(axis=1), z_out.mean(axis=1)], axis=0)  # per-scenario
    c_all = np.concatenate([c_in, c_out], axis=0)
    in_mask = np.zeros(len(z_all), dtype=bool); in_mask[:len(z_in)] = True
    z_centered = z_all - z_all.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    pc = z_centered @ Vt.T[:, :2]
    maxabs = np.max(np.abs(c_all), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sc0 = axes[0].scatter(pc[in_mask, 0], pc[in_mask, 1],
                          c=maxabs[in_mask], cmap="viridis",
                          s=22, alpha=0.85, edgecolor="black", linewidth=0.3)
    sc1 = axes[0].scatter(pc[~in_mask, 0], pc[~in_mask, 1],
                          c=maxabs[~in_mask], cmap="viridis",
                          marker="^", s=30, alpha=0.85, edgecolor="black",
                          linewidth=0.3)
    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
    axes[0].set_title("Per-scenario mean of E_ema(M_q,Y_q) — coloured by max|c_i|",
                      fontsize=10)
    axes[0].grid(alpha=0.3)
    plt.colorbar(sc0, ax=axes[0], label="max|c_i|")
    # right panel: singular values
    axes[1].bar(np.arange(len(S)) + 1, S, color="#3c8e3c",
                edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("PCA component")
    axes[1].set_ylabel("singular value")
    axes[1].set_title("Singular spectrum of JEPA target embeddings", fontsize=10)
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_fm_embedding_pca.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---- 5. Distance-vs-R² scatter --------------------------------------------
def plot_distance_vs_r2():
    d_in = DATA["dist_in"]
    d_out = DATA["dist_out"]
    r_in = DATA["r2_in"]
    r_out = DATA["r2_out"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d_in, r_in, color="#3c8e3c", s=28, alpha=0.8,
               edgecolor="black", linewidth=0.3, label="in-box")
    ax.scatter(d_out, r_out, color="#9c2c2c", marker="^", s=32, alpha=0.8,
               edgecolor="black", linewidth=0.3, label="out-box")
    # Pearson corr (treat all together)
    d_all = np.concatenate([d_in, d_out]); r_all = np.concatenate([r_in, r_out])
    finite = np.isfinite(r_all)
    corr = float(np.corrcoef(d_all[finite], r_all[finite])[0, 1])
    ax.set_xlabel("mean ‖z_pred − z_target‖₂ per scenario (JEPA pretext distance)")
    ax.set_ylabel("per-scenario R² of Y predictions")
    ax.set_title(f"Does JEPA's pretext metric predict downstream R²?  Pearson r = {corr:+.3f}",
                 fontsize=11)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(min(-3.0, r_all[finite].min() - 0.2), 1.05)
    fig.tight_layout()
    out = PLOT_DIR / "jepa_fm_distance_vs_r2.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return corr


if __name__ == "__main__":
    plot_bars()
    plot_curves()
    plot_ablations()
    plot_embedding_pca()
    corr = plot_distance_vs_r2()
    print(f"\nPearson(distance, R²) = {corr:+.3f}")
