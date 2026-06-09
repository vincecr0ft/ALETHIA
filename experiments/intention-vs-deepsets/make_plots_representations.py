"""Plots for the representation-analysis study.

Reads:  output_representation_analysis/{summary.json, embeddings.npz}
Writes: docs/research/plots/repr_*.png

Five panels, each focused on a specific question about what each FM's
latent space encodes about the SMEFT generative model:

  1. repr_c_recoverability.png   — how much of each individual Wilson
                                    coefficient is recoverable from each
                                    representation (linear and MLP probe,
                                    in-box and out-box).
  2. repr_intrinsic_dim.png      — singular spectrum and effective rank
                                    of each representation.
  3. repr_pca_by_c.png           — PC1-PC2 of each representation,
                                    coloured by max|c_i|. Visualises whether
                                    the representation organises scenarios
                                    by Wilson-coefficient scale.
  4. repr_cross_decoding.png     — directed graph / matrix: how well does
                                    a linear map from representation X
                                    reproduce representation Y?
  5. repr_y_invariance.png       — cosine similarity of each representation
                                    under per-scenario y-shuffle (the
                                    rate-blindness diagnostic).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output_representation_analysis"
PLOT_DIR = HERE.parent.parent / "docs" / "research" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "summary.json") as f:
    SUMMARY = json.load(f)
EMB = np.load(OUT_DIR / "embeddings.npz")

REPS = ["Intention_w", "DeepSets_z", "JEPA_summary_m", "JEPA_summary_s"]
LABELS = {
    "Intention_w":     "Intention\nridge w (16-d)",
    "DeepSets_z":      "DeepSets\nmean-pool z (16-d)",
    "JEPA_summary_m":  "JEPA matched\nsummary (16-d)",
    "JEPA_summary_s":  "JEPA scaled\nsummary (64-d, tx)",
}
COLORS = {
    "Intention_w":     "#2c5d8a",
    "DeepSets_z":      "#d49a3c",
    "JEPA_summary_m":  "#7eb37e",
    "JEPA_summary_s":  "#3c8e3c",
}
WC_NAMES = ["cHq3", "cHq1", "clq3", "clq1"]


# ---- 1. c-recoverability ------------------------------------------------
def plot_c_recoverability():
    c_data = SUMMARY["c_recoverability"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax_i, (split, probe) in enumerate([("in", "linear"), ("out", "linear"),
                                            ("in", "mlp"),    ("out", "mlp")]):
        ax = axes[ax_i // 2, ax_i % 2]
        x = np.arange(len(REPS))
        # joint R²
        joint = [c_data[r][f"{probe}_{split}"]["r2_joint"] for r in REPS]
        per_c = np.array([c_data[r][f"{probe}_{split}"]["r2_per_c"] for r in REPS])
        w = 0.16
        ax.bar(x - 2*w, per_c[:, 0], width=w, color="#7e2c2c", label="cHq3")
        ax.bar(x - w,   per_c[:, 1], width=w, color="#c25151", label="cHq1")
        ax.bar(x,       per_c[:, 2], width=w, color="#ec7878", label="clq3")
        ax.bar(x + w,   per_c[:, 3], width=w, color="#f5a8a8", label="clq1")
        ax.bar(x + 2*w, joint,       width=w, color="black", alpha=0.75,
               label="joint", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[r] for r in REPS], fontsize=8)
        ax.set_ylabel(f"R² of c-recovery ({probe} probe, {split}-box)",
                      fontsize=9)
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_ylim(-0.4, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for xi, v in zip(x + 2*w, joint):
            ax.text(xi, max(v + 0.03, -0.35), f"{v:+.2f}", ha="center", fontsize=7)
        if ax_i == 0:
            ax.legend(fontsize=7, ncol=5, loc="upper center",
                      bbox_to_anchor=(0.5, 1.18))
        ax.set_title(f"{probe.upper()} probe — {split}-box", fontsize=10)
    fig.suptitle("c-recoverability per Wilson-coefficient: "
                 "what does each representation know about the latent c?",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = PLOT_DIR / "repr_c_recoverability.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


# ---- 2. intrinsic dimension --------------------------------------------
def plot_intrinsic_dim():
    diag = SUMMARY["diagnostics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # singular spectra
    ax = axes[0]
    for r in REPS:
        s = np.array(diag[r]["singular_values"])
        s_norm = s / s.max()
        ax.plot(np.arange(1, len(s) + 1), s_norm, "o-",
                color=COLORS[r], label=LABELS[r].replace("\n", " "),
                markersize=4, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("PCA component index")
    ax.set_ylabel("singular value (normalised to max)")
    ax.set_title("Singular spectrum — how concentrated is the variance?",
                 fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.axhline(0.01, color="k", linestyle="--", linewidth=0.7,
               label="effective-rank cutoff (1%)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(1e-6, 2)
    # bar chart of effective rank, PR, d_90, d_99
    ax = axes[1]
    metrics = ["effective_rank_1pct", "participation_ratio",
               "d_90pct_var", "d_99pct_var"]
    labels_m = ["eff. rank\n(SV >1%)", "participation\nratio",
                "d for 90%\nvariance", "d for 99%\nvariance"]
    x = np.arange(len(REPS))
    w = 0.2
    for j, (mk, lab) in enumerate(zip(metrics, labels_m)):
        vals = [diag[r][mk] for r in REPS]
        ax.bar(x + (j - 1.5) * w, vals, width=w, label=lab,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in REPS], fontsize=8)
    ax.set_ylabel("dimensions")
    ax.set_title("Intrinsic-dimension diagnostics", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("How many effective dimensions does each representation use?",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "repr_intrinsic_dim.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


# ---- 3. PCA scatter by c -----------------------------------------------
def plot_pca_by_c():
    c_in = EMB["c_in"]; c_out = EMB["c_out"]
    maxabs_in = np.max(np.abs(c_in), axis=1)
    maxabs_out = np.max(np.abs(c_out), axis=1)
    fig, axes = plt.subplots(1, len(REPS), figsize=(15, 4.2), sharey=False)
    for ax, r in zip(axes, REPS):
        R_in = EMB[f"{r}__in"]; R_out = EMB[f"{r}__out"]
        R_all = np.vstack([R_in, R_out])
        R_c = R_all - R_all.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(R_c, full_matrices=False)
        pc = R_c @ vt.T[:, :2]
        n_in = len(R_in)
        sc0 = ax.scatter(pc[:n_in, 0], pc[:n_in, 1], c=maxabs_in,
                         cmap="viridis", s=22, alpha=0.85,
                         edgecolor="black", linewidth=0.3, vmin=0, vmax=1)
        ax.scatter(pc[n_in:, 0], pc[n_in:, 1], c=maxabs_out,
                   cmap="viridis", marker="^", s=28, alpha=0.85,
                   edgecolor="black", linewidth=0.3, vmin=0, vmax=1)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(LABELS[r], fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("PC1-PC2 projection of per-scenario representations, "
                 "coloured by max|c_i| (circle=in-box, triangle=out-box)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
    cax = fig.add_axes([0.97, 0.15, 0.015, 0.7])
    plt.colorbar(sc0, cax=cax, label="max|c_i|")
    out = PLOT_DIR / "repr_pca_by_c.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


# ---- 4. cross-decoding matrix -----------------------------------------
def plot_cross_decoding():
    cd = SUMMARY["cross_decoding"]
    # square matrix: rows = source, cols = target
    n = len(REPS)
    M = np.full((n, n), np.nan)
    for i, src in enumerate(REPS):
        for j, tgt in enumerate(REPS):
            if src == tgt:
                M[i, j] = 1.0; continue
            k = f"{src}->{tgt}"
            if k in cd:
                M[i, j] = cd[k]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="R² of linear src → tgt fit")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=10)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[r] for r in REPS], rotation=20, ha="right",
                       fontsize=8)
    ax.set_yticklabels([LABELS[r] for r in REPS], fontsize=8)
    ax.set_xlabel("target representation"); ax.set_ylabel("source representation")
    ax.set_title("Cross-decoding R²: how much of column-rep does row-rep\n"
                 "linearly contain? (>0.9 = effectively equivalent up to rotation)",
                 fontsize=10)
    fig.tight_layout()
    out = PLOT_DIR / "repr_cross_decoding.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


# ---- 5. y-permutation invariance ---------------------------------------
def plot_y_invariance():
    yi = SUMMARY["y_permutation_invariance"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [yi[r] for r in REPS]
    colors = [COLORS[r] for r in REPS]
    x = np.arange(len(REPS))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8,
               label="cos=1 (rate-blind: y-shuffle doesn't change representation)")
    ax.axhline(0.0, color="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[r] for r in REPS], fontsize=8)
    ax.set_ylabel("cos similarity under per-scenario y-shuffle")
    ax.set_ylim(0, 1.1)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:+.2f}", ha="center", fontsize=9)
    for xi, v in zip(x, vals):
        note = "uses m-y alignment" if v < 0.95 else "rate-blind"
        ax.text(xi, -0.07, note, ha="center", fontsize=7, color="dimgray")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Rate-blindness diagnostic: does shuffling y across events of\n"
                 "the same scenario change the representation?", fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "repr_y_invariance.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_c_recoverability()
    plot_intrinsic_dim()
    plot_pca_by_c()
    plot_cross_decoding()
    plot_y_invariance()
