r"""Recovery money shot (paper Figure B2): the latent reconstructs the analytic
SMEFT morphing templates.

Loads the existing trained ManifoldInformer checkpoint (no retraining) and runs
the P3/P4 probe (manifold_informer_gates.gate_P3_P4), which regresses the
encoder's Wilson-coefficient tangents/curvatures onto the analytic morphing
templates A_i, B_ij (each reduced to a scalar per operator / operator-pair by the
m-average the probe uses). Dumps the recovered-vs-analytic arrays and plots the
on-diagonal scatter.

This is a single trained model. The paper's headline R^2 = 0.99999 (tangents) /
0.954 (curvatures) are medians over 15 retrained seeds (gates_multiseed.py); the
single-seed R^2 computed here is printed and annotated so the figure is honest
about which number it shows.

OMP_NUM_THREADS=2 .venv/bin/python experiments/manifold-informer/recovery_money_shot.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import manifold_informer_gates as GA
from manifold_informer import ManifoldInformer
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.features import N_WC, WC_NAMES

OUT = HERE / "output_recovery"; OUT.mkdir(exist_ok=True)

model = ManifoldInformer(d_event=2, d_emb=16, hidden=32, alpha=1e-3, use_ema=True)
model.load_state_dict(torch.load(GA.MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print(f"loaded {GA.MODEL_PATH}")

rng = np.random.default_rng(GA.SEED)
oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
res = GA.gate_P3_P4(model, oracle, rng)

A_scalar = np.array(res["A_scalar"]);  A_pred = np.array(res["A_pred"])
B_ref = np.array(res["B_ref_flat"]);   B_pred = np.array(res["B_pred"])
p3, p4 = res["P3_r2"], res["P4_r2"]
print(f"single-seed: P3_r2={p3:.5f}  P4_r2={p4:.4f}")

# operator and pair labels
tex = {"cHq3": r"$c_{Hq}^{(3)}$", "cHq1": r"$c_{Hq}^{(1)}$",
       "clq3": r"$c_{\ell q}^{(3)}$", "clq1": r"$c_{\ell q}^{(1)}$"}
op_lbl = [tex.get(n, n) for n in WC_NAMES]
iu = np.triu_indices(N_WC)
pair_lbl = [f"{WC_NAMES[i]},{WC_NAMES[j]}" for i, j in zip(*iu)]

json.dump({"A_scalar": A_scalar.tolist(), "A_pred": A_pred.tolist(),
           "B_ref_flat": B_ref.tolist(), "B_pred": B_pred.tolist(),
           "op_labels": list(WC_NAMES), "pair_labels": pair_lbl,
           "P3_r2_single_seed": p3, "P4_r2_single_seed": p4},
          open(OUT / "recovery_arrays.json", "w"), indent=2)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
for ax, x, y, ttl, r2 in [
        (axa, A_scalar, A_pred, "tangents $A_i$ (interference)", p3),
        (axb, B_ref, B_pred, "curvatures $B_{ij}$ (BSM$^2$)", p4)]:
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    pad = 0.08 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.6)
    ax.scatter(x, y, s=46, c="C3", zorder=3, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("analytic template"); ax.set_ylabel("probe-recovered")
    ax.set_title(ttl); ax.grid(alpha=0.3)
    ax.text(0.05, 0.92, f"$R^2={r2:.5f}$" if r2 > 0.99 else f"$R^2={r2:.3f}$",
            transform=ax.transAxes, fontsize=10, va="top")
for i, l in enumerate(op_lbl):
    axa.annotate(l, (A_scalar[i], A_pred[i]), fontsize=7,
                 xytext=(4, 3), textcoords="offset points")
fig.suptitle("The latent reconstructs the analytic SMEFT morphing templates "
             "(single trained model)", fontsize=11)
fig.savefig(OUT / "recovery.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT}/recovery.png and recovery_arrays.json")
