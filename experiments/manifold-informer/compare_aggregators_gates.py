r"""Manifold-identity certificate across aggregators (the A.3 comparison).

Trains the MeanPool and Attention event-set FMs (manifold_informer_variants.py)
with the *identical* JEPA loss, data, and budget as the ManifoldInformer
(train_manifold_informer.py), then runs the SAME P1-P4 gates
(manifold_informer_gates.py) on all three models and tabulates the
certificate. The closed-form ridge ManifoldInformer is loaded from its
existing checkpoint.

This operationalises the memo's central claim (A.3: "no other FM ... can run
this test") into a measured one: does the closed-form ridge aggregator
recover the analytic tangent A_i (P3) and curvature B_ij (P4) where generic
mean-pool / attention aggregators, trained the same way on the same events,
do not? Per A.5 the result is reported as-is whichever way it falls.

Run:
    uv run --no-sync python experiments/manifold-informer/compare_aggregators_gates.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.features import N_WC
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

from manifold_informer import ManifoldInformer
from manifold_informer_variants import MeanPoolInformer, AttnInformer
from manifold_informer_gates import (
    gate_P2, gate_P3_P4, gate_P1_probe_linearity,
)
from train_manifold_informer import (
    get_or_make_dataset, train_model,
    D_EMB, D_HIDDEN, ALPHA_RIDGE, EMA_MOMENTUM, VICREG_W, VIEW_W, DENSITY_W,
    LR, BATCH_S, N_STEPS, SEED,
)

OUT_DIR = HERE / "output_aggregator_comparison"
OUT_DIR.mkdir(exist_ok=True)
RIDGE_CKPT = HERE / "output_manifold_informer" / "manifold_informer.pt"


def _common_kw():
    return dict(d_event=2, d_emb=D_EMB, hidden=D_HIDDEN,
                ema_momentum=EMA_MOMENTUM, vicreg_weight=VICREG_W,
                view_weight=VIEW_W, density_anchor_weight=DENSITY_W,
                use_ema=True)


def get_or_train_variant(name, factory, train, val):
    ckpt = OUT_DIR / f"{name}.pt"
    model = factory()
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"  [load] {name} ({model.n_params} params)")
        return model
    print(f"  [train] {name} ({model.n_params} params)")
    t0 = time.time()
    train_model(model, train, val, n_steps=N_STEPS, lr=LR,
                batch_s=BATCH_S, seed=SEED)
    torch.save(model.state_dict(), ckpt)
    print(f"    trained in {time.time()-t0:.1f}s")
    return model


def run_gates(model, oracle, seed=4242):
    """Run the three gate blocks; each gets its own fresh rng so the event
    seeds match across models (apples-to-apples)."""
    out = {}
    out["P2"] = gate_P2(model, oracle, np.random.default_rng(seed))
    out["P3_P4"] = gate_P3_P4(model, oracle, np.random.default_rng(seed + 1))
    return out


def main():
    print("# Manifold-identity certificate across aggregators\n")
    train, val = get_or_make_dataset()
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)

    # Fisher basis (shared) for the P1 probe-linearity gate.
    rng = np.random.default_rng(SEED)
    F = empirical_fisher_c(oracle,
                           rng.uniform(-0.6, 0.6, size=(400, N_WC)),
                           rng.uniform(0.3, 2.3, size=400), fd_step=1e-3)
    D_eig, V_fisher = fisher_basis(F)
    print(f"  Fisher eigenvalues: {D_eig.round(4).tolist()}  "
          f"(resolved directions: {int((D_eig > 1e-2).sum())}/{N_WC})\n")

    # ---- build the three arms ----
    models = {}
    print("# ManifoldInformer (closed-form ridge aggregator)")
    if not RIDGE_CKPT.exists():
        print(f"  ERROR: missing {RIDGE_CKPT}; run train_manifold_informer.py first")
        sys.exit(2)
    ridge = ManifoldInformer(**_common_kw())
    ridge.load_state_dict(torch.load(RIDGE_CKPT, weights_only=True))
    ridge.eval()
    print(f"  [load] ({ridge.n_params} params)")
    models["Ridge (ManifoldInformer)"] = ridge

    print("\n# Variant aggregators (same JEPA loss / data / budget)")
    models["MeanPool"] = get_or_train_variant(
        "MeanPool", lambda: MeanPoolInformer(**_common_kw()), train, val).eval()
    models["Attention"] = get_or_train_variant(
        "Attention", lambda: AttnInformer(**_common_kw()), train, val).eval()

    # ---- run the certificate on each ----
    rows = {}
    for name, model in models.items():
        print(f"\n{'='*70}\n# Gates: {name}\n{'='*70}")
        g = run_gates(model, oracle)
        p1 = gate_P1_probe_linearity(model, oracle, V_fisher,
                                     np.random.default_rng(4243))
        rows[name] = {
            "n_params": int(model.n_params),
            "P3_tangent_r2": g["P3_P4"]["P3_r2"],
            "P4_curvature_r2": g["P3_P4"]["P4_r2"],
            "P2_separation": g["P2"]["separation_ratio"],
            "P1_mse_linear": p1["mse_linear_c1"],
            "P1_mse_mlp": p1["mse_mlp_c1"],
            "P1_pass": p1["integrity_pass"],
            "P2_pass": g["P2"]["gate_pass"],
            "P3_pass": g["P3_P4"]["P3_pass"],
            "P4_pass": g["P3_P4"]["P4_pass"],
        }

    summary = {
        "fisher_eigenvalues": D_eig.tolist(),
        "resolved_directions": int((D_eig > 1e-2).sum()),
        "config": {"n_steps": N_STEPS, "batch_s": BATCH_S, "lr": LR,
                   "d_emb": D_EMB, "hidden": D_HIDDEN,
                   "vicreg_w": VICREG_W, "view_w": VIEW_W,
                   "density_w": DENSITY_W},
        "rows": rows,
    }
    (OUT_DIR / "certificate_comparison.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # ---- table ----
    print(f"\n\n{'#'*70}\n# Manifold-identity certificate — same JEPA, same events\n{'#'*70}")
    print(f"{'aggregator':<26}{'params':>7}{'P3 tan':>9}{'P4 curv':>9}"
          f"{'P2 sep':>8}{'P1 lin/mlp':>14}")
    for name, r in rows.items():
        p1s = f"{r['P1_mse_linear']:.3f}/{r['P1_mse_mlp']:.3f}"
        print(f"{name:<26}{r['n_params']:>7}{r['P3_tangent_r2']:>9.4f}"
              f"{r['P4_curvature_r2']:>9.4f}{r['P2_separation']:>8.2f}{p1s:>14}")
    print(f"\n# wrote {(OUT_DIR / 'certificate_comparison.json').relative_to(HERE.parent.parent)}")
    return summary


if __name__ == "__main__":
    main()
