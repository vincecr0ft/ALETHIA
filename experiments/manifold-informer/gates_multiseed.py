r"""T2.1 — Multi-seed robustness bands on the P1-P4 ManifoldInformer gates.

The single-seed gate run (manifold_informer_gates.py, SEED=2026 for training
+ SEED=4242 for eval) reports point estimates that feed Table 2 of the paper.
The P1 readout-linearity margin (linear MSE 0.059 <= MLP MSE 0.073) is THIN.
This wrapper answers: does P1 pass robustly across seeds, or only at the
favourable SEED=2026?

For each seed s in [SEED_LO, SEED_HI):
  1. regenerate the training + validation event datasets with sampling seed s
     (so BOTH the c-draws and the per-scenario event seeds move);
  2. construct a fresh ManifoldInformer and train it with torch seed s
     (train_model already seeds torch.manual_seed(seed), the numpy step-batch
     RNG, and the ctx/query split generator off `seed`);
  3. run the P1, P2, P3, P4 gate computations on that model with a per-seed
     eval RNG.

We collect, per seed: P1 linear MSE, P1 MLP MSE, P1 pass (linear <= 1.5*MLP),
P2 ratio, P3 R², P4 R². Report median + [p25,p75] + min/max across seeds and
the FRACTION of seeds where P1 passes.

The dataset is regenerated per seed in-memory and NOT cached to the shared
output dir (the shared npz cache is keyed only by size, not seed, so reusing
it would defeat the multi-seed point). Each trained model is written to a
per-seed temp path under output_manifold_informer_gates/_seed_models/.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    OMP_NUM_THREADS=1 uv run python experiments/manifold-informer/gates_multiseed.py
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
import train_manifold_informer as TMI
import manifold_informer_gates as GATES


# ---------------------------------------------------------------------------
# Configuration — mirror the single-seed training/eval config exactly so the
# only thing that moves between runs is the seed.
# ---------------------------------------------------------------------------
SEED_LO = 2026
SEED_HI = 2041                 # 15 seeds: 2026..2040 inclusive
N_SEEDS = SEED_HI - SEED_LO

OUT_DIR = HERE / "output_manifold_informer_gates"
OUT_DIR.mkdir(exist_ok=True)
MODEL_TMP_DIR = OUT_DIR / "_seed_models"
MODEL_TMP_DIR.mkdir(exist_ok=True)

# Training config (copied from train_manifold_informer.py module globals).
D_EMB = TMI.D_EMB
D_HIDDEN = TMI.D_HIDDEN
ALPHA_RIDGE = TMI.ALPHA_RIDGE
EMA_MOMENTUM = TMI.EMA_MOMENTUM
VICREG_W = TMI.VICREG_W
VIEW_W = TMI.VIEW_W
DENSITY_W = TMI.DENSITY_W
LR = TMI.LR
BATCH_S = TMI.BATCH_S
N_STEPS = TMI.N_STEPS
N_SCENARIOS_TRAIN = TMI.N_SCENARIOS_TRAIN
N_SCENARIOS_VAL = TMI.N_SCENARIOS_VAL
N_EVENTS_PER_VIEW = TMI.N_EVENTS_PER_VIEW
C_BOX_TRAIN = TMI.C_BOX_TRAIN


# ---------------------------------------------------------------------------
def train_one_seed(seed: int) -> ManifoldInformer:
    """Regenerate data with sampling seed `seed`, train a fresh model with
    torch/init seed `seed`. Returns the trained, eval-mode model."""
    # Data: vary the sampling seed (drives both c-draws and event seeds).
    train = TMI._generate_dataset(N_SCENARIOS_TRAIN, C_BOX_TRAIN,
                                  N_EVENTS_PER_VIEW, seed=seed)
    val = TMI._generate_dataset(N_SCENARIOS_VAL, C_BOX_TRAIN,
                                N_EVENTS_PER_VIEW, seed=seed + 10_000)

    # Model: fresh init; torch.manual_seed(seed) inside train_model controls
    # weight init order via Adam + the deepcopy'd EMA target. To make the
    # *parameter init itself* seed-dependent we seed torch before construction.
    torch.manual_seed(seed)
    model = ManifoldInformer(
        d_event=2, d_emb=D_EMB, hidden=D_HIDDEN, alpha=ALPHA_RIDGE,
        ema_momentum=EMA_MOMENTUM, vicreg_weight=VICREG_W,
        view_weight=VIEW_W, density_anchor_weight=DENSITY_W, use_ema=True,
    )
    TMI.train_model(model, train, val, n_steps=N_STEPS, lr=LR,
                    batch_s=BATCH_S, seed=seed)
    model.eval()
    return model


def run_gates_one_seed(model: ManifoldInformer, eval_seed: int) -> dict:
    """Run P1/P2/P3/P4 on a trained model with a per-seed eval RNG.

    Reuses the exact gate functions from manifold_informer_gates.py. The
    Fisher basis is rebuilt per eval seed (it is a function of the oracle
    prior draw, matching the single-seed script which builds it from the
    eval RNG)."""
    rng = np.random.default_rng(eval_seed)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)

    c_prior = rng.uniform(-0.6, 0.6, size=(400, N_WC))
    m_prior = rng.uniform(0.3, 2.3, size=400)
    F = empirical_fisher_c(oracle, c_prior, m_prior, fd_step=1e-3)
    _, V_fisher = fisher_basis(F)

    p2 = GATES.gate_P2(model, oracle, rng)
    p34 = GATES.gate_P3_P4(model, oracle, rng)
    p1 = GATES.gate_P1_probe_linearity(model, oracle, V_fisher, rng)
    return {
        "P1_mse_linear": p1["mse_linear_c1"],
        "P1_mse_mlp": p1["mse_mlp_c1"],
        # Paper's stated P1 question is "linear <= MLP" (the THIN margin).
        # Report BOTH the strict (<=) and the gate's 1.5x-slack pass.
        "P1_pass_strict": bool(p1["mse_linear_c1"] <= p1["mse_mlp_c1"]),
        "P1_pass_gate": bool(p1["integrity_pass"]),
        "P2_ratio": p2["separation_ratio"],
        "P2_pass": p2["gate_pass"],
        "P3_r2": p34["P3_r2"],
        "P3_pass": p34["P3_pass"],
        "P4_r2": p34["P4_r2"],
        "P4_pass": p34["P4_pass"],
    }


def _agg(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def main():
    print(f"# Multi-seed P1-P4 gates  ({N_SEEDS} seeds: {SEED_LO}..{SEED_HI-1})")
    t0 = time.time()
    per_seed = []
    for seed in range(SEED_LO, SEED_HI):
        ts = time.time()
        print(f"\n{'='*70}\n# SEED {seed}\n{'='*70}", flush=True)
        model = train_one_seed(seed)
        torch.save(model.state_dict(), MODEL_TMP_DIR / f"mi_seed{seed}.pt")
        # Eval seed offset from train seed so the eval draw is decoupled but
        # still deterministic per training seed.
        row = run_gates_one_seed(model, eval_seed=seed + 100_000)
        row["seed"] = seed
        row["wall_seconds"] = time.time() - ts
        per_seed.append(row)
        print(f"\n  [seed {seed}] P1 lin={row['P1_mse_linear']:.4f} "
              f"mlp={row['P1_mse_mlp']:.4f} "
              f"strict_pass={row['P1_pass_strict']} | "
              f"P2={row['P2_ratio']:.2f} P3_R2={row['P3_r2']:.6f} "
              f"P4_R2={row['P4_r2']:.4f}  ({row['wall_seconds']:.1f}s)",
              flush=True)

    # Aggregates.
    keys = ["P1_mse_linear", "P1_mse_mlp", "P2_ratio", "P3_r2", "P4_r2"]
    aggregates = {k: _agg([r[k] for r in per_seed]) for k in keys}
    n = len(per_seed)
    aggregates["P1_pass_fraction_strict"] = sum(
        r["P1_pass_strict"] for r in per_seed) / n
    aggregates["P1_pass_fraction_gate"] = sum(
        r["P1_pass_gate"] for r in per_seed) / n
    aggregates["P2_pass_fraction"] = sum(r["P2_pass"] for r in per_seed) / n
    aggregates["P3_pass_fraction"] = sum(r["P3_pass"] for r in per_seed) / n
    aggregates["P4_pass_fraction"] = sum(r["P4_pass"] for r in per_seed) / n

    summary = {
        "n_seeds": n,
        "seed_range": [SEED_LO, SEED_HI - 1],
        "config": {
            "d_emb": D_EMB, "hidden": D_HIDDEN, "alpha": ALPHA_RIDGE,
            "n_steps": N_STEPS, "batch_s": BATCH_S, "lr": LR,
            "n_train_scen": N_SCENARIOS_TRAIN, "n_val_scen": N_SCENARIOS_VAL,
            "n_events_per_view": N_EVENTS_PER_VIEW, "c_box": C_BOX_TRAIN,
            "P1_gate_def": "strict: linear<=MLP ; gate: linear<=1.5*MLP",
            "P3_probe": "RidgeCV, 4 samples (N_WC) x 16 features (d_emb)",
            "P4_probe": "RidgeCV, 10 samples (N_WC*(N_WC+1)/2) x 16 features",
        },
        "aggregates": aggregates,
        "per_seed": per_seed,
        "wall_seconds": time.time() - t0,
    }
    out_path = OUT_DIR / "gates_multiseed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Console report.
    print(f"\n\n{'='*70}\n# AGGREGATES across {n} seeds\n{'='*70}")
    def fmt(k):
        a = aggregates[k]
        return (f"  {k:16s} median={a['median']:.5f} "
                f"[p25={a['p25']:.5f}, p75={a['p75']:.5f}] "
                f"min={a['min']:.5f} max={a['max']:.5f}")
    for k in keys:
        print(fmt(k))
    print(f"\n  P1 pass fraction (strict linear<=MLP) : "
          f"{aggregates['P1_pass_fraction_strict']:.2f} "
          f"({sum(r['P1_pass_strict'] for r in per_seed)}/{n})")
    print(f"  P1 pass fraction (gate linear<=1.5MLP): "
          f"{aggregates['P1_pass_fraction_gate']:.2f} "
          f"({sum(r['P1_pass_gate'] for r in per_seed)}/{n})")
    print(f"  P2 pass fraction: {aggregates['P2_pass_fraction']:.2f}")
    print(f"  P3 pass fraction: {aggregates['P3_pass_fraction']:.2f}")
    print(f"  P4 pass fraction: {aggregates['P4_pass_fraction']:.2f}")
    print(f"\n# wrote {out_path}  (wall = {summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
