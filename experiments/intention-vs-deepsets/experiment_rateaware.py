"""Vertex-disclosure attempt: train the rate-aware Intention head against
the analytic SMEFT oracle, then run the linear-probe identifiability test.

Compares to the baseline (un-rate-aware) Intention head trained at the
same configuration, using the same held-out probe scenarios.

Outputs:
    experiments/intention-vs-deepsets/output_rateaware/
      ├── rateaware_model.pt
      ├── baseline_model.pt
      ├── results.npz
      ├── identifiability.json
      └── identifiability_probe.png
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from data_smeft import (
    make_oracle, sample_c, sample_c_shell,
    make_dataset, scenarios_to_tensors,
    N_WC, M_RANGE,
)
from intention_learned import IntentionFMLearned
from intention_rateaware import IntentionFMRateAware


# Match experiment_smeft.py exactly so the comparison is fair.
N_TRAIN_SCENARIOS = 200
N_TEST_IN = 50
N_TEST_OUT = 50
K_CTX = 12
Q_QUERY = 32
C_MAX_TRAIN = 0.7
C_OUTER = 1.0
N_META_STEPS = 1500
LR = 1e-3
BATCH_S = 32
SEED = 0

# Identifiability probe config (matches the existing full-chain probe).
WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
WITHHOLD_DIM = 2  # c_lq^(3)
WITHHOLD_BAND = (0.6, 1.0)
N_PROBE_TRAIN = 400
N_PROBE_TEST_INSIDE = 200
N_PROBE_TEST_WITHHELD = 200

OUT_DIR = HERE / "output_rateaware"
OUT_DIR.mkdir(exist_ok=True)


def r2_per_scenario(y_pred, y_true):
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def train_model(model, train_t, val_t, name, log_every=200):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for step in range(1, N_META_STEPS + 1):
        idx = rng.choice(S_train, size=BATCH_S, replace=False)
        y_pred = model(train_t["M_ctx"][idx], train_t["Y_ctx"][idx],
                       train_t["M_query"][idx])
        loss = ((y_pred - train_t["Y_query"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 50, 100, 200, 500, 1000, 1500) or step % log_every == 0:
            model.eval()
            with torch.no_grad():
                yp = model(val_t["M_ctx"], val_t["Y_ctx"], val_t["M_query"]).cpu().numpy()
            r2 = r2_per_scenario(yp, val_t["Y_query"].cpu().numpy())
            print(f"  [{name}] step {step:5d}  loss={loss.item():.6f}  "
                  f"val_R2_median={float(np.median(r2)):+.4f}")
            model.train()
    print(f"  [{name}] wall = {time.time() - t0:.1f}s")
    model.eval()


def eval_held_out(model, t):
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


def sample_c_inbox_probe(n: int, rng: np.random.Generator) -> np.ndarray:
    """c ~ U([-0.7, 0.7]^4) excluding |c_lq^(3)| in [0.6, 1.0]."""
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_MAX_TRAIN, C_MAX_TRAIN, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def sample_c_withheld_probe(n: int, rng: np.random.Generator) -> np.ndarray:
    """c in the withheld band: c_lq^(3) in +/-[0.6, 1.0], rest U([-0.7,0.7])."""
    out = rng.uniform(-C_MAX_TRAIN, C_MAX_TRAIN, size=(n, N_WC))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[:, WITHHOLD_DIM] = signs * rng.uniform(*WITHHOLD_BAND, size=n)
    return out


def implicit_w_one(model, c, oracle, rng):
    """One scenario's implicit ridge weight."""
    M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
    Y_ctx = oracle.truth(np.tile(c, (K_CTX, 1)), M_ctx)
    _, w, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    return w


def implicit_w_batch(model, cs, oracle, rng):
    return np.array([implicit_w_one(model, c, oracle, rng) for c in cs])


def run_probe(model, label: str, oracle):
    rng = np.random.default_rng(5151)
    c_train = sample_c_inbox_probe(N_PROBE_TRAIN, rng)
    w_train = implicit_w_batch(model, c_train, oracle, rng)
    probe = RidgeCV(alphas=np.logspace(-4, 2, 25))
    probe.fit(w_train, c_train)

    c_in = sample_c_inbox_probe(N_PROBE_TEST_INSIDE, rng)
    w_in = implicit_w_batch(model, c_in, oracle, rng)
    c_in_pred = probe.predict(w_in)
    r2_inside = {name: float(r2_score(c_in[:, i], c_in_pred[:, i]))
                 for i, name in enumerate(WC_NAMES)}

    c_out = sample_c_withheld_probe(N_PROBE_TEST_WITHHELD, rng)
    w_out = implicit_w_batch(model, c_out, oracle, rng)
    c_out_pred = probe.predict(w_out)
    r2_withheld = {name: float(r2_score(c_out[:, i], c_out_pred[:, i]))
                   for i, name in enumerate(WC_NAMES)}

    print(f"\n=== {label} identifiability ===")
    print(f"  ridge alpha selected: {probe.alpha_:.4f}")
    print(f"  {'op':<8s} {'inside':>10s} {'withheld':>10s}")
    for name in WC_NAMES:
        print(f"  {name:<8s} {r2_inside[name]:+10.4f} {r2_withheld[name]:+10.4f}")
    return dict(
        r2_inside=r2_inside, r2_withheld=r2_withheld,
        ridge_alpha=float(probe.alpha_),
        c_in=c_in, c_in_pred=c_in_pred,
        c_out=c_out, c_out_pred=c_out_pred,
    )


def main():
    print("# Build SMEFT datasets")
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train_scen = make_dataset(N_TRAIN_SCENARIOS,
                              lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
                              oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=1)
    test_in_scen = make_dataset(N_TEST_IN,
                                lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
                                oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=42)
    test_out_scen = make_dataset(N_TEST_OUT,
                                 lambda r: sample_c_shell(r, 1,
                                                          c_inner=C_MAX_TRAIN,
                                                          c_outer=C_OUTER)[0],
                                 oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=43)
    train_t = scenarios_to_tensors(train_scen)
    test_in_t = scenarios_to_tensors(test_in_scen)
    test_out_t = scenarios_to_tensors(test_out_scen)

    # --- baseline (un-rate-aware) on the same SMEFT data ---
    print("\n# Train baseline IntentionFM_Learned")
    baseline = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    print(f"  n_params = {baseline.n_params}")
    train_model(baseline, train_t, test_in_t, "baseline")
    torch.save(baseline.state_dict(), OUT_DIR / "baseline_model.pt")
    yp_base_in = eval_held_out(baseline, test_in_t)
    yp_base_out = eval_held_out(baseline, test_out_t)

    # --- rate-aware variant ---
    print("\n# Train IntentionFM_RateAware")
    rate_aware = IntentionFMRateAware(d_psi=16, hidden=64, alpha=1e-3)
    print(f"  n_params = {rate_aware.n_params}")
    train_model(rate_aware, train_t, test_in_t, "rate-aware")
    torch.save(rate_aware.state_dict(), OUT_DIR / "rateaware_model.pt")
    yp_ra_in = eval_held_out(rate_aware, test_in_t)
    yp_ra_out = eval_held_out(rate_aware, test_out_t)

    Y_in = test_in_t["Y_query"].cpu().numpy()
    Y_out = test_out_t["Y_query"].cpu().numpy()
    r2_base_in = r2_per_scenario(yp_base_in, Y_in)
    r2_base_out = r2_per_scenario(yp_base_out, Y_out)
    r2_ra_in = r2_per_scenario(yp_ra_in, Y_in)
    r2_ra_out = r2_per_scenario(yp_ra_out, Y_out)

    print("\n# Held-out R^2 comparison")
    for name, r in (("baseline   in ", r2_base_in), ("baseline   out", r2_base_out),
                    ("rate-aware in ", r2_ra_in),   ("rate-aware out", r2_ra_out)):
        print(f"  {name}  median={float(np.median(r)):+.4f}  p5={float(np.percentile(r, 5)):+.4f}")

    np.savez(OUT_DIR / "results.npz",
             yp_base_in=yp_base_in, yp_base_out=yp_base_out,
             yp_ra_in=yp_ra_in, yp_ra_out=yp_ra_out,
             test_in_Y_query=Y_in, test_out_Y_query=Y_out,
             test_in_c=test_in_t["c"].cpu().numpy(),
             test_out_c=test_out_t["c"].cpu().numpy())

    # --- identifiability probe ---
    print("\n# Identifiability probes")
    base_probe = run_probe(baseline, "baseline", oracle)
    rate_probe = run_probe(rate_aware, "rate-aware", oracle)

    summary = dict(
        baseline=dict(
            r2_inside=base_probe["r2_inside"],
            r2_withheld=base_probe["r2_withheld"],
            held_out={
                "median_in": float(np.median(r2_base_in)),
                "median_out": float(np.median(r2_base_out)),
                "p5_in": float(np.percentile(r2_base_in, 5)),
                "p5_out": float(np.percentile(r2_base_out, 5)),
            },
        ),
        rate_aware=dict(
            r2_inside=rate_probe["r2_inside"],
            r2_withheld=rate_probe["r2_withheld"],
            held_out={
                "median_in": float(np.median(r2_ra_in)),
                "median_out": float(np.median(r2_ra_out)),
                "p5_in": float(np.percentile(r2_ra_in, 5)),
                "p5_out": float(np.percentile(r2_ra_out, 5)),
            },
        ),
    )
    with open(OUT_DIR / "identifiability.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote identifiability.json")

    # --- plot ---
    fig, axes = plt.subplots(2, N_WC, figsize=(15, 7.5), constrained_layout=True)
    for col, name in enumerate(WC_NAMES):
        ax = axes[0, col]
        ax.scatter(base_probe["c_in"][:, col], base_probe["c_in_pred"][:, col],
                   c="C0", s=10, alpha=0.6, edgecolors="black", lw=0.2,
                   label=f"baseline R^2={base_probe['r2_inside'][name]:+.3f}")
        ax.scatter(rate_probe["c_in"][:, col], rate_probe["c_in_pred"][:, col],
                   c="C1", s=10, alpha=0.6, edgecolors="black", lw=0.2,
                   label=f"rate-aware R^2={rate_probe['r2_inside'][name]:+.3f}")
        lo, hi = -0.75, 0.75
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(f"{name}: inside training box"); ax.legend(fontsize=8)
        if col == 0:
            ax.set_ylabel("probe prediction")

        ax = axes[1, col]
        ax.scatter(base_probe["c_out"][:, col], base_probe["c_out_pred"][:, col],
                   c="C0", s=10, alpha=0.6, edgecolors="black", lw=0.2,
                   label=f"baseline R^2={base_probe['r2_withheld'][name]:+.3f}")
        ax.scatter(rate_probe["c_out"][:, col], rate_probe["c_out_pred"][:, col],
                   c="C1", s=10, alpha=0.6, edgecolors="black", lw=0.2,
                   label=f"rate-aware R^2={rate_probe['r2_withheld'][name]:+.3f}")
        lo, hi = (-1.05, 1.05) if col == WITHHOLD_DIM else (-0.75, 0.75)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(f"{name}: withheld band"); ax.legend(fontsize=8)
        ax.set_xlabel("true c")
        if col == 0:
            ax.set_ylabel("probe prediction")

    fig.suptitle(
        "Linear-probe identifiability: baseline IntentionFM vs rate-aware variant "
        "(analytic-SMEFT oracle, identical training configuration)",
        fontsize=11)
    fig.savefig(OUT_DIR / "identifiability_probe.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR / 'identifiability_probe.png'}")


if __name__ == "__main__":
    main()
