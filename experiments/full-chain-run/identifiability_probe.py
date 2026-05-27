"""Linear-probe identifiability test for the Intention FM.

The headline FM result shows good predictive R^2 on mu(m), but prediction
quality alone does not separate "learned the SMEFT polynomial structure"
from "memorised the (M_ctx, Y_ctx) -> Y_q regression". The disclosure test
is a linear probe from the FM's IMPLICIT scenario representation back to
the Wilson coefficients c.

In the Intention setting, the implicit per-scenario latent is the closed-
form ridge weight vector

    w_implicit(c) = (Psi_ctx(c)^T Psi_ctx(c) + alpha I)^{-1} Psi_ctx(c)^T Y_ctx(c)

where psi_theta is the learned MLP basis and (M_ctx, Y_ctx) is a context
of K observations drawn from the c-conditional analytic SMEFT oracle.
w_implicit lives in R^{d_psi=16}.

The probe: fit a linear map c ~ W @ w_implicit + b on training scenarios,
report per-operator R^2 on held-out scenarios (both inside the training
c-box and inside the withheld c_lq^(3) band).

Threshold for the disclosure claim, per aletheia_verification_plan.md
Component 3 test 9: per-operator R^2 > 0.5 qualitative, R^2 > 0.8
strong evidence that the FM has disclosed the SMEFT structure.
"""
from __future__ import annotations

import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM

SEED = 5151
N_WC = 4
WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
WITHHOLD_DIM = 2                       # c_lq^(3)
WITHHOLD_BAND = (0.6, 1.0)
C_TRAIN_BOX = 0.7
M_RANGE = (0.3, 2.3)
K_CTX = 12

N_TRAIN_PROBE = 400                    # scenarios with training-box c
N_TEST_INSIDE = 200                    # held-out in-box test scenarios
N_TEST_WITHHELD = 200                  # held-out inside the withheld band

OUT_DIR = HERE / "output"
PLOT_DIR = HERE.parent.parent / "docs" / "research" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def sample_c_inbox(n: int, rng: np.random.Generator) -> np.ndarray:
    """c ~ U([-0.7, 0.7]^4) excluding |c_lq^(3)| in [0.6, 1.0]."""
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def sample_c_withheld(n: int, rng: np.random.Generator) -> np.ndarray:
    """c in the withheld band: c_lq^(3) in +/-[0.6, 1.0], rest U([-0.7,0.7])."""
    out = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=(n, N_WC))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[:, WITHHOLD_DIM] = signs * rng.uniform(*WITHHOLD_BAND, size=n)
    return out


def implicit_w(model: IntentionFM, c: np.ndarray, oracle, rng) -> np.ndarray:
    """Compute w_implicit(c) by drawing a fresh K=12 context for this c."""
    M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
    Y_ctx = oracle.truth(np.tile(c, (K_CTX, 1)), M_ctx)
    _, w, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    return w


def batch_implicit_w(model, cs, oracle, rng) -> np.ndarray:
    return np.array([implicit_w(model, c, oracle, rng) for c in cs])


def main():
    rng = np.random.default_rng(SEED)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)

    # Load pretrained Intention FM (1500 steps, withheld band excluded).
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    state = torch.load(OUT_DIR / "intention_fm.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print("=== Probe: training data ===")
    c_train = sample_c_inbox(N_TRAIN_PROBE, rng)
    w_train = batch_implicit_w(model, c_train, oracle, rng)
    print(f"  N_train={len(c_train)}, w shape={w_train.shape}")
    print(f"  c std: {c_train.std(0).round(3)}")
    print(f"  w std (per dim, first 5): {w_train.std(0)[:5].round(3)}")

    # Linear probe via Ridge with CV over alpha.
    print("\n=== Fitting linear probe ===")
    probe = RidgeCV(alphas=np.logspace(-4, 2, 25))
    probe.fit(w_train, c_train)
    print(f"  ridge alpha selected: {probe.alpha_:.4f}")
    c_train_pred = probe.predict(w_train)
    for i, name in enumerate(WC_NAMES):
        r2 = r2_score(c_train[:, i], c_train_pred[:, i])
        print(f"  train  {name:10s}  R^2 = {r2:+.4f}")

    print("\n=== Test: inside training box (held-out c) ===")
    c_in = sample_c_inbox(N_TEST_INSIDE, rng)
    w_in = batch_implicit_w(model, c_in, oracle, rng)
    c_in_pred = probe.predict(w_in)
    r2_in = {}
    for i, name in enumerate(WC_NAMES):
        r2 = r2_score(c_in[:, i], c_in_pred[:, i])
        r2_in[name] = float(r2)
        print(f"  inside {name:10s}  R^2 = {r2:+.4f}")

    print("\n=== Test: inside withheld band (extrapolation) ===")
    c_out = sample_c_withheld(N_TEST_WITHHELD, rng)
    w_out = batch_implicit_w(model, c_out, oracle, rng)
    c_out_pred = probe.predict(w_out)
    r2_out = {}
    for i, name in enumerate(WC_NAMES):
        r2 = r2_score(c_out[:, i], c_out_pred[:, i])
        r2_out[name] = float(r2)
        print(f"  withheld {name:10s}  R^2 = {r2:+.4f}")

    # Save numbers.
    summary = dict(
        ridge_alpha=float(probe.alpha_),
        n_train=N_TRAIN_PROBE,
        n_test_inside=N_TEST_INSIDE,
        n_test_withheld=N_TEST_WITHHELD,
        r2_inside=r2_in,
        r2_withheld=r2_out,
        median_inside=float(np.median(list(r2_in.values()))),
        median_withheld=float(np.median(list(r2_out.values()))),
        worst_inside=float(min(r2_in.values())),
        worst_withheld=float(min(r2_out.values())),
    )
    with open(OUT_DIR / "identifiability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))

    # Plot: scatter of predicted vs true c per operator, two columns
    # (in-box, withheld band).
    fig, axes = plt.subplots(2, N_WC, figsize=(15, 7.5),
                             constrained_layout=True)
    for col, name in enumerate(WC_NAMES):
        # Top row: inside training box.
        ax = axes[0, col]
        ax.scatter(c_in[:, col], c_in_pred[:, col], c="C0", s=12,
                   alpha=0.7, edgecolors="black", lw=0.3)
        lo, hi = c_in[:, col].min() - 0.05, c_in[:, col].max() + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6,
                label="y=x")
        ax.set_title(f"{name}\ninside-box  $R^2={r2_in[name]:+.3f}$")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("probe prediction")

        # Bottom row: withheld band.
        ax = axes[1, col]
        ax.scatter(c_out[:, col], c_out_pred[:, col], c="C3", s=12,
                   alpha=0.7, edgecolors="black", lw=0.3)
        lo, hi = (-1.05, 1.05) if col == WITHHOLD_DIM else (-0.75, 0.75)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y=x")
        ax.set_title(f"withheld-band  $R^2={r2_out[name]:+.3f}$")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("true $c$")
        if col == 0:
            ax.set_ylabel("probe prediction")

    fig.suptitle(
        "Linear-probe identifiability: do we recover $c$ from the FM's "
        "implicit scenario representation $w_\\theta$?\n"
        "Top row: inside training c-box. Bottom row: inside withheld band "
        f"$|c_{{lq}}^{{(3)}}| \\in [{WITHHOLD_BAND[0]}, {WITHHOLD_BAND[1]}]$.",
        fontsize=11)
    fig.savefig(PLOT_DIR / "identifiability_probe.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {PLOT_DIR / 'identifiability_probe.png'}")


if __name__ == "__main__":
    main()
