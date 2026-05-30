"""Scaling sweep over ``d_psi`` for the Intention FM (U11).

The ML audit notes that the paper invokes "scaling-behaviour analyses
now standard in the jet-level foundation model literature" but performs
none. This sweep retrains the headline IntentionFM at six values of
``d_psi`` and reports both held-out R^2 on the standard 50-in / 50-out
benchmark *and* per-operator disclosure R^2 from the linear probe on
the implicit ridge weight ``w``. The single-figure output is one panel
of Section 7.

Why both metrics in one sweep: a model that scales on held-out R^2 but
not on disclosure R^2 is an interesting story (more capacity = better
fit but no extra structural understanding); a model that scales on
both is the cleaner foundation-model claim. We don't pre-commit either
outcome — the sweep produces whichever curve falls out.

Protocol per d_psi: identical to ``experiments/intention-vs-deepsets/
experiment_smeft.py`` (N=200 training scenarios, K=12, Q=32, 1500
Adam steps, lr=1e-3, batch=24, ``c_lq^(3)`` band withheld).

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/full-chain-run/scaling_sweep.py
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
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from torch.optim import Adam

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM


SEED = 2026
N_WC = 4
WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
WITHHOLD_DIM = 2
WITHHOLD_BAND = (0.6, 1.0)
C_TRAIN_BOX = 0.7
M_RANGE = (0.3, 2.3)
K_CTX = 12
Q_CTX = 32

# Headline-matched training schedule (cf. experiment_smeft.py).
N_PRETRAIN_SCEN = 1500
PRETRAIN_BATCH = 24
PRETRAIN_STEPS = 1500
LR = 1e-3

# Disclosure-probe schedule (cf. identifiability_probe.py).
N_PROBE_TRAIN = 400
N_PROBE_TEST_IN = 200
N_PROBE_TEST_OUT = 200

# Held-out R^2 schedule.
N_HELDOUT_IN = 50
N_HELDOUT_OUT = 50

D_PSI_SWEEP = tuple(
    int(x) for x in os.environ.get("D_PSI_SWEEP", "4,8,16,32,64,128").split(",")
)

OUT = HERE / "output_scaling"
OUT.mkdir(parents=True, exist_ok=True)


def sample_c_training(n, rng):
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def sample_c_withheld(n, rng):
    out = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=(n, N_WC))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[:, WITHHOLD_DIM] = signs * rng.uniform(*WITHHOLD_BAND, size=n)
    return out


def heldout_r2(model, oracle, cs, rng):
    """Median R^2 across scenarios in ``cs``. Each scenario uses a fresh
    K=12 context and Q=32 query at independent m-draws."""
    r2s = []
    for c in cs:
        M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
        M_q = rng.uniform(*M_RANGE, size=Q_CTX)
        Y_ctx = oracle.truth(np.tile(c, (K_CTX, 1)), M_ctx)
        Y_q = oracle.truth(np.tile(c, (Q_CTX, 1)), M_q)
        yp = model.predict_np(M_ctx, Y_ctx, M_q)
        ss_res = float(np.sum((yp - Y_q) ** 2))
        ss_tot = float(np.sum((Y_q - Y_q.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        r2s.append(r2)
    return r2s


def disclosure_r2(model, oracle, rng):
    """Per-operator R^2 on inside and withheld bands, using the
    implicit ridge weight w as the probe input."""

    def implicit_w_batch(cs):
        ws = []
        for c in cs:
            M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
            Y_ctx = oracle.truth(np.tile(c, (K_CTX, 1)), M_ctx)
            _, w, _ = model.A_inv_and_w(M_ctx, Y_ctx)
            ws.append(w)
        return np.array(ws)

    # Train probe on in-box scenarios.
    c_tr = sample_c_training(N_PROBE_TRAIN, rng)
    w_tr = implicit_w_batch(c_tr)
    probe = RidgeCV(alphas=np.logspace(-4, 2, 25))
    probe.fit(w_tr, c_tr)

    c_in = sample_c_training(N_PROBE_TEST_IN, rng)
    w_in = implicit_w_batch(c_in)
    c_in_pred = probe.predict(w_in)
    r2_in = {n: float(r2_score(c_in[:, i], c_in_pred[:, i]))
             for i, n in enumerate(WC_NAMES)}

    c_out = sample_c_withheld(N_PROBE_TEST_OUT, rng)
    w_out = implicit_w_batch(c_out)
    c_out_pred = probe.predict(w_out)
    r2_out = {n: float(r2_score(c_out[:, i], c_out_pred[:, i]))
              for i, n in enumerate(WC_NAMES)}

    return r2_in, r2_out, float(probe.alpha_)


def pretrain(d_psi: int, oracle, rng) -> IntentionFM:
    torch.manual_seed(SEED)
    model = IntentionFM(d_psi=d_psi, hidden=64, alpha=1e-3)
    opt = Adam(model.parameters(), lr=LR)
    c_pool = sample_c_training(N_PRETRAIN_SCEN, rng)
    for step in range(PRETRAIN_STEPS):
        idx = rng.choice(N_PRETRAIN_SCEN, size=PRETRAIN_BATCH, replace=False)
        cs = c_pool[idx]
        M_ctx = rng.uniform(*M_RANGE, size=(PRETRAIN_BATCH, K_CTX))
        M_q = rng.uniform(*M_RANGE, size=(PRETRAIN_BATCH, Q_CTX))
        C_ctx = np.repeat(cs, K_CTX, axis=0)
        C_q = np.repeat(cs, Q_CTX, axis=0)
        Y_ctx = oracle.truth(C_ctx, M_ctx.reshape(-1)
                             ).reshape(PRETRAIN_BATCH, K_CTX)
        Y_q = oracle.truth(C_q, M_q.reshape(-1)
                           ).reshape(PRETRAIN_BATCH, Q_CTX)
        y_pred = model(
            torch.from_numpy(M_ctx).float(),
            torch.from_numpy(Y_ctx).float(),
            torch.from_numpy(M_q).float(),
        )
        loss = F.mse_loss(y_pred, torch.from_numpy(Y_q).float())
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def n_params(model: IntentionFM) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print(f"# Scaling sweep over d_psi: {D_PSI_SWEEP}")
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    rng = np.random.default_rng(SEED)

    # Held-out scenario pools (shared across d_psi for a fair comparison).
    c_heldout_in = sample_c_training(N_HELDOUT_IN, rng)
    c_heldout_out = sample_c_withheld(N_HELDOUT_OUT, rng)

    results = {}
    for d_psi in D_PSI_SWEEP:
        print(f"\n--- d_psi = {d_psi} ---")
        t0 = time.time()
        model = pretrain(d_psi, oracle, rng)
        wall = time.time() - t0
        np_ = n_params(model)
        print(f"  trained {np_} params in {wall:.1f}s")

        # Held-out R^2.
        rng_h = np.random.default_rng(SEED + d_psi)
        r2_in = heldout_r2(model, oracle, c_heldout_in, rng_h)
        r2_out = heldout_r2(model, oracle, c_heldout_out, rng_h)
        print(f"  in  : median={np.median(r2_in):+.4f}  "
              f"p5={np.percentile(r2_in, 5):+.4f}")
        print(f"  out : median={np.median(r2_out):+.4f}  "
              f"p5={np.percentile(r2_out, 5):+.4f}")

        # Disclosure R^2 — fresh probe per scale.
        rng_d = np.random.default_rng(SEED + 1000 + d_psi)
        disc_in, disc_out, probe_alpha = disclosure_r2(model, oracle, rng_d)
        print(f"  disclosure inside  : {disc_in}")
        print(f"  disclosure withheld: {disc_out}")

        results[d_psi] = dict(
            n_params=np_, wall_seconds=wall,
            heldout=dict(
                r2_median_in=float(np.median(r2_in)),
                r2_p5_in=float(np.percentile(r2_in, 5)),
                r2_median_out=float(np.median(r2_out)),
                r2_p5_out=float(np.percentile(r2_out, 5)),
                r2_per_scenario_in=r2_in,
                r2_per_scenario_out=r2_out,
            ),
            disclosure=dict(
                inside=disc_in, withheld=disc_out, probe_alpha=probe_alpha,
            ),
        )
        # Checkpoint per scale so the sweep is resumable.
        torch.save(model.state_dict(), OUT / f"intention_fm_d{d_psi}.pt")

    with open(OUT / "scaling_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot — two-panel figure.
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                                  constrained_layout=True)
        ds = list(results.keys())
        med_in = [results[d]["heldout"]["r2_median_in"] for d in ds]
        med_out = [results[d]["heldout"]["r2_median_out"] for d in ds]
        p5_in = [results[d]["heldout"]["r2_p5_in"] for d in ds]
        p5_out = [results[d]["heldout"]["r2_p5_out"] for d in ds]
        axes[0].plot(ds, med_in, "o-", label="median in", color="C0")
        axes[0].plot(ds, med_out, "s-", label="median out", color="C1")
        axes[0].plot(ds, p5_in, "o--", label="p5 in", color="C0", alpha=0.5)
        axes[0].plot(ds, p5_out, "s--", label="p5 out", color="C1", alpha=0.5)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel(r"$d_\psi$")
        axes[0].set_ylabel(r"held-out $R^2$")
        axes[0].set_title("Held-out R^2 vs d_psi")
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        for name in WC_NAMES:
            ys = [results[d]["disclosure"]["withheld"][name] for d in ds]
            axes[1].plot(ds, ys, "o-", label=name)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel(r"$d_\psi$")
        axes[1].set_ylabel(r"disclosure $R^2$ (withheld band)")
        axes[1].set_title("Per-operator disclosure vs d_psi")
        axes[1].grid(True, alpha=0.3); axes[1].legend()

        plot_dir = HERE.parent.parent / "docs" / "research" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_path = plot_dir / "scaling_sweep.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
