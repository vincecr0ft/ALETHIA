"""Supercharged vertex-disclosure attempt.

The toy-PDF, d_psi=16, 200-scenario, 1500-step baseline leaves the two
vertex operators (cHq3, cHq1) and the second four-fermion operator
(clq1) under-disclosed. This script tests four levers at once:

  1. CT18NNLO PDFs in place of the toy analytic PDF. The toy PDF
     misrepresents the u/d ratio at the 13% level (see
     docs/research/01-oracle/empirical-results.md sec. 2); CT18NNLO
     introduces the m-dependent PDF structure that gives c_Hq^(3) and
     c_Hq^(1) different rate signatures via their u-quark vs d-quark
     couplings, and similarly for c_lq^(3) vs c_lq^(1).
  2. Wider basis: d_psi = 32 (was 16). Doubles the rank of the closed-
     form ridge solve so a richer per-scenario summary fits in the
     implicit weight vector.
  3. Bigger MLP: hidden = 128 (was 64). More capacity for the basis to
     learn PDF-differentiated features.
  4. More data + more steps: 500 training scenarios, 3000 Adam steps,
     larger batches (48 instead of 32). Lets the basis amortise over a
     larger Wilson distribution.

Held-out R^2 and the linear-probe identifiability test both run against
the same upgraded oracle so the comparison is fair.

Outputs:
    experiments/intention-vs-deepsets/output_supercharged/
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

from data import (
    sample_c, sample_c_shell, make_scenario, make_dataset,
    scenarios_to_tensors,
    N_WC, M_RANGE,
)
from intention_learned import IntentionFMLearned
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle

# ---------- supercharged configuration ----------
PDF = "CT18NNLO"          # was "analytic"
D_PSI = 32                # was 16
HIDDEN = 128              # was 64
N_TRAIN_SCENARIOS = 500   # was 200
N_TEST_IN = 50
N_TEST_OUT = 50
K_CTX = 12
Q_QUERY = 32
C_MAX_TRAIN = 0.7
C_OUTER = 1.0
N_META_STEPS = 3000       # was 1500
LR = 1e-3
BATCH_S = 48              # was 32
SEED = 0

WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
WITHHOLD_DIM = 2
WITHHOLD_BAND = (0.6, 1.0)
N_PROBE_TRAIN = 400
N_PROBE_TEST_INSIDE = 200
N_PROBE_TEST_WITHHELD = 200

OUT_DIR = HERE / "output_supercharged"
OUT_DIR.mkdir(exist_ok=True)


def make_smeft_oracle(seed: int = 0, pdf: str = PDF):
    return AnalyticSMEFTOracle(pdf=pdf, noise_frac=0.0, seed=seed)


def r2_per_scenario(y_pred, y_true):
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def train_model(model, train_t, val_t, name):
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
        if step in (1, 50, 200, 500, 1000, 2000, 3000):
            model.eval()
            with torch.no_grad():
                yp = model(val_t["M_ctx"], val_t["Y_ctx"], val_t["M_query"]).cpu().numpy()
            r2 = r2_per_scenario(yp, val_t["Y_query"].cpu().numpy())
            print(f"  [{name}] step {step:5d}  loss={loss.item():.4f}  "
                  f"val_R2_median={float(np.median(r2)):+.4f}")
            model.train()
    print(f"  [{name}] wall = {time.time() - t0:.1f}s")
    model.eval()


def eval_held_out(model, t):
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


def sample_c_inbox_probe(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_MAX_TRAIN, C_MAX_TRAIN, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def sample_c_withheld_probe(n: int, rng: np.random.Generator) -> np.ndarray:
    out = rng.uniform(-C_MAX_TRAIN, C_MAX_TRAIN, size=(n, N_WC))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[:, WITHHOLD_DIM] = signs * rng.uniform(*WITHHOLD_BAND, size=n)
    return out


def implicit_w_one(model, c, oracle, rng):
    M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
    Y_ctx = oracle.truth(np.tile(c, (K_CTX, 1)), M_ctx)
    _, w, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    return w, Y_ctx, M_ctx


def implicit_w_batch(model, cs, oracle, rng):
    ws, Ys, Ms = [], [], []
    for c in cs:
        w, Y, M = implicit_w_one(model, c, oracle, rng)
        ws.append(w); Ys.append(Y); Ms.append(M)
    return np.array(ws), np.array(Ys), np.array(Ms)


def rate_features(Y_ctx: np.ndarray, M_ctx: np.ndarray) -> np.ndarray:
    """Multi-band rate summary per scenario: aggregates Y over different
    m-bands so the probe can read off PDF-differentiated rate channels
    that distinguish cHq3 from cHq1."""
    eps = 1e-6
    feats = []
    for Y, M in zip(Y_ctx, M_ctx):
        lo = M < 0.8
        mid = (M >= 0.8) & (M < 1.5)
        hi = M >= 1.5
        def safe_mean(arr):
            return float(np.mean(arr)) if arr.size > 0 else 0.0
        feats.append([
            np.log(max(safe_mean(Y), eps)),
            np.log(max(safe_mean(Y[lo]) if lo.any() else safe_mean(Y), eps)),
            np.log(max(safe_mean(Y[mid]) if mid.any() else safe_mean(Y), eps)),
            np.log(max(safe_mean(Y[hi]) if hi.any() else safe_mean(Y), eps)),
            float(np.log(max(np.std(Y), eps))),
        ])
    return np.array(feats)


def run_probe(model, label: str, oracle, augment_rate: bool = False):
    rng = np.random.default_rng(5151)
    c_train = sample_c_inbox_probe(N_PROBE_TRAIN, rng)
    w_train, Y_train, M_train = implicit_w_batch(model, c_train, oracle, rng)
    if augment_rate:
        rf = rate_features(Y_train, M_train)
        w_train = np.concatenate([w_train, rf], axis=1)
    probe = RidgeCV(alphas=np.logspace(-4, 2, 25))
    probe.fit(w_train, c_train)

    c_in = sample_c_inbox_probe(N_PROBE_TEST_INSIDE, rng)
    w_in, Y_in, M_in = implicit_w_batch(model, c_in, oracle, rng)
    if augment_rate:
        rf_in = rate_features(Y_in, M_in)
        w_in = np.concatenate([w_in, rf_in], axis=1)
    c_in_pred = probe.predict(w_in)
    r2_inside = {name: float(r2_score(c_in[:, i], c_in_pred[:, i]))
                 for i, name in enumerate(WC_NAMES)}

    c_out = sample_c_withheld_probe(N_PROBE_TEST_WITHHELD, rng)
    w_out, Y_out, M_out = implicit_w_batch(model, c_out, oracle, rng)
    if augment_rate:
        rf_out = rate_features(Y_out, M_out)
        w_out = np.concatenate([w_out, rf_out], axis=1)
    c_out_pred = probe.predict(w_out)
    r2_withheld = {name: float(r2_score(c_out[:, i], c_out_pred[:, i]))
                   for i, name in enumerate(WC_NAMES)}

    print(f"\n=== {label}{' + rate features' if augment_rate else ''} ===")
    print(f"  {'op':<8s} {'inside':>10s} {'withheld':>10s}")
    for name in WC_NAMES:
        print(f"  {name:<8s} {r2_inside[name]:+10.4f} {r2_withheld[name]:+10.4f}")
    return dict(r2_inside=r2_inside, r2_withheld=r2_withheld,
                ridge_alpha=float(probe.alpha_))


def main():
    print(f"# Build supercharged datasets (PDF={PDF})")
    t0 = time.time()
    oracle = make_smeft_oracle(seed=0)
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
    print(f"  dataset wall = {time.time() - t0:.1f}s")

    # --- supercharged IntentionFM ---
    print(f"\n# Train supercharged IntentionFM (d_psi={D_PSI}, hidden={HIDDEN})")
    model = IntentionFMLearned(d_psi=D_PSI, hidden=HIDDEN, alpha=1e-3)
    print(f"  n_params = {model.n_params}")
    train_model(model, train_t, test_in_t, "supercharged")
    torch.save(model.state_dict(), OUT_DIR / "supercharged_model.pt")
    yp_in = eval_held_out(model, test_in_t)
    yp_out = eval_held_out(model, test_out_t)

    Y_in = test_in_t["Y_query"].cpu().numpy()
    Y_out = test_out_t["Y_query"].cpu().numpy()
    r2_in = r2_per_scenario(yp_in, Y_in)
    r2_out = r2_per_scenario(yp_out, Y_out)

    print("\n# Held-out R^2")
    print(f"  supercharged in   median={float(np.median(r2_in)):+.4f}  "
          f"p5={float(np.percentile(r2_in, 5)):+.4f}")
    print(f"  supercharged out  median={float(np.median(r2_out)):+.4f}  "
          f"p5={float(np.percentile(r2_out, 5)):+.4f}")

    np.savez(OUT_DIR / "results.npz",
             yp_in=yp_in, yp_out=yp_out,
             test_in_Y_query=Y_in, test_out_Y_query=Y_out,
             test_in_c=test_in_t["c"].cpu().numpy(),
             test_out_c=test_out_t["c"].cpu().numpy())

    # --- identifiability probes ---
    print("\n# Identifiability probes (against CT18NNLO oracle)")
    probe_native = run_probe(model, "supercharged", oracle, augment_rate=False)
    probe_rate   = run_probe(model, "supercharged", oracle, augment_rate=True)

    summary = dict(
        config=dict(
            pdf=PDF, d_psi=D_PSI, hidden=HIDDEN, n_train=N_TRAIN_SCENARIOS,
            n_steps=N_META_STEPS, batch=BATCH_S,
            n_params=model.n_params,
        ),
        held_out=dict(
            median_in=float(np.median(r2_in)),
            median_out=float(np.median(r2_out)),
            p5_in=float(np.percentile(r2_in, 5)),
            p5_out=float(np.percentile(r2_out, 5)),
        ),
        probe_native=probe_native,
        probe_rate_augmented=probe_rate,
    )
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
