"""Multi-observable mu_FB experiment with vertex-curriculum sampling and
standardized-w linear probe.

Diagnosed problem: mu_FB carries the chirality-asymmetric signal that
distinguishes vertex from four-fermion operators, but the four-fermion
contribution scales as ``(m/Lambda)^4`` and dominates the implicit ridge
weights ``w``. The linear probe from ``w`` to ``c`` then sees vertex
shifts (~3% of mu_FB) drowned in four-fermion variations (~100x of mu_FB).

Two fixes here:
  1. Curriculum: 50% of training scenarios sampled with ``|c_lq*| < 0.1``
     (vertex-dominant), the other 50% with the standard ``[-0.7, 0.7]``
     box. This forces the encoder to allocate basis capacity to the
     vertex-scale features instead of being free to overfit to four-
     fermion-dominant scenarios.
  2. The linear-probe features ``w`` are standardised (mean zero, unit
     variance per dim) before the ridge fit. This prevents the high-
     variance four-fermion directions from drowning the small-variance
     vertex directions under RidgeCV's auto-selected regularisation.
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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from intention_multiobs import IntentionFMMulti, M_REF
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle


PDF = "CT18NNLO"
D_PSI = 32
HIDDEN = 128
N_TRAIN = 500
N_TEST_IN = 50
N_TEST_OUT = 50
K_M = 12
K_FB = 12
Q_M = 16
Q_FB = 16
M_RANGE = (0.3, 2.3)
C_MAX_TRAIN = 0.7
C_OUTER = 1.0
N_META_STEPS = 3000
LR = 1e-3
BATCH_S = 48
SEED = 0
N_WC = 4

WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
WITHHOLD_DIM = 2
WITHHOLD_BAND = (0.6, 1.0)
N_PROBE_TRAIN = 600
N_PROBE_TEST_INSIDE = 200
N_PROBE_TEST_WITHHELD = 200

# Curriculum: half of scenarios are vertex-dominated.
CURRICULUM_FRAC_VERTEX = 0.5

OUT_DIR = HERE / "output_multiobs_curriculum"
OUT_DIR.mkdir(exist_ok=True)


def sample_c_curriculum(rng, n, frac_vertex=CURRICULUM_FRAC_VERTEX, c_max=C_MAX_TRAIN):
    """Mixed sampler. Half: vertex-dominant (c_lq* small, c_Hq* normal).
    Other half: standard uniform box."""
    out = np.empty((n, N_WC))
    n_vertex = int(round(n * frac_vertex))
    # Vertex-dominant: c_lq* in [-0.1, 0.1], c_Hq* in [-0.7, 0.7]
    out[:n_vertex] = rng.uniform(-c_max, c_max, size=(n_vertex, N_WC))
    out[:n_vertex, 2] = rng.uniform(-0.1, 0.1, size=n_vertex)  # c_lq3
    out[:n_vertex, 3] = rng.uniform(-0.1, 0.1, size=n_vertex)  # c_lq1
    # Standard uniform
    out[n_vertex:] = rng.uniform(-c_max, c_max, size=(n - n_vertex, N_WC))
    rng.shuffle(out, axis=0)
    return out


def sample_c_box(rng, n, c_max=C_MAX_TRAIN):
    return rng.uniform(-c_max, c_max, size=(n, N_WC))


def sample_c_shell(rng, n, c_inner=C_MAX_TRAIN, c_outer=C_OUTER):
    out = np.empty((n, N_WC)); filled = 0
    while filled < n:
        batch = rng.uniform(-c_outer, c_outer, size=(8 * n, N_WC))
        keep = np.max(np.abs(batch), axis=1) > c_inner
        batch = batch[keep]
        take = min(n - filled, len(batch))
        out[filled:filled + take] = batch[:take]
        filled += take
    return out


def make_scenario(oracle, c, rng):
    M_ctx = rng.uniform(*M_RANGE, size=K_M)
    MF_ctx = rng.uniform(*M_RANGE, size=K_FB)
    M_q = rng.uniform(*M_RANGE, size=Q_M)
    MF_q = rng.uniform(*M_RANGE, size=Q_FB)
    Y_xs_ctx  = oracle.truth(np.tile(c, (K_M, 1)),  M_ctx)
    Y_fb_ctx  = oracle.truth_mu_fb(np.tile(c, (K_FB, 1)), MF_ctx)
    Y_xs_q    = oracle.truth(np.tile(c, (Q_M, 1)),  M_q)
    Y_fb_q    = oracle.truth_mu_fb(np.tile(c, (Q_FB, 1)), MF_q)
    return {
        "M_ctx":   M_ctx.astype(np.float32),
        "Y_m_ctx": Y_xs_ctx.astype(np.float32),
        "PT_ctx":  MF_ctx.astype(np.float32),
        "Y_pt_ctx": Y_fb_ctx.astype(np.float32),
        "M_q":   M_q.astype(np.float32),
        "Y_m_q": Y_xs_q.astype(np.float32),
        "PT_q":  MF_q.astype(np.float32),
        "Y_pt_q": Y_fb_q.astype(np.float32),
        "c": c.astype(np.float32),
    }


def make_dataset(n, c_sampler, oracle, seed):
    rng = np.random.default_rng(seed)
    cs = c_sampler(rng, n)
    return [make_scenario(oracle, c, rng) for c in cs]


def to_tensors(scenarios):
    return {k: torch.from_numpy(np.stack([s[k] for s in scenarios]))
            for k in scenarios[0]}


def r2_per_scenario(y_pred, y_true):
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def train(model, train_t, val_t):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for step in range(1, N_META_STEPS + 1):
        idx = rng.choice(S_train, size=BATCH_S, replace=False)
        y_m_p, y_fb_p = model(train_t["M_ctx"][idx],  train_t["Y_m_ctx"][idx],
                              train_t["PT_ctx"][idx], train_t["Y_pt_ctx"][idx],
                              train_t["M_q"][idx],    train_t["PT_q"][idx])
        loss = ((y_m_p - train_t["Y_m_q"][idx]) ** 2).mean() + \
               ((y_fb_p - train_t["Y_pt_q"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 50, 200, 500, 1000, 2000, 3000):
            model.eval()
            with torch.no_grad():
                yp_m, yp_fb = model(val_t["M_ctx"], val_t["Y_m_ctx"],
                                    val_t["PT_ctx"], val_t["Y_pt_ctx"],
                                    val_t["M_q"], val_t["PT_q"])
            r2_m  = r2_per_scenario(yp_m.cpu().numpy(),  val_t["Y_m_q"].cpu().numpy())
            r2_fb = r2_per_scenario(yp_fb.cpu().numpy(), val_t["Y_pt_q"].cpu().numpy())
            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"R2_xs={float(np.median(r2_m)):+.4f}  "
                  f"R2_fb={float(np.median(r2_fb)):+.4f}")
            model.train()
    print(f"  wall = {time.time() - t0:.1f}s")
    model.eval()


def sample_c_inbox_probe(n, rng):
    out = np.empty((n, N_WC)); i = 0
    while i < n:
        c = rng.uniform(-C_MAX_TRAIN, C_MAX_TRAIN, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c; i += 1
    return out


def sample_c_withheld_probe(n, rng):
    out = rng.uniform(-C_MAX_TRAIN, C_MAX_TRAIN, size=(n, N_WC))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[:, WITHHOLD_DIM] = signs * rng.uniform(*WITHHOLD_BAND, size=n)
    return out


def implicit_w_one(model, c, oracle, rng):
    M_ctx = rng.uniform(*M_RANGE, size=K_M).astype(np.float32)
    MF_ctx = rng.uniform(*M_RANGE, size=K_FB).astype(np.float32)
    Y_xs = oracle.truth(np.tile(c, (K_M, 1)), M_ctx).astype(np.float32)
    Y_fb = oracle.truth_mu_fb(np.tile(c, (K_FB, 1)), MF_ctx).astype(np.float32)
    _, w, _ = model.A_inv_and_w(M_ctx, Y_xs, MF_ctx, Y_fb)
    return w


def run_probe(model, oracle, label="curriculum+mu_FB", standardize=True):
    rng = np.random.default_rng(5151)

    c_train_inbox = sample_c_inbox_probe(N_PROBE_TRAIN // 2, rng)
    c_train_vertex = sample_c_inbox_probe(N_PROBE_TRAIN // 2, rng)
    c_train_vertex[:, 2] = rng.uniform(-0.1, 0.1, size=len(c_train_vertex))
    c_train_vertex[:, 3] = rng.uniform(-0.1, 0.1, size=len(c_train_vertex))
    c_train = np.concatenate([c_train_inbox, c_train_vertex], axis=0)
    w_train = np.array([implicit_w_one(model, c, oracle, rng) for c in c_train])

    if standardize:
        scaler = StandardScaler().fit(w_train)
        w_train_s = scaler.transform(w_train)
    else:
        scaler = None
        w_train_s = w_train

    probe = RidgeCV(alphas=np.logspace(-6, 4, 40))
    probe.fit(w_train_s, c_train)

    c_in = sample_c_inbox_probe(N_PROBE_TEST_INSIDE, rng)
    w_in = np.array([implicit_w_one(model, c, oracle, rng) for c in c_in])
    if scaler is not None:
        w_in_s = scaler.transform(w_in)
    else:
        w_in_s = w_in
    c_in_pred = probe.predict(w_in_s)
    r2_inside = {name: float(r2_score(c_in[:, i], c_in_pred[:, i]))
                 for i, name in enumerate(WC_NAMES)}

    c_out = sample_c_withheld_probe(N_PROBE_TEST_WITHHELD, rng)
    w_out = np.array([implicit_w_one(model, c, oracle, rng) for c in c_out])
    if scaler is not None:
        w_out_s = scaler.transform(w_out)
    else:
        w_out_s = w_out
    c_out_pred = probe.predict(w_out_s)
    r2_withheld = {name: float(r2_score(c_out[:, i], c_out_pred[:, i]))
                   for i, name in enumerate(WC_NAMES)}

    print(f"\n=== {label} identifiability (standardised={standardize}) ===")
    print(f"  ridge alpha selected: {probe.alpha_:.6f}")
    print(f"  {'op':<8s} {'inside':>10s} {'withheld':>10s}")
    for name in WC_NAMES:
        print(f"  {name:<8s} {r2_inside[name]:+10.4f} {r2_withheld[name]:+10.4f}")
    return dict(r2_inside=r2_inside, r2_withheld=r2_withheld,
                ridge_alpha=float(probe.alpha_))


def main():
    print(f"# Build datasets (PDF={PDF}, curriculum on vertex-dominant scenarios)")
    t0 = time.time()
    oracle = AnalyticSMEFTOracle(pdf=PDF, noise_frac=0.0)
    train_scen = make_dataset(N_TRAIN, sample_c_curriculum, oracle, seed=1)
    test_in = make_dataset(N_TEST_IN, sample_c_box, oracle, seed=42)
    test_out = make_dataset(N_TEST_OUT, sample_c_shell, oracle, seed=43)
    train_t = to_tensors(train_scen)
    test_in_t = to_tensors(test_in)
    test_out_t = to_tensors(test_out)
    print(f"  dataset wall = {time.time() - t0:.1f}s")

    print(f"\n# Train (d_psi={D_PSI}, hidden={HIDDEN})")
    model = IntentionFMMulti(d_psi=D_PSI, hidden=HIDDEN, alpha=1e-3,
                             ref_a=M_REF, ref_b=M_REF)
    print(f"  n_params = {model.n_params}")
    train(model, train_t, test_in_t)
    torch.save(model.state_dict(), OUT_DIR / "curriculum_model.pt")

    print("\n# Identifiability probes")
    probe_std  = run_probe(model, oracle, "curriculum+mu_FB", standardize=True)
    probe_raw  = run_probe(model, oracle, "curriculum+mu_FB", standardize=False)

    summary = dict(
        config=dict(pdf=PDF, d_psi=D_PSI, hidden=HIDDEN, n_train=N_TRAIN,
                    K_m=K_M, K_fb=K_FB, n_params=model.n_params,
                    curriculum_frac_vertex=CURRICULUM_FRAC_VERTEX),
        identifiability_standardised=probe_std,
        identifiability_raw=probe_raw,
    )
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
