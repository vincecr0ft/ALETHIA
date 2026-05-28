"""Multi-observable Intention head with A_FB(m_ll) as the second channel.

The lepton-p_T extension of experiment_multiobs.py left the vertex
operators below the qualitative disclosure threshold because at LO with
no transverse recoil the p_T spectrum is a different projection of the
same partonic phase space as d sigma / d m_ll. The discriminating
observable is the forward-backward asymmetry A_FB(m_ll), which carries
the chirality-asymmetric piece of the cross section that both p_T and
m_ll integrate out.

This script trains the Intention head against the combined context
(m_ll, mu_xs) and (m_ll, A_FB), then runs the linear-probe
identifiability test against CT18NNLO.
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

from intention_multiobs import IntentionFMMulti, M_REF
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle


PDF = "CT18NNLO"
D_PSI = 32
HIDDEN = 128
N_TRAIN = 500
N_TEST_IN = 50
N_TEST_OUT = 50
K_M = 12         # m_ll context points (cross section)
K_AFB = 12       # m_ll context points (A_FB)
Q_M = 16
Q_AFB = 16
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
N_PROBE_TRAIN = 400
N_PROBE_TEST_INSIDE = 200
N_PROBE_TEST_WITHHELD = 200

OUT_DIR = HERE / "output_multiobs_mufb"
OUT_DIR.mkdir(exist_ok=True)


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


def make_scenario_afb(oracle, c, rng):
    M_ctx = rng.uniform(*M_RANGE, size=K_M)
    MA_ctx = rng.uniform(*M_RANGE, size=K_AFB)
    M_q = rng.uniform(*M_RANGE, size=Q_M)
    MA_q = rng.uniform(*M_RANGE, size=Q_AFB)
    Y_xs_ctx  = oracle.truth(np.tile(c, (K_M, 1)),  M_ctx)
    Y_afb_ctx = oracle.truth_mu_fb(np.tile(c, (K_AFB, 1)), MA_ctx)
    Y_xs_q    = oracle.truth(np.tile(c, (Q_M, 1)),  M_q)
    Y_afb_q   = oracle.truth_mu_fb(np.tile(c, (Q_AFB, 1)), MA_q)
    return {
        "M_ctx":   M_ctx.astype(np.float32),
        "Y_m_ctx": Y_xs_ctx.astype(np.float32),
        "PT_ctx":  MA_ctx.astype(np.float32),     # reuse Multi class's field name
        "Y_pt_ctx": Y_afb_ctx.astype(np.float32), # ditto
        "M_q":   M_q.astype(np.float32),
        "Y_m_q": Y_xs_q.astype(np.float32),
        "PT_q":  MA_q.astype(np.float32),
        "Y_pt_q": Y_afb_q.astype(np.float32),
        "c": c.astype(np.float32),
    }


def make_dataset(n, c_sampler, oracle, seed):
    rng = np.random.default_rng(seed)
    return [make_scenario_afb(oracle, c_sampler(rng, 1)[0], rng) for _ in range(n)]


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
        y_m_p, y_afb_p = model(train_t["M_ctx"][idx],  train_t["Y_m_ctx"][idx],
                                train_t["PT_ctx"][idx], train_t["Y_pt_ctx"][idx],
                                train_t["M_q"][idx],    train_t["PT_q"][idx])
        loss = ((y_m_p - train_t["Y_m_q"][idx]) ** 2).mean() + \
               ((y_afb_p - train_t["Y_pt_q"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 50, 200, 500, 1000, 2000, 3000):
            model.eval()
            with torch.no_grad():
                yp_m, yp_afb = model(val_t["M_ctx"], val_t["Y_m_ctx"],
                                     val_t["PT_ctx"], val_t["Y_pt_ctx"],
                                     val_t["M_q"], val_t["PT_q"])
            r2_m   = r2_per_scenario(yp_m.cpu().numpy(),   val_t["Y_m_q"].cpu().numpy())
            r2_afb = r2_per_scenario(yp_afb.cpu().numpy(), val_t["Y_pt_q"].cpu().numpy())
            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"R2_xs_med={float(np.median(r2_m)):+.4f}  "
                  f"R2_afb_med={float(np.median(r2_afb)):+.4f}")
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
    MA_ctx = rng.uniform(*M_RANGE, size=K_AFB).astype(np.float32)
    Y_xs  = oracle.truth(np.tile(c, (K_M, 1)),  M_ctx).astype(np.float32)
    Y_afb = oracle.truth_mu_fb(np.tile(c, (K_AFB, 1)), MA_ctx).astype(np.float32)
    _, w, _ = model.A_inv_and_w(M_ctx, Y_xs, MA_ctx, Y_afb)
    return w


def run_probe(model, oracle, label="multi-obs+AFB"):
    rng = np.random.default_rng(5151)
    c_train = sample_c_inbox_probe(N_PROBE_TRAIN, rng)
    w_train = np.array([implicit_w_one(model, c, oracle, rng) for c in c_train])
    probe = RidgeCV(alphas=np.logspace(-4, 2, 25))
    probe.fit(w_train, c_train)

    c_in = sample_c_inbox_probe(N_PROBE_TEST_INSIDE, rng)
    w_in = np.array([implicit_w_one(model, c, oracle, rng) for c in c_in])
    c_in_pred = probe.predict(w_in)
    r2_inside = {name: float(r2_score(c_in[:, i], c_in_pred[:, i]))
                 for i, name in enumerate(WC_NAMES)}

    c_out = sample_c_withheld_probe(N_PROBE_TEST_WITHHELD, rng)
    w_out = np.array([implicit_w_one(model, c, oracle, rng) for c in c_out])
    c_out_pred = probe.predict(w_out)
    r2_withheld = {name: float(r2_score(c_out[:, i], c_out_pred[:, i]))
                   for i, name in enumerate(WC_NAMES)}

    print(f"\n=== {label} identifiability ===")
    print(f"  ridge alpha selected: {probe.alpha_:.4f}")
    print(f"  {'op':<8s} {'inside':>10s} {'withheld':>10s}")
    for name in WC_NAMES:
        print(f"  {name:<8s} {r2_inside[name]:+10.4f} {r2_withheld[name]:+10.4f}")
    return dict(r2_inside=r2_inside, r2_withheld=r2_withheld,
                ridge_alpha=float(probe.alpha_),
                c_in=c_in, c_in_pred=c_in_pred,
                c_out=c_out, c_out_pred=c_out_pred)


def main():
    print(f"# Build (m, A_FB) datasets (PDF={PDF})")
    t0 = time.time()
    oracle = AnalyticSMEFTOracle(pdf=PDF, noise_frac=0.0)
    train_scen = make_dataset(N_TRAIN, sample_c_box, oracle, seed=1)
    test_in = make_dataset(N_TEST_IN, sample_c_box, oracle, seed=42)
    test_out = make_dataset(N_TEST_OUT, sample_c_shell, oracle, seed=43)
    train_t = to_tensors(train_scen)
    test_in_t = to_tensors(test_in)
    test_out_t = to_tensors(test_out)
    print(f"  dataset wall = {time.time() - t0:.1f}s")

    print(f"\n# Train (d_psi={D_PSI}, hidden={HIDDEN})")
    # Both encoders operate on m_ll, so both refs are M_REF.
    model = IntentionFMMulti(d_psi=D_PSI, hidden=HIDDEN, alpha=1e-3,
                             ref_a=M_REF, ref_b=M_REF)
    print(f"  n_params = {model.n_params}")
    train(model, train_t, test_in_t)
    torch.save(model.state_dict(), OUT_DIR / "multiobs_afb_model.pt")

    with torch.no_grad():
        yp_m_in, yp_afb_in = model(test_in_t["M_ctx"], test_in_t["Y_m_ctx"],
                                    test_in_t["PT_ctx"], test_in_t["Y_pt_ctx"],
                                    test_in_t["M_q"], test_in_t["PT_q"])
        yp_m_out, yp_afb_out = model(test_out_t["M_ctx"], test_out_t["Y_m_ctx"],
                                      test_out_t["PT_ctx"], test_out_t["Y_pt_ctx"],
                                      test_out_t["M_q"], test_out_t["PT_q"])
    r2_m_in    = r2_per_scenario(yp_m_in.cpu().numpy(),    test_in_t["Y_m_q"].cpu().numpy())
    r2_afb_in  = r2_per_scenario(yp_afb_in.cpu().numpy(),  test_in_t["Y_pt_q"].cpu().numpy())
    r2_m_out   = r2_per_scenario(yp_m_out.cpu().numpy(),   test_out_t["Y_m_q"].cpu().numpy())
    r2_afb_out = r2_per_scenario(yp_afb_out.cpu().numpy(), test_out_t["Y_pt_q"].cpu().numpy())

    print("\n# Held-out R^2")
    for label, r in (("xs  in ", r2_m_in), ("xs  out", r2_m_out),
                     ("afb in ", r2_afb_in), ("afb out", r2_afb_out)):
        print(f"  {label}  median={float(np.median(r)):+.4f}  p5={float(np.percentile(r, 5)):+.4f}")

    probe = run_probe(model, oracle)

    summary = dict(
        config=dict(pdf=PDF, d_psi=D_PSI, hidden=HIDDEN, n_train=N_TRAIN,
                    K_m=K_M, K_afb=K_AFB, n_params=model.n_params),
        held_out={
            "xs_in_median":   float(np.median(r2_m_in)),
            "xs_out_median":  float(np.median(r2_m_out)),
            "afb_in_median":  float(np.median(r2_afb_in)),
            "afb_out_median": float(np.median(r2_afb_out)),
        },
        identifiability=dict(
            r2_inside=probe["r2_inside"],
            r2_withheld=probe["r2_withheld"],
            ridge_alpha=probe["ridge_alpha"],
        ),
    )
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
