"""T0.2 diagnostic D0 + D1-baseline: positive control.

Run the PRECURSOR learned-basis head (IntentionFM, 5328 params) through the
*current FM harness plumbing* (fit_fm_probe-style W, sigma_ctilde, _mle_on_context,
the same acquisition functions) at the published B.9 cell:
    K=12, pool P0 (30 pts), angular mu_FB observable, N_cycles=500, bimodal target.

Published precursor numbers to reproduce (stage_c_closed_loop_aggregate.json):
    contraction d1:  random -> 0.81,  param_epig_a -> 0.79
    cos_Ainv_top:    -> 1.000
    mle_err d1:      param_epig_a 1.144  vs  random 1.138

Two probe variants, to bisect Sigma-plumbing vs W-refitting:
    frozen : published W (probe_W_mass_only_v2.npz), V recomputed on mu_FB  [load_probe]
    refit  : fit_fm_probe-style W (regress precursor ridge weight -> c)     [the FM path]

Decision (per protocol D0):
    frozen reproduces 0.81/0.79  -> sigma_ctilde / MLE plumbing is sound.
    refit  reproduces 0.81/0.79  -> harness (incl. W-refit) is sound; FM degeneracy is FM-path.
    refit  does NOT but frozen does -> the W-refit (fit_fm_probe) is the regression -> D2.

Also prints the precursor D1 baseline (alpha, ||psi||, lev_p, eig(A_seed)) as the
healthy reference the FM D1 run will be compared against.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (
    N_WC, M_RANGE, SIGMA_Y, WITHHOLD_DIM, WITHHOLD_DIM_2, TARGET_C_LQ3, TARGET_C_HQ3,
    build_oracle, load_pretrained_model, load_probe, angular_fisher_V,
    build_pool_P0, ig_per_candidate, sigma_ctilde, OUT,
)
from modules.surrogate.intention import (
    epig_acquire_m, param_epig_d_acquire, param_epig_a_acquire,
)
from stage_c_design_only import _mle_on_context

N_CYCLES = int(os.environ.get("N_CYCLES", "500"))
K_CTX = int(os.environ.get("K_CTX", "12"))
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
SEED0 = int(os.environ.get("SEED0", "2026"))
ACQS = os.environ.get("ACQS", "random,param_epig_a").split(",")


def target_c() -> np.ndarray:
    c = np.zeros(N_WC)
    c[WITHHOLD_DIM] = TARGET_C_LQ3
    c[WITHHOLD_DIM_2] = TARGET_C_HQ3
    return c


def fit_precursor_probe(head, oracle, *, c_box=0.6, n_train=400, K=24, seed=4242,
                        full_range=True):
    """fit_fm_probe-style W on the PRECURSOR head: regress ridge weight w(c)->c,
    rotate by Fisher V (recomputed on mu_FB). Mirrors stage_c_fm_common.fit_fm_probe.
    full_range=True draws the fit-context over the full M_RANGE (the FIX);
    full_range=False reproduces the original thin-band [0.5,1.0] bug for A/B."""
    rng = np.random.default_rng(seed)
    V, lam = angular_fisher_V(oracle)
    m_lo, m_hi = (M_RANGE if full_range else (0.5, 1.0))
    Ws, Cs = [], []
    for _ in range(n_train):
        c = rng.uniform(-c_box, c_box, size=N_WC)
        M = rng.uniform(m_lo, m_hi, size=K)
        Y = oracle.truth_mu_fb(np.tile(c, (K, 1)), M) + rng.normal(0.0, SIGMA_Y, size=K)
        _, w, _ = head.A_inv_and_w(M, Y)
        Ws.append(w); Cs.append(c)
    Wmat = np.stack(Ws); Cmat = np.stack(Cs)
    X = np.hstack([Wmat, np.ones((len(Wmat), 1))])
    Wb, *_ = np.linalg.lstsq(X, Cmat, rcond=None)
    W = Wb[:-1].T
    return {"W": W, "V": V, "P": V.T @ W, "sigma_y": SIGMA_Y, "rank_W": int(np.linalg.matrix_rank(W))}


def acquire_one(head, M_ctx, Y_ctx, M_pool, P, sigma_y, M_target, acq, rng):
    if acq == "random":
        return M_pool[rng.choice(len(M_pool), size=1, replace=False)]
    if acq == "leverage":
        A_inv, _, _ = head.A_inv_and_w(M_ctx, Y_ctx)
        Psi = head.psi_np(M_pool)
        lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
        return M_pool[[int(np.argmax(lev))]]
    if acq == "epig":
        return M_pool[epig_acquire_m(head, M_ctx, Y_ctx, M_pool, M_target, k=1)]
    if acq == "param_epig_d":
        return M_pool[param_epig_d_acquire(head, M_ctx, Y_ctx, M_pool, P, k=1,
                                           sigma_y=sigma_y, resolved_dim=2)]
    if acq == "param_epig_a":
        return M_pool[param_epig_a_acquire(head, M_ctx, Y_ctx, M_pool, P, k=1,
                                           target_direction=1, sigma_y=sigma_y)]
    raise ValueError(acq)


def run_chain(head, oracle, c_tgt, P, sigma_y, V, acq, seed):
    rng = np.random.default_rng(seed)
    M_ctx = rng.uniform(0.5, 1.0, size=K_CTX)
    Y_ctx = oracle.truth_mu_fb(np.tile(c_tgt, (K_CTX, 1)), M_ctx)
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    pool_rng = np.random.default_rng(rng.bit_generator.random_raw())
    S_seed = np.diag(sigma_ctilde(head, M_ctx, Y_ctx, P, sigma_y, resolved_dim=2)).copy()
    cos_trace = []
    for _ in range(N_CYCLES):
        pool = build_pool_P0(pool_rng, size=30)
        ig = ig_per_candidate(head, M_ctx, Y_ctx, pool)
        A_inv, _, _ = head.A_inv_and_w(M_ctx, Y_ctx)
        Psi = head.psi_np(pool)
        n_top = max(2, int(0.1 * len(pool)))
        top = np.argsort(ig)[-n_top:]
        cs = []
        for ii, i in enumerate(top):
            for j in top[ii + 1:]:
                num = float(Psi[i] @ A_inv @ Psi[j])
                den = np.sqrt(float(Psi[i] @ A_inv @ Psi[i]) * float(Psi[j] @ A_inv @ Psi[j]))
                cs.append(num / max(den, 1e-30))
        cos_trace.append(float(np.mean(np.abs(cs))) if cs else 1.0)
        pick = acquire_one(head, M_ctx, Y_ctx, pool, P, sigma_y, M_target, acq, rng)
        Y_pick = oracle.truth_mu_fb(np.tile(c_tgt, (len(pick), 1)), pick)
        M_ctx = np.concatenate([M_ctx, pick]); Y_ctx = np.concatenate([Y_ctx, Y_pick])
    S_final = np.diag(sigma_ctilde(head, M_ctx, Y_ctx, P, sigma_y, resolved_dim=2)).copy()
    mle = _mle_on_context(oracle, c_tgt, M_ctx, Y_ctx, V)
    return {"contr_d1": float(S_final[1] / max(S_seed[1], 1e-30)),
            "contr_d0": float(S_final[0] / max(S_seed[0], 1e-30)),
            "mle_d1": float(mle[1]), "cos": float(np.mean(cos_trace))}


def d1_baseline(head, oracle, c_tgt):
    rng = np.random.default_rng(SEED0)
    M_ctx = rng.uniform(0.5, 1.0, size=K_CTX)
    Y_ctx = oracle.truth_mu_fb(np.tile(c_tgt, (K_CTX, 1)), M_ctx)
    A_inv, _, Psi_ctx = head.A_inv_and_w(M_ctx, Y_ctx)
    A = np.linalg.inv(A_inv)
    pool = build_pool_P0(rng, size=30)
    Psi_pool = head.psi_np(pool)
    lev = np.einsum("pd,de,pe->p", Psi_pool, A_inv, Psi_pool)
    return {"alpha": float(head.alpha),
            "psi_pool_norm_mean": float(np.linalg.norm(Psi_pool, axis=1).mean()),
            "psi_pool_norm_med": float(np.median(np.linalg.norm(Psi_pool, axis=1))),
            "A_seed_eig_min": float(np.linalg.eigvalsh(A).min()),
            "A_seed_eig_max": float(np.linalg.eigvalsh(A).max()),
            "lev_p_med": float(np.median(lev)),
            "lev_p_max": float(lev.max()),
            "lev_p_min": float(lev.min())}


def main():
    print(f"# D0 positive control: precursor head, K={K_CTX}, N_cycles={N_CYCLES}, "
          f"seeds={N_SEEDS}, acqs={ACQS}")
    oracle = build_oracle(seed=SEED0)
    head = load_pretrained_model()
    c_tgt = target_c()

    base = d1_baseline(head, oracle, c_tgt)
    print("\n## D1 baseline (PRECURSOR head — the healthy reference)")
    for k, v in base.items():
        print(f"    {k:22s} = {v:.4g}")

    V, _ = angular_fisher_V(oracle)
    probes = {"frozen": load_probe(oracle),
              "refit_full": fit_precursor_probe(head, oracle, full_range=True),
              "refit_thin": fit_precursor_probe(head, oracle, full_range=False)}
    print(f"\n    rank(W) frozen={np.linalg.matrix_rank(probes['frozen']['W'])} "
          f"refit_full={probes['refit_full'].get('rank_W')} "
          f"refit_thin={probes['refit_thin'].get('rank_W')}")

    results = {"config": {"N_CYCLES": N_CYCLES, "K_CTX": K_CTX, "N_SEEDS": N_SEEDS,
                          "published": {"random": {"contr_d1": 0.81, "mle_d1": 1.138},
                                        "param_epig_a": {"contr_d1": 0.79, "mle_d1": 1.144},
                                        "cos": 1.000}},
               "d1_baseline_precursor": base, "by_probe": {}}
    t0 = time.time()
    for pname, probe in probes.items():
        P, sy = probe["P"], probe["sigma_y"]
        print(f"\n## D0 probe='{pname}'")
        results["by_probe"][pname] = {}
        for acq in ACQS:
            rows = [run_chain(head, oracle, c_tgt, P, sy, V, acq, SEED0 + s)
                    for s in range(N_SEEDS)]
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg_std = {k + "_std": float(np.std([r[k] for r in rows])) for k in rows[0]}
            agg.update(agg_std)
            results["by_probe"][pname][acq] = agg
            print(f"    {acq:14s} contr_d1={agg['contr_d1']:.4f}±{agg['contr_d1_std']:.4f}  "
                  f"mle_d1={agg['mle_d1']:.4f}  cos={agg['cos']:.4f}")
    results["wall_seconds"] = time.time() - t0
    with open(OUT / "d0_positive_control.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n# wrote {OUT}/d0_positive_control.json  (wall {results['wall_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
