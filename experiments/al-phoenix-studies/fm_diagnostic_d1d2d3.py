"""T0.2 FM diagnostics D1/D2/D3 — run ONLY after D0 passes.

Localizes the FM zero-contraction to one of three layers, per the protocol:

  D1 (A^-1 layer): alpha, pooled-row norms ||psi||, eig(A_seed), lev_p over the
     candidate pool. Regularizer-dominated iff lev_p << 1 on a diverse pool
     (then cos_Ainv is just raw embedding cosine and carries no ridge-geometry
     information). Leverage is invariant to global feature rescaling; collapse
     requires alpha large relative to ||psi||^2 — both are printed.

  D2 (P = V^T W layer): rank(W); the contraction of A^-1 restricted to its own
     resolved directions across cycles, compared to the contraction of Sigma.
     If A^-1 contracts but Sigma does not, W misprojects (stale/rank-deficient).

  D3 (information layer): eig(Sigma_seed) for seed-room; confirm angular mu_FB
     observable (the head samples cos theta* from the target angular law -> the
     FM sweep IS on the angular observable by construction; we assert it).

REP env: full (default) | def_ext.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (
    N_WC, M_RANGE, build_pool_P0, ig_per_candidate, sigma_ctilde, OUT,
)
from stage_c_fm_common import (
    FMHead, build_oracle, load_full_encoder, load_deficient_encoder,
    recover_extension_dirs, fit_fm_probe, target_c,
)
from stage_c_fm import WP_QUEUE, _build_head

REP = os.environ.get("REP", "full")
K_CTX = int(os.environ.get("K_CTX", "12"))
SEED = int(os.environ.get("SEED", "2026"))
N_PROBE_CYCLES = int(os.environ.get("N_PROBE_CYCLES", "50"))


def resolved_A_contraction(A_inv_seed, A_inv_final, r_dirs):
    """Contraction of A^-1 restricted to the resolved directions r_dirs (cols)."""
    s = r_dirs.T @ A_inv_seed @ r_dirs
    f = r_dirs.T @ A_inv_final @ r_dirs
    return float(np.trace(f) / max(np.trace(s), 1e-30))


def main():
    oracle = build_oracle(seed=SEED)
    c_tgt = target_c()
    head = _build_head(REP, oracle, c_tgt)
    probe = fit_fm_probe(lambda c: _build_head(REP, oracle, c), oracle, seed=4242)
    P, V, W, sy = probe["P"], probe["V"], probe["W"], probe["sigma_y"]

    rng = np.random.default_rng(SEED)
    M_ctx = rng.uniform(0.5, 1.0, size=K_CTX)
    Y_ctx = oracle.truth_mu_fb(np.tile(c_tgt, (K_CTX, 1)), M_ctx)
    pool = build_pool_P0(rng, size=30)

    A_inv_seed, _, Psi_ctx = head.A_inv_and_w(M_ctx, Y_ctx)
    A_seed = np.linalg.inv(A_inv_seed)
    Psi_pool = head.psi_np(pool)
    lev = np.einsum("pd,de,pe->p", Psi_pool, A_inv_seed, Psi_pool)
    psi_norm = np.linalg.norm(Psi_pool, axis=1)

    # D1
    d1 = {
        "alpha": float(head.alpha),
        "psi_pool_norm_mean": float(psi_norm.mean()),
        "psi_pool_norm_med": float(np.median(psi_norm)),
        "psi_pool_norm_min": float(psi_norm.min()),
        "psi_pool_norm_max": float(psi_norm.max()),
        "psi_row_variability_cv": float(np.linalg.norm(Psi_pool - Psi_pool.mean(0), axis=1).mean()
                                        / max(psi_norm.mean(), 1e-30)),
        "A_seed_eig_min": float(np.linalg.eigvalsh(A_seed).min()),
        "A_seed_eig_max": float(np.linalg.eigvalsh(A_seed).max()),
        "alpha_over_psi2": float(head.alpha / max(psi_norm.mean() ** 2, 1e-30)),
        "lev_p_med": float(np.median(lev)),
        "lev_p_max": float(lev.max()),
        "lev_p_min": float(lev.min()),
    }
    regularizer_dominated = d1["lev_p_med"] < 0.05

    # D2: A^-1 contraction (own resolved dirs) vs Sigma contraction, over cycles
    pool_rng = np.random.default_rng(rng.bit_generator.random_raw())
    M_c, Y_c = M_ctx.copy(), Y_ctx.copy()
    S_seed = np.diag(sigma_ctilde(head, M_c, Y_c, P, sy, resolved_dim=2)).copy()
    # resolved directions of A_seed: top-2 eigenvectors (largest eigenvalues)
    w_eig, v_eig = np.linalg.eigh(A_seed)
    r_dirs = v_eig[:, np.argsort(w_eig)[::-1][:2]]
    for _ in range(N_PROBE_CYCLES):
        p = build_pool_P0(pool_rng, size=30)
        ig = ig_per_candidate(head, M_c, Y_c, p)
        pick = p[[int(np.argmax(ig))]]            # leverage-greedy, deterministic
        Yp = oracle.truth_mu_fb(np.tile(c_tgt, (len(pick), 1)), pick)
        M_c = np.concatenate([M_c, pick]); Y_c = np.concatenate([Y_c, Yp])
    A_inv_final, _, _ = head.A_inv_and_w(M_c, Y_c)
    S_final = np.diag(sigma_ctilde(head, M_c, Y_c, P, sy, resolved_dim=2)).copy()
    d2 = {
        "rank_W": int(np.linalg.matrix_rank(W)),
        "W_shape": list(W.shape),
        "A_inv_resolved_contraction": resolved_A_contraction(A_inv_seed, A_inv_final, r_dirs),
        "Sigma_contraction_d0": float(S_final[0] / max(S_seed[0], 1e-30)),
        "Sigma_contraction_d1": float(S_final[1] / max(S_seed[1], 1e-30)),
    }

    # D3
    Sig_seed_full = sigma_ctilde(head, M_ctx, Y_ctx, P, sy, resolved_dim=None)
    d3 = {
        "Sigma_seed_eig": np.linalg.eigvalsh(Sig_seed_full).tolist(),
        "Sigma_seed_diag_resolved": S_seed.tolist(),
        "observable": "angular mu_FB (cos theta* sampled from target angular law in FMHead)",
        "d_event": int(head.d_event),
    }

    print(f"# FM diagnostic D1/D2/D3 — REP={REP}, K={K_CTX}")
    print("\n## D1 (A^-1 layer)")
    for k, v in d1.items():
        print(f"    {k:24s} = {v:.4g}")
    print(f"    --> regularizer_dominated (lev_p_med<0.05): {regularizer_dominated}")
    print("\n## D2 (P=V^T W layer)")
    for k, v in d2.items():
        print(f"    {k:24s} = {v}")
    a_contracts = d2["A_inv_resolved_contraction"] < 0.9
    sig_contracts = d2["Sigma_contraction_d1"] < 0.9
    print(f"    --> A^-1 contracts: {a_contracts}  Sigma contracts: {sig_contracts}")
    if a_contracts and not sig_contracts:
        print("    --> PROJECTION FAULT: A^-1 contracts but Sigma does not -> W misprojects (D2 fires)")
    print("\n## D3 (information layer)")
    print(f"    Sigma_seed_eig = {[f'{x:.3e}' for x in d3['Sigma_seed_eig']]}")
    print(f"    observable = {d3['observable']}")

    out = {"rep": REP, "K_ctx": K_CTX, "seed": SEED,
           "D1": d1, "regularizer_dominated": bool(regularizer_dominated),
           "D2": d2, "D3": d3}
    with open(OUT / f"fm_diagnostic_d1d2d3_{REP}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n# wrote {OUT}/fm_diagnostic_d1d2d3_{REP}.json")


if __name__ == "__main__":
    main()
