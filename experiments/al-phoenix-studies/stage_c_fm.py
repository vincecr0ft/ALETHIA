"""Stage C on the ManifoldInformer (FM) representation — T0.2.

Per-chain worker: the B.9 contraction-vs-MLE closed loop, but with the FM
pooled-per-event encoder rows as the ridge feature map in place of the
precursor binned-ratio psi(m). Reuses the B.9 metrics verbatim:

  - resolved-subspace contraction Sigma_{c2}^post / Sigma_{c2}^seed
    (sigma_ctilde, generic over the head),
  - analytic per-direction MLE error on c2 (_mle_on_context from
    stage_c_design_only, representation-independent — it fits the analytic
    morphing on oracle labels),
  - top-decile-IG |cos_{A^-1}| collinearity (same code path as the precursor
    closed loop).

Env (same names as stage_c_closed_loop.py, plus REP):
  ACQ      in {random, leverage, epig, param_epig_d, param_epig_a}
  POOL     in {P0, P1, P2}
  K_CTX    in {12, 24, 48}
  SEED
  N_CYCLES
  REP      in {full, def_ext}   FM encoder variant (default full)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    M_RANGE, N_WC, OUT, WITHHOLD_DIM, WITHHOLD_DIM_2,
    TARGET_C_LQ3, TARGET_C_HQ3, build_pool_P0, build_pool_P1, build_pool_P2,
    ig_per_candidate, sigma_ctilde,
)
from modules.surrogate.intention import (  # noqa: E402
    epig_acquire_m, param_epig_d_acquire, param_epig_a_acquire,
)
from stage_c_design_only import _mle_on_context  # noqa: E402
from stage_c_fm_common import (  # noqa: E402
    FMHead, build_oracle, load_full_encoder, load_deficient_encoder,
    recover_extension_dirs, fit_fm_probe, target_c,
)
from manifold_informer import ManifoldInformer  # noqa: E402  (re-export check)

ACQ = os.environ.get("ACQ", "random")
POOL = os.environ.get("POOL", "P0")
K_CTX_ENV = int(os.environ.get("K_CTX", "12"))
SEED = int(os.environ.get("SEED", "2026"))
N_CYCLES = int(os.environ.get("N_CYCLES", "50"))
REP = os.environ.get("REP", "full")

# Working-point queue for the psi-extension residual SVD (def_ext only),
# matching integrated_loop.WORKING_POINT_QUEUE.
WP_QUEUE = [
    np.array([0.0, 0.0, 0.3, 0.0]), np.array([0.0, 0.0, 0.5, 0.0]),
    np.array([0.2, 0.0, 0.4, 0.0]), np.array([0.0, 0.0, 0.4, 0.2]),
    np.array([0.4, 0.0, 0.4, 0.0]), np.array([0.0, 0.2, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.7, 0.0]), np.array([0.3, 0.0, 0.3, 0.0]),
    np.array([0.0, 0.0, 0.6, 0.0]), np.array([0.2, 0.0, 0.6, 0.0]),
    np.array([0.0, 0.0, 0.5, 0.3]), np.array([0.3, 0.2, 0.3, 0.0]),
]


def _build_pool(name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "P0":
        return build_pool_P0(rng, size=30)
    if name == "P1":
        return build_pool_P1(rng, size=500)
    if name == "P2":
        return build_pool_P2(rng, size=500)
    raise ValueError(name)


def _build_head(rep: str, oracle, c_ang: np.ndarray) -> FMHead:
    if rep == "full":
        enc = load_full_encoder()
        return FMHead(enc, oracle, c_ang)
    if rep == "def_ext":
        enc = load_deficient_encoder()
        ext = recover_extension_dirs(enc, oracle, WP_QUEUE, n_ext=1)
        return FMHead(enc, oracle, c_ang, extension_dirs=ext)
    raise ValueError(rep)


def _acquire_one(head, M_ctx, Y_ctx, M_pool, P, sigma_y, M_target,
                 acq: str, rng: np.random.Generator) -> np.ndarray:
    if acq == "random":
        return M_pool[rng.choice(len(M_pool), size=1, replace=False)]
    if acq == "leverage":
        A_inv, _, Psi = head.A_inv_and_w(M_ctx, Y_ctx)
        Psi_pool = head.psi_np(M_pool)
        lev = np.einsum("pd,de,pe->p", Psi_pool, A_inv, Psi_pool)
        return M_pool[[int(np.argmax(lev))]]
    if acq == "epig":
        idx = epig_acquire_m(head, M_ctx, Y_ctx, M_pool, M_target, k=1)
        return M_pool[idx]
    if acq == "param_epig_d":
        idx = param_epig_d_acquire(head, M_ctx, Y_ctx, M_pool, P, k=1,
                                   sigma_y=sigma_y, resolved_dim=2)
        return M_pool[idx]
    if acq == "param_epig_a":
        idx = param_epig_a_acquire(head, M_ctx, Y_ctx, M_pool, P, k=1,
                                   target_direction=1, sigma_y=sigma_y)
        return M_pool[idx]
    raise ValueError(acq)


def main():
    oracle = build_oracle(seed=SEED)
    c_tgt = target_c()
    head = _build_head(REP, oracle, c_tgt)
    probe = fit_fm_probe(lambda c: _build_head(REP, oracle, c), oracle,
                         seed=4242)
    P = probe["P"]
    sigma_y = probe["sigma_y"]

    rng = np.random.default_rng(SEED)
    M_ctx = rng.uniform(0.5, 1.0, size=K_CTX_ENV)
    C = np.tile(c_tgt, (K_CTX_ENV, 1))
    Y_ctx = oracle.truth_mu_fb(C, M_ctx)
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    pool_rng = np.random.default_rng(rng.bit_generator.random_raw())

    S_seed = sigma_ctilde(head, M_ctx, Y_ctx, P, sigma_y, resolved_dim=2)
    diag_seed = np.diag(S_seed).copy()

    cos_top_trace = []
    cv_trace = []
    t0 = time.time()
    for c in range(N_CYCLES):
        pool = _build_pool(POOL, pool_rng)
        ig = ig_per_candidate(head, M_ctx, Y_ctx, pool)
        cv = float(ig.std() / max(ig.mean(), 1e-30))
        cv_trace.append(cv)

        # Top-decile-IG |cos_{A^-1}| collinearity (the B.9 redundancy number).
        A_inv, _, _ = head.A_inv_and_w(M_ctx, Y_ctx)
        Psi = head.psi_np(pool)
        n_top = max(2, int(0.1 * len(pool)))
        top = np.argsort(ig)[-n_top:]
        cs = []
        for ii, i in enumerate(top):
            for j in top[ii + 1:]:
                num = float(Psi[i] @ A_inv @ Psi[j])
                den = np.sqrt(float(Psi[i] @ A_inv @ Psi[i])
                              * float(Psi[j] @ A_inv @ Psi[j]))
                cs.append(num / max(den, 1e-30))
        cos_top = float(np.mean(np.abs(cs))) if cs else 1.0
        cos_top_trace.append(cos_top)

        pick_m = _acquire_one(head, M_ctx, Y_ctx, pool, P, sigma_y,
                              M_target, ACQ, rng)
        Y_pick = oracle.truth_mu_fb(np.tile(c_tgt, (len(pick_m), 1)), pick_m)
        M_ctx = np.concatenate([M_ctx, pick_m])
        Y_ctx = np.concatenate([Y_ctx, Y_pick])

    S_final = sigma_ctilde(head, M_ctx, Y_ctx, P, sigma_y, resolved_dim=2)
    diag = np.diag(S_final).copy()
    mle_err = _mle_on_context(oracle, c_tgt, M_ctx, Y_ctx, probe["V"])

    summary = {
        "rep": REP, "acq": ACQ, "pool": POOL, "K_ctx": K_CTX_ENV,
        "seed": SEED, "n_cycles": N_CYCLES,
        "Sigma_seed_diag": diag_seed.tolist(),
        "Sigma_final_diag": diag.tolist(),
        "contraction_final_d0": float(diag[0] / max(diag_seed[0], 1e-30)),
        "contraction_final_d1": float(diag[1] / max(diag_seed[1], 1e-30)),
        "mle_err_d0": float(mle_err[0]),
        "mle_err_d1": float(mle_err[1]),
        "cos_Ainv_top_mean": float(np.mean(cos_top_trace)),
        "cos_Ainv_top_last": float(cos_top_trace[-1]),
        "cv_ig_mean": float(np.mean(cv_trace)),
        "wall_seconds": time.time() - t0,
    }
    out_dir = OUT / f"fm_{REP}_{POOL}_K{K_CTX_ENV}_N{N_CYCLES}_{ACQ}_s{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[fm:{REP}] {ACQ}/{POOL}/K{K_CTX_ENV}/s{SEED}: "
          f"contr_d1={summary['contraction_final_d1']:.4f} "
          f"mle_d1={summary['mle_err_d1']:.4f} "
          f"cos={summary['cos_Ainv_top_mean']:.4f} "
          f"wall={summary['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
