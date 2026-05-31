"""Stage C, closed-loop: Phoenix-instrumented per-cycle AL run.

What the AL_separation plan §3 actually asks for: same head, same acquisition,
same oracle as the failed sweep, but on a heterogeneous pool with c̃-space
contraction + MLE scoring. We skip the pretrain (load the frozen checkpoint
from the failed sweep so the psi basis is identical) and run the closed-loop
acquisition over N cycles, emitting one Phoenix span per cycle.

Phoenix span layout (project ``alethia-al-studies``):
  chain.al.stage_c.closed_loop.run             (per chain, one root)
    chain.al.cycle                             (per cycle, k=1 pick)
      tool.al.acquire.<acq>
      tool.al.posterior_update
      chain.al.drift.eigen                     (eigen state on current A)

This is the "what an Arize dashboard sees" view. The dashboard's value isn't
that the loop separates the acquisitions — Stage A/B already say it likely
won't on this physics — but that it surfaces the *gate* signals in real time
(the EIG-spread, the redundancy, the per-direction contraction).

Run-matrix one chain per invocation, parameterised by env:
  ACQ      ∈ {random, leverage, epig, param_epig_d, param_epig_a}
  POOL     ∈ {P0, P1, P2}
  K_CTX    ∈ {12, 24}
  SEED     ∈ ...
  N_CYCLES (default 100; matches the failed sweep's count / k=1)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from _common import (  # noqa: E402
    M_RANGE, N_WC, OUT, SIGMA_Y, WITHHOLD_DIM, WITHHOLD_DIM_2,
    TARGET_C_LQ3, TARGET_C_HQ3,
    build_oracle, build_pool_P0, build_pool_P1, build_pool_P2,
    ig_per_candidate, load_pretrained_model, load_probe, sigma_ctilde,
    truth_mu_fb, tracer,
)
from modules.surrogate.intention import (  # noqa: E402
    epig_acquire_m, param_epig_d_acquire, param_epig_a_acquire,
)
from modules.surrogate.intention.eigen import eigen_state  # noqa: E402

ACQ = os.environ.get("ACQ", "random")
POOL = os.environ.get("POOL", "P2")
K_CTX_ENV = int(os.environ.get("K_CTX", "12"))
SEED = int(os.environ.get("SEED", "2026"))
N_CYCLES = int(os.environ.get("N_CYCLES", "100"))


def _build_pool(name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "P0":
        return build_pool_P0(rng, size=30)
    if name == "P1":
        return build_pool_P1(rng, size=500)
    if name == "P2":
        return build_pool_P2(rng, size=500)
    raise ValueError(name)


def _acquire_one(model, M_ctx, Y_ctx, M_pool, P, sigma_y, M_target,
                  acq: str, rng: np.random.Generator) -> np.ndarray:
    if acq == "random":
        return M_pool[rng.choice(len(M_pool), size=1, replace=False)]
    if acq == "leverage":
        A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
        Psi = model.psi_np(M_pool)
        lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
        i = int(np.argmax(lev))
        return M_pool[[i]]
    if acq == "epig":
        idx = epig_acquire_m(model, M_ctx, Y_ctx, M_pool, M_target, k=1)
        return M_pool[idx]
    if acq == "param_epig_d":
        idx = param_epig_d_acquire(model, M_ctx, Y_ctx, M_pool, P, k=1,
                                    sigma_y=sigma_y, resolved_dim=2)
        return M_pool[idx]
    if acq == "param_epig_a":
        idx = param_epig_a_acquire(model, M_ctx, Y_ctx, M_pool, P, k=1,
                                    target_direction=1, sigma_y=sigma_y)
        return M_pool[idx]
    raise ValueError(acq)


def main():
    model = load_pretrained_model()
    oracle = build_oracle(seed=SEED)
    probe = load_probe(oracle)

    rng = np.random.default_rng(SEED)
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = TARGET_C_LQ3
    target_c[WITHHOLD_DIM_2] = TARGET_C_HQ3
    M_ctx = rng.uniform(0.5, 1.0, size=K_CTX_ENV)
    Y_ctx = truth_mu_fb(oracle, target_c, M_ctx)
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)

    # Reproducible per-cycle pool. Each cycle samples a fresh pool from the
    # same builder so the chain has a chance to "discover" different
    # candidates across the run, matching the production loop's behaviour.
    # We seed each cycle from a child stream of the chain rng.
    pool_rng = np.random.default_rng(rng.bit_generator.random_raw())

    S_seed = sigma_ctilde(model, M_ctx, Y_ctx, probe["P"], probe["sigma_y"],
                          resolved_dim=2)
    diag_seed = np.diag(S_seed).copy()

    traj = {
        "cycle": [], "context_size": [],
        "Sigma_d0": [], "Sigma_d1": [],
        "contraction_d0": [], "contraction_d1": [],
        "pick_m": [], "pool_cv_ig": [], "pool_cos_Ainv_top": [],
        "eig_kappa": [], "eig_min": [], "eig_max": [],
    }

    tr = tracer()
    with tr.start_as_current_span("chain.al.stage_c.closed_loop.run") as root:
        root.set_attribute("aletheia.al.acq", ACQ)
        root.set_attribute("aletheia.al.pool", POOL)
        root.set_attribute("aletheia.al.K_ctx", K_CTX_ENV)
        root.set_attribute("aletheia.al.seed", SEED)
        root.set_attribute("aletheia.al.n_cycles", N_CYCLES)
        t0 = time.time()

        for c in range(N_CYCLES):
            with tr.start_as_current_span("chain.al.cycle") as cyc:
                cyc.set_attribute("aletheia.al.cycle.index", c)
                cyc.set_attribute("aletheia.al.cycle.context_size",
                                  int(len(M_ctx)))
                # Pool + IG spread signal.
                pool = _build_pool(POOL, pool_rng)
                ig = ig_per_candidate(model, M_ctx, Y_ctx, pool)
                cv = float(ig.std() / max(ig.mean(), 1e-30))
                cyc.set_attribute("aletheia.al.pool.cv_ig", cv)
                cyc.set_attribute("aletheia.al.pool.size", int(len(pool)))

                # Top-decile cosine in A^{-1} — the dashboard's redundancy
                # indicator from Stage B.
                A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
                Psi = model.psi_np(pool)
                n_top = max(2, int(0.1 * len(pool)))
                top = np.argsort(ig)[-n_top:]
                cs = []
                for i_idx, i in enumerate(top):
                    for j in top[i_idx + 1:]:
                        num = float(Psi[i] @ A_inv @ Psi[j])
                        den = np.sqrt(float(Psi[i] @ A_inv @ Psi[i])
                                       * float(Psi[j] @ A_inv @ Psi[j]))
                        cs.append(num / max(den, 1e-30))
                cos_top = float(np.mean(np.abs(cs))) if cs else 1.0
                cyc.set_attribute("aletheia.al.pool.cos_Ainv_top_mean", cos_top)

                # Eigen state for the chain.drift.eigen-style signal.
                eig = eigen_state(Psi[:len(M_ctx)] if False else model.psi_np(M_ctx),
                                   model.alpha)
                cyc.set_attribute("aletheia.al.eigen.kappa", float(eig.kappa))
                cyc.set_attribute("aletheia.al.eigen.min", float(eig.lam.min()))
                cyc.set_attribute("aletheia.al.eigen.max", float(eig.lam.max()))

                # Acquire one pick.
                with tr.start_as_current_span(f"tool.al.acquire.{ACQ}") as sp:
                    pick_m = _acquire_one(model, M_ctx, Y_ctx, pool, probe["P"],
                                           probe["sigma_y"], M_target, ACQ, rng)
                    sp.set_attribute("aletheia.al.pick.m", float(pick_m[0]))

                # Fold pick into context and update Σ_c̃.
                Y_pick = truth_mu_fb(oracle, target_c, pick_m)
                M_ctx = np.concatenate([M_ctx, pick_m])
                Y_ctx = np.concatenate([Y_ctx, Y_pick])
                with tr.start_as_current_span("tool.al.posterior_update") as sp:
                    S = sigma_ctilde(model, M_ctx, Y_ctx, probe["P"],
                                      probe["sigma_y"], resolved_dim=2)
                    diag = np.diag(S).copy()
                    sp.set_attribute("aletheia.al.Sigma.d0", float(diag[0]))
                    sp.set_attribute("aletheia.al.Sigma.d1", float(diag[1]))
                    sp.set_attribute(
                        "aletheia.al.contraction.d0",
                        float(diag[0] / max(diag_seed[0], 1e-30)))
                    sp.set_attribute(
                        "aletheia.al.contraction.d1",
                        float(diag[1] / max(diag_seed[1], 1e-30)))

                traj["cycle"].append(c)
                traj["context_size"].append(int(len(M_ctx)))
                traj["Sigma_d0"].append(float(diag[0]))
                traj["Sigma_d1"].append(float(diag[1]))
                traj["contraction_d0"].append(float(diag[0] / max(diag_seed[0], 1e-30)))
                traj["contraction_d1"].append(float(diag[1] / max(diag_seed[1], 1e-30)))
                traj["pick_m"].append(float(pick_m[0]))
                traj["pool_cv_ig"].append(cv)
                traj["pool_cos_Ainv_top"].append(cos_top)
                traj["eig_kappa"].append(float(eig.kappa))
                traj["eig_min"].append(float(eig.lam.min()))
                traj["eig_max"].append(float(eig.lam.max()))

        # Terminal MLE error on the resolved subspace.
        from stage_c_design_only import _mle_on_context  # noqa: E402
        mle_err = _mle_on_context(oracle, target_c, M_ctx, Y_ctx, probe["V"])
        root.set_attribute("aletheia.al.terminal.mle_err_d0", float(mle_err[0]))
        root.set_attribute("aletheia.al.terminal.mle_err_d1", float(mle_err[1]))
        root.set_attribute("aletheia.al.terminal.contraction_d1",
                           float(traj["contraction_d1"][-1]))
        root.set_attribute("aletheia.al.wall_seconds", time.time() - t0)

    summary = {
        "acq": ACQ, "pool": POOL, "K_ctx": K_CTX_ENV, "seed": SEED,
        "n_cycles": N_CYCLES,
        "Sigma_seed_diag": diag_seed.tolist(),
        "Sigma_final_diag": diag.tolist(),
        "contraction_final_d0": float(diag[0] / max(diag_seed[0], 1e-30)),
        "contraction_final_d1": float(diag[1] / max(diag_seed[1], 1e-30)),
        "mle_err_d0": float(mle_err[0]),
        "mle_err_d1": float(mle_err[1]),
        "wall_seconds": time.time() - t0,
        "trajectory": traj,
    }
    out_dir = OUT / f"closed_loop_{POOL}_K{K_CTX_ENV}_N{N_CYCLES}_{ACQ}_s{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[closed_loop] {ACQ}/{POOL}/K{K_CTX_ENV}/seed{SEED}: "
          f"contraction_d1={summary['contraction_final_d1']:.4g}  "
          f"mle_err_d1={summary['mle_err_d1']:.4g}  "
          f"wall={summary['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
