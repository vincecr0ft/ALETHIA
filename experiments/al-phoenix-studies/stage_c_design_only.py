"""Stage C, design-only: closed-form contraction & MLE sweep.

The AL_separation plan §3 specifies a full closed-loop run; this script
implements the design-only special case that the closed-form head supports.
Per the plan §3.2.1, posterior-variance contraction Σ_aa^final / Σ_aa^seed
*is* the primary metric and is design-only (label-independent) on the
linear-Gaussian head. The MLE per-direction error (§3.2.2) is the inference
estimator and depends on labels, so we evaluate it on actual oracle labels.

Configurations swept (cartesian):
  pool      ∈ {P0 (30 uniform), P1 (500 uniform), P2 (500 log+tail)}
  K_ctx     ∈ {12, 24, 48}                                   — postmortem §6.1
  budget    ∈ {1, 5}     picks per chain                     — stressed / generous
  acq       ∈ {random, leverage, epig, param_epig_d, param_epig_a@d1}
  seed      ∈ {2026, ..., 2026 + n_seeds - 1}

Emits one Phoenix span per (config, seed, acq) plus an aggregation span.

Writes output/stage_c_design_only.json with per-acquisition contraction
ratios, MLE errors, and Welch-t separation against random.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from _common import (  # noqa: E402
    M_RANGE, N_WC, OUT, WITHHOLD_DIM, WITHHOLD_DIM_2, TARGET_C_LQ3,
    TARGET_C_HQ3, build_oracle, build_pool_P0, build_pool_P1, build_pool_P2,
    ig_per_candidate, load_pretrained_model, load_probe, sigma_ctilde,
    target_context_mu_fb, truth_mu_fb, tracer,
)
from modules.surrogate.intention import (  # noqa: E402
    epig_acquire_m, param_epig_d_acquire, param_epig_a_acquire,
)


def _build_pool(name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "P0":
        return build_pool_P0(rng, size=30)
    if name == "P1":
        return build_pool_P1(rng, size=500)
    if name == "P2":
        return build_pool_P2(rng, size=500)
    raise ValueError(name)


def _seed_context(rng: np.random.Generator, oracle, K: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin seed context of size K in [0.5, 1.0] for the bimodal target."""
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = TARGET_C_LQ3
    target_c[WITHHOLD_DIM_2] = TARGET_C_HQ3
    M_ctx = rng.uniform(0.5, 1.0, size=K)
    Y_ctx = truth_mu_fb(oracle, target_c, M_ctx)
    return target_c, M_ctx, Y_ctx


def _mle_on_context(oracle, target_c: np.ndarray, M_ctx: np.ndarray,
                    Y_ctx: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Analytic morphing MLE on the active context (mu_FB observable).

    Returns per-direction error |c̃_MLE - c̃_true| in the Fisher rotation.
    """
    from scipy.optimize import least_squares
    # Build morphing at the context m-values via finite differences.
    m = M_ctx
    K = len(m)
    h = 1e-3
    zero = np.zeros((K, N_WC))
    Y_SM = oracle.truth_mu_fb(zero, m)
    A = np.empty((K, N_WC))
    for i in range(N_WC):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        A[:, i] = (oracle.truth_mu_fb(cp, m) - oracle.truth_mu_fb(cm, m)) / (2.0 * h)
    Bm = np.zeros((K, N_WC, N_WC))
    for i in range(N_WC):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        Bm[:, i, i] = (oracle.truth_mu_fb(cp, m) + oracle.truth_mu_fb(cm, m)
                       - 2.0 * Y_SM) / (h * h)
    for i in range(N_WC):
        for j in range(i + 1, N_WC):
            c_pp = zero.copy(); c_pp[:, i] += h; c_pp[:, j] += h
            c_pm = zero.copy(); c_pm[:, i] += h; c_pm[:, j] -= h
            c_mp = zero.copy(); c_mp[:, i] -= h; c_mp[:, j] += h
            c_mm = zero.copy(); c_mm[:, i] -= h; c_mm[:, j] -= h
            Hij = (oracle.truth_mu_fb(c_pp, m) - oracle.truth_mu_fb(c_pm, m)
                   - oracle.truth_mu_fb(c_mp, m) + oracle.truth_mu_fb(c_mm, m)) / (4.0 * h * h)
            Bm[:, i, j] = Hij; Bm[:, j, i] = Hij
    Bm = 0.5 * Bm

    def residuals(c):
        Y_pred = Y_SM + A @ c + np.einsum("i,kij,j->k", c, Bm, c)
        return (Y_ctx - Y_pred) / 0.05

    sol = least_squares(residuals, x0=np.zeros(N_WC), method="lm", max_nfev=200)
    c_mle = sol.x
    c_tilde_mle = V.T @ c_mle
    c_tilde_true = V.T @ target_c
    return np.abs(c_tilde_mle - c_tilde_true)


def _acquire(model, M_ctx, Y_ctx, M_pool, P, sigma_y, M_target, acq: str,
              k: int, rng: np.random.Generator) -> np.ndarray:
    if acq == "random":
        return M_pool[rng.choice(len(M_pool), size=k, replace=False)]
    if acq == "leverage":
        # Greedy D-optimal on A: pick k highest-leverage pool points,
        # updating A_inv via Sherman-Morrison.
        A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
        Psi = model.psi_np(M_pool)
        avail = np.ones(len(M_pool), dtype=bool)
        chosen = []
        for _ in range(k):
            lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
            lev = np.where(avail, lev, -np.inf)
            i = int(np.argmax(lev))
            chosen.append(i)
            avail[i] = False
            p = Psi[i]
            Ap = A_inv @ p
            A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
        return M_pool[chosen]
    if acq == "epig":
        idx = epig_acquire_m(model, M_ctx, Y_ctx, M_pool, M_target, k=k)
        return M_pool[idx]
    if acq == "param_epig_d":
        idx = param_epig_d_acquire(model, M_ctx, Y_ctx, M_pool, P, k=k,
                                    sigma_y=sigma_y, resolved_dim=2)
        return M_pool[idx]
    if acq == "param_epig_a":
        idx = param_epig_a_acquire(model, M_ctx, Y_ctx, M_pool, P, k=k,
                                    target_direction=1, sigma_y=sigma_y)
        return M_pool[idx]
    raise ValueError(acq)


def run_chain(model, oracle, probe, *, pool_name: str, K: int, k_picks: int,
              acq: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    target_c, M_ctx, Y_ctx = _seed_context(rng, oracle, K)
    pool = _build_pool(pool_name, rng)
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)

    # Pre-acquisition Σ_c̃ on resolved subspace.
    S_pre = sigma_ctilde(model, M_ctx, Y_ctx, probe["P"], probe["sigma_y"],
                          resolved_dim=2)
    diag_pre = np.diag(S_pre).copy()

    # Acquire.
    picks = _acquire(model, M_ctx, Y_ctx, pool, probe["P"], probe["sigma_y"],
                      M_target, acq, k_picks, rng)
    # Fold in oracle labels for the picks.
    Y_picks = truth_mu_fb(oracle, target_c, picks)
    M_aug = np.concatenate([M_ctx, picks])
    Y_aug = np.concatenate([Y_ctx, Y_picks])

    S_post = sigma_ctilde(model, M_aug, Y_aug, probe["P"], probe["sigma_y"],
                          resolved_dim=2)
    diag_post = np.diag(S_post).copy()
    contraction = diag_post / np.maximum(diag_pre, 1e-30)

    # MLE per-direction error on the post-acquisition context.
    mle_err = _mle_on_context(oracle, target_c, M_aug, Y_aug, probe["V"])

    return {
        "pool": pool_name,
        "K_ctx": K,
        "k_picks": k_picks,
        "acq": acq,
        "seed": seed,
        "context_size": int(len(M_aug)),
        "Sigma_pre_diag": diag_pre.tolist(),
        "Sigma_post_diag": diag_post.tolist(),
        "contraction_d0": float(contraction[0]),
        "contraction_d1": float(contraction[1]),
        "mle_err_per_direction": mle_err.tolist(),
    }


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch t-statistic and (mean_diff, se_diff)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    va = a.var(ddof=1) / n_a
    vb = b.var(ddof=1) / n_b
    se = np.sqrt(va + vb)
    md = a.mean() - b.mean()
    t = md / max(se, 1e-30)
    return float(t), float(md), float(se)


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Group by (pool, K, k_picks, acq) and report mean ± std + Welch vs random."""
    keys = [(r["pool"], r["K_ctx"], r["k_picks"], r["acq"]) for r in rows]
    grouped: dict[tuple, list[dict]] = {}
    for k, r in zip(keys, rows):
        grouped.setdefault(k, []).append(r)
    out_groups: list[dict[str, Any]] = []
    # Welch needs the random arm per config.
    cfg_random: dict[tuple, dict[str, np.ndarray]] = {}
    for (pool, K, kp, acq), rs in grouped.items():
        cd1 = np.array([r["contraction_d1"] for r in rs])
        mle1 = np.array([r["mle_err_per_direction"][1] for r in rs])
        if acq == "random":
            cfg_random[(pool, K, kp)] = {"cd1": cd1, "mle1": mle1}
    for (pool, K, kp, acq), rs in grouped.items():
        cd0 = np.array([r["contraction_d0"] for r in rs])
        cd1 = np.array([r["contraction_d1"] for r in rs])
        mle0 = np.array([r["mle_err_per_direction"][0] for r in rs])
        mle1 = np.array([r["mle_err_per_direction"][1] for r in rs])
        entry = {
            "pool": pool, "K_ctx": K, "k_picks": kp, "acq": acq,
            "n_seeds": len(rs),
            "contraction_d0_mean": float(cd0.mean()), "contraction_d0_std": float(cd0.std(ddof=1)),
            "contraction_d1_mean": float(cd1.mean()), "contraction_d1_std": float(cd1.std(ddof=1)),
            "mle_err_d0_mean": float(mle0.mean()), "mle_err_d0_std": float(mle0.std(ddof=1)),
            "mle_err_d1_mean": float(mle1.mean()), "mle_err_d1_std": float(mle1.std(ddof=1)),
        }
        if acq != "random" and (pool, K, kp) in cfg_random:
            ref = cfg_random[(pool, K, kp)]
            t_c, md_c, se_c = _welch(cd1, ref["cd1"])
            t_m, md_m, se_m = _welch(mle1, ref["mle1"])
            entry["welch_t_contraction_d1_vs_random"] = t_c
            entry["welch_md_contraction_d1_vs_random"] = md_c
            entry["welch_se_contraction_d1_vs_random"] = se_c
            entry["welch_t_mle_d1_vs_random"] = t_m
            entry["welch_md_mle_d1_vs_random"] = md_m
            entry["welch_se_mle_d1_vs_random"] = se_m
        out_groups.append(entry)
    return {"groups": out_groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["P0", "P1", "P2"])
    ap.add_argument("--K", type=int, nargs="+", default=[12, 24, 48])
    ap.add_argument("--k-picks", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--acq", nargs="+",
                    default=["random", "leverage", "epig",
                              "param_epig_d", "param_epig_a"])
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=2026)
    args = ap.parse_args()

    model = load_pretrained_model()
    oracle = build_oracle(seed=0)
    probe = load_probe(oracle)

    tr = tracer()
    rows: list[dict] = []
    t0 = time.time()
    with tr.start_as_current_span("chain.al.stage_c.design_only.run") as root:
        root.set_attribute("aletheia.al.stage", "C-design-only")
        root.set_attribute("aletheia.al.n_seeds", args.n_seeds)
        root.set_attribute("aletheia.al.pools", args.pools)
        root.set_attribute("aletheia.al.K_ctx_grid", args.K)
        root.set_attribute("aletheia.al.k_picks_grid", args.k_picks)
        root.set_attribute("aletheia.al.acquisitions", args.acq)

        for pool_name in args.pools:
            for K in args.K:
                for kp in args.k_picks:
                    for acq in args.acq:
                        for s in range(args.n_seeds):
                            seed = args.seed0 + s
                            chain = run_chain(model, oracle, probe,
                                              pool_name=pool_name,
                                              K=K, k_picks=kp,
                                              acq=acq, seed=seed)
                            rows.append(chain)
        # Aggregate now so we can attach summary attributes to the root span.
        agg = aggregate(rows)
        # Add headline: best (lowest mean contraction) per config across acqs.
        per_cfg = {}
        for g in agg["groups"]:
            key = (g["pool"], g["K_ctx"], g["k_picks"])
            per_cfg.setdefault(key, []).append(g)
        headlines = []
        for (pool, K, kp), gs in per_cfg.items():
            best = min(gs, key=lambda gg: gg["contraction_d1_mean"])
            rng_g = next((gg for gg in gs if gg["acq"] == "random"), None)
            if rng_g and best["acq"] != "random":
                lift = (rng_g["contraction_d1_mean"]
                        - best["contraction_d1_mean"]) / max(
                            rng_g["contraction_d1_std"], 1e-12)
                headlines.append({
                    "pool": pool, "K_ctx": K, "k_picks": kp,
                    "best_acq": best["acq"],
                    "best_contraction_d1": best["contraction_d1_mean"],
                    "random_contraction_d1": rng_g["contraction_d1_mean"],
                    "lift_over_random_in_random_std": float(lift),
                })
        root.set_attribute("aletheia.al.stage_c.n_chains", len(rows))
        root.set_attribute("aletheia.al.stage_c.wall_seconds",
                           time.time() - t0)
        # Compact summary string for the trace viewer.
        root.set_attribute(
            "aletheia.al.stage_c.headline_summary",
            "; ".join(
                f"{h['pool']}/K{h['K_ctx']}/k{h['k_picks']}: "
                f"best={h['best_acq']} lift={h['lift_over_random_in_random_std']:.2f}σ"
                for h in headlines))

    summary = {
        "wall_seconds": time.time() - t0,
        "n_chains": len(rows),
        "config": vars(args),
        "headlines": headlines,
        "aggregate": agg,
        "rows": rows,
    }
    out_path = OUT / "stage_c_design_only.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[stage_c_design] wrote {out_path} ({len(rows)} chains, "
          f"{time.time()-t0:.1f}s)")
    # Print headlines sorted by separation strength.
    headlines.sort(key=lambda h: -h["lift_over_random_in_random_std"])
    for h in headlines[:12]:
        print(f"  {h['pool']}/K{h['K_ctx']}/k{h['k_picks']}: "
              f"best={h['best_acq']}  "
              f"contraction(best)={h['best_contraction_d1']:.3g}  "
              f"contraction(random)={h['random_contraction_d1']:.3g}  "
              f"lift={h['lift_over_random_in_random_std']:+.2f}σ")
    return summary


if __name__ == "__main__":
    main()
