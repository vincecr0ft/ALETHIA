"""Stage B — the EIG-spread diagnostic and the go/no-go gate (AL_separation §2),
extended with the redundancy diagnostic (§3.3 fallback) that Stage A's
falsification of "CV(IG) << 1 on P0" makes mandatory.

The plan's gate is CV(IG) ≳ 1. Stage A showed P0 already meets that, so a
naive gate-pass would push us into Stage C. The redundancy diagnostic asks
the harder question: are the high-IG candidates *mutually informative for the
same direction*? If so, the IG spread is a mirage — random's coverage already
catches that direction by chance.

Emits Phoenix spans into project ``alethia-al-studies``:
  - chain.al.stage_b.eig_spread.P0
  - chain.al.stage_b.eig_spread.P1
  - chain.al.stage_b.eig_spread.P2
  - chain.al.stage_b.redundancy.P0
  - chain.al.stage_b.redundancy.P1
  - chain.al.stage_b.redundancy.P2
  - chain.al.stage_b.gate

Writes output/stage_b_summary.json with the gate decision and per-pool
distributions.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from _common import (  # noqa: E402
    M_RANGE, OUT,
    build_oracle, build_pool_P0, build_pool_P1, build_pool_P2,
    ig_per_candidate, load_pretrained_model, load_probe,
    target_context_mu_fb, tracer,
)


SEED = int(os.environ.get("SEED", "2026"))


def _ig_stats(ig: np.ndarray, top_k: int = 5) -> dict[str, float]:
    mu = float(ig.mean())
    sd = float(ig.std())
    med = float(np.median(ig))
    top = float(np.sort(ig)[-top_k:].mean())
    return {
        "ig_mean": mu, "ig_std": sd, "ig_median": med,
        "ig_min": float(ig.min()), "ig_max": float(ig.max()),
        "ig_cv": sd / mu if mu > 0 else float("inf"),
        "ig_top_k_mean": top, "top_k": top_k,
        "spread_topk_over_median": (top - med) / med if med > 0 else float("inf"),
    }


def _redundancy(model, M_ctx, Y_ctx, pool: np.ndarray, *,
                top_decile_frac: float = 0.1) -> dict[str, Any]:
    """Closed-form redundancy diagnostic for the AL_separation §3.3 fallback.

    For a candidate pair (p, q), the joint IG of acquiring both is
        IG(p, q) = IG(p) + IG(q | p)
    and the redundancy gap is
        R(p, q) = IG(p) + IG(q) - IG(p, q) ≥ 0.
    On the linear-Gaussian head this is closed-form because A is additive,
    so we compute it on the top-decile-IG candidates and report the mean
    redundancy as a fraction of the per-candidate IG. High R/IG means a
    random draw is likely to land "good enough" on the same direction.
    """
    ig = ig_per_candidate(model, M_ctx, Y_ctx, pool)
    n_top = max(2, int(np.ceil(top_decile_frac * len(pool))))
    top_idx = np.argsort(ig)[-n_top:]
    Psi = model.psi_np(pool)
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)

    # Pairwise IG(q | p) via Sherman-Morrison on A_inv.
    def cond_ig(p_idx: int, q_idx: int) -> float:
        p = Psi[p_idx]
        Ap = A_inv @ p
        denom = 1.0 + p @ Ap
        A_inv_post = A_inv - np.outer(Ap, Ap) / denom
        q = Psi[q_idx]
        lev_q = float(q @ A_inv_post @ q)
        return 0.5 * np.log1p(max(lev_q, 0.0))

    R = []
    R_norm = []
    cos_psi = []  # A_inv-inner-product cosine, matches AL_separation §3.3
    for i, p_idx in enumerate(top_idx):
        for q_idx in top_idx[i + 1:]:
            ig_pq = ig[p_idx] + cond_ig(int(p_idx), int(q_idx))
            ig_indep = ig[p_idx] + ig[q_idx]
            redundancy = ig_indep - ig_pq
            R.append(redundancy)
            R_norm.append(redundancy / max(ig[p_idx], 1e-12))
            # A_inv-inner-product cosine.
            p = Psi[p_idx]; q = Psi[q_idx]
            num = float(p @ A_inv @ q)
            den = np.sqrt(float(p @ A_inv @ p) * float(q @ A_inv @ q))
            cos_psi.append(num / max(den, 1e-30))
    R = np.array(R)
    R_norm = np.array(R_norm)
    cos_psi = np.array(cos_psi)
    return {
        "top_decile_size": int(n_top),
        "redundancy_mean": float(R.mean()),
        "redundancy_median": float(np.median(R)),
        "redundancy_over_ig_mean": float(R_norm.mean()),
        "redundancy_over_ig_max": float(R_norm.max()),
        "cos_Ainv_mean_abs": float(np.mean(np.abs(cos_psi))),
        "cos_Ainv_max_abs": float(np.max(np.abs(cos_psi))),
        # Redundancy diagnostic interpretation per AL_separation §3.3: top
        # candidates near-collinear in the A⁻¹ metric → coverage captures
        # the same information → batch-aware selection (not greedy) needed.
        "high_redundancy_warning": float(np.mean(np.abs(cos_psi))) > 0.7,
    }


def stage_b_run() -> dict:
    model = load_pretrained_model()
    oracle = build_oracle(seed=SEED)
    probe = load_probe(oracle)

    rng = np.random.default_rng(SEED)
    target_c, M_ctx, Y_ctx = target_context_mu_fb(rng, oracle)

    pools = {
        "P0": build_pool_P0(rng, size=30),
        "P1": build_pool_P1(rng, size=500),
        "P2": build_pool_P2(rng, size=500),
    }

    tr = tracer()
    summary: dict[str, Any] = {
        "seed": SEED,
        "context_size": int(len(M_ctx)),
        "M_range": list(M_RANGE),
        "pools": {},
        "redundancy": {},
    }

    with tr.start_as_current_span("chain.al.stage_b.run") as root:
        root.set_attribute("aletheia.al.stage", "B")
        root.set_attribute("aletheia.al.seed", SEED)

        for name, pool in pools.items():
            ig = ig_per_candidate(model, M_ctx, Y_ctx, pool)
            stats = _ig_stats(ig)
            with tr.start_as_current_span(f"chain.al.stage_b.eig_spread.{name}") as sp:
                sp.set_attribute("aletheia.al.pool", name)
                sp.set_attribute("aletheia.al.pool.size", int(len(pool)))
                for k, v in stats.items():
                    sp.set_attribute(f"aletheia.al.eig_spread.{k}", v)
            summary["pools"][name] = {
                **stats,
                "n_pool": int(len(pool)),
                "ig_distribution": ig.tolist(),
            }

            red = _redundancy(model, M_ctx, Y_ctx, pool)
            with tr.start_as_current_span(f"chain.al.stage_b.redundancy.{name}") as sp:
                sp.set_attribute("aletheia.al.pool", name)
                for k, v in red.items():
                    sp.set_attribute(f"aletheia.al.redundancy.{k}", v)
            summary["redundancy"][name] = red

        # Gate decision. Plan §2.3: CV(IG) ≳ 1 on a physical pool → proceed.
        # Augmented with the redundancy check: if the top candidates are
        # near-collinear in A⁻¹, the high CV is a mirage.
        def _pool_passes(name: str) -> tuple[bool, str]:
            cv = summary["pools"][name]["ig_cv"]
            high_red = summary["redundancy"][name]["high_redundancy_warning"]
            if cv < 1.0:
                return False, f"CV(IG)={cv:.3f} < 1 — flat pool"
            if high_red:
                cos = summary["redundancy"][name]["cos_Ainv_mean_abs"]
                return False, f"CV(IG)={cv:.3f} OK but mean |cos_Ainv|={cos:.3f} > 0.7 — redundant top candidates"
            return True, f"CV(IG)={cv:.3f} ≥ 1 and redundancy low — pool can separate"

        gate_per_pool = {n: _pool_passes(n) for n in pools}
        any_passes = any(passes for passes, _ in gate_per_pool.values())
        best_pool = max(
            pools.keys(),
            key=lambda n: (
                gate_per_pool[n][0],
                summary["pools"][n]["ig_cv"]
                - (10.0 if summary["redundancy"][n]["high_redundancy_warning"] else 0.0),
            ),
        )

        with tr.start_as_current_span("chain.al.stage_b.gate") as sp:
            sp.set_attribute("aletheia.al.gate.any_pool_passes", any_passes)
            sp.set_attribute("aletheia.al.gate.best_pool", best_pool)
            for n, (ok, why) in gate_per_pool.items():
                sp.set_attribute(f"aletheia.al.gate.{n}.pass", bool(ok))
                sp.set_attribute(f"aletheia.al.gate.{n}.reason", why)
            sp.set_attribute(
                "aletheia.al.gate.decision",
                "PROCEED to Stage C" if any_passes else
                "STOP — physical null; report Stage B diagnostic as the contribution"
            )

        summary["gate"] = {
            "any_pool_passes": any_passes,
            "best_pool": best_pool,
            "per_pool": {n: {"pass": ok, "reason": why}
                          for n, (ok, why) in gate_per_pool.items()},
            "decision": "PROCEED" if any_passes else "STOP",
        }

    out_path = OUT / "stage_b_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[stage_b] wrote {out_path}")
    for n in pools:
        stats = summary["pools"][n]
        red = summary["redundancy"][n]
        ok, why = gate_per_pool[n]
        print(f"  {n}: CV(IG)={stats['ig_cv']:.3f}  "
              f"top/median={stats['spread_topk_over_median']:.2f}  "
              f"|cos_Ainv|_mean={red['cos_Ainv_mean_abs']:.3f}  "
              f"gate={'PASS' if ok else 'fail'} — {why}")
    print(f"  decision: {summary['gate']['decision']}, best_pool={best_pool}")
    return summary


if __name__ == "__main__":
    stage_b_run()
