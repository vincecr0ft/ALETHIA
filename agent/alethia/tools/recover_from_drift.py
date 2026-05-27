"""recover_from_drift: combined EPIG -> oracle -> fm_update -> recalibrate.

This single-call tool runs the local-retrain chain in one shot. It exists
because Gemini sometimes parallelises the equivalent three-call sequence
(epig_select, oracle_query, fm_update) and hallucinates intermediate
values. Wrapping the three steps as one deterministic tool eliminates the
data-dependency exposure to the model while keeping each sub-step in the
Phoenix trace as its own child span (chain.recover_from_drift.*).
"""
from __future__ import annotations

import time

import numpy as np
from opentelemetry import trace

from alethia.state import STATE
from modules.surrogate.intention import epig_acquire_m, target_set_entropy

_tracer = trace.get_tracer("alethia.tools")


def recover_from_drift(k: int = 5, m_lo_tev: float = 1.0,
                       m_hi_tev: float = 2.2,
                       fidelity: str = "T1") -> dict:
    """Execute one drift-recovery cycle: pick k m-values via EPIG, query
    the oracle at them, fold the results into the FM context, refit
    conformal.

    Use this tool when ``check_drift`` returns ``action ==
    'local_retrain'``. The full sub-chain (EPIG, oracle, FM update,
    recalibration) is run server-side; Gemini does not need to pass any
    intermediate values.

    Args:
        k: number of oracle commissions to acquire.
        m_lo_tev: lower bound of the EPIG candidate pool, TeV.
        m_hi_tev: upper bound.
        fidelity: "T1" (analytic, ~ms) or "T2" (MadGraph, ~30 s/call).

    Returns:
        Combined trajectory of the recovery: picked m-values, oracle mu
        values, H_T before/after, context size before/after, oracle
        wall time, and post-update kappa.
    """
    STATE.ensure_initialised()
    M_target = np.linspace(0.4, 2.2, 50)

    with _tracer.start_as_current_span("chain.recover_from_drift") as root:
        root.set_attribute("aletheia.recover.k", int(k))
        root.set_attribute("aletheia.recover.m_lo_tev", float(m_lo_tev))
        root.set_attribute("aletheia.recover.m_hi_tev", float(m_hi_tev))
        root.set_attribute("aletheia.recover.fidelity_tier", fidelity)

        # 1) EPIG.
        with _tracer.start_as_current_span("tool.epig.select") as sp:
            rng = np.random.default_rng()
            M_pool = rng.uniform(m_lo_tev, m_hi_tev, size=200)
            idx = epig_acquire_m(STATE.model, STATE.M_ctx, STATE.Y_ctx,
                                 M_pool, M_target, k=k)
            picked_m = M_pool[idx]
            sp.set_attribute("aletheia.epig.k", int(k))
            sp.set_attribute("aletheia.epig.pool_size", 200)
            sp.set_attribute("aletheia.epig.picked_m_values",
                             [float(m) for m in picked_m])

        # 2) Oracle.
        with _tracer.start_as_current_span("tool.oracle.query") as oq:
            t0 = time.time()
            picked_y = STATE._oracle_truth(picked_m, fidelity=fidelity)
            oracle_wall = time.time() - t0
            STATE.oracle_calls += k
            oq.set_attribute("aletheia.oracle.fidelity_tier", fidelity)
            oq.set_attribute("aletheia.oracle.n_points", int(k))
            oq.set_attribute("aletheia.oracle.cost_seconds",
                             float(oracle_wall))
            oq.set_attribute("aletheia.oracle.wilson_norm",
                             float(np.linalg.norm(STATE.target_c)))
            oq.set_attribute("aletheia.oracle.cumulative_calls",
                             int(STATE.oracle_calls))

        # 3) FM update.
        with _tracer.start_as_current_span("tool.fm.update") as fu:
            H_T_pre = target_set_entropy(STATE.model, STATE.M_ctx,
                                         STATE.Y_ctx, M_target)
            ctx_in = len(STATE.M_ctx)
            STATE.M_ctx = np.concatenate([STATE.M_ctx, picked_m])
            STATE.Y_ctx = np.concatenate([STATE.Y_ctx, picked_y])
            H_T_post = target_set_entropy(STATE.model, STATE.M_ctx,
                                          STATE.Y_ctx, M_target)
            kappa_post = STATE.model.kappa_A(STATE.M_ctx)
            fu.set_attribute("aletheia.fm.context_size_in", int(ctx_in))
            fu.set_attribute("aletheia.fm.context_size_out",
                             int(len(STATE.M_ctx)))
            fu.set_attribute("aletheia.fm.target_entropy_H_T_pre",
                             float(H_T_pre))
            fu.set_attribute("aletheia.fm.target_entropy_H_T_post",
                             float(H_T_post))
            fu.set_attribute("aletheia.fm.condition_number_kappa",
                             float(kappa_post))

        # 4) Refit conformal.
        STATE.calibrator.fit(STATE.model, STATE.M_ctx, STATE.Y_ctx,
                             STATE.M_cal, STATE.Y_cal)

        root.set_attribute("aletheia.recover.H_T_pre", float(H_T_pre))
        root.set_attribute("aletheia.recover.H_T_post", float(H_T_post))
        root.set_attribute("aletheia.recover.H_T_delta",
                           float(H_T_post - H_T_pre))

    return {
        "picked_m_values": [float(m) for m in picked_m],
        "oracle_mu_values": [float(v) for v in picked_y],
        "fidelity_tier": fidelity,
        "k": int(k),
        "oracle_wall_seconds": float(oracle_wall),
        "context_size_in": int(ctx_in),
        "context_size_out": int(len(STATE.M_ctx)),
        "H_T_pre": float(H_T_pre),
        "H_T_post": float(H_T_post),
        "H_T_delta": float(H_T_post - H_T_pre),
        "kappa_after": float(kappa_post),
        "cumulative_oracle_calls": int(STATE.oracle_calls),
    }
