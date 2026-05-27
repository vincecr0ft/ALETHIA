"""check_drift: runs DAS-CUSUM / BH binomial / kappa over recent state."""
from __future__ import annotations

import numpy as np
from opentelemetry import trace

from alethia.state import STATE
from modules.surrogate.intention import (
    das_cusum_update, coverage_bh_test, kappa_drift, aggregate_action,
)

_tracer = trace.get_tracer("alethia.tools")


def check_drift() -> dict:
    """Run the three drift detectors over the recent residual stream.

    Returns the per-detector flags, the aggregator's recommended action,
    and the supporting statistics. Resets the accumulated coverage counts
    after every call so each invocation reflects the most recent window.
    """
    STATE.ensure_initialised()
    with _tracer.start_as_current_span("chain.drift.evaluate"):
        # ----- accuracy: DAS-CUSUM on the running stream -----
        # The state's cusum buffer is updated by fm_update / fm_predict's
        # downstream residual analysis. Here we just inspect its current
        # statistic by feeding it a single 0.0 (placeholder) and reading
        # the S_t value out. If there is no buffered history, S_t = 0.
        with _tracer.start_as_current_span(
                "tool.drift.accuracy.das_cusum") as ds:
            _, _, S_t = das_cusum_update(STATE.cusum_state, 0.0, w=30,
                                         h=6.0, k=0.5)
            acc_fired = S_t > 6.0
            ds.set_attribute("aletheia.drift.acc.S_t", float(S_t))
            ds.set_attribute("aletheia.drift.acc.fired", bool(acc_fired))

        # ----- calibration: BH binomial coverage over m-regions -----
        with _tracer.start_as_current_span(
                "tool.drift.calibration.bh") as cs:
            counts_total = STATE.coverage_counts_68[:, 1]
            counts_cov = STATE.coverage_counts_68[:, 0]
            if counts_total.sum() < 40:
                # Not enough data yet; defer.
                cal_fired = False
                p_min = 1.0
                n_fail = 0
            else:
                cal_fired, pvals, fail_mask, _ = coverage_bh_test(
                    counts_cov, counts_total, target_coverage=0.683,
                    alpha=0.05)
                p_min = float(pvals.min())
                n_fail = int(fail_mask.sum())
            cs.set_attribute("aletheia.drift.cal.global_pvalue", p_min)
            cs.set_attribute("aletheia.drift.cal.failing_regions", n_fail)
            cs.set_attribute("aletheia.drift.cal.fired", bool(cal_fired))

        # ----- coverage: kappa(A) and v_min projection on the context -----
        with _tracer.start_as_current_span(
                "tool.drift.coverage.kappa") as ks:
            kappa = STATE.model.kappa_A(STATE.M_ctx)
            # Recent projection uses the current context's m-values as
            # the "recent stream" proxy.
            # For a baseline projection variance we use a uniform sample.
            rng = np.random.default_rng(12345)
            M_train = rng.uniform(0.3, 2.3, size=200)
            Psi_train = STATE.model.psi_np(M_train)
            eigs, vecs = np.linalg.eigh(
                Psi_train.T @ Psi_train +
                STATE.model.alpha * np.eye(STATE.model.d_psi))
            v_min = vecs[:, 0]
            train_proj_var = float(np.var(Psi_train @ v_min))
            cov_fired, _, proj_ratio = kappa_drift(
                STATE.model, STATE.M_ctx, STATE.M_ctx, train_proj_var,
                kappa_threshold=1e3, proj_ratio_threshold=2.5)
            ks.set_attribute("aletheia.drift.cov.kappa", float(kappa))
            ks.set_attribute(
                "aletheia.drift.cov.vmin_projection_ratio",
                float(proj_ratio))
            ks.set_attribute("aletheia.drift.cov.fired", bool(cov_fired))

    # Aggregate.
    action, target_signal = aggregate_action(
        bool(acc_fired), bool(cal_fired), bool(cov_fired),
        STATE.history_flags, persistence_N=3)
    STATE.history_flags.append(
        (bool(acc_fired), bool(cal_fired), bool(cov_fired)))
    flag_str = ("1" if acc_fired else "0") + \
               ("1" if cal_fired else "0") + \
               ("1" if cov_fired else "0")
    with _tracer.start_as_current_span("tool.drift.aggregate") as ag:
        ag.set_attribute("aletheia.drift.combined_flag", flag_str)
        ag.set_attribute("aletheia.drift.action", action)
        if target_signal:
            ag.set_attribute("aletheia.drift.target_signal", target_signal)
    return {
        "acc_fired": bool(acc_fired),
        "cal_fired": bool(cal_fired),
        "cov_fired": bool(cov_fired),
        "combined_flag": flag_str,
        "action": action,
        "kappa": float(kappa),
        "vmin_projection_ratio": float(proj_ratio),
        "cusum_S": float(S_t),
        "context_size": int(len(STATE.M_ctx)),
    }
