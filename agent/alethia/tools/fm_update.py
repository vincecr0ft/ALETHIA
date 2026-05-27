"""fm_update: fold new (m, mu) observations into the FM context, recalibrate."""
from __future__ import annotations

import numpy as np
from opentelemetry import trace

from alethia.state import STATE
from modules.surrogate.intention import target_set_entropy

_tracer = trace.get_tracer("alethia.tools")


def fm_update(m_values: list[float], mu_values: list[float]) -> dict:
    """Fold new (m, mu) observations into the Intention head's context.

    Closed-form: no gradient descent. The Intention attention is exact for
    the new context. After the update, refits the conformal calibrator on
    the broader probe region (invariant: docs/research/04-epig-conformal/
    eval.md section 2).

    Args:
        m_values: list of m_ll in TeV. Must have same length as mu_values.
        mu_values: oracle-returned mu values (from oracle_query).

    Returns:
        H_T_pre, H_T_post (target-set entropy), context_size_in/out, kappa.
    """
    STATE.ensure_initialised()
    m = np.asarray(m_values, dtype=float)
    mu = np.asarray(mu_values, dtype=float)
    if m.shape != mu.shape:
        raise ValueError(
            f"shape mismatch: m {m.shape} vs mu {mu.shape}")

    # Use a fixed target set for the H_T tracker (same as run.py).
    M_target = np.linspace(0.4, 2.2, 50)
    with _tracer.start_as_current_span("tool.fm.update") as sp:
        H_T_pre = target_set_entropy(STATE.model, STATE.M_ctx,
                                     STATE.Y_ctx, M_target)
        ctx_in = len(STATE.M_ctx)
        STATE.M_ctx = np.concatenate([STATE.M_ctx, m])
        STATE.Y_ctx = np.concatenate([STATE.Y_ctx, mu])
        H_T_post = target_set_entropy(STATE.model, STATE.M_ctx,
                                      STATE.Y_ctx, M_target)
        kappa_post = STATE.model.kappa_A(STATE.M_ctx)
        # Refit the calibrator (invariant).
        STATE.calibrator.fit(STATE.model, STATE.M_ctx, STATE.Y_ctx,
                             STATE.M_cal, STATE.Y_cal)
        sp.set_attribute("aletheia.fm.context_size_in", int(ctx_in))
        sp.set_attribute("aletheia.fm.context_size_out",
                         int(len(STATE.M_ctx)))
        sp.set_attribute("aletheia.fm.target_entropy_H_T_pre",
                         float(H_T_pre))
        sp.set_attribute("aletheia.fm.target_entropy_H_T_post",
                         float(H_T_post))
        sp.set_attribute("aletheia.fm.condition_number_kappa",
                         float(kappa_post))
    return {
        "context_size_in": int(ctx_in),
        "context_size_out": int(len(STATE.M_ctx)),
        "H_T_pre": float(H_T_pre),
        "H_T_post": float(H_T_post),
        "H_T_delta": float(H_T_post - H_T_pre),
        "kappa_after": float(kappa_post),
        "n_added": int(len(m)),
    }
