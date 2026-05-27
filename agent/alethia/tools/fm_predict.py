"""fm_predict: closed-form Intention forward pass with conformal interval."""
from __future__ import annotations

import numpy as np
from opentelemetry import trace

from alethia.state import STATE

_tracer = trace.get_tracer("alethia.tools")


def fm_predict(m_values: list[float], coverage: float = 0.683) -> dict:
    """Predict the SMEFT modification factor mu(m) at the requested m-values.

    Returns the predictive mean, conformal-calibrated sigma at the requested
    coverage, and per-query leverage in the Intention head's feature space.
    The foundation model's forward pass never sees a Wilson coefficient;
    predictions are conditioned on the current context only.

    Args:
        m_values: m_ll values in TeV. Must be positive.
        coverage: nominal interval coverage (0.683 or 0.954).

    Returns:
        Dict with keys: mu (list of floats), sigma (list of floats),
        leverage (list of floats), n (int), coverage_level (float).
    """
    STATE.ensure_initialised()
    with _tracer.start_as_current_span("tool.surrogate.predict") as sp:
        m = np.asarray(m_values, dtype=float)
        mu = STATE.model.predict_np(STATE.M_ctx, STATE.Y_ctx, m)
        lev = STATE.model.leverage(STATE.M_ctx, m)
        sd = STATE.calibrator.coverage_sigma(
            STATE.model, STATE.M_ctx, STATE.Y_ctx, m, coverage)
        sp.set_attribute("aletheia.fm.n_query", len(m))
        sp.set_attribute("aletheia.fm.leverage_mean", float(lev.mean()))
        sp.set_attribute("aletheia.fm.leverage_max", float(lev.max()))
        sp.set_attribute("aletheia.fm.mu_mean", float(mu.mean()))
        sp.set_attribute("aletheia.fm.sigma_mean", float(sd.mean()))
        sp.set_attribute("aletheia.fm.coverage_level", float(coverage))
        return {
            "mu": [float(v) for v in mu],
            "sigma": [float(v) for v in sd],
            "leverage": [float(v) for v in lev],
            "n": int(len(m)),
            "coverage_level": float(coverage),
            "context_size": int(len(STATE.M_ctx)),
        }
