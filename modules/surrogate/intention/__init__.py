"""Intention foundation model for ALETHIA.

A closed-form linear-attention surrogate (Garnelo & Czarnecki 2023,
arXiv:2305.10203) with a learned `psi_theta` MLP basis. The forward pass
takes only `(M_ctx, Y_ctx, M_q)` and returns predictive mean and per-query
leverage; Wilson coefficients never enter `psi_theta`.

Submodules:

- ``intention.model``        — ``IntentionFM`` (encoder + closed-form head)
- ``intention.calibration``  — ``IntentionConformal`` (stratified split conformal)
- ``intention.acquisition``  — ``epig_acquire_m`` (closed-form EPIG over m)
- ``intention.drift``        — DAS-CUSUM, BH coverage, kappa drift, aggregator

Headline empirical evidence at
``docs/research/02-foundation-model/intention-vs-deepsets.md`` and the
end-to-end demonstration at ``docs/research/synthesis/full-chain-run.md``.
"""
from .model import IntentionFM, PsiMLP, M_REF
from .calibration import IntentionConformal
from .acquisition import epig_acquire_m, target_set_entropy
from .drift import (DASCUSUMState, das_cusum_update, coverage_bh_test,
                    kappa_drift, aggregate_action)

__all__ = [
    "IntentionFM", "PsiMLP", "M_REF",
    "IntentionConformal",
    "epig_acquire_m", "target_set_entropy",
    "DASCUSUMState", "das_cusum_update", "coverage_bh_test",
    "kappa_drift", "aggregate_action",
]
