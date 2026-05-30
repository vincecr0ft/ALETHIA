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
from .acquisition import (epig_acquire_m, epig_acquire_m_eigen, target_set_entropy,
                          param_epig_d_acquire, param_epig_a_acquire,
                          _load_probe_artifact)
from .drift import (DASCUSUMState, das_cusum_update, coverage_bh_test,
                    kappa_drift, aggregate_action)
from .eigen import (EigenState, eigen_state, eigen_resample_weights,
                    eigenvector_stability, union_subspace)
from .fisher import (empirical_fisher_c, fisher_basis, rotate_c,
                     sample_c_prior_inbox)

__all__ = [
    "IntentionFM", "PsiMLP", "M_REF",
    "IntentionConformal",
    "epig_acquire_m", "epig_acquire_m_eigen", "target_set_entropy",
    "param_epig_d_acquire", "param_epig_a_acquire",
    "DASCUSUMState", "das_cusum_update", "coverage_bh_test",
    "kappa_drift", "aggregate_action",
    "EigenState", "eigen_state", "eigen_resample_weights",
    "eigenvector_stability", "union_subspace",
    "empirical_fisher_c", "fisher_basis", "rotate_c",
    "sample_c_prior_inbox",
]
