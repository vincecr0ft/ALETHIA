"""Analytic-SMEFT oracle variant of data.py.

Identical scenario shapes (c in R^4, M in [0.3, 2.3] TeV, K_ctx, Q_query)
but the ground-truth oracle is `AnalyticSMEFTOracle` (closed-form leading-order
Drell-Yan with PDF convolution) instead of `DummyAnalyticOracle` (polynomial toy).
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/vince/ALETHIA")

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle

# Re-export the shape/sampling utilities; only the oracle factory differs.
from data import (
    sample_c, sample_c_shell, make_scenario, make_dataset,
    scenarios_to_tensors,
    N_WC, M_RANGE, M_REF, ALPHA_RIDGE,
)


def make_oracle(seed: int = 0, noise_frac: float = 0.0,
                pdf: str = "analytic") -> AnalyticSMEFTOracle:
    return AnalyticSMEFTOracle(pdf=pdf, noise_frac=noise_frac, seed=seed)
