"""
modules.surrogate
=================

Intention foundation model for the SMEFT modification factor mu(c, m), with
leverage-stratified conformal calibration and three acquisition strategies
(random, leverage, EPIG).

See ``docs/surrogate/ARCHITECTURE.md`` for the math,
``docs/surrogate/INTEGRATION.md`` for wiring this into Phoenix + Gemini, and
``docs/surrogate/TOOLS.md`` for the orchestrator-facing tool signatures.
"""
from .features import (
    phi_c, phi_x, phi_joint,
    N_WC, WC_NAMES, K_X, M_REF, PAIRS,
    D_C, D_X, D_JOINT,
)
from .ground_truth import Oracle, DummyAnalyticOracle
from .oracle_smeft import AnalyticSMEFTOracle, WC_NAME_MAP
from .oracle_madgraph import MadGraphSMEFTOracle, LHA_CODES
from .model import IntentionFM
from .calibration import ConformalCalibrator
from .acquisition import random_acquire, leverage_acquire, epig_acquire
from .evaluation import (
    empirical_coverage,
    decile_calibration,
    stratified_coverage,
)

__version__ = "0.1.0"

__all__ = [
    "phi_c", "phi_x", "phi_joint",
    "N_WC", "WC_NAMES", "K_X", "M_REF", "PAIRS",
    "D_C", "D_X", "D_JOINT",
    "Oracle", "DummyAnalyticOracle",
    "AnalyticSMEFTOracle", "WC_NAME_MAP",
    "MadGraphSMEFTOracle", "LHA_CODES",
    "IntentionFM",
    "ConformalCalibrator",
    "random_acquire", "leverage_acquire", "epig_acquire",
    "empirical_coverage", "decile_calibration", "stratified_coverage",
    "__version__",
]
