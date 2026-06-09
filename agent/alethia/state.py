"""Singleton state shared across ALETHIA agent tools.

ADK tools are stateless functions; cross-tool persistence lives here. The
state holds:

- the trained Intention foundation model (PyTorch),
- the per-context conformal calibrator,
- the current context (M_ctx, Y_ctx) accumulating EPIG-acquired observations,
- the target Wilson scenario c (the "physics question" the user is asking),
- the analytic and MadGraph oracle adapters,
- streaming drift state.

The FM and adapters are initialised lazily on first access so that
``import alethia`` is cheap; the heavy state load (PyTorch checkpoint plus
calibration set) happens at first tool invocation.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_ckpt() -> Path:
    """Intention FM checkpoint: env override > packaged copy > repo training run.

    The packaged copy (``agent/alethia/assets/intention_fm.pt``) is what ships
    in the Cloud Run image; the repo training-run path is used in local dev
    where the experiment regenerates it.
    """
    env = (os.environ.get("ALETHIA_CKPT") or "").strip()
    if env:
        return Path(env)
    packaged = Path(__file__).resolve().parent / "assets" / "intention_fm.pt"
    if packaged.exists():
        return packaged
    return REPO_ROOT / "experiments/full-chain-run/output/intention_fm.pt"


DEFAULT_CKPT = _resolve_ckpt()

# Configuration. The defaults match the engineered-drift scenario from
# the end-to-end demo at docs/research/synthesis/full-chain-run.md.
N_WC = 4
WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
M_RANGE_TEV = (0.3, 2.3)
SEED_M_RANGE = (0.5, 1.0)
N_SEED_CTX = 8
N_CAL_POINTS = 400
DEFAULT_TARGET_C = np.array([0.0, 0.0, 0.8, 0.0])  # clq3 = 0.8


@dataclass
class AletheiaState:
    """Singleton agent state."""
    target_c: np.ndarray = field(
        default_factory=lambda: DEFAULT_TARGET_C.copy())
    M_ctx: Optional[np.ndarray] = None
    Y_ctx: Optional[np.ndarray] = None
    M_cal: Optional[np.ndarray] = None
    Y_cal: Optional[np.ndarray] = None
    model: object = None
    calibrator: object = None
    analytic_oracle: object = None
    madgraph_oracle: object = None
    cusum_state: object = None
    coverage_counts_68: np.ndarray = field(
        default_factory=lambda: np.zeros((5, 2), dtype=int))
    history_flags: list = field(default_factory=list)
    oracle_calls: int = 0
    _initialised: bool = False

    def ensure_initialised(self) -> None:
        if self._initialised:
            return
        # Imports kept inside to keep `import state` fast.
        from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
        from modules.surrogate.intention import (
            IntentionFM, IntentionConformal, DASCUSUMState)

        rng = np.random.default_rng(2026)

        self.analytic_oracle = AnalyticSMEFTOracle(
            pdf="analytic", noise_frac=0.0)

        # Load Intention FM from the chain-run checkpoint.
        self.model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
        if DEFAULT_CKPT.exists():
            state = torch.load(DEFAULT_CKPT, map_location="cpu",
                               weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()

        # Seed context for the current target scenario.
        self.M_ctx = rng.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
        self.Y_ctx = self._oracle_truth(self.M_ctx, fidelity="T1")

        # Calibration set across the broader probe region.
        self.M_cal = rng.uniform(*M_RANGE_TEV, size=N_CAL_POINTS)
        self.Y_cal = self._oracle_truth(self.M_cal, fidelity="T1")
        self.calibrator = IntentionConformal(n_strata=5, noise_frac=0.05)
        self.calibrator.fit(self.model, self.M_ctx, self.Y_ctx,
                            self.M_cal, self.Y_cal)

        # Drift streaming state.
        self.cusum_state = DASCUSUMState(buf=deque(maxlen=30))
        self.coverage_counts_68 = np.zeros((5, 2), dtype=int)
        self.history_flags = []
        self.oracle_calls = 0
        self._initialised = True

    def _oracle_truth(self, m: np.ndarray, fidelity: str = "T1") -> np.ndarray:
        """Evaluate the oracle at the current target c, broadcast over m."""
        c = self.target_c
        C = np.tile(c, (len(m), 1))
        if fidelity == "T1":
            return self.analytic_oracle.truth(C, m)
        elif fidelity == "T2":
            if self.madgraph_oracle is None:
                from modules.surrogate.oracle_madgraph import (
                    MadGraphSMEFTOracle)
                self.madgraph_oracle = MadGraphSMEFTOracle(nevents=500)
            return self.madgraph_oracle.truth(C, m)
        else:
            raise ValueError(f"Unknown fidelity tier: {fidelity}")

    def set_target_c(self, c: np.ndarray) -> None:
        """Reset to a new target scenario. Clears context and recalibrates."""
        self.target_c = np.asarray(c, dtype=float).reshape(N_WC)
        # Re-seed context for the new scenario.
        rng = np.random.default_rng(2026)
        self.M_ctx = rng.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
        self.Y_ctx = self._oracle_truth(self.M_ctx, fidelity="T1")
        self.Y_cal = self._oracle_truth(self.M_cal, fidelity="T1")
        self.calibrator.fit(self.model, self.M_ctx, self.Y_ctx,
                            self.M_cal, self.Y_cal)
        self.history_flags.clear()
        self.coverage_counts_68 = np.zeros((5, 2), dtype=int)


# Module-level singleton.
STATE = AletheiaState()
