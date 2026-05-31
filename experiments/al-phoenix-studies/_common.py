"""Shared infra for the Phoenix-instrumented AL studies (Stages A/B/C).

Set up a dedicated Phoenix project ("alethia-al-studies") so traces from these
diagnostics don't mix with the main agent or the legacy full-chain sweep.
Provides:

  - tracer():                handle for span emission, scoped to this project.
  - load_pretrained_model(): an IntentionFM in eval mode (mu_FB pretraining
                             reused from the prior failed sweep — the model is
                             not the variable under test in any of these
                             studies).
  - build_oracle():          mu_FB-observable analytic SMEFT oracle.
  - load_probe():            frozen (W, V, σ_y) for the c̃-projection P = Vᵀ W,
                             with V *recomputed* on the mu_FB observable per
                             postmortem Pitfall 8.
  - build_pools():           the P0/P1/P2 candidate pools used by Stages B & C.

Imports follow the same sys.path dance the full-chain run uses so this can be
launched with `uv run python experiments/al-phoenix-studies/<stage>.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FULL_CHAIN = REPO / "experiments" / "full-chain-run"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FULL_CHAIN))

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle  # noqa: E402
from modules.surrogate.intention import IntentionFM             # noqa: E402

OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Phoenix tracer ----------

_PROJECT = "alethia-al-studies"


def tracer():
    """Phoenix tracer for the AL-studies project.

    Lazy import + idempotent registration. ``PHOENIX_TRACING=0`` returns a
    no-op tracer, matching the convention in experiments/full-chain-run/run.py.
    """
    if os.environ.get("PHOENIX_TRACING", "1").lower() in {"0", "off", "false"}:
        from opentelemetry import trace as _otel_trace
        return _otel_trace.get_tracer(f"{_PROJECT}.noop")
    if not hasattr(tracer, "_provider"):
        from phoenix.otel import register
        # Explicit endpoint default so a missing env var doesn't silently
        # send to the wrong collector.
        os.environ.setdefault(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        tracer._provider = register(
            project_name=_PROJECT,
            auto_instrument=False,
            protocol="http/protobuf",
            batch=False,
            verbose=False,
        )
    return tracer._provider.get_tracer(f"{_PROJECT}.runtime")


# ---------- Model + oracle + probe ----------

# Match the production AL sweep config: bimodal target, mu_FB observable,
# K=12 context, 4 Wilson coefficients.
N_WC = 4
WITHHOLD_DIM = 2           # clq3
WITHHOLD_DIM_2 = 0         # cHq3
TARGET_C_LQ3 = 0.8
TARGET_C_HQ3 = -0.5
M_RANGE = (0.3, 2.3)       # TeV
SEED_M_RANGE = (0.5, 1.0)  # thin seed context
K_CTX = 12
SIGMA_Y = 0.05

# Pretrained checkpoint from the failed AL sweep. Same psi basis as everything
# in §5 of the postmortem so the EIG diagnostics here measure the same physics.
DEFAULT_CKPT = (
    FULL_CHAIN / "output_bimodal_random_mu_afb_stressed_s2027" / "intention_fm.pt"
)


def build_oracle(seed: int = 0) -> AnalyticSMEFTOracle:
    return AnalyticSMEFTOracle(
        sqrt_s_gev=13000.0,
        lambda_scale_gev=1000.0,
        order="quadratic",
        pdf="analytic",
        noise_frac=0.05,
        seed=seed,
    )


def truth_mu_fb(oracle: AnalyticSMEFTOracle, c: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Vector evaluate mu_FB at a single c across an m-array."""
    C = np.tile(c, (len(m), 1))
    return oracle.truth_mu_fb(C, m)


def load_pretrained_model(ckpt: Path | None = None) -> IntentionFM:
    """Load the IntentionFM in eval mode. Raises if the checkpoint is absent
    — pretraining from scratch is intentionally not done here so the studies
    are reproducible against the same psi basis as the postmortem sweep."""
    path = ckpt or DEFAULT_CKPT
    if not path.is_file():
        raise FileNotFoundError(
            f"pretrained IntentionFM not found at {path}. "
            "Either run experiments/full-chain-run/run.py with OBSERVABLE=mu_afb "
            "BIMODAL=1 STRESSED_BUDGET=1 to pretrain, or pass --ckpt to use a "
            "different checkpoint."
        )
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def angular_fisher_V(oracle: AnalyticSMEFTOracle, n_m: int = 60
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Fisher rotation on the mu_FB observable (Pitfall 8 of the audit).

    Returns (V, lam) with V[:, a] the a-th eigenvector ordered by descending
    eigenvalue. lam is the corresponding eigenvalue spectrum — Stage A reports
    it as the data-vs-prior-dominated check called out in postmortem §4.1.
    """
    m = np.linspace(M_RANGE[0], M_RANGE[1], n_m)
    h = 1e-3
    zero = np.zeros((n_m, N_WC))
    dY = np.zeros((n_m, N_WC))
    for i in range(N_WC):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        dY[:, i] = (oracle.truth_mu_fb(cp, m) - oracle.truth_mu_fb(cm, m)) / (2.0 * h)
    F = dY.T @ dY
    lam, V = np.linalg.eigh(F)
    idx = np.argsort(lam)[::-1]
    return V[:, idx], lam[idx]


def load_probe(oracle: AnalyticSMEFTOracle, path: Path | None = None) -> dict[str, Any]:
    """Frozen INV-2 probe, with V *recomputed* on mu_FB."""
    # v2 has W of shape (4, d_psi=16) matching the production IntentionFM
    # basis; v3 is the augmented (4, 32) variant fit on a Fisher-augmented
    # feature space and is incompatible with the plain ψ A⁻¹ ψᵀ projection
    # used here.
    probe_path = path or (FULL_CHAIN / "output" / "probe_W_mass_only_v2.npz")
    if not probe_path.is_file():
        probe_path = FULL_CHAIN / "output" / "probe_W_mass_only.npz"
    npz = np.load(probe_path)
    W = np.asarray(npz["W"], dtype=np.float64)
    sigma_y = float(npz["sigma_y"]) if "sigma_y" in npz.files else 0.05
    V, lam = angular_fisher_V(oracle)
    P = V.T @ W
    return {"W": W, "V": V, "lam_fisher": lam, "sigma_y": sigma_y, "P": P,
            "probe_path": str(probe_path)}


# ---------- Candidate pools ----------

def target_context_mu_fb(rng: np.random.Generator, oracle: AnalyticSMEFTOracle
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(target_c, M_ctx, Y_ctx) for the bimodal target. The seed context is
    the same K=12 thin draw in [0.5, 1.0] used by the failed sweep."""
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = TARGET_C_LQ3
    target_c[WITHHOLD_DIM_2] = TARGET_C_HQ3
    M_ctx = rng.uniform(*SEED_M_RANGE, size=K_CTX)
    Y_ctx = truth_mu_fb(oracle, target_c, M_ctx)
    return target_c, M_ctx, Y_ctx


def build_pool_P0(rng: np.random.Generator, size: int = 30) -> np.ndarray:
    """Match the failed sweep's stressed-budget pool: 30 uniform draws on
    M_RANGE. The postmortem's c̃-space null was produced on exactly this pool."""
    return rng.uniform(*M_RANGE, size=size)


def build_pool_P1(rng: np.random.Generator, size: int = 500) -> np.ndarray:
    """Dense uniform pool spanning the full observable. Tests whether the
    failed sweep's null is a pool-density artifact (i.e. whether 30 random
    points just under-sample an underlying CV(IG) >> 0 distribution)."""
    return rng.uniform(*M_RANGE, size=size)


def build_pool_P2(rng: np.random.Generator, size: int = 500,
                  log_weight: bool = True) -> np.ndarray:
    """Pool over-sampling the high-m tail and the M_RANGE boundaries — the
    a-priori-high-leverage regions of psi(m). Built by drawing in
    log(m) with a triangular emphasis on the tail boundaries."""
    lo, hi = M_RANGE
    if log_weight:
        # log-uniform: tail emphasis comes for free, plus draw extras in
        # the top decade to give a-priori high-leverage candidates a fair shot.
        u = rng.uniform(np.log(lo), np.log(hi), size=size)
        m = np.exp(u)
        # Spike on the top 10% of M_RANGE for the design's eigen-tail.
        n_spike = max(1, size // 5)
        m_spike = rng.uniform(0.9 * hi, hi, size=n_spike)
        m = np.concatenate([m, m_spike])[:size]
        rng.shuffle(m)
        return m
    return rng.uniform(lo, hi, size=size)


# ---------- IG over a pool ----------

def ig_per_candidate(model: IntentionFM, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                      M_pool: np.ndarray) -> np.ndarray:
    """Per-candidate leverage-IG: ig_p = ½ log(1 + ψ_pᵀ A⁻¹ ψ_p).

    This is the design-only EIG of a single rank-one update on the current
    posterior. Y_ctx is ignored (label-independence); kept in the signature
    to make Stage A.1's permutation/noise variants obvious at the call site.
    """
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    psi = model.psi_np(M_pool)
    lev = np.einsum("pd,de,pe->p", psi, A_inv, psi)
    return 0.5 * np.log1p(np.maximum(lev, 0.0))


def sigma_ctilde(model: IntentionFM, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                 P: np.ndarray, sigma_y: float, resolved_dim: int | None = None,
                 ) -> np.ndarray:
    """Σ_c̃ = σ_y² P A⁻¹ Pᵀ over the (optionally restricted) resolved subspace."""
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    P_use = P if resolved_dim is None else P[:resolved_dim]
    return (sigma_y ** 2) * (P_use @ A_inv @ P_use.T)
