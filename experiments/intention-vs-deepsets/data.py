"""Scenario sampler for the Intention-vs-DeepSets FM comparison.

A *scenario* is a single SMEFT parameter vector ``c in R^4`` (cHq3, cHq1, clq3,
clq1). For each scenario we sample a context of ``K`` observations
``(M_ctx, Y_ctx)`` and a query batch ``M_query`` with ground truth
``Y_query``. The ground-truth oracle is
``modules.surrogate.ground_truth.DummyAnalyticOracle`` (polynomial toy).

Hard constraint: ``c`` is consumed only here, to produce the labels (``Y_ctx``
and ``Y_query``). It never leaves this module. The FMs operate only on
``(M_ctx, Y_ctx, M_query)``.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/vince/ALETHIA")

import numpy as np
import torch

from modules.surrogate.ground_truth import DummyAnalyticOracle

# 4-operator surrogate basis (matches DummyAnalyticOracle / features.py).
N_WC = 4
M_RANGE = (0.3, 2.3)      # TeV
M_REF = 1.0               # TeV reference for log(m / M_REF) features
ALPHA_RIDGE = 1e-3        # default ridge in closed-form attention


def make_oracle(seed: int = 0, noise_frac: float = 0.0) -> DummyAnalyticOracle:
    """Polynomial toy oracle. Noise-free by default so the comparison is
    architectural rather than denoising-driven."""
    return DummyAnalyticOracle(seed=seed, noise_frac=noise_frac)


def sample_c(rng: np.random.Generator, n: int, c_max: float = 0.7) -> np.ndarray:
    """Uniform sampling of ``c`` in the box ``[-c_max, c_max]^4``."""
    return rng.uniform(-c_max, c_max, size=(n, N_WC))


def sample_c_shell(rng: np.random.Generator, n: int,
                   c_inner: float = 0.7, c_outer: float = 1.0) -> np.ndarray:
    """Rejection-sample c in the shell ``c_inner <= max|c_i| <= c_outer``.

    Equivalently: uniform on ``[-c_outer, c_outer]^4`` rejected to the
    annulus in L-infinity. This is the extrapolation region.
    """
    out = np.empty((n, N_WC))
    filled = 0
    while filled < n:
        batch = rng.uniform(-c_outer, c_outer, size=(8 * n, N_WC))
        keep = np.max(np.abs(batch), axis=1) > c_inner
        batch = batch[keep]
        take = min(n - filled, len(batch))
        out[filled:filled + take] = batch[:take]
        filled += take
    return out


def make_scenario(oracle: DummyAnalyticOracle, c: np.ndarray, rng: np.random.Generator,
                  K_ctx: int = 12, Q_query: int = 32) -> dict:
    """Construct a single scenario's tensors.

    Parameters
    ----------
    oracle : DummyAnalyticOracle
    c      : (N_WC,) parameter vector for THIS scenario.
    rng    : np.random.Generator -- both contexts and queries are sampled
             from it so the same scenario draws different M points each call.
    K_ctx, Q_query : ints.

    Returns
    -------
    dict with keys (all numpy):
        M_ctx     (K,)
        Y_ctx     (K,)
        M_query   (Q,)
        Y_query   (Q,)
        c         (N_WC,)
    """
    M_ctx = rng.uniform(M_RANGE[0], M_RANGE[1], size=K_ctx)
    M_query = rng.uniform(M_RANGE[0], M_RANGE[1], size=Q_query)
    # The oracle expects one c per m; broadcast.
    c_ctx = np.tile(c[None, :], (K_ctx, 1))
    c_query = np.tile(c[None, :], (Q_query, 1))
    Y_ctx = oracle.truth(c_ctx, M_ctx)
    Y_query = oracle.truth(c_query, M_query)
    return {
        "M_ctx": M_ctx.astype(np.float32),
        "Y_ctx": Y_ctx.astype(np.float32),
        "M_query": M_query.astype(np.float32),
        "Y_query": Y_query.astype(np.float32),
        "c": c.astype(np.float32),
    }


def make_dataset(n_scenarios: int,
                 c_sampler,
                 oracle: DummyAnalyticOracle,
                 K_ctx: int = 12, Q_query: int = 32,
                 seed: int = 0) -> list[dict]:
    """Materialise a list of scenarios.

    ``c_sampler`` is a callable ``rng -> (N_WC,)``. The caller controls the
    c-distribution; the dataset structure is identical regardless of inside-
    vs-outside-box.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_scenarios):
        c = c_sampler(rng)
        s = make_scenario(oracle, c, rng, K_ctx=K_ctx, Q_query=Q_query)
        scenarios.append(s)
    return scenarios


def scenarios_to_tensors(scenarios: list[dict]) -> dict[str, torch.Tensor]:
    """Stack scenarios into batched tensors for training.

    Shapes:
      M_ctx   (S, K)
      Y_ctx   (S, K)
      M_query (S, Q)
      Y_query (S, Q)
      c       (S, N_WC)  -- kept for diagnostics ONLY.
    """
    return {
        "M_ctx": torch.from_numpy(np.stack([s["M_ctx"] for s in scenarios])),
        "Y_ctx": torch.from_numpy(np.stack([s["Y_ctx"] for s in scenarios])),
        "M_query": torch.from_numpy(np.stack([s["M_query"] for s in scenarios])),
        "Y_query": torch.from_numpy(np.stack([s["Y_query"] for s in scenarios])),
        "c": torch.from_numpy(np.stack([s["c"] for s in scenarios])),
    }


if __name__ == "__main__":
    oracle = make_oracle()
    rng = np.random.default_rng(0)
    inside = make_dataset(5, lambda r: sample_c(r, 1)[0], oracle, seed=1)
    outside = make_dataset(5, lambda r: sample_c_shell(r, 1)[0], oracle, seed=2)
    print("inside  c[0]  max|c|=", float(np.max(np.abs(inside[0]["c"]))))
    print("outside c[0]  max|c|=", float(np.max(np.abs(outside[0]["c"]))))
    t = scenarios_to_tensors(inside)
    for k, v in t.items():
        print(f"  {k:8s} {tuple(v.shape)} dtype={v.dtype}")
