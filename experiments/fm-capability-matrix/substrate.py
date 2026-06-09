r"""Shared SMEFT Drell-Yan substrate for the FM capability matrix.

The whole point of this experiment (see ALETHEIA_ablative_FM_survey_plan.md):
hold the *data-generating process* fixed and vary the *task*, so the matrix
measures fitness-for-purpose rather than performance on one arbitrary probe.
Every capability cell draws from the same analytic SMEFT Drell-Yan oracle
(`modules.analytic_smeft`), the same 4 Wilson coefficients
``(cHq3, cHq1, clq3, clq1)``, and the same per-event kinematics
``x = (log m_ll / 1 TeV, cos θ*_CS)``. What differs between cells is the task
posed on that substrate and the metric used to score it.

This module owns:
  * canonical config (Wilson box / extrapolation shell, event counts);
  * Wilson-point sampling in-distribution (box) and out-of-distribution (shell);
  * an event-set dataset: per scenario a Wilson point c, two independent event
    views X1, X2 at that c (RS3L-style re-simulation), and per-event exact
    log-likelihood ratios log w_c(x) (the only cheap physical scalar);
  * an SM-only event bank (c = 0) for the anomaly cell;
  * dense μ(c, m) and A_FB(c, m) profiles on an m-grid for the operator cell;
  * disk caching as .npz so every cell reads the *same* data.

A "scenario" is one Wilson working point with its event sets. Cells that need
labels read the scenario's ``c``; cells that are unsupervised ignore it. No
cell ever feeds c into a forward pass — c only shapes data and labels.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle  # noqa: E402
from modules.surrogate.oracle_events import (  # noqa: E402
    event_log_likelihood_ratio,
    _build_m_cdf,
    _sample_m_from_cdf,
    _angular_S_DoverS,
    _sample_costheta,
)
from modules.surrogate.features import N_WC, WC_NAMES  # noqa: E402

OUT = HERE / "output_matrix"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Canonical configuration — shared by every cell.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    n_wc: int = N_WC
    wc_names: tuple = WC_NAMES
    # In-distribution Wilson box (uniform per coordinate), in 1/Lambda^2 units.
    box_half: float = 0.6
    # Out-of-distribution shell: norm in [shell_lo, shell_hi], outside the box
    # corner radius, for the OOD-robustness (JEPA) cell.
    shell_lo: float = 1.1
    shell_hi: float = 1.5
    # Event-set sizes.
    n_events: int = 160          # events per view per scenario
    # Scenario counts.
    n_train: int = 450
    n_val: int = 120
    n_test: int = 120
    n_ood: int = 120             # out-of-distribution (shell) test scenarios
    # SM-only event bank for the anomaly cell (number of independent SM sets).
    n_sm_bank: int = 250
    # m-grid (TeV) for the operator cell's μ / A_FB profiles.
    m_lo: float = 0.3
    m_hi: float = 2.3
    n_m_grid: int = 40
    # Coarser grid used inside the event sampler's inverse-CDF (speed).
    n_cdf_grid: int = 100
    cdf_m_lo: float = 0.3
    cdf_m_hi: float = 2.5
    seed: int = 2026
    oracle_order: str = "quadratic"
    oracle_pdf: str = "analytic"


CONFIG = Config()


def make_oracle(cfg: Config = CONFIG) -> AnalyticSMEFTOracle:
    """Noiseless oracle (μ is exact; sampling provides the stochasticity)."""
    return AnalyticSMEFTOracle(
        order=cfg.oracle_order, pdf=cfg.oracle_pdf, noise_frac=0.0
    )


# --------------------------------------------------------------------------
# Wilson-point sampling.
# --------------------------------------------------------------------------


def sample_box(n: int, rng: np.random.Generator, cfg: Config = CONFIG) -> np.ndarray:
    """Uniform in the in-distribution box [-box_half, box_half]^n_wc."""
    return rng.uniform(-cfg.box_half, cfg.box_half, size=(n, cfg.n_wc))


def sample_shell(n: int, rng: np.random.Generator, cfg: Config = CONFIG) -> np.ndarray:
    """Wilson points on an OOD shell: direction uniform on the sphere, radius
    in [shell_lo, shell_hi]. These lie strictly outside the training box's
    inscribed sphere, so a model that only memorised the box must extrapolate.
    """
    d = rng.standard_normal(size=(n, cfg.n_wc))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.uniform(cfg.shell_lo, cfg.shell_hi, size=(n, 1))
    return d * r


# --------------------------------------------------------------------------
# Per-scenario event generation.
# --------------------------------------------------------------------------

_CDF_GRID = None


def _cdf_grid(cfg: Config = CONFIG) -> np.ndarray:
    global _CDF_GRID
    if _CDF_GRID is None:
        _CDF_GRID = np.linspace(cfg.cdf_m_lo, cfg.cdf_m_hi, cfg.n_cdf_grid)
    return _CDF_GRID


def _draw_view(oracle, c, cdf, grid, n, rng):
    """Draw n events (log m/1TeV, cos θ*) from a prebuilt marginal-m CDF.

    Mirrors oracle_events.sample_events but reuses an already-built CDF so
    two views at the same c don't each rebuild it (the dominant cost).
    """
    m = _sample_m_from_cdf(grid, cdf, n, rng)
    _, dover_s = _angular_S_DoverS(oracle, c, m)
    u = _sample_costheta(dover_s, rng)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = np.log(m / 1.0)
    out[:, 1] = u
    return out


def make_scenarios(
    oracle: AnalyticSMEFTOracle,
    C: np.ndarray,
    rng: np.random.Generator,
    cfg: Config = CONFIG,
    *,
    two_views: bool = True,
    with_logw: bool = True,
    label: str = "",
) -> dict:
    """Generate event sets for a batch of Wilson points C (n, n_wc).

    Returns a dict of arrays:
      c        : (n, n_wc)
      X1       : (n, n_events, 2)  first event view
      X2       : (n, n_events, 2)  second independent view (if two_views)
      logw1    : (n, n_events)     exact per-event log w_c(x) for X1 (if with_logw)
    """
    n = C.shape[0]
    grid = _cdf_grid(cfg)
    X1 = np.empty((n, cfg.n_events, 2), dtype=np.float32)
    X2 = np.empty((n, cfg.n_events, 2), dtype=np.float32) if two_views else None
    logw1 = np.empty((n, cfg.n_events), dtype=np.float32) if with_logw else None
    t0 = time.time()
    for i in range(n):
        c = C[i]
        # Build the marginal-m CDF once per Wilson point; both views draw from it.
        _, cdf = _build_m_cdf(oracle, c, grid, sm_only=False)
        X1[i] = _draw_view(oracle, c, cdf, grid, cfg.n_events, rng)
        if two_views:
            X2[i] = _draw_view(oracle, c, cdf, grid, cfg.n_events, rng)
        if with_logw:
            logw1[i] = event_log_likelihood_ratio(oracle, c, X1[i])
        if label and (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{label}] {i + 1}/{n}  ({dt / (i + 1):.2f}s/scn)", flush=True)
    out = {"c": C.astype(np.float32), "X1": X1}
    if two_views:
        out["X2"] = X2
    if with_logw:
        out["logw1"] = logw1
    return out


def profiles(
    oracle: AnalyticSMEFTOracle, C: np.ndarray, cfg: Config = CONFIG
) -> dict:
    """Dense μ(c, m) and A_FB(c, m) profiles over the m-grid for each c.

    Returns mu (n, n_m_grid) and afb (n, n_m_grid), plus the grid (n_m_grid,).
    Used by the operator cell (forward-map emulation) and as targets elsewhere.
    """
    m = np.linspace(cfg.m_lo, cfg.m_hi, cfg.n_m_grid)
    n = C.shape[0]
    mu = np.empty((n, cfg.n_m_grid), dtype=np.float32)
    afb = np.empty((n, cfg.n_m_grid), dtype=np.float32)
    for i in range(n):
        Ci = np.tile(C[i], (cfg.n_m_grid, 1))
        mu[i] = oracle.truth(Ci, m)
        afb[i] = oracle.truth_afb(Ci, m)
    return {"m_grid": m.astype(np.float32), "mu": mu, "afb": afb}


# --------------------------------------------------------------------------
# Build + cache the full shared dataset.
# --------------------------------------------------------------------------

CACHE = OUT / "substrate_cache.npz"


def build_and_cache(cfg: Config = CONFIG, force: bool = False) -> dict:
    """Generate every split once and cache to disk. Idempotent."""
    if CACHE.exists() and not force:
        print(f"cache exists: {CACHE}")
        return load_cache()

    rng = np.random.default_rng(cfg.seed)
    oracle = make_oracle(cfg)
    t0 = time.time()

    C_train = sample_box(cfg.n_train, rng, cfg)
    C_val = sample_box(cfg.n_val, rng, cfg)
    C_test = sample_box(cfg.n_test, rng, cfg)
    C_ood = sample_shell(cfg.n_ood, rng, cfg)
    C_sm = np.zeros((cfg.n_sm_bank, cfg.n_wc))  # SM-only bank

    print(f"generating train ({cfg.n_train}) ...", flush=True)
    train = make_scenarios(oracle, C_train, rng, cfg, with_logw=False, label="train")
    print(f"generating val ({cfg.n_val}) ...", flush=True)
    val = make_scenarios(oracle, C_val, rng, cfg, with_logw=False, label="val")
    print(f"generating test ({cfg.n_test}) ...", flush=True)
    test = make_scenarios(oracle, C_test, rng, cfg, with_logw=False, label="test")
    print(f"generating ood shell ({cfg.n_ood}) ...", flush=True)
    ood = make_scenarios(oracle, C_ood, rng, cfg, with_logw=False, label="ood")
    print(f"generating SM bank ({cfg.n_sm_bank}) ...", flush=True)
    sm = make_scenarios(oracle, C_sm, rng, cfg, two_views=False,
                        with_logw=False, label="sm")

    print("computing μ / A_FB profiles ...", flush=True)
    prof_train = profiles(oracle, C_train, cfg)
    prof_test = profiles(oracle, C_test, cfg)

    blob = {}
    for split, d in [("train", train), ("val", val), ("test", test),
                     ("ood", ood), ("sm", sm)]:
        for k, v in d.items():
            blob[f"{split}_{k}"] = v
    blob["m_grid"] = prof_train["m_grid"]
    blob["train_mu"] = prof_train["mu"]
    blob["train_afb"] = prof_train["afb"]
    blob["test_mu"] = prof_test["mu"]
    blob["test_afb"] = prof_test["afb"]
    blob["config_json"] = np.array([str(asdict(cfg))])

    np.savez_compressed(CACHE, **blob)
    print(f"cached {CACHE}  ({time.time() - t0:.1f}s total)")
    return load_cache()


def load_cache() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} missing — run substrate.build_and_cache() first."
        )
    z = np.load(CACHE, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to check timing/shapes")
    args = ap.parse_args()

    if args.smoke:
        rng = np.random.default_rng(0)
        orc = make_oracle()
        C = sample_box(4, rng)
        t0 = time.time()
        sc = make_scenarios(orc, C, rng, label="smoke")
        print("scenario keys", list(sc.keys()))
        print("X1", sc["X1"].shape, "X2", sc["X2"].shape, "logw1", sc["logw1"].shape)
        print(f"{(time.time() - t0) / 4:.2f}s/scenario")
        pr = profiles(orc, C)
        print("profiles mu", pr["mu"].shape, "afb", pr["afb"].shape)
    else:
        build_and_cache(force=args.force)
