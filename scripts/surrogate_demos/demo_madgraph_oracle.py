r"""
demo_madgraph_oracle.py
=======================

End-to-end demo of the SMEFT surrogate trained on the real MadGraph
oracle (replacing the analytic oracle entirely).

Each oracle call launches MG5_aMC v3.7.1 with SMEFTsim_U35_alphaScheme_UFO
to compute the integrated cross section in a narrow m_ll window. The
SMEFT modification factor mu(c, m) = sigma(c, m) / sigma_SM(m) is
returned to the FM. A small SM-only cache is precomputed on a logarithmic
m grid so per-query cost is one MG run after the cache is filled.

Settings chosen for runtime: with the defaults below, the demo takes
roughly 25-40 minutes wall-clock on a laptop (≈ 60 MG runs at ~20 s each).
Bump N_TRAIN / N_SLICE if you want a tighter fit and have the time.

Run: python scripts/surrogate_demos/demo_madgraph_oracle.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from modules.surrogate import (
    IntentionFM,
    ConformalCalibrator,
    MadGraphSMEFTOracle,
    N_WC,
    WC_NAMES,
)


# ---- config (knobs deliberately conservative for runtime) ----
TRAIN_BOX_C = 0.20
CAL_BOX_C = 0.35
M_RANGE_TEV = (0.4, 2.0)
N_TRAIN = 30
N_CAL = 20
N_SLICE = 20

LAMBDA_GEV = 1000.0
M_WINDOW_TEV = 0.20
NEVENTS = 1000

# Slice for the overlay plot.
SLICE_OP = "clq1"
SLICE_VAL = 0.15  # smaller than the smoke run; stay close to TRAIN_BOX_C


def draw(rng: np.random.Generator, n: int, c_box: float):
    return (
        rng.uniform(-c_box, c_box, (n, N_WC)),
        rng.uniform(*M_RANGE_TEV, n),
    )


def main() -> None:
    print("=" * 72)
    print("Intention FM vs MadGraph oracle — end-to-end demo")
    print("=" * 72)
    print(f"  Lambda = {LAMBDA_GEV/1000:.1f} TeV, m_ll in {M_RANGE_TEV} TeV")
    print(f"  N_TRAIN = {N_TRAIN}, N_CAL = {N_CAL}, N_SLICE = {N_SLICE}")
    print(f"  nevents/MG run = {NEVENTS}, m window = {M_WINDOW_TEV*1000:.0f} GeV")

    oracle = MadGraphSMEFTOracle(
        lambda_gev=LAMBDA_GEV,
        m_window_tev=M_WINDOW_TEV,
        nevents=NEVENTS,
        verbose=True,
    )

    # ---- precompute SM cache on a log-spaced m grid (one MG run each) ----
    sm_grid = np.geomspace(M_RANGE_TEV[0], M_RANGE_TEV[1], 10)
    print(f"\nPrecomputing SM cross sections on {len(sm_grid)}-point grid ...")
    t0 = time.perf_counter()
    oracle.precompute_sm(sm_grid)
    print(f"  done in {time.perf_counter()-t0:.0f} s "
          f"({(time.perf_counter()-t0)/len(sm_grid):.1f} s/pt)")

    rng = np.random.default_rng(7)

    # ---- training data ----
    print(f"\nDrawing {N_TRAIN} oracle training queries ...")
    C_tr, M_tr = draw(rng, N_TRAIN, TRAIN_BOX_C)
    t0 = time.perf_counter()
    Y_tr = oracle(C_tr, M_tr)
    t_train = time.perf_counter() - t0
    print(f"  {N_TRAIN} MG queries in {t_train:.0f} s ({t_train/N_TRAIN:.1f} s/pt)")
    print(f"  observed mu range: [{Y_tr.min():.3f}, {Y_tr.max():.3f}]")

    # ---- fit FM ----
    t0 = time.perf_counter()
    fm = IntentionFM(lam=1e-3).fit(C_tr, M_tr, Y_tr)
    print(f"\nFM fit: {1e3*(time.perf_counter()-t0):.2f} ms")
    print(f"  recovered noise_frac: {fm.noise_frac:.4f}  (MG MC noise at "
          f"nevents={NEVENTS} ~ {1/np.sqrt(NEVENTS):.3f})")

    # ---- conformal cal set (broader probe region) ----
    print(f"\nDrawing {N_CAL} cal-set queries on |c| <= {CAL_BOX_C} ...")
    C_cal, M_cal = draw(rng, N_CAL, CAL_BOX_C)
    t0 = time.perf_counter()
    Y_cal = oracle(C_cal, M_cal)
    print(f"  done in {time.perf_counter()-t0:.0f} s")
    cc = ConformalCalibrator(n_strata=5).fit(fm, C_cal, M_cal, Y_cal)

    # ---- slice for the overlay plot ----
    c_slice = np.zeros(N_WC)
    c_slice[WC_NAMES.index(SLICE_OP)] = SLICE_VAL
    m_slice_tev = np.linspace(M_RANGE_TEV[0], M_RANGE_TEV[1], N_SLICE)
    C_slice = np.tile(c_slice, (N_SLICE, 1))

    print(f"\nMG truth on the {N_SLICE}-point slice at {SLICE_OP} = +{SLICE_VAL:.2f} ...")
    t0 = time.perf_counter()
    mu_truth = oracle.truth(C_slice, m_slice_tev)
    print(f"  done in {time.perf_counter()-t0:.0f} s")

    mu_fm, _ = fm.predict(C_slice, m_slice_tev)
    sigma_cc = cc.coverage_sigma(fm, C_slice, m_slice_tev, coverage=0.683)

    rel_err = np.abs(mu_fm - mu_truth) / np.maximum(np.abs(mu_truth), 1e-6)
    print(f"  truth mu range:   [{mu_truth.min():.3f}, {mu_truth.max():.3f}]")
    print(f"  median |rel err|: {np.median(rel_err):.4f}")
    print(f"  max    |rel err|: {np.max(rel_err):.4f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(m_slice_tev, mu_truth, "ko-", lw=2.0, ms=5,
            label="MadGraph truth (SMEFTsim, CT18NNLO, LO)")
    ax.plot(m_slice_tev, mu_fm, color="C3", lw=1.8,
            label="Intention FM prediction")
    ax.fill_between(
        m_slice_tev, mu_fm - sigma_cc, mu_fm + sigma_cc,
        color="C3", alpha=0.25, label="FM 1σ (conformal)",
    )
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, label="SM (μ = 1)")
    ax.set_xlabel(r"$m_{\ell\ell}$ [TeV]")
    ax.set_ylabel(r"$\mu(c, m) = \sigma(c, m) / \sigma_{\rm SM}(m)$")
    title = (
        f"FM vs MadGraph, Drell-Yan ($\\sqrt{{s}}=13$ TeV, "
        f"$\\Lambda={LAMBDA_GEV/1000:.0f}$ TeV, "
        f"SMEFTsim NP$\\leq$2)\n"
        f"slice at {SLICE_OP} = +{SLICE_VAL:.2f};  "
        f"trained on {N_TRAIN} points, |c|≤{TRAIN_BOX_C};  "
        f"median |rel err| = {np.median(rel_err):.3f}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "demo_madgraph_oracle.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
