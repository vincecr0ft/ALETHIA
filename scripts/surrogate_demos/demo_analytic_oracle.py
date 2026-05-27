r"""
demo_analytic_oracle.py
=======================

End-to-end demo of the SMEFT surrogate against the analytic LO Drell-Yan
oracle (the real physics, replacing :class:`DummyAnalyticOracle`).

Trains :class:`IntentionFM` on ~200 noisy oracle queries, fits stratified
conformal calibration, and overlays the model's prediction (with its
calibrated 1σ band) on the analytic ground truth along a chosen
``m_ll`` slice at a fixed Wilson-coefficient vector.

The analytic SMEFT calculator here plays the role MadGraph would in a
full physics setup: it's the closed-form leading-order ground truth for
the ``pp -> ll`` cross section with dim-6 SMEFT operators. The SMEFT
*ratio* ``mu = sigma_BSM / sigma_SM`` is PDF-independent at LO, so the
analytic toy PDF is sufficient for this demonstration.

Run: python scripts/surrogate_demos/demo_analytic_oracle.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from modules.surrogate import (
    IntentionFM,
    ConformalCalibrator,
    AnalyticSMEFTOracle,
    N_WC,
    WC_NAMES,
)


# ---- config --------------------------------------------------------------
# Training/cal box on c, and m_ll range in TeV. With Lambda = 3 TeV the
# dimensionless EFT expansion parameter c * m^2 / Lambda^2 stays <~ 0.3
# across the box, keeping mu in a regime the polynomial-in-log(m) basis
# fits accurately.
TRAIN_BOX_C = 0.5
CAL_BOX_C = 1.0
M_RANGE_TEV = (0.3, 2.5)

N_TRAIN = 200
N_CAL = 200

# Physics knobs.
SQRT_S_GEV = 13000.0
LAMBDA_GEV = 3000.0  # 3 TeV EFT scale (typical LHC SMEFT reference)
ORACLE_ORDER = "quadratic"
ORACLE_NOISE_FRAC = 0.03

# Slice for the overlay plot.
SLICE_OP = "clq1"
SLICE_VAL = 0.3


def draw(rng: np.random.Generator, n: int, c_box: float):
    return (
        rng.uniform(-c_box, c_box, (n, N_WC)),
        rng.uniform(*M_RANGE_TEV, n),
    )


def main() -> None:
    print("=" * 72)
    print("Intention FM vs analytic SMEFT oracle — end-to-end demo")
    print("=" * 72)

    oracle = AnalyticSMEFTOracle(
        sqrt_s_gev=SQRT_S_GEV,
        lambda_scale_gev=LAMBDA_GEV,
        order=ORACLE_ORDER,
        pdf="CT18NNLO",       # real PDFs via vendor/lhapdf (built by `make lhapdf`).
        noise_frac=ORACLE_NOISE_FRAC,
        seed=0,
    )
    rng = np.random.default_rng(42)

    # ---- training data ----
    C_tr, M_tr = draw(rng, N_TRAIN, TRAIN_BOX_C)
    t0 = time.perf_counter()
    Y_tr = oracle(C_tr, M_tr, noise=True)
    t_oracle = time.perf_counter() - t0
    print(f"\nOracle: {N_TRAIN} noisy queries in {t_oracle:.1f}s "
          f"({1e3 * t_oracle / N_TRAIN:.1f} ms/pt, pdf={oracle.pdf!r})")
    print(f"  mu range observed:  [{Y_tr.min():.3f}, {Y_tr.max():.3f}]")

    # ---- fit FM ----
    t0 = time.perf_counter()
    fm = IntentionFM(lam=1e-3).fit(C_tr, M_tr, Y_tr)
    t_fit = (time.perf_counter() - t0) * 1e3
    print(f"\nIntentionFM fit:        {t_fit:6.2f} ms")
    print(f"  recovered noise_frac: {fm.noise_frac:.4f}  (oracle truth: {ORACLE_NOISE_FRAC})")

    # ---- calibration set on broader probe region ----
    C_cal, M_cal = draw(rng, N_CAL, CAL_BOX_C)
    Y_cal = oracle(C_cal, M_cal, noise=True)
    t0 = time.perf_counter()
    cc = ConformalCalibrator(n_strata=5).fit(fm, C_cal, M_cal, Y_cal)
    t_cc = (time.perf_counter() - t0) * 1e3
    print(f"Conformal fit:          {t_cc:6.2f} ms  ({cc.n_strata} strata)")

    # ---- evaluation along an m_ll slice at a fixed WC vector ----
    # Single operator activated; the FM has to recover the closed-form
    # SMEFT shape `1 + c f(m) + c^2 g(m)` from noisy training points.
    c_slice = np.zeros(N_WC)
    c_slice[WC_NAMES.index(SLICE_OP)] = SLICE_VAL
    m_grid_tev = np.linspace(M_RANGE_TEV[0], M_RANGE_TEV[1], 80)
    C_grid = np.tile(c_slice, (m_grid_tev.size, 1))

    mu_truth = oracle.truth(C_grid, m_grid_tev)
    mu_fm, sigma_raw = fm.predict(C_grid, m_grid_tev)
    sigma_cc = cc.coverage_sigma(fm, C_grid, m_grid_tev, coverage=0.683)

    rel_err = np.abs(mu_fm - mu_truth) / np.maximum(np.abs(mu_truth), 1e-6)
    print(f"\nSlice: c = {dict(zip(WC_NAMES, c_slice))}")
    print(f"  truth mu range:        [{mu_truth.min():.3f}, {mu_truth.max():.3f}]")
    print(f"  median |rel err|:      {np.median(rel_err):.4f}")
    print(f"  max    |rel err|:      {np.max(rel_err):.4f}")

    # ---- training-pool overlay for context ----
    # show training mu values stratified by their location, for the plot.

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(m_grid_tev, mu_truth, "k-", lw=2.2,
            label="analytic SMEFT oracle (truth)")
    ax.plot(m_grid_tev, mu_fm, color="C3", lw=1.6,
            label="Intention FM prediction")
    ax.fill_between(
        m_grid_tev, mu_fm - sigma_cc, mu_fm + sigma_cc,
        color="C3", alpha=0.25, label="FM 1σ (conformal)",
    )
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, label="SM (μ = 1)")
    ax.set_xlabel(r"$m_{\ell\ell}$ [TeV]")
    ax.set_ylabel(r"$\mu(c, m) = \sigma(c, m) / \sigma_{\rm SM}(m)$")
    wc_label = ", ".join(f"{n} = {v:+.2f}" for n, v in zip(WC_NAMES, c_slice) if v != 0)
    title = (
        f"FM vs analytic SMEFT, Drell-Yan ($\\sqrt{{s}}={SQRT_S_GEV/1000:.0f}$ TeV, "
        f"$\\Lambda={LAMBDA_GEV/1000:.0f}$ TeV, order={ORACLE_ORDER})\n"
        f"slice at {wc_label};  trained on {N_TRAIN} points, "
        f"|c|≤{TRAIN_BOX_C};  median |rel err| on slice = {np.median(rel_err):.3f}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "demo_analytic_oracle.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
