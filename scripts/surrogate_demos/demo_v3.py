r"""
demo_v3.py
==========

Full demo of the smeft_surrogate package. Reproduces the v3 plot from the
chat: conformal calibration (panel c), multi-seed active vs random
(panel d), drift signal before/after with acquired points (panels a, b).

Replaces ``DummyAnalyticOracle`` with your real oracle to make this a
physics demo. The signature is identical so no other code changes.

Run: python scripts/demo_v3.py
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from modules.surrogate import (
    IntentionFM, ConformalCalibrator, DummyAnalyticOracle,
    leverage_acquire,
    empirical_coverage, stratified_coverage,
    N_WC, WC_NAMES,
)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plotting import drift_heatmap, overlay_picks_by_round, coverage_bars, learning_curves


# ---- config ----
TRAIN_BOX, SAMPLE_BOX = 1.0, 2.0
M_RANGE     = (0.2, 2.5)
N_TRAIN, N_CAL, N_TEST = 200, 200, 2000
N_SEEDS, N_ROUNDS, N_PER_ROUND, POOL_SIZE = 5, 10, 10, 500


def draw(rng, n, box, m_range=M_RANGE):
    return (rng.uniform(-box, box, (n, N_WC)),
            rng.uniform(m_range[0], m_range[1], n))


def main():
    oracle = DummyAnalyticOracle(seed=42)

    # =============== single-seed baseline + conformal ===============
    base_rng = np.random.default_rng(42)
    C_tr, M_tr = draw(base_rng, N_TRAIN, TRAIN_BOX)
    Y_tr       = oracle(C_tr, M_tr, noise=True)
    # IMPORTANT: cal set must cover the broader probe region the model is
    # queried on, not just the training box. See ARCHITECTURE.md.
    C_cal, M_cal = draw(base_rng, N_CAL,  SAMPLE_BOX)
    Y_cal        = oracle(C_cal, M_cal, noise=True)
    C_te, M_te   = draw(base_rng, N_TEST, SAMPLE_BOX)
    Y_te         = oracle(C_te, M_te, noise=True)
    Y_te_clean   = oracle.truth(C_te, M_te)

    t0 = time.perf_counter()
    fm = IntentionFM(lam=1e-3).fit(C_tr, M_tr, Y_tr)
    t_fit = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    cc = ConformalCalibrator(n_strata=5).fit(fm, C_cal, M_cal, Y_cal)
    t_cc = (time.perf_counter() - t0) * 1e3

    print("=" * 72)
    print("smeft_surrogate demo v3: conformal calibration + active vs random")
    print("=" * 72)
    print(f"Fit time:                {t_fit:7.2f} ms")
    print(f"Conformal fit time:      {t_cc:7.2f} ms  (n_strata = {cc.n_strata})")
    print(f"Recovered noise_frac:    {fm.noise_frac:.4f}  (true {oracle.noise_frac})")

    mu_te = fm.predict(C_te, M_te, return_std=False)
    rel = np.abs(mu_te - Y_te_clean) / np.maximum(np.abs(Y_te_clean), 1e-6)
    abs_err = np.abs(mu_te - Y_te_clean)
    print()
    print("Held-out accuracy on the hard |c| <= 2 region:")
    print(f"  median |rel err| = {np.median(rel):.4f}")
    print(f"  IQR    |rel err| = [{np.quantile(rel, 0.25):.4f}, {np.quantile(rel, 0.75):.4f}]")
    print(f"  median |abs err| = {np.median(abs_err):.4f}   (baseline mu ~ 1)")

    print()
    print("Per-stratum coverage on the hard region:")
    print("  stratum  n      <lev>      raw_1s  cc_1s   raw_2s  cc_2s")
    rows_1s = stratified_coverage(fm, cc, C_te, M_te, Y_te, coverage=0.683)
    rows_2s = stratified_coverage(fm, cc, C_te, M_te, Y_te, coverage=0.954)
    for r1, r2 in zip(rows_1s, rows_2s):
        print(f"  {r1['stratum']}        {r1['n']:4d}   {r1['mean_leverage']:8.2f}   "
              f"{r1['raw_coverage']:.3f}    {r1['conformal_coverage']:.3f}   "
              f"{r2['raw_coverage']:.3f}    {r2['conformal_coverage']:.3f}")
    print("  target                              0.683   0.683   0.954   0.954")

    # Save references for plotting
    raw_cov_1s = np.array([r["raw_coverage"]       for r in rows_1s])
    cc_cov_1s  = np.array([r["conformal_coverage"] for r in rows_1s])

    # =============== multi-seed active vs random ===============
    eval_rng = np.random.default_rng(7)
    C_eval, M_eval = draw(eval_rng, 800, SAMPLE_BOX)
    Y_eval         = oracle.truth(C_eval, M_eval)

    err_r = np.zeros((N_SEEDS, N_ROUNDS + 1))
    err_a = np.zeros((N_SEEDS, N_ROUNDS + 1))
    lev_r = np.zeros((N_SEEDS, N_ROUNDS + 1))
    lev_a = np.zeros((N_SEEDS, N_ROUNDS + 1))

    acquired_history_a   = []
    fm_active_first_seed = fm_initial_first_seed = None
    C_tr0 = M_tr0 = None

    def _eval(model):
        mu = model.predict(C_eval, M_eval, return_std=False)
        err = float(np.mean(np.abs((mu - Y_eval) / np.maximum(np.abs(Y_eval), 1e-6))))
        lev = float(model.leverage(C_eval, M_eval).mean())
        return err, lev

    print()
    print(f"Multi-seed loop: {N_SEEDS} seeds × {N_ROUNDS} rounds × {N_PER_ROUND} pts/round")
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(2000 + seed)
        C0, M0 = draw(rng, N_TRAIN, TRAIN_BOX)
        Y0     = oracle(C0, M0, noise=True)
        fmr = IntentionFM(lam=1e-3).fit(C0.copy(), M0.copy(), Y0.copy())
        fma = IntentionFM(lam=1e-3).fit(C0.copy(), M0.copy(), Y0.copy())
        err_r[seed, 0], lev_r[seed, 0] = _eval(fmr)
        err_a[seed, 0], lev_a[seed, 0] = _eval(fma)

        for r in range(N_ROUNDS):
            C_pr = rng.uniform(-SAMPLE_BOX, SAMPLE_BOX, (POOL_SIZE, N_WC))
            M_pr = rng.uniform(M_RANGE[0], M_RANGE[1], POOL_SIZE)
            idx_r = rng.choice(POOL_SIZE, N_PER_ROUND, replace=False)
            fmr.update(C_pr[idx_r], M_pr[idx_r], oracle(C_pr[idx_r], M_pr[idx_r], noise=True))

            C_pa = rng.uniform(-SAMPLE_BOX, SAMPLE_BOX, (POOL_SIZE, N_WC))
            M_pa = rng.uniform(M_RANGE[0], M_RANGE[1], POOL_SIZE)
            idx_a = leverage_acquire(fma, C_pa, M_pa, N_PER_ROUND)
            C_as, M_as = C_pa[idx_a], M_pa[idx_a]
            fma.update(C_as, M_as, oracle(C_as, M_as, noise=True))

            err_r[seed, r + 1], lev_r[seed, r + 1] = _eval(fmr)
            err_a[seed, r + 1], lev_a[seed, r + 1] = _eval(fma)
            if seed == 0:
                acquired_history_a.append((r, C_as.copy(), M_as.copy()))

        if seed == 0:
            fm_active_first_seed  = fma
            fm_initial_first_seed = IntentionFM(lam=1e-3).fit(C0.copy(), M0.copy(), Y0.copy())
            C_tr0, M_tr0 = C0, M0

    print()
    print("                     random                active")
    print("                  err     lev          err     lev")
    print(f"  init           {err_r[:, 0].mean():.4f}  {lev_r[:, 0].mean():.3f}       "
          f"{err_a[:, 0].mean():.4f}  {lev_a[:, 0].mean():.3f}")
    print(f"  after {N_ROUNDS*N_PER_ROUND}   {err_r[:, -1].mean():.4f}  {lev_r[:, -1].mean():.3f}       "
          f"{err_a[:, -1].mean():.4f}  {lev_a[:, -1].mean():.3f}")
    print(f"  active gain on leverage: ×{lev_r[:, -1].mean() / lev_a[:, -1].mean():.1f}")

    # =============== plot ===============
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2)

    ng = 90
    c0_grid = np.linspace(-2.5, 2.5, ng)
    m_grid  = np.linspace(M_RANGE[0], M_RANGE[1], ng)
    lev_before = fm_initial_first_seed.leverage(
        np.column_stack([c0_grid.repeat(ng), np.zeros(ng * ng), np.zeros(ng * ng), np.zeros(ng * ng)]),
        np.tile(m_grid, ng),
    ).reshape(ng, ng).T
    # Recompute via helper for color-scale sharing
    ax0 = fig.add_subplot(gs[0, 0])
    im0, lev_b = drift_heatmap(ax0, fm_initial_first_seed, c0_grid=c0_grid, m_grid=m_grid)
    ax1 = fig.add_subplot(gs[0, 1])
    vmin = np.log10(min(lev_b.min(), fm_active_first_seed.leverage(
        np.zeros((1, N_WC)), np.array([1.0]))[0]) + 1e-6)
    vmax = np.log10(lev_b.max())
    im0.set_clim(vmin, vmax)
    im1, lev_a2 = drift_heatmap(ax1, fm_active_first_seed,
                                c0_grid=c0_grid, m_grid=m_grid, vmin=vmin, vmax=vmax)
    ax0.scatter(C_tr0[:, 0], M_tr0, c="white", s=4, alpha=0.6, label="initial training")
    ax1.scatter(C_tr0[:, 0], M_tr0, c="white", s=4, alpha=0.25)
    cmap_r = overlay_picks_by_round(ax1, acquired_history_a)
    for ax in (ax0, ax1):
        ax.axvline(-TRAIN_BOX, color="cyan", ls=":", lw=0.8)
        ax.axvline( TRAIN_BOX, color="cyan", ls=":", lw=0.8)
    ax0.set_title("Drift signal BEFORE retraining\n(log10 leverage; bright = under-explored)")
    ax1.set_title("Drift signal AFTER 10 active rounds\n(picks coloured by round, early to late)")
    fig.colorbar(im0, ax=ax0, label="log10 leverage")
    fig.colorbar(im1, ax=ax1, label="log10 leverage")
    import matplotlib.cm as mcm
    sm = mcm.ScalarMappable(cmap=cmap_r, norm=Normalize(vmin=0, vmax=N_ROUNDS - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, pad=0.02, shrink=0.6, label="acquisition round")

    ax2 = fig.add_subplot(gs[1, 0])
    coverage_bars(ax2, cc, raw_cov_1s, cc_cov_1s, target=0.683)
    ax2.set_title("Per-stratum coverage: raw vs conformal\n(target line = 0.683 for 1σ-equivalent)")

    ax3 = fig.add_subplot(gs[1, 1])
    x_calls = np.arange(N_ROUNDS + 1) * N_PER_ROUND
    learning_curves(
        ax3, x_calls=x_calls,
        curves={"random": (lev_r, "C7"),
                "active (leverage acq)": (lev_a, "C3")},
        ylabel="mean leverage on hard eval set",
        title=f"Active vs random ({N_SEEDS} seeds, ±1σ band)\n"
              f"Active reduces leverage ×{lev_r[:, -1].mean()/lev_a[:, -1].mean():.0f} more per oracle call",
    )

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "demo_v3.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
