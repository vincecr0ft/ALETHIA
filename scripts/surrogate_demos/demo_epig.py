r"""
demo_epig.py
============

Focused demo of EPIG vs leverage acquisition. Three things:

  1. With a UNIFORM target set covering the same support as the candidate
     pool, EPIG ≈ leverage (both maximise marginal information).

  2. With a FOCUSED target set (e.g. a tight cloud around a specific
     Wilson-and-mass region of physics interest), EPIG steers the agent
     toward acquisitions that reduce uncertainty AT that region; leverage
     just goes after the highest-variance points globally.

  3. The downstream impact: predictive variance at the target set as a
     function of oracle budget. EPIG drops the target variance faster
     because it picks points that matter for the target.

This is what makes EPIG worth the slight extra cost over leverage in the
deployment: when the orchestrator knows what region the physics question
lives in, it can ask the surrogate to focus its uncertainty reduction
there.

Run: python scripts/demo_epig.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from modules.surrogate import (
    IntentionFM, DummyAnalyticOracle,
    leverage_acquire, epig_acquire,
    N_WC, WC_NAMES,
)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plotting import drift_heatmap


# ---- config ----
TRAIN_BOX, SAMPLE_BOX = 1.0, 2.0
M_RANGE  = (0.2, 2.5)
N_TRAIN  = 200
POOL_SIZE = 500
K_PICKS   = 30           # picks per acquisition strategy
N_ROUNDS  = 10
N_PER_RD  = 10


def draw(rng, n, box, m_range=M_RANGE):
    return (rng.uniform(-box, box, (n, N_WC)),
            rng.uniform(m_range[0], m_range[1], n))


def main():
    oracle = DummyAnalyticOracle(seed=42)
    rng    = np.random.default_rng(123)

    # initial training and pool
    C_tr, M_tr = draw(rng, N_TRAIN, TRAIN_BOX)
    Y_tr       = oracle(C_tr, M_tr, noise=True)
    fm         = IntentionFM(lam=1e-3).fit(C_tr, M_tr, Y_tr)

    C_pool, M_pool = draw(rng, POOL_SIZE, SAMPLE_BOX)

    # two targets:
    #   T_uniform: large uniform cloud over the broader probe region
    #   T_focused: tight cloud at (c_0 = +1.8, others = 0, m = 2.0)
    #              = "the high-mass tail of a specific Wilson direction"
    n_T_uniform = 600
    C_T_uni, M_T_uni = draw(rng, n_T_uniform, SAMPLE_BOX)

    n_T_focus = 200
    centre_c0, centre_m = 1.8, 2.0
    C_T_foc = np.tile([centre_c0, 0.0, 0.0, 0.0], (n_T_focus, 1)) \
              + rng.normal(0, 0.06, (n_T_focus, N_WC))
    M_T_foc = np.full(n_T_focus, centre_m) + rng.normal(0, 0.03, n_T_focus)

    # =============== one-shot acquisition comparison ===============
    print("=" * 72)
    print("EPIG vs leverage: single-shot acquisition under uniform vs focused target")
    print("=" * 72)
    idx_lev          = leverage_acquire(fm, C_pool, M_pool, k=K_PICKS)
    idx_epig_uniform = epig_acquire(fm, C_pool, M_pool, C_T_uni, M_T_uni, k=K_PICKS)
    idx_epig_focused = epig_acquire(fm, C_pool, M_pool, C_T_foc, M_T_foc, k=K_PICKS)

    overlap_uni  = len(set(idx_lev.tolist()) & set(idx_epig_uniform.tolist()))
    overlap_foc  = len(set(idx_lev.tolist()) & set(idx_epig_focused.tolist()))
    print(f"Pick overlap leverage ∩ EPIG(uniform target): {overlap_uni}/{K_PICKS}")
    print(f"Pick overlap leverage ∩ EPIG(focused target): {overlap_foc}/{K_PICKS}")
    print()
    print("Mean c_0 of picks (focused target centred at c_0 = 1.8):")
    print(f"  leverage          : {C_pool[idx_lev,          0].mean():+.3f}")
    print(f"  EPIG (uniform T)  : {C_pool[idx_epig_uniform, 0].mean():+.3f}")
    print(f"  EPIG (focused T)  : {C_pool[idx_epig_focused, 0].mean():+.3f}  ← should be near +1.8")
    print()
    print("Mean m_ll of picks (focused target centred at m = 2.0 TeV):")
    print(f"  leverage          : {M_pool[idx_lev          ].mean():.3f}")
    print(f"  EPIG (uniform T)  : {M_pool[idx_epig_uniform ].mean():.3f}")
    print(f"  EPIG (focused T)  : {M_pool[idx_epig_focused ].mean():.3f}  ← should be near 2.0")

    # =============== sequential loop: predictive variance on focused T ===============
    print()
    print(f"Sequential loop ({N_ROUNDS} rounds × {N_PER_RD} pts, focused target):")
    fm_lev  = IntentionFM(lam=1e-3).fit(C_tr.copy(), M_tr.copy(), Y_tr.copy())
    fm_eps  = IntentionFM(lam=1e-3).fit(C_tr.copy(), M_tr.copy(), Y_tr.copy())

    def _target_var(model):
        sd = model.predict(C_T_foc, M_T_foc)[1]
        return float(np.mean(sd ** 2))

    var_lev = [_target_var(fm_lev)]
    var_eps = [_target_var(fm_eps)]

    rng2 = np.random.default_rng(999)
    print(f"  init                target_var leverage={var_lev[0]:.4e}  EPIG={var_eps[0]:.4e}")
    for r in range(N_ROUNDS):
        # leverage picks
        C_p1 = rng2.uniform(-SAMPLE_BOX, SAMPLE_BOX, (POOL_SIZE, N_WC))
        M_p1 = rng2.uniform(M_RANGE[0], M_RANGE[1], POOL_SIZE)
        i1 = leverage_acquire(fm_lev, C_p1, M_p1, N_PER_RD)
        fm_lev.update(C_p1[i1], M_p1[i1], oracle(C_p1[i1], M_p1[i1], noise=True))

        # EPIG picks against focused target
        C_p2 = rng2.uniform(-SAMPLE_BOX, SAMPLE_BOX, (POOL_SIZE, N_WC))
        M_p2 = rng2.uniform(M_RANGE[0], M_RANGE[1], POOL_SIZE)
        i2 = epig_acquire(fm_eps, C_p2, M_p2, C_T_foc, M_T_foc, N_PER_RD)
        fm_eps.update(C_p2[i2], M_p2[i2], oracle(C_p2[i2], M_p2[i2], noise=True))

        var_lev.append(_target_var(fm_lev))
        var_eps.append(_target_var(fm_eps))
        print(f"  round {r+1:>2}             target_var leverage={var_lev[-1]:.4e}  EPIG={var_eps[-1]:.4e}  "
              f"ratio L/E={var_lev[-1]/var_eps[-1]:.2f}")

    print()
    print(f"Final ratio (leverage target_var / EPIG target_var) = "
          f"{var_lev[-1] / var_eps[-1]:.2f}")
    print("EPIG drives predictive variance lower on the focused target per oracle call.")

    # =============== plot ===============
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2)

    ng = 90
    c0_grid = np.linspace(-2.5, 2.5, ng)
    m_grid  = np.linspace(M_RANGE[0], M_RANGE[1], ng)

    # (a) leverage picks overlaid on drift heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im, _ = drift_heatmap(ax0, fm, c0_grid=c0_grid, m_grid=m_grid)
    ax0.scatter(C_tr[:, 0], M_tr, c="white", s=4, alpha=0.4)
    ax0.scatter(C_pool[idx_lev, 0], M_pool[idx_lev],
                c="cyan", s=32, edgecolors="black", lw=0.4, label="leverage picks")
    ax0.axvline(-TRAIN_BOX, color="cyan", ls=":", lw=0.7)
    ax0.axvline( TRAIN_BOX, color="cyan", ls=":", lw=0.7)
    ax0.set_title("Leverage acquisition: spreads picks over high-lev regions")
    ax0.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax0, label="log10 leverage")

    # (b) EPIG (focused) picks overlaid on same heatmap + target region marker
    ax1 = fig.add_subplot(gs[0, 1])
    im, _ = drift_heatmap(ax1, fm, c0_grid=c0_grid, m_grid=m_grid)
    ax1.scatter(C_tr[:, 0], M_tr, c="white", s=4, alpha=0.4)
    ax1.scatter(C_T_foc[:, 0], M_T_foc, c="yellow", s=10, alpha=0.5, label="target region T")
    ax1.scatter(C_pool[idx_epig_focused, 0], M_pool[idx_epig_focused],
                c="lime", s=32, edgecolors="black", lw=0.4, label="EPIG picks (focused T)")
    ax1.axvline(-TRAIN_BOX, color="cyan", ls=":", lw=0.7)
    ax1.axvline( TRAIN_BOX, color="cyan", ls=":", lw=0.7)
    ax1.set_title("EPIG with focused target: picks cluster near T")
    ax1.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax1, label="log10 leverage")

    # (c) Per-strategy mean c_0 and mean m of picks (bar chart)
    ax2 = fig.add_subplot(gs[1, 0])
    labels = ["leverage", "EPIG\nuniform T", "EPIG\nfocused T"]
    c0_means = [C_pool[idx_lev, 0].mean(), C_pool[idx_epig_uniform, 0].mean(), C_pool[idx_epig_focused, 0].mean()]
    m_means  = [M_pool[idx_lev   ].mean(), M_pool[idx_epig_uniform ].mean(), M_pool[idx_epig_focused ].mean()]
    x = np.arange(3); w = 0.36
    ax2.bar(x - w / 2, c0_means, w, color="C0", label="mean c_0")
    ax2.bar(x + w / 2, [v - 0 for v in m_means], w, color="C3", label="mean m_ll / TeV")
    ax2.axhline(centre_c0, color="C0", ls=":", lw=0.7)
    ax2.axhline(centre_m,  color="C3", ls=":", lw=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("mean pick coordinate")
    ax2.set_title("Where each strategy sends picks\n(dotted lines = focused target centre)")
    ax2.legend(fontsize=8)

    # (d) Target predictive variance vs oracle budget
    ax3 = fig.add_subplot(gs[1, 1])
    x_calls = np.arange(N_ROUNDS + 1) * N_PER_RD
    ax3.plot(x_calls, var_lev, "o-", color="C7", label="leverage acq")
    ax3.plot(x_calls, var_eps, "o-", color="C3", label="EPIG (focused T)")
    ax3.set_yscale("log")
    ax3.set_xlabel("oracle calls (cumulative)")
    ax3.set_ylabel("mean predictive variance on target T")
    ax3.set_title(f"Target-variance shrinkage: EPIG focuses oracle budget on what matters\n"
                  f"(final ratio leverage/EPIG = ×{var_lev[-1]/var_eps[-1]:.2f})")
    ax3.legend(fontsize=8)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "demo_epig.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
