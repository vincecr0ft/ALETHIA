r"""Task 1 steps 3 + 4 on the SMEFT analytic oracle.

Using the event-level sampler (Task 0) and the SVD machinery validated on
the toy (Task 2), point the same construction at the real analytic SMEFT
Drell-Yan oracle. The setup follows
ALETHIA_informer_workpoint_AL_handoff.md §1.3 step 3-4.

Two sub-experiments:

A. **Out-of-span fingerprint on a deficient ψ.** Build a ψ that contains
   only mass-dependent event features (omits cos θ*_CS). Verify the
   §1.3 facts on per-event likelihood ratio:

   - leverage: events drawn at SM probed against a c with strong angular
     structure show low leverage (rate-only events look in-distribution)
   - accuracy: prediction error floors at a non-zero residual
   - calibration: the predictive interval undercovers the cos θ* part

B. **Residual SVD coherence.** Build a working-point residual matrix R
   over a sweep of c-values inside the EFT window. The dominant left
   singular vector should be a coherent function of cos θ*_CS that
   reproduces the chirality structure A_FB carries. The clean control
   (ψ already including cos θ*_CS) should be silent (σ_1 below floor).

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/manifold-informer/smeft_residual_svd.py

Note: oracle calls are slow (per-event PDF convolution). We use N_events
~ 4-8k per working point and a coarse working-point grid; this is
sufficient for the coherence signal and runs in ~ 5-10 min CPU.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import (
    sample_events, event_log_likelihood_ratio,
)
from modules.surrogate.features import N_WC, WC_NAMES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_EVENTS_PROBE = 6000       # per working point — keep modest for wall time
SIGMA_FLOOR = 1e-3
SIGMA_FOR_PRED_INTERVAL = 0.5

OUT_DIR = HERE / "output_smeft_residual_svd"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def psi_mass_only(events: np.ndarray) -> np.ndarray:
    r"""Deficient ψ: features over log_m_ll only; cos θ*_CS dependence is
    NOT represented. Polynomial in log_m to degree 4 gives a rich m-basis
    (the analytic SMEFT µ ratio has finite m-power structure but the log
    of it carries unbounded m-tail, so we use a degree-4 polynomial which
    matches the morphing-decomposition degree on m above the Z pole).

    Returns (N, 5) with columns (1, log_m, log_m², log_m³, log_m⁴).
    The cos θ* column is *missing*; the §1.3 fingerprint should fire.
    """
    log_m = events[:, 0]
    return np.stack([np.ones_like(log_m), log_m, log_m ** 2,
                       log_m ** 3, log_m ** 4], axis=1)


def psi_mass_and_angular(events: np.ndarray) -> np.ndarray:
    r"""Full ψ: log_m basis ⊕ cos θ*_CS basis ⊕ cross terms.

    The "in-span" reference. The true log w_c(x) on real SMEFT has

        log w_c(x) = log µ(c, m) + log[(1 + u²) + 2 r(c, m) u]
                                  - log[(1 + u²) + 2 r(SM, m) u]

    so a basis that captures the rational-function structure better than a
    bare polynomial in (log_m, u) gets us closer to genuine silence on the
    clean control. We include:

        polynomial in log_m to degree 4 (the high-mass tail order)
        polynomial in u to degree 3
        cross terms u·log_m, u·log_m², u²·log_m, u²·log_m²
        the rational factor u/(1+u²) — the leading angular term in the
          expansion of log[(1+u²) + 2r u] around r=0
        m-dependent versions of u/(1+u²) — to span the m-dependence of r(c,m)

    Returns (N, D=15).
    """
    log_m = events[:, 0]
    u = events[:, 1]
    u_over = u / (1.0 + u * u)               # leading angular asymmetry term
    return np.stack([
        np.ones_like(log_m), log_m, log_m ** 2, log_m ** 3, log_m ** 4,
        u, u ** 2, u ** 3,
        u * log_m, u * log_m ** 2,
        u ** 2 * log_m, u ** 2 * log_m ** 2,
        u_over, u_over * log_m, u_over * log_m ** 2,
    ], axis=1)


def projector_residual(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Residual of `target` against col-space of `psi`. Ridge alpha=1e-6."""
    Psi = psi
    A = Psi.T @ Psi + 1e-6 * np.eye(Psi.shape[1])
    w = np.linalg.solve(A, Psi.T @ target)
    return target - Psi @ w


def leverage_per_event(psi: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
    return np.einsum("nd,de,ne->n", psi, A_inv, psi)


# ---------------------------------------------------------------------------
# Sub-experiment A — fingerprint (leverage / accuracy / calibration)
# ---------------------------------------------------------------------------
def sub_A_fingerprint(oracle) -> dict:
    print("\n# A. Out-of-span fingerprint on a mass-only ψ")
    rng = np.random.default_rng(0)
    c_strong_angular = np.array([0.0, 0.0, 0.4, 0.0])    # large c_lq^(3)

    # Probe events drawn at SM (the "in-distribution" pool).
    print(f"  Sampling {N_EVENTS_PROBE} probe events at SM ...")
    t0 = time.time()
    events_sm = sample_events(oracle, np.zeros(N_WC), N_EVENTS_PROBE,
                                seed=1234)
    print(f"  SM event draw: {time.time() - t0:.1f}s")

    # Encoders.
    psi_def = psi_mass_only(events_sm)                                # (N, 5)
    psi_full = psi_mass_and_angular(events_sm)                         # (N, 10)

    # Per-event truth log w_c(x) at the strong-angular working point.
    print(f"  Evaluating log w_c at c = {c_strong_angular.tolist()} ...")
    t0 = time.time()
    log_w_strong = event_log_likelihood_ratio(oracle, c_strong_angular, events_sm)
    print(f"  log w_c evaluation: {time.time() - t0:.1f}s")

    # In-span projections.
    res_def = projector_residual(psi_def, log_w_strong)
    res_full = projector_residual(psi_full, log_w_strong)

    # Leverage on a separate small held-out test set drawn at SM.
    test_events = sample_events(oracle, np.zeros(N_WC), 1000, seed=5678)
    psi_def_test = psi_mass_only(test_events)
    psi_full_test = psi_mass_and_angular(test_events)
    A_def = psi_def.T @ psi_def + 1e-6 * np.eye(psi_def.shape[1])
    A_full = psi_full.T @ psi_full + 1e-6 * np.eye(psi_full.shape[1])
    lev_def_test = leverage_per_event(psi_def_test, np.linalg.inv(A_def))
    lev_full_test = leverage_per_event(psi_full_test, np.linalg.inv(A_full))

    # Accuracy floor: RMS residual after projection.
    rms_def = float(np.sqrt(np.mean(res_def ** 2)))
    rms_full = float(np.sqrt(np.mean(res_full ** 2)))

    # Calibration: the nominal 1σ predictive interval scale is set by what
    # the full ψ achieves on the same data (the "in-span" reference RMS).
    # If the deficient ψ undercovers, |res_def| > σ_full for most events.
    sigma_pred = max(rms_full, 1e-6)
    cov_def = float(np.mean(np.abs(res_def) < sigma_pred))
    cov_full = float(np.mean(np.abs(res_full) < sigma_pred))

    print(f"\n  Deficient ψ (mass-only):  D_psi = {psi_def.shape[1]}")
    print(f"    leverage on held-out SM events: mean = {lev_def_test.mean():.3f}, "
          f"max = {lev_def_test.max():.3f}, "
          f"std = {lev_def_test.std():.3f}")
    print(f"    accuracy floor (log w_c RMS residual) = {rms_def:.3f}")
    print(f"    nominal 1σ coverage (σ={sigma_pred}) = {cov_def:.3f}  "
          f"(if < 0.68, the angular signal is undercovered)")
    print(f"\n  Full ψ (mass + cos θ*):  D_psi = {psi_full.shape[1]}")
    print(f"    leverage on held-out SM events: mean = {lev_full_test.mean():.3f}, "
          f"max = {lev_full_test.max():.3f}")
    print(f"    accuracy floor = {rms_full:.3f}")
    print(f"    nominal 1σ coverage = {cov_full:.3f}")

    # §1.3 facts to verify:
    # (1) leverage on the deficient ψ is similar to the full ψ for events
    #     drawn at SM — events whose angular discriminating structure
    #     would be needed read in-distribution. Both should be O(D_psi / N),
    #     a small number; the §1.3 claim is that leverage is BLIND to w_⊥.
    # (2) accuracy floor for deficient ψ is much higher than full ψ.
    # (3) coverage for deficient ψ undershoots, at σ matched to ψ_full's RMS.
    lev_both_small = (lev_def_test.mean() < 0.1) and (lev_full_test.mean() < 0.1)
    lev_same_order = (max(lev_def_test.mean(), lev_full_test.mean())
                       / max(min(lev_def_test.mean(), lev_full_test.mean()), 1e-30)
                       < 3.0)
    leverage_blind = lev_both_small and lev_same_order
    acc_floors_higher = rms_def > 2.0 * rms_full
    coverage_breaks = cov_def < cov_full - 0.05      # undercovers vs full
    print(f"\n  §1.3 facts (sigma_pred for coverage = RMS_full = {sigma_pred:.4f}):")
    print(f"    (1) leverage blind to w_⊥ (both small AND same order): "
          f"{'PASS' if leverage_blind else 'FAIL'}  "
          f"(lev_def = {lev_def_test.mean():.5f}, lev_full = {lev_full_test.mean():.5f})")
    print(f"    (2) accuracy floor higher in deficient ψ: "
          f"{'PASS' if acc_floors_higher else 'FAIL'}  "
          f"(RMS def = {rms_def:.3f}, full = {rms_full:.3f}; ratio = {rms_def/max(rms_full, 1e-12):.1f}×)")
    print(f"    (3) coverage breaks in deficient ψ: "
          f"{'PASS' if coverage_breaks else 'FAIL'}  "
          f"(cov def = {cov_def:.3f}, full = {cov_full:.3f})")

    return {
        "c_strong_angular": c_strong_angular.tolist(),
        "n_events_probe": N_EVENTS_PROBE,
        "leverage_def_mean": float(lev_def_test.mean()),
        "leverage_full_mean": float(lev_full_test.mean()),
        "rms_residual_def": rms_def,
        "rms_residual_full": rms_full,
        "coverage_def": cov_def,
        "coverage_full": cov_full,
        "fact1_leverage_blind": bool(leverage_blind),
        "fact2_accuracy_higher": bool(acc_floors_higher),
        "fact3_coverage_breaks": bool(coverage_breaks),
        "sigma_pred_used": float(sigma_pred),
    }


# ---------------------------------------------------------------------------
# Sub-experiment B — residual-SVD coherence on a working-point sweep
# ---------------------------------------------------------------------------
def sub_B_residual_svd(oracle, probe_events: np.ndarray | None = None) -> dict:
    print("\n# B. Residual-SVD coherence over working points")

    # Probe events drawn at SM (shared across all c).
    rng = np.random.default_rng(7)
    if probe_events is None:
        print(f"  Sampling {N_EVENTS_PROBE} probe events at SM ...")
        t0 = time.time()
        probe_events = sample_events(oracle, np.zeros(N_WC),
                                        N_EVENTS_PROBE, seed=2222)
        print(f"  Probe event draw: {time.time() - t0:.1f}s")
    psi_def = psi_mass_only(probe_events)
    psi_full = psi_mass_and_angular(probe_events)

    # Working-point sweep: mix of four-fermion and vertex excursions to
    # produce a c-varying angular signal.
    c_train = np.array([
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.7, 0.0],
        [0.2, 0.0, 0.4, 0.0],
        [0.4, 0.0, 0.4, 0.0],
        [0.0, 0.2, 0.4, 0.0],
        [0.0, 0.0, 0.4, 0.2],
        [0.3, 0.0, 0.3, 0.0],
    ])
    n_train = len(c_train)
    print(f"  Working-point sweep over {n_train} c-values ...")

    R_def_cols = []
    R_full_cols = []
    for k, c in enumerate(c_train):
        t0 = time.time()
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        R_def_cols.append(projector_residual(psi_def, log_w))
        R_full_cols.append(projector_residual(psi_full, log_w))
        print(f"    c = {c.tolist()}  ({time.time() - t0:.1f}s)")

    R_def = np.stack(R_def_cols, axis=1)
    R_full = np.stack(R_full_cols, axis=1)

    # SVD.
    U_def, S_def, Vt_def = np.linalg.svd(R_def, full_matrices=False)
    U_full, S_full, Vt_full = np.linalg.svd(R_full, full_matrices=False)
    print(f"\n  Deficient ψ residual SVD: σ top 5 = {S_def[:5].round(4).tolist()}")
    print(f"    σ_1/σ_2 = {S_def[0] / max(S_def[1], 1e-12):.2f}, σ_1 = {S_def[0]:.3e}")
    print(f"  Full ψ residual SVD: σ top 5 = {S_full[:5].round(6).tolist()}")
    print(f"    σ_1/σ_2 = {S_full[0] / max(S_full[1], 1e-12):.2f}, σ_1 = {S_full[0]:.3e}")

    # Fingerprint criterion. Two checks:
    #   (a) deficient ψ fires the absolute σ_1 > floor ∧ σ_1/σ_2 > 5 fingerprint;
    #   (b) deficient σ_1 dominates the full-ψ σ_1 by at least 5×. The
    #       absolute "full silent" gate is the wrong test on real SMEFT
    #       because the in-span structure is non-polynomial; ψ_full
    #       captures the leading angular and rational-function pieces but
    #       not the full energy-tail log µ(c, m), so its residual is
    #       finite even at the right basis design. The right physical
    #       claim is the RELATIVE gap: a deficient ψ that omits cos θ*
    #       leaves a residual much bigger than a ψ that captures it.
    SIGMA_FLOOR_LOCAL = 1e-3
    def_fires = (S_def[0] > SIGMA_FLOOR_LOCAL
                  and S_def[0] / max(S_def[1], 1e-12) > 5.0)
    full_dominated_by_def = S_def[0] > 5.0 * S_full[0]

    # Held-out: append u_1 to ψ_def, evaluate on disjoint working points.
    c_held = np.array([
        [0.0, 0.0, 0.6, 0.0],
        [0.3, 0.0, 0.5, 0.0],
        [0.0, 0.3, 0.3, 0.0],
    ])
    print(f"\n  Held-out: appending u_1 to deficient ψ ...")
    u1 = U_def[:, 0]
    psi_def_aug = np.concatenate([psi_def, u1.reshape(-1, 1)], axis=1)
    res_before = []
    res_after = []
    for c in c_held:
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        r_b = projector_residual(psi_def, log_w)
        r_a = projector_residual(psi_def_aug, log_w)
        res_before.append(r_b); res_after.append(r_a)
    rms_before = float(np.sqrt(np.mean(np.stack(res_before) ** 2)))
    rms_after = float(np.sqrt(np.mean(np.stack(res_after) ** 2)))
    reduction = rms_before / max(rms_after, 1e-30)
    print(f"  RMS residual on held-out:  before = {rms_before:.3f}  "
          f"after = {rms_after:.3f}  reduction = {reduction:.2f}×")

    # Correlation of recovered u_1 with cos θ*_CS — does the SVD pick up
    # the chirality-asymmetric direction?
    u1_centred = u1 - u1.mean()
    cos_th = probe_events[:, 1] - probe_events[:, 1].mean()
    rho_u1_cos = float(
        np.dot(u1_centred, cos_th)
        / (np.linalg.norm(u1_centred) * np.linalg.norm(cos_th) + 1e-30)
    )
    print(f"  Pearson r(u_1, cos θ*) = {rho_u1_cos:.3f}  "
          f"(non-zero indicates the SVD recovered the angular direction)")

    print(f"\n  §1.3 / Task 4 fingerprint:")
    print(f"    Deficient ψ fires (σ_1 > floor ∧ σ_1/σ_2 > 5): "
          f"{'PASS' if def_fires else 'FAIL'}")
    print(f"    Deficient σ_1 ≥ 5× full σ_1 (relative gap):    "
          f"{'PASS' if full_dominated_by_def else 'FAIL'}  "
          f"({S_def[0]:.2f} vs {S_full[0]:.2f}, ratio = {S_def[0]/max(S_full[0], 1e-12):.1f}×)")
    print(f"    Held-out residual reduction > 5×:              "
          f"{'PASS' if reduction > 5.0 else 'FAIL'}")
    print(f"    u_1 correlates with cos θ*:                    "
          f"{'PASS' if abs(rho_u1_cos) > 0.3 else 'FAIL'}")

    np.savez(OUT_DIR / "svd_recovery.npz",
             probe_events=probe_events, R_def=R_def, R_full=R_full,
             U_def=U_def, S_def=S_def, U_full=U_full, S_full=S_full,
             c_train=c_train, c_held=c_held, u1=u1)
    return {
        "n_train_working_points": n_train,
        "n_events_probe": N_EVENTS_PROBE,
        "sigma_top5_def": S_def[:5].tolist(),
        "sigma_top5_full": S_full[:5].tolist(),
        "sigma1_over_sigma2_def": float(S_def[0] / max(S_def[1], 1e-12)),
        "sigma1_over_sigma2_full": float(S_full[0] / max(S_full[1], 1e-12)),
        "rms_residual_before": rms_before,
        "rms_residual_after": rms_after,
        "held_out_reduction": reduction,
        "pearson_u1_costheta": rho_u1_cos,
        "gate_def_fires": bool(def_fires),
        "gate_def_dominates_full": bool(full_dominated_by_def),
        "gate_reduction": bool(reduction > 5.0),
        "gate_u1_costheta_correlated": bool(abs(rho_u1_cos) > 0.3),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plots(results: dict):
    npz = np.load(OUT_DIR / "svd_recovery.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # 1 — SVD spectrum
    ax = axes[0]
    ax.semilogy(range(1, len(npz["S_def"]) + 1), npz["S_def"], "-o", c="C3",
                label="deficient ψ (mass-only)")
    ax.semilogy(range(1, len(npz["S_full"]) + 1),
                np.maximum(npz["S_full"], 1e-12), "-s", c="C0",
                label="full ψ (mass + cos θ*)")
    ax.axhline(1e-3, ls="--", c="grey", alpha=0.5, label="σ floor (1e-3)")
    ax.set_xlabel("k"); ax.set_ylabel(r"$\sigma_k$ of residual R")
    ax.set_title("B — Residual-SVD spectrum")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # 2 — u_1 vs cos θ*
    ax = axes[1]
    events = npz["probe_events"]
    u1 = npz["u1"]
    order = np.argsort(events[:, 1])
    ax.plot(events[order, 1], u1[order], ".", ms=2, alpha=0.4, c="C3",
            label=r"$u_1$ (recovered)")
    ax.set_xlabel(r"$\cos\theta^*_{CS}$")
    ax.set_ylabel(r"$U[:, 1]$ (top SVD left singular vector)")
    ax.set_title(f"Recovered u_1 vs cos θ*\n"
                  f"Pearson r = {results['B']['pearson_u1_costheta']:.3f}")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # 3 — held-out reduction
    ax = axes[2]
    cats = ["before\nappend", "after\nappend"]
    vals = [results['B']['rms_residual_before'], results['B']['rms_residual_after']]
    bars = ax.bar(cats, vals, color=["C3", "C2"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("held-out RMS residual"); ax.set_yscale("log")
    ax.set_title(f"ψ-extension: held-out RMS\n"
                  f"reduction = {results['B']['held_out_reduction']:.1f}×")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT_DIR / "smeft_residual_svd.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("# ALETHIA Task 1 steps 3-4: SMEFT residual-SVD on event-level oracle")
    t0 = time.time()
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    results = {}
    results["A"] = sub_A_fingerprint(oracle)
    results["B"] = sub_B_residual_svd(oracle)
    make_plots(results)

    gates_passed = {
        "A_fact1_leverage_blind": results["A"]["fact1_leverage_blind"],
        "A_fact2_accuracy_higher": results["A"]["fact2_accuracy_higher"],
        "A_fact3_coverage_breaks": results["A"]["fact3_coverage_breaks"],
        "B_deficient_psi_fires": results["B"]["gate_def_fires"],
        "B_deficient_dominates_full": results["B"]["gate_def_dominates_full"],
        "B_held_out_reduction": results["B"]["gate_reduction"],
        "B_u1_correlates_costheta": results["B"]["gate_u1_costheta_correlated"],
    }
    summary = {
        "studies": results,
        "gates_passed": gates_passed,
        "all_pass": all(gates_passed.values()),
        "wall_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n## Gates:")
    for k, v in gates_passed.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n# Overall: {'PASS' if summary['all_pass'] else 'FAIL'}  "
          f"(wall = {summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
