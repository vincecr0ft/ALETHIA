r"""Residual-SVD on MadGraph-generated SM events.

Consumes the per-event arrays produced by ``mg_residual_svd_parallel.py``
(saved under ``output_mg_events/probe_sm.npz`` and ``c_NN.npz``), and runs
exactly the same §1.3/§1.4 fingerprint experiment as
``smeft_residual_svd.py`` does on analytic-sampled events.

Mechanism. The probe events are drawn from MG at SM. For each working
point c_k, the per-event log-likelihood-ratio log w_{c_k}(x) is evaluated
*analytically* at the MG SM events. The residual of log w_c against the
in-span projection onto a polynomial ψ over (log m, cos θ*) is built into
a residual matrix R; SVD decomposes R = U Σ V^T and the dominant left
singular vector u_1 should pick up the cos θ* direction the deficient ψ
omits — same as on analytic events. This is the MG-fidelity check on
the analytic Task 1 step 3-4 result.

What it does NOT do: re-evaluate log w_c by re-running MG at each c. That
would be a per-event MG call (prohibitive). The per-event likelihood
ratio is defined by the differential cross section, which the analytic
oracle reproduces to per-cent precision against MG already (the existing
mg_crosscheck_afb gate at 64k events). So evaluating log w_c analytically
on MG events is the right thing.
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
from modules.surrogate.oracle_events import event_log_likelihood_ratio
from modules.surrogate.features import N_WC


EVENTS_DIR = HERE / "output_mg_events"
OUT_DIR = HERE / "output_mg_residual_svd"
OUT_DIR.mkdir(exist_ok=True)

# Mirror of the (c_idx, c) table in mg_residual_svd_parallel.py.
C_POINTS_FOR_R = [
    np.array([0.0, 0.0, 0.3, 0.0]),
    np.array([0.0, 0.0, 0.5, 0.0]),
    np.array([0.0, 0.0, 0.7, 0.0]),
    np.array([0.2, 0.0, 0.4, 0.0]),
    np.array([0.4, 0.0, 0.4, 0.0]),
    np.array([0.0, 0.2, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.4, 0.2]),
    np.array([0.3, 0.0, 0.3, 0.0]),
]


# ---------------------------------------------------------------------------
# Encoders (inlined from smeft_residual_svd.py to keep this self-contained).
# ---------------------------------------------------------------------------
def psi_mass_only(events: np.ndarray) -> np.ndarray:
    log_m = events[:, 0]
    return np.stack([np.ones_like(log_m), log_m, log_m ** 2,
                       log_m ** 3, log_m ** 4], axis=1)


def psi_mass_and_angular(events: np.ndarray) -> np.ndarray:
    log_m = events[:, 0]
    u = events[:, 1]
    u_over = u / (1.0 + u * u)
    return np.stack([
        np.ones_like(log_m), log_m, log_m ** 2, log_m ** 3, log_m ** 4,
        u, u ** 2, u ** 3,
        u * log_m, u * log_m ** 2,
        u ** 2 * log_m, u ** 2 * log_m ** 2,
        u_over, u_over * log_m, u_over * log_m ** 2,
    ], axis=1)


def projector_residual(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    A = psi.T @ psi + 1e-6 * np.eye(psi.shape[1])
    w = np.linalg.solve(A, psi.T @ target)
    return target - psi @ w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("# MG residual-SVD on real MadGraph events")
    t0 = time.perf_counter()

    # Load probe events (SM).
    probe_path = EVENTS_DIR / "probe_sm.npz"
    if not probe_path.exists():
        print(f"  MISSING {probe_path} — run mg_residual_svd_parallel.py first")
        sys.exit(2)
    probe = np.load(probe_path)
    probe_events = probe["events"]
    print(f"  SM probe events: {probe_events.shape}")

    # Sanity: the probe events should live in the analysis window.
    log_m = probe_events[:, 0]
    m_tev = np.exp(log_m)
    print(f"  m_ll range: [{m_tev.min():.3f}, {m_tev.max():.3f}] TeV  "
          f"(median {np.median(m_tev):.3f})")
    print(f"  cos θ* range: [{probe_events[:, 1].min():.3f}, "
          f"{probe_events[:, 1].max():.3f}]")

    # Encoders.
    psi_def = psi_mass_only(probe_events)
    psi_full = psi_mass_and_angular(probe_events)
    print(f"  D_psi_def = {psi_def.shape[1]}  D_psi_full = {psi_full.shape[1]}")

    # Per-c log w via the analytic oracle.
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    R_def_cols = []
    R_full_cols = []
    print(f"\n  Evaluating log w_c at MG SM events for {len(C_POINTS_FOR_R)} c-points")
    for k, c in enumerate(C_POINTS_FOR_R):
        t1 = time.perf_counter()
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        R_def_cols.append(projector_residual(psi_def, log_w))
        R_full_cols.append(projector_residual(psi_full, log_w))
        print(f"    c_{k+1:02d} = {c.tolist()}  ({time.perf_counter() - t1:.1f}s)")
    R_def = np.stack(R_def_cols, axis=1)
    R_full = np.stack(R_full_cols, axis=1)

    # SVD.
    U_def, S_def, Vt_def = np.linalg.svd(R_def, full_matrices=False)
    U_full, S_full, Vt_full = np.linalg.svd(R_full, full_matrices=False)
    print(f"\n  Deficient ψ residual SVD: σ top 5 = {S_def[:5].round(4).tolist()}")
    print(f"    σ_1 = {S_def[0]:.3e}, σ_1/σ_2 = "
          f"{S_def[0] / max(S_def[1], 1e-12):.2f}")
    print(f"  Full ψ residual SVD: σ top 5 = {S_full[:5].round(6).tolist()}")
    print(f"    σ_1 = {S_full[0]:.3e}, σ_1/σ_2 = "
          f"{S_full[0] / max(S_full[1], 1e-12):.2f}")

    # Held-out reduction.
    c_held = np.array([
        [0.0, 0.0, 0.6, 0.0],
        [0.3, 0.0, 0.5, 0.0],
        [0.0, 0.3, 0.3, 0.0],
    ])
    u1 = U_def[:, 0]
    psi_def_aug = np.concatenate([psi_def, u1.reshape(-1, 1)], axis=1)
    rms_before_list = []
    rms_after_list = []
    for c in c_held:
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        r_b = projector_residual(psi_def, log_w)
        r_a = projector_residual(psi_def_aug, log_w)
        rms_before_list.append(r_b)
        rms_after_list.append(r_a)
    rms_before = float(np.sqrt(np.mean(np.stack(rms_before_list) ** 2)))
    rms_after = float(np.sqrt(np.mean(np.stack(rms_after_list) ** 2)))
    reduction = rms_before / max(rms_after, 1e-30)
    print(f"\n  Held-out RMS:  before = {rms_before:.3f}  "
          f"after = {rms_after:.3f}  reduction = {reduction:.2f}×")

    # Correlation of u_1 with cos θ*.
    u1c = u1 - u1.mean()
    ctc = probe_events[:, 1] - probe_events[:, 1].mean()
    rho = float(np.dot(u1c, ctc)
                / (np.linalg.norm(u1c) * np.linalg.norm(ctc) + 1e-30))
    print(f"  Pearson r(u_1, cos θ*) = {rho:.3f}")

    # Gates (same as analytic version after the polish).
    SIGMA_FLOOR = 1e-3
    gates = {
        "deficient_fires": (S_def[0] > SIGMA_FLOOR
                              and S_def[0] / max(S_def[1], 1e-12) > 5.0),
        "deficient_dominates_full": S_def[0] > 5.0 * S_full[0],
        "held_out_reduction": reduction > 5.0,
        "u1_correlates_costheta": abs(rho) > 0.3,
    }
    for k, v in gates.items():
        print(f"  Gate {k}: {'PASS' if v else 'FAIL'}")

    # Compare with the analytic SVD result (if available).
    analytic_summary_path = HERE / "output_smeft_residual_svd" / "summary.json"
    analytic_compare = None
    if analytic_summary_path.exists():
        with open(analytic_summary_path) as f:
            analytic = json.load(f)
        a_def_sigma1 = analytic["studies"]["B"]["sigma_top5_def"][0]
        a_full_sigma1 = analytic["studies"]["B"]["sigma_top5_full"][0]
        a_rho = analytic["studies"]["B"]["pearson_u1_costheta"]
        a_red = analytic["studies"]["B"]["held_out_reduction"]
        print(f"\n## Comparison with analytic SVD:")
        print(f"  σ_1 def:   MG = {S_def[0]:.3f}    analytic = {a_def_sigma1:.3f}")
        print(f"  σ_1 full:  MG = {S_full[0]:.3f}    analytic = {a_full_sigma1:.3f}")
        print(f"  Pearson:   MG = {rho:.3f}          analytic = {a_rho:.3f}")
        print(f"  Reduction: MG = {reduction:.2f}×   analytic = {a_red:.2f}×")
        analytic_compare = {
            "mg_def_sigma1": float(S_def[0]),
            "analytic_def_sigma1": float(a_def_sigma1),
            "mg_full_sigma1": float(S_full[0]),
            "analytic_full_sigma1": float(a_full_sigma1),
            "mg_pearson_u1_costheta": rho,
            "analytic_pearson_u1_costheta": a_rho,
            "mg_held_out_reduction": reduction,
            "analytic_held_out_reduction": a_red,
        }

    np.savez(OUT_DIR / "svd_recovery.npz",
             probe_events=probe_events, R_def=R_def, R_full=R_full,
             U_def=U_def, S_def=S_def, U_full=U_full, S_full=S_full,
             u1=u1)

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.semilogy(range(1, len(S_def) + 1), S_def, "-o", c="C3",
                label="deficient ψ (mass-only)")
    ax.semilogy(range(1, len(S_full) + 1), np.maximum(S_full, 1e-12), "-s", c="C0",
                label="full ψ (mass + cos θ*)")
    ax.axhline(SIGMA_FLOOR, ls="--", c="grey", alpha=0.5, label="σ floor")
    ax.set_xlabel("k"); ax.set_ylabel(r"$\sigma_k$ of residual R")
    ax.set_title("MG residual-SVD spectrum")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    order = np.argsort(probe_events[:, 1])
    ax.plot(probe_events[order, 1], u1[order], ".", ms=2, alpha=0.4, c="C3",
            label="$u_1$ (MG events)")
    ax.set_xlabel(r"$\cos\theta^*_{CS}$")
    ax.set_ylabel("$U[:, 1]$")
    ax.set_title(f"u_1 vs cos θ*: Pearson r = {rho:.3f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.savefig(OUT_DIR / "mg_residual_svd.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_events_probe": int(probe_events.shape[0]),
        "n_working_points_R": len(C_POINTS_FOR_R),
        "sigma_top5_def": S_def[:5].tolist(),
        "sigma_top5_full": S_full[:5].tolist(),
        "sigma1_over_sigma2_def": float(S_def[0] / max(S_def[1], 1e-12)),
        "sigma1_over_sigma2_full": float(S_full[0] / max(S_full[1], 1e-12)),
        "rms_residual_before": rms_before,
        "rms_residual_after": rms_after,
        "held_out_reduction": reduction,
        "pearson_u1_costheta": rho,
        "gates": gates,
        "all_pass": all(gates.values()),
        "analytic_compare": analytic_compare,
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {OUT_DIR}/summary.json + mg_residual_svd.png")
    print(f"# Overall: {'PASS' if summary['all_pass'] else 'FAIL'}  "
          f"(wall = {summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
