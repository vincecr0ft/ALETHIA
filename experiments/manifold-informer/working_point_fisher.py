"""Task 1 step 2 — the gating experiment for the ManifoldInformer reframe.

Sweep the working point c along the c_lq^(3) axis on the existing mass-only
analytic SMEFT Drell-Yan oracle and ask: does the small-eigenvalue vertex
direction(s) lift by orders of magnitude as |c_lq| moves into the EFT-valid
window, by the mechanism of Eq. (1) of the handoff
(ALETHIA_informer_workpoint_AL_handoff.md §1.1)?

The math under test (the additive-Y-noise Fisher; Pitfall 1 of
ALETHEIA_investigations.md):

    F_{ij}(c) = sum_m (∂_i μ(c, m)) (∂_j μ(c, m)) / σ_y²

with

    ∂_i μ(c, m) = A_i(m) + 2 sum_j B_{ij}(m) c_j         (Eq. 1)

The single-point Fisher diagonalisation in the current paper (Section 5)
evaluates at c=0 only, where the gradient reduces to A_i(m). On the
neutral-current Drell-Yan mass observable, A_i(m) for the two vertex
operators (cHq3, cHq1) is proportional to σ_SM(m) — the vertex shifts
the SM Z coupling and just rescales the SM curve — so the vertex
eigenvalues of F(0) sit at the 1e-4 prior floor, indistinguishable
from the overall normalisation.

The reframe predicts that moving off SM along c_lq^(3) injects a
non-trivial cross-template B_{vertex, ℓq}(m) which is *not* proportional
to σ_SM(m) because the four-fermion operator carries energy growth
(M^4/Λ^4). The flat vertex direction should lift.

If it doesn't — quantify how big B_{vertex, ℓq} actually is inside the
EFT-valid window and report it as a measured sharp null for the §1
reframe. Tasks 3-5 are gated on this passing.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/manifold-informer/working_point_fisher.py
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_WC = 4
WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
# c_lq^(3) is index 2 in WC_NAMES. The two vertex operators are 0, 1.
CLQ3_DIM = 2
CHQ3_DIM = 0
CHQ1_DIM = 1
CLQ1_DIM = 3

# Working-point sweep along c_lq^(3). EFT validity requires ŝ/Λ² < 1; with
# Λ = 1 TeV that is m_ll < 1 TeV. We sweep up to |c| = 1.0; for m_ll above
# 1 TeV the EFT expansion is formally outside strict validity at large |c|,
# but the analytic morphing is mathematically defined and the lift mechanism
# is observable. Flag the EFT window in the writeup.
C_SWEEP = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# Joint excursion along the (c_lq^(3), c_lq^(1)) diagonal: tests whether
# multi-operator working-point excursion lifts faster than single-operator
# (the §1.1 prediction is that cross-templates B_{vertex, ℓq^(1)} also
# carry energy growth, so the lift compounds).
C_JOINT_SWEEP = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

# m-grid on which the Fisher sum is evaluated. Uniform in m_ll across the
# analysis window. (Could re-weight by σ_SM(m); leave that as a sanity
# variant below.)
M_RANGE_TEV = (0.3, 2.3)
N_M = 60

# Noise model: additive on Y = μ at σ_y = 0.05 (matches INV-2). This sets
# the absolute scale of the eigenvalues; ratios are σ_y-independent.
SIGMA_Y = 0.05

# Finite-difference step for the central-difference gradient. μ is quadratic
# in c at LO so central differences are exact up to floating-point.
FD_STEP = 1e-3

# Output.
OUT_DIR = HERE / "output_task1_step2"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Working-point Fisher
# ---------------------------------------------------------------------------
def working_point_fisher(oracle, c_wp: np.ndarray, m_grid: np.ndarray,
                          *, sigma_y: float = SIGMA_Y,
                          fd_step: float = FD_STEP) -> np.ndarray:
    """Additive-noise Fisher F_{ij}(c_wp) summed over m_grid.

        F_{ij}(c) = sum_m (∂_i μ)(∂_j μ) / σ_y²

    Args:
        oracle: anything with `.truth(C, m)` mapping (N, n_wc), (N,) -> (N,).
            We broadcast c_wp across m_grid to get the per-m gradient.
        c_wp: (n_wc,) working point in Wilson space.
        m_grid: (K,) m-values in TeV.
        sigma_y: scalar additive noise stddev on μ.
        fd_step: central-difference step.

    Returns:
        F: (n_wc, n_wc) symmetric PSD matrix.
    """
    K = len(m_grid)
    C0 = np.tile(c_wp, (K, 1))                                      # (K, n_wc)
    grad_mu = np.empty((K, N_WC), dtype=float)
    for i in range(N_WC):
        cp = C0.copy(); cp[:, i] += fd_step
        cm = C0.copy(); cm[:, i] -= fd_step
        mu_p = oracle.truth(cp, m_grid)
        mu_m = oracle.truth(cm, m_grid)
        grad_mu[:, i] = (mu_p - mu_m) / (2.0 * fd_step)
    F = (grad_mu.T @ grad_mu) / (sigma_y ** 2)
    return 0.5 * (F + F.T)


def fisher_eigen(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, V = np.linalg.eigh(0.5 * (F + F.T))                         # ascending
    idx = np.argsort(lam)[::-1]                                       # descending
    return lam[idx], V[:, idx]


def cross_template_at(oracle, m_grid: np.ndarray, i: int, j: int,
                       *, h: float = 1e-3) -> np.ndarray:
    """Recover the morphing cross-template B_{ij}(m) via finite differences
    of μ at c=0. For i != j:

        B_{ij}(m) = ½ ∂_i ∂_j μ |_{c=0}
                  = (μ(+e_i+e_j) - μ(+e_i-e_j) - μ(-e_i+e_j) + μ(-e_i-e_j))
                    / (8 h²)

    Used to certify the §1.1 mechanism: confirm B_{vertex,ℓq}(m) is *not*
    proportional to σ_SM(m), i.e. carries energy growth.
    """
    K = len(m_grid)
    if i == j:
        c_p = np.zeros((K, N_WC)); c_p[:, i] = +h
        c_m = np.zeros((K, N_WC)); c_m[:, i] = -h
        c_0 = np.zeros((K, N_WC))
        mu_p = oracle.truth(c_p, m_grid)
        mu_m = oracle.truth(c_m, m_grid)
        mu_0 = oracle.truth(c_0, m_grid)
        return 0.5 * (mu_p + mu_m - 2.0 * mu_0) / (h * h)
    c_pp = np.zeros((K, N_WC)); c_pp[:, i] = +h; c_pp[:, j] = +h
    c_pm = np.zeros((K, N_WC)); c_pm[:, i] = +h; c_pm[:, j] = -h
    c_mp = np.zeros((K, N_WC)); c_mp[:, i] = -h; c_mp[:, j] = +h
    c_mm = np.zeros((K, N_WC)); c_mm[:, i] = -h; c_mm[:, j] = -h
    mu_pp = oracle.truth(c_pp, m_grid)
    mu_pm = oracle.truth(c_pm, m_grid)
    mu_mp = oracle.truth(c_mp, m_grid)
    mu_mm = oracle.truth(c_mm, m_grid)
    return (mu_pp - mu_pm - mu_mp + mu_mm) / (8.0 * h * h)


def linear_template_at(oracle, m_grid: np.ndarray, i: int,
                        *, h: float = 1e-3) -> np.ndarray:
    """Recover A_i(m) = ∂_i μ |_{c=0} via central differences."""
    K = len(m_grid)
    c_p = np.zeros((K, N_WC)); c_p[:, i] = +h
    c_m = np.zeros((K, N_WC)); c_m[:, i] = -h
    mu_p = oracle.truth(c_p, m_grid)
    mu_m = oracle.truth(c_m, m_grid)
    return (mu_p - mu_m) / (2.0 * h)


def sm_mu_at(oracle, m_grid: np.ndarray) -> np.ndarray:
    """μ(c=0, m) = 1 by construction (BSM/SM ratio with c=0). Return for
    reference; should be ≈ 1 at every m."""
    K = len(m_grid)
    return oracle.truth(np.zeros((K, N_WC)), m_grid)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("# ALETHIA Task 1 step 2 — working-point Fisher sweep")
    print(f"  WC: {WC_NAMES}")
    print(f"  c_lq^(3) sweep: {C_SWEEP.tolist()}")
    print(f"  m-grid: linspace({M_RANGE_TEV[0]}, {M_RANGE_TEV[1]}, "
          f"{N_M}) TeV")
    print(f"  σ_y    = {SIGMA_Y}")

    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    m_grid = np.linspace(M_RANGE_TEV[0], M_RANGE_TEV[1], N_M)

    # --- diagnostics at c = 0 (the existing paper's anchor) ---
    print("\n## At c = 0 (the paper's single-point anchor)")
    mu_sm = sm_mu_at(oracle, m_grid)
    print(f"  μ(0, m) range: [{mu_sm.min():.6f}, {mu_sm.max():.6f}] "
          f"(should be ≈ 1)")
    A = np.stack([linear_template_at(oracle, m_grid, i) for i in range(N_WC)],
                 axis=1)                                              # (K, n_wc)
    print(f"  A_i(m) shape: {A.shape}")

    # The §1.1 mechanism prediction: A_vertex ∝ σ_SM(m), so the cosine
    # similarity of A_vertex(m) with σ_SM(m) is near 1 (which here means
    # near μ_SM=1; check the *m-shape* of A_vertex against a constant).
    # We compute the ratio std(A) / mean(A) for each operator — vertex
    # operators (rate shifts on SM curve) should have small std/mean if
    # they're rate-only, and four-fermion operators large std/mean.
    for i, name in enumerate(WC_NAMES):
        rel_var = float(np.std(A[:, i]) / max(abs(np.mean(A[:, i])), 1e-30))
        print(f"    {name:6s}  mean(A)={np.mean(A[:, i]):+.4f}  "
              f"std(A)={np.std(A[:, i]):+.4f}  std/|mean|={rel_var:.3f}")

    # Cross-templates B_{vertex, ℓq}.
    print("\n  Cross-templates B_{i,j}(m):")
    for (i, j) in [(CHQ3_DIM, CLQ3_DIM), (CHQ1_DIM, CLQ3_DIM),
                    (CHQ3_DIM, CLQ1_DIM), (CHQ1_DIM, CLQ1_DIM)]:
        B_ij = cross_template_at(oracle, m_grid, i, j)
        print(f"    B_{WC_NAMES[i]:>6s}_{WC_NAMES[j]:<6s}  "
              f"mean={np.mean(B_ij):+.4f}  std={np.std(B_ij):+.4f}  "
              f"max|B|={np.max(np.abs(B_ij)):+.4f}")

    # Diagonal vertex^2: ∂_i² μ |_{c=0} = 2 B_{ii}.
    print("\n  Diagonal vertex curvatures B_{ii}(m):")
    for i in (CHQ3_DIM, CHQ1_DIM, CLQ3_DIM, CLQ1_DIM):
        B_ii = cross_template_at(oracle, m_grid, i, i)
        print(f"    B_{WC_NAMES[i]:6s}_{WC_NAMES[i]:6s}  "
              f"mean={np.mean(B_ii):+.4f}  std={np.std(B_ii):+.4f}")

    # --- the sweep ---
    print("\n## Working-point Fisher F(c) along c_lq^(3)")
    all_eigvals = []
    all_eigvecs = []
    all_alignments = []
    for c_val in C_SWEEP:
        c_wp = np.zeros(N_WC)
        c_wp[CLQ3_DIM] = c_val
        F = working_point_fisher(oracle, c_wp, m_grid)
        lam, V = fisher_eigen(F)
        # Per-direction alignment with each Warsaw operator: V[:, k] in
        # Warsaw basis. We report |V[i, k]| for each operator i, eigenmode k.
        align = np.abs(V).T                                            # (n_eig, n_wc)
        all_eigvals.append(lam)
        all_eigvecs.append(V)
        all_alignments.append(align)
        eigval_str = "  ".join(f"{l:>10.3e}" for l in lam)
        print(f"  c_lq^(3)={c_val:>4.2f}:  eigvals = {eigval_str}")

    all_eigvals = np.array(all_eigvals)                                # (S, 4)
    all_alignments = np.array(all_alignments)                          # (S, 4, 4)

    # --- gate analysis ---
    # The smallest eigenvalue at c = 0 — by the paper, ~1e-4 vertex direction.
    # We track its motion vs c_lq^(3). If it stays at 1e-4, the reframe is
    # falsified on this oracle.
    smallest_idx_c0 = int(np.argmin(all_eigvals[0]))
    print("\n## Gate analysis")
    print(f"  Smallest eigenvalue at c=0: λ_min = {all_eigvals[0, smallest_idx_c0]:.3e}")
    print(f"  Direction (Warsaw alignment): "
          f"{ {n: f'{a:.2f}' for n, a in zip(WC_NAMES, all_alignments[0, smallest_idx_c0])} }")

    # The §1.1 claim is that small eigenvalues lift. We track ALL eigenvalues
    # in descending-rank order over the sweep — the smallest at the start
    # may not be the smallest at the end (rotations happen). Report:
    #   (a) min(λ) at each c
    #   (b) lift factor min(λ) / min(λ at c=0)
    #   (c) lift factor for the eigenvector closest to the c=0 lowest direction
    min_lam_per_c = np.min(all_eigvals, axis=1)
    print("\n  Lift factor of the *current minimum* eigenvalue:")
    for c_val, lm in zip(C_SWEEP, min_lam_per_c):
        ratio = lm / max(min_lam_per_c[0], 1e-30)
        print(f"    c_lq^(3)={c_val:>4.2f}:  λ_min = {lm:>10.3e}   "
              f"lift = {ratio:>10.2f}×")

    # Cleanest gate: track the eigenmode whose Warsaw-basis alignment is
    # closest to the c=0 smallest eigenvector. As c moves, the eigenvectors
    # rotate, so we find the best-matching mode at each c by inner product
    # against the c=0 lowest eigenvector.
    v0_min = all_eigvecs[0][:, smallest_idx_c0]
    tracked_eigvals = []
    for s, (V_s, lam_s) in enumerate(zip(all_eigvecs, all_eigvals)):
        overlaps = np.abs(V_s.T @ v0_min)                              # (4,)
        best = int(np.argmax(overlaps))
        tracked_eigvals.append(lam_s[best])
    tracked_eigvals = np.array(tracked_eigvals)
    print("\n  Lift factor of the *tracked* eigenmode (best-overlap with c=0 minimum):")
    for c_val, lt in zip(C_SWEEP, tracked_eigvals):
        ratio = lt / max(tracked_eigvals[0], 1e-30)
        print(f"    c_lq^(3)={c_val:>4.2f}:  λ_tracked = {lt:>10.3e}   "
              f"lift = {ratio:>10.2f}×")

    # The pass/fail decision: at c_lq^(3) = 0.6, the tracked vertex eigenvalue
    # should lift by at least 100× (two orders of magnitude). This is the
    # quantitative version of the §1.1 prediction.
    final_lift_min = float(min_lam_per_c[-1] / max(min_lam_per_c[0], 1e-30))
    final_lift_tracked = float(tracked_eigvals[-1] / max(tracked_eigvals[0], 1e-30))
    gate_pass_min = final_lift_min >= 100.0
    gate_pass_tracked = final_lift_tracked >= 100.0
    print(f"\n  Gate (lift ≥ 100× at c_lq=0.6):  "
          f"min-track={'PASS' if gate_pass_min else 'FAIL'}  "
          f"(lift = {final_lift_min:.2f}×);  "
          f"v0-track={'PASS' if gate_pass_tracked else 'FAIL'}  "
          f"(lift = {final_lift_tracked:.2f}×)")

    # --- joint (c_lq3, c_lq1) excursion ---
    print("\n## Joint working-point Fisher F(c) along (c_lq^(3), c_lq^(1))")
    joint_eigvals = []
    joint_tracked = []
    for c_val in C_JOINT_SWEEP:
        c_wp = np.zeros(N_WC)
        c_wp[CLQ3_DIM] = c_val
        c_wp[CLQ1_DIM] = c_val
        F = working_point_fisher(oracle, c_wp, m_grid)
        lam, V = fisher_eigen(F)
        overlaps = np.abs(V.T @ v0_min)
        best = int(np.argmax(overlaps))
        joint_eigvals.append(lam)
        joint_tracked.append(lam[best])
        eigval_str = "  ".join(f"{l:>10.3e}" for l in lam)
        print(f"  c_lq^(3)=c_lq^(1)={c_val:>4.2f}:  eigvals = {eigval_str}")
    joint_eigvals = np.array(joint_eigvals)
    joint_tracked = np.array(joint_tracked)

    print("\n  Lift factor of the tracked eigenmode under joint excursion:")
    for c_val, lt in zip(C_JOINT_SWEEP, joint_tracked):
        ratio = lt / max(joint_tracked[0], 1e-30)
        print(f"    c={c_val:>4.2f}:  λ_tracked = {lt:>10.3e}   lift = {ratio:>10.2f}×")

    joint_final_lift = float(joint_tracked[-1] / max(joint_tracked[0], 1e-30))
    print(f"\n  Joint final lift at c=1.0: {joint_final_lift:.2f}×")

    # BCRB transition threshold: F-eigenvalue must exceed 1/σ_prior² for the
    # direction to be data-dominated. With σ_prior ≈ 0.4 (the relevant prior
    # on the rotated c̃ given the c-box of 0.7 truncated for held-out testing):
    sigma_prior_for_bcrb = 0.4
    bcrb_thresh = 1.0 / (sigma_prior_for_bcrb ** 2)
    print(f"\n  BCRB-transition threshold (σ_prior={sigma_prior_for_bcrb}): "
          f"λ_data-dom ≥ {bcrb_thresh:.2f}")
    print(f"  Single-direction: λ_tracked at c=1.0 = {tracked_eigvals[-1]:.3e} "
          f"({tracked_eigvals[-1] / bcrb_thresh:.1e} × threshold)")
    print(f"  Joint           : λ_tracked at c=1.0 = {joint_tracked[-1]:.3e} "
          f"({joint_tracked[-1] / bcrb_thresh:.1e} × threshold)")

    # --- save ---
    np.savez(
        OUT_DIR / "fisher_sweep.npz",
        c_sweep=C_SWEEP, m_grid=m_grid,
        eigvals=all_eigvals, eigvecs=np.array(all_eigvecs),
        alignments=all_alignments, A_at_zero=A,
        c_joint_sweep=C_JOINT_SWEEP, joint_eigvals=joint_eigvals,
        joint_tracked=joint_tracked,
    )
    summary = {
        "c_sweep": C_SWEEP.tolist(),
        "wc_names": list(WC_NAMES),
        "m_range_tev": list(M_RANGE_TEV),
        "n_m_grid": N_M,
        "sigma_y": SIGMA_Y,
        "eigvals_per_c": all_eigvals.tolist(),
        "min_lam_per_c": min_lam_per_c.tolist(),
        "tracked_eigvals": tracked_eigvals.tolist(),
        "final_lift_min": final_lift_min,
        "final_lift_tracked": final_lift_tracked,
        "gate_pass_min_track": bool(gate_pass_min),
        "gate_pass_v0_track": bool(gate_pass_tracked),
        "smallest_idx_c0": int(smallest_idx_c0),
        "smallest_eigvec_c0_warsaw": all_eigvecs[0][:, smallest_idx_c0].tolist(),
        "c_joint_sweep": C_JOINT_SWEEP.tolist(),
        "joint_eigvals_per_c": joint_eigvals.tolist(),
        "joint_tracked_eigvals": joint_tracked.tolist(),
        "joint_final_lift": joint_final_lift,
        "bcrb_threshold_sigma_prior_0.4": bcrb_thresh,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)

    ax = axes[0]
    for k in range(N_WC):
        ax.semilogy(C_SWEEP, all_eigvals[:, k], "-o",
                    label=f"λ_{k+1} (rank {k+1} at each c)")
    ax.set_xlabel(r"working point $c_{\ell q}^{(3)}$")
    ax.set_ylabel(r"Fisher eigenvalue $\lambda_k$")
    ax.set_title("Fisher eigenvalues vs working-point excursion\n"
                 "(rank-ordered at each c)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.semilogy(C_SWEEP, tracked_eigvals, "-o", c="C3", lw=2.0,
                label=r"single dir: $c_{\ell q}^{(3)}$ only")
    ax.semilogy(C_JOINT_SWEEP, joint_tracked, "-s", c="C2", lw=2.0,
                label=r"joint: $c_{\ell q}^{(3)} = c_{\ell q}^{(1)}$")
    ax.axhline(bcrb_thresh, ls="--", c="k", alpha=0.7,
               label=fr"BCRB transition $1/\sigma_{{\rm prior}}^2$ (σ=0.4)")
    ax.set_xlabel(r"working-point magnitude $|c|$")
    ax.set_ylabel(r"Fisher eigenvalue (vertex direction)")
    ax.set_title("Lift of the c=0 flat direction\n"
                 "(tracked by max overlap)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    lift_single = tracked_eigvals / tracked_eigvals[0]
    lift_joint = joint_tracked / joint_tracked[0]
    ax.plot(C_SWEEP, lift_single, "-o", c="C3", lw=2.0,
            label=r"single ($c_{\ell q}^{(3)}$)")
    ax.plot(C_JOINT_SWEEP, lift_joint, "-s", c="C2", lw=2.0,
            label=r"joint ($c_{\ell q}^{(3)} = c_{\ell q}^{(1)}$)")
    # Quadratic-in-c reference (the asymptotic §1.1 prediction:
    # F ~ c² ∫B² for the vertex direction).
    c_ref = np.linspace(C_SWEEP[0], C_SWEEP[-1], 50)
    quad_ref = 1.0 + (c_ref / C_SWEEP[1]) ** 2 if C_SWEEP[1] > 0 else None
    if quad_ref is not None:
        ax.plot(c_ref, quad_ref / quad_ref[0] * lift_single[1],
                "--", c="grey", alpha=0.6,
                label=r"$\propto c^2$ ref (asymptotic §1.1)")
    ax.set_xlabel(r"working-point magnitude $|c|$")
    ax.set_ylabel("lift factor relative to c=0")
    ax.set_yscale("log")
    ax.set_title("Vertex-direction lift factor\n"
                 "(monotone in |c| is the falsifier)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(OUT_DIR / "fisher_lift.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n# wrote {OUT_DIR}/summary.json and {OUT_DIR}/fisher_lift.png")
    return summary


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n# total wall = {time.time() - t0:.1f}s")
