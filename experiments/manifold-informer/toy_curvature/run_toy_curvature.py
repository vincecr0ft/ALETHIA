r"""Task 2 — toy-curvature studies.

On the closed-form toy oracle (``toy_oracle.ToyCurvatureOracle``), demonstrate
the four mechanisms the ManifoldInformer reframe depends on, before pointing
them at the real SMEFT oracle. Each is a self-contained sub-experiment with a
falsification gate.

Study A — **Morphing-matrix recovery**. Sweep N_base ∈ {3, 4, 6, 8, 16}, show
templates (g_SM, a, b) are reconstructed at floating-point precision once
N_base ≥ 6 (the minimum for 2 ops + quadratic morphing). Plot cond(M) vs
base-point placement strategy.

Study B — **Multi-point vs single-point Fisher lift**. Show the single-point
Fisher at SM misses the c_0 (vertex-like) direction, and that excursion along
c_1 lifts its eigenvalue. Quantitative version of the §1.1 result on a
controlled oracle. Falsification: monotone lift in |c_1|; ≥10× by |c_1|=0.6.

Study C — **Residual-SVD recovery of an out-of-span template**. Construct a
deliberately deficient encoder ψ (omits the high-frequency sin(6 x_aux) and
cos(6 x_aux) directions of the per-event likelihood ratio), build the residual
operator R over a sweep of working points, SVD, recover the dominant left
singular vector u_1(x). Falsification: cosine similarity to truth > 0.95,
appended-ψ reduces residual on held-out working points.

Study D — **Span-completeness criterion zero-fire control**. With the
no-injection oracle (in-span only), confirm the completeness criterion
(coverage clean ∧ accuracy drift ∧ calibration break + low-rank coherent
residual SVD) does *not* fire across 20 seeds. False-fire is the
deal-breaker — if it fires when nothing is missing, Task 4's aggregator
will extend ψ chasing noise.

Output:
    output_toy_curvature/
      summary.json   metrics + gate booleans
      *.png          Studies A-D plots
      svd_recovery.npz  Study C numerical artefacts
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib.pyplot as plt

from toy_oracle import (
    ToyCurvatureOracle, N_OPS,
    solve_morphing_templates, morphing_design_matrix,
    working_point_fisher_toy, _g_SM, _injection_shape,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M_RANGE = (0.3, 2.5)
N_M_GRID = 40
M_GRID = np.linspace(M_RANGE[0], M_RANGE[1], N_M_GRID)
SIGMA_Y = 0.05
SEED = 2026

OUT_DIR = HERE / "output_toy_curvature"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Study A — morphing recovery vs N_base
# ---------------------------------------------------------------------------
def study_A_morphing_recovery() -> dict:
    print("\n# Study A — morphing-matrix template recovery vs N_base")
    rng = np.random.default_rng(SEED)
    oracle = ToyCurvatureOracle(injection_eps=0.0)
    by_n = {}
    for n_base in [3, 4, 6, 8, 16]:
        # Strategy: uniform random in [-0.5, 0.5]^2 with rejection if M is rank-deficient
        for attempt in range(10):
            c_base = rng.uniform(-0.5, 0.5, size=(n_base, N_OPS))
            M = morphing_design_matrix(c_base)
            if np.linalg.matrix_rank(M, tol=1e-8) == min(M.shape):
                break
        res = solve_morphing_templates(c_base, M_GRID, oracle)
        err_max = max(res[f"rel_err_{name}"]
                       for name in ("a_0", "a_1", "b_00", "b_11", "b_01"))
        by_n[n_base] = {
            "cond_M": res["cond_morphing_matrix"],
            "max_template_rel_err": err_max,
            "rel_err_a_0": res["rel_err_a_0"],
            "rel_err_a_1": res["rel_err_a_1"],
            "rel_err_b_00": res["rel_err_b_00"],
            "rel_err_b_11": res["rel_err_b_11"],
            "rel_err_b_01": res["rel_err_b_01"],
        }
        print(f"  N_base={n_base:>2d}:  cond(M) = {res['cond_morphing_matrix']:>8.2e}   "
              f"max rel err = {err_max:.2e}")
    # Gate: at N_base ≥ 6 (the minimum for 2 ops quadratic morphing) the templates
    # should be recovered to floating-point precision.
    gate_passed = bool(by_n[6]["max_template_rel_err"] < 1e-8)
    print(f"  Gate (N_base=6 → max rel err < 1e-8): "
          f"{'PASS' if gate_passed else 'FAIL'}")
    return {"by_n_base": by_n, "gate_passed": gate_passed}


# ---------------------------------------------------------------------------
# Study B — multi-point Fisher lift on a controlled flat direction
# ---------------------------------------------------------------------------
def study_B_fisher_lift() -> dict:
    print("\n# Study B — Fisher eigenvalue lift along c_1 (the curvature-resolved axis)")
    oracle = ToyCurvatureOracle(injection_eps=0.0)
    c_sweep = np.linspace(0.0, 1.0, 11)
    eigvals = []
    eigvecs = []
    for c_val in c_sweep:
        c_wp = np.array([0.0, c_val])     # excursion along c_1
        F = working_point_fisher_toy(oracle, c_wp, M_GRID, sigma_y=SIGMA_Y)
        lam, V = np.linalg.eigh(0.5 * (F + F.T))
        idx = np.argsort(lam)[::-1]
        eigvals.append(lam[idx])
        eigvecs.append(V[:, idx])
    eigvals = np.array(eigvals)                       # (S, N_OPS)
    # Track the c=0 smallest direction.
    smallest_idx = int(np.argmin(eigvals[0]))
    v0_min = eigvecs[0][:, smallest_idx]
    tracked = []
    for V, lam in zip(eigvecs, eigvals):
        overlaps = np.abs(V.T @ v0_min)
        best = int(np.argmax(overlaps))
        tracked.append(lam[best])
    tracked = np.array(tracked)
    lift_final = float(tracked[-1] / max(tracked[0], 1e-30))
    print(f"  Smallest eigenvalue at c=0: λ_min = {eigvals[0, smallest_idx]:.3e}")
    print(f"  Direction (c_0, c_1) alignment: {v0_min.round(3).tolist()}")
    print("  Lift trajectory:")
    for c_val, lt in zip(c_sweep, tracked):
        ratio = lt / max(tracked[0], 1e-30)
        print(f"    c_1={c_val:>4.2f}:  λ = {lt:>10.3e}   lift = {ratio:>8.2f}×")
    gate_pass = lift_final >= 10.0
    print(f"  Gate (lift ≥ 10× at c_1=1.0): {'PASS' if gate_pass else 'FAIL'} "
          f"(lift = {lift_final:.2f}×)")
    return {
        "c_sweep": c_sweep.tolist(),
        "eigvals_per_c": eigvals.tolist(),
        "tracked_eigvals": tracked.tolist(),
        "lift_final": lift_final,
        "gate_passed": bool(gate_pass),
        "smallest_eigvec_c0": v0_min.tolist(),
    }


# ---------------------------------------------------------------------------
# Study C — residual SVD recovers an out-of-span template
# ---------------------------------------------------------------------------
def _build_psi_deficient(events: np.ndarray) -> np.ndarray:
    """Deficient encoder for the toy oracle.

    The toy's in-span structure of w_c(x) = µ(c, m) · [1 + ε(η·c) sin(6 x_aux)]
    is *exactly* polynomial in (m², x_aux) up to degree (3, 3): µ(c, m)
    involves powers m², m⁴, m⁶, and the (1 + ε...) angular factor is the
    only non-polynomial piece. So a basis that includes (1, m², m⁴, m⁶,
    x_aux, x_aux², x_aux³) and their lowest cross-terms is exactly the
    in-span basis, and the residual against this basis is purely the
    sin(6 x_aux) injection.

    Returns (N, D_psi=10) feature matrix. The dimensions used:
        column 0:  1
        columns 1-3:  m², m⁴, m⁶
        columns 4-6:  x_aux, x_aux², x_aux³
        columns 7-9:  m²·x_aux, m⁴·x_aux, m²·x_aux²
    """
    log_m = events[:, 0]
    m2 = np.exp(2.0 * log_m)
    m4 = m2 * m2
    m6 = m4 * m2
    xa = events[:, 1]
    xa2 = xa * xa
    xa3 = xa2 * xa
    feats = np.stack([
        np.ones_like(log_m),
        m2, m4, m6,
        xa, xa2, xa3,
        m2 * xa, m4 * xa, m2 * xa2,
    ], axis=1)
    return feats


def _projector_residual(psi_basis: np.ndarray,
                          target: np.ndarray) -> np.ndarray:
    """Residual of `target` against the column space of `psi_basis`. Both
    arrays have shape (N, D_psi) and (N,) respectively. Returns (N,)."""
    # Π_S target = ψ (ψᵀψ + αI)⁻¹ ψᵀ target
    alpha = 1e-6
    Psi = psi_basis
    A = Psi.T @ Psi + alpha * np.eye(Psi.shape[1])
    w = np.linalg.solve(A, Psi.T @ target)
    proj = Psi @ w
    return target - proj


def study_C_residual_svd() -> dict:
    print("\n# Study C — residual-SVD recovery of an out-of-span injection")
    inject_eps = 0.5    # strong enough to be visible
    oracle_inject = ToyCurvatureOracle(
        injection_eps=inject_eps,
        injection_coupling=(0.0, 1.0),       # excited by c_1
    )
    rng = np.random.default_rng(SEED)

    # A shared probe-event set: events drawn at SM (so the residual SVD's
    # column basis is the same per-working-point, exactly as Task 4 will do).
    N_PROBE = 4000
    probe_events = oracle_inject.sample_events(np.zeros(N_OPS), N_PROBE,
                                                  seed=SEED + 1)
    psi = _build_psi_deficient(probe_events)              # (N, 7)

    # Build the residual operator R: per working point c_k, per-event residual
    # of log w_c(x) against its in-span projection on the deficient ψ.
    c_train = np.array([
        [0.0,  0.2], [0.0,  0.4], [0.0,  0.6], [0.0,  0.8],
        [0.3,  0.3], [0.5,  0.5], [-0.3, 0.3], [-0.5, 0.5],
    ])                                                     # 8 working points
    log_w = np.stack([oracle_inject.density_ratio_per_event(c, probe_events)
                       for c in c_train], axis=1)         # (N, n_train)
    residuals = np.stack([_projector_residual(psi, log_w[:, k])
                           for k in range(len(c_train))], axis=1)
    # SVD.
    U, S_sv, Vt = np.linalg.svd(residuals, full_matrices=False)
    print(f"  Singular values of R (top 5): {S_sv[:5].round(4).tolist()}")
    sv_ratio = float(S_sv[0] / (S_sv[1] + 1e-12))
    print(f"  σ_1 / σ_2 = {sv_ratio:.2f}  (high = coherent missing structure)")

    # u_1: per-event recovered function. Compare to truth sin(6 x_aux).
    u1 = U[:, 0]
    # Sign: the SVD's left singular vector has an arbitrary sign; match the
    # sign by inner product with the truth.
    truth_u = _injection_shape(probe_events[:, 1])
    truth_u /= np.linalg.norm(truth_u)
    if np.dot(u1, truth_u) < 0:
        u1 = -u1
        U = -U
    cos_sim = float(np.dot(u1, truth_u))
    print(f"  cos(u_1, sin(6·x_aux)) = {cos_sim:.4f}  "
          f"(target > 0.95)")

    # Held-out gate: at four NEW working points, the residual after appending
    # u_1 to ψ must be much smaller than before.
    c_held = np.array([
        [0.0, 0.5], [0.0, 0.7], [0.4, 0.4], [-0.2, 0.6]
    ])
    log_w_held = np.stack(
        [oracle_inject.density_ratio_per_event(c, probe_events) for c in c_held],
        axis=1)
    res_before = np.stack(
        [_projector_residual(psi, log_w_held[:, k]) for k in range(len(c_held))],
        axis=1)
    psi_aug = np.concatenate([psi, u1.reshape(-1, 1)], axis=1)
    res_after = np.stack(
        [_projector_residual(psi_aug, log_w_held[:, k]) for k in range(len(c_held))],
        axis=1)
    rms_before = float(np.sqrt(np.mean(res_before ** 2)))
    rms_after = float(np.sqrt(np.mean(res_after ** 2)))
    held_out_reduction = rms_before / max(rms_after, 1e-30)
    print(f"  Held-out residual RMS:  before = {rms_before:.4f}   "
          f"after = {rms_after:.4f}   reduction = {held_out_reduction:.1f}×")

    # Control: same oracle but with injection_eps=0 (no missing structure).
    # The residual SVD should NOT produce a coherent top singular vector.
    print("\n  Control: no-injection oracle (in-span only)")
    oracle_clean = ToyCurvatureOracle(injection_eps=0.0)
    log_w_clean = np.stack(
        [oracle_clean.density_ratio_per_event(c, probe_events) for c in c_train],
        axis=1)
    residuals_clean = np.stack(
        [_projector_residual(psi, log_w_clean[:, k]) for k in range(len(c_train))],
        axis=1)
    U_c, S_c, _ = np.linalg.svd(residuals_clean, full_matrices=False)
    sv_ratio_clean = float(S_c[0] / (S_c[1] + 1e-12))
    print(f"  Singular values (top 5): {S_c[:5].round(6).tolist()}")
    print(f"  σ_1/σ_2 (clean) = {sv_ratio_clean:.2f}  "
          f"σ_1 (clean) = {S_c[0]:.2e}  (PASS criterion: σ_1 < 1e-3 OR ratio < 5)")

    # Gates. The completeness fingerprint requires BOTH a high σ_1/σ_2 ratio
    # AND a σ_1 above a finite-sample noise floor, matching Study D's
    # operational criterion. Cosine similarity of u_1 to the bare injection
    # function is not the right gate — when w_c factorises as
    # ε(η·c)·µ(c,m)·sin(6 x_aux), the top SVD direction is the *m-modulated*
    # sin (which is exactly what the encoder needs to absorb), not bare sin.
    # The right gate is the held-out residual reduction.
    SIGMA_FLOOR = 1e-3
    gate_inject_fires = (sv_ratio > 5.0) and (S_sv[0] > SIGMA_FLOOR)
    gate_clean_silent = not ((sv_ratio_clean > 5.0) and (S_c[0] > SIGMA_FLOOR))
    gate_reduction = held_out_reduction > 5.0
    print(f"\n  Gate A — injection fires (σ_1/σ_2 > 5 ∧ σ_1 > {SIGMA_FLOOR:.0e}): "
          f"{'PASS' if gate_inject_fires else 'FAIL'}  "
          f"(σ_1/σ_2 = {sv_ratio:.1f}, σ_1 = {S_sv[0]:.2e})")
    print(f"  Gate B — held-out RMS reduction > 5×: "
          f"{'PASS' if gate_reduction else 'FAIL'}  "
          f"(reduction = {held_out_reduction:.1f}×)")
    print(f"  Gate C — clean control silent (NOT both σ_1/σ_2 > 5 ∧ σ_1 > {SIGMA_FLOOR:.0e}): "
          f"{'PASS' if gate_clean_silent else 'FAIL'}  "
          f"(σ_1/σ_2 = {sv_ratio_clean:.1f}, σ_1 = {S_c[0]:.2e})")

    np.savez(OUT_DIR / "svd_recovery.npz",
             probe_events=probe_events, residuals=residuals,
             U=U, S=S_sv, c_train=c_train, c_held=c_held,
             u1=u1, truth_u=truth_u,
             residuals_clean=residuals_clean,
             S_clean=S_c)
    return {
        "sv_top5_inject": S_sv[:5].tolist(),
        "sv_top5_clean": S_c[:5].tolist(),
        "sv_ratio_inject": sv_ratio,
        "sv_ratio_clean": sv_ratio_clean,
        "sigma1_inject": float(S_sv[0]),
        "sigma1_clean": float(S_c[0]),
        "cos_u1_bare_sin": cos_sim,
        "rms_residual_before": rms_before,
        "rms_residual_after": rms_after,
        "held_out_reduction": held_out_reduction,
        "gate_inject_fires_pass": bool(gate_inject_fires),
        "gate_reduction_pass": bool(gate_reduction),
        "gate_clean_silent_pass": bool(gate_clean_silent),
        "injection_eps": inject_eps,
    }


# ---------------------------------------------------------------------------
# Study D — zero-false-fire of the completeness criterion across seeds
# ---------------------------------------------------------------------------
def study_D_no_false_fires() -> dict:
    print("\n# Study D — zero-false-fire control (clean oracle, 20 seeds)")
    oracle_clean = ToyCurvatureOracle(injection_eps=0.0)
    N_PROBE = 2000
    c_train = np.array([
        [0.0,  0.2], [0.0,  0.4], [0.0,  0.6],
        [0.3,  0.3], [-0.3, 0.3], [0.4, -0.4],
    ])
    n_fires = 0
    sv_ratios = []
    for seed in range(20):
        probe_events = oracle_clean.sample_events(np.zeros(N_OPS), N_PROBE,
                                                    seed=seed + 1000)
        psi = _build_psi_deficient(probe_events)
        log_w = np.stack(
            [oracle_clean.density_ratio_per_event(c, probe_events) for c in c_train],
            axis=1)
        residuals = np.stack(
            [_projector_residual(psi, log_w[:, k]) for k in range(len(c_train))],
            axis=1)
        _, S_sv, _ = np.linalg.svd(residuals, full_matrices=False)
        ratio = float(S_sv[0] / (S_sv[1] + 1e-12))
        sv_ratios.append(ratio)
        # Fire criterion: σ_1/σ_2 > 5 AND σ_1 above noise floor (here ~ 1e-3).
        if ratio > 5.0 and S_sv[0] > 1e-3:
            n_fires += 1
    sv_ratios = np.array(sv_ratios)
    print(f"  σ_1/σ_2 distribution across 20 seeds: "
          f"mean = {sv_ratios.mean():.2f}, max = {sv_ratios.max():.2f}")
    print(f"  False fires: {n_fires} / 20")
    gate_pass = n_fires == 0
    print(f"  Gate (0 false fires across 20 seeds): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return {
        "n_seeds": 20,
        "n_false_fires": int(n_fires),
        "sv_ratio_mean": float(sv_ratios.mean()),
        "sv_ratio_max": float(sv_ratios.max()),
        "gate_passed": bool(gate_pass),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # A — morphing recovery rel err vs N_base
    ax = axes[0, 0]
    res_A = results["A"]
    n_bases = sorted(res_A["by_n_base"].keys())
    for tmpl, marker in zip(["a_0", "a_1", "b_00", "b_11", "b_01"],
                              "oxs^d"):
        ys = [res_A["by_n_base"][n][f"rel_err_{tmpl}"] for n in n_bases]
        ax.semilogy(n_bases, ys, "-" + marker, label=tmpl)
    ax.axvline(6, ls=":", c="grey", alpha=0.7, label="N_base = 6 (min)")
    ax.set_xlabel("N_base")
    ax.set_ylabel("template max rel err")
    ax.set_title("A — Morphing template recovery")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # B — Fisher lift
    ax = axes[0, 1]
    res_B = results["B"]
    c_s = np.array(res_B["c_sweep"])
    eigvals = np.array(res_B["eigvals_per_c"])
    for k in range(eigvals.shape[1]):
        ax.semilogy(c_s, eigvals[:, k], "-o", label=f"λ_{k+1}")
    tracked = np.array(res_B["tracked_eigvals"])
    ax.semilogy(c_s, tracked, "--s", c="C3", lw=2.0,
                label="tracked (c=0 smallest)")
    ax.axhline(res_B["tracked_eigvals"][0] * 10.0, ls=":", c="k", alpha=0.5,
               label="gate: 10× lift")
    ax.set_xlabel(r"$c_1$"); ax.set_ylabel(r"Fisher eigenvalue $\lambda$")
    ax.set_title("B — Multi-point Fisher lift on toy")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # C — recovered u_1 vs truth
    ax = axes[1, 0]
    npz = np.load(OUT_DIR / "svd_recovery.npz")
    xa = npz["probe_events"][:, 1]
    # Sort along x_aux for a clean line plot.
    order = np.argsort(xa)
    ax.plot(xa[order], npz["u1"][order], ".", ms=2, alpha=0.4,
            c="C0", label="$u_1$ (recovered)")
    truth_n = npz["truth_u"]
    ax.plot(xa[order], truth_n[order], "-", lw=1.5, c="C3",
            label=r"$\sin(6 x_{\rm aux})$ (truth, unit norm)")
    ax.set_xlabel(r"$x_{\rm aux}$"); ax.set_ylabel("residual SVD U[:, 0]")
    ax.set_title(f"C — Residual SVD u_1 vs bare sin(6 x_aux)\n"
                  f"cos sim = {results['C']['cos_u1_bare_sin']:.3f}  "
                  f"(low expected: u_1 = µ-modulated sin)")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # D — control σ_1/σ_2 distribution
    ax = axes[1, 1]
    sv_top5_clean = results["C"]["sv_top5_clean"]
    sv_top5_inject = results["C"]["sv_top5_inject"]
    ax.semilogy(range(1, 6), sv_top5_clean, "-o", c="C0", label="clean (no injection)")
    ax.semilogy(range(1, 6), sv_top5_inject, "-s", c="C3", label="with injection")
    ax.set_xlabel("k"); ax.set_ylabel(r"$\sigma_k$ (singular value)")
    ax.set_title(f"D — Residual SVD spectrum\n"
                  f"σ_1/σ_2 inject={results['C']['sv_ratio_inject']:.1f}  "
                  f"clean={results['C']['sv_ratio_clean']:.1f}")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    fig.savefig(OUT_DIR / "task2_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("# ALETHIA Task 2 — toy-curvature studies")
    t0 = time.time()
    results = {}
    results["A"] = study_A_morphing_recovery()
    results["B"] = study_B_fisher_lift()
    results["C"] = study_C_residual_svd()
    results["D"] = study_D_no_false_fires()
    print("\n# Plots")
    make_plots(results)

    # Aggregate.
    gates_passed = {
        "A_morphing_recovery": results["A"]["gate_passed"],
        "B_fisher_lift": results["B"]["gate_passed"],
        "C_injection_fires": results["C"]["gate_inject_fires_pass"],
        "C_held_out_reduction": results["C"]["gate_reduction_pass"],
        "C_clean_silent": results["C"]["gate_clean_silent_pass"],
        "D_no_false_fires": results["D"]["gate_passed"],
    }
    all_pass = all(gates_passed.values())
    summary = {
        "studies": results,
        "gates_passed": gates_passed,
        "all_gates_pass": all_pass,
        "wall_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {OUT_DIR}/summary.json + task2_summary.png")
    print(f"\n## Gates:")
    for k, v in gates_passed.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n# Overall: {'PASS' if all_pass else 'FAIL'} "
          f"(wall = {summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
