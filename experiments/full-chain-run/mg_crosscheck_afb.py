"""Gate 3 of INV-1: MadGraph A_FB cross-check.

The analytic angular extension to the rate oracle (INV-1 in
``ALETHEIA_investigations.md``) introduces ``A_FB(c, m_ll)`` as the
chirality-asymmetric observable that breaks the residual vertex-direction
degeneracy seen by ``mu(c, m_ll)`` alone. Validation gate 3 requires
that ``A_FB(c, m_ll)`` from the analytic oracle matches MadGraph to
within 1% across a small grid of ``(c, m_ll)`` points. This is the gate
that catches the cos-θ*_CS sign / dilution-direction error called out in
Pitfall 4: angular integrals are insensitive to the sign and pass gate 1
either way, so gate 3 is non-skippable.

Two run modes:

* ``--dry-run`` (default) -- emit the analytic ``A_FB`` reference table
  and the call schedule; no MG calls. Used when a follow-on machine will
  run the full sweep.
* ``--run`` -- actually execute MadGraph. Loops over the (c, m) grid and
  calls :meth:`MadGraphSMEFTOracle.truth_costhetaCS_bins`, which runs a
  single ``generate_events`` per ``(c, m)`` and post-hoc buckets its LHE
  events into the forward / backward hemispheres in the Collins-Soper
  frame. ``A_FB_MG = (sigma_F - sigma_B) / (sigma_F + sigma_B)`` is then
  compared to ``truth_afb``.

Budget:
  4 c-points x 5 m-points = 20 MG calls (one per cell, NOT 40 -- the
  forward/backward split is post-hoc from the LHE events of a single
  call). At ~10 minutes per call with ``nevents = 4000`` (statistical
  noise on ``A_FB`` ~ 1 / sqrt(N_F + N_B) ~ 1.6%, enough for the 1%
  gate when averaged over ~5k events), wall time ≈ 3.3 hours.

Smoke mode (``--smoke``) runs a single (SM, m_ll = 1 TeV) cell at low
nevents to confirm the wiring without spending the full budget.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle


# ---------------------------------------------------------------------------
# Comparison configuration -- EFT-valid per INV-1 pitfall 7.
# ---------------------------------------------------------------------------
LAMBDA_TEV: float = 2.0
SQRT_S_GEV: float = 13000.0

# 3 non-zero c-points + SM (the spec asks for 3-5; we use 3 to keep the
# budget inside ~3.3 hours wall time at nevents=4000). |c| <= 0.3 keeps
# s/Lambda^2 < 1.5 across the m-grid with Lambda = 2 TeV; mirror Boughezal
# et al. (arXiv:2303.08257).
C_POINTS: list[dict[str, float]] = [
    {"cHq3": 0.0, "cHq1": 0.0, "clq3": 0.0, "clq1": 0.0},   # SM reference
    {"cHq3": 0.0, "cHq1": 0.0, "clq3": 0.25, "clq1": 0.0},  # 4F triplet
    {"cHq3": 0.0, "cHq1": 0.0, "clq3": 0.0, "clq1": 0.25},  # 4F singlet
    {"cHq3": 0.2, "cHq1": 0.0, "clq3": 0.0, "clq1": 0.0},   # vertex triplet
]

M_GRID_TEV: np.ndarray = np.array([0.4, 0.7, 1.1, 1.6, 2.3])

# Single forward/backward split in cos-θ*_CS (the F/B classifier per
# event is ``cos θ*_CS >= 0``). The 2-bin notation is kept for the summary
# JSON so downstream tooling that reads the dry-run schema still works.
COSTHETA_BINS: list[tuple[float, float]] = [
    (-1.0, 0.0),   # backward hemisphere
    (0.0, 1.0),    # forward hemisphere
]

# nevents tuned so the per-(c,m) statistical noise on A_FB is well below
# the 1% gate. Sigma(A_FB) ≈ 1/sqrt(n_F + n_B), so n=4000 gives ~1.6% at
# the SM and tighter (because |A_FB| > 0 reduces sigma) BSM points.
MG_NEVENTS_PER_CELL: int = 4000
MG_M_WINDOW_TEV: float = 0.20   # +-100 GeV around the bin centre.
PERCENT_THRESHOLD: float = 1.0  # gate 3 passes if all |Delta A_FB| <= 1 %.


def estimate_budget(*, nevents: int = MG_NEVENTS_PER_CELL,
                    n_c: int | None = None,
                    n_m: int | None = None) -> dict:
    n_c = n_c if n_c is not None else len(C_POINTS)
    n_m = n_m if n_m is not None else len(M_GRID_TEV)
    n_bins = len(COSTHETA_BINS)
    # One MG call per (c, m) cell; the forward / backward split is
    # post-hoc from the LHE events of a single call (NOT 2 calls per
    # cell as an earlier draft of this script assumed).
    n_calls = n_c * n_m
    seconds_per_call = max(60.0, nevents * 0.15)  # ~10 min at nevents=4000.
    total_minutes = n_calls * seconds_per_call / 60.0
    return {
        "n_c_points": n_c,
        "n_m_points": n_m,
        "n_costheta_bins": n_bins,
        "n_mg_calls": n_calls,
        "estimated_seconds_per_call": seconds_per_call,
        "estimated_total_hours": total_minutes / 60.0,
    }


def analytic_reference(*,
                       c_points: list[dict[str, float]] = C_POINTS,
                       m_grid_tev: np.ndarray = M_GRID_TEV) -> dict:
    """Compute the analytic A_FB on the (c, m) grid as the target.

    Uses :meth:`AnalyticSMEFTOracle.truth_afb` (the existing wrapper around
    :func:`differential_afb`). MG values when computed (with ``--run``) are
    compared to these per (c-point, m) cell.
    """
    oracle = AnalyticSMEFTOracle(
        sqrt_s_gev=SQRT_S_GEV,
        lambda_scale_gev=LAMBDA_TEV * 1000.0,
        order="quadratic",
        pdf="auto",        # CT18NNLO if LHAPDF is available.
        noise_frac=0.0,
        seed=0,
    )
    afb = np.zeros((len(c_points), len(m_grid_tev)))
    from modules.surrogate.features import N_WC, WC_NAMES
    for i, c_dict in enumerate(c_points):
        c = np.zeros(N_WC)
        for j, name in enumerate(WC_NAMES):
            c[j] = c_dict[name]
        c_batch = np.tile(c, (len(m_grid_tev), 1))
        afb[i, :] = oracle.truth_afb(c_batch, m_grid_tev)
    return {
        "c_points": c_points,
        "m_grid_tev": list(map(float, m_grid_tev)),
        "afb_analytic": afb.tolist(),
    }


def _c_label(c_dict: dict[str, float]) -> str:
    label = ", ".join(f"{k}={v:.2g}" for k, v in c_dict.items() if v != 0.0)
    return label or "SM"


def dry_run(output_path: Path) -> None:
    """Print the budget and the call schedule without invoking MG."""
    print("INV-1 gate 3 DRY-RUN -- no MadGraph calls will be made.\n")
    budget = estimate_budget()
    print("Budget:")
    for key, value in budget.items():
        print(f"  {key:>30}  =  {value}")
    print()
    print(f"Configuration: Lambda = {LAMBDA_TEV} TeV, sqrt(s) = {SQRT_S_GEV} GeV,")
    print(f"               nevents per cell = {MG_NEVENTS_PER_CELL}")
    print(f"               pass criterion: max |Delta A_FB| <= {PERCENT_THRESHOLD} %\n")
    print("Call schedule (one MG run per (c, m); F/B from post-hoc LHE bucketing):")
    idx = 0
    for c_dict in C_POINTS:
        c_label = _c_label(c_dict)
        for m in M_GRID_TEV:
            idx += 1
            print(f"  [{idx:3d}]  c=({c_label}), m_ll = {m:.2f} TeV")
    print(f"\nTotal: {idx} MG runs (each ~{MG_NEVENTS_PER_CELL} events).")
    print()
    print("Analytic A_FB reference (this is what MG must match):")
    ref = analytic_reference()
    afb_arr = np.array(ref["afb_analytic"])
    for i, c_dict in enumerate(C_POINTS):
        c_label = _c_label(c_dict)
        print(f"  c = ({c_label})")
        for j, m in enumerate(M_GRID_TEV):
            print(f"      m = {m:.2f} TeV  ->  A_FB_analytic = {afb_arr[i, j]:+.4f}")

    summary = {
        "mode": "dry-run",
        "budget": budget,
        "lambda_tev": LAMBDA_TEV,
        "sqrt_s_gev": SQRT_S_GEV,
        "pass_threshold_percent": PERCENT_THRESHOLD,
        "analytic_reference": ref,
        "notes": (
            "Dry-run: emits the analytic A_FB target and the per-(c,m) call "
            "schedule. Run with --run to actually invoke MadGraph; one MG "
            "call per (c, m) cell with post-hoc LHE bucketing into F/B."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")


def real_run(
    output_path: Path,
    *,
    nevents: int = MG_NEVENTS_PER_CELL,
    c_points: list[dict[str, float]] | None = None,
    m_grid_tev: np.ndarray | None = None,
    smoke: bool = False,
) -> dict:
    """Invoke MadGraph for each (c, m) cell; compare to analytic A_FB.

    One ``generate_events`` per cell; the forward / backward split is
    computed post-hoc by bucketing the LHE events on
    ``cos θ*_CS`` (sign convention: z-axis along ``sign(P_z(ℓℓ))``,
    angle measured on ``ℓ⁻``; matches INV-1 Conventions and Pitfall 4).
    """
    from modules.surrogate.oracle_madgraph import MadGraphSMEFTOracle
    from modules.surrogate.features import N_WC, WC_NAMES

    c_points = c_points if c_points is not None else C_POINTS
    m_grid_tev = (m_grid_tev if m_grid_tev is not None
                  else M_GRID_TEV)

    print(f"INV-1 gate 3 REAL RUN: {len(c_points)} c-points x "
          f"{len(m_grid_tev)} m-points = {len(c_points)*len(m_grid_tev)} MG calls")
    print(f"  Lambda = {LAMBDA_TEV} TeV, sqrt(s) = {SQRT_S_GEV} GeV, "
          f"nevents = {nevents}")
    print(f"  m_window = {MG_M_WINDOW_TEV} TeV, smoke = {smoke}")

    oracle = MadGraphSMEFTOracle(
        lambda_gev=LAMBDA_TEV * 1000.0,
        m_window_tev=MG_M_WINDOW_TEV,
        nevents=int(nevents),
        verbose=True,
        # Persistent cache: the inclusive xs from each call can still
        # populate the rate cache, but the cos-θ*-binned method does
        # not currently share that cache (the per-bin event counts are
        # what matter and they are not in the rate cache schema).
        reuse_cache=False,
        keep_artifacts=False,
        # Widen the fiducial lepton acceptance so MG returns the
        # parton-level A_FB the analytic reference computes. The
        # default run_card cuts (ptl > 10 GeV, |etal| < 2.5) remove
        # forward leptons that carry most of A_FB and bias the MG
        # number low by ~50% (gate 3 would then fail purely from the
        # acceptance mismatch, not from any chiral-construction error).
        ptl_min_gev=0.0,
        etal_max=10.0,
    )

    ref = analytic_reference(c_points=c_points, m_grid_tev=m_grid_tev)
    afb_analytic = np.array(ref["afb_analytic"])

    n_c = len(c_points)
    n_m = len(m_grid_tev)
    afb_mg = np.full((n_c, n_m), np.nan)
    sigma_F = np.full((n_c, n_m), np.nan)
    sigma_B = np.full((n_c, n_m), np.nan)
    sigma_tot = np.full((n_c, n_m), np.nan)
    n_F_arr = np.full((n_c, n_m), 0, dtype=int)
    n_B_arr = np.full((n_c, n_m), 0, dtype=int)
    deltas = np.full((n_c, n_m), np.nan)
    cell_seconds = np.full((n_c, n_m), np.nan)
    failures: list[dict] = []

    t_start = time.perf_counter()
    cell_idx = 0
    n_cells = n_c * n_m
    for i, c_dict in enumerate(c_points):
        c = np.zeros(N_WC)
        for j, name in enumerate(WC_NAMES):
            c[j] = c_dict[name]
        for k, m in enumerate(m_grid_tev):
            cell_idx += 1
            label = _c_label(c_dict)
            print(f"\n  [{cell_idx}/{n_cells}] c=({label}), m_ll = "
                  f"{float(m):.2f} TeV ...", flush=True)
            t0 = time.perf_counter()
            try:
                out = oracle.truth_costhetaCS_bins(
                    c.reshape(1, -1), np.array([float(m)]))
                afb_mg[i, k] = float(out["A_FB"][0])
                sigma_F[i, k] = float(out["sigma_F"][0])
                sigma_B[i, k] = float(out["sigma_B"][0])
                sigma_tot[i, k] = float(out["sigma"][0])
                n_F_arr[i, k] = int(out["n_F"][0])
                n_B_arr[i, k] = int(out["n_B"][0])
                deltas[i, k] = afb_mg[i, k] - afb_analytic[i, k]
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "c_label": label,
                    "m_ll_tev": float(m),
                    "error": repr(exc),
                })
                print(f"      FAILED: {exc!r}", flush=True)
            cell_seconds[i, k] = time.perf_counter() - t0
            if np.isfinite(afb_mg[i, k]):
                print(f"      A_FB_MG       = {afb_mg[i, k]:+.4f}  "
                      f"(n_F={n_F_arr[i, k]}, n_B={n_B_arr[i, k]})\n"
                      f"      A_FB_analytic = {afb_analytic[i, k]:+.4f}\n"
                      f"      Delta         = {deltas[i, k]:+.4f}  "
                      f"({100*deltas[i, k]:+.2f} %)\n"
                      f"      cell time     = {cell_seconds[i, k]:.1f} s",
                      flush=True)

    total_seconds = time.perf_counter() - t_start
    finite = np.isfinite(deltas)
    max_abs_delta = float(np.nanmax(np.abs(deltas))) if finite.any() else np.nan
    median_abs_delta = (float(np.nanmedian(np.abs(deltas)))
                       if finite.any() else np.nan)
    gate_passes = (bool(finite.all() and (np.abs(deltas) <= PERCENT_THRESHOLD / 100.0).all())
                   if not smoke else None)
    n_finite = int(finite.sum())

    summary = {
        "mode": "real" if not smoke else "real-smoke",
        "lambda_tev": LAMBDA_TEV,
        "sqrt_s_gev": SQRT_S_GEV,
        "nevents_per_cell": int(nevents),
        "m_window_tev": MG_M_WINDOW_TEV,
        "pass_threshold_percent": PERCENT_THRESHOLD,
        "c_points": c_points,
        "m_grid_tev": list(map(float, m_grid_tev)),
        "afb_analytic": afb_analytic.tolist(),
        "afb_mg": afb_mg.tolist(),
        "delta": deltas.tolist(),
        "sigma_F_pb": sigma_F.tolist(),
        "sigma_B_pb": sigma_B.tolist(),
        "sigma_total_pb": sigma_tot.tolist(),
        "n_forward": n_F_arr.tolist(),
        "n_backward": n_B_arr.tolist(),
        "cell_seconds": cell_seconds.tolist(),
        "total_seconds": float(total_seconds),
        "n_cells": int(n_cells),
        "n_cells_finite": n_finite,
        "max_abs_delta": (None if not np.isfinite(max_abs_delta)
                          else float(max_abs_delta)),
        "median_abs_delta": (None if not np.isfinite(median_abs_delta)
                            else float(median_abs_delta)),
        "gate_passes": gate_passes,
        "failures": failures,
        "notes": (
            "One MG call per (c, m); LHE events bucketed post-hoc by "
            "cos θ*_CS into F/B; A_FB_MG = (F-B)/(F+B). Sign convention: "
            "CS z-axis along sign(P_z(ℓℓ)); cos θ*_CS computed on ℓ⁻ "
            "(PDG +13). Gate 3 passes if max |Delta A_FB| <= "
            f"{PERCENT_THRESHOLD}% across all cells."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")
    print(f"Total wall time: {total_seconds/60:.1f} min")
    if gate_passes is True:
        print(f"GATE 3 PASSED: max |Delta A_FB| = "
              f"{max_abs_delta*100:.2f}% <= {PERCENT_THRESHOLD}%")
    elif gate_passes is False:
        print(f"GATE 3 FAILED: max |Delta A_FB| = "
              f"{max_abs_delta*100:.2f}% > {PERCENT_THRESHOLD}%")
    else:
        print("Smoke run -- gate not evaluated.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually invoke MadGraph on the full 4 x 5 = 20-cell grid "
             "(~3.3 hours wall time). Without this flag the script "
             "defaults to the dry-run schedule + analytic reference.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single SM cell at m_ll = 1.1 TeV with nevents = 500 "
             "to exercise the full pipeline end-to-end. Does NOT evaluate "
             "the gate. Intended for verifying the MG oracle wiring before "
             "the user spends the full ~3.3h compute budget.",
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=None,
        help="Override nevents per cell (defaults to "
             f"{MG_NEVENTS_PER_CELL} for --run, 500 for --smoke).",
    )
    parser.add_argument(
        "--c-points",
        type=int,
        default=None,
        help="Use only the first N c-points (default: all 4). For ad-hoc "
             "partial sweeps.",
    )
    parser.add_argument(
        "--m-points",
        type=int,
        default=None,
        help="Use only the first N m-points (default: all 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "output" / "mg_crosscheck_afb_summary.json",
        help="Where to save the summary JSON.",
    )
    args = parser.parse_args()

    if args.smoke:
        nevents = args.nevents if args.nevents is not None else 500
        c_points = C_POINTS[:1]                  # SM only
        m_grid = np.array([1.1])                 # one bin away from the Z pole
        real_run(args.output, nevents=nevents,
                 c_points=c_points, m_grid_tev=m_grid, smoke=True)
        return

    if args.run:
        nevents = (args.nevents if args.nevents is not None
                   else MG_NEVENTS_PER_CELL)
        c_points = (C_POINTS[: args.c_points]
                    if args.c_points is not None else C_POINTS)
        m_grid = (M_GRID_TEV[: args.m_points]
                  if args.m_points is not None else M_GRID_TEV)
        real_run(args.output, nevents=nevents,
                 c_points=c_points, m_grid_tev=m_grid, smoke=False)
        return

    dry_run(args.output)


if __name__ == "__main__":
    main()
