"""4-way parallel driver for INV-1 Gate 3 (MG A_FB sweep).

Splits the 4 c-points across 4 workers, each with its own pre-copied
MG process directory under ``vendor/MG5_aMC/processes/dy_smeft{,_w1,_w2,_w3}``.
Each worker runs all 5 m-points for its assigned c-point sequentially
(~20 min per cell at nevents=8000), so wall time is ~5 * 20 min = ~1.7 h
total instead of the ~6.6 h sequential time on the same machine.

Each worker writes a per-slice JSON; the driver merges them at the end
into a single ``mg_crosscheck_afb_summary_n8k_parallel.json`` with the
gate verdict over the full grid.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

# Import shared constants from the sequential harness so they stay in sync.
sys.path.insert(0, str(HERE))
from mg_crosscheck_afb import (  # noqa: E402
    C_POINTS, M_GRID_TEV, LAMBDA_TEV, SQRT_S_GEV, MG_M_WINDOW_TEV,
    PERCENT_THRESHOLD,
)


WORKER_PROCESS_DIRS = [
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w1",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w2",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w3",
]
OUT_DIR = HERE / "output"
PARTIAL_DIR = OUT_DIR / "mg_afb_partial"
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)


def _run_one_slice(worker_idx: int, c_idx: int, nevents: int) -> dict:
    """Run all m-points for a single c-point in a fresh subprocess."""
    sys.path.insert(0, str(ROOT))
    from modules.surrogate.oracle_madgraph import MadGraphSMEFTOracle
    from modules.surrogate.features import N_WC, WC_NAMES

    process_dir = WORKER_PROCESS_DIRS[worker_idx]
    c_dict = C_POINTS[c_idx]
    c = np.zeros(N_WC)
    for j, name in enumerate(WC_NAMES):
        c[j] = c_dict[name]

    oracle = MadGraphSMEFTOracle(
        process_dir=process_dir,
        lambda_gev=LAMBDA_TEV * 1000.0,
        m_window_tev=MG_M_WINDOW_TEV,
        nevents=int(nevents),
        verbose=False,
        reuse_cache=False,
        keep_artifacts=False,
        ptl_min_gev=0.0,
        etal_max=10.0,
    )

    n_m = len(M_GRID_TEV)
    row = {
        "worker_idx": worker_idx,
        "c_idx": c_idx,
        "c_point": c_dict,
        "process_dir": str(process_dir),
        "cells": [],
    }
    t_slice = time.perf_counter()
    for k, m in enumerate(M_GRID_TEV):
        cell = {"m_ll_tev": float(m)}
        t0 = time.perf_counter()
        try:
            out = oracle.truth_costhetaCS_bins(
                c.reshape(1, -1), np.array([float(m)]))
            cell.update({
                "A_FB_MG": float(out["A_FB"][0]),
                "sigma_F": float(out["sigma_F"][0]),
                "sigma_B": float(out["sigma_B"][0]),
                "sigma": float(out["sigma"][0]),
                "n_F": int(out["n_F"][0]),
                "n_B": int(out["n_B"][0]),
                "ok": True,
            })
        except Exception as exc:  # noqa: BLE001
            cell.update({"ok": False, "error": repr(exc)})
        cell["seconds"] = time.perf_counter() - t0
        row["cells"].append(cell)
        # checkpoint per cell so partial progress survives crashes
        with open(PARTIAL_DIR / f"worker_{worker_idx}.json", "w") as f:
            json.dump(row, f, indent=2)
    row["slice_seconds"] = time.perf_counter() - t_slice
    return row


def _analytic_reference() -> np.ndarray:
    """Re-evaluate analytic A_FB on the full grid (cheap; ~seconds)."""
    from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
    from modules.surrogate.features import N_WC, WC_NAMES

    oracle = AnalyticSMEFTOracle(
        sqrt_s_gev=SQRT_S_GEV,
        lambda_scale_gev=LAMBDA_TEV * 1000.0,
        order="quadratic",
        pdf="auto",
        noise_frac=0.0,
        seed=0,
    )
    afb = np.zeros((len(C_POINTS), len(M_GRID_TEV)))
    for i, c_dict in enumerate(C_POINTS):
        c = np.zeros(N_WC)
        for j, name in enumerate(WC_NAMES):
            c[j] = c_dict[name]
        c_batch = np.tile(c, (len(M_GRID_TEV), 1))
        afb[i, :] = oracle.truth_afb(c_batch, M_GRID_TEV)
    return afb


def main(nevents: int = 8000) -> None:
    t0 = time.perf_counter()
    print(f"INV-1 gate 3 parallel sweep: 4 workers, "
          f"{len(C_POINTS)} c-points x {len(M_GRID_TEV)} m-points, "
          f"nevents = {nevents}", flush=True)
    for w, pd in enumerate(WORKER_PROCESS_DIRS):
        if not (pd / "bin" / "generate_events").exists():
            print(f"  worker {w}: MISSING process_dir {pd}", flush=True)
            sys.exit(2)
    print(f"  output -> {OUT_DIR}/mg_crosscheck_afb_summary_n8k_parallel.json",
          flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_run_one_slice, w, c_idx, nevents): (w, c_idx)
            for w, c_idx in enumerate(range(len(C_POINTS)))
        }
        for fut in as_completed(futures):
            w, c_idx = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"  worker {w} (c_idx={c_idx}) done in "
                      f"{row['slice_seconds']/60.0:.1f} min", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  worker {w} (c_idx={c_idx}) FAILED: {exc!r}",
                      flush=True)

    # Re-assemble into (c, m) arrays.
    n_c, n_m = len(C_POINTS), len(M_GRID_TEV)
    afb_mg = np.full((n_c, n_m), np.nan)
    sigma_F = np.full((n_c, n_m), np.nan)
    sigma_B = np.full((n_c, n_m), np.nan)
    sigma = np.full((n_c, n_m), np.nan)
    n_F = np.full((n_c, n_m), 0, dtype=int)
    n_B = np.full((n_c, n_m), 0, dtype=int)
    cell_seconds = np.full((n_c, n_m), np.nan)
    failures: list[dict] = []
    for row in rows:
        i = row["c_idx"]
        for k, cell in enumerate(row["cells"]):
            if cell.get("ok"):
                afb_mg[i, k] = cell["A_FB_MG"]
                sigma_F[i, k] = cell["sigma_F"]
                sigma_B[i, k] = cell["sigma_B"]
                sigma[i, k] = cell["sigma"]
                n_F[i, k] = cell["n_F"]
                n_B[i, k] = cell["n_B"]
            else:
                failures.append({
                    "c_idx": i,
                    "m_ll_tev": cell["m_ll_tev"],
                    "error": cell.get("error", "unknown"),
                })
            cell_seconds[i, k] = cell.get("seconds", float("nan"))

    afb_analytic = _analytic_reference()
    deltas = afb_mg - afb_analytic
    finite = np.isfinite(deltas)
    max_abs_delta = float(np.nanmax(np.abs(deltas))) if finite.any() else float("nan")
    median_abs_delta = float(np.nanmedian(np.abs(deltas))) if finite.any() else float("nan")
    gate_passes = bool(
        finite.all()
        and (np.abs(deltas) <= PERCENT_THRESHOLD / 100.0).all()
    )

    summary = {
        "mode": "real-parallel",
        "n_workers": 4,
        "nevents": int(nevents),
        "lambda_tev": LAMBDA_TEV,
        "sqrt_s_gev": SQRT_S_GEV,
        "pass_threshold_percent": PERCENT_THRESHOLD,
        "gate_passes": gate_passes,
        "max_abs_delta_pct": 100.0 * max_abs_delta,
        "median_abs_delta_pct": 100.0 * median_abs_delta,
        "c_points": C_POINTS,
        "m_grid_tev": M_GRID_TEV.tolist(),
        "afb_mg": afb_mg.tolist(),
        "afb_analytic": afb_analytic.tolist(),
        "deltas": deltas.tolist(),
        "sigma_F": sigma_F.tolist(),
        "sigma_B": sigma_B.tolist(),
        "sigma": sigma.tolist(),
        "n_F": n_F.tolist(),
        "n_B": n_B.tolist(),
        "cell_seconds": cell_seconds.tolist(),
        "failures": failures,
        "wall_seconds": time.perf_counter() - t0,
    }
    out_path = OUT_DIR / "mg_crosscheck_afb_summary_n8k_parallel.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {len(rows)}/{n_c} slices.  gate_passes = {gate_passes}",
          flush=True)
    print(f"  max |Δ| = {100*max_abs_delta:.3f} %, "
          f"median |Δ| = {100*median_abs_delta:.3f} %",
          flush=True)
    print(f"  wall time = {summary['wall_seconds']/60.0:.1f} min",
          flush=True)
    print(f"  summary  -> {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nevents", type=int, default=8000,
                        help="MG events per cell (default 8000).")
    args = parser.parse_args()
    main(nevents=args.nevents)
