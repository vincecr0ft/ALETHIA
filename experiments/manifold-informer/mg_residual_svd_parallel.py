"""8-way parallel MadGraph event generation for the ManifoldInformer
residual-SVD experiment.

Each of 8 workers takes one process_dir and generates ``N_EVENTS`` MG events
at one working point (c-value) over a wide m_ll window. The events are
LHE-parsed into the canonical ``(log m_ll/M_ref, cos θ*_CS)`` array and
saved per-c so the residual-SVD machinery can consume them downstream.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/manifold-informer/mg_residual_svd_parallel.py

The 9 c-values are:
  c[0]:  SM                            (the probe-event source)
  c[1-8]: 8 working points used for the residual matrix R columns

Outputs:
  output_mg_events/
    probe_sm.npz           N_EVENTS×2 SM events (for the in-span basis)
    c_<idx>.npz            N_EVENTS×2 events at working point <idx>
    summary.json           per-c MG runtimes, event counts, paths
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# 8 worker process dirs. They were created by copying dy_smeft to dy_smeft_w1..7.
WORKER_PROCESS_DIRS = [
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w1",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w2",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w3",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w4",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w5",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w6",
    ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft_w7",
]
N_WORKERS = len(WORKER_PROCESS_DIRS)

# Working points: 1 SM (the probe-event source for in-span basis) + 8 c-values
# that will be the columns of the residual matrix R. Same set as the analytic
# SVD harness so the MG result is a fidelity check.
C_POINTS = [
    # idx 0 — SM probe events
    np.array([0.0, 0.0, 0.0, 0.0]),
    # idx 1-8: residual-SVD columns
    np.array([0.0, 0.0, 0.3, 0.0]),
    np.array([0.0, 0.0, 0.5, 0.0]),
    np.array([0.0, 0.0, 0.7, 0.0]),
    np.array([0.2, 0.0, 0.4, 0.0]),
    np.array([0.4, 0.0, 0.4, 0.0]),
    np.array([0.0, 0.2, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.4, 0.2]),
    np.array([0.3, 0.0, 0.3, 0.0]),
]
N_C = len(C_POINTS)

# MG generation parameters.
N_EVENTS_PER_C = 4000
M_WINDOW_TEV = 2.0                     # full window — events span [0.3, 2.3] TeV
M_TEV_CENTER = 1.3                     # window centre in TeV
LAMBDA_TEV = 1.0

OUT_DIR = HERE / "output_mg_events"
OUT_DIR.mkdir(exist_ok=True)


def _run_one(worker_idx: int, c_idx: int) -> dict:
    """Worker: generate MG events at one working point, save (log m, cos θ*)."""
    sys.path.insert(0, str(ROOT))
    from modules.surrogate.oracle_madgraph import MadGraphSMEFTOracle

    process_dir = WORKER_PROCESS_DIRS[worker_idx]
    c = C_POINTS[c_idx]

    oracle = MadGraphSMEFTOracle(
        process_dir=process_dir,
        lambda_gev=LAMBDA_TEV * 1000.0,
        m_window_tev=M_WINDOW_TEV,
        nevents=N_EVENTS_PER_C,
        verbose=False,
        reuse_cache=False,
        keep_artifacts=False,
        # Match the parton-level fiducial used in the existing
        # mg_crosscheck_afb cross-check: no p_T cut, |η| < 10 so we
        # don't lose forward leptons that carry the bulk of A_FB.
        ptl_min_gev=0.0,
        etal_max=10.0,
    )

    t0 = time.perf_counter()
    try:
        events = oracle.sample_events_mg(
            c, m_tev_center=M_TEV_CENTER,
            m_window_tev=M_WINDOW_TEV,
            nevents=N_EVENTS_PER_C,
        )
        wall = time.perf_counter() - t0
        n_kept = int(events.shape[0])
        # Save.
        if c_idx == 0:
            out_path = OUT_DIR / "probe_sm.npz"
        else:
            out_path = OUT_DIR / f"c_{c_idx:02d}.npz"
        np.savez(out_path, events=events, c=c)
        return {
            "worker_idx": worker_idx,
            "c_idx": c_idx,
            "c": c.tolist(),
            "n_events_requested": N_EVENTS_PER_C,
            "n_events_kept": n_kept,
            "wall_seconds": wall,
            "out_path": str(out_path),
            "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "worker_idx": worker_idx,
            "c_idx": c_idx,
            "c": c.tolist(),
            "wall_seconds": time.perf_counter() - t0,
            "ok": False,
            "error": repr(exc),
        }


def main():
    print(f"# 8-way parallel MG event generation")
    print(f"  N_C = {N_C}  (1 SM probe + {N_C - 1} working points for R)")
    print(f"  N_EVENTS_PER_C = {N_EVENTS_PER_C}")
    print(f"  m_window = {M_WINDOW_TEV} TeV  (centred at {M_TEV_CENTER} TeV)")
    print(f"  Λ = {LAMBDA_TEV} TeV")
    print(f"  N_WORKERS = {N_WORKERS}")

    # Sanity: every process_dir exists.
    for w, pd in enumerate(WORKER_PROCESS_DIRS):
        if not (pd / "bin" / "generate_events").exists():
            print(f"  worker {w}: MISSING process_dir {pd}", flush=True)
            sys.exit(2)

    # 9 c-points + 8 workers: we have 1 more c-point than workers, so the
    # first worker to free up does the 9th job.
    t_start = time.perf_counter()
    rows = []
    next_c = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        # Seed: launch min(N_C, N_WORKERS) initially.
        futures = {}
        for w in range(min(N_C, N_WORKERS)):
            futures[pool.submit(_run_one, w, next_c)] = (w, next_c)
            next_c += 1
        while futures:
            done = next(as_completed(futures))
            w, c_idx = futures.pop(done)
            try:
                row = done.result()
                rows.append(row)
                if row["ok"]:
                    print(f"  worker {w}: c_idx={c_idx}  "
                          f"n={row['n_events_kept']}  "
                          f"wall={row['wall_seconds']:.1f}s  "
                          f"-> {Path(row['out_path']).name}",
                          flush=True)
                else:
                    print(f"  worker {w}: c_idx={c_idx} FAILED  "
                          f"{row.get('error', 'unknown')}",
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  worker {w}: c_idx={c_idx} EXCEPTION {exc!r}",
                      flush=True)
                rows.append({"worker_idx": w, "c_idx": c_idx, "ok": False,
                              "error": repr(exc)})
            # Free worker — assign next c if any left.
            if next_c < N_C:
                futures[pool.submit(_run_one, w, next_c)] = (w, next_c)
                next_c += 1

    wall_total = time.perf_counter() - t_start
    summary = {
        "config": {
            "n_c": N_C,
            "n_events_per_c": N_EVENTS_PER_C,
            "m_window_tev": M_WINDOW_TEV,
            "m_tev_center": M_TEV_CENTER,
            "lambda_tev": LAMBDA_TEV,
            "n_workers": N_WORKERS,
            "process_dirs": [str(p) for p in WORKER_PROCESS_DIRS],
        },
        "rows": rows,
        "wall_seconds": wall_total,
        "successes": sum(1 for r in rows if r.get("ok")),
        "failures": sum(1 for r in rows if not r.get("ok")),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n# wrote {OUT_DIR}/summary.json")
    print(f"# wall total: {wall_total:.1f}s")
    print(f"# successes: {summary['successes']} / {N_C}")
    if summary["failures"]:
        print(f"# failures: {summary['failures']}")
    return summary


if __name__ == "__main__":
    main()
