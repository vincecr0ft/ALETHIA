"""8-way parallel MG event generation at 32k events/c, high-m_ll window.

Same as ``mg_residual_svd_parallel_highm.py`` but at nevents=32000 per
working point — the statistics tier the existing INV-1 gate 3 cross-check
reported as the noise-floor that stabilises (4000 events: max |Δ| = 1.67%;
32000 events: max |Δ| = 1.11%; 64000 events: max |Δ| = 1.39% with 1 outlier).
For the residual-SVD machinery this gives an order-of-magnitude tighter
MC noise per c-point at the cost of ~8× wall.

9 c-points × 32000 events on 8 workers ≈ 8 min wall (vs ~1.5 min at 4000).

Outputs land under ``output_mg_events_highstats/``.
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

WORKER_PROCESS_DIRS = [
    ROOT / "vendor" / "MG5_aMC" / "processes" / f"dy_smeft{suf}"
    for suf in ["", "_w1", "_w2", "_w3", "_w4", "_w5", "_w6", "_w7"]
]
N_WORKERS = len(WORKER_PROCESS_DIRS)

C_POINTS = [
    np.array([0.0, 0.0, 0.0, 0.0]),
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

# High-stats settings.
N_EVENTS_PER_C = 32000
M_WINDOW_TEV = 1.8                    # [0.5, 2.3] TeV
M_TEV_CENTER = 1.4
LAMBDA_TEV = 1.0

OUT_DIR = HERE / "output_mg_events_highstats"
OUT_DIR.mkdir(exist_ok=True)


def _run_one(worker_idx: int, c_idx: int) -> dict:
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
        out_path = (OUT_DIR / "probe_sm.npz") if c_idx == 0 \
            else (OUT_DIR / f"c_{c_idx:02d}.npz")
        np.savez(out_path, events=events, c=c)
        return {"worker_idx": worker_idx, "c_idx": c_idx, "c": c.tolist(),
                "n_events_kept": int(events.shape[0]),
                "wall_seconds": wall,
                "out_path": str(out_path), "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"worker_idx": worker_idx, "c_idx": c_idx, "c": c.tolist(),
                "wall_seconds": time.perf_counter() - t0,
                "ok": False, "error": repr(exc)}


def main():
    print(f"# 8-way parallel MG event gen (high-stats, 32k events, m_ll ∈ [0.5, 2.3] TeV)")
    print(f"  N_C = {N_C}  N_EVENTS_PER_C = {N_EVENTS_PER_C}")
    for w, pd in enumerate(WORKER_PROCESS_DIRS):
        if not (pd / "bin" / "generate_events").exists():
            print(f"  MISSING {pd}"); sys.exit(2)
    t_start = time.perf_counter()
    rows = []
    next_c = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for w in range(min(N_C, N_WORKERS)):
            futures[pool.submit(_run_one, w, next_c)] = (w, next_c)
            next_c += 1
        while futures:
            done = next(as_completed(futures))
            w, c_idx = futures.pop(done)
            try:
                row = done.result(); rows.append(row)
                if row["ok"]:
                    print(f"  worker {w}: c_idx={c_idx}  "
                          f"n={row['n_events_kept']}  "
                          f"wall={row['wall_seconds']:.1f}s  "
                          f"-> {Path(row['out_path']).name}", flush=True)
                else:
                    print(f"  worker {w}: c_idx={c_idx} FAILED  "
                          f"{row.get('error')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                rows.append({"worker_idx": w, "c_idx": c_idx, "ok": False,
                              "error": repr(exc)})
                print(f"  worker {w}: EXCEPTION {exc!r}", flush=True)
            if next_c < N_C:
                futures[pool.submit(_run_one, w, next_c)] = (w, next_c)
                next_c += 1
    wall_total = time.perf_counter() - t_start
    summary = {
        "config": {"n_c": N_C, "n_events_per_c": N_EVENTS_PER_C,
                    "m_window_tev": M_WINDOW_TEV, "m_tev_center": M_TEV_CENTER,
                    "lambda_tev": LAMBDA_TEV, "n_workers": N_WORKERS},
        "rows": rows, "wall_seconds": wall_total,
        "successes": sum(1 for r in rows if r.get("ok")),
        "failures": sum(1 for r in rows if not r.get("ok")),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {OUT_DIR}/summary.json")
    print(f"# wall total: {wall_total/60:.1f} min  "
          f"successes: {summary['successes']} / {N_C}")
    return summary


if __name__ == "__main__":
    main()
