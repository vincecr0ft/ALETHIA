"""One representative closed-loop chain per acquisition, with Phoenix tracing
ON, so the alethia-al-studies project at localhost:6006 shows a viewable
dashboard.

Run after the parallel sweep — this is the human-facing artifact, not the
science. ~5 chains × ~50 cycles, ~minutes wall.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHAINS = [
    # (acq, pool, K, seed) — one per acquisition, on the production pool so
    # the dashboard reflects the same setup the failed sweep used.
    ("random",        "P0", 12, 2026),
    ("leverage",      "P0", 12, 2026),
    ("epig",          "P0", 12, 2026),
    ("param_epig_d",  "P0", 12, 2026),
    ("param_epig_a",  "P0", 12, 2026),
    # And one on the heterogeneous tail-spiked pool with larger K to show
    # the data-vs-prior regime in the dashboard.
    ("param_epig_a",  "P2", 24, 2026),
]


def run_one(acq: str, pool: str, K: int, seed: int) -> None:
    e = os.environ.copy()
    e["ACQ"] = acq; e["POOL"] = pool; e["K_CTX"] = str(K); e["SEED"] = str(seed)
    e["N_CYCLES"] = os.environ.get("N_CYCLES", "50")
    e["PHOENIX_TRACING"] = "1"
    # Keep BLAS threads sane; we run sequentially here.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "TORCH_NUM_THREADS"):
        e[k] = e.get(k, "4")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(HERE / "stage_c_closed_loop.py")],
        env=e, check=True,
    )
    print(f"  {acq:14s} {pool:>3s} K={K:2d} seed={seed}: {time.time()-t0:.1f}s")


def main():
    print(f"# Tracing demo into Phoenix project alethia-al-studies "
          f"({len(CHAINS)} chains)")
    for chain in CHAINS:
        run_one(*chain)
    print("# done — visit http://localhost:6006/projects and select "
          "alethia-al-studies to view the traces.")


if __name__ == "__main__":
    main()
