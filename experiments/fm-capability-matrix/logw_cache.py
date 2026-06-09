r"""Compute per-event log-likelihood ratios log w_c(x) for the cached event sets.

This is the morphing target the Intention ridge must regress as its VALUE V
(not the EMA self-embedding). It was dropped from substrate_cache.npz for speed
(event_log_likelihood_ratio is ~290ms/64 events); compute it once here and cache
separately so the value-ablation can use V = log w_c.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import substrate as sub
from modules.surrogate.oracle_events import event_log_likelihood_ratio

OUT = HERE / "output_matrix"
CACHE = OUT / "logw_cache.npz"


def compute(splits=("train", "test", "ood")):
    data = sub.load_cache()
    oracle = sub.make_oracle()
    blob = {}
    t0 = time.time()
    for sp in splits:
        X = data[f"{sp}_X1"]          # (S, N, 2)
        C = data[f"{sp}_c"]           # (S, 4)
        S, N, _ = X.shape
        lw = np.empty((S, N), dtype=np.float32)
        for i in range(S):
            lw[i] = event_log_likelihood_ratio(oracle, C[i], X[i])
            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                print(f"  [{sp}] {i+1}/{S}  ({dt:.0f}s)", flush=True)
        blob[f"{sp}_logw1"] = lw
        print(f"done {sp}: {lw.shape}  range [{lw.min():.2f}, {lw.max():.2f}]",
              flush=True)
    np.savez_compressed(CACHE, **blob)
    print(f"cached {CACHE}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    compute()
