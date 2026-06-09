r"""Generate MadGraph NLO event sets across Wilson points for the real-data
c-recovery test. One MG run per scenario (~35-50s), saved incrementally so a
partial run is usable. Window [0.3, 2.3] TeV (above the Z peak, the high-mass
Drell-Yan tail where SMEFT lives).
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

from modules.surrogate.oracle_madgraph import MadGraphSMEFTOracle

OUTDIR = HERE / "output_mg_cache"
OUTDIR.mkdir(exist_ok=True)

BOX_HALF = 0.6
N_TRAIN = 150
N_TEST = 50
N_SM = 25
NEVENTS = 2000
M_CENTER, M_WINDOW = 1.3, 2.0          # -> [0.3, 2.3] TeV


def main():
    rng = np.random.default_rng(2026)
    C_train = rng.uniform(-BOX_HALF, BOX_HALF, (N_TRAIN, 4))
    C_test = rng.uniform(-BOX_HALF, BOX_HALF, (N_TEST, 4))
    C_sm = np.zeros((N_SM, 4))
    # SM first (the measurement baseline), then train/test interleaved so a
    # partial run is already usable for a preview.
    jobs = [("sm", i, C_sm[i]) for i in range(N_SM)]
    tt = [("train", i, C_train[i]) for i in range(N_TRAIN)]
    te = [("test", i, C_test[i]) for i in range(N_TEST)]
    for k in range(max(N_TRAIN, N_TEST)):
        if k < N_TRAIN:
            jobs.append(tt[k])
        if k % 3 == 0 and k // 3 < N_TEST:
            jobs.append(te[k // 3])

    orc = MadGraphSMEFTOracle(nevents=NEVENTS, m_window_tev=M_WINDOW, verbose=False)
    t0 = time.time(); done = 0
    for job in jobs:
        if job == "sep":
            continue
        split, i, c = job
        path = OUTDIR / f"{split}_{i:03d}.npz"
        if path.exists():
            done += 1
            continue
        try:
            ev = orc.sample_events_mg(c, M_CENTER, M_WINDOW, nevents=NEVENTS)
        except Exception as e:
            print(f"  ! {split}_{i} failed: {e}", flush=True)
            continue
        np.savez_compressed(path, events=ev.astype(np.float32), c=c.astype(np.float32))
        done += 1
        dt = time.time() - t0
        print(f"  [{split}_{i:03d}] n_ev={ev.shape[0]} ({dt/done:.0f}s/scn, "
              f"{done} done, {dt/60:.0f}min elapsed)", flush=True)
    print(f"DONE: {done} scenarios, {(time.time()-t0)/60:.0f} min total", flush=True)


if __name__ == "__main__":
    main()
