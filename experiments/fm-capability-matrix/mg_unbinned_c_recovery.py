r"""UNBINNED real-data c-recovery: feed the raw per-event MadGraph sets directly
(no histogram), every arm consumes the same (N,2) event cloud, c is target-only.

This is the test the binned mg_c_recovery.py was not: no per-m-bin reduction, so
the per-event structure the ManifoldInformer is built to exploit is preserved.
Same arms as fair_c_recovery (event-only, learned value): Intention closed-form
ridge vs mean/attention/deepsets poolers. Held-out resolved-c R², 5 seeds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import probes
from fair_c_recovery import ARMS, train_eval
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis
import substrate as sub

MGDIR = HERE / "output_mg_cache"
OUT = HERE / "output_matrix"
N_SUB = 1500          # subsample each set to a fixed event count


def load(split, rng):
    evs, cs = [], []
    for f in sorted(MGDIR.glob(f"{split}_*.npz")):
        z = np.load(f); ev = z["events"]
        if ev.shape[0] < N_SUB:
            continue
        idx = rng.choice(ev.shape[0], N_SUB, replace=False)
        evs.append(ev[idx]); cs.append(z["c"])
    return np.array(evs, dtype=np.float32), np.array(cs, dtype=np.float32)


def main(seeds=(0, 1, 2, 3, 4), k_resolved=2):
    t0 = time.time()
    rng = np.random.default_rng(0)
    Xtr, Ctr = load("train", rng)
    Xte, Cte = load("test", rng)
    print(f"UNBINNED MG: train={len(Xtr)} test={len(Xte)} events/set={N_SUB} "
          f"(each arm gets raw (N,2) events; c target-only)", flush=True)
    if len(Xtr) < 40 or len(Xte) < 15:
        print("not enough MG scenarios"); return None

    oracle = sub.make_oracle(); r = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, r.uniform(-0.6, 0.6, (256, 4)),
                           r.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F); Vk = V[:, :k_resolved]
    Xtr_t = torch.tensor(Xtr); Xte_t = torch.tensor(Xte)
    Ytr = torch.tensor(Ctr @ Vk, dtype=torch.float32); Yte = Cte @ Vk

    res = {a: {"resolved": [], "per_coord": []} for a in ARMS}
    for s in seeds:
        for a, Arm in ARMS.items():
            pred, _ = train_eval(Arm, Xtr_t, Ytr, Xte_t,
                                 torch.tensor(Yte, dtype=torch.float32), s)
            res[a]["resolved"].append(probes.r2_score(Yte, pred))
            pred4, _ = train_eval(Arm, Xtr_t, torch.tensor(Ctr, dtype=torch.float32),
                                  Xte_t, torch.tensor(Cte, dtype=torch.float32), s)
            res[a]["per_coord"].append([probes.r2_score(Cte[:, j:j+1], pred4[:, j:j+1])
                                        for j in range(4)])
        print(f"seed {s} done", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["resolved"]); pc = np.array(res[a]["per_coord"])
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)]}
    out = {"seeds": list(seeds), "n_train": len(Xtr), "n_test": len(Xte),
           "events_per_set": N_SUB, "source": "MadGraph NLO, UNBINNED per-event",
           "summary": summary, "wall_seconds": time.time() - t0}
    (OUT / "mg_unbinned_c_recovery.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== UNBINNED real-data (MadGraph) c-recovery, {len(seeds)} seeds, "
          f"train={len(Xtr)} test={len(Xte)} ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        sm = summary[a]
        print(f"  {a:26s} resolved c̃ R²={sm['resolved_r2_mean']:.3f}±{sm['resolved_r2_std']:.3f}"
              f"  per-coord={np.round(sm['per_coord_r2_mean'],2)}")
    return out


if __name__ == "__main__":
    main()
