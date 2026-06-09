r"""Real-data (MadGraph NLO) c-recovery: Intention vs aggregator readouts, on
observables MEASURED from the events. Leak-free; c is target-only.

Each scenario is a MadGraph event set at a Wilson point. From the events we
MEASURE, per m-bin (quantile bins from the pooled SM sample, so the falling
spectrum is handled):
  μ_shape(m_bin) = (scenario count fraction) / (SM count fraction)   — shape ratio
  A_FB(m_bin)    = (N_forward - N_backward) / N_total                — from cosθ* sign
Both are functions of the events + the SM event bank — measurable without c.
Every arm gets the identical (m_bin, [μ_shape, A_FB]) context; representation
width matched; c is the held-out probe target. Same arms as the analytic test.
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
from realistic_disclosure import IntentionMO, PoolMO, train
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis
import substrate as sub

MGDIR = HERE / "output_mg_cache"
OUT = HERE / "output_matrix"
N_BINS = 8


def load(split):
    files = sorted(MGDIR.glob(f"{split}_*.npz"))
    evs, cs = [], []
    for f in files:
        z = np.load(f)
        if z["events"].shape[0] >= 200:        # need enough events to measure
            evs.append(z["events"]); cs.append(z["c"])
    return evs, np.array(cs) if cs else np.zeros((0, 4))


def measure(evs, edges, sm_frac):
    """Per scenario: μ_shape and A_FB on the m-bins. (n, N_BINS, 2)."""
    out = np.zeros((len(evs), N_BINS, 2), dtype=np.float32)
    for s, ev in enumerate(evs):
        m = np.exp(ev[:, 0]); u = ev[:, 1]
        idx = np.clip(np.digitize(m, edges[1:-1]), 0, N_BINS - 1)
        for b in range(N_BINS):
            sel = idx == b
            n = int(sel.sum())
            frac = n / len(m)
            mu_shape = frac / max(sm_frac[b], 1e-6)
            if n > 0:
                afb = (np.sign(u[sel]) > 0).mean() * 2 - 1   # (N_F-N_B)/N in [-1,1]
            else:
                afb = 0.0
            out[s, b] = [mu_shape, afb]
    return out


def main(seeds=(0, 1, 2, 3, 4), k_resolved=2):
    t0 = time.time()
    ev_tr, C_tr = load("train")
    ev_te, C_te = load("test")
    ev_sm, _ = load("sm")
    print(f"loaded MG: train={len(ev_tr)} test={len(ev_te)} sm={len(ev_sm)}", flush=True)
    if len(ev_tr) < 40 or len(ev_te) < 15 or len(ev_sm) < 3:
        print("not enough MG scenarios yet; rerun when generation has progressed.")
        return None

    # quantile m-bin edges from pooled SM events; SM bin fractions = baseline
    m_sm = np.concatenate([np.exp(e[:, 0]) for e in ev_sm])
    edges = np.quantile(m_sm, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = edges[0] - 1e-6, edges[-1] + 1e-6
    sm_idx = np.clip(np.digitize(m_sm, edges[1:-1]), 0, N_BINS - 1)
    sm_frac = np.array([(sm_idx == b).mean() for b in range(N_BINS)])
    centers = 0.5 * (edges[:-1] + edges[1:])

    Ytr = measure(ev_tr, edges, sm_frac)
    Yte = measure(ev_te, edges, sm_frac)

    oracle = sub.make_oracle(); rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-0.6, 0.6, (256, 4)),
                           rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F); Vk = V[:, :k_resolved]

    Mc = torch.tensor(np.tile(centers[:N_BINS - 2], (len(ev_tr), 1)), dtype=torch.float32)
    Mq = torch.tensor(np.tile(centers[N_BINS - 2:], (len(ev_tr), 1)), dtype=torch.float32)
    Yc = torch.tensor(Ytr[:, :N_BINS - 2], dtype=torch.float32)
    Yq = torch.tensor(Ytr[:, N_BINS - 2:], dtype=torch.float32)
    McTe = torch.tensor(np.tile(centers[:N_BINS - 2], (len(ev_te), 1)), dtype=torch.float32)
    YcTe = torch.tensor(Yte[:, :N_BINS - 2], dtype=torch.float32)

    arms = {"Intention_ridge": lambda: IntentionMO(),
            "MeanPool": lambda: PoolMO("mean"),
            "Attention": lambda: PoolMO("attn")}
    res = {a: {"resolved": [], "per_coord": []} for a in arms}
    for s in seeds:
        for a, fac in arms.items():
            m = fac(); train(m, Mc, Yc, Mq, Yq, s)
            Ztr = m.rep(Mc, Yc); Zte = m.rep(McTe, YcTe)
            rr = probes.held_out_probe(Ztr, C_tr @ Vk, Zte, C_te @ Vk, seed=s).r2
            p4 = probes.held_out_probe(Ztr, C_tr, Zte, C_te, seed=s)
            res[a]["resolved"].append(rr)
            res[a]["per_coord"].append(p4.r2_per_target)
        print(f"seed {s} done", flush=True)

    summary = {}
    for a in arms:
        rr = np.array(res[a]["resolved"]); pc = np.array(res[a]["per_coord"])
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)]}
    out = {"seeds": list(seeds), "n_train": len(ev_tr), "n_test": len(ev_te),
           "n_sm": len(ev_sm), "n_events_per_scn": int(np.median([len(e) for e in ev_tr])),
           "n_bins": N_BINS, "source": "MadGraph NLO events, measured μ_shape ⊕ A_FB",
           "fisher_eigenvalues": [float(x) for x in Dvals], "summary": summary,
           "wall_seconds": time.time() - t0}
    (OUT / "mg_c_recovery.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== REAL-DATA (MadGraph) c-recovery, n_seeds={len(seeds)}, "
          f"train={len(ev_tr)} test={len(ev_te)} ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        sm = summary[a]
        print(f"  {a:16s} resolved c̃ R²={sm['resolved_r2_mean']:.3f}±{sm['resolved_r2_std']:.3f}"
              f"  per-coord={np.round(sm['per_coord_r2_mean'],2)}")
    return out


if __name__ == "__main__":
    main()
