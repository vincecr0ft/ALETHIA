r"""Matched-input c-recovery: every arm recovers c from EVENTS ONLY.

Audit (2026-06-08): the value-ablation / morphing-gate fed the Intention arm
V = log w_c(x), the per-event log-likelihood ratio evaluated at the TRUE c.
That is the sufficient statistic for c — the answer — and no other arm got it.
So any "Intention recovers c best" from those runs is favorable-by-construction.

This test removes the leak. Every architecture is the SAME function
    events X (B, N, 2)  ->  ĉ̃   (resolved Wilson coords)
trained supervised on (events, c) pairs, evaluated held-out. c enters ONLY as
the regression target in the loss; it never enters any forward pass, and no arm
is handed log w_c, μ(c,·), or any c-derived feature. The Intention's value
channel is LEARNED from events (g_θ(x)), exactly like everyone else's encoder.

Reported: held-out R² on the Fisher-resolved subspace and per-Wilson-coordinate,
mean ± std over seeds. No ranking is asserted beyond what the matched numbers
show.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import substrate as sub
import probes
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

OUT = HERE / "output_matrix"
D = 16
STEPS = 1500
BATCH = 32


def mlp(sizes, act=nn.GELU, last=False):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last:
            L.append(act())
    return nn.Sequential(*L)


# Every arm: forward(X) -> ĉ̃ ; X is (B, N, 2) events ONLY.

class IntentionRidgeLearnedV(nn.Module):
    """Closed-form ridge with keys AND values LEARNED from events (no leak).
    w_θ = (φᵀφ + αI)⁻¹ φᵀ g, where φ(x), g(x) are both MLPs of the raw event."""
    def __init__(self, n_out, alpha=1e-3):
        super().__init__()
        self.phi = mlp([2, 64, 64, D])
        self.gval = mlp([2, 64, 1])           # learned scalar value per event
        self.readout = mlp([D, 64, n_out])    # SAME MLP readout as the poolers
        self.alpha = alpha

    def forward(self, X):
        K = self.phi(X)                        # (B,N,D)
        v = self.gval(X)                       # (B,N,1)
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ v).squeeze(-1)  # (B,D)
        return self.readout(w)


class MeanPool(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.phi = mlp([2, 64, D])
        self.readout = mlp([D, 64, n_out])

    def forward(self, X):
        return self.readout(self.phi(X).mean(1))


class AttnPool(nn.Module):
    def __init__(self, n_out, heads=4):
        super().__init__()
        self.phi = mlp([2, 64, D])
        self.seed = nn.Parameter(torch.randn(1, 1, D) * 0.1)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.readout = mlp([D, 64, n_out])

    def forward(self, X):
        h = self.phi(X); B = h.shape[0]
        z, _ = self.attn(self.seed.expand(B, -1, -1), h, h)
        return self.readout(z.squeeze(1))


class DeepSetsStd(nn.Module):
    """Mean+std pooling DeepSets (the substrate's default encoder)."""
    def __init__(self, n_out):
        super().__init__()
        self.phi = mlp([2, 64, D])
        self.readout = mlp([2 * D, 64, n_out])

    def forward(self, X):
        h = self.phi(X)
        return self.readout(torch.cat([h.mean(1), h.std(1)], -1))


ARMS = {
    "Intention_ridge_learnedV": IntentionRidgeLearnedV,
    "MeanPool": MeanPool,
    "Attention": AttnPool,
    "DeepSets_meanstd": DeepSetsStd,
}


def train_eval(Arm, Xtr, Ytr, Xte, Yte, seed):
    torch.manual_seed(seed)
    model = Arm(Ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = Xtr.shape[0]; rng = np.random.default_rng(seed)
    model.train()
    for _ in range(STEPS):
        idx = rng.choice(n, BATCH, replace=False)
        loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy()
    return pred, sum(p.numel() for p in model.parameters())


def main(seeds=(0, 1, 2, 3, 4), k_resolved=2):
    t0 = time.time()
    data = sub.load_cache()
    # INPUTS: events only. Assert nothing c-derived is in the feature tensors.
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)   # (450,160,2) events
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    assert Xtr.shape[-1] == 2, "inputs must be raw 2-D events only"
    Ctr, Cte = data["train_c"], data["test_c"]                  # labels only

    oracle = sub.make_oracle(); rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-0.6, 0.6, (256, 4)),
                           rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F); Vk = V[:, :k_resolved]
    Ytr = torch.tensor(Ctr @ Vk, dtype=torch.float32)
    Yte_res = Cte @ Vk
    # also recover full 4-D c for per-coord reporting
    Ytr4 = torch.tensor(Ctr, dtype=torch.float32)
    print(f"Fisher eigvals {np.round(Dvals,4)}; resolved top-{k_resolved}", flush=True)
    print(f"INPUTS: every arm gets X events {tuple(Xtr.shape)} only; "
          f"target = resolved c̃ (held-out). No arm gets log w_c / μ / c.", flush=True)

    res = {a: {"resolved": [], "per_coord": []} for a in ARMS}
    for s in seeds:
        for a, Arm in ARMS.items():
            pred_res, npar = train_eval(Arm, Xtr, Ytr, Xte,
                                        torch.tensor(Yte_res, dtype=torch.float32), s)
            r2_res = probes.r2_score(Yte_res, pred_res)
            pred4, _ = train_eval(Arm, Xtr, Ytr4, Xte, torch.tensor(Cte, dtype=torch.float32), s)
            per = [probes.r2_score(Cte[:, j:j+1], pred4[:, j:j+1]) for j in range(4)]
            res[a]["resolved"].append(r2_res)
            res[a]["per_coord"].append(per)
            res[a]["n_params"] = npar
        print(f"seed {s} done", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["resolved"])
        pc = np.array(res[a]["per_coord"])  # (seeds,4)
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)],
                      "n_params": int(res[a]["n_params"])}
    out = {"seeds": list(seeds), "k_resolved": k_resolved,
           "inputs": "raw events (B,N,2) only; c is target-only, never a feature",
           "fisher_eigenvalues": [float(x) for x in Dvals], "summary": summary,
           "wall_seconds": time.time() - t0}
    (OUT / "fair_c_recovery.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== FAIR c-recovery (events-only, n_seeds={len(seeds)}) ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        s = summary[a]
        print(f"  {a:26s} resolved c̃ R²={s['resolved_r2_mean']:.3f}±{s['resolved_r2_std']:.3f}"
              f"  per-coord={np.round(s['per_coord_r2_mean'],2)}  params={s['n_params']}")
    return out


if __name__ == "__main__":
    main()
