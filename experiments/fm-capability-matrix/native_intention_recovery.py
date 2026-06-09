r"""Native-interface c-recovery: the value-channel diagnosis, controlled.

The audited fair_c_recovery (events-only, leak-free) found the Intention ridge
WORST (0.286) when its value channel is a free learned scalar g_θ(x). The
analytical reading (Garnelo-Czarnecki: the least-squares/Intention module is the
exact in-context regressor, of which linear attention is the K^T K = I special
case) says the ridge readout is only the right tool when its VALUE space carries
the physics — i.e. when the context->value map linearises the target. A free
learned scalar does not; the binned observable in realistic_disclosure does, and
there the Intention wins (0.862).

This script changes EXACTLY ONE THING relative to fair_c_recovery and holds
everything else identical (same raw events, same readout MLP, same training,
same probe, same resolved subspace, leak-free, c is target-only, UNBINNED):

  the Intention's per-event VALUE becomes the c-free physical angular basis
      V(x) = [1, u, u^2],   u = cos θ*_CS
  which is exactly the Drell-Yan angular law's moment basis
      dσ/du ∝ (1+u^2) S(c,m) + 2u D(c,m),   A_FB = (3/4) D/S,
  and the KEY is the learned mass embedding ψ_θ(log m). The ridge then solves,
  in closed form per event-set, the *conditional angular-moment regression*
      E[(1,u,u^2) | log m]
  whose mass-dependence IS the morphing coordinate. No binning: every event is
  its own sample. Nothing here depends on c.

Arms (all see only raw (log m, u); c is target-only):
  Intention_physV     : K = ψ(log m), V = [1,u,u^2]   (the native construction)
  Intention_learnedV  : K = ψ(x),     V = g_θ(x)      (the fair_c_recovery loser; control)
  MeanPool / Attention: pooled readouts of φ(log m, u) (the transformer controls)

Prediction under the analysis: physV >> learnedV, and physV at least matches the
poolers on the resolved subspace, with tighter seed variance (the lower-variance
MAP/least-squares estimator). Report numbers; assert nothing else.
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
STEPS = 800
BATCH = 32
ALPHA = 1e-3
torch.set_num_threads(8)


def mlp(sizes, act=nn.GELU, last=False):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last:
            L.append(act())
    return nn.Sequential(*L)


def phys_moments(X):
    """c-free per-event angular moment basis [1, u, u^2]; u = cos θ*. (B,N,3)."""
    u = X[..., 1:2]
    return torch.cat([torch.ones_like(u), u, u * u], dim=-1)


class IntentionPhysV(nn.Module):
    """K = ψ(log m) learned, V = [1,u,u^2] c-free. w = (KᵀK+αI)⁻¹ KᵀV is the
    conditional angular-moment regression on the mass embedding (the morphing
    coordinate). Readout matches the poolers."""
    def __init__(self, n_out, n_v=3, alpha=ALPHA):
        super().__init__()
        self.psi = mlp([1, 64, 64, D])           # key from log m only
        self.n_v = n_v
        self.readout = mlp([D * n_v, 64, n_out])
        self.alpha = alpha

    def forward(self, X):
        m = X[..., 0:1]
        K = self.psi(m)                          # (B,N,D)
        V = phys_moments(X)                       # (B,N,3)  c-free
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ V)    # (B,D,3)
        return self.readout(w.reshape(w.shape[0], -1))


class IntentionLearnedV(nn.Module):
    """The fair_c_recovery loser, verbatim: keys AND value learned from events."""
    def __init__(self, n_out, alpha=ALPHA):
        super().__init__()
        self.phi = mlp([2, 64, 64, D])
        self.gval = mlp([2, 64, 1])
        self.readout = mlp([D, 64, n_out])
        self.alpha = alpha

    def forward(self, X):
        K = self.phi(X)
        v = self.gval(X)
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ v).squeeze(-1)
        return self.readout(w)


class MeanPool(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.phi = mlp([2, 64, D])
        self.readout = mlp([2 * D, 64, n_out])

    def forward(self, X):
        h = self.phi(X)
        return self.readout(torch.cat([h.mean(1), h.std(1)], -1))


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


ARMS = {
    "Intention_physV": IntentionPhysV,
    "Intention_learnedV": IntentionLearnedV,
    "MeanPool": MeanPool,
    "Attention": AttnPool,
}


def train_eval(Arm, Xtr, Ytr, Xte, seed):
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


def main(seeds=(0, 1, 2), k_resolved=2):
    t0 = time.time()
    data = sub.load_cache()
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    assert Xtr.shape[-1] == 2, "inputs must be raw 2-D events only (unbinned)"
    Ctr, Cte = data["train_c"], data["test_c"]

    oracle = sub.make_oracle(); rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-0.6, 0.6, (256, 4)),
                           rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F); Vk = V[:, :k_resolved]
    Ytr = torch.tensor(Ctr @ Vk, dtype=torch.float32)
    Yte_res = Cte @ Vk
    Ytr4 = torch.tensor(Ctr, dtype=torch.float32)
    print(f"Fisher eigvals {np.round(Dvals,4)}; resolved top-{k_resolved}", flush=True)
    print(f"INPUTS: raw unbinned events {tuple(Xtr.shape)}; c target-only; "
          f"Intention_physV value = [1,u,u^2] (c-free angular law). No log w_c.",
          flush=True)

    res = {a: {"resolved": [], "per_coord": []} for a in ARMS}
    for s in seeds:
        for a, Arm in ARMS.items():
            pred_res, npar = train_eval(Arm, Xtr, Ytr, Xte, s)
            r2_res = probes.r2_score(Yte_res, pred_res)
            pred4, _ = train_eval(Arm, Xtr, Ytr4, Xte, s)
            per = [probes.r2_score(Cte[:, j:j + 1], pred4[:, j:j + 1]) for j in range(4)]
            res[a]["resolved"].append(r2_res)
            res[a]["per_coord"].append(per)
            res[a]["n_params"] = npar
        print(f"seed {s} done", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["resolved"]); pc = np.array(res[a]["per_coord"])
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)],
                      "n_params": int(res[a]["n_params"])}
    out = {"seeds": list(seeds), "k_resolved": k_resolved,
           "inputs": "raw UNBINNED events (B,N,2); c target-only; "
                     "Intention_physV value=[1,u,u^2] c-free; leak-free",
           "fisher_eigenvalues": [float(x) for x in Dvals], "summary": summary,
           "wall_seconds": time.time() - t0}
    (OUT / "native_intention_recovery.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== NATIVE c-recovery (unbinned events, leak-free, n_seeds={len(seeds)}) ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        sm = summary[a]
        print(f"  {a:20s} resolved c̃ R²={sm['resolved_r2_mean']:.3f}±{sm['resolved_r2_std']:.3f}"
              f"  per-coord={np.round(sm['per_coord_r2_mean'],2)}  params={sm['n_params']}")
    return out


if __name__ == "__main__":
    main()
