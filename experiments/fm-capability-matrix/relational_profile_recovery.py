r"""Relational task: in-context conditional-profile recovery (the Intention's home).

Contrast with the marginal task (stacked_informer_recovery): there the target was
a mass-MARGINAL effect (four-fermion tail), where softmax attention's
data-dependent reweighting — a soft density estimator — held a ~0.02 edge. Here
the target is a CONDITIONAL MAP: given a raw event set at a working point, predict
the forward-backward asymmetry PROFILE A_FB(m) at query masses. A_FB(m) is the
oracle observable (leak-free: c is never an input; the profile is what an analysis
measures). Recovering it is in-context regression of E[angular | m] — exactly what
the closed-form ridge solves in one shot, and what self-/cross-attention must
learn through its weights.

Same raw (log m, u) inputs, matched width/heads, leak-free. Arms:
  InformerICL   : K=ψ(log m_ctx), V=g(u_ctx); w=(KᵀK+αI)⁻¹KᵀV; Â(m_q)=ψ(log m_q)·w
                  (the in-context Intention, single closed-form solve)
  TransformerICL: query tokens ψ(log m_q) CROSS-ATTEND to context tokens enc(m,u)
  MeanPool      : pool context, concat query mass, readout

Prediction (relational ⇒ whitening helps): InformerICL > TransformerICL, the
mirror image of the marginal task. Same module that LOST the marginal task should
WIN the conditional one. Report numbers; assert nothing else.
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

OUT = HERE / "output_matrix"
D = 16
VV = 4
STEPS = 1000
BATCH = 32
ALPHA = 1e-3
N_Q = 16
torch.set_num_threads(8)


def mlp(sizes, act=nn.GELU, last=False):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last:
            L.append(act())
    return nn.Sequential(*L)


class InformerICL(nn.Module):
    """In-context Intention: closed-form ridge regression of the angular value on
    the mass embedding, queried at held-out masses. The ManifoldInformer's form."""
    def __init__(self, alpha=ALPHA):
        super().__init__()
        self.psi = mlp([1, 64, 64, D])
        self.gval = mlp([1, 32, VV])
        self.readout = mlp([VV, 32, 1])
        self.alpha = alpha

    def forward(self, X, logmq):                  # X (B,N,2), logmq (Q,)
        B, N, _ = X.shape; Q = logmq.shape[0]
        K = self.psi(X[..., 0:1])                  # (B,N,D)
        V = self.gval(X[..., 1:2])                 # (B,N,VV)
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ V)         # (B,D,VV)
        Kq = self.psi(logmq.view(1, Q, 1).expand(B, Q, 1))         # (B,Q,D)
        Zq = Kq @ w                                                # (B,Q,VV)
        return self.readout(Zq).squeeze(-1)                        # (B,Q)


class TransformerICL(nn.Module):
    """Query masses cross-attend to context events (the matched control)."""
    def __init__(self, heads=4):
        super().__init__()
        self.enc = mlp([2, 64, D])
        self.qenc = mlp([1, 64, D])
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.readout = mlp([D, 32, 1])

    def forward(self, X, logmq):
        B, N, _ = X.shape; Q = logmq.shape[0]
        h = self.enc(X)                                            # (B,N,D)
        q = self.qenc(logmq.view(1, Q, 1).expand(B, Q, 1))         # (B,Q,D)
        z, _ = self.attn(q, h, h)                                  # (B,Q,D)
        return self.readout(z).squeeze(-1)


class MeanPoolICL(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = mlp([2, 64, D])
        self.readout = mlp([D + 1, 64, 1])

    def forward(self, X, logmq):
        B, N, _ = X.shape; Q = logmq.shape[0]
        z = self.enc(X).mean(1)                                    # (B,D)
        inp = torch.cat([z.view(B, 1, D).expand(B, Q, D),
                         logmq.view(1, Q, 1).expand(B, Q, 1)], -1)
        return self.readout(inp).squeeze(-1)


ARMS = {
    "InformerICL": InformerICL,
    "TransformerICL": TransformerICL,
    "MeanPool": MeanPoolICL,
}


def afb_targets(oracle, C, m_q):
    T = np.empty((len(C), len(m_q)), dtype=np.float32)
    for i in range(len(C)):
        T[i] = oracle.truth_afb(np.tile(C[i], (len(m_q), 1)), m_q)
    return T


def train_eval(Arm, Xtr, Ytr, logmq, Xte, seed):
    torch.manual_seed(seed)
    model = Arm()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = Xtr.shape[0]; rng = np.random.default_rng(seed)
    model.train()
    for _ in range(STEPS):
        idx = rng.choice(n, BATCH, replace=False)
        loss = ((model(Xtr[idx], logmq) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte, logmq).numpy()
    return pred, sum(p.numel() for p in model.parameters())


def main(seeds=(0, 1, 2, 3, 4, 5, 6, 7)):
    t0 = time.time()
    data = sub.load_cache()
    oracle = sub.make_oracle()
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    Ctr, Cte = data["train_c"], data["test_c"]

    m_q = np.linspace(0.35, 2.2, N_Q).astype(np.float32)           # TeV
    logmq = torch.tensor(np.log(m_q), dtype=torch.float32)         # ψ takes log m
    Ytr = torch.tensor(afb_targets(oracle, Ctr, m_q), dtype=torch.float32)
    Yte = afb_targets(oracle, Cte, m_q)

    print(f"TASK: predict A_FB(m) profile at {N_Q} query masses from raw events; "
          f"conditional map; c target-only (never an input). Leak-free.", flush=True)
    print(f"A_FB target spread: std={Yte.std():.4f} over {Yte.shape} (B,Q)", flush=True)

    res = {a: {"r2": []} for a in ARMS}
    for s in seeds:
        for a, Arm in ARMS.items():
            pred, npar = train_eval(Arm, Xtr, Ytr, logmq, Xte, s)
            res[a]["r2"].append(probes.r2_score(Yte, pred))
            res[a]["n_params"] = npar
        print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["r2"])
        summary[a] = {"profile_r2_mean": float(rr.mean()), "profile_r2_std": float(rr.std()),
                      "n_params": int(res[a]["n_params"])}
    out = {"seeds": list(seeds), "n_query_masses": N_Q,
           "task": "in-context A_FB(m) conditional-profile recovery from raw events; "
                   "leak-free (c target-only); relational/conditional structure",
           "summary": summary, "wall_seconds": time.time() - t0}
    (OUT / "relational_profile_recovery.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== RELATIONAL A_FB(m)-profile recovery (in-context, leak-free, "
          f"n_seeds={len(seeds)}) ===")
    for a in sorted(summary, key=lambda n: summary[n]["profile_r2_mean"], reverse=True):
        sm = summary[a]
        print(f"  {a:16s} profile R²={sm['profile_r2_mean']:.3f}±{sm['profile_r2_std']:.3f}"
              f"  params={sm['n_params']}")
    return out


if __name__ == "__main__":
    main()
