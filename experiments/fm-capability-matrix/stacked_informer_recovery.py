r"""Stacked Informer vs matched transformer, raw unbinned events (leak-free).

Correction (user, 2026-06-09): the single-block `physV` test was *pure Intention*
— one closed-form ridge solve, hence strictly linear in the value and computing
one conditional E[V|K], blind to the key marginal. That is NOT the Informer. The
Informer is the STACK: self-Intention layers (the whitened-linear-attention
drop-in of Garnelo-Czarnecki) with nonlinear FFNs between them, exactly the
closed-form analogue of a transformer's stacked self-attention blocks. The stack
is nonlinear and set-aware (each layer's w depends on the whole set through KᵀK),
so it can capture the mass-marginal signal a single solve cannot.

Containment at the architecture level: a self-Intention layer
    out_i = q_iᵀ (KᵀK + αI)⁻¹ KᵀV ,   Q=HW_q, K=HW_k, V=HW_v
contains self-attention as the KᵀK=I (whitened-keys) special case, at equal
complexity. So a depth-L Informer ⊇ a depth-L transformer.

Test: the EXACT raw-event task where the single block failed (fair_c_recovery /
native_intention_recovery: resolved c̃ from unbinned events, c target-only,
leak-free). Arms, all depth-L=2, width D, matched FFN, mean-pool + same readout:
    StackedInformer    : self-Intention blocks (whitened linear attention)
    StackedTransformer : self-attention blocks (the thing it must match/beat)
    MeanPool           : single-layer pooled baseline (the marginal-only control)
    SingleIntention    : one ridge block, rep=w  (the physV-style failure, control)

Prediction (containment + stacking): StackedInformer >= StackedTransformer, both
recover the four-fermion clq3 the single block lost, both >> the single block.
Report numbers; assert nothing else.
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
L_DEPTH = 2
FF = 32
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


class SelfIntention(nn.Module):
    """Multi-head whitened-linear-attention sublayer (Garnelo drop-in for
    self-attention): per head, out_i = q_iᵀ (K_hᵀK_h+αI)⁻¹ K_hᵀV_h; heads
    concatenated. Set->set, permutation-equivariant. Matches the transformer
    control's head count for a fair comparison."""
    def __init__(self, d=D, heads=4, alpha=ALPHA):
        super().__init__()
        assert d % heads == 0
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d)
        self.d = d; self.h = heads; self.dh = d // heads; self.alpha = alpha

    def _split(self, T):                          # (B,N,d) -> (B,h,N,dh)
        B, N, _ = T.shape
        return T.view(B, N, self.h, self.dh).transpose(1, 2)

    def forward(self, H):                         # H (B,N,d)
        B, N, _ = H.shape
        Q, K, V = self._split(self.Wq(H)), self._split(self.Wk(H)), self._split(self.Wv(H))
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(self.dh)   # (B,h,dh,dh)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ V)             # (B,h,dh,dh)
        out = (Q @ w).transpose(1, 2).reshape(B, N, self.d)            # (B,N,d)
        return self.proj(out)


class SelfAttention(nn.Module):
    """Matched self-attention sublayer (the transformer control)."""
    def __init__(self, d=D, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, H):
        z, _ = self.attn(H, H, H)
        return z


class StackedSet(nn.Module):
    """Depth-L set encoder: [sublayer + FFN, both residual] × L, mean-pool, readout.
    sublayer_cls in {SelfIntention, SelfAttention} -> Informer vs transformer."""
    def __init__(self, n_out, sublayer_cls, depth=L_DEPTH, d=D):
        super().__init__()
        self.embed = mlp([2, FF, d])
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "mix": sublayer_cls(d),
                "n1": nn.LayerNorm(d),
                "ff": mlp([d, FF, d]),
                "n2": nn.LayerNorm(d),
            }))
        self.readout = mlp([d, FF, n_out])

    def forward(self, X):                         # X (B,N,2)
        H = self.embed(X)
        for b in self.blocks:
            H = b["n1"](H + b["mix"](H))
            H = b["n2"](H + b["ff"](H))
        return self.readout(H.mean(1))


class MeanPool(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.phi = mlp([2, FF, D]); self.readout = mlp([D, FF, n_out])

    def forward(self, X):
        return self.readout(self.phi(X).mean(1))


class SingleIntention(nn.Module):
    """One ridge block, rep=w (the physV-style single-solve control)."""
    def __init__(self, n_out, alpha=ALPHA):
        super().__init__()
        self.phi = mlp([2, FF, FF, D]); self.gval = mlp([2, FF, 1])
        self.readout = mlp([D, FF, n_out]); self.alpha = alpha

    def forward(self, X):
        K = self.phi(X); v = self.gval(X)
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ v).squeeze(-1)
        return self.readout(w)


ARMS = {
    "StackedInformer": lambda n: StackedSet(n, SelfIntention),
    "StackedTransformer": lambda n: StackedSet(n, SelfAttention),
    "MeanPool": lambda n: MeanPool(n),
    "SingleIntention": lambda n: SingleIntention(n),
}


def train_eval(make, Xtr, Ytr, Xte, seed):
    torch.manual_seed(seed)
    model = make(Ytr.shape[1])
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


def main(seeds=(0, 1, 2, 3, 4, 5, 6, 7), k_resolved=2):
    t0 = time.time()
    data = sub.load_cache()
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    assert Xtr.shape[-1] == 2
    Ctr, Cte = data["train_c"], data["test_c"]

    oracle = sub.make_oracle(); rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-0.6, 0.6, (256, 4)),
                           rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F); Vk = V[:, :k_resolved]
    Ytr = torch.tensor(Ctr @ Vk, dtype=torch.float32)
    Yte_res = Cte @ Vk
    Ytr4 = torch.tensor(Ctr, dtype=torch.float32)
    print(f"Fisher eigvals {np.round(Dvals,4)}; resolved top-{k_resolved}; "
          f"depth L={L_DEPTH}", flush=True)
    print(f"INPUTS: raw UNBINNED events {tuple(Xtr.shape)}; c target-only; leak-free",
          flush=True)

    res = {a: {"resolved": [], "per_coord": []} for a in ARMS}
    for s in seeds:
        for a, make in ARMS.items():
            pred_res, npar = train_eval(make, Xtr, Ytr, Xte, s)
            r2_res = probes.r2_score(Yte_res, pred_res)
            pred4, _ = train_eval(make, Xtr, Ytr4, Xte, s)
            per = [probes.r2_score(Cte[:, j:j + 1], pred4[:, j:j + 1]) for j in range(4)]
            res[a]["resolved"].append(r2_res)
            res[a]["per_coord"].append(per)
            res[a]["n_params"] = npar
        print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["resolved"]); pc = np.array(res[a]["per_coord"])
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)],
                      "n_params": int(res[a]["n_params"])}
    out = {"seeds": list(seeds), "k_resolved": k_resolved, "depth": L_DEPTH,
           "inputs": "raw UNBINNED events (B,N,2); c target-only; leak-free; "
                     "stacked self-Intention vs matched stacked self-attention",
           "fisher_eigenvalues": [float(x) for x in Dvals], "summary": summary,
           "wall_seconds": time.time() - t0}
    (OUT / "stacked_informer_recovery.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== STACKED Informer vs transformer (unbinned events, leak-free, "
          f"L={L_DEPTH}, n_seeds={len(seeds)}) ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        sm = summary[a]
        print(f"  {a:20s} resolved c̃ R²={sm['resolved_r2_mean']:.3f}±{sm['resolved_r2_std']:.3f}"
              f"  per-coord={np.round(sm['per_coord_r2_mean'],2)}  params={sm['n_params']}")
    return out


if __name__ == "__main__":
    main()
