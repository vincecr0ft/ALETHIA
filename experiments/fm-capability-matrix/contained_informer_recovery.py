r"""Containment done correctly: the transformer must be a SUBSET of the Informer.

The screwup in stacked_informer_recovery: I compared whitened-linear-least-squares
("Informer") against SOFTMAX attention ("transformer"). Garnelo-Czarnecki proves
least-squares strictly generalises *linear* attention — softmax is a DIFFERENT
nonlinearity the bare Intention does not contain. So that was not a containment
pair, and softmax's data-dependent reweighting (a soft density estimator) is what
won the marginal task. To test the user's claim — "the transformer is a contained
subset of the Informer, it should not win" — the Informer block must actually
contain its transformer baseline.

Two containment pairs, both depth-2, multi-head, same width, same raw-event task
(resolved c̃ recovery, leak-free, c target-only):

  Garnelo pair (exact):
    LinearTransformer : out = Q (KᵀV)                          (linear attention)
    Informer_LS       : out = Q (KᵀK+αI)⁻¹ KᵀV                 ⊇ LinearTransformer
                        (whitening→I recovers linear attention)

  Constructed pair (to also contain softmax):
    SoftmaxTransformer: standard multi-head softmax attention
    Informer_Gated    : softmax-attention sublayer + g·least-squares sublayer,
                        g a learnable gate init 0 ⇒ at g=0 it IS the transformer,
                        so Informer_Gated ⊇ SoftmaxTransformer by construction.

Prediction (containment): Informer_LS ≥ LinearTransformer and Informer_Gated ≥
SoftmaxTransformer — neither transformer should win its pair. Report numbers and
the learned gate magnitudes; assert nothing else.
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
HEADS = 4
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


def _split(T, h):                                  # (B,N,d)->(B,h,N,dh)
    B, N, d = T.shape
    return T.view(B, N, h, d // h).transpose(1, 2)


def _merge(T):                                     # (B,h,N,dh)->(B,N,d)
    B, h, N, dh = T.shape
    return T.transpose(1, 2).reshape(B, N, h * dh)


class LinearAttn(nn.Module):
    """Multi-head linear attention: out_i = q_iᵀ (KᵀV). Garnelo's contained case."""
    def __init__(self, d=D, h=HEADS):
        super().__init__()
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.proj = nn.Linear(d, d); self.h = h; self.d = d

    def forward(self, H):
        Q, K, V = _split(self.Wq(H), self.h), _split(self.Wk(H), self.h), _split(self.Wv(H), self.h)
        N = H.shape[1]
        out = Q @ (K.transpose(-2, -1) @ V) / N                 # (B,h,N,dh)
        return self.proj(_merge(out))


class WhitenedLS(nn.Module):
    """Multi-head least-squares: out_i = q_iᵀ (KᵀK+αI)⁻¹ KᵀV. Superset of LinearAttn
    (whitening (KᵀK+αI)⁻¹ → I recovers linear attention up to scale)."""
    def __init__(self, d=D, h=HEADS, alpha=ALPHA):
        super().__init__()
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.proj = nn.Linear(d, d); self.h = h; self.d = d; self.dh = d // h; self.alpha = alpha

    def forward(self, H):
        Q, K, V = _split(self.Wq(H), self.h), _split(self.Wk(H), self.h), _split(self.Wv(H), self.h)
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(self.dh)
        w = torch.linalg.solve(A, K.transpose(-2, -1) @ V)      # (B,h,dh,dh)
        return self.proj(_merge(Q @ w))


class SoftmaxAttn(nn.Module):
    def __init__(self, d=D, h=HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)

    def forward(self, H):
        z, _ = self.attn(H, H, H); return z


class Block(nn.Module):
    """Encoder block. kind: 'linear'|'ls'|'softmax'|'gated'. 'gated' = softmax
    attention + a learnable-gate least-squares sublayer (init g=0 ⇒ pure transformer)."""
    def __init__(self, kind, d=D):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.mix = LinearAttn(d)
        elif kind == "ls":
            self.mix = WhitenedLS(d)
        elif kind == "softmax":
            self.mix = SoftmaxAttn(d)
        elif kind == "gated":
            self.mix = SoftmaxAttn(d)
            self.ls = WhitenedLS(d)
            self.g = nn.Parameter(torch.zeros(1))      # init 0 ⇒ exact transformer
        self.n1 = nn.LayerNorm(d); self.ff = mlp([d, FF, d]); self.n2 = nn.LayerNorm(d)

    def forward(self, H):
        if self.kind == "gated":
            H = self.n1(H + self.mix(H) + self.g * self.ls(H))
        else:
            H = self.n1(H + self.mix(H))
        return self.n2(H + self.ff(H))


class Stacked(nn.Module):
    def __init__(self, n_out, kind, depth=L_DEPTH, d=D):
        super().__init__()
        self.embed = mlp([2, FF, d])
        self.blocks = nn.ModuleList([Block(kind, d) for _ in range(depth)])
        self.readout = mlp([d, FF, n_out])

    def forward(self, X):
        H = self.embed(X)
        for b in self.blocks:
            H = b(H)
        return self.readout(H.mean(1))

    def gates(self):
        return [float(b.g.detach()) for b in self.blocks if b.kind == "gated"]


ARMS = {
    "LinearTransformer": "linear",
    "Informer_LS":       "ls",
    "SoftmaxTransformer": "softmax",
    "Informer_Gated":    "gated",
}


def train_eval(kind, Xtr, Ytr, Xte, seed):
    torch.manual_seed(seed)
    model = Stacked(Ytr.shape[1], kind)
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
    return pred, model.gates(), sum(p.numel() for p in model.parameters())


def main(seeds=(0, 1, 2, 3, 4), k_resolved=2):
    t0 = time.time()
    data = sub.load_cache()
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    Ctr, Cte = data["train_c"], data["test_c"]

    oracle = sub.make_oracle(); rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-0.6, 0.6, (256, 4)),
                           rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F); Vk = V[:, :k_resolved]
    Ytr = torch.tensor(Ctr @ Vk, dtype=torch.float32)
    Yte_res = Cte @ Vk
    Ytr4 = torch.tensor(Ctr, dtype=torch.float32)
    print(f"Fisher eigvals {np.round(Dvals,4)}; depth L={L_DEPTH}, heads={HEADS}", flush=True)
    print("CONTAINMENT pairs: Informer_LS ⊇ LinearTransformer (Garnelo); "
          "Informer_Gated ⊇ SoftmaxTransformer (gate init 0). Raw events, leak-free.",
          flush=True)

    res = {a: {"resolved": [], "per_coord": [], "gates": []} for a in ARMS}
    for s in seeds:
        for a, kind in ARMS.items():
            pred_res, gates, npar = train_eval(kind, Xtr, Ytr, Xte, s)
            r2_res = probes.r2_score(Yte_res, pred_res)
            pred4, g4, _ = train_eval(kind, Xtr, Ytr4, Xte, s)
            per = [probes.r2_score(Cte[:, j:j + 1], pred4[:, j:j + 1]) for j in range(4)]
            res[a]["resolved"].append(r2_res)
            res[a]["per_coord"].append(per)
            if gates:
                res[a]["gates"].append(gates)
            res[a]["n_params"] = npar
        print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["resolved"]); pc = np.array(res[a]["per_coord"])
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)],
                      "n_params": int(res[a]["n_params"])}
        if res[a]["gates"]:
            summary[a]["mean_abs_gate"] = float(np.abs(np.array(res[a]["gates"])).mean())
    out = {"seeds": list(seeds), "k_resolved": k_resolved, "depth": L_DEPTH, "heads": HEADS,
           "inputs": "raw events; resolved c̃; leak-free; containment pairs",
           "summary": summary, "wall_seconds": time.time() - t0}
    (OUT / "contained_informer_recovery.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== CONTAINMENT c-recovery (raw events, leak-free, L={L_DEPTH}, "
          f"n_seeds={len(seeds)}) ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        sm = summary[a]
        extra = f"  |gate|={sm['mean_abs_gate']:.3f}" if "mean_abs_gate" in sm else ""
        print(f"  {a:20s} resolved c̃ R²={sm['resolved_r2_mean']:.3f}±{sm['resolved_r2_std']:.3f}"
              f"  clq3={sm['per_coord_r2_mean'][2]:.2f}{extra}")
    print("\nContainment check:")
    print(f"  Informer_LS {summary['Informer_LS']['resolved_r2_mean']:.3f} vs "
          f"LinearTransformer {summary['LinearTransformer']['resolved_r2_mean']:.3f}")
    print(f"  Informer_Gated {summary['Informer_Gated']['resolved_r2_mean']:.3f} vs "
          f"SoftmaxTransformer {summary['SoftmaxTransformer']['resolved_r2_mean']:.3f}")
    return out


if __name__ == "__main__":
    main()
