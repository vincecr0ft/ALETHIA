r"""The discriminating test the prior comparison lacked: vary the VALUE and the
READOUT, not the set-pooler.

Diagnosis (confirmed numerically): the ManifoldInformer's ridge value V is the
EMA self-embedding, so V≈K and w_θ=(KᵀK+αI)⁻¹KᵀV → I; the summary degenerates to
mean-pool (cos(summary, mean-pool)=1.0000). Self-prediction buys no disclosure.

The fix (Intention KVQ, Garnelo & Czarnecki 2305.10203; ICL-as-kernel-regression
on manifolds, 2506.10959): make V carry the morphing target — the per-event
log-likelihood ratio log w_c(x). Then w_θ(c) is the morphing-basis coordinate, a
linear probe recovers the A_i/B_ij structure by construction, and a pooler with
no per-scenario solve should fall away.

Cast (every arm sees the SAME context = events + per-event value log w_c, and is
trained to predict log w_c at held-out query events; they differ only in how the
per-scenario coordinate is formed and read out):

  Intention_ridge_logw  : V=log w_c, closed-form LINEAR ridge  w=(KᵀK+αI)⁻¹Kᵀv
  Nonlinear_deepsets_logw: V=log w_c, NONLINEAR DeepSets pool of (φ,v) + MLP head
  Nonlinear_attn_logw   : V=log w_c, NONLINEAR attention pool of (φ,v) + MLP head
  Ridge_self            : V=self-embedding (the bug): w=(KᵀK+αI)⁻¹KᵀK, JEPA-trained
  MeanPool_novalue      : coordinate = mean φ(X), value NOT used to form it

Each coordinate is probed (held-out ridge) for Wilson-coefficient recovery, per
direction and on the Fisher-resolved subspace. Multiple seeds -> mean ± std.

Hypotheses: Intention_ridge_logw >> {Ridge_self, MeanPool_novalue} (value matters)
and > {Nonlinear_*_logw} (H5: a nonlinear readout destroys polynomial-in-c
recovery).
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
from nets import mlp
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

OUT = HERE / "output_matrix"
D = 16
N_CTX = 120
STEPS = 1200
BATCH = 32
ALPHA = 1e-3


# --------------------------------------------------------------------------
# Arms — each: forward_loss(Xc, vc, Xq, vq) -> scalar loss; coordinate(Xc, vc).
# --------------------------------------------------------------------------


class RidgeLogw(nn.Module):
    """V = log w_c, closed-form linear ridge readout (the Intention done right)."""
    def __init__(self):
        super().__init__()
        self.phi = mlp([2, 64, 64, D])
        self.alpha = ALPHA

    def _w(self, Xc, vc):
        K = self.phi(Xc)                              # (S,Nc,D)
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        b = K.transpose(-2, -1) @ vc.unsqueeze(-1)    # (S,D,1)
        return torch.linalg.solve(A, b)               # (S,D,1)

    def forward_loss(self, Xc, vc, Xq, vq):
        w = self._w(Xc, vc)
        pred = (self.phi(Xq) @ w).squeeze(-1)         # (S,Nq) linear in φ(x_q)
        return ((pred - vq) ** 2).mean()

    @torch.no_grad()
    def coordinate(self, Xc, vc):
        return self._w(Xc, vc).squeeze(-1).cpu().numpy()


class RidgeSelf(nn.Module):
    """V = self-embedding (the bug): w = (KᵀK+αI)⁻¹KᵀK, trained by JEPA self-
    prediction of query embeddings. Coordinate = w·mean(V) (≡ mean-pool)."""
    def __init__(self):
        super().__init__()
        self.phi = mlp([2, 64, 64, D])
        self.alpha = ALPHA

    def _w_and_V(self, Xc):
        K = self.phi(Xc); V = K                       # self-embedding
        A = K.transpose(-2, -1) @ K + self.alpha * torch.eye(D)
        b = K.transpose(-2, -1) @ V
        return torch.linalg.solve(A, b), V            # (S,D,D), (S,Nc,D)

    def forward_loss(self, Xc, vc, Xq, vq):
        w, _ = self._w_and_V(Xc)
        Zpred = self.phi(Xq) @ w                       # (S,Nq,D)
        return ((Zpred - self.phi(Xq).detach()) ** 2).mean()   # JEPA self-pred

    @torch.no_grad()
    def coordinate(self, Xc, vc):
        w, V = self._w_and_V(Xc)
        return torch.einsum("sdt,st->sd", w, V.mean(1)).cpu().numpy()


class NonlinearPool(nn.Module):
    """V = log w_c, NONLINEAR readout. pool in {deepsets, attn}."""
    def __init__(self, pool="deepsets"):
        super().__init__()
        self.phi = mlp([2, 64, D])
        self.g = mlp([D + 1, 64, D])                  # per-event (φ, v) -> D
        self.pool = pool
        if pool == "attn":
            self.seed = nn.Parameter(torch.randn(1, 1, D) * 0.1)
            self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.head = mlp([D + D, 64, 1])               # [φ(x_q), coord] -> v

    def _coord(self, Xc, vc):
        h = self.g(torch.cat([self.phi(Xc), vc.unsqueeze(-1)], -1))   # (S,Nc,D)
        if self.pool == "deepsets":
            return h.mean(1)
        S = h.shape[0]
        z, _ = self.attn(self.seed.expand(S, -1, -1), h, h)
        return z.squeeze(1)

    def forward_loss(self, Xc, vc, Xq, vq):
        coord = self._coord(Xc, vc)                    # (S,D)
        Kq = self.phi(Xq); S, Nq, _ = Kq.shape
        inp = torch.cat([Kq, coord.unsqueeze(1).expand(-1, Nq, -1)], -1)
        pred = self.head(inp.reshape(S * Nq, -1)).reshape(S, Nq)
        return ((pred - vq) ** 2).mean()

    @torch.no_grad()
    def coordinate(self, Xc, vc):
        return self._coord(Xc, vc).cpu().numpy()


class MeanPoolNoValue(nn.Module):
    """Coordinate = mean φ(X); the value is NOT used to form it (events only)."""
    def __init__(self):
        super().__init__()
        self.phi = mlp([2, 64, D])
        self.head = mlp([D + D, 64, 1])

    def _coord(self, Xc):
        return self.phi(Xc).mean(1)

    def forward_loss(self, Xc, vc, Xq, vq):
        coord = self._coord(Xc)
        Kq = self.phi(Xq); S, Nq, _ = Kq.shape
        inp = torch.cat([Kq, coord.unsqueeze(1).expand(-1, Nq, -1)], -1)
        pred = self.head(inp.reshape(S * Nq, -1)).reshape(S, Nq)
        return ((pred - vq) ** 2).mean()

    @torch.no_grad()
    def coordinate(self, Xc, vc):
        return self._coord(Xc).cpu().numpy()


ARMS = {
    "Intention_ridge_logw": lambda: RidgeLogw(),
    "Nonlinear_deepsets_logw": lambda: NonlinearPool("deepsets"),
    "Nonlinear_attn_logw": lambda: NonlinearPool("attn"),
    "Ridge_self_bug": lambda: RidgeSelf(),
    "MeanPool_novalue": lambda: MeanPoolNoValue(),
}


def train(model, Xc, vc, Xq, vq, *, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = Xc.shape[0]; rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        idx = rng.choice(n, BATCH, replace=False)
        loss = model.forward_loss(Xc[idx], vc[idx], Xq[idx], vq[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_seed(seed, data, logw, Vk, raw):
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    Ltr = torch.tensor(logw["train_logw1"], dtype=torch.float32)
    Lte = torch.tensor(logw["test_logw1"], dtype=torch.float32)
    # standardize the value target on train (stability); coordinate probe is invariant
    mu, sd = Ltr.mean(), Ltr.std().clamp_min(1e-6)
    Ltr = (Ltr - mu) / sd; Lte = (Lte - mu) / sd
    # context/query split
    Xc_tr, Xq_tr, vc_tr, vq_tr = Xtr[:, :N_CTX], Xtr[:, N_CTX:], Ltr[:, :N_CTX], Ltr[:, N_CTX:]
    Xc_te, vc_te = Xte[:, :N_CTX], Lte[:, :N_CTX]
    Ctr, Cte = data["train_c"], data["test_c"]
    ctil_tr, ctil_te = Ctr @ Vk, Cte @ Vk            # Fisher-resolved targets

    out = {}
    for name, factory in ARMS.items():
        m = factory()
        train(m, Xc_tr, vc_tr, Xq_tr, vq_tr, seed=seed)
        Ztr = m.coordinate(Xc_tr, vc_tr)
        Zte = m.coordinate(Xc_te, vc_te)
        # resolved-subspace recovery (top-k Fisher dirs)
        rec = probes.probe_with_floor(Ztr, ctil_tr, Zte, ctil_te,
                                      raw["train"], raw["test"], seed=seed)
        # per-Wilson-coordinate recovery (all 4), to see how many are resolved
        per = probes.held_out_probe(Ztr, Ctr, Zte, Cte, seed=seed)
        out[name] = {"resolved_r2": rec.r2, "resolved_margin": rec.margin,
                     "per_coord_r2": [float(x) for x in per.r2_per_target],
                     "all4_r2": per.r2}
    return out


def main(seeds=(0, 1, 2, 3, 4), k_resolved=2):
    t0 = time.time()
    data = sub.load_cache()
    logw = {k: np.load(OUT / "logw_cache.npz")[k]
            for k in np.load(OUT / "logw_cache.npz").files}
    oracle = sub.make_oracle()
    rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-sub.CONFIG.box_half, sub.CONFIG.box_half,
                           (256, 4)), rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, V = fisher_basis(F)
    Vk = V[:, :k_resolved]
    raw = {"train": probes.raw_event_summary(data["train_X1"]),
           "test": probes.raw_event_summary(data["test_X1"])}
    print(f"Fisher eigvals {np.round(Dvals,4)}; resolving top {k_resolved}", flush=True)

    per_seed = []
    for s in seeds:
        print(f"seed {s} ...", flush=True)
        per_seed.append(run_seed(s, data, logw, Vk, raw))

    # aggregate mean ± std
    agg = {}
    for name in ARMS:
        res_r2 = np.array([ps[name]["resolved_r2"] for ps in per_seed])
        all4 = np.array([ps[name]["all4_r2"] for ps in per_seed])
        per_coord = np.array([ps[name]["per_coord_r2"] for ps in per_seed])  # (seeds,4)
        agg[name] = {
            "resolved_r2_mean": float(res_r2.mean()), "resolved_r2_std": float(res_r2.std()),
            "all4_r2_mean": float(all4.mean()), "all4_r2_std": float(all4.std()),
            "per_coord_r2_mean": [float(x) for x in per_coord.mean(0)],
            "n_dirs_recovered": int((per_coord.mean(0) > 0.3).sum()),
        }

    result = {"seeds": list(seeds), "k_resolved": k_resolved,
              "fisher_eigenvalues": [float(x) for x in Dvals],
              "agg": agg, "per_seed": per_seed, "wall_seconds": time.time() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "value_ablation.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== value × readout ablation (n_seeds={len(seeds)}), resolved top-{k_resolved} ===")
    order = sorted(agg, key=lambda n: agg[n]["resolved_r2_mean"], reverse=True)
    for n in order:
        a = agg[n]
        print(f"  {n:26s} resolved R²={a['resolved_r2_mean']:.3f}±{a['resolved_r2_std']:.3f}  "
              f"margin {a['resolved_r2_mean']-0:.3f} | all4 R²={a['all4_r2_mean']:.3f}  "
              f"#dirs>0.3={a['n_dirs_recovered']}  per-coord={np.round(a['per_coord_r2_mean'],2)}")
    return result


if __name__ == "__main__":
    main()
