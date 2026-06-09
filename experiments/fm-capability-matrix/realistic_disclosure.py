r"""Does the Intention, wired to a MEASURABLE physics observable (no leak), win
the c-recovery / disclosure certificate against aggregator readouts?

Audit-clean inputs. The value is NOT log w_c (which needs the true c). It is the
binned differential observable a real analysis measures:
  μ(m_bin)   = dσ(c)/dσ_SM   (cross-section ratio per m-bin)
  A_FB(m_bin)= forward-backward asymmetry per m-bin
both computed at the scenario's c and then degraded with REALISTIC per-bin
finite-statistics noise (σ ∝ 1/√N_SM(bin), a fixed luminosity), so each scenario
is a noisy measurement — exactly what an experiment hands the fitter. μ and A_FB
are measurable from the events + the known SM template WITHOUT knowing c; only
the multiplicative-noise realisation differs from the true value. Multi-observable
(μ ⊕ A_FB) breaks the chirality degeneracy so more than the 2 m_ll-only Wilson
directions are resolvable.

EVERY arm receives the identical (m_bin, [μ_meas, A_FB_meas]) context and is
trained the same way (predict the observable at held-out query bins); the
per-scenario representation is then probed (held-out ridge) for the 4 Wilson
coefficients. c is target-only, never an input. The only thing that differs
between arms is how the context becomes the representation:
  Intention : closed-form ridge ψ(m) -> [μ,A_FB]  (the encoding block)
  MeanPool / Attention / DeepSets : pooled readouts (the control)
Multiple seeds -> mean ± std. Report numbers; assert nothing else.
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
from modules.analytic_smeft import differential_xs
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

OUT = HERE / "output_matrix"
D = 16
N_BINS = 12
M_LO, M_HI = 0.3, 2.3
LUMI = 5.0e4          # arbitrary luminosity scale setting the per-bin stat noise
STEPS = 1500
BATCH = 32


def bin_centers():
    edges = np.linspace(M_LO, M_HI, N_BINS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def sm_yields(oracle, centers):
    """Expected SM yield per m-bin (sets the per-bin statistical noise)."""
    res = differential_xs({}, centers * 1000.0, sqrt_s=oracle.sqrt_s,
                          lambda_scale=oracle.lam, order=oracle.order, pdf=oracle.pdf)
    sig = np.asarray(res["sm_only"], float)
    sig = np.clip(sig, 1e-12, None)
    return LUMI * sig / sig.sum() * N_BINS    # normalized to ~LUMI total, per bin


def measure(oracle, C, centers, n_sm, rng):
    """Per-scenario measured observables: μ and A_FB on the m-bins, with
    realistic per-bin statistical noise σ_μ = μ/√N_SM(bin)."""
    S = C.shape[0]
    mu = np.empty((S, N_BINS)); afb = np.empty((S, N_BINS))
    for i in range(S):
        Ci = np.tile(C[i], (N_BINS, 1))
        mu[i] = oracle.truth(Ci, centers)
        afb[i] = oracle.truth_afb(Ci, centers)
    sig_mu = mu / np.sqrt(n_sm)[None, :]
    sig_afb = 1.0 / np.sqrt(n_sm)[None, :]
    mu_meas = mu + sig_mu * rng.standard_normal(mu.shape)
    afb_meas = afb + sig_afb * rng.standard_normal(afb.shape)
    return mu_meas.astype(np.float32), afb_meas.astype(np.float32)


def mlp(sizes, act=nn.GELU, last=False):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last:
            L.append(act())
    return nn.Sequential(*L)


def _scaled(m):
    return torch.log(m)


class IntentionMO(nn.Module):
    """Closed-form ridge ψ(m) -> Y (2-D observable). Representation = w (d×2)."""
    def __init__(self, d=D, alpha=1e-3):
        super().__init__()
        self.psi = mlp([1, 64, 64, d]); self.d = d; self.alpha = alpha

    def _w(self, M, Y):                                  # M (B,K), Y (B,K,2)
        P = self.psi(_scaled(M).unsqueeze(-1))           # (B,K,d)
        A = P.transpose(-2, -1) @ P + self.alpha * torch.eye(self.d)
        return torch.linalg.solve(A, P.transpose(-2, -1) @ Y)   # (B,d,2)

    def forward(self, M, Y, Mq):
        w = self._w(M, Y)
        return self.psi(_scaled(Mq).unsqueeze(-1)) @ w    # (B,Q,2)

    @torch.no_grad()
    def rep(self, M, Y):
        return self._w(M, Y).reshape(M.shape[0], -1).numpy()


class PoolMO(nn.Module):
    """Pooled readout of (scaled m, μ, A_FB) triples. pool in {mean, attn}."""
    def __init__(self, pool="mean", d=D):
        super().__init__()
        self.enc = mlp([3, 64, d]); self.pool = pool; self.d = d
        # rep width matched to the Intention's w (d×2 = 2d): mean+std pooling.
        self.rep_dim = 2 * d
        if pool == "attn":
            self.seed = nn.Parameter(torch.randn(1, 2, d) * 0.1)   # 2 seeds -> 2d
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.dec = mlp([self.rep_dim + 1, 64, 2])

    def _summary(self, M, Y):
        h = self.enc(torch.cat([_scaled(M).unsqueeze(-1), Y], -1))    # (B,K,d)
        if self.pool == "mean":
            return torch.cat([h.mean(1), h.std(1)], -1)              # (B,2d)
        B = h.shape[0]
        z, _ = self.attn(self.seed.expand(B, -1, -1), h, h)          # (B,2,d)
        return z.reshape(B, -1)                                       # (B,2d)

    def forward(self, M, Y, Mq):
        z = self._summary(M, Y); B, Q = Mq.shape
        inp = torch.cat([z.unsqueeze(1).expand(-1, Q, -1),
                         _scaled(Mq).unsqueeze(-1)], -1)
        return self.dec(inp.reshape(B * Q, -1)).reshape(B, Q, 2)

    @torch.no_grad()
    def rep(self, M, Y):
        return self._summary(M, Y).numpy()


ARMS = {
    "Intention_ridge": lambda: IntentionMO(),
    "MeanPool": lambda: PoolMO("mean"),
    "Attention": lambda: PoolMO("attn"),
}


def train(model, M, Y, Mq, Yq, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = M.shape[0]; rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        idx = rng.choice(n, BATCH, replace=False)
        loss = ((model(M[idx], Y[idx], Mq[idx]) - Yq[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main(seeds=(0, 1, 2, 3, 4), k_resolved=2):
    t0 = time.time()
    data = sub.load_cache()
    oracle = sub.make_oracle()
    centers = bin_centers()
    n_sm = sm_yields(oracle, centers)
    Ctr, Cte = data["train_c"], data["test_c"]

    # Fisher with μ⊕A_FB to count resolvable directions on this observable set.
    rng = np.random.default_rng(0)
    cpr = rng.uniform(-0.6, 0.6, (256, 4)); mpr = rng.uniform(M_LO, M_HI, 256)
    F_mu = empirical_fisher_c(oracle, cpr, mpr, fd_step=1e-3)
    Dvals, V = fisher_basis(F_mu); Vk = V[:, :k_resolved]

    # split context/query bins (predict observable at held-out query bins)
    Mc = np.tile(centers[:N_BINS - 3], (len(Ctr), 1))
    Mq = np.tile(centers[N_BINS - 3:], (len(Ctr), 1))

    def build(C, seed):
        r = np.random.default_rng(1000 + seed)
        mu, afb = measure(oracle, C, centers, n_sm, r)
        Y = np.stack([mu, afb], -1)                       # (S,N_BINS,2)
        return Y

    print(f"Fisher(μ) eigvals {np.round(Dvals,4)}; realistic per-bin σ_μ/μ ≈ "
          f"{np.round(1/np.sqrt(n_sm),3)}", flush=True)
    print(f"INPUTS: every arm gets (m_bin, [μ_meas, A_FB_meas]); c is target-only. "
          f"No arm gets log w_c.", flush=True)

    res = {a: {"resolved": [], "per_coord": [], "obs_r2": []} for a in ARMS}
    for s in seeds:
        Ytr_full = build(Ctr, s); Yte_full = build(Cte, s)
        Mc_t = torch.tensor(Mc, dtype=torch.float32)
        Mq_t = torch.tensor(Mq, dtype=torch.float32)
        Yc = torch.tensor(Ytr_full[:, :N_BINS - 3], dtype=torch.float32)
        Yq = torch.tensor(Ytr_full[:, N_BINS - 3:], dtype=torch.float32)
        McTe = torch.tensor(np.tile(centers[:N_BINS - 3], (len(Cte), 1)), dtype=torch.float32)
        YcTe = torch.tensor(Yte_full[:, :N_BINS - 3], dtype=torch.float32)
        raw_tr = Ytr_full.reshape(len(Ctr), -1)
        raw_te = Yte_full.reshape(len(Cte), -1)
        for a, fac in ARMS.items():
            m = fac(); train(m, Mc_t, Yc, Mq_t, Yq, s)
            with torch.no_grad():
                obs_r2 = probes.r2_score(Yte_full[:, N_BINS-3:],
                    m(McTe, YcTe, torch.tensor(np.tile(centers[N_BINS-3:], (len(Cte),1)),
                      dtype=torch.float32)).numpy())
            Ztr = m.rep(Mc_t, Yc); Zte = m.rep(McTe, YcTe)
            rr = probes.held_out_probe(Ztr, Ctr @ Vk, Zte, Cte @ Vk, seed=s).r2
            p4 = probes.held_out_probe(Ztr, Ctr, Zte, Cte, seed=s)
            res[a]["resolved"].append(rr)
            res[a]["per_coord"].append(p4.r2_per_target)
            res[a]["obs_r2"].append(obs_r2)
        print(f"seed {s} done", flush=True)

    summary = {}
    for a in ARMS:
        rr = np.array(res[a]["resolved"]); pc = np.array(res[a]["per_coord"])
        summary[a] = {"resolved_r2_mean": float(rr.mean()), "resolved_r2_std": float(rr.std()),
                      "per_coord_r2_mean": [float(x) for x in pc.mean(0)],
                      "obs_pred_r2_mean": float(np.mean(res[a]["obs_r2"]))}
    out = {"seeds": list(seeds), "n_bins": N_BINS, "lumi": LUMI,
           "inputs": "(m_bin,[μ_meas,A_FB_meas]) measured w/ per-bin stat noise; "
                     "c target-only; no log w_c",
           "fisher_eigenvalues": [float(x) for x in Dvals],
           "summary": summary, "wall_seconds": time.time() - t0}
    (OUT / "realistic_disclosure.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== REALISTIC disclosure (measured μ⊕A_FB, leak-free, n_seeds={len(seeds)}) ===")
    for a in sorted(summary, key=lambda n: summary[n]["resolved_r2_mean"], reverse=True):
        s = summary[a]
        print(f"  {a:16s} resolved c̃ R²={s['resolved_r2_mean']:.3f}±{s['resolved_r2_std']:.3f}"
              f"  obs-pred R²={s['obs_pred_r2_mean']:.3f}  per-coord={np.round(s['per_coord_r2_mean'],2)}")
    return out


if __name__ == "__main__":
    main()
