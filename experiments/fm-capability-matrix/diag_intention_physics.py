r"""Diagnostic: the Intention mechanism on the physics interface vs what the
matrix tested. Confirms the regression is the harness, not the mechanism.

The matrix's representation cells used the event-set JEPA ManifoldInformer: a
generic per-event MLP encoder, NO physical observable, trained self-supervised
to predict event embeddings. It recovers c at R² ~0.1.

The Intention mechanism that worked in the project is the closed-form ridge on
PHYSICS features psi(log m) predicting the PHYSICAL OBSERVABLE mu(c,m); its
per-scenario ridge coefficients w are the representation. Here we run exactly
that on the SAME Wilson points the matrix uses, and a DeepSets baseline on the
SAME (m, mu) interface, and probe c held-out. No rigging: every model sees the
same (m, mu) context; the only difference is the aggregator.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "intention-vs-deepsets"))

import substrate as sub
import probes
from intention_learned import IntentionFMLearned

K_CTX, Q_QRY = 64, 32
M_LO, M_HI = 0.3, 2.3
NOISE = 0.05


def build_scalar(C, oracle, rng):
    """Per scenario: K context + Q query (m, mu) pairs. mu is the physical
    observable (cross-section ratio) with small multiplicative noise."""
    S = C.shape[0]
    Mc = rng.uniform(M_LO, M_HI, (S, K_CTX))
    Mq = rng.uniform(M_LO, M_HI, (S, Q_QRY))
    Yc = np.empty((S, K_CTX)); Yq = np.empty((S, Q_QRY))
    for i in range(S):
        Ci = np.tile(C[i], (K_CTX, 1))
        Yc[i] = oracle.truth(Ci, Mc[i])
        Ci = np.tile(C[i], (Q_QRY, 1))
        Yq[i] = oracle.truth(Ci, Mq[i])
    Yc *= (1 + NOISE * rng.standard_normal(Yc.shape))
    Yq *= (1 + NOISE * rng.standard_normal(Yq.shape))
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(Mc), f(Yc), f(Mq), f(Yq)


def meta_train(model, Mc, Yc, Mq, Yq, *, steps=1500, bs=64, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Mc.shape[0]; rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(n, bs, replace=False)
        pred = model(Mc[idx], Yc[idx], Mq[idx])
        loss = ((pred - Yq[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def intention_w(model, Mc, Yc):
    """Per-scenario ridge coefficients w = (Psi^T Psi + aI)^-1 Psi^T Y. This is
    the Intention representation (repr_intention in representation_analysis.py)."""
    S, K = Mc.shape; d = model.psi.d_psi
    Psi = model.psi(Mc.reshape(-1)).reshape(S, K, d)
    A = torch.einsum("skd,ske->sde", Psi, Psi) + model.alpha * torch.eye(d)
    b = torch.einsum("skd,sk->sd", Psi, Yc)
    return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1).numpy()


class DeepSetsScalar(torch.nn.Module):
    """DeepSets on the SAME (m, mu) interface: per-pair MLP -> mean pool."""
    def __init__(self, d_out=16, hidden=64):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(2, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out))
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(d_out + 1, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1))
        self.d_out = d_out

    def _summary(self, Mc, Yc):
        evt = torch.stack([torch.log(Mc), Yc], -1)
        return self.phi(evt).mean(1)

    def forward(self, Mc, Yc, Mq):
        z = self._summary(Mc, Yc)
        S, Q = Mq.shape
        zin = torch.cat([z.unsqueeze(1).expand(-1, Q, -1),
                         torch.log(Mq).unsqueeze(-1)], -1)
        return self.dec(zin.reshape(S * Q, -1)).reshape(S, Q)

    @torch.no_grad()
    def repr(self, Mc, Yc):
        return self._summary(Mc, Yc).numpy()


def main():
    t0 = time.time()
    data = sub.load_cache()
    Ctr, Cte = data["train_c"], data["test_c"]
    oracle = sub.make_oracle()
    rng = np.random.default_rng(0)

    print("building (m, mu) physics context on the matrix's Wilson points ...",
          flush=True)
    Mc_tr, Yc_tr, Mq_tr, Yq_tr = build_scalar(Ctr, oracle, rng)
    Mc_te, Yc_te, Mq_te, Yq_te = build_scalar(Cte, oracle, rng)
    raw_tr = probes.raw_event_summary(data["train_X1"])
    raw_te = probes.raw_event_summary(data["test_X1"])

    # --- Intention (closed-form ridge on physics features -> mu) ---
    intn = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    meta_train(intn, Mc_tr, Yc_tr, Mq_tr, Yq_tr)
    mu_r2 = probes.r2_score(Yq_te.numpy(), intn(Mc_te, Yc_te, Mq_te).detach().numpy())
    W_tr, W_te = intention_w(intn, Mc_tr, Yc_tr), intention_w(intn, Mc_te, Yc_te)
    intn_probe = probes.probe_with_floor(W_tr, Ctr, W_te, Cte, raw_tr, raw_te)

    # --- DeepSets on the SAME (m, mu) interface ---
    ds = DeepSetsScalar()
    meta_train(ds, Mc_tr, Yc_tr, Mq_tr, Yq_tr)
    ds_mu_r2 = probes.r2_score(Yq_te.numpy(), ds(Mc_te, Yc_te, Mq_te).detach().numpy())
    Zds_tr, Zds_te = ds.repr(Mc_tr, Yc_tr), ds.repr(Mc_te, Yc_te)
    ds_probe = probes.probe_with_floor(Zds_tr, Ctr, Zds_te, Cte, raw_tr, raw_te)

    print(f"\n--- physics interface (m, mu), same Wilson points, {time.time()-t0:.0f}s ---")
    print(f"Intention  : mu-pred R²={mu_r2:.3f}  | c-recovery R²={intn_probe.r2:.3f} "
          f"(floor {intn_probe.floor_r2:.3f}, margin {intn_probe.margin:+.3f})")
    print(f"DeepSets    : mu-pred R²={ds_mu_r2:.3f}  | c-recovery R²={ds_probe.r2:.3f} "
          f"(floor {ds_probe.floor_r2:.3f}, margin {ds_probe.margin:+.3f})")
    print(f"\nfor contrast, the matrix's event-set ManifoldInformer (no mu, JEPA):")
    print(f"  c-recovery R² ~0.10, margin ~ -0.04  (recovers c barely at floor)")
    print(f"\nper-target Intention c-recovery R²: "
          f"{[round(x,3) for x in intn_probe.r2_per_target]}")


if __name__ == "__main__":
    main()
