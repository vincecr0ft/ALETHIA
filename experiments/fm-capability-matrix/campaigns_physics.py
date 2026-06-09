r"""Corrected cross-campaign comparison on the PHYSICS interface.

The first cross-campaign run (informer_campaigns.py) tested the event-set JEPA
reframe: generic per-event encoder, no physical observable, self-supervised
embedding prediction. That is off-design for the Intention mechanism, whose
power is the closed-form ridge of physics features psi(log m) onto the physical
observable mu(c,m) (diag_intention_physics.py: it reproduces the project's
mu-pred R²=0.997 and resolved-direction c-recovery 0.85 there, vs ~0.10 for the
event-set version). That run is retracted.

This run gives EVERY architecture the identical (m, mu) physics interface and
the same resolved-subspace metric — the known-good intention-vs-deepsets setup —
and scores each across campaigns. No rigging: same data, same metric, the only
difference is the architecture. Cast (all consume forward(M_ctx,Y_ctx,M_query)
and expose a per-scenario representation):

  Intention            closed-form ridge on psi(log m) -> mu (the repo mechanism)
  DeepSets             mean-pool aggregator
  SetTransformer       attention pooling
  ParticleTransformer  self-attention + pairwise |Δlog m| bias
  DeepONet             branch/trunk operator on the binned profile
  TabPFN               in-context transformer

Campaigns (each on resolved Wilson directions c̃ = c @ V_fisher, k resolved):
  mu_fit          held-out mu-prediction R²  (the physics-fit; primary)
  recovery        held-out c̃ recovery R² (capacity-controlled probe)
  label_eff       c̃ recovery R² vs label budget (AUC of the curve)
  ood             c̃ recovery R² on the extrapolation shell
  calibration     Gaussian head on the representation -> c̃ calibration error
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
sys.path.insert(0, str(REPO / "experiments" / "intention-vs-deepsets"))

import substrate as sub
import probes
from intention_learned import IntentionFMLearned
from architectures_extended import (
    SetTransformerFM, ParticleTransformerFM, DeepONetFM, TabPFNHeadFM)
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

OUT = HERE / "output_matrix"
K_CTX, Q_QRY = 64, 32
M_LO, M_HI = 0.3, 2.3
NOISE = 0.05
STEPS = 1500


# --------------------------------------------------------------------------
# Data: (m, mu) physics context on given Wilson points.
# --------------------------------------------------------------------------


def build_scalar(C, oracle, rng):
    S = C.shape[0]
    Mc = rng.uniform(M_LO, M_HI, (S, K_CTX))
    Mq = rng.uniform(M_LO, M_HI, (S, Q_QRY))
    Yc = np.empty((S, K_CTX)); Yq = np.empty((S, Q_QRY))
    for i in range(S):
        Yc[i] = oracle.truth(np.tile(C[i], (K_CTX, 1)), Mc[i])
        Yq[i] = oracle.truth(np.tile(C[i], (Q_QRY, 1)), Mq[i])
    Yc *= (1 + NOISE * rng.standard_normal(Yc.shape))
    Yq *= (1 + NOISE * rng.standard_normal(Yq.shape))
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(Mc), f(Yc), f(Mq), f(Yq)


# --------------------------------------------------------------------------
# A DeepSets baseline on the same scalar interface (architectures_extended has
# the rest of the cast but not a plain DeepSets, so define one matching its API).
# --------------------------------------------------------------------------


class DeepSetsScalar(torch.nn.Module):
    def __init__(self, d_out=24, hidden=48):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(2, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out))
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(d_out + 1, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _summary(self, Mc, Yc):
        return self.phi(torch.stack([torch.log(Mc), Yc], -1)).mean(1)

    def forward(self, Mc, Yc, Mq):
        z = self._summary(Mc, Yc); S, Q = Mq.shape
        zin = torch.cat([z.unsqueeze(1).expand(-1, Q, -1),
                         torch.log(Mq).unsqueeze(-1)], -1)
        return self.dec(zin.reshape(S * Q, -1)).reshape(S, Q)

    @torch.no_grad()
    def representation(self, Mc, Yc):
        return self._summary(Mc, Yc)


def make_cast():
    return {
        "Intention": lambda: IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3),
        "DeepSets": lambda: DeepSetsScalar(),
        "SetTransformer": lambda: SetTransformerFM(),
        "ParticleTransformer": lambda: ParticleTransformerFM(),
        "DeepONet": lambda: DeepONetFM(),
        "TabPFN": lambda: TabPFNHeadFM(),
    }


def meta_train(model, Mc, Yc, Mq, Yq, *, steps=STEPS, bs=64, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Mc.shape[0]; rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(n, bs, replace=False)
        loss = ((model(Mc[idx], Yc[idx], Mq[idx]) - Yq[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def represent(name, model, Mc, Yc):
    if name == "Intention":
        S, K = Mc.shape; d = model.psi.d_psi
        Psi = model.psi(Mc.reshape(-1)).reshape(S, K, d)
        A = torch.einsum("skd,ske->sde", Psi, Psi) + model.alpha * torch.eye(d)
        b = torch.einsum("skd,sk->sd", Psi, Yc)
        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1).numpy()
    return model.representation(Mc, Yc).numpy()


def _gauss_head(Ztr, Ytr, steps=800, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    d, k = Ztr.shape[1], Ytr.shape[1]
    mean = torch.nn.Linear(d, k); logstd = torch.nn.Linear(d, k)
    opt = torch.optim.Adam(list(mean.parameters()) + list(logstd.parameters()), lr=lr)
    Z = torch.tensor(Ztr, dtype=torch.float32); Y = torch.tensor(Ytr, dtype=torch.float32)
    for _ in range(steps):
        m, ls = mean(Z), logstd(Z).clamp(-6, 3)
        loss = (0.5 * ((Y - m) ** 2 / torch.exp(2 * ls)) + ls).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return mean, logstd


CAMPAIGN_DIR = {"mu_fit": +1, "recovery": +1, "label_eff": +1,
                "ood": +1, "calibration": -1}


def run(seed: int = 0, k_resolved: int = 2) -> dict:
    t0 = time.time()
    data = sub.load_cache()
    Ctr, Cte, Cood = data["train_c"], data["test_c"], data["ood_c"]
    oracle = sub.make_oracle()
    rng = np.random.default_rng(seed)

    # Fisher basis on the analytic prior -> resolved Wilson directions.
    c_prior = rng.uniform(-sub.CONFIG.box_half, sub.CONFIG.box_half,
                          (256, sub.CONFIG.n_wc))
    m_prior = rng.uniform(M_LO, M_HI, 256)
    F = empirical_fisher_c(oracle, c_prior, m_prior, fd_step=1e-3)
    D, V = fisher_basis(F)                          # eigvals desc, eigvecs desc
    Vk = V[:, :k_resolved]
    ctil = {"train": Ctr @ Vk, "test": Cte @ Vk, "ood": Cood @ Vk}
    print(f"Fisher eigenvalues (desc): {np.round(D, 4)} -> using top {k_resolved}",
          flush=True)

    print("building (m, mu) physics data ...", flush=True)
    S = {}
    S["train"] = build_scalar(Ctr, oracle, rng)
    S["test"] = build_scalar(Cte, oracle, rng)
    S["ood"] = build_scalar(Cood, oracle, rng)
    raw = {"train": probes.raw_event_summary(data["train_X1"]),
           "test": probes.raw_event_summary(data["test_X1"])}

    results, meta = {}, {}
    for name, factory in make_cast().items():
        print(f"training {name} ...", flush=True)
        model = factory()
        meta_train(model, *S["train"], seed=seed)
        meta[name] = {"n_params": int(model.n_params)}

        # mu_fit: held-out prediction R² on the physical observable.
        with torch.no_grad():
            Mc, Yc, Mq, Yq = S["test"]
            mu_r2 = probes.r2_score(Yq.numpy(), model(Mc, Yc, Mq).numpy())

        # representations
        Z = {sp: represent(name, model, S[sp][0], S[sp][1])
             for sp in ["train", "test", "ood"]}

        rec = probes.probe_with_floor(Z["train"], ctil["train"], Z["test"],
                                      ctil["test"], raw["train"], raw["test"],
                                      seed=seed)
        # label efficiency on c̃
        le = []
        for b in (25, 50, 100, 200, 450):
            b = min(b, Z["train"].shape[0])
            le.append(probes.held_out_probe(Z["train"][:b], ctil["train"][:b],
                                            Z["test"], ctil["test"], seed=seed).r2)
        ood_r2 = probes.held_out_probe(Z["train"], ctil["train"], Z["ood"],
                                       ctil["ood"], seed=seed).r2
        mean, logstd = _gauss_head(Z["train"], ctil["train"], seed=seed)
        with torch.no_grad():
            Zt = torch.tensor(Z["test"], dtype=torch.float32)
            cal = probes.gaussian_coverage(ctil["test"], mean(Zt).numpy(),
                                           np.exp(logstd(Zt).clamp(-6, 3).numpy()))

        results[name] = {
            "mu_fit": mu_r2, "recovery": rec.r2, "recovery_margin": rec.margin,
            "label_eff": float(np.mean(le)), "label_eff_curve": [float(x) for x in le],
            "ood": ood_r2, "calibration": cal["calibration_error"],
        }
        print(f"  {name}: mu={mu_r2:.3f} rec={rec.r2:.3f} le={np.mean(le):.3f} "
              f"ood={ood_r2:.3f} cal={cal['calibration_error']:.3f}", flush=True)

    # ranks per campaign
    ranks = {n: {} for n in results}
    for camp, direction in CAMPAIGN_DIR.items():
        vals = {n: results[n][camp] for n in results}
        order = sorted(vals, key=lambda n: vals[n], reverse=(direction > 0))
        for r, n in enumerate(order, 1):
            ranks[n][camp] = r
    mean_rank = {n: float(np.mean(list(ranks[n].values()))) for n in results}

    out = {"interface": "(m, mu) physics observable; resolved Wilson dirs",
           "k_resolved": k_resolved, "fisher_eigenvalues": [float(x) for x in D],
           "results": results, "ranks": ranks, "mean_rank": mean_rank,
           "meta": meta, "wall_seconds": time.time() - t0,
           "campaign_dir": CAMPAIGN_DIR}
    OUT.mkdir(exist_ok=True)
    (OUT / "campaigns_physics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run()
    print("\nmean rank across campaigns (1=best):")
    for n, mr in sorted(r["mean_rank"].items(), key=lambda kv: kv[1]):
        print(f"  {n:20s} {mr:.2f}  ranks={r['ranks'][n]}")
