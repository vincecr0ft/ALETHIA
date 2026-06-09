r"""Capability cell: Contrastive / JEPA joint-embedding  ->  OOD robustness.

FM family: Contrastive / JEPA joint-embedding (analogue of RS3L self-supervised
jets). The substrate provides two independent re-simulation views X1, X2 per
Wilson scenario (same c, different Monte-Carlo seeds), enabling view-invariant
pretraining without ever seeing the Wilson label c.

HEP task: View-invariant event-set representation (two re-simulation views per
c), tested for out-of-distribution robustness: trained on the in-distribution
Wilson box, probed on an extrapolation shell.

Objective tested: does VIEW-INVARIANCE pretraining buy OOD ROBUSTNESS (smaller
in->out degradation) vs a plain supervised encoder?

The JEPA encoder is pretrained with a VICReg-style objective that penalises:
  * alignment: ||z1 - z2||^2 (pull same-scenario views together),
  * variance collapse: relu(1 - std(z)).mean() per batch (prevent collapse),
  * covariance decorrelation: off-diagonal squared elements of cov(z) (prevent
    redundancy).
Wilson coefficients NEVER enter the encoder; c is only used for the downstream
linear probe after the encoder is frozen.

Ablation: a plain supervised encoder (same DeepSetsEncoder arch, same width)
trained to regress c on the box gives R2_box_sup, R2_ood_sup. JEPA is expected
to have a smaller OOD drop because view-invariance is a weaker in-distribution
constraint and the representation generalises further.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import substrate as sub
import probes
from nets import DeepSetsEncoder, mlp, count_params, set_seed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"


# ---------------------------------------------------------------------------
# Hyper-parameters (tiny for CPU speed < 150 s).
# ---------------------------------------------------------------------------

D_Z = 32          # encoder output width
D_EMB = 48        # per-event embedding dim
HIDDEN = 64       # MLP hidden
BS = 64           # scenarios per step
STEPS_JEPA = 1200
STEPS_SUP = 1200
LR = 2e-3

# VICReg coefficients
LAM_SIM = 25.0    # alignment (invariance)
LAM_VAR = 25.0    # variance (anti-collapse)
LAM_COV = 1.0     # covariance (decorrelation)


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------

def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """VICReg on a batch of pairs (B, D).

    Alignment  : mean squared distance between same-scenario embeddings.
    Variance   : hinge on per-feature std — penalise if std < 1.
    Covariance : sum of squared off-diagonal elements of the normalised cov.
    """
    B, D = z1.shape

    # -- alignment --
    loss_sim = torch.mean((z1 - z2) ** 2)

    # -- variance --
    def var_term(z):
        z_c = z - z.mean(0, keepdim=True)
        std = torch.sqrt(z_c.var(0) + 1e-4)
        return torch.relu(1.0 - std).mean()

    loss_var = var_term(z1) + var_term(z2)

    # -- covariance --
    def cov_term(z):
        z_c = z - z.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / (B - 1)
        # zero out diagonal; square and sum off-diagonal elements
        mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        return (cov[mask] ** 2).sum() / D

    loss_cov = cov_term(z1) + cov_term(z2)

    total = LAM_SIM * loss_sim + LAM_VAR * loss_var + LAM_COV * loss_cov
    return total, {"sim": loss_sim.item(), "var": loss_var.item(),
                   "cov": loss_cov.item(), "total": total.item()}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def pretrain_jepa(
    enc: DeepSetsEncoder,
    X1_tr: torch.Tensor,
    X2_tr: torch.Tensor,
    *,
    steps: int = STEPS_JEPA,
    bs: int = BS,
    lr: float = LR,
    seed: int = 0,
) -> list[float]:
    """Siamese VICReg pretraining — c never seen."""
    set_seed(seed)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    n = X1_tr.shape[0]
    rng = np.random.default_rng(seed)
    enc.train()
    losses = []
    for step in range(steps):
        idx = rng.choice(n, min(bs, n), replace=False)
        z1 = enc(X1_tr[idx])
        z2 = enc(X2_tr[idx])
        loss, info = vicreg_loss(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(info["total"])
    return losses


def train_supervised(
    enc: DeepSetsEncoder,
    X_tr: torch.Tensor,
    C_tr: torch.Tensor,
    *,
    steps: int = STEPS_SUP,
    bs: int = BS,
    lr: float = LR,
    seed: int = 0,
) -> nn.Module:
    """Supervised regression head on top of the same encoder arch."""
    set_seed(seed)
    head = mlp([D_Z, HIDDEN, C_tr.shape[1]])
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    enc.train()
    head.train()
    for _ in range(steps):
        idx = rng.choice(n, min(bs, n), replace=False)
        z = enc(X_tr[idx])
        pred = head(z)
        loss = torch.mean((pred - C_tr[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(seed: int = 0) -> dict:
    t0 = time.time()
    set_seed(seed)

    data = sub.load_cache()

    X1_tr = torch.tensor(data["train_X1"], dtype=torch.float32)
    X2_tr = torch.tensor(data["train_X2"], dtype=torch.float32)
    X1_te = torch.tensor(data["test_X1"], dtype=torch.float32)
    X1_ood = torch.tensor(data["ood_X1"], dtype=torch.float32)

    C_tr = data["train_c"].astype(np.float32)
    C_te = data["test_c"].astype(np.float32)
    C_ood = data["ood_c"].astype(np.float32)

    # ------------------------------------------------------------------
    # 1.  JEPA pretraining (c-agnostic)
    # ------------------------------------------------------------------
    jepa_enc = DeepSetsEncoder(d_out=D_Z, d_emb=D_EMB, hidden=HIDDEN)
    jepa_losses = pretrain_jepa(jepa_enc, X1_tr, X2_tr, seed=seed)

    # Freeze and extract representations
    jepa_enc.eval()
    with torch.no_grad():
        Z_tr_jepa = jepa_enc(X1_tr).numpy()
        Z_te_jepa = jepa_enc(X1_te).numpy()
        Z_ood_jepa = jepa_enc(X1_ood).numpy()

    # ------------------------------------------------------------------
    # 2.  JEPA probes: box (test) and shell (ood)
    # ------------------------------------------------------------------
    probe_jepa_box = probes.held_out_probe(Z_tr_jepa, C_tr, Z_te_jepa, C_te, seed=seed)
    probe_jepa_ood = probes.held_out_probe(Z_tr_jepa, C_tr, Z_ood_jepa, C_ood, seed=seed)

    r2_jepa_box = probe_jepa_box.r2
    r2_jepa_ood = probe_jepa_ood.r2
    jepa_ood_drop = r2_jepa_box - r2_jepa_ood

    # Floor / margin on the box split
    raw_tr = probes.raw_event_summary(data["train_X1"])
    raw_te = probes.raw_event_summary(data["test_X1"])
    probe_jepa_floor = probes.probe_with_floor(
        Z_tr_jepa, C_tr, Z_te_jepa, C_te, raw_tr, raw_te, seed=seed
    )

    # ------------------------------------------------------------------
    # 3.  Supervised ablation (same arch, trained to regress c)
    # ------------------------------------------------------------------
    sup_enc = DeepSetsEncoder(d_out=D_Z, d_emb=D_EMB, hidden=HIDDEN)
    C_tr_t = torch.tensor(C_tr)
    train_supervised(sup_enc, X1_tr, C_tr_t, seed=seed)

    sup_enc.eval()
    with torch.no_grad():
        Z_tr_sup = sup_enc(X1_tr).numpy()
        Z_te_sup = sup_enc(X1_te).numpy()
        Z_ood_sup = sup_enc(X1_ood).numpy()

    probe_sup_box = probes.held_out_probe(Z_tr_sup, C_tr, Z_te_sup, C_te, seed=seed)
    probe_sup_ood = probes.held_out_probe(Z_tr_sup, C_tr, Z_ood_sup, C_ood, seed=seed)

    r2_sup_box = probe_sup_box.r2
    r2_sup_ood = probe_sup_ood.r2
    sup_ood_drop = r2_sup_box - r2_sup_ood

    # ------------------------------------------------------------------
    # 4.  Assemble output
    # ------------------------------------------------------------------
    n_params_jepa = count_params(jepa_enc)
    n_params_sup = count_params(sup_enc)

    result = {
        "cell": "jepa_ood_robustness",
        "fm_family": "Contrastive / JEPA joint-embedding",
        "hep_task": (
            "View-invariant event-set representation (two re-simulation views "
            "per c), tested for out-of-distribution robustness: trained on the "
            "in-distribution Wilson box, probed on an extrapolation shell."
        ),
        "metric_primary": {
            "name": "OOD retention: JEPA frozen-probe c-R2 on extrapolation shell",
            "value": float(r2_jepa_ood),
        },
        "metrics": {
            "jepa_r2_box": float(r2_jepa_box),
            "jepa_r2_ood": float(r2_jepa_ood),
            "jepa_ood_drop": float(jepa_ood_drop),
            "sup_r2_box": float(r2_sup_box),
            "sup_r2_ood": float(r2_sup_ood),
            "sup_ood_drop": float(sup_ood_drop),
            "jepa_floor_r2": float(probe_jepa_floor.floor_r2),
            "jepa_margin": float(probe_jepa_floor.margin),
            "jepa_final_loss": float(jepa_losses[-1]),
            "jepa_probe_d_z": probe_jepa_box.d_z,
            "jepa_probe_alpha": probe_jepa_box.alpha,
        },
        "ablation_isolated": (
            "view-invariance pretraining -> OOD robustness "
            "(JEPA OOD drop vs supervised-encoder OOD drop)"
        ),
        "n_params": n_params_jepa,
        "n_params_sup": n_params_sup,
        "wall_seconds": float(time.time() - t0),
        "config": {
            "d_z": D_Z,
            "d_emb": D_EMB,
            "hidden": HIDDEN,
            "steps_jepa": STEPS_JEPA,
            "steps_sup": STEPS_SUP,
            "lr": LR,
            "lam_sim": LAM_SIM,
            "lam_var": LAM_VAR,
            "lam_cov": LAM_COV,
            "n_train": int(X1_tr.shape[0]),
            "n_events": int(X1_tr.shape[1]),
        },
    }

    OUT.mkdir(exist_ok=True)
    with open(OUT / "cell_jepa.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(
        {k: r[k] for k in (
            "cell", "metric_primary", "metrics", "n_params", "wall_seconds"
        )},
        indent=2,
    ))
