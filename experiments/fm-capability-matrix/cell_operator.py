r"""Capability cell: Neural-operator / DeepONet -> forward-map emulation.

FM family: Neural-operator / PDE surrogate (analogue of DeepONet / FIM-SDE).
HEP task: Emulate the SMEFT forward map c -> mu(c, m) profile over dilepton
mass m (a fast cross-section-ratio surrogate).

The Wilson coefficient c is the operator *input* here (legitimate — this is a
surrogate, not an inference task). The m coordinate plays the role of the
continuous query point in DeepONet.

Architecture:
  - Branch net: MLP(c; 4 -> 64 -> 64 -> p)  (encodes the "input function")
  - Trunk  net: MLP(m; 1 -> 64 -> 64 -> p)  (encodes the query coordinate)
  - Output:     dot(branch, trunk) + scalar_bias  -> scalar mu_hat(c, m)

Training: MSE on standardised targets over all 40 m-grid points per scenario.
Ablation: re-train withholding interior m-band [15..24], evaluate interpolation
R2 on that band for held-out test scenarios.
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
from nets import mlp, count_params, set_seed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"

# --------------------------------------------------------------------------
# DeepONet
# --------------------------------------------------------------------------

class DeepONet(nn.Module):
    """Tiny DeepONet: branch(c) · trunk(m) + bias -> scalar mu_hat."""

    def __init__(self, p: int = 32, hidden: int = 64):
        super().__init__()
        # branch: c (4-dim) -> p
        self.branch = mlp([4, hidden, hidden, p])
        # trunk: m scalar (1-dim) -> p
        self.trunk = mlp([1, hidden, hidden, p])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        c: (batch, 4)  — Wilson coefficients per scenario
        m: (n_m,)      — m-grid query points
        Returns: (batch, n_m)  mu_hat
        """
        b = self.branch(c)                      # (batch, p)
        t = self.trunk(m.unsqueeze(-1))         # (n_m, p)
        return b @ t.T + self.bias              # (batch, n_m)


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------

def build_tensors(c_np, mu_np):
    """Return float32 tensors for c and mu."""
    c = torch.tensor(c_np, dtype=torch.float32)
    mu = torch.tensor(mu_np, dtype=torch.float32)
    return c, mu


def train_model(
    model: DeepONet,
    c_tr: torch.Tensor,
    mu_tr: torch.Tensor,
    m_grid: torch.Tensor,
    *,
    mu_mean: float,
    mu_std: float,
    steps: int = 2000,
    bs: int = 64,
    lr: float = 3e-3,
    m_mask: np.ndarray | None = None,
    seed: int = 0,
) -> DeepONet:
    """MSE training on (optionally masked) m-grid points.

    m_mask: boolean array of shape (n_m,); if given, only those m points
    are used during training (for the ablation).
    """
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    n = c_tr.shape[0]
    rng = np.random.default_rng(seed)

    # Standardise targets
    mu_tr_std = (mu_tr - mu_mean) / mu_std           # (n_scenarios, n_m)

    # Build masked m-grid tensor if needed
    if m_mask is not None:
        m_use = m_grid[m_mask]                        # (n_m_train,)
        mu_use = mu_tr_std[:, m_mask]                 # (n_scen, n_m_train)
    else:
        m_use = m_grid
        mu_use = mu_tr_std

    model.train()
    for step in range(steps):
        idx = rng.choice(n, min(bs, n), replace=False)
        c_b = c_tr[idx]          # (bs, 4)
        mu_b = mu_use[idx]       # (bs, n_m_use)

        pred = model(c_b, m_use)  # (bs, n_m_use)
        loss = ((pred - mu_b) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    return model


@torch.no_grad()
def predict(model: DeepONet, c: torch.Tensor, m_grid: torch.Tensor,
            mu_mean: float, mu_std: float) -> np.ndarray:
    """Return predictions in original (unstandardised) units."""
    model.eval()
    pred_std = model(c, m_grid)          # (n, n_m)
    return (pred_std.numpy() * mu_std + mu_mean)


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------

def run(seed: int = 0) -> dict:
    t0 = time.time()

    data = sub.load_cache()
    cfg = sub.CONFIG

    # ---- raw arrays ----
    train_c = data["train_c"].astype(np.float32)          # (450, 4)
    test_c  = data["test_c"].astype(np.float32)           # (120, 4)
    m_grid_np = data["m_grid"].astype(np.float32)         # (40,)
    train_mu = data["train_mu"].astype(np.float32)        # (450, 40)
    test_mu  = data["test_mu"].astype(np.float32)         # (120, 40)

    n_m = len(m_grid_np)

    # Standardise using TRAIN statistics
    mu_mean = float(train_mu.mean())
    mu_std  = float(train_mu.std()) + 1e-8

    # Tensors
    c_tr = torch.tensor(train_c)
    c_te = torch.tensor(test_c)
    mu_tr = torch.tensor(train_mu)
    m_ten = torch.tensor(m_grid_np)

    # ================================================================
    # 1. FULL-GRID model
    # ================================================================
    set_seed(seed)
    model_full = DeepONet(p=32, hidden=64)
    print("Training full-grid DeepONet ...", flush=True)
    model_full = train_model(
        model_full, c_tr, mu_tr, m_ten,
        mu_mean=mu_mean, mu_std=mu_std,
        steps=2000, bs=64, lr=3e-3, seed=seed,
    )

    pred_full = predict(model_full, c_te, m_ten, mu_mean, mu_std)  # (120, 40)

    profile_r2_full = probes.r2_score(test_mu.flatten(), pred_full.flatten())

    # Wasserstein: treat each scenario's 40-point profile as a 1-D distribution.
    # Stack: real (n_test * n_m, 1) vs pred (n_test * n_m, 1) — use marginal_wasserstein
    # on (n_test, n_m) reshaped as (n_test * n_m, 1) so we get one W1 number.
    real_flat = test_mu.reshape(-1, 1).astype(np.float64)
    pred_flat = pred_full.reshape(-1, 1).astype(np.float64)
    w_result  = probes.marginal_wasserstein(real_flat, pred_flat)
    profile_w1_mean = w_result["w1_mean"]

    # Also optionally A_FB
    test_afb  = data["test_afb"].astype(np.float32)          # (120, 40)
    train_afb = data["train_afb"].astype(np.float32)         # (450, 40)
    afb_mean  = float(train_afb.mean())
    afb_std   = float(train_afb.std()) + 1e-8
    mu_afb_tr = torch.tensor(train_afb)

    set_seed(seed + 1)
    model_afb = DeepONet(p=32, hidden=64)
    print("Training A_FB DeepONet ...", flush=True)
    model_afb = train_model(
        model_afb, c_tr, mu_afb_tr, m_ten,
        mu_mean=afb_mean, mu_std=afb_std,
        steps=2000, bs=64, lr=3e-3, seed=seed + 1,
    )
    pred_afb = predict(model_afb, c_te, m_ten, afb_mean, afb_std)
    afb_profile_r2 = probes.r2_score(test_afb.flatten(), pred_afb.flatten())

    print(f"Full-grid profile R2 (mu): {profile_r2_full:.4f}")
    print(f"A_FB profile R2:           {afb_profile_r2:.4f}")
    print(f"Profile W1 mean:           {profile_w1_mean:.5f}")

    # ================================================================
    # 2. ABLATION: hold out interior m-band [15..24], train on rest
    # ================================================================
    holdout_lo, holdout_hi = 15, 25   # indices [15, 24] inclusive
    m_mask_train = np.ones(n_m, dtype=bool)
    m_mask_train[holdout_lo:holdout_hi] = False
    m_mask_holdout = ~m_mask_train   # [15..24]

    set_seed(seed)
    model_abl = DeepONet(p=32, hidden=64)
    print("Training ablation DeepONet (interior band withheld) ...", flush=True)
    model_abl = train_model(
        model_abl, c_tr, mu_tr, m_ten,
        mu_mean=mu_mean, mu_std=mu_std,
        steps=2000, bs=64, lr=3e-3,
        m_mask=m_mask_train, seed=seed,
    )

    # Evaluate ONLY on held-out interior m-band
    m_holdout_ten = m_ten[m_mask_holdout]             # (10,)
    pred_abl = predict(model_abl, c_te, m_holdout_ten, mu_mean, mu_std)  # (120, 10)

    true_holdout = test_mu[:, m_mask_holdout]          # (120, 10)
    r2_m_interpolation = probes.r2_score(
        true_holdout.flatten(), pred_abl.flatten()
    )

    print(f"Ablation R2 (held-out interior m-band): {r2_m_interpolation:.4f}")
    print(f"  m held-out range: [{m_grid_np[holdout_lo]:.3f}, {m_grid_np[holdout_hi - 1]:.3f}] TeV")

    wall = time.time() - t0

    result = {
        "cell": "operator_deeponet",
        "fm_family": "Neural-operator / PDE surrogate",
        "hep_task": (
            "Emulate the SMEFT forward map c -> mu(c,m) profile over dilepton "
            "mass m (fast cross-section-ratio surrogate)"
        ),
        "metric_primary": {
            "name": "held-out forward-map profile R2 (mu(c,m))",
            "value": float(profile_r2_full),
        },
        "metrics": {
            "profile_r2_full": float(profile_r2_full),
            "r2_m_interpolation": float(r2_m_interpolation),
            "profile_w1_mean": float(profile_w1_mean),
            "afb_profile_r2": float(afb_profile_r2),
        },
        "ablation_isolated": (
            "operator emulation generalizes across the continuous m coordinate "
            "(held-out interior m band [15..24], ~0.98..1.5 TeV)"
        ),
        "n_params": count_params(model_full),
        "wall_seconds": float(wall),
        "config": {
            "p_latent": 32,
            "hidden": 64,
            "steps": 2000,
            "batch_size": 64,
            "lr": 3e-3,
            "n_train": int(train_c.shape[0]),
            "n_test": int(test_c.shape[0]),
            "n_m_grid": int(n_m),
            "m_holdout_band": [int(holdout_lo), int(holdout_hi - 1)],
            "m_holdout_tev": [
                float(m_grid_np[holdout_lo]),
                float(m_grid_np[holdout_hi - 1]),
            ],
            "mu_mean": float(mu_mean),
            "mu_std": float(mu_std),
        },
    }

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "cell_operator.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written: {out_path}")

    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(
        {k: r[k] for k in ("cell", "metric_primary", "metrics",
                            "n_params", "wall_seconds")},
        indent=2,
    ))
