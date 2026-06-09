r"""Capability cell: Anomaly detection -> model-independent new-physics search.

FM family: Anomaly detection (objective-agnostic) — analogue of the LHC Olympics
anomaly challenge. The model is an autoencoder trained *only* on Standard-Model
events; it scores event sets by their mean per-event reconstruction error. BSM
events (higher-mass / shifted-angular distributions) are harder to reconstruct,
producing a larger anomaly score.

HEP task: Unsupervised new-physics detection. Train only on Standard-Model events;
score event sets (SM or BSM) by anomaly; separate SM from BSM without ever seeing
BSM in training.

Objective tested: Unsupervised SM-only training detecting BSM across two difficulty
tiers — near-SM box (|c| <= 0.6) and far shell (|c| in 1.1–1.5). The shell tier
should be easier (higher AUC) because those Wilson points are further from SM.

Ablation isolated: unsupervised SM-only anomaly detection across difficulty tiers
(near-SM box vs far shell); shell should be easier (higher AUC).
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
# Autoencoder: small bottleneck on the 2D event space.
# --------------------------------------------------------------------------

class EventAutoencoder(nn.Module):
    """Per-event autoencoder: 2 -> hidden -> bottleneck -> hidden -> 2.

    Trained with MSE on standardized SM events. BSM events have shifted
    mass/angular distributions -> higher reconstruction error.
    """

    def __init__(self, hidden: int = 16, bottleneck: int = 1):
        super().__init__()
        self.encoder = mlp([2, hidden, bottleneck], last_act=True)
        self.decoder = mlp([bottleneck, hidden, 2])

    def forward(self, x):
        """x: (..., 2) -> recon: (..., 2)"""
        z = self.encoder(x)
        return self.decoder(z)

    def recon_error(self, x):
        """Per-event MSE reconstruction error: (..., 2) -> (...,)"""
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=-1)


# --------------------------------------------------------------------------
# Training on pooled SM events (standardized).
# --------------------------------------------------------------------------

def fit_standardizer(X_flat: np.ndarray):
    """Compute mean and std from (N, 2) flat events. Returns (mu, sigma)."""
    mu = X_flat.mean(axis=0)
    sigma = X_flat.std(axis=0).clip(1e-8)
    return mu, sigma


def train_ae(
    model: EventAutoencoder,
    X_train_flat: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    steps: int = 1500,
    bs: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
) -> EventAutoencoder:
    """Train autoencoder on standardized pooled SM events."""
    set_seed(seed)
    # Standardize
    X_std = (X_train_flat - mu) / sigma           # (N, 2)
    X_t = torch.tensor(X_std, dtype=torch.float32)
    n = X_t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        idx = rng.choice(n, min(bs, n), replace=False)
        x_batch = X_t[idx]
        loss = model.recon_error(x_batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# --------------------------------------------------------------------------
# Per-set anomaly scoring.
# --------------------------------------------------------------------------

@torch.no_grad()
def score_sets(
    model: EventAutoencoder,
    X_sets: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Mean per-event reconstruction error per set.

    X_sets: (n_sets, n_events, 2)
    Returns: (n_sets,) float64
    """
    model.eval()
    # Standardize and score all sets in one vectorized forward pass.
    X_std = (X_sets - mu) / sigma                      # (n_sets, n_events, 2)
    X_t = torch.tensor(X_std, dtype=torch.float32)     # (n_sets, n_events, 2)
    err = model.recon_error(X_t)                        # (n_sets, n_events)
    return err.mean(dim=-1).numpy().astype(np.float64)  # (n_sets,)


# --------------------------------------------------------------------------
# Main run.
# --------------------------------------------------------------------------

def run(seed: int = 0) -> dict:
    t0 = time.time()

    data = sub.load_cache()

    # SM bank: (250, 160, 2)
    sm_X1 = data["sm_X1"]                        # (250, 160, 2)
    # BSM test sets
    test_X1 = data["test_X1"]                    # (120, 160, 2) box |c|<=0.6
    ood_X1 = data["ood_X1"]                      # (120, 160, 2) shell |c| in 1.1-1.5

    # Split SM bank: 180 train / 70 held-out background.
    n_sm = sm_X1.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sm)
    n_train_sm = 180
    idx_train = perm[:n_train_sm]
    idx_bg    = perm[n_train_sm:]                # 70 held-out SM sets

    sm_train_sets = sm_X1[idx_train]             # (180, 160, 2)
    sm_bg_sets    = sm_X1[idx_bg]               # (70, 160, 2)

    # Pool training events: (180*160, 2)
    X_train_flat = sm_train_sets.reshape(-1, 2)

    # Standardizer from SM-train events only.
    mu, sigma = fit_standardizer(X_train_flat)

    # Build and train autoencoder.
    model = EventAutoencoder(hidden=16, bottleneck=1)
    set_seed(seed)
    train_ae(model, X_train_flat, mu, sigma, steps=1500, bs=1024, lr=1e-3, seed=seed)

    # Score held-out SM background sets (70 sets).
    bg_scores   = score_sets(model, sm_bg_sets, mu, sigma)   # (70,)
    # Score box BSM sets (120 sets — near-SM, harder).
    box_scores  = score_sets(model, test_X1,    mu, sigma)   # (120,)
    # Score shell BSM sets (120 sets — far from SM, easier).
    shell_scores = score_sets(model, ood_X1,    mu, sigma)   # (120,)

    # AUC (higher score = more anomalous, so these are correct orientation).
    auc_box   = probes.roc_auc(bg_scores, box_scores)
    auc_shell = probes.roc_auc(bg_scores, shell_scores)

    # Significance at 1% background working point.
    sig_box   = probes.significance_at_background(bg_scores, box_scores,   bg_eff=0.01)
    sig_shell = probes.significance_at_background(bg_scores, shell_scores, bg_eff=0.01)

    n_params = count_params(model)
    wall = time.time() - t0

    result = {
        "cell": "anomaly_detection",
        "fm_family": "Anomaly detection (objective-agnostic)",
        "hep_task": (
            "Unsupervised new-physics detection: train only on Standard-Model events, "
            "score event sets by anomaly, separate SM from BSM without ever seeing "
            "BSM in training."
        ),
        "metric_primary": {
            "name": "anomaly ROC AUC, SM vs box-BSM (held-out)",
            "value": float(auc_box),
        },
        "metrics": {
            "auc_box":           float(auc_box),
            "auc_shell":         float(auc_shell),
            "significance_box":  float(sig_box["significance"]),
            "significance_shell":float(sig_shell["significance"]),
            "sig_eff_box":       float(sig_box["sig_eff"]),
            "sig_eff_shell":     float(sig_shell["sig_eff"]),
            "n_bg":              int(len(bg_scores)),
            "n_sig":             int(len(box_scores)),
        },
        "ablation_isolated": (
            "unsupervised SM-only anomaly detection across difficulty tiers "
            "(near-SM box vs far shell); shell should be easier (higher AUC)"
        ),
        "n_params": n_params,
        "wall_seconds": float(wall),
        "config": {
            "n_sm_train_sets":  n_train_sm,
            "n_sm_bg_sets":     int(len(bg_scores)),
            "n_events_per_set": int(sm_X1.shape[1]),
            "ae_hidden":        16,
            "ae_bottleneck":    1,
            "ae_steps":         1500,
            "ae_lr":            1e-3,
        },
    }

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "cell_anomaly.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(
        {k: r[k] for k in ("cell", "metric_primary", "metrics", "n_params", "wall_seconds")},
        indent=2,
    ))
