r"""Capability cell: Masked / denoising SSL -> label efficiency.

FM family: Masked / denoising SSL (analogue of ESM-2 / Masked Particle Modeling).
HEP task: Self-supervised masked event-set modeling, then frozen-backbone
Wilson-coefficient recovery at varying label budgets.

Objective tested: does self-supervised pretraining buy LABEL EFFICIENCY vs
training from scratch?  The SSL encoder is pretrained c-agnostically (Wilson
coefficients never enter the forward pass), then frozen.  A linear probe is fit
at each budget B.  The ablation contrast is an identical DeepSetsEncoder trained
from scratch with supervised regression at each budget B (re-init, short Adam
run), holding architecture and budget identical.

SSL task: masked set modeling.  For each event set we randomly split its 160
events into a CONTEXT subset (~60%) and a MASKED subset (~40%).  Encode the
context subset with DeepSetsEncoder -> context vector z.  Predict a 2D
histogram of the masked events over a 6x6 grid of (log m, cos theta), trained
with MSE on the normalized histogram counts.  This is set-level masked modeling:
reconstruct the held-out-event distribution from context.
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
from nets import DeepSetsEncoder, count_params, set_seed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"

# Histogram grid parameters (fixed from pooled train quantiles).
_N_BINS = 6
# Edges computed once from train data in run(); stored as module globals after
# build so the _make_histogram function is pure given fixed edges.
_EDGES_M: np.ndarray | None = None
_EDGES_C: np.ndarray | None = None


def _make_histogram(X_np: np.ndarray) -> np.ndarray:
    """2D normalised histogram of event set X_np (N, 2) on the fixed grid.

    Returns a flat vector of length _N_BINS**2 (float32), normalised so it
    sums to 1 (treating it as a discrete distribution over cells).
    """
    assert _EDGES_M is not None and _EDGES_C is not None, "call _init_edges first"
    h, _ = np.histogramdd(X_np, bins=[_EDGES_M, _EDGES_C])
    h = h.astype(np.float32)
    total = h.sum()
    if total > 0:
        h /= total
    return h.ravel()


def _init_edges(X_all: np.ndarray) -> None:
    """Set global histogram edges from pooled training events (quantile-based)."""
    global _EDGES_M, _EDGES_C
    m_flat = X_all[:, :, 0].ravel()
    c_flat = X_all[:, :, 1].ravel()
    m_lo, m_hi = float(np.quantile(m_flat, 0.01)), float(np.quantile(m_flat, 0.99))
    c_lo, c_hi = float(np.quantile(c_flat, 0.01)), float(np.quantile(c_flat, 0.99))
    # Add small margin so outermost events don't fall outside the outermost bin.
    pad_m = (m_hi - m_lo) * 0.05
    pad_c = (c_hi - c_lo) * 0.05
    _EDGES_M = np.linspace(m_lo - pad_m, m_hi + pad_m, _N_BINS + 1)
    _EDGES_C = np.linspace(c_lo - pad_c, c_hi + pad_c, _N_BINS + 1)


def _build_histogram_targets(X_np: np.ndarray) -> np.ndarray:
    """Build histogram targets for every scenario in X_np (n, N, 2).

    Returns (n, _N_BINS**2) float32 array.
    """
    n = X_np.shape[0]
    targets = np.empty((n, _N_BINS ** 2), dtype=np.float32)
    for i in range(n):
        targets[i] = _make_histogram(X_np[i])
    return targets


# --------------------------------------------------------------------------
# SSL pretraining model: context encoder -> histogram predictor.
# --------------------------------------------------------------------------

class SSLMaskedModel(nn.Module):
    """Set-masked modeling: encode context -> predict histogram of masked events.

    Architecture: DeepSetsEncoder(context) -> head MLP -> softmax -> histogram.
    """

    def __init__(self, d_z: int = 32, d_emb: int = 48, hidden: int = 64,
                 n_bins: int = _N_BINS):
        super().__init__()
        self.enc = DeepSetsEncoder(d_out=d_z, d_emb=d_emb, hidden=hidden)
        n_out = n_bins ** 2
        # head: z -> distribution over histogram cells
        self.head = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_out),
        )
        self.d_z = d_z

    def forward(self, X_ctx):
        """X_ctx: (..., N_ctx, 2) -> log_probs (..., n_bins^2)."""
        z = self.enc(X_ctx)
        logits = self.head(z)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def encode(self, X):
        """Encode full event set X (..., N, 2) -> (..., d_z)."""
        self.eval()
        return self.enc(X)


def _ssl_loss(log_probs: torch.Tensor, hist_target: torch.Tensor) -> torch.Tensor:
    """KL(target || pred) = -sum(target * log_pred) for normalised histograms.

    Equivalent to cross-entropy when target sums to 1.  Using MSE on
    sqrt(histogram) (Hellinger-like) is also sensible; we use cross-entropy
    (KL-div formulation) which is standard for distribution matching.
    """
    # hist_target: (B, n_out) normalised, sums to ~1
    # Avoid log(0) by flooring target contribution where target=0.
    return -(hist_target * log_probs).sum(dim=-1).mean()


def _split_context_masked(X: torch.Tensor, mask_frac: float = 0.4,
                           rng: np.random.Generator = None):
    """Split each set in X (B, N, 2) into context and masked subsets.

    Returns X_ctx (B, N_ctx, 2), X_msk (B, N_msk, 2) with shuffled but
    consistent per-batch-item splits.  Uses numpy rng for reproducibility.
    """
    B, N, D = X.shape
    N_msk = max(1, int(round(mask_frac * N)))
    N_ctx = N - N_msk

    X_ctx_list, X_msk_list = [], []
    for b in range(B):
        if rng is not None:
            perm = rng.permutation(N)
        else:
            perm = np.random.permutation(N)
        ctx_idx = torch.as_tensor(perm[:N_ctx], dtype=torch.long)
        msk_idx = torch.as_tensor(perm[N_ctx:], dtype=torch.long)
        X_ctx_list.append(X[b][ctx_idx])
        X_msk_list.append(X[b][msk_idx])

    return torch.stack(X_ctx_list), torch.stack(X_msk_list)


def train_ssl(model: SSLMaskedModel, Xtr: torch.Tensor,
              *, steps: int = 800, bs: int = 32, lr: float = 3e-3,
              mask_frac: float = 0.4, seed: int = 0) -> SSLMaskedModel:
    """Self-supervised pretraining: masked set modeling, c-agnostic."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    model.train()

    for step in range(steps):
        idx = rng.choice(n, bs, replace=False)
        X_batch = Xtr[idx]  # (bs, 160, 2)
        X_ctx, X_msk = _split_context_masked(X_batch, mask_frac, rng)

        # Target histogram from the masked events.
        hist_np = _build_histogram_targets(X_msk.numpy())
        hist_t = torch.as_tensor(hist_np)

        log_probs = model(X_ctx)
        loss = _ssl_loss(log_probs, hist_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0:
            print(f"  [SSL] step {step+1}/{steps}  loss={loss.item():.4f}", flush=True)

    return model


# --------------------------------------------------------------------------
# Supervised scratch training (ablation contrast).
# --------------------------------------------------------------------------

class ScratchRegressor(nn.Module):
    """Identical DeepSetsEncoder + linear head, trained supervised -> c."""

    def __init__(self, n_wc: int = 4, d_z: int = 32, d_emb: int = 48,
                 hidden: int = 64):
        super().__init__()
        self.enc = DeepSetsEncoder(d_out=d_z, d_emb=d_emb, hidden=hidden)
        self.head = nn.Linear(d_z, n_wc)

    def forward(self, X):
        return self.head(self.enc(X))

    @torch.no_grad()
    def encode(self, X):
        self.enc.eval()
        return self.enc(X)


def train_scratch(model: ScratchRegressor, Xtr: torch.Tensor, Ctr: torch.Tensor,
                  *, steps: int = 600, bs: int = 32, lr: float = 3e-3,
                  seed: int = 0) -> ScratchRegressor:
    """Supervised regression from scratch on a budget of B scenarios."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    model.train()
    loss_fn = nn.MSELoss()

    for step in range(steps):
        idx = rng.choice(n, min(bs, n), replace=(n < bs))
        pred = model(Xtr[idx])
        loss = loss_fn(pred, Ctr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


# --------------------------------------------------------------------------
# Main run.
# --------------------------------------------------------------------------

def run(seed: int = 0) -> dict:
    t0 = time.time()
    data = sub.load_cache()
    cfg = sub.CONFIG

    Xtr_np = data["train_X1"]  # (450, 160, 2)
    Ctr_np = data["train_c"]   # (450, 4)
    Xte_np = data["test_X1"]   # (120, 160, 2)
    Cte_np = data["test_c"]    # (120, 4)

    Xval_np = data["val_X1"]  # (120, 160, 2) — unlabelled for SSL

    # Initialise histogram edges from training data.
    _init_edges(Xtr_np)

    Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
    Ctr = torch.tensor(Ctr_np, dtype=torch.float32)
    Xte = torch.tensor(Xte_np, dtype=torch.float32)

    # Pool train + val for SSL pretraining (c-agnostic: no labels needed).
    Xssl = torch.cat([Xtr, torch.tensor(Xval_np, dtype=torch.float32)], dim=0)

    # ------------------------------------------------------------------ #
    # 1. SSL pretraining (c-agnostic, uses train_X1 + val_X1).
    # ------------------------------------------------------------------ #
    set_seed(seed)
    d_z = 48
    ssl_model = SSLMaskedModel(d_z=d_z, d_emb=64, hidden=80)
    print("=== SSL pretraining ===", flush=True)
    ssl_model = train_ssl(ssl_model, Xssl, steps=1200, bs=32, lr=3e-3,
                          mask_frac=0.4, seed=seed)

    # ------------------------------------------------------------------ #
    # 2. Freeze SSL encoder; extract representations.
    # ------------------------------------------------------------------ #
    ssl_model.eval()
    with torch.no_grad():
        Zssl_tr = ssl_model.encode(Xtr).numpy()  # (450, d_z)
        Zssl_te = ssl_model.encode(Xte).numpy()  # (120, d_z)

    # ------------------------------------------------------------------ #
    # 3. Label-efficiency curve: frozen SSL probe at each budget.
    # ------------------------------------------------------------------ #
    budgets = [25, 50, 100, 200, 450]
    ssl_r2_by_budget = []
    scratch_r2_by_budget = []

    print("=== label-efficiency probe loop ===", flush=True)

    for B in budgets:
        # Frozen SSL probe.
        p = probes.held_out_probe(
            Zssl_tr[:B], Ctr_np[:B],
            Zssl_te, Cte_np,
            seed=seed,
        )
        ssl_r2_by_budget.append(float(p.r2))

        # ---------------------------------------------------------------- #
        # 4. Scratch regressor at the same budget (re-init + supervised).
        # ---------------------------------------------------------------- #
        scratch = ScratchRegressor(n_wc=cfg.n_wc, d_z=d_z, d_emb=64, hidden=80)
        train_scratch(scratch, Xtr[:B], Ctr[:B],
                      steps=600, bs=min(32, B), lr=3e-3, seed=seed)
        scratch.eval()
        with torch.no_grad():
            Z_sc_tr = scratch.encode(Xtr[:B]).numpy()
            Z_sc_te = scratch.encode(Xte).numpy()
        p_sc = probes.held_out_probe(
            Z_sc_tr, Ctr_np[:B],
            Z_sc_te, Cte_np,
            seed=seed,
        )
        scratch_r2_by_budget.append(float(p_sc.r2))
        print(f"  B={B:4d}  SSL-R2={ssl_r2_by_budget[-1]:.3f}  "
              f"SCRATCH-R2={scratch_r2_by_budget[-1]:.3f}", flush=True)

    ssl_minus_scratch = [float(a - b)
                         for a, b in zip(ssl_r2_by_budget, scratch_r2_by_budget)]

    # AUC over budgets (trapezoid, normalised by budget range for interpretability).
    bud_arr = np.array(budgets, float)
    auc_ssl = float(np.trapezoid(ssl_r2_by_budget, bud_arr) / (bud_arr[-1] - bud_arr[0]))
    auc_scr = float(np.trapezoid(scratch_r2_by_budget, bud_arr) / (bud_arr[-1] - bud_arr[0]))

    # ------------------------------------------------------------------ #
    # 5. probe_with_floor at full budget (B=450) for the common matrix column.
    # ------------------------------------------------------------------ #
    raw_tr = probes.raw_event_summary(Xtr_np)
    raw_te = probes.raw_event_summary(Xte_np)
    full_probe = probes.probe_with_floor(
        Zssl_tr, Ctr_np, Zssl_te, Cte_np,
        raw_tr, raw_te, seed=seed,
    )

    # ------------------------------------------------------------------ #
    # 6. Assemble result.
    # ------------------------------------------------------------------ #
    n_params_ssl = count_params(ssl_model)
    n_params_scratch = count_params(ScratchRegressor(n_wc=cfg.n_wc, d_z=d_z,
                                                      d_emb=64, hidden=80))

    result = {
        "cell": "ssl_label_efficiency",
        "fm_family": "Masked / denoising SSL",
        "hep_task": ("Self-supervised masked event-set modeling, then "
                     "frozen-backbone Wilson-coefficient recovery at varying "
                     "label budgets."),
        "metric_primary": {
            "name": "label-efficiency gain (SSL - from-scratch c-R2) at B=25",
            "value": ssl_minus_scratch[0],   # index 0 = B=25
        },
        "metrics": {
            "budgets": budgets,
            "ssl_r2_by_budget": ssl_r2_by_budget,
            "scratch_r2_by_budget": scratch_r2_by_budget,
            "ssl_minus_scratch_by_budget": ssl_minus_scratch,
            "label_efficiency_auc_ssl": auc_ssl,
            "label_efficiency_auc_scratch": auc_scr,
            "probe_floor_r2": float(full_probe.floor_r2),
            "probe_margin": float(full_probe.margin),
            "probe_r2_full_budget": float(full_probe.r2),
        },
        "ablation_isolated": (
            "self-supervised pretraining -> label efficiency "
            "(frozen-SSL vs from-scratch at matched budgets)"
        ),
        "n_params": n_params_ssl,
        "wall_seconds": time.time() - t0,
        "config": {
            "d_z": d_z,
            "n_bins": _N_BINS,
            "mask_frac": 0.4,
            "ssl_steps": 1200,
            "ssl_n_scenarios": int(Xssl.shape[0]),
            "scratch_steps": 600,
            "n_train": int(Xtr.shape[0]),
            "n_events": cfg.n_events,
            "n_params_scratch_ref": n_params_scratch,
        },
    }

    OUT.mkdir(exist_ok=True)
    with open(OUT / "cell_ssl.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(
        {k: r[k] for k in ("cell", "metric_primary", "metrics",
                            "n_params", "wall_seconds")},
        indent=2,
    ))
