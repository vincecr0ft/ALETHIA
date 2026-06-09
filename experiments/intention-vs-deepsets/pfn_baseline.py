"""Prior-Fitted-Network baseline for the Intention / DeepSets comparison.

Implements a small transformer in-context-learning (ICL) head trained
on the same SMEFT scenario prior as `experiment_smeft.py`. The forward
pass receives a sequence of context tokens (m, y, is_query=0) followed
by query tokens (m, 0, is_query=1) and predicts the y-value at the
query positions. This is the "PFN" architecture of Mueller et al. 2022
(arXiv:2112.10510), specialised to a 1D regression target.

Why a PFN baseline: the ML audit (docs/research/audit-ml.md) identifies
this as the load-bearing missing experiment — without it, the paper
cannot argue that the closed-form-ridge head in Intention is doing
work that a generic ICL transformer cannot. With it, Section 7 either:

(a) Intention beats PFN cleanly  → closed-form-ridge framing holds
(b) PFN matches Intention        → reframe ALETHIA as the closed-form
                                   specialisation of PFNs on a physics
                                   regression target.

Both outcomes strengthen the paper.

Constraint: as with all Intention/DeepSets variants, the model must NOT
see c on the forward pass.  c lives only inside the data generator.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/intention-vs-deepsets/pfn_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn

from data_smeft import (
    make_oracle, sample_c, sample_c_shell,
    make_dataset, scenarios_to_tensors,
    N_WC, M_RANGE,
)


OUTPUT_DIR = HERE / "output_pfn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration — matched to experiment_smeft.py protocol.
N_TRAIN_SCENARIOS = 200
N_TEST_IN = 50
N_TEST_OUT = 50
K_CTX = 12
Q_QUERY = 32
C_MAX_TRAIN = 0.7
C_OUTER = 1.0
N_META_STEPS = 1500
LR = 1e-3
BATCH_S = 32
SEED = 0
M_REF = 1.0   # TeV reference for log-m feature

# Tiny transformer config. Target ~5000-6000 params to match the
# 5328-param IntentionFM_Learned headline. d_model must be divisible
# by nhead.
D_MODEL = 16
N_HEAD = 4
DIM_FF = 40
N_LAYERS = 2


class PFNHead(nn.Module):
    """Transformer ICL head over an (M_ctx, Y_ctx, M_q) sequence.

    Tokenisation: each context point is a 3-vector (log_m, y, is_query=0)
    and each query point is (log_m, 0.0, is_query=1). The transformer
    encoder processes the full (K + Q)-length sequence in one pass; the
    output head reads off the last Q positions and projects to a scalar
    y_hat per query.

    No positional encoding — the underlying regression task is
    permutation-invariant in (M_ctx, Y_ctx) pairs and permutation-
    invariant across queries. The (m, y, is_query) features carry the
    only ordering information the model needs.
    """

    def __init__(self, d_model: int = D_MODEL, nhead: int = N_HEAD,
                 dim_feedforward: int = DIM_FF, n_layers: int = N_LAYERS):
        super().__init__()
        self.in_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                M_q: torch.Tensor) -> torch.Tensor:
        # Shapes: (B, K), (B, K), (B, Q)  ->  (B, Q)
        B, K = M_ctx.shape
        Q = M_q.shape[1]
        # Use log(m / M_REF) as the m-feature so the encoder sees the same
        # scale that the rest of the codebase uses.
        log_m_ctx = torch.log(M_ctx / M_REF)
        log_m_q = torch.log(M_q / M_REF)
        ctx_tokens = torch.stack([
            log_m_ctx, Y_ctx, torch.zeros_like(log_m_ctx)
        ], dim=-1)                                          # (B, K, 3)
        q_tokens = torch.stack([
            log_m_q, torch.zeros_like(log_m_q), torch.ones_like(log_m_q)
        ], dim=-1)                                          # (B, Q, 3)
        seq = torch.cat([ctx_tokens, q_tokens], dim=1)      # (B, K+Q, 3)
        seq = self.in_proj(seq)                             # (B, K+Q, d)
        out = self.encoder(seq)                             # (B, K+Q, d)
        y_q = self.out_proj(out[:, K:, :]).squeeze(-1)      # (B, Q)
        return y_q


def r2_per_scenario(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def mse_total(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def summarise(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    r2 = r2_per_scenario(y_pred, y_true)
    return {
        "name": name,
        "r2_median": float(np.median(r2)),
        "r2_p5": float(np.percentile(r2, 5)),
        "r2_mean": float(np.mean(r2)),
        "mse": mse_total(y_pred, y_true),
        "n_scenarios": int(len(r2)),
        "r2_per_scenario": r2.tolist(),
    }


def train(model: PFNHead, train_t, val_t, n_steps, lr, batch_s, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)
    losses = []
    val_hist = []
    log_steps = {1, 10, 25, 50, 100, 200, 400, 600, 1000, 1500}
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(S, size=batch_s, replace=False)
        M_ctx = train_t["M_ctx"][idx]
        Y_ctx = train_t["Y_ctx"][idx]
        M_q = train_t["M_query"][idx]
        Y_q = train_t["Y_query"][idx]
        yp = model(M_ctx, Y_ctx, M_q)
        loss = ((yp - Y_q) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step in log_steps:
            model.eval()
            with torch.no_grad():
                yp_val = model(val_t["M_ctx"], val_t["Y_ctx"],
                               val_t["M_query"]).cpu().numpy()
            r2_val = r2_per_scenario(
                yp_val, val_t["Y_query"].cpu().numpy())
            val_hist.append((step, float(np.median(r2_val))))
            print(f"  [pfn      ] step {step:5d}  loss={loss.item():.6f}  "
                  f"val_R2_median={np.median(r2_val):+.4f}")
            model.train()
    return losses, val_hist, time.time() - t0


def main():
    print("# PFN baseline — SMEFT analytic oracle")
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train_scen = make_dataset(
        N_TRAIN_SCENARIOS,
        lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
        oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=1)
    test_in_scen = make_dataset(
        N_TEST_IN,
        lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
        oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=42)
    test_out_scen = make_dataset(
        N_TEST_OUT,
        lambda r: sample_c_shell(r, 1, c_inner=C_MAX_TRAIN,
                                 c_outer=C_OUTER)[0],
        oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=43)
    train_t = scenarios_to_tensors(train_scen)
    test_in_t = scenarios_to_tensors(test_in_scen)
    test_out_t = scenarios_to_tensors(test_out_scen)

    model = PFNHead()
    n_params = model.n_params
    print(f"  PFN n_params = {n_params}")
    losses, hist, wall = train(
        model, train_t, test_in_t, N_META_STEPS, LR, BATCH_S, seed=SEED)
    print(f"  trained in {wall:.1f}s")
    model.eval()
    with torch.no_grad():
        yp_in = model(test_in_t["M_ctx"], test_in_t["Y_ctx"],
                      test_in_t["M_query"]).cpu().numpy()
        yp_out = model(test_out_t["M_ctx"], test_out_t["Y_ctx"],
                       test_out_t["M_query"]).cpu().numpy()

    res_in = summarise("PFN", yp_in,
                       test_in_t["Y_query"].cpu().numpy())
    res_out = summarise("PFN", yp_out,
                        test_out_t["Y_query"].cpu().numpy())

    print(f"  in : median R2 = {res_in['r2_median']:+.4f}  "
          f"p5 = {res_in['r2_p5']:+.4f}")
    print(f"  out: median R2 = {res_out['r2_median']:+.4f}  "
          f"p5 = {res_out['r2_p5']:+.4f}")

    summary = dict(
        config=dict(
            d_model=D_MODEL, n_head=N_HEAD, dim_feedforward=DIM_FF,
            n_layers=N_LAYERS, n_params=n_params,
            n_train=N_TRAIN_SCENARIOS, n_steps=N_META_STEPS,
            lr=LR, batch=BATCH_S, seed=SEED,
            K_ctx=K_CTX, Q_query=Q_QUERY,
            c_max_train=C_MAX_TRAIN, c_outer=C_OUTER,
        ),
        PFN=dict(
            in_=res_in, out=res_out, n_params=n_params, wall=wall,
            scaling_history=hist, losses=losses,
        ),
    )
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), OUTPUT_DIR / "pfn.pt")
    print(f"  wrote {OUTPUT_DIR / 'summary.json'}")
    print(f"  wrote {OUTPUT_DIR / 'pfn.pt'}")


if __name__ == "__main__":
    main()
