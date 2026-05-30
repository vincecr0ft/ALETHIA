"""Joint-score auxiliary loss for the Intention FM (U9, "Mining Gold" trick).

The Brehmer-Cranmer-Louppe-Pavez observation is that for any
likelihood-based simulator, the joint score
``s_i(x, c) = d log p(x | c) / d c_i`` is available analytically (or
near-analytically) at simulation time, even though the marginal
likelihood is intractable. Adding an MSE term that asks the model's
*derivative* with respect to ``c`` to match the analytic score is a
free supervision signal the headline MSE-on-mu loss throws away.

In our regression setting the analogue is direct. ``mu(c, m)`` is the
target; its analytic c-derivative
``d mu / d c_i`` at fixed ``m`` is available from the morphing
decomposition. We require that the IntentionFM's predicted ``mu``
exhibit the same c-sensitivity as the oracle, where ``c`` enters the
model only through the context labels ``Y_ctx = mu(c, M_ctx)``.

Implementation (central differences in c, end-to-end through the
ridge solve):

  c_plus  = c + eps * e_i
  c_minus = c - eps * e_i
  Y_ctx_plus  = oracle.truth(c_plus,  M_ctx)
  Y_ctx_minus = oracle.truth(c_minus, M_ctx)
  fwd_sens_i  = (model(M_ctx, Y_ctx_plus, M_q) -
                 model(M_ctx, Y_ctx_minus, M_q)) / (2 eps)
  true_sens_i = (oracle.truth(c_plus, M_q) -
                 oracle.truth(c_minus, M_q)) / (2 eps)
  L_score    += mse(fwd_sens_i, true_sens_i)

Total loss = L_mse + lambda_score * L_score. Extra cost is 4 * N_WC
extra oracle calls per scenario (8 per scenario at N_WC=4).

The resulting checkpoint is saved to a separate path so the headline
run's `output/intention_fm.pt` is not overwritten — this is an
ablation, not a replacement. Re-run `identifiability_probe.py` with
the OUT_DIR pointed at `output_score_aux/` to measure the disclosure
lift on the under-disclosed operators.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/full-chain-run/pretrain_score_aux.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM


SEED = 2026
N_WC = 4
WITHHOLD_DIM = 2               # c_lq^(3)
WITHHOLD_BAND = (0.6, 1.0)
C_TRAIN_BOX = 0.7
M_RANGE = (0.3, 2.3)

K_CTX = 12
Q_CTX = 32
N_PRETRAIN_SCEN = 1500
PRETRAIN_BATCH = 24
PRETRAIN_STEPS = 1500
LR = 1e-3

# Score-aux configuration.
LAMBDA_SCORE = float(os.environ.get("LAMBDA_SCORE", "0.5"))
FD_STEP = float(os.environ.get("FD_STEP", "1e-2"))
SCORE_EVERY = int(os.environ.get("SCORE_EVERY", "1"))  # apply every N steps

OUT = HERE / "output_score_aux"
OUT.mkdir(parents=True, exist_ok=True)


def sample_c_training(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def oracle_batch_labels(
    oracle: AnalyticSMEFTOracle, cs: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Vectorised mu(c, m) for a batch of (c, m) pairs.

    cs: (B, N_WC). M: (B, K). Returns (B, K) np.float32.
    """
    B, K = M.shape
    C = np.repeat(cs, K, axis=0)
    M_flat = M.reshape(-1)
    Y = oracle.truth(C, M_flat).reshape(B, K)
    return Y.astype(np.float32)


def score_targets(
    oracle: AnalyticSMEFTOracle,
    cs: np.ndarray, M_q: np.ndarray, *, eps: float,
) -> np.ndarray:
    """Analytic central-difference d mu / d c at the query points.

    cs: (B, N_WC). M_q: (B, Q). Returns (B, Q, N_WC) np.float32 —
    one sensitivity vector per (scenario, query, c-direction).
    """
    B, Q = M_q.shape
    out = np.empty((B, Q, N_WC), dtype=np.float32)
    for i in range(N_WC):
        cs_plus = cs.copy(); cs_plus[:, i] += eps
        cs_minus = cs.copy(); cs_minus[:, i] -= eps
        y_plus = oracle_batch_labels(oracle, cs_plus, M_q)
        y_minus = oracle_batch_labels(oracle, cs_minus, M_q)
        out[:, :, i] = (y_plus - y_minus) / (2.0 * eps)
    return out


def fwd_sens(
    model: IntentionFM, oracle: AnalyticSMEFTOracle,
    cs: np.ndarray, M_ctx: np.ndarray, M_q: torch.Tensor,
    *, eps: float,
) -> torch.Tensor:
    """Central-difference sensitivity of model prediction with respect
    to c, where c enters only through the context labels Y_ctx(c).

    Returns (B, Q, N_WC) Tensor, differentiable through psi_theta.
    """
    B, K = M_ctx.shape
    Q = M_q.shape[1]
    M_ctx_t = torch.from_numpy(M_ctx).float()
    out = []
    for i in range(N_WC):
        cs_plus = cs.copy(); cs_plus[:, i] += eps
        cs_minus = cs.copy(); cs_minus[:, i] -= eps
        Y_plus = oracle_batch_labels(oracle, cs_plus, M_ctx)
        Y_minus = oracle_batch_labels(oracle, cs_minus, M_ctx)
        Y_plus_t = torch.from_numpy(Y_plus).float()
        Y_minus_t = torch.from_numpy(Y_minus).float()
        y_plus = model(M_ctx_t, Y_plus_t, M_q)
        y_minus = model(M_ctx_t, Y_minus_t, M_q)
        out.append((y_plus - y_minus) / (2.0 * eps))
    return torch.stack(out, dim=-1)               # (B, Q, N_WC)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    opt = Adam(model.parameters(), lr=LR)

    c_pool = sample_c_training(N_PRETRAIN_SCEN, rng)

    print(f"Pretraining with score-aux: lambda={LAMBDA_SCORE} eps={FD_STEP} "
          f"every={SCORE_EVERY}")
    losses_mse = []
    losses_score = []
    losses_total = []
    t0 = time.time()
    for step in range(PRETRAIN_STEPS):
        idx = rng.choice(N_PRETRAIN_SCEN, size=PRETRAIN_BATCH, replace=False)
        cs = c_pool[idx]
        M_ctx = rng.uniform(*M_RANGE, size=(PRETRAIN_BATCH, K_CTX))
        M_q = rng.uniform(*M_RANGE, size=(PRETRAIN_BATCH, Q_CTX))

        Y_ctx = oracle_batch_labels(oracle, cs, M_ctx)
        Y_q = oracle_batch_labels(oracle, cs, M_q)
        M_ctx_t = torch.from_numpy(M_ctx).float()
        Y_ctx_t = torch.from_numpy(Y_ctx).float()
        M_q_t = torch.from_numpy(M_q).float()
        Y_q_t = torch.from_numpy(Y_q).float()

        y_pred = model(M_ctx_t, Y_ctx_t, M_q_t)
        L_mse = F.mse_loss(y_pred, Y_q_t)

        if step % SCORE_EVERY == 0 and LAMBDA_SCORE > 0:
            fwd_s = fwd_sens(model, oracle, cs, M_ctx, M_q_t, eps=FD_STEP)
            true_s = torch.from_numpy(
                score_targets(oracle, cs, M_q, eps=FD_STEP)
            ).float()
            L_score = F.mse_loss(fwd_s, true_s)
        else:
            L_score = torch.zeros((), device=y_pred.device)

        L_total = L_mse + LAMBDA_SCORE * L_score
        opt.zero_grad()
        L_total.backward()
        opt.step()

        losses_mse.append(float(L_mse.item()))
        losses_score.append(float(L_score.item()))
        losses_total.append(float(L_total.item()))

        if step % 100 == 0 or step == PRETRAIN_STEPS - 1:
            print(f"  step {step:5d}/{PRETRAIN_STEPS}  "
                  f"L_mse={L_mse.item():.6f}  "
                  f"L_score={L_score.item():.6f}  "
                  f"L_total={L_total.item():.6f}")

    wall = time.time() - t0
    print(f"Done in {wall:.1f}s")

    torch.save(model.state_dict(), OUT / "intention_fm.pt")
    summary = dict(
        lambda_score=LAMBDA_SCORE, fd_step=FD_STEP, score_every=SCORE_EVERY,
        pretrain_steps=PRETRAIN_STEPS, batch=PRETRAIN_BATCH,
        d_psi=16, lr=LR, wall_seconds=wall,
        final_L_mse=losses_mse[-1], final_L_score=losses_score[-1],
        final_L_total=losses_total[-1],
    )
    with open(OUT / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(OUT / "losses.npz",
             mse=np.array(losses_mse),
             score=np.array(losses_score),
             total=np.array(losses_total))
    print(f"Saved checkpoint to {OUT / 'intention_fm.pt'}")
    print(f"To compare disclosure lift, re-run identifiability_probe.py "
          f"with OUT_DIR={OUT}")


if __name__ == "__main__":
    main()
