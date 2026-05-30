"""INV-2 fix path B: retrain psi_theta with a c-recoverability auxiliary loss.

The pretraining objective in run.py is plain MSE on the meta-learning
prediction y_q. The closed-form ridge weight w = A^{-1} Psi^T Y_ctx is a
function of (M_ctx, Y_ctx); psi_theta is free to encode any kinematic
shapes that minimise the prediction loss, including shapes that compress
or destroy the Wilson-coefficient information w carries. The probe-
capacity diagnostic (probe_capacity_diagnostic.py) shows that higher-
capacity probes (quadratic, MLP) do not close the gap to the MLE
variance on the resolved c_tilde directions, which places the bottleneck
on psi_theta rather than the probe.

This script retrains psi_theta with a JOINT loss

    L = MSE_y(y_pred, y_truth)  +  lambda * MSE_c(c_pred, c_true)

where c_pred = w @ W_aux + b_aux for a jointly-trained linear auxiliary
probe (W_aux, b_aux) read off the ridge weight w. The MSE_c term is the
c-recoverability auxiliary that biases psi_theta toward encoding
shapes from which c can be linearly extracted. After training we save
psi_theta and the auxiliary (W_aux, b_aux) so probe_efficiency.py and
INV-3 can use them.

Run with `uv run python -u experiments/full-chain-run/retrain_with_c_recovery.py`.
Outputs intention_fm_v2.pt and aux_probe_v2.npz under output/.

Compute budget: ~3000 Adam steps at batch=32 (K=12, Q=32) on CPU ~10 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM

# -----------------------------------------------------------------
# Config (mirrors run.py defaults; keep aligned with PRETRAIN_*).
# -----------------------------------------------------------------
SEED = 2026
N_PRETRAIN_SCEN = 200
PRETRAIN_BATCH = 24
# Retrain variants (driven by env vars):
#   FINETUNE=1 INIT_CHECKPOINT=... : start from v1 ψ_θ (preserves prediction
#       quality; useful for small d_psi-preserving nudges; tried at λ_c=0.1
#       and did not change c-encoding subspace).
#   FINETUNE=0 D_PSI=32           : from-scratch with larger ψ_θ capacity;
#       the structural fix once the diagnostic shows the d_psi=16 ridge
#       weight cannot match MLE on the leading c̃ direction.
import os as _os_steps
PRETRAIN_STEPS = int(_os_steps.environ.get("RETRAIN_STEPS", "1500"))
FINETUNE = _os_steps.environ.get("FINETUNE", "1") not in ("0", "false")
INIT_CHECKPOINT = _os_steps.environ.get(
    "INIT_CHECKPOINT", "output/intention_fm.pt")
D_PSI = int(_os_steps.environ.get("D_PSI", "16"))
RETRAIN_TAG = _os_steps.environ.get("RETRAIN_TAG", "v2")
K_CTX = 12
Q_CTX = 32
LR = 5e-4
M_RANGE = (0.3, 2.3)
C_TRAIN_BOX = 0.7
WITHHOLD_DIM = 2          # c_lq^(3)
WITHHOLD_BAND = (0.6, 1.0)
SIGMA_Y_TRAIN = 0.05       # noise on Y during training; matches probe_efficiency
N_WC = 4
# Loss balance: at v1 convergence mse_y ~ 1e-3, c-residual variance ~ 0.025
# on c_tilde_1, so mse_c ~ 0.04 in c-units. lambda_c = 1.0 was making the
# c term dominate (700x larger contribution at step 0); lambda_c = 0.1
# keeps both gradients well-conditioned without overpowering the prediction
# objective during fine-tuning from v1.
LAMBDA_C = float(_os_steps.environ.get("LAMBDA_C", "0.1"))
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_c_training(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform |c_i| <= 0.7 excluding |c_lq^(3)| in [0.6, 1.0]."""
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


class IntentionFMJoint(nn.Module):
    """IntentionFM with an auxiliary linear c-recovery probe.

    Forward returns (y_pred, c_pred). The auxiliary probe (W_aux, b_aux)
    is a single linear layer over the ridge weight w.
    """

    def __init__(self, d_psi: int = 16, hidden: int = 64,
                 alpha: float = 1e-3, n_wc: int = 4):
        super().__init__()
        self.fm = IntentionFM(d_psi=d_psi, hidden=hidden, alpha=alpha)
        self.aux = nn.Linear(d_psi, n_wc)

    def forward(self, M_ctx, Y_ctx, M_q):
        """Returns (y_pred, c_pred, w)."""
        Psi_ctx = self.fm.psi(M_ctx)
        Psi_q   = self.fm.psi(M_q)
        d = Psi_ctx.shape[-1]
        I = self.fm.alpha * torch.eye(d, device=Psi_ctx.device,
                                       dtype=Psi_ctx.dtype)
        A = Psi_ctx.transpose(-2, -1) @ Psi_ctx + I
        rhs = Psi_ctx.transpose(-2, -1) @ Y_ctx.unsqueeze(-1)
        w = torch.linalg.solve(A, rhs).squeeze(-1)         # (B, d)
        y_q = (Psi_q @ w.unsqueeze(-1)).squeeze(-1)        # (B, Q)
        c_pred = self.aux(w)                                # (B, n_wc)
        return y_q, c_pred, w


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print(f"Retrain psi_theta with c-recoverability auxiliary "
          f"(lambda_c = {LAMBDA_C}).")

    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    model = IntentionFMJoint(d_psi=D_PSI, hidden=64, alpha=1e-3, n_wc=N_WC)

    if FINETUNE:
        init_ckpt_path = HERE / INIT_CHECKPOINT
        if not init_ckpt_path.exists():
            print(f"FINETUNE=1 but {init_ckpt_path} missing; falling back to "
                  "from-scratch training.")
        else:
            init_state = torch.load(init_ckpt_path, map_location="cpu",
                                    weights_only=True)
            # init_state is a plain IntentionFM state dict; load into
            # model.fm (the IntentionFM submodule), leaving model.aux at
            # its random initialisation.
            model.fm.load_state_dict(init_state)
            print(f"Fine-tuning from v1 checkpoint: {init_ckpt_path}")

    opt = Adam(model.parameters(), lr=LR)

    print("Sampling training scenario pool...")
    c_pool = sample_c_training(N_PRETRAIN_SCEN, rng)

    losses = {"mse_y": [], "mse_c": [], "total": []}
    print(f"Training for {PRETRAIN_STEPS} steps (batch={PRETRAIN_BATCH}).")
    t0 = time.time()
    for step in range(PRETRAIN_STEPS):
        idx = rng.choice(N_PRETRAIN_SCEN, size=PRETRAIN_BATCH, replace=False)
        cs = c_pool[idx]                                # (B, n_wc)
        B = PRETRAIN_BATCH
        M_ctx_np = rng.uniform(*M_RANGE, size=(B, K_CTX))
        M_q_np = rng.uniform(*M_RANGE, size=(B, Q_CTX))
        C_ctx = np.repeat(cs, K_CTX, axis=0)
        C_q = np.repeat(cs, Q_CTX, axis=0)
        Y_ctx_np = oracle.truth(C_ctx, M_ctx_np.flatten()).reshape(B, K_CTX)
        Y_q_np = oracle.truth(C_q, M_q_np.flatten()).reshape(B, Q_CTX)
        # Additive noise on Y_ctx with the same sigma the probe sees.
        Y_ctx_np = Y_ctx_np + rng.normal(
            0.0, SIGMA_Y_TRAIN, size=Y_ctx_np.shape)

        M_ctx = torch.from_numpy(M_ctx_np).float()
        Y_ctx = torch.from_numpy(Y_ctx_np).float()
        M_q = torch.from_numpy(M_q_np).float()
        Y_q = torch.from_numpy(Y_q_np).float()
        c_t = torch.from_numpy(cs).float()

        y_pred, c_pred, _ = model(M_ctx, Y_ctx, M_q)
        mse_y = F.mse_loss(y_pred, Y_q)
        mse_c = F.mse_loss(c_pred, c_t)
        loss = mse_y + LAMBDA_C * mse_c
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses["mse_y"].append(float(mse_y.item()))
        losses["mse_c"].append(float(mse_c.item()))
        losses["total"].append(float(loss.item()))

        if step % 100 == 0 or step == PRETRAIN_STEPS - 1:
            print(f"  step {step:4d}/{PRETRAIN_STEPS}  "
                  f"mse_y={mse_y.item():.4g}  mse_c={mse_c.item():.4g}  "
                  f"total={loss.item():.4g}")

    print(f"Done in {time.time() - t0:.1f}s.")

    # Save: psi_theta state dict (compatible with existing IntentionFM)
    # plus the aux probe weights.
    fm_state = model.fm.state_dict()
    fm_path = OUT_DIR / f"intention_fm_{RETRAIN_TAG}.pt"
    torch.save(fm_state, fm_path)
    print(f"Saved retrained IntentionFM (psi_theta only) to {fm_path}")

    aux_W = model.aux.weight.detach().cpu().numpy()      # (n_wc, d_psi)
    aux_b = model.aux.bias.detach().cpu().numpy()        # (n_wc,)
    np.savez(OUT_DIR / f"aux_probe_{RETRAIN_TAG}.npz",
             W=aux_W, b=aux_b,
             notes=np.array(
                 "Jointly-trained linear auxiliary probe used during "
                 "psi_theta retraining; load alongside intention_fm_v2.pt "
                 f"as a baseline starting point. lambda_c = {LAMBDA_C}, "
                 f"sigma_y_train = {SIGMA_Y_TRAIN}."))
    print(f"Saved aux probe to {OUT_DIR / f'aux_probe_{RETRAIN_TAG}.npz'}")

    # Sanity diff: per-direction in-sample c-RMSE on a batch.
    with torch.no_grad():
        cs_eval = torch.from_numpy(
            sample_c_training(200, rng)).float()
        # eval one fresh context per c
        B = 200
        M_ctx_np = rng.uniform(*M_RANGE, size=(B, K_CTX))
        M_q_np = rng.uniform(*M_RANGE, size=(B, Q_CTX))
        C_ctx = np.repeat(cs_eval.numpy(), K_CTX, axis=0)
        C_q = np.repeat(cs_eval.numpy(), Q_CTX, axis=0)
        Y_ctx_np = oracle.truth(C_ctx, M_ctx_np.flatten()).reshape(B, K_CTX)
        Y_q_np = oracle.truth(C_q, M_q_np.flatten()).reshape(B, Q_CTX)
        Y_ctx_np = Y_ctx_np + rng.normal(
            0.0, SIGMA_Y_TRAIN, size=Y_ctx_np.shape)
        _, c_pred, _ = model(
            torch.from_numpy(M_ctx_np).float(),
            torch.from_numpy(Y_ctx_np).float(),
            torch.from_numpy(M_q_np).float())
        rmse_per_dir = (
            (c_pred - cs_eval).pow(2).mean(0).sqrt().numpy())
        print(f"\nIn-sample Warsaw c-RMSE (aux probe, retrained psi_theta): "
              f"{rmse_per_dir.round(4)}")


if __name__ == "__main__":
    main()
