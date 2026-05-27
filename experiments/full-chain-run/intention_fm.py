"""IntentionFM with learned psi_theta basis for the full-chain run.

Closed-form linear attention:
    y_q = psi(M_q) (psi(M_ctx)^T psi(M_ctx) + alpha I)^{-1} psi(M_ctx)^T Y_ctx

psi_theta: R -> R^{d_psi} is a small MLP, trained end-to-end through
torch.linalg.solve on a meta-learning MSE objective. c never enters psi_theta's
forward pass.

This is the production variant used by run.py. Pretraining loop is
delegated to run.py (so we can instrument it with Phoenix spans).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


M_REF = 1.0  # TeV


class PsiMLP(nn.Module):
    """psi_theta: scalar m -> R^{d_psi}. Input is log(m / M_REF)."""

    def __init__(self, d_psi: int = 16, hidden: int = 64):
        super().__init__()
        self.d_psi = d_psi
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_psi),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: (..., ) -> psi: (..., d_psi)
        x = torch.log(m / M_REF).unsqueeze(-1)
        return self.net(x)


class IntentionFM(nn.Module):
    """Intention closed-form attention with learned basis.

    Inference takes a context (M_ctx, Y_ctx) and a query M_q; returns the
    predictive mean. Differentiable through torch.linalg.solve for
    meta-learning.
    """

    def __init__(self, d_psi: int = 16, hidden: int = 64, alpha: float = 1e-3):
        super().__init__()
        self.psi = PsiMLP(d_psi=d_psi, hidden=hidden)
        self.alpha = alpha
        self.d_psi = d_psi

    def forward(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                M_q: torch.Tensor) -> torch.Tensor:
        # Shapes: M_ctx (B, K), Y_ctx (B, K), M_q (B, Q) -> (B, Q)
        Psi_ctx = self.psi(M_ctx)        # (B, K, d)
        Psi_q   = self.psi(M_q)          # (B, Q, d)
        d = Psi_ctx.shape[-1]
        I = self.alpha * torch.eye(d, device=Psi_ctx.device, dtype=Psi_ctx.dtype)
        A = Psi_ctx.transpose(-2, -1) @ Psi_ctx + I       # (B, d, d)
        rhs = Psi_ctx.transpose(-2, -1) @ Y_ctx.unsqueeze(-1)  # (B, d, 1)
        w = torch.linalg.solve(A, rhs)                    # (B, d, 1)
        y_q = (Psi_q @ w).squeeze(-1)                     # (B, Q)
        return y_q

    # ---- numpy-side inference helpers (no grad needed) ----
    @torch.no_grad()
    def predict_np(self, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                   M_q: np.ndarray) -> np.ndarray:
        Mc = torch.from_numpy(M_ctx).float().unsqueeze(0)
        Yc = torch.from_numpy(Y_ctx).float().unsqueeze(0)
        Mq = torch.from_numpy(M_q).float().unsqueeze(0)
        return self.forward(Mc, Yc, Mq).squeeze(0).numpy()

    @torch.no_grad()
    def psi_np(self, M: np.ndarray) -> np.ndarray:
        M_t = torch.from_numpy(M).float()
        return self.psi(M_t).numpy()

    @torch.no_grad()
    def A_inv_and_w(self, M_ctx: np.ndarray, Y_ctx: np.ndarray) -> tuple:
        """Return A^{-1} and w for the current context (numpy)."""
        Psi = self.psi_np(M_ctx)                          # (K, d)
        d = Psi.shape[1]
        A = Psi.T @ Psi + self.alpha * np.eye(d)
        A_inv = np.linalg.inv(A)
        w = A_inv @ Psi.T @ Y_ctx
        return A_inv, w, Psi

    @torch.no_grad()
    def leverage(self, M_ctx: np.ndarray, M_q: np.ndarray) -> np.ndarray:
        """Per-query leverage lev(m) = psi(m)^T A_ctx^{-1} psi(m)."""
        A_inv, _, _ = self.A_inv_and_w(M_ctx, np.zeros(len(M_ctx)))
        Psi_q = self.psi_np(M_q)                          # (Q, d)
        return np.einsum("qd,de,qe->q", Psi_q, A_inv, Psi_q)

    @torch.no_grad()
    def kappa_A(self, M_ctx: np.ndarray) -> float:
        Psi = self.psi_np(M_ctx)
        d = Psi.shape[1]
        A = Psi.T @ Psi + self.alpha * np.eye(d)
        eigs = np.linalg.eigvalsh(A)
        return float(eigs.max() / max(eigs.min(), 1e-30))
