"""Per-event rate-aware Intention head (Patch 6 literal).

The augmented basis appends log(y) to the per-event feature vector:

    phi_aug(m, y) = [psi_theta(m); log(y)]    in R^{d_psi + 1}

This is the construction physics_patches.md Patch 6 prescribes. Unlike
the set-level rate-conditioned variant in intention_rateaware.py, here
the basis dimension grows by one and the ridge weight vector picks up
an explicit per-scenario rate-coupling component.

The ridge solve runs in the (d_psi + 1)-dim augmented basis. At a query
point m_q the augmented feature vector needs a value of log(y_q); we use
the unaugmented Intention prediction as that estimate, per Patch 6.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


M_REF = 1.0  # TeV


class PsiMLPPerEvent(nn.Module):
    """psi_theta: m -> R^{d_psi}. Same architecture as the baseline."""

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
        x = torch.log(m / M_REF).unsqueeze(-1)
        return self.net(x)


def _augment_with_log_y(psi: torch.Tensor, y: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """psi: (..., d), y: (...,) -> aug: (..., d+1)."""
    log_y = torch.log(torch.clamp(y, min=eps)).unsqueeze(-1)
    return torch.cat([psi, log_y], dim=-1)


class IntentionFMPerEventRate(nn.Module):
    """Patch 6 literal: per-event log(y) augmentation of the basis.

    Inference is two ridge solves:
      1. Unaugmented: w_u = (Psi^T Psi + αI)^{-1} Psi^T Y_ctx in R^d
         predicts y_q_hat = psi(m_q)^T w_u for the query rate estimate.
      2. Augmented: Psi_aug = [Psi, log Y_ctx] in (K, d+1);
         w_a = (Psi_aug^T Psi_aug + αI)^{-1} Psi_aug^T Y_ctx in R^{d+1}.
         Final prediction: y_q = [psi(m_q), log y_q_hat] dot w_a.
    """

    def __init__(self, d_psi: int = 16, hidden: int = 64, alpha: float = 1e-3):
        super().__init__()
        self.psi = PsiMLPPerEvent(d_psi=d_psi, hidden=hidden)
        self.d_psi = d_psi
        self.alpha = alpha

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _solve_unaug(self, Psi_ctx, Y_ctx):
        d = Psi_ctx.shape[-1]
        if Psi_ctx.dim() == 2:
            A = Psi_ctx.T @ Psi_ctx + self.alpha * torch.eye(
                d, dtype=Psi_ctx.dtype, device=Psi_ctx.device)
            return torch.linalg.solve(A, Psi_ctx.T @ Y_ctx)
        # batched (S, K, d)
        S = Psi_ctx.size(0)
        A = torch.einsum("skd,ske->sde", Psi_ctx, Psi_ctx)
        eye = torch.eye(d, dtype=A.dtype, device=A.device).expand_as(A)
        A = A + self.alpha * eye
        b = torch.einsum("skd,sk->sd", Psi_ctx, Y_ctx)
        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def _solve_aug(self, Psi_ctx_aug, Y_ctx):
        return self._solve_unaug(Psi_ctx_aug, Y_ctx)

    def forward(self, M_ctx, Y_ctx, M_query):
        if M_ctx.dim() == 1:
            return self._predict_one(M_ctx, Y_ctx, M_query)
        S = M_ctx.size(0); K = M_ctx.size(1); Q = M_query.size(1); d = self.d_psi
        Psi_ctx = self.psi(M_ctx.reshape(-1)).reshape(S, K, d)
        Psi_query = self.psi(M_query.reshape(-1)).reshape(S, Q, d)
        # 1. unaugmented prediction at queries
        w_u = self._solve_unaug(Psi_ctx, Y_ctx)
        y_q_hat = torch.einsum("sqd,sd->sq", Psi_query, w_u)
        # 2. augmented basis
        Psi_ctx_aug = _augment_with_log_y(Psi_ctx, Y_ctx)
        w_a = self._solve_aug(Psi_ctx_aug, Y_ctx)
        Psi_query_aug = _augment_with_log_y(Psi_query, y_q_hat)
        return torch.einsum("sqd,sd->sq", Psi_query_aug, w_a)

    def _predict_one(self, M_ctx, Y_ctx, M_query):
        Psi_ctx = self.psi(M_ctx)
        Psi_query = self.psi(M_query)
        w_u = self._solve_unaug(Psi_ctx, Y_ctx)
        y_q_hat = Psi_query @ w_u
        Psi_ctx_aug = _augment_with_log_y(Psi_ctx, Y_ctx)
        w_a = self._solve_aug(Psi_ctx_aug, Y_ctx)
        Psi_query_aug = _augment_with_log_y(Psi_query, y_q_hat)
        return Psi_query_aug @ w_a

    # ---- numpy helpers for probe ----
    @torch.no_grad()
    def A_inv_and_w(self, M_ctx: np.ndarray, Y_ctx: np.ndarray):
        """Return augmented A^{-1}, w_aug, Psi_ctx_aug."""
        Mc = torch.from_numpy(M_ctx).float()
        Yc = torch.from_numpy(Y_ctx).float()
        Psi = self.psi(Mc).numpy()
        Psi_aug = np.concatenate([Psi, np.log(np.maximum(Y_ctx, 1e-6))[:, None]], axis=1)
        d = Psi_aug.shape[1]
        A = Psi_aug.T @ Psi_aug + self.alpha * np.eye(d)
        A_inv = np.linalg.inv(A)
        w = A_inv @ Psi_aug.T @ Y_ctx
        return A_inv, w, Psi_aug
