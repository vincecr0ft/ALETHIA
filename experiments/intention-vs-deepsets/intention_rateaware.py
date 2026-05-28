"""Rate-aware Intention foundation model.

The basis psi_theta(m) is conditioned on a per-scenario *rate summary*
r = mean(Y_ctx). This gives the ridge problem a constant channel that
carries the overall rate, which is the diagnosed disclosure path for
pure-vertex Wilson operators that shift the cross section by a flat
(v / Lambda)^2 factor with no m_ll shape signature
(empirical-results.md section 4; Patch 6 of physics_patches.md).

Concretely:
    log_rate = log(max(mean(Y_ctx), eps))           per scenario, broadcast
    psi(m, log_rate) = MLP([log(m / M_REF), log_rate])
    A = Psi_ctx^T Psi_ctx + alpha I
    w = A^{-1} Psi_ctx^T Y_ctx
    y_query = Psi_query @ w

The rate summary is well-defined at the query (it depends only on
Y_ctx, not on Y_q), so no double-solve is needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


M_REF = 1.0  # TeV


class PsiMLPRateAware(nn.Module):
    """psi_theta: (m, log_rate) -> R^{d_psi}."""

    def __init__(self, d_psi: int = 16, hidden: int = 64):
        super().__init__()
        self.d_psi = d_psi
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_psi),
        )

    def forward(self, m: torch.Tensor, log_rate: torch.Tensor) -> torch.Tensor:
        """Element-wise inputs, broadcasted. m and log_rate have the same
        leading shape; output has that shape + (d_psi,)."""
        x = torch.stack([torch.log(m / M_REF), log_rate], dim=-1)
        return self.net(x)


def _log_rate_from_Y(Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-scenario log(mean(Y)).  Y is (K,) or (S, K).  Returns scalar or (S,)."""
    if Y.dim() == 1:
        return torch.log(torch.clamp(Y.mean(), min=eps))
    return torch.log(torch.clamp(Y.mean(dim=-1), min=eps))


class IntentionFMRateAware(nn.Module):
    """Closed-form linear attention with rate-conditioned learned basis."""

    def __init__(self, d_psi: int = 16, hidden: int = 64, alpha: float = 1e-3):
        super().__init__()
        self.psi = PsiMLPRateAware(d_psi=d_psi, hidden=hidden)
        self.d_psi = d_psi
        self.alpha = alpha

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                M_query: torch.Tensor) -> torch.Tensor:
        if M_ctx.dim() == 1:
            return self._predict_one(M_ctx, Y_ctx, M_query)
        S = M_ctx.size(0)
        K = M_ctx.size(1)
        Q = M_query.size(1)
        d = self.d_psi
        log_rate = _log_rate_from_Y(Y_ctx)                 # (S,)
        lr_ctx = log_rate.unsqueeze(-1).expand(-1, K)      # (S, K)
        lr_q   = log_rate.unsqueeze(-1).expand(-1, Q)      # (S, Q)
        Psi_ctx = self.psi(M_ctx.reshape(-1), lr_ctx.reshape(-1)).reshape(S, K, d)
        Psi_query = self.psi(M_query.reshape(-1), lr_q.reshape(-1)).reshape(S, Q, d)
        A = torch.einsum("skd,ske->sde", Psi_ctx, Psi_ctx)
        eye = torch.eye(d, dtype=A.dtype, device=A.device).expand_as(A)
        A = A + self.alpha * eye
        b = torch.einsum("skd,sk->sd", Psi_ctx, Y_ctx)
        w = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)   # (S, d)
        return torch.einsum("sqd,sd->sq", Psi_query, w)

    def _predict_one(self, M_ctx, Y_ctx, M_query):
        log_rate = _log_rate_from_Y(Y_ctx)
        Psi_ctx = self.psi(M_ctx, log_rate.expand_as(M_ctx))
        Psi_query = self.psi(M_query, log_rate.expand_as(M_query))
        A = Psi_ctx.T @ Psi_ctx + self.alpha * torch.eye(
            self.d_psi, dtype=Psi_ctx.dtype, device=Psi_ctx.device)
        w = torch.linalg.solve(A, Psi_ctx.T @ Y_ctx)
        return Psi_query @ w

    # -- numpy-side helpers mirroring modules/surrogate/intention/model.py --
    @torch.no_grad()
    def predict_np(self, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                   M_q: np.ndarray) -> np.ndarray:
        Mc = torch.from_numpy(M_ctx).float()
        Yc = torch.from_numpy(Y_ctx).float()
        Mq = torch.from_numpy(M_q).float()
        return self._predict_one(Mc, Yc, Mq).numpy()

    @torch.no_grad()
    def psi_np(self, M_ctx: np.ndarray, Y_ctx: np.ndarray,
               M: np.ndarray) -> np.ndarray:
        Mc = torch.from_numpy(M_ctx).float()
        Yc = torch.from_numpy(Y_ctx).float()
        Mt = torch.from_numpy(M).float()
        log_rate = _log_rate_from_Y(Yc)
        return self.psi(Mt, log_rate.expand_as(Mt)).numpy()

    @torch.no_grad()
    def A_inv_and_w(self, M_ctx: np.ndarray, Y_ctx: np.ndarray) -> tuple:
        """Return A^{-1}, w, Psi_ctx for the current context (numpy)."""
        Psi = self.psi_np(M_ctx, Y_ctx, M_ctx)               # (K, d)
        d = Psi.shape[1]
        A = Psi.T @ Psi + self.alpha * np.eye(d)
        A_inv = np.linalg.inv(A)
        w = A_inv @ Psi.T @ Y_ctx
        return A_inv, w, Psi
