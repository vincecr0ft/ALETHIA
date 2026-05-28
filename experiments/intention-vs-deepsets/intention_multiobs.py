"""Multi-observable Intention foundation model.

The encoder uses two independent feature maps:

    psi_m  : log(m / M_ref)  -> R^{d_psi}     (m_ll spectrum)
    psi_pt : log(pT / P_ref) -> R^{d_psi}     (lepton transverse momentum)

The context combines K_m mass-points and K_pt pT-points; the closed-form
ridge solves a single (d_psi, d_psi) system over the combined design
matrix. The implicit ridge weight vector w is the per-scenario summary
that the linear probe maps back to Wilson coefficients.

The point: the m_ll-only encoder cannot disclose the vertex operators
(cHq3, cHq1) because they produce nearly degenerate signatures in m_ll
at this PDF + scale. The pT spectrum carries the angular structure that
breaks the degeneracy via the chirality-dependent cos(theta*)
distribution. See modules/analytic_smeft/smeft.py:differential_xs_pt.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


M_REF = 1.0   # TeV
PT_REF = 0.5  # TeV; a typical scale for the pT spectrum


class PsiMLP1D(nn.Module):
    """Maps log(scaled_kinematic) -> R^{d_psi}."""

    def __init__(self, d_psi: int, hidden: int, ref: float = M_REF):
        super().__init__()
        self.d_psi = d_psi
        self.ref = ref
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_psi),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log(x / self.ref).unsqueeze(-1))


class IntentionFMMulti(nn.Module):
    """Closed-form linear attention over two observables.

    Forward pass shapes:
      M_ctx     (S, K_m)
      Y_m_ctx   (S, K_m)
      PT_ctx    (S, K_pt)
      Y_pt_ctx  (S, K_pt)
      M_q       (S, Q_m)
      PT_q      (S, Q_pt)

    Returns:
      (y_m_pred, y_pt_pred) of shapes (S, Q_m), (S, Q_pt).

    The ridge solve is over the combined (K_m + K_pt, d) design matrix.
    """

    def __init__(self, d_psi: int = 32, hidden: int = 128, alpha: float = 1e-3,
                 ref_a: float = M_REF, ref_b: float = PT_REF):
        super().__init__()
        # Two independent encoders; the second-channel reference is configurable
        # so the same class drives the (m, p_T) and (m, A_FB) variants.
        self.psi_m  = PsiMLP1D(d_psi=d_psi, hidden=hidden, ref=ref_a)
        self.psi_pt = PsiMLP1D(d_psi=d_psi, hidden=hidden, ref=ref_b)
        self.d_psi = d_psi
        self.alpha = alpha

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, M_ctx, Y_m_ctx, PT_ctx, Y_pt_ctx, M_q, PT_q):
        if M_ctx.dim() == 1:
            return self._predict_one(M_ctx, Y_m_ctx, PT_ctx, Y_pt_ctx, M_q, PT_q)
        S = M_ctx.size(0); Km = M_ctx.size(1); Kpt = PT_ctx.size(1)
        Qm = M_q.size(1); Qpt = PT_q.size(1); d = self.d_psi

        Psi_m_ctx  = self.psi_m(M_ctx.reshape(-1)).reshape(S, Km, d)
        Psi_pt_ctx = self.psi_pt(PT_ctx.reshape(-1)).reshape(S, Kpt, d)
        Psi_m_q  = self.psi_m(M_q.reshape(-1)).reshape(S, Qm, d)
        Psi_pt_q = self.psi_pt(PT_q.reshape(-1)).reshape(S, Qpt, d)

        # Stack context along the K axis
        Psi_ctx = torch.cat([Psi_m_ctx, Psi_pt_ctx], dim=1)         # (S, Km+Kpt, d)
        Y_ctx   = torch.cat([Y_m_ctx,   Y_pt_ctx],   dim=1)         # (S, Km+Kpt)

        A = torch.einsum("skd,ske->sde", Psi_ctx, Psi_ctx)
        eye = torch.eye(d, dtype=A.dtype, device=A.device).expand_as(A)
        A = A + self.alpha * eye
        b = torch.einsum("skd,sk->sd", Psi_ctx, Y_ctx)
        w = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)      # (S, d)

        y_m_pred  = torch.einsum("sqd,sd->sq", Psi_m_q,  w)
        y_pt_pred = torch.einsum("sqd,sd->sq", Psi_pt_q, w)
        return y_m_pred, y_pt_pred

    def _predict_one(self, M_ctx, Y_m_ctx, PT_ctx, Y_pt_ctx, M_q, PT_q):
        Psi_m_ctx  = self.psi_m(M_ctx)
        Psi_pt_ctx = self.psi_pt(PT_ctx)
        Psi_m_q  = self.psi_m(M_q)
        Psi_pt_q = self.psi_pt(PT_q)
        Psi_ctx = torch.cat([Psi_m_ctx, Psi_pt_ctx], dim=0)
        Y_ctx   = torch.cat([Y_m_ctx,   Y_pt_ctx],   dim=0)
        A = Psi_ctx.T @ Psi_ctx + self.alpha * torch.eye(
            self.d_psi, dtype=Psi_ctx.dtype, device=Psi_ctx.device)
        w = torch.linalg.solve(A, Psi_ctx.T @ Y_ctx)
        return (Psi_m_q @ w), (Psi_pt_q @ w)

    # ---- numpy helpers for probe ----
    @torch.no_grad()
    def A_inv_and_w(self, M_ctx: np.ndarray, Y_m_ctx: np.ndarray,
                    PT_ctx: np.ndarray, Y_pt_ctx: np.ndarray):
        """Implicit per-scenario w from a combined-observable context."""
        Mc  = torch.from_numpy(M_ctx).float()
        Yc_m  = torch.from_numpy(Y_m_ctx).float()
        Pc  = torch.from_numpy(PT_ctx).float()
        Yc_pt = torch.from_numpy(Y_pt_ctx).float()
        Psi_m  = self.psi_m(Mc).numpy()
        Psi_pt = self.psi_pt(Pc).numpy()
        Psi    = np.concatenate([Psi_m, Psi_pt], axis=0)
        Y      = np.concatenate([Y_m_ctx, Y_pt_ctx], axis=0)
        d = Psi.shape[1]
        A = Psi.T @ Psi + self.alpha * np.eye(d)
        A_inv = np.linalg.inv(A)
        w = A_inv @ Psi.T @ Y
        return A_inv, w, Psi
