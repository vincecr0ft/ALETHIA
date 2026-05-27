"""Learned-psi Intention foundation model.

Same closed-form linear attention as the tarball's ``IntentionFMInContext``
(arXiv:2305.10203), but with the basis ``psi(m)`` replaced by a small MLP
trained end-to-end through ``torch.linalg.solve``.

Forward pass (single scenario):

    Psi_ctx   = psi_theta(M_ctx)               (K, d_psi)
    Psi_query = psi_theta(M_query)             (Q, d_psi)
    A         = Psi_ctx^T Psi_ctx + alpha I    (d_psi, d_psi)
    b         = Psi_ctx^T Y_ctx                (d_psi,)
    w         = A^{-1} b                       (d_psi,)
    y_query   = Psi_query @ w                  (Q,)

Wilson coefficient ``c`` is NEVER on the forward pass; it only enters via
the labels ``Y_ctx`` and ``Y_query`` constructed in ``data.py``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Match the polynomial basis's reference scale (so the fixed-basis baseline
# and the learned basis see equivalent m inputs).
M_REF = 1.0


def _scaled_m(m: torch.Tensor) -> torch.Tensor:
    """Map raw m (TeV) to log(m / M_REF) for numerical stability."""
    return torch.log(m / M_REF).unsqueeze(-1)


class PsiMLP(nn.Module):
    """Small MLP psi_theta: R -> R^{d_psi}."""

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
        """m: (..., ) -> psi: (..., d_psi). Augments with a constant 1
        column inside so the basis can represent the SM-only mode without
        needing a perfectly-tuned MLP output.
        """
        x = _scaled_m(m)
        return self.net(x)


class IntentionFMLearned(nn.Module):
    """Closed-form linear attention with learned-psi basis.

    The "model parameters" are entirely inside ``self.psi``. There is no
    explicit weight matrix: the per-scenario weights are solved at
    inference time from the context. ``alpha`` is a ridge regulariser.
    """

    def __init__(self, d_psi: int = 16, hidden: int = 64, alpha: float = 1e-3):
        super().__init__()
        self.psi = PsiMLP(d_psi=d_psi, hidden=hidden)
        self.d_psi = d_psi
        self.alpha = alpha

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                M_query: torch.Tensor) -> torch.Tensor:
        """Predict ``Y_query`` from ``(M_ctx, Y_ctx)`` at ``M_query``.

        Accepts both single-scenario tensors:
            M_ctx: (K,), Y_ctx: (K,), M_query: (Q,) -> (Q,)
        and batched tensors:
            M_ctx: (S, K), Y_ctx: (S, K), M_query: (S, Q) -> (S, Q)
        """
        if M_ctx.dim() == 1:
            return self._predict_one(M_ctx, Y_ctx, M_query)
        # batched
        S = M_ctx.size(0)
        K = M_ctx.size(1)
        Q = M_query.size(1)
        d = self.d_psi
        # Flatten the batch dim for the MLP, then reshape.
        Psi_ctx = self.psi(M_ctx.reshape(-1)).reshape(S, K, d)
        Psi_query = self.psi(M_query.reshape(-1)).reshape(S, Q, d)
        # A = Psi_ctx^T Psi_ctx + alpha I  per scenario
        A = torch.einsum("skd,ske->sde", Psi_ctx, Psi_ctx)
        eye = torch.eye(d, dtype=A.dtype, device=A.device).expand_as(A)
        A = A + self.alpha * eye
        b = torch.einsum("skd,sk->sd", Psi_ctx, Y_ctx)
        # w = A^{-1} b  via solve (back-prop friendly).
        w = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)   # (S, d)
        # y_query = Psi_query @ w
        return torch.einsum("sqd,sd->sq", Psi_query, w)

    def _predict_one(self, M_ctx, Y_ctx, M_query):
        Psi_ctx = self.psi(M_ctx)
        Psi_query = self.psi(M_query)
        A = Psi_ctx.T @ Psi_ctx + self.alpha * torch.eye(self.d_psi,
                                                        dtype=Psi_ctx.dtype,
                                                        device=Psi_ctx.device)
        w = torch.linalg.solve(A, Psi_ctx.T @ Y_ctx)
        return Psi_query @ w


class IntentionFMFixed:
    """Baseline: the tarball's hand-engineered polynomial basis.

    psi(m) = {1, log(m/M_REF), ..., log^{K_X-1}(m/M_REF)}. No training. Pure
    closed-form attention. Run on numpy; the comparison code wraps this
    behind a uniform ``predict(M_ctx, Y_ctx, M_query) -> Y_query`` API.
    """

    def __init__(self, K_X: int = 5, M_REF: float = 1.0, alpha: float = 1e-3):
        self.K_X = K_X
        self.M_REF = M_REF
        self.alpha = alpha

    def psi(self, m: np.ndarray) -> np.ndarray:
        m = np.atleast_1d(m).astype(np.float64)
        log_m = np.log(m / self.M_REF)
        return np.stack([log_m ** k for k in range(self.K_X)], axis=1)

    def predict(self, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                M_query: np.ndarray) -> np.ndarray:
        Psi_ctx = self.psi(M_ctx)
        Psi_query = self.psi(M_query)
        A = Psi_ctx.T @ Psi_ctx + self.alpha * np.eye(self.K_X)
        # Use solve for parity with the learned version.
        w = np.linalg.solve(A, Psi_ctx.T @ Y_ctx.astype(np.float64))
        return Psi_query @ w


if __name__ == "__main__":
    import sys; sys.path.insert(0, "/tmp/fm_compare")
    from data import make_oracle, make_scenario
    rng = np.random.default_rng(0)
    o = make_oracle()
    c = rng.uniform(-0.5, 0.5, 4)
    s = make_scenario(o, c, rng)
    fixed = IntentionFMFixed()
    learned = IntentionFMLearned()
    yp_fixed = fixed.predict(s["M_ctx"], s["Y_ctx"], s["M_query"])
    print("fixed   MSE =", float(np.mean((yp_fixed - s["Y_query"]) ** 2)))
    with torch.no_grad():
        yp_learned = learned(torch.from_numpy(s["M_ctx"]),
                             torch.from_numpy(s["Y_ctx"]),
                             torch.from_numpy(s["M_query"]))
    print("learned MSE (untrained) =", float(((yp_learned - torch.from_numpy(s["Y_query"])) ** 2).mean()))
    print("learned n_params =", learned.n_params)
