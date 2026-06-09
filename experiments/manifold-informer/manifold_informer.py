r"""ManifoldInformer — the event-set JEPA-trained FM (Task 3).

Per ALETHIA_informer_workpoint_AL_handoff.md §1.5, the FM consumes a
variable-length set of per-event kinematic vectors x_n ∈ R^d_event (here
``d_event = 2``: ``(log m_ll/M_ref, cos θ*_CS)``) drawn at a fixed latent
working point c. The Wilson coefficient never enters the forward pass.
Training is JEPA-native: predict the embedding of held-out events from
context events; the target encoder is an EMA copy and the loss is in
latent space, not in event reconstruction. The Intention closed-form
ridge plays the role of the in-context per-scenario predictor.

Architecture sketch (the four pieces):

    1) per-event encoder phi_θ : R^d_event -> R^d_emb
       a small MLP applied per event; permutation-invariant by construction.

    2) Intention closed-form ridge as the predictor
       given a context event set X_ctx with embeddings K = phi_θ(X_ctx)
       and target embeddings V (from the EMA encoder), compute
           A = K^T K + α I
           w_θ(c) = A^{-1} K^T V   ∈ R^{d_emb × d_emb}
       and predict per-query event x_q:
           z_pred(x_q) = phi_θ(x_q) · w_θ(c)
       This is the in-context Bayesian linear regression that gives the
       per-scenario manifold coordinate, expressed in the learned basis.

    3) JEPA pretext loss
       L_JEPA = ‖z_pred(X_held) - stop_grad( phi_ema(X_held) )‖²
       plus VICReg collapse guard on the predicted embeddings.

    4) RS3L view-invariance (re-simulation augmentation)
       L_view = ‖w_θ(X_1) - w_θ(X_2)‖² for two independent event sets
       X_1, X_2 drawn at the same c. The closed-form ridge's
       label-independence already makes the design matrix A invariant
       under repeated sampling of the SAME m's; for SMEFT the events
       have different m's per draw, so this term enforces the genuine
       invariance the manifold predicts.

The Intention closed-form ridge expects scalar Y per context point in
its original incarnation. Here we use it on EMBEDDINGS: V is the matrix
of stop-gradient EMA-encoded target-event embeddings (shape K × d_emb),
A and the solve are unchanged, w_θ is now a matrix not a vector. This
is a direct generalisation that keeps the closed form intact while
matching JEPA-native semantics.

Falsification gates (P1-P4) live in ``manifold_informer_gates.py``.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-event encoder
# ---------------------------------------------------------------------------

class PerEventEncoder(nn.Module):
    """phi_θ: R^{d_event} -> R^{d_emb}.

    Applied per event; permutation-invariant by construction (the same
    weights see every event independently). The handoff calls for a
    Deep-Sets / Energy-Flow / Particle-Flow encoder (Komiske-Metodiev-
    Thaler arXiv:1810.05165); we use the simplest sufficient form (MLP
    per event) and let the aggregator handle pooling.

    Default input layout: x = (log m_ll/M_ref, cos θ*_CS), so d_event = 2.
    Trivially extends to richer per-event features (p_T, rapidity, ...).
    """

    def __init__(self, d_event: int = 2, d_emb: int = 16, hidden: int = 32):
        super().__init__()
        self.d_event = d_event
        self.d_emb = d_emb
        self.net = nn.Sequential(
            nn.Linear(d_event, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_event) -> (..., d_emb). Per-event application."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Intention closed-form ridge head (event-set version)
# ---------------------------------------------------------------------------

class IntentionRidgeHead(nn.Module):
    r"""Per-scenario closed-form ridge over event embeddings.

    Forward signature (single scenario):
        K = phi_θ(X_ctx)        (N_ctx, d_emb)   context-event embeddings
        V                       (N_ctx, d_target) target embeddings
                                                  (typically EMA-encoded
                                                  context events; same shape
                                                  as K so d_target = d_emb)
        K_q = phi_θ(X_q)        (N_q, d_emb)     query-event embeddings

    Compute:
        A      = K^T K + α I            (d_emb, d_emb)
        w_θ(c) = A^{-1} K^T V           (d_emb, d_target)   ← manifold coord
        Z_pred = K_q · w_θ(c)           (N_q, d_target)

    Batched (multi-scenario) version takes leading batch dim. Differentiable
    through torch.linalg.solve so the per-event encoder can be meta-trained
    by the JEPA loss on Z_pred.

    Args:
        alpha: ridge regulariser.
    """

    def __init__(self, alpha: float = 1e-3):
        super().__init__()
        self.alpha = alpha

    def w_implicit(self, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Solve w_θ = (K^T K + αI)^{-1} K^T V.

        K: (..., N_ctx, d_emb), V: (..., N_ctx, d_target).
        Returns w: (..., d_emb, d_target).
        """
        d = K.shape[-1]
        # A = K^T K + αI per leading batch.
        A = K.transpose(-2, -1) @ K
        eye = torch.eye(d, device=K.device, dtype=K.dtype)
        # Broadcast eye over batch dims.
        for _ in range(A.dim() - 2):
            eye = eye.unsqueeze(0)
        A = A + self.alpha * eye
        rhs = K.transpose(-2, -1) @ V
        return torch.linalg.solve(A, rhs)

    def forward(self, K: torch.Tensor, V: torch.Tensor,
                K_q: torch.Tensor) -> torch.Tensor:
        """Predict embeddings at query events.

        K:   (..., N_ctx, d_emb)
        V:   (..., N_ctx, d_target)
        K_q: (..., N_q,   d_emb)

        Returns Z_pred: (..., N_q, d_target).
        """
        w = self.w_implicit(K, V)
        return K_q @ w


# ---------------------------------------------------------------------------
# Aggregator (manifold coordinate as a single vector per scenario)
# ---------------------------------------------------------------------------

class WImplicitAggregator(nn.Module):
    """Aggregate a context event set into a single per-scenario vector.

    For the §1.5 manifold-coordinate reading, w_θ(c) IS the scenario
    summary. When V = K (the standard self-prediction setup), w_θ is
    a (d_emb, d_emb) matrix; we flatten its principal direction
    (the top eigenvector of K^T K, weighted by K^T V) to a vector of
    size d_emb that serves as the per-scenario manifold coordinate
    for the P1-P4 latent geometry probes.

    For the JEPA loss we use w_θ as a matrix (no flattening) — the
    flattened form is only for the geometric probes.
    """

    def __init__(self, ridge: IntentionRidgeHead):
        super().__init__()
        self.ridge = ridge

    def summary_vector(self, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Per-scenario manifold coordinate as a vector (for P1-P4 probes).

        K, V: (..., N_ctx, d_emb). Returns (..., d_emb).
        Uses w := w_θ(c) (d_emb, d_emb); the summary is w · (mean V).
        """
        w = self.ridge.w_implicit(K, V)                              # (..., d, d)
        V_mean = V.mean(dim=-2)                                       # (..., d)
        return torch.einsum("...dt,...t->...d", w, V_mean)


# ---------------------------------------------------------------------------
# Decoder readout (for the optional aux density anchor)
# ---------------------------------------------------------------------------

class DensityAnchorDecoder(nn.Module):
    """Tiny MLP from a predicted embedding to a scalar log w_c(x).

    Optional anchor term that ties the manifold representation to the
    per-event log-likelihood ratio (the only physical scalar we have
    cheaply available, via :func:`oracle_events.event_log_likelihood_ratio`).
    The JEPA pretext shapes the basis; this term sets its scale.
    """

    def __init__(self, d_emb: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ManifoldInformer(nn.Module):
    r"""ManifoldInformer FM (Task 3).

    Public API:
        forward(X_ctx, X_q)                              -> Z_pred  (S, N_q, d_emb)
        forward(X_ctx, X_q, return_losses=True,
                X_ctx_view2=...,
                log_w_q=None)                             -> dict of losses
        update_target()                                   EMA update
        encode_context(X_ctx)                             -> summary  (S, d_emb)
        manifold_coord(X_ctx)                             -> w_vector (S, d_emb)
        psi_np(X)                                         -> embeddings (numpy)

    Args:
        d_event: per-event feature dim (default 2 for (log m_ll, cos θ*)).
        d_emb: per-event embedding dim.
        hidden: per-event encoder hidden dim.
        alpha: ridge alpha.
        ema_momentum: EMA tau for the target encoder.
        vicreg_weight: weight on the variance + covariance regulariser.
        view_weight: weight on the RS3L view-invariance term.
        density_anchor_weight: weight on the optional decoder/log-w anchor;
            set to 0.0 to disable (pure JEPA).
        use_ema: if False, siamese (target = source); else EMA target.
    """

    def __init__(self,
                 d_event: int = 2,
                 d_emb: int = 16,
                 hidden: int = 32,
                 alpha: float = 1e-3,
                 ema_momentum: float = 0.996,
                 vicreg_weight: float = 0.04,
                 view_weight: float = 0.1,
                 density_anchor_weight: float = 0.05,
                 use_ema: bool = True):
        super().__init__()
        self.event_encoder = PerEventEncoder(d_event=d_event, d_emb=d_emb,
                                               hidden=hidden)
        self.ridge = IntentionRidgeHead(alpha=alpha)
        self.aggregator = WImplicitAggregator(self.ridge)
        self.density_decoder = DensityAnchorDecoder(d_emb=d_emb,
                                                      hidden=max(d_emb, 16))
        self.use_ema = use_ema
        if use_ema:
            self.target_encoder = copy.deepcopy(self.event_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = self.event_encoder
        self.d_event = d_event
        self.d_emb = d_emb
        self.alpha = alpha
        self.ema_momentum = ema_momentum
        self.vicreg_weight = vicreg_weight
        self.view_weight = view_weight
        self.density_anchor_weight = density_anchor_weight

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def update_target(self) -> None:
        if not self.use_ema:
            return
        tau = self.ema_momentum
        for p_src, p_tgt in zip(self.event_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(tau).add_(p_src.data, alpha=1.0 - tau)

    # ---- forward components ----

    def encode_context(self, X_ctx: torch.Tensor) -> torch.Tensor:
        """Per-scenario manifold-coordinate vector via the ridge aggregator.

        X_ctx: (S, N_ctx, d_event) -> (S, d_emb).
        Internally uses V = stop_grad(target_encoder(X_ctx)) as the target
        embedding, matching the JEPA convention.
        """
        K = self.event_encoder(X_ctx)
        with (torch.no_grad() if self.use_ema else nullcontext()):
            V = self.target_encoder(X_ctx)
        return self.aggregator.summary_vector(K, V)

    def manifold_coord(self, X_ctx: torch.Tensor) -> torch.Tensor:
        """Alias of encode_context, named for the P1-P4 probes."""
        return self.encode_context(X_ctx)

    def predict_embeddings(self, X_ctx: torch.Tensor,
                            X_q: torch.Tensor) -> torch.Tensor:
        """Predict target embeddings at query events via the closed-form ridge."""
        K = self.event_encoder(X_ctx)
        K_q = self.event_encoder(X_q)
        with (torch.no_grad() if self.use_ema else nullcontext()):
            V = self.target_encoder(X_ctx)
        return self.ridge(K, V, K_q)

    def vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Bardes et al. 2022. z flattened over batch + position."""
        z = z.reshape(-1, z.shape[-1])
        n = z.shape[0]
        std = z.std(dim=0, unbiased=False)
        var_term = torch.relu(1.0 - std).mean()
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, n - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_term = (off ** 2).sum() / float(self.d_emb)
        return var_term + cov_term

    # ---- combined forward ----

    def forward(self,
                X_ctx: torch.Tensor,
                X_q: torch.Tensor,
                *,
                X_ctx_view2: torch.Tensor | None = None,
                log_w_q: torch.Tensor | None = None,
                return_losses: bool = False):
        """If return_losses=False, returns Z_pred (S, N_q, d_emb).

        If return_losses=True, returns dict with keys
            jepa, vicreg, view, density, total, z_pred.
        ``X_ctx_view2`` (optional): a second independent event set at the
        same c, for the RS3L view-invariance term.
        ``log_w_q`` (optional): per-query log w_c(x_q) for the density
        anchor (set density_anchor_weight=0 if you don't supply it).
        """
        single = (X_ctx.dim() == 2)
        if single:
            X_ctx = X_ctx.unsqueeze(0)
            X_q = X_q.unsqueeze(0)
            if X_ctx_view2 is not None:
                X_ctx_view2 = X_ctx_view2.unsqueeze(0)
            if log_w_q is not None:
                log_w_q = log_w_q.unsqueeze(0)

        K = self.event_encoder(X_ctx)
        K_q = self.event_encoder(X_q)
        with (torch.no_grad() if self.use_ema else nullcontext()):
            V_ctx = self.target_encoder(X_ctx)
            V_q_target = self.target_encoder(X_q)
        Z_pred = self.ridge(K, V_ctx, K_q)                            # (S, N_q, d)

        if not return_losses:
            return Z_pred.squeeze(0) if single else Z_pred

        # JEPA loss (predict target-encoder embeddings of held-out events).
        jepa_loss = ((Z_pred - V_q_target.detach()) ** 2).mean()

        # VICReg on predicted embeddings.
        vic_loss = self.vicreg_loss(Z_pred)

        # RS3L view-invariance on the per-scenario summary vector.
        view_loss = torch.zeros((), device=X_ctx.device, dtype=X_ctx.dtype)
        if X_ctx_view2 is not None and self.view_weight > 0.0:
            w1 = self.aggregator.summary_vector(K, V_ctx)
            K2 = self.event_encoder(X_ctx_view2)
            with (torch.no_grad() if self.use_ema else nullcontext()):
                V2 = self.target_encoder(X_ctx_view2)
            w2 = self.aggregator.summary_vector(K2, V2)
            view_loss = ((w1 - w2) ** 2).mean()

        # Optional density anchor.
        density_loss = torch.zeros((), device=X_ctx.device, dtype=X_ctx.dtype)
        if log_w_q is not None and self.density_anchor_weight > 0.0:
            log_w_pred = self.density_decoder(Z_pred)                 # (S, N_q)
            density_loss = ((log_w_pred - log_w_q) ** 2).mean()

        total = (jepa_loss
                  + self.vicreg_weight * vic_loss
                  + self.view_weight * view_loss
                  + self.density_anchor_weight * density_loss)

        if single:
            Z_pred = Z_pred.squeeze(0)
        return {
            "jepa": jepa_loss,
            "vicreg": vic_loss,
            "view": view_loss,
            "density": density_loss,
            "total": total,
            "z_pred": Z_pred,
        }

    # ---- numpy helpers ----

    @torch.no_grad()
    def psi_np(self, X: np.ndarray) -> np.ndarray:
        """Per-event embeddings as numpy, for the linear-probe machinery."""
        Xt = torch.from_numpy(X).float()
        return self.event_encoder(Xt).cpu().numpy()

    @torch.no_grad()
    def manifold_coord_np(self, X_ctx: np.ndarray) -> np.ndarray:
        """w_θ(c) as a numpy vector per scenario.

        X_ctx: (N_ctx, d_event) -> (d_emb,) for a single scenario, or
               (S, N_ctx, d_event) -> (S, d_emb) for a batch.
        """
        Xt = torch.from_numpy(X_ctx).float()
        if Xt.dim() == 2:
            return self.manifold_coord(Xt.unsqueeze(0)).squeeze(0).cpu().numpy()
        return self.manifold_coord(Xt).cpu().numpy()


if __name__ == "__main__":
    # Smoke test on a small batch.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    S, N_ctx, N_q, d_event, d_emb = 8, 256, 64, 2, 16
    X_ctx = torch.randn(S, N_ctx, d_event)
    X_q = torch.randn(S, N_q, d_event)
    X_view2 = torch.randn(S, N_ctx, d_event)
    log_w_q = torch.randn(S, N_q) * 0.1

    model = ManifoldInformer(d_event=d_event, d_emb=d_emb, hidden=32)
    print(f"ManifoldInformer n_params = {model.n_params}")

    # Eval.
    Z = model(X_ctx, X_q)
    print(f"  eval Z shape = {tuple(Z.shape)}")

    # Train with all four terms.
    out = model(X_ctx, X_q, X_ctx_view2=X_view2, log_w_q=log_w_q,
                return_losses=True)
    print(f"  train losses:  jepa={out['jepa'].item():.4f}  "
          f"vicreg={out['vicreg'].item():.4f}  "
          f"view={out['view'].item():.4f}  "
          f"density={out['density'].item():.4f}  "
          f"total={out['total'].item():.4f}")

    model.update_target()
    print("  EMA update OK.")

    # Numpy helpers.
    psi_np = model.psi_np(rng.standard_normal((128, d_event)).astype(np.float32))
    print(f"  psi_np shape = {psi_np.shape}")
    w_np = model.manifold_coord_np(
        rng.standard_normal((4, 256, d_event)).astype(np.float32))
    print(f"  manifold_coord_np shape = {w_np.shape}")
