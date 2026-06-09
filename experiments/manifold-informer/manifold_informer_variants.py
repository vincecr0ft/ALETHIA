r"""Aggregator variants of the ManifoldInformer for the certificate comparison.

The novelty memo's A.3 claims the *manifold-identity certificate* (P3 tangent
∂w/∂c ≈ A_i, P4 curvature ∂²w/∂c² ≈ B_ij) is what distinguishes the
ManifoldInformer, and A.5 commits to reporting it honestly if a competing
architecture matches it. To make that an empirical claim rather than an
assertion, this module provides drop-in event-set FMs that share the
ManifoldInformer's JEPA machinery (same per-event encoder, EMA target,
VICReg, RS3L view term, density anchor, identical training loop and data)
and differ ONLY in how the context event set becomes the per-scenario
manifold coordinate w_θ and how query embeddings are predicted:

  MeanPoolInformer  — DeepSets aggregator: w = mean_n phi(x_n); query
                      embedding predicted by an MLP on [w, phi(x_q)].
  AttnInformer      — Set-Transformer aggregator: w = PMA(phi(X)) (pooling
                      by multihead attention to a learned seed); same MLP
                      predictor head.

Both expose the exact API the P1-P4 gates consume
(`manifold_coord(X_ctx) -> (S, d_emb)`, `forward(..., return_losses=True)`,
`update_target()`, `d_emb`), so manifold_informer_gates.gate_P3_P4 /
gate_P2 / gate_P1_probe_linearity run on them unchanged. The closed-form
ridge ManifoldInformer (manifold_informer.py) is the third arm of the
comparison.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.nn as nn

from manifold_informer import PerEventEncoder, DensityAnchorDecoder


class _JEPAInformerBase(nn.Module):
    """Shared JEPA/VICReg/view/density scaffolding; subclasses supply the
    aggregator (`_summary`) and the query predictor (`_predict`)."""

    def __init__(self, d_event=2, d_emb=16, hidden=32, ema_momentum=0.996,
                 vicreg_weight=0.04, view_weight=0.1,
                 density_anchor_weight=0.05, use_ema=True):
        super().__init__()
        self.event_encoder = PerEventEncoder(d_event=d_event, d_emb=d_emb,
                                             hidden=hidden)
        self.density_decoder = DensityAnchorDecoder(d_emb=d_emb,
                                                    hidden=max(d_emb, 16))
        self.use_ema = use_ema
        if use_ema:
            self.target_encoder = copy.deepcopy(self.event_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = self.event_encoder
        self.d_event, self.d_emb = d_event, d_emb
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
        for ps, pt in zip(self.event_encoder.parameters(),
                          self.target_encoder.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)

    # ---- aggregator + predictor: provided by subclasses ----
    def _summary(self, K: torch.Tensor) -> torch.Tensor:
        """K = phi(X_ctx): (S, N, d_emb) -> per-scenario w: (S, d_emb)."""
        raise NotImplementedError

    def _predict(self, w: torch.Tensor, K_q: torch.Tensor) -> torch.Tensor:
        """w: (S, d_emb), K_q = phi(X_q): (S, Nq, d_emb) -> Z_pred (S, Nq, d_emb)."""
        raise NotImplementedError

    # ---- manifold coordinate (what the gates probe) ----
    def manifold_coord(self, X_ctx: torch.Tensor) -> torch.Tensor:
        single = (X_ctx.dim() == 2)
        if single:
            X_ctx = X_ctx.unsqueeze(0)
        K = self.event_encoder(X_ctx)
        w = self._summary(K)
        return w.squeeze(0) if single else w

    encode_context = manifold_coord

    def vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, z.shape[-1])
        n = z.shape[0]
        std = z.std(dim=0, unbiased=False)
        var_term = torch.relu(1.0 - std).mean()
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, n - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_term = (off ** 2).sum() / float(self.d_emb)
        return var_term + cov_term

    def forward(self, X_ctx, X_q, *, X_ctx_view2=None, log_w_q=None,
                return_losses=False):
        single = (X_ctx.dim() == 2)
        if single:
            X_ctx, X_q = X_ctx.unsqueeze(0), X_q.unsqueeze(0)
            if X_ctx_view2 is not None:
                X_ctx_view2 = X_ctx_view2.unsqueeze(0)
            if log_w_q is not None:
                log_w_q = log_w_q.unsqueeze(0)

        K = self.event_encoder(X_ctx)
        K_q = self.event_encoder(X_q)
        with (torch.no_grad() if self.use_ema else nullcontext()):
            V_q_target = self.target_encoder(X_q)
        w = self._summary(K)
        Z_pred = self._predict(w, K_q)

        if not return_losses:
            return Z_pred.squeeze(0) if single else Z_pred

        jepa = ((Z_pred - V_q_target.detach()) ** 2).mean()
        vic = self.vicreg_loss(Z_pred)
        view = torch.zeros((), device=X_ctx.device, dtype=X_ctx.dtype)
        if X_ctx_view2 is not None and self.view_weight > 0.0:
            K2 = self.event_encoder(X_ctx_view2)
            view = ((w - self._summary(K2)) ** 2).mean()
        density = torch.zeros((), device=X_ctx.device, dtype=X_ctx.dtype)
        if log_w_q is not None and self.density_anchor_weight > 0.0:
            density = ((self.density_decoder(Z_pred) - log_w_q) ** 2).mean()
        total = (jepa + self.vicreg_weight * vic + self.view_weight * view
                 + self.density_anchor_weight * density)
        if single:
            Z_pred = Z_pred.squeeze(0)
        return {"jepa": jepa, "vicreg": vic, "view": view,
                "density": density, "total": total, "z_pred": Z_pred}


class _MLPPredictor(nn.Module):
    """Shared query head: Z_pred(x_q) = MLP([w, phi(x_q)])."""

    def __init__(self, d_emb, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_emb, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_emb))

    def forward(self, w, K_q):
        S, Nq, d = K_q.shape
        w_exp = w.unsqueeze(1).expand(-1, Nq, -1)
        return self.net(torch.cat([w_exp, K_q], dim=-1))


class MeanPoolInformer(_JEPAInformerBase):
    """DeepSets aggregator: w = mean over events of phi(x)."""

    def __init__(self, d_event=2, d_emb=16, hidden=32, **kw):
        super().__init__(d_event=d_event, d_emb=d_emb, hidden=hidden, **kw)
        self.predictor = _MLPPredictor(d_emb, hidden)

    def _summary(self, K):
        return K.mean(dim=-2)

    def _predict(self, w, K_q):
        return self.predictor(w, K_q)


class AttnInformer(_JEPAInformerBase):
    """Set-Transformer aggregator: w = PMA(phi(X)) (attention pooling)."""

    def __init__(self, d_event=2, d_emb=16, hidden=32, n_heads=4, **kw):
        super().__init__(d_event=d_event, d_emb=d_emb, hidden=hidden, **kw)
        self.seed = nn.Parameter(torch.randn(1, 1, d_emb) * 0.1)
        self.attn = nn.MultiheadAttention(d_emb, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_emb)
        self.predictor = _MLPPredictor(d_emb, hidden)

    def _summary(self, K):
        S = K.shape[0]
        q = self.seed.expand(S, -1, -1)
        z, _ = self.attn(q, K, K)
        return self.ln(z.squeeze(1))

    def _predict(self, w, K_q):
        return self.predictor(w, K_q)


VARIANTS = {
    "MeanPool": lambda: MeanPoolInformer(),
    "Attention": lambda: AttnInformer(),
}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    S, N, Nq = 6, 200, 50
    X = torch.from_numpy(rng.standard_normal((S, N, 2)).astype(np.float32))
    Xq = torch.from_numpy(rng.standard_normal((S, Nq, 2)).astype(np.float32))
    X2 = torch.from_numpy(rng.standard_normal((S, N, 2)).astype(np.float32))
    lw = torch.from_numpy((rng.standard_normal((S, Nq)) * 0.1).astype(np.float32))
    for name, f in VARIANTS.items():
        m = f()
        z = m(X, Xq)
        out = m(X, Xq, X_ctx_view2=X2, log_w_q=lw, return_losses=True)
        w = m.manifold_coord(X)
        m.update_target()
        print(f"{name:10s} n_params={m.n_params:6d}  Z{tuple(z.shape)}  "
              f"w{tuple(w.shape)}  total={out['total'].item():.4f}")
