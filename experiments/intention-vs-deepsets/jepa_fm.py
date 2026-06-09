"""JEPA-FM: Joint-Embedding Predictive Architecture for SMEFT in-context regression.

Adapts T-JEPA (Thimonier et al., ICLR 2025; arXiv:2410.05016) to the Intention-
vs-DeepSets setup where a *scenario* is a sample whose "features" are K context
events ``(m_i, y_i)`` and Q query events ``(m_q, y_q)``. The context encoder
embeds the K context events and pools them; the EMA target encoder embeds the
Q query events; a predictor reconstructs each target event's embedding from
the context summary plus the target's mass coordinate. The Wilson coefficient
``c`` never enters the forward pass (BRIEF constraint).

Five modules:
  E   — per-event encoder phi(m, y) -> R^{d_emb}
  E_q — query-position encoder phi_q(m_q) -> R^{d_emb}  (analog of T-JEPA's
        learnable mask token: gives the predictor a position-only embedding)
  f_θ — context aggregator: mean-pool over K embeddings + MLP refinement
  g_φ — predictor: (summary, query_pos_emb) -> predicted target embedding
  d   — auxiliary decoder: embedding -> scalar Y prediction
  E_ema — EMA copy of E (stop-grad target encoder)

Loss (training):
  L_total = L_JEPA + λ_VIC · L_VICReg + λ_aux · L_MSE
  L_JEPA   = mean ‖g_φ(z_ctx, E_q(m_q)) − stop_grad( E_ema(m_q, y_q) )‖²
  L_VICReg = max(0, 1 − std(z_pred, dim=batch)).mean()
             + off_diag(cov(z_pred))² .sum() / d_emb
  L_MSE    = mean (d(g_φ(z_ctx, E_q(m_q))) − y_q)²    (R² readout head)

Defaults give a parameter-matched comparison (~6k trainable) to IntentionFM_Learned
(5,328) and DeepSets-FM (6,545); the EMA target encoder is NOT counted toward
trainable params.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

M_REF = 1.0   # match the other models' reference scale


def _log_m(m: torch.Tensor) -> torch.Tensor:
    return torch.log(m / M_REF)


class PerEventEncoder(nn.Module):
    """phi(m, y) -> R^{d_emb}. Shared between the gradient encoder and the
    EMA target encoder (the latter is a deepcopy with frozen grads, updated
    by EMA in JEPAFM.update_target)."""

    def __init__(self, d_emb: int = 16, hidden: int = 48):
        super().__init__()
        self.d_emb = d_emb
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """m, y shape (...,) -> (..., d_emb). Stacks scaled-m and y as the
        per-event content vector."""
        x = torch.stack([_log_m(m), y], dim=-1)
        return self.net(x)


class QueryPosEncoder(nn.Module):
    """phi_q(m_q) -> R^{d_emb}. T-JEPA's mask-token analog: gives the
    predictor a position-only signal for where to predict, without leaking y."""

    def __init__(self, d_emb: int = 16, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, m_q: torch.Tensor) -> torch.Tensor:
        x = _log_m(m_q).unsqueeze(-1)
        return self.net(x)


class ContextAggregator(nn.Module):
    """Mean-pool over K event embeddings, then refine with an MLP. The
    aggregator is what makes this a DeepSets-style context encoder. Replacing
    it with a small self-attention block is the natural transformer upgrade
    if the matched-budget JEPA-FM shows promise."""

    def __init__(self, d_emb: int = 16, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """embs: (S, K, d_emb) -> (S, d_emb)."""
        pooled = embs.mean(dim=1)
        return self.net(pooled)


class Predictor(nn.Module):
    """g_φ(summary, query_pos_emb) -> predicted target embedding in the
    same space as E_ema's per-event outputs."""

    def __init__(self, d_emb: int = 16, hidden: int = 48):
        super().__init__()
        self.d_emb = d_emb
        self.net = nn.Sequential(
            nn.Linear(2 * d_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, summary: torch.Tensor,
                query_pos_emb: torch.Tensor) -> torch.Tensor:
        """summary: (S, d_emb); query_pos_emb: (S, Q, d_emb) -> (S, Q, d_emb)."""
        S, Q, d = query_pos_emb.shape
        summ_exp = summary.unsqueeze(1).expand(-1, Q, -1)
        x = torch.cat([summ_exp, query_pos_emb], dim=-1)
        return self.net(x.reshape(S * Q, -1)).reshape(S, Q, d)


class Decoder(nn.Module):
    """Auxiliary readout: predicted embedding -> scalar Y. Tiny MLP, jointly
    trained with the JEPA loss at weight λ_aux. The R² comparison uses this
    head's output."""

    def __init__(self, d_emb: int = 16, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class TransformerContextEncoder(nn.Module):
    """T-JEPA-faithful context encoder: small transformer over K event tokens
    plus a learnable [CLS] token that pools the scenario summary. This is the
    drop-in replacement for ContextAggregator that lets the JEPA recipe use
    its native ingestion path (Thimonier et al., T-JEPA ICLR 2025, §3
    "Context and Target Encoders")."""

    def __init__(self, d_emb: int = 64, n_layers: int = 2, n_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_emb))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_heads,
            dim_feedforward=ff_mult * d_emb, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_emb)

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """embs: (S, K, d_emb) -> summary (S, d_emb).
        Prepends a learnable [CLS] token, runs self-attention, returns the
        CLS output after a final LayerNorm. The CLS-pool replaces mean-pool
        as the scenario summary."""
        S = embs.shape[0]
        cls = self.cls.expand(S, -1, -1)                  # (S, 1, d_emb)
        h = torch.cat([cls, embs], dim=1)                  # (S, K+1, d_emb)
        h = self.tx(h)
        return self.norm(h[:, 0])                          # (S, d_emb)


class TransformerPredictor(nn.Module):
    """Transformer-based predictor: takes (context summary, query-pos token)
    -> predicted target embedding via self-attention. Matches T-JEPA's
    "predictor is a transformer encoder" choice, with the hidden dim downsized
    via a linear projection (Thimonier et al. §3 'Predictor')."""

    def __init__(self, d_emb: int = 64, predictor_dim: int | None = None,
                 n_layers: int = 2, n_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.d_emb = d_emb
        h = predictor_dim or max(16, d_emb // 2)
        self.h = h
        self.proj_in_summary = nn.Linear(d_emb, h)
        self.proj_in_q = nn.Linear(d_emb, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=ff_mult * h,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj_out = nn.Linear(h, d_emb)

    def forward(self, summary: torch.Tensor,
                query_pos_emb: torch.Tensor) -> torch.Tensor:
        """summary: (S, d_emb); query_pos_emb: (S, Q, d_emb) -> (S, Q, d_emb).
        For each query, attend over (summary_token, query_pos_token); return
        the post-attention query-pos token projected back to d_emb."""
        S, Q, d = query_pos_emb.shape
        s = self.proj_in_summary(summary)                       # (S, h)
        q = self.proj_in_q(query_pos_emb)                       # (S, Q, h)
        # Build (S*Q, 2, h): each row is [summary_token, query_token]
        s_exp = s.unsqueeze(1).expand(-1, Q, -1).reshape(S * Q, 1, self.h)
        q_flat = q.reshape(S * Q, 1, self.h)
        seq = torch.cat([s_exp, q_flat], dim=1)                 # (S*Q, 2, h)
        out = self.tx(seq)
        q_out = out[:, 1]                                       # (S*Q, h)
        return self.proj_out(q_out).reshape(S, Q, d)


class TransformerPerEventEncoder(nn.Module):
    """Two-MLP per-event encoder producing d_emb token embeddings. Slightly
    deeper than PerEventEncoder so the transformer has tokens with non-trivial
    per-event content."""

    def __init__(self, d_emb: int = 64, hidden: int = 128):
        super().__init__()
        self.d_emb = d_emb
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.stack([_log_m(m), y], dim=-1)
        return self.net(x)


class TransformerQueryPosEncoder(nn.Module):
    def __init__(self, d_emb: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, m_q: torch.Tensor) -> torch.Tensor:
        x = _log_m(m_q).unsqueeze(-1)
        return self.net(x)


class JEPAFM(nn.Module):
    """JEPA Foundation Model for in-context regression.

    Public API matches IntentionFMLearned / DeepSetsFM:
        eval/predict:  forward(M_ctx, Y_ctx, M_q) -> Y_pred  (S, Q) or (Q,)
        train:         forward(M_ctx, Y_ctx, M_q, Y_q, return_losses=True)
                       -> (Y_pred, jepa_loss, vic_loss, mse_loss, z_pred)
    """

    def __init__(self,
                 d_emb: int = 16,
                 hidden: int = 48,
                 ema_momentum: float = 0.996,
                 vicreg_weight: float = 0.04,
                 aux_weight: float = 0.1,
                 jepa_weight: float = 1.0,
                 use_ema: bool = True,
                 encoder: str = "deepsets",
                 n_layers: int = 2,
                 n_heads: int = 4,
                 ff_mult: int = 4,
                 predictor_dim: int | None = None):
        """encoder ∈ {"deepsets", "transformer"}. The "transformer" variant
        replaces ContextAggregator with a multi-layer self-attention encoder
        (T-JEPA-faithful) and Predictor with TransformerPredictor; the
        per-event and query-pos encoders become 3-layer MLPs to give the
        attention layer richer per-event tokens."""
        super().__init__()
        self.encoder_kind = encoder
        if encoder == "transformer":
            self.event_encoder = TransformerPerEventEncoder(d_emb=d_emb, hidden=hidden)
            self.query_pos_encoder = TransformerQueryPosEncoder(d_emb=d_emb, hidden=hidden)
            self.context_aggregator = TransformerContextEncoder(
                d_emb=d_emb, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult)
            self.predictor = TransformerPredictor(
                d_emb=d_emb, predictor_dim=predictor_dim,
                n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult)
        elif encoder == "deepsets":
            self.event_encoder = PerEventEncoder(d_emb=d_emb, hidden=hidden)
            self.query_pos_encoder = QueryPosEncoder(d_emb=d_emb, hidden=hidden)
            self.context_aggregator = ContextAggregator(d_emb=d_emb, hidden=hidden)
            self.predictor = Predictor(d_emb=d_emb, hidden=hidden)
        else:
            raise ValueError(f"unknown encoder={encoder!r}")
        self.decoder = Decoder(d_emb=d_emb, hidden=max(d_emb, 32))

        self.use_ema = use_ema
        if use_ema:
            self.target_event_encoder = copy.deepcopy(self.event_encoder)
            for p in self.target_event_encoder.parameters():
                p.requires_grad = False
        else:
            # Siamese: target = grad encoder (no EMA, full gradient flow).
            self.target_event_encoder = self.event_encoder

        self.d_emb = d_emb
        self.ema_momentum = ema_momentum
        self.vicreg_weight = vicreg_weight
        self.aux_weight = aux_weight
        self.jepa_weight = jepa_weight

    @property
    def n_params(self) -> int:
        # Trainable only — EMA target encoder is not double-counted.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of the target encoder: theta_bar <- tau*theta_bar + (1-tau)*theta."""
        if not self.use_ema:
            return
        tau = self.ema_momentum
        for p_src, p_tgt in zip(self.event_encoder.parameters(),
                                self.target_event_encoder.parameters()):
            p_tgt.data.mul_(tau).add_(p_src.data, alpha=1.0 - tau)

    # -- forward components -------------------------------------------------

    def encode_context(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor) -> torch.Tensor:
        """(S, K), (S, K) -> (S, d_emb)."""
        embs = self.event_encoder(M_ctx, Y_ctx)
        return self.context_aggregator(embs)

    def predict_embedding(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                          M_q: torch.Tensor) -> torch.Tensor:
        """(S, K), (S, K), (S, Q) -> (S, Q, d_emb)."""
        summary = self.encode_context(M_ctx, Y_ctx)
        q_pos = self.query_pos_encoder(M_q)
        return self.predictor(summary, q_pos)

    def encode_target(self, M_q: torch.Tensor, Y_q: torch.Tensor) -> torch.Tensor:
        """EMA-encoded target embeddings, with stop-grad. (S, Q) -> (S, Q, d_emb)."""
        ctx = torch.no_grad() if self.use_ema else nullcontext()
        with ctx:
            return self.target_event_encoder(M_q, Y_q)

    def vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Bardes et al. 2022. z: (S, Q, d_emb) -> scalar."""
        z = z.reshape(-1, z.shape[-1])
        n = z.shape[0]
        # Variance term: per-dim std hinged at 1 (encourage spread).
        std = z.std(dim=0, unbiased=False)
        var_term = torch.relu(1.0 - std).mean()
        # Covariance term: off-diagonal cov elements squared.
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, n - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_term = (off ** 2).sum() / float(self.d_emb)
        return var_term + cov_term

    # -- combined forward ---------------------------------------------------

    def forward(self,
                M_ctx: torch.Tensor,
                Y_ctx: torch.Tensor,
                M_q: torch.Tensor,
                Y_q: torch.Tensor | None = None,
                return_losses: bool = False):
        single = (M_ctx.dim() == 1)
        if single:
            M_ctx = M_ctx.unsqueeze(0)
            Y_ctx = Y_ctx.unsqueeze(0)
            M_q = M_q.unsqueeze(0)
            if Y_q is not None:
                Y_q = Y_q.unsqueeze(0)

        z_pred = self.predict_embedding(M_ctx, Y_ctx, M_q)        # (S, Q, d)
        y_pred = self.decoder(z_pred)                              # (S, Q)

        if not return_losses:
            return y_pred.squeeze(0) if single else y_pred

        assert Y_q is not None, "return_losses=True requires Y_q"
        z_target = self.encode_target(M_q, Y_q)                    # (S, Q, d)
        # Detach is redundant under no_grad but kept for the no-EMA siamese case
        # where we still want to stop the JEPA loss from updating the target path.
        jepa_loss = ((z_pred - z_target.detach()) ** 2).mean()
        vic_loss = self.vicreg_loss(z_pred)
        mse_loss = ((y_pred - Y_q) ** 2).mean()
        if single:
            y_pred = y_pred.squeeze(0)
        return y_pred, jepa_loss, vic_loss, mse_loss, z_pred

    # -- numpy / eval helpers -----------------------------------------------

    @torch.no_grad()
    def predict_np(self, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                   M_q: np.ndarray) -> np.ndarray:
        Mc = torch.from_numpy(M_ctx).float()
        Yc = torch.from_numpy(Y_ctx).float()
        Mq = torch.from_numpy(M_q).float()
        return self.forward(Mc, Yc, Mq).cpu().numpy()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/vince/ALETHIA/experiments/intention-vs-deepsets")
    from data import make_oracle, make_scenario
    rng = np.random.default_rng(0)
    o = make_oracle()
    c = rng.uniform(-0.5, 0.5, 4)
    s = make_scenario(o, c, rng)
    m = JEPAFM()
    print("JEPAFM n_params (trainable) =", m.n_params)
    Mc = torch.from_numpy(s["M_ctx"]).unsqueeze(0)
    Yc = torch.from_numpy(s["Y_ctx"]).unsqueeze(0)
    Mq = torch.from_numpy(s["M_query"]).unsqueeze(0)
    Yq = torch.from_numpy(s["Y_query"]).unsqueeze(0)
    out = m(Mc, Yc, Mq, Yq, return_losses=True)
    yp, jl, vl, ml, z = out
    print("y_pred shape =", tuple(yp.shape), "z_pred shape =", tuple(z.shape))
    print(f"untrained jepa={jl.item():.4f}  vic={vl.item():.4f}  mse={ml.item():.4f}")
    m.update_target()
    print("EMA update OK.")
