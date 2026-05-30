"""IntentionJEPAFM: T-JEPA-trained foundation model with the Intention basis
as the per-event embedding.

The existing IntentionFM_Learned is a closed-form ridge regression in a learned
basis psi_theta(m); training is supervised MSE backprop'd through
torch.linalg.solve. The existing JEPA-FM is a generic transformer (or DeepSets)
encoder over (log_m, y) pairs, trained with the T-JEPA recipe (predict masked
target-event embeddings via an EMA target encoder). Neither is both: this file
adds the missing piece.

Design (one sentence): replace JEPA-FM's per-event encoder phi(m, y) with an
*Intention-basis* token encoder that feeds (psi_theta(m), psi_theta(m)*y, y)
through a small MLP into d_emb, keep the transformer aggregator + predictor +
EMA target encoder + VICReg + aux-MSE machinery from jepa_fm.py, and train end-
to-end with the full T-JEPA loss. There is no closed-form ridge solve anywhere.

Why this is the right "intention-based embedding". The closed-form ridge head
computes w = (Psi^T Psi + alpha I)^{-1} Psi^T Y, then predicts Psi_q . w. The
per-event quantity Psi^T Y is sum_k y_k psi(m_k) — i.e. a y-weighted
accumulation in the Intention basis. The multiplicative feature
``psi_theta(m) * y`` is the per-token analogue of that accumulation, given to a
learned attention aggregator instead of being analytically inverted. So the
network has the same access to the ridge-solve sufficient statistics, but the
aggregator that combines them is trained instead of closed-form.

Public API matches JEPAFM exactly so the existing experiment harnesses can
treat the two interchangeably:

    forward(M_ctx, Y_ctx, M_q)                         -> Y_pred         (eval)
    forward(M_ctx, Y_ctx, M_q, Y_q, return_losses=True)
       -> (Y_pred, jepa_loss, vic_loss, mse_loss, z_pred)                (train)
    update_target()                                    -> EMA step
    encode_target(M_q, Y_q)                            -> (S, Q, d_emb)
    predict_embedding(M_ctx, Y_ctx, M_q)               -> (S, Q, d_emb)
    encode_context(M_ctx, Y_ctx)                       -> (S, d_emb)
    predict_np(M_ctx, Y_ctx, M_q)                      -> Y_pred         (numpy)
    A_inv_and_w(M_ctx, Y_ctx)                          -> (A_inv, w, Psi)
    psi_np(M)                                          -> psi (numpy)

The last two preserve the closed-form diagnostic access path that the chain
(`drift.py`, `acquisition.py`, `identifiability_probe.py`) uses. They are
*never* used for predictions in IntentionJEPAFM — the prediction path is the
trained transformer + decoder — but exposing psi_theta as numpy means the same
linear-probe disclosure machinery works without modification.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn


M_REF = 1.0   # TeV; matches IntentionFM and JEPA-FM


def _log_m(m: torch.Tensor) -> torch.Tensor:
    return torch.log(m / M_REF)


class PsiMLP(nn.Module):
    """psi_theta: scalar log(m/M_REF) -> R^{d_psi}. Same shape as
    intention_learned.PsiMLP and modules.surrogate.intention.model.PsiMLP so
    weights are interchangeable (a future warm-start path can lift psi_theta
    from a pretrained IntentionFM_Learned checkpoint)."""

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
        x = _log_m(m).unsqueeze(-1)
        return self.net(x)


class IntentionPerEventEncoder(nn.Module):
    """phi(m, y) -> R^{d_emb} via the Intention basis.

    Token feature = MLP([psi(m), psi(m)*y, y]). The psi(m)*y channel gives the
    network the per-event analogue of the closed-form ridge sufficient
    statistic Psi^T Y. The bare-y scalar lets it recover raw rate info.

    Shares the psi module with the rest of the FM (NOT a copy): one psi_theta
    is jointly used by the per-event encoder, the query-pos encoder, and the
    EMA target encoder (with its own copy of the projection head).
    """

    def __init__(self, psi: PsiMLP, d_emb: int = 16, hidden: int = 48):
        super().__init__()
        self.psi = psi
        self.d_emb = d_emb
        self.proj = nn.Sequential(
            nn.Linear(2 * psi.d_psi + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        psi_m = self.psi(m)                                        # (..., d_psi)
        psi_m_y = psi_m * y.unsqueeze(-1)                          # (..., d_psi)
        feat = torch.cat([psi_m, psi_m_y, y.unsqueeze(-1)], dim=-1)
        return self.proj(feat)


class IntentionQueryPosEncoder(nn.Module):
    """phi_q(m_q) -> R^{d_emb}, the T-JEPA mask-token analogue.

    Token feature = MLP([psi(m_q), zeros_d_psi, zeros_1]) — same input layout as
    IntentionPerEventEncoder so the basis is shared, but the y-dependent slots
    are zeroed out. Equivalent to "we know the position in m, not the label."
    """

    def __init__(self, psi: PsiMLP, d_emb: int = 16, hidden: int = 48):
        super().__init__()
        self.psi = psi
        self.d_emb = d_emb
        self.proj = nn.Sequential(
            nn.Linear(2 * psi.d_psi + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, m_q: torch.Tensor) -> torch.Tensor:
        psi_m = self.psi(m_q)                                      # (..., d_psi)
        z_psi = torch.zeros_like(psi_m)
        z_y = torch.zeros(psi_m.shape[:-1] + (1,),
                          dtype=psi_m.dtype, device=psi_m.device)
        feat = torch.cat([psi_m, z_psi, z_y], dim=-1)
        return self.proj(feat)


class TransformerContextEncoder(nn.Module):
    """Self-attention over K context tokens with a learnable CLS pool.

    Replaces JEPA-FM's mean-pool ContextAggregator. The CLS output is the
    scenario summary z_ctx that the predictor cross-attends to. Norm-first
    layers (Pre-LN) for stable training at small batch.
    """

    def __init__(self, d_emb: int = 16, n_layers: int = 1, n_heads: int = 4,
                 ff_mult: int = 2, dropout: float = 0.0):
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
        S = embs.shape[0]
        cls = self.cls.expand(S, -1, -1)                            # (S, 1, d)
        h = torch.cat([cls, embs], dim=1)                            # (S, K+1, d)
        h = self.tx(h)
        return self.norm(h[:, 0])                                    # (S, d)


class CrossAttnPredictor(nn.Module):
    """g_phi(summary, q_pos_emb) -> predicted target embedding.

    Implements one cross-attention head with the query as the q_pos token and
    the (summary, q_pos) pair as the key/value pool, then an MLP head. This is
    cheaper than two encoder layers and gives the predictor exactly the data
    it needs (context summary + position hint) without further self-attention.
    """

    def __init__(self, d_emb: int = 16, hidden: int = 32, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.d_emb = d_emb
        self.attn = nn.MultiheadAttention(d_emb, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_emb)
        self.ff = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_emb),
        )
        self.norm2 = nn.LayerNorm(d_emb)

    def forward(self, summary: torch.Tensor,
                query_pos_emb: torch.Tensor) -> torch.Tensor:
        """summary: (S, d); query_pos_emb: (S, Q, d) -> (S, Q, d)."""
        S, Q, d = query_pos_emb.shape
        # Per-query KV pool of size 2: [summary, q_pos]. Built as (S*Q, 2, d).
        s_exp = summary.unsqueeze(1).expand(-1, Q, -1).reshape(S * Q, 1, d)
        q_flat = query_pos_emb.reshape(S * Q, 1, d)
        kv = torch.cat([s_exp, q_flat], dim=1)                       # (S*Q, 2, d)
        # Query is the q_pos token.
        q = q_flat
        h, _ = self.attn(q, kv, kv, need_weights=False)
        h = self.norm1(q + h)
        h = self.norm2(h + self.ff(h))
        return h.reshape(S, Q, d)


class Decoder(nn.Module):
    """Auxiliary readout: predicted embedding -> scalar Y.

    Trained jointly via aux_weight * MSE. Inference uses this head's output
    for the R^2 comparison; the JEPA pretext is what shapes psi_theta.
    """

    def __init__(self, d_emb: int = 16, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class IntentionJEPAFM(nn.Module):
    """Foundation model with Intention-basis embeddings + JEPA-style training.

    No closed-form ridge anywhere in the prediction path; psi_theta is trained
    end-to-end as a shared basis between the per-event encoder, the query-pos
    encoder, and the EMA target encoder.

    Args:
        d_psi: width of the Intention basis psi_theta.
        d_emb: token width fed to the transformer aggregator.
        psi_hidden: hidden width of the psi_theta MLP.
        encode_hidden: hidden width of the per-event / query-pos projection.
        n_layers: transformer encoder layers.
        n_heads: attention heads (must divide d_emb).
        ff_mult: feed-forward expansion in the transformer encoder.
        predictor_hidden: hidden width of the cross-attention predictor's FF.
        decoder_hidden: hidden width of the Y readout decoder.
        ema_momentum: EMA decay tau for the target encoder.
        vicreg_weight, aux_weight, jepa_weight: loss weights.
        use_ema: if False, siamese target = source (no EMA, full grad flow).
        share_query_psi: if True (default), the query-pos encoder shares its
            psi_theta with the per-event encoder. If False, it owns a copy.
            Sharing is the right default — m-coordinate is m-coordinate.
    """

    def __init__(self,
                 d_psi: int = 16,
                 d_emb: int = 16,
                 psi_hidden: int = 32,
                 encode_hidden: int = 32,
                 n_layers: int = 1,
                 n_heads: int = 4,
                 ff_mult: int = 2,
                 predictor_hidden: int = 32,
                 decoder_hidden: int = 16,
                 ema_momentum: float = 0.996,
                 vicreg_weight: float = 0.04,
                 aux_weight: float = 0.1,
                 jepa_weight: float = 1.0,
                 use_ema: bool = True,
                 share_query_psi: bool = True):
        super().__init__()
        if d_emb % n_heads != 0:
            raise ValueError(f"d_emb={d_emb} must be divisible by n_heads={n_heads}")

        # The Intention basis itself — the "intention-based embedding".
        self.psi = PsiMLP(d_psi=d_psi, hidden=psi_hidden)

        # Per-event encoder + query-pos encoder share psi by default.
        self.event_encoder = IntentionPerEventEncoder(
            self.psi, d_emb=d_emb, hidden=encode_hidden)
        if share_query_psi:
            self.query_pos_encoder = IntentionQueryPosEncoder(
                self.psi, d_emb=d_emb, hidden=encode_hidden)
        else:
            self.query_pos_encoder = IntentionQueryPosEncoder(
                PsiMLP(d_psi=d_psi, hidden=psi_hidden),
                d_emb=d_emb, hidden=encode_hidden)

        self.context_aggregator = TransformerContextEncoder(
            d_emb=d_emb, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult)
        self.predictor = CrossAttnPredictor(
            d_emb=d_emb, hidden=predictor_hidden, n_heads=n_heads)
        self.decoder = Decoder(d_emb=d_emb, hidden=decoder_hidden)

        # EMA target encoder: separate psi + projection so it can be lagged
        # without entangling with the gradient encoder. EMA covers the whole
        # event_encoder (psi + projection).
        self.use_ema = use_ema
        if use_ema:
            self.target_event_encoder = copy.deepcopy(self.event_encoder)
            for p in self.target_event_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_event_encoder = self.event_encoder

        self.d_psi = d_psi
        self.d_emb = d_emb
        self.alpha = 1e-3  # kept only so A_inv_and_w() is well-defined for probes
        self.ema_momentum = ema_momentum
        self.vicreg_weight = vicreg_weight
        self.aux_weight = aux_weight
        self.jepa_weight = jepa_weight

    @property
    def n_params(self) -> int:
        """Trainable params only — EMA target encoder is not double-counted."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of the target event-encoder."""
        if not self.use_ema:
            return
        tau = self.ema_momentum
        for p_src, p_tgt in zip(self.event_encoder.parameters(),
                                self.target_event_encoder.parameters()):
            p_tgt.data.mul_(tau).add_(p_src.data, alpha=1.0 - tau)

    # -- pieces of the forward pass ----------------------------------------

    def encode_context(self, M_ctx: torch.Tensor,
                       Y_ctx: torch.Tensor) -> torch.Tensor:
        """(S, K), (S, K) -> (S, d_emb)."""
        embs = self.event_encoder(M_ctx, Y_ctx)                      # (S, K, d_emb)
        return self.context_aggregator(embs)

    def predict_embedding(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                          M_q: torch.Tensor) -> torch.Tensor:
        """(S, K), (S, K), (S, Q) -> (S, Q, d_emb)."""
        summary = self.encode_context(M_ctx, Y_ctx)
        q_pos = self.query_pos_encoder(M_q)
        return self.predictor(summary, q_pos)

    def encode_target(self, M_q: torch.Tensor,
                      Y_q: torch.Tensor) -> torch.Tensor:
        """EMA target encoder of (m_q, y_q). Stop-grad."""
        ctx = torch.no_grad() if self.use_ema else nullcontext()
        with ctx:
            return self.target_event_encoder(M_q, Y_q)

    def vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Bardes et al. 2022. z: (S, Q, d_emb) -> scalar. Identical to
        jepa_fm.JEPAFM.vicreg_loss; kept here so the file is self-contained."""
        z = z.reshape(-1, z.shape[-1])
        n = z.shape[0]
        std = z.std(dim=0, unbiased=False)
        var_term = torch.relu(1.0 - std).mean()
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, n - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_term = (off ** 2).sum() / float(self.d_emb)
        return var_term + cov_term

    # -- combined forward --------------------------------------------------

    def forward(self,
                M_ctx: torch.Tensor,
                Y_ctx: torch.Tensor,
                M_q: torch.Tensor,
                Y_q: torch.Tensor | None = None,
                return_losses: bool = False):
        single = (M_ctx.dim() == 1)
        if single:
            M_ctx = M_ctx.unsqueeze(0); Y_ctx = Y_ctx.unsqueeze(0)
            M_q = M_q.unsqueeze(0)
            if Y_q is not None:
                Y_q = Y_q.unsqueeze(0)

        z_pred = self.predict_embedding(M_ctx, Y_ctx, M_q)            # (S, Q, d)
        y_pred = self.decoder(z_pred)                                  # (S, Q)

        if not return_losses:
            return y_pred.squeeze(0) if single else y_pred

        assert Y_q is not None, "return_losses=True requires Y_q"
        z_target = self.encode_target(M_q, Y_q)                        # (S, Q, d)
        jepa_loss = ((z_pred - z_target.detach()) ** 2).mean()
        vic_loss = self.vicreg_loss(z_pred)
        mse_loss = ((y_pred - Y_q) ** 2).mean()
        if single:
            y_pred = y_pred.squeeze(0)
        return y_pred, jepa_loss, vic_loss, mse_loss, z_pred

    # -- numpy helpers (for the existing chain / probe / drift code) -------

    @torch.no_grad()
    def predict_np(self, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                   M_q: np.ndarray) -> np.ndarray:
        Mc = torch.from_numpy(M_ctx).float()
        Yc = torch.from_numpy(Y_ctx).float()
        Mq = torch.from_numpy(M_q).float()
        return self.forward(Mc, Yc, Mq).cpu().numpy()

    @torch.no_grad()
    def psi_np(self, M: np.ndarray, Y_ctx=None) -> np.ndarray:
        """Expose the Intention basis for linear-probe disclosure machinery.
        Y_ctx is accepted for API parity with the rate-aware Intention variant
        and ignored — psi_theta is a function of m only."""
        Mt = torch.from_numpy(M).float()
        return self.psi(Mt).cpu().numpy()

    @torch.no_grad()
    def A_inv_and_w(self, M_ctx: np.ndarray, Y_ctx: np.ndarray) -> tuple:
        """Closed-form ridge over psi_theta(M_ctx) — *not* used for prediction
        in IntentionJEPAFM (the prediction path is the trained transformer +
        decoder), but exposed so the disclosure probe and the drift detectors
        keep working unchanged.

        The probe's diagnostic value rests on w_implicit being a per-scenario
        sufficient statistic in psi-space; since psi_theta is now trained via
        JEPA pretext (not via a supervised solve), w_implicit measures
        "what the JEPA-shaped basis sees" rather than "what the prediction
        head computes". That is exactly the disentanglement the audit asked
        for.
        """
        Psi = self.psi_np(M_ctx)
        d = Psi.shape[1]
        A = Psi.T @ Psi + self.alpha * np.eye(d)
        A_inv = np.linalg.inv(A)
        w = A_inv @ Psi.T @ Y_ctx
        return A_inv, w, Psi


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/vince/ALETHIA/experiments/intention-vs-deepsets")
    from data import make_oracle, make_scenario

    rng = np.random.default_rng(0)
    o = make_oracle()
    c = rng.uniform(-0.5, 0.5, 4)
    s = make_scenario(o, c, rng)

    m = IntentionJEPAFM()
    print(f"IntentionJEPAFM n_params (trainable) = {m.n_params}")
    Mc = torch.from_numpy(s["M_ctx"]).unsqueeze(0)
    Yc = torch.from_numpy(s["Y_ctx"]).unsqueeze(0)
    Mq = torch.from_numpy(s["M_query"]).unsqueeze(0)
    Yq = torch.from_numpy(s["Y_query"]).unsqueeze(0)

    # Eval forward.
    yp = m(Mc, Yc, Mq)
    print(f"  eval y_pred shape = {tuple(yp.shape)}")

    # Train forward + losses.
    yp, jl, vl, ml, z = m(Mc, Yc, Mq, Yq, return_losses=True)
    print(f"  train z_pred shape = {tuple(z.shape)}")
    print(f"  untrained losses: jepa={jl.item():.4f}  vic={vl.item():.4f}  "
          f"mse={ml.item():.4f}")

    # EMA update.
    m.update_target()
    print("  EMA update OK.")

    # numpy helpers for the chain.
    psi = m.psi_np(s["M_ctx"])
    A_inv, w, Psi = m.A_inv_and_w(s["M_ctx"], s["Y_ctx"])
    print(f"  psi_np shape = {psi.shape}, A_inv shape = {A_inv.shape}, "
          f"w shape = {w.shape}")
    yp_np = m.predict_np(s["M_ctx"], s["Y_ctx"], s["M_query"])
    print(f"  predict_np shape = {yp_np.shape}")
