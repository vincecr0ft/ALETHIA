"""Extended architecture cast for the FM-landscape comparison.

The novelty memo (manifoldinformer_FM_landscape_and_novelty.md, A.1 item 1)
asks for a real head-to-head among architectures that *could plausibly*
disclose the SMEFT morphing manifold, evaluated by the same battery on
identical splits. The existing cast (experiment_smeft.py / representation_
analysis.py) covers the closed-form Intention head, a matched DeepSets
pooler, JEPA, and the c-consuming regressor. This module adds four
landscape entrants, each standing in for a family from Part B of the memo:

  SetTransformerFM      — attention-based set pooling (Lee et al., Set
                          Transformer; the canonical permutation-invariant
                          attention aggregator). Family: generic set FMs.
  ParticleTransformerFM — full self-attention over context events with a
                          CLS token and a pairwise |Δlog m| interaction
                          bias (Qu et al., arXiv:2202.03772). Family: B.13
                          HEP detector-level transformers.
  DeepONetFM            — branch/trunk operator head on the *binned* ratio
                          (Lu et al. DeepONet). Family: B.4 operator FMs;
                          also embodies the binned-input bottleneck of the
                          Das Bakshi competitor (binning, arXiv:2211.02058).
  TabPFNHeadFM          — meta-trained in-context transformer; queries
                          attend to (m,y) context tokens, prediction in one
                          forward pass with no per-scenario fit (Hollmann
                          et al., Nature 2025; PFN, arXiv:2112.10510).
                          Family: B.2, the closest mechanistic analogue.

Every model exposes the project-uniform meta-learning API

    forward(M_ctx, Y_ctx, M_query) -> Y_query

(single scenario: M_ctx (K,), Y_ctx (K,), M_query (Q,) -> (Q,);
 batched:        M_ctx (S,K), Y_ctx (S,K), M_query (S,Q) -> (S,Q))

so it drops straight into the train/eval loops in experiment_smeft.py, and a
`representation(M_ctx, Y_ctx) -> (S, D)` method so the same c-recoverability /
y-shuffle / effective-rank probes in representation_analysis.py apply
unchanged. Wilson coefficients c never enter any forward pass; they only
shape the labels.

Parameter counts are matched to the ~5k-param Intention/DeepSets budget
within a small factor (see __main__).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

M_REF = 1.0
# log-mass binning range for the DeepONet branch (M in [0.3, 2.3] TeV).
_LOGM_LO = math.log(0.3 / M_REF)
_LOGM_HI = math.log(2.3 / M_REF)


def _scaled_m(m: torch.Tensor) -> torch.Tensor:
    return torch.log(m / M_REF)


def _as_batched(M_ctx, Y_ctx, M_query):
    """Promote single-scenario tensors to a batch of one; report the flag."""
    if M_ctx.dim() == 1:
        return M_ctx[None], Y_ctx[None], M_query[None], True
    return M_ctx, Y_ctx, M_query, False


# --------------------------------------------------------------------------
# 1. Set Transformer  (attention-based set pooling)
# --------------------------------------------------------------------------
class _MAB(nn.Module):
    """Multihead attention block: X attends to Y (Lee et al., Eq. 6-7)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                nn.Linear(dim, dim))

    def forward(self, X, Y):
        h, _ = self.attn(X, Y, Y)
        h = self.ln0(X + h)
        return self.ln1(h + self.ff(h))


class SetTransformerFM(nn.Module):
    """Event encoder -> SAB self-attention -> PMA pooling -> decode."""

    def __init__(self, dim: int = 24, n_heads: int = 4, hidden: int = 32):
        super().__init__()
        self.enc = nn.Linear(2, dim)
        self.sab = _MAB(dim, n_heads)          # self-attention over events
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.pma = _MAB(dim, n_heads)          # pool by attention to a seed
        self.decoder = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        self.dim = dim

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _summary(self, M_ctx, Y_ctx):
        S, K = M_ctx.shape
        evt = torch.stack([_scaled_m(M_ctx), Y_ctx], dim=-1)   # (S, K, 2)
        h = self.sab(self.enc(evt), self.enc(evt))             # (S, K, dim)
        seed = self.seed.expand(S, -1, -1)
        z = self.pma(seed, h).squeeze(1)                       # (S, dim)
        return z

    def forward(self, M_ctx, Y_ctx, M_query):
        M_ctx, Y_ctx, M_query, single = _as_batched(M_ctx, Y_ctx, M_query)
        S, Q = M_query.shape
        z = self._summary(M_ctx, Y_ctx)
        mq = _scaled_m(M_query).unsqueeze(-1)                  # (S, Q, 1)
        dec_in = torch.cat([z.unsqueeze(1).expand(-1, Q, -1), mq], dim=-1)
        out = self.decoder(dec_in.reshape(S * Q, -1)).reshape(S, Q)
        return out.squeeze(0) if single else out

    @torch.no_grad()
    def representation(self, M_ctx, Y_ctx):
        M_ctx, Y_ctx, _, single = _as_batched(M_ctx, Y_ctx, M_ctx)
        z = self._summary(M_ctx, Y_ctx)
        return z.squeeze(0) if single else z


# --------------------------------------------------------------------------
# 2. Particle-Transformer-style attention (CLS token + pairwise bias)
# --------------------------------------------------------------------------
class ParticleTransformerFM(nn.Module):
    """Self-attention over event tokens with a learnable pairwise
    interaction bias U(|Δlog m|) added to the attention logits (the ParT
    hallmark), summarised through a CLS token."""

    def __init__(self, dim: int = 24, n_heads: int = 3, n_layers: int = 2,
                 hidden: int = 32):
        super().__init__()
        self.enc = nn.Linear(2, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.n_heads = n_heads
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln0": nn.LayerNorm(dim),
                "q": nn.Linear(dim, dim), "k": nn.Linear(dim, dim),
                "v": nn.Linear(dim, dim), "o": nn.Linear(dim, dim),
                "ln1": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, dim)),
            }) for _ in range(n_layers)])
        # pairwise interaction embedding U: scalar |Δlog m| -> per-head bias
        self.U = nn.Sequential(nn.Linear(1, 16), nn.GELU(),
                               nn.Linear(16, n_heads))
        self.decoder = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        self.dim = dim

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pair_bias(self, M_ctx):
        # |Δlog m| over the K events, zero-padded for the CLS slot.
        lm = _scaled_m(M_ctx)                                  # (S, K)
        d = (lm.unsqueeze(-1) - lm.unsqueeze(-2)).abs().unsqueeze(-1)  # (S,K,K,1)
        bias = self.U(d).permute(0, 3, 1, 2)                  # (S, H, K, K)
        S, H, K, _ = bias.shape
        out = bias.new_zeros(S, H, K + 1, K + 1)
        out[:, :, 1:, 1:] = bias                              # CLS row/col = 0
        return out

    def _summary(self, M_ctx, Y_ctx):
        S, K = M_ctx.shape
        evt = torch.stack([_scaled_m(M_ctx), Y_ctx], dim=-1)
        tok = torch.cat([self.cls.expand(S, -1, -1), self.enc(evt)], dim=1)
        bias = self._pair_bias(M_ctx)                         # (S, H, K+1, K+1)
        H, dh = self.n_heads, self.dim // self.n_heads
        scale = 1.0 / math.sqrt(dh)
        for L in self.layers:
            x = L["ln0"](tok)
            q = L["q"](x).reshape(S, -1, H, dh).transpose(1, 2)
            k = L["k"](x).reshape(S, -1, H, dh).transpose(1, 2)
            v = L["v"](x).reshape(S, -1, H, dh).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) * scale + bias
            att = att.softmax(dim=-1)
            h = (att @ v).transpose(1, 2).reshape(S, -1, self.dim)
            tok = tok + L["o"](h)
            tok = tok + L["ff"](L["ln1"](tok))
        return tok[:, 0]                                       # CLS summary

    def forward(self, M_ctx, Y_ctx, M_query):
        M_ctx, Y_ctx, M_query, single = _as_batched(M_ctx, Y_ctx, M_query)
        S, Q = M_query.shape
        z = self._summary(M_ctx, Y_ctx)
        mq = _scaled_m(M_query).unsqueeze(-1)
        dec_in = torch.cat([z.unsqueeze(1).expand(-1, Q, -1), mq], dim=-1)
        out = self.decoder(dec_in.reshape(S * Q, -1)).reshape(S, Q)
        return out.squeeze(0) if single else out

    @torch.no_grad()
    def representation(self, M_ctx, Y_ctx):
        M_ctx, Y_ctx, _, single = _as_batched(M_ctx, Y_ctx, M_ctx)
        z = self._summary(M_ctx, Y_ctx)
        return z.squeeze(0) if single else z


# --------------------------------------------------------------------------
# 3. DeepONet head on the binned ratio (branch / trunk operator)
# --------------------------------------------------------------------------
class DeepONetFM(nn.Module):
    """Branch encodes the binned context y-profile; trunk encodes the query
    log-mass; prediction is the branch-trunk inner product (Lu et al.).

    The binning is deliberate: it is the information bottleneck the memo
    flags (binned SMEFT input, arXiv:2211.02058) and the input form of the
    Das Bakshi competitor. This model therefore reads the morphing manifold
    only through a histogram, never event-level."""

    def __init__(self, n_bins: int = 16, p: int = 24, hidden: int = 48):
        super().__init__()
        self.n_bins = n_bins
        self.branch = nn.Sequential(
            nn.Linear(n_bins, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p))
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(1))
        self.p = p

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _bin_context(self, M_ctx, Y_ctx):
        """Mean y per fixed log-m bin; empty bins -> 0. -> (S, n_bins)."""
        S, K = M_ctx.shape
        lm = _scaled_m(M_ctx)
        edges = torch.linspace(_LOGM_LO, _LOGM_HI, self.n_bins + 1,
                               device=M_ctx.device, dtype=M_ctx.dtype)
        idx = torch.bucketize(lm, edges[1:-1])                # (S, K) in [0, n_bins-1]
        prof = M_ctx.new_zeros(S, self.n_bins)
        cnt = M_ctx.new_zeros(S, self.n_bins)
        prof.scatter_add_(1, idx, Y_ctx)
        cnt.scatter_add_(1, idx, torch.ones_like(Y_ctx))
        return prof / cnt.clamp_min(1.0)

    def forward(self, M_ctx, Y_ctx, M_query):
        M_ctx, Y_ctx, M_query, single = _as_batched(M_ctx, Y_ctx, M_query)
        S, Q = M_query.shape
        b = self.branch(self._bin_context(M_ctx, Y_ctx))      # (S, p)
        t = self.trunk(_scaled_m(M_query).reshape(S * Q, 1)).reshape(S, Q, self.p)
        out = torch.einsum("sp,sqp->sq", b, t) + self.bias
        return out.squeeze(0) if single else out

    @torch.no_grad()
    def representation(self, M_ctx, Y_ctx):
        M_ctx, Y_ctx, _, single = _as_batched(M_ctx, Y_ctx, M_ctx)
        b = self.branch(self._bin_context(M_ctx, Y_ctx))
        return b.squeeze(0) if single else b


# --------------------------------------------------------------------------
# 4. TabPFN-style in-context predictor (meta-trained, frozen at eval)
# --------------------------------------------------------------------------
class TabPFNHeadFM(nn.Module):
    """A transformer that performs in-context regression in one forward pass.

    Context tokens carry (scaled_m, y); query tokens carry (scaled_m, 0) plus
    a role flag. Queries attend to context (and to each other) through stacked
    encoder layers; the y for each query token is read off linearly. No
    per-scenario optimisation at eval time — the 'learning algorithm' is in
    the weights, exactly the TabPFN / PFN stance."""

    def __init__(self, dim: int = 32, n_heads: int = 4, n_layers: int = 3,
                 hidden: int = 48):
        super().__init__()
        # token = [scaled_m, y, is_query]
        self.embed = nn.Linear(3, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=hidden,
            activation="gelu", batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(dim, 1)
        self.dim = dim

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, M_ctx, Y_ctx, M_query):
        M_ctx, Y_ctx, M_query, single = _as_batched(M_ctx, Y_ctx, M_query)
        S, K = M_ctx.shape
        Q = M_query.shape[1]
        ctx = torch.stack(
            [_scaled_m(M_ctx), Y_ctx, torch.zeros_like(M_ctx)], dim=-1)
        qry = torch.stack(
            [_scaled_m(M_query), torch.zeros_like(M_query),
             torch.ones_like(M_query)], dim=-1)
        tok = self.embed(torch.cat([ctx, qry], dim=1))        # (S, K+Q, dim)
        h = self.tx(tok)
        out = self.readout(h[:, K:]).squeeze(-1)              # query slots
        return out.squeeze(0) if single else out

    @torch.no_grad()
    def representation(self, M_ctx, Y_ctx):
        """Mean of the encoded context tokens — the aggregate queries see."""
        M_ctx, Y_ctx, _, single = _as_batched(M_ctx, Y_ctx, M_ctx)
        S, K = M_ctx.shape
        ctx = torch.stack(
            [_scaled_m(M_ctx), Y_ctx, torch.zeros_like(M_ctx)], dim=-1)
        h = self.tx(self.embed(ctx))
        z = h.mean(dim=1)
        return z.squeeze(0) if single else z


# Registry consumed by experiment_extended_cast.py.
EXTENDED_CAST = {
    "SetTransformer_z":      lambda: SetTransformerFM(),
    "ParticleTransformer_z": lambda: ParticleTransformerFM(),
    "DeepONet_b":            lambda: DeepONetFM(),
    "TabPFN_z":              lambda: TabPFNHeadFM(),
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, Q, S = 12, 32, 4
    M_ctx = torch.from_numpy(rng.uniform(0.3, 2.3, (S, K)).astype(np.float32))
    Y_ctx = torch.from_numpy(rng.standard_normal((S, K)).astype(np.float32))
    M_q = torch.from_numpy(rng.uniform(0.3, 2.3, (S, Q)).astype(np.float32))
    for name, factory in EXTENDED_CAST.items():
        m = factory()
        y = m(M_ctx, Y_ctx, M_q)
        r = m.representation(M_ctx, Y_ctx)
        # single-scenario path
        y1 = m(M_ctx[0], Y_ctx[0], M_q[0])
        print(f"{name:24s} n_params={m.n_params:6d}  "
              f"y{tuple(y.shape)}  repr{tuple(r.shape)}  single{tuple(y1.shape)}")
