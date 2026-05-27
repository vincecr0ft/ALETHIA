"""DeepSets foundation model, parameter-matched to ``IntentionFMLearned``.

Same meta-data as the Intention FM: a context ``(M_ctx, Y_ctx)`` and a
query ``M_query``. The architectural difference is how the context is
ingested:

  - per-event encoder phi_event([m_i, y_i]) -> R^{d_set}
  - mean-pool over events -> z in R^{d_set}
  - decoder dec([z, m_query]) -> y_query

Wilson coefficients ``c`` never enter the forward pass. They affect the
labels (Y_ctx and Y_query) but the architecture is c-blind.

The dimensions are picked to match IntentionFMLearned's parameter count
(~5300) within a factor of two. Default settings produce ~5250 params.
"""
from __future__ import annotations

import torch
import torch.nn as nn

M_REF = 1.0


def _scaled_m(m: torch.Tensor) -> torch.Tensor:
    return torch.log(m / M_REF).unsqueeze(-1)


class DeepSetsFM(nn.Module):
    def __init__(self, d_set: int = 16, hidden: int = 48):
        super().__init__()
        # per-event encoder: takes scaled_m and y -> R^{d_set}
        self.phi_event = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_set),
        )
        # decoder: takes [z, scaled_m_query] -> y
        self.decoder = nn.Sequential(
            nn.Linear(d_set + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.d_set = d_set

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, M_ctx: torch.Tensor, Y_ctx: torch.Tensor,
                M_query: torch.Tensor) -> torch.Tensor:
        """Same API as IntentionFMLearned.

        Single-scenario tensors:
            M_ctx: (K,), Y_ctx: (K,), M_query: (Q,) -> (Q,)
        Batched:
            M_ctx: (S, K), Y_ctx: (S, K), M_query: (S, Q) -> (S, Q)
        """
        if M_ctx.dim() == 1:
            M_ctx = M_ctx.unsqueeze(0)
            Y_ctx = Y_ctx.unsqueeze(0)
            M_query = M_query.unsqueeze(0)
            single = True
        else:
            single = False
        S, K = M_ctx.shape
        Q = M_query.shape[1]
        # Per-event features.
        m_ctx_s = torch.log(M_ctx / M_REF)         # (S, K)
        evt_in = torch.stack([m_ctx_s, Y_ctx], dim=-1)   # (S, K, 2)
        h = self.phi_event(evt_in.reshape(S * K, 2)).reshape(S, K, -1)
        z = h.mean(dim=1)                          # (S, d_set)
        # Query: broadcast z across Q queries.
        m_q_s = torch.log(M_query / M_REF).unsqueeze(-1)   # (S, Q, 1)
        z_exp = z.unsqueeze(1).expand(-1, Q, -1)           # (S, Q, d_set)
        dec_in = torch.cat([z_exp, m_q_s], dim=-1)         # (S, Q, d_set+1)
        out = self.decoder(dec_in.reshape(S * Q, -1)).reshape(S, Q)
        if single:
            out = out.squeeze(0)
        return out


class IntentionFMRegressor(nn.Module):
    """Ceiling baseline: c IS on the forward pass.

    Direct (c, m) -> y MLP. Trained on (c, m, y) triples gathered from the
    training scenarios. This is the architecture the constraint forbids;
    it exists in this comparison only to mark "what you'd get if you
    abandoned the constraint."
    """

    def __init__(self, n_wc: int = 4, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_wc + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, c: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """c: (N, n_wc) M: (N,) -> (N,)."""
        m_s = torch.log(M / M_REF).unsqueeze(-1)
        x = torch.cat([c, m_s], dim=-1)
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    fm = DeepSetsFM()
    print("DeepSetsFM n_params      =", fm.n_params)
    ce = IntentionFMRegressor()
    print("IntentionFMRegressor     =", ce.n_params)
    import numpy as np, sys
    sys.path.insert(0, "/tmp/fm_compare")
    from data import make_oracle, make_scenario
    rng = np.random.default_rng(0)
    o = make_oracle()
    c = rng.uniform(-0.5, 0.5, 4)
    s = make_scenario(o, c, rng)
    with torch.no_grad():
        y = fm(torch.from_numpy(s["M_ctx"]),
               torch.from_numpy(s["Y_ctx"]),
               torch.from_numpy(s["M_query"]))
    print("forward OK shape =", tuple(y.shape))
