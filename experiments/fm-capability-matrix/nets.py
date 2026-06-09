r"""Shared neural building blocks for the capability-matrix cells.

Small, CPU-friendly modules over the 2-D event substrate
``x = (log m_ll / 1 TeV, cos θ*_CS)``. Cells import what they need; each cell
still owns its objective, loss, and metric. Kept deliberately tiny — the
substrate is 2-D so capacity is not the point; the *objective* is.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

D_EVENT = 2


def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class PerEventMLP(nn.Module):
    """phi: R^d_event -> R^d_emb applied per event (permutation-equivariant)."""

    def __init__(self, d_emb=32, hidden=64, d_in=D_EVENT):
        super().__init__()
        self.net = mlp([d_in, hidden, hidden, d_emb])

    def forward(self, x):  # (..., d_in) -> (..., d_emb)
        return self.net(x)


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant set encoder: per-event MLP -> mean+std pool -> MLP.

    Maps an event set (..., N, d_event) to a fixed scenario vector (..., d_out).
    The mean+std pooling is the standard DeepSets aggregator; std makes the
    summary sensitive to spread, which matters for SMEFT (the energy-growing
    tail is a variance effect, not just a mean shift).
    """

    def __init__(self, d_out=32, d_emb=48, hidden=64, d_in=D_EVENT):
        super().__init__()
        self.phi = mlp([d_in, hidden, d_emb])
        self.rho = mlp([2 * d_emb, hidden, d_out])
        self.d_out = d_out

    def forward(self, X):  # (..., N, d_in) -> (..., d_out)
        h = self.phi(X)
        pooled = torch.cat([h.mean(dim=-2), h.std(dim=-2)], dim=-1)
        return self.rho(pooled)

    @torch.no_grad()
    def encode_np(self, X_np: np.ndarray) -> np.ndarray:
        self.eval()
        return self.forward(torch.as_tensor(X_np, dtype=torch.float32)).cpu().numpy()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
