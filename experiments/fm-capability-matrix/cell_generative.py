r"""Capability cell: Autoregressive generative  ->  generation + transfer.

FM family: Autoregressive generative (analogue of OmniJet-alpha / Evo;
next-token prediction over discretised event kinematics).

HEP task: Event generation by next-token prediction, then transfer the
generative backbone's representation to Wilson-coefficient recovery.

Objective tested: does GENERATIVE (next-token) pretraining yield a
TRANSFERABLE representation?

Architecture:
  1. Tokenise each event (log m, cos theta*) into (t_m, t_cos) in [0,B)
     using quantile bin edges from the pooled train events (B=32 bins/dim).
  2. Autoregressive model (c-agnostic, trained on ALL pooled train events):
       p(t_m)         — a free B-vector marginal (length-B logit param)
       p(t_cos | t_m) — an MLP: embed(t_m) -> B-dim softmax
     Loss: cross-entropy (next-token prediction; c never enters).
  3. Generative fidelity: sample N events, compare to pooled real test events
     via energy_distance_2d and marginal_wasserstein.
  4. Transfer: per-scenario representation = mean over events of
     [embed(t_m), last-hidden of conditional MLP].  Frozen backbone.
     Probe: probe_with_floor(Z_train, train_c, Z_test, test_c, raw_tr, raw_te).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import substrate as sub
import probes
from nets import mlp, count_params, set_seed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"

# --------------------------------------------------------------------------
# Hyper-parameters (tiny, CPU-friendly).
# --------------------------------------------------------------------------
B = 32           # quantile bins per feature
D_EMB = 24       # token embedding dim
D_HIDDEN = 48    # hidden layer in conditional MLP
STEPS = 800      # training steps (fast CPU run)
BS = 512         # batch size (events, not scenarios)
LR = 5e-3
N_GEN = 9600     # events to generate for fidelity check (~80 events per test scenario)


# --------------------------------------------------------------------------
# Tokeniser (quantile-based, fitted on pooled train events).
# --------------------------------------------------------------------------

class EventTokeniser:
    """Maps each event (log m, cos theta*) to (t_m, t_cos) in [0, B)."""

    def __init__(self, B: int = B):
        self.B = B
        self.edges_m: np.ndarray | None = None   # (B+1,)
        self.edges_cos: np.ndarray | None = None  # (B+1,)
        self.centers_m: np.ndarray | None = None  # (B,)
        self.centers_cos: np.ndarray | None = None  # (B,)
        self.bin_width_m: float = 0.0
        self.bin_width_cos: float = 0.0

    def fit(self, X: np.ndarray) -> "EventTokeniser":
        """X: (n_scenarios, n_events, 2) — uses all events pooled."""
        flat = X.reshape(-1, 2)
        qs = np.linspace(0.0, 1.0, self.B + 1)
        self.edges_m = np.quantile(flat[:, 0], qs)
        self.edges_cos = np.quantile(flat[:, 1], qs)
        # Force strict edges to avoid empty bins at boundaries.
        self.edges_m[0] -= 1e-6;  self.edges_m[-1] += 1e-6
        self.edges_cos[0] -= 1e-6; self.edges_cos[-1] += 1e-6
        self.centers_m = 0.5 * (self.edges_m[:-1] + self.edges_m[1:])
        self.centers_cos = 0.5 * (self.edges_cos[:-1] + self.edges_cos[1:])
        self.bin_width_m = float(np.diff(self.edges_m).mean())
        self.bin_width_cos = float(np.diff(self.edges_cos).mean())
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X (..., 2) -> T (..., 2) int64, values in [0, B)."""
        T = np.empty(X.shape[:-1] + (2,), dtype=np.int64)
        T[..., 0] = np.clip(np.searchsorted(self.edges_m[1:], X[..., 0]), 0, self.B - 1)
        T[..., 1] = np.clip(np.searchsorted(self.edges_cos[1:], X[..., 1]), 0, self.B - 1)
        return T

    def decode(self, T: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """T (..., 2) int64 -> X (..., 2) float, optionally jitter within bin."""
        X = np.empty(T.shape[:-1] + (2,), dtype=np.float64)
        X[..., 0] = self.centers_m[T[..., 0]]
        X[..., 1] = self.centers_cos[T[..., 1]]
        if rng is not None:
            X[..., 0] += rng.uniform(-self.bin_width_m / 2, self.bin_width_m / 2,
                                     X.shape[:-1])
            X[..., 1] += rng.uniform(-self.bin_width_cos / 2, self.bin_width_cos / 2,
                                     X.shape[:-1])
        return X


# --------------------------------------------------------------------------
# Autoregressive model.
# --------------------------------------------------------------------------

class ARGenerativeModel(nn.Module):
    """Next-token AR over (t_m, t_cos): p(t_m) * p(t_cos | t_m).

    p(t_m)         : a learnable B-logit vector (marginal, no conditioning).
    p(t_cos | t_m) : embed(t_m) -> MLP -> B-dim logit -> softmax.

    Backbone for transfer = per-event [embed(t_m), hidden(t_m), embed_cos(t_cos)],
    mean-pooled over the scenario's events. Including t_cos lets the mean-pooled
    repr capture the angular forward-backward asymmetry, which is the dominant
    c-discriminating observable.
    """

    def __init__(self, B: int = B, d_emb: int = D_EMB, d_hidden: int = D_HIDDEN):
        super().__init__()
        self.B = B
        self.d_emb = d_emb
        self.d_hidden = d_hidden
        # p(t_m): free B-vector of logits
        self.logits_m = nn.Parameter(torch.zeros(B))
        # Embedding for t_m token (used in AR conditional and backbone)
        self.emb_m = nn.Embedding(B, d_emb)
        # Conditional MLP: d_emb -> d_hidden -> B logits for p(t_cos | t_m)
        self.cond_hidden = nn.Sequential(nn.Linear(d_emb, d_hidden), nn.GELU())
        self.cond_out = nn.Linear(d_hidden, B)
        # Separate embedding for t_cos — only used in backbone repr (not in AR loss)
        self.emb_cos = nn.Embedding(B, d_emb)
        # Representation width = d_emb (t_m embed) + d_hidden (cond hidden) + d_emb (t_cos embed)
        self.d_repr = d_emb + d_hidden + d_emb

    def forward(self, t_m: torch.Tensor, t_cos: torch.Tensor):
        """Cross-entropy AR loss.  t_m, t_cos: (batch,) int64."""
        # Loss for t_m: NLL of the learned marginal
        loss_m = nn.functional.cross_entropy(
            self.logits_m.unsqueeze(0).expand(len(t_m), -1), t_m
        )
        # Conditional: embed t_m -> hidden -> logits_cos
        h = self.emb_m(t_m)           # (batch, d_emb)
        hidden = self.cond_hidden(h)   # (batch, d_hidden)
        logits_cos = self.cond_out(hidden)  # (batch, B)
        loss_cos = nn.functional.cross_entropy(logits_cos, t_cos)
        return loss_m + loss_cos

    @torch.no_grad()
    def backbone_repr(self, t_m: torch.Tensor, t_cos: torch.Tensor) -> torch.Tensor:
        """Per-event backbone repr: [embed(t_m), cond_hidden(t_m), embed_cos(t_cos)].

        t_m, t_cos: (n,) int64 -> (n, d_repr)
        """
        h_m = self.emb_m(t_m)           # (n, d_emb)
        hidden = self.cond_hidden(h_m)   # (n, d_hidden)
        h_cos = self.emb_cos(t_cos)      # (n, d_emb)
        return torch.cat([h_m, hidden, h_cos], dim=-1)  # (n, d_repr)

    @torch.no_grad()
    def sample(self, n: int, rng=None) -> np.ndarray:
        """Sample n token pairs (t_m, t_cos) -> (n, 2) int64.

        Fully vectorised: multinomial draw for t_m and t_cos in one pass.
        """
        # Sample t_m from the learned marginal (vectorised).
        p_m = torch.softmax(self.logits_m, dim=0)               # (B,)
        t_m_samp = torch.multinomial(p_m, num_samples=n, replacement=True)  # (n,)

        # Batch-compute conditional logits for all sampled t_m.
        chunk = 4096
        t_cos_parts = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            tm_chunk = t_m_samp[start:end]
            h = self.emb_m(tm_chunk)
            hidden = self.cond_hidden(h)
            logits_cos = self.cond_out(hidden)                   # (chunk, B)
            probs_cos = torch.softmax(logits_cos, dim=-1)        # (chunk, B)
            t_cos_chunk = torch.multinomial(probs_cos, num_samples=1).squeeze(1)
            t_cos_parts.append(t_cos_chunk)
        t_cos_samp = torch.cat(t_cos_parts, dim=0)               # (n,)

        return np.stack([t_m_samp.cpu().numpy(),
                         t_cos_samp.cpu().numpy()], axis=1).astype(np.int64)


# --------------------------------------------------------------------------
# Training.
# --------------------------------------------------------------------------

def train_ar(model: ARGenerativeModel,
             tokens: np.ndarray,  # (N_events_total, 2) int64
             *,
             steps: int = STEPS,
             bs: int = BS,
             lr: float = LR,
             seed: int = 0) -> ARGenerativeModel:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(tokens)
    rng = np.random.default_rng(seed)
    t_m_all = torch.tensor(tokens[:, 0], dtype=torch.long)
    t_cos_all = torch.tensor(tokens[:, 1], dtype=torch.long)

    model.train()
    for step in range(steps):
        idx = rng.choice(n, bs, replace=False)
        loss = model(t_m_all[idx], t_cos_all[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# --------------------------------------------------------------------------
# Scenario-level backbone representation (frozen, mean-pooled over events).
# --------------------------------------------------------------------------

@torch.no_grad()
def scenario_repr(model: ARGenerativeModel,
                  tok: EventTokeniser,
                  X: np.ndarray,  # (n_scenarios, n_events, 2)
                  ) -> np.ndarray:
    """Mean-pool the per-event backbone repr over events.

    Returns (n_scenarios, d_repr).
    """
    model.eval()
    n_scen, n_ev, _ = X.shape
    T = tok.encode(X)                                      # (n_scen, n_ev, 2)
    t_m_flat = torch.tensor(T[:, :, 0].ravel(), dtype=torch.long)    # (n_scen*n_ev,)
    t_cos_flat = torch.tensor(T[:, :, 1].ravel(), dtype=torch.long)  # (n_scen*n_ev,)
    r_flat = model.backbone_repr(t_m_flat, t_cos_flat)    # (n_scen*n_ev, d_repr)
    r = r_flat.view(n_scen, n_ev, -1)                     # (n_scen, n_ev, d_repr)
    return r.mean(dim=1).cpu().numpy()                     # (n_scen, d_repr)


# --------------------------------------------------------------------------
# Main run function.
# --------------------------------------------------------------------------

def run(seed: int = 0) -> dict:
    t0 = time.time()
    data = sub.load_cache()
    cfg = sub.CONFIG
    set_seed(seed)
    rng_np = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1: Fit tokeniser on pooled train events.
    # ------------------------------------------------------------------
    train_X1 = data["train_X1"]   # (450, 160, 2)
    test_X1 = data["test_X1"]     # (120, 160, 2)
    train_c = data["train_c"]     # (450, 4)
    test_c = data["test_c"]       # (120, 4)

    tok = EventTokeniser(B=B)
    tok.fit(train_X1)

    # ------------------------------------------------------------------
    # Step 2: Tokenise ALL pooled train events for AR training.
    # ------------------------------------------------------------------
    # Flatten (450 * 160, 2) events
    train_flat = train_X1.reshape(-1, 2)                  # (72000, 2)
    tokens_train = tok.encode(train_flat)                  # (72000, 2) int64

    # ------------------------------------------------------------------
    # Step 3: Train the AR model.
    # ------------------------------------------------------------------
    model = ARGenerativeModel(B=B, d_emb=D_EMB, d_hidden=D_HIDDEN)
    print(f"  AR model params: {count_params(model)}", flush=True)
    model = train_ar(model, tokens_train, steps=STEPS, bs=BS, lr=LR, seed=seed)

    # ------------------------------------------------------------------
    # Step 4: Generative fidelity.
    #   Generate N_GEN events; compare to pooled real test events (c-agnostic).
    # ------------------------------------------------------------------
    model.eval()
    gen_tokens = model.sample(N_GEN, rng=None)            # (N_GEN, 2)
    gen_events = tok.decode(gen_tokens, rng=rng_np)       # (N_GEN, 2) float

    # Pooled real test events (c-agnostic cloud)
    real_test_flat = test_X1.reshape(-1, 2).astype(float)  # (19200, 2)

    ed = probes.energy_distance_2d(real_test_flat, gen_events, seed=seed)
    wass = probes.marginal_wasserstein(real_test_flat, gen_events)

    # ------------------------------------------------------------------
    # Step 5: Transfer — frozen backbone -> per-scenario repr -> probe.
    # ------------------------------------------------------------------
    Z_train = scenario_repr(model, tok, train_X1)         # (450, d_repr)
    Z_test = scenario_repr(model, tok, test_X1)           # (120, d_repr)

    raw_tr = probes.raw_event_summary(train_X1)           # (450, d_raw)
    raw_te = probes.raw_event_summary(test_X1)            # (120, d_raw)

    probe = probes.probe_with_floor(
        Z_train, train_c, Z_test, test_c, raw_tr, raw_te, seed=seed
    )

    wall = time.time() - t0

    result = {
        "cell": "ar_generative",
        "fm_family": "Autoregressive generative",
        "hep_task": ("Event generation by next-token prediction (native), with "
                     "representation transfer to Wilson-coefficient recovery as "
                     "the ablation"),
        "metric_primary": {
            "name": "generation fidelity: energy distance gen vs real (lower=better)",
            "value": float(ed),
        },
        "metrics": {
            "gen_energy_distance": float(ed),
            "gen_w1_mean": float(wass["w1_mean"]),
            "gen_w1_per_feature": [float(x) for x in wass["w1_per_feature"]],
            "transfer_probe_r2": float(probe.r2),
            "transfer_floor_r2": float(probe.floor_r2),
            "transfer_margin": float(probe.margin),
            "transfer_r2_per_target": [float(x) for x in probe.r2_per_target],
        },
        "ablation_isolated": (
            "does generative next-token pretraining yield a transferable "
            "representation? (transfer margin over random-feature floor)"
        ),
        "n_params": count_params(model),
        "wall_seconds": float(wall),
        "config": {
            "B": B,
            "d_emb": D_EMB,
            "d_hidden": D_HIDDEN,
            "d_repr": model.d_repr,
            "steps": STEPS,
            "n_train_events": int(tokens_train.shape[0]),
            "n_gen": N_GEN,
            "n_train_scenarios": int(train_X1.shape[0]),
            "n_test_scenarios": int(test_X1.shape[0]),
        },
    }

    OUT.mkdir(exist_ok=True)
    with open(OUT / "cell_generative.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(
        {k: r[k] for k in ("cell", "metric_primary", "metrics", "n_params", "wall_seconds")},
        indent=2,
    ))
