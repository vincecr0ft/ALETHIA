r"""Capability cell: Diffusion / flow generative  ->  conditional event generation.

FM family: "Diffusion / flow generative" (analogue of CaloChallenge generative
models in HEP). The objective this cell rewards is *high-fidelity continuous
generative modelling* of the per-c event distribution p(x|c) over the 2D
kinematics x = (log m_ll/1TeV, cos theta*_CS).

HEP task: Conditional generation of detector-level events x=(log m_ll, cos
theta*) given a Wilson point c, scored on distributional fidelity.

Model: a small conditional DDPM (denoising diffusion probabilistic model) with:
  - T=50 linear-beta schedule (betas in [1e-4, 0.02]).
  - eps-predictor MLP: [x_t(2) | c(4) | t_embed(8)] -> eps_hat(2).
  - Trained via standard MSE noise-prediction objective.
  - Ancestral DDPM reverse process for sampling.

Ablation isolated: continuous diffusion generation fidelity vs the real-vs-real
floor and a Gaussian baseline.
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

# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

def make_schedule(T: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear beta schedule; returns tensors for betas, alphas, alphas_bar."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


# ---------------------------------------------------------------------------
# Time embedding: sinusoidal features
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t_idx: torch.Tensor, d: int = 8, T: int = 50) -> torch.Tensor:
    """Map integer timestep indices (B,) -> (B, d) sinusoidal features."""
    t_frac = t_idx.float() / T  # (B,)
    # Half sine, half cosine at different frequencies
    half = d // 2
    freqs = torch.pow(10.0, torch.arange(half, dtype=torch.float32) / (half - 1 + 1e-9) * 4.0)
    arg = t_frac.unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (B, d)
    return emb


# ---------------------------------------------------------------------------
# Conditional eps-predictor network
# ---------------------------------------------------------------------------

class EpsPredictor(nn.Module):
    """MLP: [x_t(2) | c(4) | t_embed(8)] -> eps_hat(2).

    Kept deliberately small; the 2D problem does not need capacity.
    """

    def __init__(self, d_t_emb: int = 8, T: int = 50, hidden: int = 128):
        super().__init__()
        d_in = 2 + 4 + d_t_emb
        self.T = T
        self.d_t_emb = d_t_emb
        # Three hidden layers with GELU, no last activation
        self.net = mlp([d_in, hidden, hidden, hidden, 2])

    def forward(self, x_t: torch.Tensor, c: torch.Tensor,
                t_idx: torch.Tensor) -> torch.Tensor:
        """
        x_t : (B, 2)
        c   : (B, 4)
        t_idx: (B,) integer in [0, T-1]
        -> eps_hat (B, 2)
        """
        t_emb = sinusoidal_embedding(t_idx, self.d_t_emb, self.T)  # (B, d_t_emb)
        inp = torch.cat([x_t, c, t_emb], dim=-1)  # (B, 2+4+d_t_emb)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ddpm(
    model: EpsPredictor,
    train_X1: np.ndarray,   # (n_scen, n_ev, 2) float32
    train_c: np.ndarray,    # (n_scen, 4)  float32
    alphas_bar: torch.Tensor,
    *,
    steps: int = 3000,
    batch: int = 256,
    lr: float = 3e-4,
    seed: int = 0,
) -> EpsPredictor:
    """Standard DDPM training: MSE on predicted noise."""
    set_seed(seed)
    T = len(alphas_bar)
    n_scen, n_ev, _ = train_X1.shape
    # Flatten all (scenario, event) pairs for easy sampling
    # X_all: (n_scen*n_ev, 2),  C_all: (n_scen*n_ev, 4)
    X_all = torch.tensor(train_X1.reshape(-1, 2), dtype=torch.float32)
    C_all = torch.tensor(
        np.repeat(train_c, n_ev, axis=0), dtype=torch.float32
    )
    N = X_all.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    model.train()

    for step in range(steps):
        # Sample batch of events
        idx = rng.integers(0, N, size=batch)
        x0 = X_all[idx]      # (B, 2)
        c  = C_all[idx]      # (B, 4)

        # Sample random timesteps
        t_idx = torch.randint(0, T, (batch,))  # (B,)
        abar = alphas_bar[t_idx].unsqueeze(1)  # (B, 1)

        # Forward diffusion: x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * eps

        eps_hat = model(x_t, c, t_idx)
        loss = nn.functional.mse_loss(eps_hat, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{steps}  loss={loss.item():.4f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Sampling (ancestral DDPM reverse process)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ddpm(
    model: EpsPredictor,
    c_batch: torch.Tensor,   # (B, 4)
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    n_samples: int = 160,
) -> torch.Tensor:
    """Ancestral DDPM sampling: x_T ~ N(0,I), T reverse steps -> x_0.

    c_batch: (B, 4) — generates n_samples events for each c in the batch.
    Returns: (B, n_samples, 2)
    """
    B = c_batch.shape[0]
    T = len(betas)
    model.eval()

    # Expand c to cover all samples: (B*n_samples, 4)
    c_rep = c_batch.repeat_interleave(n_samples, dim=0)  # (B*n_samples, 4)

    x = torch.randn(B * n_samples, 2)

    for t in range(T - 1, -1, -1):
        t_idx = torch.full((B * n_samples,), t, dtype=torch.long)
        eps_hat = model(x, c_rep, t_idx)

        beta_t = betas[t]
        alpha_t = alphas[t]
        abar_t = alphas_bar[t]

        # DDPM reverse mean
        coef = beta_t / torch.sqrt(1.0 - abar_t)
        mean = (x - coef * eps_hat) / torch.sqrt(alpha_t)

        if t > 0:
            abar_prev = alphas_bar[t - 1]
            # Posterior variance (simple choice: beta_t * (1-abar_{t-1})/(1-abar_t))
            sigma2 = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(sigma2) * noise
        else:
            x = mean

    return x.view(B, n_samples, 2)  # (B, n_samples, 2)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(seed: int = 0) -> dict:
    t0 = time.time()

    set_seed(seed)
    data = sub.load_cache()
    cfg = sub.CONFIG

    train_X1 = data["train_X1"]   # (450, 160, 2)
    train_c  = data["train_c"]    # (450, 4)
    test_X1  = data["test_X1"]    # (120, 160, 2)
    test_c   = data["test_c"]     # (120, 4)

    # ------------------------------------------------------------------
    # Standardize events (per-feature mean/std from pooled train)
    # ------------------------------------------------------------------
    X_flat = train_X1.reshape(-1, 2)  # (450*160, 2)
    mu_x  = X_flat.mean(axis=0).astype(np.float32)
    std_x = X_flat.std(axis=0).astype(np.float32) + 1e-6

    train_X1_std = ((train_X1 - mu_x) / std_x).astype(np.float32)
    test_X1_std  = ((test_X1  - mu_x) / std_x).astype(np.float32)

    # ------------------------------------------------------------------
    # Build DDPM
    # ------------------------------------------------------------------
    T = 50
    betas, alphas, alphas_bar = make_schedule(T=T)

    model = EpsPredictor(d_t_emb=8, T=T, hidden=128)
    n_params = count_params(model)
    print(f"EpsPredictor n_params={n_params}", flush=True)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("Training DDPM ...", flush=True)
    train_ddpm(
        model, train_X1_std, train_c, alphas_bar,
        steps=3000, batch=256, lr=3e-4, seed=seed,
    )

    # ------------------------------------------------------------------
    # Evaluate: conditional fidelity over ~30 test scenarios
    # ------------------------------------------------------------------
    print("Evaluating conditional fidelity ...", flush=True)
    n_eval_scen = 30
    rng = np.random.default_rng(seed + 1)
    eval_idx = rng.choice(test_X1.shape[0], size=n_eval_scen, replace=False)

    c_eval = torch.tensor(test_c[eval_idx], dtype=torch.float32)  # (30, 4)

    # Generate n_samples = 160 events per scenario (same size as real)
    n_gen = cfg.n_events  # 160

    # Generate in batches of 10 to keep memory light
    batch_size = 10
    gen_batches = []
    for start in range(0, n_eval_scen, batch_size):
        end = min(start + batch_size, n_eval_scen)
        c_b = c_eval[start:end]
        gen_b = sample_ddpm(model, c_b, betas, alphas, alphas_bar,
                            n_samples=n_gen)  # (bs, 160, 2)
        gen_batches.append(gen_b.numpy())

    gen_std = np.concatenate(gen_batches, axis=0)   # (30, 160, 2) standardised
    gen_raw = gen_std * std_x + mu_x                 # (30, 160, 2) original scale

    # Per-scenario energy distance and W1
    ed_gen_list  = []
    w1_gen_list  = []
    ed_floor_list = []   # real-vs-real floor (two independent real subsets)

    for i, ei in enumerate(eval_idx):
        real_i = test_X1[ei]                    # (160, 2)
        gen_i  = gen_raw[i]                     # (160, 2)

        ed_gen_list.append(probes.energy_distance_2d(real_i, gen_i))
        w1_gen_list.append(probes.marginal_wasserstein(real_i, gen_i)["w1_mean"])

        # Real-vs-real floor: split the 160 real events into two halves
        half = len(real_i) // 2
        ed_floor_list.append(
            probes.energy_distance_2d(real_i[:half], real_i[half:])
        )

    mean_ed_gen    = float(np.mean(ed_gen_list))
    mean_ed_floor  = float(np.mean(ed_floor_list))
    mean_w1_gen    = float(np.mean(w1_gen_list))

    # ------------------------------------------------------------------
    # Gaussian baseline: N(0,I) in standardised space -> de-standardise
    # ------------------------------------------------------------------
    print("Computing Gaussian baseline ...", flush=True)
    ed_gauss_list = []
    rng_g = np.random.default_rng(seed + 2)
    for ei in eval_idx:
        real_i = test_X1[ei]
        gauss_std = rng_g.standard_normal((n_gen, 2)).astype(np.float32)
        gauss_raw = gauss_std * std_x + mu_x
        ed_gauss_list.append(probes.energy_distance_2d(real_i, gauss_raw))
    mean_ed_gauss = float(np.mean(ed_gauss_list))

    # ------------------------------------------------------------------
    # Pooled fidelity: generate for ALL test scenarios and pool
    # ------------------------------------------------------------------
    print("Computing pooled fidelity ...", flush=True)
    c_all_test = torch.tensor(test_c, dtype=torch.float32)  # (120, 4)
    pool_batches = []
    pool_bs = 10
    for start in range(0, test_c.shape[0], pool_bs):
        end = min(start + pool_bs, test_c.shape[0])
        c_b = c_all_test[start:end]
        gen_b = sample_ddpm(model, c_b, betas, alphas, alphas_bar,
                            n_samples=n_gen)  # (bs, 160, 2)
        pool_batches.append(gen_b.numpy())

    pool_gen_std = np.concatenate(pool_batches, axis=0)   # (120, 160, 2)
    pool_gen_raw = pool_gen_std * std_x + mu_x            # de-standardise

    pool_gen_flat  = pool_gen_raw.reshape(-1, 2)          # (120*160, 2)
    pool_real_flat = test_X1.reshape(-1, 2)               # (120*160, 2)
    pooled_ed = probes.energy_distance_2d(pool_real_flat, pool_gen_flat)

    wall = time.time() - t0
    print(f"\nDone. wall={wall:.1f}s", flush=True)
    print(f"  mean_ed_gen={mean_ed_gen:.4f}  floor={mean_ed_floor:.4f}"
          f"  gauss={mean_ed_gauss:.4f}  pooled_ed={pooled_ed:.4f}", flush=True)

    result = {
        "cell": "diffusion_generation",
        "fm_family": "Diffusion / flow generative",
        "hep_task": (
            "Conditional generation of detector-level events "
            "x=(log m_ll, cos theta*) given a Wilson point c, "
            "scored on distributional fidelity."
        ),
        "metric_primary": {
            "name": "conditional generation mean energy distance (lower=better)",
            "value": mean_ed_gen,
        },
        "metrics": {
            "mean_energy_distance_gen":      mean_ed_gen,
            "mean_energy_distance_realfloor": mean_ed_floor,
            "mean_energy_distance_gaussian": mean_ed_gauss,
            "mean_w1_gen":                   mean_w1_gen,
            "pooled_energy_distance_gen":    pooled_ed,
        },
        "ablation_isolated": (
            "continuous diffusion generation fidelity vs the real-vs-real "
            "floor and a Gaussian baseline"
        ),
        "n_params": n_params,
        "wall_seconds": wall,
        "config": {
            "T_steps":       T,
            "beta_start":    1e-4,
            "beta_end":      0.02,
            "train_steps":   3000,
            "batch":         256,
            "lr":            3e-4,
            "hidden":        128,
            "d_t_emb":       8,
            "n_eval_scen":   n_eval_scen,
            "n_gen_per_scen": n_gen,
            "n_train":       int(train_X1.shape[0]),
            "n_events":      cfg.n_events,
        },
    }

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "cell_diffusion.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written: {out_path}", flush=True)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(
        {k: r[k] for k in ("cell", "metric_primary", "metrics",
                           "n_params", "wall_seconds")},
        indent=2,
    ))
