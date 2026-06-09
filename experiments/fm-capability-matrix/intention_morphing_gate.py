r"""Direct A_i / B_ij recovery from the corrected Intention coordinate.

The value-ablation showed w_θ(V=log w_c) recovers c far better than the
self-embedding bug. This closes the loop the framing leans on: does a LINEAR
probe of the Intention coordinate recover the analytic morphing tangent A_i(m)
and curvature B_ij(m) themselves — the P3/P4 quantities — on the resolved
subspace?

Protocol (the v2 form, the landscape memo §4): fit ONE global linear decoder
D: w_θ -> μ(c, m)-profile on held-out train scenarios (the linear readout). With
D frozen, finite-difference w_θ in c with MATCHED sampling seeds (so the
difference is geometry, not sampling noise), map the c-derivatives through D, and
compare to the analytic templates as vectors over m via scale-free cosine:
  D · ∂w_θ/∂c_i        vs  A_i(m) = ∂μ/∂c_i        (tangent, P3)
  D · ∂²w_θ/∂c_i∂c_j   vs  B_ij(m) = ∂²μ/∂c_i∂c_j  (curvature, P4)
on the Fisher-resolved directions (clq3, clq1). Control: the V=self / mean-pool
coordinate, which should fail — the certificate is sensitive to value/readout.
Multiple model seeds -> mean ± std.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import substrate as sub
from intention_value_ablation import RidgeLogw, MeanPoolNoValue, train, N_CTX
from modules.surrogate.oracle_events import (
    sample_events, event_log_likelihood_ratio)
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

OUT = HERE / "output_matrix"
M_GRID = np.linspace(0.3, 2.3, 40)
RESOLVED = [2, 3]          # clq3, clq1 — the Fisher-resolvable four-fermion dirs
H = 0.15                    # finite-difference step in c (μ is exactly quadratic)
N_EV = 512                 # events per coordinate eval
SAMPLE_SEEDS = (101, 202, 303)   # matched-sampling seeds, averaged


# --- analytic morphing templates over m ---------------------------------


def analytic_A(oracle, i, d=1e-3):
    cp = np.zeros((len(M_GRID), 4)); cp[:, i] = +d
    cm = np.zeros((len(M_GRID), 4)); cm[:, i] = -d
    return (oracle.truth(cp, M_GRID) - oracle.truth(cm, M_GRID)) / (2 * d)


def analytic_B(oracle, i, j, d=1e-3):
    if i == j:
        cp = np.zeros((len(M_GRID), 4)); cp[:, i] = +d
        cm = np.zeros((len(M_GRID), 4)); cm[:, i] = -d
        c0 = np.zeros((len(M_GRID), 4))
        return (oracle.truth(cp, M_GRID) + oracle.truth(cm, M_GRID)
                - 2 * oracle.truth(c0, M_GRID)) / (d * d)
    cpp = np.zeros((len(M_GRID), 4)); cpp[:, i] = +d; cpp[:, j] = +d
    cpm = np.zeros((len(M_GRID), 4)); cpm[:, i] = +d; cpm[:, j] = -d
    cmp = np.zeros((len(M_GRID), 4)); cmp[:, i] = -d; cmp[:, j] = +d
    cmm = np.zeros((len(M_GRID), 4)); cmm[:, i] = -d; cmm[:, j] = -d
    return (oracle.truth(cpp, M_GRID) - oracle.truth(cpm, M_GRID)
            - oracle.truth(cmp, M_GRID) + oracle.truth(cmm, M_GRID)) / (4 * d * d)


# --- model coordinate at an arbitrary c, matched sampling ----------------


def coord_at(model, oracle, c, sample_seed):
    """w_θ(c) from N_EV events sampled with a fixed seed (matched across the ±
    perturbations) and their exact log w_c value, standardized with the SAME
    (mu, sd) used at training time."""
    ev = sample_events(oracle, c, N_EV, seed=sample_seed, m_grid=sub._cdf_grid())
    lw = event_log_likelihood_ratio(oracle, c, ev).astype(np.float32)
    lw = (lw - model._lw_mu) / model._lw_sd
    Xc = torch.tensor(ev[None], dtype=torch.float32)
    vc = torch.tensor(lw[None], dtype=torch.float32)
    return model.coordinate(Xc, vc)[0]      # (D,)


def fd_tangent(model, oracle, i):
    out = []
    for s in SAMPLE_SEEDS:
        cp = np.zeros(4); cp[i] = +H
        cm = np.zeros(4); cm[i] = -H
        out.append((coord_at(model, oracle, cp, s)
                    - coord_at(model, oracle, cm, s)) / (2 * H))
    return np.mean(out, axis=0)


def fd_curv(model, oracle, i, j):
    out = []
    for s in SAMPLE_SEEDS:
        if i == j:
            cp = np.zeros(4); cp[i] = +H
            cm = np.zeros(4); cm[i] = -H
            c0 = np.zeros(4)
            out.append((coord_at(model, oracle, cp, s) + coord_at(model, oracle, cm, s)
                        - 2 * coord_at(model, oracle, c0, s)) / (H * H))
        else:
            cpp = np.zeros(4); cpp[i] = +H; cpp[j] = +H
            cpm = np.zeros(4); cpm[i] = +H; cpm[j] = -H
            cmp = np.zeros(4); cmp[i] = -H; cmp[j] = +H
            cmm = np.zeros(4); cmm[i] = -H; cmm[j] = -H
            out.append((coord_at(model, oracle, cpp, s) - coord_at(model, oracle, cpm, s)
                        - coord_at(model, oracle, cmp, s) + coord_at(model, oracle, cmm, s))
                       / (4 * H * H))
    return np.mean(out, axis=0)


def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fit_decoder(Z, Y, alphas=(1e-2, 1e-1, 1.0, 10.0), val=0.25, seed=0):
    """Linear ridge Z->Y (μ-profile); alpha on inner val. Returns W,b,heldout-ready."""
    rng = np.random.default_rng(seed); n = len(Z); idx = rng.permutation(n)
    nv = int(val * n); vi, ti = idx[:nv], idx[nv:]
    best, bestW, bestb, besta = -1e9, None, None, alphas[0]
    Zc_mean, Yc_mean = Z[ti].mean(0), Y[ti].mean(0)
    for a in alphas:
        Zc = Z[ti] - Zc_mean; Yc = Y[ti] - Yc_mean
        W = np.linalg.solve(Zc.T @ Zc + a * np.eye(Z.shape[1]), Zc.T @ Yc)
        b = Yc_mean - Zc_mean @ W
        pred = Z[vi] @ W + b
        r2 = 1 - ((Y[vi]-pred)**2).sum() / ((Y[vi]-Y[vi].mean(0))**2).sum()
        if r2 > best:
            best, bestW, bestb, besta = r2, W, b, a
    return bestW, bestb, besta


def coords_for_split(model, X, L):
    Xc = torch.tensor(X[:, :N_CTX], dtype=torch.float32)
    vc = torch.tensor(L[:, :N_CTX], dtype=torch.float32)
    return model.coordinate(Xc, vc)


def run_model(name, factory, data, logw, oracle, A, B, seed):
    # train on cached events + logw
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Ltr = torch.tensor(logw["train_logw1"], dtype=torch.float32)
    mu, sd = Ltr.mean(), Ltr.std().clamp_min(1e-6)
    Ltr_n = (Ltr - mu) / sd
    m = factory()
    train(m, Xtr[:, :N_CTX], Ltr_n[:, :N_CTX], Xtr[:, N_CTX:], Ltr_n[:, N_CTX:], seed=seed)
    m._lw_mu = float(mu); m._lw_sd = float(sd)   # for consistent standardization in coord_at

    # held-out linear decoder D: w_θ -> μ-profile
    Ztr = coords_for_split(m, data["train_X1"], (np.asarray(Ltr_n)))
    Zte = coords_for_split(m, data["test_X1"],
                           (np.asarray((torch.tensor(logw["test_logw1"]) - mu) / sd)))
    W, b, alpha = fit_decoder(Ztr, data["train_mu"], seed=seed)
    pred_te = Zte @ W + b
    dec_r2 = float(1 - ((data["test_mu"]-pred_te)**2).sum()
                   / ((data["test_mu"]-data["test_mu"].mean(0))**2).sum())

    # finite-difference geometry, mapped through D, vs analytic templates
    tan_cos, curv_cos = {}, {}
    for i in RESOLVED:
        tan_cos[i] = cosine(fd_tangent(m, oracle, i) @ W, A[i])
    pairs = [(i, j) for k, i in enumerate(RESOLVED) for j in RESOLVED[k:]]
    for (i, j) in pairs:
        curv_cos[(i, j)] = cosine(fd_curv(m, oracle, i, j) @ W, B[(i, j)])
    return {"decoder_heldout_r2": dec_r2, "decoder_alpha": alpha,
            "tangent_cos": {str(i): tan_cos[i] for i in RESOLVED},
            "curvature_cos": {f"{i}{j}": curv_cos[(i, j)] for (i, j) in pairs}}


def main(seeds=(0, 1, 2)):
    t0 = time.time()
    data = sub.load_cache()
    logw = {k: np.load(OUT / "logw_cache.npz")[k]
            for k in np.load(OUT / "logw_cache.npz").files}
    oracle = sub.make_oracle()
    A = {i: analytic_A(oracle, i) for i in RESOLVED}
    B = {(i, j): analytic_B(oracle, i, j)
         for k, i in enumerate(RESOLVED) for j in RESOLVED[k:]}
    rng = np.random.default_rng(0)
    F = empirical_fisher_c(oracle, rng.uniform(-0.6, 0.6, (256, 4)),
                           rng.uniform(0.3, 2.3, 256), fd_step=1e-3)
    Dvals, _ = fisher_basis(F)
    print(f"Fisher eigvals {np.round(Dvals,4)}; resolved dirs (clq3,clq1)={RESOLVED}",
          flush=True)

    cast = {"Intention_ridge_logw": lambda: RidgeLogw(),
            "MeanPool_control": lambda: MeanPoolNoValue()}
    per_seed = {n: [] for n in cast}
    for s in seeds:
        for n, f in cast.items():
            print(f"seed {s} model {n} ...", flush=True)
            per_seed[n].append(run_model(n, f, data, logw, oracle, A, B, s))

    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {}
    for n in cast:
        rs = per_seed[n]
        tan = [np.mean([r["tangent_cos"][str(i)] for i in RESOLVED]) for r in rs]
        cur = [np.mean(list(r["curvature_cos"].values())) for r in rs]
        dec = [r["decoder_heldout_r2"] for r in rs]
        summary[n] = {"decoder_heldout_r2": agg(dec),
                      "tangent_cos_mean": agg(tan),
                      "curvature_cos_mean": agg(cur),
                      "per_seed": rs}
    out = {"seeds": list(seeds), "resolved_dirs": RESOLVED,
           "fisher_eigenvalues": [float(x) for x in Dvals],
           "summary": summary, "wall_seconds": time.time() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "morphing_gate.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== direct A_i/B_ij recovery via linear probe of w_θ (n_seeds={len(seeds)}) ===")
    for n in cast:
        s = summary[n]
        print(f"  {n:22s} decoder R²={s['decoder_heldout_r2']['mean']:.3f}"
              f"±{s['decoder_heldout_r2']['std']:.3f}  "
              f"tangent cos(A_i)={s['tangent_cos_mean']['mean']:.3f}"
              f"±{s['tangent_cos_mean']['std']:.3f}  "
              f"curvature cos(B_ij)={s['curvature_cos_mean']['mean']:.3f}"
              f"±{s['curvature_cos_mean']['std']:.3f}")
    return out


if __name__ == "__main__":
    main()
