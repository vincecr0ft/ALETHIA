r"""Paper-grade quantum-kernel encoding comparison: multi-seed + resolvable subspace.

Extends cell_quantum_kernel.py from a single-seed point estimate to a comparison
with error bars and clean signal. Two moves, both addressing the single-cell
caveat that at 2-of-4 Fisher-resolvable directions the disclosure test has little
power to separate encodings:

  (1) Multi-seed via event bootstrap. The cached substrate is fixed, but each
      scenario is an event SET; resampling its events (with replacement) gives a
      genuine data-level perturbation of the mean density matrix rho_c at
      near-zero cost (no oracle regeneration). Every encoding is rebuilt and
      re-probed per bootstrap seed, including a fresh alignment-train for the
      trained sub-rows. Reports mean +/- std of the c-recovery margin.

  (2) Resolvable-subspace scoring. The morphing manifold only carries signal on
      the Fisher-resolved directions (eigenvalues ~[31, 10, 3e-4, 2e-4] -> 2 of
      4). Scoring disclosure on the full 4-vector dilutes the encoding difference
      with 2 noise directions. We compute the Fisher eigenbasis once (finite-diff
      of the analytic per-event log-likelihood-ratio on SM events, the same
      construction as manifold_informer_gates), rotate c into it, and ALSO probe
      the top-k resolvable coordinates, where the amplitude-alignment effect is
      cleanest.

The scientific claim under test: the amplitude-aligned spinor encoding
(arXiv:2602.21311) discloses the morphing manifold better than a generic IQP
encoding, robustly across seeds and most sharply on the resolvable subspace.

Usage: ../../.venv/bin/python qk_scan.py [--seeds N] [--qubits Q]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import substrate as sub
import probes
from cell_quantum_kernel import (
    IQPFeatureMap, SpinorFeatureMap, train_scales,
    rbf_rff_features, _median_gamma, _to_py,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"


# --------------------------------------------------------------------------
# Fisher eigenbasis (resolvable directions) from analytic scores.
# --------------------------------------------------------------------------

def fisher_basis(oracle, sm_events: np.ndarray, n_wc: int, *,
                 eps: float = 1e-2, n_pool: int = 4000, seed: int = 0):
    """Fisher information eigenbasis at c=0 from finite-diff per-event scores.

    score_i(x) = d/dc_i log w_c(x) |_{c=0}  (central difference);
    F_ij = E_x[score_i score_j] over SM events; eigh(F) gives the resolvable
    directions (large eigenvalues) — the only subspace carrying recoverable
    morphing signal.
    """
    from modules.surrogate.oracle_events import event_log_likelihood_ratio
    rng = np.random.default_rng(seed)
    X = sm_events.reshape(-1, 2)
    if X.shape[0] > n_pool:
        X = X[rng.choice(X.shape[0], n_pool, replace=False)]
    scores = []
    for i in range(n_wc):
        cp = np.zeros(n_wc); cp[i] = eps
        cm = np.zeros(n_wc); cm[i] = -eps
        sp = np.asarray(event_log_likelihood_ratio(oracle, cp, X), float)
        sm = np.asarray(event_log_likelihood_ratio(oracle, cm, X), float)
        scores.append((sp - sm) / (2 * eps))
    S = np.stack(scores, axis=1)                          # (Nx, n_wc)
    F = S.T @ S / S.shape[0]
    w, V = np.linalg.eigh(F)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


# --------------------------------------------------------------------------
# Encodings under test (each returns (Z_train, Z_test)).
# --------------------------------------------------------------------------

def _bootstrap_events(X: np.ndarray, rng) -> np.ndarray:
    """Resample events within each scenario with replacement. X: (S, N, 2)."""
    S, N, _ = X.shape
    idx = rng.integers(0, N, size=(S, N))
    return np.take_along_axis(X, idx[:, :, None], axis=1)


def encodings(q: int):
    """Factory dict: name -> fn(Xtr, Xte, Ctr, seed) -> (Ztr, Zte, info)."""
    iqp = IQPFeatureMap(n_qubits=q, depth=2)
    spin = SpinorFeatureMap(n_qubits=q)
    feat_dim = 2 * iqp.dim * iqp.dim

    def iqp_fixed(Xtr, Xte, Ctr, seed):
        s1 = np.ones(iqp.q)
        return iqp.scenario_features(Xtr, s1, 1.0), iqp.scenario_features(Xte, s1, 1.0), {}

    def iqp_trained(Xtr, Xte, Ctr, seed):
        sc, al = train_scales(iqp, Xtr, Ctr, maxiter=40, seed=seed)
        s1 = np.array(sc["s1"])
        return iqp.scenario_features(Xtr, s1, sc["s2"]), iqp.scenario_features(Xte, s1, sc["s2"]), {"align": al}

    def spinor_fixed(Xtr, Xte, Ctr, seed):
        s1 = np.ones(spin.q)
        return spin.scenario_features(Xtr, s1, 1.0), spin.scenario_features(Xte, s1, 1.0), {}

    def spinor_trained(Xtr, Xte, Ctr, seed):
        sc, al = train_scales(spin, Xtr, Ctr, maxiter=40, seed=seed)
        s1 = np.array(sc["s1"])
        return spin.scenario_features(Xtr, s1, sc["s2"]), spin.scenario_features(Xte, s1, sc["s2"]), {"align": al}

    def classical_rbf(Xtr, Xte, Ctr, seed):
        g = _median_gamma(Xtr, seed=seed)
        return rbf_rff_features(Xtr, feat_dim, g, seed=seed), rbf_rff_features(Xte, feat_dim, g, seed=seed), {"gamma": g}

    return {
        "iqp_fixed": iqp_fixed,
        "iqp_trained": iqp_trained,
        "spinor_fixed": spinor_fixed,
        "spinor_trained": spinor_trained,
        "classical_rbf": classical_rbf,
    }


# --------------------------------------------------------------------------
# Scan.
# --------------------------------------------------------------------------

def run(seeds: int = 8, q: int = 3) -> dict:
    t0 = time.time()
    data = sub.load_cache()
    Xtr0, Ctr = data["train_X1"], data["train_c"]
    Xte0, Cte = data["test_X1"], data["test_c"]
    n_wc = Ctr.shape[1]

    # Fisher basis -> resolvable subspace (computed once).
    oracle = sub.make_oracle()
    fw, fV = fisher_basis(oracle, data["sm_X1"], n_wc)
    resolv = int(np.sum(fw > 0.01 * fw.max()))
    resolv = max(1, min(resolv, n_wc))
    Rsub = fV[:, :resolv]                                  # resolvable eigenvectors
    Ctr_res = Ctr @ Rsub                                   # (S, resolv)
    Cte_res = Cte @ Rsub

    enc = encodings(q)
    names = list(enc.keys())
    full = {n: [] for n in names}
    res = {n: [] for n in names}
    per_target = {n: [] for n in names}

    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        Xtr = _bootstrap_events(Xtr0, rng) if s > 0 else Xtr0  # seed 0 = unperturbed
        Xte = _bootstrap_events(Xte0, rng) if s > 0 else Xte0
        raw_tr = probes.raw_event_summary(Xtr)
        raw_te = probes.raw_event_summary(Xte)
        for n in names:
            Ztr, Zte, _info = enc[n](Xtr, Xte, Ctr, s)
            pf = probes.probe_with_floor(Ztr, Ctr, Zte, Cte, raw_tr, raw_te, seed=s)
            pr = probes.probe_with_floor(Ztr, Ctr_res, Zte, Cte_res, raw_tr, raw_te, seed=s)
            full[n].append(pf.margin)
            res[n].append(pr.margin)
            per_target[n].append(pf.r2_per_target)
        print(f"  seed {s+1}/{seeds} done ({time.time()-t0:.0f}s)", flush=True)

    def stat(d):
        return {n: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "vals": [float(x) for x in v]} for n, v in d.items()}

    summary = {
        "config": {"seeds": seeds, "n_qubits": q, "feature_dim": 2 * (2 ** q) ** 2,
                   "n_train": int(Ctr.shape[0]), "n_test": int(Cte.shape[0]),
                   "n_events": int(Xtr0.shape[1])},
        "fisher": {"eigenvalues": [float(x) for x in fw],
                   "n_resolvable": resolv,
                   "note": "resolvable = #eigenvalues > 1% of max"},
        "margin_full": stat(full),
        "margin_resolvable": stat(res),
        "per_target_r2_mean": {n: np.mean(per_target[n], axis=0).tolist() for n in names},
        "wall_seconds": time.time() - t0,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "qk_scan.json").write_text(json.dumps(summary, indent=2, default=_to_py))
    _plot(summary, names)
    _print_table(summary, names)
    return summary


def _print_table(summary, names):
    print("\nFisher eigenvalues:", [f"{x:.3g}" for x in summary["fisher"]["eigenvalues"]],
          f"-> {summary['fisher']['n_resolvable']} resolvable")
    print(f"\n{'encoding':<16}{'margin (full)':>22}{'margin (resolvable)':>24}")
    print("-" * 62)
    for n in names:
        mf, sf = summary["margin_full"][n]["mean"], summary["margin_full"][n]["std"]
        mr, sr = summary["margin_resolvable"][n]["mean"], summary["margin_resolvable"][n]["std"]
        print(f"{n:<16}{mf:>+12.4f} ± {sf:<6.4f}{mr:>+14.4f} ± {sr:<6.4f}")


def _plot(summary, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"  (plot skipped: {e})")
        return
    x = np.arange(len(names))
    w = 0.38
    mf = [summary["margin_full"][n]["mean"] for n in names]
    sf = [summary["margin_full"][n]["std"] for n in names]
    mr = [summary["margin_resolvable"][n]["mean"] for n in names]
    sr = [summary["margin_resolvable"][n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w / 2, mf, w, yerr=sf, capsize=3, label="all 4 directions",
           color="#6e6e74")
    ax.bar(x + w / 2, mr, w, yerr=sr, capsize=3,
           label=f"resolvable subspace ({summary['fisher']['n_resolvable']} dir)",
           color="#26679e")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("c-recovery margin over random-feature floor")
    ax.set_title("Quantum-kernel encoding comparison "
                 f"({summary['config']['seeds']} bootstrap seeds, "
                 f"q={summary['config']['n_qubits']})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "qk_scan.png", dpi=140)
    print(f"  wrote {OUT / 'qk_scan.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--qubits", type=int, default=3)
    args = ap.parse_args()
    run(seeds=args.seeds, q=args.qubits)
