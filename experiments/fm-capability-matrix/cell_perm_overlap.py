r"""G3 cell: a NON-LOCAL overlap observable whose exact ceiling we test against the
random-feature floor — the make-or-break gate of the multi-copy program.

==============================================================================
WHY THIS CELL EXISTS  (read this before the code)
==============================================================================

G1 (cell_swap_overlap.py) established that a two-copy SWAP estimator of the overlap
kernel K_ss' = Tr(rho_s rho_s') reaches a given disclosure with far fewer copies
than the optimal single-copy (classical-shadow) estimator, and that the separation
is alignment-gated (spinor yes, IQP no). But the M=infinity CEILING of that kernel
sits BELOW the random-feature floor (~ -0.18 on the resolvable subspace). So G1 is a
sample-complexity advantage in reaching a sub-floor target — not yet an advantageous
disclosure. The whole program is pointless unless some overlap observable has an
exact ceiling that CLEARS the floor. That is G3, and this cell is it.

The diagnosis. The fidelity-mean kernel is

    K_ss' = Tr(rho_s rho_s') = (1/N^2) sum_{n,n'} |<psi(x_n^s)|psi(x_{n'}^{s'})>|^2

i.e. the ARITHMETIC mean of squared single-event overlaps. It is a LOCAL, degree-2
functional of the single-event marginal rho_s, and it is exactly the class the
classical-ML-on-quantum-data dequantization theorems cover (Huang-Chen-Preskill,
PRX Quantum 4, 040337, 2023, arXiv:2210.14894: average-case, LOCAL observables).
Mean-pooling over events destroys inter-event coherence, and a local degree-2
summary is provably classically learnable. No surprise it does not beat a classical
random-feature floor built from the same events.

The G3 observable. Replace the arithmetic mean with the BOSONIC / SYMMETRIC k-event
overlap: encode k events as a single symmetrised (bosonic) state |Sym_k(.)>, and use
the squared overlap of two such states as the kernel entry. For normalised
single-event states the bosonic squared overlap is

    |<Sym_k(A)|Sym_k(B)>|^2 = |per(M_AB)|^2 / ( per(G_AA) per(G_BB) ),

where M_AB[i,j] = <a_i|b_j> is the k x k CROSS-event overlap matrix and G_** are the
within-set Gram matrices. This object is:
  (a) genuinely NON-LOCAL: a global observable on the k-event register, outside the
      scope of the local-observable dequantization theorems;
  (b) #P-HARD classically for growing k: the permanent is the BosonSampling quantity
      (Aaronson-Arkhipov, arXiv:1011.3245), so there is an independent classical-
      hardness anchor, not just "we hope it is hard";
  (c) a strict generalisation of G1: at k=1 it reduces EXACTLY to the arithmetic
      mean |<a|b>|^2 = Tr(rho_s rho_s'), so k is a LOCALITY DIAL from the sub-floor
      kernel (k=1) up to a maximally non-local one (large k).

The make-or-break question this cell asks:

    As the locality dial k grows, does the EXACT (infinite-copy) c-recovery ceiling
    of the bosonic-overlap kernel RISE, and does it clear the random-feature floor?

If yes: there exists a non-local overlap observable that discloses the SMEFT
manifold better than the classical floor, on the only axis (non-local / #P-hard)
that the dequantization theorems leave open. That is the advantageous-disclosure
result the program needs, and it earns the multi-copy estimation question (does a
collective two-copy measurement reach this ceiling sample-efficiently — G3b) and the
hardware step (G5).

If no (ceiling stays flat / below floor as k grows): the bosonic observable buys no
disclosure either, and the SMEFT kernel is dequantizable even at the non-local
level. That is a clean, strong conditional null on a #P-hard observable — a much
sharper statement than the G1 sub-floor caveat.

==============================================================================
SIMULATION STATUS
==============================================================================

Exact statevector, pure numpy. The permanent is computed by a vectorised Ryser
formula (O(2^k k) per matrix). This cell tests the INFORMATION question (the exact
ceiling); the sample-complexity / multi-copy-estimation question for this observable
is a separate follow-up (G3b), the analogue of what cell_swap_overlap did for the
k=1 kernel. No shots, no hardware here.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import substrate as sub
import probes
from cell_quantum_kernel import SpinorFeatureMap, IQPFeatureMap, _to_py
from qk_scan import fisher_basis

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"


# --------------------------------------------------------------------------
# Vectorised Ryser permanent for a batch of k x k matrices.
# --------------------------------------------------------------------------

def _ryser_masks(k: int):
    """Precompute, for every nonempty column subset S of [k], the membership row
    (1.0 where j in S) and the Ryser sign (-1)^(k-|S|). Returns (masks, signs)
    with masks shape (2^k - 1, k)."""
    masks = []
    signs = []
    for r in range(1, k + 1):
        for cols in combinations(range(k), r):
            m = np.zeros(k)
            m[list(cols)] = 1.0
            masks.append(m)
            signs.append((-1.0) ** (k - r))
    return np.asarray(masks), np.asarray(signs)


def batch_permanent(A: np.ndarray, masks: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Permanent of each k x k matrix in A (shape (B, k, k)) by Ryser's formula.

    per(A) = (-1)^k sum_{S subseteq [k]} (-1)^|S| prod_i ( sum_{j in S} A[i,j] ).
    Implemented as: for each nonempty column subset, sum the selected columns
    (B, k), take the product over rows (B,), weight by the Ryser sign, accumulate.
    Vectorised over the batch and over all 2^k-1 subsets.
    """
    # colsums[b, s, i] = sum_{j in S_s} A[b, i, j]
    colsums = np.einsum("bij,sj->bsi", A, masks.astype(A.dtype))   # (B, 2^k-1, k)
    prods = colsums.prod(axis=2)                                   # (B, 2^k-1)
    out = (prods * signs[None, :]).sum(axis=1)                     # (B,)
    return out


# --------------------------------------------------------------------------
# Bosonic / symmetric k-event overlap kernel (the G3 observable).
# --------------------------------------------------------------------------

def sample_subsets(n_scen: int, n_evt: int, k: int, R: int,
                   rng: np.random.Generator) -> np.ndarray:
    """For every scenario, R subsets of k distinct event indices. (n_scen, R, k)."""
    out = np.empty((n_scen, R, k), dtype=np.int64)
    for s in range(n_scen):
        for r in range(R):
            out[s, r] = rng.choice(n_evt, size=k, replace=False)
    return out


def scenario_subset_states(fmap, sets: np.ndarray, subsets: np.ndarray,
                           s1: np.ndarray, s2: float) -> np.ndarray:
    """Per-scenario, per-replicate bosonic-input states. Returns U of shape
    (S, R, k, dim): U[s, r] are the k single-event statevectors of subset r."""
    S, R, k = subsets.shape
    D = fmap.dim
    U = np.empty((S, R, k, D), dtype=np.complex128)
    for s in range(S):
        psi = fmap.states(sets[s], s1, s2)             # (N, dim)
        U[s] = psi[subsets[s]]                         # (R, k, dim)
    return U


def within_set_norms(U: np.ndarray, masks, signs, floor: float = 1e-9) -> np.ndarray:
    """per(G_ss) for each (scenario, replicate). U: (S, R, k, dim) -> (S, R) real."""
    S, R, k, D = U.shape
    flat = U.reshape(S * R, k, D)
    G = np.einsum("bik,bjk->bij", flat, flat.conj())   # (S*R, k, k) within-set Gram
    per = batch_permanent(G, masks, signs).real        # real, >= 0 for a Gram
    return np.clip(per.reshape(S, R), floor, None)


def boson_overlap_gram(Ua: np.ndarray, Ub: np.ndarray,
                       norm_a: np.ndarray, norm_b: np.ndarray,
                       masks, signs, chunk: int = 64) -> np.ndarray:
    """Bosonic-overlap Gram K[i,j] = mean_r |per(M_ij^r)|^2 / (norm_a[i,r] norm_b[j,r]),
    with M_ij^r[p,q] = <Ua[i,r,p] | Ub[j,r,q]>. Averaged over the R paired replicates.

    Ua: (Sa, R, k, dim), Ub: (Sb, R, k, dim). Returns (Sa, Sb) real kernel in [0,1].
    Row-chunked over Sa to bound the (chunk*Sb, k, k) batch memory.
    """
    Sa, R, k, D = Ua.shape
    Sb = Ub.shape[0]
    K = np.zeros((Sa, Sb))
    for r in range(R):
        Ar = Ua[:, r]                                  # (Sa, k, dim)
        Br = Ub[:, r]                                  # (Sb, k, dim)
        na = norm_a[:, r]                              # (Sa,)
        nb = norm_b[:, r]                              # (Sb,)
        for i0 in range(0, Sa, chunk):
            i1 = min(i0 + chunk, Sa)
            # cross-overlap matrices M[i,j,p,q] = <Ar[i,p] | Br[j,q]>
            M = np.einsum("ipd,jqd->ijpq", Ar[i0:i1], Br.conj())   # (c, Sb, k, k)
            c = i1 - i0
            per = batch_permanent(M.reshape(c * Sb, k, k), masks, signs)
            fid = (np.abs(per) ** 2).reshape(c, Sb)
            K[i0:i1] += fid / (na[i0:i1, None] * nb[None, :])
    return K / R


# --------------------------------------------------------------------------
# Main: locality-dial (k) sweep of the exact ceiling vs the floor.
# --------------------------------------------------------------------------

def run(q: int = 3, k_list=(1, 2, 3, 4, 6), R: int = 16, seeds: int = 4,
        encoding: str = "spinor", budget: int | None = None) -> dict:
    """If ``budget`` is set, the number of subsets per k is R_k = max(2, budget//k)
    so the EVENT BUDGET R_k * k is held ~constant across k. This disentangles the
    locality effect (genuine non-locality of the observable) from the trivial
    "larger k consumes more events" effect — the control that makes a rising
    ceiling-vs-k curve mean something."""
    t0 = time.time()
    data = sub.load_cache()
    Xtr, Ctr = data["train_X1"], data["train_c"]
    Xte, Cte = data["test_X1"], data["test_c"]
    S_tr, N_evt, _ = Xtr.shape
    S_te = Xte.shape[0]
    n_wc = Ctr.shape[1]
    raw_tr = probes.raw_event_summary(Xtr)
    raw_te = probes.raw_event_summary(Xte)

    # Fisher-resolvable subspace (the directions that carry morphing signal).
    oracle = sub.make_oracle()
    fw, fV = fisher_basis(oracle, data["sm_X1"], n_wc)
    resolv = max(1, min(int(np.sum(fw > 0.01 * fw.max())), n_wc))
    Rsub = fV[:, :resolv]
    Ctr_res, Cte_res = Ctr @ Rsub, Cte @ Rsub

    fmap = (SpinorFeatureMap(n_qubits=q) if encoding == "spinor"
            else IQPFeatureMap(n_qubits=q, depth=2))
    s1 = np.ones(fmap.q)

    def score(Ktt, Ket):
        full = probes.probe_with_floor(Ktt, Ctr, Ket, Cte, raw_tr, raw_te,
                                        seed=0).margin
        res = probes.probe_with_floor(Ktt, Ctr_res, Ket, Cte_res, raw_tr, raw_te,
                                      seed=0).margin
        return full, res

    sweep = {"k": list(k_list), "R_k": [], "full": [], "full_std": [], "res": [],
             "res_std": [], "floor_r2": None, "budget": budget}
    for k in k_list:
        masks, signs = _ryser_masks(k)
        R_k = max(2, budget // k) if budget else R
        sweep["R_k"].append(R_k)
        f_seeds, r_seeds = [], []
        floor_val = None
        for sd in range(seeds):
            rng = np.random.default_rng(4200 + sd)
            sub_tr = sample_subsets(S_tr, N_evt, k, R_k, rng)
            sub_te = sample_subsets(S_te, N_evt, k, R_k, rng)
            Utr = scenario_subset_states(fmap, Xtr, sub_tr, s1, 1.0)
            Ute = scenario_subset_states(fmap, Xte, sub_te, s1, 1.0)
            ntr = within_set_norms(Utr, masks, signs)
            nte = within_set_norms(Ute, masks, signs)
            Ktt = boson_overlap_gram(Utr, Utr, ntr, ntr, masks, signs)
            Ket = boson_overlap_gram(Ute, Utr, nte, ntr, masks, signs)
            f, r = score(Ktt, Ket)
            f_seeds.append(f); r_seeds.append(r)
            # floor is identical across k (raw events, fixed width = Gram width S_tr)
            floor_val = probes.probe_with_floor(
                Ktt, Ctr_res, Ket, Cte_res, raw_tr, raw_te, seed=0).floor_r2
        sweep["full"].append(float(np.mean(f_seeds)))
        sweep["full_std"].append(float(np.std(f_seeds)))
        sweep["res"].append(float(np.mean(r_seeds)))
        sweep["res_std"].append(float(np.std(r_seeds)))
        sweep["floor_r2"] = float(floor_val)
        print(f"  [k-sweep q={q} {encoding}] k={k} R_k={R_k} (budget={R_k*k} evt)"
              f"  res_margin={sweep['res'][-1]:+.4f} +/- {sweep['res_std'][-1]:.4f}"
              f"  full={sweep['full'][-1]:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    best_res = max(sweep["res"]) if sweep["res"] else float("nan")
    clears = best_res > 0.0
    k1_res = sweep["res"][0] if sweep["res"] else float("nan")
    rises = best_res > k1_res + 1e-3
    verdict = {
        "best_res_margin": float(best_res),
        "k1_res_margin": float(k1_res),
        "ceiling_rises_with_k": bool(rises),
        "ceiling_clears_floor": bool(clears),
        "decision": (
            "PROCEED to G3b/G5 (non-local ceiling clears the floor)" if clears else
            ("PARTIAL: ceiling rises with locality but does not yet clear the floor"
             if rises else
             "STOP / strong conditional null (non-local observable buys no disclosure)")),
        "note": ("k is the locality dial: k=1 is exactly the arithmetic-mean kernel "
                 "Tr(rho_s rho_s') (G1's sub-floor target), large k is the bosonic / "
                 "#P-hard permanent overlap. A positive res_margin at any k>1 is the "
                 "advantageous-disclosure result the program needs."),
    }

    result = {
        "cell": "perm_overlap",
        "fm_family": "Bosonic (symmetric k-event) permanent-overlap kernel",
        "hep_task": ("SMEFT manifold disclosure via a non-local overlap observable: "
                     "Wilson-coefficient recovery from the bosonic k-event overlap"),
        "encoding": encoding,
        "metric_primary": {
            "name": "best exact c-recovery margin over floor across the locality dial k",
            "value": float(best_res),
        },
        "fisher": {"eigenvalues": [float(x) for x in fw], "n_resolvable": resolv},
        "locality_sweep": sweep,
        "verdict": verdict,
        "config": {"q": q, "R": R, "seeds": seeds, "k_list": list(k_list),
                   "n_train": S_tr, "n_test": S_te, "n_events": N_evt},
        "wall_seconds": time.time() - t0,
    }
    OUT.mkdir(exist_ok=True)
    tag = f"_{encoding}"
    (OUT / f"cell_perm_overlap{tag}.json").write_text(
        json.dumps(result, indent=2, default=_to_py))
    _plot(result, tag)
    _print_table(result)
    return result


def _print_table(r):
    sw = r["locality_sweep"]
    print(f"\nencoding={r['encoding']}  q={r['config']['q']}  "
          f"Fisher resolvable={r['fisher']['n_resolvable']}  "
          f"floor_r2={sw['floor_r2']:+.4f}")
    print(f"\n{'k (locality)':>13}{'res margin':>14}{'full margin':>14}")
    print("-" * 41)
    for i, k in enumerate(sw["k"]):
        print(f"{k:>13}{sw['res'][i]:>+13.4f} {sw['full'][i]:>+13.4f}")
    print(f"\nDECISION: {r['verdict']['decision']}")


def _plot(r, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"  (plot skipped: {e})")
        return
    sw = r["locality_sweep"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.errorbar(sw["k"], sw["res"], yerr=sw["res_std"], marker="o", capsize=3,
                color="#26679e", label="bosonic overlap (resolvable subspace)")
    ax.axhline(0.0, ls="--", color="k", lw=0.9, label="random-feature floor")
    ax.set_xlabel("locality dial k  (k=1: arithmetic-mean kernel; large k: #P-hard permanent)")
    ax.set_ylabel("exact c-recovery margin over floor")
    ax.set_title(f"G3 locality sweep ({r['encoding']}, q={r['config']['q']})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"cell_perm_overlap{tag}.png", dpi=140)
    print(f"  wrote {OUT / f'cell_perm_overlap{tag}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--R", type=int, default=16, help="bosonic subsets per scenario")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--encoding", choices=("spinor", "iqp"), default="spinor")
    ap.add_argument("--budget", type=int, default=None,
                    help="hold event budget R_k*k ~constant: R_k = budget//k")
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run(q=2, k_list=(1, 2, 3), R=6, seeds=2, encoding=args.encoding)
    else:
        ks = tuple(k for k in (1, 2, 3, 4, 6, 8) if k <= args.kmax)
        run(q=args.q, k_list=ks, R=args.R, seeds=args.seeds,
            encoding=args.encoding, budget=args.budget)
