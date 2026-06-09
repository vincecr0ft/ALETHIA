r"""G1 cell: multi-copy (SWAP-test) vs single-copy (classical-shadow) estimation
of the amplitude-overlap kernel  ->  SMEFT manifold disclosure under a finite
copy budget.

==============================================================================
WHY THIS CELL EXISTS  (read this before the code)
==============================================================================

The sibling `cell_quantum_kernel.py` showed that a state-fidelity quantum kernel,
mean-embedded over an event set, recovers the SMEFT Wilson coefficients NO BETTER
than a matched classical RBF kernel and never beats a random-feature floor, with
no kernel concentration. That null is not a failure: it is exactly what the
quantum-learning theory predicts for a SINGLE-COPY estimator. The provable
quantum advantages in learning about quantum systems (Huang et al., Science 376,
1182, 2022, arXiv:2112.00778; Chen-Cotler-Huang-Li, FOCS 2021, arXiv:2111.05881)
are NOT a property of "using a quantum feature map." They are a property of having
QUANTUM MEMORY: the ability to hold and coherently process two or more copies of a
state at once. The advantage is a SAMPLE-COMPLEXITY statement, invisible in the
infinite-copy exact-statevector limit.

cell_quantum_kernel works in that infinite-copy limit (it forms the exact density
matrix rho_c and takes exact inner products), so by construction it cannot see the
separation. This cell puts the copy budget back as the manipulated variable.

The physical object. The SMEFT cross section is |M_SM + sum_i c_i M_i|^2, so the
morphing templates are amplitude OVERLAPS A_i ~ Re<M_SM|M_i>, B_ij ~ Re<M_i|M_j>.
Our scenario summary is the mean density matrix rho_s = mean_n |psi(x_n)><psi(x_n)|
of an event set, and the disclosure kernel is the overlap K_ss' = Tr(rho_s rho_s').
This is a QUADRATIC functional of the state — the same class of object (an overlap
of amplitude vectors) that the coherent-amplitude-evaluation program prepares and
measures directly on a quantum device (arXiv:2602.21311). The central question of
this cell:

    At a fixed budget of M state copies per scenario, does a TWO-COPY estimator of
    the overlap kernel (a SWAP test) recover the Wilson coefficients better than
    the best SINGLE-COPY estimator (classical shadows), and does that gap WIDEN
    with qubit number?

Why two-copy should win, quantitatively. A SWAP test on rho_s (x) rho_s' outputs a
Bernoulli with P(coincidence) = (1 + Tr(rho_s rho_s'))/2, so M_pair pairs estimate
the overlap with variance (1 - F^2)/M_pair, INDEPENDENT of Hilbert-space dimension.
A single-copy classical-shadow estimate of the same quadratic overlap Tr(rho rho')
carries the shadow-norm variance, which for a global (Clifford/Haar) shadow grows
~ 2^q / M. So at fixed M the SWAP estimate of the kernel is exponentially cleaner
as q grows. If the kernel entries drive c-recovery, the recovery margin should be
(a) flat in q for the SWAP estimator and (b) decaying in q for the shadow
estimator. The crossover, if it exists, is the dimension-resolved fingerprint of
the multi-copy advantage — on a substrate where we own the analytic ground truth.

What this cell is NOT. It is not a quantum-advantage claim. The exact (infinite-M)
kernel still only ties the classical baseline (that is the cell_quantum_kernel
result, reproduced here as the `exact` mode). The hypothesis under test is strictly
about SAMPLE COMPLEXITY: that at small M the two-copy estimator reaches the same
disclosure with fewer copies than any single-copy strategy. That is the only ground
on which a separation is provable, so it is the only honest place to look.

==============================================================================
DECISION RULE  (this is a make-or-break gate, G1 of the standalone program)
==============================================================================

PROCEED to G2+ iff, on the Fisher-resolvable subspace, the SWAP mode reaches a
c-recovery margin at a copy budget M where the SHADOW mode cannot — i.e. there is
an M-window with swap_margin(M) > shadow_margin(M) beyond seed noise — AND that gap
widens (or shadow degrades while swap stays flat) as q grows in the q-sweep.

STOP and report a conditional null iff SWAP ties SHADOW at every M and every q:
then multi-copy memory buys nothing for this overlap target, and the SMEFT kernel
is dequantizable in the sample-complexity sense too. Either outcome is publishable;
the second is "dequantization confirmed on a new substrate, including in sample
complexity," which is a clean result, not a failed experiment.

==============================================================================
SIMULATION STATUS
==============================================================================

Everything here is EXACT few-qubit statevector simulation in numpy/scipy. No
hardware, no PennyLane. The SWAP test is simulated by binomial sampling around the
exact overlap (the device would measure the same Bernoulli). The classical shadow
is simulated by drawing global Haar single-copy measurements and applying the
exact inverse channel rho_hat = (D+1) v v^dag - I per snapshot. The ONLY step of
the standalone program that needs real hardware is G5 (running the SWAP test on a
device and measuring how fast the advantage collapses under noise, cf.
arXiv:2204.13691); this cell is its classically-simulable precondition.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

import substrate as sub
import probes
from cell_quantum_kernel import SpinorFeatureMap, IQPFeatureMap, _to_py
from qk_scan import fisher_basis

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"


# --------------------------------------------------------------------------
# Exact per-scenario density matrices and the exact overlap Gram.
# --------------------------------------------------------------------------

def scenario_density_vecs(fmap, sets: np.ndarray, s1: np.ndarray,
                          s2: float) -> np.ndarray:
    """vec(rho_s) for every scenario, rho_s = mean_n |psi(x_n)><psi(x_n)|.

    Returns V: (S, D*D) complex, V[s] = rho_s.ravel(). Tr(rho_s rho_s') is then
    Re(<V[s], V[s']>) because both rho are Hermitian, so the overlap Gram is a
    single matmul (identical inner-product structure to cell_quantum_kernel).

    Convention note (verified, harmless): psi.conj().T @ psi stores conj(rho) =
    rho^T, not rho itself. Every downstream use is an overlap Tr(rho_s rho_s'),
    and Tr(conj rho . conj sigma) = Tr(rho sigma) is real, so the Gram is
    identical to using the true rho. The shadow path reconstructs the true rho;
    the two conventions agree under the trace (see verify_quantum_pennylane.py).
    """
    S, N, _ = sets.shape
    D = fmap.dim
    V = np.empty((S, D * D), dtype=np.complex128)
    for s in range(S):
        psi = fmap.states(sets[s], s1, s2)            # (N, D)
        rho = (psi.conj().T @ psi) / N                # (D, D) Hermitian, tr=1
        V[s] = rho.ravel()
    return V


def overlap_gram(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """Exact overlap Gram K[i,j] = Tr(rho^a_i rho^b_j) = Re(<Va_i, Vb_j>)."""
    return (Va @ Vb.conj().T).real


# --------------------------------------------------------------------------
# Two-copy estimator: simulated SWAP test (dimension-independent variance).
# --------------------------------------------------------------------------

def swap_estimate(K_exact: np.ndarray, n_copies: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Estimate the overlap Gram from M = n_copies SWAP-test shots per entry.

    A SWAP test on rho (x) sigma yields P(coincidence) = (1 + Tr(rho sigma))/2.
    With M shots, p_hat ~ Binomial(M, p)/M and F_hat = 2 p_hat - 1, giving
    Var(F_hat) = (1 - F^2)/M, INDEPENDENT of Hilbert-space dimension. Each pair
    of copies is consumed once, so M copies of each state give M/2 ... we charge
    M shots = M copy-pairs for a like-for-like copy budget with the shadow mode
    (both spend M state-preparations per scenario; see _copies_note).
    """
    p = np.clip((1.0 + K_exact) / 2.0, 0.0, 1.0)
    counts = rng.binomial(n_copies, p)
    return 2.0 * (counts / n_copies) - 1.0


def swap_grams(K_tt_exact, K_et_exact, n_copies, rng):
    """SWAP-estimated train-train (symmetrised) and test-train Grams."""
    n = K_tt_exact.shape[0]
    Ktt = swap_estimate(K_tt_exact, n_copies, rng)
    iu = np.triu_indices(n, k=1)
    Ktt[(iu[1], iu[0])] = Ktt[iu]                     # symmetrise upper -> lower
    Ket = swap_estimate(K_et_exact, n_copies, rng)
    return Ktt, Ket


# --------------------------------------------------------------------------
# Single-copy estimator: classical shadows (global Haar), variance ~ D/M.
# --------------------------------------------------------------------------

def shadow_density_vecs(fmap, sets: np.ndarray, s1: np.ndarray, s2: float,
                        n_copies: int, rng: np.random.Generator) -> tuple:
    """Two independent classical-shadow estimates rho_hat_A, rho_hat_B per
    scenario, each from n_copies/2 global single-copy measurements.

    Global (Haar) shadow snapshot: draw U ~ Haar(D), measure one copy of rho_s in
    the U-rotated computational basis (outcome b ~ |<b|U|psi>|^2 for a random
    event psi), invert with rho_hat = (D+1) v v^dag - I where v = U^dag|b>. The
    estimator is unbiased for rho_s. Returning TWO independent halves lets the
    overlap Gram Tr(rho_hat_A rho_hat_B) be unbiased even on the diagonal (the
    standard split-shadow trick for quadratic functionals).

    Variance of the resulting overlap estimate grows ~ D/M (the global shadow
    norm), the single-copy penalty the SWAP test does not pay.
    """
    S, N, _ = sets.shape
    D = fmap.dim
    half = max(1, n_copies // 2)
    VA = np.empty((S, D * D), dtype=np.complex128)
    VB = np.empty((S, D * D), dtype=np.complex128)
    eyeflat = np.eye(D, dtype=np.complex128).ravel()
    for s in range(S):
        psi_all = fmap.states(sets[s], s1, s2)        # (N, D) pure event states
        for bank, Vout in ((0, VA), (1, VB)):
            ev = rng.integers(0, N, size=half)        # sample events (defines rho_s)
            psi = psi_all[ev]                         # (half, D)
            U = unitary_group.rvs(D, size=half, random_state=rng)  # (half, D, D)
            if half == 1:
                U = U[None]
            Upsi = np.einsum("mij,mj->mi", U, psi)    # (half, D)
            probs = (Upsi.conj() * Upsi).real
            probs /= probs.sum(1, keepdims=True)
            # Vectorised Born sampling (inverse-CDF) — replaces a per-snapshot
            # rng.choice loop; validated identical in verify_quantum_pennylane.py.
            cum = np.cumsum(probs, axis=1)
            b = (rng.random(half)[:, None] < cum).argmax(1)
            v = np.conj(U[np.arange(half), b, :])     # (half, D), v_m = U^dag|b_m>
            # snapshot rho_hat^(m) = (D+1) v v^dag - I; average over the half.
            outer = np.einsum("mi,mj->ij", v, v.conj()) / half   # mean v v^dag
            rho_hat = (D + 1.0) * outer.ravel() - eyeflat
            Vout[s] = rho_hat
    return VA, VB


def shadow_grams(fmap, Xtr, Xte, s1, s2, n_copies, rng):
    """Shadow-estimated train-train and test-train overlap Grams (unbiased via
    independent A/B shadow banks)."""
    VA_tr, VB_tr = shadow_density_vecs(fmap, Xtr, s1, s2, n_copies, rng)
    VA_te, _VB_te = shadow_density_vecs(fmap, Xte, s1, s2, n_copies, rng)
    Ktt = overlap_gram(VA_tr, VB_tr)                  # unbiased incl. diagonal
    Ket = overlap_gram(VA_te, VB_tr)
    return Ktt, Ket


# --------------------------------------------------------------------------
# Geometric difference (Huang et al., arXiv:2011.01938) — secondary diagnostic.
# --------------------------------------------------------------------------

def geometric_difference(K_quantum: np.ndarray, K_classical: np.ndarray,
                         lam: float = 1e-3) -> float:
    r"""Asymmetric geometric difference g(C -> Q) of two (train) Gram matrices.

    Both kernels are trace-normalised to N, then
        g = sqrt( || sqrt(K_Q) (K_C + lam I)^{-1} sqrt(K_Q) ||_inf ).
    A large g (growing like sqrt(N)) is a NECESSARY condition for the quantum
    kernel to have any prediction advantage over the classical one on some
    labelling; g = O(1) means the classical kernel can mimic it. This is a
    data-dependent screen, not a guarantee — reported alongside the c-recovery
    margins, never instead of them.
    """
    N = K_quantum.shape[0]

    def _norm(K):
        K = 0.5 * (K + K.T)
        tr = np.trace(K)
        return N * K / tr if tr > 0 else K

    KQ = _norm(K_quantum)
    KC = _norm(K_classical)
    w, U = np.linalg.eigh(KQ)
    sqrtKQ = (U * np.sqrt(np.clip(w, 0, None))) @ U.T
    Cinv = np.linalg.inv(KC + lam * np.eye(N))
    M = sqrtKQ @ Cinv @ sqrtKQ
    return float(np.sqrt(np.max(np.abs(np.linalg.eigvalsh(0.5 * (M + M.T))))))


# --------------------------------------------------------------------------
# Classical RBF overlap Gram (matched control, for the geometric difference).
# --------------------------------------------------------------------------

def rbf_gram(raw_a: np.ndarray, raw_b: np.ndarray, gamma: float) -> np.ndarray:
    """RBF Gram exp(-gamma ||raw_a_i - raw_b_j||^2) on raw per-scenario summaries."""
    d2 = (np.sum(raw_a ** 2, 1)[:, None] + np.sum(raw_b ** 2, 1)[None, :]
          - 2.0 * raw_a @ raw_b.T)
    return np.exp(-gamma * np.clip(d2, 0, None))


# --------------------------------------------------------------------------
# Score one (mode, M) configuration through the shared held-out probe.
# --------------------------------------------------------------------------

def _probe_from_grams(Ktt, Ket, Ctr, Cte, raw_tr, raw_te, seed):
    """Empirical-kernel-map ridge: scenario feature = its row of the overlap
    Gram against the train anchors. Reuses probes.probe_with_floor verbatim, so
    the random-feature floor and capacity control are identical to every other
    disclosure cell. Returns the ProbeResult (carries .margin)."""
    return probes.probe_with_floor(Ktt, Ctr, Ket, Cte, raw_tr, raw_te, seed=seed)


# --------------------------------------------------------------------------
# Main experiment: copy-budget sweep (fixed q) + qubit sweep (fixed M).
# --------------------------------------------------------------------------

def run(q: int = 3, copies=(16, 64, 256, 1024, 4096),
        q_sweep=(2, 3, 4), q_sweep_copies: int = 128,
        seeds: int = 4, encoding: str = "spinor") -> dict:
    t0 = time.time()
    data = sub.load_cache()
    Xtr, Ctr = data["train_X1"], data["train_c"]
    Xte, Cte = data["test_X1"], data["test_c"]
    n_wc = Ctr.shape[1]
    raw_tr = probes.raw_event_summary(Xtr)
    raw_te = probes.raw_event_summary(Xte)

    # Fisher-resolvable subspace (the directions that carry morphing signal).
    oracle = sub.make_oracle()
    fw, fV = fisher_basis(oracle, data["sm_X1"], n_wc)
    resolv = max(1, min(int(np.sum(fw > 0.01 * fw.max())), n_wc))
    Rsub = fV[:, :resolv]
    Ctr_res, Cte_res = Ctr @ Rsub, Cte @ Rsub

    def make_map(qq):
        return (SpinorFeatureMap(n_qubits=qq) if encoding == "spinor"
                else IQPFeatureMap(n_qubits=qq, depth=2))

    def grams_exact(fmap):
        s1 = np.ones(fmap.q)
        V = scenario_density_vecs(fmap, Xtr, s1, 1.0)
        Vte = scenario_density_vecs(fmap, Xte, s1, 1.0)
        return overlap_gram(V, V), overlap_gram(Vte, V), s1

    def score(Ktt, Ket):
        full = _probe_from_grams(Ktt, Ket, Ctr, Cte, raw_tr, raw_te, 0).margin
        res = _probe_from_grams(Ktt, Ket, Ctr_res, Cte_res, raw_tr, raw_te, 0).margin
        return full, res

    # ---- (1) Copy-budget sweep at fixed q ----
    fmap = make_map(q)
    Ktt_x, Ket_x, s1 = grams_exact(fmap)
    exact_full, exact_res = score(Ktt_x, Ket_x)

    sweep = {"copies": list(copies), "exact": {"full": exact_full, "res": exact_res},
             "swap": {"full": [], "full_std": [], "res": [], "res_std": []},
             "shadow": {"full": [], "full_std": [], "res": [], "res_std": []}}

    for M in copies:
        sw_f, sw_r, sh_f, sh_r = [], [], [], []
        for sd in range(seeds):
            rng = np.random.default_rng(7000 + sd)
            Ktt_s, Ket_s = swap_grams(Ktt_x, Ket_x, M, rng)
            f, r = score(Ktt_s, Ket_s)
            sw_f.append(f); sw_r.append(r)
            Ktt_h, Ket_h = shadow_grams(fmap, Xtr, Xte, s1, 1.0, M, rng)
            f, r = score(Ktt_h, Ket_h)
            sh_f.append(f); sh_r.append(r)
        for key, vals in (("swap", (sw_f, sw_r)), ("shadow", (sh_f, sh_r))):
            sweep[key]["full"].append(float(np.mean(vals[0])))
            sweep[key]["full_std"].append(float(np.std(vals[0])))
            sweep[key]["res"].append(float(np.mean(vals[1])))
            sweep[key]["res_std"].append(float(np.std(vals[1])))
        print(f"  [M-sweep q={q}] M={M:>5}  swap_res={sweep['swap']['res'][-1]:+.4f}"
              f"  shadow_res={sweep['shadow']['res'][-1]:+.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- (2) Qubit sweep at fixed M (the separation fingerprint) ----
    qscan = {"qubits": list(q_sweep), "copies": q_sweep_copies,
             "exact_res": [], "swap_res": [], "swap_res_std": [],
             "shadow_res": [], "shadow_res_std": [], "geom_diff": []}
    for qq in q_sweep:
        fm = make_map(qq)
        Ktt_x2, Ket_x2, s1q = grams_exact(fm)
        _, ex_r = score(Ktt_x2, Ket_x2)
        qscan["exact_res"].append(ex_r)
        gamma = 1.0 / (2.0 * np.median(np.sum((raw_tr[:200] - raw_tr[:200].mean(0)) ** 2, 1)) + 1e-9)
        qscan["geom_diff"].append(
            geometric_difference(Ktt_x2, rbf_gram(raw_tr, raw_tr, gamma)))
        sw_r, sh_r = [], []
        for sd in range(seeds):
            rng = np.random.default_rng(9000 + sd)
            Ktt_s, Ket_s = swap_grams(Ktt_x2, Ket_x2, q_sweep_copies, rng)
            sw_r.append(score(Ktt_s, Ket_s)[1])
            Ktt_h, Ket_h = shadow_grams(fm, Xtr, Xte, s1q, 1.0, q_sweep_copies, rng)
            sh_r.append(score(Ktt_h, Ket_h)[1])
        qscan["swap_res"].append(float(np.mean(sw_r)))
        qscan["swap_res_std"].append(float(np.std(sw_r)))
        qscan["shadow_res"].append(float(np.mean(sh_r)))
        qscan["shadow_res_std"].append(float(np.std(sh_r)))
        print(f"  [q-sweep M={q_sweep_copies}] q={qq}  swap_res={qscan['swap_res'][-1]:+.4f}"
              f"  shadow_res={qscan['shadow_res'][-1]:+.4f}"
              f"  g={qscan['geom_diff'][-1]:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- Decision ----
    swap_beats = [s - h for s, h in zip(sweep["swap"]["res"], sweep["shadow"]["res"])]
    widening = (qscan["swap_res"][-1] - qscan["shadow_res"][-1]) > \
               (qscan["swap_res"][0] - qscan["shadow_res"][0])
    verdict = {
        "swap_minus_shadow_res_by_M": swap_beats,
        "max_swap_advantage_res": float(max(swap_beats)) if swap_beats else 0.0,
        "gap_widens_with_q": bool(widening),
        "decision": ("PROCEED (multi-copy gap present)" if max(swap_beats, default=0) > 0.02
                     and widening else
                     "STOP / conditional null (swap ties shadow)"),
        "note": ("PROCEED needs both a copy-budget window where swap beats shadow on "
                 "the resolvable subspace AND a gap that widens with q. Thresholds "
                 "(0.02 margin) are heuristic; read the curves, not just the flag."),
    }

    result = {
        "cell": "swap_overlap",
        "fm_family": "Multi-copy overlap kernel (SWAP test) vs single-copy (classical shadow)",
        "hep_task": ("SMEFT manifold disclosure under a finite copy budget: "
                     "Wilson-coefficient recovery from a sample-estimated overlap kernel"),
        "encoding": encoding,
        "metric_primary": {
            "name": "max SWAP-over-shadow c-recovery-margin gap on resolvable subspace",
            "value": verdict["max_swap_advantage_res"],
        },
        "fisher": {"eigenvalues": [float(x) for x in fw], "n_resolvable": resolv},
        "copy_budget_sweep": sweep,
        "qubit_sweep": qscan,
        "verdict": verdict,
        "config": {"q": q, "seeds": seeds, "n_train": int(Xtr.shape[0]),
                   "n_test": int(Xte.shape[0]), "n_events": int(Xtr.shape[1]),
                   "copies": list(copies), "q_sweep": list(q_sweep),
                   "q_sweep_copies": q_sweep_copies},
        "wall_seconds": time.time() - t0,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "cell_swap_overlap.json").write_text(json.dumps(result, indent=2, default=_to_py))
    _plot(result)
    _print_table(result)
    return result


def _print_table(r):
    sw = r["copy_budget_sweep"]
    print(f"\nencoding={r['encoding']}  Fisher resolvable={r['fisher']['n_resolvable']}"
          f"  exact(res)={sw['exact']['res']:+.4f}")
    print(f"\n{'M copies':>9}{'swap (res)':>16}{'shadow (res)':>16}{'swap-shadow':>14}")
    print("-" * 55)
    for i, M in enumerate(sw["copies"]):
        s, h = sw["swap"]["res"][i], sw["shadow"]["res"][i]
        print(f"{M:>9}{s:>+13.4f}   {h:>+13.4f}   {s-h:>+11.4f}")
    qs = r["qubit_sweep"]
    print(f"\n{'q':>3}{'swap (res)':>14}{'shadow (res)':>15}{'geom diff':>12}")
    print("-" * 45)
    for i, qq in enumerate(qs["qubits"]):
        print(f"{qq:>3}{qs['swap_res'][i]:>+13.4f}  {qs['shadow_res'][i]:>+13.4f}"
              f"  {qs['geom_diff'][i]:>10.2f}")
    print(f"\nDECISION: {r['verdict']['decision']}")


def _plot(r):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"  (plot skipped: {e})")
        return
    sw = r["copy_budget_sweep"]
    qs = r["qubit_sweep"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    M = sw["copies"]
    ax1.errorbar(M, sw["swap"]["res"], yerr=sw["swap"]["res_std"], marker="o",
                 capsize=3, label="two-copy (SWAP)", color="#26679e")
    ax1.errorbar(M, sw["shadow"]["res"], yerr=sw["shadow"]["res_std"], marker="s",
                 capsize=3, label="single-copy (shadow)", color="#b5651d")
    ax1.axhline(sw["exact"]["res"], ls="--", color="k", lw=0.9,
                label=f"exact (M=inf) {sw['exact']['res']:+.3f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("copy budget M per scenario")
    ax1.set_ylabel("c-recovery margin (resolvable subspace)")
    ax1.set_title(f"Copy-budget sweep (q={r['config']['q']}, {r['encoding']})")
    ax1.legend(frameon=False, fontsize=8)
    q = qs["qubits"]
    ax2.errorbar(q, qs["swap_res"], yerr=qs["swap_res_std"], marker="o", capsize=3,
                 label="two-copy (SWAP)", color="#26679e")
    ax2.errorbar(q, qs["shadow_res"], yerr=qs["shadow_res_std"], marker="s", capsize=3,
                 label="single-copy (shadow)", color="#b5651d")
    ax2.plot(q, qs["exact_res"], ls="--", color="k", lw=0.9, marker="x", label="exact")
    ax2.set_xlabel("qubits q  (Hilbert dim 2^q)")
    ax2.set_ylabel("c-recovery margin (resolvable subspace)")
    ax2.set_title(f"Qubit sweep at fixed M={qs['copies']} (separation fingerprint)")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cell_swap_overlap.png", dpi=140)
    print(f"  wrote {OUT / 'cell_swap_overlap.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=3, help="qubits for the copy-budget sweep")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--encoding", choices=("spinor", "iqp"), default="spinor")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run")
    args = ap.parse_args()
    if args.smoke:
        run(q=2, copies=(16, 128), q_sweep=(2, 3), q_sweep_copies=64,
            seeds=2, encoding=args.encoding)
    else:
        run(q=args.q, seeds=args.seeds, encoding=args.encoding)
