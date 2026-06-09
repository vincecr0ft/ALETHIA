r"""Capability cell: Quantum kernel  ->  SMEFT manifold disclosure (c-recovery).

FM family favoured: kernel methods with a *quantum* feature map. This is the
quantum sibling of the in-context / Intention closed-form ridge (cell_pfn): the
Intention head is kernel ridge regression with a learned classical feature map;
here the feature map is a data-encoding quantum circuit and the kernel is the
state fidelity. The question this cell poses is the disclosure question, not a
generation/forecast one: does a quantum reproducing-kernel Hilbert space *span
the SMEFT morphing manifold* — i.e. recover the Wilson coefficients from a
frozen representation — better than a random-feature floor and a matched
classical kernel?

Why a quantum kernel is a natural object here. The SMEFT cross section is a
squared amplitude, |M_SM + sum_i c_i M_i|^2, so the morphing tangents and
curvatures are *overlaps* of amplitude vectors (A_i ~ Re<M_SM|M_i>,
B_ij ~ Re<M_i|M_j>). A fidelity quantum kernel |<psi(x)|psi(x')>|^2 computes
exactly that kind of overlap, so the morphing structure and the kernel's
inner-product structure are the same shape. Whether the *chosen encoding* aligns
its RKHS with the morphing manifold is the empirical question (expressivity is
not alignment — Kuebler/Schuld), and that is what the trained sub-row tests.

Construction.
  * Per-event feature map: an IQP / ZZ-feature-map encoding (Havlicek et al.,
    Nature 2019) of x = (log m_ll, cos theta*) into n_qubits, depth L. The
    data-dependent part is diagonal, so the n<=8-qubit statevector is *exact*
    in numpy — no quantum hardware, no shot noise (that is a deliberate first
    cut; shot noise / concentration is a follow-up).
  * Set -> scenario: kernel mean embedding. A scenario is an event set; its
    quantum summary is the average density matrix rho_c = mean_n |psi(x_n)><psi(x_n)|,
    the state describing the event ensemble. Its (real) entries are the
    per-scenario feature vector. The induced scenario kernel is Tr(rho_c rho_c'),
    the fidelity kernel mean-embedded over the set (the "tall-data set SBI"
    summary).
  * Scoring: the SAME held-out, capacity-controlled ridge probe with a matched
    random-feature floor used by every disclosure cell (probes.probe_with_floor).
    margin = probe_r2 - floor_r2 > 0 means the quantum representation beats a
    random projection of the raw events at the same width.

Two sub-rows, the fixed-vs-trained contrast that makes this a result rather than
a gimmick (it is the same axis as the V=K degeneracy in the classical head):
  * fixed   — a generic encoding (unit angle scales): does a generic quantum
              RKHS already contain the morphing templates? (random-feature-like)
  * trained — encoding angle scales optimised by kernel-target alignment
              (Hubregtsen et al.): the embedding kernel that genuinely *learns
              the manifold metric*, the fair analogue of the trained classical
              phi_theta.

Metric (native to the objective): the trained quantum kernel's c-recovery
margin over the random-feature floor. A matched classical RBF kernel-mean-
embedding (same construction, same width) is reported as the head-to-head
control, and a kernel-concentration diagnostic flags the standard quantum-kernel
failure mode (Thanasilp et al.).
"""
from __future__ import annotations

import json
import time
from functools import reduce
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import substrate as sub
import probes

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"


# --------------------------------------------------------------------------
# Quantum feature map (exact statevector, pure numpy).
# --------------------------------------------------------------------------

class _FidelityMap:
    """Shared set->scenario step: mean density-matrix kernel-mean embedding.

    Any subclass exposing ``.dim`` and ``.states(X, s1, s2) -> (M, dim) complex``
    gets the per-scenario feature vector rho_c = mean_n |psi(x_n)><psi(x_n)|,
    flattened to (Re, Im). The induced scenario kernel is Tr(rho_c rho_c'), the
    fidelity kernel mean-embedded over the event set.
    """

    def scenario_features(self, sets: np.ndarray, s1: np.ndarray,
                          s2: float) -> np.ndarray:
        S, N, _ = sets.shape
        feats = np.empty((S, 2 * self.dim * self.dim), dtype=np.float64)
        for s in range(S):                                 # loop S (small), vectorise N
            psi = self.states(sets[s], s1, s2)             # (N, dim)
            rho = (psi.conj().T @ psi) / N                 # (dim, dim), Hermitian, tr=1
            feats[s] = np.concatenate([rho.real.ravel(), rho.imag.ravel()])
        return feats


class IQPFeatureMap(_FidelityMap):
    r"""Exact IQP / ZZ-feature-map statevector encoder for 2-D events.

    State for one event x:  psi(x) = (U_phi(x) H^{otimes q})^L |0>^q,
    with U_phi(x) diagonal in the computational basis:
        U_phi(x) = exp(i [ sum_j a_j(x) Z_j + sum_{(j,k)} a_jk(x) Z_j Z_k ]).
    Single-qubit angle a_j(x) = s1_j * x[feat_j]; entangling angle on a linear
    chain a_jk(x) = s2 * (pi - x[feat_j])(pi - x[feat_k]) (Havlicek form). The
    feature->qubit assignment is round-robin over the 2 data features.

    Diagonal data encoding => the whole map is matrix-free except the fixed H^q,
    so it is exact and fast for q <= ~8.
    """

    def __init__(self, n_qubits: int = 3, depth: int = 2):
        self.q = int(n_qubits)
        self.L = int(depth)
        self.dim = 2 ** self.q
        # Z eigenvalues per (basis state, qubit): +1 if bit==0 else -1.
        b = np.arange(self.dim)
        self.Z = np.stack([1 - 2 * ((b >> j) & 1) for j in range(self.q)],
                          axis=1).astype(np.float64)         # (dim, q)
        # Linear-chain entangling pairs and their Z_j Z_k eigenvalues.
        self.pairs = [(j, j + 1) for j in range(self.q - 1)]
        self.zz = [self.Z[:, j] * self.Z[:, k] for (j, k) in self.pairs]  # each (dim,)
        # Feature index assigned to each qubit (round-robin over 2 features).
        self.feat = [j % 2 for j in range(self.q)]
        # Fixed H^{otimes q}.
        H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        self.Hq = reduce(np.kron, [H] * self.q)             # (dim, dim), symmetric

    def _phase(self, X: np.ndarray, s1: np.ndarray, s2: float) -> np.ndarray:
        """Per-event encoding phase, shape (M, dim). X: (M, 2)."""
        # Single-qubit: (M, q) angles  ->  (M, dim) via Z.
        a1 = X[:, self.feat] * s1[None, :]                  # (M, q)
        single = a1 @ self.Z.T                              # (M, dim)
        # Entangling chain.
        pair = np.zeros((X.shape[0], self.dim))
        for (j, k), zz in zip(self.pairs, self.zz):
            a_jk = s2 * (np.pi - X[:, self.feat[j]]) * (np.pi - X[:, self.feat[k]])
            pair += a_jk[:, None] * zz[None, :]
        return single + pair

    def states(self, X: np.ndarray, s1: np.ndarray, s2: float) -> np.ndarray:
        """Statevectors psi(x) for a batch of events. X: (M, 2) -> (M, dim) complex."""
        phase = self._phase(X, s1, s2)
        ph = np.exp(1j * phase)                             # (M, dim)
        amp = np.zeros((X.shape[0], self.dim), dtype=np.complex128)
        amp[:, 0] = 1.0                                     # |0...0>
        for _ in range(self.L):
            amp = amp @ self.Hq                             # Hq symmetric
            amp = amp * ph
        return amp

# --------------------------------------------------------------------------
# Spinor / amplitude-aligned feature map (arXiv:2602.21311).
# --------------------------------------------------------------------------

class SpinorFeatureMap(_FidelityMap):
    r"""Amplitude-aligned encoder: Weyl-spinor / Bloch states for 2-D events.

    Motivated by Coherent Quantum Evaluation of Collider Amplitudes (2602.21311),
    where each massless leg's normalised Weyl spinor is a single-qubit Bloch
    state prepared by a U_3 rotation, and helicity amplitudes are overlaps
    <ij>, [ij] of those spinors. For Drell-Yan the SMEFT morphing templates are
    amplitude overlaps (A_i ~ Re<M_SM|M_i>, B_ij ~ Re<M_i|M_j>), so a feature map
    built from spinor states puts those overlaps natively in the fidelity kernel
    — the alignment the generic IQP map lacks.

    For our reduced kinematics x = (log m_ll, cos theta*), the scattering polar
    angle theta = arccos(cos theta*) sets the angle-qubit spinors (reproducing
    the (1 +/- cos theta) angular brackets), the energy proxy en = sigmoid(log m)
    sets the energy-qubit spinors (the sqrt(s)/EFT energy-growth axis), and
    energy-dependent azimuthal phases (the dotted-vs-undotted lambda/lambda-tilde
    distinction) plus a chain of CNOTs build the angle x energy interference
    cross-terms a single product state cannot carry.

    Per qubit j:  |q_j> = cos(alpha_j/2)|0> + e^{i beta_j} sin(alpha_j/2)|1>,
      even j (angle qubit) : alpha = s1[j]*theta,        beta = s2 * log m_ll
      odd  j (energy qubit): alpha = s1[j]*pi*sigmoid(m), beta = s2 * theta
    then a CNOT chain (j -> j+1) entangles them. Exact statevector, q <= ~8.
    """

    def __init__(self, n_qubits: int = 3):
        self.q = int(n_qubits)
        self.dim = 2 ** self.q
        # CNOT-chain permutations (control j -> target j+1), qubit 0 = MSB.
        b = np.arange(self.dim)
        self.perms = []
        for c, t in [(j, j + 1) for j in range(self.q - 1)]:
            cbit = (b >> (self.q - 1 - c)) & 1
            flip = b ^ (1 << (self.q - 1 - t))
            self.perms.append(np.where(cbit == 1, flip, b))

    def states(self, X: np.ndarray, s1: np.ndarray, s2: float) -> np.ndarray:
        """Spinor statevectors psi(x). X: (M, 2) -> (M, dim) complex."""
        M = X.shape[0]
        theta = np.arccos(np.clip(X[:, 1], -1.0, 1.0))      # scattering angle
        logm = X[:, 0]
        en = 1.0 / (1.0 + np.exp(-logm))                    # energy proxy in (0,1)
        amp = np.ones((M, 1), dtype=np.complex128)
        for j in range(self.q):
            if j % 2 == 0:
                alpha = s1[j] * theta
                beta = s2 * logm
            else:
                alpha = s1[j] * np.pi * en
                beta = s2 * theta
            q0 = np.cos(alpha / 2.0)
            q1 = np.exp(1j * beta) * np.sin(alpha / 2.0)
            qj = np.stack([q0, q1], axis=-1)                # (M, 2) complex
            amp = (amp[:, :, None] * qj[:, None, :]).reshape(M, -1)  # kron
        for perm in self.perms:                             # entangle (CNOT chain)
            amp = amp[:, perm]
        return amp


# --------------------------------------------------------------------------
# Kernel-target alignment (for the trained sub-row).
# --------------------------------------------------------------------------

def _centered_gram(Z: np.ndarray) -> np.ndarray:
    Zc = Z - Z.mean(0, keepdims=True)
    return Zc @ Zc.T


def kernel_target_alignment(Z: np.ndarray, Y: np.ndarray) -> float:
    """Centered kernel-target alignment <K, K_Y>_F / (||K|| ||K_Y||) in [0,1]."""
    K = _centered_gram(Z)
    Yc = Y - Y.mean(0, keepdims=True)
    KY = Yc @ Yc.T
    num = float(np.sum(K * KY))
    den = float(np.linalg.norm(K) * np.linalg.norm(KY)) + 1e-30
    return num / den


def train_scales(fmap: IQPFeatureMap, sets: np.ndarray, C: np.ndarray,
                 *, maxiter: int = 60, seed: int = 0):
    """Optimise (s1_a, s1_b, s2) encoding angle scales by kernel-target alignment.

    Three parameters only: a shared scale per data feature and one entangling
    scale. Clipped to [0.05, 5]. Returns (scales_dict, best_alignment).
    """
    q = fmap.q

    def unpack(p):
        s1a, s1b, s2 = np.clip(np.abs(p), 0.05, 5.0)
        s1 = np.array([s1a if (j % 2 == 0) else s1b for j in range(q)])
        return s1, float(s2)

    def neg_align(p):
        s1, s2 = unpack(p)
        Z = fmap.scenario_features(sets, s1, s2)
        return -kernel_target_alignment(Z, C)

    res = minimize(neg_align, np.array([1.0, 1.0, 1.0]),
                   method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-4})
    s1, s2 = unpack(res.x)
    return {"s1": s1.tolist(), "s2": s2}, float(-res.fun)


# --------------------------------------------------------------------------
# Classical RBF kernel-mean-embedding control (matched width).
# --------------------------------------------------------------------------

def rbf_rff_features(sets: np.ndarray, D: int, gamma: float,
                     *, seed: int = 0) -> np.ndarray:
    """Classical RBF random-Fourier-feature kernel mean embedding, width D.

    Same set->scenario construction (average the per-event feature map) but with
    a classical RBF kernel — the head-to-head classical control at matched width.
    """
    rng = np.random.default_rng(seed)
    S, N, d = sets.shape
    W = rng.standard_normal((d, D)) * np.sqrt(2.0 * gamma)
    bvec = rng.uniform(0.0, 2.0 * np.pi, size=D)
    feats = np.empty((S, D))
    scale = np.sqrt(2.0 / D)
    for s in range(S):
        proj = sets[s] @ W + bvec                          # (N, D)
        feats[s] = scale * np.cos(proj).mean(0)
    return feats


def _median_gamma(sets: np.ndarray, n_pairs: int = 4000, seed: int = 0) -> float:
    """Median-heuristic RBF bandwidth from pooled events."""
    rng = np.random.default_rng(seed)
    X = sets.reshape(-1, sets.shape[-1])
    idx = rng.choice(X.shape[0], size=min(2 * n_pairs, X.shape[0]), replace=False)
    A = X[idx[:len(idx) // 2]]
    B = X[idx[len(idx) // 2:]]
    d2 = np.sum((A[:len(B)] - B[:len(A)]) ** 2, axis=1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return 1.0 / (2.0 * med)


# --------------------------------------------------------------------------
# Cell.
# --------------------------------------------------------------------------

def _concentration(Z: np.ndarray) -> tuple[float, float]:
    """Off-diagonal spread of the normalised scenario kernel (concentration
    diagnostic). A kernel concentrating to a constant (std -> 0) is the standard
    quantum-kernel failure mode (Thanasilp et al.)."""
    Zc = Z - Z.mean(0, keepdims=True)
    K = Zc @ Zc.T
    dn = np.sqrt(np.clip(np.diag(K), 1e-30, None))
    Kn = K / np.outer(dn, dn)
    iu = np.triu_indices_from(Kn, k=1)
    off = Kn[iu]
    return float(off.mean()), float(off.std())


def run(seed: int = 0, n_qubits: int = 3, depth: int = 2) -> dict:
    t0 = time.time()
    data = sub.load_cache()

    Xtr, Ctr = data["train_X1"], data["train_c"]
    Xte, Cte = data["test_X1"], data["test_c"]
    raw_tr = probes.raw_event_summary(Xtr)
    raw_te = probes.raw_event_summary(Xte)

    fmap = IQPFeatureMap(n_qubits=n_qubits, depth=depth)
    feat_dim = 2 * fmap.dim * fmap.dim

    # --- Fixed quantum kernel (unit angle scales) ---
    s1_fixed = np.ones(fmap.q)
    Ztr_fix = fmap.scenario_features(Xtr, s1_fixed, 1.0)
    Zte_fix = fmap.scenario_features(Xte, s1_fixed, 1.0)
    probe_fix = probes.probe_with_floor(Ztr_fix, Ctr, Zte_fix, Cte,
                                        raw_tr, raw_te, seed=seed)
    cmean, cstd = _concentration(Ztr_fix)

    # --- Trained quantum kernel (angle scales by kernel-target alignment) ---
    scales, align = train_scales(fmap, Xtr, Ctr, seed=seed)
    s1_tr = np.array(scales["s1"])
    Ztr_tr = fmap.scenario_features(Xtr, s1_tr, scales["s2"])
    Zte_tr = fmap.scenario_features(Xte, s1_tr, scales["s2"])
    probe_tr = probes.probe_with_floor(Ztr_tr, Ctr, Zte_tr, Cte,
                                       raw_tr, raw_te, seed=seed)

    # --- Spinor / amplitude-aligned quantum kernel (arXiv:2602.21311) ---
    smap = SpinorFeatureMap(n_qubits=n_qubits)
    s1_sp_fixed = np.ones(smap.q)
    Ztr_sp_fix = smap.scenario_features(Xtr, s1_sp_fixed, 1.0)
    Zte_sp_fix = smap.scenario_features(Xte, s1_sp_fixed, 1.0)
    probe_sp_fix = probes.probe_with_floor(Ztr_sp_fix, Ctr, Zte_sp_fix, Cte,
                                           raw_tr, raw_te, seed=seed)
    sp_scales, sp_align = train_scales(smap, Xtr, Ctr, seed=seed)
    s1_sp = np.array(sp_scales["s1"])
    Ztr_sp = smap.scenario_features(Xtr, s1_sp, sp_scales["s2"])
    Zte_sp = smap.scenario_features(Xte, s1_sp, sp_scales["s2"])
    probe_sp = probes.probe_with_floor(Ztr_sp, Ctr, Zte_sp, Cte,
                                       raw_tr, raw_te, seed=seed)

    # --- Classical RBF kernel-mean-embedding control (matched width) ---
    gamma = _median_gamma(Xtr, seed=seed)
    Ztr_rbf = rbf_rff_features(Xtr, feat_dim, gamma, seed=seed)
    Zte_rbf = rbf_rff_features(Xte, feat_dim, gamma, seed=seed)
    probe_rbf = probes.probe_with_floor(Ztr_rbf, Ctr, Zte_rbf, Cte,
                                        raw_tr, raw_te, seed=seed)

    result = {
        "cell": "quantum_kernel",
        "fm_family": "Quantum kernel (fidelity feature map)",
        "hep_task": ("SMEFT manifold disclosure: Wilson-coefficient recovery "
                     "from a quantum kernel-mean-embedding of one event set"),
        "metric_primary": {
            "name": ("best quantum-kernel c-recovery margin over floor "
                     "(higher=better)"),
            "value": max(probe_tr.margin, probe_sp.margin),
        },
        "metrics": {
            # Native disclosure score (what the matrix's capacity column reads):
            # the best quantum encoding (generic IQP vs amplitude-aligned spinor).
            "probe_r2_heldout": (probe_sp.r2 if probe_sp.margin >= probe_tr.margin
                                 else probe_tr.r2),
            "probe_floor_r2": probe_tr.floor_r2,
            "probe_margin": max(probe_tr.margin, probe_sp.margin),
            "best_encoding": ("spinor_amplitude" if probe_sp.margin >= probe_tr.margin
                              else "iqp"),
            # Amplitude-aligned spinor kernel (arXiv:2602.21311).
            "spinor_fixed_margin": probe_sp_fix.margin,
            "spinor_trained_r2": probe_sp.r2,
            "spinor_trained_floor_r2": probe_sp.floor_r2,
            "spinor_trained_margin": probe_sp.margin,
            "spinor_trained_alignment": sp_align,
            "spinor_r2_per_target": probe_sp.r2_per_target,
            "spinor_trained_scales_s1": sp_scales["s1"],
            "spinor_trained_scales_s2": sp_scales["s2"],
            # Fixed vs trained generic-IQP quantum kernel.
            "qk_fixed_r2": probe_fix.r2,
            "qk_fixed_floor_r2": probe_fix.floor_r2,
            "qk_fixed_margin": probe_fix.margin,
            "qk_trained_r2": probe_tr.r2,
            "qk_trained_floor_r2": probe_tr.floor_r2,
            "qk_trained_margin": probe_tr.margin,
            "qk_trained_alignment": align,
            "qk_trained_scales_s1": scales["s1"],
            "qk_trained_scales_s2": scales["s2"],
            "qk_r2_per_target": probe_tr.r2_per_target,
            # Classical RBF kernel control at matched width.
            "classical_rbf_r2": probe_rbf.r2,
            "classical_rbf_floor_r2": probe_rbf.floor_r2,
            "classical_rbf_margin": probe_rbf.margin,
            "classical_rbf_gamma": gamma,
            # Concentration diagnostic (fixed QK).
            "kernel_offdiag_mean": cmean,
            "kernel_offdiag_std": cstd,
        },
        "ablation_isolated": ("quantum RKHS vs random-feature floor and a "
                              "matched classical RBF kernel; fixed vs "
                              "alignment-trained encoding"),
        "n_params": 3,  # the trained encoding has 3 angle-scale parameters
        "wall_seconds": time.time() - t0,
        "config": {"n_qubits": n_qubits, "depth": depth,
                   "feature_dim": feat_dim, "d_z": probe_tr.d_z,
                   "n_train": int(Xtr.shape[0]), "n_events": int(Xtr.shape[1])},
    }
    OUT.mkdir(exist_ok=True)
    with open(OUT / "cell_quantum_kernel.json", "w") as f:
        json.dump(result, f, indent=2, default=_to_py)
    return result


def _to_py(o):
    """JSON fallback: coerce numpy scalars/arrays to native Python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("cell", "metric_primary", "metrics",
                                        "n_params", "wall_seconds")}, indent=2,
                     default=_to_py))
