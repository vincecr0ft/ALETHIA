r"""Cross-check the pure-numpy quantum implementations against PennyLane.

What is verified (and why each is the right object):

  1. IQP / spinor FEATURE MAPS -> event-event fidelity matrix |<psi(x)|psi(x')>|^2.
     The fidelity kernel is exactly what every quantum-kernel cell consumes, and it
     is invariant to qubit relabeling and global phase, so matching it against an
     independent PennyLane circuit is a convention-robust proof that our hand-rolled
     statevector implements the intended unitary. A wrong circuit gives a wrong
     kernel.
  2. SWAP-test identity P(ancilla=0) = (1 + Tr(rho sigma))/2. The swap_estimate()
     sampler relies on this; we confirm it with an explicit controlled-SWAP circuit
     in PennyLane (mixed-state device) against our overlap_gram().
  3. Classical-shadow inverse channel rho_hat = (D+1) v v^dag - I is UNBIASED:
     averaging many global-Haar snapshots reconstructs rho, and the split-bank
     overlap Tr(rho_hat_A rho_hat_B) converges to Tr(rho sigma). (PennyLane's
     built-in classical_shadow uses local Pauli measurements, a different
     estimator, so unbiasedness/convergence is the correct check here.)

Run: ../../.venv/bin/python verify_quantum_pennylane.py
Exit code 0 and "ALL CHECKS PASSED" mean the numpy cells are trustworthy.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from cell_quantum_kernel import IQPFeatureMap, SpinorFeatureMap
from cell_swap_overlap import (
    scenario_density_vecs, overlap_gram, shadow_density_vecs,
)

TOL = 1e-9


# --------------------------------------------------------------------------
# PennyLane reference circuits (independent re-implementation).
# --------------------------------------------------------------------------

def pl_iqp_states(X, q, depth, s1, s2):
    """PennyLane statevectors for the IQP/ZZ map, one per event row of X."""
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def circ(x):
        for _ in range(depth):
            for w in range(q):
                qml.Hadamard(w)
            for j in range(q):                       # exp(i a_j Z_j) = RZ(-2 a_j)
                qml.RZ(-2.0 * s1[j] * x[j % 2], wires=j)
            for j in range(q - 1):                   # exp(i a_jk ZZ) = IsingZZ(-2 a_jk)
                a_jk = s2 * (np.pi - x[j % 2]) * (np.pi - x[(j + 1) % 2])
                qml.IsingZZ(-2.0 * a_jk, wires=[j, j + 1])
        return qml.state()

    return np.stack([np.asarray(circ(x)) for x in X])


def pl_spinor_states(X, q, s1, s2):
    """PennyLane statevectors for the spinor/Bloch map, one per event row of X."""
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def circ(x):
        theta = np.arccos(np.clip(x[1], -1.0, 1.0))
        logm = x[0]
        en = 1.0 / (1.0 + np.exp(-logm))
        for j in range(q):
            if j % 2 == 0:
                alpha, beta = s1[j] * theta, s2 * logm
            else:
                alpha, beta = s1[j] * np.pi * en, s2 * theta
            qml.RY(alpha, wires=j)                    # RZ(beta) RY(alpha)|0> = our qubit
            qml.RZ(beta, wires=j)
        for j in range(q - 1):
            qml.CNOT(wires=[j, j + 1])
        return qml.state()

    return np.stack([np.asarray(circ(x)) for x in X])


def fidelity_matrix(states):
    """G[n,m] = |<psi_n|psi_m>|^2 for a stack of statevectors."""
    G = states.conj() @ states.T
    return (G * G.conj()).real


# --------------------------------------------------------------------------
# Checks.
# --------------------------------------------------------------------------

def _events(rng, n=6):
    """A few physical (log m, cos theta*) events spanning the kinematic range."""
    logm = rng.uniform(-1.0, 0.8, size=n)
    cost = rng.uniform(-0.95, 0.95, size=n)
    return np.stack([logm, cost], axis=1)


def check_feature_maps():
    rng = np.random.default_rng(0)
    X = _events(rng, 6)
    ok = True
    for q in (2, 3, 4):
        # --- IQP ---
        depth = 2
        s1 = rng.uniform(0.3, 2.0, size=q)
        s2 = float(rng.uniform(0.3, 2.0))
        imap = IQPFeatureMap(n_qubits=q, depth=depth)
        G_ours = fidelity_matrix(imap.states(X, s1, s2))
        G_pl = fidelity_matrix(pl_iqp_states(X, q, depth, s1, s2))
        d_iqp = float(np.max(np.abs(G_ours - G_pl)))
        # --- Spinor ---
        s1s = rng.uniform(0.3, 2.0, size=q)
        s2s = float(rng.uniform(0.3, 2.0))
        smap = SpinorFeatureMap(n_qubits=q)
        G_ours_s = fidelity_matrix(smap.states(X, s1s, s2s))
        G_pl_s = fidelity_matrix(pl_spinor_states(X, q, s1s, s2s))
        d_spin = float(np.max(np.abs(G_ours_s - G_pl_s)))
        passed = d_iqp < TOL and d_spin < TOL
        ok = ok and passed
        print(f"  q={q}: IQP fidelity-matrix max|diff|={d_iqp:.2e}  "
              f"spinor max|diff|={d_spin:.2e}  {'OK' if passed else 'FAIL'}")
    return ok


def check_swap_identity():
    """P(ancilla=0) of a controlled-SWAP test == (1 + Tr(rho sigma))/2 == our Gram."""
    rng = np.random.default_rng(1)
    X = _events(rng, 12)
    q = 3
    smap = SpinorFeatureMap(n_qubits=q)
    s1 = np.ones(q)
    # Two scenarios = two event halves -> two mean density matrices.
    V = scenario_density_vecs(smap, X[None, :6], s1, 1.0)[0].reshape(2 ** q, 2 ** q)
    W = scenario_density_vecs(smap, X[None, 6:], s1, 1.0)[0].reshape(2 ** q, 2 ** q)
    tr_ours = float((V * W.T).real.sum())            # Tr(rho sigma)

    anc = 0
    reg_a = list(range(1, 1 + q))
    reg_b = list(range(1 + q, 1 + 2 * q))
    dev = qml.device("default.mixed", wires=1 + 2 * q)

    @qml.qnode(dev)
    def swaptest():
        qml.QubitDensityMatrix(V, wires=reg_a)
        qml.QubitDensityMatrix(W, wires=reg_b)
        qml.Hadamard(anc)
        for a, b in zip(reg_a, reg_b):
            qml.CSWAP(wires=[anc, a, b])
        qml.Hadamard(anc)
        return qml.probs(wires=anc)

    p0 = float(swaptest()[0])
    pred = (1.0 + tr_ours) / 2.0
    d = abs(p0 - pred)
    passed = d < 1e-8
    print(f"  SWAP test: P(anc=0)={p0:.10f}  (1+Tr(rho sigma))/2={pred:.10f}  "
          f"|diff|={d:.2e}  {'OK' if passed else 'FAIL'}")
    return passed


def check_shadow_unbiased():
    """Global-Haar shadow rho_hat reconstructs rho; split-bank overlap -> Tr(rho sigma)."""
    rng = np.random.default_rng(2)
    X = _events(rng, 40)
    q = 2
    smap = SpinorFeatureMap(n_qubits=q)
    D = smap.dim
    s1 = np.ones(q)
    sets = X[None]                                   # one scenario, 40 events
    # CONVENTION: scenario_density_vecs stores conj(rho) = rho^T (it uses
    # psi.conj().T @ psi). That is overlap-invariant — Tr(conj rho . conj sigma)
    # = Tr(rho sigma) — so every Gram in the cells is correct. But the shadow
    # estimator reconstructs the TRUE rho = mean_n |psi_n><psi_n|, so the
    # Frobenius reference here must be the true rho, not the stored conjugate.
    psi = smap.states(X, s1, 1.0)
    rho = np.einsum("ni,nj->ij", psi, psi.conj()) / len(X)        # true rho
    sigma = scenario_density_vecs(smap, X[None, :20], s1, 1.0)[0].reshape(D, D)
    tr_true = float((scenario_density_vecs(smap, sets, s1, 1.0)[0].reshape(D, D)
                     * sigma.T).real.sum())          # Tr(rho sigma), conj-invariant

    # Reconstruction: average many snapshots of the SAME scenario.
    M = 200000
    VA, VB = shadow_density_vecs(smap, sets, s1, 1.0, M, rng)
    rho_hat = (0.5 * (VA[0] + VB[0])).reshape(D, D)
    frob = float(np.linalg.norm(rho_hat - rho))

    # Overlap convergence: split-bank estimate over many scenarios of rho vs sigma.
    sigsets = X[None, :20]
    n_rep, acc = 40, []
    for r in range(n_rep):
        rr = np.random.default_rng(100 + r)
        A, _ = shadow_density_vecs(smap, sets, s1, 1.0, 4000, rr)
        _, B = shadow_density_vecs(smap, sigsets, s1, 1.0, 4000, rr)
        acc.append(float((A[0] * B[0].conj()).real.sum()))
    ov_mean, ov_err = float(np.mean(acc)), float(np.std(acc) / np.sqrt(n_rep))
    d_ov = abs(ov_mean - tr_true)
    # Both are Monte-Carlo estimators: reconstruction Frobenius scales ~ sqrt(D/M)
    # (global-shadow variance), so the threshold is statistical, not machine-eps.
    frob_tol = 5.0 * np.sqrt(D / M)
    passed = frob < frob_tol and d_ov < 4 * ov_err + 5e-3
    print(f"  shadow reconstruction ||rho_hat-rho||_F={frob:.2e} (M={M}, tol={frob_tol:.2e})")
    print(f"  shadow overlap est={ov_mean:.4f}+/-{ov_err:.4f}  true Tr(rho sigma)={tr_true:.4f}"
          f"  |diff|={d_ov:.2e}  {'OK' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("PennyLane", qml.__version__)
    print("\n[1] feature maps vs PennyLane (fidelity kernel):")
    a = check_feature_maps()
    print("\n[2] SWAP-test identity:")
    b = check_swap_identity()
    print("\n[3] classical-shadow unbiasedness:")
    c = check_shadow_unbiased()
    print("\n" + ("ALL CHECKS PASSED" if (a and b and c) else "*** SOME CHECKS FAILED ***"))
    raise SystemExit(0 if (a and b and c) else 1)
