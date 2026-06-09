"""
Rung 0 of the Phase-3 ramp: the minimal working SMEFT Drell-Yan oracle.

ONE dimension-six Wilson coefficient c, the dilepton-mass-binned rate, and the
exact morphing identity of the paper's Eq. (eq:morphing) restricted to n=1:

    sigma(c, m) = sigma_SM(m) + A(m) c + B(m) c^2          (quadratic in c)

This is the single-coefficient core that the paper's "Polynomial-toy
verification" appendix (App. B.10) compares architectures on, written here as a
self-contained, runnable seed. Everything later in the paper is a ramp on this:
  rung 1: n=2 coefficients (interference B_ij, the vertex/four-fermion degeneracy)
  rung 2: add MC-style Poisson noise (the per-cent oracle/MC agreement of sec:theory)
  rung 3: the simple-learner panel (can a generic learner match the structure?)
  rung 4: the learned manifold (ManifoldInformer) vs this structured oracle
  rung 5: active learning on top (the contraction-vs-MLE story)

numpy only. Deterministic. Run: python3 rung0_single_coefficient.py
"""
import numpy as np

# ---- physics-flavoured single-coefficient oracle -------------------------
# m_ll grid (TeV); a steeply falling Drell-Yan continuum, EFT scale Lambda.
LAMBDA = 2.0
m = np.linspace(0.3, 2.5, 24)                 # dilepton invariant-mass bins
sigma_SM = m ** (-3.0)                         # falling continuum (arb. units)

# A four-fermion operator: interference ~ (m/Lambda)^2, square ~ (m/Lambda)^4,
# so sensitivity grows into the high-mass tail (the paper's (m/Lambda)^4 motif).
a1 = 0.9
b2 = 0.6
A = sigma_SM * a1 * (m / LAMBDA) ** 2           # linear template  (interference)
B = sigma_SM * b2 * (m / LAMBDA) ** 4           # quadratic template (BSM^2)

def sigma(c):
    """Exact differential rate at coefficient c, per m-bin."""
    return sigma_SM + A * c + B * c * c

# ---- [1] exact 3-sample morphing ----------------------------------------
# A degree-2 polynomial in c is pinned by N = C(1+2,2) = 3 samples. Pick a
# minimal basis of three couplings, build the Vandermonde morphing matrix M.
c_basis = np.array([-1.0, 0.0, 1.5])
M = np.vstack([np.ones_like(c_basis), c_basis, c_basis ** 2]).T   # rows (1, c, c^2)
samples = np.array([sigma(c) for c in c_basis])                   # 3 x len(m)

def morph(c_query):
    """Reconstruct sigma(c_query) from the 3-sample basis, no new oracle call."""
    w = np.array([1.0, c_query, c_query ** 2]) @ np.linalg.inv(M)  # morphing weights
    return w @ samples

c_test = np.array([-0.7, -0.25, 0.13, 0.6, 0.95, 1.3, 2.4, 5.0])  # incl. extrapolation
errs = [np.max(np.abs(morph(c) - sigma(c))) for c in c_test]
print(f"[1] exact morphing from 3 samples (cond(M) = {np.linalg.cond(M):.2f})")
print(f"    worst |morphed - true| over c in {{{', '.join(f'{c:g}' for c in c_test)}}}: "
      f"{max(errs):.2e}")

# ---- [2] template recovery: M^{-1} extracts (sigma_SM, A, B) --------------
T_rec = np.linalg.inv(M) @ samples            # rows: T0, T1, T2
tmpl_err = max(np.max(np.abs(T_rec[0] - sigma_SM)),
               np.max(np.abs(T_rec[1] - A)),
               np.max(np.abs(T_rec[2] - B)))
print(f"[2] template recovery max|T_rec - T_true| = {tmpl_err:.2e}  "
      f"(SM / interference / BSM^2 separated exactly)")

# ---- [3] Fisher information for c, and where it lives ---------------------
# Event counts are Poisson: per-bin variance ~ rate, so var(bin) = sigma/N_ref.
# Fisher per bin = (d sigma / d c)^2 / var = N_ref (d sigma/d c)^2 / sigma. With
# d sigma/d c|_SM = A ~ sigma_SM (m/Lambda)^2 and sigma_SM ~ m^-3, the per-bin
# information grows ~ m into the tail -- the paper's high-mass-sensitivity motif.
N_ref = 1.0e4                                   # reference luminosity (events at SM)
dsig_dc = A + 2.0 * B * 0.0                      # d sigma / d c at c=0 (SM)
var = sigma_SM / N_ref                           # Poisson per-bin variance
fisher_bin = dsig_dc ** 2 / var
F0 = np.sum(fisher_bin)
cr = 1.0 / np.sqrt(F0)
hi = m >= np.quantile(m, 2.0 / 3.0)              # top third in mass
frac_hi = np.sum(fisher_bin[hi]) / F0
print(f"[3] Fisher F(c=0) = {F0:.3e}   Cramer-Rao  sigma_c >= {cr:.3e}  (Poisson, N_ref={N_ref:.0e})")
print(f"    high-mass third of bins carries {100 * frac_hi:.0f}% of the information")

print("\nrung-0 seed OK: one coefficient, rate quadratic in c, 3 oracle runs pin it"
      "\nexactly, and the sensitivity concentrates in the high-mass tail.")
