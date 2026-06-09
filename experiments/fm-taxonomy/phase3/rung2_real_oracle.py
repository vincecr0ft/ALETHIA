"""
Rung 2: rung-1b on the REAL analytic SMEFT oracle.

Replaces the toy Jacobians of rung 1b with the actual ALETHIA oracle
(modules.analytic_smeft): differential_xs (d sigma/d m_ll) and differential_afb
(A_FB), CT18NNLO PDFs, Lambda = 1 TeV. Two coefficients:

  c_lq^(3)    four-fermion  -- rate grows ~ (m/Lambda)^4 in the tail
  c_phi_q^(3) vertex        -- flat ~2.5% rate shift, DEGENERATE with the
                               luminosity normalisation on the mass spectrum;
                               only A_FB distinguishes it.

Parameters estimated: theta = (c_lq3, c_phiq3, alpha), where alpha is a floated
log-normalisation NUISANCE. Floating alpha is what turns the flat vertex rate
shift into a genuine degeneracy on the mass observable (the paper's
Fisher-eigenvalue collapse).

Design = allocate a unit of measurement to a cell (observable in {mass-rate,
A_FB} x m-bin). Poisson per-bin variance from the SM event yield. Mass cells are
abundant (fine binning); A_FB cells are scarce (few bins, costlier per event) --
so uniform allocation under-covers the only channel that breaks the degeneracy.

Reported: marginal Cramer-Rao sigma on each coefficient (the inverse-Fisher
diagonal of the full 3-parameter model -- this is exactly the paper's
"contraction"), for random vs D-optimal vs A-optimal, in the mass-only pool and
the mass+AFB pool.

python3 rung2_real_oracle.py
"""
import numpy as np
from modules.analytic_smeft import differential_xs, differential_afb

LAM = 1000.0                       # Lambda = 1 TeV (lambda_scale in GeV)
C_TRUE = {"c_lq^(3)": 0.5, "c_phi_q^(3)": 0.5}
PARAMS = ["c_lq^(3)", "c_phi_q^(3)"]          # + alpha nuisance appended below
H = 1e-3                                       # finite-difference step in c

# ---- binning: mass abundant, A_FB scarce ---------------------------------
m_mass = np.linspace(420.0, 2380.0, 40)        # 40 mass-rate bins
m_afb = np.linspace(600.0, 1800.0, 6)          # 6 A_FB bins (scarce channel)
BINW = 50.0                                     # GeV, nominal bin width
LUMI = 3.0e6                                    # pb^-1 scale (sets event yields)
AFB_COST = 8.0                                  # events-per-quantum penalty for A_FB

def xs(c, m):
    return differential_xs(c, m, lambda_scale=LAM)["differential_xs"]

def afb(c, m):
    return differential_afb(c, m, lambda_scale=LAM)["A_FB"]

# ---- oracle-computed sensitivities at the truth (local Fisher design) -----
def dc(fn, m):
    """Central finite-difference d fn / d c_k at C_TRUE, for each coefficient."""
    g = []
    for k in PARAMS:
        cp = dict(C_TRUE); cp[k] += H
        cm = dict(C_TRUE); cm[k] -= H
        g.append((fn(cp, m) - fn(cm, m)) / (2 * H))
    return np.array(g)                          # (nparam, nbin)

R0 = xs(C_TRUE, m_mass)                          # rate at truth (mass bins)
dR = dc(xs, m_mass)                              # d rate / d c   (2, 40)
A0 = afb(C_TRUE, m_afb)                           # A_FB at truth
dA = dc(afb, m_afb)                              # d A_FB / d c   (2, 6)

# SM yields per bin set the Poisson variance
N_mass = LUMI * xs({}, m_mass) * BINW            # expected counts, mass bins
N_afb = LUMI * xs({}, m_afb) * BINW / AFB_COST   # fewer effective events for A_FB

# ---- per-cell Fisher contributions g_cell = J J^T / var (3x3, incl. alpha) -
# mass-rate cell i: observable R; theta-gradient = (dR/dc1, dR/dc2, dR/dalpha=R).
# var of a relative-rate measurement ~ R^2 / N  (Poisson).
def mass_cells():
    cells = []
    for i in range(len(m_mass)):
        J = np.array([dR[0, i], dR[1, i], R0[i]])      # d/dc1, d/dc2, d/dalpha
        var = R0[i] ** 2 / N_mass[i]
        cells.append(J[:, None] * J[None, :] / var)
    return cells

# A_FB cell j: observable A; alpha-independent (ratio). var ~ (1-A^2)/N.
def afb_cells():
    cells = []
    for j in range(len(m_afb)):
        J = np.array([dA[0, j], dA[1, j], 0.0])
        var = (1.0 - A0[j] ** 2) / N_afb[j]
        cells.append(J[:, None] * J[None, :] / var)
    return cells

MASS = mass_cells()
AFB = afb_cells()
PRIOR = np.diag([1e-4, 1e-4, 1e-6])              # weak ridge prior on theta

def marginal_sigma(F):
    """Marginal CR sigma on (c_lq3, c_phiq3) from the full 3-param Fisher."""
    cov = np.linalg.inv(F)
    return np.sqrt(np.diag(cov))[:2]

def design_run(pool, acq, seed, budget=24, n_seed=3):
    rng = np.random.default_rng(seed)
    cells = list(range(len(pool)))
    F = PRIOR.copy()
    chosen = list(rng.choice(cells, n_seed, replace=False))
    for i in chosen:
        F = F + pool[i]
    for _ in range(budget):
        if acq == "random":
            j = int(rng.integers(len(pool)))
        else:
            best, bj = -np.inf, 0
            Finv = np.linalg.inv(F)
            for j in range(len(pool)):
                Fn = F + pool[j]
                if acq == "dopt":
                    val = np.linalg.slogdet(Fn)[1]                 # max log|F|
                elif acq == "aopt":
                    val = -np.trace(np.linalg.inv(Fn)[:2, :2])     # min coeff variance
                if val > best:
                    best, bj = val, j
            j = bj
        F = F + pool[j]
        chosen.append(j)
    n_afb = sum(1 for k in chosen if k >= len(MASS)) if pool is COMBINED else 0
    return marginal_sigma(F), n_afb

COMBINED = MASS + AFB
POOLS = {"mass-only": MASS, "mass+AFB": COMBINED}
ACQS = ["random", "dopt", "aopt"]
SEEDS = range(4000, 4030)

print(f"Real SMEFT oracle (Lambda=1 TeV). Estimating (c_lq3, c_phiq3) with a")
print(f"floated normalisation nuisance. {len(m_mass)} mass cells, {len(m_afb)} A_FB cells "
      f"(A_FB = {100*len(m_afb)/len(COMBINED):.0f}% of combined pool).")
for pool_name, pool in POOLS.items():
    print(f"\n=== pool: {pool_name} ===")
    print(f"{'acq':>8} | {'sigma(c_lq3)':>13} {'sigma(c_phiq3)':>15} {'#AFB':>5}")
    print("-" * 50)
    base = None
    for acq in ACQS:
        sigs, nafb = [], []
        for s in SEEDS:
            sg, na = design_run(pool, acq, s)
            sigs.append(sg); nafb.append(na)
        sigs = np.array(sigs)
        s1, s2 = sigs[:, 0].mean(), sigs[:, 1].mean()
        print(f"{acq:>8} | {s1:13.4e} {s2:15.4e} {np.mean(nafb):5.1f}")
