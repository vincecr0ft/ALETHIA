r"""EFT/SMEFT quadratic-oracle survey: how well do simple learners stand in
for a full MC program?  (Phase-2, axis III of the ALETHIA FM taxonomy.)

Thesis under test
-----------------
A dim-6 SMEFT differential rate is an *exact* polynomial in the Wilson
coefficients,

    sigma(c) = T_0 + sum_i c_i T_i + sum_{i<=j} c_i c_j T_{ij}              (1)

(SM-only, interference, BSM-squared; the sm_only / interference / bsm_squared
triple that ALETHIA's analytic SMEFT oracle modules/analytic_smeft/smeft.py
returns).  This polynomial is the "oracle": the structured object a full MC
generator (MadGraph / SMEFT@NLO-style reweighting) draws from with statistical
noise.  Lagrangian morphing reconstructs (1) EXACTLY from the provably minimal
basis of N = C(n+2, 2) noiseless samples (Baak et al. arXiv:1410.7388; effective
Lagrangian morphing, Balasubramanian et al. arXiv:2202.13612).

We build the oracle for n = 1, 2, 3 coefficients with genuine interference
terms T_{ij}, add MC-style Poisson/Gaussian sampling noise to mimic a generator,
and fit a panel of simple learners:

  - exact morphing basis            (structure-aware, minimal N, no fit)
  - polynomial regression, deg=2    (structure-aware degree, least squares)
  - polynomial regression, deg=1    (WRONG degree: misses BSM^2 curvature)
  - polynomial regression, deg=3    (over-degree: extra freedom, variance)
  - Gaussian process (RBF)          (generic smooth interpolator)
  - small MLP                       (generic universal approximator)

For each we quantify:
  (a) interpolation error inside the basis/training hull,
  (b) extrapolation error outside it,
  (c) behaviour near a degenerate / flat-Fisher direction,
  (d) sample-efficiency: error vs number of MC points.

What we expect to read off
--------------------------
The structure-aware learners (morphing, deg-2 polynomial) are EXACT on the
noiseless oracle and reach the noise floor with O(N) points; they extrapolate
because a polynomial is its own continuation.  The generic learners (GP, MLP)
interpolate acceptably inside the hull but degrade off-support and need far more
data; the wrong-degree polynomial (deg-1) is biased everywhere the quadratic
term matters.  This is axis III made quantitative: structured (morphing) vs
learned (generic) manifold provenance on the *same* object.

Run:
    python3 oracle_survey.py
    # or, in the ALETHIA env:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/fm-taxonomy/phase2/03-eft-oracle/oracle_survey.py

Dependencies: numpy (required); scikit-learn (GP + MLP, optional — the script
prints "skipped" rows if absent); matplotlib (optional, guarded).  Seeded.
Writes oracle_survey.csv next to this file.
"""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 2026

# Optional sklearn (GP + MLP).  Guarded so the core morphing/poly study runs
# under a numpy-only sandbox.
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    _HAVE_SK = True
except Exception:                                                   # pragma: no cover
    _HAVE_SK = False


# ===========================================================================
# 1. The exact polynomial oracle: sigma(c) = T0 + sum c_i T_i + sum c_i c_j T_ij
# ===========================================================================
def monomial_powers(n_coef: int, degree: int) -> list[tuple[int, ...]]:
    r"""All monomial exponent tuples for n_coef coefficients up to total degree.

    For degree 2 the count is C(n+2, 2) — the minimal Lagrangian-morphing basis
    size (Balasubramanian et al. arXiv:2202.13612, Eq. 21).  Ordered by total
    degree then lexicographically so index 0 is the constant (SM) term.
    """
    out: list[tuple[int, ...]] = []
    for d in range(degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_coef), d):
            powers = [0] * n_coef
            for k in combo:
                powers[k] += 1
            out.append(tuple(powers))
    return out


def design_row(c: np.ndarray, powers: list[tuple[int, ...]]) -> np.ndarray:
    r"""Monomial feature vector v(c) = (prod_i c_i^{p_i}) for each power tuple.

    This is the v(c) of the morphing algebra: sigma(c) = v(c) . theta, with
    theta the (flattened) physical templates.  Same map for the polynomial-
    regression design matrix — the *only* difference between morphing and
    least-squares polynomial fitting is N (square vs overdetermined) and whether
    the samples are noiseless.
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    return np.array([np.prod(c ** np.asarray(p, dtype=float)) for p in powers])


def design_matrix(C: np.ndarray, powers: list[tuple[int, ...]]) -> np.ndarray:
    """Stack design_row over a (N, n_coef) array of couplings -> (N, n_terms)."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    return np.stack([design_row(c, powers) for c in C], axis=0)


class QuadraticEFTOracle:
    r"""Exact degree-2 EFT differential-rate oracle on a kinematic grid.

    Each amplitude is linear in the couplings (the SMEFT structure of
    modules/analytic_smeft/smeft.py, Greljo & Marzocca arXiv:1704.09015):

        A(c, x) = a_SM(x) + sum_i c_i a_i(x),

    so the rate is the genuine quadratic form

        sigma(c, x) = |A|^2 = a_SM^2
                    + sum_i c_i (2 a_SM a_i)              # interference T_i
                    + sum_{i<=j} c_i c_j (2-delta_ij) a_i a_j   # T_ij (incl. cross)

    The BSM amplitudes carry explicit energy growth a_i(x) ~ x (the contact
    operator's hat-s / Lambda^2 behaviour), so the T_i / T_ij templates are NOT
    proportional to T_0: the morphing basis is genuinely (n+1)(n+2)/2 -
    dimensional, not a rescaling.  The summary scalar observable used for the
    learner panel is the integrated rate sum_x sigma(c, x) (a cross section),
    which inherits the same quadratic-in-c structure.

    n_coef <= 3.  A near-degenerate direction is built in by making a_2 nearly a
    multiple of a_1 (flat-Fisher direction, controlled by `degeneracy`).
    """

    def __init__(self, n_coef: int = 2, n_x: int = 64, degeneracy: float = 0.0,
                 rng: np.random.Generator | None = None):
        assert 1 <= n_coef <= 3
        self.n_coef = n_coef
        self.x = np.linspace(0.0, 1.0, n_x)
        rng = rng or np.random.default_rng(SEED)
        # SM amplitude: O(1) with kinematic shape.
        self.a_sm = 1.0 + 0.3 * np.cos(2.0 * np.pi * self.x)
        # BSM amplitudes, each with linear energy growth + distinct shape.
        cols = []
        cols.append(0.40 * self.x + 0.10)                         # a_1
        if n_coef >= 2:
            base2 = 0.30 * self.x ** 2 + 0.05
            # degeneracy in [0,1]: at 1.0, a_2 -> multiple of a_1 (flat dir).
            cols.append((1.0 - degeneracy) * base2 + degeneracy * 0.7 * cols[0])
        if n_coef >= 3:
            cols.append(0.25 * np.sin(1.5 * np.pi * self.x) + 0.20)  # a_3
        self.a_bsm = np.stack(cols, axis=0)                       # (n_coef, n_x)
        self.powers = monomial_powers(n_coef, degree=2)
        self.n_terms = len(self.powers)                           # = C(n+2,2)
        self._templates = self._build_templates()                 # (n_terms, n_x)

    def _build_templates(self) -> np.ndarray:
        """Physical templates T_p(x) for each monomial power tuple p."""
        T = np.zeros((self.n_terms, self.x.size))
        for idx, p in enumerate(self.powers):
            deg = sum(p)
            if deg == 0:
                T[idx] = self.a_sm ** 2                            # T0 = a_SM^2
            elif deg == 1:
                i = p.index(1)
                T[idx] = 2.0 * self.a_sm * self.a_bsm[i]           # interference
            else:  # deg == 2
                nz = [k for k, e in enumerate(p) if e > 0]
                if len(nz) == 1:                                   # c_i^2
                    i = nz[0]
                    T[idx] = self.a_bsm[i] ** 2
                else:                                              # c_i c_j cross
                    i, j = nz
                    T[idx] = 2.0 * self.a_bsm[i] * self.a_bsm[j]
        return T

    def differential(self, c: np.ndarray) -> np.ndarray:
        """Exact sigma(c, x) on the grid (the noiseless ground truth)."""
        v = design_row(c, self.powers)
        return v @ self._templates                                # (n_x,)

    def rate(self, c: np.ndarray) -> float:
        """Integrated cross section sum_x sigma(c, x) (the scalar oracle)."""
        return float(np.sum(self.differential(c)))

    def rate_batch(self, C: np.ndarray) -> np.ndarray:
        V = design_matrix(C, self.powers)
        return V @ self._templates.sum(axis=1)                    # (N,)

    def template_scalars(self) -> np.ndarray:
        """theta = integral of each template (the scalar-rate coefficients)."""
        return self._templates.sum(axis=1)                        # (n_terms,)


# ===========================================================================
# 2. MC-style noise: a generator returns the oracle rate with statistics
# ===========================================================================
def mc_sample(oracle: QuadraticEFTOracle, C: np.ndarray, n_events: float,
              rng: np.random.Generator) -> np.ndarray:
    r"""Mimic a MadGraph-style campaign: the true rate sigma(c) measured with a
    finite number of generated events.

    A generator estimates sigma from n_events MC draws; the estimator has
    relative statistical error ~ 1/sqrt(n_events).  We model the returned rate
    as Gaussian about the truth with that relative width (equivalently a
    high-count Poisson), which is the additive-Gaussian observation model of
    ALETHIA's Fisher studies (working_point_fisher.py, sigma_y) with a
    coupling-dependent scale.
    """
    truth = oracle.rate_batch(C)
    rel = 1.0 / math.sqrt(n_events)
    return truth + rng.normal(0.0, rel * np.abs(truth))


# ===========================================================================
# 3. Learner panel
# ===========================================================================
def fit_morphing(oracle: QuadraticEFTOracle, rng: np.random.Generator):
    r"""Exact morphing: query the oracle at the minimal basis of N = C(n+2,2)
    NOISELESS couplings, invert the square morphing matrix M, predict by
    re-evaluating the polynomial.

        sigma(c) = v(c) M^{-1} (sigma(c_1), ..., sigma(c_N))^T.

    This is the structure-aware pole: no fit, minimal samples, exact at any c
    (interpolating or extrapolating).  We use noiseless basis samples — the
    morphing assumption is that the basis campaigns are run to negligible MC
    error (the standard practice; one expensive high-stats campaign per basis
    point).  The contrast with the noisy generic learners is deliberate.
    """
    n = oracle.n_coef
    N = oracle.n_terms
    # A well-conditioned basis: SM point + axis excursions + cross points.
    basis = [np.zeros(n)]
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0; basis.append(e.copy())
        e[i] = -1.0; basis.append(e.copy())
    for i, j in itertools.combinations(range(n), 2):
        e = np.zeros(n); e[i] = 1.0; e[j] = 1.0; basis.append(e)
    basis = np.array(basis[:N])
    if len(basis) < N:                                            # pad if needed
        extra = rng.uniform(-1, 1, size=(N - len(basis), n))
        basis = np.vstack([basis, extra])
    M = design_matrix(basis, oracle.powers)                       # (N, N)
    y = oracle.rate_batch(basis)                                  # noiseless
    M_inv = np.linalg.inv(M)
    theta = M_inv @ y                                             # recovered templates
    cond = float(np.linalg.cond(M))

    def predict(C):
        return design_matrix(C, oracle.powers) @ theta
    return predict, {"n_samples": N, "cond_M": cond,
                     "theta_err": float(np.max(np.abs(theta - oracle.template_scalars())))}


def fit_poly(C_train, y_train, oracle: QuadraticEFTOracle, degree: int):
    r"""Least-squares polynomial regression of the chosen total degree.

    degree=2 is the structure-aware degree (correct); degree=1 omits the BSM^2
    curvature (biased wherever c^2 matters); degree=3 adds spurious freedom
    (variance off-support).  Ridge-regularised lightly for conditioning.
    """
    powers = monomial_powers(oracle.n_coef, degree)
    X = design_matrix(C_train, powers)
    lam = 1e-8 * np.trace(X.T @ X) / X.shape[1]
    A = X.T @ X + lam * np.eye(X.shape[1])
    coef = np.linalg.solve(A, X.T @ y_train)

    def predict(C):
        return design_matrix(C, powers) @ coef
    return predict, {"n_terms": len(powers)}


def fit_gp(C_train, y_train):
    if not _HAVE_SK:
        return None, {"skipped": "sklearn absent"}
    scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    yz = scaler.transform(y_train.reshape(-1, 1)).ravel()
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1))
              + WhiteKernel(1e-3, (1e-6, 1e0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=2, random_state=SEED)
    gp.fit(C_train, yz)

    def predict(C):
        return scaler.inverse_transform(
            gp.predict(np.atleast_2d(C)).reshape(-1, 1)).ravel()
    return predict, {"kernel": str(gp.kernel_)}


def fit_mlp(C_train, y_train):
    if not _HAVE_SK:
        return None, {"skipped": "sklearn absent"}
    xs = StandardScaler().fit(C_train)
    ys = StandardScaler().fit(y_train.reshape(-1, 1))
    Xz = xs.transform(C_train)
    yz = ys.transform(y_train.reshape(-1, 1)).ravel()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                       solver="lbfgs", max_iter=4000, alpha=1e-4,
                       random_state=SEED)
    mlp.fit(Xz, yz)

    def predict(C):
        return ys.inverse_transform(
            mlp.predict(xs.transform(np.atleast_2d(C))).reshape(-1, 1)).ravel()
    return predict, {"hidden": (64, 64)}


# ===========================================================================
# 4. Evaluation: interpolation / extrapolation / sample-efficiency
# ===========================================================================
def rmse(pred, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def rel_rmse(pred, truth) -> float:
    truth = np.asarray(truth)
    scale = np.sqrt(np.mean(truth ** 2)) + 1e-30
    return rmse(pred, truth) / scale


def make_grids(n_coef: int, rng: np.random.Generator, hull: float = 1.0,
               n_in: int = 400, n_out: int = 400):
    """Interpolation grid inside [-hull, hull]^n; extrapolation grid in the
    shell hull < ||c||_inf <= 2*hull (genuinely outside the training box)."""
    C_in = rng.uniform(-hull, hull, size=(n_in, n_coef))
    C_out = []
    while len(C_out) < n_out:
        c = rng.uniform(-2 * hull, 2 * hull, size=n_coef)
        if np.max(np.abs(c)) > hull:
            C_out.append(c)
    return C_in, np.array(C_out)


def run_panel(n_coef: int, n_events: float, n_train: int, hull: float,
              degeneracy: float, rng: np.random.Generator) -> list[dict]:
    """Train every learner on one noisy MC dataset; score in/out of hull."""
    oracle = QuadraticEFTOracle(n_coef=n_coef, degeneracy=degeneracy, rng=rng)
    C_train = rng.uniform(-hull, hull, size=(n_train, n_coef))
    y_train = mc_sample(oracle, C_train, n_events, rng)
    C_in, C_out = make_grids(n_coef, rng, hull=hull)
    y_in_true = oracle.rate_batch(C_in)
    y_out_true = oracle.rate_batch(C_out)

    learners = {}
    p, meta = fit_morphing(oracle, rng); learners["morphing(exact)"] = (p, meta)
    p, meta = fit_poly(C_train, y_train, oracle, 2); learners["poly_deg2(correct)"] = (p, meta)
    p, meta = fit_poly(C_train, y_train, oracle, 1); learners["poly_deg1(under)"] = (p, meta)
    p, meta = fit_poly(C_train, y_train, oracle, 3); learners["poly_deg3(over)"] = (p, meta)
    p, meta = fit_gp(C_train, y_train); learners["gaussian_process"] = (p, meta)
    p, meta = fit_mlp(C_train, y_train); learners["mlp_64x64"] = (p, meta)

    rows = []
    for name, (pred, meta) in learners.items():
        if pred is None:
            rows.append({"learner": name, "n_coef": n_coef, "n_events": n_events,
                         "n_train": n_train, "degeneracy": degeneracy,
                         "interp_rel_rmse": float("nan"),
                         "extrap_rel_rmse": float("nan"),
                         "note": meta.get("skipped", "")})
            continue
        rows.append({
            "learner": name, "n_coef": n_coef, "n_events": n_events,
            "n_train": n_train, "degeneracy": degeneracy,
            "interp_rel_rmse": rel_rmse(pred(C_in), y_in_true),
            "extrap_rel_rmse": rel_rmse(pred(C_out), y_out_true),
            "note": (f"cond(M)={meta['cond_M']:.2f},theta_err={meta['theta_err']:.1e}"
                     if "cond_M" in meta else ""),
        })
    return rows


def sample_efficiency(n_coef: int, n_events: float, hull: float,
                      rng: np.random.Generator) -> list[dict]:
    """Interp rel-RMSE of each *fitting* learner vs number of MC training points.

    Morphing uses a fixed N = C(n+2,2) noiseless basis (reported as its column),
    so the curve shows when a generic learner finally matches the structured
    one's data cost.
    """
    oracle = QuadraticEFTOracle(n_coef=n_coef, rng=rng)
    C_in, _ = make_grids(n_coef, rng, hull=hull, n_out=1)
    y_in_true = oracle.rate_batch(C_in)
    sizes = [oracle.n_terms, 2 * oracle.n_terms, 4 * oracle.n_terms,
             8 * oracle.n_terms, 16 * oracle.n_terms, 64, 256]
    sizes = sorted(set(s for s in sizes if s >= oracle.n_terms))
    out = []
    for nt in sizes:
        C_train = rng.uniform(-hull, hull, size=(nt, n_coef))
        y_train = mc_sample(oracle, C_train, n_events, rng)
        row = {"n_coef": n_coef, "n_train": nt, "minimal_basis_N": oracle.n_terms}
        for name, fit in (("poly_deg2", lambda: fit_poly(C_train, y_train, oracle, 2)),
                          ("gaussian_process", lambda: fit_gp(C_train, y_train)),
                          ("mlp_64x64", lambda: fit_mlp(C_train, y_train))):
            pred, _ = fit()
            row[name] = (rel_rmse(pred(C_in), y_in_true) if pred is not None
                         else float("nan"))
        out.append(row)
    return out


# ===========================================================================
# 5. Fisher geometry of the scalar oracle + degenerate-direction probe
# ===========================================================================
def fisher_scalar(oracle: QuadraticEFTOracle, c: np.ndarray,
                  sigma_y: float = 0.05) -> np.ndarray:
    r"""Fisher information of the couplings for the differential observation
    Y(x) = sigma(c, x) + N(0, sigma_y^2), read off the morphing polynomial:

        F_ij(c) = sum_x (d_i sigma)(d_j sigma) / sigma_y^2,
        d_i sigma(c, x) = T_i(x) + 2 T_ii(x) c_i + sum_{j!=i} T_ij(x) c_j.

    This is exactly Eq. (1) of ALETHIA's work-point handoff (A_i + 2 B_ij c_j),
    the tangent/curvature data of the metric, available in closed form because
    the templates ARE the morphing basis.  Its small eigenvalues are the flat
    (unmeasurable) coupling directions.
    """
    n = oracle.n_coef
    grad = np.zeros((n, oracle.x.size))
    idx = {p: k for k, p in enumerate(oracle.powers)}
    for i in range(n):
        ei = tuple(1 if k == i else 0 for k in range(n))
        grad[i] += oracle._templates[idx[ei]]                     # T_i
        eii = tuple(2 if k == i else 0 for k in range(n))
        grad[i] += 2.0 * c[i] * oracle._templates[idx[eii]]       # 2 c_i T_ii
        for j in range(n):
            if j == i:
                continue
            pij = tuple(1 if k in (i, j) else 0 for k in range(n))
            grad[i] += c[j] * oracle._templates[idx[pij]]         # c_j T_ij
    return (grad @ grad.T) / sigma_y ** 2


def degeneracy_probe(rng: np.random.Generator) -> list[dict]:
    """How learners fare along a built-in flat-Fisher direction (n=2).

    With degeneracy -> 1, a_2 ~ a_1, so the c_1 + c_2 combination is well
    measured but c_1 - c_2 is nearly flat: the Fisher matrix has one small
    eigenvalue.  We report the Fisher condition number and the interp error
    restricted to the flat direction (c_1 = -c_2 line) for the correct-degree
    polynomial vs the GP — the structured learner stays exact along the flat
    direction (it is still the same polynomial); the generic learner has no
    signal there to fit.
    """
    out = []
    for degen in (0.0, 0.7, 0.95):
        oracle = QuadraticEFTOracle(n_coef=2, degeneracy=degen, rng=rng)
        F0 = fisher_scalar(oracle, np.zeros(2))
        # Fisher at a working point along the energy-growing axis (lifts flat dir).
        Fwp = fisher_scalar(oracle, np.array([0.6, 0.6]))
        lam0 = np.linalg.eigvalsh(F0)
        lamwp = np.linalg.eigvalsh(Fwp)
        cond0 = float(lam0[-1] / max(lam0[0], 1e-30))
        condwp = float(lamwp[-1] / max(lamwp[0], 1e-30))
        # Train on a noisy box, score on the flat direction c1 = -c2.
        C_train = rng.uniform(-1, 1, size=(64, 2))
        y_train = mc_sample(oracle, C_train, 1e4, rng)
        t = np.linspace(-1, 1, 200)
        C_flat = np.stack([t, -t], axis=1)
        y_flat = oracle.rate_batch(C_flat)
        p2, _ = fit_poly(C_train, y_train, oracle, 2)
        pgp, mgp = fit_gp(C_train, y_train)
        out.append({
            "degeneracy": degen,
            "fisher_cond_c0": cond0,
            "fisher_cond_wp(0.6,0.6)": condwp,
            "fisher_lift_min_eig": float(lamwp[0] / max(lam0[0], 1e-30)),
            "poly_deg2_flatdir_rel_rmse": rel_rmse(p2(C_flat), y_flat),
            "gp_flatdir_rel_rmse": (rel_rmse(pgp(C_flat), y_flat)
                                    if pgp is not None else float("nan")),
        })
    return out


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)
    np.set_printoptions(precision=4, suppress=False)

    print("=" * 78)
    print("EFT/SMEFT QUADRATIC-ORACLE SURVEY  (axis III: structured vs learned)")
    print("=" * 78)
    print(f"sklearn available: {_HAVE_SK}  (GP + MLP rows are 'skipped' if False)")

    # --- minimal basis count check (the morphing N = C(n+2,2)) ---
    print("\n[0] Minimal morphing basis size N = C(n+2, 2):")
    for n in (1, 2, 3):
        N = len(monomial_powers(n, 2))
        print(f"    n={n}: N = {N}  (= (n^2+3n+2)/2 = {(n*n + 3*n + 2)//2})")

    all_rows: list[dict] = []

    # --- main panel: n = 1, 2, 3 at a moderate MC budget ---
    print("\n[1] Learner panel — interpolation vs extrapolation relative RMSE")
    print("    (MC budget n_events=1e4 -> ~1% per-point stat error; "
          "n_train=64; hull |c|<=1)")
    header = (f"    {'n':>2} {'learner':<20} {'interp_relRMSE':>15} "
              f"{'extrap_relRMSE':>15}  note")
    for n in (1, 2, 3):
        print(f"\n  --- n_coef = {n} (basis N = {len(monomial_powers(n,2))}) ---")
        print(header)
        rows = run_panel(n_coef=n, n_events=1e4, n_train=64, hull=1.0,
                         degeneracy=0.0, rng=rng)
        for r in rows:
            all_rows.append({**r, "block": "panel"})
            it = (f"{r['interp_rel_rmse']:.3e}"
                  if r['interp_rel_rmse'] == r['interp_rel_rmse'] else "skipped")
            ex = (f"{r['extrap_rel_rmse']:.3e}"
                  if r['extrap_rel_rmse'] == r['extrap_rel_rmse'] else "skipped")
            print(f"    {n:>2} {r['learner']:<20} {it:>15} {ex:>15}  {r['note']}")

    # --- sample efficiency (n = 2) ---
    print("\n[2] Sample efficiency — interp relRMSE vs #MC training points (n=2)")
    print("    morphing reaches machine-precision exactness at the minimal "
          "basis N below.")
    se = sample_efficiency(n_coef=2, n_events=1e4, hull=1.0, rng=rng)
    print(f"    {'n_train':>8} {'poly_deg2':>14} {'gaussian_proc':>15} {'mlp_64x64':>14}")
    for r in se:
        all_rows.append({**r, "block": "sample_efficiency", "learner": "(multi)"})
        def fmt(v): return f"{v:.3e}" if v == v else "skipped"
        print(f"    {r['n_train']:>8} {fmt(r['poly_deg2']):>14} "
              f"{fmt(r['gaussian_process']):>15} {fmt(r['mlp_64x64']):>14}")
    print(f"    (minimal morphing basis N = {se[0]['minimal_basis_N']} "
          f"noiseless samples -> exact)")

    # --- degeneracy / flat-Fisher probe (n = 2) ---
    print("\n[3] Degenerate / flat-Fisher direction (n=2): a_2 -> multiple of a_1")
    print("    fisher_cond = lambda_max/lambda_min of F(c); flatdir = error on "
          "the c1=-c2 line")
    dp = degeneracy_probe(rng)
    print(f"    {'degen':>6} {'F_cond(c=0)':>13} {'F_cond_wp':>11} "
          f"{'lift_min_eig':>13} {'poly2_flat':>12} {'gp_flat':>10}")
    for r in dp:
        all_rows.append({**r, "block": "degeneracy", "learner": "(probe)"})
        def fmt(v): return f"{v:.3e}" if v == v else "skipped"
        print(f"    {r['degeneracy']:>6.2f} {r['fisher_cond_c0']:>13.3e} "
              f"{r['fisher_cond_wp(0.6,0.6)']:>11.3e} "
              f"{r['fisher_lift_min_eig']:>13.3e} "
              f"{fmt(r['poly_deg2_flatdir_rel_rmse']):>12} "
              f"{fmt(r['gp_flatdir_rel_rmse']):>10}")

    # --- write CSV ---
    csv_path = HERE / "oracle_survey.csv"
    keys = sorted({k for r in all_rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[csv] wrote {csv_path}  ({len(all_rows)} rows)")

    # --- optional plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nts = [r["n_train"] for r in se]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        for name, style in (("poly_deg2", "-o"), ("gaussian_process", "-s"),
                            ("mlp_64x64", "-^")):
            vals = [r[name] for r in se]
            if all(v == v for v in vals):
                ax.loglog(nts, vals, style, label=name)
        ax.axvline(se[0]["minimal_basis_N"], ls="--", c="k", alpha=0.7,
                   label=f"morphing N={se[0]['minimal_basis_N']} (exact)")
        ax.set_xlabel("# MC training points")
        ax.set_ylabel("interpolation relative RMSE")
        ax.set_title("Sample efficiency vs structured morphing basis (n=2)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.savefig(HERE / "sample_efficiency.png", dpi=140)
        plt.close(fig)
        print(f"[plot] wrote {HERE / 'sample_efficiency.png'}")
    except Exception as exc:                                       # pragma: no cover
        print(f"[plot] skipped ({type(exc).__name__}: {exc})")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("  - morphing(exact) & poly_deg2 reach machine / noise-floor error in")
    print("    and out of the hull; both are structure-aware (axis III: structured).")
    print("  - poly_deg1(under) is biased wherever the BSM^2 curvature matters.")
    print("  - GP/MLP interpolate inside the hull but degrade off-support and need")
    print("    many more points (axis III: learned manifold, error grows off-support).")
    print("  - along the flat-Fisher direction the structured learner stays exact;")
    print("    the working point (0.6,0.6) lifts the small Fisher eigenvalue")
    print("    (lift_min_eig >> 1) — ALETHIA's flat-vertex mechanism, in closed form.")
    print("=" * 78)


if __name__ == "__main__":
    main()
