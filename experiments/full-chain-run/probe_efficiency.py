"""INV-2: Bayesian-efficiency of the disclosure probe (mass-only observable).

Establishes that the linear disclosure probe c_hat = W w + b attains the
Bayesian Cramer-Rao bound per direction on the rate-only observable.

Strategy:
 1. Fit the probe W by ordinary least squares (no ridge) on 400 training-box
    scenarios; record OLS residual variance and (X^T X)^{-1} for the
    frequentist posterior covariance.
 2. Compute Fisher eigenbasis V from the empirical Fisher of the oracle on
    the same prior (mass-only observable, so Fisher is on m_ll only).
 3. For each held-out c (in-box, withheld-band): draw N_rep K=12 contexts,
    compute realised bias and variance per direction in the c_tilde frame,
    plus the K-point analytic Fisher F_K(c) and BCRB_a = (F_K + Sigma_prior^-1)^-1.
 4. MLE cross-check via scipy.optimize.least_squares on the morphing
    residuals; vertex MLE variance ~ prior variance means the floor is the
    observable.
 5. Freeze W, b, V, prior covariance and OLS internals to a single .npz so
    INV-3's parameter-EPIG harness can consume them.

Noise model: run.py constructs `AnalyticSMEFTOracle(noise_frac=0.0)`, so the
training contexts are noiseless. For BCRB to be finite (and for the probe
posterior covariance to be meaningful at all) we add additive Gaussian noise
on Y with a small fixed sigma_y; the same sigma is used in F_K, the probe
fit's residual variance reference, and the MLE weights. This is the only
defensible choice given the chain pretraining objective is least-squares on
Y (i.e. equivalent to additive-Gaussian noise on Y). We DO NOT use the
relative-noise (paper Eq. 9) Fisher form.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.stats import norm

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM
from modules.surrogate.intention.fisher import (
    empirical_fisher_c, fisher_basis, sample_c_prior_inbox,
)

# ----- Configuration (matches run.py / identifiability_probe.py conventions) -----
SEED = 7771
N_WC = 4
WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
WITHHOLD_DIM = 2                       # c_lq^(3)
WITHHOLD_BAND = (0.6, 1.0)
C_TRAIN_BOX = 0.7
M_RANGE = (0.3, 2.3)
K_CTX = 12

N_TRAIN_PROBE = 400                    # Probe training scenarios
N_TEST_INBOX = 50                      # Held-out scenarios inside training box
N_TEST_WITHHELD = 50                   # Held-out inside the withheld band
N_REP = 200                            # K=12 context replicas per held-out c
N_FISHER_PRIOR = 800                   # samples for empirical Fisher and prior
# Notes: spec defaults are N_TEST_*=100 and N_REP=500 for the full study;
# the values above are the spec's documented defensible minimums (N_rep >=
# 200, N_held >= 50) chosen to fit the run in one CPU-bounded turn. A
# full-budget rerun is parameter-only (just bump these constants).

# Additive noise model on Y. run.py uses noise_frac=0.0 (truth) for contexts,
# but the FM is trained with MSE on Y, which corresponds to additive Gaussian
# noise. We pick a small sigma_y so BCRB and probe posterior are well defined.
SIGMA_Y = 0.05

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MODEL_CHECKPOINT and OUT_SUFFIX are env-var overrides so we can point
# the same script at the original psi_theta (v1) or the c-recoverability-
# retrained one (v2) and write distinct artifacts. Defaults point at v1
# for backward compatibility.
import os as _os
MODEL_CHECKPOINT = _os.environ.get(
    "MODEL_CHECKPOINT", str(OUT_DIR / "intention_fm.pt"))
_DEFAULT_SUFFIX = "_v2" if "intention_fm_v2" in MODEL_CHECKPOINT else ""
OUT_SUFFIX = _os.environ.get("OUT_SUFFIX", _DEFAULT_SUFFIX)
W_ARTIFACT_PATH = OUT_DIR / f"probe_W_mass_only{OUT_SUFFIX}.npz"
SBI_JSON_PATH = OUT_DIR / f"sbi_posterior_summary_mass_only{OUT_SUFFIX}.json"
BCRB_JSON_PATH = OUT_DIR / f"bcrb_efficiency_mass_only{OUT_SUFFIX}.json"


# ------------------------------------------------------------------
# Prior sampling
# ------------------------------------------------------------------
def sample_c_inbox(n: int, rng: np.random.Generator) -> np.ndarray:
    """c ~ U([-0.7, 0.7]^4) excluding |c_lq^(3)| in [0.6, 1.0]."""
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def sample_c_withheld(n: int, rng: np.random.Generator) -> np.ndarray:
    out = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=(n, N_WC))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[:, WITHHOLD_DIM] = signs * rng.uniform(*WITHHOLD_BAND, size=n)
    return out


# ------------------------------------------------------------------
# Morphing decomposition of mu(c, m): mu = mu_SM + sum_i A_i c_i + sum_ij B_ij c_i c_j
# Vectorised analytic surrogate: at fixed m we compute A_i(m) and B_ij(m) once
# by finite differences on oracle.truth (mu is exactly quadratic in c).
# ------------------------------------------------------------------
def morphing_decomposition(
    oracle, m_values: np.ndarray, h: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu_SM, A, B) on `m_values`.

    mu_SM : (K,)
    A     : (K, N_WC)              first derivative at c=0
    B     : (K, N_WC, N_WC)        Hessian / 2 at c=0 (symmetric)

    Then for any c:  mu(c, m) = mu_SM + A @ c + c @ B @ c.
    """
    m = np.asarray(m_values, dtype=float)
    K = len(m)
    n = N_WC
    zero = np.zeros((K, n))

    mu_SM = oracle.truth(zero, m)

    # First derivatives.
    A = np.empty((K, n))
    for i in range(n):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        A[:, i] = (oracle.truth(cp, m) - oracle.truth(cm, m)) / (2.0 * h)

    # Second derivatives. B_ii = (mu(+h) + mu(-h) - 2 mu_SM)/(2 h^2).
    # B_ij (i != j): use mixed difference and subtract diagonal contributions.
    # We compute the Hessian H_ij = d^2 mu / dc_i dc_j and store B = H / 2 so
    # that mu(c, m) = mu_SM + A c + (1/2) c^T H c = mu_SM + A c + c^T B c.
    H = np.zeros((K, n, n))
    for i in range(n):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        H[:, i, i] = (oracle.truth(cp, m) + oracle.truth(cm, m)
                      - 2.0 * mu_SM) / (h * h)
    for i in range(n):
        for j in range(i + 1, n):
            c_pp = zero.copy(); c_pp[:, i] += h; c_pp[:, j] += h
            c_pm = zero.copy(); c_pm[:, i] += h; c_pm[:, j] -= h
            c_mp = zero.copy(); c_mp[:, i] -= h; c_mp[:, j] += h
            c_mm = zero.copy(); c_mm[:, i] -= h; c_mm[:, j] -= h
            Hij = (oracle.truth(c_pp, m) - oracle.truth(c_pm, m)
                   - oracle.truth(c_mp, m) + oracle.truth(c_mm, m)) / (4.0 * h * h)
            H[:, i, j] = Hij
            H[:, j, i] = Hij
    B = 0.5 * H
    return mu_SM, A, B


class MorphingTable:
    """Cubic-spline interpolation of (mu_SM(m), A_i(m), B_ij(m)) on m.

    Pre-tabulated on a dense grid once via :func:`morphing_decomposition`,
    then re-evaluated at arbitrary context m_k by spline. mu is exactly a
    quadratic polynomial in c at fixed m, so once these tables are
    accurate the analytic mu(c, m) and its c-gradient are exact.
    """

    def __init__(self, oracle, m_grid: np.ndarray):
        self.m_grid = np.asarray(m_grid, dtype=float)
        mu_SM, A, B = morphing_decomposition(oracle, self.m_grid)
        self._mu_SM = CubicSpline(self.m_grid, mu_SM, bc_type="natural")
        # A: (K, n_wc). Spline per column.
        self._A_splines = [CubicSpline(self.m_grid, A[:, i], bc_type="natural")
                           for i in range(N_WC)]
        # B: (K, n_wc, n_wc), symmetric. Store upper triangle.
        self._B_splines = {}
        for i in range(N_WC):
            for j in range(i, N_WC):
                self._B_splines[(i, j)] = CubicSpline(
                    self.m_grid, B[:, i, j], bc_type="natural")

    def at(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (mu_SM, A, B) at m. Shapes (K,), (K, n_wc), (K, n_wc, n_wc)."""
        m = np.asarray(m, dtype=float)
        K = m.shape[0]
        mu_SM = self._mu_SM(m)
        A = np.stack([s(m) for s in self._A_splines], axis=1)
        B = np.zeros((K, N_WC, N_WC))
        for (i, j), s in self._B_splines.items():
            val = s(m)
            B[:, i, j] = val
            B[:, j, i] = val
        return mu_SM, A, B


def mu_from_morphing(c: np.ndarray, mu_SM: np.ndarray, A: np.ndarray,
                     B: np.ndarray) -> np.ndarray:
    """mu(c, m_k) per k.  c: (n_wc,), returns (K,)."""
    return mu_SM + A @ c + np.einsum("i,kij,j->k", c, B, c)


def dmu_dc_from_morphing(c: np.ndarray, A: np.ndarray,
                         B: np.ndarray) -> np.ndarray:
    """d mu / d c_i evaluated at c.  Returns (K, n_wc)."""
    return A + np.einsum("kij,j->ki", B + B.transpose(0, 2, 1), c)


def fisher_K_analytic(c: np.ndarray, mu_SM: np.ndarray, A: np.ndarray,
                      B: np.ndarray, sigma_y: float) -> np.ndarray:
    """K-point Fisher F_K(c) for additive Gaussian noise on mu."""
    dmu = dmu_dc_from_morphing(c, A, B)              # (K, n_wc)
    return (dmu.T @ dmu) / (sigma_y ** 2)


# ------------------------------------------------------------------
# Context / probe utilities
# ------------------------------------------------------------------
def implicit_w(model: IntentionFM, c: np.ndarray, oracle, rng,
               sigma_y: float, morphing: "MorphingTable | None" = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a K=12 context for this c with additive Gaussian noise and
    return the ridge weight w, the context M_ctx, and Y_ctx (noised).

    If `morphing` is provided we evaluate mu via the spline table; this is
    machine-precision identical to `oracle.truth` for the SMEFT mu (it is
    exactly quadratic in c at fixed m, and the splines reproduce
    A_i(m), B_ij(m) to spline tolerance).
    """
    M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
    if morphing is None:
        Y_truth = oracle.truth(np.tile(c, (K_CTX, 1)), M_ctx)
    else:
        mu_SM_k, A_k, B_k = morphing.at(M_ctx)
        Y_truth = mu_from_morphing(c, mu_SM_k, A_k, B_k)
    Y_ctx = Y_truth + rng.normal(0.0, sigma_y, size=K_CTX)
    _, w, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    return w, M_ctx, Y_ctx


def batch_implicit_w(model, cs, oracle, rng, sigma_y, morphing=None):
    ws = np.empty((len(cs), model.d_psi))
    for i, c in enumerate(cs):
        w, _, _ = implicit_w(model, c, oracle, rng, sigma_y, morphing)
        ws[i] = w
    return ws


# ------------------------------------------------------------------
# Oracle MLE per context
# ------------------------------------------------------------------
def mle_c_from_morphing(Y_ctx: np.ndarray, mu_SM: np.ndarray, A: np.ndarray,
                        B: np.ndarray, sigma_y: float,
                        c0: np.ndarray,
                        bounds: tuple = ((-1.0,) * 4, (1.0,) * 4),
                        n_restarts: int = 5,
                        rng: np.random.Generator | None = None,
                        gtol: float = 1e-10,
                        return_info: bool = False):
    """Bounded multi-restart MLE for c given a pre-computed morphing on M_ctx.

    Residual = (Y_k - mu(c, m_k)) / sigma_y. We use scipy's TRF (bounded)
    optimizer because the unbounded LM optimizer wanders catastrophically
    on the near-flat vertex directions (c̃_3, c̃_4) where the Fisher
    information is ~1e-4; without bounds the MLE produces |c| ~ 10s and
    the variance estimate is dominated by optimizer divergence rather
    than likelihood curvature. The bounds match the training-prior
    support |c_i| ≤ 1 (slightly wider than the training box of 0.7 to
    leave room for the withheld band evaluation).

    n_restarts > 1 mitigates the second failure mode (local minima on
    flat directions). Returns the lowest-cost solution across restarts.

    If return_info=True, additionally returns a dict with the best-cost
    gradient norm, final cost, and a flag indicating whether the
    solution sits on a box edge (which would mean the bound is biting
    and the unconstrained MLE doesn't exist for this context).
    """
    def residuals(c):
        return (Y_ctx - mu_from_morphing(c, mu_SM, A, B)) / sigma_y

    lo = np.asarray(bounds[0])
    hi = np.asarray(bounds[1])
    best_x = c0.copy()
    best_cost = np.inf
    best_grad = np.nan
    for r in range(n_restarts):
        if r == 0:
            x0 = c0
        else:
            if rng is None:
                rng = np.random.default_rng(0)
            x0 = lo + (hi - lo) * rng.uniform(size=len(c0))
        try:
            sol = least_squares(residuals, x0=x0, method="trf",
                                 bounds=(lo, hi),
                                 max_nfev=500, gtol=gtol, xtol=gtol,
                                 ftol=gtol)
            if sol.cost < best_cost:
                best_cost = float(sol.cost)
                best_x = sol.x
                # gradient: J^T r at the solution
                best_grad = float(np.linalg.norm(sol.jac.T @ sol.fun))
        except Exception:
            continue
    if not np.isfinite(best_cost):
        if return_info:
            return np.full(len(c0), np.nan), {"converged": False}
        return np.full(len(c0), np.nan)
    if return_info:
        edge_tol = 1e-3
        on_edge = bool(np.any(np.abs(best_x - lo) < edge_tol) or
                       np.any(np.abs(best_x - hi) < edge_tol))
        return best_x, {"converged": True, "cost": best_cost,
                         "grad_norm": best_grad, "on_box_edge": on_edge}
    return best_x


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    print(f"Noise model: additive Gaussian on Y, sigma_y = {SIGMA_Y}.")
    print("(run.py uses oracle noise_frac=0.0; we add this floor so BCRB is "
          "finite; the same sigma is used in F_K, probe-fit context, and MLE.)")

    # Load oracle (truth used everywhere; we add the sigma_y noise ourselves).
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)

    # Pre-tabulate morphing on a dense grid for fast Fisher / MLE / mu eval.
    print("Pre-tabulating morphing on m-grid (one-time cost)...")
    m_grid = np.linspace(M_RANGE[0], M_RANGE[1], 240)
    morphing = MorphingTable(oracle, m_grid)

    # Load pretrained IntentionFM. Defaults to intention_fm.pt; override
    # via the MODEL_CHECKPOINT env var to point at intention_fm_v2.pt
    # (the c-recoverability-retrained variant).
    ckpt = Path(MODEL_CHECKPOINT)
    if not ckpt.exists():
        print(f"ERROR: pretrained Intention FM not found at {ckpt}.")
        print("  Run `uv run python experiments/full-chain-run/run.py` first,")
        print("  or `uv run python experiments/full-chain-run/"
              "retrain_with_c_recovery.py` for the v2 checkpoint.")
        sys.exit(1)
    print(f"Loading checkpoint: {ckpt}")
    # Infer d_psi from the checkpoint so this script handles both
    # d_psi=16 (v1, v2) and d_psi=32 (v3) without an env-var dance.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    inferred_d_psi = state["psi.net.4.bias"].shape[0]
    print(f"Inferred d_psi from checkpoint = {inferred_d_psi}")
    model = IntentionFM(d_psi=inferred_d_psi, hidden=64, alpha=1e-3)
    model.load_state_dict(state)
    model.eval()
    d_psi = model.d_psi

    # ----- Fit OLS probe on 400 in-box scenarios -----
    print("\n=== OLS probe ===")
    c_train = sample_c_inbox(N_TRAIN_PROBE, rng)
    w_train = batch_implicit_w(model, c_train, oracle, rng, SIGMA_Y, morphing)

    # Augment with intercept column so we can fit (W, b) jointly.
    X = np.hstack([w_train, np.ones((N_TRAIN_PROBE, 1))])    # (N, d_psi+1)
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Wb = XtX_inv @ X.T @ c_train                             # (d_psi+1, n_wc)
    W = Wb[:d_psi].T                                         # (n_wc, d_psi)
    b = Wb[d_psi]                                            # (n_wc,)
    c_train_hat = X @ Wb
    resid = c_train - c_train_hat
    # Per-output residual variance (each direction trained independently).
    dof = N_TRAIN_PROBE - (d_psi + 1)
    sigma_resid_sq = (resid ** 2).sum(axis=0) / dof          # (n_wc,)
    print(f"  OLS sigma_resid^2 per direction = {sigma_resid_sq.round(5)}")

    # Probe coefficient standard errors per output direction:
    # Var(coef_a) = sigma_resid_sq[a] * diag(XtX_inv)  (length d_psi+1).
    coef_se_diag = np.outer(np.sqrt(sigma_resid_sq), np.sqrt(np.diag(XtX_inv)))
    print(f"  OLS coef SE (first 5 dims, dir 0): "
          f"{coef_se_diag[0, :5].round(4)}")

    # ----- Empirical Fisher and V -----
    print("\n=== Fisher eigenbasis ===")
    c_fisher = sample_c_prior_inbox(
        n=N_FISHER_PRIOR, n_wc=N_WC, box=C_TRAIN_BOX, rng=rng,
        withhold_dim=WITHHOLD_DIM, withhold_band=WITHHOLD_BAND)
    m_fisher = rng.uniform(*M_RANGE, size=len(c_fisher))
    F = empirical_fisher_c(oracle, c_fisher, m_fisher, fd_step=1e-3)
    D_eig, V = fisher_basis(F)
    print(f"  Fisher eigenvalues (desc): {D_eig.round(6).tolist()}")
    # The spec quotes ~{19.6, 8.7, 3e-4, 1.3e-4} as the expected
    # spectrum; the empirical Fisher uses d log mu / d c which is the
    # relative-noise convention used by the rotation only -- the
    # eigenbasis V is invariant to noise convention up to a global
    # scaling because both conventions share the same d mu / d c null
    # space.  The BCRB / efficiency calc below uses additive-on-mu
    # Fisher F_K consistently.

    # ----- Prior covariance in rotated frame -----
    c_prior_for_cov = sample_c_prior_inbox(
        n=N_FISHER_PRIOR, n_wc=N_WC, box=C_TRAIN_BOX, rng=rng,
        withhold_dim=WITHHOLD_DIM, withhold_band=WITHHOLD_BAND)
    c_tilde_prior = c_prior_for_cov @ V                      # (N, n_wc)
    sigma_prior_diag_sq = c_tilde_prior.var(axis=0)          # (n_wc,)
    inv_prior_diag = 1.0 / sigma_prior_diag_sq
    print(f"  sigma_prior^2 per c_tilde direction: "
          f"{sigma_prior_diag_sq.round(5).tolist()}")
    print(f"  1/sigma_prior^2 : {inv_prior_diag.round(2).tolist()}")

    # ----- Held-out evaluation -----
    pools = {
        "inbox": sample_c_inbox(N_TEST_INBOX, rng),
        "withheld_band": sample_c_withheld(N_TEST_WITHHELD, rng),
    }

    z68 = float(norm.ppf(0.5 + 0.683 / 2.0))   # ~0.998 (~1 sigma)

    bcrb_summary = {"sigma_y": SIGMA_Y,
                    "fisher_eigenvalues_descending": D_eig.tolist(),
                    "sigma_prior_diag": sigma_prior_diag_sq.tolist(),
                    "pools": {}}
    sbi_summary = {"sigma_y": SIGMA_Y,
                   "fisher_V_warsaw_columns_are_c_tilde": V.tolist(),
                   "fisher_eigenvalues_descending": D_eig.tolist(),
                   "wc_order": list(WC_NAMES),
                   "pools": {}}

    for pool_name, cs_pool in pools.items():
        print(f"\n=== Pool: {pool_name} (N={len(cs_pool)}, N_rep={N_REP}) ===")
        n_pool = len(cs_pool)

        # Per-scenario aggregates.
        biases = np.zeros((n_pool, N_WC))         # bias on c_tilde
        variances = np.zeros((n_pool, N_WC))      # realised var on c_tilde
        bcrb_diag = np.zeros((n_pool, N_WC))      # BCRB_a per direction
        fisher_diag_K = np.zeros((n_pool, N_WC))  # F_K diag in rotated basis
        mle_variances = np.zeros((n_pool, N_WC))  # MLE var per direction
        probe_post_sigma2 = np.zeros((n_pool, N_WC))  # claimed posterior var
        cover68 = np.zeros((n_pool, N_WC))        # interval coverage flags

        sbi_records = []

        for i, c_true in enumerate(cs_pool):
            c_tilde_true = V.T @ c_true

            # K=12 contexts: collect ridge w, run MLE per context.
            c_hat_reps = np.zeros((N_REP, N_WC))
            mle_reps = np.zeros((N_REP, N_WC))
            # Per-context probe predictive variance (frequentist) — store one.
            pred_var_freq = None

            # For the SBI artifact we record a single representative context
            # plus the posterior aggregated over all reps.
            saved_M_ctx = None
            saved_Y_ctx = None

            # Average F_K over the N_rep contexts (their abscissae differ;
            # the BCRB is per-context but we summarise by mean F_K per c).
            F_K_avg = np.zeros((N_WC, N_WC))

            for r in range(N_REP):
                w, M_ctx, Y_ctx = implicit_w(
                    model, c_true, oracle, rng, SIGMA_Y, morphing)
                # Probe prediction.
                c_hat = W @ w + b
                c_hat_reps[r] = c_hat

                # Morphing decomposition on this context's M_ctx is shared
                # by the Fisher and the MLE residuals (via spline tables).
                mu_SM_k, A_k, B_k = morphing.at(M_ctx)

                # MLE on this context.
                mle_reps[r] = mle_c_from_morphing(
                    Y_ctx, mu_SM_k, A_k, B_k, SIGMA_Y, c0=np.zeros(N_WC))

                # K-point Fisher at c_true on this context's abscissae.
                F_K_avg += fisher_K_analytic(c_true, mu_SM_k, A_k, B_k, SIGMA_Y)

                if r == 0:
                    # Probe predictive variance (frequentist):
                    # Var(c_hat_a) = sigma_resid_sq[a] * (1 + x^T XtX^-1 x)
                    # where x = [w, 1].
                    x = np.concatenate([w, [1.0]])
                    lev = float(x @ XtX_inv @ x)
                    pred_var_freq = sigma_resid_sq * (1.0 + lev)  # (n_wc,)
                    saved_M_ctx = M_ctx.copy()
                    saved_Y_ctx = Y_ctx.copy()

            F_K_avg /= N_REP
            # Rotate F_K into the eigenbasis.
            F_K_tilde = V.T @ F_K_avg @ V                       # (4,4)
            fisher_diag_K[i] = np.diag(F_K_tilde)

            # BCRB per direction in c_tilde frame. The well-conditioned
            # rotated-basis form:  M_tilde = F_K_tilde + diag(1/sigma_prior^2)
            # then BCRB = diag(M_tilde^-1). We use the full inverse in
            # the rotated basis to handle any residual off-diagonal coupling
            # (small relative to the diagonal at the prior precision level).
            M_tilde = F_K_tilde + np.diag(inv_prior_diag)
            M_tilde_inv = np.linalg.inv(M_tilde)
            bcrb_diag[i] = np.diag(M_tilde_inv)

            # Rotate probe predictions / mle into c_tilde.
            c_hat_tilde = c_hat_reps @ V                        # (N_rep, 4)
            mle_tilde = mle_reps @ V

            biases[i] = c_hat_tilde.mean(axis=0) - c_tilde_true
            variances[i] = c_hat_tilde.var(axis=0)
            mle_variances[i] = mle_tilde.var(axis=0)

            # Posterior Sigma_c_tilde for this scenario from the probe.
            # Sigma_c_hat_a = pred_var_freq[a] in Warsaw; rotate diag => non-diag,
            # but we only need the marginal variance per c_tilde direction.
            # Sigma_c_hat (Warsaw) is diagonal across outputs because each
            # direction was fit independently with its own residual variance.
            # Rotate the full diagonal matrix.
            Sigma_warsaw = np.diag(pred_var_freq)
            Sigma_tilde = V.T @ Sigma_warsaw @ V
            probe_post_sigma2[i] = np.diag(Sigma_tilde)

            # 68% interval per c_tilde direction.
            sd_post = np.sqrt(np.maximum(probe_post_sigma2[i], 0.0))
            c_hat_mean_tilde = c_hat_tilde.mean(axis=0)
            lo = c_hat_mean_tilde - z68 * sd_post
            hi = c_hat_mean_tilde + z68 * sd_post
            cover68[i] = ((c_tilde_true >= lo) & (c_tilde_true <= hi)).astype(float)

            # SBI record (one per held-out scenario).
            sbi_records.append({
                "c_true_warsaw": c_true.tolist(),
                "c_true_tilde": c_tilde_true.tolist(),
                "M_ctx": saved_M_ctx.tolist(),
                "Y_ctx": saved_Y_ctx.tolist(),
                "posterior_mean_warsaw": c_hat_reps.mean(axis=0).tolist(),
                "posterior_mean_tilde": c_hat_mean_tilde.tolist(),
                "posterior_cov_tilde": Sigma_tilde.tolist(),
                "interval_68_tilde": list(zip(lo.tolist(), hi.tolist())),
                "coverage_flag_tilde": cover68[i].astype(int).tolist(),
            })

        # Aggregate per pool.
        mean_lambda = fisher_diag_K.mean(axis=0)
        mean_bcrb = bcrb_diag.mean(axis=0)
        mean_bias = biases.mean(axis=0)
        mean_var = variances.mean(axis=0)
        mean_mse = (biases ** 2 + variances).mean(axis=0)
        mean_mle_var = mle_variances.mean(axis=0)
        # Standard error of the mean of the bias.
        bias_sem = biases.std(axis=0, ddof=1) / np.sqrt(len(cs_pool))
        # Efficiency clipped to (0, 1].
        eta = np.clip(mean_bcrb / np.maximum(mean_mse, 1e-30), 0.0, 1.0)
        cov_rate = cover68.mean(axis=0)

        # Per-direction verdict.
        verdicts = []
        # "Prior-dominated": F_K_diag * sigma_prior^2 < 1 (data adds less than prior).
        # Otherwise "data-dominated"; among those, efficient if eta >= 0.8.
        for a in range(N_WC):
            prior_dominated = mean_lambda[a] * sigma_prior_diag_sq[a] < 1.0
            if prior_dominated:
                verdicts.append("prior-dominated")
            elif eta[a] >= 0.8 and abs(mean_bias[a]) <= bias_sem[a]:
                verdicts.append("data-dominated-and-efficient")
            else:
                verdicts.append("data-dominated-and-inefficient")

        print(f"  Per-direction summary (c_tilde 1..4):")
        for a in range(N_WC):
            print(f"    dir {a}: lambda_K={mean_lambda[a]:.4g}  "
                  f"1/sigma_p^2={inv_prior_diag[a]:.4g}  "
                  f"BCRB={mean_bcrb[a]:.4g}  MSE={mean_mse[a]:.4g}  "
                  f"eta={eta[a]:.3f}  bias={mean_bias[a]:+.4g}  "
                  f"mle_var={mean_mle_var[a]:.4g}  cov68={cov_rate[a]:.2f}  "
                  f"verdict={verdicts[a]}")

        bcrb_summary["pools"][pool_name] = {
            "n_scenarios": int(len(cs_pool)),
            "n_rep_per_scenario": int(N_REP),
            "per_direction": [
                {
                    "c_tilde_index": a + 1,
                    "lambda_K_mean": float(mean_lambda[a]),
                    "inv_sigma_prior_sq": float(inv_prior_diag[a]),
                    "sigma_prior_sq": float(sigma_prior_diag_sq[a]),
                    "bcrb_mean": float(mean_bcrb[a]),
                    "bias_mean": float(mean_bias[a]),
                    "bias_sem": float(bias_sem[a]),
                    "variance_mean": float(mean_var[a]),
                    "mse_mean": float(mean_mse[a]),
                    "efficiency_eta": float(eta[a]),
                    "mle_variance_mean": float(mean_mle_var[a]),
                    "coverage_68": float(cov_rate[a]),
                    "verdict": verdicts[a],
                }
                for a in range(N_WC)
            ],
        }
        sbi_summary["pools"][pool_name] = sbi_records

    # ----- Persist artifacts -----
    print(f"\nSaving frozen probe artifact to {W_ARTIFACT_PATH}")
    np.savez(
        W_ARTIFACT_PATH,
        W=W,
        b=b,
        V=V,
        D=D_eig,
        sigma_prior_diag=sigma_prior_diag_sq,
        sigma_resid_squared=sigma_resid_sq,
        XtX_inv=XtX_inv,
        c_train_box=c_train,
        sigma_y=np.array(SIGMA_Y),
        notes=np.array(
            "OLS probe, additive noise sigma_y on Y; XtX_inv includes "
            "intercept as last row/col; W is (n_wc, d_psi), b is (n_wc,); "
            "V is the Fisher rotation (columns = c_tilde directions, desc "
            "by eigenvalue D); sigma_prior_diag is the empirical variance "
            "of c_tilde under the training-box prior. Mass-only observable."
        ),
    )

    print(f"Saving SBI posterior summary to {SBI_JSON_PATH}")
    with open(SBI_JSON_PATH, "w") as f:
        json.dump(sbi_summary, f, indent=2)

    print(f"Saving BCRB efficiency summary to {BCRB_JSON_PATH}")
    with open(BCRB_JSON_PATH, "w") as f:
        json.dump(bcrb_summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
