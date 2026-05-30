"""ALETHIA full-chain demonstration run.

ONE active scenario: a target Wilson configuration in the withheld band
|c_lq^(3)| in [0.6, 1.0]. The Intention FM has been pretrained on the
analytic SMEFT oracle with that band excluded. The loop watches the FM
predict in this scenario from a thin seed context; drift detectors fire
because the FM's pretrained psi_theta basis is weak in the band and the
seed context's m-support is narrow; EPIG selects more oracle queries in
the high-leverage m-regions; the context grows; predictions and coverage
recover.

Run with:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/full-chain-run/run.py
"""
from __future__ import annotations

import os, sys, time, json, logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from phoenix.otel import register

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import (
    IntentionFM, M_REF,
    IntentionConformal,
    epig_acquire_m, epig_acquire_m_eigen, target_set_entropy,
    param_epig_d_acquire, param_epig_a_acquire, _load_probe_artifact,
    DASCUSUMState, das_cusum_update,
    coverage_bh_test, kappa_drift, aggregate_action,
    eigenvector_stability,
)

_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "alethia")
_TRACING_ON = os.environ.get("PHOENIX_TRACING", "1").lower() not in {
    "0", "false", "off"}
if _TRACING_ON:
    TRACER_PROVIDER = register(project_name=_PROJECT_NAME,
                               auto_instrument=False,
                               protocol="http/protobuf")
    tracer = TRACER_PROVIDER.get_tracer("alethia.full_chain")
else:
    # No-op tracer for multi-seed parallel sweeps where flooding the
    # collector adds nothing and contends on the HTTP exporter.
    from opentelemetry import trace as _otel_trace
    tracer = _otel_trace.get_tracer("alethia.full_chain.noop")

# ----- Configuration -----
SEED = int(os.environ.get("SEED", "2026"))
N_WC = 4
WITHHOLD_DIM = 2               # clq3
WITHHOLD_BAND = (0.6, 1.0)
# Optional second withhold dim, used by the bimodal drift experiment
# (docs/research/upgrade-architecture.md §6). When BIMODAL=1 a second
# operator is also excluded from the training distribution and added
# to the target scenario; otherwise these are no-ops.
BIMODAL = os.environ.get("BIMODAL", "0").lower() in {"1", "true", "yes"}
WITHHOLD_DIM_2 = 0             # cHq3
WITHHOLD_BAND_2 = (0.4, 0.8)
TARGET_C_2 = -0.5              # c_Hq^(3) value at the bimodal target
C_TRAIN_BOX = 0.7
M_RANGE = (0.3, 2.3)
M_REF_TEV = 1.0

K_CTX = 12
Q_CTX = 32
N_PRETRAIN_SCEN = 1500
PRETRAIN_BATCH = 24
# Pretrain steps overridable via env so smoke runs on the angular
# observable (much slower per-eval than mass) can shorten warm-up
# without touching the production default. INV-3 smoke uses ~200.
PRETRAIN_STEPS = int(os.environ.get("PRETRAIN_STEPS", "1500"))
LR = 1e-3

# ----- Loop configuration -----
# Loop length and oracle budget likewise overridable for smoke tests.
N_LOOP_CYCLES = int(os.environ.get("N_LOOP_CYCLES", "400"))
PROBES_PER_CYCLE = 24
N_SEED_CTX = 8                 # thin seed: FM has narrow m-support
SEED_M_RANGE = (0.5, 1.0)      # seed context clustered in low-m only
N_CAL_POINTS = 400             # cal set drawn over full M_RANGE
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "500"))
# Stressed-budget mode (INV-3 of ALETHEIA_investigations.md, Run matrix):
# k=1 per fire, pool ~30. Default keeps the original k=5 / pool=300 budget
# so existing runs are unaffected.
STRESSED_BUDGET = os.environ.get("STRESSED_BUDGET", "0").lower() in {
    "1", "true", "yes"}
K_EPIG = 1 if STRESSED_BUDGET else 5
POOL_SIZE = 30 if STRESSED_BUDGET else 300
ACQUISITION = os.environ.get("ACQUISITION", "epig").lower()
_ACQUISITION_MODES = {"epig", "random", "leverage", "epig_eigen",
                      "param_epig_d", "param_epig_a"}
assert ACQUISITION in _ACQUISITION_MODES, (
    f"ACQUISITION must be one of {sorted(_ACQUISITION_MODES)}, "
    f"got {ACQUISITION!r}")
# Observable selector (INV-1 / INV-3): "mass" (default) is the rate ratio
# mu(c, m) and "mu_afb" is the BSM-relative FB asymmetry mu_FB(c, m). When
# mu_afb is selected the Fisher V is recomputed on the fly per Pitfall 8;
# the disclosure probe W is reused as frozen by INV-2 per Pitfall 6.
OBSERVABLE = os.environ.get("OBSERVABLE", "mass").lower()
_OBSERVABLES = {"mass", "mu_afb"}
assert OBSERVABLE in _OBSERVABLES, (
    f"OBSERVABLE must be one of {sorted(_OBSERVABLES)}, got {OBSERVABLE!r}")
# Parameter-space EPIG (INV-3) needs the frozen probe artifact (P = V^T W
# and sigma_y) produced by INV-2's probe_efficiency.py. Path overridable via
# PROBE_ARTIFACT env var; resolved subspace via PARAM_EPIG_R (default 2 for
# mass-only — the two four-fermion directions); target direction for the
# A-optimal mode via PARAM_EPIG_A (default 1, the second-most-resolved c̃).
PROBE_ARTIFACT_PATH = os.environ.get(
    "PROBE_ARTIFACT",
    str(HERE / "output" / "probe_W_mass_only_v2.npz"))
PARAM_EPIG_R = int(os.environ.get("PARAM_EPIG_R", "2"))
PARAM_EPIG_A = int(os.environ.get("PARAM_EPIG_A", "1"))

# Lazy probe-artifact load: parameter-EPIG modes always need it; the
# c̃-space scoring trace (H, per-direction error) also needs (W, sigma_y)
# to compute P A^{-1} P^T at every cycle, so load it whenever available
# under the angular observable too. Falls back to None gracefully if the
# file is absent and the mode is not parameter-EPIG.
_PROBE: dict | None = None
if ACQUISITION in ("param_epig_d", "param_epig_a"):
    _PROBE = _load_probe_artifact(PROBE_ARTIFACT_PATH)
else:
    try:
        _PROBE = _load_probe_artifact(PROBE_ARTIFACT_PATH)
    except FileNotFoundError:
        _PROBE = None

# ----- Thresholds (calibrated in agent c second-pass empirical) -----
CUSUM_H = 6.0
CUSUM_W = 30
CUSUM_K = 0.5
BH_ALPHA = 0.05
KAPPA_THRESHOLD = 1e3          # lowered from 1e4 to be sensitive to thin contexts
PROJ_RATIO_THRESHOLD = 2.5
PERSISTENCE_N = 3
CAL_PERSISTENCE_ESCALATE = 4   # consecutive cal fires that escalate to local_retrain
COOLDOWN_CYCLES = 3

# ----- Logging -----
_suffix = ACQUISITION if ACQUISITION != "epig" else ""
if BIMODAL:
    _suffix = f"bimodal_{_suffix}" if _suffix else "bimodal"
# Observable suffix: angular runs land under output_<...>_afb_<...> so
# their summaries don't clobber the mass-only runs.
if OBSERVABLE != "mass":
    _suffix = f"{_suffix}_{OBSERVABLE}" if _suffix else OBSERVABLE
# Stressed-budget suffix likewise keeps the directory distinct.
if STRESSED_BUDGET:
    _suffix = f"{_suffix}_stressed" if _suffix else "stressed"
# Seed suffix for multi-seed averaging. Default seed (2026) gets no
# suffix so existing single-seed outputs are preserved.
_seed_suffix = "" if SEED == 2026 else f"_s{SEED}"
OUT_BASENAME = (
    "output" + _seed_suffix if not _suffix
    else f"output_{_suffix}{_seed_suffix}"
)
OUT = HERE / OUT_BASENAME
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=OUT / "run.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
)
log = logging.getLogger("full_chain")
log.info("=== ALETHIA full-chain run starting ===")


def sample_c_training(n: int, rng: np.random.Generator) -> np.ndarray:
    """c ~ U([-0.7, 0.7]^4) minus the withheld bands.

    Default (BIMODAL=0): only ``|c_lq^(3)|`` is excluded.
    BIMODAL=1: ``|c_Hq^(3)|`` is also excluded simultaneously, producing
    a 2D withhold for the bimodal drift experiment.
    """
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) >= WITHHOLD_BAND[0]:
            continue
        if BIMODAL and abs(c[WITHHOLD_DIM_2]) >= WITHHOLD_BAND_2[0]:
            continue
        out[i] = c
        i += 1
    return out


def sample_c_broad(n: int, rng: np.random.Generator) -> np.ndarray:
    """c sampled from the full probe region (including withheld band)."""
    return rng.uniform(-1.0, 1.0, size=(n, N_WC))


def _oracle_truth_fn(oracle):
    """Return the noiseless-Y function for the current OBSERVABLE.

    - ``mass``:    ``oracle.truth(C, m)`` — the rate ratio μ(c, m_ll).
    - ``mu_afb``:  ``oracle.truth_mu_fb(C, m)`` — the BSM-relative FB
      asymmetry numerator μ_FB(c, m_ll). This is the multiplicative
      analogue of μ for the chirality-asymmetric piece of the cross
      section (see ``AnalyticSMEFTOracle.truth_mu_fb`` docstring) and is
      the observable INV-3 needs to activate the vertex directions.
    """
    if OBSERVABLE == "mass":
        return oracle.truth
    if OBSERVABLE == "mu_afb":
        return oracle.truth_mu_fb
    raise AssertionError(f"unhandled OBSERVABLE={OBSERVABLE!r}")


def mu_at(oracle, c: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    C = np.tile(c, (len(m_values), 1))
    return _oracle_truth_fn(oracle)(C, m_values)


def _angular_fisher_V(oracle, *, n_m: int = 60,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    r"""Recompute the Fisher rotation V for the angular observable.

    Pitfall 8 of ALETHEIA_investigations.md: V is per-Fisher and the
    mass-only V does not apply to the mass+angular observable. Here we
    build a K-point Fisher F = sum_k (∂_i Y)(∂_j Y) / σ_y² at c=0 over a
    uniform m-grid in M_RANGE, with Y the active OBSERVABLE; the
    additive-on-Y noise model is matched everywhere (Pitfall 1).

    Returns (V, λ) where V[:, a] is the a-th eigenvector ordered by
    descending eigenvalue λ.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    truth = _oracle_truth_fn(oracle)
    m = np.linspace(M_RANGE[0], M_RANGE[1], n_m)
    h = 1e-3
    zero = np.zeros((n_m, N_WC))
    dY = np.zeros((n_m, N_WC))
    for i in range(N_WC):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        dY[:, i] = (truth(cp, m) - truth(cm, m)) / (2.0 * h)
    F = dY.T @ dY
    lam, V = np.linalg.eigh(F)
    # eigh returns ascending; reverse for descending so the resolved
    # directions sit at the leading rows of P = V^T W.
    idx = np.argsort(lam)[::-1]
    return V[:, idx], lam[idx]


def _morphing_at_m(oracle, m_values: np.ndarray, h: float = 1e-3
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Finite-difference morphing decomposition Y(c, m) at the active
    OBSERVABLE: Y = Y_SM + A·c + cᵀ B c, with A: (K, N_WC), B: (K, N_WC,
    N_WC). Used by the oracle MLE (Pitfall 5: reuse morphing, don't
    rebuild) and by the c̃-space scoring trace.

    Y_SM, A, B are evaluated at c=0; at fixed m, mu(c, m) is exactly
    quadratic in c at LO and the analytic μ_FB is too, so this captures
    the observable exactly.
    """
    truth = _oracle_truth_fn(oracle)
    m = np.asarray(m_values, dtype=float)
    K = len(m)
    n = N_WC
    zero = np.zeros((K, n))
    Y_SM = truth(zero, m)
    A = np.empty((K, n))
    for i in range(n):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        A[:, i] = (truth(cp, m) - truth(cm, m)) / (2.0 * h)
    H = np.zeros((K, n, n))
    for i in range(n):
        cp = zero.copy(); cp[:, i] += h
        cm = zero.copy(); cm[:, i] -= h
        H[:, i, i] = (truth(cp, m) + truth(cm, m) - 2.0 * Y_SM) / (h * h)
    for i in range(n):
        for j in range(i + 1, n):
            c_pp = zero.copy(); c_pp[:, i] += h; c_pp[:, j] += h
            c_pm = zero.copy(); c_pm[:, i] += h; c_pm[:, j] -= h
            c_mp = zero.copy(); c_mp[:, i] -= h; c_mp[:, j] += h
            c_mm = zero.copy(); c_mm[:, i] -= h; c_mm[:, j] -= h
            Hij = (truth(c_pp, m) - truth(c_pm, m)
                   - truth(c_mp, m) + truth(c_mm, m)) / (4.0 * h * h)
            H[:, i, j] = Hij
            H[:, j, i] = Hij
    return Y_SM, A, 0.5 * H


class _MorphingTable:
    """Cubic-spline interpolant of (Y_SM(m), A_i(m), B_ij(m)) for the
    active OBSERVABLE. Built once per run on a dense m-grid; reused at
    every cycle for the oracle MLE and c̃-space scoring."""

    def __init__(self, oracle, n_grid: int = 160):
        from scipy.interpolate import CubicSpline
        self.m_grid = np.linspace(M_RANGE[0], M_RANGE[1], n_grid)
        Y_SM, A, B = _morphing_at_m(oracle, self.m_grid)
        self._y_sm = CubicSpline(self.m_grid, Y_SM, bc_type="natural")
        self._A = [CubicSpline(self.m_grid, A[:, i], bc_type="natural")
                   for i in range(N_WC)]
        self._B = {}
        for i in range(N_WC):
            for j in range(i, N_WC):
                self._B[(i, j)] = CubicSpline(
                    self.m_grid, B[:, i, j], bc_type="natural")

    def at(self, m: np.ndarray):
        m = np.asarray(m, dtype=float)
        K = m.shape[0]
        Y_SM = self._y_sm(m)
        A = np.stack([s(m) for s in self._A], axis=1)
        B = np.zeros((K, N_WC, N_WC))
        for (i, j), s in self._B.items():
            v = s(m)
            B[:, i, j] = v
            B[:, j, i] = v
        return Y_SM, A, B


def _mle_c(Y_obs: np.ndarray, Y_SM: np.ndarray, A: np.ndarray, B: np.ndarray,
           c0: np.ndarray, sigma_y: float) -> np.ndarray:
    """Oracle MLE for c given context residuals against the morphing.

    Additive Gaussian noise on Y, matching the noise model used in F_K
    and the probe (Pitfall 1). Residuals = (Y_obs - Y_pred) / σ_y.
    """
    from scipy.optimize import least_squares
    def residuals(c):
        Y_pred = Y_SM + A @ c + np.einsum("i,kij,j->k", c, B, c)
        return (Y_obs - Y_pred) / sigma_y
    try:
        sol = least_squares(residuals, x0=c0, method="lm", max_nfev=200)
        return sol.x
    except Exception:
        return np.full(N_WC, np.nan)


def _ctilde_metrics(model: IntentionFM,
                    M_ctx: np.ndarray, Y_ctx: np.ndarray,
                    target_c: np.ndarray,
                    W: np.ndarray, b: np.ndarray, V: np.ndarray,
                    sigma_y: float, resolved_dim: int,
                    morphing: "_MorphingTable | None",
                    ) -> dict:
    r"""Per-cycle c̃-space scoring trace (INV-3 Run matrix).

    Returns a dict with:
      - ``H_ctilde``: ``½ log det Σ`` over the resolved c̃ subspace, with
        Σ = σ_y² P A⁻¹ Pᵀ, P = Vᵀ W restricted to its leading
        ``resolved_dim`` rows. Resolved-subspace posterior entropy
        (Pitfall 3: stays well-conditioned because r is the data-dominated
        subspace identified in INV-2).
      - ``c_tilde_err``: per-direction posterior error on the leading
        ``resolved_dim`` rotated coefficients against the oracle MLE on
        the current context. The MLE is fit by least squares against the
        analytic morphing of the active OBSERVABLE (Pitfall 5).
      - ``Sigma_diag``: diagonal of Σ (for downstream calibration).
    """
    A_inv, w_ridge, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    P_full = V.T @ W                                     # (n_wc, d_psi)
    P_r = P_full[:resolved_dim]
    Sigma = (sigma_y ** 2) * (P_r @ A_inv @ P_r.T)
    sign, logdet = np.linalg.slogdet(
        Sigma + 1e-15 * np.eye(resolved_dim))
    H = 0.5 * float(logdet) if sign > 0 else float("nan")

    target_c_tilde = V.T @ target_c
    out: dict = {
        "H_ctilde": H,
        "Sigma_diag": np.diag(Sigma).tolist(),
        "target_c_tilde": target_c_tilde.tolist(),
    }

    if morphing is not None:
        Y_SM_ctx, A_morph, B_morph = morphing.at(M_ctx)
        c_mle = _mle_c(Y_ctx, Y_SM_ctx, A_morph, B_morph,
                       c0=np.zeros(N_WC), sigma_y=sigma_y)
        c_tilde_mle = V.T @ c_mle
        # Per-direction error: posterior mean error |c̃_a^MLE - c̃_a^true|
        # for a ∈ [0, resolved_dim). The MLE is the oracle reference
        # (asymptotically efficient) per INV-2's cross-check.
        err = np.abs(c_tilde_mle[:resolved_dim] - target_c_tilde[:resolved_dim])
        out["c_tilde_err"] = err.tolist()
        out["c_tilde_mle"] = c_tilde_mle[:resolved_dim].tolist()
    else:
        out["c_tilde_err"] = [float("nan")] * resolved_dim
        out["c_tilde_mle"] = [float("nan")] * resolved_dim
    return out


# --------------------- Pretraining ---------------------

def pretrain_intention(oracle, rng) -> tuple:
    log.info("Pretraining: steps=%d batch=%d K=%d Q=%d",
             PRETRAIN_STEPS, PRETRAIN_BATCH, K_CTX, Q_CTX)
    torch.manual_seed(SEED)
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    opt = Adam(model.parameters(), lr=LR)

    pretrain_wall = 0.0
    losses = []
    with tracer.start_as_current_span("chain.pretrain") as span:
        span.set_attribute("aletheia.pretrain.n_steps", PRETRAIN_STEPS)
        span.set_attribute("aletheia.pretrain.batch", PRETRAIN_BATCH)
        span.set_attribute("aletheia.pretrain.d_psi", 16)
        span.set_attribute("aletheia.pretrain.withhold_dim_name", "c_lq^(3)")
        span.set_attribute("aletheia.pretrain.withhold_band_lo",
                           WITHHOLD_BAND[0])
        span.set_attribute("aletheia.pretrain.withhold_band_hi",
                           WITHHOLD_BAND[1])

        c_pool = sample_c_training(N_PRETRAIN_SCEN, rng)
        truth_fn = _oracle_truth_fn(oracle)
        t0 = time.time()
        for step in range(PRETRAIN_STEPS):
            idx = rng.choice(N_PRETRAIN_SCEN, size=PRETRAIN_BATCH,
                             replace=False)
            cs = c_pool[idx]
            B, K, Q = PRETRAIN_BATCH, K_CTX, Q_CTX
            M_ctx_np = rng.uniform(*M_RANGE, size=(B, K))
            M_q_np = rng.uniform(*M_RANGE, size=(B, Q))
            C_ctx = np.repeat(cs, K, axis=0)
            C_q = np.repeat(cs, Q, axis=0)
            Y_ctx_np = truth_fn(C_ctx, M_ctx_np.flatten()).reshape(B, K)
            Y_q_np = truth_fn(C_q, M_q_np.flatten()).reshape(B, Q)

            M_ctx = torch.from_numpy(M_ctx_np).float()
            Y_ctx = torch.from_numpy(Y_ctx_np).float()
            M_q = torch.from_numpy(M_q_np).float()
            Y_q = torch.from_numpy(Y_q_np).float()

            y_pred = model(M_ctx, Y_ctx, M_q)
            loss = F.mse_loss(y_pred, Y_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

            if step % 100 == 0 or step == PRETRAIN_STEPS - 1:
                log.info("  step %4d/%d loss=%.6f", step, PRETRAIN_STEPS,
                         loss.item())
        pretrain_wall = time.time() - t0
        span.set_attribute("aletheia.pretrain.wall_seconds", pretrain_wall)
        span.set_attribute("aletheia.pretrain.final_loss", losses[-1])
        log.info("Pretrain done in %.1fs, final loss=%.6f",
                 pretrain_wall, losses[-1])
    return model, pretrain_wall, np.array(losses)


# --------------------- Loop ---------------------

class RegionTagger:
    """Tag m-values into 5 buckets over M_RANGE."""

    def __init__(self):
        self.edges = np.linspace(M_RANGE[0], M_RANGE[1], 6)

    def tag(self, m: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges[1:-1], m), 0,
                       len(self.edges) - 2)


def run_loop(oracle, model: IntentionFM, rng) -> dict:
    log.info("=== Loop starting ===")

    # ONE target scenario in the withheld band(s).
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = 0.8   # clq3 = +0.8 in the withheld band
    if BIMODAL:
        target_c[WITHHOLD_DIM_2] = TARGET_C_2
    log.info("Target scenario c=%s (BIMODAL=%s)", target_c.tolist(), BIMODAL)

    # c̃-space scoring requires (W, b, V, σ_y) and a morphing table for
    # the active OBSERVABLE. W and σ_y come from the frozen INV-2 probe
    # artifact (Pitfall 6); V is recomputed per-observable on the fly
    # (Pitfall 8) when OBSERVABLE != mass. The morphing is built once.
    morphing_local: "_MorphingTable | None" = None
    W_probe = b_probe = V_probe = None
    sigma_y_probe = float("nan")
    P_for_acq: np.ndarray | None = None
    if _PROBE is not None:
        try:
            morphing_local = _MorphingTable(oracle, n_grid=120)
            W_probe = np.asarray(_PROBE["W"], dtype=np.float64)
            b_probe = np.asarray(_PROBE.get("b", np.zeros(N_WC)),
                                  dtype=np.float64)
            sigma_y_probe = float(_PROBE.get("sigma_y", 0.05))
            if OBSERVABLE == "mass":
                V_probe = np.asarray(_PROBE["V"], dtype=np.float64)
            else:
                V_probe, lam_obs = _angular_fisher_V(oracle)
                log.info("Recomputed Fisher V for OBSERVABLE=%s; "
                         "eigvals=%s (Pitfall 8)",
                         OBSERVABLE, lam_obs.tolist())
            P_for_acq = V_probe.T @ W_probe
        except Exception as exc:
            log.warning("c̃-space scoring disabled: %s", exc)
            morphing_local = None
            P_for_acq = None
    if morphing_local is None:
        log.info("c̃-space scoring trace inactive: probe artifact "
                 "missing or morphing build failed.")

    # Thin seed context: K=8 points clustered in [0.5, 1.0] only.
    M_ctx = rng.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
    Y_ctx = mu_at(oracle, target_c, M_ctx)
    log.info("Seed context: K=%d, m in %s", N_SEED_CTX, SEED_M_RANGE)

    # Calibration set: broader probe region (m in full M_RANGE for target c).
    # Per the invariant: cal set covers the distribution the model will be
    # queried on. For a single-scenario loop, the cal set is m-values across
    # the full M_RANGE for the same target c.
    M_cal = rng.uniform(*M_RANGE, size=N_CAL_POINTS)
    Y_cal = mu_at(oracle, target_c, M_cal)

    cc = IntentionConformal(n_strata=5, noise_frac=0.05)
    cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)

    # Target set for H_T (the headline monitor): fine m-grid.
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    Y_target_truth = mu_at(oracle, target_c, M_target)

    # Region tagger and baseline projection variance.
    tagger = RegionTagger()
    n_regions = 5
    train_M_sample = rng.uniform(*M_RANGE, size=300)
    Psi_train = model.psi_np(train_M_sample)
    eigs_train, vecs_train = np.linalg.eigh(
        Psi_train.T @ Psi_train + model.alpha * np.eye(model.d_psi))
    v_min_train = vecs_train[:, 0]
    train_proj_var = float(np.var(Psi_train @ v_min_train))

    # Streaming state.
    cusum_state = DASCUSUMState(
        buf=DASCUSUMState().buf.__class__(maxlen=CUSUM_W))
    coverage_counts_68 = np.zeros((n_regions, 2), dtype=int)
    coverage_counts_95 = np.zeros((n_regions, 2), dtype=int)
    history_flags = []

    # Trajectories. The H_ctilde / c_tilde_err_d{1,2} lists are the
    # INV-3 c̃-space scoring trace; they default to NaN per cycle when
    # the probe artifact is unavailable.
    traj = dict(
        H_T=[], kappa=[], cov_68_by_region=[], cov_95_by_region=[],
        context_size=[], oracle_calls=[], drift_flags=[], action=[],
        cusum_S=[], cal_pvalue_min=[], proj_ratio=[],
        H_ctilde=[], c_tilde_err_d1=[], c_tilde_err_d2=[],
        c_tilde_mle_d1=[], c_tilde_mle_d2=[],
    )
    oracle_calls = 0
    consec_recovered = 0
    cycles_done = 0
    last_action_cycle = -100
    consec_cal_fires = 0
    rmse_band_trace = []
    # Previous-cycle eigenvector blocks for stability tracking on the
    # chain.drift.eigen span. None on the first cycle.
    prev_U_top: np.ndarray | None = None
    prev_U_low: np.ndarray | None = None

    t0 = time.time()
    for cycle in range(N_LOOP_CYCLES):
        with tracer.start_as_current_span("chain.cycle") as cyc_span:
            cyc_span.set_attribute("aletheia.cycle.index", cycle)
            cyc_span.set_attribute("aletheia.cycle.context_size", len(M_ctx))

            # Sample probe m-values across full M_RANGE. Bias 50% to the
            # tails [1.5, 2.3] where seed context didn't reach.
            n_tail = int(0.5 * PROBES_PER_CYCLE)
            probe_m_tail = rng.uniform(1.5, M_RANGE[1], n_tail)
            probe_m_bulk = rng.uniform(*M_RANGE, PROBES_PER_CYCLE - n_tail)
            probe_m = np.concatenate([probe_m_tail, probe_m_bulk])
            rng.shuffle(probe_m)
            probe_y_truth = mu_at(oracle, target_c, probe_m)

            # FM predict.
            with tracer.start_as_current_span("tool.surrogate.predict") as sp:
                mu_pred = model.predict_np(M_ctx, Y_ctx, probe_m)
                lev = model.leverage(M_ctx, probe_m)
                sd_68 = cc.coverage_sigma(
                    model, M_ctx, Y_ctx, probe_m, 0.683)
                sd_95 = cc.coverage_sigma(
                    model, M_ctx, Y_ctx, probe_m, 0.954)
                sp.set_attribute("aletheia.fm.n_query", PROBES_PER_CYCLE)
                sp.set_attribute("aletheia.fm.leverage_mean",
                                 float(lev.mean()))
                sp.set_attribute("aletheia.fm.leverage_max",
                                 float(lev.max()))
                sp.set_attribute("aletheia.fm.mu_mean", float(mu_pred.mean()))
                sp.set_attribute("aletheia.fm.sigma_mean", float(sd_68.mean()))

            # Standardised residuals; per-region coverage tally.
            std_resid = (probe_y_truth - mu_pred) / np.maximum(sd_68, 1e-12)
            tags = tagger.tag(probe_m)
            cov68_hit = np.abs(probe_y_truth - mu_pred) < sd_68
            cov95_hit = np.abs(probe_y_truth - mu_pred) < sd_95
            for s in range(n_regions):
                mask = (tags == s)
                if mask.any():
                    coverage_counts_68[s, 0] += int(cov68_hit[mask].sum())
                    coverage_counts_68[s, 1] += int(mask.sum())
                    coverage_counts_95[s, 0] += int(cov95_hit[mask].sum())
                    coverage_counts_95[s, 1] += int(mask.sum())

            # Drift detection.
            with tracer.start_as_current_span("chain.drift.evaluate"):
                with tracer.start_as_current_span(
                        "tool.drift.accuracy.das_cusum") as ds:
                    z_mean = float(np.median(std_resid))
                    cusum_state, acc_fired, S_t = das_cusum_update(
                        cusum_state, z_mean, w=CUSUM_W, h=CUSUM_H, k=CUSUM_K)
                    ds.set_attribute("aletheia.drift.acc.S_t", S_t)
                    ds.set_attribute("aletheia.drift.acc.fired", acc_fired)

                with tracer.start_as_current_span(
                        "tool.drift.calibration.bh") as cs_:
                    cal_fired, pvals, fail_mask, bh_thr = coverage_bh_test(
                        coverage_counts_68[:, 0], coverage_counts_68[:, 1],
                        target_coverage=0.683, alpha=BH_ALPHA)
                    cs_.set_attribute("aletheia.drift.cal.global_pvalue",
                                      float(pvals.min()))
                    cs_.set_attribute("aletheia.drift.cal.failing_regions",
                                      int(fail_mask.sum()))
                    cs_.set_attribute("aletheia.drift.cal.fired", cal_fired)

                with tracer.start_as_current_span(
                        "tool.drift.coverage.kappa") as ks:
                    cov_fired, kappa, proj_ratio, eig = kappa_drift(
                        model, M_ctx, probe_m, train_proj_var,
                        kappa_threshold=KAPPA_THRESHOLD,
                        proj_ratio_threshold=PROJ_RATIO_THRESHOLD)
                    ks.set_attribute("aletheia.drift.cov.kappa", kappa)
                    ks.set_attribute(
                        "aletheia.drift.cov.vmin_projection_ratio",
                        proj_ratio)
                    ks.set_attribute("aletheia.drift.cov.fired", cov_fired)

                # Emit chain.drift.eigen span carrying the full
                # spectrum and the top/low eigenvector blocks. See
                # docs/research/upgrade-architecture.md §2.2.
                with tracer.start_as_current_span(
                        "chain.drift.eigen") as es_eig:
                    k_top = min(3, eig.d)
                    U_top, lam_top = eig.high_subspace(k_top)
                    U_low, lam_low = eig.low_subspace(k_top)
                    # Stability against previous cycle's blocks.
                    if prev_U_top is not None and prev_U_low is not None:
                        stab_top = eigenvector_stability(
                            prev_U_top, U_top).tolist()
                        stab_low = eigenvector_stability(
                            prev_U_low, U_low).tolist()
                    else:
                        stab_top = [1.0] * k_top
                        stab_low = [1.0] * k_top
                    es_eig.set_attribute(
                        "aletheia.eigen.lambda", eig.lam.tolist())
                    es_eig.set_attribute(
                        "aletheia.eigen.lam_top_k", lam_top.tolist())
                    es_eig.set_attribute(
                        "aletheia.eigen.lam_low_k", lam_low.tolist())
                    # Flatten d x k matrices column-major for span attrs.
                    es_eig.set_attribute(
                        "aletheia.eigen.U_top_flat",
                        U_top.flatten(order="F").tolist())
                    es_eig.set_attribute(
                        "aletheia.eigen.U_low_flat",
                        U_low.flatten(order="F").tolist())
                    es_eig.set_attribute("aletheia.eigen.d", eig.d)
                    es_eig.set_attribute("aletheia.eigen.k", k_top)
                    # Carry cycle index directly so the Phoenix rubric
                    # in eigen_query.py can filter by --since-cycle.
                    es_eig.set_attribute("aletheia.cycle.index", cycle)
                    es_eig.set_attribute(
                        "aletheia.eigen.kappa", float(eig.kappa))
                    es_eig.set_attribute(
                        "aletheia.eigen.lis_rank", eig.lis_rank())
                    es_eig.set_attribute(
                        "aletheia.eigen.stability_top", stab_top)
                    es_eig.set_attribute(
                        "aletheia.eigen.stability_low", stab_low)
                    prev_U_top, prev_U_low = U_top, U_low

            # Persistence escalation: 4+ consecutive cal fires => escalate.
            consec_cal_fires = consec_cal_fires + 1 if cal_fired else 0
            forced_local = consec_cal_fires >= CAL_PERSISTENCE_ESCALATE

            with tracer.start_as_current_span("tool.drift.aggregate") as ag:
                action, target_signal = aggregate_action(
                    acc_fired, cal_fired, cov_fired,
                    history_flags, persistence_N=PERSISTENCE_N)
                history_flags.append((acc_fired, cal_fired, cov_fired))
                if forced_local and action in ("recal", "watch", "noop"):
                    action = "local_retrain"
                    target_signal = "cal_persist"
                flag_str = ("1" if acc_fired else "0") + \
                           ("1" if cal_fired else "0") + \
                           ("1" if cov_fired else "0")
                ag.set_attribute("aletheia.drift.combined_flag", flag_str)
                ag.set_attribute("aletheia.drift.action", action)
                if target_signal:
                    ag.set_attribute("aletheia.drift.target_signal",
                                     target_signal)

            # Act.
            if action in ("local_retrain", "global_retrain"):
                if cycle - last_action_cycle < COOLDOWN_CYCLES:
                    action = "cooled"
                elif oracle_calls + K_EPIG > ORACLE_BUDGET:
                    action = "budget_exhausted"
                else:
                    last_action_cycle = cycle
                    consec_cal_fires = 0
                    with tracer.start_as_current_span(
                            "tool.epig.select") as es:
                        # Pool of candidates: emphasise high-leverage regions.
                        M_pool = rng.uniform(*M_RANGE, size=POOL_SIZE)
                        if ACQUISITION == "random":
                            picked = rng.choice(len(M_pool), size=K_EPIG, replace=False)
                            picked_m = M_pool[picked]
                        elif ACQUISITION == "leverage":
                            # D-optimal greedy: pick k pool points with highest leverage(M_pool, M_target).
                            A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
                            Psi_pool = model.psi_np(M_pool)
                            lev_P = np.einsum("pd,de,pe->p", Psi_pool, A_inv, Psi_pool)
                            picked = np.argsort(lev_P)[::-1][:K_EPIG]
                            picked_m = M_pool[picked]
                        elif ACQUISITION == "epig_eigen":
                            # Resample the pool from the low-eigen subspace
                            # of A on the current context, then EPIG-score
                            # the resampled pool. See architecture §3.
                            picked_m = epig_acquire_m_eigen(
                                model, M_ctx, Y_ctx, M_pool, M_target,
                                eigen=eig, k_low=3, k_pick=K_EPIG, rng=rng)
                        elif ACQUISITION in ("param_epig_d", "param_epig_a"):
                            # INV-3 parameter-space EPIG over c̃-subspace.
                            # P = V^T W and σ_y come from the frozen INV-2
                            # probe artifact; V is observable-dependent
                            # (Pitfall 8) so use the in-loop ``P_for_acq``
                            # which recomputes V for the current
                            # OBSERVABLE while keeping W frozen
                            # (Pitfall 6).
                            P_acq = (P_for_acq if P_for_acq is not None
                                     else _PROBE["P"])
                            picked_idx = (
                                param_epig_d_acquire(
                                    model, M_ctx, Y_ctx, M_pool,
                                    P=P_acq, k=K_EPIG,
                                    sigma_y=sigma_y_probe,
                                    resolved_dim=PARAM_EPIG_R)
                                if ACQUISITION == "param_epig_d"
                                else param_epig_a_acquire(
                                    model, M_ctx, Y_ctx, M_pool,
                                    P=P_acq, k=K_EPIG,
                                    target_direction=PARAM_EPIG_A,
                                    sigma_y=sigma_y_probe))
                            picked_m = M_pool[picked_idx]
                        else:
                            picked = epig_acquire_m(
                                model, M_ctx, Y_ctx, M_pool, M_target,
                                k=K_EPIG)
                            picked_m = M_pool[picked]
                        es.set_attribute("aletheia.epig.k", K_EPIG)
                        es.set_attribute("aletheia.epig.pool_size", POOL_SIZE)
                        es.set_attribute("aletheia.epig.acquisition", ACQUISITION)
                        es.set_attribute("aletheia.epig.picked_m_values",
                                         np.asarray(picked_m).round(3).tolist())

                    with tracer.start_as_current_span(
                            "tool.oracle.query") as oq:
                        picked_y = mu_at(oracle, target_c, picked_m)
                        oq.set_attribute("aletheia.oracle.fidelity_tier",
                                         "T1")
                        oq.set_attribute("aletheia.oracle.n_points", K_EPIG)
                        oq.set_attribute("aletheia.oracle.wilson_norm",
                                         float(np.linalg.norm(target_c)))

                    with tracer.start_as_current_span(
                            "tool.fm.update") as fu:
                        H_T_pre = target_set_entropy(
                            model, M_ctx, Y_ctx, M_target)
                        ctx_in = len(M_ctx)
                        M_ctx = np.concatenate([M_ctx, picked_m])
                        Y_ctx = np.concatenate([Y_ctx, picked_y])
                        H_T_post = target_set_entropy(
                            model, M_ctx, Y_ctx, M_target)
                        fu.set_attribute("aletheia.fm.context_size_in",
                                         ctx_in)
                        fu.set_attribute("aletheia.fm.context_size_out",
                                         len(M_ctx))
                        fu.set_attribute(
                            "aletheia.fm.target_entropy_H_T_pre", H_T_pre)
                        fu.set_attribute(
                            "aletheia.fm.target_entropy_H_T_post",
                            H_T_post)
                        fu.set_attribute(
                            "aletheia.fm.condition_number_kappa",
                            model.kappa_A(M_ctx))

                    cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)
                    oracle_calls += K_EPIG
            elif action in ("recal", "recal_then_check"):
                cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)

            # Track.
            H_T_cur = target_set_entropy(model, M_ctx, Y_ctx, M_target)
            kappa_cur = model.kappa_A(M_ctx)
            cov_68_per = np.where(
                coverage_counts_68[:, 1] > 0,
                coverage_counts_68[:, 0] /
                np.maximum(coverage_counts_68[:, 1], 1),
                np.nan)
            cov_95_per = np.where(
                coverage_counts_95[:, 1] > 0,
                coverage_counts_95[:, 0] /
                np.maximum(coverage_counts_95[:, 1], 1),
                np.nan)
            # RMSE on the target set (the band the FM should be predicting):
            mu_band = model.predict_np(M_ctx, Y_ctx, M_target)
            rmse_band = float(np.sqrt(np.mean((mu_band - Y_target_truth) ** 2)))
            rmse_band_trace.append(rmse_band)

            traj["H_T"].append(H_T_cur)
            traj["kappa"].append(kappa_cur)
            traj["cov_68_by_region"].append(cov_68_per.copy())
            traj["cov_95_by_region"].append(cov_95_per.copy())
            traj["context_size"].append(len(M_ctx))
            traj["oracle_calls"].append(oracle_calls)
            traj["drift_flags"].append(flag_str)
            traj["action"].append(action)
            traj["cusum_S"].append(S_t)
            traj["cal_pvalue_min"].append(float(pvals.min()))
            traj["proj_ratio"].append(proj_ratio)

            # c̃-space scoring trace (INV-3 Run matrix). NaNs if the probe
            # artifact and morphing were unavailable; otherwise H =
            # ½ log det Σ on the resolved subspace, plus per-direction
            # MLE error on c̃_1 and c̃_2 (the two leading rotated coords).
            if (morphing_local is not None and W_probe is not None
                    and V_probe is not None):
                ct = _ctilde_metrics(
                    model, M_ctx, Y_ctx, target_c,
                    W=W_probe, b=b_probe, V=V_probe,
                    sigma_y=sigma_y_probe,
                    resolved_dim=max(2, PARAM_EPIG_R),
                    morphing=morphing_local)
                traj["H_ctilde"].append(float(ct["H_ctilde"]))
                err = ct["c_tilde_err"]
                mle = ct["c_tilde_mle"]
                traj["c_tilde_err_d1"].append(float(err[0]) if len(err) > 0
                                              else float("nan"))
                traj["c_tilde_err_d2"].append(float(err[1]) if len(err) > 1
                                              else float("nan"))
                traj["c_tilde_mle_d1"].append(float(mle[0]) if len(mle) > 0
                                              else float("nan"))
                traj["c_tilde_mle_d2"].append(float(mle[1]) if len(mle) > 1
                                              else float("nan"))
            else:
                traj["H_ctilde"].append(float("nan"))
                traj["c_tilde_err_d1"].append(float("nan"))
                traj["c_tilde_err_d2"].append(float("nan"))
                traj["c_tilde_mle_d1"].append(float("nan"))
                traj["c_tilde_mle_d2"].append(float("nan"))

            cyc_span.set_attribute("aletheia.cycle.H_T", H_T_cur)
            cyc_span.set_attribute("aletheia.cycle.kappa", kappa_cur)
            cyc_span.set_attribute("aletheia.cycle.oracle_calls",
                                   oracle_calls)
            cyc_span.set_attribute("aletheia.cycle.rmse_target_band",
                                   rmse_band)

            # Rolling window cov check: every 30 cycles inspect aggregate
            # cov_68 over [0.65, 0.72] band.
            if cycle % 30 == 29:
                tot_cov = coverage_counts_68[:, 0].sum()
                tot_n = max(coverage_counts_68[:, 1].sum(), 1)
                rolling_68 = tot_cov / tot_n
                if abs(rolling_68 - 0.683) < 0.03:
                    consec_recovered += 1
                else:
                    consec_recovered = 0
                # Reset for the next window.
                coverage_counts_68 = np.zeros((n_regions, 2), dtype=int)
                coverage_counts_95 = np.zeros((n_regions, 2), dtype=int)

            cycles_done = cycle + 1

            if cycle % 25 == 0:
                log.info(
                    "[c%4d] H_T=%.2f kappa=%.1f oracle=%d ctx=%d flag=%s "
                    "act=%-15s S=%.2f p=%.3f proj=%.2f rmse=%.3f",
                    cycle, H_T_cur, kappa_cur, oracle_calls, len(M_ctx),
                    flag_str, action, S_t, float(pvals.min()), proj_ratio,
                    rmse_band)

            if (oracle_calls >= ORACLE_BUDGET
                    and consec_recovered >= 1):
                log.info("[c%d] budget reached and recovered", cycle)
                break
            if consec_recovered >= 2 and oracle_calls > 50:
                log.info("[c%d] coverage recovered for 2 windows", cycle)
                break

    wall = time.time() - t0
    log.info("Loop done: %d cycles in %.1fs, oracle=%d, ctx=%d",
             cycles_done, wall, oracle_calls, len(M_ctx))

    final_mu_band = model.predict_np(M_ctx, Y_ctx, M_target)
    final_lev_band = model.leverage(M_ctx, M_target)
    return dict(
        traj=traj, M_ctx=M_ctx, Y_ctx=Y_ctx,
        M_cal=M_cal, Y_cal=Y_cal,
        target_c=target_c, M_target=M_target,
        Y_target_truth=Y_target_truth,
        final_mu_band=final_mu_band, final_lev_band=final_lev_band,
        cycles_done=cycles_done, oracle_calls=oracle_calls,
        wall_seconds=wall, rmse_band_trace=rmse_band_trace,
    )


def main():
    rng = np.random.default_rng(SEED)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    t_total = time.time()

    model, pretrain_wall, pretrain_losses = pretrain_intention(oracle, rng)

    # BEFORE snapshot: FM's prediction with only the thin seed context.
    rng_b = np.random.default_rng(SEED + 1)
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = 0.8
    M_band_eval = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 100)
    Y_band_truth = mu_at(oracle, target_c, M_band_eval)
    M_seed_b = rng_b.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
    Y_seed_b = mu_at(oracle, target_c, M_seed_b)
    mu_before = model.predict_np(M_seed_b, Y_seed_b, M_band_eval)

    out = run_loop(oracle, model, rng)

    mu_after = model.predict_np(out["M_ctx"], out["Y_ctx"], M_band_eval)

    # Persist everything.
    np.savez(
        OUT / "trajectory.npz",
        **{k: np.array(v) for k, v in out["traj"].items()},
        rmse_band_trace=np.array(out["rmse_band_trace"]),
        M_ctx_final=out["M_ctx"], Y_ctx_final=out["Y_ctx"],
        M_band_eval=M_band_eval, Y_band_truth=Y_band_truth,
        mu_before=mu_before, mu_after=mu_after,
        M_target=out["M_target"], Y_target_truth=out["Y_target_truth"],
        final_mu_band=out["final_mu_band"],
        final_lev_band=out["final_lev_band"],
        target_c=target_c,
        pretrain_losses=pretrain_losses,
    )

    torch.save(model.state_dict(), OUT / "intention_fm.pt")

    # Summary.
    final_cov_68 = float(np.nanmean(out["traj"]["cov_68_by_region"][-1])
                         if len(out["traj"]["cov_68_by_region"]) > 0
                         else float("nan"))
    final_kappa = out["traj"]["kappa"][-1]
    final_H_T = out["traj"]["H_T"][-1]
    summary = dict(
        cycles_done=out["cycles_done"],
        oracle_calls=out["oracle_calls"],
        loop_wall_seconds=out["wall_seconds"],
        pretrain_wall_seconds=pretrain_wall,
        total_wall_seconds=time.time() - t_total,
        final_cov_68_avg_over_regions=final_cov_68,
        final_kappa=final_kappa,
        final_H_T=final_H_T,
        final_context_size=int(out["traj"]["context_size"][-1]),
        before_band_rmse=float(np.sqrt(
            np.mean((mu_before - Y_band_truth) ** 2))),
        after_band_rmse=float(np.sqrt(
            np.mean((mu_after - Y_band_truth) ** 2))),
        n_drift_events=int(sum(
            1 for a in out["traj"]["action"]
            if a in ("local_retrain", "global_retrain", "recal",
                     "recal_then_check"))),
        n_local_retrain=int(sum(1 for a in out["traj"]["action"]
                                if a == "local_retrain")),
        n_recal=int(sum(1 for a in out["traj"]["action"] if a == "recal")),
        engineered_band=list(WITHHOLD_BAND),
        engineered_dim_name="c_lq^(3)",
        engineered_target_c=target_c.tolist(),
        d_psi=16, pretrain_steps=PRETRAIN_STEPS,
        oracle_budget=ORACLE_BUDGET, K_epig=K_EPIG,
        cusum_h=CUSUM_H, bh_alpha=BH_ALPHA,
        kappa_threshold=KAPPA_THRESHOLD,
        proj_ratio_threshold=PROJ_RATIO_THRESHOLD,
    )
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=== Run complete ===")
    log.info(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
