"""FM-representation adapter for the B.9 contraction-vs-MLE sweep (T0.2).

The B.9 machinery (stage_c_closed_loop.py, _common.py) is generic over the
*feature map*: every acquisition rule and the contraction / MLE metrics only
ever touch

    model.psi_np(M)            -> (n, D) ridge feature rows for candidates M
    model.A_inv_and_w(M, Y)    -> (A^{-1}, w, Psi) closed-form ridge solve
    model.alpha                -> ridge regulariser

In B.9 the candidate is a scalar mass m and psi(m) is a single MLP row
(IntentionFM, the precursor binned-ratio head). Here we swap in the trained
ManifoldInformer encoder. A candidate is still a mass m, but the FM consumes
per-event features x = (log m / M_ref, cos theta*). To produce one ridge row
per candidate mass we

    1. sample n_ev events whose m is pinned to the candidate mass and whose
       cos theta* is drawn from the *target working point's* angular density
       (so the angular structure cos theta* actually appears in the row, which
       is the whole point of the event-level encoder vs the mass-only head);
    2. encode each event with the trained per-event encoder phi_theta;
    3. mean-pool the per-event R^16 embeddings to one R^16 ridge row.

The pooling is deterministic per mass (seeded from the mass) so psi_np is a
pure function and the design matrix A is well defined and reproducible, exactly
as psi(m) is in the precursor head.

Two encoder variants are supported, matching integrated_loop.py:
    - "full"      : the trained d_event=2 encoder (already carries cos theta*).
    - "def_ext"   : the deficient d_event=1 encoder + psi-extension, i.e. the
                    deficient encoder's R^16 rows with the residual-SVD u1
                    direction appended (the "post-extension" representation the
                    work order asks for). u1 is recovered from the residual
                    operator on probe events, identical to integrated_loop's
                    extend_psi.

The probe W (FM-ridge-weight -> c-tilde) is representation specific and is fit
fresh here: the precursor probe_W_mass_only lives in the precursor R^16 and is
meaningless in the FM R^16. We fit W by regressing the per-c FM ridge weight
onto the Wilson coefficients across a training grid of working points, then
rotate by the SAME Fisher V (recomputed on mu_FB) used by B.9.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MI_DIR = REPO / "experiments" / "manifold-informer"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MI_DIR))

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle      # noqa: E402
from modules.surrogate.oracle_events import (                        # noqa: E402
    sample_events, event_log_likelihood_ratio, _angular_S_DoverS,
    _sample_costheta,
)
from manifold_informer import ManifoldInformer                       # noqa: E402

# Reuse B.9 constants so the physics setup is identical.
from _common import (                                                # noqa: E402
    N_WC, WITHHOLD_DIM, WITHHOLD_DIM_2, TARGET_C_LQ3, TARGET_C_HQ3,
    M_RANGE, SIGMA_Y, build_oracle, angular_fisher_V,
)

D_EMB = 16
D_HIDDEN = 32
ALPHA_RIDGE = 1e-3
FULL_CKPT = MI_DIR / "output_manifold_informer" / "manifold_informer.pt"
DEF_CKPT = MI_DIR / "output_integrated_loop" / "manifold_informer_deficient.pt"

# Events sampled per candidate mass to form one pooled ridge row.
N_EV_PER_CAND = 64
# Probe events used to recover the psi-extension u1 direction.
N_PROBE_EVENTS = 4000


def target_c() -> np.ndarray:
    c = np.zeros(N_WC)
    c[WITHHOLD_DIM] = TARGET_C_LQ3
    c[WITHHOLD_DIM_2] = TARGET_C_HQ3
    return c


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------
def load_full_encoder() -> ManifoldInformer:
    m = ManifoldInformer(d_event=2, d_emb=D_EMB, hidden=D_HIDDEN,
                         alpha=ALPHA_RIDGE, use_ema=True)
    m.load_state_dict(torch.load(FULL_CKPT, map_location="cpu",
                                 weights_only=True))
    m.eval()
    return m


def load_deficient_encoder() -> ManifoldInformer:
    m = ManifoldInformer(d_event=1, d_emb=D_EMB, hidden=D_HIDDEN,
                         alpha=ALPHA_RIDGE, use_ema=True)
    m.load_state_dict(torch.load(DEF_CKPT, map_location="cpu",
                                 weights_only=True))
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Per-candidate event sampling pinned to a mass, angular law from target c
# ---------------------------------------------------------------------------
def _events_at_mass(oracle: AnalyticSMEFTOracle, c_ang: np.ndarray,
                    m_tev: float, n_ev: int, seed: int) -> np.ndarray:
    """n_ev events all at mass m_tev, cos theta* drawn from the c_ang angular
    density at that mass. Returns (n_ev, 2) = (log m / M_ref, cos theta*).

    cos theta* law: dsigma/du propto (1+u^2) + 2 (D/S) u with D/S = (4/3) A_FB
    at (c_ang, m). This is exactly oracle_events._sample_costheta on the per-m
    D/S, so the angular structure that the d_event=2 encoder sees is the real
    target-working-point angular law, not a placeholder.
    """
    rng = np.random.default_rng(seed)
    m_arr = np.full(n_ev, float(m_tev))
    _, DoverS = _angular_S_DoverS(oracle, c_ang, m_arr)              # (n_ev,)
    u = _sample_costheta(DoverS, rng)
    out = np.empty((n_ev, 2), dtype=np.float64)
    out[:, 0] = np.log(np.maximum(m_arr, 1e-12) / 1.0)
    out[:, 1] = u
    return out


def _mass_seed(m_tev: float, base: int) -> int:
    """Deterministic per-mass seed so psi_np is a pure function of (M, base)."""
    return (base ^ (int(round(float(m_tev) * 1e6)) & 0x7FFFFFFF)) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# FM head: IntentionFM-compatible adapter on pooled per-event embeddings
# ---------------------------------------------------------------------------
class FMHead:
    """Adapter exposing the IntentionFM interface on the FM representation.

    psi_np(M): for each candidate mass m, sample N_EV events pinned to m
    (angular law = target c), encode per-event, mean-pool to one R^D row.
    Optionally append the psi-extension direction(s) (a constant per-row
    coefficient times the pooled cos-theta residual feature) -> R^{D+ext}.

    The pooling seed is fixed per (mass, pool_base) so the same mass always
    maps to the same row within a chain — A is then a deterministic function
    of the context masses, matching the precursor head's psi(m).
    """

    def __init__(self, encoder: ManifoldInformer, oracle: AnalyticSMEFTOracle,
                 c_ang: np.ndarray, *, n_ev: int = N_EV_PER_CAND,
                 pool_base: int = 12345, extension_dirs: np.ndarray | None = None):
        self.encoder = encoder
        self.oracle = oracle
        self.c_ang = np.asarray(c_ang, dtype=float)
        self.n_ev = n_ev
        self.pool_base = pool_base
        self.alpha = float(encoder.alpha)
        self.d_event = encoder.d_event
        # extension_dirs: (n_ext, d_emb_full_row) appended after pooling.
        self.extension_dirs = extension_dirs
        self.d_psi = D_EMB + (0 if extension_dirs is None
                              else extension_dirs.shape[0])

    @torch.no_grad()
    def _pooled_row(self, m_tev: float) -> np.ndarray:
        ev = _events_at_mass(self.oracle, self.c_ang, m_tev, self.n_ev,
                             self._row_seed(m_tev))
        d = self.encoder.d_event
        Xt = torch.from_numpy(ev[:, :d]).float()
        emb = self.encoder.event_encoder(Xt).cpu().numpy().astype(np.float64)
        row = emb.mean(axis=0)                                       # (d_emb,)
        if self.extension_dirs is not None:
            # psi-extension: append projection of the pooled row onto each
            # residual direction. With u1 a unit residual direction in R^d_emb,
            # the appended coordinate is (row . u1) — the learned-basis
            # coordinate along the recovered missing direction. Matches the
            # integrated_loop extend_psi semantics (append u1 as a new column
            # of psi; here psi is the pooled row so the appended entry is the
            # row's coordinate along u1).
            extra = self.extension_dirs @ row                       # (n_ext,)
            row = np.concatenate([row, extra])
        return row

    def _row_seed(self, m_tev: float) -> int:
        return _mass_seed(m_tev, self.pool_base)

    def psi_np(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_1d(np.asarray(M, dtype=float))
        return np.stack([self._pooled_row(float(m)) for m in M], axis=0)

    def A_inv_and_w(self, M_ctx: np.ndarray, Y_ctx: np.ndarray):
        Psi = self.psi_np(M_ctx)                                     # (K, D)
        D = Psi.shape[1]
        A = Psi.T @ Psi + self.alpha * np.eye(D)
        A_inv = np.linalg.inv(A)
        w = A_inv @ Psi.T @ np.asarray(Y_ctx, dtype=float)
        return A_inv, w, Psi


# ---------------------------------------------------------------------------
# psi-extension direction from the residual operator (def encoder)
# ---------------------------------------------------------------------------
@torch.no_grad()
def recover_extension_dirs(encoder: ManifoldInformer,
                           oracle: AnalyticSMEFTOracle,
                           c_queue: list[np.ndarray], *,
                           n_probe: int = N_PROBE_EVENTS, seed: int = 7777,
                           n_ext: int = 1) -> np.ndarray:
    """Recover the top residual-SVD direction(s) u1 in the encoder's R^d_emb,
    exactly as integrated_loop assembles R and ψ-extends.

    Build psi = per-event embeddings on SM probe events; for each working point
    c in the queue, residual of log w_c against the psi span is one column of R;
    SVD(R), take the top-n_ext left singular vectors. These live in R^{N_probe}
    (per-event space), but extend_psi appends u1 as a new psi column there. For
    the pooled-row adapter we instead need the direction in R^d_emb. We obtain
    it by regressing u1 (per-event) back onto the per-event embeddings:
    u1_emb = argmin || psi @ a - u1 ||, i.e. a = (psi^T psi)^-1 psi^T u1; the
    appended pooled coordinate is then row . a, which is the pooled projection
    onto the same recovered missing direction.

    Returns (n_ext, d_emb).
    """
    c_sm = np.zeros(N_WC)
    probe_events = sample_events(oracle, c_sm, n_probe, seed=seed)
    d = encoder.d_event
    Xt = torch.from_numpy(probe_events[:, :d]).float()
    psi = encoder.event_encoder(Xt).cpu().numpy().astype(np.float64)  # (Np, d_emb)
    A = psi.T @ psi + 1e-6 * np.eye(psi.shape[1])
    cols = []
    for c in c_queue:
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        w = np.linalg.solve(A, psi.T @ log_w)
        r = log_w - psi @ w
        cols.append(r)
    R = np.stack(cols, axis=1)                                       # (Np, Ncyc)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    dirs = []
    for j in range(min(n_ext, U.shape[1])):
        u = U[:, j]
        a = np.linalg.solve(A, psi.T @ u)                            # (d_emb,)
        n = np.linalg.norm(a)
        dirs.append(a / n if n > 0 else a)
    return np.stack(dirs, axis=0)


# ---------------------------------------------------------------------------
# Probe W: fit FM-ridge-weight -> c-tilde, rotate by Fisher V (recomputed)
# ---------------------------------------------------------------------------
def fit_fm_probe(head_factory, oracle: AnalyticSMEFTOracle, *,
                 c_box: float = 0.6, n_train: int = 400, K_ctx: int = 24,
                 seed: int = 4242) -> dict:
    """Fit W mapping the FM ridge weight w_theta(c) (R^D) to the Wilson vector.

    For each training working point c (drawn in a box), build a context of K
    masses, label with mu_FB(c, m), solve the FM ridge for w_theta(c) in R^D,
    and stack. Then least-squares fit c ~ W w_theta(c) over the grid, giving
    W of shape (N_WC, D). V is the Fisher rotation recomputed on mu_FB (the
    SAME V as B.9). P = V^T W.

    head_factory(c_ang) -> FMHead: builds a head whose angular law is the given
    c (the row sampler needs an angular working point; we use each training c
    as its own angular law so w_theta(c) is internally consistent).
    """
    rng = np.random.default_rng(seed)
    V, lam = angular_fisher_V(oracle)
    Ws = []
    Cs = []
    # NOTE (T0.2 / diagnostic D2): the fit-context MUST span the full M_RANGE,
    # not the thin seed band [0.5,1.0]. The ridge weight w(c) lives in a
    # context-dependent subspace; the sweep grows the context across the full
    # [0.3,2.3] pool, so a W fit on thin-band weights reads c̃ from the wrong
    # subspace and Σ = σ² P A⁻¹ Pᵀ stops tracking the contraction (it reads
    # ≈1.000 in every arm). This exactly reproduced the FM "zero contraction"
    # on the precursor head in d0_positive_control.py. The working precursor
    # probe (probe_efficiency.py:implicit_w) draws the context over the full
    # M_RANGE, adds an intercept column, and uses noised labels — matched here.
    for _ in range(n_train):
        c = rng.uniform(-c_box, c_box, size=N_WC)
        head = head_factory(c)
        M_ctx = rng.uniform(*M_RANGE, size=K_ctx)
        C = np.tile(c, (K_ctx, 1))
        Y = oracle.truth_mu_fb(C, M_ctx) + rng.normal(0.0, SIGMA_Y, size=K_ctx)
        _, w, _ = head.A_inv_and_w(M_ctx, Y)
        Ws.append(w)
        Cs.append(c)
    Wmat = np.stack(Ws, axis=0)                                     # (n_train, D)
    Cmat = np.stack(Cs, axis=0)                                     # (n_train, N_WC)
    # Augmented OLS with intercept: c ≈ W w + b. Keep only the slope W (the
    # offset b does not enter the covariance Σ = σ² P A⁻¹ Pᵀ).
    X = np.hstack([Wmat, np.ones((len(Wmat), 1))])                  # (n_train, D+1)
    Wb, *_ = np.linalg.lstsq(X, Cmat, rcond=None)                  # (D+1, N_WC)
    W = Wb[:-1].T                                                   # (N_WC, D)
    P = V.T @ W
    return {"W": W, "V": V, "lam_fisher": lam, "sigma_y": SIGMA_Y, "P": P,
            "rank_W": int(np.linalg.matrix_rank(W))}
