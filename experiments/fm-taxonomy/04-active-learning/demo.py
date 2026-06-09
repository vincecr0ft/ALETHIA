"""Active-learning vs random, in closed form, mirroring the ALETHIA head.

Self-contained (numpy only). Reproduces the conceptual distinctions of
writeup.md on a toy Bayesian linear-Gaussian inference problem of the same
shape as experiments/al-phoenix-studies/:

    y = phi(x)^T w + eps,   eps ~ N(0, sigma_y^2),   prior w ~ N(0, tau^2 I)

Posterior precision (design matrix) A = sigma_y^{-2} sum phi phi^T + tau^{-2} I,
posterior covariance of w is sigma_y^2 A^{-1} (up to the prior term). Every
acquisition score is a closed-form rank-one (Sherman-Morrison) update of A^{-1}
-- exactly the O(D^2)/cycle machinery the IntentionFM head exposes
(ig_per_candidate, sigma_ctilde, epig_acquire_m, param_epig_a_acquire in
modules/surrogate/intention/acquisition.py).

Two experiments:

  E1  WELL-POSED regime. A random feature map; the target direction is
      data-identifiable. Information-gain acquisition (D-optimal / BALD =
      leverage; EPIG on a target set) contracts the posterior AND reduces
      parameter / predictive error faster than random. This is the textbook
      "targeted beats random" result.

  E2  ALETHIA-SHAPED regime. A near-rank-deficient feature map makes the
      high-leverage candidates collinear in the A^{-1} metric (cos_Ainv -> 1)
      and the target direction weakly identified. D-optimal still contracts
      the *reported* posterior covariance faster than random, yet its point
      estimate on the targeted direction is NOT better -- it confidently
      shrinks around a biased centre. This is the documented contraction-vs-MLE
      gap. EPIG (prediction-targeted) tracks predictive error better than
      D-optimal here, illustrating the parameter-vs-prediction split.

Run:  python demo.py        (or: uv run python demo.py)
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Closed-form Bayesian linear-Gaussian learner (the "head").
# --------------------------------------------------------------------------


class LinearGaussianHead:
    """Posterior over w for y = phi^T w + eps. Mirrors A_inv_and_w."""

    def __init__(self, Phi_basis_fn, sigma_y: float, tau: float):
        self.phi = Phi_basis_fn          # x (n,) -> Phi (n, D)
        self.sigma_y = float(sigma_y)
        self.tau = float(tau)

    def A_inv_and_w(self, X_ctx, Y_ctx):
        """Return (A_inv, w_map, A). A = sy^-2 Phi^T Phi + tau^-2 I."""
        Phi = self.phi(X_ctx)                       # (K, D)
        D = Phi.shape[1]
        A = (Phi.T @ Phi) / self.sigma_y**2 + np.eye(D) / self.tau**2
        A_inv = np.linalg.inv(A)
        b = (Phi.T @ Y_ctx) / self.sigma_y**2
        w_map = A_inv @ b
        return A_inv, w_map, A


# --------------------------------------------------------------------------
# Acquisition scores (all closed form; one rank-one update each).
# --------------------------------------------------------------------------


def score_leverage(head, A_inv, X_pool):
    """BALD / D-optimal: 0.5 log(1 + phi^T A^{-1} phi). (_common.ig_per_candidate)"""
    Psi = head.phi(X_pool)
    lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
    return 0.5 * np.log1p(np.maximum(lev, 0.0))


def score_param_a(head, A_inv, X_pool, p_a):
    """A-optimal on direction p_a: -0.5 log(1 - sy^2 u_a^2 / ((1+lev) Sigma_aa)).

    Matches param_epig_a_acquire's score.
    """
    sy2 = head.sigma_y**2
    Psi = head.phi(X_pool)
    AP = Psi @ A_inv
    lev = np.einsum("pd,pd->p", AP, Psi)
    u_a = AP @ p_a
    Sigma_aa = sy2 * float(p_a @ A_inv @ p_a)
    arg = 1.0 - sy2 * (u_a**2) / ((1.0 + lev) * max(Sigma_aa, 1e-30))
    return -0.5 * np.log(np.maximum(arg, 1e-12))


def score_epig(head, A_inv, X_pool, X_target):
    """Predictive EPIG over a target set. Matches epig_acquire_m's score."""
    Psi_P = head.phi(X_pool)
    Psi_T = head.phi(X_target)
    lev_T = np.einsum("id,de,ie->i", Psi_T, A_inv, Psi_T)
    lev_P = np.einsum("pd,de,pe->p", Psi_P, A_inv, Psi_P)
    K_TP = Psi_T @ A_inv @ Psi_P.T                  # (n_T, n_P)
    var_red = K_TP**2 / (lev_P[None, :] + 1.0)
    denom = np.maximum(lev_T[:, None] - var_red, 1e-12)
    log_ratio = np.log(np.maximum(lev_T[:, None], 1e-12) / denom)
    return 0.5 * log_ratio.mean(axis=0)


def cos_ainv_top_decile(head, A_inv, X_pool):
    """Mean |cos| of top-decile-leverage candidates in the A^{-1} metric.

    The Stage-B redundancy diagnostic: ~1.0 means the high-IG subspace is 1-D.
    """
    Psi = head.phi(X_pool)
    lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
    k = max(2, len(lev) // 10)
    top = np.argsort(lev)[::-1][:k]
    G = Psi[top] @ A_inv @ Psi[top].T               # Gram in A^{-1} metric
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = np.abs(G / np.outer(d, d))
    iu = np.triu_indices_from(C, k=1)
    return float(C[iu].mean())


# --------------------------------------------------------------------------
# One AL loop. Returns per-round metrics.
# --------------------------------------------------------------------------


def run_loop(head, w_true, X_ctx, Y_ctx, X_pool, X_target, p_a,
             acq, rng, n_rounds, sigma_y, target_pred_x):
    """Greedy round-by-round acquisition. acq in {random,leverage,param_a,epig}."""
    X_ctx = list(X_ctx)
    Y_ctx = list(Y_ctx)
    avail = np.ones(len(X_pool), dtype=bool)
    rows = []
    for _ in range(n_rounds):
        A_inv, w_map, _ = head.A_inv_and_w(np.array(X_ctx), np.array(Y_ctx))
        # metric 1: posterior covariance along targeted direction (contraction)
        var_a = sigma_y**2 * float(p_a @ A_inv @ p_a)
        # metric 2: estimator error along targeted direction (the "MLE" gap)
        err_a = abs(float(p_a @ w_map) - float(p_a @ w_true))
        # metric 3: predictive RMSE on the target eval points
        Phi_pred = head.phi(target_pred_x)
        y_pred = Phi_pred @ w_map
        y_true = Phi_pred @ w_true
        pred_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        rows.append((var_a, err_a, pred_rmse))

        if not avail.any():
            break
        if acq == "random":
            cand = rng.choice(np.flatnonzero(avail))
        else:
            if acq == "leverage":
                s = score_leverage(head, A_inv, X_pool)
            elif acq == "param_a":
                s = score_param_a(head, A_inv, X_pool, p_a)
            elif acq == "epig":
                s = score_epig(head, A_inv, X_pool, X_target)
            else:
                raise ValueError(acq)
            s = np.where(avail, s, -np.inf)
            cand = int(np.argmax(s))
        x_new = X_pool[cand]
        y_new = float(head.phi(np.array([x_new]))[0] @ w_true
                      + rng.normal(0, sigma_y))
        X_ctx.append(x_new)
        Y_ctx.append(y_new)
        avail[cand] = False
    return np.array(rows)


# --------------------------------------------------------------------------
# Feature maps.
# --------------------------------------------------------------------------


def make_rff(D, freqs, phases):
    """Random Fourier features: well-conditioned, target identifiable."""
    def phi(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.sqrt(2.0 / D) * np.cos(np.outer(x, freqs) + phases)
    return phi


def make_collinear(D, freqs, phases, leak):
    """Near-rank-deficient map: all but a tiny `leak` of the variance lives in
    a single dominant feature, so high-leverage pool points become collinear in
    the A^{-1} metric -- the ALETHIA cos_Ainv -> 1 / weak-identifiability regime.
    """
    base = make_rff(D, freqs, phases)

    def phi(x):
        P = base(x)
        dom = P[:, :1]                      # one dominant direction
        rest = P[:, 1:] * leak             # everything else suppressed
        return np.concatenate([dom, rest], axis=1)
    return phi


# --------------------------------------------------------------------------


def setup(rng, D, x_lo, x_hi, n_pool, n_target, sigma_y, tau, collinear, leak):
    freqs = rng.uniform(0.5, 6.0, size=D)
    phases = rng.uniform(0, 2 * np.pi, size=D)
    phi = (make_collinear(D, freqs, phases, leak) if collinear
           else make_rff(D, freqs, phases))
    head = LinearGaussianHead(phi, sigma_y, tau)
    w_true = rng.normal(0, tau, size=D)
    X_pool = rng.uniform(x_lo, x_hi, size=n_pool)
    X_target = rng.uniform(x_hi - 0.25 * (x_hi - x_lo), x_hi, size=n_target)
    target_pred_x = np.linspace(x_hi - 0.25 * (x_hi - x_lo), x_hi, 60)
    # targeted parameter direction: the feature loading at a high-x eval point
    p_a = phi(np.array([x_hi]))[0]
    p_a = p_a / np.linalg.norm(p_a)
    # thin seed context in the low-x region (like ALETHIA's seed draw)
    K = 6
    X_ctx = rng.uniform(x_lo, x_lo + 0.25 * (x_hi - x_lo), size=K)
    Y_ctx = phi(X_ctx) @ w_true + rng.normal(0, sigma_y, size=K)
    return head, w_true, X_ctx, Y_ctx, X_pool, X_target, p_a, target_pred_x


def aggregate(head_kw, acqs, n_seeds, n_rounds, base_seed):
    """Average final metrics over seeds; return dict acq -> (var, err, rmse) arrays."""
    out = {a: [] for a in acqs}
    cos_vals = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        (head, w_true, X_ctx, Y_ctx, X_pool, X_target,
         p_a, target_pred_x) = setup(rng, **head_kw)
        A_inv0, _, _ = head.A_inv_and_w(X_ctx, Y_ctx)
        cos_vals.append(cos_ainv_top_decile(head, A_inv0, X_pool))
        for a in acqs:
            rng_a = np.random.default_rng(base_seed + s)  # same labels/noise
            rows = run_loop(head, w_true, X_ctx, Y_ctx, X_pool, X_target, p_a,
                            a, rng_a, n_rounds, head_kw["sigma_y"], target_pred_x)
            out[a].append(rows[-1])
    agg = {a: np.array(v) for a, v in out.items()}
    return agg, float(np.mean(cos_vals))


def zscore(target, ref):
    """Signed z of mean(target)-mean(ref); negative = target smaller (better)."""
    d = target.mean() - ref.mean()
    sd = np.sqrt(target.var(ddof=1) / len(target) + ref.var(ddof=1) / len(ref))
    return d / sd if sd > 0 else 0.0


def report(name, agg, cos_top, acqs):
    print(f"\n=== {name} ===")
    print(f"top-decile mean |cos_Ainv| at seed context: {cos_top:.3f}  "
          f"({'collinear/redundant' if cos_top > 0.7 else 'spread'})")
    ref = agg["random"]
    hdr = f"{'acq':>10} | {'Var(c_a)':>10} {'err(c_a)':>10} {'pred_RMSE':>10}"
    hdr += f" | {'zVar':>7} {'zErr':>7} {'zRMSE':>7}"
    print(hdr)
    print("-" * len(hdr))
    for a in acqs:
        v = agg[a]
        var_m, err_m, rmse_m = v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()
        if a == "random":
            zs = "   ref     ref     ref"
        else:
            zs = (f"{zscore(v[:, 0], ref[:, 0]):>7.1f} "
                  f"{zscore(v[:, 1], ref[:, 1]):>7.1f} "
                  f"{zscore(v[:, 2], ref[:, 2]):>7.1f}")
        print(f"{a:>10} | {var_m:>10.4g} {err_m:>10.4g} {rmse_m:>10.4g} | {zs}")
    print("z<0 => acquisition beats random on that metric (smaller is better); "
          "z>0 => worse.")


def main():
    n_seeds, n_rounds = 40, 30
    acqs = ["random", "leverage", "param_a", "epig"]

    e1 = dict(D=24, x_lo=0.0, x_hi=6.0, n_pool=200, n_target=40,
              sigma_y=0.05, tau=1.0, collinear=False, leak=0.0)
    agg1, cos1 = aggregate(e1, acqs, n_seeds, n_rounds, base_seed=1000)
    report("E1 WELL-POSED (target identifiable, features spread)",
           agg1, cos1, acqs)

    e2 = dict(D=24, x_lo=0.0, x_hi=6.0, n_pool=200, n_target=40,
              sigma_y=0.05, tau=1.0, collinear=True, leak=0.02)
    agg2, cos2 = aggregate(e2, acqs, n_seeds, n_rounds, base_seed=2000)
    report("E2 ALETHIA-SHAPED (collinear high-leverage ridge, weak ID)",
           agg2, cos2, acqs)

    print("\nReading:")
    print(" E1: leverage/param_a/epig should show zVar<0 AND zErr<0/zRMSE<0 ")
    print("     -> contraction propagates to the estimate. Textbook AL win.")
    print(" E2: leverage should show strongly zVar<0 (faster contraction) but ")
    print("     zErr>=0 (no better, often worse estimate) -> the contraction-")
    print("     vs-MLE gap. cos_Ainv ~ 1 flags the 1-D high-IG ridge that makes")
    print("     greedy parameter-info acquisition shrink around a biased centre.")


if __name__ == "__main__":
    main()
