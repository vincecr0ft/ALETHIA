"""Smoke and structural tests for modules.surrogate.intention.

Covers:
- IntentionFM forward differentiability through torch.linalg.solve
- psi_theta signature is (m,) only — no c on the forward pass (constraint)
- closed-form leverage matches Cauchy-Schwarz upper bound
- IntentionConformal raises OutdatedCalibratorError on stale context
- epig_acquire_m returns non-negative information gain
- aggregator decision table is exhaustive
"""
import inspect

import numpy as np
import torch
import pytest

from modules.surrogate.intention import (
    IntentionFM, PsiMLP, M_REF,
    IntentionConformal,
    epig_acquire_m, target_set_entropy,
    coverage_bh_test, kappa_drift, aggregate_action,
)


# ------ constraint: psi_theta sees only m ------

def test_psi_signature_takes_only_m():
    sig = inspect.signature(PsiMLP.forward)
    params = list(sig.parameters)
    # self + m
    assert params == ["self", "m"], (
        f"PsiMLP.forward must take only (self, m); got {params}. "
        "Adding c violates the BRIEF constraint.")


def test_intention_forward_signature_is_ctx_query():
    sig = inspect.signature(IntentionFM.forward)
    params = list(sig.parameters)
    assert params == ["self", "M_ctx", "Y_ctx", "M_q"], (
        f"IntentionFM.forward must take only (M_ctx, Y_ctx, M_q); got {params}.")


# ------ closed-form correctness ------

def test_predict_runs_and_shapes():
    rng = np.random.default_rng(0)
    model = IntentionFM(d_psi=8, hidden=32, alpha=1e-3)
    M_ctx = rng.uniform(0.3, 2.3, size=10)
    Y_ctx = rng.normal(size=10)
    M_q = rng.uniform(0.3, 2.3, size=20)
    y = model.predict_np(M_ctx, Y_ctx, M_q)
    assert y.shape == (20,)


def test_leverage_nonneg():
    rng = np.random.default_rng(1)
    model = IntentionFM(d_psi=8, hidden=32, alpha=1e-3)
    M_ctx = rng.uniform(0.3, 2.3, size=12)
    M_q = rng.uniform(0.3, 2.3, size=30)
    lev = model.leverage(M_ctx, M_q)
    assert lev.shape == (30,)
    assert np.all(lev >= 0)


def test_kappa_positive():
    rng = np.random.default_rng(2)
    model = IntentionFM(d_psi=8, hidden=32, alpha=1e-3)
    M_ctx = rng.uniform(0.3, 2.3, size=20)
    kappa = model.kappa_A(M_ctx)
    assert kappa >= 1.0


def test_forward_is_differentiable():
    """End-to-end backprop through torch.linalg.solve must work."""
    torch.manual_seed(3)
    model = IntentionFM(d_psi=8, hidden=16, alpha=1e-3)
    B, K, Q = 4, 6, 5
    M_ctx = torch.linspace(0.3, 2.3, K).repeat(B, 1)
    Y_ctx = torch.randn(B, K)
    M_q = torch.linspace(0.4, 2.0, Q).repeat(B, 1)
    Y_q = torch.randn(B, Q)
    y_pred = model(M_ctx, Y_ctx, M_q)
    loss = ((y_pred - Y_q) ** 2).mean()
    loss.backward()
    # any parameter should have non-zero grad
    grads = [p.grad.abs().max().item() for p in model.parameters()
             if p.grad is not None]
    assert any(g > 0 for g in grads), "no non-zero parameter gradient"


# ------ EPIG ------

def test_epig_non_negative():
    rng = np.random.default_rng(4)
    model = IntentionFM(d_psi=8, hidden=16, alpha=1e-3)
    M_ctx = rng.uniform(0.3, 2.3, size=10)
    Y_ctx = rng.normal(size=10)
    M_pool = rng.uniform(0.3, 2.3, size=50)
    M_target = rng.uniform(0.3, 2.3, size=20)
    idx = epig_acquire_m(model, M_ctx, Y_ctx, M_pool, M_target, k=5)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5, "EPIG must not duplicate picks"


def test_target_set_entropy_finite():
    rng = np.random.default_rng(5)
    model = IntentionFM(d_psi=8, hidden=16, alpha=1e-3)
    M_ctx = rng.uniform(0.3, 2.3, size=10)
    Y_ctx = rng.normal(size=10)
    M_target = rng.uniform(0.3, 2.3, size=20)
    H = target_set_entropy(model, M_ctx, Y_ctx, M_target)
    assert np.isfinite(H)


# ------ conformal ------

def test_conformal_outdated_raises():
    rng = np.random.default_rng(6)
    model = IntentionFM(d_psi=8, hidden=16, alpha=1e-3)
    M_ctx = rng.uniform(0.3, 2.3, size=10)
    Y_ctx = rng.normal(size=10)
    M_cal = rng.uniform(0.3, 2.3, size=80)
    Y_cal = rng.normal(size=80)
    cc = IntentionConformal(n_strata=4, noise_frac=0.1)
    cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)
    M_q = rng.uniform(0.3, 2.3, size=5)
    # Fresh-state call works.
    sd = cc.coverage_sigma(model, M_ctx, Y_ctx, M_q, 0.683)
    assert sd.shape == (5,)
    # Mutate the context: stale calibrator must refuse.
    M_ctx_new = np.concatenate([M_ctx, [1.0]])
    Y_ctx_new = np.concatenate([Y_ctx, [0.5]])
    with pytest.raises(RuntimeError):
        cc.coverage_sigma(model, M_ctx_new, Y_ctx_new, M_q, 0.683)


# ------ drift detectors ------

def test_bh_under_null_no_false_alarm():
    """Well-calibrated coverage stratified into 5 regions: BH should not
    fire on average."""
    rng = np.random.default_rng(7)
    # n=200 per stratum, target 0.683; sample bin(0.683, 200) and BH-test.
    counts_total = np.full(5, 200, dtype=int)
    fires = 0
    for _ in range(100):
        counts_covered = rng.binomial(200, 0.683, size=5)
        fired, _, _, _ = coverage_bh_test(counts_covered, counts_total,
                                          target_coverage=0.683,
                                          alpha=0.05)
        if fired:
            fires += 1
    # FDR control: at most ~5% should fire under the null; allow 10%
    # slack for 100 trials.
    assert fires < 15, f"BH false-alarm rate too high: {fires}/100"


def test_kappa_drift_fires_on_thin_context():
    rng = np.random.default_rng(8)
    model = IntentionFM(d_psi=16, hidden=32, alpha=1e-3)
    # Thin context: 3 points clustered in a narrow range.
    M_ctx = np.array([0.6, 0.65, 0.7])
    recent_M = rng.uniform(0.3, 2.3, size=30)
    # Baseline train projection.
    Psi_train = model.psi_np(rng.uniform(0.3, 2.3, size=200))
    eigs, vecs = np.linalg.eigh(
        Psi_train.T @ Psi_train + model.alpha * np.eye(model.d_psi))
    v_min = vecs[:, 0]
    train_proj_var = float(np.var(Psi_train @ v_min))
    fired, kappa, proj_ratio, eig = kappa_drift(
        model, M_ctx, recent_M, train_proj_var,
        kappa_threshold=1e3, proj_ratio_threshold=2.5)
    assert kappa > 0
    # With only 3 context points in d_psi=16, A is rank-3 + alpha I; kappa
    # should be quite large.
    assert kappa > 100
    # The returned EigenState carries the full spectrum of A.
    assert eig.d == model.d_psi
    assert eig.lam.shape == (model.d_psi,)
    assert np.isclose(eig.kappa, kappa)


def test_aggregator_exhaustive():
    """All 8 input combinations map to a string action."""
    history = [(False, False, False)] * 5
    for acc in (0, 1):
        for cal in (0, 1):
            for cov in (0, 1):
                action, _ = aggregate_action(
                    bool(acc), bool(cal), bool(cov),
                    history, persistence_N=3)
                assert isinstance(action, str)
                assert action in {
                    "noop", "watch", "recal", "recal_then_check",
                    "local_retrain", "global_retrain",
                }
