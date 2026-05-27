r"""
Intention foundation model: closed-form Bayesian linear regression on the
joint Wilson-times-observable feature space.

The model is exactly the Q (K^T K + lam I)^{-1} K^T V form, with K = Phi
(training feature matrix), V = Y (training targets), Q = phi(x*) (query
feature row). Predictive mean is unique; predictive variance combines
aleatoric (heteroscedastic on |mu|) and epistemic (leverage) parts.

This is the surrogate. It is deliberately deterministic: no SGD, no
randomness, no LLM. Gemini calls .predict / .leverage / .update / .acquire
as deterministic tools and runs the orchestration logic outside.

Cost: fit is :math:`O(N d^2 + d^3)`; predict is :math:`O(N d^2)`; leverage
is :math:`O(N d^2)`; update is a refit; acquire (in acquisition.py) uses
Sherman-Morrison so each pick is :math:`O(d^2)`.
"""
from __future__ import annotations

import numpy as np

from .features import phi_joint


class IntentionFM:
    """Closed-form ridge regression with calibrated heteroscedastic noise."""

    def __init__(self, lam: float = 1e-3):
        self.lam = lam
        # Fitted state. None before .fit() is called.
        self.w:          np.ndarray | None = None     # (D_JOINT,)
        self.A_inv:      np.ndarray | None = None     # (D_JOINT, D_JOINT)
        self.noise_frac: float | None      = None
        self.C_train:    np.ndarray | None = None
        self.M_train:    np.ndarray | None = None
        self.Y_train:    np.ndarray | None = None

    # ---- fit ----
    def fit(self, C: np.ndarray, M: np.ndarray, Y: np.ndarray) -> "IntentionFM":
        Phi   = phi_joint(C, M)
        d     = Phi.shape[1]
        A     = Phi.T @ Phi + self.lam * np.eye(d)
        self.A_inv = np.linalg.inv(A)
        self.w     = self.A_inv @ Phi.T @ Y
        mu_tr      = Phi @ self.w
        # Robust heteroscedastic noise fraction via MAD on relative residuals.
        rel = (Y - mu_tr) / np.maximum(np.abs(mu_tr), 1e-3)
        mad = float(np.median(np.abs(rel - np.median(rel))))
        self.noise_frac = max(1.4826 * mad, 1e-3)
        self.C_train, self.M_train, self.Y_train = C, M, Y
        return self

    # ---- predict ----
    def predict(self, C: np.ndarray, M: np.ndarray, return_std: bool = True):
        """Predictive mean and (optionally) per-point std.

        ``sigma_total^2 = (noise_frac * |mu|)^2 * (1 + leverage)``.
        """
        Phi  = phi_joint(C, M)
        mu   = Phi @ self.w
        if not return_std:
            return mu
        lev  = np.einsum("nd,de,ne->n", Phi, self.A_inv, Phi)
        aleo = (self.noise_frac * np.abs(mu)) ** 2
        return mu, np.sqrt(aleo * (1.0 + lev))

    # ---- leverage (drift signal) ----
    def leverage(self, C: np.ndarray, M: np.ndarray) -> np.ndarray:
        Phi = phi_joint(C, M)
        return np.einsum("nd,de,ne->n", Phi, self.A_inv, Phi)

    # ---- online update by refit (still sub-millisecond at this size) ----
    def update(self, C_new: np.ndarray, M_new: np.ndarray, Y_new: np.ndarray) -> "IntentionFM":
        return self.fit(
            np.vstack([self.C_train, C_new]),
            np.concatenate([self.M_train, M_new]),
            np.concatenate([self.Y_train, Y_new]),
        )

    # ---- serialization helpers (for persisting model versions across deploys) ----
    def state_dict(self) -> dict:
        return {
            "lam":         self.lam,
            "w":           self.w,
            "A_inv":       self.A_inv,
            "noise_frac":  self.noise_frac,
            "C_train":     self.C_train,
            "M_train":     self.M_train,
            "Y_train":     self.Y_train,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "IntentionFM":
        m = cls(lam=state["lam"])
        for k, v in state.items():
            setattr(m, k, v)
        return m
