r"""Bridge from the analytic SMEFT Drell-Yan calculator to the surrogate's
:class:`Oracle` protocol.

The Intention FM consumes a scalar ``mu(c, m) = sigma_BSM(c, m) / sigma_SM(m)``
at continuous ``(c, m)`` points; there is no binning anywhere in the model
(see ``docs/surrogate/REFERENCE.md``). This adapter wraps the analytic LO
calculator's pointwise :func:`differential_xs` and exposes it as an
:class:`Oracle`.

The four Wilson coefficients the surrogate sees -- ``("cHq3", "cHq1",
"clq3", "clq1")`` -- map one-to-one to Warsaw-basis names in the
analytic calculator. Other dim-6 operators it supports are held at zero
here; extend ``WC_NAME_MAP`` if you want to expand ``N_WC``.

Mass convention: the surrogate uses ``m_ll`` in TeV (``M_REF = 1`` TeV in
:mod:`modules.surrogate.features`); the analytic calculator uses GeV.
Conversion happens inside this adapter so callers stay in TeV throughout.

The SMEFT cross-section ratio is PDF-independent at LO, so the demo can
use the analytic toy PDF without losing accuracy on ``mu``; switch to
``pdf="CT18NNLO"`` if you also want absolute cross-sections.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from modules.analytic_smeft import differential_xs
from modules.analytic_smeft.pdfs import PDFSet

from .features import N_WC, WC_NAMES


# Surrogate Wilson-coefficient names -> Warsaw-basis names in analytic_smeft.
WC_NAME_MAP: dict[str, str] = {
    "cHq3": "c_phi_q^(3)",
    "cHq1": "c_phi_q^(1)",
    "clq3": "c_lq^(3)",
    "clq1": "c_lq^(1)",
}
assert tuple(WC_NAME_MAP) == WC_NAMES, (
    f"WC_NAME_MAP order {tuple(WC_NAME_MAP)} must match surrogate WC_NAMES {WC_NAMES}"
)


class AnalyticSMEFTOracle:
    r"""Pointwise ``mu(c, m)`` from the analytic SMEFT Drell-Yan calculator.

    Implements the :class:`Oracle` protocol:

    * ``__call__(c, m, *, noise=True) -> (n,)``  — noisy observation.
    * ``truth(c, m)                 -> (n,)``    — noiseless physical value.

    Args:
      sqrt_s_gev: hadronic centre-of-mass energy in GeV.
      lambda_scale_gev: EFT scale Lambda in GeV.
      order: ``"linear"`` or ``"quadratic"`` in the EFT expansion.
      pdf: PDF spec passed to :func:`differential_xs`. ``"analytic"`` is
        fastest and is exact for ``mu`` ratios at LO.
      noise_frac: relative MC-like noise. ``mu_noisy = mu + N(0, noise_frac * |mu|)``.
      seed: RNG seed for the noise stream.
    """

    def __init__(
        self,
        *,
        sqrt_s_gev: float = 13000.0,
        lambda_scale_gev: float = 1000.0,
        order: str = "quadratic",
        pdf: Union[str, PDFSet] = "analytic",
        noise_frac: float = 0.05,
        seed: int = 0,
    ) -> None:
        if order not in ("linear", "quadratic"):
            raise ValueError(f"order must be 'linear' or 'quadratic', got {order!r}.")
        self.sqrt_s = float(sqrt_s_gev)
        self.lam = float(lambda_scale_gev)
        self.order = order
        self.pdf = pdf
        self.noise_frac = float(noise_frac)
        self._noise_rng = np.random.default_rng(seed)

    def truth(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Noiseless ``mu(c, m) = sigma(c, m) / sigma_SM(m)``.

        ``c`` is ``(n, N_WC)`` in canonical surrogate order; ``m`` is ``(n,)``
        in TeV. Returns shape ``(n,)``.
        """
        c = np.atleast_2d(np.asarray(c, dtype=float))
        m_tev = np.atleast_1d(np.asarray(m, dtype=float))
        if c.shape[1] != N_WC:
            raise ValueError(f"c must have {N_WC} columns, got {c.shape[1]}")
        if c.shape[0] != m_tev.shape[0]:
            raise ValueError(
                f"c has {c.shape[0]} rows but m has {m_tev.shape[0]} entries"
            )

        mu = np.empty(c.shape[0], dtype=float)
        for i in range(c.shape[0]):
            wc = {WC_NAME_MAP[name]: float(c[i, j]) for j, name in enumerate(WC_NAMES)}
            res = differential_xs(
                wc,
                np.array([m_tev[i] * 1000.0]),  # TeV -> GeV
                sqrt_s=self.sqrt_s,
                lambda_scale=self.lam,
                order=self.order,
                pdf=self.pdf,
            )
            mu[i] = float(res["differential_xs"][0] / res["sm_only"][0])
        return mu

    def __call__(
        self,
        c: np.ndarray,
        m: np.ndarray,
        *,
        noise: bool = True,
    ) -> np.ndarray:
        mu = self.truth(c, m)
        if noise and self.noise_frac > 0.0:
            mu = mu + self._noise_rng.normal(0.0, self.noise_frac * np.abs(mu))
        return mu
