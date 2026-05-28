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

from modules.analytic_smeft import differential_afb, differential_xs, differential_xs_pt
from modules.analytic_smeft.pdfs import PDFSet, get_pdf

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
        # Resolve the PDF spec once at construction so every truth() call
        # reuses the same PDFSet instance instead of re-running mkPDF.
        self.pdf: Union[str, PDFSet] = get_pdf(pdf) if isinstance(pdf, str) else pdf
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

    def truth_pt(self, c: np.ndarray, pt: np.ndarray) -> np.ndarray:
        r"""Noiseless ``mu_pT(c, pT) = (d sigma_BSM / d pT) / (d sigma_SM / d pT)``.

        Companion to :meth:`truth` exposing the lepton transverse-momentum
        spectrum. The angular integral that defines ``d sigma / d pT`` carries
        different operator weights from ``d sigma / d m_ll`` because the SMEFT
        chirality structure shows up in the cos(theta*) distribution; this is
        the multi-observable identifiability channel that breaks the
        ``c_Hq^(3)``--``c_Hq^(1)`` and ``c_lq^(3)``--``c_lq^(1)`` degeneracies
        m_ll alone cannot resolve.

        ``c`` is ``(n, N_WC)`` in canonical surrogate order; ``pt`` is
        ``(n,)`` in TeV. Returns shape ``(n,)``.
        """
        c = np.atleast_2d(np.asarray(c, dtype=float))
        pt_tev = np.atleast_1d(np.asarray(pt, dtype=float))
        if c.shape[1] != N_WC:
            raise ValueError(f"c must have {N_WC} columns, got {c.shape[1]}")
        if c.shape[0] != pt_tev.shape[0]:
            raise ValueError(
                f"c has {c.shape[0]} rows but pt has {pt_tev.shape[0]} entries"
            )
        mu = np.empty(c.shape[0], dtype=float)
        for i in range(c.shape[0]):
            wc = {WC_NAME_MAP[name]: float(c[i, j]) for j, name in enumerate(WC_NAMES)}
            res = differential_xs_pt(
                wc,
                np.array([pt_tev[i] * 1000.0]),     # TeV -> GeV
                sqrt_s=self.sqrt_s,
                lambda_scale=self.lam,
                order=self.order,
                pdf=self.pdf,
            )
            mu[i] = float(res["differential_xs"][0] / res["sm_only"][0])
        return mu

    def truth_mu_fb(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        r"""Noiseless ``mu_FB(c, m_ll)`` = BSM-relative FB asymmetry numerator.

        Defined analogously to ``truth(c, m)``:
        ``mu_FB(c, m_ll) = (sigma_F(c, m_ll) - sigma_B(c, m_ll)) /
                          (sigma_F_SM(m_ll) - sigma_B_SM(m_ll))``.

        This is the multiplicative analogue of ``mu = sigma_BSM / sigma_SM``
        but for the chirality-asymmetric piece of the cross section. Unlike
        the dimensionless ``A_FB(c, m)`` ratio (which has a complicated
        SM baseline curve the encoder must learn), ``mu_FB`` divides out
        the SM kinematic dependence by construction, so the encoder only
        needs to learn the operator-induced multiplicative shift.

        Returns shape ``(n,)``.
        """
        c = np.atleast_2d(np.asarray(c, dtype=float))
        m_tev = np.atleast_1d(np.asarray(m, dtype=float))
        if c.shape[1] != N_WC:
            raise ValueError(f"c must have {N_WC} columns, got {c.shape[1]}")
        if c.shape[0] != m_tev.shape[0]:
            raise ValueError(
                f"c has {c.shape[0]} rows but m has {m_tev.shape[0]} entries"
            )
        mu_fb = np.empty(c.shape[0], dtype=float)
        for i in range(c.shape[0]):
            wc = {WC_NAME_MAP[name]: float(c[i, j]) for j, name in enumerate(WC_NAMES)}
            res = differential_afb(
                wc,
                np.array([m_tev[i] * 1000.0]),     # TeV -> GeV
                sqrt_s=self.sqrt_s,
                lambda_scale=self.lam,
                order=self.order,
                pdf=self.pdf,
            )
            num_total = float(res["afb_numerator"][0])
            # SM-only FB numerator: A_FB_SM * sigma_SM.
            num_sm = float(res["A_FB_SM"][0]) * float(res["sm_xs"][0])
            mu_fb[i] = num_total / max(abs(num_sm), 1e-30) * (1.0 if num_sm > 0 else -1.0)
        return mu_fb

    def truth_afb(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        r"""Noiseless ``A_FB(c, m_ll)`` (forward-backward asymmetry).

        The chirality-asymmetric piece of the cross section, ``A_FB(m_ll)``,
        is the observable that distinguishes left-handed from right-handed
        quark couplings. The vertex operators ``c_phi_q^(3)`` and
        ``c_phi_q^(1)`` modify these couplings differently and the four-
        fermion operators ``c_lq^(3)`` and ``c_lq^(1)`` likewise produce
        distinct chirality structures; ``A_FB(m_ll)`` is therefore the
        physical observable that breaks the residual identifiability
        degeneracy reported by the kinematic-only Intention head.

        Returns ``A_FB(c, m_ll)`` in ``[-1, 1]`` at every requested ``m_ll``.
        """
        c = np.atleast_2d(np.asarray(c, dtype=float))
        m_tev = np.atleast_1d(np.asarray(m, dtype=float))
        if c.shape[1] != N_WC:
            raise ValueError(f"c must have {N_WC} columns, got {c.shape[1]}")
        if c.shape[0] != m_tev.shape[0]:
            raise ValueError(
                f"c has {c.shape[0]} rows but m has {m_tev.shape[0]} entries"
            )
        afb = np.empty(c.shape[0], dtype=float)
        for i in range(c.shape[0]):
            wc = {WC_NAME_MAP[name]: float(c[i, j]) for j, name in enumerate(WC_NAMES)}
            res = differential_afb(
                wc,
                np.array([m_tev[i] * 1000.0]),     # TeV -> GeV
                sqrt_s=self.sqrt_s,
                lambda_scale=self.lam,
                order=self.order,
                pdf=self.pdf,
            )
            afb[i] = float(res["A_FB"][0])
        return afb
