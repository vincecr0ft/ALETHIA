"""Analytic leading-order SMEFT Drell-Yan cross section calculator.

Closed-form neutral-current Drell-Yan (pp -> l+l-) at parton level, with
dimension-6 SMEFT operators in the Warsaw basis. No Pythia, Delphes, or
MadGraph — numpy + scipy only.

Public API:

* :func:`differential_xs` — pointwise ``d sigma / d m_ll`` at continuous ``m_ll``.
  Surrogates and integrators consume this directly.
* :func:`simulate_analytic_smeft` — bin-averaged ``d sigma / d{m_ll, pT_l}``
  over user-supplied bin edges. Convenience wrapper.
"""

from .smeft import differential_xs, simulate_analytic_smeft

__all__ = ["differential_xs", "simulate_analytic_smeft"]
