"""ALETHIA physics modules.

Standalone, framework-agnostic building blocks (no ADK / Phoenix imports) that
are wired into the agent as ADK tools:

  * ``analytic_smeft``  — leading-order SMEFT Drell-Yan cross sections
  * ``surrogate``       — Intention foundation model + conformal calibration
                          + leverage / EPIG acquisition (see
                          ``docs/surrogate/`` for the math and integration
                          brief)
  * ``drift``           — drift detection                  (planned)
"""
