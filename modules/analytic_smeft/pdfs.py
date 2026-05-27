"""Parton distribution functions for the analytic SMEFT Drell-Yan calculator.

Two interchangeable backends behind a common :class:`PDFSet` interface:

* :class:`LHAPDFSet` — wraps the ``lhapdf`` library (e.g. ``CT18NNLO``) when it
  is installed. Use this for quantitative work.
* :class:`AnalyticPDF` — a self-contained toy ansatz ``x f(x) = N x^a (1-x)^b``
  with valence distributions normalised to quark counting. It has **no DGLAP
  evolution** (scale-independent) and is accurate only to order-of-magnitude
  for high-mass Drell-Yan. It exists so the module runs with numpy + scipy
  alone, and so the PDF choice is a swappable parameter.

Flavours use PDG ids: d=1, u=2, s=3, c=4, b=5; antiquarks are negative.
``xf(pid, x, Q)`` returns ``x * f(x)`` (the momentum-weighted PDF), accepting a
scalar or an array of ``x``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.special import beta as _beta


class PDFSet:
    """Abstract PDF backend. Subclasses implement :meth:`xf`."""

    name = "abstract"

    def xf(self, pid: int, x, Q: float):
        """Return ``x * f_pid(x, Q)``; ``x`` may be a scalar or ndarray."""
        raise NotImplementedError


class AnalyticPDF(PDFSet):
    """Toy proton PDF: ``x f(x) = N x^a (1-x)^b``, scale-independent.

    Valence ``u_v`` and ``d_v`` are normalised to integral 2 and 1 (quark
    counting). The light sea shares one shape; strange/charm/bottom are scaled
    fractions of it. Deliberately simple — see module docstring.
    """

    name = "analytic"

    # Exponents of x f(x) (not of the bare PDF).
    _A_VALENCE = 0.5
    _B_UVALENCE = 3.0
    _B_DVALENCE = 4.0
    _A_SEA = -0.2         # x f_sea ~ x^-0.2  =>  f_sea ~ x^-1.2 (realistic small-x rise)
    _B_SEA = 8.0
    _SEA_NORM = 0.055     # overall light-sea normalisation of x f
    _STRANGE_FRAC = 0.60  # s / light-sea suppression
    _CHARM_FRAC = 0.35
    _BOTTOM_FRAC = 0.10

    def __init__(self) -> None:
        # integral(f_v) = N * Beta(a, b+1)  since x f_v = N x^a (1-x)^b.
        self._n_uv = 2.0 / _beta(self._A_VALENCE, self._B_UVALENCE + 1.0)
        self._n_dv = 1.0 / _beta(self._A_VALENCE, self._B_DVALENCE + 1.0)

    def _x_valence(self, x, norm: float, b: float):
        return norm * x**self._A_VALENCE * (1.0 - x) ** b

    def _x_sea(self, x):
        return self._SEA_NORM * x**self._A_SEA * (1.0 - x) ** self._B_SEA

    def xf(self, pid: int, x, Q: float = 0.0):
        x = np.asarray(x, dtype=float)
        inside = (x > 0.0) & (x < 1.0)
        xs = np.where(inside, x, 0.5)  # dummy value where masked, avoids 0**neg

        sea = self._x_sea(xs)
        apid = abs(pid)
        if apid == 2:      # up
            val = self._x_valence(xs, self._n_uv, self._B_UVALENCE) if pid > 0 else 0.0
            result = val + sea
        elif apid == 1:    # down
            val = self._x_valence(xs, self._n_dv, self._B_DVALENCE) if pid > 0 else 0.0
            result = val + sea
        elif apid == 3:    # strange (sea only)
            result = self._STRANGE_FRAC * sea
        elif apid == 4:    # charm (sea only)
            result = self._CHARM_FRAC * sea
        elif apid == 5:    # bottom (sea only)
            result = self._BOTTOM_FRAC * sea
        else:              # gluon and everything else: irrelevant at LO DY
            result = np.zeros_like(xs)

        result = np.where(inside, result, 0.0)
        return float(result) if result.ndim == 0 else result


def _vendored_lhapdf_site() -> Optional[str]:
    """Locate a project-vendored LHAPDF install, preferring the MG-managed one.

    Two candidate paths, checked in order:

    1. ``vendor/MG5_aMC/HEPTools/lhapdf6_py3/`` — installed by MG's
       ``install lhapdf6`` command. This is the canonical path; MG owns
       LHAPDF for both Python (analytic xs) and Fortran (MG event gen).
    2. ``vendor/lhapdf/`` — legacy standalone build via
       ``scripts/install_lhapdf.sh``. Kept as fallback so the package stays
       functional during the migration; the script is on a deprecation path.

    Returns the ``site-packages`` directory holding the ``lhapdf`` module,
    else ``None``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        # MG-managed install. Distro python may pick dist-packages over
        # site-packages depending on activation env, so look for both.
        "vendor/MG5_aMC/HEPTools/lhapdf6_py3/lib/python*/site-packages",
        "vendor/MG5_aMC/HEPTools/lhapdf6_py3/lib/python*/dist-packages",
        # Legacy standalone build (deprecated; kept as fallback).
        "vendor/lhapdf/lib/python*/site-packages",
    ]
    for pattern in candidates:
        for site in sorted(repo_root.glob(pattern)):
            if (site / "lhapdf").is_dir():
                return str(site)
    return None


def _import_lhapdf():
    """Import ``lhapdf``, falling back to the project-vendored build.

    Raises ``ImportError`` if lhapdf is reachable by neither route.
    """
    site = _vendored_lhapdf_site()
    if site:
        if site not in sys.path:
            sys.path.insert(0, site)
        # Point LHAPDF at the vendored PDF data directory.
        share = Path(site).parents[2] / "share" / "LHAPDF"
        if share.is_dir():
            current = os.environ.get("LHAPDF_DATA_PATH", "")
            if str(share) not in current.split(os.pathsep):
                os.environ["LHAPDF_DATA_PATH"] = (
                    f"{share}{os.pathsep}{current}".rstrip(os.pathsep)
                )
    import lhapdf

    return lhapdf


class LHAPDFSet(PDFSet):
    """Wrapper around an ``lhapdf`` set (default ``CT18NNLO``).

    Uses ``lhapdf`` from the environment if importable, otherwise the
    project-vendored build under ``vendor/lhapdf`` (see ``make lhapdf``).
    """

    def __init__(self, name: str = "CT18NNLO", member: int = 0) -> None:
        lhapdf = _import_lhapdf()  # raises ImportError when lhapdf is unavailable
        self._pdf = lhapdf.mkPDF(name, member)
        self.name = name

    def xf(self, pid: int, x, Q: float):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        flat = np.atleast_1d(x)
        out = np.array(
            [self._pdf.xfxQ(pid, float(xi), float(Q)) if 0.0 < xi < 1.0 else 0.0
             for xi in flat]
        )
        return float(out[0]) if scalar else out.reshape(x.shape)


def get_pdf(spec: Union[str, PDFSet] = "auto") -> PDFSet:
    """Resolve a PDF spec to a :class:`PDFSet`.

    * a :class:`PDFSet` instance is returned unchanged;
    * ``"analytic"`` -> :class:`AnalyticPDF`;
    * ``"auto"`` -> :class:`LHAPDFSet` (CT18NNLO) if ``lhapdf`` imports, else
      :class:`AnalyticPDF`;
    * any other string -> :class:`LHAPDFSet` with that set name.
    """
    if isinstance(spec, PDFSet):
        return spec
    if spec == "analytic":
        return AnalyticPDF()
    if spec == "auto":
        try:
            return LHAPDFSet("CT18NNLO")
        except Exception:
            return AnalyticPDF()
    return LHAPDFSet(spec)
