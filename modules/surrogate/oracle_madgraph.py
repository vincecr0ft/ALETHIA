r"""MadGraph5_aMC@NLO oracle for the SMEFT modification factor mu(c, m).

Per-query MG runs against SMEFTsim_U35_alphaScheme_UFO. Each oracle call
sets the four Wilson coefficients (cHq3, cHq1, clq3, clq1) in
``param_card.dat``, narrows the dilepton-mass window in ``run_card.dat`` to
a band around the requested mass, launches ``generate_events`` in a
scrubbed environment, and parses the integrated cross section from the
banner. The SM cross section at each mass is cached, so subsequent
queries at the same m_tev hit the cache directly.

Configuration assumptions, reflecting the smoke-tested setup in
``vendor/MG5_aMC/processes/dy_smeft``:

* MG5_aMC v3.7.1 at ``vendor/MG5_aMC``
* SMEFTsim_U35_alphaScheme_UFO at ``vendor/MG5_aMC/models/``
* process: ``p p > mu+ mu- NP<=2`` (quadratic-EFT-order kept)
* sqrt(s) = 13 TeV, PDF = CT18NNLO (LHAID 14000)
* default lepton cuts: ``ptl > 10 GeV``, ``|etal| < 2.5`` (the physical
  acceptance an LHC analysis would apply)
* mass window per query: ``[m - m_window/2, m + m_window/2]`` in TeV

This is slow: each ``generate_events`` call is on the order of 10-60 s
depending on ``nevents``. With ``nevents=1000`` (the default here) MC
statistical noise on the parsed cross section is ~3%.

The adapter is otherwise drop-in compatible with the existing
:class:`AnalyticSMEFTOracle` and satisfies the surrogate's :class:`Oracle`
protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from .features import N_WC, WC_NAMES


# Surrogate WC name -> SMEFT block lhacode in the SMEFTsim param_card.
LHA_CODES: dict[str, int] = {
    "cHq3": 25,
    "cHq1": 24,
    "clq3": 36,
    "clq1": 35,
}
assert tuple(LHA_CODES) == WC_NAMES, (
    f"LHA_CODES order {tuple(LHA_CODES)} must match WC_NAMES {WC_NAMES}"
)

# Repo paths.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROCESS_DIR = _REPO_ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft"
_DEFAULT_CLEAN_ENV = _REPO_ROOT / "scripts" / "clean_env.sh"


# Regex that matches a single SMEFT-block line: leading whitespace, the
# lhacode, the (possibly-scientific) numeric value, then a `# <name>` tail.
_SMEFT_LINE_RE = re.compile(
    r"^(\s*)(\d+)(\s+)([+\-]?[\d\.eE+\-]+)(\s*#\s*\S+\s*)$"
)


def _zero_smeft_block(param_card_text: str) -> str:
    """Set every Wilson coefficient in the ``Block smeft`` to 0."""
    lines = param_card_text.splitlines(keepends=True)
    in_block = False
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.lstrip().lower()
        if stripped.startswith("block smeft "):
            in_block = True
            out.append(raw)
            continue
        if in_block and stripped.startswith("block "):
            in_block = False
            out.append(raw)
            continue
        if in_block:
            m = _SMEFT_LINE_RE.match(line)
            if m:
                pre1, code, pre2, _val, tail = m.groups()
                out.append(f"{pre1}{code}{pre2}0.000000e+00{tail}\n")
                continue
        out.append(raw)
    return "".join(out)


def _set_smeft_value(param_card_text: str, lhacode: int, value: float) -> str:
    """Set the single SMEFT-block line with the given lhacode to ``value``."""
    lines = param_card_text.splitlines(keepends=True)
    in_block = False
    out: list[str] = []
    target = str(lhacode)
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.lstrip().lower()
        if stripped.startswith("block smeft "):
            in_block = True
            out.append(raw)
            continue
        if in_block and stripped.startswith("block "):
            in_block = False
        if in_block:
            m = _SMEFT_LINE_RE.match(line)
            if m and m.group(2) == target:
                pre1, code, pre2, _val, tail = m.groups()
                out.append(f"{pre1}{code}{pre2}{value:.6e}{tail}\n")
                continue
        out.append(raw)
    return "".join(out)


def _set_run_card_mmll(run_card_text: str, m_lo_gev: float, m_hi_gev: float) -> str:
    """Replace the ``mmll`` and ``mmllmax`` lines in the run card."""
    out: list[str] = []
    for raw in run_card_text.splitlines(keepends=True):
        if " = mmll " in raw and "mmllmax" not in raw:
            out.append(f" {m_lo_gev:.4f} = mmll    "
                       "! min invariant mass of l+l- (same flavour) lepton pair\n")
        elif " = mmllmax" in raw:
            out.append(f" {m_hi_gev:.4f} = mmllmax "
                       "! max invariant mass of l+l- (same flavour) lepton pair\n")
        else:
            out.append(raw)
    return "".join(out)


_XSEC_RE = re.compile(r"#\s*Integrated weight \(pb\)\s*:\s*([+\-]?[\d\.eE+\-]+)")


def _patch_me5_config(process_dir: Path) -> None:
    """Force the process's me5_configuration.txt to disable browser-opening.

    MG5_aMC defaults `automatic_html_opening = True`, which spawns a browser
    tab per `generate_events` call. The fix sets both
    `automatic_html_opening = False` and `web_browser = None`. Called once
    per oracle init; idempotent. No-op if the keys are already set.
    """
    cfg = process_dir / "Cards" / "me5_configuration.txt"
    if not cfg.exists():
        return
    text = cfg.read_text()
    desired = {
        "automatic_html_opening": "False",
        "web_browser":            "None",
    }
    out = []
    seen = {k: False for k in desired}
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        # Match either "# key = ..." or "key = ..." and force to the desired value.
        for key, val in desired.items():
            if (stripped.startswith(f"{key} ") or stripped.startswith(f"# {key} ")
                    or stripped.startswith(f"{key}=")):
                # Skip any comment-prefix, normalise to "key = val\n".
                out.append(f"{key} = {val}\n")
                seen[key] = True
                break
        else:
            out.append(line)
    # If a key was missing entirely from the file, append it.
    for key, val in desired.items():
        if not seen[key]:
            out.append(f"{key} = {val}\n")
    new_text = "".join(out)
    if new_text != text:
        cfg.write_text(new_text)


_CACHE_VERSION = 1


def _cache_key(c: np.ndarray, m_tev: float, lambda_gev: float,
               m_window_tev: float, nevents: int) -> str:
    """Stable cache key from the oracle-call parameters."""
    payload = {
        "v":      _CACHE_VERSION,
        "c":      [round(float(x), 9) for x in np.asarray(c).ravel()],
        "m_tev":  round(float(m_tev), 9),
        "wnd":    round(float(m_window_tev), 9),
        "lam":    round(float(lambda_gev), 9),
        "nev":    int(nevents),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _parse_xs_from_banner(banner_path: Path) -> float:
    """Parse the integrated cross section in pb from a run banner."""
    text = banner_path.read_text()
    m = _XSEC_RE.search(text)
    if not m:
        raise RuntimeError(f"No 'Integrated weight (pb)' line in {banner_path}")
    return float(m.group(1))


class MadGraphSMEFTOracle:
    r"""Per-query MG oracle returning the SMEFT modification factor.

    For each row of ``c`` and matching entry of ``m`` (TeV), one MG run is
    launched. The integrated cross section over the narrow ``m`` window is
    divided by the cached SM cross section in the same window to give
    ``mu = sigma(c, m) / sigma_SM(m)``.

    Args:
      process_dir: path to the pre-generated MG process directory. Defaults
        to ``vendor/MG5_aMC/processes/dy_smeft``.
      clean_env: path to the bash scrubber used to sanitise the environment
        before invoking MG (strips conda contamination). Defaults to
        ``scripts/clean_env.sh``.
      lambda_gev: EFT scale in GeV; written into ``Block smeftcutoff``.
      m_window_tev: full width of the ``m_ll`` integration window per query.
      nevents: MG ``nevents``; trades MC noise for speed.
      verbose: if True, prints per-query timings to stdout.
    """

    def __init__(
        self,
        process_dir: Union[str, Path] = _DEFAULT_PROCESS_DIR,
        clean_env: Union[str, Path] = _DEFAULT_CLEAN_ENV,
        *,
        lambda_gev: float = 1000.0,
        m_window_tev: float = 0.100,
        nevents: int = 1000,
        verbose: bool = False,
        reuse_cache: bool = True,
        keep_artifacts: bool = False,
        cache_path: Union[str, Path, None] = None,
    ) -> None:
        """
        Extra knobs (all opt-in defaults match prior behaviour where possible):

        - ``reuse_cache``: read/write a persistent disk cache keyed on the
          full oracle-call parameters. Subsequent calls with identical
          (c, m, lambda, m_window, nevents) return the cached mu without
          touching MadGraph. Default True.
        - ``keep_artifacts``: if False (the default), delete the entire
          ``Events/run_NNN`` and ``HTML/run_NNN`` directories after each
          parse — only the cache entry persists. If True, retain the
          banner.txt of the latest run for debugging.
        - ``cache_path``: where to persist the cache as JSONL. Defaults
          to ``<process_dir>/.alethia_cache/oracle.jsonl``.
        """
        self.process_dir = Path(process_dir).resolve()
        self.clean_env = Path(clean_env).resolve()
        if not (self.process_dir / "bin" / "generate_events").exists():
            raise FileNotFoundError(
                f"MG process directory has no bin/generate_events: {self.process_dir}"
            )
        if not self.clean_env.exists():
            raise FileNotFoundError(f"clean_env wrapper not found: {self.clean_env}")
        self.lambda_gev = float(lambda_gev)
        self.m_window_tev = float(m_window_tev)
        self.nevents = int(nevents)
        self.verbose = bool(verbose)
        self.reuse_cache = bool(reuse_cache)
        self.keep_artifacts = bool(keep_artifacts)
        self._sigma_sm_cache: dict[float, float] = {}

        # Defensively disable browser-opening on every init. MG resets the
        # config in some workflows; this keeps it pinned.
        _patch_me5_config(self.process_dir)

        # Persistent cache (key -> mu).
        if cache_path is None:
            cache_path = self.process_dir / ".alethia_cache" / "oracle.jsonl"
        self.cache_path = Path(cache_path).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, float] = {}
        if self.reuse_cache and self.cache_path.exists():
            self._load_cache()

    # ---- cache I/O ----
    def _load_cache(self) -> None:
        with self.cache_path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if row.get("v") != _CACHE_VERSION:
                    continue
                k = row.get("key")
                mu = row.get("mu")
                if k is not None and mu is not None:
                    self._cache[k] = float(mu)

    def _append_cache(self, key: str, c: np.ndarray, m_tev: float,
                      mu: float, sigma_full_pb: float,
                      sigma_sm_pb: float) -> None:
        row = {
            "v":             _CACHE_VERSION,
            "key":           key,
            "c":             [float(x) for x in np.asarray(c).ravel()],
            "m_tev":         float(m_tev),
            "m_window_tev":  float(self.m_window_tev),
            "lambda_gev":    float(self.lambda_gev),
            "nevents":       int(self.nevents),
            "mu":            float(mu),
            "sigma_full_pb": float(sigma_full_pb),
            "sigma_sm_pb":   float(sigma_sm_pb),
            "ts":            datetime.now(timezone.utc).isoformat(),
        }
        with self.cache_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def cache_stats(self) -> dict:
        return {
            "entries": int(len(self._cache)),
            "path": str(self.cache_path),
            "size_bytes": (self.cache_path.stat().st_size
                           if self.cache_path.exists() else 0),
        }

    def clear_cache(self) -> int:
        """Wipe both the in-memory and on-disk cache. Returns # entries cleared."""
        n = len(self._cache)
        self._cache.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()
        return n

    def prune_artifacts(self) -> dict:
        """Delete every ``Events/run_NNN`` and ``HTML/run_NNN`` directory.

        MG accumulates these per call; the parsed cross section is the
        only thing the surrogate cares about, and it lives in the cache.
        Returns counts so the caller can audit.
        """
        ev_root = self.process_dir / "Events"
        html_root = self.process_dir / "HTML"
        ev_count = 0
        for p in (ev_root.iterdir() if ev_root.exists() else []):
            if p.is_dir() and p.name.startswith("run_"):
                shutil.rmtree(p, ignore_errors=True)
                ev_count += 1
        html_count = 0
        for p in (html_root.iterdir() if html_root.exists() else []):
            if p.is_dir() and p.name.startswith("run_"):
                shutil.rmtree(p, ignore_errors=True)
                html_count += 1
        return {"events_pruned": ev_count, "html_pruned": html_count}

    # ---- card editors ----
    def _write_cards(self, c: np.ndarray, m_tev: float) -> None:
        """Set param_card to (zeroed + four targeted) WCs and the lambda
        scale; narrow run_card's m_ll window to ``[m - w/2, m + w/2]``."""
        cards = self.process_dir / "Cards"

        param_text = (cards / "param_card.dat").read_text()
        param_text = _zero_smeft_block(param_text)
        for j, name in enumerate(WC_NAMES):
            param_text = _set_smeft_value(param_text, LHA_CODES[name], float(c[j]))
        # Lambda goes in Block smeftcutoff, lhacode 1.
        param_text = re.sub(
            r"(Block smeftcutoff[^\n]*\n\s*1\s+)[+\-]?[\d\.eE+\-]+",
            rf"\g<1>{self.lambda_gev:.6e}",
            param_text,
        )
        (cards / "param_card.dat").write_text(param_text)

        m_gev = float(m_tev) * 1000.0
        half = self.m_window_tev * 1000.0 * 0.5
        run_text = (cards / "run_card.dat").read_text()
        run_text = _set_run_card_mmll(run_text, m_gev - half, m_gev + half)
        # Ensure nevents is at our chosen value.
        run_text = re.sub(
            r"^\s*\d+\s*=\s*nevents.*$",
            f"  {self.nevents}  = nevents ! Number of unweighted events requested",
            run_text,
            count=1,
            flags=re.M,
        )
        (cards / "run_card.dat").write_text(run_text)

    # ---- MG invocation ----
    def _launch_and_parse(self) -> float:
        """Run ``generate_events -f`` in the scrubbed env, parse xs, then
        prune the run artefacts. With ``keep_artifacts=False`` (default),
        the entire Events/run_NNN and HTML/run_NNN directories are deleted
        once the cross section has been read; only the persistent cache
        retains the result.
        """
        events_dir = self.process_dir / "Events"
        html_dir = self.process_dir / "HTML"
        before = set(p.name for p in events_dir.iterdir()) if events_dir.exists() else set()

        t0 = time.perf_counter()
        cmd = ["bash", "-c",
               f". {self.clean_env} && cd {self.process_dir} && "
               "./bin/generate_events -f"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"generate_events failed:\nstdout:\n{exc.stdout.decode(errors='replace')}\n"
                f"stderr:\n{exc.stderr.decode(errors='replace')}"
            ) from exc

        after = set(p.name for p in events_dir.iterdir())
        new = sorted(after - before)
        if not new:
            raise RuntimeError("generate_events produced no new Events/run_* directory")
        latest = new[-1]
        run_dir = events_dir / latest
        banner = next(run_dir.glob("*_banner.txt"))
        xs = _parse_xs_from_banner(banner)

        # Aggressive cleanup so the process dir stays bounded under heavy use.
        if self.keep_artifacts:
            # Keep the banner of the latest run; drop event files and HTML.
            for f in run_dir.iterdir():
                if "banner" not in f.name:
                    f.unlink()
            (html_dir / latest).exists() and shutil.rmtree(
                html_dir / latest, ignore_errors=True)
            # Drop older Events runs entirely.
            for older in sorted(after - {latest})[:-1]:
                shutil.rmtree(events_dir / older, ignore_errors=True)
            for p in (html_dir.iterdir() if html_dir.exists() else []):
                if p.is_dir() and p.name.startswith("run_") and p.name != latest:
                    shutil.rmtree(p, ignore_errors=True)
        else:
            # Nuke EVERY run_NNN: in Events and in HTML. The result lives
            # in the persistent cache.
            for p in events_dir.iterdir():
                if p.is_dir() and p.name.startswith("run_"):
                    shutil.rmtree(p, ignore_errors=True)
            for p in (html_dir.iterdir() if html_dir.exists() else []):
                if p.is_dir() and p.name.startswith("run_"):
                    shutil.rmtree(p, ignore_errors=True)

        if self.verbose:
            print(f"  MG query: xs = {xs:.6e} pb in {time.perf_counter()-t0:.1f}s",
                  flush=True)
        return xs

    def _xs_for(self, c: np.ndarray, m_tev: float) -> float:
        self._write_cards(c, m_tev)
        return self._launch_and_parse()

    def _sigma_sm_at(self, m_tev: float) -> float:
        """SM cross section over the ``[m - w/2, m + w/2]`` window.

        Exact-cached entries (e.g. from :meth:`precompute_sm`) are returned
        as-is. Otherwise log-linear interpolation of the precomputed grid
        is used when ``m_tev`` lies inside it; otherwise a fresh MG run
        fills the cache for this exact ``m_tev``.
        """
        key = round(float(m_tev), 6)
        if key in self._sigma_sm_cache:
            return self._sigma_sm_cache[key]
        # Try interpolating from precomputed grid points, if it brackets m.
        keys = sorted(self._sigma_sm_cache)
        if keys and keys[0] <= key <= keys[-1]:
            log_m = np.log(np.asarray(keys, dtype=float))
            log_x = np.log(np.asarray([self._sigma_sm_cache[k] for k in keys], dtype=float))
            return float(np.exp(np.interp(np.log(key), log_m, log_x)))
        # Out of range: fall back to a fresh MG run and cache.
        self._sigma_sm_cache[key] = self._xs_for(np.zeros(N_WC), m_tev)
        return self._sigma_sm_cache[key]

    def precompute_sm(self, m_grid_tev: np.ndarray) -> dict[float, float]:
        """Populate the SM-xs cache on a fixed grid of ``m_tev`` values.

        After this call, :meth:`truth` and ``__call__`` no longer trigger
        an MG run for the SM denominator at any ``m`` inside the grid;
        log-linear interpolation is used between grid points.
        """
        m_grid = np.atleast_1d(np.asarray(m_grid_tev, dtype=float))
        for m_tev in m_grid:
            key = round(float(m_tev), 6)
            if key not in self._sigma_sm_cache:
                self._sigma_sm_cache[key] = self._xs_for(np.zeros(N_WC), m_tev)
        return dict(self._sigma_sm_cache)

    # ---- Oracle protocol ----
    def truth(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Pointwise mu(c, m). One MG run per cache-miss row of c (plus one
        cached SM run per unique m). c shape ``(n, N_WC)``, m shape
        ``(n,)`` in TeV. Returns shape ``(n,)``. When ``reuse_cache=True``
        (default), repeat calls with identical parameters return cached mu
        without launching MadGraph.
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
            key = _cache_key(c[i], m_tev[i], self.lambda_gev,
                             self.m_window_tev, self.nevents)
            if self.reuse_cache and key in self._cache:
                mu[i] = self._cache[key]
                if self.verbose:
                    print(f"  MG cache HIT  key={key}  mu={mu[i]:.6e}",
                          flush=True)
                continue
            sm = self._sigma_sm_at(m_tev[i])
            full = self._xs_for(c[i], m_tev[i])
            mu[i] = full / sm
            if self.reuse_cache:
                self._cache[key] = float(mu[i])
                self._append_cache(key, c[i], m_tev[i], mu[i], full, sm)
        return mu

    def __call__(
        self,
        c: np.ndarray,
        m: np.ndarray,
        *,
        noise: bool = True,
    ) -> np.ndarray:
        """MG already has MC statistical noise baked in at ``nevents``; the
        ``noise`` flag is accepted for protocol compatibility but does not
        change the call (use a larger ``nevents`` in the constructor for
        less noise)."""
        return self.truth(c, m)
