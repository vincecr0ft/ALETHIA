"""Phoenix-side rubric: query `chain.drift.eigen` spans for the LIS gap.

Implements the MCP rubric specified in
[upgrade-plan.md §5.2](../../docs/research/upgrade-plan.md) and
[upgrade-architecture.md §2.3](../../docs/research/upgrade-architecture.md):

    since cycle T0, return drift firings where
      any(lambda_j < theta * lambda_max)
    group by the eigenvector u_j responsible
    return the union of {u_j} as the "current LIS gap subspace"

The script POSTs a GraphQL query against the Phoenix instance pointed
to by ``PHOENIX_COLLECTOR_ENDPOINT`` (defaulting to localhost:6006),
pulls the spans named ``chain.drift.eigen`` since a cycle threshold,
parses the ``aletheia.eigen.*`` attributes, identifies cycles whose
trailing eigenvalues fall below the configured ratio threshold, and
returns the orthonormalised union of the implicated eigenvectors.

Standalone — does NOT touch the Phoenix Python client (we ship only
``arize-phoenix-otel`` for tracing). Uses ``urllib.request`` so the
script runs in any environment that can see the collector.

CLI:

    python eigen_query.py \
        --since-cycle 0 \
        --theta 1e-3 \
        --top-k 3 \
        --project alethia

Prints JSON to stdout. Designed to be callable from a notebook, from
an MCP server wrapping this entry point, or directly from
`bimodal_drift.py` to feed eigen-redirected acquisition off real
recorded span history rather than the current-cycle decomposition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from modules.surrogate.intention.eigen import union_subspace


# Phoenix GraphQL query: pull spans of a given name from a project,
# project-out the attributes we need. `attributes` is returned as a
# JSON-encoded string (nested dicts mirror the OTel attribute path
# segments split on '.'), so the client parses it post-fetch.
_GRAPHQL_QUERY = """
query DriftEigenSpans($projectName: String!) {
  getProjectByName(name: $projectName) {
    name
    spans(
      first: 1000,
      filterCondition: "name == 'chain.drift.eigen'"
    ) {
      edges {
        node {
          spanId
          name
          startTime
          attributes
        }
      }
    }
  }
}
"""


def _endpoint() -> str:
    return (os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
            or "http://localhost:6006").rstrip("/")


def fetch_eigen_spans(project: str = "alethia") -> list[dict[str, Any]]:
    """Fetch all chain.drift.eigen spans from a Phoenix project.

    Returns a list of dicts: {span_id, start_time, attrs (dict)}.
    Raises RuntimeError if the collector is unreachable or returns errors.
    """
    url = _endpoint() + "/graphql"
    body = json.dumps({
        "query": _GRAPHQL_QUERY,
        "variables": {"projectName": project},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )
    api_key = os.environ.get("PHOENIX_API_KEY")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Phoenix at {url}: {exc}. Set "
            f"PHOENIX_COLLECTOR_ENDPOINT or run `make phoenix-up`."
        ) from exc

    if "errors" in payload and payload["errors"]:
        raise RuntimeError(
            f"GraphQL error: {payload['errors']}"
        )

    project = (payload.get("data", {}) or {}).get("getProjectByName")
    if not project:
        return []
    span_edges = (project.get("spans") or {}).get("edges", [])
    out = []
    for e in span_edges:
        n = e["node"]
        # Phoenix returns `attributes` as a JSON-encoded string whose
        # decoded value is a nested dict keyed by attribute-path segments
        # (split on '.'). `_attr` understands both nested and flat shapes.
        attrs_raw = n.get("attributes")
        if isinstance(attrs_raw, str):
            attrs = json.loads(attrs_raw) if attrs_raw else {}
        else:
            attrs = attrs_raw or {}
        out.append({
            "span_id": n.get("spanId"),
            "start_time": n.get("startTime"),
            "attrs": attrs,
        })
    return out


def _attr(attrs: dict, key: str, default=None):
    """Phoenix sometimes nests attributes ('aletheia.eigen.lambda' is a
    flat key; some versions split on dots into nested dicts). Try both."""
    if key in attrs:
        return attrs[key]
    parts = key.split(".")
    node = attrs
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def select_lis_gap(
    spans: list[dict[str, Any]],
    *,
    theta: float = 1e-3,
    since_cycle: int = 0,
    top_k: int = 3,
) -> dict[str, Any]:
    """The MCP rubric proper. Walks spans, identifies cycles where any
    trailing eigenvalue falls below ``theta * lam_max``, collects the
    corresponding low-eigenvector blocks, and orthonormalises their
    union into the "current LIS gap subspace".

    Returns:
        {
          "n_spans_seen": int,
          "fired_cycles": list[int],      # cycle indices (if attr present)
          "U_low_union": list[list[float]],  # (d, r) orthonormalised
          "lambda_min_per_fired": list[float],
          "theta": float,
          "top_k": int,
        }
    """
    fired_cycles: list[int] = []
    lambda_min: list[float] = []
    U_blocks: list[np.ndarray] = []
    d_seen: int | None = None
    for s in spans:
        a = s["attrs"]
        lam = _attr(a, "aletheia.eigen.lambda")
        if lam is None:
            continue
        lam = np.asarray(lam, dtype=float)
        if lam.size == 0:
            continue
        # cycle index lives on the parent chain.cycle span; we don't
        # always have it here, so fall back to start_time as a proxy.
        cycle = _attr(a, "aletheia.cycle.index")
        if cycle is None:
            cycle = s.get("start_time")
        try:
            cycle_int = int(cycle) if cycle is not None else -1
        except (TypeError, ValueError):
            cycle_int = -1
        if cycle_int >= 0 and cycle_int < since_cycle:
            continue

        lam_max = float(lam.max())
        if lam_max <= 0:
            continue
        if not np.any(lam < theta * lam_max):
            continue

        # Pull U_low_flat: stored column-major as flat list.
        U_low_flat = _attr(a, "aletheia.eigen.U_low_flat")
        d = _attr(a, "aletheia.eigen.d")
        k = _attr(a, "aletheia.eigen.k")
        if U_low_flat is None or d is None or k is None:
            continue
        d, k = int(d), int(k)
        if d_seen is None:
            d_seen = d
        elif d_seen != d:
            # Inconsistent psi dimension across cycles — would be a bug
            # in run.py. Skip rather than crash.
            continue
        U_low = np.asarray(U_low_flat, dtype=float).reshape(d, k, order="F")
        U_blocks.append(U_low[:, :min(top_k, k)])
        fired_cycles.append(cycle_int)
        lambda_min.append(float(lam.min()))

    if not U_blocks:
        return {
            "n_spans_seen": len(spans),
            "fired_cycles": [],
            "U_low_union": [],
            "lambda_min_per_fired": [],
            "theta": float(theta),
            "top_k": int(top_k),
        }

    U_union = union_subspace(U_blocks)
    return {
        "n_spans_seen": len(spans),
        "fired_cycles": fired_cycles,
        "U_low_union": U_union.tolist(),
        "lambda_min_per_fired": lambda_min,
        "theta": float(theta),
        "top_k": int(top_k),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MCP-rubric eigen query against Phoenix")
    parser.add_argument("--project", default="alethia",
                        help="Phoenix project name")
    parser.add_argument("--since-cycle", type=int, default=0,
                        help="ignore drift firings before this cycle")
    parser.add_argument("--theta", type=float, default=1e-3,
                        help="trigger eigenvalue ratio: lam_j < theta * lam_max")
    parser.add_argument("--top-k", type=int, default=3,
                        help="how many bottom-eigenvalue directions to pull per firing")
    parser.add_argument("--output", default="-",
                        help="output JSON path or '-' for stdout")
    args = parser.parse_args(argv)

    spans = fetch_eigen_spans(project=args.project)
    result = select_lis_gap(
        spans, theta=args.theta, since_cycle=args.since_cycle,
        top_k=args.top_k)

    blob = json.dumps(result, indent=2)
    if args.output == "-":
        print(blob)
    else:
        with open(args.output, "w") as f:
            f.write(blob)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
