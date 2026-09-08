#!/usr/bin/env python3
"""
Cross-source consistency check for the Fluxion release scorecard.

Issue #3535 surfaced a divergence between the two committed sources that
feed ``SCORECARD.md``'s headline pass rate / MAE:

* ``validation/performance_history.latest.json`` (timestamp 2026-09-07;
  preferred -- newer)
* ``docs/ASHRAE140_RESULTS.md``               (timestamp 2026-08-16; fallback)

The scorecard generator now prefers the perf-history snapshot for those
headline figures, so any future divergence between the two sources is a
silent drift bug. This gate catches it:

    python3 scripts/check_scorecard_data_sources_consistent.py

Exit codes:
    0 -- the two sources agree within tolerance on both headline figures.
    1 -- divergence exceeds the tolerance on either pass-rate or MAE.
    2 -- a source file is missing or could not be parsed.

Tolerances (chosen to absorb normal floating-point rounding between the
docs markdown (one decimal) and the JSON snapshot (full precision), plus
the natural run-to-run noise of a single validation pass):

* pass_rate : 0.1 pp  (i.e. ``abs(docs - json) <= 0.1``)
* mae       : 0.5 pp  (i.e. ``abs(docs - json) <= 0.5``)

When the two sources disagree beyond tolerance, the gate prints which
source is newer (the canonical one) and which to regenerate from.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERF_SNAPSHOT = REPO_ROOT / "validation" / "performance_history.latest.json"
ASHRAE_DOC = REPO_ROOT / "docs" / "ASHRAE140_RESULTS.md"
PERF_SNAPSHOT_ATTR = "validation/performance_history.latest.json"
ASHRAE_DOC_ATTR = "docs/ASHRAE140_RESULTS.md"

# Tolerances -- see module docstring.
PASS_RATE_TOLERANCE_PP = 0.1
MAE_TOLERANCE_PP = 0.5

# Pattern for the perf-history snapshot timestamp (``...T...+00:00``).
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

# Re-use the same regex the scorecard generator uses for the ASHRAE doc.
_PASS_RATE_RE = re.compile(r"\|\s*Pass Rate\s*\|\s*([0-9.]+)\s*%?\s*\|", re.IGNORECASE)
_MAE_RE = re.compile(
    r"\|\s*Mean Absolute Error\s*\|\s*([0-9.]+)\s*%?\s*\|", re.IGNORECASE
)
_GENERATED_RE = re.compile(r"\*Generated:\s*([0-9-]+ [0-9:]+ UTC)\*")


def _load_perf_snapshot() -> dict | None:
    """Return the parsed perf-history snapshot or ``None`` if unavailable.

    We deliberately do NOT raise here -- a missing/corrupt snapshot is a
    legitimate state (the ASHRAE doc then stands as the sole source), and
    the scorecard generator already handles it. The gate's job is to flag
    *divergence*, not absence.
    """
    if not PERF_SNAPSHOT.exists():
        return None
    try:
        return json.loads(PERF_SNAPSHOT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_ashrae_headline() -> tuple[float | None, float | None, str]:
    """Return ``(pass_rate, mae, generated_utc)`` from the ASHRAE doc.

    Parsed the same way ``scripts/generate_scorecard.py::parse_ashrae``
    parses it (first ``## Summary`` table row that matches the key), so a
    fix to one is reflected in the other.
    """
    if not ASHRAE_DOC.exists():
        return None, None, ""
    text = ASHRAE_DOC.read_text(encoding="utf-8")

    pr_m = _PASS_RATE_RE.search(text)
    mae_m = _MAE_RE.search(text)
    gen_m = _GENERATED_RE.search(text)

    def _f(m: re.Match[str] | None) -> float | None:
        return float(m.group(1)) if m else None

    return _f(pr_m), _f(mae_m), gen_m.group(1) if gen_m else ""


def _compare(a: float | None, b: float | None, tol: float) -> tuple[bool, float | None]:
    """Return ``(within_tolerance, abs_diff)``. ``None`` on either side
    means "missing" and we conservatively treat it as outside the
    tolerance (so the gate fails loud rather than silent-green)."""
    if a is None or b is None:
        return False, None
    diff = abs(a - b)
    return diff <= tol, diff


def _timestamp_perf(entry: dict | None) -> str:
    if not entry:
        return "(missing)"
    ts = str(entry.get("timestamp", ""))
    if _ISO_TS_RE.match(ts):
        return ts[:10]  # YYYY-MM-DD prefix
    return ts or "(no timestamp)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check that the scorecard's two committed sources "
        "agree on the headline pass-rate and MAE."
    )
    ap.add_argument(
        "--pass-rate-tolerance",
        type=float,
        default=PASS_RATE_TOLERANCE_PP,
        help=f"pass-rate divergence tolerance in pp "
        f"(default: {PASS_RATE_TOLERANCE_PP})",
    )
    ap.add_argument(
        "--mae-tolerance",
        type=float,
        default=MAE_TOLERANCE_PP,
        help=f"MAE divergence tolerance in pp (default: {MAE_TOLERANCE_PP})",
    )
    args = ap.parse_args(argv)

    print("=== Scorecard data-source consistency (Issue #3535) ===")
    print(f"Repo: {REPO_ROOT}")
    print(
        f"Sources: {PERF_SNAPSHOT_ATTR} (preferred, newer) | "
        f"{ASHRAE_DOC_ATTR} (fallback, older)"
    )
    print(f"Tolerances: pass_rate ≤ {args.pass_rate_tolerance} pp | "
          f"MAE ≤ {args.mae_tolerance} pp")
    print()

    perf = _load_perf_snapshot()
    doc_pr, doc_mae, doc_gen = _load_ashrae_headline()

    perf_pr = perf.get("pass_rate") if perf else None
    perf_mae = perf.get("mae") if perf else None

    ts_perf = _timestamp_perf(perf)
    ts_doc = doc_gen.split(" ")[0] if doc_gen else "(missing)"

    print(f"Perf-history timestamp: {ts_perf}")
    print(f"ASHRAE-doc timestamp:   {ts_doc}")
    print()

    failures: list[str] = []

    print(
        f"  pass_rate:  perf-history = "
        f"{perf_pr if perf_pr is not None else '(missing)':>6}  |  "
        f"ASHRAE doc = "
        f"{doc_pr if doc_pr is not None else '(missing)':>6}",
        end="",
    )
    pr_ok, pr_diff = _compare(perf_pr, doc_pr, args.pass_rate_tolerance)
    if pr_diff is None:
        print("  | MISSING ON ONE SIDE")
        failures.append("pass-rate: missing on one side")
    elif pr_ok:
        print(f"  | diff = {pr_diff:.3f} pp  ✓")
    else:
        print(f"  | diff = {pr_diff:.3f} pp  ✗ (>{args.pass_rate_tolerance} pp)")
        failures.append(
            f"pass-rate divergence {pr_diff:.3f} pp > "
            f"{args.pass_rate_tolerance} pp tolerance"
        )

    print(
        f"  mae:        perf-history = "
        f"{perf_mae if perf_mae is not None else '(missing)':>6}  |  "
        f"ASHRAE doc = "
        f"{doc_mae if doc_mae is not None else '(missing)':>6}",
        end="",
    )
    mae_ok, mae_diff = _compare(perf_mae, doc_mae, args.mae_tolerance)
    if mae_diff is None:
        print("  | MISSING ON ONE SIDE")
        failures.append("MAE: missing on one side")
    elif mae_ok:
        print(f"  | diff = {mae_diff:.3f} pp  ✓")
    else:
        print(f"  | diff = {mae_diff:.3f} pp  ✗ (>{args.mae_tolerance} pp)")
        failures.append(
            f"MAE divergence {mae_diff:.3f} pp > "
            f"{args.mae_tolerance} pp tolerance"
        )

    print()
    if not failures:
        print("PASS: the two committed sources agree within tolerance.")
        print(
            f"  Scorecard headline reflects {PERF_SNAPSHOT_ATTR} "
            f"(newer; {ts_perf}) with {ASHRAE_DOC_ATTR} "
            f"({ts_doc}) as the documented fallback."
        )
        return 0

    print(f"FAIL: {len(failures)} divergence(s) detected:")
    for f in failures:
        print(f"  - {f}")
    print()
    print("Diagnosis (Issue #3535): the two committed sources feed the")
    print("scorecard headline (pass-rate, MAE, max-deviation) and they")
    print("disagree. The scorecard generator prefers the perf-history")
    print("snapshot (newer), so the scorecard will show the perf-history")
    print("value. The ASHRAE doc is the fallback only when the snapshot is")
    print("missing/corrupt.")
    print()
    print("Resolution path:")
    print(
        "  1. If the divergence is real (the validation run actually moved):"
    )
    print(f"     - regenerate docs/ASHRAE140_RESULTS.md from the latest run")
    print(
        f"       (the perf-history snapshot is canonical -- issue #3535 scope guard)."
    )
    print("  2. If the snapshot is stale (validation run not yet snapshotted):")
    print(
        f"     - re-run the validation harness so "
        f"{PERF_SNAPSHOT_ATTR} is refreshed."
    )
    print(
        "  3. If the ASHRAE doc is stale and should NOT be regenerated:"
    )
    print(
        f"     - regenerate the scorecard explicitly: "
        f"`python3 scripts/generate_scorecard.py`."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)