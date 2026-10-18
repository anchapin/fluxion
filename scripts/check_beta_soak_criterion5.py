#!/usr/bin/env python3
"""Beta-soak Criterion 5 tracked-cohort gate (ADR-0025, applied 2026-10-18).

Applies the two-tier tracked-cohort semantics Alex accepted for Criterion 4
(``docs/adr/0025-beta-soak-criterion4-tracked-cohort.md``, Option C) to
Criterion 5 of the nightly ASHRAE 140 gauge soak:
``known_issues_regression::issue_1457_case_600_series_tracking``.

Route decision (round 64): the Criterion 5 test is the #1457 known-gap
tracker — a single ``#[test]`` with **no** ``#[ignore]`` that asserts the 14
remaining out-of-band Case 600-series metrics. Its failing signature
(peak cooling OVER, peak heating UNDER, annual cooling UNDER, free-float
min-temp too warm) is the documented discrete-node / 1-hour-timestep
solar-injection limitation routed to the GaugeSolver (#1465 / #1462 /
#3251 Phase 3, production path not yet wired for low-mass 5R1C). That is
the same structural band-gap class ADR-0025 Option C already covers, so
the accepted ADR is *applied*, not re-decided.

* **Hard tier** (fail-closed) — the executed test set in the Criterion 5
  log must be exactly the one tracked test, and it must still execute
  (not be ``#[ignore]``d and not pass).
* **Tracked tier** — the 14 metric values the test prints under
  ``--nocapture`` are diffed against the provenance-stamped baseline
  ``tests/reference_data/gauge_soak/criterion5_tracked_cohort_baseline.csv``
  at the Issue #4281 output-diff gate's 0.1% relative epsilon.

Verdicts (``CRITERION5_VERDICT``): ``green`` / ``tracked-drift`` /
``re-pin-required`` / ``set-drift`` / ``hard-fail`` — same semantics as
the Criterion 4 checker. Exit codes: 0 for ``green`` /
``tracked-drift`` / ``re-pin-required`` (the workflow enforces the night
verdict separately, after the streak file is written); 1 for
``set-drift`` / ``hard-fail`` / unreadable inputs.

Usage::

    python3 scripts/check_beta_soak_criterion5.py \
        --log criterion-5-output.log \
        --baseline tests/reference_data/gauge_soak/criterion5_tracked_cohort_baseline.csv \
        [--github-env "$GITHUB_ENV"] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Issue #4281 output-diff gate epsilon, reused per ADR-0025 Option C.
RELATIVE_EPSILON = 0.001

# The single tracked test. The workflow's cargo filter
# (`known_issues_regression::issue_1457_case_600_series_tracking`) must
# execute exactly this set.
TRACKED_TEST = (
    "known_issues_regression::issue_1457_case_600_series_tracking::"
    "test_issue1457_remaining_600_series_metrics"
)

# Statuses from `cargo test` summary lines.
_STATUS_RE = re.compile(
    r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)(?:,\s+(.*))?$"
)

# The measured-value lines the test prints under --nocapture:
#   `Case610 peak_heating: 4.90 kW [ref 4.30-5.70] FAIL`
#   `Case600FF min_free_float: -12.34C [ref -18.80..-15.60] PASS`
_VALUE_LINE_RE = re.compile(
    r"^Case(?P<case>\d+(?:FF)?)\s+(?P<metric>[a-z_]+):\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>MWh|kW|C)\b"
)


@dataclass
class TestOutcome:
    name: str
    status: str  # ok | FAILED | ignored


@dataclass
class Verdict:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    drift: list[dict] = field(default_factory=list)
    re_pins: list[str] = field(default_factory=list)


def parse_log(text: str) -> list[TestOutcome]:
    """Return the per-test outcomes from a cargo test log section."""
    outcomes: list[TestOutcome] = []
    for line in text.splitlines():
        m = _STATUS_RE.match(line.strip())
        if m:
            outcomes.append(TestOutcome(name=m.group(1), status=m.group(2)))
    return outcomes


def parse_baseline(
    path: Path,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], dict[str, str]]]:
    """Read the tracked-cohort baseline CSV.

    Returns (values, meta): keyed by (case, metric), with ``meta``
    carrying the recorded unit per key.
    """
    values: dict[tuple[str, str], float] = {}
    meta: dict[tuple[str, str], dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("case,"):
            continue
        parts = line.split(",")
        if len(parts) != 4:
            raise ValueError(f"baseline row malformed: {line!r}")
        case, metric, value, unit = parts
        key = (case, metric)
        values[key] = float(value)
        meta[key] = {"unit": unit}
    if not values:
        raise ValueError("baseline contains no tracked rows")
    return values, meta


def parse_measured_values(text: str) -> dict[tuple[str, str], float]:
    """Extract printed measured values keyed by (case, metric)."""
    out: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        m = _VALUE_LINE_RE.match(line.strip())
        if m:
            out[(m.group("case"), m.group("metric"))] = float(m.group("value"))
    return out


def relative_drift(new: float, base: float) -> float:
    if base == 0.0:
        return 0.0 if new == 0.0 else float("inf")
    return abs(new - base) / abs(base)


def evaluate(
    outcomes: list[TestOutcome],
    measured: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
) -> Verdict:
    seen = {o.name: o.status for o in outcomes}
    if set(seen) != {TRACKED_TEST}:
        missing = {TRACKED_TEST} - set(seen)
        extra = set(seen) - {TRACKED_TEST}
        reasons = []
        if missing:
            reasons.append(f"tracked test missing from log (fail-closed): {sorted(missing)}")
        if extra:
            reasons.append(f"tests not in the Criterion 5 cohort: {sorted(extra)}")
        return Verdict("set-drift", reasons=reasons)

    status = seen[TRACKED_TEST]
    if status == "ok":
        # The #1457 metrics went in-band: celebrate, then re-pin. The
        # cohort leaves via a re-pin commit, never by widening a band.
        return Verdict("re-pin-required", re_pins=[TRACKED_TEST])
    if status == "ignored":
        return Verdict(
            "hard-fail",
            reasons=[f"{TRACKED_TEST} became ignored (fail-closed)"],
        )

    # status == FAILED: the expected state for the known-gap tracker.
    # Drift-check every printed value against the baseline.
    drift: list[dict] = []
    for key, base_value in baseline.items():
        value = measured.get(key)
        if value is None:
            drift.append(
                {
                    "metric": f"{key[0]}/{key[1]}",
                    "reason": "no measured value line found in log",
                    "baseline": base_value,
                }
            )
            continue
        rel = relative_drift(value, base_value)
        if rel > RELATIVE_EPSILON:
            drift.append(
                {
                    "metric": f"{key[0]}/{key[1]}",
                    "measured": value,
                    "baseline": base_value,
                    "relative_drift": rel,
                }
            )
    if drift:
        return Verdict("tracked-drift", drift=drift)
    return Verdict("green")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "tests/reference_data/gauge_soak/criterion5_tracked_cohort_baseline.csv"
        ),
    )
    ap.add_argument("--github-env", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"criterion5 gate: cannot read log {args.log}: {exc}", file=sys.stderr)
        return 1
    try:
        baseline, _meta = parse_baseline(args.baseline)
    except (OSError, ValueError) as exc:
        print(f"criterion5 gate: cannot read baseline: {exc}", file=sys.stderr)
        return 1

    outcomes = parse_log(log_text)
    if not outcomes:
        print(
            "criterion5 gate: no test outcomes in log (fail-closed)",
            file=sys.stderr,
        )
        return 1

    measured = parse_measured_values(log_text)
    verdict = evaluate(outcomes, measured, baseline)

    if args.github_env is not None:
        with open(args.github_env, "a", encoding="utf-8") as f:
            f.write(f"CRITERION5_VERDICT={verdict.verdict}\n")

    payload = {
        "verdict": verdict.verdict,
        "reasons": verdict.reasons,
        "drift": verdict.drift,
        "re_pin_required": verdict.re_pins,
        "executed_tests": len(outcomes),
        "tracked_metric_count": len(baseline),
        "epsilon_relative": RELATIVE_EPSILON,
    }
    print(json.dumps(payload, indent=2))

    if verdict.verdict in {"set-drift", "hard-fail"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
