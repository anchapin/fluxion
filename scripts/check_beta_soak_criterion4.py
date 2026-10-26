#!/usr/bin/env python3
"""β-soak Criterion 4 tracked-cohort gate (ADR-0025, Option C).

Implements the two-tier Criterion 4 semantics Alex approved on
2026-10-18 for the nightly ASHRAE 140 gauge soak
(``docs/adr/0025-beta-soak-criterion4-tracked-cohort.md``):

* **Hard tier** — the six currently-passing 600-series tests must stay
  green and the twelve quarantined ``#[ignore]``d tests must stay
  ignored. Any new failure, any unexpected execution set, or any
  forced-out-of-quarantine run is an infrastructure-grade failure.
* **Tracked tier** — the seven tracked members (re-pinned 2026-10-26,
  loop round 102; was "nine" at the 2026-10-18 record) run every night; their
  measured values (the ``Case NNN ...`` lines the tests print under
  ``--nocapture``) are diffed against the provenance-stamped baseline
  ``tests/reference_data/gauge_soak/criterion4_tracked_cohort_baseline.csv``
  at the Issue #4281 output-diff gate's 0.1% relative epsilon.

Verdicts (``CRITERION4_VERDICT``):

* ``green`` — night counts.
* ``tracked-drift`` — a tracked test failed with a value drift beyond
  epsilon (or a value could not be parsed). The night is failed by the
  workflow's night-verdict step, but **after** the streak state file is
  written (the ADR's fix for the abort-eats-the-streak bug).
* ``re-pin-required`` — a tracked test turned green. The night fails;
  the cohort member must be re-pinned (it leaves the tracked cohort and
  becomes a hard gate).
* ``set-drift`` — the executed/ignored test set does not match the
  baseline cohort. Fail-closed; the checker exits non-zero.
* ``hard-fail`` — a hard-tier test failed. Fail-closed; non-zero exit.

Exit codes: 0 for ``green`` / ``tracked-drift`` / ``re-pin-required``
(the workflow enforces the night verdict separately so the streak file
gets written); 1 for ``set-drift`` / ``hard-fail`` / unreadable inputs.

Usage::

    python3 scripts/check_beta_soak_criterion4.py \
        --log criterion-4-output.log \
        --baseline tests/reference_data/gauge_soak/criterion4_tracked_cohort_baseline.csv \
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

# Statuses from `cargo test` summary lines.
_STATUS_RE = re.compile(
    r"^test\s+(ashrae_140_case_600_series::\S+)\s+\.\.\.\s+"
    r"(ok|FAILED|ignored)(?:,\s+(.*))?$"
)

# The measured-value lines the tests print (`Case 630 Annual Heating:
# 29.23 MWh (Ref: 4.36-5.14)` / `Case 600FF Max Temp: 43.93°C (...)`).
_VALUE_LINE_RE = re.compile(
    r"^Case\s+(?P<case>\d+(?:FF)?)\s+(?P<metric>[A-Za-z ]+?):\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>MWh|kW|°C)\b"
)

# metric key in the baseline CSV, derived from the printed line's case +
# metric text (the tests print exactly one line per metric).
_METRIC_KEYS = {
    ("600FF", "Min Temp"): "min_free_float_temp",
    ("600FF", "Max Temp"): "max_free_float_temp",
    ("630", "Annual Heating"): "annual_heating",
    ("630", "Annual Cooling"): "annual_cooling",
    ("630", "Peak Cooling"): "peak_cooling",
    ("640", "Annual Cooling"): "annual_cooling",
    ("650", "Annual Cooling"): "annual_cooling",
    ("650", "Peak Cooling"): "peak_cooling",
    ("650FF", "Min Temp"): "min_free_float_temp",
    ("650FF", "Max Temp"): "max_free_float_temp",
}


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
) -> tuple[
    dict[str, float],
    dict[str, dict[str, str]],
    dict[str, str],
]:
    """Read the tracked-cohort baseline CSV.

    Returns (values, meta, hard_tier): ``values`` maps tracked test_name
    -> measured value; ``meta`` maps test_name -> {metric, unit};
    ``hard_tier`` maps test_name -> recorded status (``pass`` or
    ``ignored``) for the non-tracked cohort members.
    """
    values: dict[str, float] = {}
    meta: dict[str, dict[str, str]] = {}
    hard_tier: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if parts[0] == "test_name":
            continue
        if parts[0] == "hard":
            if len(parts) != 3 or parts[2] not in {"pass", "ignored"}:
                raise ValueError(f"hard-tier row malformed: {line!r}")
            hard_tier[parts[1]] = parts[2]
            continue
        if len(parts) != 4:
            raise ValueError(f"baseline row malformed: {line!r}")
        name, metric, value, unit = parts
        values[name] = float(value)
        meta[name] = {"metric": metric, "unit": unit}
    return values, meta, hard_tier


def parse_measured_values(text: str) -> dict[tuple[str, str], float]:
    """Extract printed measured values keyed by (case, metric text)."""
    out: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        m = _VALUE_LINE_RE.match(line.strip())
        if m:
            out[(m.group("case"), m.group("metric").strip())] = float(
                m.group("value")
            )
    return out


# Reverse lookup: baseline metric key -> the printed (case, metric) pair
# that carries that value. Kept explicit rather than guessed.
_METRIC_KEY_TO_PRINT: dict[str, tuple[str, str]] = {
    v: k for k, v in _METRIC_KEYS.items()
}


def relative_drift(new: float, base: float) -> float:
    if base == 0.0:
        return 0.0 if new == 0.0 else float("inf")
    return abs(new - base) / abs(base)


def evaluate(
    outcomes: list[TestOutcome],
    measured: dict[tuple[str, str], float],
    baseline: dict[str, float],
    baseline_meta: dict[str, dict[str, str]],
    hard_tier: dict[str, str],
) -> Verdict:
    baseline_tracked = set(baseline)
    seen = {o.name: o.status for o in outcomes}

    expected_all = baseline_tracked | set(hard_tier)
    missing = expected_all - set(seen)
    extra = set(seen) - expected_all
    if missing or extra:
        reasons = []
        if missing:
            reasons.append(f"tests missing from log (fail-closed): {sorted(missing)}")
        if extra:
            reasons.append(f"tests not in baseline cohort: {sorted(extra)}")
        return Verdict("set-drift", reasons=reasons)

    drift: list[dict] = []
    re_pins: list[str] = []
    hard_failures: list[str] = []

    for name, status in seen.items():
        if name in baseline_tracked:
            # Tracked tier: must fail, with value drift-checked.
            if status == "ok":
                re_pins.append(name)
                continue
            if status == "ignored":
                hard_failures.append(
                    f"{name} became ignored while tracked (fail-closed)"
                )
                continue
            key = _METRIC_KEY_TO_PRINT.get(baseline_meta[name]["metric"])
            # Disambiguate per-case metrics sharing a key (min/max temp,
            # 600FF vs 650FF): the case id comes from the test name.
            case = name.split("::")[1].removeprefix("case_").upper()
            print_key = (case, key[1]) if key else None
            value = measured.get(print_key) if print_key else None
            if value is None:
                drift.append(
                    {
                        "test": name,
                        "reason": "no measured value line found in log",
                        "baseline": baseline[name],
                    }
                )
                continue
            rel = relative_drift(value, baseline[name])
            if rel > RELATIVE_EPSILON:
                drift.append(
                    {
                        "test": name,
                        "measured": value,
                        "baseline": baseline[name],
                        "relative_drift": rel,
                    }
                )
        else:
            # Hard tier, machine-checked against the recorded state:
            # a recorded 'pass' must still print 'ok'; a recorded
            # 'ignored' must still print 'ignored' (a forced execution
            # means the #[ignore] was removed without a re-pin).
            recorded = hard_tier[name]
            if recorded == "pass" and status == "FAILED":
                hard_failures.append(f"{name} regressed (hard tier)")
            elif recorded == "ignored" and status != "ignored":
                hard_failures.append(
                    f"{name} left quarantine without a re-pin "
                    f"(recorded ignored, ran as {status})"
                )
    if hard_failures:
        return Verdict("hard-fail", reasons=hard_failures)
    if re_pins:
        return Verdict("re-pin-required", re_pins=re_pins)
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
            "tests/reference_data/gauge_soak/criterion4_tracked_cohort_baseline.csv"
        ),
    )
    ap.add_argument("--github-env", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"criterion4 gate: cannot read log {args.log}: {exc}", file=sys.stderr)
        return 1
    try:
        baseline, baseline_meta, hard_tier = parse_baseline(args.baseline)
    except (OSError, ValueError) as exc:
        print(f"criterion4 gate: cannot read baseline: {exc}", file=sys.stderr)
        return 1

    outcomes = parse_log(log_text)
    if not outcomes:
        print(
            "criterion4 gate: no ashrae_140_case_600_series outcomes in log "
            "(fail-closed)",
            file=sys.stderr,
        )
        return 1

    measured = parse_measured_values(log_text)
    verdict = evaluate(outcomes, measured, baseline, baseline_meta, hard_tier)

    if args.github_env is not None:
        with open(args.github_env, "a", encoding="utf-8") as f:
            f.write(f"CRITERION4_VERDICT={verdict.verdict}\n")

    payload = {
        "verdict": verdict.verdict,
        "reasons": verdict.reasons,
        "drift": verdict.drift,
        "re_pin_required": verdict.re_pins,
        "executed_tests": len(outcomes),
        "tracked_cohort_size": len(baseline),
        "hard_tier_size": len(hard_tier),
        "epsilon_relative": RELATIVE_EPSILON,
    }
    print(json.dumps(payload, indent=2))

    if verdict.verdict in {"set-drift", "hard-fail"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
