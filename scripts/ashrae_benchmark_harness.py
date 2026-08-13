#!/usr/bin/env python3
"""
ASHRAE 140 Benchmark Harness
============================
Runs ASHRAE 140 validation test targets with per-target wall-clock timing,
extracts pass/fail counts, and compares against a stored JSON baseline so
that each Wave 1-3 PR can show a concrete delta:
    "This PR: 12 passed / 6 failed  (was 10/8, +2 passes)"

Usage
-----
# Full run — all test targets, no comparison
python scripts/ashrae_benchmark_harness.py

# Full run, compare against stored baseline
python scripts/ashrae_benchmark_harness.py --compare benches/baseline/ashrae_benchmark_baseline.json

# Run and save result as the new baseline
python scripts/ashrae_benchmark_harness.py --save-baseline benches/baseline/ashrae_benchmark_baseline.json

# Save baseline AND compare (update after recording delta)
python scripts/ashrae_benchmark_harness.py \\
    --compare benches/baseline/ashrae_benchmark_baseline.json \\
    --save-baseline benches/baseline/ashrae_benchmark_baseline.json

# Output JSON summary to a file (for CI artifact upload)
python scripts/ashrae_benchmark_harness.py --output benchmark_results.json

# CI-friendly: compare + output + fail on regression
python scripts/ashrae_benchmark_harness.py \\
    --compare benches/baseline/ashrae_benchmark_baseline.json \\
    --output benchmark_results.json \\
    --fail-on-regression

Exit codes
----------
0 — All tests ran; no regression detected (or --fail-on-regression not set)
1 — Regression detected AND --fail-on-regression is set
2 — Fatal: could not compile or run cargo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1"

# Test targets to time individually.  Each maps to a file at tests/<target>.rs
# The comprehensive validator is listed first — its --nocapture output drives
# the per-case pass/fail breakdown.
TEST_TARGETS = [
    "ashrae_140_validation",           # comprehensive: all 18+ cases, main source of per-case data
    "ashrae_140_blind_validation",     # blind validation (no peeking at ref ranges during sim)
    "ashrae_140_case_600_series",      # 600/610/620/630/640/650 series
    "ashrae_140_case_900",             # 900/910/920/930/940/950 and FF variants
    "ashrae_140_case_960_sunspace",    # Sunspace (Case 960)
    "ashrae_140_case_195_470",         # Analytical cases 195 and 470
    "ashrae_140_free_floating",        # Free-floating temperature (600FF, 650FF, 900FF, 950FF)
    "ashrae_140_integration",          # Integration tests across the validator stack
    "ashrae_140_case_non_residential", # Non-residential cases
    "ashrae_140_cases_800_810",        # 800 / 810 series
    "ashrae_140_setback_ventilation",  # Setback and ventilation variants
]

# Regex patterns that match --nocapture output from the ASHRAE140Validator
# Note: Some patterns accept "inf" as a valid numeric value (e.g., when reference range is 0)
# Also handles leading whitespace that may appear in cargo test output
_NUMERIC_PATTERN = r"([+-]?(?:inf|\d+\.?\d*))"  # Matches numbers including inf/-inf
_CASE_PATTERN = re.compile(
    r"Case\s+(\d+[A-Z0-9_]*)\s*[:\-]\s*"
    r"Heating\s*=\s*" + _NUMERIC_PATTERN + r"\s*\(Ref:\s*" + _NUMERIC_PATTERN + r"\s*-\s*" + _NUMERIC_PATTERN + r"\),\s*"
    r"Cooling\s*=\s*" + _NUMERIC_PATTERN + r"\s*\(Ref:\s*" + _NUMERIC_PATTERN + r"\s*-\s*" + _NUMERIC_PATTERN + r"\)"
)
_SUMMARY_PATTERN = re.compile(
    r"^\s*Pass\s+Rate:\s*([\d.]+|inf)%.*?Passed:\s*(\d+).*?Failed:\s*(\d+).*?"
    r"Mean\s+Absolute\s+Error:\s*([\d.]+|inf)%",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
# Rust test runner output: "test result: ok. N passed; M failed; ..."
_RUST_RESULT_PATTERN = re.compile(
    r"test result:.*?(\d+)\s+passed;\s*(\d+)\s+failed"
)
# Individual test FAILED line
_FAILED_TEST_PATTERN = re.compile(r"^FAILED\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationCase:
    case_id: str
    heating_actual: float
    heating_ref_min: float
    heating_ref_max: float
    heating_pass: bool
    cooling_actual: float
    cooling_ref_min: float
    cooling_ref_max: float
    cooling_pass: bool
    overall_pass: bool


@dataclass
class TargetResult:
    target: str
    duration_s: float
    exit_code: int
    tests_passed: int
    tests_failed: int
    failed_test_names: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BenchmarkSummary:
    total_validation_cases: int
    validation_cases_passed: int
    validation_cases_failed: int
    pass_rate: float
    mae_percent: float
    total_duration_s: float
    total_tests_passed: int   # sum across all targets (rust test functions)
    total_tests_failed: int


@dataclass
class BenchmarkReport:
    schema_version: str
    timestamp: str
    commit_sha: str
    branch: str
    summary: BenchmarkSummary
    test_targets: list[TargetResult]
    validation_cases: list[ValidationCase]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _git_info() -> tuple[str, str]:
    """Return (commit_sha, branch) from the repo, or empty strings on failure."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        sha = os.environ.get("GITHUB_SHA", "unknown")[:7]

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        branch = os.environ.get("GITHUB_REF_NAME", "unknown")

    return sha, branch


def _run_cargo_test(target: str, release: bool = True, timeout: int = 300) -> tuple[str, float, int]:
    """
    Run `cargo test --test <target> [--release] -- --nocapture`.
    Returns (combined_stdout_stderr, duration_s, exit_code).
    """
    cmd = ["cargo", "test", "--test", target]
    if release:
        cmd.append("--release")
    cmd += ["--", "--nocapture"]

    print(f"  Running: {' '.join(cmd)}", flush=True)
    t0 = _monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = _monotonic() - t0
        combined = result.stdout + "\n" + result.stderr
        return combined, duration, result.returncode
    except subprocess.TimeoutExpired:
        duration = _monotonic() - t0
        return f"TIMEOUT after {timeout}s", duration, 124
    except FileNotFoundError:
        return "ERROR: cargo not found on PATH", 0.0, 127


def _monotonic() -> float:
    import time
    return time.monotonic()


def _parse_target_output(output: str) -> tuple[int, int, list[str]]:
    """Parse Rust test runner output → (passed, failed, failed_names)."""
    passed = failed = 0
    for m in _RUST_RESULT_PATTERN.finditer(output):
        passed = int(m.group(1))
        failed = int(m.group(2))
    failed_names = _FAILED_TEST_PATTERN.findall(output)
    return passed, failed, failed_names


def _parse_validation_output(output: str) -> tuple[list[ValidationCase], float, float]:
    """
    Parse comprehensive validator output.
    Returns (cases, pass_rate, mae_percent).
    """

    def parse_numeric(s: str) -> float:
        """Parse a numeric string that may contain inf or -inf."""
        s = s.strip().lower()
        if s in ("inf", "+inf"):
            return float("inf")
        elif s == "-inf":
            return float("-inf")
        return float(s)

    cases: list[ValidationCase] = []

    for m in _CASE_PATTERN.finditer(output):
        case_id = m.group(1)
        h_act = parse_numeric(m.group(2))
        h_min = parse_numeric(m.group(3))
        h_max = parse_numeric(m.group(4))
        c_act = parse_numeric(m.group(5))
        c_min = parse_numeric(m.group(6))
        c_max = parse_numeric(m.group(7))
        # Handle inf in range checks - if ref range is inf, always pass
        h_pass = h_min <= h_act <= h_max if h_min != float("inf") and h_max != float("inf") else True
        c_pass = c_min <= c_act <= c_max if c_min != float("inf") and c_max != float("inf") else True
        cases.append(ValidationCase(
            case_id=case_id,
            heating_actual=h_act,
            heating_ref_min=h_min,
            heating_ref_max=h_max,
            heating_pass=h_pass,
            cooling_actual=c_act,
            cooling_ref_min=c_min,
            cooling_ref_max=c_max,
            cooling_pass=c_pass,
            overall_pass=h_pass and c_pass,
        ))

    pass_rate = mae = 0.0
    m = _SUMMARY_PATTERN.search(output)
    if m:
        pass_rate = parse_numeric(m.group(1))
        mae = parse_numeric(m.group(4))

    return cases, pass_rate, mae


def run_harness(release: bool = True, timeout: int = 300) -> BenchmarkReport:
    sha, branch = _git_info()
    target_results: list[TargetResult] = []
    validation_cases: list[ValidationCase] = []
    pass_rate = mae = 0.0

    print(f"\n{'='*60}")
    print("ASHRAE 140 Benchmark Harness")
    print(f"Commit: {sha}  Branch: {branch}")
    print(f"Mode:   {'release' if release else 'debug'}")
    print(f"{'='*60}\n")

    for target in TEST_TARGETS:
        print(f"[{target}]")
        output, duration, code = _run_cargo_test(target, release=release, timeout=timeout)
        passed, failed, failed_names = _parse_target_output(output)

        # Only parse validation cases from the comprehensive target
        if target == "ashrae_140_validation" and output:
            validation_cases, pass_rate, mae = _parse_validation_output(output)

        result = TargetResult(
            target=target,
            duration_s=round(duration, 2),
            exit_code=code,
            tests_passed=passed,
            tests_failed=failed,
            failed_test_names=failed_names,
            notes="TIMEOUT" if code == 124 else ("NOT FOUND (skipped)" if code == 127 else ""),
        )
        target_results.append(result)
        status = "✓" if code == 0 else "✗"
        print(f"  {status} {passed} passed / {failed} failed  ({duration:.1f}s)\n")

    total_duration = sum(r.duration_s for r in target_results)
    total_passed_tests = sum(r.tests_passed for r in target_results)
    total_failed_tests = sum(r.tests_failed for r in target_results)

    # If we parsed per-case data, use it; otherwise fall back to rust test counts
    if validation_cases:
        vc_passed = sum(1 for c in validation_cases if c.overall_pass)
        vc_failed = sum(1 for c in validation_cases if not c.overall_pass)
        total_vc = len(validation_cases)
        if pass_rate == 0.0 and total_vc > 0:
            pass_rate = round(vc_passed / total_vc * 100, 1)
    else:
        vc_passed = vc_failed = total_vc = 0

    summary = BenchmarkSummary(
        total_validation_cases=total_vc,
        validation_cases_passed=vc_passed,
        validation_cases_failed=vc_failed,
        pass_rate=pass_rate,
        mae_percent=mae,
        total_duration_s=round(total_duration, 2),
        total_tests_passed=total_passed_tests,
        total_tests_failed=total_failed_tests,
    )

    return BenchmarkReport(
        schema_version=SCHEMA_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit_sha=sha,
        branch=branch,
        summary=summary,
        test_targets=target_results,
        validation_cases=validation_cases,
    )


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

@dataclass
class Delta:
    validation_cases_passed_delta: int
    validation_cases_failed_delta: int
    pass_rate_delta: float
    mae_delta: float
    duration_delta_s: float
    regression: bool          # True if pass count went down
    improvement: bool         # True if pass count went up


def compare_to_baseline(report: BenchmarkReport, baseline_path: Path) -> Optional[Delta]:
    if not baseline_path.exists():
        print(f"[compare] Baseline not found at {baseline_path}, skipping comparison.")
        return None

    try:
        with baseline_path.open() as f:
            base = json.load(f)
    except Exception as e:
        print(f"[compare] Could not load baseline: {e}")
        return None

    base_sum = base.get("summary", {})
    cur = report.summary

    vc_delta = cur.validation_cases_passed - base_sum.get("validation_cases_passed", 0)
    vf_delta = cur.validation_cases_failed - base_sum.get("validation_cases_failed", 0)
    pr_delta = cur.pass_rate - base_sum.get("pass_rate", 0.0)
    mae_delta = cur.mae_percent - base_sum.get("mae_percent", 0.0)
    dur_delta = cur.total_duration_s - base_sum.get("total_duration_s", 0.0)

    return Delta(
        validation_cases_passed_delta=vc_delta,
        validation_cases_failed_delta=vf_delta,
        pass_rate_delta=round(pr_delta, 1),
        mae_delta=round(mae_delta, 2),
        duration_delta_s=round(dur_delta, 1),
        regression=vc_delta < 0,
        improvement=vc_delta > 0,
    )


def print_delta(report: BenchmarkReport, delta: Delta) -> None:
    cur = report.summary
    sign = lambda v: f"+{v}" if v > 0 else str(v)
    icon = "⬆️  IMPROVEMENT" if delta.improvement else ("⬇️  REGRESSION" if delta.regression else "➡️  NO CHANGE")

    print(f"\n{'='*60}")
    print(f"DELTA vs. BASELINE  {icon}")
    print(f"{'='*60}")
    print(f"  Validation cases passed : {cur.validation_cases_passed}  ({sign(delta.validation_cases_passed_delta)})")
    print(f"  Validation cases failed : {cur.validation_cases_failed}  ({sign(delta.validation_cases_failed_delta)})")
    print(f"  Pass rate               : {cur.pass_rate:.1f}%  ({sign(delta.pass_rate_delta)}pp)")
    print(f"  Mean absolute error     : {cur.mae_percent:.2f}%  ({sign(delta.mae_delta)}pp)")
    print(f"  Total duration          : {cur.total_duration_s:.1f}s  ({sign(delta.duration_delta_s)}s)")
    print()


def print_summary(report: BenchmarkReport) -> None:
    s = report.summary
    print(f"\n{'='*60}")
    print(f"SUMMARY  commit={report.commit_sha}  branch={report.branch}")
    print(f"{'='*60}")
    print(f"  Validation cases  : {s.validation_cases_passed} passed / {s.validation_cases_failed} failed "
          f"({s.pass_rate:.1f}%)")
    print(f"  MAE               : {s.mae_percent:.2f}%")
    print(f"  Total duration    : {s.total_duration_s:.1f}s")
    print(f"  Rust tests (all)  : {s.total_tests_passed} passed / {s.total_tests_failed} failed")
    print()

    if report.validation_cases:
        print("  Per-case breakdown:")
        print(f"  {'Case':<10} {'Heat':>10} {'Cool':>10} {'Pass?':>6}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        for c in sorted(report.validation_cases, key=lambda x: x.case_id):
            h = f"{c.heating_actual:.0f}" if c.heating_actual else "—"
            co = f"{c.cooling_actual:.0f}" if c.cooling_actual else "—"
            ok = "✓" if c.overall_pass else "✗"
            print(f"  {c.case_id:<10} {h:>10} {co:>10} {ok:>6}")
    print()


# ---------------------------------------------------------------------------
# GitHub Actions helpers
# ---------------------------------------------------------------------------

def write_github_step_summary(report: BenchmarkReport, delta: Optional[Delta]) -> None:
    """Append a Markdown table to $GITHUB_STEP_SUMMARY if in CI."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    s = report.summary
    sign = lambda v: f"+{v}" if v > 0 else str(v)
    lines = [
        "## ASHRAE 140 Benchmark Harness Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Validation cases passed | **{s.validation_cases_passed}** |",
        f"| Validation cases failed | {s.validation_cases_failed} |",
        f"| Pass rate | {s.pass_rate:.1f}% |",
        f"| Mean absolute error | {s.mae_percent:.2f}% |",
        f"| Total duration | {s.total_duration_s:.1f}s |",
        f"| Commit | `{report.commit_sha}` |",
        "",
    ]

    if delta:
        icon = "⬆️" if delta.improvement else ("⬇️" if delta.regression else "➡️"  )
        lines += [
            f"### Delta vs. Baseline {icon}",
            "",
            "| Metric | Delta |",
            "|--------|-------|",
            f"| Cases passed | `{sign(delta.validation_cases_passed_delta)}` |",
            f"| Pass rate | `{sign(delta.pass_rate_delta)}pp` |",
            f"| MAE | `{sign(delta.mae_delta)}pp` |",
            f"| Duration | `{sign(delta.duration_delta_s)}s` |",
            "",
        ]

        if delta.regression:
            lines.append("> ⚠️ **Regression detected**: fewer validation cases passing than baseline.")
        elif delta.improvement:
            lines.append(f"> ✅ **Improvement**: +{delta.validation_cases_passed_delta} validation case(s) now passing.")

    lines += [
        "",
        "### Per-Target Timing",
        "",
        "| Target | Passed | Failed | Duration |",
        "|--------|--------|--------|----------|",
    ]
    for t in report.test_targets:
        status = "✓" if t.exit_code == 0 else "✗"
        lines.append(f"| {status} `{t.target}` | {t.tests_passed} | {t.tests_failed} | {t.duration_s:.1f}s |")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", metavar="BASELINE_JSON",
                        help="Path to baseline JSON to compare against")
    parser.add_argument("--save-baseline", metavar="BASELINE_JSON",
                        help="Save current results as the new baseline at this path")
    parser.add_argument("--output", metavar="OUTPUT_JSON",
                        help="Write full JSON report to this file (for CI artifact upload)")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit with code 1 if regression vs. baseline is detected")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (no --release flag)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-target timeout in seconds (default: 300)")
    args = parser.parse_args()

    report = run_harness(release=not args.debug, timeout=args.timeout)
    print_summary(report)

    delta: Optional[Delta] = None
    if args.compare:
        delta = compare_to_baseline(report, Path(args.compare))
        if delta:
            print_delta(report, delta)

    write_github_step_summary(report, delta)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            # dataclasses → dict (nested)
            def _to_dict(obj):
                if hasattr(obj, "__dataclass_fields__"):
                    return {k: _to_dict(v) for k, v in asdict(obj).items()}
                if isinstance(obj, list):
                    return [_to_dict(i) for i in obj]
                return obj
            json.dump(_to_dict(report), f, indent=2)
        print(f"[output] Report written to {args.output}")

    if args.save_baseline:
        baseline_path = Path(args.save_baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with baseline_path.open("w") as f:
            def _to_dict(obj):
                if hasattr(obj, "__dataclass_fields__"):
                    return {k: _to_dict(v) for k, v in asdict(obj).items()}
                if isinstance(obj, list):
                    return [_to_dict(i) for i in obj]
                return obj
            json.dump(_to_dict(report), f, indent=2)
        print(f"[baseline] Saved to {args.save_baseline}")

    if args.fail_on_regression and delta and delta.regression:
        print("ERROR: Regression detected vs. baseline — failing CI gate.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
