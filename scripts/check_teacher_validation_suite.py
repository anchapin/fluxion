#!/usr/bin/env python3
"""Teacher Validation Suite umbrella gate (Refs #3986 / #4116).

Aggregates the three sub-suites of the #3986 teacher validation suite
(ASHRAE 140 fabric + 1052-RP conduction + PCM test-box) into a single
nightly verdict. Registered in `release_gates.yaml::ci.nightly_authority`
as advisory-only per the user's PR-C scope decision.

Sub-suites:
- `ashrae_140_fabric` — ThermalSelector wiring through ASHRAE 140 (PR-A #4119)
- `conduction_1052rp` — ASHRAE 1052-RP analytical conduction (PR #3981)
- `pcm_test_box`     — Phase-change material test-box skeleton (PR-B #4120)

Usage (typical):
    python3 scripts/check_teacher_validation_suite.py \\
        --log ashrae_140_fabric=ashrae.log \\
        --log conduction_1052rp=1052.log \\
        --log pcm_test_box=pcm.log

Exit code:
    0  if every sub-suite PASSes (overall verdict: PASS)
    1  if any sub-suite FAILs (overall verdict: FAIL)
    2  on argument / parse errors (see stderr)

Filtering:
    --sub-suite NAME  aggregates only one sub-suite (used for nightly
                      debugging / re-runs). When set, the other sub-suites
                      are NOT evaluated.

Per-sub-suite parsing:
    Each sub-suite's cargo test output has the standard `test result:`
    line. PASS iff the line is `test result: ok.`; FAIL iff the line is
    `test result: FAILED.` (this includes the case where 0 tests failed
    in some edge configurations — the prefix drives the verdict).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

# ---------- Data classes ----------

# Status enum values: PASS / FAIL. We intentionally do NOT include KNOWN_FAIL
# in this iteration — the issue scope explicitly said "sub-suites get stubbed
# PASS-with-known-empty until everything's ready" and PR-B+1 (real PCM physics)
# is the first time a KNOWN_FAIL distinction would be needed. Adding it later
# is a non-breaking extension (just one more enum value + a field on the
# baseline JSON).

SUB_SUITE_NAMES: tuple[str, ...] = (
    "ashrae_140_fabric",
    "conduction_1052rp",
    "pcm_test_box",
)


@dataclasses.dataclass(frozen=True)
class SubSuiteResult:
    """Per-sub-suite verdict + raw counts parsed from cargo test output."""

    name: str
    observed_pass: int
    observed_fail: int
    observed_ignored: int
    status: str  # "PASS" or "FAIL"

    @property
    def observed_total(self) -> int:
        return self.observed_pass + self.observed_fail


@dataclasses.dataclass(frozen=True)
class AggregateVerdict:
    """Aggregated verdict across all evaluated sub-suites."""

    overall_status: str  # "PASS" or "FAIL"
    per_sub_suite: list[SubSuiteResult]
    filtered_out: list[str] = dataclasses.field(default_factory=list)


# ---------- Pure functions (Slice 1-6) ----------

# Matches cargo's standard `test result:` summary line. Captures:
#   1. overall status word ("ok" or "FAILED")
#   2. passed count
#   3. failed count
#   4. ignored count
_TEST_RESULT_RE = re.compile(
    r"^test result:\s*"
    r"(?P<status>\w+)\.\s*"
    r"(?P<passed>\d+)\s+passed;\s*"
    r"(?P<failed>\d+)\s+failed;\s*"
    r"(?P<ignored>\d+)\s+ignored",
    re.MULTILINE,
)


def parse_sub_suite_result(sub_suite_name: str, log_text: str) -> SubSuiteResult:
    """Parse one sub-suite's cargo test output into a SubSuiteResult.

    Status rule: PASS iff `test result: ok.` is present; FAIL otherwise
    (the absence of the `test result:` line, OR `test result: FAILED.`,
    both yield FAIL — a log with no summary line is treated as a hard
    failure to surface broken pipelines).
    """
    match = _TEST_RESULT_RE.search(log_text)
    if match is None:
        return SubSuiteResult(
            name=sub_suite_name,
            observed_pass=0,
            observed_fail=0,
            observed_ignored=0,
            status="FAIL",
        )
    status_word = match.group("status").lower()
    return SubSuiteResult(
        name=sub_suite_name,
        observed_pass=int(match.group("passed")),
        observed_fail=int(match.group("failed")),
        observed_ignored=int(match.group("ignored")),
        status="PASS" if status_word == "ok" else "FAIL",
    )


def aggregate_verdict(sub_results: list[SubSuiteResult]) -> AggregateVerdict:
    """Aggregate per-sub-suite verdicts.

    Conservative policy: a single FAIL flips the overall verdict to FAIL.
    This ensures teacher-path regressions cannot be masked by other
    sub-suites passing. (Refined in future revisions to support KNOWN_FAIL
    entries alongside PASS for the PCM test-box once PR-B+1 lands.)
    """
    if not sub_results:
        return AggregateVerdict(overall_status="FAIL", per_sub_suite=[])
    overall = (
        "PASS"
        if all(r.status == "PASS" for r in sub_results)
        else "FAIL"
    )
    return AggregateVerdict(
        overall_status=overall,
        per_sub_suite=list(sub_results),
    )


# ---------- CLI (Slice 7-9) ----------

def _parse_log_args(log_args: list[list[str]]) -> dict[str, Path]:
    """Parse `--log NAME=PATH [NAME=PATH ...]` arguments into a mapping.

    `log_args` is a list of lists because each `--log` flag accepts one or
    more NAME=PATH entries (`nargs="+"`). Empty strings, malformed entries,
    and non-existent paths raise `argparse.ArgumentTypeError` so the user's
    bad input surfaces in the standard argparse error path (exit code 2,
    clear message).
    """
    mapping: dict[str, Path] = {}
    for flag_args in log_args:
        for entry in flag_args:
            if "=" not in entry:
                raise argparse.ArgumentTypeError(
                    f"--log entries must be in NAME=PATH form, got: {entry!r}"
                )
            name, path_str = entry.split("=", 1)
            name = name.strip()
            path = Path(path_str.strip())
            if not name:
                raise argparse.ArgumentTypeError(
                    f"--log NAME part is empty in {entry!r}"
                )
            if name not in SUB_SUITE_NAMES:
                raise argparse.ArgumentTypeError(
                    f"--log NAME {name!r} is not a known sub-suite "
                    f"(expected one of: {', '.join(SUB_SUITE_NAMES)})"
                )
            if not path.exists():
                raise argparse.ArgumentTypeError(
                    f"--log PATH does not exist for sub-suite {name!r}: {path}"
                )
            if name in mapping:
                raise argparse.ArgumentTypeError(
                    f"--log NAME {name!r} specified more than once"
                )
            mapping[name] = path
    return mapping


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher Validation Suite umbrella gate (Refs #3986 / #4116). "
            "Aggregates three sub-suites into a single nightly verdict."
        ),
    )
    parser.add_argument(
        "--log",
        action="append",
        nargs="+",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Path to a captured cargo test log for a sub-suite. "
            "Repeatable; one or more NAME=PATH entries per --log flag. "
            "NAME must be one of "
            f"{', '.join(SUB_SUITE_NAMES)}."
        ),
    )
    parser.add_argument(
        "--sub-suite",
        choices=SUB_SUITE_NAMES,
        default=None,
        help=(
            "Optional filter: aggregate only this sub-suite. The other "
            "sub-suites are reported as 'filtered' in the output but "
            "do NOT contribute to the verdict. Useful for nightly "
            "debugging / re-runs of a single sub-suite."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Parse --log arguments.
    try:
        log_paths = _parse_log_args(args.log)
    except argparse.ArgumentTypeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Determine which sub-suites to evaluate.
    if args.sub_suite is not None:
        evaluate = [args.sub_suite]
        filtered_out = [s for s in SUB_SUITE_NAMES if s != args.sub_suite]
    else:
        evaluate = list(SUB_SUITE_NAMES)
        filtered_out = []

    # Require a log for each evaluated sub-suite.
    missing = [s for s in evaluate if s not in log_paths]
    if missing:
        print(
            f"ERROR: missing --log entries for sub-suites: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    # Parse each evaluated sub-suite.
    sub_results: list[SubSuiteResult] = []
    for sub_suite in evaluate:
        log_text = log_paths[sub_suite].read_text(encoding="utf-8", errors="replace")
        sub_results.append(parse_sub_suite_result(sub_suite, log_text))

    verdict = aggregate_verdict(sub_results)
    verdict = dataclasses.replace(verdict, filtered_out=filtered_out)

    # Render verdict to stdout for the CI log.
    print("Teacher Validation Suite (Issue #3986 / PR-C #4116)")
    print("=" * 60)
    for sub in verdict.per_sub_suite:
        print(
            f"  {sub.name:<22} {sub.status:<5} "
            f"(pass={sub.observed_pass}, fail={sub.observed_fail}, "
            f"ignored={sub.observed_ignored})"
        )
    for name in verdict.filtered_out:
        print(f"  {name:<22} FILTERED (--sub-suite filter applied)")
    print("-" * 60)
    print(f"  OVERALL: {verdict.overall_status}")

    # Exit 0 on PASS, non-zero on FAIL. The CI workflow checks the exit code.
    return 0 if verdict.overall_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
