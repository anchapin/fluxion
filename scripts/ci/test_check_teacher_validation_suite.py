#!/usr/bin/env python3
"""Behavior tests for scripts/check_teacher_validation_suite.py (Refs #3986 / #4116).

These tests exercise the pure-function verdict logic and the CLI's main()
entry point. They follow the shape of test_check_strict_energy_gate_regression.py
which is the closest analog in this repo.

The umbrella gate aggregates three sub-suites (Refs #3986):
- ASHRAE 140 fabric (PR-A #4119) — ThermalSelector wiring through ASHRAE 140
- 1052-RP conduction (PR-#3981) — ASHRAE 1052-RP analytical conduction
- PCM test-box (PR-B #4120) — Phase-change material test-box skeleton

Each sub-suite's verdict is one of: PASS / FAIL. The aggregate verdict is:
- PASS iff all sub-suites PASS
- FAIL iff at least one sub-suite FAILs (including any "KNOWN-FAIL" expectations)

Sub-suite naming follows the sub-suite CLI flag convention:
- `--sub-suite ashrae_140_fabric`
- `--sub-suite conduction_1052rp`
- `--sub-suite pcm_test_box`

The script is registered in `release_gates.yaml::ci.nightly_authority` as an
advisory-only nightly gate per the user's PR-C scope decision (Refs #4116).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Path setup: scripts/ci/test_*.py files are pytest-discovered by the
# scripts-tests workflow. The umbrella script lives one directory up at
# scripts/check_teacher_validation_suite.py.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
UMBRElla_SCRIPT = SCRIPTS_DIR / "check_teacher_validation_suite.py"
CONFTEST_PATH = Path(__file__).resolve().parent / "conftest.py"


# ---------- Sample cargo test output fixtures ----------
# These mirror the exact `cargo test -- --list`-style output that the
# umbrella gate parses (one `test result:` line per sub-suite).

PASSING_ASHRAE_LOG = """\
     Running tests/all_tests/main.rs (target/debug/deps/all_tests-abc123)
running 7 tests
test ashrae_140_validator_selector_parity::case_600_default_vs_new_with_default_parity ... ok
test ashrae_140_validator_selector_parity::case_900_explicit_legacy_selector_runs ... ok
test ashrae_140_validator_selector_parity::multi_zone_explicit_legacy_selector_accepted ... ok
test ashrae_140_validator_selector_parity::multi_zone_new_delegates_to_new_with_selector_default ... ok
test ashrae_140_validator_selector_parity::multi_zone_selector_round_trip ... ok
test ashrae_140_validator_selector_parity::new_delegates_to_new_with_selector_default ... ok
test ashrae_140_validator_selector_parity::selector_round_trip ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
"""

FAILING_1052_LOG = """\
     Running tests/all_tests/main.rs (target/debug/deps/all_tests-def456)
running 6 tests
test conduction_1052rp_analytical::test_neumann_solution ... ok
test conduction_1052rp_analytical::test_three_layer_wall ... FAILED
test conduction_1052rp_analytical::test_two_layer_wall ... ok
test conduction_1052rp_analytical::test_baseline_construction ... ok
test conduction_1052rp_analytical::test_high_mass_check ... ok
test conduction_1052rp_analytical::test_thermal_bridge ... ok

test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
"""

PASSING_PCM_LOG = """\
     Running tests/all_tests/main.rs (target/debug/deps/all_tests-ghi789)
running 8 tests
test teacher_validation_pcm_box::apparent_cp_latent_band_height ... ok
test teacher_validation_pcm_box::apparent_cp_sensible_outside_melting_band ... ok
test teacher_validation_pcm_box::enthalpy_linear_above_liquidus ... ok
test teacher_validation_pcm_box::enthalpy_linear_below_solidus ... ok
test teacher_validation_pcm_box::rt27_constructs_with_nominal_properties ... ok
test teacher_validation_pcm_box::test_box_apparent_cp_at_wall_delegates_to_material ... ok
test teacher_validation_pcm_box::test_box_constructs_with_default_pcm ... ok
test teacher_validation_pcm_box::test_box_solid_fraction_returns_none_without_reference_data ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.45s
"""


def _run_script(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run the umbrella script with the given args and capture output.

    Sets cwd to the repo root so the script can find relative paths
    (tests/reference_data/, scripts/, etc.).
    """
    repo_root = SCRIPTS_DIR.parent
    return subprocess.run(
        [sys.executable, str(UMBRElla_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        input=stdin,
        check=False,
    )


# ---------- Slice 1 (RED→GREEN): parse_sub_suite_result_passing_cargo_log ----------
# Single-sub-suite result type and parser.

def test_parse_passing_cargo_log_returns_pass() -> None:
    """A clean cargo test output with all-passing tests yields a PASS
    `SubSuiteResult` for the given sub-suite."""
    from check_teacher_validation_suite import parse_sub_suite_result

    result = parse_sub_suite_result(
        sub_suite_name="ashrae_140_fabric",
        log_text=PASSING_ASHRAE_LOG,
    )
    assert result.name == "ashrae_140_fabric"
    assert result.observed_pass == 7
    assert result.observed_fail == 0
    assert result.observed_ignored == 0
    assert result.status == "PASS"


# ---------- Slice 2 (RED→GREEN): parse_sub_suite_result_failing_cargo_log ----------

def test_parse_failing_cargo_log_returns_fail() -> None:
    """A cargo test output with at least one failure yields FAIL status."""
    from check_teacher_validation_suite import parse_sub_suite_result

    result = parse_sub_suite_result(
        sub_suite_name="conduction_1052rp",
        log_text=FAILING_1052_LOG,
    )
    assert result.name == "conduction_1052rp"
    assert result.observed_pass == 4
    assert result.observed_fail == 1
    assert result.status == "FAIL"


def test_parse_failing_log_does_not_silently_pass() -> None:
    """A log line `test result: FAILED` must not be reported as PASS even if
    the line is followed by `N passed` (N > 0). The status is driven by the
    `test result:` prefix."""
    from check_teacher_validation_suite import parse_sub_suite_result

    result = parse_sub_suite_result(
        sub_suite_name="conduction_1052rp",
        log_text=FAILING_1052_LOG,
    )
    assert result.status != "PASS"


# ---------- Slice 3 (RED→GREEN): aggregate_verdict_all_pass_returns_overall_pass ----------

def test_aggregate_all_pass_returns_overall_pass() -> None:
    """When every sub-suite PASSes, the aggregate verdict is PASS."""
    from check_teacher_validation_suite import (
        parse_sub_suite_result,
        aggregate_verdict,
    )

    sub_results = [
        parse_sub_suite_result("ashrae_140_fabric", PASSING_ASHRAE_LOG),
        parse_sub_suite_result("conduction_1052rp", PASSING_ASHRAE_LOG),  # any passing log
        parse_sub_suite_result("pcm_test_box", PASSING_PCM_LOG),
    ]
    verdict = aggregate_verdict(sub_results)
    assert verdict.overall_status == "PASS"


# ---------- Slice 4 (RED→GREEN): aggregate_verdict_any_fail_returns_overall_fail ----------

def test_aggregate_any_fail_returns_overall_fail() -> None:
    """When at least one sub-suite FAILs, the aggregate verdict is FAIL."""
    from check_teacher_validation_suite import (
        parse_sub_suite_result,
        aggregate_verdict,
    )

    sub_results = [
        parse_sub_suite_result("ashrae_140_fabric", PASSING_ASHRAE_LOG),
        parse_sub_suite_result("conduction_1052rp", FAILING_1052_LOG),
        parse_sub_suite_result("pcm_test_box", PASSING_PCM_LOG),
    ]
    verdict = aggregate_verdict(sub_results)
    assert verdict.overall_status == "FAIL"


# ---------- Slice 5 (RED→GREEN): aggregate_verdict_mixed_pass_and_fail_returns_overall_fail ----------

def test_aggregate_mixed_pass_and_fail_returns_overall_fail() -> None:
    """Even with 2 of 3 sub-suites passing, a single FAIL flips the overall
    verdict to FAIL. This is the conservative aggregation policy: a teacher-path
    regression must not be masked by other suites passing."""
    from check_teacher_validation_suite import (
        parse_sub_suite_result,
        aggregate_verdict,
    )

    sub_results = [
        parse_sub_suite_result("ashrae_140_fabric", PASSING_ASHRAE_LOG),
        parse_sub_suite_result("conduction_1052rp", PASSING_ASHRAE_LOG),
        parse_sub_suite_result("pcm_test_box", FAILING_1052_LOG),
    ]
    verdict = aggregate_verdict(sub_results)
    assert verdict.overall_status == "FAIL"


# ---------- Slice 6 (RED→GREEN): aggregate_verdict_includes_per_sub_suite_detail ----------

def test_aggregate_verdict_includes_per_sub_suite_detail() -> None:
    """The aggregate verdict struct exposes per-sub-suite detail so the CI
    log can render which sub-suite(s) failed without re-parsing."""
    from check_teacher_validation_suite import (
        parse_sub_suite_result,
        aggregate_verdict,
    )

    sub_results = [
        parse_sub_suite_result("ashrae_140_fabric", PASSING_ASHRAE_LOG),
        parse_sub_suite_result("conduction_1052rp", FAILING_1052_LOG),
        parse_sub_suite_result("pcm_test_box", PASSING_PCM_LOG),
    ]
    verdict = aggregate_verdict(sub_results)
    assert len(verdict.per_sub_suite) == 3
    by_name = {r.name: r for r in verdict.per_sub_suite}
    assert by_name["ashrae_140_fabric"].status == "PASS"
    assert by_name["conduction_1052rp"].status == "FAIL"
    assert by_name["pcm_test_box"].status == "PASS"


# ---------- Slice 7 (RED→GREEN): main_returns_zero_when_all_subsuites_pass ----------

def test_main_returns_zero_when_all_subsuites_pass(
    tmp_path: Path,
) -> None:
    """When all three sub-suite logs report PASS, main() exits 0 (gate green)."""
    log_ashrae = tmp_path / "ashrae.log"
    log_1052 = tmp_path / "1052.log"
    log_pcm = tmp_path / "pcm.log"
    log_ashrae.write_text(PASSING_ASHRAE_LOG)
    log_1052.write_text(PASSING_ASHRAE_LOG)  # re-use passing log shape
    log_pcm.write_text(PASSING_PCM_LOG)

    result = _run_script(
        "--log",
        f"ashrae_140_fabric={log_ashrae}",
        "--log",
        f"conduction_1052rp={log_1052}",
        "--log",
        f"pcm_test_box={log_pcm}",
    )
    assert result.returncode == 0, f"expected 0, got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"


# ---------- Slice 8 (RED→GREEN): main_returns_nonzero_when_any_subsuite_fails ----------

def test_main_returns_nonzero_when_any_subsuite_fails(
    tmp_path: Path,
) -> None:
    """When at least one sub-suite FAILs, main() exits non-zero (gate red)."""
    log_ashrae = tmp_path / "ashrae.log"
    log_1052 = tmp_path / "1052.log"
    log_pcm = tmp_path / "pcm.log"
    log_ashrae.write_text(PASSING_ASHRAE_LOG)
    log_1052.write_text(FAILING_1052_LOG)
    log_pcm.write_text(PASSING_PCM_LOG)

    result = _run_script(
        "--log",
        f"ashrae_140_fabric={log_ashrae}",
        "--log",
        f"conduction_1052rp={log_1052}",
        "--log",
        f"pcm_test_box={log_pcm}",
    )
    assert result.returncode != 0, f"expected non-zero, got 0\nstdout: {result.stdout}\nstderr: {result.stderr}"
    # The verdict must be visible in stdout for the CI log.
    assert "FAIL" in result.stdout


# ---------- Slice 9 (RED→GREEN): main_with_sub_suite_filter_runs_only_one ----------

def test_main_with_sub_suite_filter_only_runs_one(
    tmp_path: Path,
) -> None:
    """`--sub-suite <name>` filters the aggregation to a single sub-suite.
    Used by nightly debugging / re-runs; main() must not require logs for
    the other sub-suites in this mode."""
    log_ashrae = tmp_path / "ashrae.log"
    log_ashrae.write_text(PASSING_ASHRAE_LOG)

    result = _run_script(
        "--sub-suite",
        "ashrae_140_fabric",
        "--log",
        f"ashrae_140_fabric={log_ashrae}",
    )
    # Single passing sub-suite → overall PASS → exit 0.
    assert result.returncode == 0, f"expected 0, got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    # The output should mention only the filtered sub-suite, not the others.
    assert "ashrae_140_fabric" in result.stdout
    # The other two sub-suites should NOT be in the output (or explicitly marked filtered).
    assert "conduction_1052rp" not in result.stdout or "filtered" in result.stdout.lower()
    assert "pcm_test_box" not in result.stdout or "filtered" in result.stdout.lower()
