"""
Tests for ``scripts/ashrae_benchmark_harness.py`` — Issue #1847.

This module exercises the benchmark-config parsing, the regex-driven
loaders, and the baseline-comparison delta calculations.  All
``subprocess`` calls are mocked so no cargo invocation occurs.
"""

from __future__ import annotations

import io
import json
import subprocess
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ashrae_benchmark_harness as abh


# ---------------------------------------------------------------------------
# _parse_target_output — Rust-test-runner output.
# ---------------------------------------------------------------------------


def test_parse_target_output_extracts_passed_failed_and_names():
    out = """
running 47 tests
test result: ok. 45 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out

FAILED tests::test_one
FAILED tests::test_two
"""
    passed, failed, names = abh._parse_target_output(out)
    assert passed == 45
    assert failed == 2
    assert names == ["tests::test_one", "tests::test_two"]


def test_parse_target_output_no_results_defaults_to_zero():
    passed, failed, names = abh._parse_target_output("no rust output here")
    assert (passed, failed, names) == (0, 0, [])


# ---------------------------------------------------------------------------
# _parse_validation_output — Cases / Pass Rate / MAE (bullet 3 of issue).
# ---------------------------------------------------------------------------


class TestParseValidationOutput:
    """Grouped regex edge cases (issue: "MAE/pass-rate regex parsing highest-value")."""

    def test_basic_case_is_parsed(self):
        out = (
            "Case 600 : Heating=2.00 (Ref: 1.00-3.00), "
            "Cooling=1.20 (Ref: 0.80-1.60)\n"
            "Pass Rate: 90.0% ... Passed: 9 ... Failed: 1\n"
            "Mean Absolute Error: 2.50%\n"
        )
        cases, pass_rate, mae = abh._parse_validation_output(out)
        assert len(cases) == 1
        case = cases[0]
        assert case.case_id == "600"
        assert case.heating_pass is True
        assert case.cooling_pass is True
        assert case.overall_pass is True
        assert pass_rate == 90.0
        assert mae == 2.50

    def test_inf_in_reference_range_always_passes(self):
        """``inf`` as a reference bound means "no upper/lower limit" — must pass."""
        out = "Case 600 : Heating=inf (Ref: 0.00-inf), Cooling=inf (Ref: 0.00-inf)"
        cases, _, _ = abh._parse_validation_output(out)
        assert cases[0].heating_pass is True
        assert cases[0].cooling_pass is True
        assert cases[0].overall_pass is True

    def test_inf_pass_rate_and_mae_supported(self):
        """``Pass Rate: inf%`` — the issue's headline edge case."""
        out = (
            "Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)\n"
            "  Pass Rate: inf% ... Passed: 10 ... Failed: 0\n"
            "  Mean Absolute Error: inf%\n"
        )
        cases, pass_rate, mae = abh._parse_validation_output(out)
        assert pass_rate == float("inf")
        assert mae == float("inf")

    def test_leading_whitespace_in_summary_is_parsed(self):
        """rust test output often prefixes 'Pass Rate' with indentation."""
        out = (
            "Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)\n"
            "\n        Pass Rate: 80.0% ... Passed: 8 ... Failed: 2\n"
            "        Mean Absolute Error: 4.20%\n"
        )
        _, pass_rate, mae = abh._parse_validation_output(out)
        assert pass_rate == 80.0
        assert mae == 4.20

    def test_failing_case_marks_overall_pass_false(self):
        out = "Case 600 : Heating=99.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)"
        cases, _, _ = abh._parse_validation_output(out)
        assert cases[0].heating_pass is False
        assert cases[0].overall_pass is False

    def test_multiple_cases_track_each_independently(self):
        out = (
            "Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)\n"
            "Case 900 : Heating=99.0 (Ref: 4.50-5.50), Cooling=8.0 (Ref: 7.50-8.50)\n"
        )
        cases, _, _ = abh._parse_validation_output(out)
        assert [c.case_id for c in cases] == ["600", "900"]
        assert cases[0].overall_pass is True
        assert cases[1].overall_pass is False

    def test_no_summary_falls_back_to_zero(self):
        out = "Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)"
        _, pass_rate, mae = abh._parse_validation_output(out)
        assert pass_rate == 0.0
        assert mae == 0.0

    def test_no_cases_returns_empty_list(self):
        cases, _, _ = abh._parse_validation_output("nothing here at all")
        assert cases == []


# ---------------------------------------------------------------------------
# _git_info — both env fallback and exception fallback paths.
# ---------------------------------------------------------------------------


def test_git_info_uses_subprocess(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_check_output(cmd, **kwargs):
        # Mirror real subprocess.check_output(text=True) shape: it returns str.
        if "rev-parse" in cmd and "--short" in cmd:
            return "abc1234\n"
        if "rev-parse" in cmd and "--abbrev-ref" in cmd:
            return "main\n"
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    sha, branch = abh._git_info()
    assert sha == "abc1234"
    assert branch == "main"


def test_git_info_falls_back_to_env_on_subprocess_exception(monkeypatch):
    """When ``git rev-parse`` fails (no .git dir), env vars drive the result."""
    def fake_check_output(cmd, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    monkeypatch.setenv("GITHUB_SHA", "deadbeefcafe")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature/test")
    sha, branch = abh._git_info()
    assert sha == "deadbee"  # truncated to 7 chars
    assert branch == "feature/test"


def test_git_info_handles_subprocess_called_process(monkeypatch):
    def fake_check_output(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    # The production code is:
    #   sha = os.environ.get("GITHUB_SHA", "unknown")[:7]
    # An empty env value yields "" via .get(); we exercise that path here and
    # separately exercise the "unknown" default via test_git_info_uses_env when
    # the env var is unset.
    monkeypatch.setenv("GITHUB_SHA", "")
    sha, branch = abh._git_info()
    assert sha == ""
    # ``branch`` follows the same pattern with ``GITHUB_REF_NAME``.
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    sha, branch = abh._git_info()
    assert branch == "unknown"


def test_git_info_unknown_default_when_no_env(monkeypatch):
    """When both subprocess and env are unusable, the literal ``"unknown"`` default applies."""

    def fake_check_output(cmd, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    sha, branch = abh._git_info()
    assert sha == "unknown"
    assert branch == "unknown"


# ---------------------------------------------------------------------------
# _run_cargo_test — three return paths.
# ---------------------------------------------------------------------------


def test_run_cargo_test_success(monkeypatch):
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "stdout content"
    fake_proc.stderr = "stderr content"
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)

    out, duration, code = abh._run_cargo_test("ashrae_140_validation")
    assert code == 0
    assert "stdout content" in out
    assert "stderr content" in out
    assert duration >= 0


def test_run_cargo_test_timeout_returns_124(monkeypatch):
    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired("cargo", 5)

    monkeypatch.setattr("subprocess.run", fake_run)
    out, duration, code = abh._run_cargo_test("ashrae_140_validation", timeout=5)
    assert code == 124
    assert "TIMEOUT" in out


def test_run_cargo_test_missing_binary_returns_127(monkeypatch):
    def fake_run(*a, **kw):
        raise FileNotFoundError("cargo not found")

    monkeypatch.setattr("subprocess.run", fake_run)
    out, duration, code = abh._run_cargo_test("ashrae_140_validation")
    assert code == 127
    assert "ERROR" in out


# ---------------------------------------------------------------------------
# run_harness — full pipeline with all subprocesses mocked (Issue bullet 4).
# ---------------------------------------------------------------------------


def _make_run(returncode=0, stdout="", stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_run_harness_aggregates_per_target_data(monkeypatch):
    # git_info is replaced with deterministic values.
    monkeypatch.setattr(abh, "_git_info", lambda: ("abc1234", "develop"))

    # Both cases are within tolerance so both pass.
    summary_output = """
Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
Case 900 : Heating=5.10 (Ref: 4.50-5.50), Cooling=8.10 (Ref: 7.50-8.50)
Pass Rate: 100.0% ... Passed: 2 ... Failed: 0
Mean Absolute Error: 1.50%
test result: ok. 2 passed; 0 failed
"""

    def fake_run(cmd, **kwargs):
        target = cmd[3] if len(cmd) > 3 else "?"
        if target == "ashrae_140_validation":
            return _make_run(0, stdout=summary_output, stderr="")
        return _make_run(0, stdout="test result: ok. 5 passed; 0 failed", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    report = abh.run_harness(release=False, timeout=60)
    assert report.schema_version == abh.SCHEMA_VERSION
    assert report.commit_sha == "abc1234"
    assert report.branch == "develop"
    assert len(report.test_targets) == len(abh.TEST_TARGETS)
    # Comprehensive target parsed two validation cases.
    assert len(report.validation_cases) == 2
    # Both cases pass, so the harness reports 2/2.
    assert report.summary.validation_cases_passed == 2
    assert report.summary.validation_cases_failed == 0
    # Summary pass_rate from the regex parser wins over the per-case fallback.
    assert report.summary.pass_rate == 100.0
    assert report.summary.mae_percent == pytest.approx(1.50)
    # Aggregated rust-test counts: 2 from comprehensive + 5×N from the rest.
    assert report.summary.total_tests_passed >= 2


def test_run_harness_summary_pass_rate_fallback_when_no_match(monkeypatch):
    """When the regex summary isn't in the output, derive pass_rate from per-case data."""
    monkeypatch.setattr(abh, "_git_info", lambda: ("abc", "main"))
    summary_output = """
Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
Case 900 : Heating=99.0 (Ref: 4.50-5.50), Cooling=8.10 (Ref: 7.50-8.50)
"""

    def fake_run(cmd, **kwargs):
        target = cmd[3] if len(cmd) > 3 else "?"
        if target == "ashrae_140_validation":
            return _make_run(0, stdout=summary_output, stderr="")
        return _make_run(0, stdout="test result: ok. 1 passed; 0 failed", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    report = abh.run_harness(release=False, timeout=60)
    # 1 of 2 cases passes → 50.0%.
    assert report.summary.pass_rate == pytest.approx(50.0)


def test_run_harness_marks_target_not_found_when_cargo_missing(monkeypatch):
    monkeypatch.setattr(abh, "_git_info", lambda: ("abc", "main"))

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no cargo")

    monkeypatch.setattr("subprocess.run", fake_run)
    report = abh.run_harness()
    assert all(t.exit_code == 127 for t in report.test_targets)
    assert all("NOT FOUND" in t.notes for t in report.test_targets)


def test_run_harness_handles_per_target_timeouts(monkeypatch):
    monkeypatch.setattr(abh, "_git_info", lambda: ("abc", "main"))

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr("subprocess.run", fake_run)
    report = abh.run_harness(timeout=60)
    assert all(t.exit_code == 124 for t in report.test_targets)


# ---------------------------------------------------------------------------
# compare_to_baseline — Delta computation.
# ---------------------------------------------------------------------------


def _make_report(
    passed: int = 10,
    failed: int = 5,
    pass_rate: float = 66.7,
    mae: float = 2.5,
    duration: float = 100.0,
    tests_passed: int = 30,
    tests_failed: int = 5,
) -> abh.BenchmarkReport:
    summary = abh.BenchmarkSummary(
        total_validation_cases=15,
        validation_cases_passed=passed,
        validation_cases_failed=failed,
        pass_rate=pass_rate,
        mae_percent=mae,
        total_duration_s=duration,
        total_tests_passed=tests_passed,
        total_tests_failed=tests_failed,
    )
    return abh.BenchmarkReport(
        schema_version=abh.SCHEMA_VERSION,
        timestamp="2026-01-01T00:00:00Z",
        commit_sha="abc",
        branch="main",
        summary=summary,
        test_targets=[],
        validation_cases=[],
    )


def test_compare_to_baseline_regression(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "summary": {
            "validation_cases_passed": 12,
            "validation_cases_failed": 3,
            "pass_rate": 80.0,
            "mae_percent": 2.0,
            "total_duration_s": 90.0,
        }
    }))
    cur = _make_report(passed=10, failed=5, pass_rate=66.7, mae=3.0, duration=110.0)
    delta = abh.compare_to_baseline(cur, baseline)
    assert delta is not None
    assert delta.validation_cases_passed_delta == -2
    assert delta.pass_rate_delta == pytest.approx(-13.3)
    assert delta.mae_delta == pytest.approx(1.0)
    assert delta.duration_delta_s == pytest.approx(20.0)
    assert delta.regression is True
    assert delta.improvement is False


def test_compare_to_baseline_improvement(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "summary": {
            "validation_cases_passed": 8,
            "validation_cases_failed": 7,
            "pass_rate": 53.3,
            "mae_percent": 5.0,
            "total_duration_s": 110.0,
        }
    }))
    cur = _make_report(passed=12, failed=3, pass_rate=80.0, mae=2.5, duration=90.0)
    delta = abh.compare_to_baseline(cur, baseline)
    assert delta.validation_cases_passed_delta == 4
    assert delta.improvement is True
    assert delta.regression is False


def test_compare_to_baseline_no_baseline_file(tmp_path: Path):
    cur = _make_report()
    assert abh.compare_to_baseline(cur, tmp_path / "missing.json") is None


def test_compare_to_baseline_invalid_json(tmp_path: Path):
    baseline = tmp_path / "broken.json"
    baseline.write_text("not-json-{")
    cur = _make_report()
    assert abh.compare_to_baseline(cur, baseline) is None


# ---------------------------------------------------------------------------
# print_delta / print_summary — exercise without crashing.
# ---------------------------------------------------------------------------


def test_print_delta_smoke(tmp_path: Path, capsys):
    cur = _make_report(passed=10)
    delta = abh.Delta(
        validation_cases_passed_delta=2,
        validation_cases_failed_delta=-2,
        pass_rate_delta=10.0, mae_delta=-0.5,
        duration_delta_s=-5.0,
        regression=False, improvement=True,
    )
    abh.print_delta(cur, delta)
    out = capsys.readouterr().out
    assert "DELTA" in out
    assert "IMPROVEMENT" in out
    assert "+2" in out


def test_print_summary_with_cases(capsys):
    report = abh.BenchmarkReport(
        schema_version="1",
        timestamp="t",
        commit_sha="c",
        branch="b",
        summary=abh.BenchmarkSummary(
            total_validation_cases=2,
            validation_cases_passed=1,
            validation_cases_failed=1,
            pass_rate=50.0,
            mae_percent=2.0,
            total_duration_s=5.0,
            total_tests_passed=10,
            total_tests_failed=2,
        ),
        test_targets=[],
        validation_cases=[
            abh.ValidationCase(
                case_id="600",
                heating_actual=2.0, heating_ref_min=1.0, heating_ref_max=3.0,
                heating_pass=True,
                cooling_actual=1.0, cooling_ref_min=0.5, cooling_ref_max=1.5,
                cooling_pass=True,
                overall_pass=True,
            ),
            abh.ValidationCase(
                case_id="900",
                heating_actual=5.0, heating_ref_min=4.5, heating_ref_max=5.5,
                heating_pass=True,
                cooling_actual=8.0, cooling_ref_min=7.5, cooling_ref_max=8.5,
                cooling_pass=True,
                overall_pass=True,
            ),
        ],
    )
    abh.print_summary(report)
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "Case" in out and "Pass?" in out
    assert "600" in out and "900" in out


# ---------------------------------------------------------------------------
# write_github_step_summary — exercises summary + delta + per-target table.
# ---------------------------------------------------------------------------


def test_write_github_step_summary_appends_markdown(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    report = abh.BenchmarkReport(
        schema_version="1",
        timestamp="t",
        commit_sha="abc1234",
        branch="main",
        summary=abh.BenchmarkSummary(
            total_validation_cases=3,
            validation_cases_passed=3,
            validation_cases_failed=0,
            pass_rate=100.0,
            mae_percent=1.5,
            total_duration_s=42.0,
            total_tests_passed=33,
            total_tests_failed=0,
        ),
        test_targets=[
            abh.TargetResult(
                target="ashrae_140_validation",
                duration_s=12.0, exit_code=0,
                tests_passed=10, tests_failed=0,
            ),
            abh.TargetResult(
                target="ashrae_140_case_900",
                duration_s=8.0, exit_code=1,
                tests_passed=5, tests_failed=1, failed_test_names=["x"],
            ),
        ],
        validation_cases=[],
    )
    delta = abh.Delta(
        validation_cases_passed_delta=1,
        validation_cases_failed_delta=0,
        pass_rate_delta=5.0, mae_delta=-0.5,
        duration_delta_s=-3.0,
        regression=False, improvement=True,
    )
    abh.write_github_step_summary(report, delta)
    text = summary_path.read_text()
    assert "ASHRAE 140 Benchmark Harness Results" in text
    assert "ashrae_140_validation" in text
    assert "Improvement" in text or "✅" in text
    assert "Regression" not in text  # no regression block


def test_write_github_step_summary_no_env_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    report = _make_report()
    # Should silently no-op without raising.
    abh.write_github_step_summary(report, None)


def test_write_github_step_summary_regression_block(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    report = abh.BenchmarkReport(
        schema_version="1",
        timestamp="t",
        commit_sha="c",
        branch="b",
        summary=abh.BenchmarkSummary(
            total_validation_cases=4,
            validation_cases_passed=2,
            validation_cases_failed=2,
            pass_rate=50.0,
            mae_percent=4.0,
            total_duration_s=10.0,
            total_tests_passed=10,
            total_tests_failed=2,
        ),
        test_targets=[],
        validation_cases=[],
    )
    delta = abh.Delta(
        validation_cases_passed_delta=-2,
        validation_cases_failed_delta=2,
        pass_rate_delta=-30.0,
        mae_delta=2.0,
        duration_delta_s=2.0,
        regression=True, improvement=False,
    )
    abh.write_github_step_summary(report, delta)
    text = summary_path.read_text()
    assert "Regression" in text or "⏬" in text or "⬇️" in text
