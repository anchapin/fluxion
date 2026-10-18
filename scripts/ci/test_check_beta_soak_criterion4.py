"""Unit tests for scripts/check_beta_soak_criterion4.py (ADR-0025 Option C).

Run from scripts/: /tmp/ci-test-venv/bin/python -m pytest ci/test_check_beta_soak_criterion4.py -q --no-cov
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "check_beta_soak_criterion4.py"
_spec = importlib.util.spec_from_file_location("check_beta_soak_criterion4", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

BASELINE = (
    REPO_ROOT
    / "tests"
    / "reference_data"
    / "gauge_soak"
    / "criterion4_tracked_cohort_baseline.csv"
)


def _names(log_text: str) -> list[mod.TestOutcome]:
    return mod.parse_log(log_text)


GREEN_LOG = """\
test ashrae_140_case_600_series::case_600ff::test_max_temperature ... FAILED
Case 600FF Max Temp: 43.93°C (Ref: 62.40 to 68.40)
test ashrae_140_case_600_series::case_600ff::test_min_temperature ... ok
Case 600FF Min Temp: -12.01°C (Ref: -13.80 to -9.90)
test ashrae_140_case_600_series::case_610::test_annual_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_610::test_annual_cooling ... ignored, physics gap
test ashrae_140_case_600_series::case_610::test_peak_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_610::test_peak_cooling ... ignored, physics gap
test ashrae_140_case_600_series::case_620::test_annual_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_620::test_annual_cooling ... ignored, physics gap
test ashrae_140_case_600_series::case_620::test_peak_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_620::test_peak_cooling ... ignored, physics gap
test ashrae_140_case_600_series::case_630::test_annual_heating ... FAILED
Case 630 Annual Heating: 29.23 MWh (Ref: 4.36-5.14)
test ashrae_140_case_600_series::case_630::test_annual_cooling ... FAILED
Case 630 Annual Cooling: 6.44 MWh (Ref: 2.57-3.07)
test ashrae_140_case_600_series::case_630::test_peak_cooling ... FAILED
Case 630 Peak Cooling: 13.43 kW (Ref: 3.53-4.21)
test ashrae_140_case_600_series::case_630::test_peak_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_640::test_annual_cooling ... FAILED
Case 640 Annual Cooling: 6.83 MWh (Ref: 5.24-5.89)
test ashrae_140_case_600_series::case_640::test_annual_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_640::test_peak_heating ... ignored, physics gap
test ashrae_140_case_600_series::case_640::test_peak_cooling ... ignored, physics gap
test ashrae_140_case_600_series::case_650::test_annual_cooling ... FAILED
Case 650 Annual Cooling: 3.85 MWh (Ref: 4.19-4.95)
test ashrae_140_case_600_series::case_650::test_peak_cooling ... FAILED
Case 650 Peak Cooling: 14.57 kW (Ref: 5.04-6.29)
test ashrae_140_case_600_series::case_650::test_annual_heating_near_zero ... ok
test ashrae_140_case_600_series::case_650ff::test_min_temperature ... FAILED
Case 650FF Min Temp: -17.89°C (Ref: -17.80 to -16.70)
test ashrae_140_case_600_series::case_650ff::test_max_temperature ... FAILED
Case 650FF Max Temp: 43.33°C (Ref: 61.10 to 66.80)
test ashrae_140_case_600_series::free_float_hvac_guard::builders_match_free_float_classification ... ok
test ashrae_140_case_600_series::free_float_hvac_guard::case_600ff_zero_hvac_energy ... ok
test ashrae_140_case_600_series::free_float_hvac_guard::case_650ff_zero_hvac_energy ... ok
test ashrae_140_case_600_series::test_case_640_hourly_peak_week_diagnostic ... ok
test other_suite::case_a::test_b ... ok
"""

# Compact log: only the cohort members (subset fixture for drift edits).
TRACKED = {
    "ashrae_140_case_600_series::case_600ff::test_max_temperature": "43.93",
    "ashrae_140_case_600_series::case_630::test_annual_heating": "29.23",
    "ashrae_140_case_600_series::case_630::test_annual_cooling": "6.44",
    "ashrae_140_case_600_series::case_630::test_peak_cooling": "13.43",
    "ashrae_140_case_600_series::case_640::test_annual_cooling": "6.83",
    "ashrae_140_case_600_series::case_650::test_annual_cooling": "3.85",
    "ashrae_140_case_600_series::case_650::test_peak_cooling": "14.57",
    "ashrae_140_case_600_series::case_650ff::test_min_temperature": "-17.89",
    "ashrae_140_case_600_series::case_650ff::test_max_temperature": "43.33",
}


def test_baseline_csv_is_parseable_and_complete():
    values, meta, hard = mod.parse_baseline(BASELINE)
    assert len(values) == 9
    assert len(hard) == 18
    assert sum(1 for s in hard.values() if s == "pass") == 6
    assert sum(1 for s in hard.values() if s == "ignored") == 12
    for name, m in meta.items():
        assert m["unit"] in {"degC", "MWh", "kW"}


def test_value_line_regex_parses_all_units():
    v = mod.parse_measured_values(GREEN_LOG)
    assert v[("600FF", "Max Temp")] == 43.93
    assert v[("650FF", "Min Temp")] == -17.89
    assert v[("630", "Annual Heating")] == 29.23
    assert v[("650", "Peak Cooling")] == 14.57


def test_green_log_yields_green():
    outcomes = _names(GREEN_LOG)
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    measured = mod.parse_measured_values(GREEN_LOG)
    verdict = mod.evaluate(outcomes, measured, baseline, meta, hard)
    assert verdict.verdict == "green"


def test_tracked_value_drift_fails_night():
    drifted = GREEN_LOG.replace("29.23 MWh", "30.00 MWh")
    outcomes = _names(drifted)
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    measured = mod.parse_measured_values(drifted)
    verdict = mod.evaluate(outcomes, measured, baseline, meta, hard)
    assert verdict.verdict == "tracked-drift"
    assert verdict.drift[0]["test"].endswith("case_630::test_annual_heating")
    # A change inside epsilon does NOT drift.
    inside = GREEN_LOG.replace("29.23 MWh", "29.26 MWh")  # 0.103% -> drift
    inside_ok = GREEN_LOG.replace("29.23 MWh", "29.25 MWh")  # 0.0685% -> ok
    v_in = mod.evaluate(
        _names(inside), mod.parse_measured_values(inside), baseline, meta, hard
    )
    v_ok = mod.evaluate(
        _names(inside_ok), mod.parse_measured_values(inside_ok), baseline, meta, hard
    )
    assert v_in.verdict == "tracked-drift"
    assert v_ok.verdict == "green"


def test_tracked_test_turning_green_requires_repin():
    flipped = GREEN_LOG.replace(
        "test ashrae_140_case_600_series::case_650::test_annual_cooling ... FAILED",
        "test ashrae_140_case_600_series::case_650::test_annual_cooling ... ok",
    )
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    verdict = mod.evaluate(
        _names(flipped), mod.parse_measured_values(flipped), baseline, meta, hard
    )
    assert verdict.verdict == "re-pin-required"
    assert verdict.re_pins == [
        "ashrae_140_case_600_series::case_650::test_annual_cooling"
    ]


def test_hard_tier_regression_is_fail_closed():
    regressed = GREEN_LOG.replace(
        "test ashrae_140_case_600_series::free_float_hvac_guard::case_600ff_zero_hvac_energy ... ok",
        "test ashrae_140_case_600_series::free_float_hvac_guard::case_600ff_zero_hvac_energy ... FAILED",
    )
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    verdict = mod.evaluate(
        _names(regressed), mod.parse_measured_values(regressed), baseline, meta, hard
    )
    assert verdict.verdict == "hard-fail"


def test_quarantine_removal_is_fail_closed():
    unignored = GREEN_LOG.replace(
        "test ashrae_140_case_600_series::case_610::test_annual_heating ... ignored, physics gap",
        "test ashrae_140_case_600_series::case_610::test_annual_heating ... ok",
    )
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    verdict = mod.evaluate(
        _names(unignored), mod.parse_measured_values(unignored), baseline, meta, hard
    )
    assert verdict.verdict == "hard-fail"
    assert "left quarantine" in verdict.reasons[0]


def test_set_drift_is_fail_closed():
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    # Drop a test entirely.
    dropped = "\n".join(
        line
        for line in GREEN_LOG.splitlines()
        if "case_650ff::test_max_temperature" not in line
    )
    verdict = mod.evaluate(
        _names(dropped), mod.parse_measured_values(dropped), baseline, meta, hard
    )
    assert verdict.verdict == "set-drift"
    # Add a new test.
    added = GREEN_LOG + "\ntest ashrae_140_case_600_series::case_new::test_x ... ok\n"
    verdict = mod.evaluate(
        _names(added), mod.parse_measured_values(added), baseline, meta, hard
    )
    assert verdict.verdict == "set-drift"


def test_empty_log_is_fail_closed(tmp_path: Path):
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n", encoding="utf-8")
    rc = mod.main(
        ["--log", str(empty), "--baseline", str(BASELINE), "--github-env",
         str(tmp_path / "env")]
    )
    assert rc == 1


def test_relative_drift_zero_baseline():
    assert mod.relative_drift(0.0, 0.0) == 0.0
    assert mod.relative_drift(1.0, 0.0) == float("inf")


def test_github_env_written(tmp_path: Path):
    log = tmp_path / "c4.log"
    log.write_text(GREEN_LOG, encoding="utf-8")
    env = tmp_path / "gh_env"
    rc = mod.main(["--log", str(log), "--baseline", str(BASELINE),
                   "--github-env", str(env)])
    assert rc == 0
    assert "CRITERION4_VERDICT=green" in env.read_text(encoding="utf-8")


def test_real_nightly_route_log_if_present():
    """Integration: the record-time route reproduces green (run locally)."""
    log_path = Path("/tmp/r63-c4-nightly-route.log")
    if not log_path.exists():
        pytest.skip("record-time criterion-4 log not present on this machine")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    baseline, meta, hard = mod.parse_baseline(BASELINE)
    verdict = mod.evaluate(
        _names(text), mod.parse_measured_values(text), baseline, meta, hard
    )
    assert verdict.verdict == "green"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
