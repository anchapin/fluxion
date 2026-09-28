"""Tests for ``scripts/check_ashrae_140_fabric_regression.py`` -- Issue #4208.

The fabric harness gate parses the ``[#3986-A+2 Case ...]`` measurement /
parity lines printed by
``tests/all_tests/ashrae_140_fabric_multiselector.rs`` and fails on
regression beyond the recorded baseline at
``tests/reference_data/ashrae_140_fabric/baseline.json``.

The tests below drive the pure functions (``parse_measurements``,
``aggregate``) and the ``main()`` exit-code contract (0 = PASS,
1 = REGRESSION, 2 = argument/baseline error) against hermetic
``tmp_path`` fixtures -- a realistic three-cases x two-selectors harness
output, a planted component regression, renamed tags / metric keys, and
missing files. The end-to-end against the real tree is the direct
script invocation in ``ashrae_140_fabric.yml`` (path-filtered required
check, Issue #3810 cohort).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "check_ashrae_140_fabric_regression"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the fabric regression gate."""
    return load_script(SCRIPT_NAME)


def _baseline_fixture() -> dict:
    """Baseline JSON mirroring the real
    ``tests/reference_data/ashrae_140_fabric/baseline.json`` shape."""
    return {
        "captured_commit": "test-fixture",
        "regression_tolerance_rel_pct": 10.0,
        "regression_tolerance_ratio_abs": 0.1,
        "metrics": {
            "case_600_5r1c": {"H_mwh": 4.901, "C_mwh": 2.620},
            "case_600_9r4c": {"H_mwh": 4.558, "C_mwh": 4.858},
            "case_900_5r1c": {"H_mwh": 1.633, "C_mwh": 0.910},
            "case_900_9r4c": {"H_mwh": 1.633, "C_mwh": 0.910},
            "case_950_5r1c": {"H_mwh": 0.000, "C_mwh": 0.250},
            "case_950_9r4c": {"H_mwh": 0.000, "C_mwh": 0.250},
        },
        "parity_ratios": {
            "case_600": {"ratio_H": 1.0753, "ratio_C": 0.5393},
            "case_900": {"ratio_H": 1.0000, "ratio_C": 1.0000},
            "case_950": {"ratio_H": 1.0000, "ratio_C": 1.0000},
        },
    }


def _harness_log(values: dict[tuple[str, str, str], float] | None = None,
                 parity: dict[str, tuple[float, float]] | None = None) -> str:
    """Render a realistic harness log: 3 cases x 2 selectors x H/C plus
    the per-case parity lines. ``values`` overrides individual
    ``(case, selector, metric)`` measurements; ``parity`` overrides
    ``case -> (ratio_H, ratio_C)``."""
    base_values = {
        ("600", "5r1c", "H"): 4.901, ("600", "5r1c", "C"): 2.620,
        ("600", "9r4c", "H"): 4.558, ("600", "9r4c", "C"): 4.858,
        ("900", "5r1c", "H"): 1.633, ("900", "5r1c", "C"): 0.910,
        ("900", "9r4c", "H"): 1.633, ("900", "9r4c", "C"): 0.910,
        ("950", "5r1c", "H"): 0.000, ("950", "5r1c", "C"): 0.250,
        ("950", "9r4c", "H"): 0.000, ("950", "9r4c", "C"): 0.250,
    }
    base_values.update(values or {})
    base_parity = {
        "600": (1.0753, 0.5393),
        "900": (1.0000, 1.0000),
        "950": (1.0000, 1.0000),
    }
    base_parity.update(parity or {})

    lines: list[str] = []
    for case in ("600", "900", "950"):
        for selector in ("5r1c", "9r4c"):
            for metric in ("H", "C"):
                v = base_values[(case, selector, metric)]
                lines.append(
                    f"[#3986-A+2 Case {case} / {selector}] {metric}={v:.3f} MWh"
                )
        rh, rc = base_parity[case]
        lines.append(
            f"[#3986-A+2 Case {case} / parity] ratio_H={rh:.4f} ratio_C={rc:.4f}"
        )
    return "\n".join(lines) + "\n"


def _write_repo(tmp_path: Path, log_text: str,
                baseline: dict | None = None) -> tuple[Path, Path]:
    """Write a mock (log, baseline.json) pair; return their paths."""
    log = tmp_path / "fabric_output.txt"
    log.write_text(log_text, encoding="utf-8")
    base = tmp_path / "baseline.json"
    if baseline is not None:
        base.write_text(json.dumps(baseline), encoding="utf-8")
    return log, base


# ---------------------------------------------------------------------------
# parse_measurements(): realistic three-cases x two-selectors output
# ---------------------------------------------------------------------------


def test_parse_measurements_consumes_full_harness_output(checker):
    measurements, parity = checker.parse_measurements(_harness_log())

    # 3 cases x 2 selectors x 2 metrics = 12 measurement lines.
    assert len(measurements) == 12
    for case in ("600", "900", "950"):
        for selector in ("5r1c", "9r4c"):
            for metric in ("H", "C"):
                assert (case, selector, metric) in measurements
    assert measurements[("600", "5r1c", "H")] == pytest.approx(4.901)
    assert measurements[("950", "9r4c", "C")] == pytest.approx(0.250)

    # 3 parity lines.
    assert set(parity) == {"600", "900", "950"}
    assert parity["600"]["ratio_H"] == pytest.approx(1.0753)
    assert parity["600"]["ratio_C"] == pytest.approx(0.5393)


# ---------------------------------------------------------------------------
# aggregate(): PASS / REGRESSION verdicts
# ---------------------------------------------------------------------------


def test_aggregate_pass_on_in_tolerance_log(checker):
    measurements, parity = checker.parse_measurements(_harness_log())
    verdict, log_lines = checker.aggregate(
        measurements, parity, _baseline_fixture())
    assert verdict == checker.PASS
    assert any("PASS: fabric harness holds" in line for line in log_lines)


def test_aggregate_fails_on_planted_component_regression(checker):
    """A single component drifted 20% beyond the 10% tolerance fails."""
    log = _harness_log(values={("900", "5r1c", "H"): 1.633 * 1.20})
    measurements, parity = checker.parse_measurements(log)
    verdict, log_lines = checker.aggregate(
        measurements, parity, _baseline_fixture())
    assert verdict == checker.REGRESSION
    joined = "\n".join(log_lines)
    assert "case_900_5r1c" in joined
    assert "value drifted" in joined


def test_aggregate_fails_on_renamed_selector_tag(checker):
    """A renamed selector tag (`5R1C` vs `5r1c`) drops the measurement --
    missing lines are themselves a regression (filter/format drift)."""
    log = _harness_log().replace(
        "[#3986-A+2 Case 600 / 5r1c]", "[#3986-A+2 Case 600 / 5R1C]")
    measurements, parity = checker.parse_measurements(log)
    assert ("600", "5r1c", "H") not in measurements
    verdict, log_lines = checker.aggregate(
        measurements, parity, _baseline_fixture())
    assert verdict == checker.REGRESSION
    assert any("missing measurement" in line for line in log_lines)


def test_aggregate_fails_on_renamed_metric_key(checker):
    """A renamed metric key (`Heating=` vs `H=`) is not parsed -- the
    missing measurement fails the gate."""
    log = _harness_log().replace(
        "[#3986-A+2 Case 900 / 9r4c] H=", "[#3986-A+2 Case 900 / 9r4c] Heating=")
    measurements, _ = checker.parse_measurements(log)
    assert ("900", "9r4c", "H") not in measurements
    verdict, _ = checker.aggregate(
        measurements, checker.parse_measurements(log)[1], _baseline_fixture())
    assert verdict == checker.REGRESSION


def test_aggregate_fails_on_missing_parity_line(checker):
    log = "\n".join(
        line for line in _harness_log().splitlines()
        if "[#3986-A+2 Case 950 / parity]" not in line) + "\n"
    measurements, parity = checker.parse_measurements(log)
    assert "950" not in parity
    verdict, log_lines = checker.aggregate(
        measurements, parity, _baseline_fixture())
    assert verdict == checker.REGRESSION
    assert any("missing parity line for case 950" in line for line in log_lines)


def test_aggregate_fails_on_parity_ratio_drift(checker):
    """A 10pp+ shift in a selector-parity ratio means a selector silently
    changed paths -- fails even when absolute values hold."""
    log = _harness_log(parity={"600": (1.3000, 0.5393)})
    measurements, parity = checker.parse_measurements(log)
    verdict, log_lines = checker.aggregate(
        measurements, parity, _baseline_fixture())
    assert verdict == checker.REGRESSION
    assert any("ratio_H drifted" in line for line in log_lines)


# ---------------------------------------------------------------------------
# main(): exit-code contract 0 / 1 / 2
# ---------------------------------------------------------------------------


def test_main_exit_0_on_clean_log(checker, tmp_path, capsys):
    log, base = _write_repo(tmp_path, _harness_log(), _baseline_fixture())
    assert checker.main([str(log), "--baseline", str(base)]) == 0
    assert "PASS: fabric harness holds" in capsys.readouterr().out


def test_main_exit_1_on_regression(checker, tmp_path, capsys):
    log_text = _harness_log(values={("600", "9r4c", "C"): 4.858 * 1.25})
    log, base = _write_repo(tmp_path, log_text, _baseline_fixture())
    assert checker.main([str(log), "--baseline", str(base)]) == 1
    out = capsys.readouterr().out
    assert "FAIL:" in out
    assert "case_600_9r4c" in out


def test_main_exit_2_on_missing_log(checker, tmp_path, capsys):
    _, base = _write_repo(tmp_path, _harness_log(), _baseline_fixture())
    missing = tmp_path / "nope.txt"
    assert checker.main([str(missing), "--baseline", str(base)]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_main_exit_2_on_missing_baseline(checker, tmp_path, capsys):
    log, _ = _write_repo(tmp_path, _harness_log(), baseline=None)
    missing = tmp_path / "nope.json"
    assert checker.main([str(log), "--baseline", str(missing)]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_main_exit_2_on_invalid_baseline_json(checker, tmp_path, capsys):
    log, base = _write_repo(tmp_path, _harness_log(), _baseline_fixture())
    base.write_text("{not valid json", encoding="utf-8")
    assert checker.main([str(log), "--baseline", str(base)]) == 2
    assert "ERROR" in capsys.readouterr().err
