"""
Tests for ``scripts/s3_worker.py`` — Issue #2830.

Targets the ``parse_cargo_output`` helper, specifically the per-case
aggregation path that previously carried a latent ``NameError`` (it
referenced ``work_unit`` from inside a function that never received it).
Adds coverage for the ``case_id`` filter parameter so the regression
cannot return silently.
"""

from __future__ import annotations

import pytest
import s3_worker

# ---------------------------------------------------------------------------
# parse_cargo_output — MAE/Pass-Rate regex (Issue #2830).
# ---------------------------------------------------------------------------


def test_parse_cargo_output_simple_mae_header():
    out = """
test result: ok.
Mean Absolute Error: 4.21%
Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
Pass Rate: 90.0% ... Passed: 18 ... Failed: 2
"""
    metrics = s3_worker.parse_cargo_output(out)
    assert metrics["heating_mae"] == pytest.approx(4.21)
    assert metrics["overall_pass"] is True


def test_parse_cargo_output_no_match_yields_zero_metrics():
    metrics = s3_worker.parse_cargo_output("nothing to see here")
    assert metrics == {
        "heating_mae": 0.0,
        "cooling_mae": 0.0,
        "peak_heating_mae": 0.0,
        "peak_cooling_mae": 0.0,
        "temperature_mae": 0.0,
        "overall_pass": False,
    }


def test_parse_cargo_output_low_pass_rate_marks_failure():
    out = "Mean Absolute Error: 5.0% Pass Rate: 60.0% Passed: 6 Failed: 4"
    metrics = s3_worker.parse_cargo_output(out)
    assert metrics["overall_pass"] is False  # < 80% threshold


def test_parse_cargo_output_case_pattern_does_not_raise_name_error():
    """Regression test for Issue #2830.

    Before the fix, every per-case match would raise ``NameError: name
    'work_unit' is not defined`` because ``parse_cargo_output`` referenced
    ``work_unit.case_id`` without ever receiving it. ``run_simulation``'s
    broad ``except Exception`` swallowed the failure, so the per-case
    aggregation path had zero coverage and silently produced no errors.
    The fix threads ``case_id`` through as an explicit parameter; this
    test asserts the path actually executes on matching cargo output.
    """
    out = """
Case 600 : Heating=5.00 (Ref: 1.00-3.00), Cooling=2.00 (Ref: 0.50-1.50)
Pass Rate: 50.0% Passed: 1 Failed: 1
"""
    metrics = s3_worker.parse_cargo_output(out, case_id="600")
    # Heating: ref avg = (1+3)/2 = 2.0, sim = 5.0, |5-2|/2 = 150%
    assert metrics["heating_mae"] == pytest.approx(150.0)
    # Cooling: ref avg = (0.5+1.5)/2 = 1.0, sim = 2.0, |2-1|/1 = 100%
    assert metrics["cooling_mae"] == pytest.approx(100.0)
    assert metrics["overall_pass"] is False


def test_parse_cargo_output_case_filter_excludes_non_matching_cases():
    """``case_id`` selects only matching cases; non-matching cases are ignored."""
    out = """
Case 195 : Heating=0.50 (Ref: 0.00-1.00), Cooling=0.10 (Ref: -0.20-0.40)
Case 470 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
"""
    metrics_195 = s3_worker.parse_cargo_output(out, case_id="195")
    metrics_470 = s3_worker.parse_cargo_output(out, case_id="470")
    # Heating midpoint for 195: (0+1)/2 = 0.5, sim = 0.5, error = 0
    assert metrics_195["heating_mae"] == pytest.approx(0.0)
    # Heating midpoint for 470: (1+3)/2 = 2.0, sim = 2.0, error = 0
    assert metrics_470["heating_mae"] == pytest.approx(0.0)
    assert metrics_195["cooling_mae"] == pytest.approx(0.0)
    assert metrics_470["cooling_mae"] == pytest.approx(0.0)


def test_parse_cargo_output_empty_case_id_skips_per_case_aggregation():
    """When ``case_id`` is empty, the ``case_id and ...`` filter short-circuits
    and no per-case entries contribute to MAE — matching the documented
    behavior in :func:`scripts.autonomous_parameter_sweep.parse_cargo_output`
    where ``case_filter=""`` disables the per-case branch entirely.

    The global ``Mean Absolute Error`` header is still parsed correctly,
    so the function returns without raising ``NameError``.
    """
    out = """
Mean Absolute Error: 3.50%
Case 195 : Heating=1.00 (Ref: 0.00-2.00), Cooling=1.00 (Ref: 0.00-2.00)
Case 600 : Heating=5.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
"""
    metrics = s3_worker.parse_cargo_output(out, case_id="")
    # Header regex still populates heating_mae (set by ``mae_pattern`` loop
    # before the per-case branch).
    assert metrics["heating_mae"] == pytest.approx(3.50)
    # Per-case branch is skipped entirely — cooling_mae stays at the default.
    assert metrics["cooling_mae"] == 0.0


def test_parse_cargo_output_default_case_id_is_empty_string():
    """Backward-compatible default — callers may still invoke with one arg."""
    out = """
Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.20 (Ref: 0.80-1.60)
"""
    # Single-argument call must still work (legacy callers + tests).
    metrics = s3_worker.parse_cargo_output(out)
    # Heating midpoint = 2.0, sim = 2.0 — error is 0
    assert metrics["heating_mae"] == pytest.approx(0.0)
