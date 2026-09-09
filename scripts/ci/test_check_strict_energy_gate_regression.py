"""
Tests for ``scripts/check_strict_energy_gate_regression.py`` -- Issue #1333 / #2506.

Regression guard for the ASHRAE 140 strict ±15% annual-energy gate.
Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_root_hygiene.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``REPO_ROOT`` constant so the default
  baseline path can be substituted with a per-test fixture, then
* drive ``gap_pct_of_mid`` / ``parse_measured`` / ``main()`` through
  clean (in-band), known-fail (still within tolerance), and regression
  (beyond tolerance) scenarios.

The script's two key surfaces are pure functions -- ``gap_pct_of_mid``
and ``parse_measured`` -- plus a CLI ``main()`` that consumes a
``--baseline`` JSON and a captured cargo log file. Each test plants
both inputs in ``tmp_path`` and invokes ``main()`` via
``sys.argv`` injection.

Issue #3572 extended the gate from 2 cases (600/900) to 8 cases
(600/800/810/900/920/950/960/970). The legacy 2-case tests below pin
their scope via ``--require-cases 600,900`` so they keep their original
intent; new tests at the bottom exercise the full 8-case coverage
that the production strict-energy-gate workflow consumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_strict_energy_gate_regression"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the strict-energy regression checker."""
    return load_script(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# gap_pct_of_mid — pure-function gate arithmetic
# ---------------------------------------------------------------------------


def test_gap_zero_when_value_inside_band(checker):
    """In-band value → gap 0.0 (no distance outside the band)."""
    assert checker.gap_pct_of_mid(5.0, 4.0, 6.0) == 0.0


def test_gap_zero_at_band_edges(checker):
    """Band edges are inclusive → gap 0.0 at both lo and hi."""
    assert checker.gap_pct_of_mid(4.0, 4.0, 6.0) == 0.0
    assert checker.gap_pct_of_mid(6.0, 4.0, 6.0) == 0.0


def test_gap_positive_when_value_below_band(checker):
    """Value below band → gap > 0, expressed as % of band midpoint."""
    # band = [4, 6], midpoint = 5, value = 3 -> (4 - 3) / 5 * 100 = 20
    assert checker.gap_pct_of_mid(3.0, 4.0, 6.0) == pytest.approx(20.0)


def test_gap_positive_when_value_above_band(checker):
    """Value above band → gap > 0, expressed as % of band midpoint."""
    # band = [4, 6], midpoint = 5, value = 8 -> (8 - 6) / 5 * 100 = 40
    assert checker.gap_pct_of_mid(8.0, 4.0, 6.0) == pytest.approx(40.0)


def test_gap_infinite_when_band_midpoint_is_zero(checker):
    """Degenerate band (mid <= 0) → gap is infinite (script handles it)."""
    # band = [-2, 0], midpoint = -1, value = 1 -> (1 - 0) / -1 * 100 = -100 (negative!)
    # Script returns inf for mid <= 0 regardless.
    assert checker.gap_pct_of_mid(1.0, -2.0, 0.0) == float("inf")


# ---------------------------------------------------------------------------
# parse_measured — log-line → metric dict
# ---------------------------------------------------------------------------


def test_parse_measured_extracts_both_cases(checker):
    """Lines for Case 600 and Case 900 both yield entries."""
    log = (
        "[#1147 Case 600 strict] H=5.236 MWh (band 4.314-5.836), "
        "C=2.455 MWh (band 4.275-5.784)\n"
        "[#1147 Case 900 strict] H=1.754 MWh (band 1.364-1.846), "
        "C=0.689 MWh (band 2.465-3.335)\n"
    )
    measured = checker.parse_measured(log)
    assert set(measured.keys()) == {"600", "900"}
    assert measured["600"]["H"] == pytest.approx(5.236)
    assert measured["600"]["C"] == pytest.approx(2.455)
    assert measured["600"]["hlo"] == pytest.approx(4.314)
    assert measured["900"]["hhi"] == pytest.approx(1.846)


def test_parse_measured_handles_no_matches(checker):
    """Empty / irrelevant log → empty dict."""
    assert checker.parse_measured("") == {}
    assert checker.parse_measured("no strict lines here\n") == {}


def test_parse_measured_last_match_wins(checker):
    """Duplicate Case 600 strict lines → the last one wins."""
    log = (
        "[#1147 Case 600 strict] H=5.236 MWh (band 4.314-5.836), "
        "C=2.455 MWh (band 4.275-5.784)\n"
        "[#1147 Case 600 strict] H=9.999 MWh (band 4.314-5.836), "
        "C=2.455 MWh (band 4.275-5.784)\n"
    )
    measured = checker.parse_measured(log)
    assert measured["600"]["H"] == pytest.approx(9.999)


# ---------------------------------------------------------------------------
# main() — end-to-end scenarios driven through tmp_path fixtures
# ---------------------------------------------------------------------------


def _write_baseline(tmp_path: Path) -> Path:
    """Plant a baseline JSON mirroring the production file's shape."""
    payload = {
        "captured_commit": "test",
        "regression_tolerance_pp": 5.0,
        "metrics": {
            "case_600_heating": {
                "published_range_mwh": [4.36, 5.79],
                "band_mwh": [4.314, 5.836],
                "value_mwh": 5.236,
                "gap_pct_of_mid": 0.0,
                "status": "pass",
            },
            "case_600_cooling": {
                "published_range_mwh": [3.92, 6.14],
                "band_mwh": [4.275, 5.784],
                "value_mwh": 2.455,
                "gap_pct_of_mid": 36.1865,
                "status": "known_fail",
            },
            "case_900_heating": {
                "published_range_mwh": [1.17, 2.04],
                "band_mwh": [1.364, 1.846],
                "value_mwh": 1.754,
                "gap_pct_of_mid": 0.0,
                "status": "pass",
            },
            "case_900_cooling": {
                "published_range_mwh": [2.13, 3.67],
                "band_mwh": [2.465, 3.335],
                "value_mwh": 0.689,
                "gap_pct_of_mid": 61.241379,
                "status": "known_fail",
            },
        },
    }
    path = tmp_path / "strict_energy_gate_baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_log(
    tmp_path: Path,
    case: str,
    h: float, hlo: float, hhi: float,
    c: float, clo: float, chi: float,
    *,
    case900: tuple[float, float, float, float, float, float] | None = None,
) -> Path:
    """Plant a captured cargo log with one or two ``Case <n> strict`` lines.

    The script requires BOTH Case 600 and Case 900 lines unless
    ``--require-cases`` is overridden; by default we emit a benign
    Case 900 (in-band heating, known-fail cooling within tolerance) so
    the focus of each test stays on the Case 600 metric under
    investigation. Pass ``case900=None`` (and don't write a line) to
    exercise the missing-case parse-failure path.
    """
    lines = [
        f"[#1147 Case {case} strict] "
        f"H={h:.3f} MWh (band {hlo:.3f}-{hhi:.3f}), "
        f"C={c:.3f} MWh (band {clo:.3f}-{chi:.3f})"
    ]
    if case900 is not None:
        h9, hlo9, hhi9, c9, clo9, chi9 = case900
        lines.append(
            f"[#1147 Case 900 strict] "
            f"H={h9:.3f} MWh (band {hlo9:.3f}-{hhi9:.3f}), "
            f"C={c9:.3f} MWh (band {clo9:.3f}-{chi9:.3f})"
        )
    path = tmp_path / "captured.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Benign Case 900 values that mirror the production baseline (in-band H,
# known-fail C within tolerance).
_CASE_900_BASELINE = (
    1.754, 1.364, 1.846,   # H = 1.754 in [1.364, 1.846]
    0.689, 2.465, 3.335,   # C = 0.689 well below [2.465, 3.335]
)


def _invoke(
    checker,
    log_path: Path,
    baseline_path: Path,
    require_cases: str = "600,900",
) -> int:
    """Invoke ``checker.main()`` with synthetic argv.

    The script is a CLI; rather than spawn a subprocess we patch
    ``sys.argv`` and call ``main()`` in-process. ``monkeypatch`` is
    expected to undo the change after the test exits.

    Issue #3572: the production default is now all eight cases
    (600/800/810/900/920/950/960/970). The legacy 600/900 tests below
    pass ``require_cases="600,900"`` so they keep their original
    intent; new tests at the bottom exercise the full default scope.
    """
    saved = sys.argv[:]
    sys.argv[:] = [
        SCRIPT_NAME,
        str(log_path),
        "--baseline",
        str(baseline_path),
        "--require-cases",
        require_cases,
    ]
    try:
        return checker.main()
    finally:
        sys.argv[:] = saved


def test_main_passes_when_measured_matches_baseline(checker, tmp_path):
    """In-band heating + known-fail cooling within tolerance → exit 0."""
    baseline = _write_baseline(tmp_path)
    # Case 600 heating: value 5.236 inside [4.314, 5.836] → gap 0 (PASS).
    # Case 600 cooling: value 2.455 outside [4.275, 5.784]; gap ≈ 36.2 pp
    # which equals the baseline gap → KNOWN-FAIL (no regression).
    log = _write_log(
        tmp_path,
        case="600",
        h=5.236, hlo=4.314, hhi=5.836,
        c=2.455, clo=4.275, chi=5.784,
        case900=_CASE_900_BASELINE,
    )
    assert _invoke(checker, log, baseline) == 0


def test_main_returns_one_when_gap_worsens_beyond_tolerance(checker, tmp_path, capsys):
    """Known-fail metric grows worse than baseline + 5 pp → exit 1.

    This is the regression the gate exists to catch (issue #2506
    acceptance criterion): a regression beyond the documented tolerance
    trips the gate even if the baseline gap was non-zero.
    """
    baseline = _write_baseline(tmp_path)
    # Case 600 cooling: gap should be ~36.2 pp at baseline. Push the value
    # way below the band so the gap blows past baseline + 5 pp tolerance.
    log = _write_log(
        tmp_path,
        case="600",
        h=5.236, hlo=4.314, hhi=5.836,  # H still in-band
        c=0.500, clo=4.275, chi=5.784,   # C far below band
        case900=_CASE_900_BASELINE,
    )
    rc = _invoke(checker, log, baseline)
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "REGRESSION" in out
    assert "case_600_cooling" in out


def test_main_returns_one_when_known_pass_metric_exits_band(checker, tmp_path, capsys):
    """Previously-passing metric now outside the band → exit 1.

    Stricter rule per the script: any movement outside the band by more
    than the tolerance is a regression even if base_gap == 0. Push
    Case 600 heating above the band to trigger this.
    """
    baseline = _write_baseline(tmp_path)
    # band = [4.314, 5.836], midpoint = 5.075, value = 8.0 → gap ≈ 42.7 pp
    # which is well beyond the 5 pp tolerance for a metric that was
    # previously PASSING (gap 0).
    log = _write_log(
        tmp_path,
        case="600",
        h=8.000, hlo=4.314, hhi=5.836,   # H way above band
        c=2.455, clo=4.275, chi=5.784,   # C still known-fail within tolerance
        case900=_CASE_900_BASELINE,
    )
    rc = _invoke(checker, log, baseline)
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "REGRESSION" in out
    assert "case_600_heating" in out


def test_main_returns_one_when_log_missing_required_case(checker, tmp_path, capsys):
    """Log is missing Case 900 → exit 1 (parse failure is itself a regression).

    Issue #2506 explicit requirement: the script must enforce that BOTH
    required cases appear in the log; a missing line means the cargo
    filter / test name drifted.
    """
    baseline = _write_baseline(tmp_path)
    log = _write_log(
        tmp_path,
        case="600",
        h=5.236, hlo=4.314, hhi=5.836,
        c=2.455, clo=4.275, chi=5.784,
        # case900 omitted → log is missing Case 900.
    )
    rc = _invoke(checker, log, baseline)
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "900" in out
    assert "could not parse" in out


# ---------------------------------------------------------------------------
# Issue #3572: extend the gate to eight cases (600/800/810/900/920/950/960/970)
# ---------------------------------------------------------------------------


def _write_baseline_8_cases(tmp_path: Path) -> Path:
    """Plant a baseline JSON covering all eight Issue #3572 cases.

    Each metric is seeded with the current engine-measured value so
    the eight-case tests are reproducible without touching the real
    baseline file. Heating metrics for cases currently in band are
    marked ``pass``; cooling metrics (and any heating metric outside
    the band) are marked ``known_fail`` with the exact gap percentage
    so a future run with the same measured value yields KNOWN-FAIL
    (no regression).
    """
    # (case, h_value, h_lo, h_hi, c_value, c_lo, c_hi)
    measurements = [
        ("600", 5.182, 4.314, 5.836, 2.546, 4.275, 5.784),
        ("800", 5.453, 4.378, 5.923, 2.007, 4.888, 6.612),
        ("810", 1.633, 3.357, 4.543, 0.910, 3.740, 5.060),
        ("900", 1.633, 1.364, 1.846, 0.910, 2.465, 3.335),
        ("920", 2.400, 3.213, 4.347, 1.085, 2.189, 2.961),
        ("950", 0.000, 0.000, 0.000, 0.028, 0.557, 0.753),
        ("960", 2.924, 1.742, 2.357, 0.144, 1.840, 2.490),
        ("970", 3.580, 10.540, 14.260, 1.654, 7.391, 9.999),
    ]
    metrics: dict = {}
    for case, hv, hl, hh, cv, cl, ch in measurements:
        for kind, val, lo, hi in (
            ("heating", hv, hl, hh),
            ("cooling", cv, cl, ch),
        ):
            mid = 0.5 * (lo + hi)
            if mid > 0 and val >= lo and val <= hi:
                gap = 0.0
                status = "pass"
            else:
                if mid <= 0:
                    # Degenerate midpoint (Case 950 heating). With the
                    # post-#3572 fix, an in-band degenerate value still
                    # reports gap 0; the script returns 0 for any value
                    # inside [lo, hi] regardless of midpoint.
                    if val >= lo and val <= hi:
                        gap = 0.0
                        status = "pass"
                    else:
                        gap = float("inf")
                        status = "known_fail"
                elif val < lo:
                    gap = (lo - val) / mid * 100.0
                    status = "known_fail"
                else:
                    gap = (val - hi) / mid * 100.0
                    status = "known_fail"
            metrics[f"case_{case}_{kind}"] = {
                "published_range_mwh": [lo, hi],
                "band_mwh": [lo, hi],
                "value_mwh": val,
                "gap_pct_of_mid": gap,
                "status": status,
            }
    payload = {
        "captured_commit": "test-issue-3572",
        "regression_tolerance_pp": 5.0,
        "metrics": metrics,
    }
    path = tmp_path / "strict_energy_gate_baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_log_8_cases(
    tmp_path: Path,
    *,
    drop_case: str | None = None,
    overrides: dict | None = None,
) -> Path:
    """Plant a captured cargo log containing a strict line per Issue #3572 case.

    ``drop_case`` removes the listed case from the log (default: all eight
    present) so the missing-required-case path is reachable. ``overrides``
    maps case id -> (h, c) tuples to exercise drift/regression scenarios
    on a specific case while keeping the other seven at their benign
    baseline values.
    """
    overrides = dict(overrides or {})
    measurements = [
        ("600", 5.182, 4.314, 5.836, 2.546, 4.275, 5.784),
        ("800", 5.453, 4.378, 5.923, 2.007, 4.888, 6.612),
        ("810", 1.633, 3.357, 4.543, 0.910, 3.740, 5.060),
        ("900", 1.633, 1.364, 1.846, 0.910, 2.465, 3.335),
        ("920", 2.400, 3.213, 4.347, 1.085, 2.189, 2.961),
        ("950", 0.000, 0.000, 0.000, 0.028, 0.557, 0.753),
        ("960", 2.924, 1.742, 2.357, 0.144, 1.840, 2.490),
        ("970", 3.580, 10.540, 14.260, 1.654, 7.391, 9.999),
    ]
    lines: list[str] = []
    for case, hv, hl, hh, cv, cl, ch in measurements:
        if drop_case is not None and case == drop_case:
            continue
        if case in overrides:
            ovr = overrides[case]
            if isinstance(ovr, dict):
                hv = ovr.get("h", hv)
                cv = ovr.get("c", cv)
            else:
                hv, cv = ovr  # type: ignore[misc]
        lines.append(
            f"[#1147 Case {case} strict] "
            f"H={hv:.3f} MWh (band {hl:.3f}-{hh:.3f}), "
            f"C={cv:.3f} MWh (band {cl:.3f}-{ch:.3f})"
        )
    path = tmp_path / "captured.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_measured_consumes_all_eight_cases(checker):
    """Issue #3572 acceptance: regex matches the 8 strict cases (600/800/810/900/920/950/960/970)."""
    log = "\n".join(
        f"[#1147 Case {case} strict] "
        f"H={hv:.3f} MWh (band {hl:.3f}-{hh:.3f}), "
        f"C={cv:.3f} MWh (band {cl:.3f}-{ch:.3f})"
        for case, hv, hl, hh, cv, cl, ch in [
            ("600", 5.182, 4.314, 5.836, 2.546, 4.275, 5.784),
            ("800", 5.453, 4.378, 5.923, 2.007, 4.888, 6.612),
            ("810", 1.633, 3.357, 4.543, 0.910, 3.740, 5.060),
            ("900", 1.633, 1.364, 1.846, 0.910, 2.465, 3.335),
            ("920", 2.400, 3.213, 4.347, 1.085, 2.189, 2.961),
            ("950", 0.000, 0.000, 0.000, 0.028, 0.557, 0.753),
            ("960", 2.924, 1.742, 2.357, 0.144, 1.840, 2.490),
            ("970", 3.580, 10.540, 14.260, 1.654, 7.391, 9.999),
        ]
    )
    measured = checker.parse_measured(log)
    assert set(measured.keys()) == {"600", "800", "810", "900", "920", "950", "960", "970"}
    # Spot-check numeric extraction across the full 8-case set.
    assert measured["950"]["hlo"] == 0.000
    assert measured["950"]["hhi"] == 0.000
    assert measured["970"]["hlo"] == 10.540
    assert measured["970"]["chi"] == 9.999


def test_gap_handles_degenerate_band_in_band_value(checker):
    """Issue #3572: in-band value against degenerate band → gap 0 (not inf).

    Case 950 heating per ASHRAE 140-2023 §B8.5 publishes [0, 0] MWh. The
    pre-#3572 implementation returned inf for any value when the band
    midpoint collapsed to zero; the post-#3572 fix returns 0 for an
    in-band value regardless of midpoint.
    """
    assert checker.gap_pct_of_mid(0.0, 0.0, 0.0) == 0.0


def test_gap_keeps_inf_for_out_of_band_degenerate(checker):
    """Issue #3572: out-of-band value against degenerate band → gap inf.

    Regression guard for the degenerate-band fix: only IN-band values
    against a degenerate midpoint are accepted, OUT-of-band values
    still report infinity (no false-positive PASS).
    """
    assert checker.gap_pct_of_mid(0.5, 0.0, 0.0) == float("inf")
    assert checker.gap_pct_of_mid(-0.1, 0.0, 0.0) == float("inf")


def test_main_passes_when_all_eight_cases_match_baseline(checker, tmp_path):
    """Issue #3572: full eight-case log + matching baseline → exit 0."""
    baseline = _write_baseline_8_cases(tmp_path)
    log = _write_log_8_cases(tmp_path)
    assert _invoke(checker, log, baseline, require_cases=",".join(checker.SUPPORTED_CASES)) == 0


def test_main_returns_one_when_log_missing_one_of_eight_cases(checker, tmp_path, capsys):
    """Issue #3572 acceptance: missing any of the 8 required cases → exit 1.

    A drift in the cargo test filter / test name that drops one of the
    strict lines is itself a gate-coverage regression. Test by dropping
    Case 970 (the newest strict case).
    """
    baseline = _write_baseline_8_cases(tmp_path)
    log = _write_log_8_cases(tmp_path, drop_case="970")
    rc = _invoke(checker, log, baseline, require_cases=",".join(checker.SUPPORTED_CASES))
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1 when 970 missing, got {rc}\noutput:\n{out}"
    assert "970" in out
    assert "could not parse" in out


def test_main_returns_one_when_new_case_metric_regresses(checker, tmp_path, capsys):
    """Issue #3572: a newly-tracked metric (Case 800 cooling) regresses → exit 1.

    Push Case 800 cooling (currently 50.1 pp UNDER band, tracked as
    ``known_fail``) further out so the gap exceeds baseline + 5 pp
    tolerance. The gate must trip even though other seven cases are
    unchanged.
    """
    baseline = _write_baseline_8_cases(tmp_path)
    # band = [4.888, 6.612], midpoint = 5.75. Push value way down.
    log = _write_log_8_cases(
        tmp_path,
        overrides={"800": {"h": 5.453, "c": 0.500}},
    )
    rc = _invoke(checker, log, baseline, require_cases=",".join(checker.SUPPORTED_CASES))
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1 on Case 800 C regression, got {rc}\noutput:\n{out}"
    assert "REGRESSION" in out
    assert "case_800_cooling" in out


def test_main_returns_one_when_new_case_pass_metric_exits_band(checker, tmp_path, capsys):
    """Issue #3572: a newly-tracked PASS metric (Case 800 heating) exits band → exit 1.

    Case 800 heating is currently PASS (in-band, gap 0). Push it way
    above the band so the gap exceeds the 5 pp tolerance — the stricter
    rule for previously-PASS metrics trips the gate.
    """
    baseline = _write_baseline_8_cases(tmp_path)
    # band = [4.378, 5.923], midpoint = 5.1505. Push to 12.0 → gap huge.
    log = _write_log_8_cases(
        tmp_path,
        overrides={"800": {"h": 12.0, "c": 2.007}},
    )
    rc = _invoke(checker, log, baseline, require_cases=",".join(checker.SUPPORTED_CASES))
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1 on Case 800 H regression, got {rc}\noutput:\n{out}"
    assert "REGRESSION" in out
    assert "case_800_heating" in out


def test_self_test_passes_when_all_eight_cases_present(checker):
    """Issue #3572: --self-test exits 0 against a synthetic 8-case log."""
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--self-test"]
    try:
        rc = checker.self_test()
    finally:
        sys.argv[:] = saved
    assert rc == 0
