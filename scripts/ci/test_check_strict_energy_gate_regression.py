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


def _invoke(checker, log_path: Path, baseline_path: Path) -> int:
    """Invoke ``checker.main()`` with synthetic argv.

    The script is a CLI; rather than spawn a subprocess we patch
    ``sys.argv`` and call ``main()`` in-process. ``monkeypatch`` is
    expected to undo the change after the test exits.
    """
    saved = sys.argv[:]
    sys.argv[:] = [
        SCRIPT_NAME,
        str(log_path),
        "--baseline",
        str(baseline_path),
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
