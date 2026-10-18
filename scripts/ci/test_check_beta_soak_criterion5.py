"""Unit tests for scripts/check_beta_soak_criterion5.py (ADR-0025 applied).

Run from scripts/: /tmp/ci-test-venv/bin/python -m pytest ci/test_check_beta_soak_criterion5.py -q --no-cov
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "check_beta_soak_criterion5.py"
_spec = importlib.util.spec_from_file_location("check_beta_soak_criterion5", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

TRACKED_TEST = mod.TRACKED_TEST

# Baseline fixture matching the committed baseline's shape (values from the
# round-64 record run; the tests only need self-consistent numbers).
BASELINE_ROWS = [
    ("610", "peak_heating", "4.90", "kW"),
    ("610", "peak_cooling", "3.10", "kW"),
    ("620", "annual_cooling", "1.50", "MWh"),
    ("620", "peak_cooling", "1.90", "kW"),
    ("630", "peak_heating", "3.80", "kW"),
    ("630", "peak_cooling", "3.60", "kW"),
    ("640", "annual_heating", "2.20", "MWh"),
    ("640", "annual_cooling", "3.30", "MWh"),
    ("640", "peak_heating", "3.50", "kW"),
    ("640", "peak_cooling", "4.20", "kW"),
    ("650", "annual_cooling", "2.10", "MWh"),
    ("650", "peak_cooling", "3.90", "kW"),
    ("600FF", "min_free_float", "-12.01", "degC"),
    ("650FF", "min_free_float", "-17.45", "degC"),
]

FAIL_LOG = """\
test {tracked} ... FAILED
Case610 peak_heating: 4.90 kW [ref 4.30-5.70] FAIL
Case610 peak_cooling: 3.10 kW [ref 2.20-2.90] FAIL
Case620 annual_cooling: 1.50 MWh [ref 3.20-5.00] FAIL
Case620 peak_cooling: 1.90 kW [ref 2.50-3.50] FAIL
Case630 peak_heating: 3.80 kW [ref 4.70-6.10] FAIL
Case630 peak_cooling: 3.60 kW [ref 1.80-2.40] FAIL
Case640 annual_heating: 2.20 MWh [ref 2.75-3.80] FAIL
Case640 annual_cooling: 3.30 MWh [ref 5.95-8.10] FAIL
Case640 peak_heating: 3.50 kW [ref 4.30-5.70] FAIL
Case640 peak_cooling: 4.20 kW [ref 2.80-3.70] FAIL
Case650 annual_cooling: 2.10 MWh [ref 4.82-7.06] FAIL
Case650 peak_cooling: 3.90 kW [ref 1.90-2.50] FAIL
Case600FF min_free_float: -12.01C [ref -18.80..-15.60] FAIL
Case650FF min_free_float: -17.45C [ref -23.00..-21.00] FAIL
""".format(tracked=TRACKED_TEST)


def _baseline_csv(tmp_path: Path, rows=BASELINE_ROWS) -> Path:
    lines = ["case,metric,measured_value,unit"]
    lines += [",".join(r) for r in rows]
    p = tmp_path / "criterion5_baseline.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _run(tmp_path: Path, log_text: str, rows=BASELINE_ROWS):
    baseline = _baseline_csv(tmp_path, rows)
    log = tmp_path / "c5.log"
    log.write_text(log_text, encoding="utf-8")
    return mod.main(
        ["--log", str(log), "--baseline", str(baseline), "--json"]
    )


def test_parse_value_lines():
    measured = mod.parse_measured_values(FAIL_LOG)
    assert measured[("610", "peak_heating")] == 4.90
    assert measured[("600FF", "min_free_float")] == -12.01
    assert len(measured) == 14


def test_green_when_values_match():
    assert _run(Path(_mktmp()), FAIL_LOG) == 0


def test_tracked_drift_beyond_epsilon():
    drifted = FAIL_LOG.replace("Case650 annual_cooling: 2.10", "Case650 annual_cooling: 4.50")
    tmp = Path(_mktmp())
    assert _run(tmp, drifted) == 0  # exit 0; verdict carries the drift
    baseline = _baseline_csv(tmp)
    log = tmp / "c5.log"
    log.write_text(drifted, encoding="utf-8")
    outcomes = mod.parse_log(drifted)
    measured = mod.parse_measured_values(drifted)
    values, _meta = mod.parse_baseline(baseline)
    verdict = mod.evaluate(outcomes, measured, values)
    assert verdict.verdict == "tracked-drift"


def test_within_epsilon_is_green():
    # 0.1% relative drift epsilon: 2.10 * 1.0009 is within, 2.2 is beyond.
    drifted = FAIL_LOG.replace("Case650 annual_cooling: 2.10", "Case650 annual_cooling: 2.101")
    assert _run(Path(_mktmp()), drifted) == 0
    outcomes = mod.parse_log(drifted)
    measured = mod.parse_measured_values(drifted)
    values, _meta = mod.parse_baseline(_baseline_csv(Path(_mktmp())))
    verdict = mod.evaluate(outcomes, measured, values)
    assert verdict.verdict == "green"


def test_re_pin_required_when_test_passes():
    log = FAIL_LOG.replace(" ... FAILED", " ... ok")
    tmp = Path(_mktmp())
    assert _run(tmp, log) == 0
    outcomes = mod.parse_log(log)
    values, _meta = mod.parse_baseline(_baseline_csv(Path(_mktmp())))
    verdict = mod.evaluate(outcomes, mod.parse_measured_values(log), values)
    assert verdict.verdict == "re-pin-required"
    assert verdict.re_pins == [TRACKED_TEST]


def test_hard_fail_when_test_becomes_ignored():
    log = FAIL_LOG.replace(" ... FAILED", " ... ignored, physics gap")
    tmp = Path(_mktmp())
    assert _run(tmp, log) == 1
    outcomes = mod.parse_log(log)
    values, _meta = mod.parse_baseline(_baseline_csv(Path(_mktmp())))
    verdict = mod.evaluate(outcomes, mod.parse_measured_values(log), values)
    assert verdict.verdict == "hard-fail"


def test_set_drift_on_extra_test():
    log = FAIL_LOG + "test known_issues_regression::solar_issues::test_x ... ok\n"
    tmp = Path(_mktmp())
    assert _run(tmp, log) == 1
    outcomes = mod.parse_log(log)
    values, _meta = mod.parse_baseline(_baseline_csv(Path(_mktmp())))
    verdict = mod.evaluate(outcomes, mod.parse_measured_values(log), values)
    assert verdict.verdict == "set-drift"


def test_set_drift_on_missing_test():
    log = "nothing here\n"
    tmp = Path(_mktmp())
    assert _run(tmp, log) == 1


def test_missing_value_line_is_tracked_drift():
    log = f"test {TRACKED_TEST} ... FAILED\n"
    tmp = Path(_mktmp())
    assert _run(tmp, log) == 0
    outcomes = mod.parse_log(log)
    values, _meta = mod.parse_baseline(_baseline_csv(Path(_mktmp())))
    verdict = mod.evaluate(outcomes, mod.parse_measured_values(log), values)
    assert verdict.verdict == "tracked-drift"
    assert all("no measured value line" in d["reason"] for d in verdict.drift)
    assert len(verdict.drift) == 14


def test_committed_baseline_parses():
    baseline = (
        REPO_ROOT
        / "tests"
        / "reference_data"
        / "gauge_soak"
        / "criterion5_tracked_cohort_baseline.csv"
    )
    values, meta = mod.parse_baseline(baseline)
    assert len(values) == 14
    # The 12 energy metrics + 2 free-float min-temps from the test body.
    assert {k[1] for k in values} == {
        "peak_heating",
        "peak_cooling",
        "annual_cooling",
        "annual_heating",
        "min_free_float",
    }
    assert meta[("600FF", "min_free_float")]["unit"] == "degC"


def _mktmp() -> str:
    import tempfile

    return tempfile.mkdtemp()
