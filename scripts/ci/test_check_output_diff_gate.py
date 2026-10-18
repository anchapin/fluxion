"""
Tests for ``scripts/check_output_diff_gate.py`` -- Issue #4281.

Regression guard for the release-over-release output-diff gate. Mirrors
the ``load_script`` + ``tmp_path`` pattern: drive ``main()`` through the
re-record, pass, drift-beyond-epsilon, metric-set-change, from-zero and
unit-mismatch scenarios on synthetic strict-gate logs.

The synthetic logs follow the same grammar the strict checker parses:
``[#1147 Case 600 strict] H=... MWh (band ...), C=... MWh (band ...)`` and
``Case 600FF strict Tmin=... band [...], Tmax=... band [...]``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_output_diff_gate"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of the output-diff gate."""
    return load_script(SCRIPT_NAME)


ANNUAL_CASES = (
    "600", "800", "810", "900", "920", "950", "960", "970",
    "610", "620", "630", "640", "650", "910", "930", "940", "195",
)
FF_CASES = ("600FF", "650FF", "900FF", "950FF")


def synth_log(h_overrides: dict[str, float] | None = None,
              ff_overrides: dict[str, float] | None = None,
              drop: tuple[str, str] = ()) -> str:
    """Build a synthetic strict-gate log covering every required case."""
    h_overrides = h_overrides or {}
    ff_overrides = ff_overrides or {}
    lines: list[str] = []
    for case in ANNUAL_CASES:
        if (case, "all") in drop:
            continue
        h = h_overrides.get(case, 5.000)
        c = 2.000
        if (case, "H") in drop:
            lines.append(f"[#1147 Case {case} strict] C={c} MWh (band 1.5-2.5)")
            continue
        lines.append(
            f"[#1147 Case {case} strict] H={h} MWh (band 4.0-6.0), "
            f"C={c} MWh (band 1.5-2.5)"
        )
    for case in FF_CASES:
        tmin = ff_overrides.get(f"{case}_Tmin", 20.000)
        tmax = ff_overrides.get(f"{case}_Tmax", 40.000)
        if (case, "Tmin") in drop:
            lines.append(
                f"Case {case} strict Tmax={tmax} band [35.0 to 45.0]"
            )
            continue
        lines.append(
            f"Case {case} strict Tmin={tmin} band [15.0 to 25.0], "
            f"Tmax={tmax} band [35.0 to 45.0]"
        )
    return "\n".join(lines) + "\n"


def run_gate(gate, tmp_path: Path, log_text: str, extra_args: list[str]) -> int:
    log = tmp_path / "strict_gate_output.txt"
    log.write_text(log_text)
    baseline = tmp_path / "output_diff_baseline.csv"
    argv = [str(log), "--baseline", str(baseline)] + extra_args
    return gate.main(argv)


def test_update_baseline_then_pass(gate, tmp_path):
    """Re-record then rerun an identical log: PASS, provenance present."""
    log_text = synth_log()
    assert run_gate(gate, tmp_path, log_text, ["--update-baseline"]) == 0
    baseline = tmp_path / "output_diff_baseline.csv"
    text = baseline.read_text()
    assert text.startswith("# repo_sha=")
    assert "# recorded_utc=" in text
    assert "# source=strict-gate log" in text
    rows = [r for r in csv.reader(text.splitlines()) if r and not r[0].startswith("#")]
    # 17 annual cases x 2 metrics + 4 FF cases x 2 metrics = 42 rows + header
    assert len(rows) == 43
    assert rows[0] == ["case", "metric", "value", "unit"]
    assert run_gate(gate, tmp_path, log_text, []) == 0


def test_drift_beyond_epsilon_fails(gate, tmp_path):
    """A value moved >0.1% relative to baseline fails, worst first."""
    assert run_gate(gate, tmp_path, synth_log(), ["--update-baseline"]) == 0
    drifted = synth_log(h_overrides={"600": 5.010})  # +0.2% on one metric
    rc = run_gate(gate, tmp_path, drifted, [])
    assert rc == 1


def test_drift_within_epsilon_passes(gate, tmp_path):
    """A value moved within epsilon (0.05% here) stays green."""
    assert run_gate(gate, tmp_path, synth_log(), ["--update-baseline"]) == 0
    nudged = synth_log(h_overrides={"600": 5.0025})  # +0.05%
    assert run_gate(gate, tmp_path, nudged, []) == 0


def test_missing_metric_fails_closed(gate, tmp_path):
    """A dropped measurement line is a coverage regression, not drift."""
    assert run_gate(gate, tmp_path, synth_log(), ["--update-baseline"]) == 0
    rc = run_gate(gate, tmp_path, synth_log(drop=(("650", "H"),)), [])
    assert rc == 1


def test_from_zero_movement_fails(gate, tmp_path):
    """A metric that leaves an exact-zero baseline fails closed."""
    assert run_gate(gate, tmp_path, synth_log(), ["--update-baseline"]) == 0
    baseline = tmp_path / "output_diff_baseline.csv"
    text = baseline.read_text().replace("600,H,5.000000", "600,H,0.000000")
    baseline.write_text(text)
    rc = run_gate(gate, tmp_path, synth_log(), [])
    assert rc == 1


def test_unit_mismatch_fails(gate, tmp_path):
    """A unit disagreement between baseline and log fails closed."""
    assert run_gate(gate, tmp_path, synth_log(), ["--update-baseline"]) == 0
    baseline = tmp_path / "output_diff_baseline.csv"
    text = baseline.read_text().replace("600,H,5.000000,MWh", "600,H,5.000000,kWh")
    baseline.write_text(text)
    rc = run_gate(gate, tmp_path, synth_log(), [])
    assert rc == 1


def test_missing_baseline_fails_closed(gate, tmp_path, capsys):
    """No baseline file: explicit failure naming the fix, never a pass."""
    assert run_gate(gate, tmp_path, synth_log(), []) == 1
    assert "baseline not found" in capsys.readouterr().out


def test_incomplete_log_fails_closed(gate, tmp_path, capsys):
    """A log missing a required case is rejected before any diff."""
    rc = run_gate(gate, tmp_path, synth_log(drop=(("910", "all"),)), ["--update-baseline"])
    assert rc == 1
    assert "missing measurement lines" in capsys.readouterr().out
