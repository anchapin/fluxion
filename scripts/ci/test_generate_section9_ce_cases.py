"""Tests for scripts/generate_section9_ce_cases.py (Issue #4296).

Guards the generated ASHRAE 140 Section 9 CE100-CE200 case file: the
committed JSON must match the generator output byte-for-byte, and the
structural invariants the §9 harness will rely on must hold.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_section9_ce_cases.py"
OUTPUT = REPO_ROOT / "data" / "reference" / "ashrae140" / "section9_ce_cases.json"

spec = importlib.util.spec_from_file_location("gen_ce", SCRIPT)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def _doc() -> dict:
    return json.loads(OUTPUT.read_text())


def test_committed_output_matches_generator() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_case_family_and_count() -> None:
    doc = _doc()
    ids = [c["case_id"] for c in doc["cases"]]
    assert ids == [
        "CE100", "CE110", "CE120", "CE130", "CE140",
        "CE150", "CE160", "CE165", "CE170", "CE180",
        "CE185", "CE190", "CE195", "CE200",
    ]


def test_series_split_dry_wet() -> None:
    doc = _doc()
    dry = [c["case_id"] for c in doc["cases"] if c["series"] == "dry"]
    wet = [c["case_id"] for c in doc["cases"] if c["series"] == "wet"]
    assert dry == ["CE100", "CE110", "CE120", "CE130", "CE140"]
    assert wet == [
        "CE150", "CE160", "CE165", "CE170", "CE180", "CE185", "CE190", "CE195",
    ]


def test_no_invented_analytical_targets() -> None:
    """The #4285 licensed-edition gate: analytical slots stay null."""
    doc = _doc()
    for case in doc["cases"]:
        assert case["acceptance"]["analytical"] is None
        assert "4285" in case["acceptance"]["analytical_note"]


def test_comparative_bindings_match_bounds_csv() -> None:
    doc = _doc()
    expected = gen.read_bound_cases(
        REPO_ROOT
        / "tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv"
    )
    bound_ids = {c["case_id"] for c in doc["cases"] if c["acceptance"]["comparative"]}
    assert bound_ids == set(expected)


def test_ce200_ahri_conditions() -> None:
    doc = _doc()
    ce200 = next(c for c in doc["cases"] if c["case_id"] == "CE200")
    cond = ce200["conditions"]
    assert cond["internal_sensible_gains_W"] == 6120.0
    assert cond["internal_latent_gains_W"] == 1817.0
    assert cond["cooling_setpoint_C"] == 26.7
    assert cond["outdoor_dry_bulb_C"] == 35.0
    assert ce200["weather_file"] == "CE200A"


def test_engine_coverage_matches_the_engine_run() -> None:
    """The §9 engine-run harness (#4296) dispatches every CE case.

    The claim in the generated file must match the case table and point at
    the harness that produces it; the recorded per-case rows live in
    ``tests/validation/hvac_equipment_rated_performance/data/
    section9_engine_run_results.json``.
    """
    doc = _doc()
    coverage = doc["engine_coverage"]
    assert coverage["cases_run"] == len(doc["cases"]) == 14
    assert "section9_engine_run.rs" in coverage["note"]
    assert "section9_engine_run_results.json" in coverage["note"]


def test_analytical_slot_still_null_after_the_harness_landed() -> None:
    """The harness scores the comparative slot; the analytical slot stays null
    until the licensed 140-2023 §9 materials are transcribed (#4285)."""
    doc = _doc()
    for case in doc["cases"]:
        assert case["acceptance"]["analytical"] is None
