"""Tests for scripts/generate_section9b_cases.py (Issue #4273).

Guards the generated ASHRAE 140 Section 9b comparative case table: the
committed JSON must match the generator output byte-for-byte, every band must
trace back to the transcribed Task-22 bounds CSV, every §9a case must appear
exactly once, and the "not a pass/fail criterion" labelling must be present.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_section9b_cases.py"
OUTPUT = (
    REPO_ROOT / "data" / "reference" / "ashrae140" / "section9b_ce_comparative_cases.json"
)
CE_CASE_TABLE = REPO_ROOT / "data" / "reference" / "ashrae140" / "section9_ce_cases.json"
BOUNDS_CSV = (
    REPO_ROOT
    / "tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv"
)

spec = importlib.util.spec_from_file_location("gen_ce9b", SCRIPT)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def _doc() -> dict:
    return json.loads(OUTPUT.read_text())


def _csv_rows() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for row in gen.read_bound_rows(BOUNDS_CSV):
        rows[(row["case_id"], row["metric"])] = row
    return rows


def test_committed_output_matches_generator() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_generator_is_idempotent() -> None:
    """Two consecutive runs produce byte-identical output."""
    first = gen.build()
    second = gen.build()
    assert json.dumps(first, indent=2, ensure_ascii=False) == json.dumps(
        second, indent=2, ensure_ascii=False
    )
    text = json.dumps(first, indent=2, ensure_ascii=False) + "\n"
    assert text == OUTPUT.read_text()


def test_every_case_appears_exactly_once() -> None:
    doc = _doc()
    ce = json.loads(CE_CASE_TABLE.read_text())
    ids = [c["case_id"] for c in doc["cases"]]
    assert ids == [c["case_id"] for c in ce["cases"]]
    assert len(ids) == len(set(ids)) == 14


def test_every_band_traces_to_the_bounds_csv() -> None:
    """Each band's recorded endpoints and unit come from the CSV, verbatim."""
    doc = _doc()
    csv_rows = _csv_rows()
    seen = 0
    for case in doc["cases"]:
        for band in case["bands"]:
            key = (case["case_id"], band["metric"])
            assert key in csv_rows, f"band not in the CSV: {key}"
            row = csv_rows[key]
            assert band["recorded_low"] == row["low"]
            assert band["recorded_high"] == row["high"]
            assert band["recorded_unit"] == row["unit"]
            assert band["provenance"]["source_table"] == row["source_table"]
            assert band["provenance"]["source_page"] == row["source_page"]
            assert band["provenance"]["status"] == row["status"]
            assert band["provenance"]["source_program"] == row["source_program"]
            seen += 1
    assert seen == len(csv_rows)


def test_si_normalisation_matches_the_loader() -> None:
    """SI endpoints use the loader's documented conversions."""
    doc = _doc()
    for case in doc["cases"]:
        for band in case["bands"]:
            expected_low, unit_si = gen.to_si(
                band["recorded_low"], band["recorded_unit"], band["metric"]
            )
            expected_high, _ = gen.to_si(
                band["recorded_high"], band["recorded_unit"], band["metric"]
            )
            assert band["band_low_si"] == expected_low
            assert band["band_high_si"] == expected_high
            assert band["unit_si"] == unit_si
    # Spot-check the documented factors on one row of each dimension.
    annual = next(
        b
        for c in _doc()["cases"]
        for b in c["bands"]
        if b["metric"] == "annual_cooling"
    )
    assert annual["unit_si"] == "J"
    assert annual["band_low_si"] == annual["recorded_low"] * 1.0e6 * 3600.0
    peak = next(
        b for c in _doc()["cases"] for b in c["bands"] if b["metric"] == "peak_cooling"
    )
    assert peak["unit_si"] == "W"
    assert peak["band_low_si"] == peak["recorded_low"] * 1.0e3
    temp = next(
        b
        for c in _doc()["cases"]
        for b in c["bands"]
        if b["metric"] == "max_zone_temp"
    )
    assert temp["unit_si"] == "K"
    assert temp["band_low_si"] == temp["recorded_low"] + 273.15


def test_bands_carry_source_and_publication_citation() -> None:
    doc = _doc()
    csv_source = gen.read_csv_header_source(BOUNDS_CSV)
    publication = doc["_sources"]["publication"]
    assert publication["publication_key"] == "iea_task22_e100"
    assert "HVAC BESTEST, Volume 1" in publication["citation"]
    assert "10.2172/1244668" in csv_source
    for case in doc["cases"]:
        for band in case["bands"]:
            assert "comparative_bounds_ce100_ce200.csv" in band["band_source"]
            assert csv_source in band["band_source"]


def test_band_kind_label_is_present_and_exact() -> None:
    """The band kind says, verbatim, that this is not a pass/fail criterion."""
    doc = _doc()
    expected = "comparative evidence range, not a pass/fail criterion"
    assert doc["_band_kind"] == expected
    for case in doc["cases"]:
        for band in case["bands"]:
            assert band["band_kind"] == expected


def test_missing_bands_are_explicit_notes_naming_the_source() -> None:
    """No silent null: a case without a band names the missing source."""
    doc = _doc()
    csv_source = (
        "tests/validation/hvac_equipment_rated_performance/data/"
        "comparative_bounds_ce100_ce200.csv"
    )
    banded = [c for c in doc["cases"] if c["bands"]]
    unbanded = [c for c in doc["cases"] if not c["bands"]]
    assert len(banded) == 9
    assert len(unbanded) == 5
    assert [c["case_id"] for c in unbanded] == [
        "CE165",
        "CE180",
        "CE185",
        "CE190",
        "CE195",
    ]
    for case in unbanded:
        assert case["comparative_note"] is not None
        assert csv_source in case["comparative_note"]
        assert "No band is invented" in case["comparative_note"]
        # Every metric of the transcribed vocabulary is called out, too.
        assert {m["metric"] for m in case["missing_metric_notes"]} == {
            "annual_cooling",
            "peak_cooling",
            "max_zone_temp",
        }
        for note in case["missing_metric_notes"]:
            assert csv_source in note["comparative_note"]


def test_no_pass_fail_verdict_language_in_the_table() -> None:
    """The table must not claim a criterion. Only the explicit disclaimer
    sentence may contain the word 'pass'."""
    text = OUTPUT.read_text()
    for line in text.splitlines():
        if "pass" in line.lower():
            assert (
                "not a pass/fail criterion" in line
                or "no pass/fail criterion" in line
            ), line


def test_series_is_labelled_unimplemented() -> None:
    """The §9.3 comparative test series is not implemented in this tree."""
    doc = _doc()
    assert "not implemented" in doc["_scope"]
    assert "no pass/fail criterion" in doc["_scope"]
    for case in doc["cases"]:
        assert "not implemented" in case["comparative_series_status"]
