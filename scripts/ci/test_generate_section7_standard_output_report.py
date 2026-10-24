"""
Tests for ``scripts/generate_section7_standard_output_report.py`` -- the
ASHRAE 140 Section 7 Standard Output Report generator (Issue #4276, Section 7
slice).

The generator replaces the invented pass-rate scorecard shape with the
artifacts the standard actually asks for:

* a FLAT results CSV (one row per Section 7 case, one column per required
  reported value fluxion produced), and
* a CONFORMANCE TABLE (per case: ``conformant`` / ``incomplete`` /
  ``not implemented``), with the Annex B8 example-result spread carried as
  DIAGNOSTIC context and explicitly labelled "not a criterion".

The load-bearing properties pinned here:

* the flat CSV has exactly one row per Section 7 case and carries only
  values fluxion produced (never a fabricated substitute);
* a case whose required output is missing is classified ``incomplete`` and
  its cell is EMPTY -- the classification never leaks into a value;
* no aggregate score / grade / verdict appears anywhere in the outputs;
* the Annex B8 spread is labelled diagnostic, not a criterion;
* the generator is deterministic and its ``--check`` mode detects drift;
* the generated conformance states agree with the committed Rust conformance
  artifact for the implemented cases (no second, disagreeing truth).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "generate_section7_standard_output_report"


@pytest.fixture
def gen(load_script):
    """Freshly-loaded copy of the Section 7 report generator."""
    return load_script(SCRIPT_NAME)


@pytest.fixture(scope="module")
def repo_data() -> Path:
    """The repository root (for the committed inputs and artifacts)."""
    return Path(__file__).resolve().parents[2]


def _run_main(gen, *argv):
    """Invoke ``gen.main()`` with a synthetic argv; return the exit code."""
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, *argv]
    try:
        rc = gen.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved
    return rc


def _read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def _generate(gen, repo_data: Path):
    """Generate the three artifacts from the committed inputs."""
    return gen.generate(
        repo_data / "data" / "reference" / "section7" / "workflow_results.csv",
        repo_data / "data" / "reference" / "section7" / "section621_outputs.csv",
        repo_data
        / "data"
        / "reference"
        / "ashrae140"
        / "annex_b8_section7_statistics.json",
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_case_catalog_covers_the_standard_case_set(gen):
    """Every Section 7 case of the standard is in the catalog, plus 970."""
    catalog = [case_id for case_id, _title in gen.SECTION7_CASES]
    assert "195" in catalog
    assert "995" in catalog
    assert "600FF" in catalog
    assert "980FF" in catalog
    assert len(catalog) == len(set(catalog))
    # fluxion's extra case is flagged as not defined in 140-2017.
    extras = [case_id for case_id, _title in gen.FLUXION_EXTRA_CASES]
    assert extras == ["970"]
    assert "970" not in catalog


def test_implemented_case_set_matches_the_emitter(gen):
    """The implemented set is exactly the emitter's pinned case set."""
    assert set(gen.IMPLEMENTED_CASES) == {
        "600",
        "610",
        "620",
        "630",
        "640",
        "650",
        "900",
        "910",
        "920",
        "930",
        "940",
        "950",
        "960",
        "970",
        "600FF",
        "650FF",
        "900FF",
        "950FF",
    }


# ---------------------------------------------------------------------------
# Flat results CSV
# ---------------------------------------------------------------------------


def test_flat_csv_has_one_row_per_case(gen, repo_data):
    csv_text, _md, _json = _generate(gen, repo_data)
    rows = _read_csv(csv_text)
    expected = [case_id for case_id, _ in gen.SECTION7_CASES] + [
        case_id for case_id, _ in gen.FLUXION_EXTRA_CASES
    ]
    assert [row["case_id"] for row in rows] == expected


def test_flat_csv_columns_are_one_per_reported_value(gen, repo_data):
    csv_text, _md, _json = _generate(gen, repo_data)
    header = next(csv.reader(csv_text.splitlines()))
    assert header[:3] == ["case_id", "case_title", "case_status"]
    # Every requirement id appears in the column names, except 6.2.1.1.5:
    # it is a sign convention over the 6.2.1.1.1-6.2.1.1.4 loads ("entered
    # as positive (>= 0) values"), not a reported value, so it has no
    # column of its own; its conformance cell is derived from the signs
    # of those columns (see test_requirement_cells_are_four_state and the
    # conformance artifact consistency test).
    for requirement in gen.REQUIREMENT_IDS:
        if requirement == "6.2.1.1.5":
            continue
        assert any(column.startswith(requirement) for column in header), requirement
    # 12 monthly columns per monthly series, 90 temperature bins.
    monthly = [c for c in header if c.startswith("6.2.1.2.4_monthly_heating_kwh_")]
    assert len(monthly) == 12
    bins = [c for c in header if c.startswith("6.2.1.7_zone_air_temp_bin_hours_c_")]
    assert len(bins) == 90


def test_flat_csv_carries_only_produced_values(gen, repo_data):
    """A case that did not run carries no values at all."""
    csv_text, _md, _json = _generate(gen, repo_data)
    rows = _read_csv(csv_text)
    value_columns = [c for c in rows[0] if c not in ("case_id", "case_title", "case_status")]
    for row in rows:
        if row["case_status"] == "not implemented":
            assert all(row[column] == "" for column in value_columns), row["case_id"]


def test_flat_csv_monthly_loads_are_converted_to_kwh(gen, repo_data):
    """Monthly loads are reported in kWh (source CSV carries MWh)."""
    results = gen.load_results(
        repo_data / "data" / "reference" / "section7" / "workflow_results.csv"
    )
    csv_text, _md, _json = _generate(gen, repo_data)
    rows = _read_csv(csv_text)
    row = next(r for r in rows if r["case_id"] == "600")
    source = results["600"]["monthly_htg_cons"].split(",")
    for index, raw in enumerate(source):
        expected = float(raw) * gen.MONTHLY_MWH_TO_KWH
        actual = float(row[f"6.2.1.2.4_monthly_heating_kwh_{index + 1:02d}_jan"]
                       if index == 0 else
                       row[[c for c in row if c.startswith("6.2.1.2.4_monthly_heating_kwh_")][index]])
        assert actual == pytest.approx(expected, rel=1e-6)


def test_flat_csv_peak_occurrence_is_empty_when_peak_is_zero(gen, repo_data):
    """A zero peak has no occurrence -- the emitter's sentinel is dropped."""
    csv_text, _md, _json = _generate(gen, repo_data)
    rows = _read_csv(csv_text)
    free_float = next(r for r in rows if r["case_id"] == "900FF")
    assert free_float["6.2.1.1.3_peak_heating_kw"] == "0.000"
    assert free_float["6.2.1.1.3_peak_heating_time"] == ""
    assert free_float["6.2.1.1.4_peak_cooling_time"] == ""


# ---------------------------------------------------------------------------
# Conformance table
# ---------------------------------------------------------------------------


def test_case_states_are_the_three_allowed_states(gen, repo_data):
    _csv, md_text, json_text = _generate(gen, repo_data)
    payload = json.loads(json_text)
    allowed = {"conformant", "incomplete", "not implemented"}
    for case in payload["cases"]:
        assert case["status"] in allowed, case["case"]
    assert "not implemented" in md_text


def test_requirement_cells_are_the_four_allowed_states(gen, repo_data):
    _csv, _md, json_text = _generate(gen, repo_data)
    payload = json.loads(json_text)
    allowed = {"report", "partial", "missing", "na"}
    for case in payload["cases"]:
        for requirement, state in case["requirements"].items():
            assert state in allowed, (case["case"], requirement, state)


def test_missing_required_output_is_incomplete_not_a_fabricated_value(
    gen, repo_data, tmp_path
):
    """Drop a required output -> the case is ``incomplete`` and the cell empty."""
    source = repo_data / "data" / "reference" / "section7" / "workflow_results.csv"
    lines = source.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    rows = [line for line in lines[1:] if line.strip()]

    # Blank the Case 610 companion value (Section 6.2.1.3) and confirm the
    # case drops to ``incomplete`` with an empty report cell.
    companion_source = repo_data / "data" / "reference" / "section7" / "section621_outputs.csv"
    companion_lines = companion_source.read_text(encoding="utf-8").splitlines()
    broken_companion = tmp_path / "companion.csv"
    out = [companion_lines[0]]
    for line in companion_lines[1:]:
        if line.startswith("610,"):
            fields = line.split(",")
            fields[1] = ""
            line = ",".join(fields)
        out.append(line)
    broken_companion.write_text("\n".join(out) + "\n", encoding="utf-8")

    csv_text, _md, json_text = gen.generate(
        source,
        broken_companion,
        repo_data
        / "data"
        / "reference"
        / "ashrae140"
        / "annex_b8_section7_statistics.json",
    )
    payload = json.loads(json_text)
    case = next(c for c in payload["cases"] if c["case"] == "610")
    assert case["status"] == "incomplete"
    assert case["requirements"]["6.2.1.3"] == "missing"
    row = next(r for r in _read_csv(csv_text) if r["case_id"] == "610")
    assert row["6.2.1.3_transmitted_solar_shaded_south_kwh_m2"] == ""


def test_generated_states_agree_with_the_committed_rust_artifact(gen, repo_data):
    """The generator must not create a second, disagreeing truth."""
    _csv, _md, json_text = _generate(gen, repo_data)
    payload = json.loads(json_text)
    generated = {case["case"]: case for case in payload["cases"]}

    committed_path = repo_data / "data" / "reference" / "section7" / "section7_conformance.csv"
    committed_rows = list(csv.reader(committed_path.read_text(encoding="utf-8").splitlines()))
    header = committed_rows[0]
    requirements = header[2:]
    for row in committed_rows[1:]:
        record = dict(zip(header, row))
        case = generated[record["case"]]
        assert case["status"] == record["status"], record["case"]
        for requirement in requirements:
            assert case["requirements"][requirement] == record[requirement], (
                record["case"],
                requirement,
            )


def test_annex_b8_is_labelled_diagnostic_not_a_criterion(gen, repo_data):
    _csv, md_text, json_text = _generate(gen, repo_data)
    assert "not a criterion" in md_text.lower()
    assert "diagnostic" in md_text.lower()
    payload = json.loads(json_text)
    assert "not a criterion" in payload["annex_b8_role"].lower()
    assert "diagnostic" in payload["annex_b8_role"].lower()
    # The Annex B8 spread is carried next to the values, per case.
    case = next(c for c in payload["cases"] if c["case"] == "600")
    spread = case["annex_b8_diagnostic"]["annual_heating_mwh"]
    assert spread["min"] is not None and spread["max"] is not None


# ---------------------------------------------------------------------------
# No score anywhere
# ---------------------------------------------------------------------------


def test_no_score_like_quantity_in_the_outputs(gen, repo_data):
    csv_text, md_text, json_text = _generate(gen, repo_data)
    forbidden = (
        "pass rate",
        "pass_rate",
        "pass-rate",
        "compliance score",
        "compliance_score",
        "%",
        "score",
    )
    for text in (csv_text, md_text, json_text):
        lowered = text.lower()
        for token in forbidden:
            assert token not in lowered, token


# ---------------------------------------------------------------------------
# Determinism and the --check mode
# ---------------------------------------------------------------------------


def test_generation_is_deterministic(gen, repo_data):
    first = _generate(gen, repo_data)
    second = _generate(gen, repo_data)
    assert first == second


def test_check_mode_detects_drift_and_passes_when_fresh(gen, repo_data, tmp_path):
    csv_out = tmp_path / "report.csv"
    md_out = tmp_path / "conformance.md"
    json_out = tmp_path / "conformance.json"
    args = [
        "--output-csv",
        str(csv_out),
        "--output-md",
        str(md_out),
        "--output-json",
        str(json_out),
    ]
    assert _run_main(gen, *args) == 0
    assert csv_out.exists() and md_out.exists() and json_out.exists()
    assert _run_main(gen, "--check", *args) == 0

    md_out.write_text(md_out.read_text(encoding="utf-8") + "\ndrift\n", encoding="utf-8")
    assert _run_main(gen, "--check", *args) == 1


def test_missing_input_is_a_hard_error(gen, repo_data, tmp_path):
    rc = _run_main(
        gen,
        "--results-csv",
        str(tmp_path / "does-not-exist.csv"),
        "--output-csv",
        str(tmp_path / "out.csv"),
        "--output-md",
        str(tmp_path / "out.md"),
        "--output-json",
        str(tmp_path / "out.json"),
    )
    assert rc == 2
