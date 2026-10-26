"""
Tests for ``scripts/generate_section9_standard_output_report.py`` -- the
ASHRAE 140 Section 9 Standard Output Report generator (Issue #4276, Section 9
slice).

The generator is the Section 9 analogue of the Section 7 report generator
(``scripts/generate_section7_standard_output_report.py``): it replaces an
invented pass-rate shape with the artifacts the standard asks for --

* a FLAT results CSV (one row per Section 9 CE case, one column per
  reported value fluxion produced), and
* a CONFORMANCE TABLE (per case: ``conformant`` / ``incomplete`` /
  ``not implemented``), with the performance-map rating tables and the
  Task-22 ensemble ranges carried as analytical reference / diagnostic
  context and explicitly labelled "not a criterion".

The load-bearing properties pinned here:

* the flat CSV has exactly one row per Section 9 CE case and carries only
  values fluxion produced (never a fabricated substitute);
* a case that did not run is classified ``not implemented`` and every one of
  its cells is EMPTY;
* a missing required output is classified ``incomplete`` and its cell is
  EMPTY -- the classification never leaks into a value;
* no aggregate score / rate / verdict appears anywhere in the outputs, and
  the engine-run file's own band verdicts are not carried over;
* the reported values agree with the committed engine-run results and the
  reference values agree with the committed performance map and case table;
* the generator is deterministic and its ``--check`` mode detects drift.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "generate_section9_standard_output_report"

CASES_REL = Path("data") / "reference" / "ashrae140" / "section9_ce_cases.json"
MAP_REL = Path("data") / "reference" / "ashrae140" / "section9_ce100_performance_map.json"
RESULTS_REL = (
    Path("tests")
    / "validation"
    / "hvac_equipment_rated_performance"
    / "data"
    / "section9_engine_run_results.json"
)


@pytest.fixture
def gen(load_script):
    """Freshly-loaded copy of the Section 9 report generator."""
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
    return gen.generate(repo_data / CASES_REL, repo_data / MAP_REL, repo_data / RESULTS_REL)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _drop_run_field(repo_data: Path, tmp_path: Path, case_id: str, field: str, only_one=False):
    """A copy of the engine-run results with ``field`` removed from a case.

    ``only_one`` removes the field from a single run (the first one) instead
    of from every run of the case.
    """
    payload = json.loads((repo_data / RESULTS_REL).read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["case_id"] != case_id:
            continue
        if only_one:
            if case["runs"]:
                case["runs"][0].pop(field, None)
        else:
            for run in case["runs"]:
                run.pop(field, None)
    return _write_json(tmp_path / f"{case_id}_{field}.json", payload)


def _drop_case(repo_data: Path, tmp_path: Path, case_id: str):
    """A copy of the engine-run results with one case removed entirely."""
    payload = json.loads((repo_data / RESULTS_REL).read_text(encoding="utf-8"))
    payload["cases"] = [c for c in payload["cases"] if c["case_id"] != case_id]
    return _write_json(tmp_path / f"without_{case_id}.json", payload)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_case_catalog_is_the_full_ce_family(gen, repo_data):
    """Every CE case of the committed case table is reported, exactly once."""
    case_table = json.loads((repo_data / CASES_REL).read_text(encoding="utf-8"))
    expected = sorted(case["case_id"] for case in case_table["cases"])
    assert expected == [f"CE{n}" for n in (100, 110, 120, 130, 140, 150, 160, 165,
                                           170, 180, 185, 190, 195, 200)]
    _csv, _md, json_text = _generate(gen, repo_data)
    payload = json.loads(json_text)
    assert [case["case"] for case in payload["cases"]] == expected


def test_requirement_catalog_is_derived_and_labelled(gen):
    """The requirement ids are this report's own labels, not clause numbers."""
    assert gen.REQUIREMENT_IDS
    assert len(set(gen.REQUIREMENT_IDS)) == len(gen.REQUIREMENT_IDS)
    for requirement in gen.REQUIREMENTS:
        assert requirement.id.startswith("9a.")
        assert requirement.title
        assert requirement.basis
    # The derivation is stated once and reused by both artifacts.
    assert "DERIVED" in gen.DERIVATION
    assert "section9_ce_cases.json" in gen.DERIVATION
    assert "section9_ce100_performance_map.json" in gen.DERIVATION


# ---------------------------------------------------------------------------
# Flat results CSV
# ---------------------------------------------------------------------------


def test_flat_csv_has_one_row_per_case(gen, repo_data):
    csv_text, _md, _json = _generate(gen, repo_data)
    rows = _read_csv(csv_text)
    case_table = json.loads((repo_data / CASES_REL).read_text(encoding="utf-8"))
    expected = sorted(case["case_id"] for case in case_table["cases"])
    assert [row["case_id"] for row in rows] == expected


def test_flat_csv_columns_are_one_per_reported_value(gen, repo_data):
    csv_text, _md, _json = _generate(gen, repo_data)
    header = next(csv.reader(csv_text.splitlines()))
    assert header[:3] == ["case_id", "case_title", "case_status"]
    # Every derived requirement contributes at least one column.
    for requirement in gen.REQUIREMENTS:
        assert any(column.startswith(requirement.id) for column in header), requirement.id
    # The per-entering-condition outputs have one column per entering
    # wet-bulb column the performance map tabulates.
    performance_map = json.loads((repo_data / MAP_REL).read_text(encoding="utf-8"))
    columns = gen.map_ewb_columns(performance_map, gen.DEFAULT_MAP_TABLE)
    assert columns == [15.0, 17.2, 19.4, 21.7]
    for column in header:
        if "_ewb_" in column:
            assert column.rsplit("_ewb_", 1)[1] in {"15", "17.2", "19.4", "21.7"}


def test_flat_csv_values_agree_with_the_committed_engine_run_results(gen, repo_data):
    """Every reported value is the committed engine-run value (W -> kW)."""
    csv_text, _md, _json = _generate(gen, repo_data)
    rows = {row["case_id"]: row for row in _read_csv(csv_text)}
    results = json.loads((repo_data / RESULTS_REL).read_text(encoding="utf-8"))
    for case in results["cases"]:
        row = rows[case["case_id"]]
        for requirement in gen.PER_EWB_REQUIREMENTS:
            token = gen.requirement_token(requirement)
            name = f"{requirement.id}_{token}" if token else requirement.id
            for run in case["runs"]:
                ewb = gen._ewb_token(float(run["entering_wet_bulb_C"]))
                column = f"{name}_ewb_{ewb}"
                expected = float(run[requirement.field_name]) / requirement.scale
                # The CSV reports six decimal places.
                assert float(row[column]) == pytest.approx(expected, abs=1e-6), (
                    case["case_id"],
                    column,
                )


def test_flat_csv_carries_only_produced_values(gen, repo_data, tmp_path):
    """A case that did not run carries no values at all."""
    broken = _drop_case(repo_data, tmp_path, "CE165")
    csv_text, _md, json_text = gen.generate(
        repo_data / CASES_REL, repo_data / MAP_REL, broken
    )
    rows = _read_csv(csv_text)
    value_columns = [c for c in rows[0] if c not in ("case_id", "case_title", "case_status")]
    for row in rows:
        if row["case_status"] == "not implemented":
            assert all(row[column] == "" for column in value_columns), row["case_id"]


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


def test_not_implemented_case_is_classified_and_carries_no_values(gen, repo_data, tmp_path):
    """A case with no engine-run result is 'not implemented' and empty."""
    broken = _drop_case(repo_data, tmp_path, "CE165")
    csv_text, md_text, json_text = gen.generate(
        repo_data / CASES_REL, repo_data / MAP_REL, broken
    )
    payload = json.loads(json_text)
    case = next(c for c in payload["cases"] if c["case"] == "CE165")
    assert case["status"] == "not implemented"
    assert case["ran"] is False
    for requirement, state in case["requirements"].items():
        # 'missing' for every applicable requirement, 'na' for the ones
        # that do not apply to this case.
        assert state in {"missing", "na"}, requirement
    row = next(r for r in _read_csv(csv_text) if r["case_id"] == "CE165")
    value_columns = [c for c in row if c not in ("case_id", "case_title", "case_status")]
    assert all(row[column] == "" for column in value_columns)
    assert "CE165" in md_text


def test_missing_required_output_is_incomplete_not_a_fabricated_value(
    gen, repo_data, tmp_path
):
    """Drop a required output -> the case is 'incomplete' and the cell empty."""
    broken = _drop_run_field(repo_data, tmp_path, "CE100", "cop")
    csv_text, _md, json_text = gen.generate(
        repo_data / CASES_REL, repo_data / MAP_REL, broken
    )
    payload = json.loads(json_text)
    case = next(c for c in payload["cases"] if c["case"] == "CE100")
    assert case["status"] == "incomplete"
    assert case["requirements"]["9a.cop"] == "missing"
    row = next(r for r in _read_csv(csv_text) if r["case_id"] == "CE100")
    for column in row:
        if column.startswith("9a.cop"):
            assert row[column] == ""


def test_output_produced_for_only_some_runs_is_partial(gen, repo_data, tmp_path):
    """A value missing from one run of the envelope is 'partial'."""
    broken = _drop_run_field(repo_data, tmp_path, "CE100", "cop", only_one=True)
    _csv, _md, json_text = gen.generate(
        repo_data / CASES_REL, repo_data / MAP_REL, broken
    )
    payload = json.loads(json_text)
    case = next(c for c in payload["cases"] if c["case"] == "CE100")
    assert case["requirements"]["9a.cop"] == "partial"
    assert case["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Reference values: analytical reference / diagnostic, never a criterion
# ---------------------------------------------------------------------------


def test_reference_values_are_labelled_not_a_criterion(gen, repo_data):
    _csv, md_text, json_text = _generate(gen, repo_data)
    assert "not a criterion" in md_text.lower()
    assert "analytical reference" in md_text.lower()
    payload = json.loads(json_text)
    for role in payload["reference_roles"].values():
        assert "not a criterion" in role.lower()


def test_reference_values_agree_with_the_committed_performance_map(gen, repo_data):
    """The analytical reference is the map's own rating at the case condition."""
    _csv, md_text, json_text = _generate(gen, repo_data)
    payload = json.loads(json_text)
    performance_map = json.loads((repo_data / MAP_REL).read_text(encoding="utf-8"))
    table = performance_map["tables"][gen.DEFAULT_MAP_TABLE]
    for case in payload["cases"]:
        reference = case["reference_diagnostic"]["map"]
        odb = str(reference["outdoor_dry_bulb_C"])
        edb = str(reference["entering_dry_bulb_C"])
        for point in reference["points"]:
            ewb = str(point["entering_wet_bulb_C"])
            source = table[odb][ewb]
            assert point["total_capacity_kW"] == pytest.approx(
                source["total_capacity_kW"]
            )
            assert point["sensible_capacity_kW"] == pytest.approx(
                source["sensible_capacity_kW_by_edb"][edb]
            )
            assert point["compressor_power_kW"] == pytest.approx(
                source["compressor_power_kW"]
            )


def test_comparative_ranges_come_from_the_committed_case_table(gen, repo_data):
    """The diagnostic ranges are carried verbatim from the case table."""
    _csv, _md, json_text = _generate(gen, repo_data)
    payload = json.loads(json_text)
    case_table = json.loads((repo_data / CASES_REL).read_text(encoding="utf-8"))
    bounds = {
        case["case_id"]: ((case.get("acceptance") or {}).get("comparative") or {}).get(
            "bounds"
        )
        for case in case_table["cases"]
    }
    for case in payload["cases"]:
        comparative = case["reference_diagnostic"]["comparative"]
        if not bounds[case["case"]]:
            assert comparative == {}
            continue
        for metric, value in bounds[case["case"]].items():
            assert comparative["bounds"][metric] == value
            assert comparative["source"]
            assert "not a pass/fail criterion" in comparative["role"]


def test_derivation_of_the_required_output_list_is_stated_in_the_header(
    gen, repo_data
):
    _csv, md_text, json_text = _generate(gen, repo_data)
    assert "How the required-output list was derived" in md_text
    assert "DERIVED" in md_text
    assert "section9_ce_cases.json" in md_text
    assert "section9_ce100_performance_map.json" in md_text
    payload = json.loads(json_text)
    assert "DERIVED" in payload["derivation"]


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
        "compliance rate",
        "compliance_rate",
        "%",
        "score",
    )
    for text in (csv_text, md_text, json_text):
        lowered = text.lower()
        for token in forbidden:
            assert token not in lowered, token


def test_engine_band_verdicts_are_not_carried_into_the_report(gen, repo_data):
    """The engine-run file's own band verdicts stay out of this report."""
    _csv, md_text, json_text = _generate(gen, repo_data)
    for text in (md_text, json_text):
        for token in ("known_fail", "not_reproducible", "no_band", "gap_pct_of_mid"):
            assert token not in text, token


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


def test_committed_artifacts_are_up_to_date(gen):
    """The committed Section 9 artifacts match a fresh run of the generator."""
    assert _run_main(gen, "--check") == 0


def test_missing_input_is_a_hard_error(gen, repo_data, tmp_path):
    rc = _run_main(
        gen,
        "--results-json",
        str(tmp_path / "does-not-exist.json"),
        "--output-csv",
        str(tmp_path / "out.csv"),
        "--output-md",
        str(tmp_path / "out.md"),
        "--output-json",
        str(tmp_path / "out.json"),
    )
    assert rc == 2
