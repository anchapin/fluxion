#!/usr/bin/env python3
"""
ASHRAE 140 Section 9 Standard Output Report generator (Issue #4276, Section 9
slice).

Emits the Section 9 (Space-Cooling Equipment Performance, Cases CE100-CE200)
reporting artifacts the standard asks for, in the same shape as the Section 7
slice (``scripts/generate_section7_standard_output_report.py``):

  1. A FLAT results CSV -- one row per Section 9 CE case, one column per
     reported value fluxion actually produces
     (``data/reference/section9/section9_standard_output_report.csv``).
  2. A CONFORMANCE TABLE -- one row per Section 9 CE case, classified
     ``conformant`` / ``incomplete`` / ``not implemented``, with a per
     requirement cell state (``report`` / ``partial`` / ``missing`` / ``na``)
     (``data/reference/section9/section9_standard_output_conformance.md`` and
     ``.json``).

There is deliberately NO aggregate score, NO grade, NO percentage and NO
pass/fail verdict anywhere in the outputs: Standard 140 defines no such
quantity (see ``data/reference/ashrae140/analytical_acceptance_criteria.json``
-> ``_standard_structure.consequence``: "Standard 140 is a method of test, not
a rating or compliance threshold").  Every case is classified individually.

Sources (all committed, all read fresh on every run -- no number is carried
forward from a previous run of this script):

  * ``data/reference/ashrae140/section9_ce_cases.json`` -- the generated
    Section 9 CE100-CE200 case table (case ids, titles, series, input
    conditions, weather-file binding, and the per-case reference bands).
  * ``data/reference/ashrae140/section9_ce100_performance_map.json`` -- the
    CE100 performance map (Tables 9-5 through 9-10 of 140-2023), the
    analytical/reference rating values the engine evaluates.
  * ``tests/validation/hvac_equipment_rated_performance/data/
    section9_engine_run_results.json`` -- the recorded Section 9 engine-run
    results (the steady-state dispatch of every case at every entering
    wet-bulb column the map tabulates).

The required-output list is DERIVED, not transcribed: the 140-2023 Section 9
output requirements live in the licensed normative materials (Issue #4285),
so the repository carries no explicit Section 9 output-requirement list.  The
derivation rule is stated in the conformance Markdown header and in
``DERIVATION`` below; it uses only the case definitions and the performance
map.

Usage:
    python3 scripts/generate_section9_standard_output_report.py
    python3 scripts/generate_section9_standard_output_report.py --check
    python3 scripts/generate_section9_standard_output_report.py \\
        --output-csv /tmp/out.csv --output-md /tmp/out.md --output-json /tmp/out.json

Exit codes:
    0  success (or ``--check`` found no drift)
    1  ``--check`` detected drift against the committed artifacts
    2  a required input could not be read/parsed
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Pin stdout / stderr to UTF-8 so the glyphs in the console output work on
# Windows runners whose default stdio codec is cp1252.  No-op on POSIX.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

REPO = Path(__file__).resolve().parent.parent

CASES_JSON = REPO / "data" / "reference" / "ashrae140" / "section9_ce_cases.json"
MAP_JSON = (
    REPO / "data" / "reference" / "ashrae140" / "section9_ce100_performance_map.json"
)
RESULTS_JSON = (
    REPO
    / "tests"
    / "validation"
    / "hvac_equipment_rated_performance"
    / "data"
    / "section9_engine_run_results.json"
)
OUTPUT_CSV = (
    REPO / "data" / "reference" / "section9" / "section9_standard_output_report.csv"
)
OUTPUT_MD = (
    REPO
    / "data"
    / "reference"
    / "section9"
    / "section9_standard_output_conformance.md"
)
OUTPUT_JSON = (
    REPO
    / "data"
    / "reference"
    / "section9"
    / "section9_standard_output_conformance.json"
)

# The performance-map table the engine evaluates (the adjusted-net table,
# Tables 9-5 through 9-10 of 140-2023).  Read from the map artefact, never
# hard-coded.
DEFAULT_MAP_TABLE = "adjusted_net_Table_26e"

# ---------------------------------------------------------------------------
# Requirement catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """One derived Section 9 required output.

    ``field`` is the engine-run field the requirement is read from (``None``
    when the steady-state dispatch cannot produce it at all), ``scale`` is
    the divisor applied to report the value in the requirement's unit (the
    engine records powers and capacities in W, the report in kW), and
    ``per_ewb`` says whether the requirement is reported once per entering
    wet-bulb column of the map or once per case.
    """

    id: str
    title: str
    unit: str
    basis: str
    field_name: str | None
    scale: float = 1.0
    per_ewb: bool = True
    token: str = ""


# The derived Section 9 required-output list.  The ids are DERIVED ids
# ("9a.<quantity>"), NOT clause numbers of the standard: the standard's
# Section 9 output requirements are in the licensed 140-2023 normative
# materials (Issue #4285) and are therefore not transcribed anywhere in this
# repository.  See DERIVATION.
REQUIREMENTS: list[Requirement] = [
    Requirement(
        id="9a.case_inputs",
        title=(
            "Case input conditions: internal sensible gains, internal latent "
            "gains, cooling setpoint and outdoor dry-bulb temperature"
        ),
        unit="mixed",
        basis="case definition (section9_ce_cases.json -> conditions)",
        field_name=None,
        per_ewb=False,
    ),
    Requirement(
        id="9a.entering_dry_bulb",
        title="Entering dry-bulb temperature (the case cooling setpoint)",
        unit="degC",
        basis="case definition (cooling setpoint) and engine-run result",
        field_name="entering_dry_bulb_C",
        per_ewb=False,
    ),
    Requirement(
        id="9a.entering_wet_bulb",
        title="Entering wet-bulb temperature",
        unit="degC",
        basis="performance map entering-condition columns (engine run)",
        field_name="entering_wet_bulb_C",
        per_ewb=True,
    ),
    Requirement(
        id="9a.total_capacity",
        title="Total cooling capacity at the case condition",
        unit="kW",
        basis="engine dispatch (performance-map rating at the condition)",
        field_name="total_capacity_w",
        scale=1000.0,
    ),
    Requirement(
        id="9a.sensible_capacity",
        title="Sensible cooling capacity at the case condition",
        unit="kW",
        basis="engine dispatch (performance-map rating at the condition)",
        field_name="sensible_capacity_w",
        scale=1000.0,
    ),
    Requirement(
        id="9a.latent_capacity",
        title="Latent cooling capacity at the case condition",
        unit="kW",
        basis="engine dispatch (total minus sensible capacity)",
        field_name="latent_capacity_w",
        scale=1000.0,
    ),
    Requirement(
        id="9a.compressor_power",
        title="Compressor power at the case condition",
        unit="kW",
        basis="engine dispatch (performance-map rating at the condition)",
        field_name="compressor_power_w",
        scale=1000.0,
    ),
    Requirement(
        id="9a.delivered_capacity",
        title="Delivered cooling capacity at the operating point",
        unit="kW",
        basis="engine dispatch (steady-state delivery at the case load)",
        field_name="delivered_capacity_w",
        scale=1000.0,
    ),
    Requirement(
        id="9a.delivered_power",
        title="Delivered compressor power at the operating point",
        unit="kW",
        basis="engine dispatch (steady-state delivery at the case load)",
        field_name="delivered_power_w",
        scale=1000.0,
    ),
    Requirement(
        id="9a.cop",
        title=(
            "Coefficient of performance at the operating point (total "
            "capacity divided by compressor plus indoor fan power)"
        ),
        unit="-",
        basis="engine dispatch",
        field_name="cop",
    ),
    Requirement(
        id="9a.plr",
        title="Part load ratio at the operating point",
        unit="-",
        basis="engine dispatch (load balance)",
        field_name="plr",
    ),
    Requirement(
        id="9a.shr",
        title="Sensible heat ratio at the operating point",
        unit="-",
        basis="engine dispatch",
        field_name="shr",
    ),
    Requirement(
        id="9a.peak_cooling_power",
        title=(
            "Peak cooling power at the case condition (compressor power plus "
            "the map's indoor fan heat, adjusted-net basis)"
        ),
        unit="kW",
        basis="engine dispatch",
        field_name="peak_cooling_power_kw",
    ),
    Requirement(
        id="9a.required_load",
        title="Required cooling load of the case (internal sensible plus latent gains)",
        unit="kW",
        basis="engine dispatch (case load balance)",
        field_name="required_load_kw",
    ),
    Requirement(
        id="9a.capacity_shortfall",
        title="Capacity shortfall of the delivered load against the required load",
        unit="W",
        basis="engine dispatch (load balance)",
        field_name="capacity_shortfall_w",
    ),
    Requirement(
        id="9a.annual_cooling",
        title="Annual cooling energy of the case",
        unit="MWh",
        basis=(
            "case definition: the case binds an annual weather file and the "
            "case table names the annual_cooling metric for the case"
        ),
        field_name=None,
        per_ewb=False,
    ),
    Requirement(
        id="9a.max_zone_temp",
        title="Maximum zone temperature of the case",
        unit="degC",
        basis=(
            "case definition: the case table names the max_zone_temp metric "
            "for the case (CE150 only)"
        ),
        field_name=None,
        per_ewb=False,
    ),
]

# The unit token used in the flat CSV's column names (``kW`` -> ``kw``,
# ``degC`` -> ``C``, ``W`` -> ``w``, ``-`` -> no token at all).
UNIT_TOKENS = {
    "kW": "kw",
    "W": "w",
    "MWh": "mwh",
    "degC": "C",
    "-": "",
    "mixed": "",
}


def requirement_token(requirement: "Requirement") -> str:
    """The unit token appended to a requirement's column names."""
    return UNIT_TOKENS.get(requirement.unit, requirement.unit.lower())


REQUIREMENT_IDS = [requirement.id for requirement in REQUIREMENTS]

# The per-entering-condition requirements, in catalog order.
PER_EWB_REQUIREMENTS = [req for req in REQUIREMENTS if req.per_ewb]

# The case-level requirements that carry a column of their own.
CASE_LEVEL_REQUIREMENTS = [req for req in REQUIREMENTS if not req.per_ewb]

# ---------------------------------------------------------------------------
# Derivation of the required-output list
# ---------------------------------------------------------------------------

DERIVATION = (
    "The 140-2023 Section 9 output requirements live in the licensed "
    "normative materials (Issue #4285), so this repository carries no "
    "explicit Section 9 required-output list. The required-output list used "
    "here is therefore DERIVED from the two committed Section 9 source "
    "artefacts -- the case definitions "
    "(data/reference/ashrae140/section9_ce_cases.json) and the performance "
    "map (data/reference/ashrae140/section9_ce100_performance_map.json) -- by "
    "the following rule. (1) INPUTS: every case definition names the case's "
    "input conditions (internal sensible gains, internal latent gains, "
    "cooling setpoint, outdoor dry-bulb temperature) and the weather file "
    "the case binds, so the case inputs are required outputs of the case "
    "record. (2) ENTERING CONDITIONS: the performance map tabulates the "
    "entering dry-bulb and entering wet-bulb columns of the rating tables, "
    "so the entering conditions are required outputs. (3) RATED AND "
    "DISPATCHED PERFORMANCE: the performance map tabulates the rated total "
    "capacity, sensible capacity, latent capacity and compressor power at "
    "each (outdoor dry-bulb, entering wet-bulb, entering dry-bulb) point, "
    "and the steady-state dispatch of a case reports the delivered capacity "
    "and power, the coefficient of performance, the part load ratio, the "
    "sensible heat ratio, the peak cooling power, the required cooling load "
    "and the capacity shortfall at that point, so all of them are required "
    "outputs. (4) ANNUAL CASE OUTPUTS: every case definition binds an annual "
    "TMY2 weather file and the case table names the annual_cooling metric "
    "(MWh) for the case, and for CE150 also the max_zone_temp metric "
    "(degC), so those are required outputs as well. The derived requirement "
    "ids below are this report's own labels -- they are NOT clause numbers "
    "of the standard, which are not transcribed in this repository."
)

# ---------------------------------------------------------------------------
# Notes carried in the generated artifacts
# ---------------------------------------------------------------------------

NOTES: list[str] = [
    "Case states are conformant / incomplete / not implemented. No aggregate "
    "grade, rate or pass/fail verdict is reported anywhere in these "
    "artifacts: the standard defines no such quantity.",
    "Requirement cell states are report (required, produced), partial "
    "(required, produced for a subset), missing (required, not produced) and "
    "na (not required for this case).",
    DERIVATION,
    "The per-case entering wet-bulb is not transcribed for any CE case: the "
    "140-2023 Section 9 input specification is licensed (Issue #4285). The "
    "engine harness therefore dispatches every case at every entering "
    "wet-bulb column the performance map tabulates (15.0 / 17.2 / 19.4 / "
    "21.7 degC) and reports the envelope; the entering wet-bulb requirement "
    "is therefore 'partial' for every case (the envelope is reported, the "
    "case's own entering condition is not transcribed).",
    "The annual case outputs (annual cooling energy, maximum zone "
    "temperature) are not produced by this tree: the normative CE weather "
    "files (CE100A, CE110A, CE165A, CE200A) are licensed ASHRAE "
    "supplemental material that is never committed, and the steady-state "
    "equipment dispatch has no duty cycle and no zone-temperature "
    "trajectory. The affected requirement cells are 'missing' and the "
    "corresponding report cells are empty -- no value is invented.",
    "The performance-map columns are the analytical reference: Tables 9-5 "
    "through 9-10 of 140-2023 are the standard's own rating tables, and the "
    "engine evaluates the same map, so the reference value and the engine "
    "value coincide. They are shown next to the reported values as the "
    "analytical reference; no acceptance criterion is derived from them.",
    "The comparative columns are the IEA SHC Task 22 ensemble evidence "
    "ranges, transcribed per case from "
    "tests/validation/hvac_equipment_rated_performance/data/"
    "comparative_bounds_ce100_ce200.csv and carried in the case table. They "
    "are diagnostic context, not a criterion; the five cases the bounds file "
    "does not cover (CE165, CE180, CE185, CE190, CE195) carry no range and "
    "no range is invented for them.",
    "The engine-run results file records its own per-case band bookkeeping "
    "(band kind, band edges, direction, gap, band status). That bookkeeping "
    "is deliberately NOT carried into this report: this report classifies "
    "conformance of the reported outputs only, and carries the reference "
    "ranges themselves, without any verdict attached to them.",
    "Powers and capacities are reported in kW; the engine records them in W, "
    "so the values are divided by 1000 rather than re-derived. The capacity "
    "shortfall is reported in W, as the engine records it. Reported values "
    "carry six decimal places.",
    "A case that did not run carries no value at all in the flat CSV -- not "
    "even its input conditions -- so an empty cell can never be read as a "
    "produced value.",
    "The Section 9b comparative test series is not implemented in this tree "
    "and is not part of the CE100-CE200 catalog reported here.",
]

UNIT_CONVERSIONS: list[dict[str, str]] = [
    {
        "field": "cooling capacities, compressor power, delivered capacity and power, "
        "required cooling load",
        "source": "section9_engine_run_results.json runs (…_w fields)",
        "source_unit": "W",
        "reported_unit": "kW",
        "factor": "/1000",
    },
]

# The roles the reference values play in this report.  Neither role is a
# criterion.
REFERENCE_ROLES: dict[str, str] = {
    "map_tables": (
        "analytical reference -- the standard's own rating tables (Tables 9-5 "
        "through 9-10 of 140-2023), shown next to the reported value; not a "
        "criterion"
    ),
    "comparative_bounds": (
        "diagnostic context -- the IEA SHC Task 22 ensemble evidence range "
        "(min/max), carried verbatim from the case table; not a criterion"
    ),
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_float(text: object) -> float | None:
    if text is None:
        return None
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        value = float(text)
    else:
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _fmt(value: float | None, places: int = 6) -> str:
    """Format a reported value; an absent value is an EMPTY cell."""
    if value is None:
        return ""
    return f"{value:.{places}f}"


def _fmt_input(value: object) -> str:
    """Format a case-definition input value (kept as the source spelled it)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _ewb_token(value: float) -> str:
    """The entering wet-bulb column as it appears in a column name."""
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> dict[str, dict]:
    """Load the generated Section 9 case table."""
    if not path.exists():
        raise FileNotFoundError(f"case table not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"case table has no cases: {path}")
    table: dict[str, dict] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not case_id:
            raise ValueError(f"case table entry without a case_id: {path}")
        if case_id in table:
            raise ValueError(f"duplicate case id in the case table: {case_id}")
        table[case_id] = case
    return table


def load_map(path: Path) -> dict:
    """Load the CE100 performance map (the analytical reference)."""
    if not path.exists():
        raise FileNotFoundError(f"performance map not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> dict[str, dict]:
    """Load the recorded Section 9 engine-run results."""
    if not path.exists():
        raise FileNotFoundError(f"engine-run results not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"engine-run results have no cases: {path}")
    table: dict[str, dict] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not case_id:
            raise ValueError(f"engine-run result without a case_id: {path}")
        table[case_id] = case
    return table


def map_ewb_columns(performance_map: dict, table_name: str) -> list[float]:
    """The entering wet-bulb columns the map tabulates, read from the map."""
    tables = performance_map.get("tables")
    if not isinstance(tables, dict) or table_name not in tables:
        raise ValueError(f"performance map has no table {table_name}")
    table = tables[table_name]
    if not isinstance(table, dict) or not table:
        raise ValueError(f"performance map table {table_name} is empty")
    first_row = next(iter(table.values()))
    columns: list[float] = []
    for key in first_row:
        try:
            columns.append(float(key))
        except (TypeError, ValueError):
            continue
    if not columns:
        raise ValueError(f"performance map table {table_name} has no entering columns")
    return sorted(columns)


def map_reference_point(
    performance_map: dict, table_name: str, odb_c: float, edb_c: float, ewb_c: float
) -> dict | None:
    """The map's rating at one (ODB, EWB, EDB) point, or None when absent."""
    tables = performance_map.get("tables") or {}
    table = tables.get(table_name) or {}
    row = table.get(f"{odb_c:g}") or table.get(str(odb_c))
    if not isinstance(row, dict):
        return None
    point = row.get(f"{ewb_c:g}") or row.get(str(ewb_c))
    if not isinstance(point, dict):
        return None
    reference = {
        "total_capacity_kW": _parse_float(point.get("total_capacity_kW")),
        "compressor_power_kW": _parse_float(point.get("compressor_power_kW")),
        "apparatus_dew_point_C": _parse_float(point.get("apparatus_dew_point_C")),
    }
    sensible = point.get("sensible_capacity_kW_by_edb")
    if isinstance(sensible, dict):
        value = sensible.get(f"{edb_c:g}", sensible.get(str(edb_c)))
        reference["sensible_capacity_kW"] = _parse_float(value)
    else:
        reference["sensible_capacity_kW"] = None
    return reference


# ---------------------------------------------------------------------------
# Case records
# ---------------------------------------------------------------------------


@dataclass
class CaseData:
    """Everything the report needs to know about one Section 9 CE case."""

    case_id: str
    title: str
    series: str
    weather_file: str
    conditions: dict
    ran: bool
    status: str = ""
    runs: list[dict] = field(default_factory=list)
    ewb_columns: list[float] = field(default_factory=list)
    cells: dict[str, str] = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    comparative: dict = field(default_factory=dict)


def _case_runs_by_ewb(case: CaseData) -> dict[float, dict]:
    """The case's engine runs keyed by their entering wet-bulb column."""
    runs: dict[float, dict] = {}
    for run in case.runs:
        ewb = _parse_float(run.get("entering_wet_bulb_C"))
        if ewb is None:
            continue
        runs[ewb] = run
    return runs


def _applies(requirement: Requirement, case: CaseData) -> bool:
    """Does this derived requirement apply to this case?"""
    if requirement.id == "9a.max_zone_temp":
        # The case table names the max_zone_temp metric only for the case
        # that carries it (CE150); the requirement is 'na' elsewhere.
        return "max_zone_temp" in case.comparative
    return True


def _evaluate(requirement: Requirement, case: CaseData) -> str:
    """One requirement cell: report / partial / missing / na."""
    if not _applies(requirement, case):
        return "na"
    if not case.ran:
        return "missing"
    if requirement.id == "9a.case_inputs":
        # The case inputs come from the case definition, not from a run.
        keys = (
            "internal_sensible_gains_W",
            "internal_latent_gains_W",
            "cooling_setpoint_C",
            "outdoor_dry_bulb_C",
        )
        if all(_parse_float(case.conditions.get(key)) is not None for key in keys):
            return "report"
        return "missing"
    if requirement.field_name is None:
        # An annual case output: the steady-state dispatch cannot produce
        # it (no weather file, no duty cycle, no zone trajectory).
        return "missing"
    runs = _case_runs_by_ewb(case)
    if not runs:
        return "missing"
    present = 0
    for run in runs.values():
        if _parse_float(run.get(requirement.field_name)) is not None:
            present += 1
    if present == 0:
        return "missing"
    if present < len(runs):
        return "partial"
    if requirement.id == "9a.entering_wet_bulb":
        # The envelope over the map's tabulated entering columns is reported,
        # but the case's own entering condition is not transcribed.
        transcribed = case.conditions.get("entering_wet_bulb_C")
        if transcribed is None:
            return "partial"
    return "report"


def _case_status(case: CaseData) -> str:
    """conformant / incomplete / not implemented (no score, no verdict)."""
    if not case.ran:
        return "not implemented"
    for requirement in REQUIREMENTS:
        if not _applies(requirement, case):
            continue
        if case.cells[requirement.id] != "report":
            return "incomplete"
    return "conformant"


def build_cases(
    case_table: dict[str, dict],
    results: dict[str, dict],
    performance_map: dict,
    table_name: str = DEFAULT_MAP_TABLE,
) -> list[CaseData]:
    """Build the per-case records: every Section 9 CE case, run or not."""
    columns = map_ewb_columns(performance_map, table_name)
    cases: list[CaseData] = []
    for case_id in sorted(case_table):
        definition = case_table[case_id]
        result = results.get(case_id)
        conditions = dict(definition.get("conditions") or {})
        comparative = {}
        acceptance = definition.get("acceptance") or {}
        comparative_block = acceptance.get("comparative") or {}
        bounds = comparative_block.get("bounds") or {}
        if bounds:
            comparative = {
                "source": comparative_block.get("source"),
                "role": comparative_block.get("role"),
                "bounds": bounds,
            }
        case = CaseData(
            case_id=case_id,
            title=definition.get("name", case_id),
            series=definition.get("series", ""),
            weather_file=definition.get("weather_file", ""),
            conditions=conditions,
            ran=result is not None,
            runs=list((result or {}).get("runs") or []),
            ewb_columns=list(columns),
            comparative=comparative,
        )
        # The case's own entering wet-bulb, when the case table transcribes
        # one (it transcribes none today -- the input specification is
        # licensed).
        transcribed = (
            (definition.get("equipment_performance") or {}).get("entering_wet_bulb_C")
        )
        if transcribed is not None:
            case.conditions["entering_wet_bulb_C"] = transcribed
        case.reference = _case_reference(case, performance_map, table_name)
        case.cells = {
            requirement.id: _evaluate(requirement, case) for requirement in REQUIREMENTS
        }
        case.status = _case_status(case)
        cases.append(case)
    return cases


def _case_reference(
    case: CaseData, performance_map: dict, table_name: str
) -> dict:
    """The analytical reference for one case: the map's rating points."""
    odb = _parse_float(case.conditions.get("outdoor_dry_bulb_C"))
    edb = _parse_float(case.conditions.get("cooling_setpoint_C"))
    points: list[dict] = []
    if odb is not None and edb is not None:
        for ewb in case.ewb_columns:
            point = map_reference_point(performance_map, table_name, odb, edb, ewb)
            if point is None:
                continue
            entry = {"entering_wet_bulb_C": ewb}
            entry.update(point)
            points.append(entry)
    return {
        "table": table_name,
        "table_source": performance_map.get("_source"),
        "outdoor_dry_bulb_C": odb,
        "entering_dry_bulb_C": edb,
        "points": points,
        "ari_rating_point": performance_map.get("ari_rating_point"),
    }


# ---------------------------------------------------------------------------
# Flat report rows
# ---------------------------------------------------------------------------


def flat_columns(ewb_columns: list[float]) -> list[str]:
    """The flat report's column order (one column per reported value)."""
    columns = ["case_id", "case_title", "case_status"]
    columns += [
        "9a.case_inputs_internal_sensible_gains_W",
        "9a.case_inputs_internal_latent_gains_W",
        "9a.case_inputs_cooling_setpoint_C",
        "9a.case_inputs_outdoor_dry_bulb_C",
        "9a.case_inputs_weather_file",
        "9a.case_inputs_series",
    ]
    columns.append("9a.entering_dry_bulb_C")
    for requirement in PER_EWB_REQUIREMENTS:
        for ewb in ewb_columns:
            token = requirement_token(requirement)
            name = f"{requirement.id}_{token}" if token else requirement.id
            columns.append(f"{name}_ewb_{_ewb_token(ewb)}")
    columns += [
        "9a.annual_cooling_mwh",
        "9a.max_zone_temp_C",
    ]
    return columns


def flat_values(case: CaseData) -> dict[str, str]:
    """The flat report's cells for one case, keyed by column name.

    A cell is empty when the case did not run or when fluxion produced no
    such value -- nothing is invented here.
    """
    values: dict[str, str] = {}
    if not case.ran:
        return values

    values["9a.case_inputs_internal_sensible_gains_W"] = _fmt_input(
        case.conditions.get("internal_sensible_gains_W")
    )
    values["9a.case_inputs_internal_latent_gains_W"] = _fmt_input(
        case.conditions.get("internal_latent_gains_W")
    )
    values["9a.case_inputs_cooling_setpoint_C"] = _fmt_input(
        case.conditions.get("cooling_setpoint_C")
    )
    values["9a.case_inputs_outdoor_dry_bulb_C"] = _fmt_input(
        case.conditions.get("outdoor_dry_bulb_C")
    )
    values["9a.case_inputs_weather_file"] = case.weather_file
    values["9a.case_inputs_series"] = case.series

    runs = _case_runs_by_ewb(case)
    entering_dry_bulb = _parse_float(case.conditions.get("cooling_setpoint_C"))
    values["9a.entering_dry_bulb_C"] = _fmt(entering_dry_bulb, 3)

    for requirement in PER_EWB_REQUIREMENTS:
        for ewb in case.ewb_columns:
            token = requirement_token(requirement)
            name = f"{requirement.id}_{token}" if token else requirement.id
            column = f"{name}_ewb_{_ewb_token(ewb)}"
            run = runs.get(ewb)
            if run is None:
                values[column] = ""
                continue
            raw = _parse_float(run.get(requirement.field_name))
            if raw is None:
                values[column] = ""
                continue
            values[column] = _fmt(raw / requirement.scale)

    # The annual case outputs are not produced by the steady-state dispatch.
    values["9a.annual_cooling_mwh"] = ""
    values["9a.max_zone_temp_C"] = ""
    return values


def render_flat_csv(cases: list[CaseData], ewb_columns: list[float]) -> str:
    columns = flat_columns(ewb_columns)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for case in cases:
        values = flat_values(case)
        row = [case.case_id, case.title, case.status]
        for column in columns[3:]:
            row.append(values.get(column, ""))
        writer.writerow(row)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Conformance table
# ---------------------------------------------------------------------------


def _spread(values: list[float]) -> str:
    """A min-to-max spread of reference values (diagnostic only)."""
    finite = [value for value in values if value is not None]
    if not finite:
        return ""
    return f"{min(finite):g} to {max(finite):g}"


def _reference_columns(case: CaseData) -> list[str]:
    """The reference (diagnostic) columns of one case's conformance row."""
    points = case.reference.get("points") or []
    total = [_parse_float(point.get("total_capacity_kW")) for point in points]
    sensible = [_parse_float(point.get("sensible_capacity_kW")) for point in points]
    compressor = [_parse_float(point.get("compressor_power_kW")) for point in points]
    dew_point = [_parse_float(point.get("apparatus_dew_point_C")) for point in points]
    bounds = case.comparative.get("bounds") or {}

    def _bound(metric: str) -> str:
        entry = bounds.get(metric)
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return ""
        low = _parse_float(entry[0])
        high = _parse_float(entry[1])
        if low is None or high is None:
            return ""
        return f"{low:g} to {high:g}"

    return [
        _spread(total),
        _spread(sensible),
        _spread(compressor),
        _spread(dew_point),
        _bound("peak_cooling"),
        _bound("annual_cooling"),
        _bound("max_zone_temp"),
    ]


REFERENCE_HEADERS = [
    "Analytical reference total capacity at the case condition (kW)",
    "Analytical reference sensible capacity at the case condition (kW)",
    "Analytical reference compressor power at the case condition (kW)",
    "Analytical reference apparatus dew point at the case condition (degC)",
    "Comparative evidence range, peak cooling (kW)",
    "Comparative evidence range, annual cooling (MWh)",
    "Comparative evidence range, maximum zone temperature (degC)",
]


def coverage_by_requirement(cases: list[CaseData]) -> list[dict]:
    """How many cases sit in each cell state, per requirement.

    This is coverage bookkeeping -- the number of cases in each state.  No
    rate is derived from it.
    """
    summary: list[dict] = []
    for requirement in REQUIREMENTS:
        counts = {"report": 0, "partial": 0, "missing": 0, "na": 0}
        for case in cases:
            counts[case.cells[requirement.id]] += 1
        summary.append(
            {
                "requirement": requirement.id,
                "title": requirement.title,
                "unit": requirement.unit,
                "counts": counts,
            }
        )
    return summary


def case_status_counts(cases: list[CaseData]) -> dict[str, int]:
    """The number of cases in each case state (coverage bookkeeping)."""
    counts = {"conformant": 0, "incomplete": 0, "not implemented": 0}
    for case in cases:
        counts[case.status] += 1
    return counts


def render_conformance_markdown(cases: list[CaseData]) -> str:
    lines: list[str] = []
    lines.append("# Section 9 Standard Output Report -- conformance table")
    lines.append("")
    lines.append(
        "ASHRAE Standard 140 Section 9 (Space-Cooling Equipment Performance) "
        "case set CE100-CE200, reported in the Annex A2 Standard Output "
        "Report shape: one row per case, one column per reported value, and "
        "a per-requirement conformance state."
    )
    lines.append("")
    lines.append(
        "Case states: `conformant` (ran with the specified inputs and produced "
        "every required output), `incomplete` (ran, but at least one required "
        "output is missing or only partial), `not implemented` (did not run). "
        "Requirement cell states: `report` (required and produced), `partial` "
        "(required and produced for a subset), `missing` (required and not "
        "produced), `na` (not required for this case)."
    )
    lines.append("")
    lines.append(
        "**No aggregate grade, rate or pass/fail verdict appears in this "
        "report.** Standard 140 is a method of test, not a rating or "
        "compliance threshold: it defines no such quantity, and each case is "
        "classified individually as conformant, incomplete or not "
        "implemented."
    )
    lines.append("")
    lines.append("## How the required-output list was derived")
    lines.append("")
    lines.append(DERIVATION)
    lines.append("")
    lines.append(
        "**The reference columns are diagnostic context and analytical "
        "reference, not criteria.** The 'Analytical reference' columns are "
        "the standard's own rating tables (Tables 9-5 through 9-10 of "
        "140-2023) at the case condition; the engine evaluates the same map, "
        "so the reference and the reported value coincide. The 'Comparative "
        "evidence range' columns are the IEA SHC Task 22 ensemble min/max "
        "ranges carried verbatim from the case table; the case table labels "
        "them 'evidence range (ensemble min/max), not a pass/fail criterion'. "
        "Neither role is used to derive a verdict here, and no conclusion is "
        "drawn from either."
    )
    lines.append("")
    lines.append("## Reported values and conformance")
    lines.append("")
    header = ["Case", "Title", "Status"]
    header += REQUIREMENT_IDS
    header += REFERENCE_HEADERS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for case in cases:
        row = [case.case_id, case.title, case.status]
        row += [case.cells[requirement] for requirement in REQUIREMENT_IDS]
        row += _reference_columns(case)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Coverage bookkeeping")
    lines.append("")
    lines.append(
        "The table below counts, per requirement, how many cases sit in each "
        "cell state. These are counts of cases, not a rate; no aggregate "
        "quantity is derived from them."
    )
    lines.append("")
    lines.append("| Requirement | report | partial | missing | na |")
    lines.append("|---|---|---|---|---|")
    for entry in coverage_by_requirement(cases):
        counts = entry["counts"]
        lines.append(
            f"| {entry['requirement']} | {counts['report']} | {counts['partial']} | "
            f"{counts['missing']} | {counts['na']} |"
        )
    lines.append("")
    status_counts = case_status_counts(cases)
    lines.append(
        "Case states: "
        + ", ".join(
            f"{state} {status_counts[state]}" for state in
            ("conformant", "incomplete", "not implemented")
        )
        + ". Every Section 9 case is dispatched by the engine harness, so no "
        "case is 'not implemented'; every case is 'incomplete' for the same "
        "documented reason -- the annual case outputs are not produced by "
        "this tree and the per-case entering wet-bulb is only partially "
        "transcribed. Every steady-state dispatch output is reported for "
        "every case."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in NOTES:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(
        "- `data/reference/ashrae140/section9_ce_cases.json` -- the generated "
        "Section 9 CE100-CE200 case table (case definitions, input "
        "conditions, weather-file binding, per-case reference bands)."
    )
    lines.append(
        "- `data/reference/ashrae140/section9_ce100_performance_map.json` -- "
        "the CE100 performance map (Tables 9-5 through 9-10 of 140-2023), the "
        "analytical reference."
    )
    lines.append(
        "- `tests/validation/hvac_equipment_rated_performance/data/"
        "section9_engine_run_results.json` -- the recorded Section 9 "
        "engine-run results (the steady-state dispatch of every case at "
        "every entering wet-bulb column the map tabulates)."
    )
    lines.append(
        "- The flat results CSV for this case set: "
        "`data/reference/section9/section9_standard_output_report.csv`."
    )
    lines.append("")
    return "\n".join(lines)


def render_conformance_json(cases: list[CaseData]) -> str:
    payload = {
        "schema_version": "1",
        "scope": {
            "standard": "ANSI/ASHRAE Standard 140",
            "section": "Section 9 -- Space-Cooling Equipment Performance",
            "case_set": "CE100-CE200 (140-2023 Section 9.2 analytical verification)",
            "output_requirements": (
                "derived from the case definitions and the performance map; "
                "the 140-2023 Section 9 output requirements are in the "
                "licensed normative materials (Issue #4285)"
            ),
        },
        "states": {
            "case": ["conformant", "incomplete", "not implemented"],
            "requirement": ["report", "partial", "missing", "na"],
        },
        "sources": {
            "case_table": "data/reference/ashrae140/section9_ce_cases.json",
            "performance_map": (
                "data/reference/ashrae140/section9_ce100_performance_map.json"
            ),
            "engine_run_results": (
                "tests/validation/hvac_equipment_rated_performance/data/"
                "section9_engine_run_results.json"
            ),
            "flat_report_csv": (
                "data/reference/section9/section9_standard_output_report.csv"
            ),
        },
        "reference_roles": REFERENCE_ROLES,
        "derivation": DERIVATION,
        "unit_conversions": UNIT_CONVERSIONS,
        "notes": NOTES,
        "coverage": {
            "role": (
                "counts of cases per cell state -- coverage bookkeeping, not "
                "a rate"
            ),
            "case_status_counts": case_status_counts(cases),
            "by_requirement": coverage_by_requirement(cases),
        },
        "requirements": [
            {
                "id": requirement.id,
                "title": requirement.title,
                "unit": requirement.unit,
                "basis": requirement.basis,
            }
            for requirement in REQUIREMENTS
        ],
        "cases": [
            {
                "case": case.case_id,
                "title": case.title,
                "series": case.series,
                "weather_file": case.weather_file,
                "status": case.status,
                "ran": case.ran,
                "conditions": case.conditions,
                "entering_wet_bulb_sweep_C": case.ewb_columns,
                "requirements": {
                    requirement: case.cells[requirement]
                    for requirement in REQUIREMENT_IDS
                },
                "reference_diagnostic": {
                    "map": case.reference,
                    "comparative": case.comparative,
                },
            }
            for case in cases
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def generate(
    cases_json: Path,
    map_json: Path,
    results_json: Path,
) -> tuple[str, str, str]:
    """Generate the three artifacts as text (pure; no writes)."""
    case_table = load_cases(cases_json)
    performance_map = load_map(map_json)
    results = load_results(results_json)
    cases = build_cases(case_table, results, performance_map)
    columns = map_ewb_columns(performance_map, DEFAULT_MAP_TABLE)
    return (
        render_flat_csv(cases, columns),
        render_conformance_markdown(cases),
        render_conformance_json(cases),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the ASHRAE 140 Section 9 Standard Output Report "
            "(flat results CSV + conformance table)."
        )
    )
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=CASES_JSON,
        help="the generated Section 9 CE100-CE200 case table",
    )
    parser.add_argument(
        "--map-json",
        type=Path,
        default=MAP_JSON,
        help="the CE100 performance map (analytical reference)",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=RESULTS_JSON,
        help="the recorded Section 9 engine-run results",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="flat results CSV output path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=OUTPUT_MD,
        help="conformance table (markdown) output path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help="conformance table (JSON) output path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed artifacts differ from the regenerated ones",
    )
    args = parser.parse_args(argv)

    try:
        csv_text, md_text, json_text = generate(
            args.cases_json, args.map_json, args.results_json
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    outputs = (
        (args.output_csv, csv_text),
        (args.output_md, md_text),
        (args.output_json, json_text),
    )

    if args.check:
        drifted = []
        for path, text in outputs:
            if not path.exists():
                drifted.append((path, "missing"))
                continue
            if path.read_text(encoding="utf-8") != text:
                drifted.append((path, "differs"))
        if drifted:
            for path, reason in drifted:
                print(f"DRIFT: {path} {reason}")
            print(
                "Regenerate with: python3 scripts/"
                "generate_section9_standard_output_report.py"
            )
            return 1
        for path, _text in outputs:
            print(f"OK: {path} is up to date")
        return 0

    for path, text in outputs:
        _write(path, text)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
