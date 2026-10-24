#!/usr/bin/env python3
"""
ASHRAE 140 Section 7 Standard Output Report generator (Issue #4276, Section 7
slice).

Emits the Section 7 (Building Thermal Envelope and Fabric Load) reporting
artifacts the standard actually asks for, replacing the invented pass-rate
scorecard shape with:

  1. A FLAT results CSV -- one row per Section 7 case, one column per
     required reported value fluxion actually produces
     (``data/reference/section7/section7_standard_output_report.csv``).
  2. A CONFORMANCE TABLE -- one row per Section 7 case, classified
     ``conformant`` / ``incomplete`` / ``not implemented``, with a per
     requirement cell state (``report`` / ``partial`` / ``missing`` / ``na``)
     (``data/reference/section7/section7_standard_output_conformance.md`` and
     ``.json``).

There is deliberately NO aggregate score, NO grade and NO pass/fail verdict
anywhere in the outputs: the standard defines no such quantity, and the
Annex B8 example-result spread is a DIAGNOSTIC range, not a pass/fail band
(Annex B8 says so verbatim; see ``data/reference/ashrae140/
annex_b8_section7_statistics.json``).  The Annex B8 spread is therefore
carried next to the reported values as diagnostic context and is labelled as
such in every artifact.

Sources (all committed, all read fresh on every run -- no number is carried
forward from a previous run of this script):

  * ``data/reference/section7/workflow_results.csv`` -- the Section 7 results
    CSV produced by the Section 7 emitter
    (``tests/ashrae_140_section7_gsr_emitter.rs``), which runs the engine
    blind/spec-only on the canonical strict-gate weather.  This is the
    "current results data on this tree".
  * ``data/reference/section7/section621_outputs.csv`` -- the emitter's
    companion artifact for the Section 6.2.1.3/.4/.5 (Cases 610/620/630) and
    6.2.1.6 (Case 960 sun zone) outputs.
  * ``data/reference/ashrae140/annex_b8_section7_statistics.json`` -- the
    Annex B8 example-result statistics, used as DIAGNOSTIC context only.

Requirement numbering: the Section 7 case set is the case set of ANSI/ASHRAE
Standard 140-2017 Section 5.2 (called "Section 7" by fluxion's 140-2001
lineage).  Its Class I output requirements live in Section 6.2.1 of the
in-repo standard (``docs/140_2017_a_20200901.pdf``); the requirement ids used
here (``6.2.1.1.1`` ... ``6.2.1.8``) are the same ids the committed
conformance artifact ``data/reference/section7/section7_conformance.csv``
uses, so the two artifacts stay comparable.

Usage:
    python3 scripts/generate_section7_standard_output_report.py
    python3 scripts/generate_section7_standard_output_report.py --check
    python3 scripts/generate_section7_standard_output_report.py --regenerate
    python3 scripts/generate_section7_standard_output_report.py \\
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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Pin stdout / stderr to UTF-8 so the glyphs in the console output work on
# Windows runners whose default stdio codec is cp1252.  No-op on POSIX.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

REPO = Path(__file__).resolve().parent.parent

RESULTS_CSV = REPO / "data" / "reference" / "section7" / "workflow_results.csv"
COMPANION_CSV = REPO / "data" / "reference" / "section7" / "section621_outputs.csv"
ANNEX_B8 = (
    REPO / "data" / "reference" / "ashrae140" / "annex_b8_section7_statistics.json"
)
OUTPUT_CSV = (
    REPO / "data" / "reference" / "section7" / "section7_standard_output_report.csv"
)
OUTPUT_MD = (
    REPO
    / "data"
    / "reference"
    / "section7"
    / "section7_standard_output_conformance.md"
)
OUTPUT_JSON = (
    REPO
    / "data"
    / "reference"
    / "section7"
    / "section7_standard_output_conformance.json"
)

STANDARD_PDF = "docs/140_2017_a_20200901.pdf"

# ---------------------------------------------------------------------------
# Case catalog
# ---------------------------------------------------------------------------

# The Section 7 (140-2017 Section 5.2) case set, transcribed from the
# in-repo standard: Annex B1 Table B1-2 (Section 5.2.1 low-mass series),
# Table B1-3 (Section 5.2.2 / 5.2.3 high-mass and in-depth series) and the
# Section 5.2.2.3 free-float subsections.  Titles are the standard's own
# case titles.
SECTION7_CASES: list[tuple[str, str]] = [
    ("195", "Solid Conduction Test"),
    ("200", "Surface Convection/Infrared Radiation"),
    ("210", "Interior Infrared Radiation"),
    ("215", "Exterior Infrared Radiation"),
    ("220", "In-Depth Series Base Case"),
    ("230", "Infiltration"),
    ("240", "Internal Gains"),
    ("250", "Exterior Shortwave Absorptance"),
    ("270", "South Solar Gains"),
    ("280", "Cavity Albedo"),
    ("290", "South Shading"),
    ("300", "East/West Window Orientation"),
    ("310", "East/West Shading"),
    ("320", "Thermostat"),
    ("395", "Solid Conduction Test"),
    ("400", "Opaque Windows High-Conductance Wall Elements with Deadband Thermostat"),
    ("410", "Infiltration"),
    ("420", "Internal Gains"),
    ("430", "Exterior Shortwave Absorptance"),
    ("440", "Cavity Albedo"),
    ("450", "Constant Combined Interior and Exterior Surface Coefficients"),
    ("460", "Constant Combined Interior Surface Coefficients"),
    ("470", "Constant Combined Exterior Surface Coefficients"),
    ("600", "Base Case"),
    ("610", "South Shading"),
    ("620", "East/West Window Orientation"),
    ("630", "East/West Shading"),
    ("640", "Thermostat Setback"),
    ("650", "Night Ventilation"),
    ("660", "Low-Emissivity Windows with Argon Gas"),
    ("670", "Single Pane Windows"),
    ("680", "Increased Exterior Wall and Roof Insulation"),
    ("685", 'Base Case with "20,20" Thermostat'),
    ("695", 'Increased Exterior Wall and Roof Insulation with "20,20" Thermostat'),
    ("800", "High Mass without Solar Gains"),
    ("810", "High Mass Cavity Albedo"),
    ("900", "High Mass Base Building"),
    ("910", "High Mass South Shading"),
    ("920", "High Mass East/West Window Orientation"),
    ("930", "High Mass East/West Shading"),
    ("940", "High Mass Thermostat Setback"),
    ("950", "High Mass Night Ventilation"),
    ("960", "Sunspace"),
    ("980", "High Mass Increased Exterior Wall and Roof Insulation"),
    ("985", 'High Mass with "20,20" Thermostat'),
    ("995", 'High Mass Increased Exterior Wall and Roof Insulation with "20,20" Thermostat'),
    ("600FF", "Free-Float Low Mass"),
    ("650FF", "Free-Float Night Ventilation"),
    ("680FF", "Free-Float Increased Exterior Wall and Roof Insulation"),
    ("900FF", "Free-Float High Mass"),
    ("950FF", "Free-Float High Mass Night Ventilation"),
    ("980FF", "Free-Float High Mass Increased Exterior Wall and Roof Insulation"),
]

# fluxion implements one case that the 140-2017 standard does not define:
# the five-zone cross-coupling case 970.  It is reported in the table and
# flagged, because the standard has no output requirement to check it
# against beyond the generic 6.2.1.1 set.
FLUXION_EXTRA_CASES: list[tuple[str, str]] = [
    ("970", "Five zone cross coupling (fluxion case; not defined in 140-2017)"),
]

# The cases the Section 7 emitter runs (tests/ashrae_140_section7_gsr_emitter.rs).
IMPLEMENTED_CASES = (
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
)

# ---------------------------------------------------------------------------
# Requirement catalog (ANSI/ASHRAE Standard 140-2017 Section 6.2.1)
# ---------------------------------------------------------------------------

REQUIREMENTS: list[tuple[str, str, str]] = [
    (
        "6.2.1.1.1",
        "Annual heating load",
        "MWh",
    ),
    (
        "6.2.1.1.2",
        "Annual sensible cooling loads",
        "MWh",
    ),
    (
        "6.2.1.1.3",
        "Annual hourly integrated peak heating load with month, day and hour "
        "of occurrence",
        "kW",
    ),
    (
        "6.2.1.1.4",
        "Annual hourly integrated peak sensible cooling load with month, day "
        "and hour of occurrence",
        "kW",
    ),
    (
        "6.2.1.1.5",
        "All heating and cooling loads of 6.2.1.1.1 through 6.2.1.1.4 "
        "entered as positive (>= 0) values",
        "-",
    ),
    (
        "6.2.1.2.1",
        "Annual incident unshaded total solar radiation (diffuse and "
        "direct-beam) on horizontal, north, east, south and west surfaces",
        "kWh/m2",
    ),
    (
        "6.2.1.2.2",
        "Unshaded annual transmitted total solar radiation (diffuse and "
        "direct-beam) through south windows",
        "kWh/m2",
    ),
    (
        "6.2.1.2.3",
        "Sky temperature output: annual mean, annual hourly integrated "
        "minimum and maximum with month, day and hour of occurrence",
        "degC",
    ),
    (
        "6.2.1.2.4",
        "Monthly conditioned zone loads: monthly heating and cooling loads, "
        "monthly hourly integrated peak heating and cooling loads with day "
        "and hour of occurrence",
        "kWh / kW",
    ),
    (
        "6.2.1.3",
        "Case 610: annual transmitted solar radiation through the shaded "
        "south window with horizontal overhang",
        "kWh/m2",
    ),
    (
        "6.2.1.4",
        "Case 620: unshaded annual transmitted solar radiation through the "
        "west window",
        "kWh/m2",
    ),
    (
        "6.2.1.5",
        "Case 630: annual transmitted solar radiation through the shaded "
        "west window with overhang and vertical fins",
        "kWh/m2",
    ),
    (
        "6.2.1.6.min",
        "All free-float cases: annual hourly integrated minimum zone air "
        "temperature with month, day and hour of occurrence",
        "degC",
    ),
    (
        "6.2.1.6.max",
        "All free-float cases: annual hourly integrated maximum zone air "
        "temperature with month, day and hour of occurrence",
        "degC",
    ),
    (
        "6.2.1.6.mean",
        "All free-float cases: annual mean zone air temperature",
        "degC",
    ),
    (
        "6.2.1.7",
        "Case 900FF only: annual hourly integrated 1 degC zone air "
        "temperature bin frequencies",
        "hours",
    ),
    (
        "6.2.1.8",
        "Daily hourly output for the cases and output types listed in "
        "Table 6-1",
        "mixed",
    ),
]

REQUIREMENT_IDS = [req_id for req_id, _title, _unit in REQUIREMENTS]

# Cases that carry a Section 6.2.1.8 (Table 6-1) daily-hourly requirement.
DAILY_HOURLY_CASES = {
    "600",
    "640",
    "660",
    "670",
    "680",
    "685",
    "695",
    "900",
    "940",
    "980",
    "985",
    "995",
    "600FF",
    "650FF",
    "680FF",
    "900FF",
    "950FF",
    "980FF",
}

MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

# The emitter's contract CSV carries the monthly loads in MWh and the
# monthly peak powers in W (the emitter's own field names say "_mwh" and
# "_kw", but the peak array is filled with the raw W value before the
# annual-peak /1000 conversion -- see tests/ashrae_140_section7_gsr_emitter.rs).
# Section 6.2.1.2.4 asks for kWh and kW, so the flat report converts:
MONTHLY_MWH_TO_KWH = 1000.0
MONTHLY_W_TO_KW = 1.0 / 1000.0

# The emitter's temperature-bin histogram is a 90-slot array whose slot i
# counts the hours whose zone-air temperature floor falls in the 1 degC bin
# with lower bound (i - 19) degC, i.e. lower bounds -19..+70 degC.  Hours
# below -19 degC are clamped into the first bin and no hour reaches the
# +70 degC bin.  Section 6.2.1.7.1 words the range as "-20 degC to 70 degC",
# which is the same 90 bins read with the -20 degC lower bound.
TEMP_BIN_LOWER_BOUNDS = list(range(-19, 71))

# The emitter's day-profile days (the GSR contract days).
PROFILE_DAYS = (("0201", "Feb 1"), ("0504", "May 4"), ("0714", "Jul 14"))

# ---------------------------------------------------------------------------
# Notes carried in the generated artifacts
# ---------------------------------------------------------------------------

NOTES: list[str] = [
    "Case states are conformant / incomplete / not implemented. No aggregate "
    "grade or pass/fail verdict is reported anywhere in these "
    "artifacts: the standard defines no such quantity.",
    "Requirement cell states are report (required, produced), partial "
    "(required, produced for a subset), missing (required, not produced) and "
    "na (not required for this case).",
    "The Annex B8 columns are the min/max spread of the six example results "
    "published in informative Annex B8. Annex B8 states verbatim that these "
    "statistics 'do not have any substantial importance and are not to be "
    "interpreted as acceptance criteria'. They are diagnostic context only, "
    "not a criterion.",
    "Monthly loads are reported in kWh and monthly peak powers in kW, as "
    "Section 6.2.1.2.4 requires; the emitter's contract CSV carries them in "
    "MWh and W respectively, so the values are converted (x1000 and /1000) "
    "rather than re-derived.",
    "A monthly or annual peak occurrence is left empty when the matching peak "
    "value is zero: the emitter writes a sentinel timestamp for months and "
    "cases without load, and a sentinel is not an occurrence.",
    "Section 6.2.1.8 (Table 6-1) is held at 'partial' for every case that "
    "carries the requirement: the day cells of Table 6-1 in the in-repo copy "
    "of the standard carry unresolved tracked changes, so the required day "
    "list cannot be confirmed against a clean copy of the standard.",
    "The Section 6.2.1.7.1 temperature-bin histogram is the emitter's 90-slot "
    "array: slot i counts the hours whose zone-air temperature floor falls "
    "in the 1 degC bin with lower bound (i - 19) degC, i.e. lower bounds "
    "-19..+70 degC, with hours below -19 degC clamped into the first bin. "
    "The standard words the range as '-20 degC to 70 degC' (the same 90 bins "
    "read with the -20 degC lower bound), so the coldest 1 degC band is "
    "reported one bin high.",
    "Case 970 is a fluxion case with no 140-2017 definition; it is evaluated "
    "against the generic Section 6.2.1.1 output set only.",
]

UNIT_CONVERSIONS: list[dict[str, str]] = [
    {
        "field": "monthly heating and cooling loads",
        "source": "workflow_results.csv monthly_htg_cons / monthly_clg_cons",
        "source_unit": "MWh",
        "reported_unit": "kWh",
        "factor": "x1000",
    },
    {
        "field": "monthly hourly integrated peak heating and cooling loads",
        "source": "workflow_results.csv monthly_htg_peak_val / "
        "monthly_clg_peak_val",
        "source_unit": "W",
        "reported_unit": "kW",
        "factor": "/1000",
    },
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _split_csv_line(line: str) -> list[str]:
    """RFC-4180-ish field splitter (quotes, embedded commas)."""
    fields: list[str] = []
    cur: list[str] = []
    in_quotes = False
    chars = list(line)
    i = 0
    while i < len(chars):
        c = chars[i]
        if in_quotes:
            if c == '"':
                if i + 1 < len(chars) and chars[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    in_quotes = False
            else:
                cur.append(c)
        else:
            if c == '"':
                in_quotes = True
            elif c == ",":
                fields.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        i += 1
    fields.append("".join(cur))
    return fields


def _parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _packed_numbers(text: str | None) -> list[float] | None:
    """Parse a packed ``label...,v1,v2,...`` cell into its numeric tail.

    The emitter packs array fields as comma-joined values with one or two
    leading non-numeric label elements (e.g. ``0201,signed_kWh,0.0,...``).
    """
    if text is None:
        return None
    tokens = [t.strip() for t in text.split(",")]
    values: list[float] = []
    for token in reversed(tokens):
        parsed = _parse_float(token)
        if parsed is None:
            break
        values.append(parsed)
    values.reverse()
    return values


def _packed_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    tokens = [t.strip() for t in text.split(",")]
    values: list[int] = []
    for token in reversed(tokens):
        try:
            values.append(int(token))
        except (TypeError, ValueError):
            break
    values.reverse()
    return values


def _timestamp_ok(text: str | None) -> bool:
    """``DD-MMM-HH:MM`` shape check (the emitter's occurrence form)."""
    if not text:
        return False
    if len(text) != 12:
        return False
    return (
        text[2] == "-"
        and text[6] == "-"
        and text[9] == ":"
        and text[0:2].isdigit()
        and text[3:6].isalpha()
        and text[7:9].isdigit()
        and text[10:12].isdigit()
    )


def load_results(path: Path) -> dict[str, dict[str, str]]:
    """Load the emitter's contract CSV into ``{case_id: {column: value}}``.

    The contract header is dotted
    (``bestest_building_thermal_envelope_and_fabric_load_reporting.<field>``);
    the short field name is used as the key.
    """
    if not path.exists():
        raise FileNotFoundError(f"results CSV not found: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"results CSV is empty: {path}")
    header = _split_csv_line(lines[0])
    short = [h.rsplit(".", 1)[-1] for h in header]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = _split_csv_line(line)
        if len(fields) < len(short):
            continue
        case_id = fields[0].split(" ")[0].strip()
        if not case_id:
            continue
        rows[case_id] = {short[i]: fields[i] for i in range(len(short))}
    if not rows:
        raise ValueError(f"results CSV has no data rows: {path}")
    return rows


def load_companion(path: Path) -> dict[str, dict[str, str]]:
    """Load the emitter's Section 6.2.1 companion CSV (Cases 610/620/630, 960)."""
    if not path.exists():
        raise FileNotFoundError(f"companion CSV not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"companion CSV is empty: {path}")
    header = _split_csv_line(lines[0])
    short = [h.rsplit(".", 1)[-1] for h in header]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = _split_csv_line(line)
        if len(fields) < len(short):
            continue
        case_id = fields[0].strip()
        rows[case_id] = {short[i]: fields[i] for i in range(len(short))}
    return rows


def load_annex_b8(path: Path) -> dict:
    """Load the Annex B8 example-result statistics (diagnostic context only)."""
    if not path.exists():
        raise FileNotFoundError(f"Annex B8 statistics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Requirement applicability and evaluation
# ---------------------------------------------------------------------------


def is_free_float(case_id: str) -> bool:
    return case_id.endswith("FF")


def applies(requirement: str, case_id: str) -> bool:
    """Does this Section 6.2.1 requirement apply to this case?

    Applicability follows the requirement text and the Section 5.2.x
    "Output Requirements" subsections of the in-repo standard.  Case 970 has
    no 140-2017 definition and is evaluated against 6.2.1.1 only.
    """
    ff = is_free_float(case_id)
    if requirement in ("6.2.1.1.1", "6.2.1.1.2", "6.2.1.1.3", "6.2.1.1.4", "6.2.1.1.5"):
        return not ff
    if requirement == "6.2.1.2.1":
        return case_id == "600"
    if requirement == "6.2.1.2.2":
        # "(also for Cases 660 and 670)"
        return case_id in ("600", "660", "670")
    if requirement == "6.2.1.2.3":
        return case_id == "600"
    if requirement == "6.2.1.2.4":
        # "(also for Case 900)"
        return case_id in ("600", "900")
    if requirement == "6.2.1.3":
        return case_id == "610"
    if requirement == "6.2.1.4":
        return case_id == "620"
    if requirement == "6.2.1.5":
        return case_id == "630"
    if requirement in ("6.2.1.6.min", "6.2.1.6.max", "6.2.1.6.mean"):
        # Free-float cases, plus the sun zone of Case 960.
        return ff or case_id == "960"
    if requirement == "6.2.1.7":
        return case_id == "900FF"
    if requirement == "6.2.1.8":
        return case_id in DAILY_HOURLY_CASES
    return False


@dataclass
class CaseData:
    """Everything the report needs to know about one Section 7 case."""

    case_id: str
    title: str
    ran: bool
    status: str  # "conformant" | "incomplete" | "not implemented"
    row: dict[str, str]
    companion: dict[str, str]
    cells: dict[str, str]  # requirement id -> report/partial/missing/na


def _monthly_arrays(row: dict[str, str]) -> dict[str, list | None]:
    return {
        "htg_cons": _packed_numbers(row.get("monthly_htg_cons")),
        "clg_cons": _packed_numbers(row.get("monthly_clg_cons")),
        "htg_peak_val": _packed_numbers(row.get("monthly_htg_peak_val")),
        "clg_peak_val": _packed_numbers(row.get("monthly_clg_peak_val")),
        "htg_peak_time": (row.get("monthly_htg_peak_time") or "").split(","),
        "clg_peak_time": (row.get("monthly_clg_peak_time") or "").split(","),
    }


def _monthly_arrays_ok(row: dict[str, str]) -> bool:
    arrays = _monthly_arrays(row)
    for key in ("htg_cons", "clg_cons", "htg_peak_val", "clg_peak_val"):
        values = arrays[key]
        if not values or len(values) != 12:
            return False
        if not all(v == v and v not in (float("inf"), float("-inf")) and v >= 0.0 for v in values):
            return False
    for key in ("htg_peak_time", "clg_peak_time"):
        stamps = arrays[key]
        if len(stamps) != 12 or not all(_timestamp_ok(s) for s in stamps):
            return False
    return True


def evaluate(requirement: str, case: "CaseData") -> str:
    """One requirement cell: report / partial / missing / na."""
    if not applies(requirement, case.case_id):
        return "na"
    if not case.ran:
        return "missing"
    row = case.row
    companion = case.companion

    if requirement in ("6.2.1.1.1", "6.2.1.1.2"):
        key = "annual_heating" if requirement == "6.2.1.1.1" else "annual_cooling"
        value = _parse_float(row.get(key))
        return "report" if value is not None and value >= 0.0 else "missing"
    if requirement == "6.2.1.1.3":
        value = _parse_float(row.get("peak_heating_value"))
        return (
            "report"
            if value is not None and _timestamp_ok(row.get("peak_heating_time"))
            else "missing"
        )
    if requirement == "6.2.1.1.4":
        value = _parse_float(row.get("peak_cooling_value"))
        return (
            "report"
            if value is not None and _timestamp_ok(row.get("peak_cooling_time"))
            else "missing"
        )
    if requirement == "6.2.1.1.5":
        # Every listed load entered as a positive (>= 0) value.
        annual_h = _parse_float(row.get("annual_heating"))
        annual_c = _parse_float(row.get("annual_cooling"))
        peak_h = _parse_float(row.get("peak_heating_value"))
        peak_c = _parse_float(row.get("peak_cooling_value"))
        scalars = (annual_h, annual_c, peak_h, peak_c)
        if any(v is None or v < 0.0 for v in scalars):
            return "missing"
        arrays = _monthly_arrays(row)
        for key in ("htg_cons", "clg_cons", "htg_peak_val", "clg_peak_val"):
            values = arrays[key]
            if not values or len(values) != 12:
                return "missing"
            if any(v < 0.0 for v in values):
                return "missing"
        return "report"
    if requirement == "6.2.1.2.1":
        keys = (
            "horizontal_incident_solar_radiation",
            "north_incident_solar_radiation",
            "east_incident_solar_radiation",
            "south_incident_solar_radiation",
            "west_incident_solar_radiation",
        )
        for key in keys:
            value = _parse_float(row.get(key))
            if value is None or value < 0.0:
                return "missing"
        return "report"
    if requirement == "6.2.1.2.2":
        value = _parse_float(row.get("zone_total_transmitted_solar_radiation"))
        return "report" if value is not None and value >= 0.0 else "missing"
    if requirement == "6.2.1.2.3":
        mean = _parse_float(row.get("sky_avg_temp"))
        tmin = _parse_float(row.get("sky_min_temp"))
        tmax = _parse_float(row.get("sky_max_temp"))
        imin = _parse_float(row.get("sky_min_index_position"))
        imax = _parse_float(row.get("sky_max_index_position"))
        if None in (mean, tmin, tmax, imin, imax):
            return "missing"
        if not (0.0 <= imin <= 8759.0 and 0.0 <= imax <= 8759.0):
            return "missing"
        return "report"
    if requirement == "6.2.1.2.4":
        return "report" if _monthly_arrays_ok(row) else "missing"
    if requirement == "6.2.1.3":
        value = _parse_float(companion.get("req_6213_shaded_south_transmitted_kwh_m2"))
        return "report" if value is not None and value >= 0.0 else "missing"
    if requirement == "6.2.1.4":
        value = _parse_float(companion.get("req_6214_west_transmitted_kwh_m2"))
        return "report" if value is not None and value >= 0.0 else "missing"
    if requirement == "6.2.1.5":
        value = _parse_float(companion.get("req_6215_shaded_west_transmitted_kwh_m2"))
        return "report" if value is not None and value >= 0.0 else "missing"
    if requirement in ("6.2.1.6.min", "6.2.1.6.max", "6.2.1.6.mean"):
        if case.case_id == "960":
            # Case 960's requirement is the sun zone only (5.2.2.2.7.7.b).
            tmin = _parse_float(companion.get("sun_zone_min_temp_C"))
            tmax = _parse_float(companion.get("sun_zone_max_temp_C"))
            tmean = _parse_float(companion.get("sun_zone_mean_temp_C"))
            imin = _parse_float(companion.get("sun_zone_min_index"))
            imax = _parse_float(companion.get("sun_zone_max_index"))
            if None in (tmin, tmax, tmean, imin, imax):
                return "missing"
            if not (0.0 <= imin <= 8759.0 and 0.0 <= imax <= 8759.0):
                return "missing"
            return "report"
        tmin = _parse_float(row.get("min_temp"))
        tmax = _parse_float(row.get("max_temp"))
        tmean = _parse_float(row.get("avg_temp"))
        imin = _parse_float(row.get("min_index_position"))
        imax = _parse_float(row.get("max_index_position"))
        if None in (tmin, tmax, tmean, imin, imax):
            return "missing"
        if not (0.0 <= imin <= 8759.0 and 0.0 <= imax <= 8759.0):
            return "missing"
        return "report"
    if requirement == "6.2.1.7":
        bins = _packed_ints(row.get("temp_bins"))
        if not bins or len(bins) != 90:
            return "missing"
        if sum(bins) != 8760:
            return "missing"
        return "report"
    if requirement == "6.2.1.8":
        # The emitter produces hourly profiles for the GSR contract days.
        # Table 6-1's day cells carry unresolved tracked changes in the
        # in-repo standard, so the requirement can never be certified
        # "report" from this tree: it is "partial" when hourly data exists
        # and "missing" when it does not.
        keys = (
            "sens_htg_clg_0201",
            "sens_htg_clg_0714",
            "site_sky_temp_0201",
            "site_sky_temp_0504",
            "site_sky_temp_0714",
            "temp_0201",
            "temp_0714",
        )
        present = False
        for key in keys:
            values = _packed_numbers(row.get(key))
            if values and len(values) >= 24:
                present = True
        return "partial" if present else "missing"
    return "na"


def case_status(case: "CaseData") -> str:
    """conformant / incomplete / not implemented (no score, no verdict)."""
    if not case.ran:
        return "not implemented"
    for requirement in REQUIREMENT_IDS:
        if applies(requirement, case.case_id):
            if evaluate(requirement, case) != "report":
                return "incomplete"
    return "conformant"


# ---------------------------------------------------------------------------
# Flat report rows
# ---------------------------------------------------------------------------


def _fmt(value: float | None, places: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}"


def _fmt_factor(value: float | None, factor: float, places: int = 3) -> str:
    if value is None:
        return ""
    return f"{value * factor:.{places}f}"


def flat_values(case: CaseData) -> dict[str, str]:
    """The flat report's cells for one case, keyed by column name.

    A cell is empty when the case did not run or when fluxion produced no
    such value -- nothing is invented here.
    """
    values: dict[str, str] = {}
    if not case.ran:
        return values
    row = case.row
    companion = case.companion

    # 6.2.1.1 -- annual loads and peaks.
    values["6.2.1.1.1_annual_heating_mwh"] = row.get("annual_heating", "")
    values["6.2.1.1.2_annual_sensible_cooling_mwh"] = row.get("annual_cooling", "")
    peak_h = _parse_float(row.get("peak_heating_value"))
    peak_c = _parse_float(row.get("peak_cooling_value"))
    values["6.2.1.1.3_peak_heating_kw"] = row.get("peak_heating_value", "")
    values["6.2.1.1.3_peak_heating_time"] = (
        row.get("peak_heating_time", "") if peak_h else ""
    )
    values["6.2.1.1.4_peak_cooling_kw"] = row.get("peak_cooling_value", "")
    values["6.2.1.1.4_peak_cooling_time"] = (
        row.get("peak_cooling_time", "") if peak_c else ""
    )

    # 6.2.1.2.1 -- incident unshaded total solar per orientation.
    for label, key in (
        ("horizontal", "horizontal_incident_solar_radiation"),
        ("north", "north_incident_solar_radiation"),
        ("east", "east_incident_solar_radiation"),
        ("south", "south_incident_solar_radiation"),
        ("west", "west_incident_solar_radiation"),
    ):
        values[f"6.2.1.2.1_incident_solar_{label}_kwh_m2"] = row.get(key, "")

    # 6.2.1.2.2 -- unshaded annual transmitted solar through south windows.
    values["6.2.1.2.2_transmitted_solar_south_kwh_m2"] = row.get(
        "zone_total_transmitted_solar_radiation", ""
    )

    # 6.2.1.2.3 -- sky temperature.
    values["6.2.1.2.3_sky_mean_temp_C"] = row.get("sky_avg_temp", "")
    values["6.2.1.2.3_sky_min_temp_C"] = row.get("sky_min_temp", "")
    values["6.2.1.2.3_sky_min_index"] = row.get("sky_min_index_position", "")
    values["6.2.1.2.3_sky_max_temp_C"] = row.get("sky_max_temp", "")
    values["6.2.1.2.3_sky_max_index"] = row.get("sky_max_index_position", "")

    # 6.2.1.2.4 -- monthly loads (MWh -> kWh) and monthly peaks (W -> kW).
    arrays = _monthly_arrays(row)
    for index, month in enumerate(MONTHS):
        htg = arrays["htg_cons"][index] if arrays["htg_cons"] else None
        clg = arrays["clg_cons"][index] if arrays["clg_cons"] else None
        htg_peak = arrays["htg_peak_val"][index] if arrays["htg_peak_val"] else None
        clg_peak = arrays["clg_peak_val"][index] if arrays["clg_peak_val"] else None
        htg_time = (
            arrays["htg_peak_time"][index].strip()
            if len(arrays["htg_peak_time"]) > index
            else ""
        )
        clg_time = (
            arrays["clg_peak_time"][index].strip()
            if len(arrays["clg_peak_time"]) > index
            else ""
        )
        suffix = f"_{index + 1:02d}_{month.lower()}"
        values[f"6.2.1.2.4_monthly_heating_kwh{suffix}"] = _fmt_factor(
            htg, MONTHLY_MWH_TO_KWH
        )
        values[f"6.2.1.2.4_monthly_cooling_kwh{suffix}"] = _fmt_factor(
            clg, MONTHLY_MWH_TO_KWH
        )
        values[f"6.2.1.2.4_monthly_peak_heating_kw{suffix}"] = _fmt_factor(
            htg_peak, MONTHLY_W_TO_KW
        )
        values[f"6.2.1.2.4_monthly_peak_cooling_kw{suffix}"] = _fmt_factor(
            clg_peak, MONTHLY_W_TO_KW
        )
        # A zero monthly peak has no occurrence: the emitter writes a
        # sentinel timestamp for it.
        values[f"6.2.1.2.4_monthly_peak_heating_time{suffix}"] = (
            htg_time if htg_peak else ""
        )
        values[f"6.2.1.2.4_monthly_peak_cooling_time{suffix}"] = (
            clg_time if clg_peak else ""
        )

    # 6.2.1.3 / .4 / .5 -- case-window transmitted solar.
    values["6.2.1.3_transmitted_solar_shaded_south_kwh_m2"] = companion.get(
        "req_6213_shaded_south_transmitted_kwh_m2", ""
    )
    values["6.2.1.4_transmitted_solar_west_kwh_m2"] = companion.get(
        "req_6214_west_transmitted_kwh_m2", ""
    )
    values["6.2.1.5_transmitted_solar_shaded_west_kwh_m2"] = companion.get(
        "req_6215_shaded_west_transmitted_kwh_m2", ""
    )

    # 6.2.1.6 -- free-float zone air temperatures (Case 960: sun zone).
    if case.case_id == "960":
        values["6.2.1.6.mean_zone_air_temp_C"] = companion.get(
            "sun_zone_mean_temp_C", ""
        )
        values["6.2.1.6.min_zone_air_temp_C"] = companion.get(
            "sun_zone_min_temp_C", ""
        )
        values["6.2.1.6.min_index"] = companion.get("sun_zone_min_index", "")
        values["6.2.1.6.max_zone_air_temp_C"] = companion.get(
            "sun_zone_max_temp_C", ""
        )
        values["6.2.1.6.max_index"] = companion.get("sun_zone_max_index", "")
    else:
        values["6.2.1.6.mean_zone_air_temp_C"] = row.get("avg_temp", "")
        values["6.2.1.6.min_zone_air_temp_C"] = row.get("min_temp", "")
        values["6.2.1.6.min_index"] = row.get("min_index_position", "")
        values["6.2.1.6.max_zone_air_temp_C"] = row.get("max_temp", "")
        values["6.2.1.6.max_index"] = row.get("max_index_position", "")

    # 6.2.1.7 -- 1 degC zone air temperature bin frequencies.
    bins = _packed_ints(row.get("temp_bins"))
    for offset, lower in enumerate(TEMP_BIN_LOWER_BOUNDS):
        value = bins[offset] if bins and len(bins) == len(TEMP_BIN_LOWER_BOUNDS) else None
        values[f"6.2.1.7_zone_air_temp_bin_hours_c_{lower}"] = (
            str(value) if value is not None else ""
        )

    # 6.2.1.8 -- daily hourly output (24 hourly integrated values per day).
    # The emitter packs each series as "<day>,<label>,v1,...,v24"; the flat
    # report keeps the 24 hourly values (comma joined, one cell) and drops
    # the emitter's label elements.
    for label, source_keys in (
        ("zone_temp", (("0201", "temp_0201"), ("0714", "temp_0714"))),
        (
            "sky_temp",
            (
                ("0201", "site_sky_temp_0201"),
                ("0504", "site_sky_temp_0504"),
                ("0714", "site_sky_temp_0714"),
            ),
        ),
        (
            "load",
            (("0201", "sens_htg_clg_0201"), ("0714", "sens_htg_clg_0714")),
        ),
    ):
        for day, key in source_keys:
            series = _packed_numbers(row.get(key))
            values[f"6.2.1.8_hourly_{label}_{day}"] = (
                ",".join(_fmt(v, 4 if label == "load" else 2) for v in series)
                if series and len(series) == 24
                else ""
            )
    for day, _label in PROFILE_DAYS:
        for orientation, surface in (
            ("horizontal", "roof"),
            ("south", "south"),
            ("west", "west"),
        ):
            key = f"surf_out_inst_slr_rad_{day}_zone_surface_{surface}"
            values[f"6.2.1.8_hourly_incident_solar_{orientation}_{day}_wh_m2"] = row.get(
                key, ""
            )
        key = f"surf_win_trans_rad_{day}_zone_surface_south"
        values[f"6.2.1.8_hourly_transmitted_solar_south_{day}_wh_m2"] = row.get(key, "")
    return values


# The flat report's column order.  Built from the requirement catalog so the
# CSV and the conformance table cannot drift apart.
def flat_columns() -> list[str]:
    columns = ["case_id", "case_title", "case_status"]
    columns += [
        "6.2.1.1.1_annual_heating_mwh",
        "6.2.1.1.2_annual_sensible_cooling_mwh",
        "6.2.1.1.3_peak_heating_kw",
        "6.2.1.1.3_peak_heating_time",
        "6.2.1.1.4_peak_cooling_kw",
        "6.2.1.1.4_peak_cooling_time",
    ]
    for label in ("horizontal", "north", "east", "south", "west"):
        columns.append(f"6.2.1.2.1_incident_solar_{label}_kwh_m2")
    columns.append("6.2.1.2.2_transmitted_solar_south_kwh_m2")
    columns += [
        "6.2.1.2.3_sky_mean_temp_C",
        "6.2.1.2.3_sky_min_temp_C",
        "6.2.1.2.3_sky_min_index",
        "6.2.1.2.3_sky_max_temp_C",
        "6.2.1.2.3_sky_max_index",
    ]
    for index, month in enumerate(MONTHS):
        suffix = f"_{index + 1:02d}_{month.lower()}"
        columns += [
            f"6.2.1.2.4_monthly_heating_kwh{suffix}",
            f"6.2.1.2.4_monthly_cooling_kwh{suffix}",
            f"6.2.1.2.4_monthly_peak_heating_kw{suffix}",
            f"6.2.1.2.4_monthly_peak_cooling_kw{suffix}",
            f"6.2.1.2.4_monthly_peak_heating_time{suffix}",
            f"6.2.1.2.4_monthly_peak_cooling_time{suffix}",
        ]
    columns += [
        "6.2.1.3_transmitted_solar_shaded_south_kwh_m2",
        "6.2.1.4_transmitted_solar_west_kwh_m2",
        "6.2.1.5_transmitted_solar_shaded_west_kwh_m2",
        "6.2.1.6.mean_zone_air_temp_C",
        "6.2.1.6.min_zone_air_temp_C",
        "6.2.1.6.min_index",
        "6.2.1.6.max_zone_air_temp_C",
        "6.2.1.6.max_index",
    ]
    for lower in TEMP_BIN_LOWER_BOUNDS:
        columns.append(f"6.2.1.7_zone_air_temp_bin_hours_c_{lower}")
    for day, _label in PROFILE_DAYS:
        for orientation in ("horizontal", "south", "west"):
            columns.append(
                f"6.2.1.8_hourly_incident_solar_{orientation}_{day}_wh_m2"
            )
        columns.append(f"6.2.1.8_hourly_transmitted_solar_south_{day}_wh_m2")
    for key in ("temp_0201", "temp_0714"):
        columns.append(f"6.2.1.8_hourly_zone_temp_{key[5:]}")
    for key in ("site_sky_temp_0201", "site_sky_temp_0504", "site_sky_temp_0714"):
        columns.append(f"6.2.1.8_hourly_sky_temp_{key[-4:]}")
    for key in ("sens_htg_clg_0201", "sens_htg_clg_0714"):
        columns.append(f"6.2.1.8_hourly_load_{key[-4:]}")
    return columns


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_cases(
    results: dict[str, dict[str, str]],
    companion: dict[str, dict[str, str]],
) -> list[CaseData]:
    """Build the per-case records: every Section 7 case, run or not."""
    catalog = SECTION7_CASES + FLUXION_EXTRA_CASES
    cases: list[CaseData] = []
    for case_id, title in catalog:
        row = results.get(case_id)
        ran = row is not None
        case = CaseData(
            case_id=case_id,
            title=title,
            ran=ran,
            status="",
            row=row or {},
            companion=companion.get(case_id, {}),
            cells={},
        )
        case.cells = {
            requirement: evaluate(requirement, case) for requirement in REQUIREMENT_IDS
        }
        case.status = case_status(case)
        cases.append(case)
    return cases


def render_flat_csv(cases: list[CaseData]) -> str:
    columns = flat_columns()
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


def annex_b8_diagnostic(annex: dict, case_id: str) -> dict[str, dict[str, float]]:
    """The Annex B8 example-result spread for one case (diagnostic only)."""
    diagnostic: dict[str, dict[str, float]] = {}
    mapping = {
        "annual_heating_mwh": "annual_heating_mwh",
        "annual_sensible_cooling_mwh": "annual_sensible_cooling_mwh",
        "peak_heating_kw": "peak_heating_kw",
        "peak_sensible_cooling_kw": "peak_sensible_cooling_kw",
        "free_float_temperature_c": "free_float_temperature_c",
    }
    for group, source in mapping.items():
        table = annex.get(group)
        if not isinstance(table, dict):
            continue
        entry = table.get(case_id)
        if not isinstance(entry, dict):
            continue
        diagnostic[group] = {
            "min": entry.get("min"),
            "max": entry.get("max"),
            "mean": entry.get("mean"),
        }
    return diagnostic


def _spread(entry: dict[str, float] | None) -> str:
    if not entry:
        return ""
    low = entry.get("min")
    high = entry.get("max")
    if low is None or high is None:
        return ""
    return f"{low} to {high}"


def render_conformance_markdown(cases: list[CaseData], annex: dict) -> str:
    lines: list[str] = []
    lines.append("# Section 7 Standard Output Report -- conformance table")
    lines.append("")
    lines.append(
        "ASHRAE Standard 140 Section 7 (Building Thermal Envelope and Fabric "
        "Load) case set, evaluated against the Class I output requirements of "
        f"Section 6.2.1 of the in-repo standard (`{STANDARD_PDF}`)."
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
        "**No aggregate grade or pass/fail verdict appears in this "
        "report.** The standard defines no such quantity; each case is "
        "classified individually as conformant, incomplete or not "
        "implemented."
    )
    lines.append("")
    lines.append(
        "**Annex B8 columns are diagnostic context, not a criterion.** They "
        "are the min/max spread of the example results published in "
        "informative Annex B8, which states verbatim that those statistics "
        "'do not have any substantial importance and are not to be "
        "interpreted as acceptance criteria'. They are shown next to the "
        "reported values so a reader can see where fluxion sits relative to "
        "the published example results; no conclusion is drawn from them."
    )
    lines.append("")
    lines.append("## Reported values and conformance")
    lines.append("")
    header = ["Case", "Title", "Status"]
    header += REQUIREMENT_IDS
    header += [
        "Annex B8 annual heating (MWh)",
        "Annex B8 annual cooling (MWh)",
        "Annex B8 peak heating (kW)",
        "Annex B8 peak cooling (kW)",
        "Annex B8 free-float temperature (degC)",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for case in cases:
        diagnostic = annex_b8_diagnostic(annex, case.case_id)
        row = [case.case_id, case.title, case.status]
        row += [case.cells[requirement] for requirement in REQUIREMENT_IDS]
        row += [
            _spread(diagnostic.get("annual_heating_mwh")),
            _spread(diagnostic.get("annual_sensible_cooling_mwh")),
            _spread(diagnostic.get("peak_heating_kw")),
            _spread(diagnostic.get("peak_sensible_cooling_kw")),
            _spread(diagnostic.get("free_float_temperature_c")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in NOTES:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(
        "- `data/reference/section7/workflow_results.csv` -- Section 7 results "
        "CSV produced by the Section 7 emitter "
        "(`tests/ashrae_140_section7_gsr_emitter.rs`)."
    )
    lines.append(
        "- `data/reference/section7/section621_outputs.csv` -- the emitter's "
        "companion artifact for the Section 6.2.1.3/.4/.5 and 6.2.1.6 "
        "(Case 960 sun zone) outputs."
    )
    lines.append(
        "- `data/reference/ashrae140/annex_b8_section7_statistics.json` -- the "
        "Annex B8 example-result statistics (diagnostic context only)."
    )
    lines.append(
        "- The flat results CSV for this case set: "
        "`data/reference/section7/section7_standard_output_report.csv`."
    )
    lines.append("")
    return "\n".join(lines)


def render_conformance_json(cases: list[CaseData], annex: dict) -> str:
    payload = {
        "schema_version": "1",
        "scope": {
            "standard": "ANSI/ASHRAE Standard 140",
            "section": "Section 7 -- Building Thermal Envelope and Fabric Load",
            "case_set": "140-2017 Section 5.2 (fluxion's Section 7 case set)",
            "output_requirements": "140-2017 Section 6.2.1 (Class I)",
            "standard_pdf": STANDARD_PDF,
        },
        "states": {
            "case": ["conformant", "incomplete", "not implemented"],
            "requirement": ["report", "partial", "missing", "na"],
        },
        "sources": {
            "results_csv": "data/reference/section7/workflow_results.csv",
            "companion_csv": "data/reference/section7/section621_outputs.csv",
            "annex_b8_statistics": (
                "data/reference/ashrae140/annex_b8_section7_statistics.json"
            ),
            "flat_report_csv": (
                "data/reference/section7/section7_standard_output_report.csv"
            ),
        },
        "annex_b8_role": (
            "diagnostic context only -- NOT a criterion; Annex B8 states "
            "that its example-result statistics are not to be interpreted as "
            "acceptance criteria"
        ),
        "unit_conversions": UNIT_CONVERSIONS,
        "notes": NOTES,
        "requirements": [
            {"id": req_id, "title": title, "unit": unit}
            for req_id, title, unit in REQUIREMENTS
        ],
        "cases": [
            {
                "case": case.case_id,
                "title": case.title,
                "status": case.status,
                "ran": case.ran,
                "requirements": {
                    requirement: case.cells[requirement]
                    for requirement in REQUIREMENT_IDS
                },
                "annex_b8_diagnostic": annex_b8_diagnostic(annex, case.case_id),
            }
            for case in cases
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def generate(
    results_csv: Path,
    companion_csv: Path,
    annex_path: Path,
) -> tuple[str, str, str]:
    """Generate the three artifacts as text (pure; no writes)."""
    results = load_results(results_csv)
    companion = load_companion(companion_csv)
    annex = load_annex_b8(annex_path)
    cases = build_cases(results, companion)
    return (
        render_flat_csv(cases),
        render_conformance_markdown(cases, annex),
        render_conformance_json(cases, annex),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def regenerate_results(results_csv: Path, companion_csv: Path) -> int:
    """Re-run the Section 7 emitter so the results data is current.

    The emitter is a Rust integration test with two ignored "emit artifact"
    tests; running them rewrites the committed results CSV and the
    companion CSV from a fresh engine run.
    """
    commands = [
        [
            "cargo",
            "test",
            "--release",
            "--test",
            "ashrae_140_section7_gsr_emitter",
            "--",
            "--ignored",
            "emit_section7_gsr_csv",
            "--nocapture",
        ],
        [
            "cargo",
            "test",
            "--release",
            "--test",
            "ashrae_140_section7_gsr_emitter",
            "--",
            "--ignored",
            "emit_section621_outputs",
            "--nocapture",
        ],
    ]
    for command in commands:
        print("  $ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=REPO)
        if result.returncode != 0:
            print(
                f"emitter run failed (exit {result.returncode}); the results "
                "data was not refreshed",
                file=sys.stderr,
            )
            return result.returncode
    for path in (results_csv, companion_csv):
        if not path.exists():
            print(f"emitter did not produce {path}", file=sys.stderr)
            return 2
    return 0


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the ASHRAE 140 Section 7 Standard Output Report "
            "(flat results CSV + conformance table)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_CSV,
        help="Section 7 results CSV produced by the Section 7 emitter",
    )
    parser.add_argument(
        "--companion-csv",
        type=Path,
        default=COMPANION_CSV,
        help="Section 6.2.1 companion CSV produced by the Section 7 emitter",
    )
    parser.add_argument(
        "--annex-b8",
        type=Path,
        default=ANNEX_B8,
        help="Annex B8 example-result statistics (diagnostic context only)",
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
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="re-run the Section 7 emitter first so the results data is current",
    )
    args = parser.parse_args(argv)

    if args.regenerate:
        print("Regenerating the Section 7 results data ...")
        rc = regenerate_results(args.results_csv, args.companion_csv)
        if rc != 0:
            return 2

    try:
        csv_text, md_text, json_text = generate(
            args.results_csv, args.companion_csv, args.annex_b8
        )
    except (FileNotFoundError, ValueError) as exc:
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
                "generate_section7_standard_output_report.py"
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
