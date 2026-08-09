#!/usr/bin/env python3
"""
Issue #2453 / #2448 seasonal-attribution analyser.

Runs the Issue #2453 per-month diagnostic test
``tests/case_900_series_seasonal_attribution.rs`` and prints a per-month
deviation table against the ASHRAE 140 monthly reference CSV
``tests/reference_data/ashrae140/monthly/case_900_monthly_reference.csv``.

The per-case reference (Cases 910/920/930/940) is approximated using the same
degree-day monthly fraction as Case 900, scaled to the case-specific annual
midpoint. This is the same interim methodology used in
``case_900_monthly_reference.csv`` (see that file's header) and is the only
reference available until the Case 900 hourly EnergyPlus CSV is regenerated
(the ``tests/reference_data/energyplus_models/ashrae_140_case_900.idf`` file
currently fails EnergyPlus 25.2 validation per issue #1331 follow-up).

Usage
-----

From the repository root::

    # Step 1: Run the diagnostic test, capture stdout to a file
    cargo test --release -p fluxion \\
        --test case_900_series_seasonal_attribution \\
        test_case_900_series_seasonal_attribution \\
        -- --ignored --nocapture \\
        2>/dev/null | grep -E '^\\[#2453|^  Case|^  Month|^    ' \\
        > /tmp/issue-2453-attribution.txt

    # Step 2: Run the analyser
    python3 scripts/issue-2448-seasonal-attribution.py \\
        --input /tmp/issue-2453-attribution.txt

The analyser prints:

  - For each of Cases 900, 910, 920, 930, 940: a per-month table of
    (Q_solar, Q_internal, Q_conduction, Q_infiltration, H, C) plus the
    deviation of H and C against the monthly reference (midpoint of the
    CDD/HDD-scaled band).

  - The seasonal-direction summary: in which months the over-prediction is
    heating-dominated vs cooling-dominated, and the magnitude of the worst
    month.

Why this exists
---------------
The bidirectional over-prediction (annual H AND annual C both above band)
is the textbook signature of solar mass-node over-injection on a long
integration horizon (KNOWN_ISSUES.md §LIMIT-05). The diagnostic test
localises the over-injection to a specific season and term. The Python
analyser turns the test's per-month tables into a per-month deviation view
that can be diffed across PRs to verify the eventual GaugeSolver fix
(#1465 / #1462) closes the gap.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Annual ASHRAE 140 reference (inter-program min/max, midpoint) for each
# high-mass HVAC case. Source: NREL/TP-472-6231 (1995 BESTEST) Table 3-2,
# reconciled in tests/reference_data/zone_balance/PROVENANCE.md (issue #1408).
ANNUAL_REF = {
    "900": {
        "h_min": 1.17, "h_max": 2.04, "c_min": 2.13, "c_max": 3.67,
    },
    "910": {
        "h_min": 1.51, "h_max": 2.28, "c_min": 0.82, "c_max": 1.88,
    },
    "920": {
        "h_min": 3.26, "h_max": 4.30, "c_min": 1.84, "c_max": 3.31,
    },
    "930": {
        "h_min": 4.14, "h_max": 5.34, "c_min": 1.04, "c_max": 2.24,
    },
    "940": {
        "h_min": 0.79, "h_max": 1.41, "c_min": 2.08, "c_max": 3.55,
    },
}

# Same HDD/CDD monthly fractions as the interim
# case_900_monthly_reference.csv (degree-day share, sums to 1.0; reproduces
# the same per-month shape for Cases 910-940 because the degree-day
# distribution is dominated by outdoor temperature, not by the building
# being conditioned).
HEATING_FRACTION = [
    0.14960,  # Jan
    0.15420,  # Feb
    0.13277,  # Mar
    0.08598,  # Apr
    0.05532,  # May
    0.01882,  # Jun
    0.01103,  # Jul
    0.00679,  # Aug
    0.02779,  # Sep
    0.08062,  # Oct
    0.13289,  # Nov
    0.14418,  # Dec
]
COOLING_FRACTION = [
    0.00000,  # Jan
    0.00000,  # Feb
    0.00648,  # Mar
    0.02000,  # Apr
    0.08131,  # May
    0.18779,  # Jun
    0.26421,  # Jul
    0.28090,  # Aug
    0.13569,  # Sep
    0.02159,  # Oct
    0.00190,  # Nov
    0.00010,  # Dec
]

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@dataclass
class MonthlyRow:
    h_kwh: float = 0.0
    c_kwh: float = 0.0
    solar_kwh: float = 0.0
    internal_kwh: float = 0.0
    conduction_kwh: float = 0.0
    infiltration_kwh: float = 0.0


@dataclass
class CaseAttribution:
    case_id: str
    annual_h_mwh: float
    annual_c_mwh: float
    annual_solar_mwh: float
    annual_internal_mwh: float
    annual_conduction_mwh: float
    annual_infiltration_mwh: float
    monthly: list[MonthlyRow] = field(default_factory=lambda: [MonthlyRow() for _ in range(12)])


HEADER_RE = re.compile(
    r"^\[#2453 Case (?P<case_id>\d+)\] annual H=(?P<h_mwh>[\d.]+) MWh "
    r"\(ref \[(?P<h_min>[\d.]+), (?P<h_max>[\d.]+)\], mid (?P<h_mid>[\d.]+)\) "
    r"\| C=(?P<c_mwh>[\d.]+) MWh \(ref \[(?P<c_min>[\d.]+), (?P<c_max>[\d.]+)\], "
    r"mid (?P<c_mid>[\d.]+)\)$"
)

ANNUAL_TOTALS_RE = re.compile(
    r"^    annual solar=(?P<s>[\d.]+) MWh, internal=(?P<i>[\d.]+) MWh, "
    r"conduction=(?P<c>[\-\d.]+) MWh, infiltration=(?P<f>[\d.]+) MWh$"
)

MONTH_ROW_RE = re.compile(
    r"^    (?P<month>\w{3})\s+\|\s+(?P<h>[\-\d.]+)\s+(?P<c>[\-\d.]+)\s+\|\s+"
    r"(?P<sol>[\-\d.]+)\s+(?P<int>[\-\d.]+)\s+(?P<cond>[\-\d.]+)\s+(?P<inf>[\-\d.]+)$"
)


def parse_input(text: str) -> list[CaseAttribution]:
    """Parse the diagnostic test's per-month table output."""
    cases: list[CaseAttribution] = []
    current: Optional[CaseAttribution] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = HEADER_RE.match(line)
        if m:
            current = CaseAttribution(
                case_id=m.group("case_id"),
                annual_h_mwh=float(m.group("h_mwh")),
                annual_c_mwh=float(m.group("c_mwh")),
                annual_solar_mwh=0.0,
                annual_internal_mwh=0.0,
                annual_conduction_mwh=0.0,
                annual_infiltration_mwh=0.0,
            )
            cases.append(current)
            continue
        if current is None:
            continue
        m = ANNUAL_TOTALS_RE.match(line)
        if m:
            current.annual_solar_mwh = float(m.group("s"))
            current.annual_internal_mwh = float(m.group("i"))
            current.annual_conduction_mwh = float(m.group("c"))
            current.annual_infiltration_mwh = float(m.group("f"))
            continue
        m = MONTH_ROW_RE.match(line)
        if m:
            label = m.group("month")
            try:
                idx = MONTH_LABELS.index(label)
            except ValueError:
                continue
            current.monthly[idx] = MonthlyRow(
                h_kwh=float(m.group("h")),
                c_kwh=float(m.group("c")),
                solar_kwh=float(m.group("sol")),
                internal_kwh=float(m.group("int")),
                conduction_kwh=float(m.group("cond")),
                infiltration_kwh=float(m.group("inf")),
            )
    return cases


def expected_monthly_midpoint(case_id: str) -> tuple[list[float], list[float]]:
    """Return the (H, C) per-month midpoints derived from the case annual
    midpoint scaled by the HDD/CDD monthly fraction."""
    if case_id not in ANNUAL_REF:
        return [0.0] * 12, [0.0] * 12
    ref = ANNUAL_REF[case_id]
    h_mid = (ref["h_min"] + ref["h_max"]) / 2.0
    c_mid = (ref["c_min"] + ref["c_max"]) / 2.0
    h = [h_mid * f for f in HEATING_FRACTION]
    c = [c_mid * f for f in COOLING_FRACTION]
    return h, c


def direction_label(engine_h: float, engine_c: float, ref_h: float, ref_c: float) -> str:
    """Describe which way the per-month over-prediction goes."""
    if engine_h <= 0.001 and engine_c <= 0.001:
        return "(off)"
    if engine_h > engine_c and engine_h > ref_h * 1.05:
        return "H-ovr"
    if engine_c > engine_h and engine_c > ref_c * 1.05:
        return "C-ovr"
    if engine_h > ref_h * 1.05 and engine_c > ref_c * 1.05:
        return "H+C-ovr"
    if engine_h < ref_h * 0.95:
        return "H-und"
    if engine_c < ref_c * 0.95:
        return "C-und"
    return "(in band)"


def print_case(case: CaseAttribution) -> None:
    ref_h_mid, ref_c_mid = expected_monthly_midpoint(case.case_id)
    ref = ANNUAL_REF[case.case_id]
    print(f"\n  Case {case.case_id} (H ref {ref['h_min']}-{ref['h_max']} MWh, "
          f"C ref {ref['c_min']}-{ref['c_max']} MWh)")
    print(f"    Engine annual: H={case.annual_h_mwh:6.3f} MWh  "
          f"C={case.annual_c_mwh:6.3f} MWh  "
          f"ΔH={(case.annual_h_mwh - ref['h_min']) / max(ref['h_min'], 0.01) * 100:+6.1f}%  "
          f"ΔC={(case.annual_c_mwh - ref['c_min']) / max(ref['c_min'], 0.01) * 100:+6.1f}%")
    print(f"    Engine annual breakdown: solar={case.annual_solar_mwh:.2f} MWh  "
          f"internal={case.annual_internal_mwh:.2f} MWh  "
          f"conduction={case.annual_conduction_mwh:.2f} MWh  "
          f"infiltration={case.annual_infiltration_mwh:.2f} MWh")
    print("    Month | H_eng H_ref   C_eng C_ref | direction | "
          "Q_sol  Q_int  Q_cond  Q_inf  (kWh)")
    for m in range(12):
        row = case.monthly[m]
        h_ref_mwh = ref_h_mid[m]
        c_ref_mwh = ref_c_mid[m]
        h_ref_kwh = h_ref_mwh * 1000.0
        c_ref_kwh = c_ref_mwh * 1000.0
        direction = direction_label(row.h_kwh, row.c_kwh, h_ref_kwh, c_ref_kwh)
        print(
            f"      {MONTH_LABELS[m]}  | {row.h_kwh:6.1f} {h_ref_kwh:6.1f}  "
            f"{row.c_kwh:6.1f} {c_ref_kwh:6.1f} | {direction:9s} | "
            f"{row.solar_kwh:6.1f}  {row.internal_kwh:6.1f}  "
            f"{row.conduction_kwh:7.1f}  {row.infiltration_kwh:6.1f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the captured test stdout (lines starting with [#2453, Case, Month or 4-space indent)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line per-case summary at the end (useful for CI)",
    )
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2
    text = args.input.read_text(encoding="utf-8")
    cases = parse_input(text)
    if not cases:
        print("ERROR: no [#2453 Case ...] header lines found in input", file=sys.stderr)
        return 1
    print("=" * 78)
    print("  Issue #2453 / #2448 per-month seasonal attribution")
    print("  (CTF solver path; matches the production validator output)")
    print("=" * 78)
    for case in cases:
        print_case(case)
    if args.summary:
        print("\n#2453 Summary: 5-case annual over-prediction (CTF path)")
        for case in cases:
            ref = ANNUAL_REF[case.case_id]
            h_mid = (ref["h_min"] + ref["h_max"]) / 2.0
            c_mid = (ref["c_min"] + ref["c_max"]) / 2.0
            d_h = (case.annual_h_mwh - h_mid) / h_mid * 100.0
            d_c = (case.annual_c_mwh - c_mid) / c_mid * 100.0
            tag = "BIDIR" if case.annual_h_mwh > ref["h_max"] and case.annual_c_mwh > ref["c_max"] else ""
            print(
                f"  Case {case.case_id}: H={case.annual_h_mwh:.2f} MWh ({d_h:+.0f}%)  "
                f"C={case.annual_c_mwh:.2f} MWh ({d_c:+.0f}%)  {tag}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
