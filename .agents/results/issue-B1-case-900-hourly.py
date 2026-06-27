#!/usr/bin/env python3
"""
Issue #1329 (B1) — Python verification artefact for the ASHRAE 140 Case 900
IDF + per-surface roof-solar hourly reference CSV.

Per the issue scope: "Save Python verification artifact at
.agents/results/issue-B1-case-900-hourly.py".

This script re-loads the two deliverables produced by
``tools/generate_case_900_idf.py`` and prints:
  * row/column counts (8760 / 10)
  * min/max/mean per column
  * physical sanity envelope vs ASHRAE clear-sky (max total <= 1100 W/m2,
    max beam <= 1050 W/m2, ground == 0 always for a horizontal surface)
  * formula correctness: beam = DNI * max(cos(zenith), 0),
                         sky_diffuse = DHI,
                         ground_diffuse = 0,
                         total = beam + sky_diffuse + ground_diffuse
  * annual energy totals on the roof (kWh/m2/yr, MWh/yr)
  * cross-check vs the existing south-wall reference
    (surface_irradiance_south.csv — roof should receive more beam than
    the south wall because the horizontal surface sees beam whenever the
    sun is above the horizon, not just when its azimuth is in the south
    90 deg half-plane).

Run from repo root::

    python3 .agents/results/issue-B1-case-900-hourly.py
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOF_CSV = REPO / "tests/reference_data/solar/case_900_roof_solar_hourly.csv"
SOUTH_CSV = REPO / "tests/reference_data/solar/surface_irradiance_south.csv"
IDF_PATH = REPO / "tests/reference_data/energyplus_models/ashrae_140_case_900.idf"

EXPECTED_ROWS = 8760
EXPECTED_HEADER_TOKENS = [
    "hour(1-8760)",
    "beam_irradiance(W/m2)",
    "sky_diffuse_irradiance(W/m2)",
    "ground_diffuse_irradiance(W/m2)",
    "total_irradiance(W/m2)",
    "solar_zenith(deg)",
    "solar_altitude(deg)",
    "dni_wm2",
    "dhi_wm2",
    "ghi_wm2",
]
ROOF_AREA_M2 = 6.0 * 8.0  # 48 m2 (ASHRAE 140 Case 600/900 box)


def _strip_metadata(rows: list[list[str]]) -> list[list[str]]:
    return [r for r in rows if r and not r[0].lstrip().startswith("#")]


def load_roof() -> tuple[list[str], list[list[str]]]:
    with open(ROOF_CSV) as f:
        rows = list(csv.reader(f))
    rows = _strip_metadata(rows)
    header, data = rows[0], rows[1:]
    return header, data


def load_south() -> tuple[list[str], list[list[str]]]:
    with open(SOUTH_CSV) as f:
        rows = list(csv.reader(f))
    rows = _strip_metadata(rows)
    return rows[0], rows[1:]


def col(data: list[list[str]], header: list[str], name: str) -> list[float]:
    return [float(r[header.index(name)]) for r in data]


def main() -> int:
    print("=" * 70)
    print(" Issue #1329 (B1) — Case 900 IDF + Roof-Solar CSV verification")
    print("=" * 70)
    print()

    # 1. File presence
    print("[1] Artefact presence")
    print(f"    IDF    : {IDF_PATH}    exists={IDF_PATH.exists()}    "
          f"size={IDF_PATH.stat().st_size if IDF_PATH.exists() else 0} B")
    print(f"    CSV    : {ROOF_CSV}    exists={ROOF_CSV.exists()}    "
          f"size={ROOF_CSV.stat().st_size if ROOF_CSV.exists() else 0} B")
    print()

    # 2. CSV row/column shape
    print("[2] CSV shape")
    header, data = load_roof()
    print(f"    header columns: {len(header)}  (expected {len(EXPECTED_HEADER_TOKENS)})")
    print(f"    data rows    : {len(data)}     (expected {EXPECTED_ROWS})")
    header_ok = header == EXPECTED_HEADER_TOKENS
    print(f"    header matches expected schema: {header_ok}")
    rows_ok = len(data) == EXPECTED_ROWS
    print(f"    row count == 8760              : {rows_ok}")
    print()

    # 3. Per-column stats
    print("[3] Per-column stats")
    beam = col(data, header, "beam_irradiance(W/m2)")
    sky = col(data, header, "sky_diffuse_irradiance(W/m2)")
    ground = col(data, header, "ground_diffuse_irradiance(W/m2)")
    total = col(data, header, "total_irradiance(W/m2)")
    zenith = col(data, header, "solar_zenith(deg)")
    altitude = col(data, header, "solar_altitude(deg)")
    dni = col(data, header, "dni_wm2")
    dhi = col(data, header, "dhi_wm2")
    ghi = col(data, header, "ghi_wm2")
    for name, vals in [
        ("beam", beam),
        ("sky_diffuse", sky),
        ("ground", ground),
        ("total", total),
        ("zenith (deg)", zenith),
        ("altitude (deg)", altitude),
        ("DNI", dni),
        ("DHI", dhi),
        ("GHI", ghi),
    ]:
        print(
            f"    {name:14s} min={min(vals):9.2f} "
            f"max={max(vals):9.2f} mean={statistics.mean(vals):9.2f}"
        )
    print()

    # 4. Physical envelope (clear-sky plausibility)
    print("[4] Physical envelope (clear-sky plausibility)")
    print(f"    max total irradiance   : {max(total):.2f} W/m2 "
          "(envelope < 1367 extraterr, < 1100 Colorado clear-sky)")
    print(f"    max beam irradiance    : {max(beam):.2f} W/m2 "
          "(envelope < 1050)")
    print(f"    max sky diffuse        : {max(sky):.2f} W/m2 "
          "(envelope < 600)")
    print(f"    ground == 0 everywhere : {all(g == 0 for g in ground)}")
    print()

    # 5. Formula correctness
    print("[5] Formula correctness (spec-derived, not from E+ output)")
    max_err_beam = 0.0
    for b, d, z in zip(beam, dni, zenith):
        expected = 0.0 if (d <= 0 or z >= 90) else d * math.cos(math.radians(z))
        err = abs(b - expected)
        if err > max_err_beam:
            max_err_beam = err
    ok_beam = max_err_beam < 0.01
    print(
        f"    beam == DNI*max(cos(zenith), 0) within 0.01 W/m2 : "
        f"{ok_beam} (max err {max_err_beam:.6f} W/m2)"
    )

    max_err_sky = max(abs(s - d) for s, d in zip(sky, dhi))
    ok_sky = max_err_sky < 0.01
    print(
        f"    sky_diffuse == DHI within 0.01 W/m2              : "
        f"{ok_sky} (max err {max_err_sky:.6f} W/m2)"
    )

    max_err_total = max(abs(b + s + g - t) for b, s, g, t in zip(beam, sky, ground, total))
    ok_total = max_err_total < 0.01
    print(
        f"    total == beam + sky + ground within 0.01 W/m2    : "
        f"{ok_total} (max err {max_err_total:.6f} W/m2)"
    )
    print()

    # 6. Annual energy totals
    print("[6] Annual energy on the roof")
    annual_kwh_m2_total = sum(total) / 1000.0
    annual_kwh_m2_beam = sum(beam) / 1000.0
    annual_kwh_m2_sky = sum(sky) / 1000.0
    annual_mwh_roof = annual_kwh_m2_total * ROOF_AREA_M2 / 1000.0
    print(f"    annual total irradiance : {annual_kwh_m2_total:8.2f} kWh/m2/yr")
    print(f"    annual beam irradiance  : {annual_kwh_m2_beam:8.2f} kWh/m2/yr")
    print(f"    annual sky diffuse      : {annual_kwh_m2_sky:8.2f} kWh/m2/yr")
    print(
        f"    annual roof energy      : {annual_mwh_roof:8.2f} MWh/yr  "
        f"(roof area = {ROOF_AREA_M2} m2)"
    )
    print()

    # 7. Cross-check vs existing south-wall reference
    print("[7] Cross-check vs south-wall reference (surface_irradiance_south.csv)")
    sw_h, sw_d = load_south()
    sw_beam_annual = sum(float(r[sw_h.index("beam_irradiance(W/m2)")]) for r in sw_d) / 1000.0
    sw_ground_annual = sum(
        float(r[sw_h.index("ground_diffuse_irradiance(W/m2)")]) for r in sw_d
    ) / 1000.0
    print(f"    south wall annual beam        : {sw_beam_annual:8.2f} kWh/m2/yr")
    print(f"    south wall annual ground diff : {sw_ground_annual:8.2f} kWh/m2/yr")
    print(
        f"    roof annual beam             : {annual_kwh_m2_beam:8.2f} kWh/m2/yr  "
        "(expect > south wall: roof sees beam whenever sun is up)"
    )
    print()

    # 8. Acceptance summary
    print("[8] Acceptance summary")
    checks = {
        "row count == 8760": rows_ok,
        "header matches schema": header_ok,
        "max total < 1100 W/m2 (clear-sky envelope)": max(total) < 1100,
        "max beam < 1050 W/m2 (clear-sky envelope)": max(beam) < 1050,
        "ground_reflected == 0 for horizontal surface": all(g == 0 for g in ground),
        "beam formula = DNI*cos(zenith) within 0.01": ok_beam,
        "sky_diffuse == DHI within 0.01": ok_sky,
        "total = beam + sky + ground within 0.01": ok_total,
        "roof beam > south wall beam (sanity)": annual_kwh_m2_beam > sw_beam_annual,
    }
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    all_pass = all(checks.values())
    print()
    print(f"    overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())