#!/usr/bin/env python3
"""Generate ASHRAE 140 Case 800 / 810 EnergyPlus reference data for Fluxion #2953.

Produces four CSVs in ``tests/reference_data/zone_balance/``:

1. ``case_800_energy_hourly.csv`` — Hourly time series for Case 800
   (single-stage heat pump, light-mass single-story). 8760 rows.
2. ``case_810_energy_hourly.csv`` — Hourly time series for Case 810
   (comprehensive HVAC, high-mass commercial). 8760 rows.

3. ``case_800_energy_reference.csv`` — Annual + peak energy summary reference
   (4 metric rows: annual_heating, annual_cooling, peak_heating, peak_cooling)
   in the schema of ``case_600/900/920/950/960_energy_reference.csv``.
4. ``case_810_energy_reference.csv`` — Same schema, Case 810.

All four are driven from the same EPW used to regenerate all other reference
data (USA_CO_Golden-NREL.724666_TMY3.epw, see ARCHITECTURE.md §Reference Data).

ASHRAE 140 spec for Cases 800 / 810
-----------------------------------
ASHRAE Standard 140-2023 §5.2 (HVAC equipment validation) extends the base
BESTEST cases with HVAC sub-cases. Fluxion's implementation
(``src/validation/ashrae140/cases/series_800.rs``) maps the spec as:

* **Case 800** — light-mass single-story (232 m², lightweight construction)
  with a single-stage air-source heat pump (12 kW heating / 10 kW cooling,
  COP 3.2 / 3.5). Heating setpoint 20°C, cooling 24°C.
* **Case 810** — high-mass commercial (1500 m², high-mass construction) with a
  comprehensive HVAC plant (120 kW heating / 150 kW cooling, COP 4.0 / 6.0,
  75% heat recovery, VAV, economizer). Heating setpoint 21°C, cooling 24°C.

The geometry is shared with Case 600 (Case 800) and Case 900 (Case 810)
respectively — Cases 800/810 add the HVAC equipment layer on top of the
base BESTEST envelope + infiltration model.

Usage
-----
Requires EnergyPlus >= 24.1.0 installed at /usr/local/EnergyPlus-25-2-0/
(or edit EP_PATH below). From the repository root::

    python3 tests/reference_data/zone_balance/generate_case_800_810_energy.py

The script will:

1. Verify each case's EnergyPlus IDF exists at the expected path.
   Cases 800/810 IDFs are follow-up work (#2953 follow-up); the script will
   skip any case whose IDF is missing and print a clear warning.
2. Run EnergyPlus against each available IDF.
3. Extract hourly variables from ``eplusout.sql`` (Zone Mean Air Temperature,
   Site Outdoor Air Drybulb Temperature, Zone Air System Sensible Heating
   Energy, Zone Air System Sensible Cooling Energy).
4. Write the hourly CSV with 8760 rows + 5 metadata comment lines + header.
5. Compute annual heating, annual cooling, peak heating, peak cooling from
   the hourly series and write the summary reference CSV in the standard
   schema.

CSV format
----------
The hourly CSV (per zone, 8760 rows)::

    # EnergyPlus Version: 25.2.0
    # Case 800: ASHRAE 140 §5.2 light-mass single-story + single-stage heat pump
    # EPW: USA_CO_Golden-NREL.724666_TMY3.epw
    # Generated: <UTC ISO 8601>
    # Columns: hour(1-8760), T_zone(C), T_out(C), Q_heat(W), Q_cool(W)
    hour,T_zone(C),T_out(C),Q_heat(W),Q_cool(W)
    1,20.00,-3.00,235.41,0.00
    ...

The summary CSV mirrors ``case_600_energy_reference.csv``::

    # EnergyPlus Reference: ASHRAE 140 Case 800 — Annual + Peak Energy
    # Source: ASHRAE Standard 140-2023 §5.2 (HVAC equipment validation)
    # ...
    metric,unit,ref_min,ref_max,ref_midpoint,tolerance_pct,accept_min,accept_max,notes
    annual_heating,MWh,...,...,±15%,...,...,Single-stage heat pump (ref x-y MWh)
    annual_cooling,MWh,...,...,±15%,...,...,Single-stage heat pump (ref x-y MWh)
    peak_heating,kW,...,...,±15%,...,...,Winter peak after cold clear night
    peak_cooling,kW,...,...,±15%,...,...,Summer peak at solar noon

References
----------
* ASHRAE Standard 140-2023 §5.2 (HVAC equipment validation cases)
* ASHRAE Standard 140-2023 Annex B (BESTEST base cases — Case 600 / 900
  geometry shared with Case 800 / 810)
* fluxion issue #2953 (script creation; CSV outputs are follow-up)
* fluxion issue #1331 (parent issue: regenerate Case 800 / 810 reference data)
* fluxion issue #1168 (linked: reference data regeneration tracker)
* ``src/validation/ashrae140/cases/series_800.rs`` — fluxion's Case 800/810
  spec implementation (heat-pump + comprehensive HVAC parameters)
"""

import csv
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EP_PATH = Path("/usr/local/EnergyPlus-25-2-0/energyplus")
EPW = Path(
    "/usr/local/EnergyPlus-25-2-0/WeatherData/USA_CO_Golden-NREL.724666_TMY3.epw"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
IDF_DIR = REPO_ROOT / "tests/reference_data/energyplus_models"
OUTPUT_DIR = REPO_ROOT / "tests/reference_data/zone_balance"

# Case definitions: (case_id, idf_path, hourly_csv_path, geometry_label)
# Geometry labels mirror Case 600 (lightweight) and Case 900 (high-mass) — the
# fluxion Case 800 / 810 specs share those envelopes and add HVAC equipment.
CASES = [
    (
        "800",
        IDF_DIR / "ashrae_140_case_800.idf",
        OUTPUT_DIR / "case_800_energy_hourly.csv",
        "light-mass single-story + single-stage heat pump",
    ),
    (
        "810",
        IDF_DIR / "ashrae_140_case_810.idf",
        OUTPUT_DIR / "case_810_energy_hourly.csv",
        "high-mass commercial + comprehensive HVAC plant",
    ),
]

TOLERANCE_PCT = 15  # ±15% annual energy per ASHRAE 140 acceptance criteria (#1147)

# Conservation target: each case's annual heating + annual cooling must be
# self-consistent. EnergyPlus is the authority; if a future regeneration
# moves the numbers outside this band it indicates a regression in either the
# envelope (Case 600/900 shared geometry) or the HVAC equipment (Case 800/810
# novel layer). Bands below are placeholder envelopes pending first E+ run.
CONSERVATION_TARGET_MWH = (0.5, 50.0)


def run_energyplus(idf_path: Path, work_dir: Path) -> bool:
    """Run EnergyPlus against the given IDF, return True on success."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EP_PATH),
        "-w", str(EPW),
        "-d", str(work_dir),
        "-p", "eplus",
        str(idf_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"EnergyPlus failed for {idf_path.name}")
        print("STDOUT (tail):", result.stdout[-2000:])
        print("STDERR (tail):", result.stderr[-1000:])
        return False
    return True


def extract_hourly(sql_path: Path) -> list[dict]:
    """Extract hourly T_zone, T_out, Q_heat, Q_cool from eplusout.sql.

    E+ 25.x emits two Time rows per simulated hour (one warmup summary row
    with IntervalType=-1, one regular IntervalType=1). For each hour h, the
    canonical TimeIndex for zone/outdoor temperature is odd (1, 3, 5, ...)
    and the corresponding hourly heating/cooling energy TI is +1 (even).
    """
    conn = sqlite3.connect(sql_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows: list[dict] = []
    try:
        def get_series(var_name: str, key_value: str | None = None) -> dict[int, float]:
            """Get hourly time series for a variable (key_value optional)."""
            q = (
                "SELECT r.TimeIndex, r.Value "
                "FROM ReportData r "
                "JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex "
                "WHERE d.Name = ?"
            )
            params: list = [var_name]
            if key_value:
                q += " AND d.KeyValue = ?"
                params.append(key_value)
            out: dict[int, float] = {}
            for r in cur.execute(q, params):
                out[r["TimeIndex"]] = r["Value"]
            return out

        t_out = get_series("Site Outdoor Air Drybulb Temperature", "Environment")
        # Verify we got 8760 outdoor readings on odd TI.
        odd_indices = sorted(ti for ti in t_out.keys() if ti % 2 == 1)
        n_hours = len(odd_indices)
        if n_hours != 8760:
            print(
                f"WARNING: expected 8760 odd-TI outdoor readings, got {n_hours}"
            )

        t_zone = get_series("Zone Mean Air Temperature", "ZONE1")
        q_heat = get_series("Zone Air System Sensible Heating Energy", "ZONE1")
        q_cool = get_series("Zone Air System Sensible Cooling Energy", "ZONE1")

        for h, idx_t in enumerate(odd_indices, start=1):
            idx_h = idx_t + 1  # heating/cooling TI is +1 from outdoor/zone TI
            rows.append(
                {
                    "hour": h,
                    "T_zone(C)": t_zone.get(idx_t, 0.0),
                    "T_out(C)": t_out.get(idx_t, 0.0),
                    "Q_heat(W)": q_heat.get(idx_h, 0.0) / 3600.0,
                    "Q_cool(W)": q_cool.get(idx_h, 0.0) / 3600.0,
                }
            )
    finally:
        conn.close()
    return rows


def write_hourly_csv(
    rows: list[dict], output_path: Path, case_id: str, geometry_label: str
) -> None:
    """Write the hourly time-series CSV for a case."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "# EnergyPlus Version: 25.2.0",
                f"# Case {case_id}: ASHRAE 140 §5.2 {geometry_label}",
                f"# EPW: {EPW.name}",
                f"# Generated: {datetime.now(timezone.utc).isoformat()}",
                "# Columns: hour(1-8760), T_zone(C), T_out(C), "
                "Q_heat(W), Q_cool(W) [hourly mean power]",
            ]
        )
        w.writerow(
            ["hour", "T_zone(C)", "T_out(C)", "Q_heat(W)", "Q_cool(W)"]
        )
        for r in rows:
            w.writerow(
                [
                    r["hour"],
                    f"{r['T_zone(C)']:.4f}",
                    f"{r['T_out(C)']:.4f}",
                    f"{r['Q_heat(W)']:.4f}",
                    f"{r['Q_cool(W)']:.4f}",
                ]
            )
    print(f"Wrote {output_path} ({len(rows)} rows)")


def compute_summary(rows: list[dict]) -> dict:
    """Compute annual heating, cooling, and peak values from hourly rows."""
    annual_heat_kwh = sum(r["Q_heat(W)"] for r in rows) / 1000.0  # Wh → kWh
    annual_cool_kwh = sum(r["Q_cool(W)"] for r in rows) / 1000.0
    peak_heat_kw = max(r["Q_heat(W)"] for r in rows) / 1000.0  # W → kW
    peak_cool_kw = max(r["Q_cool(W)"] for r in rows) / 1000.0
    return {
        "annual_heating_MWh": annual_heat_kwh / 1000.0,
        "annual_cooling_MWh": annual_cool_kwh / 1000.0,
        "peak_heating_kW": peak_heat_kw,
        "peak_cooling_kW": peak_cool_kw,
        "total_rows": len(rows),
    }


def write_reference_summary(
    case_id: str,
    geometry_label: str,
    summary: dict,
    notes: dict[str, str],
) -> Path:
    """Write summary reference CSV matching case_600/900_energy_reference.csv schema.

    The summary band (ref_min/ref_max) is taken from the EnergyPlus run with
    a ±15% envelope (matching the acceptance criterion in the existing
    reference CSVs and ASHRAE 140-2023 §5.2 inter-program guidance). The
    band width is *narrower* than a multi-engine inter-program band because
    Case 800/810 are HVAC-equipment-only variants of the published
    Case 600/900 reference (the envelope + infiltration model is shared, so
    only the HVAC layer introduces variance).
    """
    path = OUTPUT_DIR / f"case_{case_id}_energy_reference.csv"

    # Derive reference bands from the EnergyPlus run with ±TOLERANCE_PCT on the
    # midpoint. The midpoint equals the EnergyPlus value itself; the band is
    # the ±15% acceptance envelope that ``zone_balance_eplus_isolation.rs``
    # uses for Case 600 / 900 / 920 / 950 / 960.
    metrics = []
    for metric, unit, ref_mid, note in [
        (
            "annual_heating", "MWh",
            summary["annual_heating_MWh"],
            notes["annual_heating"],
        ),
        (
            "annual_cooling", "MWh",
            summary["annual_cooling_MWh"],
            notes["annual_cooling"],
        ),
        (
            "peak_heating", "kW",
            summary["peak_heating_kW"],
            notes["peak_heating"],
        ),
        (
            "peak_cooling", "kW",
            summary["peak_cooling_kW"],
            notes["peak_cooling"],
        ),
    ]:
        ref_min = ref_mid * (1.0 - TOLERANCE_PCT / 100.0)
        ref_max = ref_mid * (1.0 + TOLERANCE_PCT / 100.0)
        accept_min = ref_min
        accept_max = ref_max
        metrics.append(
            (metric, unit, ref_min, ref_max, ref_mid, TOLERANCE_PCT,
             accept_min, accept_max, note)
        )

    # Sanity-check the conservation target. If heating + cooling leaves the
    # band, the IDF is probably mis-configured (e.g. setpoints swapped) and
    # the engineer needs to inspect the run.
    total_mwh = summary["annual_heating_MWh"] + summary["annual_cooling_MWh"]
    if not (
        CONSERVATION_TARGET_MWH[0] <= total_mwh <= CONSERVATION_TARGET_MWH[1]
    ):
        print(
            f"WARNING: Case {case_id} annual total {total_mwh:.3f} MWh "
            f"outside expected conservation band "
            f"{CONSERVATION_TARGET_MWH} MWh — review IDF / E+ run output."
        )

    with open(path, "w", newline="") as f:
        f.write(
            f"# EnergyPlus Reference: ASHRAE 140 Case {case_id} — "
            "Annual + Peak Energy\n"
        )
        f.write(
            f"# Source: ASHRAE Standard 140-2023 §5.2 (HVAC equipment "
            f"validation) — {geometry_label}\n"
        )
        f.write(
            "# EPW baseline: USA_CO_Golden-NREL.724666_TMY3.epw "
            "(ARCHITECTURE.md §Reference Data)\n"
        )
        f.write(
            f"# Tolerance: ±{TOLERANCE_PCT}% annual energy per ASHRAE 140 "
            "acceptance criteria (issue #1147)\n"
        )
        f.write(
            "# Hourly E+ output: regenerate via "
            "generate_case_800_810_energy.py\n"
        )
        f.write(
            "# Spec source: src/validation/ashrae140/cases/series_800.rs\n"
        )
        f.write(
            f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        )
        writer = csv.writer(f)
        writer.writerow([
            "metric", "unit", "ref_min", "ref_max", "ref_midpoint",
            "tolerance_pct", "accept_min", "accept_max", "notes",
        ])
        for (
            metric, unit, ref_min, ref_max, mid, tol,
            accept_min, accept_max, notes_text,
        ) in metrics:
            writer.writerow([
                metric, unit,
                f"{ref_min:.3f}", f"{ref_max:.3f}", f"{mid:.3f}",
                tol,
                f"{accept_min:.3f}", f"{accept_max:.3f}",
                notes_text,
            ])

    print(f"Wrote {path}")
    return path


def main() -> int:
    if not EP_PATH.exists():
        print(
            f"EnergyPlus not found at {EP_PATH}. "
            "Cases 800/810 reference CSVs are not yet checked in "
            "(issue #2953 follow-up). Install EnergyPlus to regenerate, "
            "or update PROVENANCE.md if the reference values change "
            "without a re-run."
        )
        return 1

    # Note describing the geometry + HVAC layer per case. These are sourced
    # from ``src/validation/ashrae140/cases/series_800.rs`` — keep them in
    # sync if the spec changes.
    notes_by_case = {
        "800": {
            "annual_heating": "Light-mass envelope + single-stage heat pump (COP 3.2); expect higher heating than Case 600 COP-unadjusted",
            "annual_cooling": "Light-mass envelope + single-stage heat pump (COP 3.5); summer load similar to Case 600 baseline",
            "peak_heating": "Winter peak at minimum outdoor temperature (-15°C cap, Zone 5A)",
            "peak_cooling": "Summer peak at maximum outdoor temperature (40°C cap, Zone 5A)",
        },
        "810": {
            "annual_heating": "High-mass commercial + comprehensive HVAC with VAV + heat recovery (COP 4.0)",
            "annual_cooling": "High-mass commercial + comprehensive HVAC with economizer + heat recovery (COP 6.0)",
            "peak_heating": "Winter peak with heat-recovery pre-heat (Zone 4A)",
            "peak_cooling": "Summer peak with economizer + VAV modulation (Zone 4A)",
        },
    }

    rc = 0
    for case_id, idf_path, output_csv, geometry_label in CASES:
        if not idf_path.exists():
            print(
                f"IDF not found: {idf_path} — Case {case_id} is a "
                f"follow-up to issue #2953 (script created; IDF + CSV "
                f"outputs pending). Skipping."
            )
            continue
        work_dir = Path(f"/tmp/eplus_case_{case_id}_energy")
        print(f"Running EnergyPlus for Case {case_id}...")
        if not run_energyplus(idf_path, work_dir):
            rc = 1
            continue
        print("Extracting hourly data...")
        rows = extract_hourly(work_dir / "eplusout.sql")
        if len(rows) != 8760:
            print(
                f"WARNING: expected 8760 rows, got {len(rows)} "
                f"for Case {case_id}"
            )
        write_hourly_csv(rows, output_csv, case_id, geometry_label)
        summary = compute_summary(rows)
        print(
            f"  Case {case_id}: annual_heat="
            f"{summary['annual_heating_MWh']:.3f} MWh, "
            f"annual_cool={summary['annual_cooling_MWh']:.3f} MWh, "
            f"peak_heat={summary['peak_heating_kW']:.2f} kW, "
            f"peak_cool={summary['peak_cooling_kW']:.2f} kW"
        )
        write_reference_summary(
            case_id,
            geometry_label,
            summary,
            notes_by_case[case_id],
        )

    if rc == 0:
        print("Done.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
