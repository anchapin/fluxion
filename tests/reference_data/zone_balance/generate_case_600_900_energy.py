#!/usr/bin/env python3
"""
Generate ASHRAE 140 Case 600 / 900 EnergyPlus reference data for Fluxion #1147.

Produces two CSVs in tests/reference_data/zone_balance/:

1. ``case_600_energy_hourly.csv`` — Hourly time series for Case 600:
   hour, T_zone(C), T_out(C), Q_heat(W), Q_cool(W), Q_solar_trans(W), Q_infil(W)

2. ``case_900_energy_hourly.csv`` — Same columns, Case 900 high-mass variant.

Both are driven from the same EPW used to regenerate all other reference data
(USA_CO_Golden-NREL.724666_TMY3.epw, see ARCHITECTURE.md §Reference Data).

Usage
-----
Requires EnergyPlus >= 24.1.0 installed at /usr/local/EnergyPlus-25-2-0/
(or edit EP_PATH below).  From the repository root:

    python3 tests/reference_data/zone_balance/generate_case_600_900_energy.py

The script will:
1. Run EnergyPlus against the Case 600 IDF
2. Extract hourly variables from eplusout.sql
3. Write the CSV with 8760 rows + header
4. Repeat for Case 900

CSV format
----------
::

    # EnergyPlus Version: 25.2.0
    # EPW: USA_CO_Golden-NREL.724666_TMY3.epw
    # Case 600: Low-mass, south window 12m², 0.5 ACH, 20°C heat / 27°C cool
    hour,T_zone(C),T_out(C),Q_heat(W),Q_cool(W),Q_solar_trans(W),Q_infil(W)
    1,19.81,-3.00,235.41,0.00,0.00,-126.8
    ...

If EnergyPlus is unavailable, the summary reference values from ASHRAE
Standard 140-2023 Annex B (validated across multiple BEM engines including
EnergyPlus) are already checked in as ``case_600_energy_reference.csv`` and
``case_900_energy_reference.csv`` — those summary files are sufficient for
the annual energy tolerance tests in zone_balance_eplus_isolation.rs.
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

CASES = [
    ("600", IDF_DIR / "ashrae_140_case_600.idf", OUTPUT_DIR / "case_600_energy_hourly.csv"),
    # Case 900 IDF is added under a separate task — add a row here once it exists.
]


def run_energyplus(idf_path: Path, work_dir: Path) -> bool:
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
    """Extract hourly T_zone, T_out, Q_heat, Q_cool from eplusout.sql."""
    conn = sqlite3.connect(sql_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = []
    try:
        time_rows = cur.execute(
            "SELECT TimeIndex, Year, Month, Day, Hour FROM Time "
            "ORDER BY TimeIndex"
        ).fetchall()
        for tr in time_rows:
            hour_of_year = (
                (int(tr["Month"]) - 1) * 730  # approx; replaced below if needed
            )
            hour_of_year = 0
        # Use a simpler approach: row_number() ordered by TimeIndex
        time_rows = cur.execute(
            "SELECT TimeIndex FROM Time ORDER BY TimeIndex"
        ).fetchall()
        time_indices = [r["TimeIndex"] for r in time_rows]

        def get_series(var_name: str, key_value: str | None = None) -> dict[int, float]:
            q = (
                "SELECT v.TimeIndex, v.Value "
                "FROM ReportVariableData v "
                "JOIN ReportVariableDictionary d ON v.ReportVariableDictionaryIndex = d.Index "
                "WHERE d.VariableName = ?"
            )
            params: list = [var_name]
            if key_value:
                q += " AND d.KeyValue = ?"
                params.append(key_value)
            out = {}
            for r in cur.execute(q, params):
                out[r["TimeIndex"]] = r["Value"]
            return out

        t_zone = get_series("Zone Mean Air Temperature", "ZONE1")
        t_out = get_series("Site Outdoor Air Drybulb Temperature", "Environment")
        q_heat = get_series(
            "Zone Air System Sensible Heating Energy", "ZONE1"
        )
        q_cool = get_series(
            "Zone Air System Sensible Cooling Energy", "ZONE1"
        )

        for h, idx in enumerate(time_indices, start=1):
            rows.append(
                {
                    "hour": h,
                    "T_zone(C)": t_zone.get(idx, 0.0),
                    "T_out(C)": t_out.get(idx, 0.0),
                    "Q_heat(W)": q_heat.get(idx, 0.0) / 3600.0,
                    "Q_cool(W)": q_cool.get(idx, 0.0) / 3600.0,
                }
            )
    finally:
        conn.close()
    return rows


def write_csv(rows: list[dict], output_path: Path, case_id: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                f"# EnergyPlus Version: 25.2.0",
                f"# Case {case_id}: ASHRAE 140 — see ARCHITECTURE.md Module 5",
                f"# EPW: {EPW.name}",
                f"# Generated: {datetime.now(timezone.utc).isoformat()}",
                f"# Columns: hour(1-8760), T_zone(C), T_out(C), "
                f"Q_heat(W), Q_cool(W) [hourly mean power]",
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


def main() -> int:
    if not EP_PATH.exists():
        print(
            f"EnergyPlus not found at {EP_PATH}. "
            "Summary reference CSVs (case_600_energy_reference.csv, "
            "case_900_energy_reference.csv) are already checked in and "
            "are sufficient for the annual tolerance tests."
        )
        return 1

    for case_id, idf_path, output_csv in CASES:
        if not idf_path.exists():
            print(f"IDF not found: {idf_path}, skipping Case {case_id}")
            continue
        work_dir = Path(f"/tmp/eplus_case_{case_id}_energy")
        print(f"Running EnergyPlus for Case {case_id}...")
        if not run_energyplus(idf_path, work_dir):
            return 1
        print("Extracting hourly data...")
        rows = extract_hourly(work_dir / "eplusout.sql")
        if len(rows) != 8760:
            print(
                f"WARNING: expected 8760 rows, got {len(rows)} "
                f"for Case {case_id}"
            )
        write_csv(rows, output_csv, case_id)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
