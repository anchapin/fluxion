#!/usr/bin/env python3
"""
Generate zone balance reference data for Fluxion issue #969.

Creates an annual (8760-hour) time series with:
- Zone temperature locked at 20°C via ZoneHVAC:IdealLoadsAirSystem
- Outdoor temperature from Denver TMY3 (varies hourly)
- Q_cond: total envelope conduction (sum of all surface inside-face rates, W)
- Q_solar = 0 (all surfaces NoSun)
- Q_vent = 0 (all surfaces NoWind, no mechanical ventilation)
- Q_int = 0 (no internal gains)
- Q_heat = zone heating energy rate (W) = heating energy [J] / 3600
- Q_cool = zone cooling energy rate (W) = cooling energy [J] / 3600

CSV columns: hour, T_zone(C), T_out(C), Q_cond(W), Q_solar(W), Q_vent(W), Q_int(W), Q_heat(W), Q_cool(W)
"""

import csv
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EPW = Path(
    "/usr/local/EnergyPlus-25-2-0/WeatherData/USA_CO_Golden-NREL.724666_TMY3.epw"
)
MODEL_DIR = Path("/tmp/eplus_fixed_zone_temp")
IDF_PATH = Path("tests/reference_data/energyplus_models/fixed_inputs_zone_temp.idf")
OUTPUT_CSV = Path("tests/reference_data/zone_balance/fixed_inputs_zone_temp.csv")


def run_energyplus(idf_text: str) -> bool:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    idf_path = MODEL_DIR / "in.idf"
    idf_path.write_text(idf_text)

    cmd = [
        "energyplus",
        "-w",
        str(EPW),
        "-d",
        str(MODEL_DIR),
        "-p",
        "eplus",
        str(idf_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            "STDOUT:",
            result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        )
        print(
            "STDERR:",
            result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
        )
        return False
    return True


def extract_csv() -> dict:
    """Extract all needed variables from eplusout.sql."""
    db_path = MODEL_DIR / "eplusout.sql"
    conn = sqlite3.connect(db_path)

    # Build time maps: hour -> TimeIndex for odd (zone temp, conduction) and even (heating) rows
    time_maps = get_time_map(conn)
    odd_map = time_maps["odd"]
    even_map = time_maps["even"]

    # Outdoor drybulb (key="Environment") - use odd TimeIndex
    T_out = get_series_by_key(
        conn, "Site Outdoor Air Drybulb Temperature", "Environment"
    )

    # Zone temperature - use odd TimeIndex
    T_zone = get_series_by_key(conn, "Zone Mean Air Temperature", "ZONE1")

    # Surface conduction — sum inside-face rates for all surfaces - use odd TimeIndex
    surfaces = ["SOUTHWALL", "NORTHWALL", "EASTWALL", "WESTWALL", "ROOF", "FLOOR"]
    Q_cond = {}
    for h in odd_map.keys():
        Q_cond[h] = 0.0
    for surf in surfaces:
        series = get_series_by_key(
            conn, "Surface Inside Face Conduction Heat Transfer Rate", surf
        )
        for time_idx, q in series.items():
            h = (time_idx + 1) // 2  # Convert TimeIndex to hour
            if h in odd_map:
                Q_cond[h] = Q_cond.get(h, 0.0) + q

    # Heating/cooling energy (J per hour) → convert to W - use even TimeIndex
    Q_heat_raw = get_series_by_key(
        conn, "Zone Air System Sensible Heating Energy", "ZONE1"
    )
    Q_cool_raw = get_series_by_key(
        conn, "Zone Air System Sensible Cooling Energy", "ZONE1"
    )

    # Convert J to W: power [W] = energy [J] / 3600 [s/h]
    Q_heat = {}
    Q_cool = {}
    for time_idx, val in Q_heat_raw.items():
        h = (time_idx + 1) // 2
        Q_heat[h] = val / 3600.0
    for time_idx, val in Q_cool_raw.items():
        h = (time_idx + 1) // 2
        Q_cool[h] = val / 3600.0

    conn.close()
    return dict(
        odd_map=odd_map,
        even_map=even_map,
        T_out=T_out,
        T_zone=T_zone,
        Q_cond=Q_cond,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
    )


def get_series_by_key(conn: sqlite3.Connection, var_name: str, key: str) -> dict:
    """Returns {TimeIndex: value} for a variable with given key."""
    query = """
        SELECT r.TimeIndex, r.Value
        FROM ReportData AS r
        JOIN ReportDataDictionary AS d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
        WHERE d.Name = ?
          AND d.KeyValue = ?
        ORDER BY r.TimeIndex
    """
    rows = conn.execute(query, [var_name, key]).fetchall()
    return {time_idx: float(val) for time_idx, val in rows}


def get_time_map(conn: sqlite3.Connection) -> dict:
    """Map hour of year (1-8760) -> TimeIndex.

    E+ stores 2 rows per hour with different IntervalType values.
    Zone temp (T_zone) and surface conduction use IntervalType=-1 (odd TimeIndex).
    Hourly meters (Q_heat, Q_cool) use IntervalType=1 (even TimeIndex).
    We build two maps: hour -> odd_TimeIndex and hour -> even_TimeIndex.
    """
    query = """
        SELECT TimeIndex, EnvironmentPeriodIndex, Hour, Day, Month, IntervalType
        FROM Time
        WHERE EnvironmentPeriodIndex = 1
        ORDER BY TimeIndex
    """
    rows = conn.execute(query).fetchall()
    odd_map = {}
    even_map = {}
    for row in rows:
        time_idx, env_idx, hour, day, month, interval_type = row
        if env_idx != 1:
            continue
        hour_of_year = (time_idx + 1) // 2
        if interval_type == -1:
            odd_map[hour_of_year] = time_idx
        else:
            even_map[hour_of_year] = time_idx
    return {"odd": odd_map, "even": even_map}


def write_csv(data: dict, output_path: Path):
    """Write the reference CSV with 8760 rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "# EnergyPlus Version: 25.2.0",
                "# Model: Single-zone box (6×8×2.7m), ZoneHVAC IdealLoadsAirSystem, fixed 20°C setpoint",
                "# EPW: USA_CO_Golden-NREL.724666_TMY3.epw",
                f"# Generated: {datetime.now(timezone.utc).isoformat()}",
                "# Parameters: All surfaces NoSun/NoWind, R-20 insulated envelope",
                "# Columns: hour(1-8760), T_zone(C), T_out(C), Q_cond(W), Q_solar(W), Q_vent(W), Q_int(W), Q_heat(W), Q_cool(W)",
                "# Notes: Q_solar=0 (NoSun), Q_vent=0 (NoWind, no mechanical vent), Q_int=0 (no internal gains)",
                "#        Q_cond is sum of inside-face conduction for all 6 surfaces",
                "#        Q_heat/Q_cool = ideal loads output to maintain T_zone at 20°C setpoint",
                "#        Q_heat and Q_cool are computed from hourly cumulative energy [J] divided by 3600 to get [W]",
            ]
        )
        writer.writerow(
            [
                "hour",
                "T_zone(C)",
                "T_out(C)",
                "Q_cond(W)",
                "Q_solar(W)",
                "Q_vent(W)",
                "Q_int(W)",
                "Q_heat(W)",
                "Q_cool(W)",
            ]
        )

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        odd_map = data["odd_map"]
        T_zone = data["T_zone"]
        T_out = data["T_out"]
        Q_cond = data["Q_cond"]
        Q_heat = data["Q_heat"]
        Q_cool = data["Q_cool"]

        for h in range(1, 8761):
            odd_idx = odd_map.get(h)
            writer.writerow(
                [
                    h,
                    f"{T_zone.get(odd_idx, 20.0):.4f}" if odd_idx else "20.0000",
                    f"{T_out.get(odd_idx, 0.0):.4f}" if odd_idx else "0.0000",
                    f"{Q_cond.get(h, 0.0):.4f}",
                    "0.0000",
                    "0.0000",
                    "0.0000",
                    f"{Q_heat.get(h, 0.0):.4f}",
                    f"{Q_cool.get(h, 0.0):.4f}",
                ]
            )

    print(f"Wrote {output_path}")


def main():
    idf_text = IDF_PATH.read_text()
    print("Running EnergyPlus simulation...")
    if not run_energyplus(idf_text):
        sys.exit(1)
    print("Extracting CSV...")
    data = extract_csv()
    write_csv(data, OUTPUT_CSV)
    print("Done.")


if __name__ == "__main__":
    main()
