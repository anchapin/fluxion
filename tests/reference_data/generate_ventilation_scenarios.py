#!/usr/bin/env python3
"""
Generate expanded EnergyPlus reference data for Ventilation module validation.

Scenarios:
  1. Denver, 0.5 ACH  (verify existing)
  2. Denver, 1.0 ACH
  3. Denver, 0.1 ACH  (tight envelope)
  4. Tampa  (hot-humid), 0.5 ACH
  5. Dulles (cold), 0.5 ACH

Output per scenario:
  hour, T_out, wind_speed, ACH, C_vent, Q_vent

Q_vent = C_vent * (T_zone - T_out)  [W]
  T_zone = Zone Mean Air Temperature from E+ simulation
  C_vent = ACH * VOL * rho * cp / 3600  [W/K]

Prerequisites:
  - EnergyPlus 25.2.0 on PATH
  - EPW files in /usr/local/EnergyPlus-25-2-0/WeatherData/

Usage:
  python generate_ventilation_scenarios.py
"""

import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EPLUS_DIR = Path("/usr/local/EnergyPlus-25-2-0")
EPLUS = "energyplus"
EPLUS_VERSION = "25.2.0"

VENT_DIR = SCRIPT_DIR / "ventilation"
MODEL_DIR = SCRIPT_DIR / "energyplus_models"

for d in [VENT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Geometry constants ─────────────────────────────────────────────────────
W, D, H = 6.0, 8.0, 2.7  # m
VOL = W * D * H  # 129.6 m³
RHO = 1.2  # kg/m³
CP = 1000.0  # J/(kg·K)

# ── Scenario definitions ───────────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "denver_05ach",
        "label": "Denver, 0.5 ACH",
        "epw": "USA_CO_Golden-NREL.724666_TMY3.epw",
        "ach": 0.5,
        "city": "Denver",
        "climate": "Mixed-Dry",
    },
    {
        "name": "denver_10ach",
        "label": "Denver, 1.0 ACH",
        "epw": "USA_CO_Golden-NREL.724666_TMY3.epw",
        "ach": 1.0,
        "city": "Denver",
        "climate": "Mixed-Dry",
    },
    {
        "name": "denver_01ach",
        "label": "Denver, 0.1 ACH (tight)",
        "epw": "USA_CO_Golden-NREL.724666_TMY3.epw",
        "ach": 0.1,
        "city": "Denver",
        "climate": "Mixed-Dry",
    },
    {
        "name": "tampa_05ach",
        "label": "Tampa (hot-humid), 0.5 ACH",
        "epw": "USA_FL_Tampa.Intl.AP.722110_TMY3.epw",
        "ach": 0.5,
        "city": "Tampa",
        "climate": "Hot-Humid",
    },
    {
        "name": "dulles_05ach",
        "label": "Dulles (cold), 0.5 ACH",
        "epw": "USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw",
        "ach": 0.5,
        "city": "Dulles",
        "climate": "Cold",
    },
]


# ── IDF template ────────────────────────────────────────────────────────────
IDF_TEMPLATE = """\
Version, {version};

SimulationControl,
  No, No, No, No, Yes;

RunPeriod,
  AnnualRun,
  1, 1, , 12, 31, , Tuesday, Yes, Yes, No, Yes, Yes;

Timestep, 1;

Building,
  RefBox_Vent,
  0.0, City, 0.04, 0.4, FullExterior, 25;

Zone,
  ZONE1, 0, 0, 0, 0, 1, 1, , {volume}, , ;

Material,
  EXT_GYP, MediumRough, 0.016, 0.16, 800, 1090;
Material,
  EXT_INSULATION, MediumRough, 0.050, 0.04, 12, 840;
Material,
  INT_GYP, MediumSmooth, 0.013, 0.16, 800, 1090;

Construction,
  EXTWALL, EXT_GYP, EXT_INSULATION, INT_GYP;

Material,
  ROOF_INSUL, MediumRough, 0.080, 0.04, 12, 840;
Construction,
  ROOF, ROOF_INSUL;

Material,
  FLOOR_INSUL, MediumRough, 0.080, 0.04, 12, 840;
Construction,
  FLOOR, FLOOR_INSUL;

GlobalGeometryRules,
  UpperLeftCorner, ClockWise, World;

BuildingSurface:Detailed,
  SouthWall, Wall, EXTWALL, ZONE1, , Outdoors, , SunExposed, WindExposed, 0.5, 4,
  0, 0, {h},  6, 0, {h},  6, 0, 0,  0, 0, 0;

BuildingSurface:Detailed,
  NorthWall, Wall, EXTWALL, ZONE1, , Outdoors, , SunExposed, WindExposed, 0.5, 4,
  6, {d}, {h},  0, {d}, {h},  0, {d}, 0,  6, {d}, 0;

BuildingSurface:Detailed,
  EastWall, Wall, EXTWALL, ZONE1, , Outdoors, , SunExposed, WindExposed, 0.5, 4,
  {w}, 0, {h},  {w}, {d}, {h},  {w}, {d}, 0,  {w}, 0, 0;

BuildingSurface:Detailed,
  WestWall, Wall, EXTWALL, ZONE1, , Outdoors, , SunExposed, WindExposed, 0.5, 4,
  0, {d}, {h},  0, 0, {h},  0, 0, 0,  0, {d}, 0;

BuildingSurface:Detailed,
  Roof, Roof, ROOF, ZONE1, , Outdoors, , SunExposed, WindExposed, , 4,
  0, 0, {h},  {w}, 0, {h},  {w}, {d}, {h},  0, {d}, {h};

BuildingSurface:Detailed,
  Floor, Floor, FLOOR, ZONE1, , Ground, , NoSun, NoWind, , 4,
  0, 0, 0,  {w}, 0, 0,  {w}, {d}, 0,  0, {d}, 0;

Site:GroundTemperature:BuildingSurface,
  18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;

ZoneInfiltration:DesignFlowRate,
  Infil_Z1_ACH,
  ZONE1,
  ,
  AirChanges/Hour,
  ,
  ,
  ,
  {ach},
  ;

Output:SQLite, SimpleAndTabular;

Output:Variable, *, Site Outdoor Air Drybulb Temperature, Hourly;
Output:Variable, *, Site Wind Speed, Hourly;
Output:Variable, ZONE1, Zone Mean Air Temperature, Hourly;
"""


def build_idf(scenario: dict) -> str:
    """Render IDF template for a scenario."""
    return IDF_TEMPLATE.format(
        version=EPLUS_VERSION,
        w=W,
        d=D,
        h=H,
        volume=VOL,
        ach=scenario["ach"],
    )


def compute_cvent(ach: float) -> float:
    """Ventilation conductance in W/K."""
    return ach * VOL * RHO * CP / 3600.0


def run_simulation(idf_path: Path, epw_path: Path, run_dir: Path) -> bool:
    """Run E+ and return True on success."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [EPLUS, "-d", str(run_dir), "-w", str(epw_path), str(idf_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


def extract_simulation_data(run_dir: Path, ach: float) -> list[dict]:
    """Extract hourly data from E+ SQLite output."""
    sql_path = run_dir / "eplusout.sql"
    if not sql_path.exists():
        print(f"    SQL file not found: {sql_path}")
        return []

    conn = sqlite3.connect(str(sql_path))
    cur = conn.cursor()

    def fetch_col(var_name: str) -> dict:
        cur.execute(
            """
            SELECT t.Hour, r.Value
            FROM ReportData r
            JOIN Time t ON r.TimeIndex = t.TimeIndex
            JOIN ReportDataDictionary rdd
              ON r.ReportDataDictionaryIndex = rdd.ReportDataDictionaryIndex
            WHERE rdd.Name = :varname
              AND rdd.ReportingFrequency = 'Hourly'
            ORDER BY t.TimeIndex
        """,
            {"varname": var_name},
        )
        return {int(hour): float(val) for hour, val in cur.fetchall()}

    outdoor_temps = fetch_col("Site Outdoor Air Drybulb Temperature")
    winds = fetch_col("Site Wind Speed")
    zone_temps = fetch_col("Zone Mean Air Temperature")

    conn.close()

    cvent = compute_cvent(ach)
    results = []
    for hour in range(1, 8761):
        t_out = outdoor_temps.get(hour, 20.0)
        ws = winds.get(hour, 0.0)
        t_zone = zone_temps.get(hour, t_out)
        q_vent = cvent * (t_zone - t_out)
        results.append(
            {
                "hour": hour,
                "T_out": round(t_out, 1),
                "wind_speed": round(ws, 1),
                "ACH": ach,
                "C_vent": round(cvent, 4),
                "Q_vent": round(q_vent, 2),
            }
        )
    return results


def write_csv(csv_path: Path, rows: list[dict], scenario: dict):
    """Write CSV with metadata header."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cvent = compute_cvent(scenario["ach"])
    params = (
        f"Volume={VOL} m3, ACH={scenario['ach']}, "
        f"rho_air={RHO} kg/m3, cp={CP} J/(kg.K), "
        f"C_vent={cvent:.4f} W/K, climate={scenario['climate']}"
    )
    lines = [
        f"# EnergyPlus Version: {EPLUS_VERSION}\n",
        f"# Model: Single-zone box ({W}x{D}x{H}m), {scenario['label']}\n",
        f"# EPW: {scenario['epw']}\n",
        f"# Generated: {now}\n",
        f"# Parameters: {params}\n",
        f"# Rows: {len(rows)}\n",
        "hour(1-8760),T_out(C),wind_speed(m/s),ACH(1/h),C_vent(W/K),Q_vent(W)\n",
    ]
    for r in rows:
        lines.append(
            f"{r['hour']},{r['T_out']},{r['wind_speed']},"
            f"{r['ACH']},{r['C_vent']},{r['Q_vent']}\n"
        )
    csv_path.write_text("".join(lines))


def write_idf(idf_path: Path, scenario: dict):
    """Write scenario IDF file."""
    idf_path.write_text(build_idf(scenario))


def main():
    print("=" * 70)
    print("Generating expanded ventilation reference data")
    print("=" * 70)

    for sc in SCENARIOS:
        name = sc["name"]
        epw_path = EPLUS_DIR / "WeatherData" / sc["epw"]
        idf_path = MODEL_DIR / f"ventilation_{name}.idf"
        csv_path = VENT_DIR / f"infiltration_{name}.csv"
        run_dir = Path("/tmp") / f"fluxion_vent_{name}"

        print(f"\n--- Scenario: {sc['label']} ---")
        print(f"  EPW: {epw_path}")

        if not epw_path.exists():
            print(f"  ERROR: EPW not found: {epw_path}")
            continue

        # Write IDF
        write_idf(idf_path, sc)
        print(f"  IDF written: {idf_path.name}")

        # Run simulation
        if run_simulation(idf_path, epw_path, run_dir):
            print("  E+ simulation completed")
            rows = extract_simulation_data(run_dir, sc["ach"])
            if rows:
                write_csv(csv_path, rows, sc)
                print(f"  CSV written: {csv_path.name} ({len(rows)} rows)")
            else:
                print("  WARNING: No data extracted from SQL")
        else:
            print("  ERROR: E+ simulation failed")
            shutil.rmtree(run_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
