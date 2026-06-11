#!/usr/bin/env python3
"""
Generate fixed-zone conduction reference data for Fluxion issue #946.

Creates a 200mm concrete wall test with:
- Fixed zone air temperature at 20°C (ideal loads)
- Concrete: k=1.73, ρ=2243, cp=837 (matching WallSpec exactly)
- No solar exposure on test wall
- 15-min timesteps, 72 hours (Jan 1-3)
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
OUTPUT_DIR = Path("/tmp/eplus_fixed_zone")
CSV_PATH = Path("tests/reference_data/conduction/step_response_fixed_zone_20c.csv")

# ── IDF ─────────────────────────────────────────────────────────────────────

IDF = """\
Version, 25.2;

! ── Simulation Control ──────────────────────────────────────────────────
SimulationControl,
  No,   ! Do Zone Sizing
  No,   ! Do System Sizing
  No,   ! Do Plant Sizing
  No,   ! Run Simulation for Sizing Periods
  Yes;  ! Run Simulation for Weather File Run Periods

RunPeriod,
  FixedZoneRun,
  1, 1, ,   ! Start: Jan 1
  1, 3, ,   ! End: Jan 3
  Tuesday,
  No, No, No, No, No;

Timestep, 4;  ! 15-min timesteps

! ── Building ────────────────────────────────────────────────────────────
Building,
  FixedZone_Conduction,
  0.0,        ! North Axis
  City,       ! Terrain
  0.04,       ! Loads Convergence Tolerance
  0.4,        ! Temp Convergence Tolerance {deltaC}
  FullExterior,
  25;         ! Max Warmup Days

! ── Zone ────────────────────────────────────────────────────────────────
Zone,
  ZONE1,
  0, 0, 0, 0, 1, 1, , 129.6, 48.0, ;

! ── Schedule: Always 20°C ───────────────────────────────────────────────
ScheduleTypeLimits,
  Temperature,     ! Name
  -100,            ! Lower Limit
  200,             ! Upper Limit
  CONTINUOUS;      ! Numeric Type

Schedule:Compact,
  Always20C,
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,
  20.0;

! ── Zone HVAC: Ideal Loads (native E+ objects, no ExpandObjects needed) ──
ZoneHVAC:IdealLoadsAirSystem,
  ZONE1_IdealLoads,       ! Name
  ,                        ! Availability Schedule Name
  25,                      ! Maximum Heating Supply Air Temperature {C}
  15,                      ! Minimum Cooling Supply Air Temperature {C}
  ,                        ! Maximum Heating Supply Air Humidity Ratio
  ,                        ! Minimum Cooling Supply Air Humidity Ratio
  ,                        ! Heating Limit
  ,                        ! Maximum Heating Air Flow Rate {m3/s}
  ,                        ! Maximum Sensible Heating Capacity {W}
  ,                        ! Cooling Limit
  ,                        ! Maximum Cooling Air Flow Rate {m3/s}
  ,                        ! Maximum Total Cooling Capacity {W}
  ,                        ! Heating Availability Schedule Name
  ,                        ! Cooling Availability Schedule Name
  ,                        ! Dehumidification Control Type
  ,                        ! Cooling Sensible Heat Ratio
  ,                        ! Dehumidification Setpoint {C}
  ,                        ! Humidification Control Type
  ,                        ! Humidification Setpoint {C}
  ,                        ! Outdoor Air Method
  ,                        ! Outdoor Air Flow Rate {m3/s}
  ,                        ! Outdoor Air Schedule Name
  ,                        ! Design Specification Outdoor Air Object Name
  ,                        ! Design Specification Zone Air Distribution Object Name
  ;                        ! Zone Heat Gain Schedule Name (schedules and gains for ventilation/recovery)

ZoneControl:Thermostat,
  ZONE1_Thermostat,       ! Name
  ZONE1,                   ! Zone Name
  ,                        ! Control Type Schedule Name
  1,                       ! Control 1 Object Type
  ThermostatSetpoint:SingleHeating,  ! Control 1 Name
  ,                        ! Control 2 Object Type
  ,                        ! Control 2 Name
  ,                        ! Control 3 Object Type
  ,                        ! Control 3 Name
  ,                        ! Control 4 Object Type
  ;                        ! Control 4 Name

ThermostatSetpoint:SingleHeating,
  HeatSetpoint,           ! Name
  Always20C;               ! Setpoint Temperature Schedule Name

ZoneControl:Thermostat,
  ZONE1_CoolThermostat,
  ZONE1,
  ,
  1,
  ThermostatSetpoint:SingleCooling,
  , , , , , , , ;

ThermostatSetpoint:SingleCooling,
  CoolSetpoint,
  Always20C;

! ── Connect ideal loads to zone ──────────────────────────────────────────
ZoneHVAC:EquipmentConnections,
  ZONE1_EquipConn,        ! Name
  ZONE1,                   ! Zone Name
  ,                        ! Zone Conditioning Equipment List Name
  ,                        ! Zone Air Inlet Port List Name (unused for ideal loads)
  ,                        ! Zone Air Node Name
  ZONE1_IdealLoads,        ! Zone Return Air Node Name
  ;                        ! Zone Exhaust Air Node Name

ZoneHVAC:EquipmentList,
  ZONE1_EquipList,
  ZoneHVAC:IdealLoadsAirSystem,
  ZONE1_IdealLoads,
  1;

! ── Materials (matching WallSpec exactly) ────────────────────────────────
! 200mm concrete: k=1.73 W/(m·K), ρ=2243 kg/m³, cp=837 J/(kg·K)
Material,
  CONCRETE_200,
  MediumRough,
  0.200,       ! Thickness {m}
  1.730,       ! Conductivity {W/m-K}
  2243,        ! Density {kg/m3}
  837;         ! Specific Heat {J/kg-K}

! Insulated wall for non-test surfaces (R-20 approx)
Material,
  HIGH_INSUL,
  MediumRough,
  0.200,
  0.01,
  12,
  840;

Construction,
  CONCRETE_WALL,
  CONCRETE_200;

Construction,
  INSUL_WALL,
  HIGH_INSUL;

! ── Ground temperature ──────────────────────────────────────────────────
Site:GroundTemperature:BuildingSurface,
  18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;

! ── Global Geometry Rules ───────────────────────────────────────────────
GlobalGeometryRules,
  UpperLeftCorner,
  ClockWise,
  World;

! ── Surfaces ────────────────────────────────────────────────────────────
! South wall — TEST wall (200mm concrete, outdoors, NO solar)
BuildingSurface:Detailed,
  SouthWall,
  Wall,
  CONCRETE_WALL,
  ZONE1,
  ,
  Outdoors,
  ,
  NoSun,         ! No solar exposure — pure conduction test
  WindExposed,
  ,
  4,
  0, 0, 2.7,
  6, 0, 2.7,
  6, 0, 0,
  0, 0, 0;

! North wall — insulated
BuildingSurface:Detailed,
  NorthWall, Wall, INSUL_WALL, ZONE1, , Outdoors, , NoSun, WindExposed, ,
  4, 6, 8, 2.7,  0, 8, 2.7,  0, 8, 0,  6, 8, 0;

! East wall — insulated
BuildingSurface:Detailed,
  EastWall, Wall, INSUL_WALL, ZONE1, , Outdoors, , NoSun, WindExposed, ,
  4, 6, 0, 2.7,  6, 8, 2.7,  6, 8, 0,  6, 0, 0;

! West wall — insulated
BuildingSurface:Detailed,
  WestWall, Wall, INSUL_WALL, ZONE1, , Outdoors, , NoSun, WindExposed, ,
  4, 0, 8, 2.7,  0, 0, 2.7,  0, 0, 0,  0, 8, 0;

! Roof — insulated
BuildingSurface:Detailed,
  Roof, Roof, INSUL_WALL, ZONE1, , Outdoors, , NoSun, WindExposed, ,
  4, 0, 0, 2.7,  6, 0, 2.7,  6, 8, 2.7,  0, 8, 2.7;

! Floor — ground contact, insulated
BuildingSurface:Detailed,
  Floor, Floor, INSUL_WALL, ZONE1, , Ground, , NoSun, NoWind, ,
  4, 0, 0, 0,  6, 0, 0,  6, 8, 0,  0, 8, 0;

! ── Output ──────────────────────────────────────────────────────────────
Output:SQLite, SimpleAndTabular;

Output:Variable, *, Site Outdoor Air Drybulb Temperature, Timestep;
Output:Variable, ZONE1, Zone Mean Air Temperature, Timestep;
Output:Variable, SouthWall, Surface Inside Face Temperature, Timestep;
Output:Variable, SouthWall, Surface Outside Face Temperature, Timestep;
Output:Variable, SouthWall, Surface Inside Face Conduction Heat Transfer Rate per Area, Timestep;
Output:Variable, SouthWall, Surface Outside Face Conduction Heat Transfer Rate per Area, Timestep;
"""


def run_energyplus(idf_text: str) -> bool:
    """Run EnergyPlus with the given IDF text."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idf_path = OUTPUT_DIR / "in.idf"
    idf_path.write_text(idf_text)

    cmd = ["energyplus", "-w", str(EPW), "-d", str(OUTPUT_DIR), str(idf_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        err_file = OUTPUT_DIR / "eplusout.err"
        if err_file.exists():
            print("ERROR — last 30 lines of eplusout.err:")
            for line in err_file.read_text().splitlines()[-30:]:
                print(f"  {line}")
        return False

    print("Success.")
    return True


def extract_csv():
    """Extract reference CSV from E+ SQL output."""
    sql_path = OUTPUT_DIR / "eplusout.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL not found: {sql_path}")

    conn = sqlite3.connect(str(sql_path))
    conn.row_factory = sqlite3.Row

    def get_series(var_name: str, key: str | None = None) -> dict:
        if key:
            rows = conn.execute(
                "SELECT ReportDataDictionaryIndex FROM ReportDataDictionary "
                "WHERE Name = ? AND UPPER(KeyValue) = UPPER(?)",
                (var_name, key),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ReportDataDictionaryIndex FROM ReportDataDictionary WHERE Name = ?",
                (var_name,),
            ).fetchall()
        assert rows, f"Variable not found: {key}/{var_name}"
        rddi = rows[0]["ReportDataDictionaryIndex"]
        data = conn.execute(
            "SELECT TimeIndex, Value FROM ReportData WHERE ReportDataDictionaryIndex = ? ORDER BY TimeIndex",
            (rddi,),
        ).fetchall()
        return {r["TimeIndex"]: r["Value"] for r in data}

    def get_time_map() -> dict:
        rows = conn.execute(
            "SELECT TimeIndex, Month, Day, Hour, Minute FROM Time ORDER BY TimeIndex"
        ).fetchall()
        return {
            r["TimeIndex"]: (r["Month"], r["Day"], r["Hour"], r["Minute"]) for r in rows
        }

    t_ext = get_series("Site Outdoor Air Drybulb Temperature")
    t_zone = get_series("Zone Mean Air Temperature", "ZONE1")
    t_surf_in = get_series("Surface Inside Face Temperature", "SouthWall")
    t_surf_out = get_series("Surface Outside Face Temperature", "SouthWall")
    q_in = get_series(
        "Surface Inside Face Conduction Heat Transfer Rate per Area", "SouthWall"
    )
    q_out = get_series(
        "Surface Outside Face Conduction Heat Transfer Rate per Area", "SouthWall"
    )

    time_map = get_time_map()
    n_pts = len(t_ext)
    print(f"Timesteps: {n_pts} (expected ~288 for 72h × 4/h)")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = [
        "# EnergyPlus Version: 25.2.0\n",
        "# Model: 200mm concrete south wall, fixed zone T=20°C via ideal loads, no solar on test wall\n",
        "# EPW: USA_CO_Golden-NREL.724666_TMY3.epw\n",
        f"# Generated: {now}\n",
        "# Concrete: k=1.73 W/(m·K), ρ=2243 kg/m³, cp=837 J/(kg·K), L=200mm\n",
        "# Timestep: 15 min | Run period: Jan 1-3 (72h) | T_zone fixed at 20°C\n",
        "# h_interior (E+ default): ~8.29 W/(m²·K) | h_exterior: ~29.3 W/(m²·K)\n",
        f"# Rows: {n_pts}\n",
    ]

    rows = []
    for idx in sorted(t_ext.keys()):
        if idx not in time_map:
            continue
        month, day, hour, minute = time_map[idx]
        elapsed_h = (day - 1) * 24 + hour + minute / 60.0
        rows.append(
            [
                round(elapsed_h, 4),
                round(t_ext.get(idx, 0), 4),
                round(t_zone.get(idx, 0), 4),
                round(t_surf_in.get(idx, 0), 4),
                round(t_surf_out.get(idx, 0), 4),
                round(q_in.get(idx, 0), 4),
                round(q_out.get(idx, 0), 4),
            ]
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        for line in header:
            f.write(line)
        writer = csv.writer(f)
        writer.writerow(
            [
                "hour(0-72)",
                "T_ext(C)",
                "T_zone(C)",
                "T_surface_inside(C)",
                "T_surface_outside(C)",
                "heat_flux_inside(W/m2)",
                "heat_flux_outside(W/m2)",
            ]
        )
        writer.writerows(rows)

    print(f"Written: {CSV_PATH} ({len(rows)} rows)")

    # Verify zone temp is fixed
    zone_temps = [t_zone.get(idx, 0) for idx in sorted(t_zone.keys())]
    min_tz, max_tz = min(zone_temps), max(zone_temps)
    print(f"Zone temperature range: [{min_tz:.2f}, {max_tz:.2f}] °C (should be ~20.0)")

    # Quick stats
    fluxes = [q_in.get(idx, 0) for idx in sorted(q_in.keys())]
    print(f"Inside heat flux range: [{min(fluxes):.2f}, {max(fluxes):.2f}] W/m²")

    conn.close()


if __name__ == "__main__":
    print("Generating fixed-zone conduction reference data...")
    if not run_energyplus(IDF):
        sys.exit(1)
    extract_csv()
    print("Done.")
