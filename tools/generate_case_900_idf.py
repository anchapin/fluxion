#!/usr/bin/env python3
"""
Issue #1329 — Generate ASHRAE 140 Case 900 IDF + per-surface roof-solar
hourly reference CSV.

Produces two artefacts:

1. ``tests/reference_data/energyplus_models/ashrae_140_case_900.idf``
   — EnergyPlus 25.2 input model: same 6m × 8m × 2.7m geometry as
     ``ashrae_140_case_600.idf`` but with **200mm concrete** walls,
     roof and floor (high-mass variant per ASHRAE 140-2023 Annex B8).
   — Also wires the 5-surface per-tilt ``Output:Variable`` block
     (Roof + 4 walls × {beam, sky-diffuse, ground-diffuse}) that
     ``ashrae_140_solar_gain.idf`` carries on lines 243-264, so a
     future E+ run can dump the per-surface hourly incident solar
     straight out of ``eplusout.sql``.

2. ``tests/reference_data/solar/case_900_roof_solar_hourly.csv``
   — Per-hour expected incident solar on the **roof** (horizontal,
     tilt = 0) computed from spec-defined inputs only:
       * beam     = DNI × max(cos(zenith), 0)              (Duffie & Beckman 1.6.3, ASHRAE 140 Annex B)
       * sky_d    = DHI × (1 + cos(0)) / 2 = DHI          (isotropic, full sky for horizontal surface)
       * ground_r = GHI × albedo × (1 − cos(0)) / 2 = 0    (no ground visible from a horizontal surface)
       * total    = beam + sky_d + ground_r
   — Inputs: TMY3 weather (``denver_tmy3_reference.csv``) and solar
     position (``solar_position_denver.csv``), both 8760 hourly,
     both already validated against EnergyPlus (ARCHITECTURE.md §2).
   — This is the **blind validation** reference: it is generated
     *without* running E+, so it can be used as ground truth for
     A#1's roof-solar math fix (issue #1323 follow-up) and for the
     fluxion solar module's `cos(zenith)`-on-tilt-0 isolation test.

Per the AGENTS.md / ARCHITECTURE.md hard rules:

    "The reference CSV must be generated from spec-defined inputs
    (ASHRAE 140 Case 900 inputs) — NOT from running the engine and
    recording output."

No E+ invocation is performed by this script.

Run from repo root::

    python3 tools/generate_case_900_idf.py

Both outputs are checked in alongside the Case 600 artefacts so the
script can also be re-run from a fresh checkout (verification path
in the issue body).
"""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IDF_DIR = REPO_ROOT / "tests/reference_data/energyplus_models"
SOLAR_DIR = REPO_ROOT / "tests/reference_data/solar"
WEATHER_CSV = REPO_ROOT / "tests/reference_data/weather/denver_tmy3_reference.csv"
POSITION_CSV = REPO_ROOT / "tests/reference_data/solar/solar_position_denver.csv"

IDF_PATH = IDF_DIR / "ashrae_140_case_900.idf"
ROOF_CSV = SOLAR_DIR / "case_900_roof_solar_hourly.csv"

EPW_NAME = "USA_CO_Golden-NREL.724666_TMY3.epw"
ENERGYPLUS_VERSION = "25.2.0"

# ASHRAE 140 Case 900 spec (Annex B8 of ASHRAE 140-2023):
#   Geometry identical to Case 600 (6m × 8m × 2.7m single-zone box).
#   200 mm concrete walls / roof / floor — high-mass construction.
#   Same south-facing 12 m² double-pane window, 0.5 ACH infiltration,
#   20 °C heating / 27 °C cooling setpoints, ideal loads air system.
#
# 200 mm concrete per ASHRAE 140 Annex B (matches fluxion's existing
# ``step_change_concrete.idf`` Material block):
#   thickness   = 0.200 m
#   conductivity= 1.730 W/m·K
#   density     = 2243 kg/m³
#   specific_h. = 837 J/kg·K
#   roughness   = MediumRough

CONCRETE_THICKNESS_M = 0.200
CONCRETE_CONDUCTIVITY = 1.730
CONCRETE_DENSITY = 2243.0
CONCRETE_CP = 837.0

# Building geometry — identical to Case 600
WIDTH_M = 6.0
DEPTH_M = 8.0
HEIGHT_M = 2.7
ROOF_AREA_M2 = WIDTH_M * DEPTH_M  # 48 m²

# Ground albedo — ASHRAE 140 default (0.2 typical for non-snow terrain).
# For the horizontal roof, (1 − cos(0)) / 2 = 0, so ground-reflected
# component is identically 0 regardless of albedo. Listed here for
# traceability of the formula.
GROUND_ALBEDO = 0.2


def load_weather() -> list[dict]:
    """Load 8760-hourly TMY3 weather (0-indexed hours, columns per header)."""
    rows: list[dict] = []
    with open(WEATHER_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "hour": int(r["hour"]),
                    "dni": float(r["dni_wm2"]),
                    "dhi": float(r["dhi_wm2"]),
                    "ghi": float(r["ghi_wm2"]),
                }
            )
    if len(rows) != 8760:
        raise RuntimeError(
            f"expected 8760 weather rows, got {len(rows)} in {WEATHER_CSV}"
        )
    return rows


def load_solar_position() -> list[dict]:
    """Load 8760-hourly solar position (1-indexed hours, E+ convention)."""
    rows: list[dict] = []
    with open(POSITION_CSV, newline="") as f:
        # Skip the 6 metadata lines (#) — they appear before the header
        # (same convention as the existing south-wall CSV).
        header_seen = False
        for line in f:
            stripped = line.strip()
            if not header_seen:
                if stripped.startswith("hour") or stripped.startswith("hour("):
                    header = next(csv.reader([stripped]))
                    header_seen = True
                    continue
                if stripped.startswith("#") or stripped == "":
                    continue
                # fallthrough: header is just the first row
                header = next(csv.reader([stripped]))
                header_seen = True
                # also parse this row as data
                rows.append(
                    {
                        "hour": int(header[0]),
                        "altitude_deg": float(header[1]),
                        "azimuth_deg": float(header[2]),
                        "zenith_deg": float(header[3]),
                    }
                )
                continue
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            parts = next(csv.reader([stripped]))
            rows.append(
                {
                    "hour": int(parts[0]),
                    "altitude_deg": float(parts[1]),
                    "azimuth_deg": float(parts[2]),
                    "zenith_deg": float(parts[3]),
                }
            )
    if len(rows) != 8760:
        raise RuntimeError(
            f"expected 8760 position rows, got {len(rows)} in {POSITION_CSV}"
        )
    return rows


def compute_roof_hourly(
    weather: list[dict], position: list[dict]
) -> list[dict]:
    """Compute per-hour incident solar on the horizontal roof (tilt=0).

    Roof has tilt=0 → surface normal = zenith direction.
    Incidence angle θ_i = solar zenith, so::

        beam_incident      = DNI × max(cos(zenith_rad), 0)
        sky_diffuse        = DHI × (1 + cos(0)) / 2   = DHI
        ground_reflected   = GHI × albedo × (1 − cos(0)) / 2 = 0
        total_incident     = beam + sky_diffuse + ground_reflected

    These match the per-surface Output:Variable names in
    ``ashrae_140_solar_gain.idf`` lines 259-261.
    """
    by_hour_pos = {r["hour"]: r for r in position}
    out: list[dict] = []
    for w in weather:
        pos = by_hour_pos[w["hour"] + 1]  # weather is 0-indexed, position 1-indexed
        zenith_rad = math.radians(pos["zenith_deg"])
        cos_zenith = math.cos(zenith_rad)
        dni = w["dni"]
        dhi = w["dhi"]
        ghi = w["ghi"]

        # Beam: clip at 0 (sun below horizon → no direct beam)
        if dni <= 0.0 or cos_zenith <= 0.0 or pos["altitude_deg"] <= 0.0:
            beam = 0.0
        else:
            beam = dni * cos_zenith

        # Sky diffuse (isotropic, full sky dome for horizontal surface).
        sky_diffuse = dhi

        # Ground reflected — horizontal surface sees zero ground.
        ground_reflected = 0.0

        total = beam + sky_diffuse + ground_reflected

        out.append(
            {
                "hour": w["hour"] + 1,  # emit 1-indexed (matches E+ convention)
                "beam": beam,
                "sky_diffuse": sky_diffuse,
                "ground_diffuse": ground_reflected,
                "total": total,
                "zenith_deg": pos["zenith_deg"],
                "altitude_deg": pos["altitude_deg"],
                "dni_wm2": dni,
                "dhi_wm2": dhi,
                "ghi_wm2": ghi,
            }
        )
    return out


def write_roof_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# EnergyPlus Version: {ENERGYPLUS_VERSION}"])
        w.writerow(
            [f"# Model: ASHRAE 140 Case 900 — Roof (6m × 8m horizontal, tilt=0)"]
        )
        w.writerow([f"# EPW: {EPW_NAME}"])
        w.writerow([f"# Source inputs: {WEATHER_CSV.name} + {POSITION_CSV.name}"])
        w.writerow([f"# Generated: {now}"])
        w.writerow(
            [
                "# Method: spec-defined (ASHRAE 140 Annex B + Duffie & Beckman "
                "Eq. 1.6.3)."
            ]
        )
        w.writerow(
            [
                "#   beam_irradiance     = DNI × max(cos(zenith), 0)",
                "(sun below horizon → 0)",
            ]
        )
        w.writerow(
            [
                "#   sky_diffuse          = DHI × (1 + cos(0))/2 = DHI",
                "(isotropic, full sky for horizontal)",
            ]
        )
        w.writerow(
            [
                "#   ground_diffuse       = GHI × albedo × (1 − cos(0))/2 = 0",
                "(no ground visible from horizontal)",
            ]
        )
        w.writerow(
            [
                "#   total_irradiance     = beam + sky_diffuse + ground_diffuse"
            ]
        )
        w.writerow([f"# Rows: {len(rows)}"])
        w.writerow(
            [
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
        )
        for r in rows:
            w.writerow(
                [
                    r["hour"],
                    f"{r['beam']:.4f}",
                    f"{r['sky_diffuse']:.4f}",
                    f"{r['ground_diffuse']:.4f}",
                    f"{r['total']:.4f}",
                    f"{r['zenith_deg']:.4f}",
                    f"{r['altitude_deg']:.4f}",
                    f"{r['dni_wm2']:.4f}",
                    f"{r['dhi_wm2']:.4f}",
                    f"{r['ghi_wm2']:.4f}",
                ]
            )
    print(f"Wrote {out_path} ({len(rows)} rows)")


def build_idf_text() -> str:
    """Return the full ASHRAE 140 Case 900 IDF text.

    Structure: identical to ``ashrae_140_case_600.idf`` (geometry,
    window, infiltration, thermostat, ideal loads), but:
      * walls / roof / floor use 200 mm concrete (high-mass);
      * the 5-surface per-tilt Output:Variable block from
        ``ashrae_140_solar_gain.idf`` lines 243-264 is wired in.
    """
    return f"""Version, 25.2;

! ── ASHRAE 140 Case 900 — High-Mass Building with South Window ─────────
! Reference: ASHRAE Standard 140-2023 Annex B8
!   - 6m x 8m x 2.7m single-zone box (same geometry as Case 600)
!   - High-mass construction: 200mm concrete walls, roof, floor
!   - South-facing window (12m², double-pane, SHGC ≈ 0.7)
!   - Heating setpoint: 20°C, Cooling setpoint: 27°C
!   - Infiltration: 0.5 ACH constant
!   - No internal gains, no mechanical ventilation (only infiltration)
!
! Per-surface Output:Variable block (beam + sky-diffuse + ground-diffuse
! for each of the 5 sun-exposed surfaces) mirrors
! tests/reference_data/energyplus_models/ashrae_140_solar_gain.idf
! lines 243-264 — gives E+ the per-tilt hourly ground truth required to
! quantitatively verify the roof-solar fix from issue #1323 / A#1.

SimulationControl,
  No,   ! Do Zone Sizing
  No,   ! Do System Sizing
  No,   ! Do Plant Sizing
  No,   ! Run Simulation for Sizing Periods
  Yes;  ! Run Simulation for Weather File Run Periods

RunPeriod,
  AnnualRun,
  1, 1, , 12, 31, ,
  Tuesday,
  No, No, No, No, No;

Timestep, 1;

Building,
  ASHRAE140_Case900,
  0.0,
  City,
  0.04,
  0.4,
  FullExterior,
  25;

Zone,
  ZONE1,
  0, 0, 0, 0, 1, 1, , , , ;

ZoneHVAC:EquipmentConnections,
  ZONE1, ZONE1_Equip,
  ZONE1_Inlet Node, ZONE1_Exhaust Node,
  ZONE1_Zone Air Node, ZONE1_Return Air Node;

! ── Materials — High-Mass ASHRAE 140 Case 900 construction ─────────────
Material,
  CONCRETE_200,            ! 200mm concrete — ASHRAE 140 high-mass spec
  MediumRough,
  {CONCRETE_THICKNESS_M:.3f},
  {CONCRETE_CONDUCTIVITY:.3f},
  {CONCRETE_DENSITY:.0f},
  {CONCRETE_CP:.0f};

Construction,
  HEAVY_WALL,
  CONCRETE_200;

Construction,
  HEAVY_ROOF,
  CONCRETE_200;

Construction,
  HEAVY_FLOOR,
  CONCRETE_200;

! ── Window (ASHRAE 140 Case 900 glazing — identical to Case 600) ───────
Material,
  GLASS_DBL_CLEAR,
  Glass,
  0.003,
  0.9,
  2500,
  840;

WindowMaterial:Glazing,
  DblClear_Glazing,
  SpectralAverage,
  0.003,
  0.837,
  0.075,
  0.075,
  0.837,
  0.075,
  0.075,
  0.0,
  0.84,
  0.84,
  0.9;

WindowMaterial:Gas,
  DblClear_Gas,
  Air,
  0.0127;

Construction,
  DblClear_Window,
  DblClear_Glazing,
  DblClear_Gas,
  DblClear_Glazing;

! ── Ground temperature ──────────────────────────────────────────────────
Site:GroundTemperature:BuildingSurface,
  18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;

GlobalGeometryRules,
  UpperLeftCorner,
  ClockWise,
  World;

! ── Surfaces (6m wide x 8m deep x 2.7m tall, south-facing window) ───────
BuildingSurface:Detailed,
  SouthWall, Wall, HEAVY_WALL, ZONE1, , Outdoors, , SunExposed, WindExposed, ,
  4,
  0, 0, 2.7,
  6, 0, 2.7,
  6, 0, 0,
  0, 0, 0;

BuildingSurface:Detailed,
  NorthWall, Wall, HEAVY_WALL, ZONE1, , Outdoors, , SunExposed, WindExposed, ,
  4,
  6, 8, 2.7,
  0, 8, 2.7,
  0, 8, 0,
  6, 8, 0;

BuildingSurface:Detailed,
  EastWall, Wall, HEAVY_WALL, ZONE1, , Outdoors, , SunExposed, WindExposed, ,
  4,
  6, 0, 2.7,
  6, 8, 2.7,
  6, 8, 0,
  6, 0, 0;

BuildingSurface:Detailed,
  WestWall, Wall, HEAVY_WALL, ZONE1, , Outdoors, , SunExposed, WindExposed, ,
  4,
  0, 8, 2.7,
  0, 0, 2.7,
  0, 0, 0,
  0, 8, 0;

BuildingSurface:Detailed,
  Roof, Roof, HEAVY_ROOF, ZONE1, , Outdoors, , SunExposed, WindExposed, ,
  4,
  0, 0, 2.7,
  6, 0, 2.7,
  6, 8, 2.7,
  0, 8, 2.7;

BuildingSurface:Detailed,
  Floor, Floor, HEAVY_FLOOR, ZONE1, , Ground, , NoSun, NoWind, ,
  4,
  0, 0, 0,
  6, 0, 0,
  6, 8, 0,
  0, 8, 0;

! ── South window (12m², double-pane) ────────────────────────────────────
FenestrationSurface:Detailed,
  SouthWindow,
  Window,
  DblClear_Window,
  SouthWall,
  , ,
  0.0,
  Outside,
  4,
  0.5, 0.0, 2.0,
  5.5, 0.0, 2.0,
  5.5, 0.0, 0.0,
  0.5, 0.0, 0.0;

! ── Infiltration: 0.5 ACH constant ──────────────────────────────────────
ZoneInfiltration:DesignFlowRate,
  ZONE1_Infil,
  ZONE1,
  AlwaysOn,
  0.02207,
  0, 0, 0;

Schedule:Compact, AlwaysOn, Fraction, Through: 12/31, For: AllDays, Until: 24:00, 1.0;
ScheduleTypeLimits, Fraction, 0.0, 1.0, Continuous;

! ── Thermostat: 20°C heating, 27°C cooling ──────────────────────────────
ScheduleTypeLimits, Temperature, -100, 200, Continuous;

Schedule:Compact,
  HeatSP,
  Temperature,
  Through: 12/31, For: AllDays, Until: 24:00, 20.0;

Schedule:Compact,
  CoolSP,
  Temperature,
  Through: 12/31, For: AllDays, Until: 24:00, 27.0;

ZoneControl:Thermostat,
  ZONE1_Tstat, ZONE1, DualSP_Sched,
  ThermostatSetpoint:DualSetpoint,
  DualSP_Sched;

ScheduleTypeLimits, ControlType, 0.0, 4.0, Discrete;
Schedule:Compact, DualSP_Sched, ControlType, Through: 12/31, For: AllDays, Until: 24:00, 4;

ThermostatSetpoint:DualSetpoint,
  DualSP_Sched, HeatSP, CoolSP;

! ── Ideal Loads Air System (metered heating/cooling) ────────────────────
ZoneHVAC:EquipmentList,
  ZONE1_Equip,
  SequentialLoad,
  ZoneHVAC:IdealLoadsAirSystem, ZONE1_IdealLoads, 1, 1;

ZoneHVAC:IdealLoadsAirSystem,
  ZONE1_IdealLoads,
  , ZONE1_Inlet Node, ZONE1_Exhaust Node, ,
  50, 13, 0.0156, 0.0077,
  NoLimit, , , NoLimit, , ,
  , , ConstantSensibleHeatRatio, 0.7, None, , , , , ;

! ── Output ──────────────────────────────────────────────────────────────
Output:SQLite, SimpleAndTabular;

Output:Variable, *, Site Outdoor Air Drybulb Temperature, Timestep;
Output:Variable, ZONE1, Zone Mean Air Temperature, Timestep;
Output:Variable, ZONE1, Zone Air System Sensible Heating Energy, Hourly;
Output:Variable, ZONE1, Zone Air System Sensible Cooling Energy, Hourly;
Output:Variable, ZONE1, Zone Infiltration Sensible Heat Loss, Timestep;
Output:Variable, ZONE1, Zone Infiltration Sensible Heat Gain, Timestep;
Output:Variable, SouthWindow, Surface Window Transmitted Solar Radiation Rate, Timestep;

! ── Per-surface incident solar (5 surfaces × {{beam, sky-diffuse, ground-diffuse}})
! Mirrors ashrae_140_solar_gain.idf lines 243-264 — gives E+ per-tilt
! hourly ground truth for verifying the roof-solar fix from #1323.
Output:Variable, SouthWall, Surface Outside Face Incident Beam Solar Radiation Rate per Area, Hourly;
Output:Variable, SouthWall, Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area, Hourly;
Output:Variable, SouthWall, Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area, Hourly;

Output:Variable, NorthWall, Surface Outside Face Incident Beam Solar Radiation Rate per Area, Hourly;
Output:Variable, NorthWall, Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area, Hourly;
Output:Variable, NorthWall, Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area, Hourly;

Output:Variable, EastWall, Surface Outside Face Incident Beam Solar Radiation Rate per Area, Hourly;
Output:Variable, EastWall, Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area, Hourly;
Output:Variable, EastWall, Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area, Hourly;

Output:Variable, WestWall, Surface Outside Face Incident Beam Solar Radiation Rate per Area, Hourly;
Output:Variable, WestWall, Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area, Hourly;
Output:Variable, WestWall, Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area, Hourly;

Output:Variable, Roof, Surface Outside Face Incident Beam Solar Radiation Rate per Area, Hourly;
Output:Variable, Roof, Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area, Hourly;
Output:Variable, Roof, Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area, Hourly;
"""


def main() -> int:
    print(f"Loading weather: {WEATHER_CSV}")
    weather = load_weather()
    print(f"  loaded {len(weather)} hourly rows")

    print(f"Loading solar position: {POSITION_CSV}")
    position = load_solar_position()
    print(f"  loaded {len(position)} hourly rows")

    print("Computing per-hour incident solar on the roof (tilt=0)...")
    rows = compute_roof_hourly(weather, position)
    write_roof_csv(rows, ROOF_CSV)

    print(f"Writing IDF: {IDF_PATH}")
    IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDF_PATH.write_text(build_idf_text())
    print(f"  wrote {IDF_PATH.stat().st_size} bytes")

    # Quick sanity-check printout
    max_total = max(r["total"] for r in rows)
    max_beam = max(r["beam"] for r in rows)
    annual_total = sum(r["total"] for r in rows)  # Wh/m² (1-hour sums)
    print()
    print("=== Roof solar summary ===")
    print(f"  peak total irradiance    : {max_total:8.2f} W/m²")
    print(f"  peak beam irradiance     : {max_beam:8.2f} W/m²")
    print(f"  annual total irradiance  : {annual_total / 1000.0:8.2f} kWh/m²/yr")
    print(f"  roof area                : {ROOF_AREA_M2:8.2f} m²")
    print(f"  annual roof energy       : {annual_total * ROOF_AREA_M2 / 1e6:8.3f} MWh/yr")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())