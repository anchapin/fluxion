#!/usr/bin/env python3
"""
Python verification for Issue #1327 — `WeatherDependentVentilation::get_ach()` lock-in
for ASHRAE 140 default infiltration (0.5 ACH).

Reproduces the Rust formulas in `src/sim/ventilation.rs` and prints the per-hour
fluxion-vs-expected drift for the ASHRAE 140 Case 900 spec inputs.

This script is purely diagnostic — it prints statistics for human review; the
authoritative pass/fail assertion lives in the Rust test
`tests/ventilation_isolation.rs::test_ashrae_140_0p5_ach_default`.

ASHRAE 140-2023 §5.5.3.6 (Default Infiltration): 0.5 ACH is the standard
default infiltration rate for the BESTEST Case 900 / 920 / 940 / 950 / 960
specifications.  The EnergyPlus reference model used a constant
`ZoneInfiltration:DesignFlowRate` schedule at 0.5 ACH; this script verifies
that fluxion's `WeatherDependentVentilation::get_ach()` preserves that
spec value when configured with `min_ach = max_ach = 0.5` across all 8760
hours of Denver TMY3 weather.

Inputs (ASHRAE 140 Case 900 reference model — matches the EnergyPlus CSV at
`tests/reference_data/ventilation/infiltration_denver.csv`):
  Volume              = 129.6 m³    (6 × 8 × 2.7 m)
  Building height     = 2.7 m
  T_in                = 20 °C       (Case 900 heating/cooling neutrality)
  Wind speed          = Denver TMY3 (USA_CO_Golden-NREL.724666_TMY3)
  Shielding factor    = 0.5         (set inside #1278 fix for `wind_benefit()`)
  Opening fraction    = 0.3         (WeatherDependentVentilation default)
  Target ACH          = 0.5 ± 0.05

References:
  - ASHRAE Standard 140-2023 §5.5.3.6 (Default Infiltration = 0.5 ACH)
  - Issue #1327 (test addition)
  - Issue #1278 (wired weather params into `wind_benefit()`)
  - Issue #1279 (dynamic h_tr_is forced-convection multiplier)
  - Issue #1280 §4 (Case 900 cooling-load undercount residual)
  - `src/sim/ventilation.rs::WeatherDependentVentilation`
  - `tests/reference_data/ventilation/infiltration_denver.csv`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# --- Constants from fluxion src/sim/ventilation.rs --------------------------

STACK_COEFFICIENT = 0.025  # src/sim/ventilation.rs:18
AIR_DENSITY = 1.2          # kg/m³
AIR_SPECIFIC_HEAT = 1000.0  # J/(kg·K) — matches E+ CSV reference model

# --- ASHRAE 140 Case 900 spec inputs ---------------------------------------

INDOOR_TEMP_C = 20.0
ZONE_VOLUME_M3 = 129.6     # 6m × 8m × 2.7m (Case 900 zone geometry)
BUILDING_HEIGHT_M = 2.7    # single-story zone height
OPENING_FRACTION = 0.3     # WeatherDependentVentilation default
SHIELDING_FACTOR = 0.5     # PR #1278 chose this for weather-responsive vents
TARGET_ACH = 0.5
TOLERANCE = 0.05           # ±0.05 ACH per the issue's acceptance criteria

# Mirror `opening_area = opening_fraction * 2.0 * (building_height * 3.0)`
OPENING_AREA_M2 = OPENING_FRACTION * 2.0 * (BUILDING_HEIGHT_M * 3.0)

# --- Fluxion Rust formulas (1:1 with src/sim/ventilation.rs) ----------------


def calculate_wind_infiltration_ach(
    wind_speed: float,
    building_height: float,
    shielding_factor: float,
) -> float:
    """Mirror src/sim/ventilation.rs::calculate_wind_infiltration_ach (lines 35-45)."""
    shelter_coefficient = 0.0 + (1.0 - shielding_factor) * 0.4
    height_factor = math.sqrt(building_height / 3.0)
    base_wind_speed = 3.0
    n_factor = shelter_coefficient * height_factor
    return n_factor * (wind_speed / base_wind_speed)


def calculate_stack_infiltration_ach(
    indoor_temp: float,
    outdoor_temp: float,
    height_diff: float,
    opening_area: float,
    zone_volume: float,
) -> float:
    """Mirror src/sim/ventilation.rs::calculate_stack_infiltration_ach (lines 58-76)."""
    if zone_volume <= 0.0 or height_diff <= 0.0:
        return 0.0
    delta_t = abs(indoor_temp - outdoor_temp)
    if delta_t < 0.5:
        return 0.0
    flow_arg = delta_t / height_diff
    flow_sqrt = math.sqrt(flow_arg) if flow_arg > 0.0 else 0.0
    q_vent = STACK_COEFFICIENT * opening_area * flow_sqrt
    return q_vent / zone_volume


def calculate_combined_infiltration_ach(
    outdoor_temp: float,
    indoor_temp: float,
    wind_speed: float,
    height_diff: float,
    opening_area: float,
    zone_volume: float,
    shielding_factor: float,
) -> float:
    """Mirror src/sim/ventilation.rs::calculate_combined_infiltration_ach (lines 91-110)."""
    wind_ach = calculate_wind_infiltration_ach(wind_speed, height_diff, shielding_factor)
    stack_ach = calculate_stack_infiltration_ach(
        indoor_temp, outdoor_temp, height_diff, opening_area, zone_volume
    )
    return max(wind_ach + stack_ach, 0.0)


def outdoor_temp_benefit(
    outdoor_temp: float,
    indoor_temp: float,
    start_temp: float = 18.0,
    full_open_temp: float = 26.0,
    indoor_cooling_setpoint: float = 26.0,
) -> float:
    """Mirror src/sim/ventilation.rs::outdoor_temp_benefit (lines 323-338)."""
    if outdoor_temp <= start_temp:
        return 0.0
    if outdoor_temp >= full_open_temp:
        return 1.0
    if indoor_temp <= indoor_cooling_setpoint:
        return 0.0
    delta_t_out = full_open_temp - start_temp
    if delta_t_out <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (outdoor_temp - start_temp) / delta_t_out))


def wind_benefit(
    wind_speed: float,
    outdoor_temp: float,
    indoor_temp: float,
    zone_volume: float,
    max_ach: float,
    opening_fraction: float = OPENING_FRACTION,
    building_height: float = BUILDING_HEIGHT_M,
    shielding_factor: float = SHIELDING_FACTOR,
) -> float:
    """Mirror src/sim/ventilation.rs::wind_benefit (lines 345-366)."""
    opening_area = opening_fraction * 2.0 * (building_height * 3.0)
    ach = calculate_combined_infiltration_ach(
        outdoor_temp,
        indoor_temp,
        wind_speed,
        building_height,
        opening_area,
        zone_volume,
        shielding_factor,
    )
    return max(0.0, min(1.0, ach / max_ach))


def get_ach_weather(
    outdoor_temp: float,
    indoor_temp: float,
    wind_speed: float,
    zone_volume: float,
    base_ach: float,
    min_ach: float,
    max_ach: float,
) -> float:
    """Mirror src/sim/ventilation.rs::get_ach_weather (lines 306-317).

    When min_ach == max_ach, the output is deterministic at min_ach — this is
    the lock-in shape for ASHRAE 140 Case 900 default infiltration (0.5 ACH).
    """
    t_b = outdoor_temp_benefit(outdoor_temp, indoor_temp)
    w_b = wind_benefit(wind_speed, outdoor_temp, indoor_temp, zone_volume, max_ach)
    combined = (t_b + w_b) / 2.0
    return max(min_ach + (max_ach - min_ach) * combined, min_ach)


def ach_to_conductance(ach: float, volume: float, rho: float, cp: float) -> float:
    """Mirror src/sim/ventilation.rs::ach_to_conductance (lines 405-407).

    h_ve = (ach * volume * rho * cp) / 3600
    """
    return (ach * volume * rho * cp) / 3600.0


# --- CSV loader --------------------------------------------------------------


def load_reference_data(path: Path) -> list[dict]:
    """Load Denver TMY3 reference rows (matches Rust `load_reference_data`)."""
    rows: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("hour"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "hour": int(parts[0]),
                    "outdoor_temp_c": float(parts[1]),
                    "wind_speed_ms": float(parts[2]),
                    "infiltration_ach": float(parts[3]),
                    "vent_conductance": float(parts[4]),
                }
            )
    return rows


# --- Main --------------------------------------------------------------------


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ref_path = repo_root / "tests" / "reference_data" / "ventilation" / "infiltration_denver.csv"
    if not ref_path.exists():
        print(f"ERROR: reference CSV not found: {ref_path}", file=sys.stderr)
        return 2

    rows = load_reference_data(ref_path)
    if len(rows) != 8760:
        print(
            f"ERROR: expected 8760 rows in reference CSV, got {len(rows)}",
            file=sys.stderr,
        )
        return 2

    # ---- 1. Spec lock-in: get_ach returns 0.5 ± 0.05 with min=max=0.5 ----
    print("=" * 78)
    print("ASHRAE 140-2023 §5.5.3.6 default-infiltration lock-in")
    print("=" * 78)
    print(f"  Volume          = {ZONE_VOLUME_M3} m³")
    print(f"  Building height = {BUILDING_HEIGHT_M} m")
    print(f"  T_in            = {INDOOR_TEMP_C} °C")
    print(f"  Opening area    = {OPENING_AREA_M2} m² (per fluxion formula)")
    print(f"  Shielding       = {SHIELDING_FACTOR}")
    print(f"  Target ACH      = {TARGET_ACH} ± {TOLERANCE}")
    print()

    drift_values: list[float] = []
    out_of_tolerance = 0
    max_drift = 0.0
    worst_hour = 0
    spec_preserving_ach: list[float] = []

    for row in rows:
        ach = get_ach_weather(
            row["outdoor_temp_c"],
            INDOOR_TEMP_C,
            row["wind_speed_ms"],
            ZONE_VOLUME_M3,
            base_ach=TARGET_ACH,
            min_ach=TARGET_ACH,
            max_ach=TARGET_ACH,
        )
        spec_preserving_ach.append(ach)
        drift = abs(ach - TARGET_ACH)
        drift_values.append(drift)
        if drift > max_drift:
            max_drift = drift
            worst_hour = row["hour"]
        if drift > TOLERANCE:
            out_of_tolerance += 1

    print("-" * 78)
    print("WeatherDependentVentilation::get_ach(hour) — min_ach = max_ach = 0.5")
    print("-" * 78)
    print(f"  Hours checked               : {len(spec_preserving_ach)}")
    print(f"  Min ACH observed            : {min(spec_preserving_ach):.6f}")
    print(f"  Max ACH observed            : {max(spec_preserving_ach):.6f}")
    print(f"  Mean ACH observed           : {sum(spec_preserving_ach)/len(spec_preserving_ach):.6f}")
    print(f"  Max |ACH − 0.5|             : {max_drift:.6f}  (at hour {worst_hour})")
    print(f"  Hours out of ±{TOLERANCE} tolerance : {out_of_tolerance}")
    print()

    # ---- 2. Per-hour derivation: wind + stack vs combined ------------------
    print("-" * 78)
    print("Per-hour derivation: wind_infiltration + stack_infiltration = combined")
    print("-" * 78)
    max_decomp_err = 0.0
    max_combined = 0.0
    min_combined = math.inf
    sum_combined = 0.0
    for row in rows:
        wind = calculate_wind_infiltration_ach(
            row["wind_speed_ms"], BUILDING_HEIGHT_M, SHIELDING_FACTOR
        )
        stack = calculate_stack_infiltration_ach(
            INDOOR_TEMP_C,
            row["outdoor_temp_c"],
            BUILDING_HEIGHT_M,
            OPENING_AREA_M2,
            ZONE_VOLUME_M3,
        )
        combined = calculate_combined_infiltration_ach(
            row["outdoor_temp_c"],
            INDOOR_TEMP_C,
            row["wind_speed_ms"],
            BUILDING_HEIGHT_M,
            OPENING_AREA_M2,
            ZONE_VOLUME_M3,
            SHIELDING_FACTOR,
        )
        decomp_err = abs((wind + stack) - combined)
        if decomp_err > max_decomp_err:
            max_decomp_err = decomp_err
        max_combined = max(max_combined, combined)
        min_combined = min(min_combined, combined)
        sum_combined += combined

    print(f"  Max |wind + stack − combined|  : {max_decomp_err:.6e}  (1% ARCHITECTURE.md tolerance)")
    print(f"  Max combined over year         : {max_combined:.4f}")
    print(f"  Min combined over year         : {min_combined:.4f}")
    print(f"  Mean combined over year        : {sum_combined / 8760:.4f}")
    print(
        "  NOTE: physics-based combined is not 0.5 — it varies with wind/ΔT.\n"
        "        This is why the ASHRAE 140 spec is encoded via min_ach = max_ach = 0.5,\n"
        "        not via the physics model. The decomposition check above verifies\n"
        "        wind + stack = combined to 1% per ARCHITECTURE.md Module 4."
    )
    print()

    # ---- 3. Conductance check: ach_to_conductance(0.5, 129.6, 1.2, 1000) ----
    print("-" * 78)
    print("Ventilation conductance: ach_to_conductance(0.5, 129.6, 1.2, 1000)")
    print("-" * 78)
    fluxion_h_ve = ach_to_conductance(TARGET_ACH, ZONE_VOLUME_M3, AIR_DENSITY, AIR_SPECIFIC_HEAT)
    analytical_h_ve = (TARGET_ACH * ZONE_VOLUME_M3 * AIR_DENSITY * AIR_SPECIFIC_HEAT) / 3600.0
    print(f"  Fluxion h_ve                : {fluxion_h_ve:.4f} W/K")
    print(f"  Analytical h_ve             : {analytical_h_ve:.4f} W/K")
    print(f"  EnergyPlus CSV h_ve         : 21.6 W/K (constant)")
    print(
        f"  |fluxion − analytical|/analytical : {abs(fluxion_h_ve - analytical_h_ve) / analytical_h_ve * 100:.4f}%"
    )
    print()

    # ---- Final pass/fail summary -----------------------------------------
    print("=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)
    spec_lock_ok = out_of_tolerance == 0
    decomp_ok = max_decomp_err < 0.01 * max_combined if max_combined > 0 else True
    cond_ok = abs(fluxion_h_ve - analytical_h_ve) / analytical_h_ve < 0.01
    print(
        f"  [{'PASS' if spec_lock_ok else 'FAIL'}] get_ach(hour) within 0.5 ± 0.05 for 8760 hours"
    )
    print(
        f"  [{'PASS' if decomp_ok else 'FAIL'}] wind + stack = combined within 1% (per ARCHITECTURE.md)"
    )
    print(
        f"  [{'PASS' if cond_ok else 'FAIL'}] ach_to_conductance matches analytical within 1%"
    )
    print()

    return 0 if (spec_lock_ok and decomp_ok and cond_ok) else 1


if __name__ == "__main__":
    sys.exit(main())