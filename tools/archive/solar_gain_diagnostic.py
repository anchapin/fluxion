#!/usr/bin/env python3
"""
Solar Gain Diagnostic Analysis for ASHRAE 140 Cases

This script analyzes solar gain calculations for South-facing (Case 900) vs
East/West-facing (Case 920) buildings to identify the root cause of cooling
energy underestimation.

Key Questions:
1. Is total solar gain correct for E/W orientations?
2. Is the beam/diffuse split correct?
3. Is the SHGC calculation correct at different incidence angles?
"""

import math

# ASHRAE 140 Case 900/920 specifications
CASE_900_SPEC = {
    "case_id": "900",
    "window_area": 12.0,  # m² (South-facing)
    "window_orientation": "South",
    "floor_area": 48.0,  # m²
}

CASE_920_SPEC = {
    "case_id": "920",
    "window_area": 12.0,  # m² (6 m² East + 6 m² West)
    "window_orientation": "East+West",
    "floor_area": 48.0,  # m²
}

# Window properties (ASHRAE 140 double-pane clear glass)
WINDOW_SHGC = 0.789  # Solar Heat Gain Coefficient at normal incidence
WINDOW_TRANS = 0.86156  # Normal transmittance

# ASHRAE 140 SHGC ratio lookup table (Issue #299)
SHGC_RATIO_TABLE = [
    (0, 1.000),
    (10, 0.995),
    (20, 0.985),
    (30, 0.970),
    (40, 0.940),
    (50, 0.890),
    (60, 0.810),
    (70, 0.680),
    (80, 0.450),
    (90, 0.000),
]


def interpolate_shgc_ratio(angle_deg):
    """Interpolate SHGC ratio from ASHRAE 140 lookup table."""
    if angle_deg <= 0:
        return 1.0
    if angle_deg >= 90:
        return 0.0

    for i in range(len(SHGC_RATIO_TABLE) - 1):
        angle_low, ratio_low = SHGC_RATIO_TABLE[i]
        angle_high, ratio_high = SHGC_RATIO_TABLE[i + 1]

        if angle_low <= angle_deg <= angle_high:
            t = (angle_deg - angle_low) / (angle_high - angle_low)
            return ratio_low * (1 - t) + ratio_high * t

    return 1.0


def calculate_effective_shgc(orientation, hour):
    """
    Calculate effective SHGC for a given orientation and hour.

    This is a simplified model - actual calculation uses solar position,
    incidence angle, etc.
    """
    # Simplified solar position model
    # Solar azimuth: East (90°) at 6am, South (180°) at noon, West (270°) at 6pm
    solar_azimuth = 90 + (hour - 6) * 15  # degrees from North

    if orientation == "South":
        surface_azimuth = 180
    elif orientation == "East":
        surface_azimuth = 90
    elif orientation == "West":
        surface_azimuth = 270
    else:
        return 0

    # Incidence angle (simplified - assumes sun at 45° altitude)
    # cos(incidence) = cos(altitude) * cos(azimuth_diff)
    altitude_rad = math.radians(45)
    azimuth_diff_rad = math.radians(abs(solar_azimuth - surface_azimuth))
    cos_incidence = math.cos(altitude_rad) * math.cos(azimuth_diff_rad)
    incidence_angle = math.degrees(math.acos(max(0, cos_incidence)))

    # Effective SHGC = SHGC_normal * SHGC_ratio(incidence_angle)
    shgc_ratio = interpolate_shgc_ratio(incidence_angle)
    effective_shgc = WINDOW_SHGC * shgc_ratio

    return effective_shgc, incidence_angle


def analyze_daily_solar_gain(case_spec):
    """Analyze solar gain profile for a full day."""
    print(f"\n{'=' * 70}")
    print(f"Daily Solar Gain Analysis - Case {case_spec['case_id']}")
    print(f"{'=' * 70}")
    print(f"Window Area: {case_spec['window_area']} m²")
    print(f"Orientation: {case_spec['window_orientation']}")
    print(f"Floor Area: {case_spec['floor_area']} m²")
    print(f"Window SHGC (normal): {WINDOW_SHGC}")
    print(f"{'=' * 70}\n")

    orientations = []
    if case_spec["window_orientation"] == "South":
        orientations = [("South", case_spec["window_area"])]
    elif case_spec["window_orientation"] == "East+West":
        orientations = [
            ("East", case_spec["window_area"] / 2),
            ("West", case_spec["window_area"] / 2),
        ]

    total_daily_gain = 0

    print(f"{'Hour':>4} | {'Solar Az':>8} | ", end="")
    for orient, _ in orientations:
        print(f"{orient:>12} | ", end="")
    print(f"{'Total (W)':>12}")
    print("-" * (4 + 8 + 12 * len(orientations) + 12 + 10))

    for hour in range(6, 20):  # 6am to 7pm
        solar_azimuth = 90 + (hour - 6) * 15

        hour_total = 0
        hour_data = f"{hour:4d} | {solar_azimuth:8.0f}° | "

        for orient, area in orientations:
            eff_shgc, incidence = calculate_effective_shgc(orient, hour)

            # Simplified solar irradiance model (DNI = 800 W/m² at peak)
            # Peak at solar noon, zero at night
            dni = 800 * max(0, math.sin(math.radians((hour - 6) * 15)))

            # Solar gain = Area × DNI × cos(incidence) × SHGC
            # Simplified: assume cos(incidence) ≈ 0.7 for reasonable angles
            if incidence < 90:
                cos_inc = max(0.1, math.cos(math.radians(incidence)))
                gain = area * dni * cos_inc * eff_shgc
            else:
                gain = 0

            hour_total += gain
            hour_data += f"{gain:10.1f} W | "

        total_daily_gain += hour_total
        hour_data += f"{hour_total:10.1f}"
        print(hour_data)

    print(
        f"\nTotal Daily Solar Gain: {total_daily_gain:.1f} Wh = {total_daily_gain / 1000:.2f} kWh"
    )
    print(f"Daily Average: {total_daily_gain / 14:.1f} W")

    return total_daily_gain


def compare_cases():
    """Compare Case 900 (South) vs Case 920 (E/W)."""
    print("\n" + "=" * 70)
    print("ASHRAE 140 Solar Gain Diagnostic Analysis")
    print("=" * 70)
    print("\nHYPOTHESIS: E/W solar gain is underestimated due to:")
    print("  1. Incorrect SHGC at high incidence angles (afternoon sun)")
    print("  2. Incorrect beam/diffuse split")
    print("  3. Incorrect solar distribution (70% to mass too high)")
    print("=" * 70)

    # Analyze both cases
    gain_900 = analyze_daily_solar_gain(CASE_900_SPEC)
    gain_920 = analyze_daily_solar_gain(CASE_920_SPEC)

    # Compare
    print(f"\n{'=' * 70}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 70}")
    print(f"Case 900 (South) Daily Gain: {gain_900:.1f} Wh")
    print(f"Case 920 (E/W)  Daily Gain: {gain_920:.1f} Wh")
    print(f"Ratio (920/900): {gain_920 / gain_900:.2f}")

    # Expected ratio based on cooling energy reference
    # Case 900 cooling: 8.00-10.50 MWh
    # Case 920 cooling: 6.50-8.50 MWh
    # Expected ratio: ~0.75-0.85
    expected_ratio = 0.80
    print(f"Expected Ratio (from cooling ref): ~{expected_ratio:.2f}")

    if gain_920 / gain_900 < expected_ratio * 0.8:
        print("\n⚠️  WARNING: E/W solar gain is TOO LOW!")
        print("   This could explain the cooling energy underestimation.")
    elif gain_920 / gain_900 > expected_ratio * 1.2:
        print("\n⚠️  WARNING: E/W solar gain is TOO HIGH!")
    else:
        print("\n✓ Solar gain ratio appears reasonable.")

    print(f"\n{'=' * 70}")
    print("RECOMMENDATION")
    print(f"{'=' * 70}")
    print("1. Extract hourly solar gain profiles from actual simulation")
    print("2. Compare with this simplified model")
    print("3. Check if E/W solar gain is underestimated in summer months")
    print("4. Verify SHGC calculation at high incidence angles (>60°)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    compare_cases()
