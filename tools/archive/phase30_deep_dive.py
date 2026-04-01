#!/usr/bin/env python3
"""
Deep Dive Analysis: Compare Fluxion solar calculations against theoretical expectations.

This script analyzes:
1. Incidence angle distribution by orientation
2. Ground-reflected radiation calculation
3. Beam/diffuse split validation
4. SHGC effectiveness at different angles
"""

import csv
import math
from collections import defaultdict
from pathlib import Path


def read_solar_data(filepath):
    """Read solar diagnostic data."""
    data = []
    if not Path(filepath).exists():
        return data

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "month": int(row["Month"]),
                    "day": int(row["Day"]),
                    "hour": float(row["HourOfDay"]),
                    "orientation": row["Orientation"],
                    "dni": float(row["DNI"]),
                    "dhi": float(row["DHI"]),
                    "ghi": float(row["GHI"]),
                    "beam_irradiance": float(row["BeamIrradiance_Wm2"]),
                    "diffuse_irradiance": float(row["DiffuseIrradiance_Wm2"]),
                    "ground_reflected_irradiance": float(row["GroundReflected_Wm2"]),
                    "incidence_angle": float(row["IncidenceAngle_deg"]),
                    "shgc_effective": float(row["SHGC_Effective"]),
                    "beam_gain_w": float(row["BeamGain_W"]),
                    "diffuse_gain_w": float(row["DiffuseGain_W"]),
                    "ground_gain_w": float(row["GroundGain_W"]),
                    "total_gain_w": float(row["TotalGain_W"]),
                    "outdoor_temp": float(row["OutdoorTemp_C"]),
                }
            )
    return data


def calculate_expected_incidence_angle(orientation, month, day, hour, latitude=39.7):
    """Calculate theoretical incidence angle for verification."""
    # Simplified solar position calculation
    day_of_year = (month - 1) * 30 + day

    # Declination angle
    declination = 23.45 * math.sin(math.radians(360 / 365 * (284 + day_of_year)))

    # Hour angle
    hour_angle = 15 * (hour - 12)

    # Solar altitude and azimuth
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)

    sin_alt = math.sin(lat_rad) * math.sin(dec_rad) + math.cos(lat_rad) * math.cos(
        dec_rad
    ) * math.cos(ha_rad)
    altitude = math.degrees(math.asin(max(0, sin_alt)))

    if altitude <= 0:
        return 90.0  # Sun below horizon

    # Solar azimuth (from North, clockwise)
    cos_az = (
        math.sin(dec_rad) * math.cos(lat_rad)
        - math.cos(dec_rad) * math.sin(lat_rad) * math.cos(ha_rad)
    ) / math.cos(math.radians(altitude))
    cos_az = max(-1, min(1, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if hour < 12:
        azimuth = 360 - azimuth

    # Surface azimuth (from North, clockwise)
    surface_azimuth = {
        "North": 0.0,
        "East": 90.0,
        "South": 180.0,
        "West": 270.0,
    }.get(orientation, 180.0)

    # Incidence angle for vertical surface
    # cos(θ) = cos(alt) × cos(azimuth_diff)
    azimuth_diff = math.radians(abs(azimuth - surface_azimuth))
    cos_incidence = math.cos(math.radians(altitude)) * math.cos(azimuth_diff)
    cos_incidence = max(0, cos_incidence)

    return math.degrees(math.acos(cos_incidence))


def analyze_incidence_angles(data_by_case):
    """Analyze incidence angle distribution and compare with theory."""
    print("\n" + "=" * 80)
    print("INCIDENCE ANGLE ANALYSIS - Theoretical vs Actual")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        cooling_data = [
            r for r in data if 5 <= r["month"] <= 9 and r["total_gain_w"] > 0
        ]

        if not cooling_data:
            continue

        print(f"\n{case_id} (Cooling Season):")
        print(
            f"{'Orientation':>10} | {'Count':>6} | {'Avg Actual':>10} | {'Avg Theory':>10} | {'Diff':>8}"
        )
        print("-" * 60)

        by_orient = defaultdict(list)
        for r in cooling_data:
            by_orient[r["orientation"]].append(r)

        for orient, rows in sorted(by_orient.items()):
            actual_angles = [r["incidence_angle"] for r in rows]
            avg_actual = sum(actual_angles) / len(actual_angles)

            # Calculate theoretical angles
            theoretical_angles = []
            for r in rows:
                theory = calculate_expected_incidence_angle(
                    orient, r["month"], r["day"], r["hour"]
                )
                theoretical_angles.append(theory)

            avg_theory = (
                sum(theoretical_angles) / len(theoretical_angles)
                if theoretical_angles
                else 0
            )
            diff = avg_actual - avg_theory

            status = "✓" if abs(diff) < 5 else "⚠️"

            print(
                f"{orient:>10} | {len(rows):6d} | {avg_actual:10.1f} | {avg_theory:10.1f} | {diff:+8.1f} {status}"
            )


def analyze_ground_reflected(data_by_case):
    """Analyze ground-reflected radiation calculation."""
    print("\n" + "=" * 80)
    print("GROUND-REFLECTED RADIATION ANALYSIS")
    print("=" * 80)

    print("\nGround-reflected formula: GHI × albedo × (1 - cos(tilt)) / 2")
    print("For vertical surfaces (tilt=90°): factor = 0.5")
    print("For horizontal surfaces (tilt=0°): factor = 0.0")

    for case_id, data in data_by_case.items():
        cooling_data = [
            r for r in data if 5 <= r["month"] <= 9 and r["total_gain_w"] > 0
        ]

        if not cooling_data:
            continue

        print(f"\n{case_id}:")

        by_orient = defaultdict(list)
        for r in cooling_data:
            by_orient[r["orientation"]].append(r)

        for orient, rows in sorted(by_orient.items()):
            # Calculate expected ground-reflected
            total_ghi = sum(r["ghi"] for r in rows)
            expected_ground = (
                total_ghi * 0.2 * 0.5
            )  # albedo=0.2, factor=0.5 for vertical

            actual_ground = sum(r["ground_reflected_irradiance"] for r in rows)

            ratio = actual_ground / expected_ground if expected_ground > 0 else 0

            print(
                f"  {orient:10s}: Expected={expected_ground / 1000:.1f} kWh, "
                f"Actual={actual_ground / 1000:.1f} kWh, "
                f"Ratio={ratio:.2f}"
            )


def analyze_beam_diffuse_ratio(data_by_case):
    """Analyze beam/diffuse ratio against theoretical expectations."""
    print("\n" + "=" * 80)
    print("BEAM/DIFFUSE RATIO ANALYSIS")
    print("=" * 80)

    print("\nExpected pattern for vertical surfaces:")
    print("  - South: High beam fraction (direct sun at noon)")
    print("  - East: High beam fraction in morning, low in afternoon")
    print("  - West: Low beam fraction in morning, high in afternoon")

    for case_id, data in data_by_case.items():
        cooling_data = [
            r for r in data if 5 <= r["month"] <= 9 and r["total_gain_w"] > 0
        ]

        if not cooling_data:
            continue

        print(f"\n{case_id}:")

        by_orient = defaultdict(list)
        for r in cooling_data:
            by_orient[r["orientation"]].append(r)

        for orient, rows in sorted(by_orient.items()):
            total_beam = sum(r["beam_gain_w"] for r in rows)
            total_diffuse = sum(r["diffuse_gain_w"] for r in rows)
            total = total_beam + total_diffuse

            if total > 0:
                beam_ratio = total_beam / total
                diffuse_ratio = total_diffuse / total

                # Expected beam ratio for vertical surfaces in summer
                expected_beam = {
                    "South": 0.75,
                    "East": 0.65,
                    "West": 0.65,
                    "North": 0.30,
                }.get(orient, 0.5)

                diff = beam_ratio - expected_beam
                status = "✓" if abs(diff) < 0.1 else "⚠️"

                print(
                    f"  {orient:10s}: Beam={beam_ratio:.2f} (expected ~{expected_beam:.2f}), "
                    f"Diffuse={diffuse_ratio:.2f} {status}"
                )


def analyze_time_of_day_pattern(data_by_case):
    """Analyze solar gain by time of day."""
    print("\n" + "=" * 80)
    print("TIME-OF-DAY PATTERN ANALYSIS (July 15)")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        july_15 = [r for r in data if r["month"] == 7 and r["day"] == 15]

        if not july_15:
            continue

        print(f"\n{case_id}:")

        # Group by hour and orientation
        by_hour_orient = defaultdict(lambda: defaultdict(list))
        for r in july_15:
            by_hour_orient[int(r["hour"])][r["orientation"]].append(r)

        # Analyze morning (6-9am) vs afternoon (3-6pm)
        morning_east = []
        afternoon_west = []

        for hour in range(6, 10):
            if "East" in by_hour_orient[hour]:
                for r in by_hour_orient[hour]["East"]:
                    morning_east.append(r["beam_irradiance"])

        for hour in range(15, 19):
            if "West" in by_hour_orient[hour]:
                for r in by_hour_orient[hour]["West"]:
                    afternoon_west.append(r["beam_irradiance"])

        if morning_east and afternoon_west:
            avg_morning_east = sum(morning_east) / len(morning_east)
            avg_afternoon_west = sum(afternoon_west) / len(afternoon_west)

            print(f"  Morning East beam (6-9am): {avg_morning_east:.1f} W/m²")
            print(f"  Afternoon West beam (3-6pm): {avg_afternoon_west:.1f} W/m²")
            print(f"  Ratio (West/East): {avg_afternoon_west / avg_morning_east:.2f}")

            # Expected: Afternoon should be similar or slightly higher due to higher ambient temp
            if avg_afternoon_west / avg_morning_east < 0.8:
                print("  ⚠️  Afternoon West beam is too low!")
            elif avg_afternoon_west / avg_morning_east > 1.2:
                print("  ⚠️  Afternoon West beam is too high!")
            else:
                print("  ✓ Ratio is reasonable")


def identify_root_cause(data_by_case):
    """Generate root cause hypothesis based on analysis."""
    print("\n" + "=" * 80)
    print("ROOT CAUSE IDENTIFICATION")
    print("=" * 80)

    hypotheses = []

    for case_id, data in data_by_case.items():
        cooling_data = [
            r for r in data if 5 <= r["month"] <= 9 and r["total_gain_w"] > 0
        ]

        if not cooling_data:
            continue

        by_orient = defaultdict(list)
        for r in cooling_data:
            by_orient[r["orientation"]].append(r)

        # Check 1: Incidence angles
        for orient in ["East", "West", "South"]:
            if orient not in by_orient:
                continue

            actual_angles = [r["incidence_angle"] for r in by_orient[orient]]
            avg_actual = sum(actual_angles) / len(actual_angles)

            # Calculate theoretical
            theoretical = []
            for r in by_orient[orient]:
                theory = calculate_expected_incidence_angle(
                    orient, r["month"], r["day"], r["hour"]
                )
                theoretical.append(theory)
            avg_theory = sum(theoretical) / len(theoretical)

            if abs(avg_actual - avg_theory) > 10:
                hypotheses.append(
                    f"⚠️  INCIDENCE ANGLE ERROR: {orient} avg={avg_actual:.1f}°, expected={avg_theory:.1f}°"
                )

        # Check 2: Ground-reflected too high
        for orient in ["East", "West", "South"]:
            if orient not in by_orient:
                continue

            total_ghi = sum(r["ghi"] for r in by_orient[orient])
            expected_ground = total_ghi * 0.2 * 0.5
            actual_ground = sum(
                r["ground_reflected_irradiance"] for r in by_orient[orient]
            )

            if actual_ground > expected_ground * 1.2:
                hypotheses.append(
                    f"⚠️  GROUND-REFLECTED TOO HIGH: {orient} actual={actual_ground / 1000:.1f} kWh, expected={expected_ground / 1000:.1f} kWh"
                )

        # Check 3: Beam fraction too high
        for orient in ["East", "West"]:
            if orient not in by_orient:
                continue

            total_beam = sum(r["beam_gain_w"] for r in by_orient[orient])
            total_diffuse = sum(r["diffuse_gain_w"] for r in by_orient[orient])
            total = total_beam + total_diffuse

            if total > 0:
                beam_ratio = total_beam / total
                if beam_ratio > 0.75:  # Expected ~0.65 for E/W
                    hypotheses.append(
                        f"⚠️  BEAM FRACTION TOO HIGH: {orient} beam_ratio={beam_ratio:.2f}, expected ~0.65"
                    )

    if hypotheses:
        print("\nIdentified Issues:")
        for h in hypotheses:
            print(f"  {h}")
    else:
        print("\n✓ No obvious issues found in solar calculation components")
        print("  The remaining 22% error may be due to:")
        print("    - Thermal mass modeling (5R1C limitations)")
        print("    - HVAC system modeling differences")
        print("    - Internal loads scheduling")
        print("    - Infiltration modeling")


def main():
    print("=" * 80)
    print("Phase 30 Deep Dive Analysis")
    print("=" * 80)

    # Read data
    data_by_case = {}

    data_paths = {
        "Case 900": "/tmp/solar_diagnostics/case_900_solar.csv",
        "Case 920": "/tmp/solar_diagnostics/case_920_solar.csv",
    }

    for case_id, path in data_paths.items():
        data = read_solar_data(path)
        if data:
            data_by_case[case_id] = data
            print(f"  Loaded {case_id}: {len(data)} records")

    if not data_by_case:
        print("\nNo data found!")
        return

    # Run analyses
    analyze_incidence_angles(data_by_case)
    analyze_ground_reflected(data_by_case)
    analyze_beam_diffuse_ratio(data_by_case)
    analyze_time_of_day_pattern(data_by_case)
    identify_root_cause(data_by_case)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
