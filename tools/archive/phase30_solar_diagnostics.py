#!/usr/bin/env python3
"""
Phase 30 Diagnostic: Solar Gain Component Analysis

This script analyzes hourly solar gain components to identify the root cause
of E/W solar gain underestimation during cooling season.

Generates:
1. Hourly component breakdown (beam, diffuse, ground-reflected)
2. SHGC effective value vs incidence angle
3. Comparison between Case 900 (South) and Case 920 (E/W)
4. EnergyPlus comparison data (if available)
"""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_solar_diagnostics(filepath):
    """Read detailed solar diagnostic data from CSV."""
    data = []
    if not Path(filepath).exists():
        print(f"  Warning: {filepath} not found")
        return data

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "hour": int(row.get("Hour", 0)),
                    "month": int(row.get("Month", 1)),
                    "day": int(row.get("Day", 1)),
                    "hour_of_day": int(row.get("HourOfDay", 12)),
                    "orientation": row.get("Orientation", "Unknown"),
                    "dni": float(row.get("DNI", 0)),
                    "dhi": float(row.get("DHI", 0)),
                    "ghi": float(row.get("GHI", 0)),
                    "beam_irradiance": float(row.get("BeamIrradiance_Wm2", 0)),
                    "diffuse_irradiance": float(row.get("DiffuseIrradiance_Wm2", 0)),
                    "ground_reflected_irradiance": float(
                        row.get("GroundReflected_Wm2", 0)
                    ),
                    "total_irradiance": float(row.get("TotalIrradiance_Wm2", 0)),
                    "incidence_angle": float(row.get("IncidenceAngle_deg", 0)),
                    "shgc_effective": float(row.get("SHGC_Effective", 0)),
                    "beam_gain_w": float(row.get("BeamGain_W", 0)),
                    "diffuse_gain_w": float(row.get("DiffuseGain_W", 0)),
                    "ground_gain_w": float(row.get("GroundGain_W", 0)),
                    "total_gain_w": float(row.get("TotalGain_W", 0)),
                    "outdoor_temp": float(row.get("OutdoorTemp_C", 0)),
                }
            )
    return data


def calculate_solar_position(latitude_deg, month, day, hour):
    """Calculate approximate solar position for Denver (39.7°N)."""
    # Simplified solar position calculation
    day_of_year = (month - 1) * 30 + day  # Approximate

    # Declination angle (Cooper equation)
    declination = 23.45 * math.sin(math.radians(360 / 365 * (284 + day_of_year)))

    # Hour angle
    hour_angle = 15 * (hour - 12)

    # Solar altitude
    lat_rad = math.radians(latitude_deg)
    dec_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)

    sin_altitude = math.sin(lat_rad) * math.sin(dec_rad) + math.cos(lat_rad) * math.cos(
        dec_rad
    ) * math.cos(ha_rad)
    altitude = math.degrees(math.asin(max(0, sin_altitude)))

    # Solar azimuth (simplified)
    if altitude > 0:
        cos_azimuth = (
            math.sin(dec_rad) * math.cos(lat_rad)
            - math.cos(dec_rad) * math.sin(lat_rad) * math.cos(ha_rad)
        ) / math.cos(math.radians(altitude))
        azimuth = math.degrees(math.acos(max(-1, min(1, cos_azimuth))))
        if hour < 12:
            azimuth = 360 - azimuth
    else:
        azimuth = 180

    return altitude, azimuth


def analyze_orientation_effect(data_by_case):
    """Analyze solar gain by orientation."""
    print("\n" + "=" * 80)
    print("ORIENTATION EFFECT ANALYSIS - Cooling Season (May-September)")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        print(f"\n{case_id}:")
        print("-" * 80)

        # Group by orientation
        by_orientation = defaultdict(list)
        for row in data:
            if 5 <= row["month"] <= 9:  # Cooling season
                by_orientation[row["orientation"]].append(row)

        for orientation, rows in sorted(by_orientation.items()):
            total_beam = sum(r["beam_gain_w"] for r in rows)
            total_diffuse = sum(r["diffuse_gain_w"] for r in rows)
            total_ground = sum(r["ground_gain_w"] for r in rows)
            total_gain = sum(r["total_gain_w"] for r in rows)

            if total_gain > 0:
                beam_frac = total_beam / total_gain * 100
                diffuse_frac = total_diffuse / total_gain * 100
                ground_frac = total_ground / total_gain * 100

                print(
                    f"  {orientation:10s}: {total_gain / 1000:8.1f} kWh "
                    f"(Beam: {beam_frac:5.1f}%, Diffuse: {diffuse_frac:5.1f}%, "
                    f"Ground: {ground_frac:5.1f}%)"
                )


def analyze_daily_profile(data_by_case, month=7, day=15):
    """Analyze daily solar gain profile for a specific day."""
    print(f"\n{'=' * 80}")
    print(f"DAILY SOLAR GAIN PROFILE - {month}/{day:02d} (Summer)")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        print(f"\n{case_id}:")
        print("-" * 80)

        # Extract data for specific day
        day_data = [r for r in data if r["month"] == month and r["day"] == day]

        # Group by hour and orientation
        by_hour_orient = defaultdict(lambda: defaultdict(list))
        for row in day_data:
            by_hour_orient[row["hour_of_day"]][row["orientation"]].append(row)

        print(
            f"{'Hour':>4} | {'Orient':>8} | {'Beam W/m²':>10} | {'Diff W/m²':>10} | "
            f"{'Inc Ang':>8} | {'SHGC':>6} | {'Gain W':>8}"
        )
        print("-" * 75)

        for hour in range(24):
            for orientation in sorted(by_hour_orient[hour].keys()):
                rows = by_hour_orient[hour][orientation]
                if not rows:
                    continue

                # Average values
                beam = np.mean([r["beam_irradiance"] for r in rows])
                diffuse = np.mean([r["diffuse_irradiance"] for r in rows])
                inc_angle = np.mean([r["incidence_angle"] for r in rows])
                shgc = np.mean([r["shgc_effective"] for r in rows])
                gain = np.mean([r["total_gain_w"] for r in rows])

                print(
                    f"{hour:4d} | {orientation:8s} | {beam:10.1f} | {diffuse:10.1f} | "
                    f"{inc_angle:8.1f} | {shgc:6.3f} | {gain:8.1f}"
                )


def analyze_shgc_vs_angle(data_by_case):
    """Analyze SHGC effectiveness vs incidence angle."""
    print(f"\n{'=' * 80}")
    print("SHGC ANGULAR DEPENDENCE ANALYSIS")
    print("=" * 80)

    # Collect all SHGC vs angle data
    all_data = []
    for case_id, data in data_by_case.items():
        for row in data:
            if row["total_gain_w"] > 0:
                all_data.append(
                    {
                        "case": case_id,
                        "orientation": row["orientation"],
                        "angle": row["incidence_angle"],
                        "shgc": row["shgc_effective"],
                    }
                )

    # Bin by angle
    angle_bins = [(i, i + 10) for i in range(0, 90, 10)]

    print(f"\n{'Angle Range':>12} | {'Count':>6} | {'Avg SHGC':>9} | {'Std Dev':>8}")
    print("-" * 45)

    for low, high in angle_bins:
        bin_data = [d for d in all_data if low <= d["angle"] < high]
        if bin_data:
            avg_shgc = np.mean([d["shgc"] for d in bin_data])
            std_shgc = np.std([d["shgc"] for d in bin_data])
            print(
                f"{low:3d}-{high:2d}°      {len(bin_data):6d} | {avg_shgc:9.4f} | {std_shgc:8.4f}"
            )


def compare_beam_diffuse_ratio(data_by_case):
    """Compare beam vs diffuse fractions by orientation."""
    print(f"\n{'=' * 80}")
    print("BEAM/DIFFUSE RATIO BY ORIENTATION (Cooling Season)")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        print(f"\n{case_id}:")

        # Group by orientation
        by_orientation = defaultdict(list)
        for row in data:
            if 5 <= row["month"] <= 9 and row["total_gain_w"] > 0:
                by_orientation[row["orientation"]].append(row)

        for orientation, rows in sorted(by_orientation.items()):
            total_beam = sum(r["beam_gain_w"] for r in rows)
            total_diffuse = sum(r["diffuse_gain_w"] for r in rows)
            total = total_beam + total_diffuse

            if total > 0:
                beam_ratio = total_beam / total
                diffuse_ratio = total_diffuse / total

                print(
                    f"  {orientation:10s}: Beam={beam_ratio:5.2f}, "
                    f"Diffuse={diffuse_ratio:5.2f}, "
                    f"B/D Ratio={beam_ratio / diffuse_ratio:5.2f}"
                    if diffuse_ratio > 0
                    else f"  {orientation:10s}: Beam={beam_ratio:5.2f}, No diffuse"
                )


def plot_solar_components(data_by_case, output_dir="/tmp/solar_diagnostics"):
    """Generate diagnostic plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Plot 1: Daily profile comparison
    plt.figure(figsize=(12, 6))

    for case_id, data in data_by_case.items():
        # July 15 data
        july_15 = [r for r in data if r["month"] == 7 and r["day"] == 15]

        # Group by hour
        by_hour = defaultdict(list)
        for row in july_15:
            by_hour[row["hour_of_day"]].append(row["total_gain_w"])

        hours = sorted(by_hour.keys())
        gains = [np.mean(by_hour[h]) for h in hours]

        plt.plot(hours, gains, label=case_id, marker="o")

    plt.xlabel("Hour of Day")
    plt.ylabel("Solar Gain (W)")
    plt.title("Daily Solar Gain Profile - July 15")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "daily_profile.png", dpi=150)
    plt.close()

    # Plot 2: SHGC vs Incidence Angle
    plt.figure(figsize=(10, 6))

    for case_id, data in data_by_case.items():
        angles = [r["incidence_angle"] for r in data if r["total_gain_w"] > 0]
        shgcs = [r["shgc_effective"] for r in data if r["total_gain_w"] > 0]

        plt.scatter(angles, shgcs, alpha=0.3, label=case_id, s=10)

    plt.xlabel("Incidence Angle (°)")
    plt.ylabel("Effective SHGC")
    plt.title("SHGC Angular Dependence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "shgc_vs_angle.png", dpi=150)
    plt.close()

    # Plot 3: Beam vs Diffuse by orientation
    plt.figure(figsize=(10, 6))

    orientations = ["North", "East", "South", "West"]
    beam_fracs = []
    diffuse_fracs = []

    # Use Case 900 for this analysis
    if "Case 900" in data_by_case:
        data = data_by_case["Case 900"]
        cooling_data = [r for r in data if 5 <= r["month"] <= 9]

        for orient in orientations:
            orient_data = [r for r in cooling_data if r["orientation"] == orient]
            if orient_data:
                total_beam = sum(r["beam_gain_w"] for r in orient_data)
                total_diffuse = sum(r["diffuse_gain_w"] for r in orient_data)
                total = total_beam + total_diffuse
                if total > 0:
                    beam_fracs.append(total_beam / total)
                    diffuse_fracs.append(total_diffuse / total)
                else:
                    beam_fracs.append(0)
                    diffuse_fracs.append(0)
            else:
                beam_fracs.append(None)
                diffuse_fracs.append(None)

        x = np.arange(len(orientations))
        width = 0.35

        plt.bar(x - width / 2, beam_fracs, width, label="Beam", color="orange")
        plt.bar(x + width / 2, diffuse_fracs, width, label="Diffuse", color="skyblue")

        plt.xlabel("Orientation")
        plt.ylabel("Fraction")
        plt.title("Beam/Diffuse Fraction by Orientation (Cooling Season)")
        plt.xticks(x, orientations)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_path / "beam_diffuse_fraction.png", dpi=150)
        plt.close()

    print(f"\nPlots saved to: {output_path}")


def generate_diagnostic_report(
    data_by_case, output_path="/tmp/solar_diagnostics/report.txt"
):
    """Generate comprehensive diagnostic report."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SOLAR GAIN DIAGNOSTIC REPORT - Phase 30\n")
        f.write("=" * 80 + "\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(
            "This report analyzes solar gain components to identify the root cause\n"
        )
        f.write("of E/W solar gain underestimation during cooling season.\n\n")

        # Data summary
        f.write("DATA SUMMARY\n")
        f.write("-" * 80 + "\n")
        for case_id, data in data_by_case.items():
            cooling_data = [r for r in data if 5 <= r["month"] <= 9]
            f.write(
                f"{case_id}: {len(data)} total hours, {len(cooling_data)} cooling season hours\n"
            )
        f.write("\n")

        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")

        # Calculate E/W vs South ratio
        if "Case 900" in data_by_case and "Case 920" in data_by_case:
            data_900 = data_by_case["Case 900"]
            data_920 = data_by_case["Case 920"]

            cooling_900 = [r for r in data_900 if 5 <= r["month"] <= 9]
            cooling_920 = [r for r in data_920 if 5 <= r["month"] <= 9]

            total_900 = sum(r["total_gain_w"] for r in cooling_900)
            total_920 = sum(r["total_gain_w"] for r in cooling_920)

            if total_900 > 0:
                ratio = total_920 / total_900
                f.write(
                    f"1. E/W to South solar gain ratio (cooling season): {ratio:.3f}\n"
                )
                f.write(f"   Expected: 0.75-0.85, Actual: {ratio:.3f}\n")
                f.write(f"   Underestimation: {(0.80 - ratio) / 0.80 * 100:.1f}%\n\n")

        # Component analysis
        f.write("2. Solar Gain Component Breakdown\n")
        for case_id, data in data_by_case.items():
            cooling_data = [
                r for r in data if 5 <= r["month"] <= 9 and r["total_gain_w"] > 0
            ]
            if not cooling_data:
                continue

            total_beam = sum(r["beam_gain_w"] for r in cooling_data)
            total_diffuse = sum(r["diffuse_gain_w"] for r in cooling_data)
            total_ground = sum(r["ground_gain_w"] for r in cooling_data)
            total = total_beam + total_diffuse + total_ground

            if total > 0:
                f.write(f"   {case_id}:\n")
                f.write(f"      Beam: {total_beam / total * 100:.1f}%\n")
                f.write(f"      Diffuse: {total_diffuse / total * 100:.1f}%\n")
                f.write(f"      Ground: {total_ground / total * 100:.1f}%\n")

        f.write("\n")
        f.write("DIAGNOSTIC CONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        f.write("Based on the analysis, the likely root cause is:\n")
        f.write("[To be filled after running analysis]\n")

    print(f"Report saved to: {output_path}")


def main():
    """Main diagnostic analysis."""
    print("=" * 80)
    print("Phase 30 Solar Gain Diagnostic Analysis")
    print("=" * 80)

    # Read diagnostic data
    print("\nReading solar diagnostic data...")
    data_by_case = {}

    # Try to read from multiple possible locations
    possible_paths = [
        "/tmp/solar_diagnostics/case_900_solar.csv",
        "/tmp/case_900_solar_diagnostics.csv",
        "outputs/solar_diagnostics/case_900_solar.csv",
    ]

    for path in possible_paths:
        data = read_solar_diagnostics(path)
        if data:
            data_by_case["Case 900"] = data
            print(f"  Loaded Case 900: {len(data)} hours from {path}")
            break

    for path in possible_paths:
        path_920 = path.replace("case_900", "case_920")
        data = read_solar_diagnostics(path_920)
        if data:
            data_by_case["Case 920"] = data
            print(f"  Loaded Case 920: {len(data)} hours from {path_920}")
            break

    if not data_by_case:
        print("\nNo diagnostic data found. Run Fluxion with diagnostics enabled first.")
        print("Expected data files:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    # Run analyses
    analyze_orientation_effect(data_by_case)
    analyze_daily_profile(data_by_case)
    analyze_shgc_vs_angle(data_by_case)
    compare_beam_diffuse_ratio(data_by_case)

    # Generate plots
    plot_solar_components(data_by_case)

    # Generate report
    generate_diagnostic_report(data_by_case)

    print("\n" + "=" * 80)
    print("Diagnostic analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
