#!/usr/bin/env python3
"""
Phase 30 Solar Diagnostic Analysis - Simple Version
No external dependencies beyond standard library.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path


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
                    "month": int(row.get("Month", 1)),
                    "day": int(row.get("Day", 1)),
                    "hour": float(row.get("HourOfDay", 12)),
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


def analyze_cooling_season_ratio(data_by_case):
    """Analyze E/W to South solar gain ratio during cooling season."""
    print("\n" + "=" * 80)
    print("COOLING SEASON SOLAR GAIN RATIO ANALYSIS")
    print("=" * 80)

    cooling_totals = {}

    for case_id, data in data_by_case.items():
        cooling_data = [r for r in data if 5 <= r["month"] <= 9]
        total_gain = sum(r["total_gain_w"] for r in cooling_data)
        cooling_totals[case_id] = total_gain / 1000.0  # kWh

        print(f"\n{case_id}:")
        print(f"  Cooling season solar gain: {total_gain / 1000.0:.1f} kWh")

        # By orientation
        by_orient = defaultdict(float)
        for r in cooling_data:
            by_orient[r["orientation"]] += r["total_gain_w"]

        for orient, gain in sorted(by_orient.items()):
            print(f"    {orient}: {gain / 1000.0:.1f} kWh")

    # Calculate ratios
    if "Case 900" in cooling_totals and "Case 920" in cooling_totals:
        south = cooling_totals["Case 900"]
        ew = cooling_totals["Case 920"]
        ratio = ew / south if south > 0 else 0

        print(f"\n{'=' * 80}")
        print("KEY METRIC: E/W to South Ratio (Cooling Season)")
        print(f"{'=' * 80}")
        print(f"  Case 900 (South): {south:.1f} kWh")
        print(f"  Case 920 (E/W):   {ew:.1f} kWh")
        print(f"  Ratio (E/W ÷ South): {ratio:.3f}")
        print("  Expected range: 0.75-0.85")
        print(f"  Deviation: {(ratio - 0.80) / 0.80 * 100:+.1f}% from expected 0.80")

        if ratio > 0.90:
            print("\  ⚠️  WARNING: Ratio is TOO HIGH - E/W solar gain is overestimated!")
            print("     This suggests the model is NOT properly accounting for:")
            print("     - Higher incidence angles on E/W surfaces during summer")
            print("     - Lower SHGC effectiveness at high incidence angles")
            print("     - Potential Perez sky model issues for vertical surfaces")
        elif ratio < 0.70:
            print(
                "\n  ⚠️  WARNING: Ratio is TOO LOW - E/W solar gain is underestimated!"
            )
        else:
            print("\n  ✓ Ratio is within acceptable range")


def analyze_daily_profile(data_by_case, month=7, day=15):
    """Analyze daily solar gain profile for July 15."""
    print(f"\n{'=' * 80}")
    print("DAILY SOLAR GAIN PROFILE - July 15 (Peak Summer)")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        # Extract July 15 data
        july_15 = [r for r in data if r["month"] == month and r["day"] == day]

        if not july_15:
            continue

        # Group by hour and orientation
        by_hour_orient = defaultdict(lambda: defaultdict(float))
        for row in july_15:
            by_hour_orient[int(row["hour"])][row["orientation"]] += row["total_gain_w"]

        print(f"\n{case_id}:")
        print(f"{'Hour':>4} | ", end="")

        orientations = sorted(set(r["orientation"] for r in july_15))
        for orient in orientations:
            print(f"{orient:>10} | ", end="")
        print()
        print("-" * (6 + 13 * len(orientations)))

        for hour in range(24):
            print(f"{hour:4d} | ", end="")
            for orient in orientations:
                gain = by_hour_orient[hour][orient]
                print(f"{gain / 1000.0:10.3f} | ", end="")  # kWh
            print()

        # Daily total
        daily_total = sum(r["total_gain_w"] for r in july_15)
        print(f"\n  Daily total: {daily_total / 1000.0:.2f} kWh")


def analyze_shgc_effectiveness(data_by_case):
    """Analyze SHGC effectiveness vs incidence angle."""
    print(f"\n{'=' * 80}")
    print("SHGC EFFECTIVENESS VS INCIDENCE ANGLE")
    print("=" * 80)

    # Collect all data
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

    # Bin by angle (10-degree bins)
    angle_bins = [(i, i + 10) for i in range(0, 90, 10)]

    print(
        f"\n{'Angle':>10} | {'Count':>6} | {'Avg SHGC':>9} | {'Std Dev':>8} | {'Expected':>9}"
    )
    print("-" * 55)

    # Expected SHGC ratios from ASHRAE 140 table
    expected = {
        (0, 10): 0.995,
        (10, 20): 0.985,
        (20, 30): 0.970,
        (30, 40): 0.940,
        (40, 50): 0.890,
        (50, 60): 0.810,
        (60, 70): 0.680,
        (70, 80): 0.450,
        (80, 90): 0.100,
    }

    for low, high in angle_bins:
        bin_data = [d for d in all_data if low <= d["angle"] < high]
        if bin_data:
            avg_shgc = sum(d["shgc"] for d in bin_data) / len(bin_data)
            variance = sum((d["shgc"] - avg_shgc) ** 2 for d in bin_data) / len(
                bin_data
            )
            std_shgc = variance**0.5
            exp_val = expected.get((low, high), 0.0)

            match = "✓" if abs(avg_shgc - exp_val) < 0.05 else "⚠️"

            print(
                f"{low:3d}-{high:2d}°     {len(bin_data):6d} | {avg_shgc:9.4f} | {std_shgc:8.4f} | {exp_val:9.3f} {match}"
            )


def analyze_beam_diffuse_split(data_by_case):
    """Analyze beam vs diffuse fraction by orientation."""
    print(f"\n{'=' * 80}")
    print("BEAM/DIFFUSE SPLIT BY ORIENTATION (Cooling Season)")
    print("=" * 80)

    for case_id, data in data_by_case.items():
        cooling_data = [
            r for r in data if 5 <= r["month"] <= 9 and r["total_gain_w"] > 0
        ]

        if not cooling_data:
            continue

        print(f"\n{case_id}:")

        # Group by orientation
        by_orient = defaultdict(list)
        for r in cooling_data:
            by_orient[r["orientation"]].append(r)

        for orient, rows in sorted(by_orient.items()):
            total_beam = sum(r["beam_gain_w"] for r in rows)
            total_diffuse = sum(r["diffuse_gain_w"] for r in rows)
            total_ground = sum(r["ground_gain_w"] for r in rows)
            total = total_beam + total_diffuse + total_ground

            if total > 0:
                beam_pct = total_beam / total * 100
                diffuse_pct = total_diffuse / total * 100
                ground_pct = total_ground / total * 100

                print(
                    f"  {orient:10s}: Beam={beam_pct:5.1f}%, Diffuse={diffuse_pct:5.1f}%, Ground={ground_pct:5.1f}%"
                )


def generate_diagnostic_report(
    data_by_case, output_path="/tmp/solar_diagnostics/diagnostic_report.json"
):
    """Generate JSON diagnostic report."""
    report = {
        "summary": {},
        "cooling_season_ratios": {},
        "shgc_analysis": {},
        "recommendations": [],
    }

    # Calculate key metrics
    if "Case 900" in data_by_case and "Case 920" in data_by_case:
        cooling_900 = [r for r in data_by_case["Case 900"] if 5 <= r["month"] <= 9]
        cooling_920 = [r for r in data_by_case["Case 920"] if 5 <= r["month"] <= 9]

        total_900 = sum(r["total_gain_w"] for r in cooling_900)
        total_920 = sum(r["total_gain_w"] for r in cooling_920)

        ratio = total_920 / total_900 if total_900 > 0 else 0

        report["summary"] = {
            "case_900_cooling_kwh": total_900 / 1000.0,
            "case_920_cooling_kwh": total_920 / 1000.0,
            "ew_to_south_ratio": ratio,
            "expected_ratio_min": 0.75,
            "expected_ratio_max": 0.85,
        }

        # Determine root cause hypothesis
        if ratio > 0.90:
            report["recommendations"] = [
                "E/W solar gain is TOO HIGH relative to South",
                "Likely causes:",
                "  1. Incidence angle calculation may be incorrect for vertical E/W surfaces",
                "  2. SHGC angular dependence table may not be aggressive enough",
                "  3. Perez sky model may overestimate diffuse for vertical surfaces",
                "  4. Beam/diffuse split may favor beam too much for E/W orientations",
            ]

            # Analyze which component is most likely at fault
            # Check average incidence angles
            angles_900 = [
                r["incidence_angle"] for r in cooling_900 if r["total_gain_w"] > 0
            ]
            angles_920 = [
                r["incidence_angle"] for r in cooling_920 if r["total_gain_w"] > 0
            ]

            if angles_900 and angles_920:
                avg_angle_900 = sum(angles_900) / len(angles_900)
                avg_angle_920 = sum(angles_920) / len(angles_920)

                report["shgc_analysis"] = {
                    "avg_incidence_angle_south": avg_angle_900,
                    "avg_incidence_angle_ew": avg_angle_920,
                }

                # E/W should have higher average incidence angles in summer
                if avg_angle_920 < avg_angle_900 * 1.1:
                    report["recommendations"].append(
                        f"\n⚠️  CRITICAL: E/W incidence angles (avg {avg_angle_920:.1f}°) are not "
                        f"significantly higher than South ({avg_angle_900:.1f}°)"
                    )
                    report["recommendations"].append(
                        "   This suggests the incidence angle calculation is the ROOT CAUSE"
                    )

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDiagnostic report saved to: {output_path}")
    return report


def main():
    """Main diagnostic analysis."""
    print("=" * 80)
    print("Phase 30 Solar Gain Diagnostic Analysis")
    print("=" * 80)

    # Read diagnostic data
    print("\nReading solar diagnostic data...")
    data_by_case = {}

    data_paths = {
        "Case 900": "/tmp/solar_diagnostics/case_900_solar.csv",
        "Case 920": "/tmp/solar_diagnostics/case_920_solar.csv",
        "Case 930": "/tmp/solar_diagnostics/case_930_solar.csv",
    }

    for case_id, path in data_paths.items():
        data = read_solar_diagnostics(path)
        if data:
            data_by_case[case_id] = data
            print(f"  Loaded {case_id}: {len(data)} records")

    if not data_by_case:
        print("\nNo diagnostic data found!")
        return

    # Run analyses
    analyze_cooling_season_ratio(data_by_case)
    analyze_daily_profile(data_by_case)
    analyze_shgc_effectiveness(data_by_case)
    analyze_beam_diffuse_split(data_by_case)

    # Generate report
    report = generate_diagnostic_report(data_by_case)

    print("\n" + "=" * 80)
    print("Diagnostic analysis complete!")
    print("=" * 80)

    # Print key findings
    if "summary" in report:
        print("\nKEY FINDING:")
        print(f"  E/W to South ratio: {report['summary']['ew_to_south_ratio']:.3f}")
        print(
            f"  Expected: {report['summary']['expected_ratio_min']}-{report['summary']['expected_ratio_max']}"
        )

        if report["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  {rec}")


if __name__ == "__main__":
    main()
