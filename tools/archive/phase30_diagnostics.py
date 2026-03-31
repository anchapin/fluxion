#!/usr/bin/env python3
"""
Phase 30 Diagnostic Analysis Script

This script analyzes hourly simulation data to understand why cooling energy
is underestimated, especially for E/W window cases (920, 930) vs S-facing cases (900).

Usage:
    python tools/phase30_diagnostics.py --case 900 --output diagnostic_900.csv
    python tools/phase30_diagnostics.py --case 920 --output diagnostic_920.csv
    python tools/phase30_diagnostics.py --compare 900 920 930 --output comparison.csv
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class HourlyDiagnosticData:
    """Container for hourly diagnostic data."""

    timestep: int
    month: int
    day: int
    hour: int

    # Weather
    dry_bulb_temp_c: float = 0.0
    dni: float = 0.0  # Direct Normal Irradiance (W/m²)
    dhi: float = 0.0  # Diffuse Horizontal Irradiance (W/m²)

    # Solar gains
    solar_gain_w: float = 0.0  # Total solar gain through windows (W)
    opaque_irradiance_wm2: float = 0.0  # Solar irradiance on opaque surfaces (W/m²)

    # Temperatures
    zone_temp_c: float = 0.0
    mass_temp_c: float = 0.0
    outdoor_temp_c: float = 0.0
    sol_air_temp_c: float = 0.0

    # HVAC
    hvac_cooling_w: float = 0.0  # HVAC cooling power (W)
    hvac_heating_w: float = 0.0  # HVAC heating power (W)
    hvac_runtime: bool = False  # Is HVAC running?

    # Loads
    internal_gains_w: float = 0.0
    envelope_conduction_w: float = 0.0

    # Energy (cumulative)
    cumulative_cooling_kwh: float = 0.0
    cumulative_heating_kwh: float = 0.0
    cumulative_solar_kwh: float = 0.0


@dataclass
class CaseComparison:
    """Container for comparing multiple cases."""

    case_id: str
    annual_cooling_mwh: float
    annual_heating_mwh: float
    peak_cooling_w: float
    peak_heating_w: float

    # Summer week averages (typical design week)
    summer_week_avg_solar_w: float = 0.0
    summer_week_avg_cooling_w: float = 0.0
    summer_week_avg_zone_temp_c: float = 0.0
    summer_week_avg_mass_temp_c: float = 0.0

    # Solar gain distribution
    total_solar_kwh: float = 0.0
    solar_to_air_fraction: float = 0.0
    solar_to_mass_fraction: float = 0.0


def timestep_to_date(timestep: int) -> Tuple[int, int, int, int]:
    """Convert timestep (0-8759) to (year, month, day, hour)."""
    # Assume non-leap year starting Jan 1
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    total_hours = timestep
    day_of_year = total_hours // 24
    hour = total_hours % 24

    month = 0
    day = day_of_year
    for m, days in enumerate(days_in_month):
        if day < days:
            month = m + 1
            day += 1  # 1-indexed
            break
        day -= days

    return (2024, month, day, hour)


def analyze_summer_week(data: List[HourlyDiagnosticData]) -> Dict:
    """Analyze a typical summer design week (July 24-30, days 205-211)."""
    summer_week_data = []

    for row in data:
        # Day of year for July 24-30
        # Jan(31) + Feb(28) + Mar(31) + Apr(30) + May(31) + Jun(30) = 181
        # July 24 = 181 + 24 = 205
        day_of_year = ((row.month - 1) * 30) + row.day  # Approximate
        if 205 <= day_of_year <= 211:
            summer_week_data.append(row)

    if not summer_week_data:
        return {}

    return {
        "avg_solar_w": sum(r.solar_gain_w for r in summer_week_data)
        / len(summer_week_data),
        "avg_cooling_w": sum(r.hvac_cooling_w for r in summer_week_data)
        / len(summer_week_data),
        "avg_zone_temp_c": sum(r.zone_temp_c for r in summer_week_data)
        / len(summer_week_data),
        "avg_mass_temp_c": sum(r.mass_temp_c for r in summer_week_data)
        / len(summer_week_data),
        "peak_solar_w": max(r.solar_gain_w for r in summer_week_data),
        "peak_cooling_w": max(r.hvac_cooling_w for r in summer_week_data),
        "hours_cooling_active": sum(
            1 for r in summer_week_data if r.hvac_cooling_w > 0
        ),
    }


def analyze_daily_patterns(data: List[HourlyDiagnosticData], case_id: str):
    """Analyze daily patterns in solar gain and cooling."""
    print(f"\n{'=' * 80}")
    print(f"DAILY PATTERN ANALYSIS - Case {case_id}")
    print(f"{'=' * 80}\n")

    # Group by hour of day
    hourly_averages = {}
    for row in data:
        hour = row.hour
        if hour not in hourly_averages:
            hourly_averages[hour] = {
                "solar_w": [],
                "cooling_w": [],
                "zone_temp_c": [],
                "mass_temp_c": [],
                "outdoor_temp_c": [],
            }
        hourly_averages[hour]["solar_w"].append(row.solar_gain_w)
        hourly_averages[hour]["cooling_w"].append(row.hvac_cooling_w)
        hourly_averages[hour]["zone_temp_c"].append(row.zone_temp_c)
        hourly_averages[hour]["mass_temp_c"].append(row.mass_temp_c)
        hourly_averages[hour]["outdoor_temp_c"].append(row.outdoor_temp_c)

    print(
        f"{'Hour':<6} {'Avg Solar (W)':<15} {'Avg Cooling (W)':<18} {'Avg Zone (°C)':<15} {'Avg Mass (°C)':<15} {'Avg Outdoor (°C)':<18}"
    )
    print(f"{'-' * 90}")

    for hour in sorted(hourly_averages.keys()):
        avg_solar = sum(hourly_averages[hour]["solar_w"]) / len(
            hourly_averages[hour]["solar_w"]
        )
        avg_cooling = sum(hourly_averages[hour]["cooling_w"]) / len(
            hourly_averages[hour]["cooling_w"]
        )
        avg_zone = sum(hourly_averages[hour]["zone_temp_c"]) / len(
            hourly_averages[hour]["zone_temp_c"]
        )
        avg_mass = sum(hourly_averages[hour]["mass_temp_c"]) / len(
            hourly_averages[hour]["mass_temp_c"]
        )
        avg_outdoor = sum(hourly_averages[hour]["outdoor_temp_c"]) / len(
            hourly_averages[hour]["outdoor_temp_c"]
        )

        print(
            f"{hour:<6} {avg_solar:<15.2f} {avg_cooling:<18.2f} {avg_zone:<15.2f} {avg_mass:<15.2f} {avg_outdoor:<18.2f}"
        )

    # Find peak hours
    peak_solar_hour = max(
        hourly_averages.keys(),
        key=lambda h: sum(hourly_averages[h]["solar_w"])
        / len(hourly_averages[h]["solar_w"]),
    )
    peak_cooling_hour = max(
        hourly_averages.keys(),
        key=lambda h: sum(hourly_averages[h]["cooling_w"])
        / len(hourly_averages[h]["cooling_w"]),
    )

    print(f"\nPeak Solar Gain Hour: {peak_solar_hour}:00")
    print(f"Peak Cooling Demand Hour: {peak_cooling_hour}:00")

    # Check for lag between solar peak and cooling peak
    lag_hours = (peak_cooling_hour - peak_solar_hour) % 24
    print(f"Thermal Lag (solar peak to cooling peak): {lag_hours} hours")

    return hourly_averages


def analyze_orientation_effect(cases_data: Dict[str, List[HourlyDiagnosticData]]):
    """Compare solar gain patterns between different window orientations."""
    print(f"\n{'=' * 80}")
    print(f"WINDOW ORIENTATION EFFECT ANALYSIS")
    print(f"{'=' * 80}\n")

    # Case 900: South windows
    # Case 920: East + West windows
    # Case 930: East + West + South windows

    print(
        f"{'Case':<8} {'Windows':<25} {'Annual Cool (MWh)':<20} {'Summer Avg Solar (W)':<20} {'Peak Solar (W)':<15}"
    )
    print(f"{'-' * 90}")

    for case_id, data in sorted(cases_data.items()):
        summer = analyze_summer_week(data)
        annual_cooling = sum(r.hvac_cooling_w for r in data) / 1e6  # Convert Wh to MWh

        windows = {
            "900": "South (12 m²)",
            "920": "East + West (6 m² each)",
            "930": "E + W + S (6+6+12 m²)",
        }.get(case_id, "Unknown")

        peak_solar = max(r.solar_gain_w for r in data) if data else 0

        print(
            f"{case_id:<8} {windows:<25} {annual_cooling:<20.3f} {summer.get('avg_solar_w', 0):<20.2f} {peak_solar:<15.2f}"
        )

    print(f"\n{'=' * 80}")
    print(f"KEY INSIGHTS:")
    print(f"{'=' * 80}")

    # Compare 900 (South) vs 920 (E+W)
    if "900" in cases_data and "920" in cases_data:
        data_900 = cases_data["900"]
        data_920 = cases_data["920"]

        summer_900 = analyze_summer_week(data_900)
        summer_920 = analyze_summer_week(data_920)

        print(f"\n1. South vs East/West Solar Gain:")
        print(
            f"   - Case 900 (South): Avg summer solar = {summer_900.get('avg_solar_w', 0):.2f} W"
        )
        print(
            f"   - Case 920 (E+W): Avg summer solar = {summer_920.get('avg_solar_w', 0):.2f} W"
        )

        # Morning vs afternoon analysis
        morning_solar_920 = sum(
            r.solar_gain_w for r in data_920 if 6 <= r.hour <= 10
        ) / max(1, sum(1 for r in data_920 if 6 <= r.hour <= 10))
        afternoon_solar_920 = sum(
            r.solar_gain_w for r in data_920 if 14 <= r.hour <= 18
        ) / max(1, sum(1 for r in data_920 if 14 <= r.hour <= 18))

        print(f"\n2. East/West Asymmetry (Case 920):")
        print(f"   - Morning solar (6-10h, East): {morning_solar_920:.2f} W")
        print(f"   - Afternoon solar (14-18h, West): {afternoon_solar_920:.2f} W")
        print(
            f"   - Afternoon/Morning ratio: {afternoon_solar_920 / max(1, morning_solar_920):.2f}"
        )

        print(f"\n3. Cooling Response:")
        print(f"   - Case 900 cooling: {summer_900.get('avg_cooling_w', 0):.2f} W avg")
        print(f"   - Case 920 cooling: {summer_920.get('avg_cooling_w', 0):.2f} W avg")

        # Thermal lag comparison
        hourly_900 = {}
        hourly_920 = {}
        for row in data_900:
            if row.hour not in hourly_900:
                hourly_900[row.hour] = []
            hourly_900[row.hour].append(row.solar_gain_w)
        for row in data_920:
            if row.hour not in hourly_920:
                hourly_920[row.hour] = []
            hourly_920[row.hour].append(row.solar_gain_w)

        peak_hour_900 = max(
            hourly_900.keys(), key=lambda h: sum(hourly_900[h]) / len(hourly_900[h])
        )
        peak_hour_920 = max(
            hourly_920.keys(), key=lambda h: sum(hourly_920[h]) / len(hourly_920[h])
        )

        print(f"\n4. Peak Solar Timing:")
        print(f"   - Case 900 (South): Peak at {peak_hour_900}:00")
        print(f"   - Case 920 (E+W): Peak at {peak_hour_920}:00")


def export_to_csv(data: List[HourlyDiagnosticData], output_path: Path):
    """Export hourly data to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestep",
                "month",
                "day",
                "hour",
                "dry_bulb_temp_c",
                "dni_wm2",
                "dhi_wm2",
                "solar_gain_w",
                "opaque_irradiance_wm2",
                "zone_temp_c",
                "mass_temp_c",
                "outdoor_temp_c",
                "sol_air_temp_c",
                "hvac_cooling_w",
                "hvac_heating_w",
                "hvac_runtime",
                "internal_gains_w",
                "envelope_conduction_w",
                "cumulative_cooling_kwh",
                "cumulative_heating_kwh",
                "cumulative_solar_kwh",
            ]
        )

        for row in data:
            writer.writerow(
                [
                    row.timestep,
                    row.month,
                    row.day,
                    row.hour,
                    row.dry_bulb_temp_c,
                    row.dni,
                    row.dhi,
                    row.solar_gain_w,
                    row.opaque_irradiance_wm2,
                    row.zone_temp_c,
                    row.mass_temp_c,
                    row.outdoor_temp_c,
                    row.sol_air_temp_c,
                    row.hvac_cooling_w,
                    row.hvac_heating_w,
                    row.hvac_runtime,
                    row.internal_gains_w,
                    row.envelope_conduction_w,
                    row.cumulative_cooling_kwh,
                    row.cumulative_heating_kwh,
                    row.cumulative_solar_kwh,
                ]
            )

    print(f"\nExported {len(data)} records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 30 Diagnostic Analysis for Cooling Energy Fix"
    )
    parser.add_argument(
        "--case", type=str, help="ASHRAE 140 case ID (e.g., 900, 920, 930)"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        help="Compare multiple cases (e.g., --compare 900 920 930)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase30_diagnostic.csv"),
        help="Output CSV file path",
    )

    args = parser.parse_args()

    if args.compare:
        # Comparison mode
        print(f"Comparing cases: {args.compare}")
        cases_data = {}

        for case_id in args.compare:
            # In a real implementation, we would load simulation data here
            # For now, we'll create a placeholder
            print(f"  Loading case {case_id}...")
            cases_data[case_id] = []

        analyze_orientation_effect(cases_data)

    elif args.case:
        # Single case mode
        print(f"Analyzing case {args.case}...")

        # In a real implementation, we would load simulation data here
        # For now, create placeholder data
        data = []

        analyze_daily_patterns(data, args.case)
        export_to_csv(data, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    exit(main())
