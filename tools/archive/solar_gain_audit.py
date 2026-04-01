#!/usr/bin/env python3
"""
Solar Gain Audit Script for Phase 30: Cooling Energy Fix

This script extracts hourly solar gain profiles from Fluxion simulations
and compares them against expected values to identify underestimation issues.

Usage:
    python tools/solar_gain_audit.py --case 900 --output solar_gain_analysis.csv
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SolarGainData:
    """Container for hourly solar gain data."""

    timestep: int
    month: int
    day: int
    hour: int
    dni: float  # Direct Normal Irradiance (W/m²)
    dhi: float  # Diffuse Horizontal Irradiance (W/m²)
    solar_gain_w: float  # Total solar gain through windows (W)
    zone_temp_c: float  # Zone air temperature (°C)
    hvac_cooling_w: float  # HVAC cooling power (W)


def load_simulation_results(results_path: Path) -> Optional[Dict]:
    """Load simulation results from JSON file."""
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return None

    with open(results_path, "r") as f:
        return json.load(f)


def extract_hourly_data(results: Dict) -> List[SolarGainData]:
    """Extract hourly solar gain data from simulation results."""
    hourly_data = []

    # Check if results contain hourly data
    if "hourly_results" not in results:
        print("Warning: No hourly_results found in simulation output")
        return []

    for timestep, hour_data in enumerate(results["hourly_results"]):
        data = SolarGainData(
            timestep=timestep,
            month=hour_data.get("month", 0),
            day=hour_data.get("day", 0),
            hour=hour_data.get("hour", 0),
            dni=hour_data.get("dni", 0.0),
            dhi=hour_data.get("dhi", 0.0),
            solar_gain_w=hour_data.get("solar_gain_w", 0.0),
            zone_temp_c=hour_data.get("zone_temp_c", 0.0),
            hvac_cooling_w=hour_data.get("hvac_cooling_w", 0.0),
        )
        hourly_data.append(data)

    return hourly_data


def calculate_daily_totals(hourly_data: List[SolarGainData]) -> Dict[int, Dict]:
    """Calculate daily totals for solar gain and cooling energy."""
    daily_totals = {}

    for data in hourly_data:
        # Create day key (assume day of year from timestep)
        day_of_year = (data.timestep // 24) + 1

        if day_of_year not in daily_totals:
            daily_totals[day_of_year] = {
                "month": data.month,
                "day": data.day,
                "solar_gain_kwh": 0.0,
                "cooling_kwh": 0.0,
                "avg_temp_c": 0.0,
                "hours": 0,
            }

        daily_totals[day_of_year]["solar_gain_kwh"] += data.solar_gain_w / 1000.0
        daily_totals[day_of_year]["cooling_kwh"] += data.hvac_cooling_w / 1000.0
        daily_totals[day_of_year]["avg_temp_c"] += data.zone_temp_c
        daily_totals[day_of_year]["hours"] += 1

    # Average temperatures
    for day_data in daily_totals.values():
        if day_data["hours"] > 0:
            day_data["avg_temp_c"] /= day_data["hours"]

    return daily_totals


def analyze_solar_gains(hourly_data: List[SolarGainData], case_id: str):
    """Analyze solar gain patterns and identify issues."""
    print(f"\n{'=' * 60}")
    print(f"SOLAR GAIN ANALYSIS - Case {case_id}")
    print(f"{'=' * 60}\n")

    if not hourly_data:
        print("No data to analyze")
        return

    # Monthly aggregation
    monthly_solar = {}
    monthly_cooling = {}

    for data in hourly_data:
        month = data.month
        if month not in monthly_solar:
            monthly_solar[month] = 0.0
            monthly_cooling[month] = 0.0

        monthly_solar[month] += data.solar_gain_w / 1000.0  # kWh
        monthly_cooling[month] += data.hvac_cooling_w / 1000.0  # kWh

    print("Monthly Solar Gain and Cooling Energy:")
    print(f"{'Month':<10} {'Solar Gain (kWh)':<20} {'Cooling (kWh)':<20}")
    print(f"{'-' * 50}")

    total_solar = 0.0
    total_cooling = 0.0

    for month in sorted(monthly_solar.keys()):
        solar = monthly_solar[month]
        cooling = monthly_cooling[month]
        total_solar += solar
        total_cooling += cooling
        print(f"{month:<10} {solar:<20.2f} {cooling:<20.2f}")

    print(f"{'-' * 50}")
    print(f"{'TOTAL':<10} {total_solar:<20.2f} {total_cooling:<20.2f}")

    # Calculate solar gain ratio
    if total_solar > 0:
        cooling_to_solar_ratio = total_cooling / total_solar
        print(f"\nCooling-to-Solar Ratio: {cooling_to_solar_ratio:.3f}")
        print("  (Higher ratio = more cooling needed per unit solar gain)")

    # Peak solar gain analysis
    max_solar = max(hourly_data, key=lambda x: x.solar_gain_w)
    print("\nPeak Solar Gain:")
    print(
        f"  Timestep: {max_solar.timestep} (Month {max_solar.month}, Day {max_solar.day}, Hour {max_solar.hour})"
    )
    print(f"  Solar Gain: {max_solar.solar_gain_w:.2f} W")
    print(f"  DNI: {max_solar.dni:.2f} W/m²")
    print(f"  DHI: {max_solar.dhi:.2f} W/m²")
    print(f"  Zone Temp: {max_solar.zone_temp_c:.2f} °C")

    # Summer week analysis (typical design week)
    print("\nSummer Week Analysis (Day 180-186, June 29 - July 5):")
    summer_week_solar = 0.0
    summer_week_cooling = 0.0
    for data in hourly_data:
        day_of_year = (data.timestep // 24) + 1
        if 180 <= day_of_year <= 186:
            summer_week_solar += data.solar_gain_w / 1000.0
            summer_week_cooling += data.hvac_cooling_w / 1000.0

    print(f"  Solar Gain: {summer_week_solar:.2f} kWh")
    print(f"  Cooling: {summer_week_cooling:.2f} kWh")


def export_to_csv(hourly_data: List[SolarGainData], output_path: Path):
    """Export hourly data to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestep",
                "month",
                "day",
                "hour",
                "dni_wm2",
                "dhi_wm2",
                "solar_gain_w",
                "zone_temp_c",
                "hvac_cooling_w",
            ]
        )

        for data in hourly_data:
            writer.writerow(
                [
                    data.timestep,
                    data.month,
                    data.day,
                    data.hour,
                    data.dni,
                    data.dhi,
                    data.solar_gain_w,
                    data.zone_temp_c,
                    data.hvac_cooling_w,
                ]
            )

    print(f"\nExported {len(hourly_data)} records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Solar Gain Audit for Phase 30: Cooling Energy Fix"
    )
    parser.add_argument(
        "--case",
        type=str,
        default="900",
        help="ASHRAE 140 case ID (e.g., 900, 920, 930)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("simulation_results.json"),
        help="Path to simulation results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("solar_gain_analysis.csv"),
        help="Output CSV file path",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading simulation results from: {args.results}")
    results = load_simulation_results(args.results)

    if results is None:
        print("Failed to load simulation results. Exiting.")
        return 1

    # Extract hourly data
    hourly_data = extract_hourly_data(results)

    if not hourly_data:
        print("No hourly data found. Exiting.")
        return 1

    # Analyze solar gains
    analyze_solar_gains(hourly_data, args.case)

    # Export to CSV
    export_to_csv(hourly_data, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
