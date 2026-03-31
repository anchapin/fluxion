#!/usr/bin/env python3
"""
Extract EnergyPlus reference data from existing SQL outputs.

This script queries eplusout.sql files to extract hourly data
for comparison with Fluxion ASHRAE 140 validation results.
"""

import json
import sqlite3
import sys
from pathlib import Path


def extract_case_data(sql_path: Path, case_id: str):
    """Extract hourly data from EnergyPlus SQL output for a specific case."""
    print(f"Extracting data for {case_id} from {sql_path}")

    conn = sqlite3.connect(sql_path)
    cursor = conn.cursor()

    # Get available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Available tables: {tables}")

    # Extract hourly data
    hourly_data = {
        "case_id": case_id,
        "hourly": {
            "outdoor_drybulb": [],
            "zone_air_temp": [],
            "heating_energy": [],
            "cooling_energy": [],
            "hours": [],
        },
    }

    # Get outdoor air temperature (Site Sky Temperature is used in these test cases)
    try:
        cursor.execute(
            """
            SELECT ReportDataDictionary.ReportDataDictionaryIndex,
                   Time.Month,
                   Time.Day,
                   Time.Hour,
                   ReportData.Value
            FROM ReportData
            JOIN ReportDataDictionary ON ReportData.ReportDataDictionaryIndex = ReportDataDictionary.ReportDataDictionaryIndex
            JOIN Time ON ReportData.TimeIndex = Time.TimeIndex
            WHERE ReportDataDictionary.Name = 'Site Sky Temperature'
            ORDER BY Time.Month, Time.Day, Time.Hour
        """
        )
        rows = cursor.fetchall()
        for row in rows:
            idx, month, day, hour, value = row
            hour_of_year = (month - 1) * 730 + (day - 1) * 24 + (hour - 1)
            hourly_data["hourly"]["outdoor_drybulb"].append(value)
            if len(hourly_data["hourly"]["hours"]) == 0:
                hourly_data["hourly"]["hours"].append(hour_of_year)
        print(f"Extracted {len(rows)} outdoor temperature values")
    except Exception as e:
        print(f"Warning: Could not extract outdoor temperature: {e}")

    # Get zone air temperature
    try:
        cursor.execute(
            """
            SELECT ReportDataDictionary.ReportDataDictionaryIndex,
                   Time.Month,
                   Time.Day,
                   Time.Hour,
                   ReportData.Value
            FROM ReportData
            JOIN ReportDataDictionary ON ReportData.ReportDataDictionaryIndex = ReportDataDictionary.ReportDataDictionaryIndex
            JOIN Time ON ReportData.TimeIndex = Time.TimeIndex
            WHERE ReportDataDictionary.Name LIKE '%Zone Air Temperature%'
            ORDER BY Time.Month, Time.Day, Time.Hour
        """
        )
        rows = cursor.fetchall()
        for row in rows:
            idx, month, day, hour, value = row
            hourly_data["hourly"]["zone_air_temp"].append(value)
            if len(hourly_data["hourly"]["hours"]) < len(rows):
                hourly_data["hourly"]["hours"].append(
                    (month - 1) * 730 + (day - 1) * 24 + (hour - 1)
                )
        print(f"Extracted {len(rows)} zone temperature values")
    except Exception as e:
        print(f"Warning: Could not extract zone temperature: {e}")

    # Get heating energy
    try:
        cursor.execute(
            """
            SELECT ReportDataDictionary.ReportDataDictionaryIndex,
                   Time.Month,
                   Time.Day,
                   Time.Hour,
                   ReportData.Value
            FROM ReportData
            JOIN ReportDataDictionary ON ReportData.ReportDataDictionaryIndex = ReportDataDictionary.ReportDataDictionaryIndex
            JOIN Time ON ReportData.TimeIndex = Time.TimeIndex
            WHERE ReportDataDictionary.Name = 'DistrictHeatingWater:Facility'
            ORDER BY Time.Month, Time.Day, Time.Hour
        """
        )
        rows = cursor.fetchall()
        for row in rows:
            idx, month, day, hour, value = row
            hourly_data["hourly"]["heating_energy"].append(value if value else 0.0)
            if len(hourly_data["hourly"]["hours"]) < len(rows):
                hourly_data["hourly"]["hours"].append(
                    (month - 1) * 730 + (day - 1) * 24 + (hour - 1)
                )
        print(f"Extracted {len(rows)} heating energy values")
    except Exception as e:
        print(f"Warning: Could not extract heating energy: {e}")

    # Get cooling energy
    try:
        cursor.execute(
            """
            SELECT ReportDataDictionary.ReportDataDictionaryIndex,
                   Time.Month,
                   Time.Day,
                   Time.Hour,
                   ReportData.Value
            FROM ReportData
            JOIN ReportDataDictionary ON ReportData.ReportDataDictionaryIndex = ReportDataDictionary.ReportDataDictionaryIndex
            JOIN Time ON ReportData.TimeIndex = Time.TimeIndex
            WHERE ReportDataDictionary.Name = 'DistrictCooling:Facility'
            ORDER BY Time.Month, Time.Day, Time.Hour
        """
        )
        rows = cursor.fetchall()
        for row in rows:
            idx, month, day, hour, value = row
            hourly_data["hourly"]["cooling_energy"].append(value if value else 0.0)
            if len(hourly_data["hourly"]["hours"]) < len(rows):
                hourly_data["hourly"]["hours"].append(
                    (month - 1) * 730 + (day - 1) * 24 + (hour - 1)
                )
        print(f"Extracted {len(rows)} cooling energy values")
    except Exception as e:
        print(f"Warning: Could not extract cooling energy: {e}")

    conn.close()

    # Calculate annual totals
    annual_heating = sum(hourly_data["hourly"]["heating_energy"])
    annual_cooling = sum(hourly_data["hourly"]["cooling_energy"])

    hourly_data["annual"] = {
        "heating_kwh": annual_heating / 1000.0,  # Convert J to kWh
        "cooling_kwh": annual_cooling / 1000.0,
    }

    print(
        f"Annual totals: Heating={hourly_data['annual']['heating_kwh']:.2f} kWh, Cooling={hourly_data['annual']['cooling_kwh']:.2f} kWh"
    )

    return hourly_data


def extract_all_cases(benchmarks_dir: Path, cases: list[str], output_dir: Path):
    """Extract data for all specified cases from benchmarks directory."""
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    for case_id in cases:
        # Look for SQL file in various possible locations
        sql_paths = [
            benchmarks_dir / f"bestest_gsr/case_{case_id}/run/eplusout.sql",
            benchmarks_dir / f"case_{case_id}/run/eplusout.sql",
        ]

        sql_path = None
        for path in sql_paths:
            if path.exists():
                sql_path = path
                break

        if not sql_path:
            print(f"Warning: No SQL file found for case {case_id}")
            continue

        try:
            data = extract_case_data(sql_path, case_id)
            results[case_id] = data

            # Save JSON output
            json_path = output_dir / f"{case_id}_reference.json"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved reference data to {json_path}")

        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            import traceback

            traceback.print_exc()

    # Save combined results
    combined_path = output_dir / "all_cases_reference.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved combined results to {combined_path}")

    return results


def main():
    benchmarks_dir = Path("benchmarks/outputs")
    output_dir = Path("refdata/energyplus_reference")

    # Cases with discrepancies from Phase 8 validation
    discrepancy_cases = [
        "600",
        "610",
        "620",
        "630",
        "640",
        "650",
        "900",
        "910",
        "920",
        "930",
        "940",
        "950",
    ]

    # Also include free-floating cases
    all_cases = discrepancy_cases + ["600FF", "900FF"]

    print("Extracting EnergyPlus reference data for ASHRAE 140 validation cases...")
    print(f"Cases: {all_cases}")
    print(f"Output directory: {output_dir}")
    print()

    results = extract_all_cases(benchmarks_dir, all_cases, output_dir)

    print("\n" + "=" * 60)
    print("Summary of Extracted Data:")
    print("=" * 60)
    for case_id in all_cases:
        if case_id in results:
            data = results[case_id]
            annual = data["annual"]
            print(
                f"{case_id}: H={annual['heating_kwh']:.2f} kWh, C={annual['cooling_kwh']:.2f} kWh"
            )
    print()


if __name__ == "__main__":
    main()
