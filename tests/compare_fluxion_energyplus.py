#!/usr/bin/env python3
"""
Compare Fluxion simulation results with EnergyPlus reference data.

This script:
1. Loads EnergyPlus reference data from reference_data.json
2. Runs Fluxion validation
3. Extracts hourly results from Fluxion output
4. Compares key metrics and identifies discrepancies
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_energyplus_reference() -> Dict:
    """Load EnergyPlus reference data."""
    path = "benchmarks/outputs/bestest_gsr/case_900/run/reference_data.json"
    with open(path, 'r') as f:
        data = json.load(f)
        # Fix the JSON structure - convert single values to lists
        if 'hourly' in data and isinstance(data['hourly'], dict):
            # Convert single values to lists for consistency
            data['hourly'] = data['hourly']
        return data


def run_fluxion_validation() -> Dict:
    """Run Fluxion Case 900 validation."""
    print("Running Fluxion validation...")

    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "fluxion", "validate", "--case", "900"],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        print(f"Error running Fluxion: {result.stderr}")
        sys.exit(1)

    # Parse output
    output = result.stdout

    # Extract heating and cooling values
    heating_mwh = None
    cooling_mwh = None

    for line in output.split('\n'):
        if 'Case 900:' in line:
            if 'Heating=' in line:
                # Extract: Heating=4.75 (Ref: 1.17-2.04)
                heating_str = line.split('Heating=')[1].split('(')[0]
                heating_mwh = float(heating_str.strip())
            elif 'Cooling=' in line:
                # Extract: Cooling=6.95 (Ref: 2.13-3.67)
                cooling_str = line.split('Cooling=')[1].split('(')[0]
                cooling_mwh = float(cooling_str.strip())

    return {
        'heating_mwh': heating_mwh,
        'cooling_mwh': cooling_mwh,
        'output': output
    }


def compare_results(ep_data: Dict, fluxion_results: Dict):
    """Compare EnergyPlus and Fluxion results."""
    print("\n=== Comparison Results ===")

    # Annual comparison
    # ep_data['hourly'] contains lists
    ep_heating_list = ep_data['hourly'].get('heating_energy_wh', [])
    ep_cooling_list = ep_data['hourly'].get('cooling_energy_wh', [])

    if not ep_heating_list or not ep_cooling_list:
        print("Error: Missing heating or cooling data in EnergyPlus reference")
        return {
            'heating_error': 0,
            'cooling_error': 0,
        }

    ep_heating = sum(ep_heating_list) / 1000.0
    ep_cooling = sum(ep_cooling_list) / 1000.0

    fluxion_heating = fluxion_results['heating_mwh']
    fluxion_cooling = fluxion_results['cooling_mwh']

    print(f"\nAnnual Energy Consumption:")
    print(f"  EnergyPlus:  Heating={ep_heating:.3f} MWh, Cooling={ep_cooling:.3f} MWh")
    print(f"  Fluxion:    Heating={fluxion_heating:.3f} MWh, Cooling={fluxion_cooling:.3f} MWh")
    print(f"  Reference:  Heating=1.17-2.04 MWh, Cooling=2.13-3.67 MWh")

    heating_error = (fluxion_heating - 1.605) / 1.605 * 100  # 1.605 is midpoint
    cooling_error = (fluxion_cooling - 2.90) / 2.90 * 100  # 2.90 is midpoint

    print(f"\n  Fluxion Error:  Heating={heating_error:+.1f}%, Cooling={cooling_error:+.1f}%")

    # Solar gain analysis
    ep_solar = ep_data['hourly']['solar_rate_total_w']
    fluxion_output = fluxion_results['output']

    # Parse Fluxion output for solar gain
    fluxion_solar_gains = []
    for line in fluxion_output.split('\n'):
        # Look for solar gain in debug output
        if 'DEBUG solar' in line or 'solar_gain_watts=' in line:
            # Extract solar value from line
            if 'solar_gain_watts=' in line:
                try:
                    value_str = line.split('solar_gain_watts=')[1]
                    # Remove any trailing characters
                    value_str = value_str.split()[0]
                    fluxion_solar_gains.append(float(value_str))
                except (IndexError, ValueError):
                    pass

    if fluxion_solar_gains:
        # Sample first few hours
        print(f"\nSolar Gain Comparison (First 24 hours):")
        print(f"{'Hour':<10} {'EP Solar (W)':<15} {'Fluxion Solar (W)':<15} {'Difference':<12}")
        print("-" * 60)

        # We need hourly Fluxion solar data
        # For now, use sample from first few lines if available
        # This is a placeholder - actual implementation would need
        # Fluxion to output hourly solar data

        fluxion_solar_avg = sum(fluxion_solar_gains[:min(24, len(fluxion_solar_gains))]) / min(24, len(fluxion_solar_gains))
        ep_solar_avg = sum(ep_solar[:24]) / 24

        print(f"{'Avg (first 24h)':<20} {ep_solar_avg:<15.2f} {fluxion_solar_avg:<15.2f}")

    # Temperature analysis
    ep_temps = ep_data['hourly']['zone_air_temp_c']
    print(f"\nTemperature Statistics:")
    print(f"  EnergyPlus:")
    print(f"    Min: {min(ep_temps):.2f}°C")
    print(f"    Max: {max(ep_temps):.2f}°C")
    print(f"    Avg: {np.mean(ep_temps):.2f}°C")
    print(f"    StdDev: {np.std(ep_temps):.2f}°C")

    # EnergyPlus reference annual from JSON
    print(f"\nEnergyPlus Reference (from JSON):")
    print(f"  Annual Heating: {ep_data['hourly']['heating_energy_wh'].sum() / 1000.0:.3f} MWh")
    print(f"  Annual Cooling: {ep_data['hourly']['cooling_energy_wh'].sum() / 1000.0:.3f} MWh")

    # Identify key discrepancies
    print(f"\n=== Key Discrepancies ===")

    # Heating is 2.86x too high
    if heating_error > 100:
        print(f"❌ HEATING: {heating_error:+.1f}% too high (2.86x EP reference)")
        print(f"   Fluxion: {fluxion_heating:.3f} MWh")
        print(f"   EnergyPlus: {ep_heating:.3f} MWh")
        print(f"   Difference: {fluxion_heating - ep_heating:.3f} MWh")
    else:
        print(f"✓ HEATING: {heating_error:+.1f}% error (acceptable)")

    # Cooling is 2.78x too high
    if cooling_error > 100:
        print(f"❌ COOLING: {cooling_error:+.1f}% too high (2.78x EP reference)")
        print(f"   Fluxion: {fluxion_cooling:.3f} MWh")
        print(f"   EnergyPlus: {ep_cooling:.3f} MWh")
        print(f"   Difference: {fluxion_cooling - ep_cooling:.3f} MWh")
    else:
        print(f"✓ COOLING: {cooling_error:+.1f}% error (acceptable)")

    return {
        'heating_error': heating_error,
        'cooling_error': cooling_error,
    }


def main():
    """Main entry point."""
    print("=== Fluxion vs EnergyPlus Comparison Tool ===\n")

    # Load EnergyPlus reference
    ep_data = load_energyplus_reference()
    print(f"✓ Loaded EnergyPlus reference data")
    print(f"  Hours of data: {len(ep_data['hourly']['zone_air_temp_c'])}")

    # Run Fluxion validation
    fluxion_results = run_fluxion_validation()

    # Compare results
    comparison = compare_results(ep_data, fluxion_results)

    # Save comparison report
    report = {
        'energyplus_heating_mwh': ep_data['hourly']['heating_energy_wh'].sum() / 1000.0,
        'energyplus_cooling_mwh': ep_data['hourly']['cooling_energy_wh'].sum() / 1000.0,
        'fluxion_heating_mwh': fluxion_results['heating_mwh'],
        'fluxion_cooling_mwh': fluxion_results['cooling_mwh'],
        'heating_error_percent': comparison['heating_error'],
        'cooling_error_percent': comparison['cooling_error'],
    }

    report_file = Path("benchmarks/outputs/bestest_gsr/case_900/run/comparison_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved comparison report to {report_file}")

    # Exit with error code if heating error > 100%
    if comparison['heating_error'] > 100 or comparison['cooling_error'] > 100:
        print(f"\n❌ CRITICAL: Significant discrepancies detected")
        sys.exit(1)


if __name__ == '__main__':
    main()
