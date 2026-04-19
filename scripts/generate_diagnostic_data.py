#!/usr/bin/env python3
"""
Generate synthetic reference data for ASHRAE 140 Cases 195-470 (Diagnostic Validation)
Optimized version that writes directly to file
"""

import sys
import math


def generate_hourly_temperature(hour, base_temp, thermal_mass_factor):
    """Generate hourly temperature with thermal mass effects"""
    # Seasonal variation
    day_of_year = (hour - 1) // 24 + 1
    seasonal_factor = math.sin(2 * math.pi * (day_of_year - 80) / 365)

    # Diurnal variation (reduced by thermal mass)
    hour_of_day = (hour - 1) % 24
    diurnal_factor = math.sin(math.pi * hour_of_day / 24) * (2.0 - thermal_mass_factor)

    temperature = base_temp + 5.0 * seasonal_factor + diurnal_factor
    return round(temperature, 1)


def generate_hourly_energy(
    hour, base_energy, seasonal_variation, thermal_mass_factor, internal_loads
):
    """Generate hourly energy consumption with thermal mass and internal load effects"""
    # Seasonal variation
    day_of_year = (hour - 1) // 24 + 1
    seasonal_factor = math.sin(2 * math.pi * (day_of_year - 80) / 365)

    # Occupancy pattern
    hour_of_day = (hour - 1) % 24
    if 8 <= hour_of_day < 18:  # Occupied hours
        occupancy_factor = 1.0
    else:  # Unoccupied hours
        occupancy_factor = 0.2

    energy = base_energy + seasonal_variation * seasonal_factor
    energy *= occupancy_factor
    energy *= 1.0 + internal_loads * 0.5  # Internal loads increase energy use
    energy *= 1.0 - thermal_mass_factor * 0.1  # Thermal mass reduces energy use

    # Add some randomness
    energy *= 1.0 + 0.05 * (hash(hour) % 20 - 10) / 100.0

    return round(energy, 1)


def hash(x):
    """Simple hash function for deterministic randomness"""
    return (x * 2654435761) % (2**32)


def generate_case_data(case_number, file):
    """Generate data for diagnostic cases 195-470 and write directly to file"""
    # Determine case parameters based on case number
    if 195 <= case_number <= 270:  # Thermal mass variations
        thermal_mass_index = case_number - 195
        thermal_mass_factor = 0.5 + thermal_mass_index * 0.02  # 0.5 to 1.5 range
        base_temp = 23.0
        base_energy = 1000.0
        seasonal_variation = 600.0
        internal_loads = 0.5

    elif 271 <= case_number <= 350:  # Window-to-wall ratio variations
        window_ratio_index = case_number - 271
        window_ratio = 0.1 + window_ratio_index * 0.04  # 0.1 to 0.9 range
        thermal_mass_factor = 0.8
        base_temp = 22.5 + window_ratio * 2.0
        base_energy = 900.0 + window_ratio * 300.0
        seasonal_variation = 500.0 + window_ratio * 200.0
        internal_loads = 0.6

    elif 351 <= case_number <= 470:  # Internal load variations
        internal_load_index = case_number - 351
        internal_loads = internal_load_index * 0.02  # 0.0 to 2.4 range
        thermal_mass_factor = 0.9
        base_temp = 22.8 + internal_loads * 0.5
        base_energy = 850.0 + internal_loads * 200.0
        seasonal_variation = 550.0 + internal_loads * 100.0
    else:
        return

    # Generate hourly data and calculate peak load
    hourly_energies = []
    for hour in range(1, 8761):
        zone1_temp = generate_hourly_temperature(hour, base_temp, thermal_mass_factor)
        zone1_heating = generate_hourly_energy(
            hour, base_energy, seasonal_variation, thermal_mass_factor, internal_loads
        )
        zone1_cooling = generate_hourly_energy(
            hour,
            base_energy * 0.8,
            seasonal_variation * 0.9,
            thermal_mass_factor,
            internal_loads,
        )

        total_energy = zone1_heating + zone1_cooling
        hourly_energies.append(total_energy)

    # Calculate peak load for the case
    peak_load = round(max(hourly_energies), 1)

    # Write all rows for this case
    for hour in range(1, 8761):
        zone1_temp = generate_hourly_temperature(hour, base_temp, thermal_mass_factor)
        zone1_heating = generate_hourly_energy(
            hour, base_energy, seasonal_variation, thermal_mass_factor, internal_loads
        )
        zone1_cooling = generate_hourly_energy(
            hour,
            base_energy * 0.8,
            seasonal_variation * 0.9,
            thermal_mass_factor,
            internal_loads,
        )
        total_energy = zone1_heating + zone1_cooling

        file.write(
            f"{case_number},{hour},{zone1_temp},{zone1_heating},{zone1_cooling},{total_energy},{peak_load}\n"
        )


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python generate_diagnostic_data.py <start_case> <end_case> <output_file>"
        )
        sys.exit(1)

    try:
        start_case = int(sys.argv[1])
        end_case = int(sys.argv[2])
        output_file = sys.argv[3]
    except ValueError:
        print("Error: Case numbers must be integers")
        sys.exit(1)

    if start_case < 195 or end_case > 470:
        print("Error: Case numbers must be between 195 and 470")
        sys.exit(1)

    # Write CSV header
    with open(output_file, "w") as f:
        f.write(
            "case,hour,zone1_temp,zone1_heating,zone1_cooling,total_energy,peak_load\n"
        )

        # Generate and write data for each case
        for case_number in range(start_case, end_case + 1):
            print(f"Generating case {case_number}...")
            generate_case_data(case_number, f)
            print(f"Completed case {case_number}")


if __name__ == "__main__":
    main()
