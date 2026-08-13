#!/usr/bin/env python3
"""
Generate synthetic reference data for ASHRAE 140 Cases 800-810 (HVAC Equipment)
"""

import math
import sys


def generate_hourly_temperature(hour, base_temp, seasonal_variation):
    """Generate hourly temperature with seasonal and diurnal variations"""
    # Seasonal variation (winter vs summer)
    day_of_year = (hour - 1) // 24 + 1
    seasonal_factor = math.sin(
        2 * math.pi * (day_of_year - 80) / 365
    )  # Winter low, summer high

    # Diurnal variation (day vs night)
    hour_of_day = (hour - 1) % 24
    diurnal_factor = math.sin(math.pi * hour_of_day / 24)  # Night low, day high

    temperature = (
        base_temp + seasonal_variation * seasonal_factor + 2.0 * diurnal_factor
    )
    return round(temperature, 1)


def generate_hourly_energy(hour, base_energy, seasonal_variation, occupancy_pattern):
    """Generate hourly energy consumption with occupancy patterns"""
    # Seasonal variation
    day_of_year = (hour - 1) // 24 + 1
    seasonal_factor = math.sin(2 * math.pi * (day_of_year - 80) / 365)

    # Occupancy pattern (8am-6pm = occupied)
    hour_of_day = (hour - 1) % 24
    if 8 <= hour_of_day < 18:  # Occupied hours
        occupancy_factor = occupancy_pattern
    else:  # Unoccupied hours
        occupancy_factor = 0.2

    energy = base_energy + seasonal_variation * seasonal_factor
    energy *= occupancy_factor

    # Add some randomness
    energy *= 1.0 + 0.1 * (hash(hour) % 20 - 10) / 100.0

    return round(energy, 1)


def hash(x):
    """Simple hash function for deterministic randomness"""
    return (x * 2654435761) % (2**32)


def generate_case_800_data():
    """Generate data for Case 800 (Base HVAC equipment case)"""
    data = []
    for hour in range(1, 8761):
        # Zone 1 data
        zone1_temp = generate_hourly_temperature(hour, 21.0, 5.0)
        zone1_heating = generate_hourly_energy(hour, 1200.0, 800.0, 1.0)
        zone1_cooling = generate_hourly_energy(hour, 1300.0, 900.0, 1.0)

        # Zone 2 data
        zone2_temp = generate_hourly_temperature(hour, 20.5, 4.5)
        zone2_heating = generate_hourly_energy(hour, 1100.0, 700.0, 1.0)
        zone2_cooling = generate_hourly_energy(hour, 1200.0, 800.0, 1.0)

        total_energy = zone1_heating + zone1_cooling + zone2_heating + zone2_cooling

        data.append(
            {
                "case": 800,
                "hour": hour,
                "zone1_temp": zone1_temp,
                "zone1_heating": zone1_heating,
                "zone1_cooling": zone1_cooling,
                "zone2_temp": zone2_temp,
                "zone2_heating": zone2_heating,
                "zone2_cooling": zone2_cooling,
                "total_energy": total_energy,
            }
        )

    return data


def generate_case_data(case_number):
    """Generate data for HVAC equipment cases 801-810"""
    data = []

    # Base parameters that vary by case
    if case_number == 801:  # Variable speed HVAC
        base_temp = 22.0
        temp_variation = 4.0
        base_energy = 900.0
        energy_variation = 600.0
        efficiency = 0.8
    elif case_number == 802:  # High efficiency HVAC
        base_temp = 21.5
        temp_variation = 4.2
        base_energy = 800.0
        energy_variation = 500.0
        efficiency = 0.9
    elif case_number == 803:  # Heat pump system
        base_temp = 21.8
        temp_variation = 4.8
        base_energy = 1000.0
        energy_variation = 700.0
        efficiency = 0.85
    elif case_number == 804:  # Radiant heating/cooling
        base_temp = 20.8
        temp_variation = 3.5
        base_energy = 1100.0
        energy_variation = 750.0
        efficiency = 0.82
    elif case_number == 805:  # DOAS with FCU
        base_temp = 21.2
        temp_variation = 4.1
        base_energy = 950.0
        energy_variation = 650.0
        efficiency = 0.87
    elif case_number == 806:  # VRF system
        base_temp = 21.6
        temp_variation = 4.3
        base_energy = 850.0
        energy_variation = 550.0
        efficiency = 0.92
    elif case_number == 807:  # Chilled beam system
        base_temp = 20.9
        temp_variation = 3.8
        base_energy = 1050.0
        energy_variation = 720.0
        efficiency = 0.84
    elif case_number == 808:  # Geothermal heat pump
        base_temp = 21.4
        temp_variation = 4.0
        base_energy = 750.0
        energy_variation = 450.0
        efficiency = 0.95
    elif case_number == 809:  # Solar assisted HVAC
        base_temp = 21.7
        temp_variation = 4.4
        base_energy = 920.0
        energy_variation = 620.0
        efficiency = 0.88
    elif case_number == 810:  # Hybrid system
        base_temp = 21.1
        temp_variation = 4.2
        base_energy = 880.0
        energy_variation = 580.0
        efficiency = 0.90
    else:
        return data

    for hour in range(1, 8761):
        # Zone 1 data
        zone1_temp = generate_hourly_temperature(hour, base_temp, temp_variation)
        zone1_heating = (
            generate_hourly_energy(hour, base_energy, energy_variation, 1.0)
            * efficiency
        )
        zone1_cooling = (
            generate_hourly_energy(hour, base_energy + 100, energy_variation, 1.0)
            * efficiency
        )

        # Zone 2 data
        zone2_temp = generate_hourly_temperature(
            hour, base_temp - 0.5, temp_variation - 0.3
        )
        zone2_heating = (
            generate_hourly_energy(hour, base_energy - 50, energy_variation - 50, 1.0)
            * efficiency
        )
        zone2_cooling = (
            generate_hourly_energy(hour, base_energy + 50, energy_variation - 50, 1.0)
            * efficiency
        )

        total_energy = zone1_heating + zone1_cooling + zone2_heating + zone2_cooling

        data.append(
            {
                "case": case_number,
                "hour": hour,
                "zone1_temp": zone1_temp,
                "zone1_heating": zone1_heating,
                "zone1_cooling": zone1_cooling,
                "zone2_temp": zone2_temp,
                "zone2_heating": zone2_heating,
                "zone2_cooling": zone2_cooling,
                "total_energy": total_energy,
            }
        )

    return data


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_reference_data.py <start_case> <end_case>")
        sys.exit(1)

    try:
        start_case = int(sys.argv[1])
        end_case = int(sys.argv[2])
    except ValueError:
        print("Error: Case numbers must be integers")
        sys.exit(1)

    if start_case < 800 or end_case > 810:
        print("Error: Case numbers must be between 800 and 810")
        sys.exit(1)

    # Write CSV header
    print(
        "case,hour,zone1_temp,zone1_heating,zone1_cooling,zone2_temp,zone2_heating,zone2_cooling,total_energy"
    )

    # Generate and write data for each case
    for case_number in range(start_case, end_case + 1):
        if case_number == 800:
            data = generate_case_800_data()
        else:
            data = generate_case_data(case_number)

        for row in data:
            print(
                f"{row['case']},{row['hour']},{row['zone1_temp']},{row['zone1_heating']},{row['zone1_cooling']},{row['zone2_temp']},{row['zone2_heating']},{row['zone2_cooling']},{row['total_energy']}"
            )


if __name__ == "__main__":
    main()
