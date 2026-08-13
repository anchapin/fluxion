#!/usr/bin/env python3
import argparse
import csv
import json
import os

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_fluxion_data(csv_path):
    hours = []
    hvac_watts = []
    zone_temps = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hours.append(int(row["Hour"]))
            # HVAC_Watts column may contain semicolon-separated values for multiple zones
            # We take the first zone for Case 900
            hvac_val = float(row["HVAC_Watts"].split(";")[0])
            hvac_watts.append(hvac_val)

            zone_temp = float(row["Zone_Temps"].split(";")[0])
            zone_temps.append(zone_temp)

    return np.array(hours), np.array(hvac_watts), np.array(zone_temps)


def load_reference_data(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: Reference file {json_path} not found. Using dummy data.")
        return np.arange(8760), np.zeros(8760), np.zeros(8760)

    with open(json_path, "r") as f:
        data = json.load(f)

    # EnergyPlus heating/cooling energy is in Joules per hour
    # Convert to Watts by dividing by 3600
    heating_j = np.array(data["hourly"].get("heating_energy", []))
    cooling_j = np.array(data["hourly"].get("cooling_energy", []))

    # Ensure they have data, or fallback to zeros
    num_raw = max(len(heating_j), len(cooling_j))
    if num_raw == 0:
        print("Warning: No heating/cooling energy data in reference. Using zeros.")
        return np.arange(8760), np.zeros(8760), np.zeros(8760)

    if len(heating_j) < num_raw:
        heating_j = np.zeros(num_raw)
    if len(cooling_j) < num_raw:
        cooling_j = np.zeros(num_raw)

    # HVAC power in Watts (positive = heating, negative = cooling)
    # reporting frequency determines the divisor
    if num_raw > 8760:
        steps_per_hour = num_raw // 8760
        # Energy reported per timestep (e.g., Joules per 10 mins)
        # 1 hour = 3600 s. Each timestep is 3600 / steps_per_hour seconds.
        divisor = 3600.0 / steps_per_hour
        hvac_watts_raw = (heating_j - cooling_j) / divisor

        zone_temps_raw = np.array(data["hourly"].get("zone_air_temp", []))
        if len(zone_temps_raw) < num_raw:
            zone_temps_raw = np.full(num_raw, 20.0)

        hvac_watts = hvac_watts_raw.reshape(8760, steps_per_hour).mean(axis=1)
        zone_temps = zone_temps_raw.reshape(8760, steps_per_hour).mean(axis=1)
    else:
        hvac_watts = (heating_j - cooling_j) / 3600.0
        zone_temps_raw = np.array(data["hourly"].get("zone_air_temp", []))
        if len(zone_temps_raw) < 8760:
            zone_temps = np.full(8760, 20.0)
        else:
            zone_temps = zone_temps_raw

    return np.arange(8760), hvac_watts, zone_temps


def analyze_peaks(flux_hours, flux_hvac, ref_hours, ref_hvac):
    # Find heating peak
    flux_peak_heat = np.max(flux_hvac)
    flux_peak_heat_hour = flux_hours[np.argmax(flux_hvac)]

    ref_peak_heat = np.max(ref_hvac)
    ref_peak_heat_hour = ref_hours[np.argmax(ref_hvac)]

    # Find cooling peak (most negative)
    flux_peak_cool = np.min(flux_hvac)
    flux_peak_cool_hour = flux_hours[np.argmin(flux_hvac)]

    ref_peak_cool = np.min(ref_hvac)
    ref_peak_cool_hour = ref_hours[np.argmin(ref_hvac)]

    print("=== Peak Analysis ===")
    print("Heating Peak:")
    print(f"  Fluxion:   {flux_peak_heat:8.2f} W at hour {flux_peak_heat_hour}")
    print(f"  Reference: {ref_peak_heat:8.2f} W at hour {ref_peak_heat_hour}")
    print(
        f"  Error:     {flux_peak_heat - ref_peak_heat:8.2f} W ({(flux_peak_heat / ref_peak_heat - 1) * 100:+.1f}%)"
    )
    print(f"  Shift:     {flux_peak_heat_hour - ref_peak_heat_hour:8d} hours")

    print("\nCooling Peak:")
    print(f"  Fluxion:   {-flux_peak_cool:8.2f} W at hour {flux_peak_cool_hour}")
    print(f"  Reference: {-ref_peak_cool:8.2f} W at hour {ref_peak_cool_hour}")
    print(
        f"  Error:     {abs(flux_peak_cool) - abs(ref_peak_cool):8.2f} W ({(flux_peak_cool / ref_peak_cool - 1) * 100:+.1f}%)"
    )
    print(f"  Shift:     {flux_peak_cool_hour - ref_peak_cool_hour:8d} hours")


def plot_peak_days(flux_hours, flux_hvac, ref_hours, ref_hvac, output_prefix):
    # Peak Heating Day: Jan 4 (Hours 72-96)
    # Peak Cooling Day: Jul 27 (Hours 4968-4992)

    days = [
        ("Peak Heating Day (Jan 4)", 72, 96),
        ("Peak Cooling Day (Jul 27)", 4968, 4992),
    ]

    for title, start, end in days:
        plt.figure(figsize=(10, 6))
        plt.plot(flux_hours[start:end], flux_hvac[start:end], "b-", label="Fluxion")
        plt.plot(
            ref_hours[start:end],
            ref_hvac[start:end],
            "r--",
            label="Reference (EnergyPlus)",
        )

        plt.title(title)
        plt.xlabel("Hour of Year")
        plt.ylabel("HVAC Power (Watts, + = Heating, - = Cooling)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = f"{output_prefix}_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare Fluxion vs Reference peak profiles."
    )
    parser.add_argument(
        "--fluxion",
        type=str,
        default="case_900_peak_hourly.csv",
        help="Path to Fluxion CSV",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="refdata/energyplus_reference/900_reference.json",
        help="Path to Reference JSON",
    )
    parser.add_argument(
        "--output", type=str, default="peak_comparison", help="Prefix for output plots"
    )

    args = parser.parse_args()

    if not os.path.exists(args.fluxion):
        print(f"Error: {args.fluxion} not found. Run the diagnostic test first.")
        return

    flux_hours, flux_hvac, flux_temps = load_fluxion_data(args.fluxion)
    ref_hours, ref_hvac, ref_temps = load_reference_data(args.reference)

    analyze_peaks(flux_hours, flux_hvac, ref_hours, ref_hvac)

    if HAS_MATPLOTLIB:
        try:
            plot_peak_days(flux_hours, flux_hvac, ref_hours, ref_hvac, args.output)
        except Exception as e:
            print(f"Note: Could not generate plots: {e}")
    else:
        print("Note: Matplotlib not found. Skipping plots.")


if __name__ == "__main__":
    main()
