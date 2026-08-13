import argparse

import numpy as np
import pandas as pd


def analyze_profiles(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Add day column
    df["day"] = df["hour"] // 24

    # Calculate diurnal swing for each day
    daily_stats = df.groupby("day")["air_temp"].agg(["min", "max"])
    daily_stats["swing"] = daily_stats["max"] - daily_stats["min"]

    avg_swing = daily_stats["swing"].mean()
    max_swing_val = daily_stats["swing"].max()
    min_swing_val = daily_stats["swing"].min()

    print("Diurnal Temperature Swing Analysis (Case 900FF):")
    print(f"  Average Swing: {avg_swing:.2f}°C")
    print(f"  Max Swing: {max_swing_val:.2f}°C")
    print(f"  Min Swing: {min_swing_val:.2f}°C")
    print("  Reference Swing (approximate): ~19.6°C")
    swing_error = (avg_swing - 19.6) / 19.6 * 100
    print(f"  Swing Error: {swing_error:.1f}%")

    # Phase lag analysis
    # We look for the hour of peak solar gain and peak air temperature for each day
    lags = []
    for day in range(365):
        day_df = df[df["day"] == day]
        if day_df["solar_gain"].max() > 100:  # Only days with significant solar gain
            peak_solar_hour = day_df.loc[day_df["solar_gain"].idxmax(), "hour"] % 24
            peak_air_hour = day_df.loc[day_df["air_temp"].idxmax(), "hour"] % 24

            # Simple lag calculation
            lag = peak_air_hour - peak_solar_hour
            # Adjust for day wrap around if needed
            if lag < -12:
                lag += 24
            if lag > 12:
                lag -= 24

            lags.append(lag)

    if lags:
        avg_lag = np.mean(lags)
        print("\nPhase Lag Analysis:")
        print(f"  Average Phase Lag (Solar Peak to Air Peak): {avg_lag:.2f} hours")
        print(f"  Standard Deviation: {np.std(lags):.2f} hours")
    else:
        print("\nPhase Lag Analysis: No days with significant solar gain found.")

    # Nighttime cooling rate residual
    # Look for hour 0-5 (nighttime) and check cooling rate (dT/dt)
    night_df = df[df["hour"] % 24 < 6]
    night_df["dt"] = night_df["air_temp"].diff()
    avg_cooling_rate = night_df[night_df["dt"] < 0]["dt"].mean()
    print("\nNighttime Cooling Rate Analysis (Hours 0-5):")
    print(f"  Average Nighttime Cooling Rate: {avg_cooling_rate:.2f}°C/hour")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze free-float temperature profiles."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="case_900ff_profile_hourly.csv",
        help="Path to CSV file",
    )
    args = parser.parse_args()

    analyze_profiles(args.csv)
