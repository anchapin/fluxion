#!/usr/bin/env python3
"""
Data Collection Tool for Fluxion.
Ingests real-world building data and reference simulation outputs (ASHRAE 140).
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fluxion.data_collection")

class ASHRAELoader:
    """Loads ASHRAE 140 reference data."""

    @staticmethod
    def load_case_600(filepath: str) -> pd.DataFrame:
        """
        Loads ASHRAE 140 Case 600 simulation results from a CSV file.
        Expected columns: Date/Time, Zone Mean Air Temperature [C], Zone Total Heating Energy [J], Zone Total Cooling Energy [J]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ASHRAE data file not found: {filepath}")

        # Load CSV, assuming EnergyPlus output format
        # EnergyPlus CSVs usually have a header row.
        df = pd.read_csv(filepath)

        # trim whitespace from columns
        df.columns = df.columns.str.strip()

        return df

class WeatherDataLoader:
    """Loads EPW weather data."""

    @staticmethod
    def load_epw(filepath: str) -> pd.DataFrame:
        """
        Loads EPW weather file.
        EPW format usually has 8 header lines.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weather file not found: {filepath}")

        # EPW column names (standard subset)
        names = [
            "Year", "Month", "Day", "Hour", "Minute", "Data Source and Uncertainty Flags",
            "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity",
            "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation",
            "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity",
            "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation",
            "Global Horizontal Illuminance", "Direct Normal Illuminance", "Diffuse Horizontal Illuminance",
            "Zenith Luminance", "Wind Direction", "Wind Speed", "Total Sky Cover",
            "Opaque Sky Cover", "Visibility", "Ceiling Height", "Present Weather Observation",
            "Present Weather Codes", "Precipitable Water", "Aerosol Optical Depth",
            "Snow Depth", "Days Since Last Snowfall", "Albedo", "Liquid Precipitation Depth",
            "Liquid Precipitation Quantity"
        ]

        # Read CSV with pandas, skipping first 8 rows of metadata
        # header=None because we provide names.
        df = pd.read_csv(filepath, skiprows=8, header=None, names=names)
        return df

class DataPreprocessor:
    """Aligns and preprocesses data."""

    @staticmethod
    def process(building_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aligns building and weather data and converts to numpy arrays.
        Returns:
            X: Inputs (N, 2) - [Dry Bulb Temperature, Global Horizontal Radiation]
            y: Outputs (N, 10) - [Zone Load/Temp duplicated 10 times]
        """
        # Ensure lengths match or align by index.
        # WARNING: This assumes that the building simulation data and the weather data
        # start at the same time and have the same frequency (e.g., hourly).
        # In a production pipeline, this should align using timestamps.

        n_samples = min(len(building_df), len(weather_df))
        building_df = building_df.iloc[:n_samples]
        weather_df = weather_df.iloc[:n_samples]

        # Prepare Inputs X: (N, 2)
        # using Dry Bulb Temp and Global Horizontal Radiation
        # Ensure columns exist
        if 'Dry Bulb Temperature' not in weather_df.columns or 'Global Horizontal Radiation' not in weather_df.columns:
             raise ValueError("Weather data missing required columns: 'Dry Bulb Temperature' or 'Global Horizontal Radiation'")

        X = weather_df[['Dry Bulb Temperature', 'Global Horizontal Radiation']].values.astype(np.float32)

        # Prepare Outputs y: (N, 10)
        # We need to extract the relevant metric from building_df.
        # Let's assume we want 'Zone Mean Air Temperature [C]' or Loads.

        target_col = None
        for col in building_df.columns:
            if 'Temperature' in col and 'Zone' in col:
                target_col = col
                break

        if target_col is None:
            # Fallback to first numeric column that is not time
            numeric_cols = building_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
            else:
                raise ValueError("No suitable target column found in building data.")

        logger.info(f"Using target column: {target_col}")
        y_single = building_df[target_col].values.reshape(-1, 1).astype(np.float32)

        # Broadcast to 10 zones
        y = np.repeat(y_single, 10, axis=1)

        return X, y

def main():
    parser = argparse.ArgumentParser(description="Fluxion Data Collection Tool")
    parser.add_argument("--ashrae-file", required=True, help="Path to ASHRAE 140 CSV file")
    parser.add_argument("--weather-file", required=True, help="Path to EPW weather file")
    parser.add_argument("--out", default="assets/real_building_data.npz", help="Output .npz file path")

    args = parser.parse_args()

    try:
        logger.info(f"Loading ASHRAE data from {args.ashrae_file}...")
        building_df = ASHRAELoader.load_case_600(args.ashrae_file)

        logger.info(f"Loading Weather data from {args.weather_file}...")
        weather_df = WeatherDataLoader.load_epw(args.weather_file)

        logger.info("Preprocessing data...")
        X, y = DataPreprocessor.process(building_df, weather_df)

        logger.info(f"Saving data to {args.out}...")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        np.savez(args.out, X=X, y=y)
        logger.info("Done.")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
