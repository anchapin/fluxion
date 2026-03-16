import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import sys
from unittest.mock import patch

# Add tools directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.data_collection import ASHRAELoader, WeatherDataLoader, DataPreprocessor, main

class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ashrae_file = os.path.join(self.test_dir, "ashrae.csv")
        self.weather_file = os.path.join(self.test_dir, "weather.epw")
        self.output_file = os.path.join(self.test_dir, "output.npz")

        # Create dummy ASHRAE CSV
        df_ashrae = pd.DataFrame({
            "Date/Time": pd.date_range(start="2023-01-01", periods=24, freq="h"),
            "Zone Mean Air Temperature [C]": np.random.uniform(18, 24, 24),
            "Zone Total Heating Energy [J]": np.random.uniform(0, 100, 24)
        })
        df_ashrae.to_csv(self.ashrae_file, index=False)

        # Create dummy EPW file
        # 8 header lines + data
        with open(self.weather_file, "w") as f:
            for i in range(8):
                f.write(f"Header line {i}\n")

            # Data columns (subset for brevity, matching indices in WeatherDataLoader)
            # 35 columns
            # Set Dry Bulb (6) and Global Horizontal (13)
            # Year, Month, Day, Hour, Minute are 0-4

            # Write 24 rows
            for i in range(24):
                row = [0] * 35
                row[0] = 2023
                row[1] = 1
                row[2] = 1
                row[3] = i + 1
                row[6] = 5.0 + i * 0.5 # Dry Bulb
                row[13] = 100.0 + i * 10 # Radiation
                f.write(",".join(map(str, row)) + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ashrae_loader(self):
        df = ASHRAELoader.load_case_600(self.ashrae_file)
        self.assertEqual(len(df), 24)
        self.assertIn("Zone Mean Air Temperature [C]", df.columns)

    def test_weather_loader(self):
        df = WeatherDataLoader.load_epw(self.weather_file)
        self.assertEqual(len(df), 24)
        self.assertIn("Dry Bulb Temperature", df.columns)
        self.assertIn("Global Horizontal Radiation", df.columns)

    def test_preprocessor(self):
        building_df = ASHRAELoader.load_case_600(self.ashrae_file)
        weather_df = WeatherDataLoader.load_epw(self.weather_file)

        X, y = DataPreprocessor.process(building_df, weather_df)

        self.assertEqual(X.shape, (24, 2))
        self.assertEqual(y.shape, (24, 10))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_main(self):
        # Run main with arguments
        test_args = [
            "tools/data_collection.py",
            "--ashrae-file", self.ashrae_file,
            "--weather-file", self.weather_file,
            "--out", self.output_file
        ]
        with patch.object(sys, 'argv', test_args):
            # We import main inside the test to avoid running it on import
            # But main() is not called on import in my implementation
            from tools.data_collection import main as run_main
            run_main()

        self.assertTrue(os.path.exists(self.output_file))
        data = np.load(self.output_file)
        self.assertIn("X", data)
        self.assertIn("y", data)
        self.assertEqual(data["X"].shape, (24, 2))
        self.assertEqual(data["y"].shape, (24, 10))

if __name__ == "__main__":
    unittest.main()
