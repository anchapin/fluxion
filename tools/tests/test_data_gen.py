import unittest
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.data_gen import geometry, simulation, weather, main

class TestDataGen(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_output"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_weather_download_mock(self):
        """Test weather download logic with mocked URL retrieval."""
        with patch("urllib.request.urlretrieve") as mock_retrieve:
            weather.download_standard_files(self.test_dir)

            # Check if urlretrieve was called for each file
            self.assertTrue(mock_retrieve.called)
            self.assertEqual(mock_retrieve.call_count, len(weather.WEATHER_FILES))

            # Verify file check logic (create a dummy file)
            dummy_file = os.path.join(self.test_dir, list(weather.WEATHER_FILES.keys())[0])
            with open(dummy_file, 'w') as f:
                f.write("dummy")

            mock_retrieve.reset_mock()
            weather.download_standard_files(self.test_dir)
            # Should have skipped the existing file
            self.assertEqual(mock_retrieve.call_count, len(weather.WEATHER_FILES) - 1)

    def test_geometry_creation_mock(self):
        """Test geometry creation using the MockOpenStudio."""
        # Force the module to use MockOpenStudio
        # Since we can't easily un-import, we rely on the fact that openstudio isn't installed here.
        model = geometry.create_shoebox_model(width=10, length=10, height=3)
        self.assertIsInstance(model, geometry.MockOpenStudio.model.Model)

    @patch("subprocess.run")
    def test_simulation_run_mock(self, mock_subprocess):
        """Test simulation execution flow (mocking subprocess for 'real' run attempt, but we expect MockModel bypass)."""
        model = geometry.create_shoebox_model()
        weather_path = os.path.join(self.test_dir, "test.epw")

        # This should trigger the "Mock model detected" branch in simulation.py
        success = simulation.run_simulation(model, weather_path, self.test_dir, "run1")
        self.assertTrue(success)

        # Verify dummy outputs were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "run1", "eplusout.sql")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "run1", "eplusout.csv")))

        # Subprocess should NOT be called because it was a mock model
        mock_subprocess.assert_not_called()

if __name__ == '__main__':
    unittest.main()
