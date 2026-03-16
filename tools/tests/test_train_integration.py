import os
import shutil
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.data_gen import geometry, simulation
from tools import train_surrogate

class TestTrainIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_integration_data")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_synthetic_data(self):
        # 1. Generate Data
        model = geometry.create_shoebox_model(width=10, length=10)
        params = {"u_value": 2.0, "hvac_setpoint": 22.0, "width": 10.0, "length": 10.0}

        # Run simulation (mock)
        simulation.run_simulation(
            model,
            "dummy_weather.epw",
            str(self.test_dir),
            "run1",
            params=params
        )

        # 2. Load Data
        X, y = train_surrogate.load_data_from_synthetic_dir(self.test_dir)

        # 3. Verify
        # X shape: (8760, 3) -> [u, setpoint, outdoor]
        # y shape: (8760, 1) -> [load]
        self.assertEqual(X.shape, (8760, 3))
        self.assertEqual(y.shape, (8760, 1))

        # Check values
        # U-value (col 0) should be 2.0
        self.assertTrue(np.allclose(X[:, 0], 2.0))
        # Setpoint (col 1) should be 22.0
        self.assertTrue(np.allclose(X[:, 1], 22.0))

        # Load (y) should be somewhat realistic (mock logic)
        # Not checking exact values, but ensuring not all zeros or NaNs
        self.assertFalse(np.isnan(y).any())
        self.assertNotEqual(np.max(np.abs(y)), 0.0)

if __name__ == "__main__":
    unittest.main()
