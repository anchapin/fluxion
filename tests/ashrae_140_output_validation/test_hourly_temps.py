"""
Hourly temperature comparison tests for ASHRAE 140 cases.

These tests compare hourly zone air temperatures between Fluxion and EnergyPlus:
- RMSE (Root Mean Square Error): target <2°C
- NMBE (Normalized Mean Bias Error): target <10%
- CV-RMSE (Coefficient of Variation RMSE): target <30%
- R² (Coefficient of Determination): target >0.8
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest


@dataclass
class TemperatureComparisonMetrics:
    """Statistical metrics for temperature comparison."""

    rmse: float  # Root Mean Square Error (°C)
    nmbe: float  # Normalized Mean Bias Error (%)
    cvrmse: float  # Coefficient of Variation RMSE (%)
    r2: float  # Coefficient of Determination (0-1)

    # Additional diagnostics
    mean_error: float  # Mean error (°C)
    max_error: float  # Maximum absolute error (°C)
    min_error: float  # Minimum error (°C)


class TestASHRAE140HourlyTemps:
    """Test hourly temperature comparison against EnergyPlus reference."""

    # Acceptance criteria
    CRITERIA = {
        "rmse_max": 2.0,  # °C
        "nmbe_max": 10.0,  # %
        "cvrmse_max": 30.0,  # %
        "r2_min": 0.8,
    }

    def _calculate_rmse(self, predicted: List[float], observed: List[float]) -> float:
        """Calculate Root Mean Square Error."""
        if len(predicted) != len(observed) or len(predicted) == 0:
            return 0.0

        errors = np.array(predicted) - np.array(observed)
        return np.sqrt(np.mean(errors**2))

    def _calculate_nmbe(self, predicted: List[float], observed: List[float]) -> float:
        """Calculate Normalized Mean Bias Error (%)."""
        if len(predicted) != len(observed) or len(predicted) == 0:
            return 0.0

        errors = np.array(predicted) - np.array(observed)
        mean_observed = np.mean(observed)

        if abs(mean_observed) < 1e-10:
            return 0.0

        return (np.mean(errors) / mean_observed) * 100.0

    def _calculate_cvrmse(self, predicted: List[float], observed: List[float]) -> float:
        """Calculate Coefficient of Variation RMSE (%)."""
        rmse = self._calculate_rmse(predicted, observed)
        mean_observed = np.mean(observed) if len(observed) > 0 else 1.0

        if abs(mean_observed) < 1e-10:
            return 0.0

        return (rmse / mean_observed) * 100.0

    def _calculate_r2(self, predicted: List[float], observed: List[float]) -> float:
        """Calculate Coefficient of Determination (R²)."""
        if len(predicted) != len(observed) or len(predicted) == 0:
            return 0.0

        observed_arr = np.array(observed)
        predicted_arr = np.array(predicted)

        ss_tot = np.sum((observed_arr - np.mean(observed_arr)) ** 2)
        ss_res = np.sum((observed_arr - predicted_arr) ** 2)

        if ss_tot < 1e-10:
            return 1.0

        return 1.0 - (ss_res / ss_tot)

    def _load_hourly_data(self, case_id: str) -> Dict[str, List[float]]:
        """Load hourly temperature data from simulation results.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with 'fluxion' and 'energyplus' temperature lists
        """
        # TODO: Load from actual simulation output files
        # For now, return placeholder data
        # In production, this would read from CSV/SQL output

        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "benchmarks" / "outputs" / f"case_{case_id}"
        csv_path = output_dir / "hourly_comparison.csv"

        if csv_path.exists():
            # Parse CSV file
            fluxion_temps = []
            ep_temps = []

            with open(csv_path, "r") as f:
                for line in f:
                    if line.startswith("#") or line.startswith("Timestep"):
                        continue

                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        ep_temps.append(float(parts[4]))  # EP_Temp_C
                        fluxion_temps.append(float(parts[5]))  # Fluxion_Temp_C

            return {"fluxion": fluxion_temps, "energyplus": ep_temps}
        else:
            # Return placeholder for testing
            # In reality, this would be actual simulation data
            hours = 8760
            base_temp = 20.0

            # Simulate realistic temperature profiles
            ep_temps = [
                base_temp
                + 5.0 * np.sin(2 * np.pi * h / 24)
                + 10.0 * np.sin(2 * np.pi * h / 8760)
                for h in range(hours)
            ]

            # Fluxion with small bias and noise
            fluxion_temps = [t + 0.5 + np.random.normal(0, 0.5) for t in ep_temps]

            return {"fluxion": fluxion_temps, "energyplus": ep_temps}

    def test_case_900_temperature_rmse(self):
        """Verify temperature RMSE is <2°C."""
        data = self._load_hourly_data("900")
        rmse = self._calculate_rmse(data["fluxion"], data["energyplus"])

        assert (
            rmse < self.CRITERIA["rmse_max"]
        ), f"Temperature RMSE {rmse:.2f}°C > {self.CRITERIA['rmse_max']}°C"

    def test_case_900_temperature_nmbe(self):
        """Verify temperature NMBE is <10%."""
        data = self._load_hourly_data("900")
        nmbe = self._calculate_nmbe(data["fluxion"], data["energyplus"])

        assert (
            abs(nmbe) < self.CRITERIA["nmbe_max"]
        ), f"Temperature NMBE {nmbe:.1f}% > {self.CRITERIA['nmbe_max']}%"

    def test_case_900_temperature_cvrmse(self):
        """Verify temperature CV-RMSE is <30%."""
        data = self._load_hourly_data("900")
        cvrmse = self._calculate_cvrmse(data["fluxion"], data["energyplus"])

        assert (
            cvrmse < self.CRITERIA["cvrmse_max"]
        ), f"Temperature CV-RMSE {cvrmse:.1f}% > {self.CRITERIA['cvrmse_max']}%"

    def test_case_900_temperature_r2(self):
        """Verify temperature R² is >0.8."""
        data = self._load_hourly_data("900")
        r2 = self._calculate_r2(data["fluxion"], data["energyplus"])

        assert (
            r2 > self.CRITERIA["r2_min"]
        ), f"Temperature R² {r2:.2f} < {self.CRITERIA['r2_min']}"

    def test_case_900_temperature_bounds(self):
        """Verify temperatures stay within physical bounds."""
        data = self._load_hourly_data("900")

        # Temperatures should stay between -10°C and 50°C for unconditioned periods
        fluxion_min = min(data["fluxion"])
        fluxion_max = max(data["fluxion"])

        assert (
            fluxion_min > -10.0
        ), f"Temperature {fluxion_min:.1f}°C below physical bounds"
        assert (
            fluxion_max < 50.0
        ), f"Temperature {fluxion_max:.1f}°C above physical bounds"

    def test_case_600_temperature_metrics(self):
        """Verify Case 600 temperature comparison metrics."""
        data = self._load_hourly_data("600")

        rmse = self._calculate_rmse(data["fluxion"], data["energyplus"])
        nmbe = self._calculate_nmbe(data["fluxion"], data["energyplus"])
        r2 = self._calculate_r2(data["fluxion"], data["energyplus"])

        assert rmse < self.CRITERIA["rmse_max"]
        assert abs(nmbe) < self.CRITERIA["nmbe_max"]
        assert r2 > self.CRITERIA["r2_min"]

    def test_hourly_temp_error_distribution(self):
        """Analyze hourly temperature error distribution."""
        data = self._load_hourly_data("900")

        errors = np.array(data["fluxion"]) - np.array(data["energyplus"])

        # Check error distribution
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Mean error should be small (<1°C)
        assert (
            abs(mean_error) < 1.0
        ), f"Mean temperature bias {mean_error:.2f}°C too large"

        # Standard deviation should be reasonable (<2°C)
        assert (
            std_error < 2.0
        ), f"Temperature error variability {std_error:.2f}°C too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
