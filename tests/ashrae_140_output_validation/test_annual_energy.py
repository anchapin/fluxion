"""
Annual energy comparison tests for ASHRAE 140 cases.

These tests compare annual HVAC energy consumption between Fluxion and EnergyPlus:
- Annual heating energy (target: <50% error)
- Annual cooling energy (target: <50% error)
- Total site energy (target: <30% error)
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest


@dataclass
class EnergyComparisonResult:
    """Result of energy comparison between Fluxion and EnergyPlus."""

    case_id: str
    fluxion_heating_mwh: float
    ep_heating_mwh: float
    heating_error_pct: float

    fluxion_cooling_mwh: float
    ep_cooling_mwh: float
    cooling_error_pct: float

    passed: bool
    message: str = ""


class TestASHRAE140AnnualEnergy:
    """Test annual energy comparison against EnergyPlus reference."""

    # Reference values from ASHRAE 140 / EnergyPlus
    REFERENCE_VALUES = {
        "900": {"heating_mwh": 1.66, "cooling_mwh": 2.49},
        "900FF": {"heating_mwh": 0.0, "cooling_mwh": 0.0},  # Free-floating
        "600": {"heating_mwh": 1.33, "cooling_mwh": 2.17},
        "600FF": {"heating_mwh": 0.0, "cooling_mwh": 0.0},
    }

    # Tolerance percentages by case type
    TOLERANCES = {
        "baseline": 50.0,  # ±50% for baseline cases
        "free_floating": 100.0,  # ±100% for free-floating (no HVAC)
    }

    def _run_fluxion_simulation(self, case_id: str) -> Dict[str, float]:
        """Run Fluxion simulation and extract annual energy.

        Args:
            case_id: Case identifier (e.g., "900", "600")

        Returns:
            Dictionary with heating_mwh and cooling_mwh
        """
        project_root = Path(__file__).parent.parent.parent
        cmd = [
            "cargo",
            "test",
            f"test_case_{case_id}_annual_energy",
            "--",
            "--nocapture",
        ]

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for simulation
        )

        if result.returncode != 0:
            raise RuntimeError(f"Fluxion simulation failed: {result.stderr}")

        # Parse energy values from test output
        # Expected format: "Heating: X.XXX MWh, Cooling: X.XXX MWh"
        heating_mwh = cooling_mwh = 0.0

        for line in result.stdout.split("\n"):
            if "Heating:" in line and "MWh" in line:
                # Extract value
                parts = line.split("Heating:")[1].split("MWh")[0].strip()
                heating_mwh = float(parts)
            if "Cooling:" in line and "MWh" in line:
                parts = line.split("Cooling:")[1].split("MWh")[0].strip()
                cooling_mwh = float(parts)

        return {"heating_mwh": heating_mwh, "cooling_mwh": cooling_mwh}

    def _calculate_error(self, fluxion: float, reference: float) -> float:
        """Calculate percentage error.

        Args:
            fluxion: Fluxion value
            reference: Reference (EnergyPlus) value

        Returns:
            Percentage error (0-100+)
        """
        if reference == 0.0:
            return 0.0 if fluxion == 0.0 else 100.0
        return abs(fluxion - reference) / reference * 100.0

    @pytest.mark.parametrize(
        "case_id,ref_heating,ref_cooling,tolerance",
        [
            ("900", 1.66, 2.49, 50.0),
            ("600", 1.33, 2.17, 50.0),
        ],
    )
    def test_annual_hvac_energy(
        self, case_id: str, ref_heating: float, ref_cooling: float, tolerance: float
    ):
        """Compare annual HVAC energy against EnergyPlus reference.

        Args:
            case_id: Case identifier
            ref_heating: Reference heating energy (MWh)
            ref_cooling: Reference cooling energy (MWh)
            tolerance: Acceptable error percentage
        """
        # Run Fluxion simulation
        try:
            fluxion_results = self._run_fluxion_simulation(case_id)
        except Exception as e:
            pytest.fail(f"Failed to run Fluxion simulation: {e}")

        heating_error = self._calculate_error(
            fluxion_results["heating_mwh"], ref_heating
        )
        cooling_error = self._calculate_error(
            fluxion_results["cooling_mwh"], ref_cooling
        )

        # Check heating energy
        if ref_heating > 0:
            assert heating_error < tolerance, (
                f"Heating energy error {heating_error:.1f}% > {tolerance}%\n"
                f"  Fluxion: {fluxion_results['heating_mwh']:.3f} MWh\n"
                f"  EnergyPlus: {ref_heating:.3f} MWh"
            )

        # Check cooling energy
        if ref_cooling > 0:
            assert cooling_error < tolerance, (
                f"Cooling energy error {cooling_error:.1f}% > {tolerance}%\n"
                f"  Fluxion: {fluxion_results['cooling_mwh']:.3f} MWh\n"
                f"  EnergyPlus: {ref_cooling:.3f} MWh"
            )

    def test_case_900_heating_energy(self):
        """Verify Case 900 annual heating energy."""
        ref_heating = self.REFERENCE_VALUES["900"]["heating_mwh"]
        tolerance = self.TOLERANCES["baseline"]

        results = self._run_fluxion_simulation("900")
        error = self._calculate_error(results["heating_mwh"], ref_heating)

        assert error < tolerance, f"Case 900 heating error {error:.1f}% > {tolerance}%"

    def test_case_900_cooling_energy(self):
        """Verify Case 900 annual cooling energy."""
        ref_cooling = self.REFERENCE_VALUES["900"]["cooling_mwh"]
        tolerance = self.TOLERANCES["baseline"]

        results = self._run_fluxion_simulation("900")
        error = self._calculate_error(results["cooling_mwh"], ref_cooling)

        assert error < tolerance, f"Case 900 cooling error {error:.1f}% > {tolerance}%"

    def test_case_900ff_free_floating(self):
        """Verify Case 900FF has zero HVAC energy (free-floating)."""
        # Free-floating case should have no HVAC energy
        results = self._run_fluxion_simulation("900FF")

        assert results["heating_mwh"] < 0.01, "Case 900FF should have zero heating"
        assert results["cooling_mwh"] < 0.01, "Case 900FF should have zero cooling"

    def test_case_600_heating_energy(self):
        """Verify Case 600 annual heating energy."""
        ref_heating = self.REFERENCE_VALUES["600"]["heating_mwh"]
        tolerance = self.TOLERANCES["baseline"]

        results = self._run_fluxion_simulation("600")
        error = self._calculate_error(results["heating_mwh"], ref_heating)

        assert error < tolerance, f"Case 600 heating error {error:.1f}% > {tolerance}%"

    def test_case_600_cooling_energy(self):
        """Verify Case 600 annual cooling energy."""
        ref_cooling = self.REFERENCE_VALUES["600"]["cooling_mwh"]
        tolerance = self.TOLERANCES["baseline"]

        results = self._run_fluxion_simulation("600")
        error = self._calculate_error(results["cooling_mwh"], ref_cooling)

        assert error < tolerance, f"Case 600 cooling error {error:.1f}% > {tolerance}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
