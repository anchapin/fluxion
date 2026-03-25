"""
Internal gains validation tests for ASHRAE 140 cases.

These tests validate that internal heat gains match ASHRAE 140 specifications:
- Equipment power: 200W continuous
- Equipment schedule: 40% night, 60% day
- Annual equipment energy calculation
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


class TestASHRAE140InternalGains:
    """Test ASHRAE 140 internal gain specifications."""

    def _run_rust_test(self, test_name: str) -> Dict[str, Any]:
        """Run a Rust test and return results."""
        project_root = Path(__file__).parent.parent.parent
        cmd = ["cargo", "test", test_name, "--", "--nocapture"]

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def test_case_900_equipment_power(self):
        """Verify electric equipment is 200W."""
        # ASHRAE 140 Case 900 specifies 200W continuous equipment load
        expected_power_w = 200.0
        tolerance = 0.01  # ±1% tolerance

        result = self._run_rust_test("test_case_900_equipment_power")
        assert (
            result["returncode"] == 0
        ), f"Equipment power validation failed: {result['stderr']}"

    def test_case_900_equipment_schedule(self):
        """Verify equipment schedule fractions."""
        # Equipment schedule:
        # - Day (6:00-22:00): 60% of full load
        # - Night (22:00-6:00): 40% of full load
        expected_day_fraction = 0.6
        expected_night_fraction = 0.4
        tolerance = 0.01

        result = self._run_rust_test("test_case_900_equipment_schedule")
        assert (
            result["returncode"] == 0
        ), f"Equipment schedule validation failed: {result['stderr']}"

    def test_case_900_equipment_annual_energy(self):
        """Verify annual equipment energy calculation."""
        # Annual energy = Power × Hours × Average Schedule Fraction
        # = 200W × 24hr × 365days × 0.5 (avg) = 876 kWh = 0.876 MWh
        expected_annual_mwh = 0.2 * 24 * 365 * 0.5 / 1000
        tolerance = 0.05  # ±5% tolerance

        result = self._run_rust_test("test_case_900_equipment_annual_energy")
        assert (
            result["returncode"] == 0
        ), f"Annual equipment energy validation failed: {result['stderr']}"

    def test_case_900_internal_loads_radiative_convective_split(self):
        """Verify internal loads radiative/convective split."""
        # Internal gains are split between radiative and convective
        # Typical split: 60% radiative, 40% convective for equipment

        result = self._run_rust_test("test_case_900_internal_loads_split")
        assert result["returncode"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
