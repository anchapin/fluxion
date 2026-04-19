"""
Construction and material validation tests for ASHRAE 140 cases.

These tests validate that building envelope constructions match ASHRAE 140 specifications:
- U-values (thermal transmittance)
- Heat capacities (thermal mass)
- Layer thicknesses and properties
- Window properties (U, SHGC, Tvis)
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


class TestASHRAE140Constructions:
    """Test ASHRAE 140 construction specifications."""

    def _run_rust_test(self, test_name: str) -> Dict[str, Any]:
        """Run a Rust test and return results."""
        project_root = Path(__file__).parent.parent.parent
        cmd = ["cargo", "test", test_name, "--", "--nocapture"]

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def test_case_900_wall_u_value(self):
        """Verify wall U-value matches ASHRAE 140 spec."""
        # ASHRAE 140 Case 900 high-mass wall U-value: 0.514 W/m²K
        result = self._run_rust_test("test_high_mass_wall_u_value_ashrae_140")
        assert (
            result["returncode"] == 0
        ), f"Wall U-value validation failed: {result['stderr']}"

    def test_case_900_roof_u_value(self):
        """Verify roof U-value matches ASHRAE 140 spec."""
        # ASHRAE 140 Case 900 roof U-value: 0.514 W/m²K
        result = self._run_rust_test("test_high_mass_roof_u_value_ashrae_140")
        assert (
            result["returncode"] == 0
        ), f"Roof U-value validation failed: {result['stderr']}"

    def test_case_900_floor_u_value(self):
        """Verify floor U-value matches ASHRAE 140 spec."""
        # ASHRAE 140 insulated floor U-value
        result = self._run_rust_test("test_high_mass_floor_u_value_ashrae_140")
        assert (
            result["returncode"] == 0
        ), f"Floor U-value validation failed: {result['stderr']}"

    def test_case_900_window_shgc(self):
        """Verify window SHGC is 0.789."""
        # Solar Heat Gain Coefficient for double clear glass: 0.789
        result = self._run_rust_test("test_window_properties_double_clear")
        assert (
            result["returncode"] == 0
        ), f"Window SHGC validation failed: {result['stderr']}"

    def test_case_600_wall_u_value(self):
        """Verify Case 600 wall U-value."""
        # Case 600 low-mass wall U-value: 0.514 W/m²K
        result = self._run_rust_test("test_low_mass_wall_u_value_ashrae_140")
        assert result["returncode"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
