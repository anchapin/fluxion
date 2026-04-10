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
        """Verify wall U-value is 0.51 W/m²K."""
        # ASHRAE 140 Case 900 high-mass wall U-value: 0.51 W/m²K
        # This includes interior and exterior film coefficients

        result = self._run_rust_test("test_case_900_wall_u_value")
        assert result["returncode"] == 0, (
            f"Wall U-value validation failed: {result['stderr']}"
        )

    def test_case_900_wall_heat_capacity(self):
        """Verify wall heat capacity matches high-mass spec."""
        # High-mass wall: 100mm concrete block
        # Heat capacity = thickness × density × specific_heat
        # = 0.1 m × 800 kg/m³ × 1000 J/kg-K = 80 kJ/m²K

        result = self._run_rust_test("test_case_900_wall_heat_capacity")
        assert result["returncode"] == 0, (
            f"Wall heat capacity validation failed: {result['stderr']}"
        )

    def test_case_900_roof_u_value(self):
        """Verify roof U-value is 0.32 W/m²K."""
        # ASHRAE 140 Case 900 roof U-value: 0.32 W/m²K

        result = self._run_rust_test("test_case_900_roof_u_value")
        assert result["returncode"] == 0, (
            f"Roof U-value validation failed: {result['stderr']}"
        )

    def test_case_900_floor_u_value(self):
        """Verify floor U-value is 0.38 W/m²K."""
        # ASHRAE 140 Case 900 floor U-value: 0.38 W/m²K

        result = self._run_rust_test("test_case_900_floor_u_value")
        assert result["returncode"] == 0, (
            f"Floor U-value validation failed: {result['stderr']}"
        )

    def test_case_900_window_u_value(self):
        """Verify window U-value is 3.0 W/m²K."""
        # ASHRAE 140 specifies double clear glass with U-value = 3.0 W/m²K

        result = self._run_rust_test("test_case_900_window_u_value")
        assert result["returncode"] == 0, (
            f"Window U-value validation failed: {result['stderr']}"
        )

    def test_case_900_window_shgc(self):
        """Verify window SHGC is 0.789."""
        # Solar Heat Gain Coefficient for double clear glass: 0.789

        result = self._run_rust_test("test_case_900_window_shgc")
        assert result["returncode"] == 0, (
            f"Window SHGC validation failed: {result['stderr']}"
        )

    def test_case_900_window_tvis(self):
        """Verify window visible transmittance is 0.86156."""
        # Visible transmittance for double clear glass: 0.86156

        result = self._run_rust_test("test_case_900_window_tvis")
        assert result["returncode"] == 0, (
            f"Window Tvis validation failed: {result['stderr']}"
        )

    def test_case_600_wall_u_value(self):
        """Verify Case 600 wall U-value."""
        # Case 600 low-mass wall U-value: different from Case 900
        # Low-mass construction uses lighter materials
        result = self._run_rust_test("test_case_600_wall_u_value")
        assert result["returncode"] == 0

    def test_case_600_wall_heat_capacity(self):
        """Verify Case 600 wall heat capacity (low-mass)."""
        # Low-mass wall has lower heat capacity than high-mass
        # Typically plasterboard + insulation + wood
        result = self._run_rust_test("test_case_600_wall_heat_capacity")
        assert result["returncode"] == 0

    def test_construction_layer_thicknesses(self):
        """Verify construction layer thicknesses."""
        # Each construction assembly has multiple layers with specific thicknesses
        # Wall layers (high-mass): concrete block, insulation, etc.
        # Roof layers: membrane, insulation, deck
        # Floor layers: carpet, concrete slab, etc.

        result = self._run_rust_test("test_case_900_construction_layers")
        assert result["returncode"] == 0

    def test_material_thermal_properties(self):
        """Verify material thermal properties."""
        # Materials have conductivity, density, specific heat
        # These are used to calculate CTF coefficients

        result = self._run_rust_test("test_case_900_materials")
        assert result["returncode"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
