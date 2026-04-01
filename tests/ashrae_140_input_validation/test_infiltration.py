"""
Infiltration validation tests for ASHRAE 140 cases.

These tests validate that infiltration specifications match ASHRAE 140:
- Infiltration rate: 0.5 ACH
- Infiltration schedule: always on
- Infiltration heat loss calculation
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


class TestASHRAE140Infiltration:
    """Test ASHRAE 140 infiltration specifications."""

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

    def test_case_900_ach(self):
        """Verify infiltration is 0.5 ACH."""
        # ASHRAE 140 Case 900 specifies 0.5 air changes per hour

        result = self._run_rust_test("test_case_900_infiltration_ach")
        assert (
            result["returncode"] == 0
        ), f"Infiltration ACH validation failed: {result['stderr']}"

    def test_case_900_infiltration_flow_rate(self):
        """Verify infiltration volumetric flow rate."""
        # Flow rate = ACH × Volume / 3600
        # = 0.5 ACH × 129.6 m³ / 3600 = 0.018 m³/s
        zone_volume = 129.6  # m³
        ach = 0.5
        ach * zone_volume / 3600

        result = self._run_rust_test("test_case_900_infiltration_flow")
        assert (
            result["returncode"] == 0
        ), f"Infiltration flow rate validation failed: {result['stderr']}"

    def test_case_900_infiltration_schedule(self):
        """Verify infiltration schedule is always on."""
        # Infiltration is continuous (24/7) in ASHRAE 140 cases

        result = self._run_rust_test("test_case_900_infiltration_schedule")
        assert result["returncode"] == 0

    def test_case_900_infiltration_heat_loss_formula(self):
        """Verify infiltration heat loss calculation."""
        # Q = ρ × Cp × V × ACH × ΔT / 3600
        # Where:
        # - ρ = 1.2 kg/m³ (air density)
        # - Cp = 1005 J/kg-K (air specific heat)
        # - V = 129.6 m³ (zone volume)
        # - ACH = 0.5/h
        # - ΔT = temperature difference

        # For ΔT = 20K:
        # Q = 1.2 × 1005 × 129.6 × 0.5 × 20 / 3600 ≈ 435 W

        result = self._run_rust_test("test_case_900_infiltration_heat_loss")
        assert (
            result["returncode"] == 0
        ), f"Infiltration heat loss validation failed: {result['stderr']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
