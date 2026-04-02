"""
HVAC and thermostat validation tests for ASHRAE 140 cases.

These tests validate that HVAC systems and thermostats match ASHRAE 140:
- Heating setpoint: 20°C
- Cooling setpoint: 27°C
- Ideal air loads configuration
- HVAC capacity (autosized)
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


class TestASHRAE140HVAC:
    """Test ASHRAE 140 HVAC specifications."""

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

    def test_case_900_heating_setpoint(self):
        """Verify heating setpoint is 20°C."""
        # ASHRAE 140 Case 900 heating setpoint: 20°C

        result = self._run_rust_test("test_case_900_heating_setpoint")
        assert result["returncode"] == 0, (
            f"Heating setpoint validation failed: {result['stderr']}"
        )

    def test_case_900_cooling_setpoint(self):
        """Verify cooling setpoint is 27°C."""
        # ASHRAE 140 Case 900 cooling setpoint: 27°C

        result = self._run_rust_test("test_case_900_cooling_setpoint")
        assert result["returncode"] == 0, (
            f"Cooling setpoint validation failed: {result['stderr']}"
        )

    def test_case_900_ideal_air_loads(self):
        """Verify ideal air loads is enabled."""
        # ASHRAE 140 uses ideal air loads (infinite capacity, 100% efficient)
        # This means HVAC meets demand exactly with no capacity limitations

        result = self._run_rust_test("test_case_900_ideal_air_loads")
        assert result["returncode"] == 0, (
            f"Ideal air loads validation failed: {result['stderr']}"
        )

    def test_case_900_hvac_capacity_autosize(self):
        """Verify HVAC capacity is autosized."""
        # Ideal air loads are autosized to meet peak demand
        # No explicit capacity limit

        result = self._run_rust_test("test_case_900_hvac_autosize")
        assert result["returncode"] == 0

    def test_case_900_thermostat_schedule(self):
        """Verify thermostat schedule is constant."""
        # Setpoints are constant 24/7 (no setback in baseline cases)

        result = self._run_rust_test("test_case_900_thermostat_schedule")
        assert result["returncode"] == 0

    def test_case_640_setback_schedule(self):
        """Verify Case 640 thermostat setback schedule."""
        # Case 640 has heating setback to 10°C overnight (23:00-07:00)

        result = self._run_rust_test("test_case_640_setback")
        assert result["returncode"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
