"""
Weather and location validation tests for ASHRAE 140 cases.

These tests validate that weather and location match ASHRAE 140:
- Denver TMY3 weather file
- Climate zone: 5B
- Design day temperatures
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


class TestASHRAE140Weather:
    """Test ASHRAE 140 weather specifications."""

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

    def test_case_900_location(self):
        """Verify location is Denver, CO."""
        # ASHRAE 140 uses Denver, Colorado as the reference location

        result = self._run_rust_test("test_case_900_location")
        assert (
            result["returncode"] == 0
        ), f"Location validation failed: {result['stderr']}"

    def test_case_900_climate_zone(self):
        """Verify climate zone is 5B."""
        # Denver climate zone: 5B (Cool - Dry, High Elevation)

        result = self._run_rust_test("test_case_900_climate_zone")
        assert (
            result["returncode"] == 0
        ), f"Climate zone validation failed: {result['stderr']}"

    def test_case_900_design_days(self):
        """Verify design day temperatures."""
        # Denver design days:
        # - Winter: -17.9°C (0% annual exceedance)
        # - Summer: 33.9°C (1% annual exceedance)

        result = self._run_rust_test("test_case_900_design_days")
        assert (
            result["returncode"] == 0
        ), f"Design day validation failed: {result['stderr']}"

    def test_case_900_weather_file(self):
        """Verify Denver TMY3 weather file is used."""
        # Weather file: Denver TMY3 (NREL)
        # Path: benchmarks/weather/Denver.epw

        result = self._run_rust_test("test_case_900_weather_file")
        assert result["returncode"] == 0

    def test_case_900_weather_data_completeness(self):
        """Verify weather data is complete (8760 hours)."""
        # TMY3 weather file should have 8760 hourly records

        result = self._run_rust_test("test_case_900_weather_completeness")
        assert result["returncode"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
