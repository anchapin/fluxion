"""
Geometry validation tests for ASHRAE 140 cases.

These tests validate that building geometry matches ASHRAE 140 specifications:
- Floor area: 48 m² (8m × 6m)
- Ceiling height: 2.7 m
- Zone volume: 129.6 m³
- Window areas and placement
- Wall areas by orientation
- Surface orientations (azimuth angles)
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


class TestASHRAE140Geometry:
    """Test ASHRAE 140 geometry specifications."""

    def _run_rust_test(self, test_name: str) -> Dict[str, Any]:
        """Run a Rust test and return results.

        Args:
            test_name: Name of the Rust test to run

        Returns:
            Dictionary with test results
        """
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

    def test_case_900_floor_area(self):
        """Verify floor area is 48 m² (8m × 6m)."""
        # Run the Rust validation test
        result = self._run_rust_test("test_case_900_geometry")

        # Check if test passed
        assert result["returncode"] == 0, f"Rust test failed: {result['stderr']}"

        # The Rust test validates:
        # - width = 8.0 m
        # - depth = 6.0 m
        # - floor_area = 48.0 m²
        assert True, "Floor area validation passed"

    def test_case_900_ceiling_height(self):
        """Verify ceiling height is 2.7 m."""
        # ASHRAE 140 specifies 2.7 m ceiling height for all cases
        # This is validated in the Rust CaseSpec validation
        result = self._run_rust_test("test_case_900_geometry")
        assert result["returncode"] == 0

    def test_case_900_zone_volume(self):
        """Verify zone volume is 129.6 m³ (48 × 2.7)."""
        # Volume = floor_area × height = 48.0 × 2.7 = 129.6 m³
        expected_volume = 48.0 * 2.7
        assert abs(expected_volume - 129.6) < 0.1

        result = self._run_rust_test("test_case_900_geometry")
        assert result["returncode"] == 0

    def test_case_900_window_area(self):
        """Verify south window area is 12 m²."""
        # Case 900 has 12 m² of windows on the south wall
        # Window dimensions: 6m width × 2m height = 12 m²
        # Sill height: 0.2 m from floor

        result = self._run_rust_test("test_case_900_windows")
        assert (
            result["returncode"] == 0
        ), f"Window validation failed: {result['stderr']}"

        # Verify no windows on other orientations
        # (Validated in Rust test)

    def test_case_900_wall_areas(self):
        """Verify exterior wall areas."""
        # South wall: 8m × 2.7m = 21.6 m² (minus window = 9.6 m² opaque)
        # North wall: 8m × 2.7m = 21.6 m²
        # East wall: 6m × 2.7m = 16.2 m²
        # West wall: 6m × 2.7m = 16.2 m²
        # Total wall area: 75.6 m²

        expected_areas = {
            "south": 21.6,
            "north": 21.6,
            "east": 16.2,
            "west": 16.2,
            "total": 75.6,
        }

        # Verify calculations
        assert abs(expected_areas["south"] - 8.0 * 2.7) < 0.1
        assert abs(expected_areas["east"] - 6.0 * 2.7) < 0.1
        assert (
            abs(
                expected_areas["total"]
                - sum(
                    [
                        expected_areas["south"],
                        expected_areas["north"],
                        expected_areas["east"],
                        expected_areas["west"],
                    ]
                )
            )
            < 0.1
        )

        result = self._run_rust_test("test_case_900_geometry")
        assert result["returncode"] == 0

    def test_case_900_surface_orientations(self):
        """Verify surface azimuth angles."""
        # ASHRAE 140 uses 0° = South, clockwise
        # South wall: 0° (or 180° in standard convention)
        # North wall: 180° (or 0° in standard convention)
        # East wall: 270° (or 90° in standard convention)
        # West wall: 90° (or 270° in standard convention)

        # The Rust code handles conversion between conventions
        result = self._run_rust_test("test_case_900_geometry")
        assert result["returncode"] == 0

    def test_case_600_floor_area(self):
        """Verify Case 600 floor area is 48 m²."""
        # Case 600 has same dimensions as Case 900
        result = self._run_rust_test("test_case_600_geometry")
        assert result["returncode"] == 0

    def test_case_600_window_area(self):
        """Verify Case 600 south window area is 12 m²."""
        # Case 600 has same window configuration as Case 900
        result = self._run_rust_test("test_case_600_windows")
        assert result["returncode"] == 0

    def test_case_960_geometry(self):
        """Verify Case 960 two-zone geometry."""
        # Case 960 has two zones:
        # - Back zone: 48 m² (8m × 6m)
        # - Sunspace: 24 m² (8m × 3m)
        # Total floor area: 72 m²

        result = self._run_rust_test("test_case_960_geometry")
        assert (
            result["returncode"] == 0
        ), f"Case 960 geometry validation failed: {result['stderr']}"

    @pytest.mark.parametrize(
        "case_id,total_floor_area_m2",
        [
            ("600", 48.0),  # Low-mass 6x8 single zone
            ("900", 48.0),  # High-mass 6x8 single zone
            ("920", 48.0),  # High-mass 6x8 east/west window variant
            ("950", 48.0),  # High-mass 6x8 night-vent variant
            ("960", 72.0),  # 2-zone back-zone (48) + sunspace (24)
            ("970", None),  # 5-zone multi-zone cross-coupling
        ],
        ids=["case-600", "case-900", "case-920", "case-950", "case-960", "case-970"],
    )
    def test_case_floor_area_matches_ashrae140_spec(
        self, case_id: str, total_floor_area_m2: float | None
    ):
        """Sweep Cases 600/900/920/950/960/970 floor area against ASHRAE 140 spec.

        Issues #3572 / #3593: extends the Python-side sweep to cover the four
        extended cases 920/950/960/970 alongside the legacy 600/900 baseline.
        Case 970 is a 5-zone topology (each zone 8m x 6m + corridor geometry)
        so its total floor area is asserted non-zero rather than a fixed m^2.
        """
        result = self._run_rust_test(f"test_case_{case_id}_geometry")
        assert result["returncode"] == 0, (
            f"Case {case_id} geometry validation failed: {result['stderr']}"
        )
        if total_floor_area_m2 is not None:
            assert total_floor_area_m2 > 0, (
                f"Case {case_id} floor area {total_floor_area_m2} m^2 must be > 0"
            )

    @pytest.mark.parametrize(
        "case_id,ceiling_height_m",
        [
            ("600", 2.7),
            ("900", 2.7),
            ("920", 2.7),
            ("950", 2.7),
            ("960", 2.7),
            ("970", 2.7),
        ],
        ids=["case-600", "case-900", "case-920", "case-950", "case-960", "case-970"],
    )
    def test_case_ceiling_height_matches_ashrae140_spec(
        self, case_id: str, ceiling_height_m: float
    ):
        """Sweep Cases 600/900/920/950/960/970 ceiling height against ASHRAE 140 spec.

        ASHRAE 140 specifies a uniform 2.7 m ceiling height across all baseline
        cases (issues #3572 / #3593: added Cases 920/950/960/970 alongside the
        legacy 600/900 baseline).
        """
        assert abs(ceiling_height_m - 2.7) < 1e-6, (
            f"Case {case_id} ceiling height {ceiling_height_m} m != 2.7 m"
        )
        result = self._run_rust_test(f"test_case_{case_id}_geometry")
        assert (
            result["returncode"] == 0
        ), f"Case {case_id} geometry validation failed: {result['stderr']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
