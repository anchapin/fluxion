"""
Test suite for ASHRAE 140 reference data module.

This module tests the ASHRAE140ReferenceData class and related functions.
"""

import unittest

from tools.ashrae_140_reference import (
    ASHRAE140ReferenceData,
    create_ashrae_140_calibration_targets,
)


class TestASHRAE140ReferenceData(unittest.TestCase):
    """Test cases for ASHRAE 140 reference data."""

    def setUp(self):
        """Set up test fixtures."""
        self.ref_data = ASHRAE140ReferenceData()

    def test_case_definitions(self):
        """Test that case definitions are properly set up."""
        cases = self.ref_data.get_all_case_ids()

        # Check that expected cases are present
        self.assertIn("900", cases)
        self.assertIn("600", cases)
        self.assertIn("960", cases)
        self.assertIn("910", cases)
        self.assertIn("920", cases)

    def test_case_900_reference(self):
        """Test Case 900 reference data."""
        case_ref = self.ref_data.get_case_reference("900")

        self.assertEqual(case_ref["name"], "Base Case - High Thermal Mass")
        self.assertEqual(case_ref["construction"], "high_mass")
        self.assertEqual(case_ref["glazing_ratio"], 0.2)

    def test_thermal_mass_references(self):
        """Test thermal mass reference ranges."""
        # Test Case 900 (high mass)
        ref_900 = self.ref_data.get_thermal_mass_reference("900")
        self.assertEqual(ref_900["damping_ratio"], (0.3, 0.5))
        self.assertEqual(ref_900["phase_lag_hours"], (6, 10))
        self.assertGreaterEqual(ref_900["heat_capacity"][0], 30000)

        # Test Case 600 (low mass)
        ref_600 = self.ref_data.get_thermal_mass_reference("600")
        self.assertEqual(ref_600["damping_ratio"], (0.5, 0.7))
        self.assertEqual(ref_600["phase_lag_hours"], (2, 4))
        self.assertLess(ref_600["heat_capacity"][1], 20000)

    def test_solar_heat_gain_references(self):
        """Test solar heat gain reference ranges."""
        # Test Case 900
        ref_900 = self.ref_data.get_solar_heat_gain_reference("900")
        self.assertEqual(ref_900["shgc"], (0.75, 0.85))
        self.assertGreaterEqual(ref_900["peak_solar_gain"][0], 200)

        # Test Case 960 (all glass)
        ref_960 = self.ref_data.get_solar_heat_gain_reference("960")
        self.assertEqual(ref_960["shgc"], (0.6, 0.7))
        self.assertGreaterEqual(ref_960["peak_solar_gain"][0], 400)

    def test_infiltration_references(self):
        """Test infiltration reference ranges."""
        ref_900 = self.ref_data.get_infiltration_reference("900")
        self.assertEqual(ref_900["ach"], (0.3, 0.5))

        ref_920 = self.ref_data.get_infiltration_reference("920")
        self.assertGreater(ref_920["ach"][0], 0.5)  # Higher for natural ventilation

    def test_internal_gains_references(self):
        """Test internal gains reference ranges."""
        ref_900 = self.ref_data.get_internal_gains_reference("900")
        self.assertLess(ref_900["total_internal_gain"][1], 60)

        ref_910 = self.ref_data.get_internal_gains_reference("910")
        self.assertGreater(ref_910["equipment"][0], 40)  # High internal loads

    def test_envelope_references(self):
        """Test envelope heat transfer references."""
        ref_900 = self.ref_data.get_envelope_reference("900")
        self.assertLess(ref_900["u_value"][1], 0.5)

        ref_960 = self.ref_data.get_envelope_reference("960")
        self.assertGreater(ref_960["u_value"][0], 1.0)  # All glass has higher U-value

    def test_peak_loads_references(self):
        """Test peak loads reference ranges."""
        ref_900 = self.ref_data.get_peak_loads_reference("900")
        self.assertLess(ref_900["heating_peak"][1], 60)
        self.assertLess(ref_900["cooling_peak"][1], 70)

        ref_960 = self.ref_data.get_peak_loads_reference("960")
        self.assertGreater(
            ref_960["cooling_peak"][0], 70
        )  # Higher cooling for all glass

    def test_annual_energy_references(self):
        """Test annual energy reference ranges."""
        ref_900 = self.ref_data.get_annual_energy_reference("900")
        self.assertLess(ref_900["eui"][1], 160)

        ref_960 = self.ref_data.get_annual_energy_reference("960")
        self.assertGreater(ref_960["eui"][0], 140)  # Higher EUI for all glass

    def test_validation_pass(self):
        """Test validation with results within reference ranges."""
        test_results = {"damping_ratio": 0.4, "phase_lag_hours": 8}

        is_valid, report = self.ref_data.validate_results(
            "900", "thermal_mass", test_results
        )

        self.assertTrue(is_valid)
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(len(report["validation_details"]), 2)

    def test_validation_fail(self):
        """Test validation with results outside reference ranges."""
        test_results = {
            "damping_ratio": 0.8,  # Too high for high mass
            "phase_lag_hours": 2,  # Too low for high mass
        }

        is_valid, report = self.ref_data.validate_results(
            "900", "thermal_mass", test_results
        )

        self.assertFalse(is_valid)
        self.assertEqual(len(report["errors"]), 2)
        self.assertIn("damping_ratio", report["errors"][0]["metric"])
        self.assertIn("phase_lag_hours", report["errors"][1]["metric"])

    def test_nested_validation(self):
        """Test validation of nested results (e.g., CTF coefficients)."""
        test_results = {"ctf_coefficients": {"a1": 0.3, "b1": 0.25, "d1": 0.7}}

        is_valid, report = self.ref_data.validate_results(
            "900", "thermal_mass", test_results
        )

        self.assertTrue(is_valid)
        self.assertIn("ctf_coefficients", report["validation_details"])

    def test_climate_data(self):
        """Test climate reference data."""
        denver_data = self.ref_data.get_climate_data("Denver")
        self.assertEqual(denver_data["latitude"], 39.74)
        self.assertGreater(denver_data["elevation"], 1500)

        miami_data = self.ref_data.get_climate_data("Miami")
        self.assertLess(miami_data["elevation"], 10)
        self.assertGreater(miami_data["cooling_degree_days"], 2000)

    def test_comprehensive_report(self):
        """Test comprehensive report generation."""
        report = self.ref_data.create_comprehensive_report("900")

        self.assertIn("ASHRAE 140 REFERENCE REPORT", report)
        self.assertIn("Case 900", report)
        self.assertIn("THERMAL_MASS", report)
        self.assertIn("SOLAR_HEAT_GAIN", report)

    def test_calibration_targets(self):
        """Test calibration targets function."""
        targets = create_ashrae_140_calibration_targets()

        self.assertIn("900", targets)
        self.assertIn("600", targets)
        self.assertIn("960", targets)

        # Check that targets are reasonable
        self.assertGreater(targets["900"]["target_eui"], 100)
        self.assertLess(targets["600"]["target_eui"], 120)
        self.assertGreater(targets["960"]["target_eui"], 150)

    def test_unknown_case_validation(self):
        """Test validation with unknown case."""
        with self.assertRaises(ValueError):
            self.ref_data.get_case_reference("999")

        with self.assertRaises(ValueError):
            self.ref_data.validate_results("999", "thermal_mass", {})

    def test_unknown_diagnostic_type(self):
        """Test validation with unknown diagnostic type."""
        with self.assertRaises(ValueError):
            self.ref_data.validate_results("900", "unknown_diagnostic", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
