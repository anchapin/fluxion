"""
Test suite for parameter validation module.

This module tests the BuildingParameterValidator and ASHRAE140Validator classes.
"""

import unittest

from tools.parameter_validation import ASHRAE140Validator, BuildingParameterValidator


class TestParameterValidation(unittest.TestCase):
    """Test cases for parameter validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = BuildingParameterValidator("standard")
        self.strict_validator = BuildingParameterValidator("strict")
        self.relaxed_validator = BuildingParameterValidator("relaxed")
        self.ashrae_validator = ASHRAE140Validator("base")

    def test_standard_validation_ranges(self):
        """Test standard validation ranges."""
        # Check that standard ranges are set correctly
        ranges = self.validator.ranges

        # Test u_value range
        self.assertEqual(ranges["u_value"], (0.15, 4.0))

        # Test setpoint ranges
        self.assertEqual(ranges["heating_setpoint"], (16.0, 24.0))
        self.assertEqual(ranges["cooling_setpoint"], (23.0, 30.0))

    def test_strict_validation_ranges(self):
        """Test strict validation ranges."""
        ranges = self.strict_validator.ranges

        # Strict ranges should be narrower
        self.assertEqual(ranges["u_value"], (0.2, 3.5))
        self.assertEqual(ranges["heating_setpoint"], (18.0, 22.0))

    def test_relaxed_validation_ranges(self):
        """Test relaxed validation ranges."""
        ranges = self.relaxed_validator.ranges

        # Relaxed ranges should be wider
        self.assertEqual(ranges["u_value"], (0.1, 5.0))
        self.assertEqual(ranges["heating_setpoint"], (15.0, 25.0))

    def test_valid_parameters(self):
        """Test validation of valid parameters."""
        valid_params = [0.5, 20.0, 26.0]  # u_value, heating, cooling
        is_valid, errors = self.validator.validate(valid_params)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_u_value(self):
        """Test validation with invalid u_value."""
        invalid_params = [0.05, 20.0, 26.0]  # u_value too low
        is_valid, errors = self.validator.validate(invalid_params)

        self.assertFalse(is_valid)
        self.assertIn("u_value", errors)
        self.assertIn("0.05", errors["u_value"])

    def test_invalid_setpoints(self):
        """Test validation with invalid setpoints."""
        # Heating too high, cooling too low, and cooling < heating
        invalid_params = [0.5, 25.0, 22.0]
        is_valid, errors = self.validator.validate(invalid_params)

        self.assertFalse(is_valid)
        self.assertIn("heating_setpoint", errors)
        self.assertIn("cooling_setpoint", errors)
        self.assertIn("setpoint_order", errors)

    def test_setpoint_difference_constraint(self):
        """Test setpoint difference constraint."""
        # Setpoints too close together
        close_params = [0.5, 22.0, 22.5]  # Only 0.5°C difference
        is_valid, errors = self.validator.validate(close_params)

        self.assertFalse(is_valid)
        self.assertIn("setpoint_difference", errors)

    def test_get_optimization_bounds(self):
        """Test getting optimization bounds."""
        bounds = self.validator.get_optimization_bounds()

        self.assertEqual(len(bounds), 3)  # Default 3 parameters
        self.assertEqual(bounds[0], (0.15, 4.0))  # u_value
        self.assertEqual(bounds[1], (16.0, 24.0))  # heating_setpoint
        self.assertEqual(bounds[2], (23.0, 30.0))  # cooling_setpoint

    def test_custom_parameter_names(self):
        """Test validation with custom parameter names."""
        custom_params = [0.5, 20.0, 26.0, 0.5]
        custom_names = [
            "u_value",
            "heating_setpoint",
            "cooling_setpoint",
            "infiltration_rate",
        ]

        is_valid, errors = self.validator.validate(custom_params, custom_names)
        self.assertTrue(is_valid)

    def test_clamp_parameters(self):
        """Test parameter clamping."""
        invalid_params = [0.05, 15.0, 35.0]  # All out of bounds
        clamped = self.validator.clamp_parameters(invalid_params)

        # Should be clamped to valid ranges
        self.assertGreaterEqual(clamped[0], 0.15)  # u_value min
        self.assertLessEqual(clamped[0], 4.0)  # u_value max
        self.assertGreaterEqual(clamped[1], 16.0)  # heating min
        self.assertLessEqual(clamped[1], 24.0)  # heating max
        self.assertGreaterEqual(clamped[2], 23.0)  # cooling min
        self.assertLessEqual(clamped[2], 30.0)  # cooling max

    def test_setpoint_adjustment(self):
        """Test automatic setpoint adjustment."""
        # Invalid setpoints (cooling < heating)
        invalid_params = [0.5, 22.0, 21.0]
        clamped = self.validator.clamp_parameters(invalid_params)

        # Should be adjusted so cooling > heating
        self.assertLess(clamped[1], clamped[2])  # heating < cooling
        # The exact difference depends on the clamping logic, but it should be positive
        self.assertGreater(clamped[2] - clamped[1], 0)  # positive difference

    def test_ashrae_140_ranges(self):
        """Test ASHRAE 140 specific ranges."""
        ranges = self.ashrae_validator.ranges

        # ASHRAE 140 should have stricter ranges
        self.assertEqual(ranges["u_value"], (0.2, 2.5))
        self.assertEqual(ranges["heating_setpoint"], (18.0, 22.0))

    def test_ashrae_140_reference_values(self):
        """Test ASHRAE 140 reference values."""
        reference = self.ashrae_validator.get_ashrae_140_reference()

        self.assertEqual(reference["u_value"], 1.5)
        self.assertEqual(reference["heating_setpoint"], 20.0)
        self.assertEqual(reference["cooling_setpoint"], 26.0)

    def test_validation_report(self):
        """Test validation report generation."""
        valid_params = [0.5, 20.0, 26.0]
        report = self.validator.create_validation_report(valid_params)

        self.assertIn("PARAMETER VALIDATION REPORT", report)
        self.assertIn("VALID", report)
        self.assertIn("u_value", report)
        self.assertIn("heating_setpoint", report)
        self.assertIn("cooling_setpoint", report)

    def test_invalid_parameter_report(self):
        """Test validation report with invalid parameters."""
        invalid_params = [0.05, 30.0, 20.0]
        report = self.validator.create_validation_report(invalid_params)

        self.assertIn("INVALID", report)
        self.assertIn("Validation Errors", report)
        self.assertIn("u_value", report)
        self.assertIn("heating_setpoint", report)
        self.assertIn("cooling_setpoint", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
