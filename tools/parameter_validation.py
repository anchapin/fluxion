"""
Parameter Validation Module

This module provides comprehensive validation and realistic bounds checking
for building energy simulation parameters. It ensures that optimization
and calibration processes operate within physically meaningful ranges.

Features:
- Physical parameter bounds for building energy simulations
- ASHRAE 140 compliant validation ranges
- Customizable validation profiles
- Integration with optimization algorithms

Usage:
    from tools.parameter_validation import BuildingParameterValidator

    # Create validator with standard bounds
    validator = BuildingParameterValidator()

    # Validate parameters
    params = [0.5, 20.0, 26.0]  # u_value, heating_setpoint, cooling_setpoint
    is_valid, errors = validator.validate(params)

    # Get bounds for optimization
    bounds = validator.get_optimization_bounds()
"""

from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildingParameterValidator:
    """
    Validator for building energy simulation parameters.

    Provides realistic physical bounds and validation for common
    building energy simulation parameters based on ASHRAE standards
    and industry best practices.
    """

    def __init__(self, validation_profile: str = "standard"):
        """
        Initialize validator with specified profile.

        Args:
            validation_profile: Validation profile ('standard', 'strict', 'relaxed')
        """
        self.validation_profile = validation_profile
        self._setup_validation_ranges()

    def _setup_validation_ranges(self):
        """Setup parameter ranges based on validation profile."""
        if self.validation_profile == "strict":
            # ASHRAE 140 compliant strict ranges
            self.ranges = {
                "u_value": (0.2, 3.5),  # W/m²K - realistic building envelope
                "heating_setpoint": (18.0, 22.0),  # °C - comfortable heating range
                "cooling_setpoint": (24.0, 28.0),  # °C - comfortable cooling range
                "infiltration_rate": (0.1, 0.8),  # ACH - air changes per hour
                "thermal_mass": (50.0, 500.0),  # kJ/m²K - building heat capacity
                "solar_heat_gain": (0.2, 0.8),  # SHGC - solar heat gain coefficient
                "window_area_ratio": (0.1, 0.6),  # Window-to-wall ratio
                "occupancy_density": (5.0, 20.0),  # m²/person
                "equipment_power": (5.0, 30.0),  # W/m²
                "lighting_power": (5.0, 20.0),  # W/m²
            }
        elif self.validation_profile == "relaxed":
            # Wider ranges for exploratory analysis
            self.ranges = {
                "u_value": (0.1, 5.0),  # W/m²K
                "heating_setpoint": (15.0, 25.0),  # °C
                "cooling_setpoint": (22.0, 32.0),  # °C
                "infiltration_rate": (0.05, 1.5),  # ACH
                "thermal_mass": (20.0, 800.0),  # kJ/m²K
                "solar_heat_gain": (0.1, 0.9),  # SHGC
                "window_area_ratio": (0.05, 0.8),  # Window-to-wall ratio
                "occupancy_density": (2.0, 30.0),  # m²/person
                "equipment_power": (2.0, 50.0),  # W/m²
                "lighting_power": (2.0, 30.0),  # W/m²
            }
        else:  # standard profile
            # Balanced ranges suitable for most applications
            self.ranges = {
                "u_value": (0.15, 4.0),  # W/m²K
                "heating_setpoint": (16.0, 24.0),  # °C
                "cooling_setpoint": (23.0, 30.0),  # °C
                "infiltration_rate": (0.1, 1.0),  # ACH
                "thermal_mass": (30.0, 600.0),  # kJ/m²K
                "solar_heat_gain": (0.15, 0.85),  # SHGC
                "window_area_ratio": (0.1, 0.7),  # Window-to-wall ratio
                "occupancy_density": (3.0, 25.0),  # m²/person
                "equipment_power": (3.0, 40.0),  # W/m²
                "lighting_power": (3.0, 25.0),  # W/m²
            }

        # Additional constraints
        self.constraints = {
            "setpoint_difference": 2.0,  # Minimum difference between heating and cooling setpoints
            "u_value_relationships": {
                "wall": (0.1, 2.0),
                "roof": (0.1, 1.5),
                "floor": (0.1, 1.8),
                "window": (0.8, 3.5),
            },
        }

    def validate(
        self, parameters: List[float], parameter_names: Optional[List[str]] = None
    ) -> Tuple[bool, Dict]:
        """
        Validate a set of parameters.

        Args:
            parameters: List of parameter values
            parameter_names: Optional list of parameter names (order matters)
                           If None, assumes standard order: [u_value, heating_setpoint, cooling_setpoint]

        Returns:
            Tuple of (is_valid, errors) where errors is a dict of validation errors
        """
        errors = {}

        if parameter_names is None:
            # Default parameter order
            parameter_names = ["u_value", "heating_setpoint", "cooling_setpoint"]

        # Check parameter count
        if len(parameters) != len(parameter_names):
            errors["count_mismatch"] = (
                f"Expected {len(parameter_names)} parameters, got {len(parameters)}"
            )
            return False, errors

        # Check each parameter against its range
        for i, (param_name, param_value) in enumerate(zip(parameter_names, parameters)):
            if param_name not in self.ranges:
                errors[f"unknown_parameter_{i}"] = f"Unknown parameter: {param_name}"
                continue

            min_val, max_val = self.ranges[param_name]
            if not (min_val <= param_value <= max_val):
                errors[param_name] = (
                    f"Value {param_value} outside range [{min_val}, {max_val}]"
                )

        # Check additional constraints
        self._check_additional_constraints(parameters, parameter_names, errors)

        return len(errors) == 0, errors

    def _check_additional_constraints(
        self, parameters: List[float], parameter_names: List[str], errors: Dict
    ):
        """Check constraints that involve multiple parameters."""

        # Check setpoint difference constraint
        if (
            "heating_setpoint" in parameter_names
            and "cooling_setpoint" in parameter_names
        ):
            heat_idx = parameter_names.index("heating_setpoint")
            cool_idx = parameter_names.index("cooling_setpoint")

            heating = parameters[heat_idx]
            cooling = parameters[cool_idx]

            if cooling <= heating:
                errors["setpoint_order"] = (
                    f"Cooling setpoint ({cooling}) must be greater than heating setpoint ({heating})"
                )

            min_diff = self.constraints["setpoint_difference"]
            if cooling - heating < min_diff:
                errors["setpoint_difference"] = (
                    f"Setpoint difference ({cooling - heating:.1f}) < minimum ({min_diff})"
                )

        # Add more complex constraints as needed

    def get_optimization_bounds(
        self, parameter_names: Optional[List[str]] = None
    ) -> List[Tuple[float, float]]:
        """
        Get bounds suitable for optimization algorithms.

        Args:
            parameter_names: List of parameter names (order matters)
                           If None, returns bounds for standard parameters

        Returns:
            List of (min, max) tuples in the same order as parameter_names
        """
        if parameter_names is None:
            parameter_names = ["u_value", "heating_setpoint", "cooling_setpoint"]

        bounds = []
        for param_name in parameter_names:
            if param_name in self.ranges:
                bounds.append(self.ranges[param_name])
            else:
                # Default bounds for unknown parameters
                bounds.append((0.0, 1.0))
                logger.warning(
                    f"Using default bounds for unknown parameter: {param_name}"
                )

        return bounds

    def get_parameter_info(self) -> Dict:
        """Get information about all validated parameters."""
        info = {}
        for param_name, (min_val, max_val) in self.ranges.items():
            info[param_name] = {
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "profile": self.validation_profile,
            }
        return info

    def clamp_parameters(
        self, parameters: List[float], parameter_names: Optional[List[str]] = None
    ) -> List[float]:
        """
        Clamp parameter values to valid ranges.

        Args:
            parameters: List of parameter values
            parameter_names: Optional list of parameter names

        Returns:
            List of clamped parameter values
        """
        if parameter_names is None:
            parameter_names = ["u_value", "heating_setpoint", "cooling_setpoint"]

        clamped = []
        for i, (param_name, param_value) in enumerate(zip(parameter_names, parameters)):
            if param_name in self.ranges:
                min_val, max_val = self.ranges[param_name]
                clamped_value = max(min_val, min(param_value, max_val))
                clamped.append(clamped_value)
            else:
                clamped.append(param_value)

        # Apply additional constraints
        clamped = self._apply_constraints_post_clamping(clamped, parameter_names)

        return clamped

    def _apply_constraints_post_clamping(
        self, parameters: List[float], parameter_names: List[str]
    ) -> List[float]:
        """Apply constraints after individual parameter clamping."""

        # Ensure setpoint difference constraint
        if (
            "heating_setpoint" in parameter_names
            and "cooling_setpoint" in parameter_names
        ):
            heat_idx = parameter_names.index("heating_setpoint")
            cool_idx = parameter_names.index("cooling_setpoint")

            heating = parameters[heat_idx]
            cooling = parameters[cool_idx]

            # Ensure cooling > heating with minimum difference
            if cooling <= heating:
                min_diff = self.constraints["setpoint_difference"]
                heat_min, heat_max = self.ranges["heating_setpoint"]
                cool_min, cool_max = self.ranges["cooling_setpoint"]

                # Try to find valid setpoints with minimum difference
                # Start by adjusting cooling up and heating down
                new_cooling = cooling + min_diff
                new_heating = heating - min_diff

                # Ensure they stay within individual bounds
                new_heating = max(heat_min, min(heat_max, new_heating))
                new_cooling = max(cool_min, min(cool_max, new_cooling))

                # If still not valid, find the closest valid combination
                if new_cooling <= new_heating:
                    # Set cooling to heating + min_diff, adjusting within bounds
                    new_heating = min(heating, cool_max - min_diff)
                    new_heating = max(heat_min, new_heating)
                    new_cooling = new_heating + min_diff
                    new_cooling = min(cool_max, new_cooling)

                parameters[heat_idx] = new_heating
                parameters[cool_idx] = new_cooling

        return parameters

    def create_validation_report(
        self, parameters: List[float], parameter_names: Optional[List[str]] = None
    ) -> str:
        """Create a detailed validation report."""
        is_valid, errors = self.validate(parameters, parameter_names)

        report = [
            "=" * 60,
            "PARAMETER VALIDATION REPORT",
            "=" * 60,
            f"Validation Profile: {self.validation_profile.upper()}",
            f"Overall Status: {'VALID' if is_valid else 'INVALID'}",
            "",
        ]

        if parameter_names is None:
            parameter_names = ["u_value", "heating_setpoint", "cooling_setpoint"]

        # Parameter details
        report.append("Parameter Details:")
        report.append("-" * 40)
        for i, (param_name, param_value) in enumerate(zip(parameter_names, parameters)):
            if param_name in self.ranges:
                min_val, max_val = self.ranges[param_name]
                status = "✓" if (min_val <= param_value <= max_val) else "✗"
                report.append(
                    f"  {param_name:20}: {param_value:8.3f} [{min_val:6.2f}, {max_val:6.2f}] {status}"
                )
            else:
                report.append(
                    f"  {param_name:20}: {param_value:8.3f} [Unknown parameter]"
                )

        # Errors
        if errors:
            report.append("")
            report.append("Validation Errors:")
            report.append("-" * 40)
            for error_key, error_msg in errors.items():
                report.append(f"  • {error_key}: {error_msg}")

        report.append("=" * 60)
        return "\n".join(report)


class ASHRAE140Validator(BuildingParameterValidator):
    """
    ASHRAE 140 specific validator with reference ranges.

    Provides validation ranges based on ASHRAE Standard 140
    "Best Practices for Building Energy Simulation" reference cases.
    """

    def __init__(self, case_type: str = "base"):
        """
        Initialize ASHRAE 140 validator.

        Args:
            case_type: Type of ASHRAE 140 case ('base', 'high_mass', 'low_mass', 'all_glass')
        """
        super().__init__("strict")  # Start with strict profile
        self.case_type = case_type
        self._setup_ashrae_140_ranges()

    def _setup_ashrae_140_ranges(self):
        """Setup ASHRAE 140 specific ranges."""
        # Base ranges from ASHRAE 140
        ashrae_base = {
            "u_value": (0.2, 2.5),  # Typical building envelope
            "heating_setpoint": (18.0, 22.0),  # ASHRAE comfort range
            "cooling_setpoint": (24.0, 28.0),  # ASHRAE comfort range
            "infiltration_rate": (0.2, 0.6),  # Typical airtightness
            "thermal_mass": (100.0, 400.0),  # Typical construction
            "solar_heat_gain": (0.3, 0.6),  # Typical glazing
        }

        # Case-specific adjustments
        if self.case_type == "high_mass":
            ashrae_base["thermal_mass"] = (300.0, 600.0)
        elif self.case_type == "low_mass":
            ashrae_base["thermal_mass"] = (50.0, 200.0)
        elif self.case_type == "all_glass":
            ashrae_base["u_value"] = (1.0, 3.0)
            ashrae_base["solar_heat_gain"] = (0.4, 0.7)

        # Update ranges
        for key, value in ashrae_base.items():
            if key in self.ranges:
                self.ranges[key] = value

    def get_ashrae_140_reference(self) -> Dict:
        """Get ASHRAE 140 reference values for the case type."""
        references = {
            "base": {
                "u_value": 1.5,
                "heating_setpoint": 20.0,
                "cooling_setpoint": 26.0,
                "infiltration_rate": 0.4,
                "thermal_mass": 200.0,
                "solar_heat_gain": 0.45,
            },
            "high_mass": {
                "u_value": 1.5,
                "heating_setpoint": 20.0,
                "cooling_setpoint": 26.0,
                "infiltration_rate": 0.4,
                "thermal_mass": 400.0,
                "solar_heat_gain": 0.45,
            },
            "low_mass": {
                "u_value": 1.5,
                "heating_setpoint": 20.0,
                "cooling_setpoint": 26.0,
                "infiltration_rate": 0.4,
                "thermal_mass": 100.0,
                "solar_heat_gain": 0.45,
            },
            "all_glass": {
                "u_value": 2.0,
                "heating_setpoint": 20.0,
                "cooling_setpoint": 26.0,
                "infiltration_rate": 0.4,
                "thermal_mass": 150.0,
                "solar_heat_gain": 0.6,
            },
        }

        return references.get(self.case_type, references["base"])


def create_physical_constraints() -> Dict:
    """Create dictionary of physical constraints for building parameters."""
    return {
        "temperature": {
            "min": -20.0,  # °C - minimum realistic indoor temperature
            "max": 50.0,  # °C - maximum realistic indoor temperature
        },
        "humidity": {
            "min": 10.0,  # % - minimum realistic relative humidity
            "max": 90.0,  # % - maximum realistic relative humidity
        },
        "airflow": {
            "min": 0.05,  # m³/s - minimum ventilation airflow
            "max": 10.0,  # m³/s - maximum realistic airflow for small buildings
        },
        "pressure": {
            "min": 95000.0,  # Pa - minimum atmospheric pressure
            "max": 105000.0,  # Pa - maximum atmospheric pressure
        },
    }


if __name__ == "__main__":
    # Demonstration of parameter validation
    print("Building Parameter Validation Demo")
    print("=" * 50)

    # Test standard validator
    validator = BuildingParameterValidator("standard")

    # Test valid parameters
    valid_params = [0.5, 20.0, 26.0]
    is_valid, errors = validator.validate(valid_params)
    print(f"Valid parameters {valid_params}: {is_valid}")

    # Test invalid parameters
    invalid_params = [
        0.05,
        30.0,
        25.0,
    ]  # u_value too low, heating too high, cooling too low
    is_valid, errors = validator.validate(invalid_params)
    print(f"Invalid parameters {invalid_params}: {is_valid}")
    print(f"Errors: {errors}")

    # Get optimization bounds
    bounds = validator.get_optimization_bounds()
    print(f"Optimization bounds: {bounds}")

    # Test clamping
    clamped = validator.clamp_parameters(invalid_params)
    print(f"Clamped parameters: {clamped}")

    # Test ASHRAE 140 validator
    print("\nASHRAE 140 Validation:")
    ashrae_validator = ASHRAE140Validator("base")
    ashrae_params = [1.5, 20.0, 26.0, 0.4, 200.0]
    ashrae_names = [
        "u_value",
        "heating_setpoint",
        "cooling_setpoint",
        "infiltration_rate",
        "thermal_mass",
    ]
    is_valid, errors = ashrae_validator.validate(ashrae_params, ashrae_names)
    print(f"ASHRAE 140 parameters valid: {is_valid}")

    # Create validation report
    report = validator.create_validation_report(valid_params)
    print("\nValidation Report:")
    print(report)
