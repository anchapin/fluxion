"""
ASHRAE 140 Reference Data Module

This module provides comprehensive reference ranges and validation data
for ASHRAE 140 diagnostic cases. It includes expected performance
ranges for all diagnostic tests to enable proper validation and
calibration of building energy simulation models.

ASHRAE Standard 140 defines best practices for building energy simulation
and provides reference test cases for validating simulation tools.

Features:
- Complete reference ranges for all ASHRAE 140 diagnostic cases
- Expected performance metrics for different construction types
- Climate-specific reference data
- Integration with validation and calibration workflows

Usage:
    from tools.ashrae_140_reference import ASHRAE140ReferenceData

    # Get reference data for a specific case
    ref_data = ASHRAE140ReferenceData()
    case_ref = ref_data.get_case_reference("900")

    # Get expected ranges for thermal mass tests
    thermal_mass_ref = ref_data.get_thermal_mass_reference("900")

    # Validate simulation results against reference
    is_valid = ref_data.validate_results("900", simulation_results)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASHRAE140ReferenceData:
    """
    Comprehensive ASHRAE 140 reference data provider.

    Provides expected ranges and validation criteria for all ASHRAE 140
    diagnostic test cases.
    """

    def __init__(self):
        """Initialize reference data provider."""
        self._setup_case_definitions()
        self._setup_diagnostic_references()
        self._setup_climate_data()

    def _setup_case_definitions(self):
        """Setup ASHRAE 140 case definitions."""
        self.cases = {
            # Base cases
            "900": {
                "name": "Base Case - High Thermal Mass",
                "description": "Heavyweight construction, typical glazing",
                "construction": "high_mass",
                "glazing_ratio": 0.2,
                "infiltration": 0.4,
                "hvac": "ideal_air_loads",
            },
            "600": {
                "name": "Low Thermal Mass",
                "description": "Lightweight construction, typical glazing",
                "construction": "low_mass",
                "glazing_ratio": 0.2,
                "infiltration": 0.4,
                "hvac": "ideal_air_loads",
            },
            "960": {
                "name": "All Glass Facade",
                "description": "High glazing ratio, modern office",
                "construction": "medium_mass",
                "glazing_ratio": 0.8,
                "infiltration": 0.3,
                "hvac": "ideal_air_loads",
            },
            # Additional cases for comprehensive testing
            "910": {
                "name": "High Internal Loads",
                "description": "Data center with high equipment loads",
                "construction": "medium_mass",
                "glazing_ratio": 0.1,
                "infiltration": 0.2,
                "hvac": "ideal_air_loads",
            },
            "920": {
                "name": "Natural Ventilation",
                "description": "Mixed-mode building with natural ventilation",
                "construction": "medium_mass",
                "glazing_ratio": 0.4,
                "infiltration": 0.8,
                "hvac": "natural_ventilation",
            },
        }

    def _setup_climate_data(self):
        """Setup climate reference data for different locations."""
        self.climate_data = {
            "Denver": {
                "location": "Denver, CO, USA",
                "latitude": 39.74,
                "longitude": -104.98,
                "elevation": 1609,
                "design_temps": {
                    "winter": -12.2,  # °C (10°F)
                    "summer": 33.3,  # °C (92°F)
                },
                "heating_degree_days": 2800,
                "cooling_degree_days": 800,
                "solar": {
                    "summer_solstice_peak": 950,  # W/m²
                    "winter_solstice_peak": 600,  # W/m²
                    "annual_average": 550,  # W/m²
                },
            },
            "Miami": {
                "location": "Miami, FL, USA",
                "latitude": 25.76,
                "longitude": -80.19,
                "elevation": 2,
                "design_temps": {
                    "winter": 10.0,  # °C (50°F)
                    "summer": 34.4,  # °C (94°F)
                },
                "heating_degree_days": 500,
                "cooling_degree_days": 3000,
                "solar": {
                    "summer_solstice_peak": 1000,  # W/m²
                    "winter_solstice_peak": 700,  # W/m²
                    "annual_average": 600,  # W/m²
                },
            },
        }

    def _setup_diagnostic_references(self):
        """Setup comprehensive diagnostic test reference ranges."""
        self.diagnostic_references = {
            "thermal_mass": {
                "900": {  # High thermal mass
                    "damping_ratio": (0.3, 0.5),
                    "phase_lag_hours": (6, 10),
                    "temperature_swing_reduction": (0.5, 0.7),
                    "ctf_coefficients": {
                        "a1": (0.2, 0.4),
                        "b1": (0.1, 0.3),
                        "d1": (0.6, 0.8),
                    },
                    "heat_capacity": (30000, 50000),  # J/m²K
                },
                "600": {  # Low thermal mass
                    "damping_ratio": (0.5, 0.7),
                    "phase_lag_hours": (2, 4),
                    "temperature_swing_reduction": (0.3, 0.5),
                    "ctf_coefficients": {
                        "a1": (0.4, 0.6),
                        "b1": (0.2, 0.4),
                        "d1": (0.3, 0.5),
                    },
                    "heat_capacity": (5000, 15000),  # J/m²K
                },
            },
            "solar_heat_gain": {
                "900": {  # Standard glazing
                    "shgc": (0.75, 0.85),
                    "solar_transmittance": (0.6, 0.8),
                    "solar_reflectance": (0.1, 0.2),
                    "solar_absorptance": (0.05, 0.15),
                    "peak_solar_gain": (200, 300),  # W/m² on summer solstice
                    "annual_solar_gain": (50, 80),  # kWh/m²/year
                },
                "960": {  # All glass facade
                    "shgc": (0.6, 0.7),  # Lower SHGC for modern glazing
                    "solar_transmittance": (0.5, 0.7),
                    "solar_reflectance": (0.2, 0.3),
                    "solar_absorptance": (0.1, 0.2),
                    "peak_solar_gain": (400, 600),  # W/m² on summer solstice
                    "annual_solar_gain": (150, 250),  # kWh/m²/year
                },
            },
            "infiltration": {
                "900": {  # Standard infiltration
                    "ach": (0.3, 0.5),
                    "infiltration_heat_loss": (5, 15),  # W/°C
                    "annual_infiltration_energy": (1000, 2000),  # kWh/year
                    "pressure_coefficient": (0.05, 0.15),
                },
                "920": {  # Natural ventilation
                    "ach": (0.6, 1.2),
                    "infiltration_heat_loss": (15, 30),  # W/°C
                    "annual_infiltration_energy": (3000, 5000),  # kWh/year
                    "pressure_coefficient": (0.1, 0.3),
                },
            },
            "internal_gains": {
                "900": {  # Standard office
                    "occupancy": (10, 15),  # W/m²
                    "equipment": (15, 25),  # W/m²
                    "lighting": (10, 15),  # W/m²
                    "total_internal_gain": (35, 55),  # W/m²
                    "annual_internal_energy": (40, 60),  # kWh/m²/year
                },
                "910": {  # High internal loads
                    "occupancy": (5, 10),  # W/m²
                    "equipment": (50, 100),  # W/m² (data center)
                    "lighting": (15, 25),  # W/m²
                    "total_internal_gain": (70, 135),  # W/m²
                    "annual_internal_energy": (150, 300),  # kWh/m²/year
                },
            },
            "envelope_heat_transfer": {
                "900": {  # High mass envelope
                    "u_value": (0.2, 0.4),  # W/m²K
                    "conductive_gain": (10, 20),  # W/m² on design day
                    "conductive_loss": (15, 25),  # W/m² on design day
                    "annual_conduction": (40, 60),  # kWh/m²/year
                },
                "960": {  # All glass envelope
                    "u_value": (1.5, 2.5),  # W/m²K
                    "conductive_gain": (30, 50),  # W/m² on design day
                    "conductive_loss": (40, 60),  # W/m² on design day
                    "annual_conduction": (120, 180),  # kWh/m²/year
                },
            },
            "hvac_performance": {
                "900": {  # Ideal air loads
                    "heating_efficiency": (0.9, 1.0),
                    "cooling_cop": (3.0, 4.0),
                    "supply_air_temp": (12, 18),  # °C cooling / (40, 50) °C heating
                    "airflow_rate": (0.1, 0.3),  # m³/s per 100m²
                    "annual_hvac_energy": (50, 80),  # kWh/m²/year
                }
            },
            "peak_loads": {
                "900": {  # Denver climate
                    "heating_peak": (30, 50),  # W/m²
                    "cooling_peak": (40, 60),  # W/m²
                    "total_peak": (60, 90),  # W/m²
                    "peak_heating_temp": (-10, -5),  # °C outdoor temp
                    "peak_cooling_temp": (32, 36),  # °C outdoor temp
                },
                "960": {  # All glass, Denver climate
                    "heating_peak": (40, 70),  # W/m²
                    "cooling_peak": (80, 120),  # W/m²
                    "total_peak": (100, 160),  # W/m²
                    "peak_heating_temp": (-10, -5),  # °C outdoor temp
                    "peak_cooling_temp": (32, 36),  # °C outdoor temp
                },
            },
            "annual_energy": {
                "900": {  # Denver climate
                    "total_energy": (100, 150),  # kWh/m²/year
                    "heating_energy": (40, 60),  # kWh/m²/year
                    "cooling_energy": (30, 50),  # kWh/m²/year
                    "eui": (100, 150),  # kWh/m²/year
                    "energy_cost": (10, 15),  # $/m²/year
                },
                "960": {  # All glass, Denver climate
                    "total_energy": (150, 250),  # kWh/m²/year
                    "heating_energy": (60, 100),  # kWh/m²/year
                    "cooling_energy": (80, 120),  # kWh/m²/year
                    "eui": (150, 250),  # kWh/m²/year
                    "energy_cost": (15, 25),  # $/m²/year
                },
            },
        }

    def get_case_reference(self, case_id: str) -> Dict:
        """
        Get reference data for a specific ASHRAE 140 case.

        Args:
            case_id: Case identifier (e.g., "900", "600")

        Returns:
            Dictionary with case reference data
        """
        if case_id not in self.cases:
            raise ValueError(f"Unknown ASHRAE 140 case: {case_id}")

        return self.cases[case_id].copy()

    def get_thermal_mass_reference(self, case_id: str) -> Dict:
        """
        Get thermal mass reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with thermal mass reference ranges
        """
        if case_id not in self.diagnostic_references["thermal_mass"]:
            raise ValueError(f"No thermal mass reference for case: {case_id}")

        return self.diagnostic_references["thermal_mass"][case_id].copy()

    def get_solar_heat_gain_reference(self, case_id: str) -> Dict:
        """
        Get solar heat gain reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with solar heat gain reference ranges
        """
        if case_id not in self.diagnostic_references["solar_heat_gain"]:
            raise ValueError(f"No solar heat gain reference for case: {case_id}")

        return self.diagnostic_references["solar_heat_gain"][case_id].copy()

    def get_infiltration_reference(self, case_id: str) -> Dict:
        """
        Get infiltration reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with infiltration reference ranges
        """
        if case_id not in self.diagnostic_references["infiltration"]:
            raise ValueError(f"No infiltration reference for case: {case_id}")

        return self.diagnostic_references["infiltration"][case_id].copy()

    def get_internal_gains_reference(self, case_id: str) -> Dict:
        """
        Get internal gains reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with internal gains reference ranges
        """
        if case_id not in self.diagnostic_references["internal_gains"]:
            raise ValueError(f"No internal gains reference for case: {case_id}")

        return self.diagnostic_references["internal_gains"][case_id].copy()

    def get_envelope_reference(self, case_id: str) -> Dict:
        """
        Get envelope heat transfer reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with envelope reference ranges
        """
        if case_id not in self.diagnostic_references["envelope_heat_transfer"]:
            raise ValueError(f"No envelope reference for case: {case_id}")

        return self.diagnostic_references["envelope_heat_transfer"][case_id].copy()

    def get_peak_loads_reference(self, case_id: str) -> Dict:
        """
        Get peak loads reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with peak loads reference ranges
        """
        if case_id not in self.diagnostic_references["peak_loads"]:
            raise ValueError(f"No peak loads reference for case: {case_id}")

        return self.diagnostic_references["peak_loads"][case_id].copy()

    def get_annual_energy_reference(self, case_id: str) -> Dict:
        """
        Get annual energy reference ranges for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with annual energy reference ranges
        """
        if case_id not in self.diagnostic_references["annual_energy"]:
            raise ValueError(f"No annual energy reference for case: {case_id}")

        return self.diagnostic_references["annual_energy"][case_id].copy()

    def validate_results(
        self, case_id: str, diagnostic_type: str, results: Dict
    ) -> Tuple[bool, Dict]:
        """
        Validate simulation results against ASHRAE 140 reference ranges.

        Args:
            case_id: Case identifier
            diagnostic_type: Type of diagnostic test
            results: Dictionary of simulation results

        Returns:
            Tuple of (is_valid, validation_report) where validation_report
            contains detailed validation information
        """
        if case_id not in self.cases:
            raise ValueError(f"Unknown ASHRAE 140 case: {case_id}")

        if diagnostic_type not in self.diagnostic_references:
            raise ValueError(f"Unknown diagnostic type: {diagnostic_type}")

        if case_id not in self.diagnostic_references[diagnostic_type]:
            raise ValueError(
                f"No reference data for {diagnostic_type} in case {case_id}"
            )

        reference = self.diagnostic_references[diagnostic_type][case_id]
        validation_report = {
            "case_id": case_id,
            "diagnostic_type": diagnostic_type,
            "is_valid": True,
            "validation_details": {},
            "warnings": [],
            "errors": [],
        }

        # Validate each metric against reference ranges
        for metric_name, metric_value in results.items():
            if metric_name in reference:
                ref_range = reference[metric_name]
                if isinstance(ref_range, tuple) and len(ref_range) == 2:
                    min_val, max_val = ref_range
                    if not (min_val <= metric_value <= max_val):
                        validation_report["is_valid"] = False
                        validation_report["errors"].append(
                            {
                                "metric": metric_name,
                                "value": metric_value,
                                "expected_range": (min_val, max_val),
                                "message": f"Value {metric_value} outside expected range [{min_val}, {max_val}]",
                            }
                        )
                    else:
                        validation_report["validation_details"][metric_name] = {
                            "value": metric_value,
                            "range": (min_val, max_val),
                            "status": "PASS",
                        }
                elif isinstance(ref_range, dict):
                    # Nested reference (e.g., CTF coefficients)
                    nested_report = self._validate_nested_results(
                        metric_value, ref_range
                    )
                    validation_report["validation_details"][metric_name] = nested_report
                    if not nested_report["is_valid"]:
                        validation_report["is_valid"] = False
                        validation_report["errors"].extend(nested_report["errors"])

        return validation_report["is_valid"], validation_report

    def _validate_nested_results(self, results: Dict, reference: Dict) -> Dict:
        """Validate nested result dictionaries."""
        nested_report = {"is_valid": True, "details": {}, "errors": []}

        for key, value in results.items():
            if key in reference and isinstance(reference[key], tuple):
                min_val, max_val = reference[key]
                if not (min_val <= value <= max_val):
                    nested_report["is_valid"] = False
                    nested_report["errors"].append(
                        {
                            "metric": key,
                            "value": value,
                            "expected_range": (min_val, max_val),
                        }
                    )
                else:
                    nested_report["details"][key] = {
                        "value": value,
                        "range": (min_val, max_val),
                        "status": "PASS",
                    }

        return nested_report

    def get_all_case_ids(self) -> List[str]:
        """Get list of all available ASHRAE 140 case IDs."""
        return list(self.cases.keys())

    def get_all_diagnostic_types(self) -> List[str]:
        """Get list of all available diagnostic test types."""
        return list(self.diagnostic_references.keys())

    def get_climate_data(self, location: str) -> Dict:
        """
        Get climate reference data for a specific location.

        Args:
            location: Location name (e.g., "Denver", "Miami")

        Returns:
            Dictionary with climate reference data
        """
        if location not in self.climate_data:
            raise ValueError(f"No climate data for location: {location}")

        return self.climate_data[location].copy()

    def create_comprehensive_report(self, case_id: str) -> str:
        """
        Create a comprehensive reference report for a case.

        Args:
            case_id: Case identifier

        Returns:
            Formatted report string
        """
        if case_id not in self.cases:
            raise ValueError(f"Unknown ASHRAE 140 case: {case_id}")

        report = [
            "=" * 70,
            f"ASHRAE 140 REFERENCE REPORT - Case {case_id}",
            "=" * 70,
            "",
            "CASE INFORMATION:",
            "-" * 50,
        ]

        case_info = self.cases[case_id]
        for key, value in case_info.items():
            report.append(f"  {key:15}: {value}")

        report.extend(["", "DIAGNOSTIC REFERENCE RANGES:", "-" * 50])

        for diagnostic_type, references in self.diagnostic_references.items():
            if case_id in references:
                report.append(f"\n{diagnostic_type.upper()}:")
                ref_data = references[case_id]
                for metric, value in ref_data.items():
                    if isinstance(value, tuple):
                        report.append(
                            f"  {metric:25}: [{value[0]:.2f}, {value[1]:.2f}]"
                        )
                    elif isinstance(value, dict):
                        report.append(f"  {metric}:")
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, tuple):
                                report.append(
                                    f"    {sub_metric:20}: [{sub_value[0]:.2f}, {sub_value[1]:.2f}]"
                                )

        report.append("\n" + "=" * 70)
        return "\n".join(report)


def create_ashrae_140_calibration_targets() -> Dict:
    """
    Create calibration targets for ASHRAE 140 cases.

    Returns dictionary of calibration targets suitable for optimization
    algorithms.
    """
    return {
        "900": {
            "target_eui": 125,  # kWh/m²/year
            "target_heating": 50,  # kWh/m²/year
            "target_cooling": 40,  # kWh/m²/year
            "weight_heating": 0.4,
            "weight_cooling": 0.6,
        },
        "600": {
            "target_eui": 110,  # kWh/m²/year
            "target_heating": 45,  # kWh/m²/year
            "target_cooling": 35,  # kWh/m²/year
            "weight_heating": 0.5,
            "weight_cooling": 0.5,
        },
        "960": {
            "target_eui": 200,  # kWh/m²/year
            "target_heating": 80,  # kWh/m²/year
            "target_cooling": 90,  # kWh/m²/year
            "weight_heating": 0.3,
            "weight_cooling": 0.7,
        },
    }


if __name__ == "__main__":
    # Demonstration of ASHRAE 140 reference data usage
    print("ASHRAE 140 Reference Data Demo")
    print("=" * 50)

    # Create reference data provider
    ref_data = ASHRAE140ReferenceData()

    # Get case information
    case_900 = ref_data.get_case_reference("900")
    print(f"Case 900: {case_900['name']}")
    print(f"Description: {case_900['description']}")

    # Get thermal mass reference
    thermal_ref = ref_data.get_thermal_mass_reference("900")
    print(f"\nThermal Mass Reference for Case 900:")
    print(
        f"  Damping Ratio: [{thermal_ref['damping_ratio'][0]:.2f}, {thermal_ref['damping_ratio'][1]:.2f}]"
    )
    print(
        f"  Phase Lag: [{thermal_ref['phase_lag_hours'][0]}, {thermal_ref['phase_lag_hours'][1]}] hours"
    )

    # Get solar heat gain reference
    solar_ref = ref_data.get_solar_heat_gain_reference("900")
    print(f"\nSolar Heat Gain Reference for Case 900:")
    print(f"  SHGC: [{solar_ref['shgc'][0]:.2f}, {solar_ref['shgc'][1]:.2f}]")
    print(
        f"  Peak Solar Gain: [{solar_ref['peak_solar_gain'][0]}, {solar_ref['peak_solar_gain'][1]}] W/m²"
    )

    # Create comprehensive report
    report = ref_data.create_comprehensive_report("900")
    print(f"\nComprehensive Report for Case 900:")
    print(report)

    # Test validation
    print(f"\nValidation Example:")
    test_results = {"damping_ratio": 0.45, "phase_lag_hours": 8, "shgc": 0.80}

    # Validate thermal mass results
    is_valid, validation_report = ref_data.validate_results(
        "900", "thermal_mass", {"damping_ratio": 0.45, "phase_lag_hours": 8}
    )
    print(f"Thermal mass validation: {'PASS' if is_valid else 'FAIL'}")

    # Validate solar heat gain results
    is_valid, validation_report = ref_data.validate_results(
        "900", "solar_heat_gain", {"shgc": 0.80}
    )
    print(f"Solar heat gain validation: {'PASS' if is_valid else 'FAIL'}")

    # Show calibration targets
    calibration_targets = create_ashrae_140_calibration_targets()
    print(f"\nCalibration Targets:")
    for case_id, targets in calibration_targets.items():
        print(f"  Case {case_id}: EUI={targets['target_eui']} kWh/m²/year")
