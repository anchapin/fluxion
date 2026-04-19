//! Shared test fixtures for high-mass validation tests in Phase 44
//!
//! This file provides common test fixtures and utilities used across
//! high-mass validation tests to reduce duplication and ensure consistency.

use crate::physics::thermal_mass::construction::{ConstructionType, MaterialLayer};
use crate::physics::thermal_mass::diagnostics::{ThermalMassDiagnostics, ThermalMassReport};
use crate::sim::construction::ConstructionLayer;
use crate::validation::ashrae140::WeatherData;
use crate::validation::high_mass::test_cases::{
    BuildingConfig, HighMassValidationCase, ReferenceResults, ValidationTolerance,
};
use crate::validation::report::ValidationResult;
use crate::validation::tolerance::ValidationTolerance as Tol;
use std::sync::Arc;

/// Creates a standard heavyweight building configuration for testing
pub fn standard_heavyweight_config() -> BuildingConfig {
    BuildingConfig {
        construction_type: ConstructionType::HeavyWeight,
        floor_area: 232.0,
        u_value: 0.35,
        window_wall_ratio: 0.15,
        infiltration_rate: 0.3,
    }
}

/// Creates a standard mediumweight building configuration for testing
pub fn standard_mediumweight_config() -> BuildingConfig {
    BuildingConfig {
        construction_type: ConstructionType::MediumWeight,
        floor_area: 500.0,
        u_value: 0.40,
        window_wall_ratio: 0.30,
        infiltration_rate: 0.5,
    }
}

/// Creates a standard lightweight building configuration for testing
pub fn standard_lightweight_config() -> BuildingConfig {
    BuildingConfig {
        construction_type: ConstructionType::Lightweight,
        floor_area: 150.0,
        u_value: 0.50,
        window_wall_ratio: 0.25,
        infiltration_rate: 0.5,
    }
}

/// Creates standard reference results for testing (ASHRAE 140 Case 900 baseline)
pub fn standard_reference_results() -> ReferenceResults {
    ReferenceResults {
        hourly_temperatures: vec![20.0; 8760],
        hourly_heating: vec![0.8; 8760],
        hourly_cooling: vec![0.3; 8760],
        annual_heating: 7008.0,
        annual_cooling: 2628.0,
    }
}

/// Creates standard validation tolerance for testing
pub fn standard_validation_tolerance() -> ValidationTolerance {
    ValidationTolerance {
        nmbe_limit: 5.0,
        cv_rmse_limit: 10.0,
        mae_limit: 0.1,
    }
}

/// Creates a standard high-mass validation case for testing
pub fn standard_high_mass_case(case_id: &str) -> HighMassValidationCase {
    HighMassValidationCase::new(
        case_id.to_string(),
        standard_heavyweight_config(),
        WeatherData::default(),
        standard_reference_results(),
        standard_validation_tolerance(),
        format!("Standard test case {}", case_id),
    )
}

/// Creates construction layers for heavyweight construction (concrete + insulation)
pub fn heavyweight_construction_layers() -> Vec<ConstructionLayer> {
    vec![
        ConstructionLayer {
            name: "Concrete".to_string(),
            conductivity: 1.7,
            density: 2300.0,
            specific_heat: 840.0,
            thickness: 0.2, // 20cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Insulation".to_string(),
            conductivity: 0.04,
            density: 50.0,
            specific_heat: 840.0,
            thickness: 0.05, // 5cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
    ]
}

/// Creates construction layers for mediumweight construction (brick + insulation + frame)
pub fn mediumweight_construction_layers() -> Vec<ConstructionLayer> {
    vec![
        ConstructionLayer {
            name: "Brick Veneer".to_string(),
            conductivity: 0.8,
            density: 1800.0,
            specific_heat: 840.0,
            thickness: 0.1, // 10cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Insulation".to_string(),
            conductivity: 0.04,
            density: 50.0,
            specific_heat: 840.0,
            thickness: 0.08, // 8cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Wood Frame".to_string(),
            conductivity: 0.12,
            density: 600.0,
            specific_heat: 1200.0,
            thickness: 0.05, // 5cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
    ]
}

/// Creates construction layers for lightweight construction (wood frame + insulation + plasterboard)
pub fn lightweight_construction_layers() -> Vec<ConstructionLayer> {
    vec![
        ConstructionLayer {
            name: "Fiberglass Insulation".to_string(),
            conductivity: 0.04,
            density: 50.0,
            specific_heat: 840.0,
            thickness: 0.1, // 10cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Wood Frame".to_string(),
            conductivity: 0.12,
            density: 600.0,
            specific_heat: 1200.0,
            thickness: 0.05, // 5cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Plasterboard".to_string(),
            conductivity: 0.25,
            density: 800.0,
            specific_heat: 1000.0,
            thickness: 0.015, // 1.5cm
            emissivity: 0.9,
            absorptance: 0.7,
        },
    ]
}

/// Creates a thermal mass diagnostics analyzer with standard parameters
pub fn standard_thermal_mass_diagnostics() -> ThermalMassDiagnostics {
    ThermalMassDiagnostics::new(3600, 25.0) // 1-hour timestep, 25 W/m²K heat loss
}

/// Creates a thermal mass diagnostics analyzer with construction layers
pub fn thermal_mass_diagnostics_with_layers(
    layers: Vec<ConstructionLayer>,
) -> ThermalMassDiagnostics {
    ThermalMassDiagnostics::with_construction_layers(layers, 25.0)
}

/// Helper function to assert that thermal mass report values are within expected ranges
pub fn assert_thermal_mass_report_reasonable(report: &ThermalMassReport, expected_type: &str) {
    // Basic sanity checks
    assert!(
        report.effective_capacitance > 0.0,
        "Effective capacitance must be positive"
    );
    assert!(report.time_constant > 0.0, "Time constant must be positive");
    assert!(
        report.damping_factor > 0.0 && report.damping_factor < 1.0,
        "Damping factor must be between 0 and 1"
    );
    assert!(
        !report.classification.is_empty(),
        "Classification must not be empty"
    );

    // Type-specific checks
    match expected_type {
        "lightweight" => {
            assert!(
                report.effective_capacitance < 100.0,
                "Lightweight construction should have capacitance < 100 kJ/m²K"
            );
            assert!(
                report.time_constant < 4.0,
                "Lightweight construction should have time constant < 4 hours"
            );
        }
        "mediumweight" => {
            assert!(
                report.effective_capacitance >= 100.0 && report.effective_capacitance < 200.0,
                "Mediumweight construction should have capacitance between 100-200 kJ/m²K"
            );
            assert!(
                report.time_constant >= 4.0 && report.time_constant < 8.0,
                "Mediumweight construction should have time constant between 4-8 hours"
            );
        }
        "heavyweight" => {
            assert!(
                report.effective_capacitance >= 200.0,
                "Heavyweight construction should have capacitance >= 200 kJ/m²K"
            );
            assert!(
                report.time_constant >= 8.0,
                "Heavyweight construction should have time constant >= 8 hours"
            );
        }
        _ => panic!("Unknown construction type: {}", expected_type),
    }
}

/// Helper function to create a mock simulation result that matches reference data
pub fn create_matching_simulation_results(
    reference: &ReferenceResults,
) -> crate::validation::high_mass::test_cases::SimulationResults {
    crate::validation::high_mass::test_cases::SimulationResults {
        hourly_temperatures: reference.hourly_temperatures.clone(),
        hourly_heating: reference.hourly_heating.clone(),
        hourly_cooling: reference.hourly_cooling.clone(),
    }
}

/// Helper function to calculate expected time constant from capacitance and heat loss coefficient
pub fn calculate_expected_time_hours(
    capacitance_kj_m2k: f64,
    heat_loss_coefficient_w_m2k: f64,
) -> f64 {
    // Convert kJ to J, then calculate time constant in seconds, then convert to hours
    let capacitance_j = capacitance_kj_m2k * 1000.0;
    let time_constant_seconds = capacitance_j / heat_loss_coefficient_w_m2k;
    time_constant_seconds / 3600.0
}

/// Helper function to calculate expected damping factor for 1-hour timestep
pub fn calculate_expected_damping_factor(time_constant_hours: f64) -> f64 {
    let timestep_hours = 1.0; // 1 hour timestep
    (-timestep_hours / time_constant_hours).exp()
}
