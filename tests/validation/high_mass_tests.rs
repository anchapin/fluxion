//! High-mass validation tests for Phase 44
//!
//! This test file covers MASS-01, MASS-02, and MASS-03 requirements
//! for high-mass physics validation completion.

use crate::physics::thermal_mass::construction::ConstructionType;
use crate::physics::thermal_mass::diagnostics::{ThermalMassDiagnostics, ThermalMassReport};
use crate::sim::construction::ConstructionLayer;
use crate::validation::ashrae140::WeatherData;
use crate::validation::high_mass::test_cases::{
    create_high_mass_validation_cases, HighMassValidationCase, ValidationStatus,
    ValidationTolerance,
};
use crate::validation::report::ValidationResult;
use crate::validation::tolerance::ValidationTolerance as Tol;
use anyhow::Result;
use std::sync::Arc;

/// Test MASS-01: High-mass validation within tolerance
#[test]
fn test_high_mass_validation_within_tolerance() -> Result<()> {
    // Create a high-mass validation case
    let case = HighMassValidationCase::new(
        "TEST-001".to_string(),
        crate::validation::high_mass::test_cases::BuildingConfig {
            construction_type: ConstructionType::HeavyWeight,
            floor_area: 232.0,
            u_value: 0.35,
            window_wall_ratio: 0.15,
            infiltration_rate: 0.3,
        },
        WeatherData::default(),
        crate::validation::high_mass::test_cases::ReferenceResults {
            hourly_temperatures: vec![20.0; 8760],
            hourly_heating: vec![0.8; 8760],
            hourly_cooling: vec![0.3; 8760],
            annual_heating: 7008.0,
            annual_cooling: 2628.0,
        },
        ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        },
        "Test high-mass validation case".to_string(),
    );

    // Execute the validation case
    let result = case.execute()?;

    // Verify that the validation passes (since we're using identical data)
    assert_eq!(result.status, ValidationStatus::Pass);

    Ok(())
}

/// Test MASS-02: Thermal mass diagnostics generation
#[test]
fn test_thermal_mass_diagnostics_generation() -> Result<()> {
    // Create construction layers for heavyweight construction
    let construction_layers = vec![
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
    ];

    // Create thermal mass diagnostics analyzer
    let diagnostics = ThermalMassDiagnostics::with_construction_layers(
        construction_layers,
        25.0, // W/m²K heat loss coefficient
    );

    // Run analysis
    let report = diagnostics.analyze();

    // Verify that all required metrics are present and reasonable
    assert!(
        report.effective_capacitance > 100.0,
        "Effective capacitance should be > 100 kJ/m²K for heavyweight"
    );
    assert!(
        report.time_constant > 5.0,
        "Time constant should be > 5 hours for heavyweight"
    );
    assert!(
        report.damping_factor > 0.0 && report.damping_factor < 1.0,
        "Damping factor should be between 0 and 1"
    );
    assert!(
        !report.classification.is_empty(),
        "Classification should not be empty"
    );

    // For heavyweight construction, we expect Heavy or VeryHeavy classification
    assert!(
        report.classification == "Heavy" || report.classification == "VeryHeavy",
        "Heavyweight construction should classify as Heavy or VeryHeavy"
    );

    Ok(())
}

/// Test MASS-03: Construction-type physics application
#[test]
fn test_construction_type_physics_application() -> Result<()> {
    // Test all construction types
    let test_cases = [
        (ConstructionType::Lightweight, 50.0, 2.0, 0.3),
        (ConstructionType::MediumWeight, 150.0, 6.0, 0.5),
        (ConstructionType::HeavyWeight, 300.0, 12.0, 0.7),
    ];

    for (construction_type, expected_capacitance, expected_time_constant, expected_damping) in
        test_cases.iter()
    {
        let props = construction_type.thermal_mass_properties();

        // Allow some tolerance for floating point comparisons
        assert!(
            (props.effective_capacitance - *expected_capacitance).abs() < 1.0,
            "Effective capacitance mismatch for {:?}",
            construction_type
        );
        assert!(
            (props.time_constant - *expected_time_constant).abs() < 0.5,
            "Time constant mismatch for {:?}",
            construction_type
        );
        assert!(
            (props.damping_factor - *expected_damping).abs() < 0.05,
            "Damping factor mismatch for {:?}",
            construction_type
        );
    }

    // Test custom construction type
    let custom_layers = vec![
        crate::physics::thermal_mass::construction::MaterialLayer::new(
            "Concrete", 1.7, 2300.0, 840.0, 0.2,
        ),
        crate::physics::thermal_mass::construction::MaterialLayer::new(
            "Insulation",
            0.04,
            50.0,
            840.0,
            0.05,
        ),
    ];

    let custom_type = ConstructionType::Custom(custom_layers);
    let props = custom_type.thermal_mass_properties();

    // Custom construction should have reasonable thermal mass properties
    assert!(
        props.effective_capacitance > 0.0,
        "Custom construction should have positive effective capacitance"
    );
    assert!(
        props.time_constant > 0.0,
        "Custom construction should have positive time constant"
    );
    assert!(
        props.damping_factor > 0.0 && props.damping_factor < 1.0,
        "Custom construction damping factor should be between 0 and 1"
    );

    Ok(())
}

/// Test that high-mass validation cases can be created and executed
#[test]
fn test_high_mass_validation_cases_creation_and_execution() -> Result<()> {
    // Create all predefined high-mass validation cases
    let cases = create_high_mass_validation_cases();

    // Verify we have the expected cases
    assert_eq!(
        cases.len(),
        3,
        "Should have 3 predefined high-mass validation cases"
    );

    let case_ids: Vec<String> = cases.iter().map(|c| c.case_id.clone()).collect();
    assert!(
        case_ids.contains(&"600".to_string()),
        "Should include Case 600"
    );
    assert!(
        case_ids.contains(&"650".to_string()),
        "Should include Case 650"
    );
    assert!(
        case_ids.contains(&"900".to_string()),
        "Should include Case 900"
    );

    // Execute each case and verify basic properties
    for case in cases {
        // Verify construction type matches case expectations
        match case.case_id.as_str() {
            "600" | "900" => {
                assert!(
                    matches!(
                        case.building_config.construction_type,
                        ConstructionType::HeavyWeight
                    ),
                    "Case {} should use HeavyWeight construction",
                    case.case_id
                );
            }
            "650" => {
                assert!(
                    matches!(
                        case.building_config.construction_type,
                        ConstructionType::MediumWeight
                    ),
                    "Case 650 should use MediumWeight construction"
                );
            }
            _ => panic!("Unexpected case ID: {}", case.case_id),
        }

        // Verify that we can create construction layers
        let layers = case.create_construction_layers();
        assert!(
            !layers.is_empty(),
            "Case {} should have construction layers",
            case.case_id
        );

        // Verify that we can run thermal mass diagnostics
        let heat_loss_coefficient =
            crate::validation::high_mass::test_cases::calculate_heat_loss_coefficient(
                case.building_config.u_value,
                case.building_config.floor_area,
                case.building_config.window_wall_ratio,
            );

        let diagnostics =
            ThermalMassDiagnostics::with_construction_layers(layers, heat_loss_coefficient);
        let report = diagnostics.analyze();

        assert!(
            report.effective_capacitance > 0.0,
            "Case {} should have positive effective capacitance",
            case.case_id
        );
        assert!(
            report.time_constant > 0.0,
            "Case {} should have positive time constant",
            case.case_id
        );
        assert!(
            report.damping_factor > 0.0 && report.damping_factor < 1.0,
            "Case {} damping factor should be between 0 and 1",
            case.case_id
        );
    }

    Ok(())
}
