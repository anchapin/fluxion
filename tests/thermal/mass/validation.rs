//! Comprehensive validation test suite for thermal mass validation
//!
//! This module provides comprehensive tests for thermal mass validation,
//! addressing MASS-03 (construction-type physics) requirements from ASHRAE 140.

use crate::thermal::mass::types::{ConstructionType, HighMassCase, ValidationResult};
use crate::thermal::mass::validator::ThermalMassValidator;

/// Test basic validation functionality
#[test]
fn test_basic_validation() {
    // Create a test case with known reference loads
    let reference = vec![100.0, 200.0, 150.0, 180.0, 160.0];
    // Simulation results slightly lower (within tolerance)
    let simulated = vec![95.0, 190.0, 145.0, 175.0, 155.0];
    let tolerance = 15.0;

    let validator = ThermalMassValidator::new(reference, simulated, tolerance)
        .expect("Failed to create validator");

    let result = validator.validate();
    assert!(
        result.passes,
        "Validation should pass for results within tolerance"
    );
    assert!(
        result.nmbe.abs() <= tolerance,
        "NMBE should be within tolerance"
    );
    assert!(
        result.cv_rmse <= tolerance,
        "CV(RMSE) should be within tolerance"
    );
}

/// Test validation failure detection
#[test]
fn test_validation_failure() {
    let reference = vec![100.0, 200.0, 150.0, 180.0];
    // Simulation results with 50% error (outside 15% tolerance)
    let simulated = vec![50.0, 250.0, 75.0, 90.0];
    let tolerance = 15.0;

    let validator = ThermalMassValidator::new(reference, simulated, tolerance)
        .expect("Failed to create validator");

    let result = validator.validate();
    assert!(
        !result.passes,
        "Validation should fail for results outside tolerance"
    );
}

/// Test all construction types for MASS-03 requirement
#[test]
fn test_construction_types() {
    // Test Light construction type
    let light_case = HighMassCase::new(
        ConstructionType::Light,
        50.0,
        840.0,
        vec![100.0, 150.0, 200.0, 180.0, 160.0],
        10.0,
    );
    assert_eq!(light_case.construction, ConstructionType::Light);
    assert!((light_case.mass_per_area - 50.0).abs() < 0.1);

    // Test Medium construction type
    let medium_case = HighMassCase::new(
        ConstructionType::Medium,
        150.0,
        840.0,
        vec![100.0, 150.0, 200.0, 180.0, 160.0],
        10.0,
    );
    assert_eq!(medium_case.construction, ConstructionType::Medium);
    assert!((medium_case.mass_per_area - 150.0).abs() < 0.1);

    // Test Heavy construction type
    let heavy_case = HighMassCase::new(
        ConstructionType::Heavy,
        300.0,
        840.0,
        vec![100.0, 150.0, 200.0, 180.0, 160.0],
        15.0,
    );
    assert_eq!(heavy_case.construction, ConstructionType::Heavy);
    assert!((heavy_case.mass_per_area - 300.0).abs() < 0.1);

    // Test VeryHeavy construction type
    let very_heavy_case = HighMassCase::new(
        ConstructionType::VeryHeavy,
        600.0,
        840.0,
        vec![100.0, 150.0, 200.0, 180.0, 160.0],
        15.0,
    );
    assert_eq!(very_heavy_case.construction, ConstructionType::VeryHeavy);
    assert!((very_heavy_case.mass_per_area - 600.0).abs() < 0.1);
}

/// Test tolerance bands for different construction types
#[test]
fn test_tolerance_bands() {
    // Light mass - stricter tolerance
    let light_ref = vec![100.0, 150.0, 200.0];
    let light_sim = vec![98.0, 148.0, 198.0]; // 2% error
    let validator = ThermalMassValidator::new(light_ref, light_sim, 10.0)
        .expect("Failed to create light validator");
    let result = validator.validate();
    assert!(
        result.passes,
        "Light mass with 2% error should pass 10% tolerance"
    );

    // Heavy mass - more lenient tolerance
    let heavy_ref = vec![100.0, 150.0, 200.0];
    let heavy_sim = vec![90.0, 135.0, 180.0]; // 10% error
    let heavy_validator = ThermalMassValidator::new(heavy_ref, heavy_sim, 15.0)
        .expect("Failed to create heavy validator");
    let heavy_result = heavy_validator.validate();
    assert!(
        heavy_result.passes,
        "Heavy mass with 10% error should pass 15% tolerance"
    );
}

/// Test NMBE calculation
#[test]
fn test_nmbe_calculation() {
    // Perfect match - NMBE should be 0
    let reference = vec![100.0, 200.0, 300.0];
    let simulated = vec![100.0, 200.0, 300.0];
    let validator =
        ThermalMassValidator::new(reference, simulated, 10.0).expect("Failed to create validator");
    let result = validator.validate();
    assert!(
        (result.nmbe).abs() < 0.01,
        "NMBE should be ~0 for perfect match"
    );

    // Positive bias (simulation higher than reference)
    let ref2 = vec![100.0, 200.0, 300.0];
    let sim2 = vec![110.0, 220.0, 330.0]; // 10% positive bias
    let validator2 =
        ThermalMassValidator::new(ref2, sim2, 10.0).expect("Failed to create validator");
    let result2 = validator2.validate();
    assert!(result2.nmbe > 0.0, "NMBE should be positive for high bias");

    // Negative bias (simulation lower than reference)
    let ref3 = vec![100.0, 200.0, 300.0];
    let sim3 = vec![90.0, 180.0, 270.0]; // 10% negative bias
    let validator3 =
        ThermalMassValidator::new(ref3, sim3, 10.0).expect("Failed to create validator");
    let result3 = validator3.validate();
    assert!(result3.nmbe < 0.0, "NMBE should be negative for low bias");
}

/// Test CV(RMSE) calculation
#[test]
fn test_cv_rmse_calculation() {
    // Perfect match - CV(RMSE) should be 0
    let reference = vec![100.0, 200.0, 300.0];
    let simulated = vec![100.0, 200.0, 300.0];
    let validator =
        ThermalMassValidator::new(reference, simulated, 10.0).expect("Failed to create validator");
    let result = validator.validate();
    assert!(
        (result.cv_rmse).abs() < 0.01,
        "CV(RMSE) should be ~0 for perfect match"
    );

    // Small variance in results
    let ref2 = vec![100.0, 200.0, 300.0];
    let sim2 = vec![105.0, 195.0, 305.0]; // 5% variance
    let validator2 =
        ThermalMassValidator::new(ref2, sim2, 10.0).expect("Failed to create validator");
    let result2 = validator2.validate();
    assert!(
        result2.cv_rmse > 0.0,
        "CV(RMSE) should be positive for variance"
    );
}

/// Test empty data error handling
#[test]
fn test_empty_data_error() {
    let result = ThermalMassValidator::new(vec![], vec![], 10.0);
    assert!(result.is_err(), "Should error on empty data");
}

/// Test mismatched length error handling
#[test]
fn test_mismatched_length_error() {
    let reference = vec![100.0, 200.0, 300.0];
    let simulated = vec![100.0, 200.0]; // Different length
    let result = ThermalMassValidator::new(reference, simulated, 10.0);
    assert!(result.is_err(), "Should error on mismatched lengths");
}

/// Test validation result structure
#[test]
fn test_validation_result_structure() {
    let reference = vec![100.0, 200.0, 300.0, 250.0, 150.0];
    let simulated = vec![105.0, 210.0, 315.0, 240.0, 145.0]; // ~5% error
    let validator =
        ThermalMassValidator::new(reference, simulated, 10.0).expect("Failed to create validator");
    let result = validator.validate();

    // Check result fields exist and are valid
    assert!(result.nmbe.is_finite(), "NMBE should be finite");
    assert!(result.cv_rmse.is_finite(), "CV(RMSE) should be finite");
    assert!(
        result.max_deviation.is_finite(),
        "Max deviation should be finite"
    );
}

/// Test high-mass case with ASHRAE 140 reference values
#[test]
fn test_ashrae_140_high_mass_case() {
    // Case 305: Heavy mass - January reference data (from high_mass.rs)
    let reference_loads = vec![
        72.4, 71.2, 70.1, 69.2, 70.2, 73.8, 82.5, 94.2, 105.4, 112.8, 118.2, 120.1, 118.5, 115.2,
        112.4, 108.5, 104.2, 101.5, 98.8, 96.2, 93.5, 91.2, 88.4, 85.8,
    ];

    // Simulated results within 10% tolerance
    let simulated_loads = vec![
        70.1, 69.0, 68.0, 67.2, 68.2, 71.6, 80.1, 91.4, 102.3, 109.5, 114.7, 116.5, 115.0, 111.8,
        109.1, 105.3, 101.1, 98.5, 95.9, 93.3, 90.7, 88.5, 85.7, 83.2,
    ];

    let validator = ThermalMassValidator::new(reference_loads, simulated_loads, 10.0)
        .expect("Failed to create validator");

    let result = validator.validate();
    // Should pass since simulated is within 10% of reference
    assert!(
        result.passes,
        "ASHRAE 140 high-mass case should pass within 10% tolerance"
    );
}

/// Test validation with custom tolerance bands
#[test]
fn test_custom_tolerance_bands() {
    let reference = vec![100.0, 200.0, 300.0];
    let simulated = vec![95.0, 195.0, 295.0]; // 5% error

    // Use stricter NMBE tolerance but lenient CV(RMSE) tolerance
    let validator = ThermalMassValidator::with_tolerances(reference, simulated, 5.0, 30.0)
        .expect("Failed to create validator with custom tolerances");

    let result = validator.validate();
    // Should fail since 5% error exceeds 5% NMBE tolerance
    assert!(!result.passes, "Should fail with strict 5% NMBE tolerance");
}

/// Test default tolerance values
#[test]
fn test_default_tolerances() {
    let validator = ThermalMassValidator::new(vec![100.0, 200.0], vec![100.0, 200.0], 10.0)
        .expect("Failed to create validator");

    let tolerance_bands = validator.tolerance_bands();
    assert_eq!(tolerance_bands.nmbe, 10.0);
    assert_eq!(tolerance_bands.cv_rmse, 10.0);
}

/// Test with realistic hourly data (24 hours)
#[test]
fn test_hourly_data_validation() {
    // 24-hour profile with daily temperature swing
    let reference: Vec<f64> = (0..24)
        .map(|i| {
            let base = 100.0;
            let amplitude = 50.0;
            base + amplitude * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin()
        })
        .collect();

    // Simulated with small random noise (within tolerance)
    let simulated: Vec<f64> = reference
        .iter()
        .map(|v| v * 1.03) // 3% higher
        .collect();

    let validator =
        ThermalMassValidator::new(reference, simulated, 10.0).expect("Failed to create validator");

    let result = validator.validate();
    assert!(
        result.passes,
        "Hourly data with 3% error should pass 10% tolerance"
    );
}

/// Test maximum deviation calculation
#[test]
fn test_max_deviation() {
    let reference = vec![100.0, 150.0, 200.0, 250.0, 300.0];
    let simulated = vec![90.0, 155.0, 210.0, 240.0, 310.0];

    let validator =
        ThermalMassValidator::new(reference, simulated, 20.0).expect("Failed to create validator");

    let result = validator.validate();
    // Maximum deviation should be 10% (300 -> 310)
    assert!(
        (result.max_deviation - 10.0).abs() < 0.1,
        "Max deviation should be 10%"
    );
}
