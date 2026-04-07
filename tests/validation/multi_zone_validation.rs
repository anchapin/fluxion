//! Multi-zone validation integration tests
//!
//! This module provides comprehensive integration tests for multi-zone validation
//! functionality, particularly focusing on ASHRAE 140 Case 960 and Case 970.

use crate::validation::ashrae_140_multi_zone::{Case960Reference, Case970Reference};
use crate::validation::reference_data::{
    calculate_mbe, calculate_percentage_difference, calculate_rmse, load_case_960_reference,
    load_case_970_reference, load_csv_reference, load_multi_zone_reference, parse_hourly_data,
    within_tolerance, ReferenceData, ReferenceDataError,
};
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

/// Test reference data loading for Case 960
#[test]
fn test_load_case_960_reference() {
    let reference = load_case_960_reference().unwrap();

    // Verify reference values match ASHRAE 140-2017
    assert_eq!(reference.annual_heating, 2.05);
    assert_eq!(reference.annual_cooling, 2.165);
    assert_eq!(reference.peak_heating, 5.0);
    assert_eq!(reference.peak_cooling, 2.0);

    // Verify tolerances
    assert_eq!(reference.energy_tolerance, 0.15);
    assert_eq!(reference.load_tolerance, 0.10);
    assert_eq!(reference.temperature_tolerance, 1.0);

    // Verify zone temperatures are populated
    assert!(!reference.zone_temperatures.is_empty());
    assert!(reference.zone_temperatures.contains_key(&4380)); // Winter design day
    assert!(reference.zone_temperatures.contains_key(&5000)); // Summer design day
    assert!(reference.zone_temperatures.contains_key(&8760)); // Annual average
}

/// Test reference data loading for Case 970
#[test]
fn test_load_case_970_reference() {
    let reference = load_case_970_reference().unwrap();

    // Verify placeholder reference values
    assert_eq!(reference.annual_heating, 15.0);
    assert_eq!(reference.annual_cooling, 12.0);
    assert_eq!(reference.peak_heating, 7.5);
    assert_eq!(reference.peak_cooling, 6.8);

    // Verify tolerances
    assert_eq!(reference.energy_tolerance, 0.15);
    assert_eq!(reference.load_tolerance, 0.10);
    assert_eq!(reference.temperature_tolerance, 1.5);

    // Verify zone temperatures are populated
    assert!(!reference.zone_temperatures.is_empty());
}

/// Test multi-zone reference loading by case ID
#[test]
fn test_load_multi_zone_reference() {
    // Test Case 960
    let reference_960 = load_multi_zone_reference("960").unwrap();
    assert_eq!(reference_960.case_id, "960");
    assert_eq!(reference_960.annual_heating, 2.05);
    assert_eq!(reference_960.annual_cooling, 2.165);

    // Test Case 970
    let reference_970 = load_multi_zone_reference("970").unwrap();
    assert_eq!(reference_970.case_id, "970");
    assert_eq!(reference_970.annual_heating, 15.0);
    assert_eq!(reference_970.annual_cooling, 12.0);

    // Test unsupported case
    let unsupported = load_multi_zone_reference("999");
    assert!(unsupported.is_err());
    if let Err(ReferenceDataError::UnsupportedCase(case_id)) = unsupported {
        assert_eq!(case_id, "999");
    } else {
        panic!("Expected UnsupportedCase error");
    }
}

/// Test CSV reference data loading
#[test]
fn test_csv_reference_loading() {
    // Create a temporary CSV file with complete data
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(
        temp_file,
        "case_id,annual_heating,annual_cooling,peak_heating,peak_cooling,energy_tolerance,load_tolerance,temperature_tolerance"
    )
    .unwrap();
    writeln!(temp_file, "960,12.4,8.7,5.2,4.8,0.15,0.10,1.0").unwrap();
    writeln!(temp_file, "970,15.0,12.0,7.5,6.8,0.15,0.10,1.5").unwrap();

    let path = temp_file.path().to_str().unwrap().to_string();
    let references = load_csv_reference(&path).unwrap();

    assert_eq!(references.len(), 2);

    // Verify Case 960 data
    let case_960 = &references[0];
    assert_eq!(case_960.case_id, "960");
    assert_eq!(case_960.annual_heating, 12.4);
    assert_eq!(case_960.annual_cooling, 8.7);
    assert_eq!(case_960.peak_heating, 5.2);
    assert_eq!(case_960.peak_cooling, 4.8);
    assert_eq!(case_960.energy_tolerance, 0.15);
    assert_eq!(case_960.load_tolerance, 0.10);
    assert_eq!(case_960.temperature_tolerance, 1.0);

    // Verify Case 970 data
    let case_970 = &references[1];
    assert_eq!(case_970.case_id, "970");
    assert_eq!(case_970.annual_heating, 15.0);
    assert_eq!(case_970.annual_cooling, 12.0);
    assert_eq!(case_970.peak_heating, 7.5);
    assert_eq!(case_970.peak_cooling, 6.8);
}

/// Test CSV loading with nonexistent file
#[test]
fn test_csv_reference_loading_nonexistent() {
    let result = load_csv_reference("/nonexistent/path.csv");
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ReferenceDataError::FileNotFound(_)
    ));
}

/// Test hourly data parsing from CSV
#[test]
fn test_parse_hourly_data() {
    let csv_content = "temperature\n20.5\n21.3\n19.8\n";
    let values = parse_hourly_data(csv_content).unwrap();

    assert_eq!(values.len(), 3);
    assert_eq!(values[0], 20.5);
    assert_eq!(values[1], 21.3);
    assert_eq!(values[2], 19.8);
}

/// Test hourly data parsing with invalid values
#[test]
fn test_parse_hourly_data_invalid() {
    let csv_content = "temperature\ninvalid\n20.5\nnot_a_number\n19.8\n";
    let values = parse_hourly_data(csv_content).unwrap();

    // Should only parse valid numeric values
    assert_eq!(values.len(), 2);
    assert_eq!(values[0], 20.5);
    assert_eq!(values[1], 19.8);
}

/// Test percentage difference calculation
#[test]
fn test_percentage_difference_calculation() {
    // Test exact 10% difference
    let pct = calculate_percentage_difference(2.2, 2.0);
    assert!(pct >= 9.99 && pct <= 10.01);

    // Test exact 10% difference (below)
    let pct = calculate_percentage_difference(1.8, 2.0);
    assert!(pct >= 9.99 && pct <= 10.01);

    // Test no difference
    let pct = calculate_percentage_difference(2.0, 2.0);
    assert_eq!(pct, 0.0);

    // Test zero reference handling
    let pct = calculate_percentage_difference(0.0, 0.0);
    assert_eq!(pct, 0.0);

    // Test infinite result for non-zero actual with zero reference
    let pct = calculate_percentage_difference(5.0, 0.0);
    assert!(pct.is_infinite());
}

/// Test RMSE calculation
#[test]
fn test_rmse_calculation() {
    let actual = vec![1.0, 2.0, 3.0, 4.0];
    let reference = vec![1.1, 1.9, 3.1, 3.9];
    let rmse = calculate_rmse(&actual, &reference).unwrap();

    // Manual calculation: sqrt(((0.1)^2 + (-0.1)^2 + (-0.1)^2 + (0.1)^2) / 4)
    // = sqrt((0.01 + 0.01 + 0.01 + 0.01) / 4) = sqrt(0.04 / 4) = sqrt(0.01) = 0.1
    assert!(rmse >= 0.099 && rmse <= 0.101);
}

/// Test RMSE calculation with length mismatch
#[test]
fn test_rmse_calculation_length_mismatch() {
    let actual = vec![1.0, 2.0, 3.0];
    let reference = vec![1.0, 2.0];
    let result = calculate_rmse(&actual, &reference);

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ReferenceDataError::InvalidValue(_)
    ));
}

/// Test RMSE calculation with empty arrays
#[test]
fn test_rmse_calculation_empty_arrays() {
    let actual = vec![];
    let reference = vec![];
    let result = calculate_rmse(&actual, &reference);

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ReferenceDataError::InvalidValue(_)
    ));
}

/// Test MBE calculation
#[test]
fn test_mbe_calculation() {
    let actual = vec![1.0, 2.0, 3.0, 4.0];
    let reference = vec![1.1, 1.9, 3.1, 3.9];
    let mbe = calculate_mbe(&actual, &reference).unwrap();

    // Manual calculation: (1.0-1.1 + 2.0-1.9 + 3.0-3.1 + 4.0-3.9) / 4
    // = (-0.1 + 0.1 + -0.1 + 0.1) / 4 = 0.0 / 4 = 0.0
    assert!(mbe >= -0.001 && mbe <= 0.001);
}

/// Test MBE calculation with biased data
#[test]
fn test_mbe_calculation_biased() {
    let actual = vec![1.0, 2.0, 3.0, 4.0];
    let reference = vec![0.8, 1.8, 2.8, 3.8]; // All 0.2 below actual
    let mbe = calculate_mbe(&actual, &reference).unwrap();

    // Manual calculation: (0.2 + 0.2 + 0.2 + 0.2) / 4 = 0.2
    assert!(mbe >= 0.199 && mbe <= 0.201);
}

/// Test tolerance checking
#[test]
fn test_tolerance_checking() {
    // Test exact match
    assert!(within_tolerance(2.05, 2.05, 0.15));

    // Test within 15% tolerance (upper bound)
    assert!(within_tolerance(2.3575, 2.05, 0.15));

    // Test within 15% tolerance (lower bound)
    assert!(within_tolerance(1.7425, 2.05, 0.15));

    // Test outside tolerance (upper bound)
    assert!(!within_tolerance(2.4, 2.05, 0.15));

    // Test outside tolerance (lower bound)
    assert!(!within_tolerance(1.7, 2.05, 0.15));

    // Test zero values
    assert!(within_tolerance(0.0, 0.0, 0.15));
    assert!(!within_tolerance(1.0, 0.0, 0.15));
}

/// Test tolerance checking with different tolerances
#[test]
fn test_tolerance_checking_different_tolerances() {
    // Test with 10% tolerance
    assert!(within_tolerance(5.5, 5.0, 0.10));
    assert!(within_tolerance(4.5, 5.0, 0.10));
    assert!(!within_tolerance(5.6, 5.0, 0.10));
    assert!(!within_tolerance(4.4, 5.0, 0.10));

    // Test with 5% tolerance
    assert!(within_tolerance(5.25, 5.0, 0.05));
    assert!(within_tolerance(4.75, 5.0, 0.05));
    assert!(!within_tolerance(5.3, 5.0, 0.05));
    assert!(!within_tolerance(4.7, 5.0, 0.05));
}

/// Test validation workflow integration
#[test]
fn test_case_960_validation_workflow() {
    // Load reference data
    let reference = load_case_960_reference().unwrap();

    // Simulate actual results that should pass validation
    let actual_heating = reference.annual_heating * 1.10; // 10% above reference
    let actual_cooling = reference.annual_cooling * 0.95; // 5% below reference

    // Test tolerance checking
    assert!(within_tolerance(
        actual_heating,
        reference.annual_heating,
        reference.energy_tolerance
    ));
    assert!(within_tolerance(
        actual_cooling,
        reference.annual_cooling,
        reference.energy_tolerance
    ));

    // Test percentage difference calculation
    let heating_pct = calculate_percentage_difference(actual_heating, reference.annual_heating);
    let cooling_pct = calculate_percentage_difference(actual_cooling, reference.annual_cooling);

    assert!(heating_pct >= 9.99 && heating_pct <= 10.01);
    assert!(cooling_pct >= 4.99 && cooling_pct <= 5.01);
}

/// Test multi-zone result aggregation
#[test]
fn test_multi_zone_result_aggregation() {
    // Load both case references
    let case_960 = load_case_960_reference().unwrap();
    let case_970 = load_case_970_reference().unwrap();

    // Simulate actual results
    let actual_960_heating = case_960.annual_heating * 1.05; // 5% above
    let actual_970_heating = case_970.annual_heating * 0.98; // 2% below

    // Calculate percentage differences
    let pct_960 = calculate_percentage_difference(actual_960_heating, case_960.annual_heating);
    let pct_970 = calculate_percentage_difference(actual_970_heating, case_970.annual_heating);

    // Verify both are within tolerance
    assert!(within_tolerance(
        actual_960_heating,
        case_960.annual_heating,
        case_960.energy_tolerance
    ));
    assert!(within_tolerance(
        actual_970_heating,
        case_970.annual_heating,
        case_970.energy_tolerance
    ));

    // Verify percentage differences
    assert!(pct_960 >= 4.99 && pct_960 <= 5.01);
    assert!(pct_970 >= 1.99 && pct_970 <= 2.01);
}

/// Test validation report generation
#[test]
fn test_validation_report_generation() {
    // This would be integrated with the actual validation framework
    // For now, test that we can create a comprehensive validation summary

    let reference_960 = load_case_960_reference().unwrap();
    let reference_970 = load_case_970_reference().unwrap();

    // Simulate validation results
    let heating_960_actual = reference_960.annual_heating * 1.08;
    let cooling_960_actual = reference_960.annual_cooling * 0.97;

    let heating_970_actual = reference_970.annual_heating * 1.05;
    let cooling_970_actual = reference_970.annual_cooling * 0.99;

    // Calculate metrics
    let heating_960_pct =
        calculate_percentage_difference(heating_960_actual, reference_960.annual_heating);
    let cooling_960_pct =
        calculate_percentage_difference(cooling_960_actual, reference_960.annual_cooling);
    let heating_970_pct =
        calculate_percentage_difference(heating_970_actual, reference_970.annual_heating);
    let cooling_970_pct =
        calculate_percentage_difference(cooling_970_actual, reference_970.annual_cooling);

    // Verify all results are within tolerance
    assert!(within_tolerance(
        heating_960_actual,
        reference_960.annual_heating,
        reference_960.energy_tolerance
    ));
    assert!(within_tolerance(
        cooling_960_actual,
        reference_960.annual_cooling,
        reference_960.energy_tolerance
    ));
    assert!(within_tolerance(
        heating_970_actual,
        reference_970.annual_heating,
        reference_970.energy_tolerance
    ));
    assert!(within_tolerance(
        cooling_970_actual,
        reference_970.annual_cooling,
        reference_970.energy_tolerance
    ));

    // All percentage differences should be reasonable
    assert!(heating_960_pct < 10.0);
    assert!(cooling_960_pct < 10.0);
    assert!(heating_970_pct < 10.0);
    assert!(cooling_970_pct < 10.0);
}

/// Test error handling for invalid reference data
#[test]
fn test_invalid_reference_data() {
    let mut invalid_data = ReferenceData::default();

    // Test negative annual heating
    invalid_data.annual_heating = -1.0;
    let result = invalid_data.validate_reference_format();
    assert!(result.is_err());

    // Test invalid energy tolerance
    let mut invalid_data = ReferenceData::default();
    invalid_data.energy_tolerance = 1.5; // > 1.0
    let result = invalid_data.validate_reference_format();
    assert!(result.is_err());

    // Test zero load tolerance
    let mut invalid_data = ReferenceData::default();
    invalid_data.load_tolerance = 0.0;
    let result = invalid_data.validate_reference_format();
    assert!(result.is_err());

    // Test negative temperature tolerance
    let mut invalid_data = ReferenceData::default();
    invalid_data.temperature_tolerance = -1.0;
    let result = invalid_data.validate_reference_format();
    assert!(result.is_err());
}

/// Test tolerance boundary cases
#[test]
fn test_tolerance_boundary_cases() {
    // Test exactly at tolerance boundary (should pass with epsilon)
    let reference = 100.0;
    let tolerance = 0.15; // 15%

    // Exact upper boundary: 100 * 1.15 = 115
    assert!(within_tolerance(115.0, reference, tolerance));

    // Exact lower boundary: 100 * 0.85 = 85
    assert!(within_tolerance(85.0, reference, tolerance));

    // Just outside upper boundary
    assert!(!within_tolerance(115.01, reference, tolerance));

    // Just outside lower boundary
    assert!(!within_tolerance(84.99, reference, tolerance));

    // Test with very small reference value
    let small_reference = 0.001;
    let small_tolerance = 0.15;

    assert!(within_tolerance(0.00115, small_reference, small_tolerance));
    assert!(within_tolerance(0.00085, small_reference, small_tolerance));
}
