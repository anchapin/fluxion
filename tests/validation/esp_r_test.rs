// tests/validation/esp_r_test.rs
use std::io::Write;
/// ESP-r integration tests
///
/// Comprehensive test suite for ESP-r validation functionality
use std::path::PathBuf;
use tempfile::NamedTempFile;

#[test]
fn test_esp_r_validator_creation() {
    // Create a temporary CSV file for testing
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "temperature,heating,cooling").unwrap();
    writeln!(temp_file, "20.5,1000.0,500.0").unwrap();
    writeln!(temp_file, "21.0,1100.0,450.0").unwrap();

    let path = temp_file.path().to_path_buf();

    // Test validator creation
    let validator = fluxion::validation::esp_r::EspRValidator::new(path, 0.5);

    assert_eq!(validator.tolerance, 0.5);
    assert!(validator.reference_path.exists());
}

#[test]
fn test_esp_r_parser() {
    // Create test CSV data
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "temperature,heating,cooling").unwrap();
    writeln!(temp_file, "20.5,1000.0,500.0").unwrap();
    writeln!(temp_file, "21.0,1100.0,450.0").unwrap();
    writeln!(temp_file, "19.8,950.0,550.0").unwrap();

    let path = temp_file.path();

    // Test parsing
    let esp_r_data = fluxion::validation::esp_r::parser::parse_esp_r_output(path).unwrap();

    assert_eq!(esp_r_data.hourly_temperatures.len(), 3);
    assert_eq!(esp_r_data.hourly_heating.len(), 3);
    assert_eq!(esp_r_data.hourly_cooling.len(), 3);

    assert_eq!(esp_r_data.hourly_temperatures[0], 20.5);
    assert_eq!(esp_r_data.hourly_heating[1], 1100.0);
    assert_eq!(esp_r_data.hourly_cooling[2], 550.0);
}

#[test]
fn test_comparison_logic() {
    // Create test data
    let fluxion_results = fluxion::validation::ValidationResults::default();

    let esp_r_data = fluxion::validation::esp_r::parser::EspRData {
        hourly_temperatures: vec![20.5, 21.0, 19.8],
        hourly_heating: vec![1000.0, 1100.0, 950.0],
        hourly_cooling: vec![500.0, 450.0, 550.0],
    };

    // Test comparison
    let results =
        fluxion::validation::esp_r::comparison::compare_results(&fluxion_results, &esp_r_data, 0.5);

    // Basic validation - should have comparison results
    assert!(!results.is_empty());

    // Check that results have expected structure
    for result in results {
        assert!(!result.zone_id.is_empty());
        assert!(result.temp_difference.is_finite());
        assert!(result.heating_difference.is_finite());
    }
}

#[test]
fn test_report_generation() {
    // Create comparison results
    let comparison_results = vec![
        fluxion::validation::esp_r::comparison::ComparisonResult {
            zone_id: "zone_1".to_string(),
            temp_difference: 0.2,
            heating_difference: 10.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
        fluxion::validation::esp_r::comparison::ComparisonResult {
            zone_id: "zone_2".to_string(),
            temp_difference: 0.6,
            heating_difference: 150.0,
            temp_within_tolerance: false,
            heating_within_tolerance: false,
        },
    ];

    // Generate report
    let report =
        fluxion::validation::reports::cross_validation::generate_report(comparison_results, 0.5);

    // Validate report structure
    assert_eq!(report.zone_results.len(), 2);
    assert!(!report.summary_statistics.mean_temp_difference.is_nan());
    assert!(!report.summary_statistics.pass_rate.is_nan());

    // Check pass/fail logic
    assert!(!report.overall_pass); // Should fail due to zone_2
}

#[test]
fn test_markdown_report_generation() {
    // Create a simple report
    let comparison_results = vec![fluxion::validation::esp_r::comparison::ComparisonResult {
        zone_id: "test_zone".to_string(),
        temp_difference: 0.1,
        heating_difference: 5.0,
        temp_within_tolerance: true,
        heating_within_tolerance: true,
    }];

    let report =
        fluxion::validation::reports::cross_validation::generate_report(comparison_results, 0.5);

    // Generate markdown
    let markdown =
        fluxion::validation::reports::cross_validation::generate_markdown_report(&report);

    // Validate markdown structure
    assert!(markdown.contains("# Cross-Validation Report"));
    assert!(markdown.contains("**Overall Status:**"));
    assert!(markdown.contains("**Summary Statistics:**"));
    assert!(markdown.contains("**Zone Results:**"));
    assert!(markdown.contains("test_zone"));
    assert!(markdown.contains("✅"));
}
