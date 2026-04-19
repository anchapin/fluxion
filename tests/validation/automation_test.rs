// tests/validation/automation_test.rs
use std::fs::{self, File};
use std::io::Write;
/// Automation workflow tests
///
/// Test suite for test automation runner functionality
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_runner_configuration() {
    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        PathBuf::from("tests/fixtures"),
        PathBuf::from("target/test_output"),
        0.5,
        true,
        "markdown".to_string(),
    );

    assert_eq!(config.tolerance, 0.5);
    assert!(config.verbose);
    assert_eq!(config.format, "markdown");
}

#[test]
fn test_runner_initialization() {
    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        PathBuf::from("tests/fixtures"),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let mut runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Should initialize successfully
    assert!(runner.initialize().is_ok());
}

#[test]
fn test_test_case_discovery() {
    // Create a temporary test case directory
    let temp_dir = tempdir().unwrap();
    let test_case_dir = temp_dir.path().join("test_case_1");
    fs::create_dir_all(&test_case_dir).unwrap();

    // Create a reference file
    let reference_path = test_case_dir.join("reference.csv");
    let mut file = File::create(&reference_path).unwrap();
    writeln!(file, "temperature,heating,cooling").unwrap();
    writeln!(file, "20.5,1000.0,500.0").unwrap();

    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        temp_dir.path().to_path_buf(),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Should discover the test case
    let test_cases = runner.discover_test_cases().unwrap();
    assert_eq!(test_cases.len(), 1);
    assert!(test_cases[0].ends_with("test_case_1"));
}

#[test]
fn test_test_data_validation() {
    // Create a temporary test case directory
    let temp_dir = tempdir().unwrap();
    let test_case_dir = temp_dir.path().join("valid_test");
    fs::create_dir_all(&test_case_dir).unwrap();

    // Create a reference file with content
    let reference_path = test_case_dir.join("reference.csv");
    let mut file = File::create(&reference_path).unwrap();
    writeln!(file, "temperature,heating,cooling").unwrap();
    writeln!(file, "20.5,1000.0,500.0").unwrap();

    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        temp_dir.path().to_path_buf(),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Should validate test data successfully
    assert!(runner.validate_test_data(&test_case_dir).is_ok());
}

#[test]
fn test_invalid_test_case_detection() {
    // Create a temporary test case directory without reference file
    let temp_dir = tempdir().unwrap();
    let test_case_dir = temp_dir.path().join("invalid_test");
    fs::create_dir_all(&test_case_dir).unwrap();

    // No reference.csv file

    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        temp_dir.path().to_path_buf(),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Should fail validation
    let result = runner.validate_test_data(&test_case_dir);
    assert!(result.is_err());
}

#[test]
fn test_empty_reference_file_detection() {
    // Create a temporary test case directory with empty reference file
    let temp_dir = tempdir().unwrap();
    let test_case_dir = temp_dir.path().join("empty_test");
    fs::create_dir_all(&test_case_dir).unwrap();

    // Create empty reference file
    let reference_path = test_case_dir.join("reference.csv");
    File::create(&reference_path).unwrap(); // Empty file

    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        temp_dir.path().to_path_buf(),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Should fail validation due to empty file
    let result = runner.validate_test_data(&test_case_dir);
    assert!(result.is_err());
}

#[test]
fn test_report_generation_formats() {
    // Create a simple test runner
    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        PathBuf::from("tests/fixtures"),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Create dummy reports for testing
    let reports = vec![fluxion::validation::reports::CrossValidationReport {
        overall_pass: true,
        zone_results: vec![],
        summary_statistics: fluxion::validation::reports::SummaryStatistics {
            mean_temp_difference: 0.1,
            max_temp_difference: 0.2,
            mean_heating_difference: 5.0,
            max_heating_difference: 10.0,
            pass_rate: 1.0,
        },
    }];

    // Test markdown generation
    let markdown_report = runner.generate_combined_report(&reports).unwrap();
    assert!(markdown_report.contains("# Combined Cross-Validation Report"));
    assert!(markdown_report.contains("Summary:"));

    // Test JSON generation (would need to change config format)
    let json_config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        PathBuf::from("tests/fixtures"),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "json".to_string(),
    );

    let json_runner = fluxion::validation::automation::runner::TestRunner::new(json_config);
    let json_report = json_runner.generate_combined_report(&reports).unwrap();

    // Should be valid JSON
    assert!(json_report.contains("["));
    assert!(json_report.contains("]"));
}

#[test]
fn test_report_saving() {
    // Create a temporary output directory
    let temp_dir = tempdir().unwrap();
    let output_dir = temp_dir.path().to_path_buf();

    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        PathBuf::from("tests/fixtures"),
        output_dir.clone(),
        0.5,
        false,
        "markdown".to_string(),
    );

    let runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Create a simple report
    let report_content = "# Test Report\n\nThis is a test report.".to_string();

    // Save the report
    assert!(runner
        .save_report(&report_content, "test_report.md")
        .is_ok());

    // Verify the file was created
    let report_path = output_dir.join("test_report.md");
    assert!(report_path.exists());

    // Verify content
    let saved_content = fs::read_to_string(report_path).unwrap();
    assert_eq!(saved_content, report_content);
}

#[test]
fn test_cleanup_functionality() {
    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        PathBuf::from("tests/fixtures"),
        PathBuf::from("target/test_output"),
        0.5,
        false,
        "markdown".to_string(),
    );

    let mut runner = fluxion::validation::automation::runner::TestRunner::new(config);

    // Initialize to create temp dir
    runner.initialize().unwrap();

    // Cleanup should succeed
    assert!(runner.cleanup().is_ok());

    // Temp dir should be cleaned up
    assert!(runner.temp_dir.is_none());
}
