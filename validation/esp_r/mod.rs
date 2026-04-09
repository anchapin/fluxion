// validation/esp_r/mod.rs
use std::error::Error;
/// ESP-r integration module for cross-validation
///
/// This module provides functionality to validate Fluxion simulation results
/// against ESP-r reference data using configurable tolerance bands.
use std::error::Error;
use std::path::PathBuf;

/// ESP-r output parser module
pub mod parser;

/// Comparison logic module
pub mod comparison;

/// Test automation module
pub mod test_automation;

/// CLI integration module
pub mod cli_integration;

/// Framework integration module
pub mod integration;

/// Examples module
pub mod examples;

/// Main ESP-r validator struct
#[derive(Debug)]
pub struct EspRValidator {
    /// Path to ESP-r reference output file
    pub reference_path: PathBuf,
    /// Temperature tolerance for comparison (in °C)
    pub tolerance: f64,
}

impl EspRValidator {
    /// Create a new ESP-r validator
    ///
    /// # Arguments
    /// * `reference_path` - Path to ESP-r CSV output file
    /// * `tolerance` - Temperature tolerance for comparison
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// let validator = EspRValidator::new(PathBuf::from("reference.csv"), 0.5);
    /// ```
    pub fn new(reference_path: PathBuf, tolerance: f64) -> Self {
        Self {
            reference_path,
            tolerance,
        }
    }

    /// Validate Fluxion results against ESP-r reference data
    ///
    /// # Arguments
    /// * `fluxion_results` - Fluxion multi-zone validation results to compare
    ///
    /// # Returns
    /// Cross-validation report with comparison results
    ///
    /// # Example
    /// ```
    /// use fluxion::validation::MultiZoneValidationResults;
    /// let fluxion_results = MultiZoneValidationResults::default();
    /// let report = validator.validate(&fluxion_results)?;
    /// ```
    pub fn validate(
        &self,
        fluxion_results: &crate::validation::MultiZoneValidationResults,
    ) -> Result<crate::validation::reports::CrossValidationReport, Box<dyn Error>> {
        // Parse ESP-r reference data
        let esp_r_data = parser::parse_esp_r_output(&self.reference_path)?;

        // Compare Fluxion results with ESP-r data
        let comparison_results =
            comparison::compare_results(fluxion_results, &esp_r_data, self.tolerance);

        // Generate cross-validation report
        let report = crate::validation::reports::cross_validation::generate_report(
            comparison_results,
            self.tolerance,
        );

        Ok(report)
    }
}

/// Run automated ESP-r cross-validation test
///
/// # Arguments
/// * `esp_r_output_path` - Path to ESP-r CSV output file
/// * `fluxion_results_path` - Path to Fluxion validation results JSON file
/// * `tolerance` - Temperature tolerance for comparison
/// * `report_format` - Output format (JSON or Markdown)
///
/// # Returns
/// Test result with pass/fail status and report
pub fn run_automated_test(
    esp_r_output_path: PathBuf,
    fluxion_results_path: PathBuf,
    tolerance: f64,
    report_format: test_automation::ReportFormat,
) -> Result<test_automation::EspRTestResult, Box<dyn Error>> {
    let config = test_automation::EspRTestConfig {
        esp_r_output_path,
        fluxion_results_path,
        tolerance,
        report_format,
    };

    test_automation::run_cross_validation_test(&config)
}

// Re-export key types for easy access
pub use test_automation::EspRTestConfig;
pub use test_automation::EspRTestResult;
pub use test_automation::ReportFormat;

// Re-export CLI integration types
pub use cli_integration::run_cli_validation;
pub use cli_integration::EspRCliConfig;
pub use cli_integration::EspRCliResult;

// Re-export integration types
pub use integration::create_integration_adapter;
pub use integration::run_as_integration_tool;
pub use integration::EspRValidationAdapter;

// Re-export examples
pub use examples::advanced_cross_validation_example;
pub use examples::basic_cross_validation_example;
pub use examples::error_handling_example;
pub use examples::report_generation_example;
pub use examples::run_all_examples;
pub use integration::EspRValidationAdapter;
