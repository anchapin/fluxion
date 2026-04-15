// validation/esp_r/mod.rs
use serde::{Deserialize, Serialize};
use std::error::Error;
/// ESP-r integration module for cross-validation
///
/// This module provides functionality to validate Fluxion simulation results
/// against ESP-r reference data using configurable tolerance bands.
use std::path::PathBuf;

/// ESP-r specific validation report
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EspRValidationReport {
    /// Zone validation results
    pub zone_results: Vec<comparison::ComparisonResult>,
    /// Validation statistics
    pub statistics: ValidationStatistics,
    /// Overall pass/fail status
    pub overall_pass: bool,
    /// Average temperature difference
    pub average_temperature_difference: f64,
}

/// Validation statistics
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ValidationStatistics {
    /// Mean temperature difference across all zones
    pub mean_temp_difference: f64,
    /// Maximum temperature difference across all zones
    pub max_temp_difference: f64,
    /// Mean heating load difference across all zones
    pub mean_heating_difference: f64,
    /// Maximum heating load difference across all zones
    pub max_heating_difference: f64,
}

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
    ) -> Result<EspRValidationReport, Box<dyn Error>> {
        // Parse ESP-r reference data
        let esp_r_data = parser::parse_esp_r_output(&self.reference_path)?;

        // Compare Fluxion results with ESP-r data
        let comparison_results =
            comparison::compare_results(fluxion_results, &esp_r_data, self.tolerance);

        // Calculate statistics
        let statistics = ValidationStatistics {
            mean_temp_difference: comparison_results
                .iter()
                .map(|r| r.temp_difference)
                .sum::<f64>()
                / comparison_results.len() as f64,
            max_temp_difference: comparison_results
                .iter()
                .map(|r| r.temp_difference)
                .fold(f64::MIN, f64::max),
            mean_heating_difference: comparison_results
                .iter()
                .map(|r| r.heating_difference)
                .sum::<f64>()
                / comparison_results.len() as f64,
            max_heating_difference: comparison_results
                .iter()
                .map(|r| r.heating_difference)
                .fold(f64::MIN, f64::max),
        };

        // Generate ESP-r validation report
        let average_temp_diff = comparison_results
            .iter()
            .map(|r| r.temp_difference)
            .sum::<f64>()
            / comparison_results.len() as f64;
        let overall_pass = comparison_results
            .iter()
            .all(|r| r.temp_within_tolerance && r.heating_within_tolerance);

        let report = EspRValidationReport {
            zone_results: comparison_results,
            statistics,
            average_temperature_difference: average_temp_diff,
            overall_pass,
        };

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

// Re-export comparison types
pub use comparison::ComparisonResult;
