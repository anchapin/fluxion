// validation/esp_r/mod.rs
use std::error::Error;
/// ESP-r integration module for cross-validation
///
/// This module provides functionality to validate Fluxion simulation results
/// against ESP-r reference data using configurable tolerance bands.
use std::path::PathBuf;

/// ESP-r output parser module
pub mod parser;

/// Comparison logic module
pub mod comparison;

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
    /// * `fluxion_results` - Fluxion validation results to compare
    ///
    /// # Returns
    /// Cross-validation report with comparison results
    ///
    /// # Example
    /// ```
    /// use fluxion::validation::ValidationResults;
    /// let fluxion_results = ValidationResults::default();
    /// let report = validator.validate(&fluxion_results)?;
    /// ```
    pub fn validate(
        &self,
        fluxion_results: &crate::validation::ValidationResults,
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
