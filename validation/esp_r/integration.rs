// validation/esp_r/integration.rs
/// ESP-r framework integration
///
/// This module provides adapter patterns and integration support for ESP-r
/// validation within the broader validation framework.
use std::error::Error;
use std::path::PathBuf;

/// Adapter for integrating ESP-r validation with the main framework
#[derive(Debug)]
pub struct EspRValidationAdapter {
    /// Validation configuration
    pub config: EspRIntegrationConfig,
    /// Temperature tolerance for comparison
    pub tolerance: f64,
    /// Output report format
    pub report_format: crate::validation::esp_r::test_automation::ReportFormat,
}

/// Configuration for ESP-r integration
#[derive(Debug, Clone)]
pub struct EspRIntegrationConfig {
    /// Path to ESP-r reference output file
    pub reference_path: PathBuf,
    /// Additional configuration parameters
    pub additional_params: Option<EspRAdditionalParams>,
}

/// Additional parameters for ESP-r validation
#[derive(Debug, Clone)]
pub struct EspRAdditionalParams {
    /// Whether to include detailed zone-by-zone reporting
    pub detailed_reporting: bool,
    /// Custom tolerance bands for specific zones
    pub zone_specific_tolerances: std::collections::HashMap<String, f64>,
}

impl EspRValidationAdapter {
    /// Create a new ESP-r validation adapter
    ///
    /// # Arguments
    /// * `config` - Integration configuration
    /// * `tolerance` - Temperature tolerance for comparison
    /// * `report_format` - Output format for reports
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use fluxion::validation::esp_r::integration::{EspRValidationAdapter, EspRIntegrationConfig};
    ///
    /// let config = EspRIntegrationConfig {
    ///     reference_path: PathBuf::from("reference.csv"),
    ///     additional_params: None,
    /// };
    ///
    /// let adapter = EspRValidationAdapter::new(config, 0.5,
    ///     fluxion::validation::esp_r::test_automation::ReportFormat::Markdown);
    /// ```
    pub fn new(
        config: EspRIntegrationConfig,
        tolerance: f64,
        report_format: crate::validation::esp_r::test_automation::ReportFormat,
    ) -> Self {
        Self {
            config,
            tolerance,
            report_format,
        }
    }

    /// Run ESP-r validation as part of the integrated framework
    ///
    /// # Arguments
    /// * `fluxion_results` - Fluxion multi-zone validation results
    ///
    /// # Returns
    /// Cross-validation report
    ///
    /// # Example
    /// ```
    /// use fluxion::validation::MultiZoneValidationResults;
    /// let fluxion_results = MultiZoneValidationResults::default();
    /// let report = adapter.run_validation(&fluxion_results)?;
    /// ```
    pub fn run_validation(
        &self,
        fluxion_results: &crate::validation::MultiZoneValidationResults,
    ) -> Result<crate::validation::reports::CrossValidationReport, Box<dyn Error>> {
        // Create ESP-r validator
        let validator = crate::validation::esp_r::EspRValidator::new(
            self.config.reference_path.clone(),
            self.tolerance,
        );

        // Run validation
        let report = validator.validate(fluxion_results)?;

        Ok(report)
    }

    /// Run validation with custom configuration
    ///
    /// # Arguments
    /// * `fluxion_results` - Fluxion multi-zone validation results
    /// * `custom_tolerance` - Optional custom tolerance override
    ///
    /// # Returns
    /// Cross-validation report
    pub fn run_validation_with_config(
        &self,
        fluxion_results: &crate::validation::MultiZoneValidationResults,
        custom_tolerance: Option<f64>,
    ) -> Result<crate::validation::reports::CrossValidationReport, Box<dyn Error>> {
        let tolerance = custom_tolerance.unwrap_or(self.tolerance);

        let validator = crate::validation::esp_r::EspRValidator::new(
            self.config.reference_path.clone(),
            tolerance,
        );

        let report = validator.validate(fluxion_results)?;

        Ok(report)
    }
}

/// Create an integration adapter with default configuration
///
/// # Arguments
/// * `reference_path` - Path to ESP-r reference output file
/// * `tolerance` - Temperature tolerance for comparison
///
/// # Returns
/// Configured ESP-r validation adapter
pub fn create_integration_adapter(
    reference_path: PathBuf,
    tolerance: f64,
) -> EspRValidationAdapter {
    let config = EspRIntegrationConfig {
        reference_path,
        additional_params: None,
    };

    EspRValidationAdapter::new(
        config,
        tolerance,
        crate::validation::esp_r::test_automation::ReportFormat::Markdown,
    )
}

/// Run ESP-r validation as an integration tool
///
/// # Arguments
/// * `reference_path` - Path to ESP-r reference output file
/// * `fluxion_results` - Fluxion multi-zone validation results
/// * `tolerance` - Temperature tolerance for comparison
///
/// # Returns
/// Cross-validation report
pub fn run_as_integration_tool(
    reference_path: PathBuf,
    fluxion_results: &crate::validation::MultiZoneValidationResults,
    tolerance: f64,
) -> Result<crate::validation::reports::CrossValidationReport, Box<dyn Error>> {
    let adapter = create_integration_adapter(reference_path, tolerance);
    adapter.run_validation(fluxion_results)
}

/// Generate integration report in the specified format
///
/// # Arguments
/// * `report` - Cross-validation report
/// * `format` - Output format
///
/// # Returns
/// Formatted report string
pub fn generate_integration_report(
    report: &crate::validation::reports::CrossValidationReport,
    format: crate::validation::esp_r::test_automation::ReportFormat,
) -> String {
    match format {
        crate::validation::esp_r::test_automation::ReportFormat::Json => {
            serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string())
        }
        crate::validation::esp_r::test_automation::ReportFormat::Markdown => {
            crate::validation::reports::generate_markdown_report(report)
        }
    }
}
