// validation/integration/mod.rs
/// Validation framework integration layer
///
/// This module provides a unified interface for integrating multiple validation tools
/// including ESP-r, EnergyPlus, and other reference tools.
use std::error::Error;
use std::path::PathBuf;

/// Validation tools supported by the framework
#[derive(Debug, Clone)]
pub enum ValidationTool {
    /// ESP-r cross-validation tool
    EspR,
    /// EnergyPlus validation tool (placeholder for future implementation)
    EnergyPlus,
    /// TRNSYS validation tool (placeholder for future implementation)
    Trnsys,
}

/// Configuration for validation integration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Path to reference data
    pub reference_path: PathBuf,
    /// Temperature tolerance for comparison
    pub tolerance: f64,
    /// Output format for reports
    pub report_format: ReportFormat,
}

/// Output format for validation reports
#[derive(Debug, Clone, PartialEq)]
pub enum ReportFormat {
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
}

/// Main validation integration struct
#[derive(Debug)]
pub struct ValidationIntegration {
    /// List of validation tools to run
    pub tools: Vec<ValidationTool>,
    /// Validation configuration
    pub config: ValidationConfig,
    /// Report format
    pub report_format: ReportFormat,
}

impl ValidationIntegration {
    /// Create a new ValidationIntegration instance
    ///
    /// # Arguments
    /// * `tools` - List of validation tools to include
    /// * `config` - Validation configuration
    /// * `report_format` - Output format for reports
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use fluxion::validation::integration::{ValidationIntegration, ValidationTool, ValidationConfig, ReportFormat};
    ///
    /// let config = ValidationConfig {
    ///     reference_path: PathBuf::from("reference.csv"),
    ///     tolerance: 0.5,
    ///     report_format: ReportFormat::Markdown,
    /// };
    ///
    /// let integration = ValidationIntegration::new(
    ///     vec![ValidationTool::EspR],
    ///     config,
    ///     ReportFormat::Markdown
    /// );
    /// ```
    pub fn new(
        tools: Vec<ValidationTool>,
        config: ValidationConfig,
        report_format: ReportFormat,
    ) -> Self {
        Self {
            tools,
            config,
            report_format,
        }
    }

    /// Run integrated validation across all configured tools
    ///
    /// # Returns
    /// Comprehensive validation report combining results from all tools
    ///
    /// # Example
    /// ```
    /// let report = integration.run_integrated_validation()?;
    /// ```
    pub fn run_integrated_validation(
        &self,
    ) -> Result<ComprehensiveValidationReport, Box<dyn Error>> {
        let mut tool_results = Vec::new();

        // Run each validation tool
        for tool in &self.tools {
            match tool {
                ValidationTool::EspR => {
                    let esp_r_result = self.run_esp_r_validation()?;
                    tool_results.push(ToolResult {
                        tool: ValidationTool::EspR,
                        result: esp_r_result,
                    });
                }
                ValidationTool::EnergyPlus => {
                    // Placeholder for future implementation
                    return Err("EnergyPlus integration not yet implemented".into());
                }
                ValidationTool::Trnsys => {
                    // Placeholder for future implementation
                    return Err("TRNSYS integration not yet implemented".into());
                }
            }
        }

        // Generate comprehensive report
        let comprehensive_report = self.generate_comprehensive_report(tool_results);

        Ok(comprehensive_report)
    }

    /// Run ESP-r specific validation
    ///
    /// # Returns
    /// ESP-r validation result
    fn run_esp_r_validation(&self) -> Result<EspRValidationResult, Box<dyn Error>> {
        // Create ESP-r validator
        let validator = crate::validation::esp_r::EspRValidator::new(
            self.config.reference_path.clone(),
            self.config.tolerance,
        );

        // For now, use default fluxion results - this will be parameterized later
        let fluxion_results = crate::validation::MultiZoneValidationResults::default();

        // Run validation
        let report = validator.validate(&fluxion_results)?;

        // Convert to our result format
        let result = EspRValidationResult {
            overall_pass: report.overall_pass,
            summary_statistics: report.summary_statistics,
            zone_count: report.zone_results.len(),
        };

        Ok(result)
    }

    /// Generate comprehensive report combining all tool results
    ///
    /// # Arguments
    /// * `tool_results` - Results from individual validation tools
    ///
    /// # Returns
    /// Comprehensive validation report
    fn generate_comprehensive_report(
        &self,
        tool_results: Vec<ToolResult>,
    ) -> ComprehensiveValidationReport {
        let mut report = ComprehensiveValidationReport {
            overall_status: OverallStatus::Pass,
            tool_results,
            summary: Summary {
                total_tools: self.tools.len(),
                passed_tools: 0,
                failed_tools: 0,
            },
        };

        // Count passed/failed tools
        for tool_result in &report.tool_results {
            match tool_result.result {
                ToolResultType::EspR(ref esp_r_result) => {
                    if esp_r_result.overall_pass {
                        report.summary.passed_tools += 1;
                    } else {
                        report.summary.failed_tools += 1;
                        report.overall_status = OverallStatus::Fail;
                    }
                }
            }
        }

        report
    }
}

/// Result from a specific validation tool
#[derive(Debug)]
pub struct ToolResult {
    /// Validation tool that produced this result
    pub tool: ValidationTool,
    /// Result data
    pub result: ToolResultType,
}

/// Enum for different types of tool results
#[derive(Debug)]
pub enum ToolResultType {
    /// ESP-r validation result
    EspR(EspRValidationResult),
    /// EnergyPlus validation result (placeholder)
    EnergyPlus,
    /// TRNSYS validation result (placeholder)
    Trnsys,
}

/// ESP-r specific validation result
#[derive(Debug)]
pub struct EspRValidationResult {
    /// Overall pass/fail status
    pub overall_pass: bool,
    /// Summary statistics
    pub summary_statistics: crate::validation::reports::SummaryStatistics,
    /// Number of zones validated
    pub zone_count: usize,
}

/// Comprehensive validation report
#[derive(Debug)]
pub struct ComprehensiveValidationReport {
    /// Overall validation status
    pub overall_status: OverallStatus,
    /// Results from individual tools
    pub tool_results: Vec<ToolResult>,
    /// Summary information
    pub summary: Summary,
}

/// Overall validation status
#[derive(Debug, PartialEq)]
pub enum OverallStatus {
    /// Validation passed
    Pass,
    /// Validation failed
    Fail,
}

/// Summary information for comprehensive report
#[derive(Debug)]
pub struct Summary {
    /// Total number of validation tools run
    pub total_tools: usize,
    /// Number of tools that passed
    pub passed_tools: usize,
    /// Number of tools that failed
    pub failed_tools: usize,
}

/// Convenience function to run ESP-r validation as part of integration
pub fn run_esp_r_integration(
    reference_path: PathBuf,
    tolerance: f64,
    report_format: ReportFormat,
) -> Result<EspRValidationResult, Box<dyn Error>> {
    let config = ValidationConfig {
        reference_path,
        tolerance,
        report_format,
    };

    let integration =
        ValidationIntegration::new(vec![ValidationTool::EspR], config, ReportFormat::Markdown);

    match integration.run_integrated_validation()? {
        ComprehensiveValidationReport { tool_results, .. } => {
            if let Some(ToolResult {
                result: ToolResultType::EspR(esp_r_result),
                ..
            }) = tool_results.first()
            {
                Ok(esp_r_result.clone())
            } else {
                Err("No ESP-r results found".into())
            }
        }
    }
}
