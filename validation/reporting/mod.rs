// validation/reporting/mod.rs
/// Validation reporting module
///
/// This module provides comprehensive reporting capabilities for all validation modules
/// including ASHRAE 140, climate zone validation, and occupancy pattern validation
pub mod generator;

/// Reporting configuration
#[derive(Debug, Clone)]
pub struct ReportingConfig {
    /// Output directory for reports
    pub output_dir: String,
    /// Report format (markdown, html, json)
    pub format: ReportFormat,
    /// Include detailed diagnostics
    pub include_diagnostics: bool,
    /// Generate comprehensive reports
    pub comprehensive: bool,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            output_dir: "validation/reports".to_string(),
            format: ReportFormat::Markdown,
            include_diagnostics: true,
            comprehensive: true,
        }
    }
}

/// Supported report formats
#[derive(Debug, Clone, PartialEq)]
pub enum ReportFormat {
    Markdown,
    Html,
    Json,
    Comprehensive,
}

/// Main reporting struct
pub struct ValidationReporter {
    config: ReportingConfig,
}

impl ValidationReporter {
    /// Create a new validation reporter
    pub fn new(config: ReportingConfig) -> Self {
        Self { config }
    }

    /// Generate comprehensive validation report
    pub fn generate_comprehensive_report(&self) -> Result<String, String> {
        // This will be implemented to generate reports from all validation modules
        Ok("Comprehensive report generation placeholder".to_string())
    }
}
