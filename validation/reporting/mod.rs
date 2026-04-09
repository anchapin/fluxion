// validation/reporting/mod.rs
pub mod cli;
pub mod examples;
/// Validation reporting module
///
/// This module provides comprehensive reporting capabilities for all validation modules
/// including ASHRAE 140, climate zone validation, and occupancy pattern validation
pub mod generator;

use std::fs;

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
    pub fn generate_comprehensive_report(&self) -> Result<ComprehensiveValidationReport, String> {
        let generator = generator::ComprehensiveReportGenerator::new();
        generator.generate_report()
    }

    /// Generate comprehensive report and export to JSON
    pub fn generate_json_report(&self, path: &str) -> Result<(), String> {
        let generator = generator::ComprehensiveReportGenerator::new();
        let report = generator.generate_report()?;
        generator.export_json(&report, path)
    }

    /// Generate comprehensive report and export to HTML
    pub fn generate_html_report(&self, path: &str) -> Result<(), String> {
        let generator = generator::ComprehensiveReportGenerator::new();
        let report = generator.generate_report()?;
        let html_content = generator.generate_html(&report)?;
        fs::write(path, html_content).map_err(|e| format!("Failed to write HTML report: {}", e))?;
        Ok(())
    }

    /// Generate comprehensive report and export to Markdown
    pub fn generate_markdown_report(&self, path: &str) -> Result<(), String> {
        let generator = generator::ComprehensiveReportGenerator::new();
        let report = generator.generate_report()?;
        let markdown_content = generator.generate_markdown(&report)?;
        fs::write(path, markdown_content)
            .map_err(|e| format!("Failed to write Markdown report: {}", e))?;
        Ok(())
    }
}
