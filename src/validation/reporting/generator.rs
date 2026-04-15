// validation/reporting/generator.rs
/// Validation report generator
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Validation report generator
#[derive(Debug, Default)]
pub struct ValidationReportGenerator;

impl ValidationReportGenerator {
    /// Generate a validation report from results
    pub fn generate_report(&self, results: &str, format: &str) -> Result<String, Box<dyn Error>> {
        match format {
            "json" => self.generate_json_report(results),
            "markdown" => self.generate_markdown_report(results),
            "html" => self.generate_html_report(results),
            _ => Err("Unsupported format".into()),
        }
    }

    /// Generate JSON report
    fn generate_json_report(&self, results: &str) -> Result<String, Box<dyn Error>> {
        Ok(results.to_string())
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self, results: &str) -> Result<String, Box<dyn Error>> {
        Ok(format!(
            "# Validation Report\n\n```json\n{}\n```\n",
            results
        ))
    }

    /// Generate HTML report
    fn generate_html_report(&self, results: &str) -> Result<String, Box<dyn Error>> {
        Ok(format!(
            "<html><body><h1>Validation Report</h1><pre>{}</pre></body></html>",
            results
        ))
    }
}

/// Report generation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportOptions {
    /// Output format
    pub format: String,

    /// Include detailed statistics
    pub include_stats: bool,

    /// Include raw data
    pub include_raw_data: bool,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            format: "markdown".to_string(),
            include_stats: true,
            include_raw_data: false,
        }
    }
}
