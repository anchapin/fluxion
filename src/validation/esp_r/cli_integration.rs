// validation/esp_r/cli_integration.rs
/// CLI integration layer for ESP-r cross-validation
///
/// Provides a user-friendly command-line interface for running ESP-r cross-validation
/// tests and generating reports with configurable tolerance and output formats.
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::{Path, PathBuf};

/// Parse Fluxion configuration from file
fn parse_fluxion_config(config_path: &PathBuf) -> Result<serde_json::Value, Box<dyn Error>> {
    let config_content = std::fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;
    Ok(config)
}

/// Run Fluxion simulation and return results
fn run_fluxion_simulation(
    config: &serde_json::Value,
) -> Result<crate::validation::MultiZoneValidationResults, Box<dyn Error>> {
    // In a real implementation, this would run the actual simulation
    // For now, return a mock result
    let mut results = crate::validation::MultiZoneValidationResults::default();
    results.add_zone_result("Zone1".to_string(), vec![22.0, 22.1, 22.2]);
    Ok(results)
}

/// CLI configuration for ESP-r cross-validation
#[derive(Debug, Serialize, Deserialize)]
pub struct EspRCliConfig {
    /// Path to ESP-r output CSV file
    pub esp_r_output: PathBuf,
    /// Path to Fluxion configuration JSON file
    pub fluxion_config: PathBuf,
    /// Validation tolerance in °C (default: 0.1)
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// Output format (JSON or Markdown, default: Markdown)
    #[serde(default = "default_format")]
    pub output_format: ReportFormat,
    /// Output file path (optional - if None, output to stdout)
    pub output_path: Option<PathBuf>,
}

fn default_tolerance() -> f64 {
    0.1
}

fn default_format() -> ReportFormat {
    ReportFormat::Markdown
}

/// Supported report formats for CLI output
#[derive(Debug, Serialize, Deserialize, Clone, clap::ValueEnum)]
pub enum ReportFormat {
    JSON,
    Markdown,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportFormat::JSON => write!(f, "json"),
            ReportFormat::Markdown => write!(f, "markdown"),
        }
    }
}

impl std::str::FromStr for ReportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(ReportFormat::JSON),
            "markdown" => Ok(ReportFormat::Markdown),
            other => Err(format!("Invalid format: {}", other)),
        }
    }
}

/// Main CLI validation result structure
#[derive(Debug, Serialize, Deserialize)]
pub struct EspRCliResult {
    /// Whether validation passed overall
    pub passed: bool,
    /// Overall pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Validation report
    pub report: crate::validation::esp_r::EspRValidationReport,
    /// Execution timestamp
    pub timestamp: u64,
    /// Any errors encountered
    pub errors: Option<String>,
}

impl EspRCliResult {
    /// Create a new CLI result
    pub fn new(
        passed: bool,
        pass_rate: f64,
        report: crate::validation::esp_r::EspRValidationReport,
        errors: Option<String>,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            passed,
            pass_rate,
            report,
            timestamp,
            errors,
        }
    }
}

/// Run CLI validation with given configuration
///
/// # Arguments
/// * `config` - CLI configuration
///
/// # Returns
/// CLI result with pass/fail status and report
pub fn run_cli_validation(config: &EspRCliConfig) -> Result<EspRCliResult, Box<dyn Error>> {
    // Validate input files exist
    validate_input_files(&config.esp_r_output, &config.fluxion_config)?;

    // Parse Fluxion configuration
    let fluxion_config = parse_fluxion_config(&config.fluxion_config)?;

    // Create ESP-r validator
    let validator =
        crate::validation::esp_r::EspRValidator::new(config.esp_r_output.clone(), config.tolerance);

    // Run Fluxion simulation to get results
    let fluxion_results = run_fluxion_simulation(&fluxion_config)?;

    // Run cross-validation
    let report = validator.validate(&fluxion_results)?;

    // Calculate pass rate
    let total_zones = report.zone_results.len();
    let passed_zones = report
        .zone_results
        .iter()
        .filter(|z| z.temp_within_tolerance && z.heating_within_tolerance)
        .count();
    let pass_rate = if total_zones > 0 {
        passed_zones as f64 / total_zones as f64
    } else {
        0.0
    };

    // Determine overall test result
    let passed = pass_rate >= 0.95; // 95% pass rate required

    // Print summary to console
    print_summary(&config, passed, pass_rate);

    // Save report if output path specified
    let result = EspRCliResult::new(passed, pass_rate, report.clone(), None);
    if let Some(output_path) = &config.output_path {
        save_report(&config, &result, output_path)?;
    }

    Ok(result)
}

/// Validate that input files exist and are accessible
fn validate_input_files(esp_r_output: &Path, fluxion_config: &Path) -> Result<(), Box<dyn Error>> {
    if !esp_r_output.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("ESP-r output file not found: {}", esp_r_output.display()),
        )));
    }

    if !fluxion_config.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Fluxion config file not found: {}",
                fluxion_config.display()
            ),
        )));
    }

    Ok(())
}

/// Save validation report to file
fn save_report(
    config: &EspRCliConfig,
    result: &EspRCliResult,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    match config.output_format {
        ReportFormat::JSON => {
            let json_string = serde_json::to_string_pretty(result)?;
            std::fs::write(output_path, json_string)?;
        }
        ReportFormat::Markdown => {
            let markdown_report = generate_markdown_report(result)?;
            std::fs::write(output_path, markdown_report)?;
        }
    }

    println!("Report saved to: {}", output_path.display());
    Ok(())
}

/// Generate Markdown formatted report
pub fn generate_markdown_report(result: &EspRCliResult) -> Result<String, Box<dyn Error>> {
    let timestamp = chrono::NaiveDateTime::from_timestamp_opt(result.timestamp as i64, 0)
        .unwrap()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    let mut report = String::new();

    // Header
    report.push_str(&format!("# ESP-r Cross-Validation CLI Report\n\n"));
    report.push_str(&format!("**Timestamp:** {}\n\n", timestamp));

    // Test result summary
    report.push_str(&format!("## Test Results\n\n"));
    report.push_str(&format!(
        "**Overall Status:** {}\n\n",
        if result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    ));
    report.push_str(&format!(
        "**Pass Rate:** {:.1}%\n\n",
        result.pass_rate * 100.0
    ));

    // Zone results table
    report.push_str(&format!("## Zone Results\n\n"));
    report.push_str(&format!("| Zone ID | Temp Within Tolerance | Heating Within Tolerance | Temp Difference | Heating Difference |\n"));
    report.push_str(&format!("|----------|------------------------|----------------------------|------------------|---------------------|\n"));

    for zone_result in &result.report.zone_results {
        report.push_str(&format!(
            "| {} | {} | {} | {:.2}°C | {:.2} W |\n",
            zone_result.zone_id,
            if zone_result.temp_within_tolerance {
                "✅"
            } else {
                "❌"
            },
            if zone_result.heating_within_tolerance {
                "✅"
            } else {
                "❌"
            },
            zone_result.temp_difference,
            zone_result.heating_difference,
        ));
    }

    // Statistics
    report.push_str(&format!("\n## Statistics\n\n"));
    report.push_str(&format!(
        "- **Mean Temperature Difference:** {:.2}°C\n\n",
        result.report.statistics.mean_temp_difference
    ));
    report.push_str(&format!(
        "- **Max Temperature Difference:** {:.2}°C\n\n",
        result.report.statistics.max_temp_difference
    ));
    report.push_str(&format!(
        "- **Mean Heating Difference:** {:.2} W\n\n",
        result.report.statistics.mean_heating_difference
    ));
    report.push_str(&format!(
        "- **Max Heating Difference:** {:.2} W\n\n",
        result.report.statistics.max_heating_difference
    ));

    Ok(report)
}

/// Print validation summary to console
fn print_summary(config: &EspRCliConfig, passed: bool, pass_rate: f64) {
    println!("ESP-r Cross-Validation Summary");
    println!("==============================");
    println!("ESP-r Output: {}", config.esp_r_output.display());
    println!("Fluxion Config: {}", config.fluxion_config.display());
    println!("Tolerance: {}°C", config.tolerance);
    println!("Format: {}", config.output_format);
    println!();
    println!("Results:");
    println!(
        "  Status: {}",
        if passed { "✅ PASSED" } else { "❌ FAILED" }
    );
    println!("  Pass Rate: {:.1}%", pass_rate * 100.0);
    println!();
}

/// Helper function to run validation and return formatted output
pub fn run_validation_and_format_output(config: &EspRCliConfig) -> Result<String, Box<dyn Error>> {
    let result = run_cli_validation(config)?;

    match config.output_format {
        ReportFormat::JSON => {
            let json_string = serde_json::to_string_pretty(&result)?;
            Ok(json_string)
        }
        ReportFormat::Markdown => {
            let markdown_report = generate_markdown_report(&result)?;
            Ok(markdown_report)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_config_serialization() {
        let config = EspRCliConfig {
            esp_r_output: PathBuf::from("test.csv"),
            fluxion_config: PathBuf::from("config.json"),
            tolerance: 0.5,
            output_format: ReportFormat::JSON,
            output_path: Some(PathBuf::from("output.json")),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EspRCliConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.tolerance, 0.5);
        assert!(matches!(deserialized.output_format, ReportFormat::JSON));
    }

    #[test]
    fn test_report_format_display() {
        assert_eq!(ReportFormat::JSON.to_string(), "json");
        assert_eq!(ReportFormat::Markdown.to_string(), "markdown");
    }

    #[test]
    fn test_report_format_from_str() {
        assert!(matches!(
            "json".parse::<ReportFormat>(),
            Ok(ReportFormat::JSON)
        ));
        assert!(matches!(
            "markdown".parse::<ReportFormat>(),
            Ok(ReportFormat::Markdown)
        ));
        assert!("invalid".parse::<ReportFormat>().is_err());
    }
}
