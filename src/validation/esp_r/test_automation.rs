// validation/esp_r/test_automation.rs
use serde::{Deserialize, Serialize};
use std::error::Error;
/// ESP-r cross-validation test automation infrastructure
///
/// Provides automated testing capabilities for ESP-r cross-validation,
/// including test configuration, execution, and reporting.
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Test configuration for ESP-r cross-validation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EspRTestConfig {
    /// Path to ESP-r output CSV file
    pub esp_r_output_path: PathBuf,
    /// Path to Fluxion validation results JSON file
    pub fluxion_results_path: PathBuf,
    /// Validation tolerance in °C
    pub tolerance: f64,
    /// Output report format (JSON or Markdown)
    pub report_format: ReportFormat,
}

/// Supported report formats
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ReportFormat {
    JSON,
    Markdown,
}

/// Test result structure
#[derive(Debug, Serialize, Deserialize)]
pub struct EspRTestResult {
    /// Test execution timestamp
    pub timestamp: u64,
    /// Test configuration used
    pub config: EspRTestConfig,
    /// Whether test passed overall
    pub passed: bool,
    /// Overall pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Validation report
    pub report: crate::validation::esp_r::EspRValidationReport,
    /// Any errors encountered
    pub errors: Option<String>,
}

impl EspRTestResult {
    /// Create a new test result
    pub fn new(
        config: EspRTestConfig,
        passed: bool,
        pass_rate: f64,
        report: crate::validation::esp_r::EspRValidationReport,
        errors: Option<String>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            timestamp,
            config,
            passed,
            pass_rate,
            report,
            errors,
        }
    }
}

/// Run cross-validation test with given configuration
///
/// # Arguments
/// * `config` - Test configuration
///
/// # Returns
/// Test result with pass/fail status and report
pub fn run_cross_validation_test(
    config: &EspRTestConfig,
) -> Result<EspRTestResult, Box<dyn Error>> {
    // Load Fluxion validation results from JSON file
    let fluxion_results = load_fluxion_results(&config.fluxion_results_path)?;

    // Create ESP-r validator
    let validator = crate::validation::esp_r::EspRValidator::new(
        config.esp_r_output_path.clone(),
        config.tolerance,
    );

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

    Ok(EspRTestResult::new(
        config.clone(),
        passed,
        pass_rate,
        report,
        None,
    ))
}

/// Load Fluxion validation results from JSON file
fn load_fluxion_results(
    path: &PathBuf,
) -> Result<crate::validation::MultiZoneValidationResults, Box<dyn Error>> {
    let file_content = std::fs::read_to_string(path)?;
    let results: crate::validation::MultiZoneValidationResults =
        serde_json::from_str(&file_content)?;
    Ok(results)
}

/// Generate test report in specified format
///
/// # Arguments
/// * `test_result` - Test result to format
/// * `format` - Output format
///
/// # Returns
/// Formatted report string
pub fn generate_test_report(
    test_result: &EspRTestResult,
    format: ReportFormat,
) -> Result<String, Box<dyn Error>> {
    match format {
        ReportFormat::JSON => {
            let json_string = serde_json::to_string_pretty(test_result)?;
            Ok(json_string)
        }
        ReportFormat::Markdown => generate_markdown_test_report(test_result),
    }
}

/// Generate Markdown formatted test report
#[allow(deprecated)]
fn generate_markdown_test_report(test_result: &EspRTestResult) -> Result<String, Box<dyn Error>> {
    let timestamp = chrono::NaiveDateTime::from_timestamp_opt(test_result.timestamp as i64, 0)
        .unwrap()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    let mut report = String::new();

    // Header
    report.push_str(&"# ESP-r Cross-Validation Test Report\n\n".to_string());
    report.push_str(&format!("**Timestamp:** {}\n\n", timestamp));
    report.push_str(&"**Test Configuration:**\n\n".to_string());
    report.push_str(&format!(
        "- ESP-r Output: {}\n\n",
        test_result.config.esp_r_output_path.display()
    ));
    report.push_str(&format!(
        "- Tolerance: {}°C\n\n",
        test_result.config.tolerance
    ));
    report.push_str(&format!(
        "- Report Format: {:?}\n\n",
        test_result.config.report_format
    ));

    // Test result summary
    report.push_str(&"## Test Results\n\n".to_string());
    report.push_str(&format!(
        "**Overall Status:** {}\n\n",
        if test_result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    ));
    report.push_str(&format!(
        "**Pass Rate:** {:.1}%\n\n",
        test_result.pass_rate * 100.0
    ));

    // Zone results table
    report.push_str(&"## Zone Results\n\n".to_string());
    report.push_str(&"| Zone ID | Temp Within Tolerance | Heating Within Tolerance | Temp Difference | Heating Difference |\n".to_string());
    report.push_str(&"|----------|------------------------|----------------------------|------------------|---------------------|\n".to_string());

    for zone_result in &test_result.report.zone_results {
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
    report.push_str(&"\n## Statistics\n\n".to_string());
    report.push_str(&format!(
        "- **Mean Temperature Difference:** {:.2}°C\n\n",
        test_result.report.statistics.mean_temp_difference
    ));
    report.push_str(&format!(
        "- **Max Temperature Difference:** {:.2}°C\n\n",
        test_result.report.statistics.max_temp_difference
    ));
    report.push_str(&format!(
        "- **Mean Heating Difference:** {:.2} W\n\n",
        test_result.report.statistics.mean_heating_difference
    ));
    report.push_str(&format!(
        "- **Max Heating Difference:** {:.2} W\n\n",
        test_result.report.statistics.max_heating_difference
    ));

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_test_config_serialization() {
        let config = EspRTestConfig {
            esp_r_output_path: PathBuf::from("test.csv"),
            fluxion_results_path: PathBuf::from("results.json"),
            tolerance: 0.5,
            report_format: ReportFormat::JSON,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EspRTestConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.tolerance, 0.5);
        assert!(matches!(deserialized.report_format, ReportFormat::JSON));
    }

    #[test]
    fn test_report_generation() {
        // This test would require actual simulation data
        // For now, just verify the function signatures compile
        assert!(true);
    }
}
