use crate::validation::report::BenchmarkReport;
use serde_json;

/// Baseline performance metrics for guardrail comparison.
#[derive(Debug, Clone)]
pub struct GuardrailBaseline {
    pub mae: f64,
    pub max_deviation: f64,
    pub pass_rate: f64,
    pub validation_time_seconds: f64,
}

impl GuardrailBaseline {
    /// Loads baseline metrics from a JSON file.
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read baseline file: {}", e))?;
        #[derive(serde::Deserialize)]
        struct BaselineJSON {
            mae: f64,
            max_deviation: f64,
            pass_rate: f64,
            validation_time_seconds: f64,
        }
        let json: BaselineJSON = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse baseline JSON: {}", e))?;
        Ok(GuardrailBaseline {
            mae: json.mae,
            max_deviation: json.max_deviation,
            pass_rate: json.pass_rate,
            validation_time_seconds: json.validation_time_seconds,
        })
    }
}

/// Checks a validation report against guardrail baseline thresholds.
///
/// Returns (success, list_of_failures). Success is true only if no thresholds are violated.
pub fn check(report: &BenchmarkReport, baseline: &GuardrailBaseline) -> (bool, Vec<String>) {
    let mut failures = Vec::new();
    let mae = report.mae();
    let max_dev = report.max_deviation();
    let pass_rate = report.pass_rate();
    let duration = report.duration_seconds();

    // MAE threshold: >2% increase causes failure
    if mae > baseline.mae * 1.02 {
        failures.push(format!(
            "MAE {:.2}% exceeds 2% threshold over baseline {:.2}%",
            mae, baseline.mae
        ));
    }

    // MaxDev threshold: >10% increase causes failure
    if max_dev > baseline.max_deviation * 1.10 {
        failures.push(format!(
            "Max Deviation {:.2}% exceeds 10% threshold over baseline {:.2}%",
            max_dev, baseline.max_deviation
        ));
    }

    // PassRate threshold: drop >5 percentage points causes failure
    if pass_rate < baseline.pass_rate - 5.0 {
        failures.push(format!(
            "Pass Rate {:.1}% dropped >5pp from baseline {:.1}%",
            pass_rate, baseline.pass_rate
        ));
    }

    // Duration threshold: >110% is a warning, not a failure
    if duration > baseline.validation_time_seconds * 1.10 {
        // Only print a warning; not a failure
        eprintln!(
            "Warning: Validation time {:.2}s is >10% slower than baseline {:.2}s",
            duration, baseline.validation_time_seconds
        );
    }

    let success = failures.is_empty();
    (success, failures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::report::{MetricType, ValidationResult, ValidationStatus};

    #[test]
    fn test_guardrails_all_pass() {
        let baseline = GuardrailBaseline {
            mae: 5.0,
            max_deviation: 10.0,
            pass_rate: 90.0,
            validation_time_seconds: 60.0,
        };

        let mut report = BenchmarkReport::default();
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 5.0,
            ref_min: 4.5,
            ref_max: 5.5,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            actual: 5.0,
            min: 4.5,
            max: 5.5,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        report.results.push(result);

        let (success, failures) = check(&report, &baseline);
        assert!(success, "Expected all guardrails to pass");
        assert!(failures.is_empty());
    }

    #[test]
    fn test_guardrails_mae_failure() {
        let baseline = GuardrailBaseline {
            mae: 5.0,
            max_deviation: 10.0,
            pass_rate: 90.0,
            validation_time_seconds: 60.0,
        };

        let mut report = BenchmarkReport::default();
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 6.0,
            ref_min: 5.0,
            ref_max: 7.0,
            percent_error: 6.0,
            status: ValidationStatus::Warning,
            actual: 6.0,
            min: 5.0,
            max: 7.0,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        report.results.push(result);

        let (success, failures) = check(&report, &baseline);
        assert!(!success, "Expected MAE guardrail to fail");
        assert!(failures.iter().any(|f| f.contains("MAE")));
    }

    #[test]
    fn test_guardrails_max_deviation_failure() {
        let baseline = GuardrailBaseline {
            mae: 5.0,
            max_deviation: 10.0,
            pass_rate: 90.0,
            validation_time_seconds: 60.0,
        };

        let mut report = BenchmarkReport::default();
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 12.0,
            ref_min: 10.0,
            ref_max: 14.0,
            percent_error: 12.0,
            status: ValidationStatus::Warning,
            actual: 12.0,
            min: 10.0,
            max: 14.0,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        report.results.push(result);

        let (success, failures) = check(&report, &baseline);
        assert!(!success, "Expected max deviation guardrail to fail");
        assert!(failures.iter().any(|f| f.contains("Max Deviation")));
    }

    #[test]
    fn test_guardrails_pass_rate_failure() {
        let baseline = GuardrailBaseline {
            mae: 5.0,
            max_deviation: 10.0,
            pass_rate: 90.0,
            validation_time_seconds: 60.0,
        };

        let mut report = BenchmarkReport::default();
        let pass_result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 5.0,
            ref_min: 4.5,
            ref_max: 5.5,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            actual: 5.0,
            min: 4.5,
            max: 5.5,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        let fail_result = ValidationResult {
            case_id: "610".to_string(),
            metric: MetricType::AnnualCooling,
            fluxion_value: 10.0,
            ref_min: 4.5,
            ref_max: 5.5,
            percent_error: 50.0,
            status: ValidationStatus::Fail,
            actual: 10.0,
            min: 4.5,
            max: 5.5,
            metric_type: MetricType::AnnualCooling,
            per_program: None,
        };
        report.results.push(pass_result);
        report.results.push(fail_result);

        let (success, failures) = check(&report, &baseline);
        assert!(!success, "Expected pass rate guardrail to fail");
        assert!(failures.iter().any(|f| f.contains("Pass Rate")));
    }

    #[test]
    fn test_guardrails_duration_warning_only() {
        let baseline = GuardrailBaseline {
            mae: 5.0,
            max_deviation: 10.0,
            pass_rate: 90.0,
            validation_time_seconds: 60.0,
        };

        let mut report = BenchmarkReport::default();
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 5.0,
            ref_min: 4.5,
            ref_max: 5.5,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            actual: 5.0,
            min: 4.5,
            max: 5.5,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        report.results.push(result);

        let (success, failures) = check(&report, &baseline);
        assert!(success, "Duration should only produce warning, not failure");
        assert!(failures.is_empty());
    }

    #[test]
    fn test_guardrails_baseline_load_missing_file() {
        let result = GuardrailBaseline::load("/nonexistent/path/baseline.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read baseline file"));
    }

    #[test]
    fn test_guardrails_baseline_load_invalid_json() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "fluxion_test_invalid_baseline_{}.json",
            std::process::id()
        ));
        {
            let mut file = std::fs::File::create(&file_path).unwrap();
            writeln!(file, "not valid json").unwrap();
        }

        let result = GuardrailBaseline::load(file_path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Failed to parse baseline JSON"));

        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_guardrails_baseline_load_valid() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "fluxion_test_valid_baseline_{}.json",
            std::process::id()
        ));
        {
            let mut file = std::fs::File::create(&file_path).unwrap();
            write!(
                file,
                r#"{{"mae": 5.0, "max_deviation": 10.0, "pass_rate": 90.0, "validation_time_seconds": 60.0}}"#
            )
            .unwrap();
        }

        let result = GuardrailBaseline::load(file_path.to_str().unwrap());
        assert!(result.is_ok());
        let baseline = result.unwrap();
        assert_eq!(baseline.mae, 5.0);
        assert_eq!(baseline.max_deviation, 10.0);
        assert_eq!(baseline.pass_rate, 90.0);
        assert_eq!(baseline.validation_time_seconds, 60.0);

        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_guardrails_multiple_failures() {
        let baseline = GuardrailBaseline {
            mae: 5.0,
            max_deviation: 10.0,
            pass_rate: 90.0,
            validation_time_seconds: 60.0,
        };

        let mut report = BenchmarkReport::default();
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 15.0,
            ref_min: 10.0,
            ref_max: 20.0,
            percent_error: 15.0,
            status: ValidationStatus::Warning,
            actual: 15.0,
            min: 10.0,
            max: 20.0,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        report.results.push(result);

        let (success, failures) = check(&report, &baseline);
        assert!(!success);
        assert!(failures.len() >= 2, "Expected at least 2 failures");
    }
}
