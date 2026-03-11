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
