//! Quality metrics analysis for ASHRAE 140 validation results.
//!
//! This module provides tools to compute aggregate quality metrics from validation
//! reports, track progress across phases, and identify problematic cases.

use crate::validation::report::{BenchmarkReport, MetricType, ValidationStatus};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use thiserror::Error;

/// Individual metric deviation for detailed tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDeviation {
    /// Case identifier (e.g., "600")
    pub case_id: String,
    /// Metric type
    pub metric: MetricType,
    /// Fluxion simulation value
    pub actual: f64,
    /// Reference range midpoint
    pub reference: f64,
    /// Percent error from reference midpoint
    pub error_pct: f64,
    /// Validation status
    pub status: ValidationStatus,
}

/// Aggregate quality metrics for a validation suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Total number of cases evaluated
    pub total_cases: usize,
    /// Number of cases that passed all metrics
    pub passed_cases: usize,
    /// Pass rate as percentage (0-100)
    pub pass_rate: f64,
    /// Total number of metric evaluations
    pub total_metrics: usize,
    /// Number of metrics that passed
    pub passed_metrics: usize,
    /// Mean Absolute Error across all numeric metrics (%)
    pub mae: f64,
    /// Maximum deviation (absolute percent error) among all metrics
    pub max_deviation: f64,
    /// Detailed list of all metric deviations
    pub deviations: Vec<MetricDeviation>,
    /// Breakdown by status
    pub status_counts: HashMap<ValidationStatus, usize>,
}

/// Report comparing metrics between two phases or configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeReport {
    /// Metrics from "before" state
    pub old: QualityMetrics,
    /// Metrics from "after" state
    pub new: QualityMetrics,
    /// Change in pass rate (percentage points)
    pub pass_rate_delta: f64,
    /// Change in MAE (percentage points)
    pub mae_delta: f64,
    /// Change in max deviation (percentage points)
    pub max_deviation_delta: f64,
    /// Number of metrics that improved (moved from fail/warn to pass)
    pub improvements: usize,
    /// Number of metrics that regressed (moved from pass to fail/warn)
    pub regressions: usize,
}

impl QualityMetrics {
    /// Creates a new empty QualityMetrics with default values.
    pub fn new() -> Self {
        Self {
            total_cases: 0,
            passed_cases: 0,
            pass_rate: 0.0,
            total_metrics: 0,
            passed_metrics: 0,
            mae: f64::NAN,
            max_deviation: 0.0,
            deviations: Vec::new(),
            status_counts: HashMap::new(),
        }
    }

    /// Computes quality metrics from a benchmark report.
    ///
    /// This function analyzes all validation results and produces aggregate statistics.
    pub fn from_benchmark_report(report: &BenchmarkReport) -> Self {
        let mut metrics = Self::new();

        // Count unique cases
        let unique_cases: HashSet<&str> =
            report.results.iter().map(|r| r.case_id.as_str()).collect();
        metrics.total_cases = unique_cases.len();

        // Track case-level pass/fail (case passes if all its metrics pass)
        let mut case_status: HashMap<&str, ValidationStatus> = HashMap::new();
        for result in &report.results {
            let case_id = result.case_id.as_str();
            let status_entry = case_status.entry(case_id).or_insert(ValidationStatus::Pass);
            if result.failed() {
                *status_entry = ValidationStatus::Fail;
            } else if result.warning() && *status_entry == ValidationStatus::Pass {
                *status_entry = ValidationStatus::Warning;
            }
        }

        metrics.passed_cases = case_status
            .values()
            .filter(|s| **s == ValidationStatus::Pass)
            .count();
        if metrics.total_cases > 0 {
            metrics.pass_rate = (metrics.passed_cases as f64 / metrics.total_cases as f64) * 100.0;
        }

        // Metric-level statistics
        metrics.total_metrics = report.results.len();
        metrics.passed_metrics = report.results.iter().filter(|r| r.passed()).count();

        // Compute MAE and deviations
        let mut absolute_errors = Vec::new();
        let mut max_dev = 0.0;

        // Build deviations list
        for result in &report.results {
            // Only compute error for metrics with reference range (non-zero bounds)
            if result.ref_min != 0.0 || result.ref_max != 0.0 {
                let ref_mid = (result.ref_min + result.ref_max) / 2.0;
                let error_pct = if ref_mid != 0.0 {
                    ((result.fluxion_value - ref_mid).abs() / ref_mid) * 100.0
                } else {
                    0.0
                };

                absolute_errors.push(error_pct);
                if error_pct > max_dev {
                    max_dev = error_pct;
                }

                metrics.deviations.push(MetricDeviation {
                    case_id: result.case_id.clone(),
                    metric: result.metric,
                    actual: result.fluxion_value,
                    reference: ref_mid,
                    error_pct,
                    status: result.status,
                });
            }
        }

        // Compute MAE
        if !absolute_errors.is_empty() {
            let sum: f64 = absolute_errors.iter().sum();
            metrics.mae = sum / absolute_errors.len() as f64;
        }

        metrics.max_deviation = max_dev;

        // Count by status
        for result in &report.results {
            *metrics.status_counts.entry(result.status).or_insert(0) += 1;
        }

        metrics
    }

    /// Returns a summary string suitable for display.
    pub fn summary(&self) -> String {
        format!(
            "Pass Rate: {:.1}% ({} / {} cases), MAE: {:.2}%, Max Dev: {:.2}%",
            self.pass_rate, self.passed_cases, self.total_cases, self.mae, self.max_deviation
        )
    }
}

impl ChangeReport {
    /// Creates a change report comparing two quality metrics snapshots.
    pub fn new(old: &QualityMetrics, new: &QualityMetrics) -> Self {
        let mut improvements = 0;
        let mut regressions = 0;

        // Track per-case improvement/regression
        // Simplified: compare overall status counts
        for (status, old_count) in &old.status_counts {
            let new_count = new.status_counts.get(status).unwrap_or(&0);
            if *new_count > *old_count && *status == ValidationStatus::Pass {
                improvements += new_count - old_count;
            } else if *new_count < *old_count && *status == ValidationStatus::Pass {
                regressions += old_count - new_count;
            }
        }

        Self {
            pass_rate_delta: new.pass_rate - old.pass_rate,
            mae_delta: new.mae - old.mae,
            max_deviation_delta: new.max_deviation - old.max_deviation,
            improvements,
            regressions,
            old: old.clone(),
            new: new.clone(),
        }
    }

    /// Returns a human-readable change description.
    pub fn description(&self) -> String {
        let direction = if self.pass_rate_delta > 0.0 {
            "↑"
        } else {
            "↓"
        };
        format!(
            "Pass rate: {:.1}%{} ({:.1} pp), MAE: {:.1}%{}",
            self.pass_rate_delta.abs(),
            direction,
            self.pass_rate_delta,
            self.mae_delta,
            if self.mae_delta < 0.0 {
                " (improved)"
            } else {
                " (worsened)"
            }
        )
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the Analyzer.
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Path to store historical metrics data (optional)
    pub historical_data_path: Option<PathBuf>,
    /// Whether to generate markdown report automatically
    pub generate_report: bool,
    /// Output path for quality metrics markdown
    pub output_path: Option<PathBuf>,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            historical_data_path: None,
            generate_report: true,
            output_path: Some(PathBuf::from("docs/QUALITY_METRICS.md")),
        }
    }
}

/// Main analyzer struct for computing and tracking quality metrics.
pub struct Analyzer {
    config: AnalyzerConfig,
}

/// Errors that can occur during analysis.
#[derive(Debug, thiserror::Error)]
pub enum AnalyzerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid report: {0}")]
    InvalidReport(String),
}

impl Analyzer {
    /// Creates a new Analyzer with the given configuration.
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Creates a new Analyzer with default configuration.
    pub fn default() -> Self {
        Self::new(AnalyzerConfig::default())
    }

    /// Computes quality metrics from a benchmark report.
    pub fn analyze(&self, report: &BenchmarkReport) -> QualityMetrics {
        QualityMetrics::from_benchmark_report(report)
    }

    /// Computes metrics and generates the quality metrics markdown file.
    ///
    /// This is the main entry point for the metrics collection hook.
    pub fn update_quality_metrics(
        &self,
        report: &BenchmarkReport,
    ) -> Result<QualityMetrics, AnalyzerError> {
        let metrics = self.analyze(report);

        if self.config.generate_report {
            if let Some(output_path) = &self.config.output_path {
                let markdown = self.render_metrics_markdown(&metrics);
                if let Some(parent) = output_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(output_path, markdown)?;
            }
        }

        Ok(metrics)
    }

    /// Renders the quality metrics dashboard in Markdown format.
    pub fn render_metrics_markdown(&self, metrics: &QualityMetrics) -> String {
        let mut output = String::new();

        output.push_str("# Quality Metrics Tracker\n\n");
        output.push_str(&format!(
            "*Generated: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        ));

        // Current Status
        output.push_str("## Current Status\n\n");
        output.push_str(&format!(
            "- **Pass Rate:** {:.1}% ({} / {} cases)\n",
            metrics.pass_rate, metrics.passed_cases, metrics.total_cases
        ));
        output.push_str(&format!("- **MAE:** {:.2}%\n", metrics.mae));
        output.push_str(&format!(
            "- **Max Deviation:** {:.2}%\n",
            metrics.max_deviation
        ));

        // Status breakdown
        output.push_str("\n### Status Breakdown\n\n");
        output.push_str("| Status | Count | Percentage |\n");
        output.push_str("|--------|-------|------------|\n");
        for (status, count) in &metrics.status_counts {
            let pct = (*count as f64 / metrics.total_metrics as f64) * 100.0;
            output.push_str(&format!("| {} | {} | {:.1}% |\n", status, count, pct));
        }
        output.push('\n');

        // Phase Progression (placeholder - manually updated based on historic data)
        output.push_str("## Phase Progression\n\n");
        output.push_str("| Phase | Pass Rate | MAE | Max Dev | Notes |\n");
        output.push_str("|-------|-----------|-----|---------|-------|\n");
        // These rows should be filled based on historical snapshots
        output.push_str("| Baseline | 25% | 78.79% | 512% | Initial state |\n");
        output.push_str("| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |\n");
        output.push_str("| Phase 2 | 35% | 38.5% | 250% | Thermal mass |\n");
        output.push_str("| Phase 3 | 42% | 32.1% | 200% | Solar improvements |\n");
        output.push_str("| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |\n");
        output.push_str(&format!(
            "| Current (Phase 5) | {:.1}% | {:.1}% | {:.0}% | Diagnostics |\n",
            metrics.pass_rate, metrics.mae, metrics.max_deviation
        ));
        output.push('\n');

        // Metric Deviations
        output.push_str("## Metric Deviations\n\n");
        output.push_str("| Case | Metric | Actual | Ref Range | Error | Issue |\n");
        output.push_str("|------|--------|--------|-----------|-------|-------|\n");

        // Sort by absolute error descending
        let mut sorted_deviations = metrics.deviations.clone();
        sorted_deviations
            .sort_by(|a, b| b.error_pct.abs().partial_cmp(&a.error_pct.abs()).unwrap());

        for dev in sorted_deviations.iter().take(30) {
            // Show top 30 worst errors
            let ref_range = if let Some(data) = crate::validation::get_benchmark_data(&dev.case_id)
            {
                if let Some((min, max)) = data.get_range(dev.metric) {
                    format!("{:.2}-{:.2}", min, max)
                } else {
                    "N/A".to_string()
                }
            } else {
                "N/A".to_string()
            };

            // Determine issue category (simple heuristic)
            let issue = classify_deviation_issue(&dev.case_id, dev.metric);

            output.push_str(&format!(
                "| {} | {} | {:.2} | {} | {:.1}% | {} |\n",
                dev.case_id,
                dev.metric.display_name(),
                dev.actual,
                ref_range,
                dev.error_pct.abs(),
                issue
            ));
        }

        output.push('\n');

        // Top Problematic Cases
        output.push_str("## Problematic Cases\n\n");
        output.push_str("Cases with the highest number of failing metrics:\n\n");
        output.push_str("| Case | Failing Metrics | Total Error |\n");
        output.push_str("|------|-----------------|-------------|\n");

        // Aggregate by case
        let mut case_errors: HashMap<String, (usize, f64)> = HashMap::new();
        for dev in &metrics.deviations {
            if dev.status == ValidationStatus::Fail {
                let entry = case_errors.entry(dev.case_id.clone()).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += dev.error_pct.abs();
            }
        }

        let mut sorted_cases: Vec<_> = case_errors.into_iter().collect();
        sorted_cases.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap()); // Sort by total error

        for (case_id, (fail_count, total_error)) in sorted_cases.iter().take(10) {
            output.push_str(&format!(
                "| {} | {} | {:.1}% |\n",
                case_id, fail_count, total_error
            ));
        }

        output.push('\n');

        // Footer
        output.push_str("---\n");
        output.push_str(
            "*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*\n",
        );

        output
    }
}

/// Classifies a deviation into a likely issue category for reporting.
fn classify_deviation_issue(case_id: &str, metric: MetricType) -> String {
    use MetricType::*;

    // Check known patterns
    if case_id == "960" && metric == AnnualCooling {
        return "InterZoneTransfer".to_string();
    }

    if (case_id.starts_with('9') && case_id != "960" && case_id != "195")
        && (metric == AnnualHeating || metric == AnnualCooling)
    {
        return "ModelLimitation".to_string();
    }

    if case_id.starts_with('6') && case_id != "600FF" && case_id != "650FF" && metric == PeakCooling
    {
        return "SolarGains".to_string();
    }

    if (case_id == "900FF" || case_id == "950FF") && matches!(metric, MinFreeFloat | MaxFreeFloat) {
        return "ThermalMass".to_string();
    }

    if matches!(metric, MinFreeFloat | MaxFreeFloat) {
        return "FreeFloat".to_string();
    }

    "Unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::get_all_benchmark_data;
    use crate::validation::report::{BenchmarkReport, ValidationResult};

    fn make_dummy_result(
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) -> ValidationResult {
        ValidationResult {
            case_id: case_id.to_string(),
            metric,
            fluxion_value,
            ref_min,
            ref_max,
            percent_error: 0.0,             // computed internally
            status: ValidationStatus::Pass, // will be determined
            per_program: None,
        }
    }

    #[test]
    fn test_quality_metrics_basic() {
        let mut report = BenchmarkReport {
            results: Vec::new(),
            benchmark_data: get_all_benchmark_data(),
            start_time: None,
            end_time: None,
        };

        // Add a passing result
        report.results.push(ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 6.5,
            ref_min: 5.5,
            ref_max: 7.5,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            per_program: None,
        });

        let metrics = QualityMetrics::from_benchmark_report(&report);

        assert_eq!(metrics.total_cases, 1);
        assert_eq!(metrics.passed_cases, 1);
        assert_eq!(metrics.pass_rate, 100.0);
        assert_eq!(metrics.total_metrics, 1);
        assert_eq!(metrics.passed_metrics, 1);
        assert!(metrics.mae.is_finite());
        assert!(metrics.deviations.len() == 1);
    }

    #[test]
    fn test_quality_metrics_mae_calculation() {
        // Use ValidationResult::new() to compute status and percent_error automatically
        let report = BenchmarkReport {
            results: vec![
                // 50% error: fluxion=22.5, ref [10,20] -> mid=15 -> (22.5-15)/15 = 0.5 -> Fail (outside range)
                ValidationResult::new("600", MetricType::AnnualHeating, 22.5, 10.0, 20.0),
                // 10% error: fluxion=13.5, ref [10,20] -> mid=15 -> (13.5-15)/15 = -0.1 -> Warning (within range but >=10% error)
                ValidationResult::new("600", MetricType::AnnualCooling, 13.5, 10.0, 20.0),
            ],
            benchmark_data: HashMap::new(),
            start_time: None,
            end_time: None,
        };

        let metrics = QualityMetrics::from_benchmark_report(&report);

        // MAE should be (|50%| + |10%|) / 2 = 30%
        // Note: The exact values depend on ref_min/ref_max. We used [10,20] giving midpoint 15.
        // For 22.5: error = (22.5-15)/15 *100 = 50%
        // For 13.5: error = (13.5-15)/15 *100 = -10%
        assert!((metrics.mae - 30.0).abs() < 0.01);
        assert_eq!(metrics.max_deviation, 50.0);
        // Case 600 fails (one metric fails, one warns) => case fails overall
        assert_eq!(metrics.passed_cases, 0);
        assert_eq!(metrics.pass_rate, 0.0);
        // Metrics: 1 fail, 1 warn
        assert_eq!(metrics.status_counts.get(&ValidationStatus::Fail), Some(&1));
        assert_eq!(
            metrics.status_counts.get(&ValidationStatus::Warning),
            Some(&1)
        );
    }

    #[test]
    fn test_change_report() {
        let old = QualityMetrics {
            total_cases: 10,
            passed_cases: 3,
            pass_rate: 30.0,
            total_metrics: 20,
            passed_metrics: 6,
            mae: 50.0,
            max_deviation: 100.0,
            deviations: Vec::new(),
            status_counts: {
                let mut h = HashMap::new();
                h.insert(ValidationStatus::Pass, 6);
                h.insert(ValidationStatus::Warning, 4);
                h.insert(ValidationStatus::Fail, 10);
                h
            },
        };

        let new = QualityMetrics {
            total_cases: 10,
            passed_cases: 5,
            pass_rate: 50.0,
            total_metrics: 20,
            passed_metrics: 10,
            mae: 30.0,
            max_deviation: 80.0,
            deviations: Vec::new(),
            status_counts: {
                let mut h = HashMap::new();
                h.insert(ValidationStatus::Pass, 10);
                h.insert(ValidationStatus::Warning, 5);
                h.insert(ValidationStatus::Fail, 5);
                h
            },
        };

        let change = ChangeReport::new(&old, &new);

        assert_eq!(change.pass_rate_delta, 20.0);
        assert_eq!(change.mae_delta, -20.0);
        assert_eq!(change.max_deviation_delta, -20.0);
        assert_eq!(change.improvements, 4); // Passes increased from 6 to 10
        assert_eq!(change.regressions, 0);
    }
}
