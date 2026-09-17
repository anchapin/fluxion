//! `BenchmarkReport` and delta/sensitivity report types — the per-test-case
//! report machinery consumed by `ashrae_140_validator` and the multi-zone
//! validators. Extracted from `src/validation/report.rs` at Issue #3788
//! decomposition time (the parent landed at 4136/4136 lines, see
//! `tests/reference_data/module_size/report_ratchet.json`) so the parent
//! `report/mod.rs` is small enough to satisfy the Issue #3457 module-size
//! ratchet. Public API is preserved unchanged; every
//! `crate::validation::report::X` path continues to work because `mod.rs`
//! re-exports the public symbols defined here.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use plotters::backend::BitMapBackend;
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::*;
use plotters::style::colors::WHITE;

use crate::validation::multi_reference::{MultiReferenceDB, ProgramRange};
use crate::validation::statistical::{StatisticalMetrics, ValidationGroup};

use super::{
    compute_status, BenchmarkData, Case960Report, Case970Report, Interpretation, MetricType,
    MultiZoneSummary, MultiZoneValidationReport, ReportHeader, ValidationResult, ValidationStatus,
};

/// Comprehensive validation report for ASHRAE 140 test cases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Section 8.1 compliance report header
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_header: Option<ReportHeader>,
    /// All validation results
    pub results: Vec<ValidationResult>,
    /// Benchmark data for each case
    pub benchmark_data: HashMap<String, BenchmarkData>,
    /// Interpretation guidance for failed metrics (not serialized for backwards compatibility)
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub interpretations: HashMap<String, Interpretation>,
    /// Start time for performance measurement (not serialized)
    #[serde(skip)]
    pub start_time: Option<Instant>,
    /// End time for performance measurement (not serialized)
    #[serde(skip)]
    pub end_time: Option<Instant>,
    /// Statistical metrics for the report (NMBE, CV(RMSE), etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_metrics: Option<StatisticalMetrics>,
    /// Per-case p-values from statistical tests
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_p_values: Option<Vec<f64>>,
    /// BH-corrected status for each test (true = rejected null hypothesis)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_corrected: Option<Vec<bool>>,
    /// Group-level validation results (PASS/FAIL per validation group)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_validation: Option<HashMap<ValidationGroup, bool>>,
}

/// Delta test result with statistical significance testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaTestResult {
    /// Metric name
    pub metric_name: String,
    /// Delta value (mean difference)
    pub delta_value: f64,
    /// P-value for statistical significance
    pub p_value: Option<f64>,
    /// Whether the difference is statistically significant
    pub is_significant: bool,
    /// Confidence interval for the delta
    pub confidence_interval: Option<(f64, f64)>,
}

/// Sensitivity analysis result with normalized metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityResult {
    /// Parameter name
    pub parameter_name: String,
    /// Raw coefficient
    pub coefficient: f64,
    /// Normalized coefficient (divided by parameter range)
    pub normalized_coefficient: f64,
    /// Parameter ranking (1 = most sensitive)
    pub ranking: usize,
}

impl BenchmarkReport {
    /// Creates a new empty validation report.
    pub fn new() -> Self {
        Self {
            report_header: None,
            results: Vec::new(),
            benchmark_data: HashMap::new(),
            interpretations: HashMap::new(),
            start_time: None,
            end_time: None,
            statistical_metrics: None,
            statistical_p_values: None,
            statistical_corrected: None,
            group_validation: None,
        }
    }

    /// Generates a JSON report.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Adds a validation result to the report.
    pub fn add_result(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    /// Adds a result using the simplified interface.
    pub fn add_result_simple(
        &mut self,
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) {
        let result = ValidationResult::new(case_id, metric, fluxion_value, ref_min, ref_max);
        self.add_result(result);
    }

    /// Adds a result with an optional peak timestamp (month, day, hour).
    ///
    /// Issue #761: ASHRAE 140-2023 Section 8.2.2 requires tracking peak load timestamps.
    pub fn add_result_with_peak_timestamp(
        &mut self,
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
        peak_timestamp: Option<(u32, u32, u32)>,
    ) {
        let mut result = ValidationResult::new(case_id, metric, fluxion_value, ref_min, ref_max);
        result.peak_timestamp = peak_timestamp;
        self.add_result(result);
    }

    /// Adds a validation result using multi-reference data, populating per-program statuses.
    ///
    /// This method looks up per-program reference ranges from the provided MultiReferenceDB,
    /// computes individual program validation statuses, determines the overall status using
    /// the rule: PASS if EnergyPlus passes, else WARN if any program passes, else FAIL.
    ///
    /// The aggregated ref_min and ref_max are computed as the envelope of all programs.
    pub fn add_result_with_multi(
        &mut self,
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        db: &MultiReferenceDB,
    ) {
        // Look up case references
        let case_refs = match db.cases.get(case_id) {
            Some(c) => c,
            None => {
                // Case not found in multi-ref DB - do NOT push a result.
                // Caller (enrich_with_multi_reference) will preserve the original result.
                return;
            }
        };

        // Get the program ranges for this metric
        let program_ranges: std::collections::HashMap<String, ProgramRange> = match metric {
            MetricType::AnnualHeating => case_refs
                .annual_heating
                .as_ref()
                .cloned()
                .unwrap_or_default(),
            MetricType::AnnualCooling => case_refs
                .annual_cooling
                .as_ref()
                .cloned()
                .unwrap_or_default(),
            MetricType::PeakHeating => case_refs.peak_heating.as_ref().cloned().unwrap_or_default(),
            MetricType::PeakCooling => case_refs.peak_cooling.as_ref().cloned().unwrap_or_default(),
            _ => {
                // For free-floating metrics, multi-reference may not be defined; fall back to no per_program
                let result = ValidationResult::new(case_id, metric, fluxion_value, 0.0, 0.0);
                self.results.push(result);
                return;
            }
        };

        // If no program ranges available, return without pushing a result.
        // Caller will preserve the original result.
        if program_ranges.is_empty() {
            return;
        }

        // Compute aggregated ref_min and ref_max as envelope of all programs
        let agg_min = program_ranges
            .values()
            .map(|r| r.min)
            .fold(f64::INFINITY, f64::min);
        let agg_max = program_ranges
            .values()
            .map(|r| r.max)
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute percent error based on aggregated midpoint
        let agg_mid = (agg_min + agg_max) / 2.0;
        let percent_error = if agg_mid != 0.0 {
            ((fluxion_value - agg_mid) / agg_mid.abs()) * 100.0
        } else {
            0.0
        };

        // Compute per-program statuses
        let mut per_program = std::collections::HashMap::new();
        for (prog_name, range) in program_ranges {
            let status = compute_status(fluxion_value, range.min, range.max);
            per_program.insert(prog_name.clone(), status);
        }

        // Determine overall status based on EnergyPlus primary, then any pass
        let overall_status = if let Some(ep_status) = per_program.get("EnergyPlus") {
            if *ep_status == ValidationStatus::Pass {
                ValidationStatus::Pass
            } else if per_program.values().any(|s| *s == ValidationStatus::Pass) {
                ValidationStatus::Warning
            } else {
                ValidationStatus::Fail
            }
        } else {
            // EnergyPlus not in the list; use aggregated envelope status
            compute_status(fluxion_value, agg_min, agg_max)
        };

        let result = ValidationResult {
            case_id: case_id.to_string(),
            metric,
            fluxion_value,
            ref_min: agg_min,
            ref_max: agg_max,
            percent_error,
            status: overall_status,
            per_program: Some(per_program),
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        };
        self.add_result(result);
    }

    /// Adds benchmark data for a case.
    pub fn add_benchmark_data(&mut self, case_id: &str, data: BenchmarkData) {
        self.benchmark_data.insert(case_id.to_string(), data);
    }

    /// Enriches existing validation results with multi-reference per-program statuses.
    ///
    /// This method processes all results currently in the report. For each result with a metric
    /// that has multi-reference data (AnnualHeating, AnnualCooling, PeakHeating, PeakCooling),
    /// it adds per-program PASS/WARN/FAIL statuses by looking up the reference ranges in the
    /// provided MultiReferenceDB. Results for metrics without multi-reference data (e.g., free-floating
    /// temperatures) or for cases not found in the database are left unchanged.
    ///
    /// The overall status for enriched results is determined by:
    /// - PASS if EnergyPlus passes
    /// - WARN if EnergyPlus fails but any other program passes
    /// - FAIL if all programs fail
    ///
    /// The aggregated ref_min and ref_max are computed as the envelope (min of mins, max of maxes)
    /// across all reference programs.
    pub fn enrich_with_multi_reference(&mut self, db: &MultiReferenceDB) {
        let mut enriched = Vec::new();

        for result in &self.results {
            // Determine if this metric can be enriched with multi-reference data
            let can_enrich = match result.metric {
                MetricType::AnnualHeating
                | MetricType::AnnualCooling
                | MetricType::PeakHeating
                | MetricType::PeakCooling => true,
                _ => false,
            };

            if can_enrich {
                // Use the add_result_with_multi method to create an enriched version
                // We create a temporary BenchmarkReport to reuse the logic
                let mut temp_report = BenchmarkReport::new();
                temp_report.add_result_with_multi(
                    &result.case_id,
                    result.metric.clone(),
                    result.fluxion_value,
                    db,
                );
                if let Some(enriched_result) = temp_report.results.into_iter().next() {
                    enriched.push(enriched_result);
                } else {
                    // Shouldn't happen, but preserve original if it does
                    enriched.push(result.clone());
                }
            } else {
                // Metrics without multi-reference (free-floating temps, etc.) stay unchanged
                enriched.push(result.clone());
            }
        }

        self.results = enriched;
    }

    /// Calculates delta analysis: difference between cases vs baseline.
    pub fn delta_analysis(&self, baseline_case: &str) -> HashMap<String, f64> {
        let mut deltas = HashMap::new();
        let baseline_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.case_id == baseline_case)
            .collect();

        for result in &self.results {
            if result.case_id != baseline_case {
                // Find matching metric in baseline
                if let Some(baseline) = baseline_results.iter().find(|b| result.metric == b.metric)
                {
                    let delta_mwh = result.fluxion_value - baseline.fluxion_value;
                    let delta_output = result.metric.to_output_units(delta_mwh);
                    let key = format!("{} - {}", result.case_id, result.metric.display_name());
                    deltas.insert(key, delta_output);
                }
            }
        }

        deltas
    }

    /// Calculates overall pass rate as a percentage.
    pub fn pass_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 100.0;
        }

        let passed = self.results.iter().filter(|r| r.passed()).count();
        (passed as f64 / self.results.len() as f64) * 100.0
    }

    /// Calculates the number of failed results.
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| r.failed()).count()
    }

    /// Calculates the number of warnings.
    pub fn warning_count(&self) -> usize {
        self.results.iter().filter(|r| r.warning()).count()
    }

    /// Calculates the Mean Absolute Error (MAE) across all results.
    pub fn mae(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let total_error: f64 = self.results.iter().map(|r| r.percent_error.abs()).sum();
        total_error / self.results.len() as f64
    }

    /// Calculates per-case MAE contribution (Issue #3171).
    ///
    /// Returns a HashMap mapping case_id to its MAE contribution, where contribution
    /// is the sum of absolute percent errors for that case divided by the total
    /// number of results. This enables contributors to identify which cases
    /// drive the overall MAE in CI output.
    pub fn case_mae_contributions(&self) -> HashMap<String, f64> {
        if self.results.is_empty() {
            return HashMap::new();
        }

        let total_count = self.results.len() as f64;
        let mut case_errors: HashMap<String, f64> = HashMap::new();

        for result in &self.results {
            let entry = case_errors.entry(result.case_id.clone()).or_insert(0.0);
            *entry += result.percent_error.abs();
        }

        for value in case_errors.values_mut() {
            *value /= total_count;
        }

        case_errors
    }

    /// Calculates the maximum deviation percentage.
    pub fn max_deviation(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.percent_error.abs())
            .fold(0.0f64, |a, b| a.max(b))
    }

    /// Returns cases with the worst performance (highest deviation).
    pub fn worst_cases(&self, top_n: usize) -> Vec<ValidationResult> {
        let mut sorted = self.results.clone();
        sorted.sort_by(|a, b| {
            b.percent_error
                .abs()
                .partial_cmp(&a.percent_error.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(top_n).collect()
    }

    /// Sets the start time for performance measurement.
    pub fn set_start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Sets the end time for performance measurement.
    pub fn set_end(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Returns the duration of the validation in seconds.
    pub fn duration_seconds(&self) -> f64 {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end.duration_since(start).as_secs_f64(),
            _ => 0.0,
        }
    }

    /// Returns the number of cases validated per second.
    pub fn cases_per_second(&self) -> f64 {
        let duration = self.duration_seconds();
        if duration > 0.0 {
            self.benchmark_data.len() as f64 / duration
        } else {
            0.0
        }
    }

    /// Generates a Markdown report.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Section 8.1 Compliance Header
        if let Some(ref header) = self.report_header {
            output.push_str("# ASHRAE 140 Validation Report\n\n");
            output.push_str("## Section 8.1 Compliance Information\n\n");
            output.push_str("| Field | Value |\n");
            output.push_str("|-------|-------|\n");
            output.push_str(&format!("| Program Name | {} |\n", header.program_name));
            output.push_str(&format!(
                "| Program Version | {} |\n",
                header.program_version
            ));
            output.push_str(&format!("| Developer | {} |\n", header.developer));
            output.push_str(&format!(
                "| Run Date | {} |\n",
                header.run_date.format("%Y-%m-%d %H:%M UTC")
            ));
            output.push_str(&format!("| ASHRAE Edition | {} |\n", header.ashrae_edition));
            output.push_str(&format!("| Weather File | {} |\n", header.weather_file_id));
            output.push_str("\n---\n\n");
        } else {
            // Fallback header when no report_header is set
            output.push_str("# ASHRAE 140 Validation Report\n\n");
        }

        // Summary statistics
        output.push_str("## Summary\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Total Results | {} |\n", self.results.len()));
        output.push_str(&format!("| Pass Rate | {:.1}% |\n", self.pass_rate()));
        output.push_str(&format!(
            "| Passed | {} |\n",
            self.results.iter().filter(|r| r.passed()).count()
        ));
        output.push_str(&format!("| Warnings | {} |\n", self.warning_count()));
        output.push_str(&format!("| Failed | {} |\n", self.fail_count()));
        output.push_str(&format!("| Mean Absolute Error | {:.2}% |\n", self.mae()));
        output.push_str(&format!(
            "| Max Deviation | {:.2}% |\n",
            self.max_deviation()
        ));
        output.push('\n');

        // Detailed results table
        output.push_str("## Detailed Results\n\n");
        output.push_str("| Case | Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |\n");
        output.push_str("|------|--------|---------|---------|---------|-----------|--------|\n");

        for result in &self.results {
            let fluxion_kwh = result.metric.to_output_units(result.fluxion_value);
            let ref_min_kwh = result.metric.to_output_units(result.ref_min);
            let ref_max_kwh = result.metric.to_output_units(result.ref_max);
            output.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {} | {} |\n",
                result.case_id,
                result.metric,
                fluxion_kwh,
                ref_min_kwh,
                ref_max_kwh,
                result.deviation_string(),
                result.status
            ));
        }

        output.push('\n');

        // Delta analysis
        if !self.benchmark_data.is_empty() {
            let baseline = self.benchmark_data.keys().next().unwrap();
            let deltas = self.delta_analysis(baseline);

            if !deltas.is_empty() {
                output.push_str("## Delta Analysis\n\n");
                output.push_str(&format!("Baseline: {}\n\n", baseline));
                output.push_str("| Case - Metric | Delta from Baseline |\n");
                output.push_str("|---------------|---------------------|\n");

                for (key, delta) in &deltas {
                    output.push_str(&format!("| {} | {:+.2} |\n", key, delta));
                }

                output.push('\n');
            }
        }

        // Worst cases
        let worst = self.worst_cases(5);
        if !worst.is_empty() {
            output.push_str("## Worst Performing Cases\n\n");
            output.push_str("| Case | Metric | Deviation | Status |\n");
            output.push_str("|------|--------|-----------|--------|\n");

            for result in worst {
                output.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    result.case_id,
                    result.metric,
                    result.deviation_string(),
                    result.status
                ));
            }

            output.push('\n');
        }

        // Interpretation guidance for failed cases
        if !self.interpretations.is_empty() {
            output.push_str("## Interpretation Guidance\n\n");
            output.push_str("The following interpretation guidance is provided for cases with failed metrics:\n\n");

            let mut case_ids: Vec<_> = self.interpretations.keys().collect();
            case_ids.sort();

            for case_id in case_ids {
                if let Some(interp) = self.interpretations.get(case_id) {
                    output.push_str(&format!("### Case {}\n\n", case_id));

                    // Root cause hypotheses
                    if !interp.root_cause_hypotheses.is_empty() {
                        output.push_str("**Root Cause Hypothesis:**\n");
                        for hypothesis in &interp.root_cause_hypotheses {
                            output.push_str(&format!("- {}\n", hypothesis));
                        }
                        output.push('\n');
                    }

                    // Parameter sensitivity
                    if !interp.parameter_sensitivity.is_empty() {
                        output.push_str("**Parameter Sensitivity:**\n");
                        for sensitivity in &interp.parameter_sensitivity {
                            output.push_str(&format!("- {}\n", sensitivity));
                        }
                        output.push('\n');
                    }

                    // Recommended next steps
                    if !interp.recommended_next_steps.is_empty() {
                        output.push_str("**Recommended Next Steps:**\n");
                        for step in &interp.recommended_next_steps {
                            output.push_str(&format!("- {}\n", step));
                        }
                        output.push('\n');
                    }

                    // What-if scenarios
                    if !interp.what_if_scenarios.is_empty() {
                        output.push_str("**What-if Scenarios:**\n");
                        for scenario in &interp.what_if_scenarios {
                            output.push_str(&format!("- {}\n", scenario));
                        }
                        output.push('\n');
                    }

                    // References
                    if !interp.references.is_empty() {
                        output.push_str("**References:**\n");
                        for ref_doc in &interp.references {
                            output.push_str(&format!("- {}\n", ref_doc));
                        }
                        output.push('\n');
                    }
                }
            }
        }

        // Legend
        output.push_str("## Legend\n\n");
        output.push_str("- **PASS**: Value within 5% of reference range\n");
        output.push_str("- **WARN**: Value within reference range but >2% deviation\n");
        output.push_str("- **FAIL**: Value outside 5% tolerance band\n");

        output
    }

    /// Generates an HTML report.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html>\n");
        html.push_str("<head>\n");
        html.push_str("  <title>ASHRAE 140 Validation Report</title>\n");
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: Arial, sans-serif; margin: 40px; }\n");
        html.push_str("    h1 { color: #333; }\n");
        html.push_str("    h2 { color: #666; border-bottom: 1px solid #ddd; }\n");
        html.push_str(
            "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n",
        );
        html.push_str("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("    th { background-color: #f2f2f2; }\n");
        html.push_str("    tr:nth-child(even) { background-color: #f9f9f9; }\n");
        html.push_str("    .pass { color: green; font-weight: bold; }\n");
        html.push_str("    .warning { color: orange; font-weight: bold; }\n");
        html.push_str("    .fail { color: red; font-weight: bold; }\n");
        html.push_str("    .positive { color: green; }\n");
        html.push_str("    .negative { color: red; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        html.push_str("  <h1>ASHRAE 140 Validation Report</h1>\n");

        // Summary statistics
        html.push_str("  <h2>Summary</h2>\n");
        html.push_str("  <table>\n");
        html.push_str("    <tr><th>Metric</th><th>Value</th></tr>\n");
        html.push_str(&format!(
            "    <tr><td>Total Results</td><td>{}</td></tr>\n",
            self.results.len()
        ));
        html.push_str(&format!(
            "    <tr><td>Pass Rate</td><td>{:.1}%</td></tr>\n",
            self.pass_rate()
        ));
        html.push_str(&format!(
            "    <tr><td>Passed</td><td>{}</td></tr>\n",
            self.results.iter().filter(|r| r.passed()).count()
        ));
        html.push_str(&format!(
            "    <tr><td>Warnings</td><td>{}</td></tr>\n",
            self.warning_count()
        ));
        html.push_str(&format!(
            "    <tr><td>Failed</td><td>{}</td></tr>\n",
            self.fail_count()
        ));
        html.push_str(&format!(
            "    <tr><td>Mean Absolute Error</td><td>{:.2}%</td></tr>\n",
            self.mae()
        ));
        html.push_str(&format!(
            "    <tr><td>Max Deviation</td><td>{:.2}%</td></tr>\n",
            self.max_deviation()
        ));
        html.push_str("  </table>\n");

        // Detailed results table
        html.push_str("  <h2>Detailed Results</h2>\n");
        html.push_str("  <table>\n");
        html.push_str("    <tr><th>Case</th><th>Metric</th><th>Fluxion</th><th>Ref Min</th><th>Ref Max</th><th>Deviation</th><th>Status</th></tr>\n");

        for result in &self.results {
            let status_class = match result.status {
                ValidationStatus::Pass => "pass",
                ValidationStatus::Warning => "warning",
                ValidationStatus::Fail => "fail",
            };

            let deviation_class = if result.percent_error > 0.0 {
                "positive"
            } else {
                "negative"
            };

            let fluxion_kwh = result.metric.to_output_units(result.fluxion_value);
            let ref_min_kwh = result.metric.to_output_units(result.ref_min);
            let ref_max_kwh = result.metric.to_output_units(result.ref_max);

            html.push_str("    <tr>\n");
            html.push_str(&format!("      <td>{}</td>\n", result.case_id));
            html.push_str(&format!("      <td>{}</td>\n", result.metric));
            html.push_str(&format!("      <td>{:.2}</td>\n", fluxion_kwh));
            html.push_str(&format!("      <td>{:.2}</td>\n", ref_min_kwh));
            html.push_str(&format!("      <td>{:.2}</td>\n", ref_max_kwh));
            html.push_str(&format!(
                "      <td class=\"{}\">{}</td>\n",
                deviation_class,
                result.deviation_string()
            ));
            html.push_str(&format!(
                "      <td class=\"{}\">{}</td>\n",
                status_class, result.status
            ));
            html.push_str("    </tr>\n");
        }

        html.push_str("  </table>\n");

        // Delta analysis
        if !self.benchmark_data.is_empty() {
            let baseline = self.benchmark_data.keys().next().unwrap();
            let deltas = self.delta_analysis(baseline);

            if !deltas.is_empty() {
                html.push_str("  <h2>Delta Analysis</h2>\n");
                html.push_str(&format!(
                    "  <p><strong>Baseline:</strong> {}</p>\n",
                    baseline
                ));
                html.push_str("  <table>\n");
                html.push_str("    <tr><th>Case - Metric</th><th>Delta from Baseline</th></tr>\n");

                for (key, delta) in &deltas {
                    let delta_class = if *delta > 0.0 { "positive" } else { "negative" };
                    html.push_str(&format!(
                        "    <tr><td>{}</td><td class=\"{}\">{:+.2}</td></tr>\n",
                        key, delta_class, delta
                    ));
                }

                html.push_str("  </table>\n");
            }
        }

        // Worst cases
        let worst = self.worst_cases(5);
        if !worst.is_empty() {
            html.push_str("  <h2>Worst Performing Cases</h2>\n");
            html.push_str("  <table>\n");
            html.push_str(
                "    <tr><th>Case</th><th>Metric</th><th>Deviation</th><th>Status</th></tr>\n",
            );

            for result in worst {
                let status_class = match result.status {
                    ValidationStatus::Pass => "pass",
                    ValidationStatus::Warning => "warning",
                    ValidationStatus::Fail => "fail",
                };

                html.push_str(&format!(
                    "    <tr><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td></tr>\n",
                    result.case_id,
                    result.metric,
                    result.deviation_string(),
                    status_class,
                    result.status
                ));
            }

            html.push_str("  </table>\n");
        }

        // Legend
        html.push_str("  <h2>Legend</h2>\n");
        html.push_str("  <ul>\n");
        html.push_str("    <li><strong>PASS</strong>: Value within 5% of reference range</li>\n");
        html.push_str(
            "    <li><strong>WARN</strong>: Value within reference range but >2% deviation</li>\n",
        );
        html.push_str("    <li><strong>FAIL</strong>: Value outside 5% tolerance band</li>\n");
        html.push_str("  </ul>\n");

        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }

    /// Generates a CSV report.
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Case,Metric,Fluxion,Ref Min,Ref Max,Percent Error,Status\n");

        // Data rows
        for result in &self.results {
            let fluxion_kwh = result.metric.to_output_units(result.fluxion_value);
            let ref_min_kwh = result.metric.to_output_units(result.ref_min);
            let ref_max_kwh = result.metric.to_output_units(result.ref_max);
            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.2},{}\n",
                result.case_id,
                result.metric,
                fluxion_kwh,
                ref_min_kwh,
                ref_max_kwh,
                result.percent_error,
                result.status
            ));
        }

        csv
    }

    /// Saves the report to a file based on the extension.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        let content = match path.extension().and_then(|e| e.to_str()) {
            Some("md") => self.to_markdown(),
            Some("html") => self.to_html(),
            Some("htm") => self.to_html(),
            Some("csv") => self.to_csv(),
            Some("txt") => self.to_markdown(),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unsupported file extension. Use .md, .html, or .csv",
                ))
            }
        };

        std::fs::write(path, content)
    }

    /// Emits a summary via `tracing` in human-readable structured form. (Issue #2500)
    pub fn print_summary(&self) {
        tracing::info!(
            total_results = self.results.len(),
            pass_rate_pct = self.pass_rate(),
            passed = self.results.iter().filter(|r| r.passed()).count(),
            warnings = self.warning_count(),
            failed = self.fail_count(),
            mae_pct = self.mae(),
            max_deviation_pct = self.max_deviation(),
            "validation report summary",
        );

        let case_mae = self.case_mae_contributions();
        let mut sorted_cases: Vec<_> = case_mae.iter().collect();
        sorted_cases.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (case_id, contribution) in sorted_cases {
            tracing::info!(
                case_id = %case_id,
                mae_contribution_pct = *contribution,
                "per_case_mae_contribution",
            );
        }
    }

    /// Emits a machine-readable summary via `tracing` for CI ingestion. (Issue #2500)
    ///
    /// The fields are recorded as structured tracing fields (and the full JSON
    /// blob is attached) so that a JSON subscriber produces records ingestible
    /// by CI scripts / Loki / Elastic without fragile regex parsing.
    pub fn print_summary_json(&self) {
        use serde_json::json;

        let passed_count = self.results.iter().filter(|r| r.passed()).count() as u32;
        let warning_count = self.warning_count() as u32;
        let fail_count = self.fail_count() as u32;

        let case_mae = self.case_mae_contributions();
        let case_mae_sorted: Vec<_> = {
            let mut v: Vec<_> = case_mae.iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            v.into_iter().map(|(k, v)| (k.clone(), *v)).collect()
        };

        let summary = json!({
            "total_results": self.results.len() as u32,
            "pass_rate": self.pass_rate(),
            "passed": passed_count,
            "warnings": warning_count,
            "failed": fail_count,
            "mae": self.mae(),
            "max_deviation": self.max_deviation(),
            "duration_seconds": self.duration_seconds(),
            "case_mae_contributions": case_mae_sorted,
        });

        tracing::info!(
            total_results = self.results.len() as u32,
            pass_rate = self.pass_rate(),
            passed = passed_count,
            warnings = warning_count,
            failed = fail_count,
            mae = self.mae(),
            max_deviation = self.max_deviation(),
            duration_seconds = self.duration_seconds(),
            json = %serde_json::to_string(&summary).unwrap(),
            "validation report summary (json)",
        );
    }

    /// Appends the report's metrics to the historical performance log.
    ///
    /// This method serializes key metrics (timestamp, MAE, max deviation, pass rate,
    /// duration, throughput) to `target/performance_history.jsonl` as a JSON line.
    /// It also attempts to include the git SHA if available from environment variables.
    /// I/O errors are handled gracefully with a warning printed to stderr.
    pub fn append_history(&self) {
        use std::fs::OpenOptions;
        use std::io::Write;

        // Collect metrics
        let timestamp = Utc::now().to_rfc3339();
        let mae = self.mae();
        let max_deviation = self.max_deviation();
        let pass_rate = self.pass_rate();
        let validation_time_seconds = self.duration_seconds();
        let throughput = self.cases_per_second();

        // Get git SHA from common CI environment variables
        let git_sha = env::var("GIT_SHA")
            .or_else(|_| env::var("GITHUB_SHA"))
            .or_else(|_| env::var("CI_COMMIT_SHA"))
            .ok()
            .map(String::from);

        // Construct JSON object
        #[derive(serde::Serialize)]
        struct HistoryEntry {
            timestamp: String,
            mae: f64,
            max_deviation: f64,
            pass_rate: f64,
            validation_time_seconds: f64,
            throughput: f64,
            git_sha: Option<String>,
        }

        let entry = HistoryEntry {
            timestamp,
            mae,
            max_deviation,
            pass_rate,
            validation_time_seconds,
            throughput,
            git_sha,
        };

        // Determine file path (target/performance_history.jsonl)
        let file_path = Path::new("target").join("performance_history.jsonl");

        // Create target directory if it doesn't exist
        if let Some(parent) = file_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                tracing::warn!(error = %e, "failed to create target directory");
                return;
            }
        }

        // Append JSON line to file
        let json_line = match serde_json::to_string(&entry) {
            Ok(line) => line,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize history entry");
                return;
            }
        };

        let mut file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
        {
            Ok(file) => file,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to open performance history file for appending",
                );
                return;
            }
        };

        if let Err(e) = writeln!(file, "{}", json_line) {
            tracing::warn!(error = %e, "failed to write to performance history");
        }
    }

    /// Generates a multi-zone validation report
    pub fn generate_multi_zone_markdown_report(&self) -> String {
        let mut output = String::new();

        output.push_str("# ASHRAE 140 Multi-Zone Validation Report\n\n");

        // Summary statistics
        output.push_str("## Summary\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Total Results | {} |\n", self.results.len()));
        output.push_str(&format!("| Pass Rate | {:.1}% |\n", self.pass_rate()));
        output.push_str(&format!(
            "| Passed | {} |\n",
            self.results.iter().filter(|r| r.passed()).count()
        ));
        output.push_str(&format!("| Warnings | {} |\n", self.warning_count()));
        output.push_str(&format!("| Failed | {} |\n", self.fail_count()));
        output.push_str(&format!("| Mean Absolute Error | {:.2}% |\n", self.mae()));
        output.push_str(&format!(
            "| Max Deviation | {:.2}% |\n",
            self.max_deviation()
        ));
        output.push('\n');

        // Multi-zone specific sections
        output.push_str("## Case 960 Results\n\n");
        output.push_str("### Two-Zone Sunspace Building Validation\n\n");

        let case_960_results: Vec<_> = self.results.iter().filter(|r| r.case_id == "960").collect();

        if !case_960_results.is_empty() {
            output.push_str("| Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |\n");
            output.push_str("|--------|---------|---------|---------|-----------|--------|\n");

            for result in case_960_results {
                let fluxion_kwh = result.metric.to_output_units(result.fluxion_value);
                let ref_min_kwh = result.metric.to_output_units(result.ref_min);
                let ref_max_kwh = result.metric.to_output_units(result.ref_max);
                output.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {:.2} | {} | {} |\n",
                    result.metric,
                    fluxion_kwh,
                    ref_min_kwh,
                    ref_max_kwh,
                    result.deviation_string(),
                    result.status
                ));
            }
            output.push('\n');
        }

        // Case 970 results
        output.push_str("## Case 970 Results\n\n");
        output.push_str("### Multi-Zone Building Framework Validation\n\n");

        let case_970_results: Vec<_> = self.results.iter().filter(|r| r.case_id == "970").collect();

        if !case_970_results.is_empty() {
            output.push_str("| Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |\n");
            output.push_str("|--------|---------|---------|---------|-----------|--------|\n");

            for result in case_970_results {
                let fluxion_kwh = result.metric.to_output_units(result.fluxion_value);
                let ref_min_kwh = result.metric.to_output_units(result.ref_min);
                let ref_max_kwh = result.metric.to_output_units(result.ref_max);
                output.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {:.2} | {} | {} |\n",
                    result.metric,
                    fluxion_kwh,
                    ref_min_kwh,
                    ref_max_kwh,
                    result.deviation_string(),
                    result.status
                ));
            }
            output.push('\n');
        }

        // Comparison table
        output.push_str("## Multi-Zone Comparison\n\n");
        output.push_str("### Cross-Case Analysis\n\n");

        let case_ids: Vec<_> = self
            .results
            .iter()
            .map(|r| r.case_id.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if case_ids.len() > 1 {
            output.push_str("| Case | Pass Rate | Max Deviation |\n");
            output.push_str("|------|-----------|---------------|\n");

            for case_id in case_ids {
                let case_results: Vec<_> = self
                    .results
                    .iter()
                    .filter(|r| r.case_id == case_id)
                    .collect();
                if !case_results.is_empty() {
                    let case_pass_rate = case_results.iter().filter(|r| r.passed()).count() as f64
                        / case_results.len() as f64
                        * 100.0;
                    let case_max_dev = case_results
                        .iter()
                        .map(|r| r.percent_error.abs())
                        .fold(0.0f64, f64::max);

                    output.push_str(&format!(
                        "| {} | {:.1}% | {:.1}% |\n",
                        case_id, case_pass_rate, case_max_dev
                    ));
                }
            }
            output.push('\n');
        }

        // Visualization section
        output.push_str("## Visualizations\n\n");
        output.push_str("### Temperature Profile Comparison\n\n");
        output.push_str("![Temperature Profile Plot](temperature_profile.png)\n\n");

        output.push_str("### Energy Consumption Comparison\n\n");
        output.push_str("![Energy Comparison Chart](energy_comparison.png)\n\n");

        output.push_str("### Inter-Zone Heat Transfer\n\n");
        output.push_str("![Heat Transfer Visualization](heat_transfer.png)\n\n");

        // Legend
        output.push_str("## Legend\n\n");
        output.push_str("- **PASS**: Value within tolerance band\n");
        output.push_str("- **WARN**: Value within reference range but >10% deviation\n");
        output.push_str("- **FAIL**: Value outside tolerance band\n");

        output
    }

    /// Generates a multi-zone CSV report
    pub fn generate_multi_zone_csv_report(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Case,Metric,Fluxion,Ref Min,Ref Max,Percent Error,Status,Case Type\n");

        // Data rows
        for result in &self.results {
            let case_type = if result.case_id == "960" {
                "Two-Zone Sunspace"
            } else if result.case_id == "970" {
                "Multi-Zone Building"
            } else {
                "Other"
            };

            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.2},{},{}\n",
                result.case_id,
                result.metric,
                result.metric.to_output_units(result.fluxion_value),
                result.metric.to_output_units(result.ref_min),
                result.metric.to_output_units(result.ref_max),
                result.percent_error,
                result.status,
                case_type
            ));
        }

        csv
    }

    /// Generates a multi-zone JSON report
    pub fn generate_multi_zone_json_report(&self) -> String {
        let multi_zone_report = MultiZoneValidationReport {
            case_960_report: self.extract_case_960_report(),
            case_970_report: self.extract_case_970_report(),
            case_980_report: Case960Report::default(),
            summary: self.generate_multi_zone_summary(),
        };

        serde_json::to_string_pretty(&multi_zone_report).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generates a comparison table for multi-zone validation
    pub fn generate_comparison_table(&self) -> String {
        let mut table = String::new();

        table.push_str("| Metric | Case 960 | Case 970 | Difference |\n");
        table.push_str("|--------|---------|---------|------------|\n");

        // Find common metrics between Case 960 and Case 970
        let case_960_metrics: std::collections::HashSet<_> = self
            .results
            .iter()
            .filter(|r| r.case_id == "960")
            .map(|r| r.metric.clone())
            .collect();

        let case_970_metrics: std::collections::HashSet<_> = self
            .results
            .iter()
            .filter(|r| r.case_id == "970")
            .map(|r| r.metric.clone())
            .collect();

        let common_metrics: Vec<_> = case_960_metrics.intersection(&case_970_metrics).collect();

        for metric in common_metrics {
            let case_960_result = self
                .results
                .iter()
                .find(|r| r.case_id == "960" && r.metric == *metric);
            let case_970_result = self
                .results
                .iter()
                .find(|r| r.case_id == "970" && r.metric == *metric);

            if let (Some(r960), Some(r970)) = (case_960_result, case_970_result) {
                let fluxion_960_kwh = r960.metric.to_output_units(r960.fluxion_value);
                let fluxion_970_kwh = r970.metric.to_output_units(r970.fluxion_value);
                let difference = fluxion_970_kwh - fluxion_960_kwh;
                table.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {:+.2} |\n",
                    metric, fluxion_960_kwh, fluxion_970_kwh, difference
                ));
            }
        }

        table
    }

    /// Extracts Case 960 specific report
    fn extract_case_960_report(&self) -> Case960Report {
        let default_result = ValidationResult {
            case_id: "960".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 0.0,
            ref_min: 0.0,
            ref_max: 0.0,
            percent_error: 0.0,
            status: ValidationStatus::Fail,
            per_program: None,
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        };

        Case960Report {
            annual_heating: self
                .results
                .iter()
                .find(|r| r.case_id == "960" && matches!(r.metric, MetricType::AnnualHeating))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            annual_cooling: self
                .results
                .iter()
                .find(|r| r.case_id == "960" && matches!(r.metric, MetricType::AnnualCooling))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            peak_heating: self
                .results
                .iter()
                .find(|r| r.case_id == "960" && matches!(r.metric, MetricType::PeakHeating))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            peak_cooling: self
                .results
                .iter()
                .find(|r| r.case_id == "960" && matches!(r.metric, MetricType::PeakCooling))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            temperature_profile: default_result.clone(),
            inter_zone_heat_transfer: default_result,
        }
    }

    /// Extracts Case 970 specific report
    fn extract_case_970_report(&self) -> Case970Report {
        let default_result = ValidationResult {
            case_id: "970".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 0.0,
            ref_min: 0.0,
            ref_max: 0.0,
            percent_error: 0.0,
            status: ValidationStatus::Fail,
            per_program: None,
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        };

        Case970Report {
            annual_heating: self
                .results
                .iter()
                .find(|r| r.case_id == "970" && matches!(r.metric, MetricType::AnnualHeating))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            annual_cooling: self
                .results
                .iter()
                .find(|r| r.case_id == "970" && matches!(r.metric, MetricType::AnnualCooling))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            peak_heating: self
                .results
                .iter()
                .find(|r| r.case_id == "970" && matches!(r.metric, MetricType::PeakHeating))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            peak_cooling: self
                .results
                .iter()
                .find(|r| r.case_id == "970" && matches!(r.metric, MetricType::PeakCooling))
                .cloned()
                .unwrap_or_else(|| default_result.clone()),
            multi_zone_coupling: default_result,
        }
    }

    /// Generates multi-zone summary statistics
    fn generate_multi_zone_summary(&self) -> MultiZoneSummary {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed()).count();
        let warning_tests = self.results.iter().filter(|r| r.warning()).count();
        let failed_tests = self.results.iter().filter(|r| r.failed()).count();
        let pass_rate = self.pass_rate();
        let mean_absolute_error = self.mae();
        let max_deviation = self.max_deviation();

        let overall_status = if failed_tests == 0 {
            ValidationStatus::Pass
        } else if passed_tests > 0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Fail
        };

        MultiZoneSummary {
            total_tests,
            passed_tests,
            warning_tests,
            failed_tests,
            pass_rate,
            mean_absolute_error,
            max_deviation,
            overall_status,
        }
    }

    /// Generates temperature profile plot (placeholder implementation)
    pub fn generate_temperature_profile_plot(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Placeholder implementation - in a real implementation, this would use plotters
        // to generate actual temperature profile charts
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;

        file.write_all(
            b"Temperature profile visualization would be generated here in a full implementation\n",
        )?;
        file.write_all(b"This placeholder represents the chart generation capability\n")?;

        Ok(())
    }

    /// Generates energy comparison chart
    pub fn generate_energy_comparison_chart(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "ASHRAE 140 Multi-Zone Energy Comparison",
                ("sans-serif", 50).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..2, 0f64..20f64)?;

        chart.configure_mesh().draw()?;

        // This would be populated with actual energy data in a real implementation
        let case_960_heating = self
            .results
            .iter()
            .find(|r| r.case_id == "960" && matches!(r.metric, MetricType::AnnualHeating))
            .map(|r| r.fluxion_value)
            .unwrap_or(0.0);

        let case_970_heating = self
            .results
            .iter()
            .find(|r| r.case_id == "970" && matches!(r.metric, MetricType::AnnualHeating))
            .map(|r| r.fluxion_value)
            .unwrap_or(0.0);

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(RED.filled())
                .data(vec![(0, case_960_heating), (1, case_970_heating)]),
        )?;

        Ok(())
    }

    /// Generates inter-zone heat transfer visualization
    pub fn generate_heat_transfer_visualization(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Inter-Zone Heat Transfer Analysis",
                ("sans-serif", 50).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f64..8760f64, -1000f64..1000f64)?;

        chart.configure_mesh().draw()?;

        // This would be populated with actual heat transfer data in a real implementation
        chart.draw_series(LineSeries::new(vec![(0.0, 0.0), (8760.0, 0.0)], &BLACK))?;

        Ok(())
    }
}

impl BenchmarkReport {
    /// Calculates monthly aggregation from hourly data.
    ///
    /// This function correctly sums hourly values into 12 months using actual hours per month.
    /// Returns a vector of 12 monthly totals (not averages).
    ///
    /// # Arguments
    /// * `hourly_data` - Slice of hourly values (typically 8760 for one year)
    ///
    /// # Returns
    /// Vector of 12 monthly totals
    pub fn calculate_monthly_aggregation(hourly_data: &[f64]) -> Vec<f64> {
        let mut monthly = vec![0.0; 12];

        // Actual hours per month (non-leap year)
        let hours_per_month = [744, 696, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744];

        // Calculate cumulative hour counts for month boundaries
        let mut month_boundaries = vec![0; 13];
        for i in 1..13 {
            month_boundaries[i] = hours_per_month[0..i].iter().sum();
        }

        // Sum hourly values into appropriate month
        for (i, &value) in hourly_data.iter().enumerate() {
            if i < 8760 {
                // Find which month this hour belongs to
                let month_idx = month_boundaries
                    .iter()
                    .position(|&boundary| i < boundary)
                    .unwrap_or(11);

                if month_idx < 12 {
                    monthly[month_idx] += value;
                }
            }
        }

        monthly
    }

    /// Performs delta test with statistical significance testing.
    ///
    /// This function compares two datasets and determines if the difference
    /// is statistically significant using a two-tailed z-test.
    ///
    /// # Arguments
    /// * `baseline` - Baseline dataset
    /// * `current` - Current dataset to compare against baseline
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% confidence)
    ///
    /// # Returns
    /// DeltaTestResult with statistical analysis
    #[allow(clippy::float_cmp)]
    // `confidence_level == 0.95` / `== 0.99` are sentinel switches on
    // a caller-supplied parameter — the comparison is against a literal
    // constant, so it is exact under `fast-math`. See Issue #3357.
    pub fn perform_delta_test(
        baseline: &[f64],
        current: &[f64],
        confidence_level: f64,
    ) -> DeltaTestResult {
        let n = baseline.len();
        let m = current.len();

        // Calculate means
        let mean_baseline: f64 = baseline.iter().sum::<f64>() / n as f64;
        let mean_current: f64 = current.iter().sum::<f64>() / m as f64;

        // Calculate standard deviations
        let std_baseline = Self::calculate_std_deviation(baseline, mean_baseline);
        let std_current = Self::calculate_std_deviation(current, mean_current);

        // Calculate delta (mean difference)
        let delta = mean_current - mean_baseline;

        // Calculate pooled standard error
        let pooled_std_error = if std_baseline > 0.0 || std_current > 0.0 {
            ((std_baseline.powi(2) / n as f64) + (std_current.powi(2) / m as f64)).sqrt()
        } else {
            0.0
        };

        // Calculate z-score (for large samples)
        let z_score = if pooled_std_error > 0.0 {
            delta / pooled_std_error
        } else {
            0.0
        };

        // Calculate p-value (two-tailed test)
        let p_value = if z_score.abs() > 0.0 {
            2.0 * (1.0 - Self::normal_cdf(z_score.abs()))
        } else {
            1.0
        };

        // Determine significance (p < 0.05 for 95% confidence)
        let is_significant = p_value < 0.05;

        // Calculate confidence interval
        let critical_value = if confidence_level == 0.95 {
            1.96 // For 95% confidence
        } else if confidence_level == 0.99 {
            2.58 // For 99% confidence
        } else {
            // Approximate for other levels
            (2.0 * (confidence_level + 0.5)).ln().sqrt()
        };

        let ci_lower = delta - critical_value * pooled_std_error;
        let ci_upper = delta + critical_value * pooled_std_error;

        DeltaTestResult {
            metric_name: "delta".to_string(),
            delta_value: delta,
            p_value: Some(p_value),
            is_significant,
            confidence_interval: Some((ci_lower, ci_upper)),
        }
    }

    /// Calculates standard deviation for a dataset.
    fn calculate_std_deviation(data: &[f64], mean: f64) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let variance: f64 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

        variance.sqrt()
    }

    /// Approximation of standard normal cumulative distribution function (CDF).
    ///
    /// Uses the Abramowitz and Stegun approximation for accuracy.
    fn normal_cdf(x: f64) -> f64 {
        // Approximation constants
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let k = 1.0 / (1.0 + p * x.abs());
        let y = 1.0 - (((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k) * (-0.5 * x * x).exp();

        0.5 * (1.0 + if x < 0.0 { -1.0 } else { 1.0 } * y)
    }

    /// Normalizes sensitivity coefficients by parameter range.
    ///
    /// This function normalizes raw sensitivity coefficients by dividing by
    /// the parameter range, enabling fair comparison of parameter importance.
    ///
    /// # Arguments
    /// * `sensitivity_results` - Raw sensitivity coefficients
    /// * `parameter_ranges` - Parameter ranges as (min, max) tuples
    ///
    /// # Returns
    /// Vector of SensitivityResult with normalized coefficients and rankings
    pub fn normalize_sensitivity(
        sensitivity_results: &[(String, f64)],
        parameter_ranges: &std::collections::HashMap<String, (f64, f64)>,
    ) -> Vec<SensitivityResult> {
        let mut normalized_results = Vec::new();

        for (param_name, coefficient) in sensitivity_results {
            // Get parameter range
            let (min_val, max_val) = parameter_ranges.get(param_name).unwrap_or(&(0.0, 1.0));

            let range = max_val - min_val;

            // Normalize coefficient by dividing by parameter range
            let normalized_coefficient = if range > 0.0 {
                coefficient.abs() / range
            } else {
                coefficient.abs()
            };

            normalized_results.push(SensitivityResult {
                parameter_name: param_name.clone(),
                coefficient: *coefficient,
                normalized_coefficient,
                ranking: 0, // Will be assigned after sorting
            });
        }

        // Sort by normalized coefficient (descending)
        normalized_results.sort_by(|a, b| {
            b.normalized_coefficient
                .partial_cmp(&a.normalized_coefficient)
                .unwrap()
        });

        // Assign rankings
        for (i, result) in normalized_results.iter_mut().enumerate() {
            result.ranking = i + 1;
        }

        normalized_results
    }

    /// Gets standard parameter ranges for Fluxion model.
    ///
    /// Returns parameter ranges for window U-value, HVAC setpoint,
    /// thermal mass, and infiltration rate.
    pub fn get_parameter_ranges() -> std::collections::HashMap<String, (f64, f64)> {
        let mut ranges = std::collections::HashMap::new();

        ranges.insert("window_u_value".to_string(), (0.1, 5.0)); // W/m²K
        ranges.insert("hvac_setpoint".to_string(), (15.0, 30.0)); // °C
        ranges.insert("thermal_mass".to_string(), (1_000_000.0, 50_000_000.0)); // J/K
        ranges.insert("infiltration_rate".to_string(), (0.1, 2.0)); // ACH

        ranges
    }
}

/// Analysis of differences between test cases (e.g., variant vs. baseline).
#[derive(Debug, Clone)]
pub struct DeltaResult {
    /// Case being analyzed (the variant)
    pub case_id: String,
    /// Baseline case for comparison
    pub baseline_id: String,
    /// Metric type
    pub metric: MetricType,
    /// Absolute difference in Fluxion values
    pub fluxion_delta: f64,
    /// Absolute difference in reference midpoint values
    pub reference_delta: f64,
    /// Percent deviation of Fluxion delta from reference delta
    pub deviation_percent: f64,
}

/// Report containing delta analysis for multiple case variants.
#[derive(Debug, Clone, Default)]
pub struct DeltaReport {
    /// All delta analysis results
    pub deltas: Vec<DeltaResult>,
}

impl DeltaReport {
    /// Creates a new empty delta report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a delta analysis result.
    pub fn add_delta(&mut self, delta: DeltaResult) {
        self.deltas.push(delta);
    }

    /// Generates a Markdown table for the delta analysis.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        output.push_str("## Delta Analysis\n\n");
        output.push_str("| Case vs Baseline | Metric | Fluxion Δ | Ref Δ | Deviation |\n");
        output.push_str("|------------------|--------|-----------|-------|-----------|\n");

        for d in &self.deltas {
            output.push_str(&format!(
                "| {} vs {} | {} | {:.2} | {:.2} | {:+.2}% |\n",
                d.case_id,
                d.baseline_id,
                d.metric,
                d.fluxion_delta,
                d.reference_delta,
                d.deviation_percent
            ));
        }

        output
    }
}
