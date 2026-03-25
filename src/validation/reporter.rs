//! Validation report generation for ASHRAE 140.
//!
//! This module provides the `ValidationReportGenerator` which produces
//! comprehensive Markdown reports from validation results.

use crate::validation::report::{BenchmarkReport, MetricType};
use crate::validation::statistical::{
    EffectDirection, StatisticalMetrics, StatisticalReport, ValidationGroup,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Baseline metrics for performance comparison.
#[derive(Debug, Clone, Deserialize)]
pub struct BaselineMetrics {
    pub mae: f64,
    pub max_deviation: f64,
    pub pass_rate: f64,
    pub validation_time_seconds: f64,
}

/// Systematic issue categories for ASHRAE 140 validation failures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SystematicIssue {
    /// Solar gain calculation problems
    SolarGains,
    /// Thermal mass dynamics issues
    ThermalMass,
    /// Inter-zone heat transfer errors
    InterZoneTransfer,
    /// HVAC load calculation errors
    HvacLoad,
    /// Weather data issues
    WeatherData,
    /// 5R1C model limitation (acceptable)
    ModelLimitation,
    /// Unknown or unclassified issue
    Unknown,
}

/// Report generator for ASHRAE 140 validation results.
pub struct ValidationReportGenerator {
    /// Output path for the generated report
    pub output_path: PathBuf,
}

impl ValidationReportGenerator {
    /// Creates a new report generator with the specified output path.
    pub fn new(output_path: PathBuf) -> Self {
        Self { output_path }
    }

    /// Appends a multi-reference comparison table to the markdown output.
    ///
    /// This table shows per-program validation results (EnergyPlus, ESP-r, TRNSYS)
    /// for each case/metric combination where multi-reference data is available.
    /// Results are grouped by case series (600, 900, special) with overall status
    /// determined by the fallback rule (PASS if EnergyPlus passes, else WARN if any
    /// program passes, else FAIL).
    fn add_multireference_table(&self, output: &mut String, report: &BenchmarkReport) {
        // Check if any results have per-program data
        let has_multiref = report.results.iter().any(|r| r.per_program.is_some());
        if !has_multiref {
            return;
        }

        output.push_str("## Multi-Reference Comparison\n\n");
        output.push_str("| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |\n");
        output.push_str("|------|--------|------------|-------|--------|---------|\n");

        // Sort results by case id and metric for consistent ordering
        let mut sorted_results: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.per_program.is_some())
            .collect();
        sorted_results.sort_by(|a, b| {
            a.case_id
                .cmp(&b.case_id)
                .then_with(|| a.metric.cmp(&b.metric))
        });

        for result in sorted_results {
            if let Some(per_prog) = &result.per_program {
                let case_cell = result.case_id.to_string();
                let metric_cell = result.metric.display_name().to_string();

                let ep = per_prog
                    .get("EnergyPlus")
                    .map(|s| format!("{} ({:.2})", s, result.fluxion_value))
                    .unwrap_or_else(|| "-".to_string());
                let espr = per_prog
                    .get("ESP-r")
                    .map(|s| format!("{} ({:.2})", s, result.fluxion_value))
                    .unwrap_or_else(|| "-".to_string());
                let trnsys = per_prog
                    .get("TRNSYS")
                    .map(|s| format!("{} ({:.2})", s, result.fluxion_value))
                    .unwrap_or_else(|| "-".to_string());

                let overall = match result.status {
                    crate::validation::report::ValidationStatus::Pass => "PASS",
                    crate::validation::report::ValidationStatus::Warning => "WARN",
                    crate::validation::report::ValidationStatus::Fail => "FAIL",
                };

                output.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    case_cell, metric_cell, ep, espr, trnsys, overall
                ));
            }
        }
        output.push('\n');
    }

    /// Generates the full validation report and writes it to the output path.
    pub fn generate(
        &self,
        report: &BenchmarkReport,
        systematic_issues: Option<&SystematicIssueMap>,
        baseline: Option<&BaselineMetrics>,
    ) -> Result<(), String> {
        let markdown = self.render_markdown(report, systematic_issues, baseline)?;

        // Ensure the output directory exists
        if let Some(parent) = self.output_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&self.output_path, markdown)
            .map_err(|e| format!("Failed to write report: {}", e))?;

        Ok(())
    }

    /// Renders a complete Markdown report from the benchmark report.
    pub fn render_markdown(
        &self,
        report: &BenchmarkReport,
        systematic_issues: Option<&SystematicIssueMap>,
        baseline: Option<&BaselineMetrics>,
    ) -> Result<String, String> {
        let mut output = String::new();

        // Header
        output.push_str("# ASHRAE Standard 140 Validation Results\n\n");
        output.push_str(&format!(
            "*Generated: {}*\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        ));

        // Summary Card
        output.push_str("## Summary\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Total Results | {} |\n", report.results.len()));
        output.push_str(&format!("| Pass Rate | {:.1}% |\n", report.pass_rate()));
        output.push_str(&format!(
            "| Passed | {} |\n",
            report.results.iter().filter(|r| r.passed()).count()
        ));
        output.push_str(&format!("| Warnings | {} |\n", report.warning_count()));
        output.push_str(&format!("| Failed | {} |\n", report.fail_count()));
        output.push_str(&format!("| Mean Absolute Error | {:.2}% |\n", report.mae()));
        output.push_str(&format!(
            "| Max Deviation | {:.2}% |\n",
            report.max_deviation()
        ));
        output.push('\n');

        // Performance Summary
        output.push_str("## Performance Summary\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!(
            "| Total Validation Duration | {:.2} seconds |\n",
            report.duration_seconds()
        ));
        output.push_str(&format!(
            "| Throughput | {:.2} cases/sec |\n",
            report.cases_per_second()
        ));
        output.push_str(&format!(
            "| Total Cases | {} |\n",
            report.benchmark_data.len()
        ));
        output.push('\n');

        // Performance Comparison (if baseline provided)
        if let Some(baseline) = baseline {
            output.push_str("## Performance Comparison\n\n");
            output.push_str("| Metric | Current | Baseline | Change | Status |\n");
            output.push_str("|--------|---------|----------|--------|--------|\n");

            // Helper to compute percent change
            let pct_change = |current: f64, base: f64| -> f64 {
                if base != 0.0 {
                    ((current - base) / base) * 100.0
                } else {
                    0.0
                }
            };

            // Mean Absolute Error (MAE)
            let mae = report.mae();
            let mae_change = pct_change(mae, baseline.mae);
            let mae_emoji = if mae_change.abs() <= 2.0 {
                "✅"
            } else if mae_change.abs() <= 10.0 {
                "⚠️"
            } else {
                "❌"
            };
            output.push_str(&format!(
                "| Mean Absolute Error (MAE) | {:.2}% | {:.2}% | {:+.2}% | {} |\n",
                mae, baseline.mae, mae_change, mae_emoji
            ));

            // Max Deviation
            let max_dev = report.max_deviation();
            let maxdev_change = pct_change(max_dev, baseline.max_deviation);
            let maxdev_emoji = if maxdev_change.abs() <= 2.0 {
                "✅"
            } else if maxdev_change.abs() <= 10.0 {
                "⚠️"
            } else {
                "❌"
            };
            output.push_str(&format!(
                "| Max Deviation | {:.2}% | {:.2}% | {:+.2}% | {} |\n",
                max_dev, baseline.max_deviation, maxdev_change, maxdev_emoji
            ));

            // Pass Rate (percentage points)
            let pass_rate = report.pass_rate();
            let passrate_change = pass_rate - baseline.pass_rate;
            let passrate_emoji = if passrate_change >= -2.0 {
                "✅"
            } else if passrate_change > -5.0 {
                "⚠️"
            } else {
                "❌"
            };
            output.push_str(&format!(
                "| Pass Rate | {:.1}% | {:.1}% | {:.1}pp | {} |\n",
                pass_rate, baseline.pass_rate, passrate_change, passrate_emoji
            ));

            // Validation Time
            let duration = report.duration_seconds();
            let time_change = pct_change(duration, baseline.validation_time_seconds);
            let time_emoji = if time_change <= 10.0 { "✅" } else { "⚠️" };
            output.push_str(&format!(
                "| Validation Time | {:.2}s | {:.2}s | {:+.1}% | {} |\n",
                duration, baseline.validation_time_seconds, time_change, time_emoji
            ));

            output.push('\n');
        }

        // Detailed Results Table - grouped by case type
        output.push_str("## Detailed Results\n\n");

        // Group cases: Baseline, High-Mass, Free-Floating, Special
        let baseline_cases = ["600", "610", "620", "630", "640", "650"];
        let high_mass_cases = ["900", "910", "920", "930", "940", "950"];
        let free_floating_cases = ["600FF", "650FF", "900FF", "950FF"];
        let special_cases = ["960", "195"];

        output.push_str("### Baseline Cases (600 Series)\n\n");
        output.push_str(
            "| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |\n",
        );
        output.push_str(
            "|------|----------------|----------------|--------------|--------------|--------|\n",
        );
        for case_id in &baseline_cases {
            self.append_case_row(&mut output, report, case_id);
        }
        output.push('\n');

        output.push_str("### High-Mass Cases (900 Series)\n\n");
        output.push_str(
            "| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |\n",
        );
        output.push_str(
            "|------|----------------|----------------|--------------|--------------|--------|\n",
        );
        for case_id in &high_mass_cases {
            self.append_case_row(&mut output, report, case_id);
        }
        output.push('\n');

        output.push_str("### Free-Floating Cases\n\n");
        output.push_str("| Case | Min Temperature | Max Temperature | Status |\n");
        output.push_str("|------|-----------------|-----------------|--------|\n");
        for case_id in &free_floating_cases {
            self.append_free_floating_row(&mut output, report, case_id);
        }
        output.push('\n');

        output.push_str("### Special Cases\n\n");
        output.push_str(
            "| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |\n",
        );
        output.push_str(
            "|------|----------------|----------------|--------------|--------------|--------|\n",
        );
        for case_id in &special_cases {
            self.append_case_row(&mut output, report, case_id);
        }
        output.push('\n');

        // Multi-reference comparison table (if available)
        self.add_multireference_table(&mut output, report);

        // Systematic Issues Section
        output.push_str("## Systematic Issues\n\n");
        if let Some(issue_map) = systematic_issues {
            if issue_map.is_empty() {
                output.push_str("*No systematic issues identified.*\n\n");
            } else {
                output.push_str(
                    "The following recurring issues are affecting validation results:\n\n",
                );
                let mut issues_by_category: HashMap<&SystematicIssue, Vec<String>> = HashMap::new();
                for (case_metric, issue) in issue_map {
                    issues_by_category
                        .entry(issue)
                        .or_default()
                        .push(case_metric.clone());
                }

                for (issue, cases) in issues_by_category.iter() {
                    output.push_str(&format!("### {}\n\n", issue_display_name(issue)));
                    output.push_str(&format!("**Affected metrics:** {} |\n", cases.join(", ")));
                    output.push_str(&format!("**Count:** {} metrics\n\n", cases.len()));
                }
            }
        } else {
            output.push_str("*Systematic issues taxonomy not yet populated.*\n\n");
        }

        // Link to Known Issues
        output.push_str("## References\n\n");
        output.push_str("- **[Quality Metrics Tracker](QUALITY_METRICS.md)** - Detailed metrics dashboard with historical progression\n");
        output.push_str("- **[Known Systematic Issues](KNOWN_ISSUES.md)** - Comprehensive issue catalog with severity, status, and resolution roadmap\n");
        output.push('\n');

        // Phase Progress
        output.push_str("## Phase Progress\n\n");
        output.push_str("| Phase | Status | Completion | Notes |\n");
        output.push_str("|-------|--------|------------|-------|\n");
        output.push_str(
            "| Phase 1: Foundation | ✅ Complete | 4/4 plans | Conductances, HVAC load fixes |\n",
        );
        output.push_str("| Phase 2: Thermal Mass | ✅ Complete | 4/4 plans | Implicit integration validated |\n");
        output.push_str("| Phase 3: Solar & External | ✅ Complete | 3/3 plans | Solar integration, mode-specific coupling |\n");
        output.push_str("| Phase 4: Multi-Zone Transfer | ✅ Complete | 6/6 plans | Inter-zone heat transfer validated |\n");
        output.push_str(
            "| Phase 5: Diagnostics & Reporting | 🔄 In Progress | 4/4 plans | Quality metrics, issue tracking |\n",
        );
        output.push_str("| Phase 6: Performance Optimization | ⏳ Pending | 0/12 requirements | GPU acceleration, throughput |\n");
        output.push_str("| Phase 7: Advanced Analysis | ⏳ Pending | 0/20 requirements | Sensitivity, visualization |\n");
        output.push('\n');

        // What's Fixed in This Phase
        output.push_str("## What's Fixed in Phase 5\n\n");
        output.push_str(
            "This phase delivered systematic diagnostics and reporting infrastructure:\n\n",
        );
        output.push_str(
            "- ✅ **REPORT-01:** Automated quality metrics computation via `analyzer.rs`\n",
        );
        output.push_str("- ✅ **REPORT-02:** Quality metrics dashboard (`QUALITY_METRICS.md`) with historical progression\n");
        output.push_str("- ✅ **REPORT-03:** Comprehensive known issues catalog (`KNOWN_ISSUES.md`) with taxonomy, severity, and GitHub links\n");
        output.push_str("- ✅ **REPORT-04:** Enhanced validation report with issue references and phase summaries\n");
        output.push('\n');

        // Legend
        output.push_str("## Legend\n\n");
        output.push_str("- **PASS**: Value within 5% of reference range\n");
        output.push_str("- **WARN**: Value within reference range but >2% deviation, or within tolerance band\n");
        output.push_str("- **FAIL**: Value outside 5% tolerance band\n");

        Ok(output)
    }

    /// Appends a single case row to the detailed results table.
    fn append_case_row(&self, output: &mut String, report: &BenchmarkReport, case_id: &str) {
        let case_results: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.case_id == case_id)
            .collect();

        // Get benchmark data for correct reference values
        // (Don't rely on result.ref_min/ref_max which may be zeroed in fallback)
        let benchmark = report.benchmark_data.get(case_id);

        let mut heating_str = String::new();
        let mut cooling_str = String::new();
        let mut peak_h_str = String::new();
        let mut peak_c_str = String::new();

        for result in &case_results {
            // Use benchmark data for reference values if available
            let (ref_min, ref_max) = match (result.metric, benchmark) {
                (MetricType::AnnualHeating, Some(b)) => {
                    (b.annual_heating_min, b.annual_heating_max)
                }
                (MetricType::AnnualCooling, Some(b)) => {
                    (b.annual_cooling_min, b.annual_cooling_max)
                }
                (MetricType::PeakHeating, Some(b)) => (b.peak_heating_min, b.peak_heating_max),
                (MetricType::PeakCooling, Some(b)) => (b.peak_cooling_min, b.peak_cooling_max),
                _ => (result.ref_min, result.ref_max), // Fallback to result values
            };

            match result.metric {
                MetricType::AnnualHeating => {
                    heating_str = format!(
                        "{:.2} MWh (Ref: {:.2}-{:.2})",
                        result.fluxion_value, ref_min, ref_max
                    );
                }
                MetricType::AnnualCooling => {
                    cooling_str = format!(
                        "{:.2} MWh (Ref: {:.2}-{:.2})",
                        result.fluxion_value, ref_min, ref_max
                    );
                }
                MetricType::PeakHeating => {
                    peak_h_str = format!(
                        "{:.2} kW (Ref: {:.2}-{:.2})",
                        result.fluxion_value, ref_min, ref_max
                    );
                }
                MetricType::PeakCooling => {
                    peak_c_str = format!(
                        "{:.2} kW (Ref: {:.2}-{:.2})",
                        result.fluxion_value, ref_min, ref_max
                    );
                }
                _ => {}
            }
        }

        // Determine overall status for this case
        let overall_status = if case_results.is_empty() {
            "❓ Unknown".to_string()
        } else {
            let passes = case_results.iter().filter(|r| r.passed()).count();
            let warnings = case_results.iter().filter(|r| r.warning()).count();
            let fails = case_results.iter().filter(|r| r.failed()).count();

            if fails == 0 && passes > 0 {
                "✅ PASS".to_string()
            } else if fails > 0 {
                "❌ FAIL".to_string()
            } else if warnings > 0 {
                "⚠️ WARN".to_string()
            } else {
                "❓ Unknown".to_string()
            }
        };

        output.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            case_id, heating_str, cooling_str, peak_h_str, peak_c_str, overall_status
        ));
    }

    /// Appends a free-floating case row.
    fn append_free_floating_row(
        &self,
        output: &mut String,
        report: &BenchmarkReport,
        case_id: &str,
    ) {
        let case_results: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.case_id == case_id)
            .collect();

        // Get benchmark data for correct reference values
        let benchmark = report.benchmark_data.get(case_id);

        let mut min_str = String::new();
        let mut max_str = String::new();

        for result in &case_results {
            // Use benchmark data for reference values if available
            let (ref_min, ref_max) = match (result.metric, benchmark) {
                (MetricType::MinFreeFloat, Some(b)) => (b.min_free_float_min, b.min_free_float_max),
                (MetricType::MaxFreeFloat, Some(b)) => (b.max_free_float_min, b.max_free_float_max),
                _ => (result.ref_min, result.ref_max), // Fallback to result values
            };

            match result.metric {
                MetricType::MinFreeFloat => {
                    min_str = format!(
                        "{:.2}°C (Ref: {:.2}-{:.2})",
                        result.fluxion_value, ref_min, ref_max
                    );
                }
                MetricType::MaxFreeFloat => {
                    max_str = format!(
                        "{:.2}°C (Ref: {:.2}-{:.2})",
                        result.fluxion_value, ref_min, ref_max
                    );
                }
                _ => {}
            }
        }

        let status = if case_results.iter().all(|r| r.passed()) {
            "✅ PASS"
        } else if case_results.iter().any(|r| r.failed()) {
            "❌ FAIL"
        } else {
            "⚠️ WARN"
        };

        output.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            case_id, min_str, max_str, status
        ));
    }
}

/// Maps case+metric pairs to their systematic issue classification.
pub type SystematicIssueMap = HashMap<String, SystematicIssue>;

impl ValidationReportGenerator {
    /// Classifies systematic issues from a benchmark report based on heuristics.
    ///
    /// This function analyzes failure patterns and assigns known issue categories.
    /// The mapping is based on current known issues from validation results.
    pub fn classify_systematic_issues(report: &BenchmarkReport) -> SystematicIssueMap {
        let mut map = SystematicIssueMap::new();

        for result in &report.results {
            if result.failed() {
                let key = format!("{} - {}", result.case_id, result.metric);
                let issue = classify_issue(result.case_id.as_str(), result.metric);
                map.insert(key, issue);
            }
        }

        map
    }
}

/// Classifies a single failed metric to a systematic issue category.
fn classify_issue(case_id: &str, metric: MetricType) -> SystematicIssue {
    // Known issue: Case 960 annual cooling over-prediction (issue #273)
    if case_id == "960" && metric == MetricType::AnnualCooling {
        return SystematicIssue::InterZoneTransfer;
    }

    // Known issue: Case 960 peak cooling within ref but high error? Already classified as InterZoneTransfer if failed

    // High-mass building annual energy over-prediction (900 series) - 5R1C model limitation
    if (case_id == "900"
        || case_id == "910"
        || case_id == "920"
        || case_id == "930"
        || case_id == "940"
        || case_id == "950"
        || case_id == "900FF"
        || case_id == "950FF")
        && (metric == MetricType::AnnualHeating || metric == MetricType::AnnualCooling)
    {
        return SystematicIssue::ModelLimitation;
    }

    // Low-mass cases peak cooling under-prediction (600-650 series) - likely solar gains
    if (case_id.starts_with('6') && case_id != "600FF" && case_id != "650FF")
        && metric == MetricType::PeakCooling
    {
        return SystematicIssue::SolarGains;
    }

    // Free-floating temperature failures in high-mass could be thermal mass dynamics
    if (case_id == "900FF" || case_id == "950FF")
        && (metric == MetricType::MinFreeFloat || metric == MetricType::MaxFreeFloat)
    {
        return SystematicIssue::ThermalMass;
    }

    // Default to unknown for unclassified failures
    SystematicIssue::Unknown
}

/// Displays a human-readable name for a systematic issue.
fn issue_display_name(issue: &SystematicIssue) -> &str {
    match issue {
        SystematicIssue::SolarGains => "Solar Gain Calculations",
        SystematicIssue::ThermalMass => "Thermal Mass Dynamics",
        SystematicIssue::InterZoneTransfer => "Inter-Zone Heat Transfer",
        SystematicIssue::HvacLoad => "HVAC Load Calculation",
        SystematicIssue::WeatherData => "Weather Data",
        SystematicIssue::ModelLimitation => "5R1C Model Limitation (Accepted)",
        SystematicIssue::Unknown => "Unknown/Unclassified",
    }
}

impl ValidationReportGenerator {
    /// Generates a report with statistical validation sections.
    ///
    /// This method extends the standard validation report with:
    /// - Statistical metrics summary (NMBE, CV(RMSE), confidence intervals)
    /// - Benjamini-Hochberg FDR correction results
    /// - Group-level validation results
    /// - Effect size analysis (Cohen's d)
    ///
    /// # Arguments
    /// * `statistical_report` - Statistical validation report with metrics and FDR results
    /// * `systematic_issues` - Optional systematic issue classification
    /// * `baseline` - Optional baseline metrics for performance comparison
    ///
    /// # Returns
    /// * Result indicating success or error message
    pub fn generate_with_statistics(
        &self,
        statistical_report: &StatisticalReport,
        systematic_issues: Option<&SystematicIssueMap>,
        baseline: Option<&BaselineMetrics>,
    ) -> Result<(), String> {
        let mut markdown =
            self.render_markdown(&statistical_report.tolerance, systematic_issues, baseline)?;

        // Append statistical sections
        markdown.push_str(&Self::format_statistical_metrics(
            &statistical_report.metrics,
        ));
        markdown.push_str(&Self::format_bh_correction_from_report(statistical_report));
        markdown.push_str(&Self::format_group_validation(
            &statistical_report.group_validation,
        ));

        // Ensure the output directory exists
        if let Some(parent) = self.output_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&self.output_path, markdown)
            .map_err(|e| format!("Failed to write report: {}", e))?;

        Ok(())
    }

    /// Exports statistical metrics to CSV format.
    ///
    /// # Arguments
    /// * `report` - Benchmark report containing statistical data
    /// * `path` - Output file path
    ///
    /// # Returns
    /// * Result indicating success or error message
    pub fn export_statistical_csv<P: AsRef<std::path::Path>>(
        report: &BenchmarkReport,
        path: P,
    ) -> Result<(), String> {
        let mut csv = String::new();

        // Header
        csv.push_str("case_id,metric_type,predicted,reference_midpoint,nmbe,cv_rmse,ci_nmbe_lower,ci_nmbe_upper,ci_cvrmse_lower,ci_cvrmse_upper,p_value,bh_corrected\n");

        // Data rows - iterate through results and add statistical data if available
        for (i, result) in report.results.iter().enumerate() {
            let predicted = result.fluxion_value;
            let ref_midpoint = (result.ref_min + result.ref_max) / 2.0;

            // Get statistical data if available
            let (nmbe, cv_rmse) = if let Some(ref metrics) = report.statistical_metrics {
                (metrics.nmbe, metrics.cv_rmse)
            } else {
                (f64::NAN, f64::NAN)
            };

            let (nmbe_lower, nmbe_upper, cvrmse_lower, cvrmse_upper) =
                if let Some(ref metrics) = report.statistical_metrics {
                    (
                        metrics.nmbe_ci.0,
                        metrics.nmbe_ci.1,
                        metrics.cv_rmse_ci.0,
                        metrics.cv_rmse_ci.1,
                    )
                } else {
                    (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
                };

            let p_value = if let Some(ref p_values) = report.statistical_p_values {
                if i < p_values.len() {
                    p_values[i]
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };

            let bh_corrected = if let Some(ref corrected) = report.statistical_corrected {
                if i < corrected.len() {
                    corrected[i]
                } else {
                    false
                }
            } else {
                false
            };

            let _cohens_d = if let Some(ref metrics) = report.statistical_metrics {
                metrics.cohens_d
            } else {
                f64::NAN
            };

            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}\n",
                result.case_id,
                result.metric,
                predicted,
                ref_midpoint,
                nmbe,
                cv_rmse,
                nmbe_lower,
                nmbe_upper,
                cvrmse_lower,
                cvrmse_upper,
                p_value,
                bh_corrected
            ));
        }

        // Write to file
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(path_ref, csv).map_err(|e| format!("Failed to write CSV: {}", e))?;

        Ok(())
    }

    /// Exports statistical metrics to JSON format.
    ///
    /// # Arguments
    /// * `report` - Benchmark report containing statistical data
    /// * `path` - Output file path
    ///
    /// # Returns
    /// * Result indicating success or error message
    pub fn export_statistical_json<P: AsRef<std::path::Path>>(
        report: &BenchmarkReport,
        path: P,
    ) -> Result<(), String> {
        let mut json_data = serde_json::json!({});

        // Add statistical metrics if available
        if let Some(ref metrics) = report.statistical_metrics {
            json_data["statistical_metrics"] = serde_json::json!({
                "nmbe": metrics.nmbe,
                "cv_rmse": metrics.cv_rmse,
                "nmbe_ci": {
                    "lower": metrics.nmbe_ci.0,
                    "upper": metrics.nmbe_ci.1
                },
                "cv_rmse_ci": {
                    "lower": metrics.cv_rmse_ci.0,
                    "upper": metrics.cv_rmse_ci.1
                },
                "cohens_d": metrics.cohens_d,
                "effect_direction": format!("{:?}", metrics.effect_direction),
                "excluded_cases": metrics.excluded_cases
            });
        }

        // Add group validation if available
        if let Some(ref group_validation) = report.group_validation {
            let mut groups_obj = serde_json::Map::new();
            for (group, &passed) in group_validation {
                let group_str = format!("{:?}", group);
                groups_obj.insert(group_str, serde_json::Value::Bool(passed));
            }
            json_data["group_validation"] = serde_json::Value::Object(groups_obj);
        }

        // Add metadata
        json_data["metadata"] = serde_json::json!({
            "alpha": 0.05,
            "fdr_method": "Benjamini-Hochberg",
            "threshold_type": "hybrid_80_percent_or_single_case",
            "excluded_cases": report.statistical_metrics.as_ref().map(|m| m.excluded_cases).unwrap_or(0)
        });

        // Write to file
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let json_string = serde_json::to_string_pretty(&json_data)
            .map_err(|e| format!("Failed to serialize JSON: {}", e))?;

        fs::write(path_ref, json_string).map_err(|e| format!("Failed to write JSON: {}", e))?;

        Ok(())
    }

    /// Formats group validation results into a Markdown table.
    ///
    /// # Arguments
    /// * `group_validation` - HashMap of validation group to pass/fail status
    ///
    /// # Returns
    /// * Markdown formatted string with group validation table
    fn format_group_validation(
        group_validation: &std::collections::HashMap<ValidationGroup, bool>,
    ) -> String {
        let mut output = String::new();

        output.push_str("## Group-Level Validation Results\n\n");
        output.push_str("| Group | Cases | Threshold | Status |\n");
        output.push_str("|-------|-------|-----------|--------|\n");

        // Sort groups for consistent output
        let mut groups: Vec<_> = group_validation.iter().collect();
        groups.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));

        for (group, &passed) in groups {
            let group_name = match group {
                ValidationGroup::Baseline => "Baseline (600-650)",
                ValidationGroup::HighMass => "High-Mass (900-950)",
                ValidationGroup::FreeFloating => "Free-Floating",
                ValidationGroup::Diagnostics => "Diagnostics",
                ValidationGroup::Equipment => "HVAC Equipment (800-810)",
            };

            let threshold = "80% pass rate"; // Simplified - actual threshold depends on case count
            let status = if passed { "✅ PASS" } else { "❌ FAIL" };

            output.push_str(&format!(
                "| {} | ≥5 | {} | {} |\n",
                group_name, threshold, status
            ));
        }

        output.push('\n');
        output
    }

    /// Formats Benjamini-Hochberg FDR correction from a StatisticalReport.
    ///
    /// # Arguments
    /// * `stat_report` - Statistical report containing FDR correction results
    ///
    /// # Returns
    /// * Markdown formatted string with BH correction table
    fn format_bh_correction_from_report(
        stat_report: &crate::validation::statistical::StatisticalReport,
    ) -> String {
        let mut output = String::new();

        output.push_str("## Benjamini-Hochberg FDR Correction\n\n");
        output.push_str("| Group | Passed Tests | Total Tests | Status |\n");
        output.push_str("|-------|--------------|-------------|--------|\n");

        // Sort groups for consistent output
        let mut groups: Vec<_> = stat_report.corrected_p_values.iter().collect();
        groups.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));

        for (group, passed_tests) in groups {
            let passed_count = passed_tests.iter().filter(|&&passed| passed).count();
            let total_count = passed_tests.len();

            let group_name = match group {
                ValidationGroup::Baseline => "Baseline (600-650)",
                ValidationGroup::HighMass => "High-Mass (900-950)",
                ValidationGroup::FreeFloating => "Free-Floating",
                ValidationGroup::Diagnostics => "Diagnostics",
                ValidationGroup::Equipment => "HVAC Equipment (800-810)",
            };

            let status = if passed_count as f64 / total_count as f64 >= 0.8 {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };

            output.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                group_name, passed_count, total_count, status
            ));
        }

        output.push('\n');
        output.push_str("**Note:** FDR correction applied separately within each validation group (α = 0.05)\n\n");

        output
    }

    /// Renders a complete Markdown report with statistical validation sections.
    ///
    /// This is similar to generate_with_statistics but returns the markdown string
    /// instead of writing to a file (useful for stdout output).
    ///
    /// # Arguments
    /// * `statistical_report` - Statistical validation report with metrics and FDR results
    /// * `systematic_issues` - Optional systematic issue classification
    /// * `baseline` - Optional baseline metrics for performance comparison
    ///
    /// # Returns
    /// * Result containing markdown string or error message
    pub fn render_markdown_with_statistics(
        &self,
        statistical_report: &StatisticalReport,
        systematic_issues: Option<&SystematicIssueMap>,
        baseline: Option<&BaselineMetrics>,
    ) -> Result<String, String> {
        let mut markdown =
            self.render_markdown(&statistical_report.tolerance, systematic_issues, baseline)?;

        // Append statistical sections
        markdown.push_str(&Self::format_statistical_metrics(
            &statistical_report.metrics,
        ));
        markdown.push_str(&Self::format_bh_correction_from_report(statistical_report));
        markdown.push_str(&Self::format_group_validation(
            &statistical_report.group_validation,
        ));

        Ok(markdown)
    }

    /// Formats statistical metrics into a Markdown table.
    ///
    /// # Arguments
    /// * `metrics` - Statistical metrics including NMBE, CV(RMSE), confidence intervals, etc.
    ///
    /// # Returns
    /// * Markdown formatted string with statistical metrics table
    pub fn format_statistical_metrics(metrics: &StatisticalMetrics) -> String {
        let mut output = String::new();

        output.push_str("## Statistical Metrics\n\n");
        output.push_str("| Metric | Value | 95% CI Lower | 95% CI Upper |\n");
        output.push_str("|--------|-------|--------------|--------------|\n");

        // NMBE
        let nmbe_val = if metrics.nmbe.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}%", metrics.nmbe)
        };
        let nmbe_lower = if metrics.nmbe_ci.0.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}%", metrics.nmbe_ci.0)
        };
        let nmbe_upper = if metrics.nmbe_ci.1.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}%", metrics.nmbe_ci.1)
        };
        output.push_str(&format!(
            "| NMBE | {} | {} | {} |\n",
            nmbe_val, nmbe_lower, nmbe_upper
        ));

        // CV(RMSE)
        let cvrmse_val = if metrics.cv_rmse.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}%", metrics.cv_rmse)
        };
        let cvrmse_lower = if metrics.cv_rmse_ci.0.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}%", metrics.cv_rmse_ci.0)
        };
        let cvrmse_upper = if metrics.cv_rmse_ci.1.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}%", metrics.cv_rmse_ci.1)
        };
        output.push_str(&format!(
            "| CV(RMSE) | {} | {} | {} |\n",
            cvrmse_val, cvrmse_lower, cvrmse_upper
        ));

        output.push('\n');

        // Effect size
        let effect_direction_str = match metrics.effect_direction {
            EffectDirection::Overprediction => "Overprediction",
            EffectDirection::Underprediction => "Underprediction",
        };
        output.push_str(&format!(
            "**Effect Size (Cohen's d):** {:.2} ({})\n\n",
            metrics.cohens_d, effect_direction_str
        ));

        // Excluded cases
        output.push_str(&format!(
            "**Excluded Cases:** {} (zero/near-zero reference values)\n\n",
            metrics.excluded_cases
        ));

        output
    }

    /// Formats Benjamini-Hochberg FDR correction results into a Markdown table.
    ///
    /// # Arguments
    /// * `p_values` - List of p-values from statistical tests
    /// * `corrected` - List of booleans indicating which tests pass BH correction (true = rejected null hypothesis)
    /// * `metric_types` - List of metric types corresponding to each p-value (for grouping)
    ///
    /// # Returns
    /// * Markdown formatted string with BH correction table
    pub fn format_bh_correction(
        p_values: &[f64],
        corrected: &[bool],
        metric_types: &[MetricType],
    ) -> String {
        let mut output = String::new();

        output.push_str("## Benjamini-Hochberg FDR Correction\n\n");
        output.push_str("| Metric | P-Value | BH Corrected |\n");
        output.push_str("|--------|----------|--------------|\n");

        // Iterate through all p-values and create table rows
        for (i, &p_val) in p_values.iter().enumerate() {
            let metric_type = if i < metric_types.len() {
                metric_types[i].display_name().to_string()
            } else {
                format!("Metric {}", i + 1)
            };

            let is_corrected = if i < corrected.len() {
                corrected[i]
            } else {
                false
            };

            let p_val_str = if p_val.is_nan() {
                "N/A".to_string()
            } else {
                format!("{:.4}", p_val)
            };

            let status = if is_corrected { "✅" } else { "❌" };

            output.push_str(&format!(
                "| {} | {} | {} |\n",
                metric_type, p_val_str, status
            ));
        }

        output.push('\n');
        output.push_str("**Note:** Applied separately within each validation group (α = 0.05)\n\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::report::{
        BenchmarkReport, MetricType, ValidationResult, ValidationStatus,
    };
    use std::fs;

    #[test]
    fn test_multireference_table() {
        // Create a BenchmarkReport with some results that have per_program data
        let mut report = BenchmarkReport::new();

        // Add a result with multi-reference enrichment
        // We'll simulate the enrichment by manually constructing a result with per_program
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 6.0,
            ref_min: 5.5,
            ref_max: 7.0,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            per_program: Some(
                vec![
                    ("EnergyPlus".to_string(), ValidationStatus::Pass),
                    ("ESP-r".to_string(), ValidationStatus::Pass),
                    ("TRNSYS".to_string(), ValidationStatus::Warning),
                ]
                .into_iter()
                .collect(),
            ),
        };
        report.add_result(result);

        // Add a result without per_program (should be skipped)
        let result2 = ValidationResult::new("900FF", MetricType::MaxFreeFloat, 45.0, 40.0, 50.0);
        report.add_result(result2);

        // Create generator and render
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_multiref.md");
        let generator = ValidationReportGenerator::new(output_path.clone());
        let markdown = generator.render_markdown(&report, None, None).unwrap();

        // Verify table appears
        assert!(markdown.contains("## Multi-Reference Comparison"));
        assert!(markdown.contains("| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |"));
        assert!(markdown.contains(
            "| 600 | Annual Heating (MWh) | PASS (6.00) | PASS (6.00) | WARN (6.00) | PASS |"
        ));

        // Extract the Multi-Reference Comparison section to verify 900FF not included
        let section_start = markdown.find("## Multi-Reference Comparison").unwrap();
        let next_section = markdown[section_start..]
            .find("\n##")
            .map(|pos| section_start + pos)
            .unwrap_or(markdown.len());
        let section = &markdown[section_start..next_section];

        // Within the multi-reference table section, 600 should appear, but 900FF should not
        assert!(section.contains("600"));
        assert!(
            !section.contains("900FF"),
            "900FF should not appear in multi-reference table but it does. Section content:\n{}",
            section
        );

        // Clean up
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_format_statistical_metrics() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        // Test 1: Basic formatting with all fields
        let metrics = StatisticalMetrics {
            nmbe: 2.34,
            cv_rmse: 8.76,
            nmbe_ci: (1.56, 3.12),
            cv_rmse_ci: (7.23, 10.29),
            cohens_d: 0.42,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        };

        let formatted = ValidationReportGenerator::format_statistical_metrics(&metrics);

        // Verify section headers
        assert!(formatted.contains("## Statistical Metrics"));
        assert!(formatted.contains("| Metric | Value | 95% CI Lower | 95% CI Upper |"));
        assert!(formatted.contains("|--------|-------|--------------|--------------|"));

        // Verify NMBE row
        assert!(formatted.contains("| NMBE | 2.34% | 1.56% | 3.12% |"));

        // Verify CV(RMSE) row
        assert!(formatted.contains("| CV(RMSE) | 8.76% | 7.23% | 10.29% |"));

        // Verify effect size
        assert!(formatted.contains("**Effect Size (Cohen's d):** 0.42 (Underprediction)"));

        // Verify excluded cases
        assert!(formatted.contains("**Excluded Cases:** 0 (zero/near-zero reference values)"));

        // Test 2: NaN handling
        let nan_metrics = StatisticalMetrics {
            nmbe: f64::NAN,
            cv_rmse: 5.5,
            nmbe_ci: (f64::NAN, 2.0),
            cv_rmse_ci: (4.5, 6.5),
            cohens_d: 0.35,
            effect_direction: EffectDirection::Overprediction,
            excluded_cases: 1,
        };

        let nan_formatted = ValidationReportGenerator::format_statistical_metrics(&nan_metrics);

        // Verify NaN values show as N/A
        assert!(nan_formatted.contains("| NMBE | N/A | N/A | 2.00% |")); // nmbe and lower CI are NaN, upper CI is 2.0
        assert!(nan_formatted.contains("| CV(RMSE) | 5.50% | 4.50% | 6.50% |")); // CV(RMSE) is not NaN

        // Verify overprediction direction
        assert!(nan_formatted.contains("(Overprediction)"));

        // Verify excluded cases count
        assert!(nan_formatted.contains("**Excluded Cases:** 1"));

        // Test 3: Effect direction variations
        let overpred_metrics = StatisticalMetrics {
            nmbe: -1.5,
            cv_rmse: 6.0,
            nmbe_ci: (-2.0, -1.0),
            cv_rmse_ci: (5.0, 7.0),
            cohens_d: -0.38,
            effect_direction: EffectDirection::Overprediction,
            excluded_cases: 0,
        };

        let overpred_formatted =
            ValidationReportGenerator::format_statistical_metrics(&overpred_metrics);
        assert!(overpred_formatted.contains("(Overprediction)"));

        let underpred_metrics = StatisticalMetrics {
            nmbe: 1.5,
            cv_rmse: 6.0,
            nmbe_ci: (1.0, 2.0),
            cv_rmse_ci: (5.0, 7.0),
            cohens_d: 0.38,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        };

        let underpred_formatted =
            ValidationReportGenerator::format_statistical_metrics(&underpred_metrics);
        assert!(underpred_formatted.contains("(Underprediction)"));
    }

    #[test]
    fn test_format_statistical_metrics_table_structure() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        let metrics = StatisticalMetrics {
            nmbe: 1.23,
            cv_rmse: 4.56,
            nmbe_ci: (0.78, 1.68),
            cv_rmse_ci: (3.89, 5.23),
            cohens_d: 0.25,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        };

        let formatted = ValidationReportGenerator::format_statistical_metrics(&metrics);

        // Test 4: Verify table structure
        let lines: Vec<&str> = formatted.lines().collect();

        // Find the table section
        let table_start = lines
            .iter()
            .position(|l| l.contains("| Metric |"))
            .expect("Should find table header");
        let table_end = lines
            .iter()
            .position(|l| l.starts_with("| NMBE |"))
            .expect("Should find NMBE row");

        // Verify header row
        assert_eq!(
            lines[table_start],
            "| Metric | Value | 95% CI Lower | 95% CI Upper |"
        );

        // Verify separator row
        assert_eq!(
            lines[table_start + 1],
            "|--------|-------|--------------|--------------|"
        );

        // Verify data rows exist
        assert!(lines.iter().any(|l| l.starts_with("| NMBE |")));
        assert!(lines.iter().any(|l| l.starts_with("| CV(RMSE) |")));

        // Verify sections appear in correct order
        let nmbe_pos = formatted.find("| NMBE |").unwrap();
        let cvrmse_pos = formatted.find("| CV(RMSE) |").unwrap();
        let effect_size_pos = formatted.find("**Effect Size (Cohen's d):**").unwrap();
        let excluded_pos = formatted.find("**Excluded Cases:**").unwrap();

        assert!(nmbe_pos < cvrmse_pos);
        assert!(cvrmse_pos < effect_size_pos);
        assert!(effect_size_pos < excluded_pos);
    }

    #[test]
    fn test_format_bh_correction() {
        // Test 1: Basic BH correction formatting
        let p_values = vec![0.0234, 0.0890, 0.1562, 0.0045, 0.1234];
        let corrected = vec![true, false, false, true, false];
        let metric_types = vec![
            MetricType::AnnualHeating,
            MetricType::AnnualCooling,
            MetricType::PeakHeating,
            MetricType::PeakCooling,
            MetricType::MinFreeFloat,
        ];

        let formatted =
            ValidationReportGenerator::format_bh_correction(&p_values, &corrected, &metric_types);

        // Verify section header
        assert!(formatted.contains("## Benjamini-Hochberg FDR Correction"));
        assert!(formatted.contains("| Metric | P-Value | BH Corrected |"));
        assert!(formatted.contains("|--------|----------|--------------|"));

        // Verify corrected tests show checkmark
        assert!(formatted.contains("✅"));
        assert!(formatted.contains("❌"));

        // Verify p-values are formatted correctly
        assert!(formatted.contains("0.0234"));
        assert!(formatted.contains("0.0890"));
        assert!(formatted.contains("0.1562"));
        assert!(formatted.contains("0.0045"));
        assert!(formatted.contains("0.1234"));

        // Verify note about per-group application
        assert!(formatted
            .contains("**Note:** Applied separately within each validation group (α = 0.05)"));

        // Test 2: Verify metric names are displayed
        assert!(formatted.contains("Annual Heating (MWh)"));
        assert!(formatted.contains("Annual Cooling (MWh)"));
        assert!(formatted.contains("Peak Heating (kW)"));
        assert!(formatted.contains("Peak Cooling (kW)"));
        assert!(formatted.contains("Min Free-Float Temp (°C)"));

        // Test 3: All corrected
        let all_corrected_p = vec![0.01, 0.02, 0.03];
        let all_corrected = vec![true, true, true];
        let all_types = vec![
            MetricType::AnnualHeating,
            MetricType::AnnualCooling,
            MetricType::PeakHeating,
        ];

        let all_corrected_formatted = ValidationReportGenerator::format_bh_correction(
            &all_corrected_p,
            &all_corrected,
            &all_types,
        );

        // All should have checkmarks
        let checkmark_count = all_corrected_formatted.matches("✅").count();
        assert_eq!(checkmark_count, 3);
        assert!(!all_corrected_formatted.contains("❌"));

        // Test 4: None corrected
        let none_corrected = vec![false, false, false];
        let none_corrected_formatted = ValidationReportGenerator::format_bh_correction(
            &all_corrected_p,
            &none_corrected,
            &all_types,
        );

        // All should have crosses
        let cross_count = none_corrected_formatted.matches("❌").count();
        assert_eq!(cross_count, 3);
        assert!(!none_corrected_formatted.contains("✅"));

        // Test 5: NaN handling in p-values
        let nan_p = vec![0.01, f64::NAN, 0.03];
        let nan_corrected = vec![true, false, true];
        let nan_types = vec![
            MetricType::AnnualHeating,
            MetricType::AnnualCooling,
            MetricType::PeakHeating,
        ];

        let nan_formatted =
            ValidationReportGenerator::format_bh_correction(&nan_p, &nan_corrected, &nan_types);
        assert!(nan_formatted.contains("N/A"));
    }

    #[test]
    fn test_format_bh_correction_grouping() {
        // Test 6: Grouping by metric type
        let p_values = vec![
            0.01, // Annual Heating
            0.02, // Peak Heating
            0.03, // Annual Cooling
            0.04, // Peak Cooling
        ];
        let corrected = vec![true, true, false, false];
        let metric_types = vec![
            MetricType::AnnualHeating,
            MetricType::PeakHeating,
            MetricType::AnnualCooling,
            MetricType::PeakCooling,
        ];

        let formatted =
            ValidationReportGenerator::format_bh_correction(&p_values, &corrected, &metric_types);

        // Verify all metrics appear
        assert!(formatted.contains("Annual Heating (MWh)"));
        assert!(formatted.contains("Peak Heating (kW)"));
        assert!(formatted.contains("Annual Cooling (MWh)"));
        assert!(formatted.contains("Peak Cooling (kW)"));

        // Verify ordering matches input
        let annual_heating_pos = formatted.find("Annual Heating (MWh)").unwrap();
        let peak_heating_pos = formatted.find("Peak Heating (kW)").unwrap();
        let annual_cooling_pos = formatted.find("Annual Cooling (MWh)").unwrap();
        let peak_cooling_pos = formatted.find("Peak Cooling (kW)").unwrap();

        assert!(annual_heating_pos < peak_heating_pos);
        assert!(peak_heating_pos < annual_cooling_pos);
        assert!(annual_cooling_pos < peak_cooling_pos);

        // Test 7: Verify table structure
        let lines: Vec<&str> = formatted.lines().collect();

        // Find the table section
        let table_start = lines
            .iter()
            .position(|l| l.contains("| Metric |"))
            .expect("Should find table header");

        // Verify header row
        assert_eq!(lines[table_start], "| Metric | P-Value | BH Corrected |");

        // Verify separator row
        assert_eq!(
            lines[table_start + 1],
            "|--------|----------|--------------|"
        );

        // Count data rows (excluding header and separator)
        let data_rows: Vec<_> = lines
            .iter()
            .skip(table_start + 2)
            .take_while(|l| l.starts_with("|"))
            .collect();

        assert_eq!(data_rows.len(), 4); // 4 p-values
    }

    #[test]
    fn test_generate_with_statistics_integration() {
        use crate::validation::statistical::{
            EffectDirection, StatisticalMetrics, StatisticalReport, ValidationGroup,
        };

        // Create a statistical report with all components
        let statistical_report = StatisticalReport {
            metrics: StatisticalMetrics {
                nmbe: 2.3,
                cv_rmse: 8.7,
                nmbe_ci: (1.5, 3.1),
                cv_rmse_ci: (7.2, 10.2),
                cohens_d: 0.42,
                effect_direction: EffectDirection::Underprediction,
                excluded_cases: 0,
            },
            tolerance: crate::validation::report::BenchmarkReport::new(),
            group_validation: {
                let mut groups = std::collections::HashMap::new();
                groups.insert(ValidationGroup::Baseline, true);
                groups.insert(ValidationGroup::HighMass, false);
                groups
            },
            corrected_p_values: std::collections::HashMap::new(),
        };

        // Create a generator with temp output
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_statistical_report.md");
        let generator = ValidationReportGenerator::new(output_path.clone());

        // Generate report with statistics
        let result = generator.generate_with_statistics(&statistical_report, None, None);

        assert!(result.is_ok(), "Should generate report successfully");

        // Verify file was created
        assert!(output_path.exists(), "Report file should exist");

        // Read and verify content
        let content = fs::read_to_string(&output_path).expect("Should read report file");

        // Verify all statistical sections are present
        assert!(content.contains("## Statistical Metrics"));
        assert!(content.contains("## Benjamini-Hochberg FDR Correction"));
        assert!(content.contains("## Group-Level Validation Results"));

        // Verify statistical metrics content
        assert!(content.contains("NMBE"));
        assert!(content.contains("CV(RMSE)"));
        assert!(content.contains("Cohen's d"));
        assert!(content.contains("Underprediction"));

        // Verify group validation content
        assert!(content.contains("Baseline"));
        assert!(content.contains("High-Mass"));

        // Verify tolerance-based sections are still present
        assert!(content.contains("## Summary"));
        assert!(content.contains("## Detailed Results"));

        // Clean up
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_generate_with_statistics_backward_compatibility() {
        use crate::validation::statistical::StatisticalReport;

        // Test that reports work with minimal statistical data
        let minimal_report = StatisticalReport {
            metrics: crate::validation::statistical::StatisticalMetrics {
                nmbe: f64::NAN,
                cv_rmse: f64::NAN,
                nmbe_ci: (f64::NAN, f64::NAN),
                cv_rmse_ci: (f64::NAN, f64::NAN),
                cohens_d: 0.0,
                effect_direction: crate::validation::statistical::EffectDirection::Underprediction,
                excluded_cases: 0,
            },
            tolerance: crate::validation::report::BenchmarkReport::new(),
            group_validation: std::collections::HashMap::new(),
            corrected_p_values: std::collections::HashMap::new(),
        };

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_minimal_statistical.md");
        let generator = ValidationReportGenerator::new(output_path.clone());

        let result = generator.generate_with_statistics(&minimal_report, None, None);

        assert!(result.is_ok(), "Should handle NaN values gracefully");

        // Verify file was created even with NaN data
        assert!(output_path.exists());

        // Clean up
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_export_statistical_csv() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        // Create a benchmark report with statistical data
        let mut report = BenchmarkReport::new();
        report.add_result(ValidationResult::new(
            "600",
            MetricType::AnnualHeating,
            6.0,
            5.5,
            7.0,
        ));
        report.add_result(ValidationResult::new(
            "600",
            MetricType::AnnualCooling,
            4.0,
            3.5,
            4.5,
        ));

        // Add statistical metrics
        report.statistical_metrics = Some(StatisticalMetrics {
            nmbe: 2.3,
            cv_rmse: 8.7,
            nmbe_ci: (1.5, 3.1),
            cv_rmse_ci: (7.2, 10.2),
            cohens_d: 0.42,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        });

        // Add p-values and BH correction
        report.statistical_p_values = Some(vec![0.023, 0.089]);
        report.statistical_corrected = Some(vec![true, false]);

        // Export to CSV
        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join("test_statistical_export.csv");

        let result = ValidationReportGenerator::export_statistical_csv(&report, &csv_path);

        assert!(result.is_ok(), "Should export CSV successfully");

        // Verify file was created
        assert!(csv_path.exists(), "CSV file should exist");

        // Verify content
        let content = fs::read_to_string(&csv_path).expect("Should read CSV file");

        // Verify header
        assert!(content.contains("case_id,metric_type,predicted,reference_midpoint,nmbe,cv_rmse"));
        assert!(content.contains("ci_nmbe_lower,ci_nmbe_upper,ci_cvrmse_lower,ci_cvrmse_upper"));
        assert!(content.contains("p_value,bh_corrected"));

        // Verify data rows
        assert!(content.contains("600,Annual Heating"));
        assert!(content.contains("600,Annual Cooling"));

        // Verify statistical values
        assert!(content.contains("2.3")); // NMBE
        assert!(content.contains("8.7")); // CV(RMSE)
        assert!(content.contains("0.023")); // p-value
                                            // Note: Cohen's d is only in JSON export, not CSV

        // Clean up
        let _ = fs::remove_file(csv_path);
    }

    #[test]
    fn test_export_statistical_json() {
        use crate::validation::statistical::{
            EffectDirection, StatisticalMetrics, ValidationGroup,
        };

        // Create a benchmark report with statistical data
        let mut report = BenchmarkReport::new();
        report.add_result(ValidationResult::new(
            "600",
            MetricType::AnnualHeating,
            6.0,
            5.5,
            7.0,
        ));

        // Add statistical metrics
        report.statistical_metrics = Some(StatisticalMetrics {
            nmbe: 2.3,
            cv_rmse: 8.7,
            nmbe_ci: (1.5, 3.1),
            cv_rmse_ci: (7.2, 10.2),
            cohens_d: 0.42,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        });

        // Add group validation
        let mut group_validation = std::collections::HashMap::new();
        group_validation.insert(ValidationGroup::Baseline, true);
        group_validation.insert(ValidationGroup::HighMass, false);
        report.group_validation = Some(group_validation);

        // Export to JSON
        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("test_statistical_export.json");

        let result = ValidationReportGenerator::export_statistical_json(&report, &json_path);

        assert!(result.is_ok(), "Should export JSON successfully");

        // Verify file was created
        assert!(json_path.exists(), "JSON file should exist");

        // Verify content
        let content = fs::read_to_string(&json_path).expect("Should read JSON file");
        let json: serde_json::Value = serde_json::from_str(&content).expect("Should parse JSON");

        // Verify structure
        assert!(json.get("statistical_metrics").is_some());
        assert!(json.get("group_validation").is_some());
        assert!(json.get("metadata").is_some());

        // Verify statistical metrics
        let metrics = &json["statistical_metrics"];
        assert_eq!(metrics["nmbe"], 2.3);
        assert_eq!(metrics["cv_rmse"], 8.7);
        assert_eq!(metrics["cohens_d"], 0.42);
        assert_eq!(metrics["excluded_cases"], 0);

        // Verify confidence intervals
        assert!(metrics.get("nmbe_ci").is_some());
        assert!(metrics.get("cv_rmse_ci").is_some());

        // Verify group validation
        let groups = &json["group_validation"];
        assert!(groups.get("Baseline").is_some());
        assert!(groups.get("HighMass").is_some());

        // Verify metadata
        let metadata = &json["metadata"];
        assert_eq!(metadata["alpha"], 0.05);
        assert_eq!(metadata["fdr_method"], "Benjamini-Hochberg");

        // Clean up
        let _ = fs::remove_file(json_path);
    }

    #[test]
    fn test_export_statistical_formats_align_with_ashrae_guideline14() {
        use crate::validation::statistical::StatisticalMetrics;

        // Test 4: Verify formats align with ASHRAE Guideline 14 conventions
        let mut report = BenchmarkReport::new();
        report.add_result(ValidationResult::new(
            "600",
            MetricType::AnnualHeating,
            6.0,
            5.5,
            7.0,
        ));

        report.statistical_metrics = Some(StatisticalMetrics {
            nmbe: 1.5,
            cv_rmse: 5.2,
            nmbe_ci: (0.8, 2.2),
            cv_rmse_ci: (4.1, 6.3),
            cohens_d: 0.28,
            effect_direction: crate::validation::statistical::EffectDirection::Underprediction,
            excluded_cases: 0,
        });

        // Export CSV and verify metric names match ASHRAE Guideline 14
        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join("test_ashrae14.csv");

        let csv_result = ValidationReportGenerator::export_statistical_csv(&report, &csv_path);
        assert!(csv_result.is_ok());

        let csv_content = fs::read_to_string(&csv_path).expect("Should read CSV");
        assert!(csv_content.contains("nmbe")); // Normalized Mean Bias Error
        assert!(csv_content.contains("cv_rmse")); // Coefficient of Variation of RMSE

        // Export JSON and verify metadata includes alpha threshold
        let json_path = temp_dir.join("test_ashrae14.json");
        let json_result = ValidationReportGenerator::export_statistical_json(&report, &json_path);
        assert!(json_result.is_ok());

        let json_content = fs::read_to_string(&json_path).expect("Should read JSON");
        let json: serde_json::Value =
            serde_json::from_str(&json_content).expect("Should parse JSON");
        assert_eq!(json["metadata"]["alpha"], 0.05); // ASHRAE Guideline 14 uses α = 0.05

        // Clean up
        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(json_path);
    }
}
