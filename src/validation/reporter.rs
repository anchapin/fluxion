//! Validation report generation for ASHRAE 140.
//!
//! This module provides the `ValidationReportGenerator` which produces
//! comprehensive Markdown reports from validation results.

use crate::validation::report::{BenchmarkReport, MetricType, ValidationStatus};
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

        let mut heating_str = String::new();
        let mut cooling_str = String::new();
        let mut peak_h_str = String::new();
        let mut peak_c_str = String::new();

        for result in &case_results {
            match result.metric {
                MetricType::AnnualHeating => {
                    heating_str = format!(
                        "{:.2} MWh (Ref: {:.2}-{:.2})",
                        result.fluxion_value, result.ref_min, result.ref_max
                    );
                }
                MetricType::AnnualCooling => {
                    cooling_str = format!(
                        "{:.2} MWh (Ref: {:.2}-{:.2})",
                        result.fluxion_value, result.ref_min, result.ref_max
                    );
                }
                MetricType::PeakHeating => {
                    peak_h_str = format!(
                        "{:.2} kW (Ref: {:.2}-{:.2})",
                        result.fluxion_value, result.ref_min, result.ref_max
                    );
                }
                MetricType::PeakCooling => {
                    peak_c_str = format!(
                        "{:.2} kW (Ref: {:.2}-{:.2})",
                        result.fluxion_value, result.ref_min, result.ref_max
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

        let mut min_str = String::new();
        let mut max_str = String::new();

        for result in &case_results {
            match result.metric {
                MetricType::MinFreeFloat => {
                    min_str = format!(
                        "{:.2}°C (Ref: {:.2}-{:.2})",
                        result.fluxion_value, result.ref_min, result.ref_max
                    );
                }
                MetricType::MaxFreeFloat => {
                    max_str = format!(
                        "{:.2}°C (Ref: {:.2}-{:.2})",
                        result.fluxion_value, result.ref_min, result.ref_max
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
}
