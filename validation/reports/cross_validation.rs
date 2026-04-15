// validation/reports/cross_validation.rs
/// Cross-validation reporting
///
/// Generates structured reports comparing Fluxion results with reference data
/// from multiple simulation tools (ESP-r, EnergyPlus, etc.)
use serde::Serialize;
use std::collections::HashMap;

/// Cross-validation report structure
#[derive(Debug, Serialize)]
pub struct CrossValidationReport {
    /// Overall validation pass/fail status
    pub overall_pass: bool,
    /// Individual zone comparison results
    pub zone_results: Vec<crate::validation::esp_r::comparison::ComparisonResult>,
    /// Summary statistics
    pub summary_statistics: SummaryStatistics,
}

/// Summary statistics for cross-validation
#[derive(Debug, Serialize)]
pub struct SummaryStatistics {
    /// Mean temperature difference
    pub mean_temp_difference: f64,
    /// Maximum temperature difference
    pub max_temp_difference: f64,
    /// Mean heating load difference
    pub mean_heating_difference: f64,
    /// Maximum heating load difference
    pub max_heating_difference: f64,
    /// Pass rate (0.0 to 1.0)
    pub pass_rate: f64,
}

/// Generate cross-validation report
///
/// # Arguments
/// * `comparison_results` - Comparison results from zone-by-zone analysis
/// * `tolerance` - Temperature tolerance used for validation
///
/// # Returns
/// Structured cross-validation report
pub fn generate_report(
    comparison_results: Vec<crate::validation::esp_r::comparison::ComparisonResult>,
    tolerance: f64,
) -> CrossValidationReport {
    // Calculate summary statistics
    let temp_diffs: Vec<f64> = comparison_results
        .iter()
        .filter(|r| r.temp_difference.is_finite())
        .map(|r| r.temp_difference)
        .collect();

    let heating_diffs: Vec<f64> = comparison_results
        .iter()
        .filter(|r| r.heating_difference.is_finite())
        .map(|r| r.heating_difference)
        .collect();

    let mean_temp_diff = temp_diffs.iter().sum::<f64>() / temp_diffs.len() as f64;
    let max_temp_diff = temp_diffs.iter().copied().fold(f64::NAN, f64::max);

    let mean_heating_diff = heating_diffs.iter().sum::<f64>() / heating_diffs.len() as f64;
    let max_heating_diff = heating_diffs.iter().copied().fold(f64::NAN, f64::max);

    let pass_count = comparison_results
        .iter()
        .filter(|r| r.temp_within_tolerance && r.heating_within_tolerance)
        .count();
    let pass_rate = pass_count as f64 / comparison_results.len() as f64;

    CrossValidationReport {
        overall_pass: pass_rate >= 0.95, // 95% pass rate required
        zone_results: comparison_results,
        summary_statistics: SummaryStatistics {
            mean_temp_difference: mean_temp_diff,
            max_temp_difference: max_temp_diff,
            mean_heating_difference: mean_heating_diff,
            max_heating_difference: max_heating_diff,
            pass_rate,
        },
    }
}

/// Generate Markdown report for human-readable output
///
/// # Arguments
/// * `report` - Cross-validation report
///
/// # Returns
/// Markdown formatted report string
pub fn generate_markdown_report(report: &CrossValidationReport) -> String {
    let mut md = String::new();

    md.push_str(&format!("# Cross-Validation Report\n\n"));
    md.push_str(&format!(
        "**Overall Status:** {}\n\n",
        if report.overall_pass {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    ));

    md.push_str("**Summary Statistics:**\n");
    md.push_str(&format!(
        "- Mean Temperature Difference: {:.2}°C\n",
        report.summary_statistics.mean_temp_difference
    ));
    md.push_str(&format!(
        "- Max Temperature Difference: {:.2}°C\n",
        report.summary_statistics.max_temp_difference
    ));
    md.push_str(&format!(
        "- Mean Heating Difference: {:.2} W\n",
        report.summary_statistics.mean_heating_difference
    ));
    md.push_str(&format!(
        "- Max Heating Difference: {:.2} W\n",
        report.summary_statistics.max_heating_difference
    ));
    md.push_str(&format!(
        "- Pass Rate: {:.1}%\n\n",
        report.summary_statistics.pass_rate * 100.0
    ));

    md.push_str("**Zone Results:**\n\n");
    md.push_str("| Zone | Temp Within Tolerance | Heating Within Tolerance | Temp Diff (°C) | Heating Diff (W) |\n");
    md.push_str("|------|----------------------|-------------------------|---------------|-----------------|\n");

    for zone in &report.zone_results {
        md.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} |\n",
            zone.zone_id,
            if zone.temp_within_tolerance {
                "✅"
            } else {
                "❌"
            },
            if zone.heating_within_tolerance {
                "✅"
            } else {
                "❌"
            },
            zone.temp_difference,
            zone.heating_difference
        ));
    }

    md
}
