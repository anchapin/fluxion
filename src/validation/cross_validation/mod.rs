//! Cross-Validation Framework for External Tool Comparison
//!
//! This module provides the core framework for comparing Fluxion validation results
//! against external building energy modeling tools like EnergyPlus and TRNSYS.

pub mod adapters;

use crate::validation::ashrae140::ASHRAE140Case;
use crate::validation::ashrae140::ASHRAE140ValidationResults;
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Validation results structure for storing simulation outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub case: ASHRAE140Case,
    pub hourly_temperatures: Vec<f64>,
    pub hourly_heating: Vec<f64>,
    pub hourly_cooling: Vec<f64>,
    pub annual_heating: f64,
    pub annual_cooling: f64,
    pub peak_heating: f64,
    pub peak_cooling: f64,
}

impl ValidationResults {
    /// Create new validation results for a specific case
    pub fn new(case: ASHRAE140Case) -> Self {
        Self {
            case,
            hourly_temperatures: Vec::new(),
            hourly_heating: Vec::new(),
            hourly_cooling: Vec::new(),
            annual_heating: 0.0,
            annual_cooling: 0.0,
            peak_heating: 0.0,
            peak_cooling: 0.0,
        }
    }

    /// Add hourly data to the results
    pub fn add_hourly_data(&mut self, hour: u32, temperature: f64, heating: f64, cooling: f64) {
        // Ensure vectors are large enough
        let index = hour as usize;
        if index >= self.hourly_temperatures.len() {
            self.hourly_temperatures.resize(index + 1, 0.0);
            self.hourly_heating.resize(index + 1, 0.0);
            self.hourly_cooling.resize(index + 1, 0.0);
        }

        self.hourly_temperatures[index] = temperature;
        self.hourly_heating[index] = heating;
        self.hourly_cooling[index] = cooling;
    }

    /// Calculate annual totals from hourly data
    pub fn calculate_annual_totals(&mut self) {
        self.annual_heating = self.hourly_heating.iter().sum();
        self.annual_cooling = self.hourly_cooling.iter().sum();
        self.peak_heating = self
            .hourly_heating
            .iter()
            .fold(0.0, |a, &b| if b > a { b } else { a });
        self.peak_cooling = self
            .hourly_cooling
            .iter()
            .fold(0.0, |a, &b| if b > a { b } else { a });
    }
}

/// Cross-validation result comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub tool_name: String,
    pub case: ASHRAE140Case,
    pub rmse: f64,
    pub percentage_difference: f64,
    pub max_deviation: f64,
    pub within_tolerance: bool,
    pub details: Vec<ComparisonDetail>,
}

/// Detailed hourly comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonDetail {
    pub hour: u32,
    pub fluxion_value: f64,
    pub reference_value: f64,
    pub difference: f64,
    pub percentage_diff: f64,
}

/// Core comparison function
pub fn compare_results(
    fluxion: &ValidationResults,
    reference: &ValidationResults,
    tool_name: &str,
) -> ComparisonReport {
    // Calculate RMSE for temperature comparison
    let rmse = calculate_rmse(&fluxion.hourly_temperatures, &reference.hourly_temperatures);

    // Calculate percentage difference for annual energy
    let annual_heating_diff =
        calculate_percentage_diff(fluxion.annual_heating, reference.annual_heating);
    let annual_cooling_diff =
        calculate_percentage_diff(fluxion.annual_cooling, reference.annual_cooling);
    let avg_percentage_diff = (annual_heating_diff + annual_cooling_diff) / 2.0;

    // Calculate max deviation
    let max_deviation = fluxion
        .hourly_temperatures
        .iter()
        .zip(reference.hourly_temperatures.iter())
        .map(|(f, r)| (f - r).abs())
        .fold(0.0f64, |a, b| a.max(b));

    // Determine if within tolerance (ASHRAE 140: 15% for energy, 1°C for temperature)
    let within_tolerance = avg_percentage_diff <= 15.0 && max_deviation <= 1.0;

    // Generate detailed comparison
    let details = fluxion
        .hourly_temperatures
        .iter()
        .zip(reference.hourly_temperatures.iter())
        .enumerate()
        .map(|(hour, (&fluxion_temp, &ref_temp))| {
            let diff = fluxion_temp - ref_temp;
            let pct_diff = if ref_temp != 0.0 {
                (diff / ref_temp).abs() * 100.0
            } else {
                0.0
            };

            ComparisonDetail {
                hour: hour as u32,
                fluxion_value: fluxion_temp,
                reference_value: ref_temp,
                difference: diff,
                percentage_diff: pct_diff,
            }
        })
        .collect();

    ComparisonReport {
        tool_name: tool_name.to_string(),
        case: fluxion.case,
        rmse,
        percentage_difference: avg_percentage_diff,
        max_deviation,
        within_tolerance,
        details,
    }
}

/// Calculate Root Mean Square Error
pub fn calculate_rmse(fluxion: &[f64], reference: &[f64]) -> f64 {
    if fluxion.len() != reference.len() || fluxion.is_empty() {
        return 0.0;
    }

    let sum_squared: f64 = fluxion
        .iter()
        .zip(reference.iter())
        .map(|(&f, &r)| {
            let diff = f - r;
            diff * diff
        })
        .sum();

    let mean_squared = sum_squared / fluxion.len() as f64;
    mean_squared.sqrt()
}

/// Calculate percentage difference
pub fn calculate_percentage_diff(fluxion: f64, reference: f64) -> f64 {
    if reference == 0.0 {
        if fluxion == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ((fluxion - reference).abs() / reference) * 100.0
    }
}

/// Trait for cross-validation with external building energy modeling tools
pub trait CrossValidationAdapter {
    /// Returns the name of the external tool (e.g., "EnergyPlus", "TRNSYS")
    fn tool_name(&self) -> &str;

    /// Loads reference results from the tool's output file
    ///
    /// # Arguments
    /// * `case` - The ASHRAE 140 case being validated
    /// * `path` - Path to the tool's output file
    ///
    /// # Returns
    /// ValidationResults struct containing the reference data
    fn load_reference_results(&self, case: ASHRAE140Case, path: &Path)
        -> Result<ValidationResults>;

    /// Compares Fluxion results against the reference results
    ///
    /// # Arguments
    /// * `fluxion` - Fluxion validation results
    /// * `reference` - Reference tool results
    ///
    /// # Returns
    /// ComparisonReport with detailed analysis
    fn compare_results(
        &self,
        fluxion: &ValidationResults,
        reference: &ValidationResults,
    ) -> ComparisonReport;

    /// Generates a human-readable comparison report
    ///
    /// # Arguments
    /// * `comparison` - The comparison results
    ///
    /// # Returns
    /// Formatted string report
    fn generate_report(&self, comparison: &ComparisonReport) -> String;

    /// Default tolerance for this tool (from ASHRAE 140 guidelines)
    fn default_tolerance(&self) -> f64 {
        0.15 // 15% tolerance per ASHRAE 140
    }
}

/// Cross-validation report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationReport {
    pub case: ASHRAE140Case,
    pub tool: String,
    pub fluxion_results: ASHRAE140ValidationResults,
    pub reference_results: ValidationResults,
    pub comparison: ComparisonReport,
    pub report: String,
}

/// Batch validation summary
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchValidationSummary {
    pub successful: usize,
    pub failed: usize,
    pub avg_rmse: f64,
    pub cases: Vec<u32>,
    pub tools: Vec<String>,
}

/// Perform cross-validation against external tool references
pub fn perform_cross_validation(
    case: ASHRAE140Case,
    tool: &str,
    reference_file: &Path,
    _tolerance: Option<f64>,
) -> Result<CrossValidationReport> {
    // Run Fluxion validation to get results
    let fluxion_results = crate::validation::ashrae140::run_validation(case)?;

    // Select appropriate adapter
    let adapter: Box<dyn CrossValidationAdapter> = match tool.to_lowercase().as_str() {
        "energyplus" => Box::new(adapters::EnergyPlusAdapter),
        "trnsys" => Box::new(adapters::TRNSYSAdapter),
        _ => return Err(anyhow!("Unsupported tool: {}", tool)),
    };

    // Load reference results from file
    let reference_results = adapter.load_reference_results(case, reference_file)?;

    // Convert ASHRAE140ValidationResults to ValidationResults for compatibility
    let fluxion_results_compat = ValidationResults {
        case: fluxion_results.case,
        hourly_temperatures: fluxion_results.hourly_temperatures.clone(),
        hourly_heating: fluxion_results.hourly_heating.clone(),
        hourly_cooling: fluxion_results.hourly_cooling.clone(),
        annual_heating: fluxion_results.annual_heating.clone(),
        annual_cooling: fluxion_results.annual_cooling.clone(),
        peak_heating: fluxion_results.peak_heating.clone(),
        peak_cooling: fluxion_results.peak_cooling.clone(),
    };

    // Compare Fluxion results against reference
    let comparison = adapter.compare_results(&fluxion_results_compat, &reference_results);

    // Generate comprehensive report
    let report = adapter.generate_report(&comparison);

    Ok(CrossValidationReport {
        case,
        tool: tool.to_string(),
        fluxion_results,
        reference_results,
        comparison,
        report,
    })
}

/// Run cross-validation for multiple cases
pub fn batch_cross_validate(
    cases: &[u32],
    tool: &str,
    reference_dir: &str,
    output_dir: &str,
    _parallel: usize,
) -> Result<BatchValidationSummary> {
    // Convert case numbers to ASHRAE140Case enum
    let case_enums: Vec<ASHRAE140Case> = cases
        .iter()
        .map(|&case_num| {
            ASHRAE140Case::from_case_id(&case_num.to_string())
                .expect(&format!("Case {} not found", case_num))
        })
        .collect();

    // Use Rayon for parallel processing
    let results: Vec<_> = case_enums
        .par_iter()
        .map(|case| {
            let reference_file =
                Path::new(reference_dir).join(format!("case_{:03}.csv", case.number()));
            perform_cross_validation(*case, tool, &reference_file, None)
        })
        .collect::<Result<Vec<_>>>()?;

    // Save individual reports
    for result in &results {
        let report_path = Path::new(output_dir).join(format!(
            "comparison_case_{}_{}.txt",
            result.case.number(),
            result.tool
        ));
        std::fs::write(report_path, &result.report)?;
    }

    // Generate summary report
    let summary = generate_batch_summary(&results);
    let summary_path = Path::new(output_dir).join("batch_summary.json");
    std::fs::write(summary_path, serde_json::to_string_pretty(&summary)?)?;

    Ok(summary)
}

/// Generate batch summary from cross-validation results
fn generate_batch_summary(results: &[CrossValidationReport]) -> BatchValidationSummary {
    let successful = results.len();
    let failed = 0; // Placeholder - will be enhanced with actual failure detection

    let avg_rmse = results.iter().map(|r| r.comparison.rmse).sum::<f64>() / results.len() as f64;

    let cases = results
        .iter()
        .map(|r| r.case.number().parse::<u32>().unwrap_or(0))
        .collect();

    let tools = results.iter().map(|r| r.tool.clone()).collect();

    BatchValidationSummary {
        successful,
        failed,
        avg_rmse,
        cases,
        tools,
    }
}
