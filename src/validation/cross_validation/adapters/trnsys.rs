//! TRNSYS Cross-Validation Adapter
//!
//! This module provides functionality to compare Fluxion results
//! against TRNSYS reference simulations.

use super::super::{ComparisonReport, CrossValidationAdapter, ValidationResults};
use crate::validation::ashrae140::ASHRAE140Case;
use anyhow::Result;
use csv::Reader;
use std::path::Path;

/// Adapter for TRNSYS cross-validation
pub struct TRNSYSAdapter;

impl CrossValidationAdapter for TRNSYSAdapter {
    fn tool_name(&self) -> &str {
        "TRNSYS"
    }

    fn load_reference_results(
        &self,
        case: ASHRAE140Case,
        path: &Path,
    ) -> Result<ValidationResults> {
        let content = std::fs::read_to_string(path)?;
        let mut results = ValidationResults::new(case);

        // Handle CSV format
        if content.contains(",") {
            let mut reader = Reader::from_path(path)?;
            for record in reader.records() {
                let record = record?;
                // Parse TRNSYS CSV format
                // Typically: Time, Zone1 Temp, Zone1 Heating, Zone1 Cooling, etc.
                if record.len() >= 4 {
                    let hour = record.get(0).unwrap().parse::<u32>().unwrap_or(0);
                    let zone1_temp = record.get(1).unwrap().parse::<f64>().unwrap_or(0.0);
                    let heating = record.get(2).unwrap().parse::<f64>().unwrap_or(0.0);
                    let cooling = record.get(3).unwrap().parse::<f64>().unwrap_or(0.0);

                    results.add_hourly_data(hour, zone1_temp, heating, cooling);
                }
            }
        }
        // Handle fixed-width format
        else {
            for (line_idx, line) in content.lines().skip(1).enumerate() {
                // Skip header
                // Parse fixed-width fields using column positions
                // This would need to be customized based on actual TRNSYS output format
                if line.len() >= 60 {
                    let hour = line[0..10].trim().parse::<u32>().unwrap_or(line_idx as u32);
                    let zone1_temp = line[10..25].trim().parse::<f64>().unwrap_or(0.0);
                    let heating = line[25..40].trim().parse::<f64>().unwrap_or(0.0);
                    let cooling = line[40..55].trim().parse::<f64>().unwrap_or(0.0);

                    results.add_hourly_data(hour, zone1_temp, heating, cooling);
                }
            }
        }

        results.calculate_annual_totals();
        Ok(results)
    }

    fn compare_results(
        &self,
        fluxion: &ValidationResults,
        reference: &ValidationResults,
    ) -> ComparisonReport {
        // TRNSYS may use slightly different tolerance guidelines
        let tolerance = 0.12; // TRNSYS-specific tolerance
        super::super::compare_results(fluxion, reference, self.tool_name())
    }

    fn generate_report(&self, comparison: &ComparisonReport) -> String {
        let detailed_results = self.format_detailed_results(&comparison.details);

        format!(
            "TRNSYS Cross-Validation Report\n
Case: {:?}\nTool: {}\nRMSE: {:.4}\nPercentage Difference: {:.2}%\nMax Deviation: {:.2}\nWithin Tolerance: {}\n\nTRNSYS-Specific Notes:\n- TRNSYS uses different numerical solvers which may cause small differences\n- Focus on overall trends rather than exact hourly matches\n\nDetailed Results:\n{}",
            comparison.case,
            self.tool_name(),
            comparison.rmse,
            comparison.percentage_difference,
            comparison.max_deviation,
            comparison.within_tolerance,
            detailed_results
        )
    }

    fn default_tolerance(&self) -> f64 {
        0.12 // TRNSYS-specific tolerance
    }
}

impl TRNSYSAdapter {
    /// Format detailed hourly results for the report
    fn format_detailed_results(&self, details: &[ComparisonDetail]) -> String {
        let mut report = String::new();
        report.push_str(
            "Hour | Fluxion Temp (°C) | Reference Temp (°C) | Difference (°C) | % Diff\n",
        );
        report.push_str(
            "---- | ----------------- | ------------------- | --------------- | ------\n",
        );

        for detail in details.iter().take(10) {
            // Show first 10 hours as sample
            report.push_str(&format!(
                "{:4} | {:.2} | {:.2} | {:.2} | {:.2}\n",
                detail.hour,
                detail.fluxion_value,
                detail.reference_value,
                detail.difference,
                detail.percentage_diff
            ));
        }

        if details.len() > 10 {
            report.push_str(&format!("... and {} more hours\n", details.len() - 10));
        }

        report
    }
}
