//! EnergyPlus Cross-Validation Adapter
//!
//! This module provides functionality to compare Fluxion results
//! against EnergyPlus reference simulations.

use super::super::{ComparisonDetail, ComparisonReport, CrossValidationAdapter, ValidationResults};
use crate::validation::ashrae140::ASHRAE140Case;
use anyhow::Result;
use csv::Reader;
use std::path::Path;

/// Adapter for EnergyPlus cross-validation
pub struct EnergyPlusAdapter;

impl CrossValidationAdapter for EnergyPlusAdapter {
    fn tool_name(&self) -> &str {
        "EnergyPlus"
    }

    fn load_reference_results(
        &self,
        case: ASHRAE140Case,
        path: &Path,
    ) -> Result<ValidationResults> {
        // Parse EnergyPlus CSV output format
        let mut reader = Reader::from_path(path)?;

        let mut results = ValidationResults::new(case);

        for record in reader.records() {
            let record = record?;

            // EnergyPlus CSV format:
            // Date/Time,Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep),
            // ZONE 1:Zone Air Temperature [C](TimeStep),
            // IDEAL LOADS AIR SYSTEM 1:Delivery Air Temperature [C](TimeStep),
            // etc.

            if record.len() >= 3 {
                let hour = record.get(0).unwrap().parse::<u32>().unwrap_or(0);
                let zone1_temp = record.get(2).unwrap().parse::<f64>().unwrap_or(0.0);
                // Parse heating and cooling from appropriate columns
                let heating = if record.len() > 5 {
                    record.get(5).unwrap().parse::<f64>().unwrap_or(0.0)
                } else {
                    0.0
                };
                let cooling = if record.len() > 6 {
                    record.get(6).unwrap().parse::<f64>().unwrap_or(0.0)
                } else {
                    0.0
                };

                results.add_hourly_data(hour, zone1_temp, heating, cooling);
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
        // Use the core comparison function with EnergyPlus-specific tolerance
        let tolerance = self.default_tolerance();
        super::super::compare_results(fluxion, reference, self.tool_name())
    }

    fn generate_report(&self, comparison: &ComparisonReport) -> String {
        let detailed_results = self.format_detailed_results(&comparison.details);

        format!(
            "EnergyPlus Cross-Validation Report\n
Case: {:?}\nTool: {}\nRMSE: {:.4}\nPercentage Difference: {:.2}%\nMax Deviation: {:.2}\nWithin Tolerance: {}\n\nDetailed Results:\n{}",
            comparison.case,
            self.tool_name(),
            comparison.rmse,
            comparison.percentage_difference,
            comparison.max_deviation,
            comparison.within_tolerance,
            detailed_results
        )
    }
}

impl EnergyPlusAdapter {
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
