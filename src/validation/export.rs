//! CSV export for ASHRAE 140 simulation diagnostics.
//!
//! Provides a command-line tool to export hourly time series data and metadata
//! for external analysis in Python, R, or Excel.

use anyhow::{Context, Result};
use csv::WriterBuilder;
use serde::Serialize;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

use crate::validation::ashrae_140_cases::CaseSpec;
use crate::validation::diagnostic::DiagnosticCollector;
use crate::validation::report::{BenchmarkReport, ValidationResult};

/// CSV exporter for ASHRAE 140 case diagnostics.
///
/// Exports hourly data to per-zone CSV files and metadata JSON.
pub struct CsvExporter {
    output_dir: PathBuf,
    delimiter: char,
}

impl CsvExporter {
    /// Creates a new CSV exporter.
    ///
    /// # Arguments
    /// * `output_dir` - Base directory where CSV files will be written (e.g., "output/csv")
    /// * `delimiter` - CSV field delimiter (default ',' for US/UK, ';' for European format)
    pub fn new(output_dir: PathBuf, delimiter: char) -> Self {
        Self {
            output_dir,
            delimiter,
        }
    }

    /// Exports hourly diagnostics data for a single case.
    ///
    /// Creates one CSV file per zone in `output_dir/{case_id}/`.
    ///
    /// # Arguments
    /// * `case_id` - Case identifier (e.g., "600", "900", "960")
    /// * `collector` - Diagnostic collector containing hourly data
    /// * `spec` - Case specification (unused currently, reserved for future)
    ///
    /// # Returns
    /// `Result<()>` indicating success or error
    pub fn export_diagnostics(
        &self,
        case_id: &str,
        collector: &DiagnosticCollector,
        _spec: &CaseSpec,
    ) -> Result<()> {
        // Create output directory: output_dir/{case_id}/
        let case_dir = self.output_dir.join(case_id);
        fs::create_dir_all(&case_dir).with_context(|| {
            format!("Failed to create output directory: {}", case_dir.display())
        })?;

        // If no hourly data collected, warn and exit early
        if collector.hourly_data.is_empty() {
            eprintln!("Warning: No hourly data collected for case {}", case_id);
            return Ok(());
        }

        let num_zones = collector.hourly_data[0].zone_temps.len();

        // Write one CSV file per zone
        for zone_idx in 0..num_zones {
            let file_path = case_dir.join(format!("case_{}_zone{}.csv", case_id, zone_idx));
            let file = File::create(&file_path)
                .with_context(|| format!("Failed to create CSV file: {}", file_path.display()))?;
            let mut writer = WriterBuilder::new()
                .delimiter(self.delimiter as u8)
                .from_writer(BufWriter::new(file));

            // Write header
            writer.write_record(&[
                "Hour",
                "Month",
                "Day",
                "HourOfDay",
                "Outdoor_Temp",
                "Zone_Temp",
                "Mass_Temp",
                "Solar_Gain",
                "Internal_Load",
                "HVAC_Heating",
                "HVAC_Cooling",
                "Infiltration_Loss",
                "Envelope_Conduction",
            ])?;

            // Write data rows
            for data in &collector.hourly_data {
                let zone_temp = data.zone_temps.get(zone_idx).copied().unwrap_or(0.0);
                let mass_temp = data.mass_temps.get(zone_idx).copied().unwrap_or(0.0);
                let solar = data.solar_gains.get(zone_idx).copied().unwrap_or(0.0);
                let internal = data.internal_loads.get(zone_idx).copied().unwrap_or(0.0);
                let heating = data.hvac_heating.get(zone_idx).copied().unwrap_or(0.0);
                let cooling = data.hvac_cooling.get(zone_idx).copied().unwrap_or(0.0);
                let infil = data.infiltration_loss.get(zone_idx).copied().unwrap_or(0.0);
                let envelope = data
                    .envelope_conduction
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0);

                writer.write_record(&[
                    data.hour.to_string(),
                    data.month.to_string(),
                    data.day.to_string(),
                    data.hour_of_day.to_string(),
                    format!("{:.2}", data.outdoor_temp),
                    format!("{:.2}", zone_temp),
                    format!("{:.2}", mass_temp),
                    format!("{:.2}", solar),
                    format!("{:.2}", internal),
                    format!("{:.2}", heating),
                    format!("{:.2}", cooling),
                    format!("{:.2}", infil),
                    format!("{:.2}", envelope),
                ])?;
            }

            writer.flush()?;
        }

        Ok(())
    }

    /// Exports metadata JSON for a case.
    ///
    /// The metadata includes:
    /// - Case specification (geometry, construction, HVAC)
    /// - Validation results (pass/fail for each metric)
    /// - Energy breakdown and peak timing (if available)
    ///
    /// # Arguments
    /// * `case_id` - Case identifier
    /// * `spec` - Case specification
    /// * `report` - Benchmark report containing validation results
    /// * `collector` - Diagnostic collector (for energy breakdown, peak timing)
    ///
    /// # Returns
    /// `Result<()>` indicating success or error
    pub fn export_metadata(
        &self,
        case_id: &str,
        spec: &CaseSpec,
        report: &BenchmarkReport,
        collector: &DiagnosticCollector,
    ) -> Result<()> {
        let case_dir = self.output_dir.join(case_id);
        let meta_path = case_dir.join("metadata.json");

        #[derive(Serialize)]
        struct Metadata {
            case_id: String,
            case_spec: CaseSpec,
            validation_results: Vec<ValidationResult>,
            energy_breakdown: Option<crate::validation::diagnostic::EnergyBreakdown>,
            peak_timing: Option<crate::validation::diagnostic::PeakTiming>,
            export_info: ExportInfo,
        }

        #[derive(Serialize)]
        struct ExportInfo {
            delimiter: char,
            columns: Vec<&'static str>,
        }

        // Extract validation results for this case
        let validation_results: Vec<ValidationResult> = report
            .results
            .iter()
            .filter(|r| r.case_id == case_id)
            .cloned()
            .collect();

        // Get energy breakdown and peak timing from collector if available
        let energy_breakdown = collector.energy_breakdowns.get(case_id).cloned();
        let peak_timing = collector.peak_timings.get(case_id).cloned();

        let metadata = Metadata {
            case_id: case_id.to_string(),
            case_spec: spec.clone(),
            validation_results,
            energy_breakdown,
            peak_timing,
            export_info: ExportInfo {
                delimiter: self.delimiter,
                columns: vec![
                    "Hour",
                    "Month",
                    "Day",
                    "HourOfDay",
                    "Outdoor_Temp",
                    "Zone_Temp",
                    "Mass_Temp",
                    "Solar_Gain",
                    "Internal_Load",
                    "HVAC_Heating",
                    "HVAC_Cooling",
                    "Infiltration_Loss",
                    "Envelope_Conduction",
                ],
            },
        };

        let json =
            serde_json::to_string_pretty(&metadata).context("Failed to serialize metadata")?;
        std::fs::write(&meta_path, json)
            .with_context(|| format!("Failed to write metadata file: {}", meta_path.display()))?;

        Ok(())
    }
}
