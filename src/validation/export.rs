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
            writer.write_record([
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
        fs::create_dir_all(&case_dir).with_context(|| {
            format!("Failed to create output directory: {}", case_dir.display())
        })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::diagnostic::{DiagnosticCollector, DiagnosticConfig, HourlyData};
    use crate::validation::report::{
        BenchmarkReport, MetricType, ValidationResult, ValidationStatus,
    };

    #[test]
    fn test_csv_exporter_creation() {
        let exporter = CsvExporter::new(PathBuf::from("/tmp/test_csv"), ',');
        assert_eq!(exporter.delimiter, ',');
    }

    #[test]
    fn test_csv_exporter_semicolon_delimiter() {
        let exporter = CsvExporter::new(PathBuf::from("/tmp/test_csv"), ';');
        assert_eq!(exporter.delimiter, ';');
    }

    #[test]
    fn test_csv_exporter_tab_delimiter() {
        let exporter = CsvExporter::new(PathBuf::from("/tmp/test_csv"), '\t');
        assert_eq!(exporter.delimiter, '\t');
    }

    fn create_test_hourly_data(num_hours: usize, num_zones: usize) -> Vec<HourlyData> {
        let mut data = Vec::new();
        for hour in 0..num_hours {
            let mut record = HourlyData::new(hour, num_zones);
            record.outdoor_temp = 20.0 + (hour as f64 % 24.0) * 0.5;
            for z in 0..num_zones {
                record.zone_temps[z] = 21.0 + (hour as f64 % 10.0) * 0.3;
                record.mass_temps[z] = 20.5 + (hour as f64 % 8.0) * 0.2;
                record.solar_gains[z] = 100.0 + (hour as f64 % 500.0);
                record.internal_loads[z] = 50.0;
                record.hvac_heating[z] = if hour % 2 == 0 { 200.0 } else { 0.0 };
                record.hvac_cooling[z] = if hour % 3 == 0 { 150.0 } else { 0.0 };
                record.infiltration_loss[z] = 30.0;
                record.envelope_conduction[z] = 45.0;
            }
            data.push(record);
        }
        data
    }

    fn create_test_collector(num_hours: usize, num_zones: usize) -> DiagnosticCollector {
        let mut collector = DiagnosticCollector::new(DiagnosticConfig::full());
        collector.hourly_data = create_test_hourly_data(num_hours, num_zones);
        collector
    }

    fn create_minimal_case_spec(case_id: &str) -> CaseSpec {
        use crate::validation::ashrae_140_cases::ASHRAE140Case;
        match case_id {
            "600" => ASHRAE140Case::Case600.spec(),
            "960" => {
                let mut spec = ASHRAE140Case::Case960.spec();
                spec.case_id = "960".to_string();
                spec
            }
            _ => ASHRAE140Case::Case600.spec(),
        }
    }

    #[test]
    fn test_export_diagnostics_single_zone() {
        let temp_dir = std::env::temp_dir().join("fluxion_export_test_single");
        let exporter = CsvExporter::new(temp_dir.clone(), ',');

        let collector = create_test_collector(24, 1);
        let spec = create_minimal_case_spec("600");

        let result = exporter.export_diagnostics("600", &collector, &spec);
        assert!(result.is_ok());

        let csv_path = temp_dir.join("600").join("case_600_zone0.csv");
        assert!(csv_path.exists());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() > 1);
        assert!(lines[0].contains("Hour"));
        assert!(lines[0].contains("Outdoor_Temp"));
        assert!(lines[0].contains("Zone_Temp"));
        assert_eq!(lines.len(), 25);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_csv_exporter_fields() {
        let exporter = CsvExporter::new(PathBuf::from("/tmp/test"), ',');
        assert_eq!(exporter.delimiter, ',');
    }

    #[test]
    fn test_csv_exporter_different_delimiters() {
        let exporter_tab = CsvExporter::new(PathBuf::from("/tmp/test"), '\t');
        assert_eq!(exporter_tab.delimiter, '\t');

        let exporter_pipe = CsvExporter::new(PathBuf::from("/tmp/test"), '|');
        assert_eq!(exporter_pipe.delimiter, '|');
    }

    #[test]
    fn test_csv_exporter_output_dir() {
        let exporter = CsvExporter::new(PathBuf::from("/custom/output"), ';');
        assert_eq!(exporter.output_dir, PathBuf::from("/custom/output"));
    }

    #[test]
    fn test_export_diagnostics_empty_hourly_data() {
        let temp_dir = std::env::temp_dir().join("fluxion_export_test_empty");
        let exporter = CsvExporter::new(temp_dir.clone(), ',');

        let mut collector = DiagnosticCollector::new(DiagnosticConfig::full());
        collector.hourly_data = vec![];
        let spec = create_minimal_case_spec("600");

        let result = exporter.export_diagnostics("600", &collector, &spec);
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_export_metadata() {
        let temp_dir = std::env::temp_dir().join("fluxion_metadata_test");
        let exporter = CsvExporter::new(temp_dir.clone(), ',');

        let collector = create_test_collector(24, 1);
        let spec = create_minimal_case_spec("600");

        let report = BenchmarkReport::default();

        let result = exporter.export_metadata("600", &spec, &report, &collector);
        assert!(result.is_ok());

        let meta_path = temp_dir.join("600").join("metadata.json");
        assert!(meta_path.exists());

        let content = std::fs::read_to_string(&meta_path).unwrap();
        assert!(content.contains("600"));
        assert!(content.contains("export_info"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_export_metadata_with_validation_results() {
        let temp_dir = std::env::temp_dir().join("fluxion_metadata_validation_test");
        let exporter = CsvExporter::new(temp_dir.clone(), ';');

        let collector = create_test_collector(24, 1);
        let spec = create_minimal_case_spec("900");

        let report = BenchmarkReport {
            results: vec![
                ValidationResult {
                    case_id: "900".to_string(),
                    metric: MetricType::AnnualHeating,
                    fluxion_value: 1.5,
                    ref_min: 1.0,
                    ref_max: 2.0,
                    percent_error: 0.0,
                    status: ValidationStatus::Pass,
                    per_program: None,
                    peak_timestamp: None,
                },
                ValidationResult {
                    case_id: "900".to_string(),
                    metric: MetricType::AnnualCooling,
                    fluxion_value: 3.0,
                    ref_min: 2.0,
                    ref_max: 4.0,
                    percent_error: 0.0,
                    status: ValidationStatus::Pass,
                    per_program: None,
                    peak_timestamp: None,
                },
            ],
            ..BenchmarkReport::default()
        };

        let result = exporter.export_metadata("900", &spec, &report, &collector);
        assert!(result.is_ok());

        let meta_path = temp_dir.join("900").join("metadata.json");
        let content = std::fs::read_to_string(&meta_path).unwrap();
        assert!(content.contains("heating"));
        assert!(content.contains("cooling"));
        assert!(content.contains("900"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_export_diagnostics_semicolon_delimiter() {
        let temp_dir = std::env::temp_dir().join("fluxion_export_semicolon");
        let exporter = CsvExporter::new(temp_dir.clone(), ';');

        let collector = create_test_collector(12, 1);
        let spec = create_minimal_case_spec("600");

        let result = exporter.export_diagnostics("600", &collector, &spec);
        assert!(result.is_ok());

        let csv_path = temp_dir.join("600").join("case_600_zone0.csv");
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let first_line = content.lines().next().unwrap();
        assert!(first_line.contains(';'));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_csv_content_has_correct_data_values() {
        let temp_dir = std::env::temp_dir().join("fluxion_export_data_check");
        let exporter = CsvExporter::new(temp_dir.clone(), ',');

        let collector = create_test_collector(24, 1);
        let spec = create_minimal_case_spec("600");

        let result = exporter.export_diagnostics("600", &collector, &spec);
        assert!(result.is_ok());

        let csv_path = temp_dir.join("600").join("case_600_zone0.csv");
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert!(lines[0].contains("Hour"));
        assert!(lines[0].contains("Month"));
        assert!(lines[0].contains("Day"));
        assert!(lines[0].contains("HourOfDay"));
        assert!(lines[0].contains("Solar_Gain"));
        assert!(lines[0].contains("HVAC_Heating"));
        assert!(lines[0].contains("HVAC_Cooling"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
