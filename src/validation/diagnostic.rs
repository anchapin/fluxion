//! Diagnostic output and debugging tools for ASHRAE 140 validation.
//!
//! This module provides detailed diagnostic output to help debug validation
//! discrepancies and understand simulation behavior.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for diagnostic output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticConfig {
    /// Enable diagnostic output
    pub enabled: bool,
    /// Output hourly data to file
    pub output_hourly: bool,
    /// Hourly output file path
    pub hourly_output_path: Option<String>,
    /// Output energy breakdown
    pub output_energy_breakdown: bool,
    /// Output peak timing information
    pub output_peak_timing: bool,
    /// Output temperature profiles
    pub output_temperature_profiles: bool,
    /// Output validation comparison table
    pub output_comparison_table: bool,
    /// Verbose output to console
    pub verbose: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_hourly: false,
            hourly_output_path: None,
            output_energy_breakdown: true,
            output_peak_timing: true,
            output_temperature_profiles: true,
            output_comparison_table: true,
            verbose: false,
        }
    }
}

impl DiagnosticConfig {
    /// Creates a new diagnostic config with all features enabled.
    pub fn full() -> Self {
        Self {
            enabled: true,
            output_hourly: true,
            hourly_output_path: Some("hourly_output.csv".to_string()),
            output_energy_breakdown: true,
            output_peak_timing: true,
            output_temperature_profiles: true,
            output_comparison_table: true,
            verbose: true,
        }
    }

    /// Creates a diagnostic config from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("ASHRAE_140_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let output_hourly = std::env::var("ASHRAE_140_HOURLY_OUTPUT")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let hourly_output_path = std::env::var("ASHRAE_140_HOURLY_PATH").ok();

        let verbose = std::env::var("ASHRAE_140_VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Self {
            enabled,
            output_hourly,
            hourly_output_path,
            output_energy_breakdown: enabled,
            output_peak_timing: enabled,
            output_temperature_profiles: enabled,
            output_comparison_table: enabled,
            verbose,
        }
    }
}

/// Hourly data for a single timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyData {
    /// Hour index (0-8759)
    pub hour: usize,
    /// Outdoor temperature (°C)
    pub outdoor_temp: f64,
    /// Zone temperatures (°C) - one per zone
    pub zone_temps: Vec<f64>,
    /// Solar gains per zone (W)
    pub solar_gains: Vec<f64>,
    /// HVAC power per zone (W, positive=heating, negative=cooling)
    pub hvac_power: Vec<f64>,
    /// Internal loads per zone (W)
    pub internal_loads: Vec<f64>,
    /// Infiltration heat loss per zone (W)
    pub infiltration_loss: Vec<f64>,
    /// Envelope conduction per zone (W)
    pub envelope_conduction: Vec<f64>,
}

impl HourlyData {
    /// Creates a new hourly data record.
    pub fn new(hour: usize, num_zones: usize) -> Self {
        Self {
            hour,
            outdoor_temp: 0.0,
            zone_temps: vec![0.0; num_zones],
            solar_gains: vec![0.0; num_zones],
            hvac_power: vec![0.0; num_zones],
            internal_loads: vec![0.0; num_zones],
            infiltration_loss: vec![0.0; num_zones],
            envelope_conduction: vec![0.0; num_zones],
        }
    }

    /// Convert to CSV row
    pub fn to_csv_row(&self) -> String {
        let zone_temps_str = self
            .zone_temps
            .iter()
            .map(|t| format!("{:.2}", t))
            .collect::<Vec<_>>()
            .join(",");

        let solar_str = self
            .solar_gains
            .iter()
            .map(|g| format!("{:.2}", g))
            .collect::<Vec<_>>()
            .join(",");

        let hvac_str = self
            .hvac_power
            .iter()
            .map(|p| format!("{:.2}", p))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            "{},{:.2},{},{},{}\n",
            self.hour, self.outdoor_temp, zone_temps_str, solar_str, hvac_str
        )
    }
}

/// Energy breakdown for a single case.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnergyBreakdown {
    /// Total envelope conduction energy (MWh)
    pub envelope_conduction_mwh: f64,
    /// Total infiltration energy (MWh)
    pub infiltration_mwh: f64,
    /// Total solar gains energy (MWh)
    pub solar_gains_mwh: f64,
    /// Total internal gains energy (MWh)
    pub internal_gains_mwh: f64,
    /// Total heating energy (MWh)
    pub heating_mwh: f64,
    /// Total cooling energy (MWh)
    pub cooling_mwh: f64,
    /// Net energy balance (MWh)
    pub net_balance_mwh: f64,
}

impl EnergyBreakdown {
    /// Creates a new empty energy breakdown.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a formatted string for display
    pub fn to_formatted_string(&self, case_id: &str) -> String {
        format!(
            "Case {} Energy Breakdown:\n\
             | Component            | Energy (MWh) |\n\
             |----------------------|---------------|\n\
             | Envelope Conduction  | {:>13.2} |\n\
             | Infiltration         | {:>13.2} |\n\
             | Solar Gains          | {:>13.2} |\n\
             | Internal Gains       | {:>13.2} |\n\
             |----------------------|---------------|\n\
             | Total Heating        | {:>13.2} |\n\
             | Total Cooling        | {:>13.2} |\n\
             | Net Balance          | {:>13.2} |",
            case_id,
            self.envelope_conduction_mwh,
            self.infiltration_mwh,
            self.solar_gains_mwh,
            self.internal_gains_mwh,
            self.heating_mwh,
            self.cooling_mwh,
            self.net_balance_mwh
        )
    }
}

/// Peak load timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakTiming {
    /// Peak heating load (kW)
    pub peak_heating_kw: f64,
    /// Hour of peak heating (0-8759)
    pub peak_heating_hour: usize,
    /// Peak cooling load (kW)
    pub peak_cooling_kw: f64,
    /// Hour of peak cooling (0-8759)
    pub peak_cooling_hour: usize,
}

impl PeakTiming {
    /// Creates a new peak timing record.
    pub fn new() -> Self {
        Self {
            peak_heating_kw: 0.0,
            peak_heating_hour: 0,
            peak_cooling_kw: 0.0,
            peak_cooling_hour: 0,
        }
    }

    /// Convert hour index to date/time string
    pub fn hour_to_datetime(hour: usize) -> String {
        let day_of_year = hour / 24 + 1;
        let hour_of_day = hour % 24;

        let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut month = 1;
        let mut day = day_of_year;
        for (i, &days) in days_in_month.iter().enumerate() {
            if day <= days as usize {
                month = i + 1;
                break;
            }
            day -= days as usize;
        }

        let month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        let month_name = month_names.get(month - 1).unwrap_or(&"?");

        format!("{} {} {:02}:00", month_name, day, hour_of_day)
    }

    /// Generate a formatted string for display
    pub fn to_formatted_string(&self, case_id: &str) -> String {
        let heating_datetime = Self::hour_to_datetime(self.peak_heating_hour);
        let cooling_datetime = Self::hour_to_datetime(self.peak_cooling_hour);

        format!(
            "Case {} Peak Load Timing:\n\
             | Type    | Peak (kW) | Time                |\n\
             |---------|-----------|---------------------|\n\
             | Heating | {:>9.2} | {} |\n\
             | Cooling | {:>9.2} | {} |",
            case_id, self.peak_heating_kw, heating_datetime, self.peak_cooling_kw, cooling_datetime
        )
    }
}

impl Default for PeakTiming {
    fn default() -> Self {
        Self::new()
    }
}

/// Temperature profile for free-floating cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureProfile {
    /// Case ID
    pub case_id: String,
    /// Minimum temperature (°C)
    pub min_temp: f64,
    /// Maximum temperature (°C)
    pub max_temp: f64,
    /// Average temperature (°C)
    pub avg_temp: f64,
    /// Temperature swing (max - min)
    pub swing: f64,
    /// Hourly temperatures (°C)
    pub hourly_temps: Vec<f64>,
}

impl TemperatureProfile {
    /// Creates a new temperature profile.
    pub fn new(case_id: &str) -> Self {
        Self {
            case_id: case_id.to_string(),
            min_temp: f64::INFINITY,
            max_temp: f64::NEG_INFINITY,
            avg_temp: 0.0,
            swing: 0.0,
            hourly_temps: Vec::new(),
        }
    }

    /// Update statistics with a new temperature reading
    pub fn update(&mut self, temp: f64) {
        self.min_temp = self.min_temp.min(temp);
        self.max_temp = self.max_temp.max(temp);
        self.hourly_temps.push(temp);
    }

    /// Finalize statistics after all data collected
    pub fn finalize(&mut self) {
        if !self.hourly_temps.is_empty() {
            self.avg_temp = self.hourly_temps.iter().sum::<f64>() / self.hourly_temps.len() as f64;
            self.swing = self.max_temp - self.min_temp;
        }
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("Hour,Zone_Temp,Outdoor_Temp\n");
        // Note: outdoor temp would need to be tracked separately
        for (hour, &temp) in self.hourly_temps.iter().enumerate() {
            csv.push_str(&format!("{},{:.2},\n", hour, temp));
        }
        csv
    }
}

/// Validation comparison row for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    /// Case ID
    pub case_id: String,
    /// Metric name
    pub metric: String,
    /// Fluxion calculated value
    pub fluxion_value: f64,
    /// Reference minimum
    pub ref_min: f64,
    /// Reference maximum
    pub ref_max: f64,
    /// Pass/Fail status
    pub status: String,
    /// Percent deviation from midpoint
    pub deviation_percent: f64,
}

impl ComparisonRow {
    /// Creates a new comparison row.
    pub fn new(
        case_id: &str,
        metric: &str,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) -> Self {
        let ref_mid = (ref_min + ref_max) / 2.0;
        let deviation_percent = if ref_mid != 0.0 {
            ((fluxion_value - ref_mid) / ref_mid.abs()) * 100.0
        } else {
            0.0
        };

        let status = if fluxion_value >= ref_min && fluxion_value <= ref_max {
            "PASS"
        } else {
            "FAIL"
        }
        .to_string();

        Self {
            case_id: case_id.to_string(),
            metric: metric.to_string(),
            fluxion_value,
            ref_min,
            ref_max,
            status,
            deviation_percent,
        }
    }

    /// Convert to Markdown table row
    pub fn to_markdown_row(&self) -> String {
        format!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.1}% | {} |",
            self.case_id,
            self.metric,
            self.fluxion_value,
            self.ref_min,
            self.ref_max,
            self.deviation_percent,
            self.status
        )
    }
}

/// Diagnostic report containing all diagnostic information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Configuration used
    pub config: DiagnosticConfig,
    /// Hourly data (if collected)
    pub hourly_data: Vec<HourlyData>,
    /// Energy breakdowns by case
    pub energy_breakdowns: HashMap<String, EnergyBreakdown>,
    /// Peak timing by case
    pub peak_timings: HashMap<String, PeakTiming>,
    /// Temperature profiles by case
    pub temperature_profiles: HashMap<String, TemperatureProfile>,
    /// Comparison table rows
    pub comparison_rows: Vec<ComparisonRow>,
}

impl DiagnosticReport {
    /// Creates a new diagnostic report.
    pub fn new(config: DiagnosticConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Add hourly data
    pub fn add_hourly_data(&mut self, data: HourlyData) {
        self.hourly_data.push(data);
    }

    /// Add energy breakdown
    pub fn add_energy_breakdown(&mut self, case_id: &str, breakdown: EnergyBreakdown) {
        self.energy_breakdowns.insert(case_id.to_string(), breakdown);
    }

    /// Add peak timing
    pub fn add_peak_timing(&mut self, case_id: &str, timing: PeakTiming) {
        self.peak_timings.insert(case_id.to_string(), timing);
    }

    /// Add temperature profile
    pub fn add_temperature_profile(&mut self, profile: TemperatureProfile) {
        self.temperature_profiles
            .insert(profile.case_id.clone(), profile);
    }

    /// Add comparison row
    pub fn add_comparison_row(&mut self, row: ComparisonRow) {
        self.comparison_rows.push(row);
    }

    /// Generate full Markdown report
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# ASHRAE 140 Diagnostic Report\n\n");

        // Energy breakdowns
        if self.config.output_energy_breakdown && !self.energy_breakdowns.is_empty() {
            output.push_str("## Energy Breakdowns\n\n");
            for (case_id, breakdown) in &self.energy_breakdowns {
                output.push_str(&breakdown.to_formatted_string(case_id));
                output.push_str("\n\n");
            }
        }

        // Peak timing
        if self.config.output_peak_timing && !self.peak_timings.is_empty() {
            output.push_str("## Peak Load Timing\n\n");
            for (case_id, timing) in &self.peak_timings {
                output.push_str(&timing.to_formatted_string(case_id));
                output.push_str("\n\n");
            }
        }

        // Temperature profiles
        if self.config.output_temperature_profiles && !self.temperature_profiles.is_empty() {
            output.push_str("## Temperature Profiles (Free-Floating Cases)\n\n");
            output.push_str(
                "| Case | Min Temp (°C) | Max Temp (°C) | Avg Temp (°C) | Swing (°C) |\n",
            );
            output.push_str("|------|---------------|---------------|---------------|------------|\n");
            for (case_id, profile) in &self.temperature_profiles {
                output.push_str(&format!(
                    "| {} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
                    case_id, profile.min_temp, profile.max_temp, profile.avg_temp, profile.swing
                ));
            }
            output.push('\n');
        }

        // Comparison table
        if self.config.output_comparison_table && !self.comparison_rows.is_empty() {
            output.push_str("## Validation Comparison Table\n\n");
            output.push_str(
                "| Case | Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |\n",
            );
            output.push_str("|------|--------|---------|---------|---------|-----------|--------|\n");
            for row in &self.comparison_rows {
                output.push_str(&row.to_markdown_row());
                output.push('\n');
            }
            output.push('\n');
        }

        output
    }

    /// Export hourly data to CSV
    pub fn export_hourly_csv<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut csv = String::from("Hour,Outdoor_Temp,Zone_Temps,Solar_Gains,HVAC_Power\n");
        for data in &self.hourly_data {
            csv.push_str(&data.to_csv_row());
        }
        std::fs::write(path, csv)
    }

    /// Save report to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        let content = match path.extension().and_then(|e| e.to_str()) {
            Some("md") => self.to_markdown(),
            Some("json") => serde_json::to_string_pretty(self).unwrap_or_default(),
            _ => self.to_markdown(),
        };
        std::fs::write(path, content)
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        if !self.config.verbose {
            return;
        }

        println!("\n=== ASHRAE 140 Diagnostic Summary ===\n");

        if !self.energy_breakdowns.is_empty() {
            println!("Energy Breakdowns:");
            for (case_id, breakdown) in &self.energy_breakdowns {
                println!(
                    "  {}: Heating={:.2} MWh, Cooling={:.2} MWh",
                    case_id, breakdown.heating_mwh, breakdown.cooling_mwh
                );
            }
            println!();
        }

        if !self.peak_timings.is_empty() {
            println!("Peak Load Timing:");
            for (case_id, timing) in &self.peak_timings {
                println!(
                    "  {}: Peak Heat={:.2} kW at hour {}, Peak Cool={:.2} kW at hour {}",
                    case_id,
                    timing.peak_heating_kw,
                    timing.peak_heating_hour,
                    timing.peak_cooling_kw,
                    timing.peak_cooling_hour
                );
            }
            println!();
        }

        if !self.temperature_profiles.is_empty() {
            println!("Free-Floating Temperature Ranges:");
            for (case_id, profile) in &self.temperature_profiles {
                println!(
                    "  {}: Min={:.1}°C, Max={:.1}°C, Swing={:.1}°C",
                    case_id, profile.min_temp, profile.max_temp, profile.swing
                );
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_config_default() {
        let config = DiagnosticConfig::default();
        assert!(!config.enabled);
        assert!(!config.output_hourly);
    }

    #[test]
    fn test_diagnostic_config_full() {
        let config = DiagnosticConfig::full();
        assert!(config.enabled);
        assert!(config.output_hourly);
        assert!(config.verbose);
    }

    #[test]
    fn test_hourly_data_creation() {
        let data = HourlyData::new(0, 2);
        assert_eq!(data.hour, 0);
        assert_eq!(data.zone_temps.len(), 2);
        assert_eq!(data.solar_gains.len(), 2);
    }

    #[test]
    fn test_energy_breakdown() {
        let mut breakdown = EnergyBreakdown::new();
        breakdown.heating_mwh = 5.0;
        breakdown.cooling_mwh = 7.0;
        breakdown.envelope_conduction_mwh = 3.0;

        let formatted = breakdown.to_formatted_string("600");
        assert!(formatted.contains("Case 600"));
        assert!(formatted.contains("5.00"));
    }

    #[test]
    fn test_peak_timing() {
        let timing = PeakTiming {
            peak_heating_kw: 5.5,
            peak_heating_hour: 500,
            peak_cooling_kw: 3.2,
            peak_cooling_hour: 4500,
        };

        let formatted = timing.to_formatted_string("600");
        assert!(formatted.contains("5.50"));
        assert!(formatted.contains("3.20"));
    }

    #[test]
    fn test_peak_timing_datetime() {
        // Hour 0 = Jan 1, 00:00
        assert_eq!(PeakTiming::hour_to_datetime(0), "Jan 1 00:00");

        // Hour 500 = Jan 21, 20:00
        assert_eq!(PeakTiming::hour_to_datetime(500), "Jan 21 20:00");

        // Hour 4380 = Jul 1, 12:00 (mid-year)
        let dt = PeakTiming::hour_to_datetime(4380);
        assert!(dt.contains("Jul"));
    }

    #[test]
    fn test_temperature_profile() {
        let mut profile = TemperatureProfile::new("600FF");
        profile.update(15.0);
        profile.update(20.0);
        profile.update(25.0);
        profile.finalize();

        assert_eq!(profile.min_temp, 15.0);
        assert_eq!(profile.max_temp, 25.0);
        assert_eq!(profile.avg_temp, 20.0);
        assert_eq!(profile.swing, 10.0);
    }

    #[test]
    fn test_comparison_row() {
        let row = ComparisonRow::new("600", "Heating", 5.0, 4.30, 5.71);
        assert_eq!(row.case_id, "600");
        assert_eq!(row.metric, "Heating");
        assert_eq!(row.status, "PASS");
    }

    #[test]
    fn test_comparison_row_fail() {
        let row = ComparisonRow::new("600", "Heating", 10.0, 4.30, 5.71);
        assert_eq!(row.status, "FAIL");
    }

    #[test]
    fn test_diagnostic_report() {
        let config = DiagnosticConfig::full();
        let mut report = DiagnosticReport::new(config);

        let breakdown = EnergyBreakdown {
            heating_mwh: 5.0,
            cooling_mwh: 7.0,
            ..Default::default()
        };
        report.add_energy_breakdown("600", breakdown);

        let timing = PeakTiming {
            peak_heating_kw: 5.5,
            peak_heating_hour: 500,
            peak_cooling_kw: 3.2,
            peak_cooling_hour: 4500,
        };
        report.add_peak_timing("600", timing);

        let markdown = report.to_markdown();
        assert!(markdown.contains("ASHRAE 140 Diagnostic Report"));
        assert!(markdown.contains("Energy Breakdowns"));
        assert!(markdown.contains("Peak Load Timing"));
    }
}