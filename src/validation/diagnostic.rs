//! Diagnostic output and debugging tools for ASHRAE 140 validation.
//!
//! This module provides diagnostic capabilities for debugging ASHRAE 140 validation
//! issues, including hourly output logging, energy breakdown, peak timing, and
//! temperature profile export.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluxion::validation::diagnostic::DiagnosticConfig;
//!
//! // Enable diagnostics via environment variable
//! std::env::set_var("ASHRAE_140_DEBUG", "1");
//!
//! // Or configure programmatically
//! let config = DiagnosticConfig {
//!     enabled: true,
//!     hourly_output: Some("hourly_output.csv".to_string()),
//!     energy_breakdown: true,
//!     peak_timing: true,
//!     ..Default::default()
//! };
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;

/// Configuration for diagnostic output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticConfig {
    /// Enable diagnostic output
    pub enabled: bool,
    /// Output file for hourly data (CSV format)
    pub hourly_output: Option<String>,
    /// Output file for temperature profiles
    pub temperature_profile: Option<String>,
    /// Enable energy breakdown output
    pub energy_breakdown: bool,
    /// Enable peak timing output
    pub peak_timing: bool,
    /// Enable validation comparison table
    pub comparison_table: bool,
    /// Verbose output (print to console)
    pub verbose: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        // Check environment variable
        let env_debug = std::env::var("ASHRAE_140_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        DiagnosticConfig {
            enabled: env_debug,
            hourly_output: std::env::var("ASHRAE_140_HOURLY_OUTPUT").ok(),
            temperature_profile: std::env::var("ASHRAE_140_TEMP_PROFILE").ok(),
            energy_breakdown: true,
            peak_timing: true,
            comparison_table: true,
            verbose: env_debug,
        }
    }
}

impl DiagnosticConfig {
    /// Creates a new diagnostic configuration with all features disabled.
    pub fn disabled() -> Self {
        DiagnosticConfig {
            enabled: false,
            hourly_output: None,
            temperature_profile: None,
            energy_breakdown: false,
            peak_timing: false,
            comparison_table: false,
            verbose: false,
        }
    }

    /// Creates a diagnostic configuration with all features enabled.
    pub fn full() -> Self {
        DiagnosticConfig {
            enabled: true,
            hourly_output: Some("hourly_output.csv".to_string()),
            temperature_profile: Some("temperature_profile.csv".to_string()),
            energy_breakdown: true,
            peak_timing: true,
            comparison_table: true,
            verbose: true,
        }
    }

    /// Creates a diagnostic configuration for debugging (console output only).
    pub fn debug() -> Self {
        DiagnosticConfig {
            enabled: true,
            hourly_output: None,
            temperature_profile: None,
            energy_breakdown: true,
            peak_timing: true,
            comparison_table: true,
            verbose: true,
        }
    }
}

/// Hourly data for a single timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyData {
    /// Hour index (0-8759)
    pub hour: usize,
    /// Month (1-12)
    pub month: u32,
    /// Day of month (1-31)
    pub day: u32,
    /// Hour of day (0-23)
    pub hour_of_day: u32,
    /// Outdoor temperature (°C)
    pub outdoor_temp: f64,
    /// Zone temperatures (°C) - one per zone
    pub zone_temps: Vec<f64>,
    /// Mass temperatures (°C) - one per zone
    pub mass_temps: Vec<f64>,
    /// Solar gains (W) - one per zone
    pub solar_gains: Vec<f64>,
    /// Internal loads (W) - one per zone
    pub internal_loads: Vec<f64>,
    /// HVAC heating power (W) - one per zone
    pub hvac_heating: Vec<f64>,
    /// HVAC cooling power (W) - one per zone
    pub hvac_cooling: Vec<f64>,
    /// Infiltration heat loss (W) - one per zone
    pub infiltration_loss: Vec<f64>,
    /// Envelope conduction (W) - one per zone
    pub envelope_conduction: Vec<f64>,
}

impl HourlyData {
    /// Creates a new hourly data record with all values initialized to zero.
    pub fn new(hour: usize, num_zones: usize) -> Self {
        let (month, day, hour_of_day) = hour_to_date(hour);

        HourlyData {
            hour,
            month,
            day,
            hour_of_day,
            outdoor_temp: 0.0,
            zone_temps: vec![0.0; num_zones],
            mass_temps: vec![0.0; num_zones],
            solar_gains: vec![0.0; num_zones],
            internal_loads: vec![0.0; num_zones],
            hvac_heating: vec![0.0; num_zones],
            hvac_cooling: vec![0.0; num_zones],
            infiltration_loss: vec![0.0; num_zones],
            envelope_conduction: vec![0.0; num_zones],
        }
    }

    /// Returns total HVAC power (heating - cooling) for all zones.
    pub fn total_hvac_power(&self) -> f64 {
        let heating: f64 = self.hvac_heating.iter().sum();
        let cooling: f64 = self.hvac_cooling.iter().sum();
        heating + cooling // cooling is negative
    }

    /// Returns total solar gains for all zones.
    pub fn total_solar_gains(&self) -> f64 {
        self.solar_gains.iter().sum()
    }

    /// Returns total internal loads for all zones.
    pub fn total_internal_loads(&self) -> f64 {
        self.internal_loads.iter().sum()
    }
}

/// Convert hour index to (month, day, hour_of_day).
fn hour_to_date(hour: usize) -> (u32, u32, u32) {
    let hour_of_day = (hour % 24) as u32;
    let day_of_year = hour / 24 + 1;

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

    (month as u32, day as u32, hour_of_day)
}

/// Energy breakdown for a case simulation.
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

    /// Calculates the total energy in the breakdown.
    pub fn total_input_mwh(&self) -> f64 {
        self.solar_gains_mwh + self.internal_gains_mwh
    }

    /// Calculates the total energy losses.
    pub fn total_loss_mwh(&self) -> f64 {
        self.envelope_conduction_mwh + self.infiltration_mwh
    }

    /// Prints the energy breakdown to stdout.
    pub fn print(&self, case_id: &str) {
        println!("\nCase {} Energy Breakdown:", case_id);
        println!(
            "  Envelope conduction: {:.3} MWh",
            self.envelope_conduction_mwh
        );
        println!("  Infiltration:        {:.3} MWh", self.infiltration_mwh);
        println!("  Solar gains:         {:.3} MWh", self.solar_gains_mwh);
        println!("  Internal gains:      {:.3} MWh", self.internal_gains_mwh);
        println!("  ─────────────────────────────────");
        println!("  Heating energy:      {:.3} MWh", self.heating_mwh);
        println!("  Cooling energy:      {:.3} MWh", self.cooling_mwh);
        println!("  Net balance:         {:.3} MWh", self.net_balance_mwh);
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
        PeakTiming {
            peak_heating_kw: 0.0,
            peak_heating_hour: 0,
            peak_cooling_kw: 0.0,
            peak_cooling_hour: 0,
        }
    }

    /// Returns a formatted string for the peak heating time.
    pub fn peak_heating_time_str(&self) -> String {
        let (month, day, hour) = hour_to_date(self.peak_heating_hour);
        format!("Month {:02} Day {:02}, {:02}:00", month, day, hour)
    }

    /// Returns a formatted string for the peak cooling time.
    pub fn peak_cooling_time_str(&self) -> String {
        let (month, day, hour) = hour_to_date(self.peak_cooling_hour);
        format!("Month {:02} Day {:02}, {:02}:00", month, day, hour)
    }

    /// Prints the peak timing to stdout.
    pub fn print(&self, case_id: &str) {
        println!("\nCase {} Peak Load Timing:", case_id);
        println!(
            "  Peak Heating: {:.2} kW at Hour {} ({})",
            self.peak_heating_kw,
            self.peak_heating_hour,
            self.peak_heating_time_str()
        );
        println!(
            "  Peak Cooling: {:.2} kW at Hour {} ({})",
            self.peak_cooling_kw,
            self.peak_cooling_hour,
            self.peak_cooling_time_str()
        );
    }
}

impl Default for PeakTiming {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation comparison table row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    /// Case identifier
    pub case_id: String,
    /// Metric type
    pub metric: String,
    /// Fluxion calculated value
    pub fluxion_value: f64,
    /// Reference minimum value
    pub ref_min: f64,
    /// Reference maximum value
    pub ref_max: f64,
    /// Pass/fail status
    pub status: String,
    /// Percent deviation from reference midpoint
    pub deviation_percent: f64,
}

/// Diagnostic collector for accumulating simulation data.
#[derive(Debug, Clone)]
pub struct DiagnosticCollector {
    /// Configuration
    pub config: DiagnosticConfig,
    /// Hourly data records
    pub hourly_data: Vec<HourlyData>,
    /// Energy breakdown per case
    pub energy_breakdowns: HashMap<String, EnergyBreakdown>,
    /// Peak timing per case
    pub peak_timings: HashMap<String, PeakTiming>,
    /// Comparison table rows
    pub comparison_rows: Vec<ComparisonRow>,
    /// Current case being simulated
    pub current_case: String,
    /// Number of zones in current case
    pub num_zones: usize,
}

impl DiagnosticCollector {
    /// Creates a new diagnostic collector with the given configuration.
    pub fn new(config: DiagnosticConfig) -> Self {
        DiagnosticCollector {
            config,
            hourly_data: Vec::new(),
            energy_breakdowns: HashMap::new(),
            peak_timings: HashMap::new(),
            comparison_rows: Vec::new(),
            current_case: String::new(),
            num_zones: 1,
        }
    }

    /// Creates a disabled diagnostic collector.
    pub fn disabled() -> Self {
        Self::new(DiagnosticConfig::disabled())
    }

    /// Creates a diagnostic collector from environment variables.
    pub fn from_env() -> Self {
        Self::new(DiagnosticConfig::default())
    }

    /// Starts a new case simulation.
    pub fn start_case(&mut self, case_id: &str, num_zones: usize) {
        if !self.config.enabled {
            return;
        }

        self.current_case = case_id.to_string();
        self.num_zones = num_zones;
        self.hourly_data.clear();

        if self.config.verbose {
            println!("\n=== Simulating Case {} ===", case_id);
        }
    }

    /// Records hourly data for a timestep.
    pub fn record_hour(&mut self, data: HourlyData) {
        if !self.config.enabled {
            return;
        }

        if self.config.verbose && data.hour.is_multiple_of(1000) {
            println!(
                "  Hour {}: Zone Temp = {:.2}°C, Outdoor = {:.2}°C",
                data.hour,
                data.zone_temps.first().unwrap_or(&0.0),
                data.outdoor_temp
            );
        }

        self.hourly_data.push(data);
    }

    /// Finalizes the current case and computes summaries.
    pub fn finalize_case(&mut self, heating_mwh: f64, cooling_mwh: f64) {
        if !self.config.enabled {
            return;
        }

        // Compute energy breakdown
        let mut breakdown = EnergyBreakdown::new();
        breakdown.heating_mwh = heating_mwh;
        breakdown.cooling_mwh = cooling_mwh;

        for data in &self.hourly_data {
            for zone_idx in 0..self.num_zones {
                breakdown.solar_gains_mwh +=
                    data.solar_gains.get(zone_idx).copied().unwrap_or(0.0) / 1_000_000.0; // Wh to MWh
                breakdown.internal_gains_mwh +=
                    data.internal_loads.get(zone_idx).copied().unwrap_or(0.0) / 1_000_000.0;
                breakdown.infiltration_mwh += data
                    .infiltration_loss
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    / 1_000_000.0;
                breakdown.envelope_conduction_mwh += data
                    .envelope_conduction
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    / 1_000_000.0;
            }
        }

        breakdown.net_balance_mwh = breakdown.solar_gains_mwh + breakdown.internal_gains_mwh
            - breakdown.heating_mwh
            + breakdown.cooling_mwh;

        // Compute peak timing
        let mut peak_timing = PeakTiming::new();
        for data in &self.hourly_data {
            let heating_kw = data.hvac_heating.iter().sum::<f64>() / 1000.0;
            let cooling_kw = data.hvac_cooling.iter().sum::<f64>().abs() / 1000.0;

            if heating_kw > peak_timing.peak_heating_kw {
                peak_timing.peak_heating_kw = heating_kw;
                peak_timing.peak_heating_hour = data.hour;
            }

            if cooling_kw > peak_timing.peak_cooling_kw {
                peak_timing.peak_cooling_kw = cooling_kw;
                peak_timing.peak_cooling_hour = data.hour;
            }
        }

        // Print summaries if enabled
        if self.config.energy_breakdown {
            breakdown.print(&self.current_case);
        }

        if self.config.peak_timing {
            peak_timing.print(&self.current_case);
        }

        // Store results
        self.energy_breakdowns
            .insert(self.current_case.clone(), breakdown);
        self.peak_timings
            .insert(self.current_case.clone(), peak_timing);
    }

    /// Adds a comparison row for the validation table.
    pub fn add_comparison(
        &mut self,
        case_id: &str,
        metric: &str,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
        status: &str,
        deviation_percent: f64,
    ) {
        if !self.config.enabled || !self.config.comparison_table {
            return;
        }

        self.comparison_rows.push(ComparisonRow {
            case_id: case_id.to_string(),
            metric: metric.to_string(),
            fluxion_value,
            ref_min,
            ref_max,
            status: status.to_string(),
            deviation_percent,
        });
    }

    /// Exports hourly data to a CSV file.
    pub fn export_hourly_csv(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Header
        writeln!(
            file,
            "Hour,Month,Day,HourOfDay,OutdoorTemp,ZoneTemp,MassTemp,SolarGain,InternalLoad,HVACHeating,HVACCooling,InfiltrationLoss,EnvelopeConduction"
        )?;

        // Data rows
        for data in &self.hourly_data {
            let zone_temp = data.zone_temps.first().unwrap_or(&0.0);
            let mass_temp = data.mass_temps.first().unwrap_or(&0.0);
            let solar = data.solar_gains.first().unwrap_or(&0.0);
            let internal = data.internal_loads.first().unwrap_or(&0.0);
            let heating = data.hvac_heating.first().unwrap_or(&0.0);
            let cooling = data.hvac_cooling.first().unwrap_or(&0.0);
            let infil = data.infiltration_loss.first().unwrap_or(&0.0);
            let envelope = data.envelope_conduction.first().unwrap_or(&0.0);

            writeln!(
                file,
                "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                data.hour,
                data.month,
                data.day,
                data.hour_of_day,
                data.outdoor_temp,
                zone_temp,
                mass_temp,
                solar,
                internal,
                heating,
                cooling,
                infil,
                envelope
            )?;
        }

        Ok(())
    }

    /// Exports temperature profile to a CSV file.
    pub fn export_temperature_profile(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Header
        write!(file, "Hour,OutdoorTemp")?;
        for i in 0..self.num_zones {
            write!(file, ",Zone{}_Temp", i)?;
        }
        writeln!(file)?;

        // Data rows
        for data in &self.hourly_data {
            write!(file, "{},{:.2}", data.hour, data.outdoor_temp)?;
            for temp in &data.zone_temps {
                write!(file, ",{:.2}", temp)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }

    /// Prints the validation comparison table.
    pub fn print_comparison_table(&self) {
        if !self.config.enabled || !self.config.comparison_table {
            return;
        }

        println!("\n=== Validation Comparison Table ===");
        println!(
            "{:<8} {:<20} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "Case", "Metric", "Fluxion", "Ref Min", "Ref Max", "Status", "Deviation"
        );
        println!("{}", "-".repeat(80));

        for row in &self.comparison_rows {
            println!(
                "{:<8} {:<20} {:>10.2} {:>10.2} {:>10.2} {:>8} {:>+9.1}%",
                row.case_id,
                row.metric,
                row.fluxion_value,
                row.ref_min,
                row.ref_max,
                row.status,
                row.deviation_percent
            );
        }
    }

    /// Saves all diagnostic outputs to files.
    pub fn save_all(&self) -> std::io::Result<()> {
        if let Some(ref path) = self.config.hourly_output {
            self.export_hourly_csv(path)?;
            if self.config.verbose {
                println!("Hourly data exported to: {}", path);
            }
        }

        if let Some(ref path) = self.config.temperature_profile {
            self.export_temperature_profile(path)?;
            if self.config.verbose {
                println!("Temperature profile exported to: {}", path);
            }
        }

        Ok(())
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_config_default() {
        let config = DiagnosticConfig::default();
        // Should be disabled by default (no env var set)
        assert!(!config.enabled || std::env::var("ASHRAE_140_DEBUG").is_ok());
    }

    #[test]
    fn test_diagnostic_config_disabled() {
        let config = DiagnosticConfig::disabled();
        assert!(!config.enabled);
        assert!(config.hourly_output.is_none());
    }

    #[test]
    fn test_diagnostic_config_full() {
        let config = DiagnosticConfig::full();
        assert!(config.enabled);
        assert!(config.hourly_output.is_some());
        assert!(config.energy_breakdown);
    }

    #[test]
    fn test_hourly_data_new() {
        let data = HourlyData::new(0, 2);
        assert_eq!(data.hour, 0);
        assert_eq!(data.month, 1);
        assert_eq!(data.day, 1);
        assert_eq!(data.hour_of_day, 0);
        assert_eq!(data.zone_temps.len(), 2);
    }

    #[test]
    fn test_hour_to_date() {
        // Hour 0 = Jan 1, 00:00
        assert_eq!(hour_to_date(0), (1, 1, 0));

        // Hour 12 = Jan 1, 12:00
        assert_eq!(hour_to_date(12), (1, 1, 12));

        // Hour 24 = Jan 2, 00:00
        assert_eq!(hour_to_date(24), (1, 2, 0));

        // Hour 744 = Feb 1, 00:00 (31 days * 24 hours)
        assert_eq!(hour_to_date(744), (2, 1, 0));

        // Hour 8760 = Last hour of year
        let (month, day, hour) = hour_to_date(8759);
        assert_eq!(month, 12);
        assert_eq!(day, 31);
        assert_eq!(hour, 23);
    }

    #[test]
    fn test_energy_breakdown() {
        let mut breakdown = EnergyBreakdown::new();
        breakdown.heating_mwh = 5.0;
        breakdown.cooling_mwh = 7.0;
        breakdown.solar_gains_mwh = 3.0;
        breakdown.internal_gains_mwh = 1.0;

        assert_eq!(breakdown.total_input_mwh(), 4.0);
    }

    #[test]
    fn test_peak_timing() {
        let mut peak = PeakTiming::new();
        peak.peak_heating_kw = 5.5;
        peak.peak_heating_hour = 123;

        let time_str = peak.peak_heating_time_str();
        assert!(time_str.contains("Month"));
    }

    #[test]
    fn test_diagnostic_collector() {
        let config = DiagnosticConfig::debug();
        let mut collector = DiagnosticCollector::new(config);

        collector.start_case("600", 1);

        let mut data = HourlyData::new(0, 1);
        data.zone_temps[0] = 20.0;
        data.outdoor_temp = 10.0;
        data.hvac_heating[0] = 5000.0;

        collector.record_hour(data);
        collector.finalize_case(0.005, 0.0);

        assert!(collector.energy_breakdowns.contains_key("600"));
        assert!(collector.peak_timings.contains_key("600"));
    }

    #[test]
    fn test_diagnostic_collector_comparison() {
        let config = DiagnosticConfig::debug();
        let mut collector = DiagnosticCollector::new(config);

        collector.add_comparison("600", "Annual Heating", 5.0, 4.30, 5.71, "PASS", 0.0);

        assert_eq!(collector.comparison_rows.len(), 1);
        assert_eq!(collector.comparison_rows[0].case_id, "600");
    }

    #[test]
    fn test_diagnostic_collector_disabled() {
        let mut collector = DiagnosticCollector::disabled();

        collector.start_case("600", 1);

        let data = HourlyData::new(0, 1);
        collector.record_hour(data);
        collector.finalize_case(5.0, 7.0);

        // Should not collect any data when disabled
        assert!(collector.hourly_data.is_empty());
    }

    #[test]
    fn test_export_hourly_csv() {
        let config = DiagnosticConfig::debug();
        let mut collector = DiagnosticCollector::new(config);

        collector.start_case("600", 1);

        let mut data = HourlyData::new(0, 1);
        data.zone_temps[0] = 20.0;
        data.outdoor_temp = 10.0;
        collector.record_hour(data);

        // Export to a temp file
        let result = collector.export_hourly_csv("/tmp/test_hourly.csv");
        assert!(result.is_ok());

        // Verify file exists
        assert!(std::path::Path::new("/tmp/test_hourly.csv").exists());
    }
}
