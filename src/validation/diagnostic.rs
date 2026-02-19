//! Diagnostic output and debugging tools for ASHRAE 140 validation.
//!
//! This module provides detailed diagnostic output to help debug validation
//! discrepancies and understand simulation behavior.
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
        // Check environment variable
        let env_debug = std::env::var("ASHRAE_140_DEBUG")
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
            enabled: env_debug,
            output_hourly,
            hourly_output_path,
            output_energy_breakdown: env_debug,
            output_peak_timing: env_debug,
            output_temperature_profiles: env_debug,
            output_comparison_table: env_debug,
            verbose,
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
        Self::default()
    }

    /// Creates a new diagnostic configuration with all features disabled.
    pub fn disabled() -> Self {
        DiagnosticConfig {
            enabled: false,
            output_hourly: false,
            hourly_output_path: None,
            output_energy_breakdown: false,
            output_peak_timing: false,
            output_temperature_profiles: false,
            output_comparison_table: false,
            verbose: false,
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
    /// Solar gains per zone (W)
    pub solar_gains: Vec<f64>,
    /// HVAC heating power (W) - one per zone
    pub hvac_heating: Vec<f64>,
    /// HVAC cooling power (W) - one per zone
    pub hvac_cooling: Vec<f64>,
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
            hvac_heating: vec![0.0; num_zones],
            hvac_cooling: vec![0.0; num_zones],
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

        let heating_str = self
            .hvac_heating
            .iter()
            .map(|p| format!("{:.2}", p))
            .collect::<Vec<_>>()
            .join(",");

        let cooling_str = self
            .hvac_cooling
            .iter()
            .map(|p| format!("{:.2}", p))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            "{},{},{},{},{:.2},{},{},{},{}\n",
            self.hour,
            self.month,
            self.day,
            self.hour_of_day,
            self.outdoor_temp,
            zone_temps_str,
            solar_str,
            heating_str,
            cooling_str
        )
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
        self.energy_breakdowns
            .insert(case_id.to_string(), breakdown);
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
            output.push_str(
                "|------|---------------|---------------|---------------|------------|\n",
            );
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
            output
                .push_str("| Case | Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |\n");
            output
                .push_str("|------|--------|---------|---------|---------|-----------|--------|\n");
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
        let mut csv = String::from("Hour,Month,Day,HourOfDay,Outdoor_Temp,Zone_Temps,Solar_Gains,HVAC_Heating,HVAC_Cooling\n");
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

        if self.config.verbose && data.hour % 1000 == 0 {
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
        if self.config.output_energy_breakdown {
            breakdown.print(&self.current_case);
        }

        if self.config.output_peak_timing {
            peak_timing.print(&self.current_case);
        }

        // Store results
        self.energy_breakdowns
            .insert(self.current_case.clone(), breakdown);
        self.peak_timings
            .insert(self.current_case.clone(), peak_timing);
    }

    /// Adds a comparison row for the validation table.
    #[allow(clippy::too_many_arguments)]
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
        if !self.config.enabled || !self.config.output_comparison_table {
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
        if !self.config.enabled || !self.config.output_comparison_table {
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
        if let Some(ref path) = self.config.hourly_output_path {
            self.export_hourly_csv(path)?;
            if self.config.verbose {
                println!("Hourly data exported to: {}", path);
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
        assert!(config.hourly_output_path.is_none());
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
        assert_eq!(data.month, 1);
        assert_eq!(data.day, 1);
        assert_eq!(data.hour_of_day, 0);
        assert_eq!(data.zone_temps.len(), 2);
        assert_eq!(data.solar_gains.len(), 2);
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

    #[test]
    fn test_diagnostic_collector() {
        let config = DiagnosticConfig::full();
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
        let config = DiagnosticConfig::full();
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
        let config = DiagnosticConfig::full();
        let mut collector = DiagnosticCollector::new(config);

        collector.start_case("600", 1);

        let mut data = HourlyData::new(0, 1);
        data.zone_temps[0] = 20.0;
        data.outdoor_temp = 10.0;
        collector.record_hour(data);

        // Export to a temp file (cross-platform)
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_hourly.csv");
        let path_str = temp_path.to_string_lossy().into_owned();

        let result = collector.export_hourly_csv(&path_str);
        assert!(result.is_ok());

        // Verify file exists
        assert!(temp_path.exists());

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }
}
