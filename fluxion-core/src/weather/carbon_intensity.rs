//! Hourly Marginal Carbon Emissions Tracking
//!
//! This module provides carbon intensity profile parsing and accumulation
//! for tracking grid carbon emissions in building energy simulations.
//!
//! # Overview
//!
//! Grid carbon intensity (kg CO2eq/kWh) varies significantly by hour due to
//! the renewable/thermal generation mix — midday solar reduces marginal emissions;
//! evening peak gas turbines increase them.
//!
//! # CSV Format
//!
//! The expected CSV format has columns: `datetime, kg_CO2eq_per_kWh`
//! - `datetime`: ISO 8601 format (e.g., `2024-01-01T00:00:00`)
//! - `kg_CO2eq_per_kWh`: Carbon intensity for that hour
//!
//! # Example
//!
//! ```ignore
//! datetime,kg_CO2eq_per_kWh
//! 2024-01-01T00:00:00,0.45
//! 2024-01-01T01:00:00,0.42
//! ```

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use thiserror::Error;

use crate::weather::interpolation::linear_interpolate;

const HOURS_PER_YEAR: usize = 8760;

#[derive(Error, Debug)]
pub enum CarbonError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV parse error at line {line}: {message}")]
    CsvParse { line: usize, message: String },
    #[error("Invalid hour {hour} (must be 0-8759)")]
    InvalidHour { hour: usize },
    #[error("Profile has {count} hours, expected {expected}")]
    InvalidHourCount { count: usize, expected: usize },
    #[error("No carbon intensity data loaded")]
    NoData,
}

/// Carbon intensity profile for grid electricity (kg CO2eq/kWh).
///
/// Stores hourly carbon intensity values for a full year (8760 hours)
/// and provides interpolation for sub-hourly timesteps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarbonIntensityProfile {
    /// Hourly carbon intensity values (kg CO2eq/kWh).
    /// Index 0 = midnight Jan 1, Index 8759 = 11pm Dec 31.
    hourly_intensity: Vec<f64>,
}

impl Default for CarbonIntensityProfile {
    fn default() -> Self {
        Self {
            hourly_intensity: vec![0.0; HOURS_PER_YEAR],
        }
    }
}

impl CarbonIntensityProfile {
    /// Create a new empty profile (all zeros).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a profile with a constant intensity for all hours.
    pub fn constant(kg_co2eq_per_kwh: f64) -> Self {
        Self {
            hourly_intensity: vec![kg_co2eq_per_kwh; HOURS_PER_YEAR],
        }
    }

    /// Get the carbon intensity for a specific hour (0-8759).
    pub fn get_hourly(&self, hour: usize) -> Result<f64, CarbonError> {
        if hour >= HOURS_PER_YEAR {
            return Err(CarbonError::InvalidHour { hour });
        }
        Ok(self.hourly_intensity[hour])
    }

    /// Get the carbon intensity for a fractional hour, with linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `hour` - Whole hour (0-8759)
    /// * `fraction` - Fraction within the hour (0.0 = start, 1.0 = end)
    ///
    /// # Returns
    ///
    /// Interpolated carbon intensity in kg CO2eq/kWh
    pub fn get_interpolated(&self, hour: usize, fraction: f64) -> Result<f64, CarbonError> {
        if hour >= HOURS_PER_YEAR {
            return Err(CarbonError::InvalidHour { hour });
        }
        let next_hour = (hour + 1) % HOURS_PER_YEAR;
        Ok(linear_interpolate(
            self.hourly_intensity[hour],
            self.hourly_intensity[next_hour],
            fraction,
        ))
    }

    /// Get the full hourly slice (for iteration/accumulation).
    pub fn hourly_values(&self) -> &[f64] {
        &self.hourly_intensity
    }

    /// Parse a carbon intensity CSV file.
    ///
    /// Expected format:
    /// ```csv
    /// datetime,kg_CO2eq_per_kWh
    /// 2024-01-01T00:00:00,0.45
    /// 2024-01-01T01:00:00,0.42
    /// ```
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file
    ///
    /// # Returns
    ///
    /// Parsed CarbonIntensityProfile
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self, CarbonError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        Self::parse_csv(reader)
    }

    /// Parse carbon intensity CSV from a reader.
    pub fn parse_csv<R: BufRead>(reader: R) -> Result<Self, CarbonError> {
        let mut profile = Self::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and header
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("datetime") || line.starts_with("DateTime") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 2 {
                return Err(CarbonError::CsvParse {
                    line: line_num + 1,
                    message: format!("Expected 2 columns, got {}", parts.len()),
                });
            }

            let intensity: f64 = parts[1].trim().parse().map_err(|_| CarbonError::CsvParse {
                line: line_num + 1,
                message: format!("Cannot parse '{}' as float", parts[1]),
            })?;

            // Parse datetime to get hour of year
            let hour = parse_datetime_to_hour(parts[0].trim())?;

            if hour >= HOURS_PER_YEAR {
                return Err(CarbonError::InvalidHour { hour });
            }

            profile.hourly_intensity[hour] = intensity;
        }

        Ok(profile)
    }

    /// Write profile to a CSV file.
    pub fn to_csv(&self, path: impl AsRef<Path>) -> Result<(), CarbonError> {
        let mut file = File::create(path.as_ref())?;
        writeln!(file, "datetime,kg_CO2eq_per_kWh")?;
        for hour in 0..HOURS_PER_YEAR {
            let datetime = hour_to_datetime(hour);
            writeln!(file, "{},{:.6}", datetime, self.hourly_intensity[hour])?;
        }
        Ok(())
    }
}

/// Parse an ISO 8601 datetime string to hour of year (0-8759).
fn parse_datetime_to_hour(datetime: &str) -> Result<usize, CarbonError> {
    // Expected format: 2024-01-01T00:00:00 or 2024-01-01 00:00:00
    let parts: Vec<&str> = datetime.split(&['-', 'T', ' ', ':'][..]).collect();
    if parts.len() < 5 {
        return Err(CarbonError::CsvParse {
            line: 0,
            message: format!("Cannot parse datetime '{}'", datetime),
        });
    }

    let month: u32 = parts[1].parse().map_err(|_| CarbonError::CsvParse {
        line: 0,
        message: format!("Cannot parse month '{}'", parts[1]),
    })?;
    let day: u32 = parts[2].parse().map_err(|_| CarbonError::CsvParse {
        line: 0,
        message: format!("Cannot parse day '{}'", parts[2]),
    })?;
    let hour: u32 = parts[3].parse().map_err(|_| CarbonError::CsvParse {
        line: 0,
        message: format!("Cannot parse hour '{}'", parts[3]),
    })?;

    let hour_of_year = date_to_hour_of_year(month, day, hour);
    Ok(hour_of_year)
}

/// Convert month, day, hour to hour of year (0-8759).
fn date_to_hour_of_year(month: u32, day: u32, hour: u32) -> usize {
    let days_before_month: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let days_before = days_before_month[(month - 1) as usize];
    let day_of_year = days_before + day - 1;
    ((day_of_year * 24) + hour) as usize
}

/// Convert hour of year (0-8759) to ISO 8601 datetime string.
fn hour_to_datetime(hour: usize) -> String {
    let hour = hour.min(8759);
    let days_per_month: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    let day_of_year = hour / 24;
    let hour_of_day = hour % 24;

    let mut month: u32 = 1;
    let mut day: u32 = 1;
    let mut days_acc: u32 = 0;

    for (i, &days_in_month) in days_per_month.iter().enumerate() {
        if days_acc + days_in_month > day_of_year as u32 {
            month = (i + 1) as u32;
            day = day_of_year as u32 - days_acc + 1;
            break;
        }
        days_acc += days_in_month;
    }

    format!("2024-{:02}-{:02}T{:02}:00:00", month, day, hour_of_day)
}

/// Carbon accumulator for tracking hourly carbon emissions.
///
/// Accumulates carbon emissions based on net grid energy and carbon intensity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CarbonAccumulator {
    /// Hourly carbon emissions (kg CO2eq)
    hourly_carbon_kg: Vec<f64>,
    /// Total carbon emissions (kg CO2eq)
    total_carbon_kg: f64,
}

impl CarbonAccumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            hourly_carbon_kg: vec![0.0; HOURS_PER_YEAR],
            total_carbon_kg: 0.0,
        }
    }

    /// Accumulate carbon for a specific hour.
    ///
    /// # Arguments
    ///
    /// * `hour` - Hour of year (0-8759)
    /// * `net_grid_kwh` - Net grid energy for the hour (kWh). Positive = import, negative = export.
    /// * `intensity` - Carbon intensity (kg CO2eq/kWh)
    ///
    /// # Returns
    ///
    /// Carbon emitted/absorbed for this hour (kg CO2eq)
    pub fn accumulate(
        &mut self,
        hour: usize,
        net_grid_kwh: f64,
        intensity: f64,
    ) -> Result<f64, CarbonError> {
        if hour >= HOURS_PER_YEAR {
            return Err(CarbonError::InvalidHour { hour });
        }
        let carbon = net_grid_kwh * intensity;
        self.hourly_carbon_kg[hour] = carbon;
        self.total_carbon_kg += carbon;
        Ok(carbon)
    }

    /// Accumulate carbon using a carbon intensity profile.
    ///
    /// # Arguments
    ///
    /// * `hour` - Hour of year (0-8759)
    /// * `net_grid_kwh` - Net grid energy for the hour (kWh)
    /// * `profile` - Carbon intensity profile
    pub fn accumulate_with_profile(
        &mut self,
        hour: usize,
        net_grid_kwh: f64,
        profile: &CarbonIntensityProfile,
    ) -> Result<f64, CarbonError> {
        let intensity = profile.get_hourly(hour)?;
        self.accumulate(hour, net_grid_kwh, intensity)
    }

    /// Get total carbon emissions (kg CO2eq).
    pub fn total_carbon_kg(&self) -> f64 {
        self.total_carbon_kg
    }

    /// Get hourly carbon values.
    pub fn hourly_values(&self) -> &[f64] {
        &self.hourly_carbon_kg
    }

    /// Get carbon for a specific hour.
    pub fn get_hourly(&self, hour: usize) -> Result<f64, CarbonError> {
        if hour >= HOURS_PER_YEAR {
            return Err(CarbonError::InvalidHour { hour });
        }
        Ok(self.hourly_carbon_kg[hour])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carbon_profile_constant() {
        let profile = CarbonIntensityProfile::constant(0.5);
        assert_eq!(profile.get_hourly(0).unwrap(), 0.5);
        assert_eq!(profile.get_hourly(100).unwrap(), 0.5);
        assert_eq!(profile.get_hourly(8759).unwrap(), 0.5);
    }

    #[test]
    fn test_carbon_profile_default() {
        let profile = CarbonIntensityProfile::new();
        assert_eq!(profile.get_hourly(0).unwrap(), 0.0);
        assert_eq!(profile.get_hourly(8759).unwrap(), 0.0);
    }

    #[test]
    fn test_carbon_profile_interpolation() {
        let mut profile = CarbonIntensityProfile::new();
        profile.hourly_intensity[0] = 0.4;
        profile.hourly_intensity[1] = 0.6;

        assert_eq!(profile.get_interpolated(0, 0.0).unwrap(), 0.4);
        assert_eq!(profile.get_interpolated(0, 1.0).unwrap(), 0.6);
        assert_eq!(profile.get_interpolated(0, 0.5).unwrap(), 0.5);
    }

    #[test]
    fn test_carbon_profile_invalid_hour() {
        let profile = CarbonIntensityProfile::new();
        assert!(profile.get_hourly(8760).is_err());
        assert!(profile.get_interpolated(8760, 0.0).is_err());
    }

    #[test]
    fn test_parse_datetime_to_hour() {
        // Jan 1, 00:00 -> hour 0
        assert_eq!(parse_datetime_to_hour("2024-01-01T00:00:00").unwrap(), 0);
        // Jan 1, 01:00 -> hour 1
        assert_eq!(parse_datetime_to_hour("2024-01-01T01:00:00").unwrap(), 1);
        // Jan 2, 00:00 -> hour 24
        assert_eq!(parse_datetime_to_hour("2024-01-02T00:00:00").unwrap(), 24);
        // Dec 31, 23:00 -> hour 8759
        assert_eq!(parse_datetime_to_hour("2024-12-31T23:00:00").unwrap(), 8759);
    }

    #[test]
    fn test_hour_to_datetime() {
        assert_eq!(hour_to_datetime(0), "2024-01-01T00:00:00");
        assert_eq!(hour_to_datetime(1), "2024-01-01T01:00:00");
        assert_eq!(hour_to_datetime(24), "2024-01-02T00:00:00");
        assert_eq!(hour_to_datetime(8759), "2024-12-31T23:00:00");
    }

    #[test]
    fn test_carbon_accumulator_import() {
        let mut acc = CarbonAccumulator::new();
        // 1 kWh imported at 0.5 kg CO2eq/kWh -> 0.5 kg
        let result = acc.accumulate(0, 1.0, 0.5).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
        assert!((acc.total_carbon_kg() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_carbon_accumulator_export() {
        let mut acc = CarbonAccumulator::new();
        // -1 kWh exported (net negative grid) at 0.5 kg CO2eq/kWh -> -0.5 kg (credit)
        let result = acc.accumulate(0, -1.0, 0.5).unwrap();
        assert!((result - (-0.5)).abs() < 1e-10);
        assert!((acc.total_carbon_kg() - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_carbon_accumulator_two_hours() {
        // Test acceptance criteria: 1 kWh at 0.5 and 0.1 produces 0.6 kg total
        let mut acc = CarbonAccumulator::new();
        acc.accumulate(0, 1.0, 0.5).unwrap(); // 0.5 kg
        acc.accumulate(1, 1.0, 0.1).unwrap(); // 0.1 kg
        assert!((acc.total_carbon_kg() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_carbon_accumulator_export_subtracts() {
        // Test: export-only hour (net negative grid) correctly subtracts from carbon total
        let mut acc = CarbonAccumulator::new();
        acc.accumulate(0, 1.0, 0.5).unwrap(); // 0.5 kg
        acc.accumulate(1, -0.5, 0.4).unwrap(); // -0.2 kg (credit)
                                               // Net: 0.5 - 0.2 = 0.3 kg
        assert!((acc.total_carbon_kg() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_carbon_accumulator_with_profile() {
        let mut profile = CarbonIntensityProfile::new();
        profile.hourly_intensity[0] = 0.5;
        profile.hourly_intensity[1] = 0.1;

        let mut acc = CarbonAccumulator::new();
        acc.accumulate_with_profile(0, 1.0, &profile).unwrap();
        acc.accumulate_with_profile(1, 1.0, &profile).unwrap();
        assert!((acc.total_carbon_kg() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_carbon_profile_csv_parse() {
        let csv_data = "\
datetime,kg_CO2eq_per_kWh
2024-01-01T00:00:00,0.45
2024-01-01T01:00:00,0.42
2024-01-01T02:00:00,0.40
";
        let profile = CarbonIntensityProfile::parse_csv(csv_data.as_bytes()).unwrap();
        assert_eq!(profile.get_hourly(0).unwrap(), 0.45);
        assert_eq!(profile.get_hourly(1).unwrap(), 0.42);
        assert_eq!(profile.get_hourly(2).unwrap(), 0.40);
    }

    #[test]
    fn test_carbon_profile_csv_roundtrip() {
        let mut profile = CarbonIntensityProfile::new();
        profile.hourly_intensity[0] = 0.5;
        profile.hourly_intensity[100] = 0.3;
        profile.hourly_intensity[8759] = 0.4;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        profile.to_csv(&temp_file).unwrap();

        temp_file.flush().unwrap();

        let parsed = CarbonIntensityProfile::from_csv(temp_file.path()).unwrap();
        assert_eq!(parsed.get_hourly(0).unwrap(), 0.5);
        assert_eq!(parsed.get_hourly(100).unwrap(), 0.3);
        assert_eq!(parsed.get_hourly(8759).unwrap(), 0.4);
    }

    #[test]
    fn test_carbon_profile_serde() {
        let mut profile = CarbonIntensityProfile::new();
        profile.hourly_intensity[0] = 0.5;
        profile.hourly_intensity[100] = 0.3;

        let json = serde_json::to_string(&profile).unwrap();
        let parsed: CarbonIntensityProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.get_hourly(0).unwrap(), 0.5);
        assert_eq!(parsed.get_hourly(100).unwrap(), 0.3);
    }

    #[test]
    fn test_carbon_accumulator_serde() {
        let mut acc = CarbonAccumulator::new();
        acc.accumulate(0, 1.0, 0.5).unwrap();
        acc.accumulate(1, 2.0, 0.3).unwrap();

        let json = serde_json::to_string(&acc).unwrap();
        let parsed: CarbonAccumulator = serde_json::from_str(&json).unwrap();
        assert!((parsed.total_carbon_kg() - 1.1).abs() < 1e-10);
    }
}
