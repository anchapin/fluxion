//! FLEXLAB Site Weather Data Loader
//!
//! Loads 1 year of measured site weather data from the FLEXLAB test cell
//! at Lawrence Berkeley National Lab (LBNL). Includes gap/quality flags
//! for each observation to indicate missing, interpolated, or suspect data.
//!
//! # Data Source
//!
//! FLEXLAB (Facility for Low Energy Experiments in Buildings) outdoor
//! weather station providing hourly measurements of:
//! - Dry-bulb temperature
//! - Relative humidity
//! - Solar irradiance (GHI, DNI, DHI)
//! - Wind speed
//!
//! # Gap/Quality Flags
//!
//! Each data point carries a [`QualityFlag`] that encodes:
//! - `Valid` — measurement present and within physical bounds
//! - `Missing` — sensor outage or missing record
//! - `Suspect` — value outside expected range for the season/location
//! - `Interpolated` — gap-filled from neighbouring observations
//!
//! # Data Path
//!
//! The canonical CSV file is expected at `data/flexlab/site_weather/site_weather_hourly.csv`
//! relative to the repository root. See `docs/validation/flexlab-dataset.md` for
//! dataset documentation and provenance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Quality flags
// ---------------------------------------------------------------------------

/// Quality flag for an individual observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityFlag {
    /// Measurement is present and within physical bounds.
    Valid,
    /// Sensor outage or missing record in the CSV.
    Missing,
    /// Value present but outside expected range for the season/location.
    Suspect,
    /// Gap-filled from neighbouring observations (linear interpolation).
    Interpolated,
}

// ---------------------------------------------------------------------------
// Weather data point
// ---------------------------------------------------------------------------

/// One hour of measured FLEXLAB site weather data with quality metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexlabWeatherRecord {
    /// Hour of year (0–8759).
    pub hour_of_year: usize,
    /// Calendar year.
    pub year: u16,
    /// Month (1–12).
    pub month: u8,
    /// Day of month (1–31).
    pub day: u8,
    /// Hour of day (0–23).
    pub hour: u8,
    /// Outdoor dry-bulb temperature [°C].
    pub dry_bulb_temp: f64,
    /// Quality flag for dry-bulb temperature.
    pub dry_bulb_flag: QualityFlag,
    /// Relative humidity [%].
    pub relative_humidity: f64,
    /// Quality flag for relative humidity.
    pub humidity_flag: QualityFlag,
    /// Global horizontal irradiance [W/m²].
    pub ghi: f64,
    /// Quality flag for GHI.
    pub ghi_flag: QualityFlag,
    /// Direct normal irradiance [W/m²].
    pub dni: f64,
    /// Quality flag for DNI.
    pub dni_flag: QualityFlag,
    /// Diffuse horizontal irradiance [W/m²].
    pub dhi: f64,
    /// Quality flag for DHI.
    pub dhi_flag: QualityFlag,
    /// Wind speed [m/s].
    pub wind_speed: f64,
    /// Quality flag for wind speed.
    pub wind_flag: QualityFlag,
}

// ---------------------------------------------------------------------------
// Loader configuration
// ---------------------------------------------------------------------------

/// Configuration for the FLEXLAB weather data loader.
#[derive(Debug, Clone)]
pub struct FlexlabWeatherConfig {
    /// Path to the site weather CSV file.
    pub csv_path: PathBuf,
    /// Minimum acceptable dry-bulb temperature [°C].
    pub min_temp_c: f64,
    /// Maximum acceptable dry-bulb temperature [°C].
    pub max_temp_c: f64,
    /// Minimum acceptable relative humidity [%].
    pub min_humidity: f64,
    /// Maximum acceptable relative humidity [%].
    pub max_humidity: f64,
    /// Maximum acceptable GHI [W/m²].
    pub max_ghi: f64,
    /// Maximum acceptable DNI [W/m²].
    pub max_dni: f64,
    /// Maximum acceptable wind speed [m/s].
    pub max_wind: f64,
}

impl Default for FlexlabWeatherConfig {
    /// Reasonable defaults for the Berkeley, CA site.
    fn default() -> Self {
        Self {
            csv_path: PathBuf::from("data/flexlab/site_weather/site_weather_hourly.csv"),
            min_temp_c: -10.0,
            max_temp_c: 45.0,
            min_humidity: 0.0,
            max_humidity: 100.0,
            max_ghi: 1400.0,
            max_dni: 1200.0,
            max_wind: 40.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Loader errors
// ---------------------------------------------------------------------------

/// Errors from loading or parsing FLEXLAB weather data.
#[derive(Debug, Clone)]
pub enum FlexlabWeatherError {
    /// The CSV file could not be read.
    FileNotFound(String),
    /// A required column is missing from the CSV header.
    MissingColumn(String),
    /// A numeric field could not be parsed.
    ParseError {
        line: usize,
        field: String,
        value: String,
    },
    /// The file contains zero valid records.
    NoData,
}

impl std::fmt::Display for FlexlabWeatherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "FLEXLAB weather file not found: {}", p),
            Self::MissingColumn(c) => write!(f, "Missing required CSV column: {}", c),
            Self::ParseError { line, field, value } => {
                write!(f, "Line {}: invalid {} value '{}'", line, field, value)
            }
            Self::NoData => write!(f, "No valid data records found in FLEXLAB weather CSV"),
        }
    }
}

impl std::error::Error for FlexlabWeatherError {}

// ---------------------------------------------------------------------------
// Gap/quality analysis helpers
// ---------------------------------------------------------------------------

/// Expected hourly temperature ranges for Berkeley, CA by month (1-indexed).
/// These are loose physical bounds used to flag suspect readings.
const MONTHLY_TEMP_MIN: [f64; 12] = [
    0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 11.0, 11.0, 10.0, 7.0, 3.0, 0.0,
];
const MONTHLY_TEMP_MAX: [f64; 12] = [
    17.0, 19.0, 22.0, 25.0, 28.0, 32.0, 33.0, 33.0, 32.0, 28.0, 21.0, 17.0,
];

/// Classify a temperature reading against seasonal bounds.
fn classify_temp(value: f64, month: u8, cfg: &FlexlabWeatherConfig) -> QualityFlag {
    if value < cfg.min_temp_c || value > cfg.max_temp_c {
        return QualityFlag::Suspect;
    }
    let idx = (month as usize).saturating_sub(1).min(11);
    if value < MONTHLY_TEMP_MIN[idx] - 10.0 || value > MONTHLY_TEMP_MAX[idx] + 10.0 {
        QualityFlag::Suspect
    } else {
        QualityFlag::Valid
    }
}

/// Classify a non-negative measurement against its physical maximum.
fn classify_nonneg(value: f64, max: f64) -> QualityFlag {
    if value < 0.0 || value > max {
        QualityFlag::Suspect
    } else {
        QualityFlag::Valid
    }
}

/// Classify humidity.
fn classify_humidity(value: f64, cfg: &FlexlabWeatherConfig) -> QualityFlag {
    if value < cfg.min_humidity || value > cfg.max_humidity {
        QualityFlag::Suspect
    } else {
        QualityFlag::Valid
    }
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

/// Map of column-name → index built from the CSV header row.
type ColumnMap = HashMap<String, usize>;

/// Build a case-insensitive column map from the header line.
fn build_column_map(header: &str) -> ColumnMap {
    header
        .split(',')
        .enumerate()
        .map(|(i, name)| (name.trim().to_lowercase(), i))
        .collect()
}

/// Look up a column index, returning an error if absent.
fn require_col(cols: &ColumnMap, name: &str) -> Result<usize, FlexlabWeatherError> {
    cols.get(name)
        .copied()
        .ok_or_else(|| FlexlabWeatherError::MissingColumn(name.to_string()))
}

/// Parse a single field as `f64`, returning `None` on failure.
fn parse_f64(fields: &[&str], idx: usize) -> Option<f64> {
    fields.get(idx)?.trim().parse::<f64>().ok()
}

/// Compute hour-of-year from calendar fields (assumes non-leap year).
fn hour_of_year(month: u8, day: u8, hour: u8) -> usize {
    let cum_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let m = (month as usize).saturating_sub(1).min(11);
    let doy = cum_days[m] + (day as usize).saturating_sub(1);
    doy * 24 + hour as usize
}

/// Parse one CSV data line into a [`FlexlabWeatherRecord`].
///
/// Missing or unparseable fields are flagged as `Missing`; numeric values
/// outside physical bounds are flagged as `Suspect`.
fn parse_record(
    fields: &[&str],
    cols: &ColumnMap,
    _line_idx: usize,
    cfg: &FlexlabWeatherConfig,
) -> Result<FlexlabWeatherRecord, FlexlabWeatherError> {
    // ── calendar fields ──────────────────────────────────────────────────
    let year = parse_f64(fields, require_col(cols, "year")?)
        .map(|v| v as u16)
        .unwrap_or(2020);
    let month = parse_f64(fields, require_col(cols, "month")?)
        .map(|v| v as u8)
        .unwrap_or(1);
    let day = parse_f64(fields, require_col(cols, "day")?)
        .map(|v| v as u8)
        .unwrap_or(1);
    let hour = parse_f64(fields, require_col(cols, "hour")?)
        .map(|v| v as u8)
        .unwrap_or(0);

    let hoy = hour_of_year(month, day, hour);

    // ── temperature ──────────────────────────────────────────────────────
    let temp_idx = cols
        .get("dry_bulb_temp")
        .or_else(|| cols.get("outdoor_temp_c"))
        .or_else(|| cols.get("temp_c"))
        .or_else(|| cols.get("temperature"))
        .copied();
    let (dry_bulb_temp, dry_bulb_flag) = match temp_idx.and_then(|i| parse_f64(fields, i)) {
        Some(t) => (t, classify_temp(t, month, cfg)),
        None => (f64::NAN, QualityFlag::Missing),
    };

    // ── humidity ─────────────────────────────────────────────────────────
    let hum_idx = cols
        .get("relative_humidity")
        .or_else(|| cols.get("rh_pct"))
        .or_else(|| cols.get("humidity"))
        .copied();
    let (relative_humidity, humidity_flag) = match hum_idx.and_then(|i| parse_f64(fields, i)) {
        Some(h) => (h, classify_humidity(h, cfg)),
        None => (f64::NAN, QualityFlag::Missing),
    };

    // ── solar: GHI ───────────────────────────────────────────────────────
    let ghi_idx = cols
        .get("ghi")
        .or_else(|| cols.get("global_horizontal_irradiance"))
        .or_else(|| cols.get("ghi_wm2"))
        .copied();
    let (ghi, ghi_flag) = match ghi_idx.and_then(|i| parse_f64(fields, i)) {
        Some(v) => (v, classify_nonneg(v, cfg.max_ghi)),
        None => (0.0, QualityFlag::Missing),
    };

    // ── solar: DNI ───────────────────────────────────────────────────────
    let dni_idx = cols
        .get("dni")
        .or_else(|| cols.get("direct_normal_irradiance"))
        .or_else(|| cols.get("dni_wm2"))
        .copied();
    let (dni, dni_flag) = match dni_idx.and_then(|i| parse_f64(fields, i)) {
        Some(v) => (v, classify_nonneg(v, cfg.max_dni)),
        None => (0.0, QualityFlag::Missing),
    };

    // ── solar: DHI ───────────────────────────────────────────────────────
    let dhi_idx = cols
        .get("dhi")
        .or_else(|| cols.get("diffuse_horizontal_irradiance"))
        .or_else(|| cols.get("dhi_wm2"))
        .copied();
    let (dhi, dhi_flag) = match dhi_idx.and_then(|i| parse_f64(fields, i)) {
        Some(v) => (v, classify_nonneg(v, cfg.max_ghi)),
        None => (0.0, QualityFlag::Missing),
    };

    // ── wind speed ───────────────────────────────────────────────────────
    let wind_idx = cols
        .get("wind_speed")
        .or_else(|| cols.get("wind_speed_ms"))
        .or_else(|| cols.get("wind"))
        .copied();
    let (wind_speed, wind_flag) = match wind_idx.and_then(|i| parse_f64(fields, i)) {
        Some(v) => (v, classify_nonneg(v, cfg.max_wind)),
        None => (0.0, QualityFlag::Missing),
    };

    Ok(FlexlabWeatherRecord {
        hour_of_year: hoy,
        year,
        month,
        day,
        hour,
        dry_bulb_temp,
        dry_bulb_flag,
        relative_humidity,
        humidity_flag,
        ghi,
        ghi_flag,
        dni,
        dni_flag,
        dhi,
        dhi_flag,
        wind_speed,
        wind_flag,
    })
}

// ---------------------------------------------------------------------------
// Gap-filling via linear interpolation
// ---------------------------------------------------------------------------

/// Attempt to fill a single `Missing` gap using the nearest valid neighbours.
///
/// Only single-hour gaps are filled; multi-hour outages retain `Missing`.
fn interpolate_gaps(records: &mut [FlexlabWeatherRecord]) {
    let len = records.len();
    if len < 3 {
        return;
    }

    // Collect indices of records whose temperature is Missing
    let missing_indices: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.dry_bulb_flag == QualityFlag::Missing)
        .map(|(i, _)| i)
        .collect();

    for &idx in &missing_indices {
        // Find previous valid
        let prev_valid = records[..idx]
            .iter()
            .rposition(|r| r.dry_bulb_flag == QualityFlag::Valid);
        // Find next valid
        let next_valid = records[idx + 1..]
            .iter()
            .position(|r| r.dry_bulb_flag == QualityFlag::Valid)
            .map(|p| p + idx + 1);

        match (prev_valid, next_valid) {
            (Some(prev), Some(next)) if next - prev == 2 => {
                // Single-hour gap — interpolate
                let t0 = records[prev].dry_bulb_temp;
                let t1 = records[next].dry_bulb_temp;
                records[idx].dry_bulb_temp = (t0 + t1) / 2.0;
                records[idx].dry_bulb_flag = QualityFlag::Interpolated;
            }
            _ => {} // Multi-hour or edge gap — leave as Missing
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Summary statistics for a loaded dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexlabWeatherSummary {
    /// Total records in the file.
    pub total_records: usize,
    /// Records with `Valid` temperature.
    pub valid_temp_count: usize,
    /// Records with `Missing` temperature.
    pub missing_temp_count: usize,
    /// Records with `Suspect` temperature.
    pub suspect_temp_count: usize,
    /// Records with `Interpolated` temperature.
    pub interpolated_temp_count: usize,
    /// Mean dry-bulb temperature [°C] over valid records.
    pub mean_temp_c: f64,
    /// Minimum dry-bulb temperature [°C].
    pub min_temp_c: f64,
    /// Maximum dry-bulb temperature [°C].
    pub max_temp_c: f64,
    /// Mean GHI [W/m²] over valid records.
    pub mean_ghi: f64,
    /// Mean wind speed [m/s] over valid records.
    pub mean_wind: f64,
}

impl FlexlabWeatherSummary {
    /// Compute summary statistics from loaded records.
    pub fn from_records(records: &[FlexlabWeatherRecord]) -> Self {
        let total = records.len();

        let valid_temps: Vec<f64> = records
            .iter()
            .filter(|r| r.dry_bulb_flag == QualityFlag::Valid)
            .map(|r| r.dry_bulb_temp)
            .collect();

        let missing = records
            .iter()
            .filter(|r| r.dry_bulb_flag == QualityFlag::Missing)
            .count();
        let suspect = records
            .iter()
            .filter(|r| r.dry_bulb_flag == QualityFlag::Suspect)
            .count();
        let interpolated = records
            .iter()
            .filter(|r| r.dry_bulb_flag == QualityFlag::Interpolated)
            .count();

        let mean_temp = if valid_temps.is_empty() {
            f64::NAN
        } else {
            valid_temps.iter().sum::<f64>() / valid_temps.len() as f64
        };
        let min_temp_c = valid_temps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_temp_c = valid_temps
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let valid_ghi: Vec<f64> = records
            .iter()
            .filter(|r| r.ghi_flag == QualityFlag::Valid)
            .map(|r| r.ghi)
            .collect();
        let mean_ghi = if valid_ghi.is_empty() {
            f64::NAN
        } else {
            valid_ghi.iter().sum::<f64>() / valid_ghi.len() as f64
        };

        let valid_wind: Vec<f64> = records
            .iter()
            .filter(|r| r.wind_flag == QualityFlag::Valid)
            .map(|r| r.wind_speed)
            .collect();
        let mean_wind = if valid_wind.is_empty() {
            f64::NAN
        } else {
            valid_wind.iter().sum::<f64>() / valid_wind.len() as f64
        };

        Self {
            total_records: total,
            valid_temp_count: valid_temps.len(),
            missing_temp_count: missing,
            suspect_temp_count: suspect,
            interpolated_temp_count: interpolated,
            mean_temp_c: mean_temp,
            min_temp_c,
            max_temp_c,
            mean_ghi,
            mean_wind,
        }
    }

    /// Fraction of records with valid temperature (0.0–1.0).
    pub fn data_availability(&self) -> f64 {
        if self.total_records == 0 {
            0.0
        } else {
            self.valid_temp_count as f64 / self.total_records as f64
        }
    }
}

/// Load FLEXLAB site weather data from the configured CSV path.
///
/// Applies gap detection and quality flagging to every observation.
/// Single-hour gaps in temperature are linearly interpolated and flagged
/// as `Interpolated`; multi-hour outages remain `Missing`.
pub fn load_flexlab_weather(
    config: &FlexlabWeatherConfig,
) -> Result<(Vec<FlexlabWeatherRecord>, FlexlabWeatherSummary), FlexlabWeatherError> {
    let content = std::fs::read_to_string(&config.csv_path).map_err(|e| {
        FlexlabWeatherError::FileNotFound(format!("{}: {}", config.csv_path.display(), e))
    })?;

    let mut records = Vec::new();
    let mut header_map: Option<ColumnMap> = None;

    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }

        if header_map.is_none() {
            header_map = Some(build_column_map(line));
            continue;
        }

        let cols = header_map.as_ref().unwrap();
        let fields: Vec<&str> = line.split(',').collect();

        match parse_record(&fields, cols, line_idx, config) {
            Ok(record) => records.push(record),
            Err(e) => {
                tracing::warn!(
                    line = line_idx + 1,
                    error = %e,
                    "flexlab weather loader parse warning",
                );
            }
        }
    }

    if records.is_empty() {
        return Err(FlexlabWeatherError::NoData);
    }

    // Sort by hour-of-year and interpolate single-hour gaps
    records.sort_by_key(|r| r.hour_of_year);
    interpolate_gaps(&mut records);

    let summary = FlexlabWeatherSummary::from_records(&records);

    Ok((records, summary))
}

/// Convenience path to the canonical FLEXLAB weather CSV.
pub fn default_csv_path() -> PathBuf {
    FlexlabWeatherConfig::default().csv_path
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hour_of_year_calculation() {
        assert_eq!(hour_of_year(1, 1, 0), 0);
        assert_eq!(hour_of_year(1, 1, 23), 23);
        assert_eq!(hour_of_year(1, 2, 0), 24);
        assert_eq!(hour_of_year(12, 31, 23), 8759);
    }

    #[test]
    fn test_quality_flag_classification() {
        let cfg = FlexlabWeatherConfig::default();

        // Valid temperature in January
        assert_eq!(classify_temp(10.0, 1, &cfg), QualityFlag::Valid);

        // Suspect: way below physical minimum
        assert_eq!(classify_temp(-100.0, 1, &cfg), QualityFlag::Suspect);

        // Suspect: way above physical maximum
        assert_eq!(classify_temp(60.0, 7, &cfg), QualityFlag::Suspect);

        // Valid humidity
        assert_eq!(classify_humidity(50.0, &cfg), QualityFlag::Valid);

        // Suspect humidity
        assert_eq!(classify_humidity(110.0, &cfg), QualityFlag::Suspect);

        // Valid solar
        assert_eq!(classify_nonneg(800.0, 1400.0), QualityFlag::Valid);

        // Suspect solar
        assert_eq!(classify_nonneg(-10.0, 1400.0), QualityFlag::Suspect);
    }

    #[test]
    fn test_build_column_map() {
        let header = "year,month,day,hour,dry_bulb_temp,relative_humidity,ghi,dni,dhi,wind_speed";
        let cols = build_column_map(header);
        assert_eq!(cols.get("year"), Some(&0));
        assert_eq!(cols.get("dry_bulb_temp"), Some(&4));
        assert_eq!(cols.get("wind_speed"), Some(&9));
    }

    #[test]
    fn test_column_aliases() {
        let header = "year,month,day,hour,outdoor_temp_c,rh_pct,global_horizontal_irradiance,direct_normal_irradiance,diffuse_horizontal_irradiance,wind_speed_ms";
        let cols = build_column_map(header);
        assert!(cols.contains_key("outdoor_temp_c"));
        assert!(cols.contains_key("rh_pct"));
        assert!(cols.contains_key("global_horizontal_irradiance"));
    }

    #[test]
    fn test_interpolate_gaps_single_hour() {
        let _cfg = FlexlabWeatherConfig::default();
        let mut records: Vec<FlexlabWeatherRecord> = Vec::new();

        // Build 5 records, hour 2 is missing
        for h in 0..5u8 {
            let mut r = FlexlabWeatherRecord {
                hour_of_year: h as usize,
                year: 2023,
                month: 6,
                day: 1,
                hour: h,
                dry_bulb_temp: 20.0 + h as f64,
                dry_bulb_flag: QualityFlag::Valid,
                relative_humidity: 50.0,
                humidity_flag: QualityFlag::Valid,
                ghi: 0.0,
                ghi_flag: QualityFlag::Valid,
                dni: 0.0,
                dni_flag: QualityFlag::Valid,
                dhi: 0.0,
                dhi_flag: QualityFlag::Valid,
                wind_speed: 2.0,
                wind_flag: QualityFlag::Valid,
            };
            if h == 2 {
                r.dry_bulb_flag = QualityFlag::Missing;
            }
            records.push(r);
        }

        interpolate_gaps(&mut records);

        // Hour 2 should be interpolated: (21.0 + 23.0) / 2 = 22.0
        assert_eq!(records[2].dry_bulb_flag, QualityFlag::Interpolated);
        assert!((records[2].dry_bulb_temp - 22.0).abs() < 0.01);
    }

    #[test]
    fn test_interpolate_gaps_multi_hour_stays_missing() {
        let mut records: Vec<FlexlabWeatherRecord> = Vec::new();

        for h in 0..6u8 {
            let mut r = FlexlabWeatherRecord {
                hour_of_year: h as usize,
                year: 2023,
                month: 6,
                day: 1,
                hour: h,
                dry_bulb_temp: 20.0,
                dry_bulb_flag: QualityFlag::Valid,
                relative_humidity: 50.0,
                humidity_flag: QualityFlag::Valid,
                ghi: 0.0,
                ghi_flag: QualityFlag::Valid,
                dni: 0.0,
                dni_flag: QualityFlag::Valid,
                dhi: 0.0,
                dhi_flag: QualityFlag::Valid,
                wind_speed: 2.0,
                wind_flag: QualityFlag::Valid,
            };
            if h == 2 || h == 3 {
                r.dry_bulb_flag = QualityFlag::Missing;
            }
            records.push(r);
        }

        interpolate_gaps(&mut records);

        // Multi-hour gap should stay Missing
        assert_eq!(records[2].dry_bulb_flag, QualityFlag::Missing);
        assert_eq!(records[3].dry_bulb_flag, QualityFlag::Missing);
    }

    #[test]
    fn test_summary_statistics() {
        let records = vec![
            FlexlabWeatherRecord {
                hour_of_year: 0,
                year: 2023,
                month: 1,
                day: 1,
                hour: 0,
                dry_bulb_temp: 10.0,
                dry_bulb_flag: QualityFlag::Valid,
                relative_humidity: 50.0,
                humidity_flag: QualityFlag::Valid,
                ghi: 100.0,
                ghi_flag: QualityFlag::Valid,
                dni: 50.0,
                dni_flag: QualityFlag::Valid,
                dhi: 50.0,
                dhi_flag: QualityFlag::Valid,
                wind_speed: 3.0,
                wind_flag: QualityFlag::Valid,
            },
            FlexlabWeatherRecord {
                hour_of_year: 1,
                year: 2023,
                month: 1,
                day: 1,
                hour: 1,
                dry_bulb_temp: f64::NAN,
                dry_bulb_flag: QualityFlag::Missing,
                relative_humidity: f64::NAN,
                humidity_flag: QualityFlag::Missing,
                ghi: 0.0,
                ghi_flag: QualityFlag::Valid,
                dni: 0.0,
                dni_flag: QualityFlag::Valid,
                dhi: 0.0,
                dhi_flag: QualityFlag::Valid,
                wind_speed: 0.0,
                wind_flag: QualityFlag::Valid,
            },
            FlexlabWeatherRecord {
                hour_of_year: 2,
                year: 2023,
                month: 1,
                day: 1,
                hour: 2,
                dry_bulb_temp: 12.0,
                dry_bulb_flag: QualityFlag::Valid,
                relative_humidity: 60.0,
                humidity_flag: QualityFlag::Valid,
                ghi: 0.0,
                ghi_flag: QualityFlag::Valid,
                dni: 0.0,
                dni_flag: QualityFlag::Valid,
                dhi: 0.0,
                dhi_flag: QualityFlag::Valid,
                wind_speed: 2.0,
                wind_flag: QualityFlag::Valid,
            },
        ];

        let summary = FlexlabWeatherSummary::from_records(&records);
        assert_eq!(summary.total_records, 3);
        assert_eq!(summary.valid_temp_count, 2);
        assert_eq!(summary.missing_temp_count, 1);
        assert!((summary.mean_temp_c - 11.0).abs() < 0.01);
        assert_eq!(summary.min_temp_c, 10.0);
        assert_eq!(summary.max_temp_c, 12.0);
        assert!((summary.data_availability() - 2.0 / 3.0).abs() < 0.01);
    }
}
