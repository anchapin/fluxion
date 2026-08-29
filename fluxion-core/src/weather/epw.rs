//! EPW (EnergyPlus Weather) file format parser.
//!
//! This module provides parsing functionality for EPW weather data files,
//! which are the standard format for weather data in building energy simulation.
//!
//! # EPW Format Overview
//!
//! EPW files consist of:
//! - Header lines (1-8): Location, design conditions, typical/extreme periods
//! - Data lines (8-8767): Hourly weather data for one year
//!
//! Each data line contains 35+ fields including temperature, solar radiation,
//! wind speed, humidity, and other meteorological parameters.
//!
//! # Supported EPW Versions
//!
//! - **EPW v2**: 8760 hourly records (standard TMY format)
//! - **EPW v3**: 35040 sub-hourly records (15-minute resolution)
//! - **AMY**: Actual Meteorological Year (similar to v2 but with actual year data)
//! - **IWEC**: International Weather for Energy Calculations (similar to v2)

use crate::weather::{HourlyWeatherData, WeatherError, WeatherSource};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

/// Maximum allowed EPW file size (50 MiB). The largest real-world EPW files
/// in `assets/weather/` are ~1.6 MB, so 50 MiB gives generous headroom while
/// still capping the parser-DoS surface flagged by #2527.
pub const MAX_EPW_SIZE_BYTES: u64 = 50 * 1024 * 1024;

/// Default allow-list directory for EPW files when `FLUXION_EPW_DIR` is
/// unset (Issue #2915). Mirrors the conventional location of bundled
/// weather assets.
pub const DEFAULT_EPW_DIR: &str = "assets/weather";

/// Validates a user-supplied EPW path against the security policy from
/// Issue #2915 before it reaches `EpwWeatherSource::from_file`. Reads the
/// allow-list directory from the `FLUXION_EPW_DIR` environment variable
/// (default `assets/weather/`).
///
/// On success returns the canonicalised absolute path. All error messages
/// are deliberately generic and omit the raw user-supplied path so that
/// attacker-controlled input is never reflected back to the caller
/// (closes the error oracle — same hardening as `validate_model_path`
/// from #2529).
///
/// # Security checks (in order)
/// 1. Path resolves to a real, non-symlink file.
/// 2. Extension is `.epw` (case-insensitive).
/// 3. Canonicalised path is inside the `FLUXION_EPW_DIR` allow-list
///    (blocks `..` traversal and any symlinks that escape the allow-list).
/// 4. File size ≤ [`MAX_EPW_SIZE_BYTES`].
pub fn validate_epw_path(p: &str) -> Result<PathBuf, String> {
    let dir = std::env::var("FLUXION_EPW_DIR").unwrap_or_else(|_| DEFAULT_EPW_DIR.to_string());
    validate_epw_path_in_dir(p, Path::new(&dir))
}

/// Parameterised core of [`validate_epw_path`]. Accepts an explicit
/// allow-list directory so it can be unit-tested without racing on the
/// process-wide `FLUXION_EPW_DIR` env var.
///
/// See [`validate_epw_path`] for the documented checks.
pub fn validate_epw_path_in_dir(p: &str, allowed_dir: &Path) -> Result<PathBuf, String> {
    let raw = Path::new(p);
    if !raw.is_file() {
        return Err("epw file not found".to_string());
    }
    // Refuse symlinks. canonicalize() resolves symlinks to their targets,
    // so checking symlink_metadata() here also rejects a symlink that
    // points to a file *inside* the allow-list (belt-and-braces against
    // future TOCTOU or symlink-swap attacks).
    let meta = std::fs::symlink_metadata(raw)
        .map_err(|_| "failed to read epw file metadata".to_string())?;
    if meta.file_type().is_symlink() {
        return Err("epw file path may not be a symbolic link".to_string());
    }
    if raw
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        != Some("epw".to_string())
    {
        return Err("invalid epw file extension (expected .epw)".to_string());
    }
    let canonical_epw =
        std::fs::canonicalize(raw).map_err(|_| "failed to canonicalize epw path".to_string())?;
    let canonical_dir = std::fs::canonicalize(allowed_dir)
        .map_err(|_| "allowed epw directory not found".to_string())?;
    if !canonical_epw.starts_with(&canonical_dir) {
        return Err("epw path outside allowed directory".to_string());
    }
    let size = meta.len();
    if size > MAX_EPW_SIZE_BYTES {
        return Err(format!(
            "epw file exceeds size limit ({} bytes)",
            MAX_EPW_SIZE_BYTES
        ));
    }
    Ok(canonical_epw)
}

/// Hourly weather record from EPW file.
///
/// Contains the standard 8760 hourly data records used in most
/// building energy simulations.
#[derive(Debug, Clone)]
pub struct HourlyRecord {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub dry_bulb_temp: f64,
    pub humidity: f64,
    pub dni: f64,
    pub dhi: f64,
    pub ghi: f64,
    pub wind_speed: f64,
    pub horizontal_infrared: f64,
    pub ground_temperature: Option<f64>,
    pub horizontal_illuminance: Option<f64>,
    pub diffuse_illuminance: Option<f64>,
    pub snow_depth: Option<f64>,
    pub snow_cover: Option<f64>,
    pub present_weather: Option<String>,
    pub present_weather_code: Option<u32>,
}

/// Sub-hourly weather record from EPW v3 file.
///
/// Contains 15-minute timestep records for higher temporal resolution.
#[derive(Debug, Clone)]
pub struct SubHourlyRecord {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub dry_bulb_temp: f64,
    pub humidity: f64,
    pub dni: f64,
    pub dhi: f64,
    pub ghi: f64,
    pub wind_speed: f64,
    pub horizontal_infrared: f64,
    pub ground_temperature: Option<f64>,
    pub horizontal_illuminance: Option<f64>,
    pub diffuse_illuminance: Option<f64>,
    pub snow_depth: Option<f64>,
    pub snow_cover: Option<f64>,
    pub present_weather: Option<String>,
    pub present_weather_code: Option<u32>,
}

/// EPW file format version.
///
/// EnergyPlus Weather (EPW) files come in multiple formats with different
/// record structures and time resolutions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpwVersion {
    /// EPW v2 (hourly)
    ///
    /// Standard EPW format with 8760 hourly records. Default for most
    /// EnergyPlus simulations and ASHRAE 140 validation.
    V2,

    /// EPW v3 (sub-hourly)
    ///
    /// Extended EPW format with 35040 sub-hourly records (15-minute timestep).
    /// Provides higher temporal resolution for detailed building simulation.
    V3,

    /// AMY (Actual Meteorological Year)
    ///
    /// Actual historical weather data instead of typical meteorological year.
    /// Has same structure as EPW v2 but contains real historical data.
    AMY,

    /// IWEC (International Weather for Energy Calculations)
    ///
    /// Weather data for international locations outside the US TMY3 coverage.
    /// Similar structure to EPW v2 with minor format variations.
    IWEC,
}

/// EPW missing-data sentinel for solar irradiance and horizontal infrared fields.
///
/// Per the EnergyPlus Auxiliary Programs reference (§ EPW Data Fields), any
/// source value of 9999 in these columns means "missing measurement".
const EPW_SOLAR_SENTINEL: f64 = 9999.0;

/// EPW missing-data sentinel for atmospheric pressure (Pa).
#[allow(dead_code)] // used in tests; will be needed when HourlyWeatherData gains a pressure field
const EPW_PRESSURE_SENTINEL: f64 = 999900.0;

/// Default sea-level atmospheric pressure used when the sentinel is encountered.
#[allow(dead_code)] // used in tests; will be needed when HourlyWeatherData gains a pressure field
const DEFAULT_ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// EPW missing-data sentinel for liquid precipitation depth (mm).
#[allow(dead_code)] // used in tests; will be needed when HourlyWeatherData gains a precipitation field
const EPW_PRECIPITATION_SENTINEL: f64 = 999.0;

/// Helper function to parse fields that may have missing data.
///
/// Returns the parsed value or the default if the field is empty or invalid.
fn parse_optional_field(field: &str, default: f64) -> f64 {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return default;
    }
    trimmed.parse::<f64>().unwrap_or(default)
}

/// Returns `true` if `value` is at or above the EPW missing-data sentinel.
///
/// EPW files use field-specific sentinel values to mark missing source data.
/// Any value greater than or equal to the sentinel should be treated as
/// "missing", matching the EnergyPlus Weather Converter behavior.
///
/// # Arguments
///
/// * `value` - The parsed numeric value
/// * `sentinel` - The sentinel threshold for this field (e.g. 9999.0 for solar)
fn is_epw_sentinel(value: f64, sentinel: f64) -> bool {
    value >= sentinel
}

/// Parse an EPW field, coercing missing-data sentinels to a safe replacement.
///
/// This wraps [`parse_optional_field`] with an additional sentinel check:
/// if the parsed value is at or above `sentinel` it is replaced with
/// `replacement`. This matches the EnergyPlus Weather Converter, which
/// replaces 9999-type sentinels with 0.0 (or a field-appropriate default)
/// rather than passing them downstream as real measurements.
///
/// # Arguments
///
/// * `field` - Raw field string from the EPW line
/// * `sentinel` - Missing-data sentinel for this field (e.g. 9999.0)
/// * `replacement` - Value to use when the field is missing or sentinel
fn parse_field_coercing_sentinel(field: &str, sentinel: f64, replacement: f64) -> f64 {
    let val = parse_optional_field(field, replacement);
    if is_epw_sentinel(val, sentinel) {
        replacement
    } else {
        val
    }
}

/// Returns `true` if the given line is an EPW header line that must be skipped
/// during data parsing.
///
/// Standard EPW files contain exactly 8 header lines before the data section:
///
/// 1. `LOCATION`
/// 2. `DESIGN CONDITIONS`
/// 3. `TYPICAL/EXTREME PERIODS`
/// 4. `GROUND TEMPERATURES`
/// 5. `HOLIDAYS/DAYLIGHT SAVINGS`
/// 6. `COMMENTS 1`
/// 7. `COMMENTS 2`
/// 8. `DATA PERIODS`
///
/// Most of these have fewer than 35 comma-separated fields and are naturally
/// filtered out by the `fields.len() < 35` guard in the parse loops. However,
/// the `GROUND TEMPERATURES` header carries monthly temperatures at multiple
/// soil depths and can easily contain 35+ fields. Without this prefix check it
/// is mis-parsed as a data record, inserting a spurious first row that shifts
/// every subsequent record by one position (Issue #1164).
///
/// EPW data lines always begin with a 4-digit year, so no legitimate data line
/// can start with any of these prefixes.
fn is_epw_header_line(line: &str) -> bool {
    const HEADER_PREFIXES: &[&str] = &[
        "LOCATION",
        "DESIGN CONDITIONS",
        "TYPICAL/EXTREME PERIODS",
        "GROUND TEMPERATURES",
        "HOLIDAYS/DAYLIGHT SAVINGS",
        "COMMENTS 1",
        "COMMENTS 2",
        "DATA PERIODS",
    ];
    HEADER_PREFIXES
        .iter()
        .any(|prefix| line.starts_with(prefix))
}

/// Detect EPW file version from header.
///
/// EPW files identify their format in the first few lines. This function
/// reads the header and determines which parser to use.
///
/// # Arguments
///
/// * `reader` - Reader for EPW file content
///
/// # Returns
///
/// Detected EPW version
pub fn detect_epw_version<R: Read>(reader: &mut R) -> Result<EpwVersion, WeatherError> {
    // Read first 1KB of file to examine header
    let mut header = [0u8; 1024];
    let bytes_read = reader
        .read(&mut header)
        .map_err(|e| WeatherError::IoError(e.to_string()))?;

    let header_str = std::str::from_utf8(&header[..bytes_read])
        .map_err(|e| WeatherError::ParseError(format!("Invalid UTF-8: {}", e)))?;

    // Check for EPW version indicators
    if header_str.contains("DATA PERIODS") {
        if header_str.contains(",15") {
            // 15-minute data → EPW v3
            return Ok(EpwVersion::V3);
        } else {
            // Hourly data → EPW v2 or AMY
            return Ok(EpwVersion::V2);
        }
    } else if header_str.contains("IWEC") {
        // IWEC format
        return Ok(EpwVersion::IWEC);
    } else if header_str.contains("TMY2") || header_str.contains("TMY3") {
        // TMY data → EPW v2 or AMY
        return Ok(EpwVersion::V2);
    }

    // Default to EPW v2 if version not detected
    Ok(EpwVersion::V2)
}

/// EPW weather data source that parses EnergyPlus Weather format files.
///
/// This struct loads and parses EPW files, extracting the required weather
/// variables for ASHRAE 140 building energy simulations.
///
/// # EPW File Format
///
/// EPW files have the following structure:
///
/// - **Lines 1-8**: Header metadata (location, design conditions, etc.)
/// - **Lines 8-8767**: Hourly weather data (8760 records)
///
/// Each data line contains 35 comma-separated fields. The most relevant for
/// building energy simulation are:
///
/// - Column 7: Dry Bulb Temperature (°C)
/// - Column 9: Relative Humidity (%)
/// - Column 11: Direct Normal Irradiance (Wh/m²)
/// - Column 12: Diffuse Horizontal Irradiance (Wh/m²)
/// - Column 13: Global Horizontal Irradiance (Wh/m²)
/// - Column 22: Wind Speed (m/s)
///
/// # Example
///
/// ```no_run
/// use fluxion_core::weather::epw::EpwWeatherSource;
/// use fluxion_core::weather::WeatherSource;
///
/// // Load an EPW file
/// let weather = EpwWeatherSource::from_file("path/to/weather.epw")
///     .expect("Failed to load EPW file");
///
/// println!("Location: {}", weather.location().unwrap());
///
/// // Get weather for a specific hour
/// let data = weather.get_hourly_data(100)
///     .expect("Failed to get weather data");
/// println!("Temperature: {}°C", data.dry_bulb_temp);
/// ```
/// Structured EPW LOCATION header (Issue #1416).
///
/// Returned by [`EpwWeatherSource::parse_location`]. Carries both the human-
/// readable `city_state` string (e.g. `"Denver, CO"`) and the explicit UTC time-
/// zone offset from EPW LOCATION column 9. The offset is positive east of
/// Greenwich (matching the longitude convention used by the rest of the solar
/// pipeline) — EPW files emitted by EnergyPlus follow the sign convention
/// `Local = UTC + offset`, so Denver is `-7.0`, New Delhi is `+5.5`, and
/// St. John's NL is `-3.5`.
#[derive(Debug, Clone, PartialEq)]
pub struct EpwLocation {
    /// "City, State" string from EPW LOCATION columns 2-3.
    pub city_state: String,
    /// UTC offset in decimal hours from EPW LOCATION column 9 (the
    /// `TimeZone` field). `None` if column 9 is missing or unparseable.
    pub utc_offset_hours: Option<f64>,
}

impl EpwLocation {
    /// Returns the human-readable "City, State" string.
    pub fn city_state(&self) -> &str {
        &self.city_state
    }
}

use std::hash::{Hash, Hasher};

impl Hash for EpwLocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.city_state.hash(state);
        self.utc_offset_hours.map(|v| v.to_bits()).hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct EpwWeatherSource {
    /// Structured location extracted from EPW header. Issue #1416 carries both
    /// the human-readable `city_state` and the UTC offset (EPW LOCATION column
    /// 9). The UTC offset is what callers should forward to
    /// `crate::solar::solar_position::calculate_solar_position` so half-hour
    /// time zones and 7.5°-offset longitudes are handled correctly.
    location: Option<EpwLocation>,
    /// Vector of hourly weather data (8760 entries)
    hourly_data: Vec<HourlyWeatherData>,
}

impl EpwWeatherSource {
    /// Creates a new EPW weather source from a file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the EPW file
    ///
    /// # Returns
    ///
    /// * `Ok(EpwWeatherSource)` - Parsed weather data source
    /// * `Err(WeatherError)` - If the file cannot be read or parsed
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The file does not exist or cannot be opened
    /// - The file format is invalid
    /// - Required data fields are missing or incomplete
    ///
    /// # Example
    ///
    /// ```no_run
    /// use fluxion_core::weather::epw::EpwWeatherSource;
    /// use fluxion_core::weather::WeatherSource;
    ///
    /// let weather = EpwWeatherSource::from_file("weather.epw")
    ///     .expect("Failed to load weather file");
    ///
    /// let data = weather.get_hourly_data(100)?;
    /// # Ok::<(), fluxion_core::weather::WeatherError>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, WeatherError> {
        let file = File::open(path).map_err(|e| WeatherError::IoError(e.to_string()))?;
        let reader = BufReader::new(file);

        Self::parse(reader)
    }

    /// Parses EPW data from a reader.
    ///
    /// This is the core parsing function that handles both header and data lines.
    ///
    /// # Arguments
    ///
    /// * `reader` - A buffered reader over the EPW file content
    ///
    /// # Returns
    ///
    /// * `Ok(EpwWeatherSource)` - Parsed weather data source
    /// * `Err(WeatherError)` - If parsing fails
    fn parse<R: BufRead>(reader: R) -> Result<Self, WeatherError> {
        let mut lines = reader.lines();

        // Parse location from header (line 1, 0-indexed)
        let location = lines
            .next()
            .ok_or_else(|| WeatherError::IncompleteData("Missing location header".to_string()))?
            .map_err(|e| {
                WeatherError::ParseError(format!("Failed to read location header: {}", e))
            })?;

        let location = Self::parse_location(&location)?;

        // Skip to data lines (lines 2-8 are additional headers)
        for _ in 0..7 {
            if lines.next().is_none() {
                return Err(WeatherError::IncompleteData(
                    "Unexpected end of file before data section".to_string(),
                ));
            }
        }

        // Parse hourly data lines
        let mut hourly_data = Vec::new();

        for (line_idx, line_result) in lines.enumerate() {
            let line = line_result.map_err(|e| {
                WeatherError::ParseError(format!(
                    "Failed to read data line {}: {}",
                    line_idx + 1,
                    e
                ))
            })?;

            // Skip comment lines (start with '!')
            if line.trim().starts_with('!') {
                continue;
            }

            let weather_data = Self::parse_data_line(&line, hourly_data.len())?;
            hourly_data.push(weather_data);
        }

        // Validate that we got the expected number of records
        if hourly_data.is_empty() {
            return Err(WeatherError::IncompleteData(
                "No valid data lines found in EPW file".to_string(),
            ));
        }

        Ok(EpwWeatherSource {
            location,
            hourly_data,
        })
    }

    /// Parses the location header line from an EPW file.
    ///
    /// The location line has the format:
    /// `LOCATION,City,StateProv,Country,DataSource,WMO,Latitude,Longitude,TimeZone,Elevation,DataPeriod`
    ///
    /// Column 9 (`TimeZone`) holds the UTC offset in decimal hours; this is now
    /// surfaced on the returned [`EpwLocation`] (Issue #1416) so callers can pass
    /// the explicit value to [`EpwWeatherSource::utc_offset_hours`] and from there
    /// to the solar-position calculator, instead of letting it infer a meridian
    /// from longitude alone.
    ///
    /// # Arguments
    ///
    /// * `line` - The location header line
    ///
    /// # Returns
    ///
    /// * `Some(EpwLocation)` - Structured location metadata (city + UTC offset)
    /// * `None` - If both city and state are missing
    fn parse_location(line: &str) -> Result<Option<EpwLocation>, WeatherError> {
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 3 {
            return Ok(None);
        }

        let city = parts[1].trim();
        let state = parts[2].trim();

        // Issue #1416: column 9 (index 9 because column 10 is "Latitude" 0-indexed
        // at position 6, no — LOCATION layout is:
        //   [0] LOCATION   [1] City   [2] StateProv   [3] Country   [4] DataSource
        //   [5] WMO   [6] Latitude   [7] Longitude   [8] TimeZone   [9] Elevation
        //   [10] DataPeriod
        // So TimeZone is at split index 8.
        let utc_offset_hours = parts
            .get(8)
            .map(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(None);

        if city.is_empty() && state.is_empty() {
            return Ok(None);
        }

        Ok(Some(EpwLocation {
            city_state: format!("{}, {}", city, state),
            utc_offset_hours,
        }))
    }

    /// Parse EPW v3 (sub-hourly) file.
    ///
    /// EPW v3 files contain 35040 sub-hourly records (15-minute timestep).
    /// This is an extension of EPW v2 with higher temporal resolution.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader for EPW v3 file content
    ///
    /// # Returns
    ///
    /// Parsed vector of sub-hourly records
    pub fn parse_epw_v3<R: Read>(reader: R) -> Result<Vec<SubHourlyRecord>, WeatherError> {
        let buffered = BufReader::new(reader);
        let mut records = Vec::new();

        for line in buffered.lines() {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;

            // Skip all 8 EPW header lines and empty lines.
            // Issue #1164: previously only LOCATION/DESIGN CONDITIONS/DATA PERIODS were
            // skipped by prefix; the GROUND TEMPERATURES header (35+ fields) slipped
            // through the field-count guard and became a spurious first record,
            // shifting all real data by one position.
            if is_epw_header_line(&line) || line.is_empty() {
                continue;
            }

            // Parse sub-hourly record
            // EPW v3 has same field structure as v2 but 4x the records
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() < 35 {
                continue; // Skip invalid lines
            }

            let record = SubHourlyRecord {
                year: fields[0].parse::<u16>().unwrap_or(2020),
                month: fields[1].parse::<u8>().unwrap_or(1),
                day: fields[2].parse::<u8>().unwrap_or(1),
                hour: fields[3].parse::<u8>().unwrap_or(0),
                minute: fields[4].parse::<u8>().unwrap_or(0),
                dry_bulb_temp: fields[6].parse::<f64>().unwrap_or(0.0),
                humidity: fields[8].parse::<f64>().unwrap_or(50.0),
                // Issue #829 fix: standard EPW v3 columns are GHI=14, DNI=15, DHI=16.
                // Issue #1415: coerce 9999 missing-data sentinels to 0.0.
                ghi: parse_field_coercing_sentinel(fields[13], EPW_SOLAR_SENTINEL, 0.0),
                dni: parse_field_coercing_sentinel(fields[14], EPW_SOLAR_SENTINEL, 0.0),
                dhi: parse_field_coercing_sentinel(fields[15], EPW_SOLAR_SENTINEL, 0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                // Issue #829 fix: HIR is column 13 (fields[12]); previously read DHI (fields[15]).
                horizontal_infrared: parse_field_coercing_sentinel(
                    fields[12],
                    EPW_SOLAR_SENTINEL,
                    0.0,
                ),
                ground_temperature: None,
                horizontal_illuminance: None,
                diffuse_illuminance: None,
                snow_depth: None,
                snow_cover: None,
                present_weather: None,
                present_weather_code: None,
            };

            records.push(record);
        }

        Ok(records)
    }

    /// Parse AMY (Actual Meteorological Year) file.
    ///
    /// AMY files contain actual historical weather data instead of typical
    /// meteorological year. They have the same structure as EPW v2.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader for AMY file content
    ///
    /// # Returns
    ///
    /// Parsed vector of hourly records
    pub fn parse_epw_amy<R: Read>(reader: R) -> Result<Vec<HourlyRecord>, WeatherError> {
        let buffered = BufReader::new(reader);
        let mut records = Vec::new();

        for line in buffered.lines() {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;

            // Skip all 8 EPW header lines and empty lines (Issue #1164).
            if is_epw_header_line(&line) || line.is_empty() {
                continue;
            }

            // Parse hourly record
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() < 35 {
                continue; // Skip invalid lines
            }

            let record = HourlyRecord {
                year: fields[0].parse::<u16>().unwrap_or(2020),
                month: fields[1].parse::<u8>().unwrap_or(1),
                day: fields[2].parse::<u8>().unwrap_or(1),
                hour: fields[3].parse::<u8>().unwrap_or(0),
                minute: fields[4].parse::<u8>().unwrap_or(0),
                dry_bulb_temp: fields[6].parse::<f64>().unwrap_or(0.0),
                humidity: fields[8].parse::<f64>().unwrap_or(50.0),
                // Issue #829 fix: standard EPW v3 columns are GHI=14, DNI=15, DHI=16.
                // Issue #1415: coerce 9999 missing-data sentinels to 0.0.
                ghi: parse_field_coercing_sentinel(fields[13], EPW_SOLAR_SENTINEL, 0.0),
                dni: parse_field_coercing_sentinel(fields[14], EPW_SOLAR_SENTINEL, 0.0),
                dhi: parse_field_coercing_sentinel(fields[15], EPW_SOLAR_SENTINEL, 0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                // Issue #829 fix: HIR is column 13 (fields[12]); previously read DHI (fields[15]).
                horizontal_infrared: parse_field_coercing_sentinel(
                    fields[12],
                    EPW_SOLAR_SENTINEL,
                    0.0,
                ),
                ground_temperature: None,
                horizontal_illuminance: None,
                diffuse_illuminance: None,
                snow_depth: None,
                snow_cover: None,
                present_weather: None,
                present_weather_code: None,
            };

            records.push(record);
        }

        Ok(records)
    }

    /// Parse IWEC (International Weather for Energy Calculations) file.
    ///
    /// IWEC files provide weather data for international locations outside
    /// US TMY3 coverage. Similar to EPW v2 with minor variations.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader for IWEC file content
    ///
    /// # Returns
    ///
    /// Parsed vector of hourly records
    pub fn parse_epw_iwec<R: Read>(reader: R) -> Result<Vec<HourlyRecord>, WeatherError> {
        let buffered = BufReader::new(reader);
        let mut records = Vec::new();

        for line in buffered.lines() {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;

            // Skip all 8 EPW header lines and empty lines (Issue #1164).
            if is_epw_header_line(&line) || line.is_empty() {
                continue;
            }

            // Parse hourly record
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() < 35 {
                continue; // Skip invalid lines
            }

            // IWEC may have different field positions - adjust indices as needed
            // For now, assume same structure as EPW v2
            let record = HourlyRecord {
                year: fields[0].parse::<u16>().unwrap_or(2020),
                month: fields[1].parse::<u8>().unwrap_or(1),
                day: fields[2].parse::<u8>().unwrap_or(1),
                hour: fields[3].parse::<u8>().unwrap_or(0),
                minute: fields[4].parse::<u8>().unwrap_or(0),
                dry_bulb_temp: fields[6].parse::<f64>().unwrap_or(0.0),
                humidity: fields[8].parse::<f64>().unwrap_or(50.0),
                // Issue #829 fix: standard EPW v3 columns are GHI=14, DNI=15, DHI=16.
                // Issue #1415: coerce 9999 missing-data sentinels to 0.0.
                ghi: parse_field_coercing_sentinel(fields[13], EPW_SOLAR_SENTINEL, 0.0),
                dni: parse_field_coercing_sentinel(fields[14], EPW_SOLAR_SENTINEL, 0.0),
                dhi: parse_field_coercing_sentinel(fields[15], EPW_SOLAR_SENTINEL, 0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                // Issue #829 fix: HIR is column 13 (fields[12]); previously read DHI (fields[15]).
                horizontal_infrared: parse_field_coercing_sentinel(
                    fields[12],
                    EPW_SOLAR_SENTINEL,
                    0.0,
                ),
                ground_temperature: None,
                horizontal_illuminance: None,
                diffuse_illuminance: None,
                snow_depth: None,
                snow_cover: None,
                present_weather: None,
                present_weather_code: None,
            };

            records.push(record);
        }

        Ok(records)
    }

    /// Parses a single hourly data line from an EPW file.
    ///
    /// The data line has 35+ comma-separated fields. We extract:
    ///
    /// - Year, Month, Day, Hour, Minute (for validation)
    /// - Dry Bulb Temperature (field 7)
    /// - Relative Humidity (field 9)
    /// - Direct Normal Irradiance (field 11)
    /// - Diffuse Horizontal Irradiance (field 12)
    /// - Global Horizontal Irradiance (field 13)
    /// - Wind Speed (field 22)
    ///
    /// # Arguments
    ///
    /// * `line` - The data line to parse
    /// * `hour_of_year` - Expected hour index (for validation)
    ///
    /// # Returns
    ///
    /// * `Ok(HourlyWeatherData)` - Parsed weather data
    /// * `Err(WeatherError)` - If parsing fails
    fn parse_data_line(line: &str, hour_of_year: usize) -> Result<HourlyWeatherData, WeatherError> {
        let fields: Vec<&str> = line.split(',').collect();

        // EPW data lines should have at least 35 fields
        if fields.len() < 35 {
            return Err(WeatherError::ParseError(format!(
                "Expected at least 35 fields, found {} on line {}",
                fields.len(),
                hour_of_year + 1
            )));
        }

        // Helper function to parse optional numeric fields
        fn parse_field(field: &str, field_name: &str) -> Result<f64, WeatherError> {
            let trimmed = field.trim();

            if trimmed.is_empty() {
                return Err(WeatherError::IncompleteData(format!(
                    "Missing {} field",
                    field_name
                )));
            }

            trimmed.parse::<f64>().map_err(|_| {
                WeatherError::ParseError(format!("Invalid {} value: '{}'", field_name, trimmed))
            })
        }

        // === Issue #829 FIX: correct EPW v3 field indices ===
        // Standard EPW v3 columns (1-indexed in spec → 0-indexed in `fields[]`):
        //   col 7  fields[6]  = Dry Bulb Temperature (°C)
        //   col 8  fields[7]  = Dew Point Temperature (°C)        ← previously misread as dry bulb
        //   col 9  fields[8]  = Relative Humidity (%)
        //   col 13 fields[12] = Horizontal Infrared Radiation Intensity (W/m²)
        //   col 14 fields[13] = Global Horizontal Radiation (Wh/m²)
        //   col 15 fields[14] = Direct Normal Radiation (Wh/m²)
        //   col 16 fields[15] = Diffuse Horizontal Radiation (Wh/m²)
        //   col 22 fields[21] = Wind Speed (m/s)
        //
        // Previously this function read dry-bulb from the dew-point column, and
        // DNI/DHI/GHI/HIR from the extraterrestrial-radiation and horizontal-IR
        // columns. That caused winter outdoor temps to be ~7 °C colder than
        // reality and the Perez sky model to receive ~1232 W/m² (Extraterrestrial
        // Direct Normal) labelled as Diffuse Horizontal, which the circumsolar
        // term then amplified to ~286 kW/m² on horizontal roof surfaces.
        let dry_bulb_temp = parse_field(fields[6], "dry bulb temperature")?;
        let humidity = parse_field(fields[8], "relative humidity")?;
        // Issue #1415: coerce EPW missing-data sentinels (9999) to 0.0 for
        // GHI/DNI/DHI and horizontal infrared, matching the EnergyPlus Weather
        // Converter. Without this, 9999 W/m² propagates into the Perez model
        // and produces nonsensical sol-air temperatures.
        let ghi = parse_field_coercing_sentinel(fields[13], EPW_SOLAR_SENTINEL, 0.0);
        let dni = parse_field_coercing_sentinel(fields[14], EPW_SOLAR_SENTINEL, 0.0);
        let dhi = parse_field_coercing_sentinel(fields[15], EPW_SOLAR_SENTINEL, 0.0);
        let wind_speed = parse_field(fields[21], "wind speed")?;
        let horizontal_infrared =
            parse_field_coercing_sentinel(fields[12], EPW_SOLAR_SENTINEL, 0.0);

        // Parse optional fields (may be missing in some EPW files)
        // Ground temperature (field 23 in 0-indexed array, if available)
        let ground_temperature = if fields.len() > 23 {
            let val = parse_optional_field(fields[23], f64::NAN);
            if val.is_nan() {
                None
            } else {
                Some(val)
            }
        } else {
            None
        };

        // Horizontal illuminance (field 24 in 0-indexed array, if available)
        let horizontal_illuminance = if fields.len() > 24 {
            let val = parse_optional_field(fields[24], f64::NAN);
            if val.is_nan() {
                None
            } else {
                Some(val)
            }
        } else {
            None
        };

        // Diffuse illuminance (field 25 in 0-indexed array, if available)
        let diffuse_illuminance = if fields.len() > 25 {
            let val = parse_optional_field(fields[25], f64::NAN);
            if val.is_nan() {
                None
            } else {
                Some(val)
            }
        } else {
            None
        };

        // Snow depth (field 26 in 0-indexed array, if available)
        let snow_depth = if fields.len() > 26 {
            let val = parse_optional_field(fields[26], f64::NAN);
            if val.is_nan() {
                None
            } else {
                Some(val)
            }
        } else {
            None
        };

        // Snow cover (field 27 in 0-indexed array, if available)
        let snow_cover = if fields.len() > 27 {
            let val = parse_optional_field(fields[27], f64::NAN);
            if val.is_nan() {
                None
            } else {
                Some(val)
            }
        } else {
            None
        };

        // Present weather observation (field 22 in 0-indexed array)
        let present_weather_code = if fields.len() > 22 {
            let val = parse_optional_field(fields[22], 999999.0);
            if val >= 999999.0 {
                None
            } else {
                Some(val as u32)
            }
        } else {
            None
        };

        // Present weather text (derived from code)
        let present_weather = present_weather_code.map(|code| {
            // Simple mapping from WMO weather codes to text
            match code {
                0 => "Clear",
                1..=3 => "Cloudy",
                4..=7 => "Fog",
                8..=10 => "Rain",
                11..=16 => "Precipitation",
                17..=20 => "Thunderstorm",
                21..=28 => "Snow",
                _ => "Unknown",
            }
            .to_string()
        });

        Ok(HourlyWeatherData {
            dry_bulb_temp,
            dni,
            dhi,
            ghi,
            wind_speed,
            humidity,
            horizontal_infrared,
            hour_of_year,
            ground_temperature,
            horizontal_illuminance,
            diffuse_illuminance,
            snow_depth,
            snow_cover,
            present_weather,
            present_weather_code,
        })
    }

    /// Returns the number of hourly data records available.
    ///
    /// For a complete EPW file, this should be 8760.
    pub fn record_count(&self) -> usize {
        self.hourly_data.len()
    }

    /// Returns the total number of solar radiation hours (hours with GHI > 0).
    ///
    /// This is useful for understanding the solar resource at a location.
    pub fn solar_hours(&self) -> usize {
        self.hourly_data.iter().filter(|d| d.ghi > 0.0).count()
    }

    /// Returns the maximum temperature in the dataset.
    pub fn max_temperature(&self) -> f64 {
        self.hourly_data
            .iter()
            .map(|d| d.dry_bulb_temp)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Returns the minimum temperature in the dataset.
    pub fn min_temperature(&self) -> f64 {
        self.hourly_data
            .iter()
            .map(|d| d.dry_bulb_temp)
            .fold(f64::INFINITY, f64::min)
    }

    /// Returns the average temperature in the dataset.
    pub fn average_temperature(&self) -> f64 {
        if self.hourly_data.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.hourly_data.iter().map(|d| d.dry_bulb_temp).sum();
        sum / self.hourly_data.len() as f64
    }
}

impl WeatherSource for EpwWeatherSource {
    fn location(&self) -> Option<String> {
        self.location.as_ref().map(|loc| loc.city_state.clone())
    }

    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
        if hour >= self.hourly_data.len() {
            return Err(WeatherError::InvalidHour(hour));
        }

        Ok(self.hourly_data[hour].clone())
    }
}

impl EpwWeatherSource {
    /// Returns the UTC time-zone offset (decimal hours) from EPW LOCATION
    /// column 9 (Issue #1416).
    ///
    /// Sign convention matches the longitude convention used by
    /// `crate::solar::solar_position::calculate_solar_position`: positive for
    /// east of Greenwich, negative for west. For Denver (`LOCATION,...,-7.0,...`)
    /// this returns `Some(-7.0)`; for New Delhi (`LOCATION,...,5.5,...`) it
    /// returns `Some(5.5)`. `None` if the EPW header has no parseable offset.
    pub fn utc_offset_hours(&self) -> Option<f64> {
        self.location.as_ref().and_then(|loc| loc.utc_offset_hours)
    }

    /// Returns the structured EPW LOCATION metadata.
    pub fn location_struct(&self) -> Option<&EpwLocation> {
        self.location.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    /// Creates a minimal valid EPW file for testing.
    fn create_test_epw() -> String {
        // Location header: Denver, CO
        let location_line =
            "LOCATION,Denver,CO,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0,1991-2005";

        // Additional headers (design conditions, periods)
        let design_conditions = "DESIGN CONDITIONS,0";
        let extreme_periods = "EXTREME PERIODS,0";
        let typical_periods = "TYPICAL/EXTREME PERIODS,0";
        let ground_temps = "GROUND TEMPERATURES,0";
        let holidays = "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0";
        let comments1 = "COMMENTS 1,Generated by Fluxion tests";
        let comments2 = "COMMENTS 2,Test data";

        // Issue #829: rewritten to match canonical EPW v3 column layout
        // (col 7 = dry bulb, col 14 = GHI, col 15 = DNI, col 16 = DHI, etc.)
        let data_lines = [
            "1991,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,0.0,-5.0,50,101325,0,0,300,0,0,0,0,0,0,0,0,3.5,180,0,0,0,0,0,0,0,0,0,0,0,0",
            "1991,1,1,2,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,-2.0,-7.0,45,101325,0,0,300,0,0,0,0,0,0,0,0,3.2,180,0,0,0,0,0,0,0,0,0,0,0,0",
            "1991,7,15,12,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,32.0,12.0,20,101325,0,0,400,900,800,100,0,0,0,0,0,2.5,180,0,0,0,0,0,0,0,0,0,0,0,0"
        ];

        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
            location_line,
            design_conditions,
            extreme_periods,
            typical_periods,
            ground_temps,
            holidays,
            comments1,
            comments2,
            data_lines.join("\n")
        )
    }

    #[test]
    fn test_parse_location() {
        let location_line =
            "LOCATION,Denver,CO,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0,1991-2005";

        let result = EpwWeatherSource::parse_location(location_line).unwrap();
        let expected = EpwLocation {
            city_state: "Denver, CO".to_string(),
            utc_offset_hours: Some(-7.0),
        };
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_parse_location_empty() {
        let location_line = "LOCATION,,,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0";

        let result = EpwWeatherSource::parse_location(location_line).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_data_line() {
        let line = "1991,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,0.0,-5.0,50,101325,0,0,300,900,800,100,0,0,0,0,0,3.5,180,0,0,0,0,0,0,0,0,0,0,0,0";

        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();

        assert_eq!(result.dry_bulb_temp, 0.0);
        assert_eq!(result.humidity, 50.0);
        assert_eq!(result.dni, 800.0);
        assert_eq!(result.dhi, 100.0);
        assert_eq!(result.ghi, 900.0);
        assert_eq!(result.wind_speed, 3.5);
        assert_eq!(result.hour_of_year, 0);
    }

    #[test]
    fn test_parse_data_line_missing_fields() {
        let line = "1991,1,1,1,0,0,99,0.0,50";

        let result = EpwWeatherSource::parse_data_line(line, 0);
        assert!(result.is_err());
        match result {
            Err(WeatherError::ParseError(_)) => {}
            _ => panic!("Expected ParseError"),
        }
    }

    // ── Issue #1415: EPW missing-data sentinel (9999) handling ───────────────

    #[test]
    fn test_is_epw_sentinel() {
        assert!(is_epw_sentinel(9999.0, EPW_SOLAR_SENTINEL));
        assert!(is_epw_sentinel(9999.5, EPW_SOLAR_SENTINEL)); // >= check
        assert!(!is_epw_sentinel(9998.9, EPW_SOLAR_SENTINEL));
        assert!(!is_epw_sentinel(0.0, EPW_SOLAR_SENTINEL));

        assert!(is_epw_sentinel(999900.0, EPW_PRESSURE_SENTINEL));
        assert!(is_epw_sentinel(999.0, EPW_PRECIPITATION_SENTINEL));
    }

    #[test]
    fn test_parse_field_coercing_sentinel_solar() {
        assert_eq!(
            parse_field_coercing_sentinel("9999", EPW_SOLAR_SENTINEL, 0.0),
            0.0
        );
        assert_eq!(
            parse_field_coercing_sentinel("850", EPW_SOLAR_SENTINEL, 0.0),
            850.0
        );
        assert_eq!(
            parse_field_coercing_sentinel("", EPW_SOLAR_SENTINEL, 0.0),
            0.0
        );
    }

    #[test]
    fn test_parse_field_coercing_sentinel_pressure() {
        assert_eq!(
            parse_field_coercing_sentinel(
                "999900",
                EPW_PRESSURE_SENTINEL,
                DEFAULT_ATMOSPHERIC_PRESSURE
            ),
            DEFAULT_ATMOSPHERIC_PRESSURE
        );
        assert_eq!(
            parse_field_coercing_sentinel(
                "101325",
                EPW_PRESSURE_SENTINEL,
                DEFAULT_ATMOSPHERIC_PRESSURE
            ),
            101325.0
        );
    }

    #[test]
    fn test_parse_field_coercing_sentinel_precipitation() {
        assert_eq!(
            parse_field_coercing_sentinel("999", EPW_PRECIPITATION_SENTINEL, 0.0),
            0.0
        );
        assert_eq!(
            parse_field_coercing_sentinel("2.5", EPW_PRECIPITATION_SENTINEL, 0.0),
            2.5
        );
    }

    /// Builds a 36-field EPW data line with the given GHI/DNI/DHI/HIR values.
    fn make_data_line(ghi: &str, dni: &str, dhi: &str, hir: &str) -> String {
        format!(
            "1991,1,1,1,0,0,0.0,-5.0,50,101325,0,0,{hir},{ghi},{dni},{dhi},0,0,0,0,0,3.5,180,0,0,0,0,0,0,0,0,0,0,0,0"
        )
    }

    #[test]
    fn test_parse_data_line_9999_ghi_becomes_zero() {
        let line = make_data_line("9999", "800", "100", "300");
        let result = EpwWeatherSource::parse_data_line(&line, 0).unwrap();
        assert_eq!(result.ghi, 0.0, "GHI=9999 sentinel must coerce to 0.0");
        assert_eq!(result.dni, 800.0, "DNI should be unaffected");
        assert_eq!(result.dhi, 100.0, "DHI should be unaffected");
    }

    #[test]
    fn test_parse_data_line_9999_dni_becomes_zero() {
        let line = make_data_line("900", "9999", "100", "300");
        let result = EpwWeatherSource::parse_data_line(&line, 0).unwrap();
        assert_eq!(result.dni, 0.0, "DNI=9999 sentinel must coerce to 0.0");
        assert_eq!(result.ghi, 900.0, "GHI should be unaffected");
        assert_eq!(result.dhi, 100.0, "DHI should be unaffected");
    }

    #[test]
    fn test_parse_data_line_9999_dhi_becomes_zero() {
        let line = make_data_line("900", "800", "9999", "300");
        let result = EpwWeatherSource::parse_data_line(&line, 0).unwrap();
        assert_eq!(result.dhi, 0.0, "DHI=9999 sentinel must coerce to 0.0");
        assert_eq!(result.ghi, 900.0, "GHI should be unaffected");
        assert_eq!(result.dni, 800.0, "DNI should be unaffected");
    }

    #[test]
    fn test_parse_data_line_9999_all_solar_become_zero() {
        let line = make_data_line("9999", "9999", "9999", "9999");
        let result = EpwWeatherSource::parse_data_line(&line, 0).unwrap();
        assert_eq!(result.ghi, 0.0);
        assert_eq!(result.dni, 0.0);
        assert_eq!(result.dhi, 0.0);
        assert_eq!(
            result.horizontal_infrared, 0.0,
            "HIR=9999 sentinel must also coerce to 0.0"
        );
    }

    #[test]
    fn test_parse_data_line_9999_horizontal_infrared_becomes_zero() {
        let line = make_data_line("900", "800", "100", "9999");
        let result = EpwWeatherSource::parse_data_line(&line, 0).unwrap();
        assert_eq!(result.horizontal_infrared, 0.0);
        assert_eq!(result.ghi, 900.0);
    }

    #[test]
    fn test_parse_data_line_valid_solar_not_affected() {
        let line = make_data_line("900", "800", "100", "300");
        let result = EpwWeatherSource::parse_data_line(&line, 0).unwrap();
        assert_eq!(result.ghi, 900.0);
        assert_eq!(result.dni, 800.0);
        assert_eq!(result.dhi, 100.0);
        assert_eq!(result.horizontal_infrared, 300.0);
    }

    // ── End Issue #1415 tests ────────────────────────────────────────────────

    #[test]
    fn test_parse_complete_epw() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);

        let source = EpwWeatherSource::parse(cursor).unwrap();

        assert_eq!(source.location(), Some("Denver, CO".to_string()));
        assert_eq!(source.record_count(), 3);

        // Check first hour
        let hour0 = source.get_hourly_data(0).unwrap();
        assert_eq!(hour0.dry_bulb_temp, 0.0);
        assert_eq!(hour0.humidity, 50.0);
        assert_eq!(hour0.wind_speed, 3.5);

        // Check second hour
        let hour1 = source.get_hourly_data(1).unwrap();
        assert_eq!(hour1.dry_bulb_temp, -2.0);
        assert_eq!(hour1.humidity, 45.0);

        // Check third hour
        let hour2 = source.get_hourly_data(2).unwrap();
        assert_eq!(hour2.dry_bulb_temp, 32.0);
        assert_eq!(hour2.dni, 800.0);
        assert_eq!(hour2.dhi, 100.0);
        assert_eq!(hour2.ghi, 900.0);
    }

    #[test]
    fn test_weather_source_trait() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();

        // Test location
        assert_eq!(source.location(), Some("Denver, CO".to_string()));

        // Test get_hourly_data
        let data = source.get_hourly_data(0).unwrap();
        assert_eq!(data.dry_bulb_temp, 0.0);

        // Test invalid hour
        let error = source.get_hourly_data(10);
        assert_eq!(error, Err(WeatherError::InvalidHour(10)));
    }

    #[test]
    fn test_weather_iterator() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();

        let mut count = 0;
        for result in source.iter_hours() {
            assert!(result.is_ok());
            count += 1;
        }

        assert_eq!(count, 3);
    }

    #[test]
    fn test_statistics_methods() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();

        assert_eq!(source.record_count(), 3);
        assert_eq!(source.solar_hours(), 1); // Only hour 2 has GHI > 0
        assert_eq!(source.max_temperature(), 32.0);
        assert_eq!(source.min_temperature(), -2.0);
        assert_eq!(source.average_temperature(), 10.0); // (0 + -2 + 32) / 3
    }

    #[test]
    fn test_parse_optional_field() {
        // Test with valid value
        assert_eq!(super::parse_optional_field("100.0", 0.0), 100.0);

        // Test with empty string (should use default)
        assert_eq!(super::parse_optional_field("", 50.0), 50.0);

        // Test with whitespace (should be trimmed to empty, use default)
        assert_eq!(super::parse_optional_field("   ", 50.0), 50.0);

        // Test with invalid number (should use default)
        assert_eq!(super::parse_optional_field("invalid", 50.0), 50.0);
    }

    #[test]
    fn test_is_epw_header_line() {
        // All 8 standard EPW header prefixes are recognised.
        assert!(super::is_epw_header_line(
            "LOCATION,Denver,CO,USA,TMY3,724666,39.74,-105.18,-7.0,1829.0"
        ));
        assert!(super::is_epw_header_line("DESIGN CONDITIONS,1"));
        assert!(super::is_epw_header_line("TYPICAL/EXTREME PERIODS,6"));
        assert!(super::is_epw_header_line(
            "GROUND TEMPERATURES,3,.5,,,,1.34,5.12"
        ));
        assert!(super::is_epw_header_line(
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0"
        ));
        assert!(super::is_epw_header_line("COMMENTS 1,Custom/User Format"));
        assert!(super::is_epw_header_line("COMMENTS 2, -- Ground temps"));
        assert!(super::is_epw_header_line(
            "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31"
        ));

        // Data lines (start with a 4-digit year) are NOT headers.
        assert!(!super::is_epw_header_line(
            "1999,1,1,1,0,?9?9?9?9E0,-3.0,-4.0,92,80600,0,0,257,0,0,0,0,0,0,0,0,0.0,9,8,16.1"
        ));
        assert!(!super::is_epw_header_line(""));
    }

    /// Builds a minimal EPW string that includes a `GROUND TEMPERATURES`
    /// header with 35+ fields — the exact shape that triggered Issue #1164.
    fn create_epw_with_ground_temps_header() -> String {
        let location = "LOCATION,Denver,CO,USA,TMY3,724666,39.74,-105.18,-7.0,1829.0";
        let design = "DESIGN CONDITIONS,1";
        let typical = "TYPICAL/EXTREME PERIODS,6";
        // GROUND TEMPERATURES with 3 depths × 12 monthly values = 36+ fields.
        let ground = "GROUND TEMPERATURES,3,0.5,,,,-0.6,1.3,5.1,8.7,15.5,19.0,20.0,18.2,14.0,8.8,3.7,0.3,2,2.1,2.6,4.7,7.1,12.3,15.6,17.3,16.9,14.5,10.9,6.9,3.7,4,4.8,4.5,5.5,6.8";
        let holidays = "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0";
        let comments1 = "COMMENTS 1,Test data";
        let comments2 = "COMMENTS 2,Issue 1164 regression";
        let data_periods = "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31";

        // Two real data rows.
        let data = [
            "1999,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9*9*9,-3.0,-4.0,92,80600,0,0,257,0,0,0,0,0,0,0,0,0.0,9,8,16.1,3300,9,999999999,89,0.0310,0,88,0.330,999.0,99.0",
            "1999,1,1,2,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9*9*9,-2.0,-6.0,77,80600,0,0,261,0,0,0,0,0,0,0,170,2.1,10,9,16.1,3000,9,999999999,89,0.0310,0,88,0.330,999.0,99.0",
        ];

        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
            location,
            design,
            typical,
            ground,
            holidays,
            comments1,
            comments2,
            data_periods,
            data.join("\n")
        )
    }

    #[test]
    fn test_parse_epw_v3_skips_ground_temps_header() {
        // Issue #1164 regression test: the GROUND TEMPERATURES header line
        // has 35+ fields and must NOT be parsed as a data record.
        let epw = create_epw_with_ground_temps_header();
        let records = EpwWeatherSource::parse_epw_v3(Cursor::new(epw)).unwrap();

        // Exactly 2 data rows, NOT 3 (the header must not sneak in).
        assert_eq!(
            records.len(),
            2,
            "GROUND TEMPERATURES header must not be parsed as a data record"
        );

        // First record must be the real hour-1 row, not the header garbage.
        assert_eq!(records[0].year, 1999);
        assert_eq!(records[0].month, 1);
        assert_eq!(records[0].day, 1);
        assert_eq!(records[0].hour, 1, "first record must be EPW hour 1");
        assert_eq!(records[0].dry_bulb_temp, -3.0);
        assert_eq!(records[0].hour, 1);
    }

    #[test]
    fn test_parse_epw_v3_matches_trait_path() {
        // parse_epw_v3 and the WeatherSource trait path (parse) must produce
        // identical field values at matching indices for the same file.
        let epw_data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/test_data/denver.epw"
        ));
        let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(&epw_data[..])).unwrap();
        let source = EpwWeatherSource::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/test_data/denver.epw"
        ))
        .unwrap();

        // Same record count (8760) — no spurious header record in parse_epw_v3.
        assert_eq!(v3.len(), source.record_count());
        assert_eq!(v3.len(), 8760);

        // Spot-check DNI/DHI/temperature alignment at several hours.
        for &h in &[0usize, 1, 7, 100, 1000, 5000, 8759] {
            let trait_data = source.get_hourly_data(h).unwrap();
            assert_eq!(v3[h].dni, trait_data.dni, "DNI mismatch at hour {}", h);
            assert_eq!(v3[h].dhi, trait_data.dhi, "DHI mismatch at hour {}", h);
            assert_eq!(
                v3[h].dry_bulb_temp, trait_data.dry_bulb_temp,
                "dry_bulb_temp mismatch at hour {}",
                h
            );
        }
    }

    #[test]
    fn test_from_file_not_found() {
        let result = EpwWeatherSource::from_file("/nonexistent/path/file.epw");
        assert!(result.is_err());
        match result {
            Err(WeatherError::IoError(_)) => {}
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_parse_incomplete_header() {
        let epw_content = "LOCATION,Denver,CO,USA";
        let cursor = Cursor::new(epw_content);

        let result = EpwWeatherSource::parse(cursor);
        assert!(result.is_err());
        match result {
            Err(WeatherError::IncompleteData(_)) => {}
            _ => panic!("Expected IncompleteData error"),
        }
    }

    #[test]
    fn test_parse_comment_lines() {
        let mut epw_content = create_test_epw();
        // Add a comment line between data lines
        epw_content = epw_content.replace("\n1991,1,1,2,0", "\n! This is a comment\n1991,1,1,2,0");

        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();

        // Should skip the comment and parse 3 data lines
        assert_eq!(source.record_count(), 3);
    }

    #[test]
    fn test_from_file_valid_epw() {
        let source = EpwWeatherSource::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/test_data/test_denver.epw"
        ));
        assert!(source.is_ok());
        let source = source.unwrap();
        assert_eq!(source.location(), Some("Denver, CO".to_string()));
        assert_eq!(source.record_count(), 6);
        assert_eq!(source.max_temperature(), 34.0);
        assert_eq!(source.min_temperature(), -6.0);
    }

    #[test]
    fn test_from_file_empty_epw() {
        let result = EpwWeatherSource::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/test_data/test_empty.epw"
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_nonexistent() {
        let result = EpwWeatherSource::from_file("/nonexistent/file.epw");
        assert!(result.is_err());
        match result {
            Err(WeatherError::IoError(_)) => {}
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_parse_location_short_header() {
        let line = "LOCATION,OnlyCity";
        let result = EpwWeatherSource::parse_location(line).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_location_with_empty_city() {
        let line = "LOCATION,,CA,USA,TMY3,000000";
        let result = EpwWeatherSource::parse_location(line).unwrap();
        // City empty but state present → returns Some with empty city.
        // No TimeZone column present in this 6-field line, so utc_offset_hours is None.
        let expected = EpwLocation {
            city_state: ", CA".to_string(),
            utc_offset_hours: None,
        };
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_detect_epw_version_v2() {
        let content = "LOCATION,Denver,CO,USA,TMY3\nDATA PERIODS,1,1,Data,Monday,1,1,12,31,24\n";
        let mut cursor = Cursor::new(content);
        let version = detect_epw_version(&mut cursor).unwrap();
        assert_eq!(version, EpwVersion::V2);
    }

    #[test]
    fn test_detect_epw_version_v3() {
        let content = "LOCATION,Denver,CO,USA,TMY3\nDATA PERIODS,1,1,Data,Monday,1,1,12,31,15\n";
        let mut cursor = Cursor::new(content);
        let version = detect_epw_version(&mut cursor).unwrap();
        assert_eq!(version, EpwVersion::V3);
    }

    #[test]
    fn test_detect_epw_version_iwec() {
        let content = "LOCATION,Denver,CO,USA,IWEC\n";
        let mut cursor = Cursor::new(content);
        let version = detect_epw_version(&mut cursor).unwrap();
        assert_eq!(version, EpwVersion::IWEC);
    }

    #[test]
    fn test_detect_epw_version_tmy() {
        let content = "LOCATION,Denver,CO,USA,TMY2\n";
        let mut cursor = Cursor::new(content);
        let version = detect_epw_version(&mut cursor).unwrap();
        assert_eq!(version, EpwVersion::V2);
    }

    #[test]
    fn test_detect_epw_version_default() {
        let content = "SOME RANDOM HEADER\n";
        let mut cursor = Cursor::new(content);
        let version = detect_epw_version(&mut cursor).unwrap();
        assert_eq!(version, EpwVersion::V2);
    }

    #[test]
    fn test_parse_epw_v3() {
        // EPW v3/AMY/IWEC parsers use fields[6] for temperature, fields[8] for humidity
        // Must have at least 35 fields to be parsed
        let content = "LOCATION,Denver,CO,USA,TMY3\nDATA PERIODS,1,1,Data,Monday,1,1,12,31,15\n1991,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,0.0,-5.0,50,101325,0,0,300,900,800,100,0,0,0,0,0,3.5,180,0,0,0,0,0,0,0,0,0,0,0,0\n";
        let cursor = Cursor::new(content);
        let records = EpwWeatherSource::parse_epw_v3(cursor).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dry_bulb_temp, 0.0);
        assert_eq!(records[0].humidity, 50.0);
        assert_eq!(records[0].dni, 800.0);
    }

    #[test]
    fn test_parse_epw_amy() {
        // Parser uses fields[6] for temp, fields[8] for humidity
        // Fields: 0=year, 1=month, 2=day, 3=hour, 4=minute, 5=?, 6=temp, 7=?, 8=humidity
        let content = "LOCATION,Denver,CO,USA,AMY\nDATA PERIODS,1,1,Data,Monday,1,1,12,31,24\n1991,1,1,1,0,0,5.0,0,40,1,0,0,0,0,0,0,0,0,0,0,0,2.0,150,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0\n";
        let cursor = Cursor::new(content);
        let records = EpwWeatherSource::parse_epw_amy(cursor).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dry_bulb_temp, 5.0);
        assert_eq!(records[0].humidity, 40.0);
    }

    #[test]
    fn test_parse_epw_iwec() {
        let content = "LOCATION,Denver,CO,USA,IWEC\nDATA PERIODS,1,1,Data,Monday,1,1,12,31,24\n1991,1,1,1,0,0,10.0,0,60,1,0,0,0,0,0,0,0,0,0,0,0,1.5,120,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0\n";
        let cursor = Cursor::new(content);
        let records = EpwWeatherSource::parse_epw_iwec(cursor).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dry_bulb_temp, 10.0);
        assert_eq!(records[0].humidity, 60.0);
    }

    #[test]
    fn test_parse_data_line_negative_temp() {
        let line = "1991,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,-25.0,-30.0,80,101325,0,0,300,0,0,0,0,0,0,0,0,5.0,180,0,0,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.dry_bulb_temp, -25.0);
        assert_eq!(result.humidity, 80.0);
    }

    #[test]
    fn test_parse_data_line_high_solar() {
        let line = "1991,7,15,12,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,35.0,12.0,20,101325,0,0,400,1150,1000,150,0,0,0,0,0,1.0,180,0,0,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.dni, 1000.0);
        assert_eq!(result.dhi, 150.0);
        assert_eq!(result.ghi, 1150.0);
    }

    #[test]
    fn test_parse_data_line_optional_fields() {
        // Fields 23-27 for optional data
        let line = "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,180,10.0,20000.0,5000.0,5.0,50.0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert!(result.ground_temperature.is_some());
        assert!(result.horizontal_illuminance.is_some());
        assert!(result.diffuse_illuminance.is_some());
        assert!(result.snow_depth.is_some());
        assert!(result.snow_cover.is_some());
    }

    #[test]
    fn test_parse_data_line_present_weather() {
        // Field 22 (0-indexed) = present weather code
        let line = "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,0,5,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert!(result.present_weather.is_some());
        assert_eq!(result.present_weather_code, Some(5));
    }

    #[test]
    fn test_present_weather_mapping_clear() {
        let line = "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.present_weather, Some("Clear".to_string()));
    }

    #[test]
    fn test_present_weather_mapping_cloudy() {
        let line = "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,0,2,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.present_weather, Some("Cloudy".to_string()));
    }

    #[test]
    fn test_present_weather_mapping_rain() {
        let line = "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,0,9,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.present_weather, Some("Rain".to_string()));
    }

    #[test]
    fn test_present_weather_mapping_unknown() {
        let line = "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,0,99,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.present_weather, Some("Unknown".to_string()));
    }

    #[test]
    fn test_statistics_empty() {
        let source = EpwWeatherSource {
            location: Some(EpwLocation {
                city_state: "Test".to_string(),
                utc_offset_hours: Some(0.0),
            }),
            hourly_data: vec![],
        };
        assert_eq!(source.record_count(), 0);
        assert_eq!(source.solar_hours(), 0);
        assert_eq!(source.average_temperature(), 0.0);
    }

    #[test]
    fn test_statistics_single_record() {
        let weather = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.0, 50.0, 0);
        let source = EpwWeatherSource {
            location: Some(EpwLocation {
                city_state: "Test".to_string(),
                utc_offset_hours: Some(0.0),
            }),
            hourly_data: vec![weather],
        };
        assert_eq!(source.record_count(), 1);
        assert_eq!(source.solar_hours(), 1);
        assert_eq!(source.max_temperature(), 25.0);
        assert_eq!(source.min_temperature(), 25.0);
        assert_eq!(source.average_temperature(), 25.0);
    }

    #[test]
    fn test_statistics_multiple_records() {
        let weather1 = HourlyWeatherData::new(10.0, 0.0, 0.0, 0.0, 2.0, 60.0, 0);
        let weather2 = HourlyWeatherData::new(20.0, 500.0, 100.0, 600.0, 3.0, 50.0, 1);
        let weather3 = HourlyWeatherData::new(30.0, 800.0, 150.0, 950.0, 4.0, 40.0, 2);
        let source = EpwWeatherSource {
            location: Some(EpwLocation {
                city_state: "Test".to_string(),
                utc_offset_hours: Some(0.0),
            }),
            hourly_data: vec![weather1, weather2, weather3],
        };
        assert_eq!(source.record_count(), 3);
        assert_eq!(source.solar_hours(), 2);
        assert_eq!(source.max_temperature(), 30.0);
        assert_eq!(source.min_temperature(), 10.0);
        assert!((source.average_temperature() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_weather_source_trait_get_hourly_data_boundary() {
        let weather = HourlyWeatherData::new(20.0, 0.0, 0.0, 0.0, 2.0, 50.0, 0);
        let source = EpwWeatherSource {
            location: Some(EpwLocation {
                city_state: "Test".to_string(),
                utc_offset_hours: Some(0.0),
            }),
            hourly_data: vec![weather],
        };
        let result = source.get_hourly_data(0);
        assert!(result.is_ok());
        let result = source.get_hourly_data(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_weather_source_trait_location_none() {
        let source = EpwWeatherSource {
            location: None,
            hourly_data: vec![],
        };
        assert_eq!(source.location(), None);
    }

    #[test]
    fn test_hourly_record_clone() {
        let record = HourlyRecord {
            year: 1991,
            month: 1,
            day: 1,
            hour: 12,
            minute: 0,
            dry_bulb_temp: 20.0,
            humidity: 50.0,
            dni: 800.0,
            dhi: 100.0,
            ghi: 900.0,
            wind_speed: 3.0,
            horizontal_infrared: 300.0,
            ground_temperature: Some(15.0),
            horizontal_illuminance: Some(50000.0),
            diffuse_illuminance: Some(20000.0),
            snow_depth: Some(0.0),
            snow_cover: Some(0.0),
            present_weather: Some("Clear".to_string()),
            present_weather_code: Some(0),
        };
        let cloned = record.clone();
        assert_eq!(cloned.dry_bulb_temp, record.dry_bulb_temp);
        assert_eq!(cloned.present_weather, record.present_weather);
    }

    #[test]
    fn test_subhourly_record_clone() {
        let record = SubHourlyRecord {
            year: 1991,
            month: 7,
            day: 15,
            hour: 12,
            minute: 15,
            dry_bulb_temp: 32.0,
            humidity: 25.0,
            dni: 900.0,
            dhi: 120.0,
            ghi: 1020.0,
            wind_speed: 2.5,
            horizontal_infrared: 350.0,
            ground_temperature: None,
            horizontal_illuminance: None,
            diffuse_illuminance: None,
            snow_depth: None,
            snow_cover: None,
            present_weather: None,
            present_weather_code: None,
        };
        let cloned = record.clone();
        assert_eq!(cloned.dry_bulb_temp, record.dry_bulb_temp);
        assert_eq!(cloned.minute, 15);
    }

    #[test]
    fn test_epw_version_equality() {
        assert_eq!(EpwVersion::V2, EpwVersion::V2);
        assert_ne!(EpwVersion::V2, EpwVersion::V3);
        assert_eq!(EpwVersion::AMY, EpwVersion::AMY);
        assert_ne!(EpwVersion::IWEC, EpwVersion::V2);
    }

    #[test]
    fn test_epw_version_debug() {
        let debug_v2 = format!("{:?}", EpwVersion::V2);
        assert!(debug_v2.contains("V2"));
        let debug_v3 = format!("{:?}", EpwVersion::V3);
        assert!(debug_v3.contains("V3"));
    }

    // -------------------------------------------------------------------------
    // Property-Based Tests (proptest)
    // Issue #1062: Property-based testing for core math & parsers
    //
    // Tests EPW parser with random valid and invalid inputs.
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_valid_data_line_parsing(
            year in 1991_i32..2025,
            month in 1_u32..13,
            day in 1_u32..29,
            hour in 0_u32..24,
            minute in 0_u32..60,
            temp in -50.0_f64..60.0,
            humidity in 1.0_f64..100.0,
        ) {
            // Generate a valid EPW data line with all required fields (minimum 35)
            // Format: year,month,day,hour,minute,EPW_flag,dry_bulb,dewpoint,humidity,wind_dir,wind_speed, ...
            // Key field indices: [6]=dry_bulb, [8]=humidity, [12]=HIR, [13]=GHI, [14]=DNI, [15]=DHI, [21]=wind_speed
            let line = format!(
                "{},{},{},{},{},?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,{},50,{},180,3.5,300,200,100,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
                year, month, day, hour, minute, temp, humidity
            );
            let result = EpwWeatherSource::parse_data_line(&line, 0);
            prop_assert!(result.is_ok(), "Valid line should parse: {}", line);
        }

        #[test]
        fn prop_invalid_data_line_returns_error(line in "[^,]{1,10},[^,]{1,10},[^,]{1,10}") {
            let result = EpwWeatherSource::parse_data_line(&line, 0);
            prop_assert!(result.is_err(), "Malformed line should fail: {}", line);
        }

        #[test]
        fn prop_truncated_data_line_rejected(line in "[0-9,y,?9*]{10,50}") {
            let result = EpwWeatherSource::parse_data_line(&line, 0);
            prop_assert!(result.is_err(), "Truncated line should fail");
        }
    }

    #[test]
    fn test_parse_data_line_with_various_missing_fields() {
        use std::io::Cursor;

        // Missing optional fields (too few commas)
        let incomplete_lines = [
            "1991,1,1,1,0",    // missing most fields
            "1991,1,1,1,0,?9", // missing fields after EPW flag
            "1991,1,1,1",      // severely truncated
            "",                // empty line
        ];

        for line in incomplete_lines.iter() {
            let result = EpwWeatherSource::parse_data_line(line, 0);
            assert!(result.is_err(), "Should fail to parse: {}", line);
        }
    }

    #[test]
    fn test_parse_location_with_various_formats() {
        // Valid formats
        let valid_locations = [
            "LOCATION,Denver,CO,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0,1991-2005",
            "LOCATION,,,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0",
            "LOCATION,Test City,,USA,AMY,123456,40.0,-100.0,-6.0,500.0",
        ];

        for loc in valid_locations.iter() {
            let result = EpwWeatherSource::parse_location(loc);
            // None of these should panic
            let _ = result;
        }

        // Invalid format should not panic
        let invalid = "NOTLOCATION";
        let result = EpwWeatherSource::parse_location(invalid);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    // -------------------------------------------------------------------------
    // Property-Based Tests (proptest)
    // Issue #1350: Malformed-row handling for the EPW parser.
    //
    // Goal: catch #1164-class regressions (off-by-one shifts from header lines
    // slipping through the field-count guard, silent garbage on non-numeric
    // fields, CRLF/LF ambiguity, BOM marker mishandling, panic on truncation,
    // etc.) WITHOUT modifying the parser. Every test below documents the
    // parser's CURRENT contract — if a future change breaks it, proptest will
    // surface the regression.
    //
    // Naming convention follows the issue's acceptance criteria:
    //   - proptest_parse_epw_truncated_*
    //   - proptest_parse_epw_ground_temperatures_header_*
    //   - proptest_parse_epw_non_numeric_*
    //   - proptest_parse_epw_crlf_*
    //   - proptest_parse_epw_bom_*
    //   - proptest_parse_epw_out_of_range_*
    //   - proptest_parse_epw_well_formed_8760_rows_*
    //
    // All four entry points (parse, parse_epw_v3, parse_epw_amy, parse_epw_iwec)
    // are exercised so that any divergence between the strict `parse()` and
    // the lenient v3/amy/iwec paths is caught.
    // -------------------------------------------------------------------------

    /// Build a 35-field EPW data row with deterministic per-column values.
    /// Used as the "happy path" reference by all property tests below.
    fn make_epw_data_row(
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        dry_bulb: f64,
        dewpoint: f64,
        humidity: f64,
        ghi: f64,
        dni: f64,
        dhi: f64,
        wind_speed: f64,
        wind_dir: f64,
        hir: f64,
    ) -> String {
        // Field layout follows the canonical EPW v3 columns used by the
        // existing parser tests in this module. 35 fields, comma-separated.
        format!(
            "{year},{month},{day},{hour},{minute},\
             ?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,\
             {dry_bulb},{dewpoint},{humidity},101325,{wind_dir},0,{hir},\
             {ghi},{dni},{dhi},0,0,0,0,0,0,{wind_speed},180,0,0,0,0,0,0,0,0,0,0,0,0",
            year = year,
            month = month,
            day = day,
            hour = hour,
            minute = minute,
            dry_bulb = dry_bulb,
            dewpoint = dewpoint,
            humidity = humidity,
            wind_dir = wind_dir,
            hir = hir,
            ghi = ghi,
            dni = dni,
            dhi = dhi,
            wind_speed = wind_speed,
        )
    }

    /// Build a standard 8-line EPW header. The GROUND TEMPERATURES line is
    /// kept short ("GROUND TEMPERATURES,0") so the field-count guard
    /// naturally filters it.
    fn make_epw_header() -> Vec<String> {
        vec![
            "LOCATION,Denver,CO,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0,1991-2005".into(),
            "DESIGN CONDITIONS,0".into(),
            "EXTREME PERIODS,0".into(),
            "TYPICAL/EXTREME PERIODS,0".into(),
            "GROUND TEMPERATURES,0".into(),
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0".into(),
            "COMMENTS 1,Generated by Fluxion tests".into(),
            "COMMENTS 2,Issue #1350 proptest corpus".into(),
        ]
    }

    /// Same as `make_epw_header` but with a GROUND TEMPERATURES line that has
    /// 36+ fields — the exact shape that previously bypassed the field-count
    /// guard and shifted every data record by one (Issue #1164).
    fn make_epw_header_with_wide_ground_temps() -> Vec<String> {
        vec![
            "LOCATION,Denver,CO,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0,1991-2005".into(),
            "DESIGN CONDITIONS,1".into(),
            "TYPICAL/EXTREME PERIODS,6".into(),
            // 36+ fields: 3 depths × 12 monthly values + header tokens
            "GROUND TEMPERATURES,3,0.5,,,,-0.6,1.3,5.1,8.7,15.5,19.0,20.0,18.2,14.0,8.8,3.7,0.3,\
             2,2.1,2.6,4.7,7.1,12.3,15.6,17.3,16.9,14.5,10.9,6.9,3.7,\
             4,4.8,4.5,5.5,6.8"
                .into(),
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0".into(),
            "COMMENTS 1,Test data".into(),
            "COMMENTS 2,Issue 1164 regression".into(),
            "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31".into(),
        ]
    }

    /// Concatenate header lines and data rows using the supplied line ending.
    fn join_epw(headers: &[String], rows: &[String], line_ending: &str) -> String {
        let mut s = String::new();
        for h in headers {
            s.push_str(h);
            s.push_str(line_ending);
        }
        for r in rows {
            s.push_str(r);
            s.push_str(line_ending);
        }
        s
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        // ---------------------------------------------------------------------
        // 1. Well-formed N-row bodies → exactly N records, no off-by-one.
        //    Uses a moderate body size so the test runs in finite time
        //    even at 10k cases; the off-by-one property (#1164) manifests
        //    at any body length, not specifically at 8760.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_well_formed_n_rows_count_invariant(
            seed in 0_u32..10_000,
            n_rows in 50_usize..200_usize,
        ) {
            // Deterministically generate `n_rows` rows from `seed` and
            // assert the count invariant across all four parser entry points.
            let headers = make_epw_header();
            let mut rows: Vec<String> = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let month = ((i / 30) % 12 + 1) as u8;
                let day = ((i % 28) + 1) as u8;
                let hour = ((i % 24) + 1) as u8;
                let temp = 15.0 + ((seed as f64) * 0.001).sin() * 10.0;
                rows.push(make_epw_data_row(
                    1991, month, day, hour, 0,
                    temp, temp - 5.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ));
            }

            let epw = join_epw(&headers, &rows, "\n");

            // The strict `parse()` path must succeed and return exactly
            // `n_rows` records — no off-by-one shift from header lines,
            // no double-counting, no losses from CRLF.
            let source = EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
                .expect("well-formed EPW should parse");
            prop_assert_eq!(source.record_count(), n_rows,
                "record count drifted off n_rows (off-by-one regression?)");

            // Spot-check the first record stays at hour_of_year=0.
            let h0 = source.get_hourly_data(0).expect("hour 0 readable");
            prop_assert_eq!(h0.hour_of_year, 0);
            prop_assert!(h0.dry_bulb_temp.is_finite(),
                "first record dry_bulb_temp must be finite");

            // The lenient v3/amy/iwec paths must also yield exactly n_rows.
            let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes()))
                .expect("v3 parser should accept well-formed body");
            prop_assert_eq!(v3.len(), n_rows);

            let amy = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes()))
                .expect("amy parser should accept well-formed body");
            prop_assert_eq!(amy.len(), n_rows);

            let iwec = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes()))
                .expect("iwec parser should accept well-formed body");
            prop_assert_eq!(iwec.len(), n_rows);
        }

        // ---------------------------------------------------------------------
        // 2. Truncated body (<100 rows) → graceful Err from strict `parse()`,
        //    Ok(N) from lenient v3/amy/iwec with N matching actual row count.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_truncated_body_under_100_rows_no_panic(
            // Range 0..99 keeps the body under 100 rows as required by the
            // issue acceptance criteria.
            n_rows in 0_usize..100_usize,
            truncate_header in any::<bool>(),
        ) {
            let headers = make_epw_header();
            let rows: Vec<String> = (0..n_rows).map(|i| {
                make_epw_data_row(
                    1991, 1, ((i % 28) + 1) as u8, ((i % 24) + 1) as u8, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                )
            }).collect();

            // Truncate the body by removing the last K rows in a random range.
            let k = if n_rows == 0 { 0 } else { (n_rows / 2) + 1 };
            let truncated: Vec<String> = rows.iter().take(n_rows.saturating_sub(k)).cloned().collect();

            // Two scenarios:
            //   truncate_header=false  → headers + truncated rows
            //   truncate_header=true   → truncate IN THE HEADER (no data section)
            let (test_headers, test_rows) = if truncate_header {
                // Keep only the first 3 header lines (LOCATION + 2 others).
                // `parse()` should refuse with IncompleteData.
                (headers.into_iter().take(3).collect::<Vec<_>>(), Vec::new())
            } else {
                (headers, truncated)
            };

            let epw = join_epw(&test_headers, &test_rows, "\n");
            let cursor = Cursor::new(epw.as_bytes());

            // `parse()` must NOT panic. Two valid outcomes:
            //   a) truncate_header=true  → Err (incomplete header)
            //   b) truncate_header=false → Ok(actual_data_row_count) or Err
            //      (when the data section is empty AND no rows were truncated
            //      to zero).
            // Either way, the function must return — never panic.
            let parse_result = std::panic::catch_unwind(|| {
                EpwWeatherSource::parse(cursor)
            });
            prop_assert!(parse_result.is_ok(), "`parse()` panicked on truncated input");
            let parse_result = parse_result.unwrap();

            if truncate_header {
                // Header truncation MUST yield Err, never silently succeed.
                prop_assert!(parse_result.is_err(),
                    "truncated header should yield Err from `parse()`, got {:?}", parse_result);
            } else {
                // Body truncation: if at least one data row remains, Ok.
                // If zero rows remain, `parse()` returns Err (no valid data lines).
                let expected_count = n_rows.saturating_sub(k);
                match parse_result {
                    Ok(source) => prop_assert_eq!(source.record_count(), expected_count,
                        "truncated body record count must match actual rows"),
                    Err(_) => prop_assert_eq!(expected_count, 0,
                        "Err is only acceptable when zero rows remain"),
                }
            }

            // The lenient v3/amy/iwec paths must accept the truncated body
            // and return exactly `expected_count` records (or Ok(0) when
            // no rows remain).
            let expected_count = if truncate_header { 0 } else {
                n_rows.saturating_sub(k)
            };

            let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes()))
                .expect("v3 should not error on truncated body");
            prop_assert_eq!(v3.len(), expected_count);

            let amy = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes()))
                .expect("amy should not error on truncated body");
            prop_assert_eq!(amy.len(), expected_count);

            let iwec = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes()))
                .expect("iwec should not error on truncated body");
            prop_assert_eq!(iwec.len(), expected_count);
        }

        // ---------------------------------------------------------------------
        // 3. Non-numeric pollution in numeric columns.
        //    Strict `parse()` MUST Err on dry_bulb/humidity/wind_speed.
        //    Lenient v3/amy/iwec MUST silently coerce to default values
        //    (no panic, no garbage that looks like a valid reading).
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_non_numeric_dry_bulb_strict_err_lenient_default(
            // A non-numeric string that no `f64::from_str` could ever parse.
            // Exclude letters that could form "inf"/"infinity"/"nan" — those
            // parse to f64::INFINITY/NaN which is technically valid.
            garbage in "[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ_]{4,12}",
        ) {
            let headers = make_epw_header();
            let rows = vec![
                make_epw_data_row(
                    1991, 1, 1, 1, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ),
                // Replace the dry_bulb field with garbage.
                {
                    let s = make_epw_data_row(
                        1991, 1, 1, 2, 0,
                        20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                    );
                    // Field index 6 (dry_bulb) is the 7th comma-separated token.
                    let mut parts: Vec<&str> = s.split(',').collect();
                    parts[6] = &garbage;
                    parts.join(",")
                },
            ];

            let epw = join_epw(&headers, &rows, "\n");
            let cursor = Cursor::new(epw.as_bytes());

            // Strict `parse()` MUST return Err (parse_field is strict).
            let parse_result = EpwWeatherSource::parse(cursor);
            prop_assert!(parse_result.is_err(),
                "non-numeric dry_bulb must produce Err from `parse()`, got {:?}", parse_result);

            // Lenient v3/amy/iwec MUST silently coerce to 0.0 (default).
            // (The polluted row is the SECOND one — index 1 — since the first
            // row is well-formed.)
            let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes()))
                .expect("v3 should not error");
            prop_assert_eq!(v3.len(), 2);
            prop_assert_eq!(v3[1].dry_bulb_temp, 0.0,
                "non-numeric dry_bulb must coerce to 0.0 in lenient v3, got {}", v3[1].dry_bulb_temp);

            let amy = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes()))
                .expect("amy should not error");
            prop_assert_eq!(amy.len(), 2);
            prop_assert_eq!(amy[1].dry_bulb_temp, 0.0);

            let iwec = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes()))
                .expect("iwec should not error");
            prop_assert_eq!(iwec.len(), 2);
            prop_assert_eq!(iwec[1].dry_bulb_temp, 0.0);
        }

        #[test]
        fn proptest_parse_epw_non_numeric_wind_speed_strict_err_lenient_default(
            // Exclude letters that could form "inf"/"nan" — those parse to
            // valid f64 INFINITY/NaN.
            garbage in "[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ_!@#$%]{4,12}",
        ) {
            let headers = make_epw_header();
            let rows = vec![
                {
                    let s = make_epw_data_row(
                        1991, 1, 1, 1, 0,
                        20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                    );
                    // Field index 21 (wind_speed) is the 22nd comma token.
                    let mut parts: Vec<&str> = s.split(',').collect();
                    parts[21] = &garbage;
                    parts.join(",")
                },
                make_epw_data_row(
                    1991, 1, 1, 2, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ),
            ];

            let epw = join_epw(&headers, &rows, "\n");
            let cursor = Cursor::new(epw.as_bytes());

            // Strict `parse()` MUST return Err.
            prop_assert!(EpwWeatherSource::parse(cursor).is_err(),
                "non-numeric wind_speed must produce Err from `parse()`");

            // Lenient v3/amy/iwec MUST coerce to 0.0.
            // (Note: parse_epw_v3 returns SubHourlyRecord, while amy/iwec
            // return HourlyRecord — collect each separately to avoid type
            // unification issues in the loop. The polluted row is index 0
            // since it's the first row in this test.)
            let v3_records = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(v3_records.len(), 2);
            prop_assert_eq!(v3_records[0].wind_speed, 0.0,
                "non-numeric wind_speed must coerce to 0.0 (v3)");

            let amy_records = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(amy_records.len(), 2);
            prop_assert_eq!(amy_records[0].wind_speed, 0.0,
                "non-numeric wind_speed must coerce to 0.0 (amy)");

            let iwec_records = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(iwec_records.len(), 2);
            prop_assert_eq!(iwec_records[0].wind_speed, 0.0,
                "non-numeric wind_speed must coerce to 0.0 (iwec)");
        }

        // GHI/DNI/DHI are LENIENT in `parse()` (parse_optional_field). So
        // non-numeric pollution in those columns MUST NOT panic — the parser
        // silently coerces invalid floats to 0.0. (Note: Rust's f64::from_str
        // DOES accept "inf", "infinity", "nan" as valid floats — we use a
        // character class that excludes those strings so the test actually
        // exercises the "non-numeric" coercion path.)
        #[test]
        fn proptest_parse_epw_non_numeric_ghi_does_not_panic(
            garbage in "[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ_]{4,8}",
        ) {
            let headers = make_epw_header();
            let rows = vec![
                {
                    let s = make_epw_data_row(
                        1991, 1, 1, 1, 0,
                        20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                    );
                    // Field index 13 (GHI).
                    let mut parts: Vec<&str> = s.split(',').collect();
                    parts[13] = &garbage;
                    parts.join(",")
                },
            ];

            let epw = join_epw(&headers, &rows, "\n");

            // Strict `parse()` does NOT enforce GHI — `parse_optional_field`
            // falls back to 0.0 on parse failure.
            let source = EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
                .expect("lenient GHI column must not poison strict parse()");
            let h0 = source.get_hourly_data(0).expect("hour 0 readable");
            prop_assert_eq!(h0.ghi, 0.0,
                "non-numeric GHI must coerce to 0.0 in strict parse(), got {}", h0.ghi);

            // Lenient v3/amy/iwec also coerce to 0.0.
            let v3_records = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(v3_records.len(), 1);
            prop_assert_eq!(v3_records[0].ghi, 0.0);

            let amy_records = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(amy_records.len(), 1);
            prop_assert_eq!(amy_records[0].ghi, 0.0);

            let iwec_records = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(iwec_records.len(), 1);
            prop_assert_eq!(iwec_records[0].ghi, 0.0);
        }

        // ---------------------------------------------------------------------
        // 4. GROUND TEMPERATURES header with 35+ fields (Issue #1164).
        //    This header previously slipped through the field-count guard
        //    and was parsed as a spurious first record. The fix was to add
        //    an explicit prefix check in `is_epw_header_line`. This test
        //    asserts the regression does NOT recur — across all four parsers.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_ground_temperatures_header_does_not_shift_records(
            // Parameterise on row count so we exercise a range of bodies.
            n_rows in 1_usize..50_usize,
        ) {
            let headers = make_epw_header_with_wide_ground_temps();
            let rows: Vec<String> = (0..n_rows).map(|i| {
                make_epw_data_row(
                    1991, 1, ((i % 28) + 1) as u8, ((i % 24) + 1) as u8, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                )
            }).collect();

            let epw = join_epw(&headers, &rows, "\n");

            // Strict `parse()` — it skips 8 lines by index. So even with the
            // wide GROUND TEMPERATURES header, it should parse `n_rows` rows.
            let source = EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
                .expect("strict parse() must accept wide GROUND TEMPERATURES header");
            prop_assert_eq!(source.record_count(), n_rows,
                "off-by-one regression: record count must equal n_rows, not n_rows+1");
            // First record must be hour 1 (not the header garbage).
            let h0 = source.get_hourly_data(0).expect("hour 0 readable");
            prop_assert_eq!(h0.hour_of_year, 0,
                "first record must be hour_of_year=0, not the GROUND TEMPERATURES header");
            prop_assert!(h0.dry_bulb_temp.is_finite(),
                "first record dry_bulb_temp must be finite (no garbage header leak)");

            // Lenient v3/amy/iwec — must use `is_epw_header_line` prefix check.
            // If the prefix check regresses, the wide header slips through
            // and pushes n_rows to n_rows+1.
            // (Each parser returns a different record type; collect separately.)
            let v3_recs = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(v3_recs.len(), n_rows,
                "off-by-one regression in v3 parser: got {} records, expected {}",
                v3_recs.len(), n_rows);
            prop_assert_eq!(v3_recs[0].year, 1991);
            prop_assert_eq!(v3_recs[0].hour, 1,
                "first record hour must be 1, not the GROUND TEMPERATURES header year");

            let amy_recs = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(amy_recs.len(), n_rows,
                "off-by-one regression in amy parser: got {} records, expected {}",
                amy_recs.len(), n_rows);
            prop_assert_eq!(amy_recs[0].year, 1991);
            prop_assert_eq!(amy_recs[0].hour, 1);

            let iwec_recs = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(iwec_recs.len(), n_rows,
                "off-by-one regression in iwec parser: got {} records, expected {}",
                iwec_recs.len(), n_rows);
            prop_assert_eq!(iwec_recs[0].year, 1991);
            prop_assert_eq!(iwec_recs[0].hour, 1);
        }

        // ---------------------------------------------------------------------
        // 5. CRLF vs LF line endings — identical parsed output.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_crlf_line_endings_match_lf(
            n_rows in 1_usize..30_usize,
            dry_bulb in -30.0_f64..45.0,
        ) {
            let headers = make_epw_header();
            let rows: Vec<String> = (0..n_rows).map(|i| {
                make_epw_data_row(
                    1991, 1, ((i % 28) + 1) as u8, ((i % 24) + 1) as u8, 0,
                    dry_bulb, dry_bulb - 5.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                )
            }).collect();

            let lf = join_epw(&headers, &rows, "\n");
            let crlf = join_epw(&headers, &rows, "\r\n");

            // Both must parse identically under strict `parse()`.
            let lf_source = EpwWeatherSource::parse(Cursor::new(lf.as_bytes()))
                .expect("LF parse");
            let crlf_source = EpwWeatherSource::parse(Cursor::new(crlf.as_bytes()))
                .expect("CRLF parse");
            prop_assert_eq!(lf_source.record_count(), crlf_source.record_count());
            prop_assert_eq!(lf_source.record_count(), n_rows);

            // Spot-check a middle hour — dry_bulb must round-trip identically.
            for h in [0usize, n_rows.saturating_sub(1)].iter().copied() {
                let lf_h = lf_source.get_hourly_data(h).unwrap();
                let crlf_h = crlf_source.get_hourly_data(h).unwrap();
                prop_assert_eq!(lf_h.dry_bulb_temp, crlf_h.dry_bulb_temp,
                    "dry_bulb mismatch at hour {} between LF and CRLF", h);
                prop_assert_eq!(lf_h.dni, crlf_h.dni);
                prop_assert_eq!(lf_h.ghi, crlf_h.ghi);
            }

            // Lenient parsers must also be CRLF-tolerant.
            let lf_v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(lf.as_bytes())).unwrap();
            let crlf_v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(crlf.as_bytes())).unwrap();
            prop_assert_eq!(lf_v3.len(), crlf_v3.len());
            prop_assert_eq!(lf_v3.len(), n_rows);

            // CR-only legacy Mac line endings — also handled by lines() which
            // trims trailing \r. We can't construct CR-only here without
            // affecting the parser semantics, so just assert no panic on CRLF.
        }

        // ---------------------------------------------------------------------
        // 6. BOM marker at file start — must not panic; first record must
        //    remain hour 1 (BOM only contaminates parts[0] of the LOCATION
        //    line which is unused).
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_bom_marker_at_start_does_not_panic(
            n_rows in 1_usize..10_usize,
            place_after_headers in any::<bool>(),
        ) {
            let bom = "\u{feff}";
            let headers = make_epw_header();
            let rows: Vec<String> = (0..n_rows).map(|i| {
                make_epw_data_row(
                    1991, 1, ((i % 28) + 1) as u8, ((i % 24) + 1) as u8, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                )
            }).collect();

            let epw = if place_after_headers {
                // BOM before the 3rd header line — must not panic and must
                // still yield n_rows records (the contaminated header line
                // has too few fields and is filtered out).
                let mut s = String::new();
                for (i, h) in headers.iter().enumerate() {
                    if i == 2 {
                        s.push_str(bom);
                    }
                    s.push_str(h);
                    s.push('\n');
                }
                for r in &rows {
                    s.push_str(r);
                    s.push('\n');
                }
                s
            } else {
                // BOM at file start — LOCATION line becomes
                // "\u{feff}LOCATION,Denver,..." but parse_location tolerates
                // this (only reads parts[1] and parts[2]).
                let mut s = String::new();
                s.push_str(bom);
                for h in &headers {
                    s.push_str(h);
                    s.push('\n');
                }
                for r in &rows {
                    s.push_str(r);
                    s.push('\n');
                }
                s
            };

            // Must not panic on any BOM position.
            let parse_result = std::panic::catch_unwind(|| {
                EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
            });
            prop_assert!(parse_result.is_ok(), "`parse()` panicked on BOM input");

            let v3_result = std::panic::catch_unwind(|| {
                EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes()))
            });
            prop_assert!(v3_result.is_ok(), "`parse_epw_v3()` panicked on BOM input");

            let amy_result = std::panic::catch_unwind(|| {
                EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes()))
            });
            prop_assert!(amy_result.is_ok(), "`parse_epw_amy()` panicked on BOM input");

            let iwec_result = std::panic::catch_unwind(|| {
                EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes()))
            });
            prop_assert!(iwec_result.is_ok(), "`parse_epw_iwec()` panicked on BOM input");
        }

        // ---------------------------------------------------------------------
        // 7. Out-of-range values (month 13, hour 25) — must not panic;
        //    u8::from_str fails → defaults are used.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_out_of_range_month_does_not_panic(
            // Accept either month=13 or month=99 to exercise both.
            month in 13_u32..100_u32,
        ) {
            let headers = make_epw_header();
            let month_str = month.to_string();
            let rows = vec![
                make_epw_data_row(
                    1991, 1, 1, 1, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ),
                {
                    let s = make_epw_data_row(
                        1991, 1, 1, 2, 0,
                        20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                    );
                    let mut parts: Vec<&str> = s.split(',').collect();
                    parts[1] = &month_str;
                    parts.join(",")
                },
            ];

            let epw = join_epw(&headers, &rows, "\n");

            // Strict `parse()` does NOT validate month — it ignores the field
            // entirely. So the parse must succeed with 2 records.
            let source = EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
                .expect("strict parse() must not error on out-of-range month");
            prop_assert_eq!(source.record_count(), 2);

            // Lenient v3/amy/iwec: u8::from_str fails on month>=100, defaults
            // to 1. For 13..=99, u8::from_str fails (overflow), defaults to 1.
            let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes()))
                .expect("v3 must not error");
            prop_assert_eq!(v3.len(), 2);
            prop_assert_eq!(v3[0].month, 1,
                "valid month=1 should round-trip; out-of-range row coerces to default");
        }

        #[test]
        fn proptest_parse_epw_out_of_range_hour_does_not_panic(
            // Accept either hour=25 or hour=99.
            hour in 25_u32..200_u32,
        ) {
            let headers = make_epw_header();
            let hour_str = hour.to_string();
            let rows = vec![
                make_epw_data_row(
                    1991, 1, 1, 1, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ),
                {
                    let s = make_epw_data_row(
                        1991, 1, 1, 1, 0,
                        20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                    );
                    let mut parts: Vec<&str> = s.split(',').collect();
                    parts[3] = &hour_str;
                    parts.join(",")
                },
            ];

            let epw = join_epw(&headers, &rows, "\n");

            // Strict parse() does not validate hour — must succeed with 2 rows.
            let source = EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
                .expect("strict parse() must not error on out-of-range hour");
            prop_assert_eq!(source.record_count(), 2);

            // Lenient v3/amy/iwec: u8::from_str fails on hour>=200, defaults
            // to 0. For 25..199, u8::from_str fails (overflow), defaults to 0.
            let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes()))
                .expect("v3 must not error");
            prop_assert_eq!(v3.len(), 2);
        }

        // ---------------------------------------------------------------------
        // 8. Wrong field count — too few (<35) or too many (>=35).
        //    Strict `parse()` MUST Err on too-few fields.
        //    Lenient v3/amy/iwec MUST skip lines with too-few fields.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_too_few_fields_strict_err_lenient_skip(
            n_fields in 0_usize..35_usize,
        ) {
            let headers = make_epw_header();
            // Build a row with exactly n_fields comma-separated tokens (always
            // <35, so the field-count guard rejects it).
            let mut parts: Vec<String> = (0..n_fields).map(|i| i.to_string()).collect();
            // Force the first token to be a year-like value so the line
            // could plausibly be a data row.
            if !parts.is_empty() {
                parts[0] = "1991".to_string();
            }
            let row = parts.join(",");
            let rows = vec![row, make_epw_data_row(
                1991, 1, 1, 2, 0,
                20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
            )];

            let epw = join_epw(&headers, &rows, "\n");
            let cursor = Cursor::new(epw.as_bytes());

            // Strict parse() — MUST Err because the first row has <35 fields.
            // (Empty n_fields → split yields [""] → 1 token < 35 → Err.)
            prop_assert!(EpwWeatherSource::parse(cursor).is_err(),
                "too-few fields must produce Err from `parse()`");

            // Lenient v3/amy/iwec MUST skip the short row and parse only the
            // second row.
            let v3_recs = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(v3_recs.len(), 1,
                "v3 must skip the too-few row and yield only the valid row");
            prop_assert_eq!(v3_recs[0].year, 1991);

            let amy_recs = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(amy_recs.len(), 1,
                "amy must skip the too-few row and yield only the valid row");
            prop_assert_eq!(amy_recs[0].year, 1991);

            let iwec_recs = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(iwec_recs.len(), 1,
                "iwec must skip the too-few row and yield only the valid row");
            prop_assert_eq!(iwec_recs[0].year, 1991);
        }

        // ---------------------------------------------------------------------
        // 9. Empty / whitespace-only data lines — must not panic.
        //    Strict `parse()` MUST Err on empty data line.
        //    Lenient v3/amy/iwec MUST skip empty lines silently.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_empty_line_mid_stream_does_not_panic(
            whitespace in "[ \\t]{0,5}",
        ) {
            let headers = make_epw_header();
            let rows = vec![
                make_epw_data_row(
                    1991, 1, 1, 1, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ),
                whitespace.clone(),
                make_epw_data_row(
                    1991, 1, 1, 2, 0,
                    20.0, 15.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
                ),
            ];

            let epw = join_epw(&headers, &rows, "\n");

            // Strict parse(): empty/whitespace data line → ParseError.
            // We must not panic regardless.
            let parse_result = std::panic::catch_unwind(|| {
                EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
            });
            prop_assert!(parse_result.is_ok(),
                "`parse()` panicked on empty data line");

            // Lenient v3/amy/iwec MUST skip empty lines.
            let v3_recs = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(v3_recs.len(), 2,
                "v3 must skip empty line and yield 2 records");

            let amy_recs = EpwWeatherSource::parse_epw_amy(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(amy_recs.len(), 2,
                "amy must skip empty line and yield 2 records");

            let iwec_recs = EpwWeatherSource::parse_epw_iwec(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert_eq!(iwec_recs.len(), 2,
                "iwec must skip empty line and yield 2 records");
        }

        // ---------------------------------------------------------------------
        // 10. Extreme but valid numeric values (1e6 deg C) — must round-trip
        //     without NaN, no panic.
        // ---------------------------------------------------------------------
        #[test]
        fn proptest_parse_epw_extreme_numeric_values_do_not_produce_nan(
            // Pick a temperature in a wide-but-finite range; the test asserts
            // that whatever value passes through the parser stays finite.
            temp in -100.0_f64..100.0_f64,
        ) {
            let headers = make_epw_header();
            let rows = vec![make_epw_data_row(
                1991, 1, 1, 1, 0,
                temp, temp - 5.0, 50.0, 800.0, 700.0, 100.0, 3.0, 180.0, 300.0,
            )];

            let epw = join_epw(&headers, &rows, "\n");

            let source = EpwWeatherSource::parse(Cursor::new(epw.as_bytes()))
                .expect("parse should succeed for finite temps");
            let h0 = source.get_hourly_data(0).unwrap();
            prop_assert!(h0.dry_bulb_temp.is_finite(),
                "dry_bulb_temp must stay finite, got {}", h0.dry_bulb_temp);
            prop_assert!(h0.ghi.is_finite());
            prop_assert!(h0.dni.is_finite());
            prop_assert!(h0.dhi.is_finite());
            prop_assert!(h0.wind_speed.is_finite());
            prop_assert!(h0.humidity.is_finite());
            prop_assert!(h0.horizontal_infrared.is_finite());

            // Lenient parsers must also stay finite.
            let v3 = EpwWeatherSource::parse_epw_v3(Cursor::new(epw.as_bytes())).unwrap();
            prop_assert!(v3[0].dry_bulb_temp.is_finite(),
                "v3 dry_bulb_temp must stay finite, got {}", v3[0].dry_bulb_temp);
        }
    }

    // ===== Issue #2915 — `validate_epw_path` security gate =====
    //
    // All cases use `validate_epw_path_in_dir` with a `tempfile` allow-list
    // so they never touch the process-wide `FLUXION_EPW_DIR` env var (and
    // therefore cannot race with each other under parallel `cargo test`).

    /// A real `.epw` file inside the allow-list directory validates and
    /// returns a canonicalised path.
    #[test]
    fn validate_epw_path_accepts_valid_epw() {
        let dir = tempfile::tempdir().unwrap();
        let epw = dir.path().join("USA_CO_Denver.epw");
        std::fs::write(&epw, b"LOCATION,Denver,CO\n").unwrap();
        let rel = epw.to_string_lossy().into_owned();
        let validated = validate_epw_path_in_dir(&rel, dir.path());
        assert!(validated.is_ok(), "valid path rejected: {validated:?}");
        let canon = validated.unwrap();
        assert!(canon.is_absolute());
        assert_eq!(canon.extension().and_then(|e| e.to_str()), Some("epw"));
    }

    /// A non-existent file is rejected with a generic "not found" message
    /// that does NOT echo the supplied path.
    #[test]
    fn validate_epw_path_rejects_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("ghost.epw");
        let err = validate_epw_path_in_dir(&missing.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(err, "epw file not found");
        // The raw user-supplied path must not be reflected back.
        assert!(!err.contains("ghost"));
    }

    /// A file with the wrong extension is rejected even if it lives inside
    /// the allow-list directory.
    #[test]
    fn validate_epw_path_rejects_wrong_extension() {
        let dir = tempfile::tempdir().unwrap();
        let txt = dir.path().join("notes.txt");
        std::fs::write(&txt, b"not an epw").unwrap();
        let err = validate_epw_path_in_dir(&txt.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(err, "invalid epw file extension (expected .epw)");
    }

    /// An uppercase `.EPW` extension is accepted (case-insensitive check).
    #[test]
    fn validate_epw_path_accepts_uppercase_extension() {
        let dir = tempfile::tempdir().unwrap();
        let epw = dir.path().join("MODEL.EPW");
        std::fs::write(&epw, b"x").unwrap();
        let res = validate_epw_path_in_dir(&epw.to_string_lossy(), dir.path());
        assert!(res.is_ok(), "uppercase .EPW should be accepted: {res:?}");
    }

    /// Path-traversal to a file OUTSIDE the allow-list directory is rejected.
    /// Uses a sibling temp dir so the `.epw` file genuinely exists but lives
    /// beyond the allow-list boundary.
    #[test]
    fn validate_epw_path_rejects_traversal_outside_allowlist() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        // Real .epw file in the *outside* dir.
        let evil = outside.path().join("evil.epw");
        std::fs::write(&evil, b"pwned").unwrap();
        // Reference it via a traversal path from inside the allowed dir.
        let rel = allowed
            .path()
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("evil.epw")
            .to_string_lossy()
            .into_owned();
        let err = validate_epw_path_in_dir(&rel, allowed.path()).unwrap_err();
        assert_eq!(err, "epw path outside allowed directory");
    }

    /// `/etc/passwd` (no `.epw` extension) is rejected — the classic
    /// traversal probe from the issue. Uses an absolute path so it is
    /// deterministic regardless of the test working directory.
    #[test]
    fn validate_epw_path_rejects_etc_passwd() {
        let dir = tempfile::tempdir().unwrap();
        // /etc/passwd exists on Linux; guard other platforms.
        if !std::path::Path::new("/etc/passwd").is_file() {
            eprintln!("skipping: /etc/passwd not present on this platform");
            return;
        }
        let err = validate_epw_path_in_dir("/etc/passwd", dir.path()).unwrap_err();
        // Fails at the extension check (no .epw). Either way, it must fail
        // and must not echo the path.
        assert!(
            err == "invalid epw file extension (expected .epw)"
                || err == "epw path outside allowed directory"
        );
        assert!(!err.contains("passwd"));
    }

    /// A symbolic link is rejected even when it points at a real `.epw`
    /// file inside the allow-list. The check happens BEFORE canonicalize
    /// so a symlink that escapes the allow-list cannot bypass the gate
    /// by pointing at an in-allowlist target.
    #[cfg(unix)]
    #[test]
    fn validate_epw_path_rejects_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.epw");
        std::fs::write(&real, b"x").unwrap();
        let link = dir.path().join("link.epw");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let err = validate_epw_path_in_dir(&link.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(err, "epw file path may not be a symbolic link");
    }

    /// A file larger than [`MAX_EPW_SIZE_BYTES`] (50 MiB) is rejected.
    /// Uses `File::set_len` to create a sparse file whose reported length
    /// exceeds the limit without actually allocating 50 MiB on disk
    /// (`metadata().len()` reports the logical size).
    #[test]
    fn validate_epw_path_rejects_oversized_file() {
        // The limit must be exactly 50 MiB (Issue #2915 acceptance).
        assert_eq!(MAX_EPW_SIZE_BYTES, 50 * 1024 * 1024);

        let dir = tempfile::tempdir().unwrap();
        let epw = dir.path().join("huge.epw");
        let f = std::fs::File::create(&epw).unwrap();
        f.set_len(MAX_EPW_SIZE_BYTES + 1).unwrap();
        drop(f);
        let err = validate_epw_path_in_dir(&epw.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(
            err,
            format!("epw file exceeds size limit ({} bytes)", MAX_EPW_SIZE_BYTES)
        );
        // Generic message: must not contain the user-supplied path.
        assert!(!err.contains("huge"));
    }

    #[test]
    fn test_epw_location_debug() {
        let loc = EpwLocation { city_state: "Denver, CO".into(), utc_offset_hours: Some(-7.0) };
        let d = format!("{:?}", loc);
        assert!(d.contains("Denver"));
        assert!(d.contains("-7"));
    }

    #[test]
    fn test_epw_location_clone() {
        let loc = EpwLocation { city_state: "Miami, FL".into(), utc_offset_hours: None };
        let cloned = loc.clone();
        assert_eq!(cloned.city_state, "Miami, FL");
        assert_eq!(cloned.utc_offset_hours, None);
        assert_eq!(cloned, loc);
    }

    #[test]
    fn test_epw_location_city_state() {
        let loc = EpwLocation { city_state: "Minneapolis, MN".into(), utc_offset_hours: Some(-6.0) };
        assert_eq!(loc.city_state(), "Minneapolis, MN");
    }

    #[test]
    fn test_epw_location_partialeq() {
        let l1 = EpwLocation { city_state: "A".into(), utc_offset_hours: Some(1.0) };
        let l2 = EpwLocation { city_state: "A".into(), utc_offset_hours: Some(1.0) };
        let l3 = EpwLocation { city_state: "B".into(), utc_offset_hours: Some(1.0) };
        assert_eq!(l1, l2);
        assert_ne!(l1, l3);
        let l4 = EpwLocation { city_state: "A".into(), utc_offset_hours: None };
        assert_ne!(l1, l4);
    }

    #[test]
    fn test_epw_version_eq() {
        assert_eq!(EpwVersion::V2, EpwVersion::V2);
        assert_ne!(EpwVersion::V2, EpwVersion::V3);
        assert_eq!(EpwVersion::AMY, EpwVersion::AMY);
        assert_eq!(EpwVersion::IWEC, EpwVersion::IWEC);
    }

    #[test]
    fn test_epw_weather_source_accessors() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();
        assert_eq!(source.location(), Some("Denver, CO".into()));
        assert_eq!(source.record_count(), 3);
        assert_eq!(source.solar_hours(), 1);
        assert_eq!(source.max_temperature(), 32.0);
        assert_eq!(source.min_temperature(), -2.0);
    }

    #[test]
    fn test_epw_weather_source_debug() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();
        let d = format!("{:?}", source);
        assert!(d.contains("Denver") || d.contains("EpwWeatherSource"));
    }

    #[test]
    fn test_epw_weather_source_get_hourly_data_out_of_range() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();
        assert!(source.get_hourly_data(3).is_err());
        assert!(source.get_hourly_data(8760).is_err());
        assert!(source.get_hourly_data(10000).is_err());
    }

    #[test]
    fn test_epw_weather_source_average_temperature() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();
        let avg = source.average_temperature();
        assert!((avg - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_sentinel_exact_boundary() {
        assert!(is_epw_sentinel(9999.0, EPW_SOLAR_SENTINEL));
        assert!(!is_epw_sentinel(9998.0, EPW_SOLAR_SENTINEL));
        assert!(is_epw_sentinel(9999.9, EPW_SOLAR_SENTINEL));
        assert!(is_epw_sentinel(10000.0, EPW_SOLAR_SENTINEL));
        assert!(is_epw_sentinel(f64::MAX, EPW_SOLAR_SENTINEL));
        assert!(!is_epw_sentinel(-f64::MAX, EPW_SOLAR_SENTINEL));
    }

    #[test]
    fn test_parse_field_sentinel_exact() {
        assert_eq!(parse_field_coercing_sentinel("", EPW_SOLAR_SENTINEL, -999.0), -999.0);
        assert_eq!(parse_field_coercing_sentinel("   ", EPW_SOLAR_SENTINEL, -1.0), -1.0);
    }

    #[test]
    fn test_epw_weather_source_iterator_bounds() {
        let epw_content = create_test_epw();
        let cursor = Cursor::new(epw_content);
        let source = EpwWeatherSource::parse(cursor).unwrap();
        let all: Vec<_> = source.iter_hours().collect();
        assert_eq!(all.len(), 3);
        assert!(all[0].is_ok());
        assert!(all[1].is_ok());
        assert!(all[2].is_ok());
        assert!(source.get_hourly_data(3).is_err());
    }

    #[test]
    fn test_statistics_zero_records() {
        let source: EpwWeatherSource = EpwWeatherSource {
            location: Some(EpwLocation { city_state: "Empty".into(), utc_offset_hours: None }),
            hourly_data: vec![],
        };
        assert_eq!(source.record_count(), 0);
        assert_eq!(source.solar_hours(), 0);
        assert!(source.max_temperature().is_infinite() && source.max_temperature().is_sign_negative());
        assert!(source.min_temperature().is_infinite() && source.min_temperature().is_sign_positive());
        assert_eq!(source.average_temperature(), 0.0);
    }

    #[test]
    fn test_statistics_all_identical_temps() {
        let w = HourlyWeatherData::new(25.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0);
        let source: EpwWeatherSource = EpwWeatherSource {
            location: None,
            hourly_data: vec![w; 5],
        };
        assert_eq!(source.max_temperature(), 25.0);
        assert_eq!(source.min_temperature(), 25.0);
        assert_eq!(source.average_temperature(), 25.0);
        assert_eq!(source.solar_hours(), 0);
    }

    #[test]
    fn test_statistics_negative_temps() {
        let w1 = HourlyWeatherData::new(-10.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0);
        let w2 = HourlyWeatherData::new(-20.0, 0.0, 0.0, 0.0, 0.0, 50.0, 1);
        let source: EpwWeatherSource = EpwWeatherSource {
            location: None,
            hourly_data: vec![w1, w2],
        };
        assert_eq!(source.max_temperature(), -10.0);
        assert_eq!(source.min_temperature(), -20.0);
        assert!((source.average_temperature() - (-15.0)).abs() < 0.01);
    }

    #[test]
    fn test_parse_location_valid_full() {
        let line = "LOCATION,Chicago,IL,USA,TMY3,725300,41.97,-87.92,-6.0,181.0,1991-2005";
        let result = EpwWeatherSource::parse_location(line).unwrap().unwrap();
        assert_eq!(result.city_state, "Chicago, IL");
        assert_eq!(result.utc_offset_hours, Some(-6.0));
    }

    #[test]
    fn test_parse_location_negative_utc() {
        let line = "LOCATION,Test,XX,TEST,TEST,000000,0.0,0.0,-12.5,0";
        let result = EpwWeatherSource::parse_location(line).unwrap().unwrap();
        assert_eq!(result.utc_offset_hours, Some(-12.5));
    }

    #[test]
    fn test_parse_location_positive_utc() {
        let line = "LOCATION,Test,XX,TEST,TEST,000000,0.0,0.0,5.5,0";
        let result = EpwWeatherSource::parse_location(line).unwrap().unwrap();
        assert_eq!(result.utc_offset_hours, Some(5.5));
    }

    #[test]
    fn test_parse_location_missing_city() {
        let line = "LOCATION,,NY,USA,TMY3,724690,39.83,-104.65,-7.0,1655.0";
        let result = EpwWeatherSource::parse_location(line).unwrap().unwrap();
        assert_eq!(result.city_state, ", NY");
    }

    #[test]
    fn test_is_epw_header_line_false_positives() {
        assert!(!super::is_epw_header_line("1991,1,1,1,0"));
        assert!(!super::is_epw_header_line("9999,1,1,1,0,0"));
        assert!(!super::is_epw_header_line("NOT A HEADER LINE HERE"));
    }

    #[test]
    fn test_parse_optional_field_whitespace() {
        assert_eq!(super::parse_optional_field("  42.5  ", 0.0), 42.5);
        assert_eq!(super::parse_optional_field("  ", 99.0), 99.0);
    }

    #[test]
    fn test_parse_data_line_optional_snow_fields() {
        let line = "1991,1,1,1,0,0,0.0,-5.0,50,101325,0,0,300,0,0,0,0,0,0,0,0,3.5,180,10.0,20000.0,5000.0,5.0,50.0,0,0,0,0,0,0,0,0,0,0";
        let result = EpwWeatherSource::parse_data_line(line, 0).unwrap();
        assert_eq!(result.snow_depth, Some(5.0));
        assert_eq!(result.snow_cover, Some(50.0));
    }

    #[test]
    fn test_hourly_record_equality() {
        let r1 = HourlyRecord {
            year: 2000, month: 6, day: 15, hour: 12, minute: 0,
            dry_bulb_temp: 30.0, humidity: 40.0, dni: 800.0, dhi: 100.0,
            ghi: 900.0, wind_speed: 3.0, horizontal_infrared: 300.0,
            ground_temperature: None, horizontal_illuminance: None,
            diffuse_illuminance: None, snow_depth: None, snow_cover: None,
            present_weather: None, present_weather_code: None,
        };
        let mut r2 = r1.clone();
        assert_eq!(r1.year, r2.year);
        r2.year = 2001;
        assert_ne!(r1.year, r2.year);
    }

    #[test]
    fn test_subhourly_record_equality() {
        let r1 = SubHourlyRecord {
            year: 2000, month: 6, day: 15, hour: 12, minute: 30,
            dry_bulb_temp: 30.0, humidity: 40.0, dni: 800.0, dhi: 100.0,
            ghi: 900.0, wind_speed: 3.0, horizontal_infrared: 300.0,
            ground_temperature: None, horizontal_illuminance: None,
            diffuse_illuminance: None, snow_depth: None, snow_cover: None,
            present_weather: None, present_weather_code: None,
        };
        let mut r2 = r1.clone();
        assert_eq!(r1.minute, r2.minute);
        r2.minute = 45;
        assert_ne!(r1.minute, r2.minute);
    }

    #[test]
    fn test_hourly_record_debug() {
        let r = HourlyRecord {
            year: 2000, month: 1, day: 1, hour: 1, minute: 0,
            dry_bulb_temp: 20.0, humidity: 50.0, dni: 0.0, dhi: 0.0,
            ghi: 0.0, wind_speed: 2.0, horizontal_infrared: 300.0,
            ground_temperature: Some(10.0), horizontal_illuminance: None,
            diffuse_illuminance: None, snow_depth: None, snow_cover: None,
            present_weather: None, present_weather_code: None,
        };
        let d = format!("{:?}", r);
        assert!(d.contains("2000"));
        assert!(d.contains("20"));
    }

    #[test]
    fn test_subhourly_record_debug() {
        let r = SubHourlyRecord {
            year: 2000, month: 7, day: 15, hour: 14, minute: 15,
            dry_bulb_temp: 35.0, humidity: 30.0, dni: 900.0, dhi: 120.0,
            ghi: 1020.0, wind_speed: 2.5, horizontal_infrared: 350.0,
            ground_temperature: None, horizontal_illuminance: None,
            diffuse_illuminance: None, snow_depth: None, snow_cover: None,
            present_weather: None, present_weather_code: None,
        };
        let d = format!("{:?}", r);
        assert!(d.contains("35"));
        assert!(d.contains("15")); // minute
    }

    #[test]
    fn test_validate_epw_path_rejects_directory() {
        let dir = tempfile::tempdir().unwrap();
        let err = validate_epw_path_in_dir(&dir.path().to_string_lossy(), dir.path()).unwrap_err();
        assert!(err.contains("not found") || err.contains("extension"));
    }

    #[test]
    fn test_validate_epw_path_accepts_relative_inside_dir() {
        let dir = tempfile::tempdir().unwrap();
        let epw = dir.path().join("rel.epw");
        std::fs::write(&epw, b"x").unwrap();
        let res = validate_epw_path_in_dir(&epw.to_string_lossy(), dir.path());
        assert!(res.is_ok(), "absolute path should work: {res:?}");
    }
}
