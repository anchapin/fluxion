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
use std::path::Path;

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
#[derive(Debug, Clone)]
pub struct EpwWeatherSource {
    /// Location extracted from EPW header (e.g., "Denver, CO")
    location: Option<String>,
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
    /// # Arguments
    ///
    /// * `line` - The location header line
    ///
    /// # Returns
    ///
    /// * `Some(String)` - Location string in "City, State" format
    /// * `None` - If location cannot be parsed
    fn parse_location(line: &str) -> Result<Option<String>, WeatherError> {
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 3 {
            return Ok(None);
        }

        let city = parts[1].trim();
        let state = parts[2].trim();

        if city.is_empty() && state.is_empty() {
            return Ok(None);
        }

        Ok(Some(format!("{}, {}", city, state)))
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
                ghi: fields[13].parse::<f64>().unwrap_or(0.0),
                dni: fields[14].parse::<f64>().unwrap_or(0.0),
                dhi: fields[15].parse::<f64>().unwrap_or(0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                // Issue #829 fix: HIR is column 13 (fields[12]); previously read DHI (fields[15]).
                horizontal_infrared: parse_optional_field(fields[12], 0.0),
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
                ghi: fields[13].parse::<f64>().unwrap_or(0.0),
                dni: fields[14].parse::<f64>().unwrap_or(0.0),
                dhi: fields[15].parse::<f64>().unwrap_or(0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                // Issue #829 fix: HIR is column 13 (fields[12]); previously read DHI (fields[15]).
                horizontal_infrared: parse_optional_field(fields[12], 0.0),
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
                ghi: fields[13].parse::<f64>().unwrap_or(0.0),
                dni: fields[14].parse::<f64>().unwrap_or(0.0),
                dhi: fields[15].parse::<f64>().unwrap_or(0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                // Issue #829 fix: HIR is column 13 (fields[12]); previously read DHI (fields[15]).
                horizontal_infrared: parse_optional_field(fields[12], 0.0),
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
        let ghi = parse_optional_field(fields[13], 0.0);
        let dni = parse_optional_field(fields[14], 0.0);
        let dhi = parse_optional_field(fields[15], 0.0);
        let wind_speed = parse_field(fields[21], "wind speed")?;
        let horizontal_infrared = parse_optional_field(fields[12], 0.0);

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
        self.location.clone()
    }

    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
        if hour >= self.hourly_data.len() {
            return Err(WeatherError::InvalidHour(hour));
        }

        Ok(self.hourly_data[hour].clone())
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
        assert_eq!(result, Some("Denver, CO".to_string()));
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
        assert_eq!(result, Some(", CA".to_string()));
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
            location: Some("Test".to_string()),
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
            location: Some("Test".to_string()),
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
            location: Some("Test".to_string()),
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
            location: Some("Test".to_string()),
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
}
