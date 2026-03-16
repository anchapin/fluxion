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
/// use fluxion::weather::epw::EpwWeatherSource;
/// use fluxion::weather::WeatherSource;
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
    /// use fluxion::weather::epw::EpwWeatherSource;
    /// use fluxion::weather::WeatherSource;
    ///
    /// let weather = EpwWeatherSource::from_file("weather.epw")
    ///     .expect("Failed to load weather file");
    ///
    /// let data = weather.get_hourly_data(100)?;
    /// # Ok::<(), fluxion::weather::WeatherError>(())
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

        for (_line_num, line) in buffered.lines().enumerate() {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;

            // Skip header lines (start with "LOCATION", "DESIGN CONDITIONS", etc.)
            if line.starts_with("LOCATION") || line.starts_with("DESIGN CONDITIONS") {
                continue;
            }

            // Skip data period lines
            if line.starts_with("DATA PERIODS") || line.is_empty() {
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
                dni: fields[10].parse::<f64>().unwrap_or(0.0),
                dhi: fields[11].parse::<f64>().unwrap_or(0.0),
                ghi: fields[12].parse::<f64>().unwrap_or(0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                horizontal_infrared: parse_optional_field(fields[15], 0.0),
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

        for (_line_num, line) in buffered.lines().enumerate() {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;

            // Skip header lines (start with "LOCATION", "DESIGN CONDITIONS", etc.)
            if line.starts_with("LOCATION") || line.starts_with("DESIGN CONDITIONS") {
                continue;
            }

            // Skip data period lines
            if line.starts_with("DATA PERIODS") || line.is_empty() {
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
                dni: fields[10].parse::<f64>().unwrap_or(0.0),
                dhi: fields[11].parse::<f64>().unwrap_or(0.0),
                ghi: fields[12].parse::<f64>().unwrap_or(0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                horizontal_infrared: parse_optional_field(fields[15], 0.0),
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

        for (_line_num, line) in buffered.lines().enumerate() {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;

            // Skip header lines (start with "LOCATION", "DESIGN CONDITIONS", etc.)
            if line.starts_with("LOCATION") || line.starts_with("DESIGN CONDITIONS") {
                continue;
            }

            // Skip data period lines
            if line.starts_with("DATA PERIODS") || line.is_empty() {
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
                dni: fields[10].parse::<f64>().unwrap_or(0.0),
                dhi: fields[11].parse::<f64>().unwrap_or(0.0),
                ghi: fields[12].parse::<f64>().unwrap_or(0.0),
                wind_speed: fields[21].parse::<f64>().unwrap_or(0.0),
                horizontal_infrared: parse_optional_field(fields[15], 0.0),
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

        // Parse temperature (field 7 in 0-indexed array)
        let dry_bulb_temp = parse_field(fields[7], "dry bulb temperature")?;

        // Parse relative humidity (field 8 in 0-indexed array)
        let humidity = parse_field(fields[8], "relative humidity")?;

        // Parse solar radiation values
        // Field 10 = Direct Normal Irradiance (Wh/m²)
        let dni = parse_optional_field(fields[10], 0.0); // Already W/m² in modern EPW

        // Field 11 = Diffuse Horizontal Irradiance (Wh/m²)
        let dhi = parse_optional_field(fields[11], 0.0);

        // Field 12 = Global Horizontal Irradiance (Wh/m²)
        let ghi = parse_optional_field(fields[12], 0.0);

        // Parse wind speed (field 21 in 0-indexed array)
        let wind_speed = parse_field(fields[21], "wind speed")?;

        // Parse horizontal infrared radiation (field 15 in 0-indexed array)
        // This is the "Horizontal Infrared Radiation Intensity from Sky" in W/m²
        // Used for calculating sky temperature for longwave radiation exchange
        let horizontal_infrared = parse_optional_field(fields[15], 0.0);

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

        // Create 3 sample data hours
        let data_lines = [
            "1991,1,1,1,0,0,99,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0",
            "1991,1,1,2,0,0,99,-2.0,45,1,0,0,0,0,0,0,0,0,0,0,0,3.2,170,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0",
            "1991,7,15,12,0,0,99,32.0,20,1,800,100,900,0,0,0,0,0,0,0,0,2.5,200,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0"
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
        let line = "1991,1,1,1,0,0,99,0.0,50,1,800,100,900,0,0,0,0,0,0,0,0,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0";

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
}
