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

use crate::weather::{HourlyWeatherData, WeatherError, WeatherSource};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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

        // Parse hourly data lines (variable number, typically 8760)
        let mut hourly_data = Vec::new();

        while let Some(line_result) = lines.next() {
            let line = line_result.map_err(|e| {
                WeatherError::ParseError(format!("Failed to read data line: {}", e))
            })?;

            // Skip comment lines (start with '!')
            if line.trim().starts_with('!') {
                continue;
            }

            let line_idx = hourly_data.len();
            let weather_data = Self::parse_data_line(&line, line_idx)?;
            hourly_data.push(weather_data);
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

        // Parse temperature (column 7, 0-indexed = field 6)
        let dry_bulb_temp = parse_field(fields[6], "dry bulb temperature")?;

        // Parse relative humidity (column 9, 0-indexed = field 8)
        let humidity = parse_field(fields[8], "relative humidity")?;

        // Parse solar radiation values (convert from Wh/m² to W/m²)
        // Column 11 = Direct Normal Irradiance (Wh/m²)
        let dni = parse_optional_field(fields[10], 0.0); // Already W/m² in modern EPW

        // Column 12 = Diffuse Horizontal Irradiance (Wh/m²)
        let dhi = parse_optional_field(fields[11], 0.0);

        // Column 13 = Global Horizontal Irradiance (Wh/m²)
        let ghi = parse_optional_field(fields[12], 0.0);

        // Parse wind speed (column 22, 0-indexed = field 21)
        let wind_speed = parse_field(fields[21], "wind speed")?;

        Ok(HourlyWeatherData {
            dry_bulb_temp,
            dni,
            dhi,
            ghi,
            wind_speed,
            humidity,
            hour_of_year,
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
        let data_lines = vec![
            "1991,1,1,1,0,0,0.0,0.0,50,1,0,0,0,0,0,0,0,0,0,0,0,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0",
            "1991,1,1,2,0,0,-2.0,0.0,45,1,0,0,0,0,0,0,0,0,0,0,0,3.2,170,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0",
            "1991,7,15,12,0,0,32.0,0.0,20,1,800,100,900,0,0,0,0,0,0,0,0,2.5,200,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0",
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
        let line = "1991,1,1,1,0,0,0.0,0.0,50,1,800,100,900,0,0,0,0,0,0,0,0,3.5,180,9999,9999,0,0,0,0,0,0,0,0,0,0,0,0";

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
            if result.is_ok() {
                count += 1;
            } else {
                // Stop counting once we hit invalid hours
                break;
            }
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
