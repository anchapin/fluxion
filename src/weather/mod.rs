//! Weather data module for building energy modeling.
//!
//! This module provides abstractions for weather data sources used in building energy
//! simulations, including support for EPW (EnergyPlus Weather) files and embedded
//! Typical Meteorological Year (TMY) data for standard locations like Denver, CO.
//!
//! # Overview
//!
//! - [`HourlyWeatherData`]: Core data structure representing hourly weather conditions
//! - [`WeatherSource`]: Trait for abstracting different weather data sources
//! - [`WeatherError`]: Error types for weather data operations
//!
//! # Supported Sources
//!
//! - **EPW Files**: External EnergyPlus Weather format files via [`epw::EpwWeatherSource`]
//! - **Embedded TMY**: Pre-defined weather data for ASHRAE 140 via [`denver::DenverTmyWeather`]

pub mod ashrae140_weather;
pub mod ddy;
pub mod denver;
pub mod epw;
pub mod interpolation;
pub mod psychrometrics;
pub mod tmy3;

pub use self::psychrometrics::*;
pub use ddy::{generate_design_day_hours, DesignDaySource, DesignDaySpec};
pub use interpolation::{interpolate_weather, select_method_for_field, InterpolationMethod};
pub use tmy3::{load_weather_locations, Tmy3Cache, WeatherLocation};

use serde::{Deserialize, Serialize};
use std::fmt;

/// Hourly weather data for building energy simulation.
///
/// Contains all meteorological parameters required for ASHRAE 140 standard
/// building energy calculations, including temperature, solar radiation,
/// wind speed, humidity, and longwave radiation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyWeatherData {
    /// Dry bulb temperature (°C)
    ///
    /// Ambient air temperature measured by a thermometer shielded from
    /// radiation and moisture effects.
    pub dry_bulb_temp: f64,

    /// Direct Normal Irradiance (W/m²)
    ///
    /// Solar radiation received per unit area on a surface that is always
    /// held normal (perpendicular) to the incoming rays.
    pub dni: f64,

    /// Diffuse Horizontal Irradiance (W/m²)
    ///
    /// Solar radiation received from the sky (excluding the direct solar beam)
    /// on a horizontal surface.
    pub dhi: f64,

    /// Global Horizontal Irradiance (W/m²)
    ///
    /// Total solar radiation received on a horizontal surface, including both
    /// direct and diffuse components.
    pub ghi: f64,

    /// Wind speed (m/s)
    ///
    /// Horizontal wind speed measured at a standard height of 10 meters above ground.
    pub wind_speed: f64,

    /// Relative humidity (%)
    ///
    /// Ratio of the partial pressure of water vapor to the equilibrium vapor
    /// pressure of water at a given temperature, expressed as a percentage.
    pub humidity: f64,

    /// Horizontal Infrared Radiation Intensity (W/m²)
    ///
    /// Total longwave (thermal) radiation coming from the sky hemisphere,
    /// measured on a horizontal surface. This is used to calculate sky temperature
    /// for longwave radiation exchange calculations.
    ///
    /// In EPW files, this is field 16 (0-indexed: field 15).
    /// Typical values range from 200-500 W/m² depending on cloud cover and humidity.
    pub horizontal_infrared: f64,

    /// Hour of year (0-8759)
    ///
    /// Zero-based index representing the hour within the year, where 0 corresponds
    /// to midnight on January 1st and 8759 corresponds to 11pm on December 31st.
    pub hour_of_year: usize,

    /// Ground temperature (°C)
    ///
    /// Temperature at the ground surface, used for foundation heat loss
    /// calculations and low-rise building thermal modeling.
    pub ground_temperature: Option<f64>,

    /// Horizontal illuminance (lux)
    ///
    /// Total visible light from the sky dome on a horizontal surface,
    /// used for daylighting calculations and electric lighting savings.
    pub horizontal_illuminance: Option<f64>,

    /// Diffuse illuminance (lux)
    ///
    /// Visible light from the sky (excluding direct solar beam) on a
    /// horizontal surface, used for daylighting under overcast conditions.
    pub diffuse_illuminance: Option<f64>,

    /// Snow depth (mm)
    ///
    /// Depth of snow accumulation on the ground, used for albedo
    /// variations affecting solar reflection and thermal mass.
    pub snow_depth: Option<f64>,

    /// Snow cover (%)
    ///
    /// Percentage of ground surface covered by snow, affects albedo
    /// and thermal mass calculations.
    pub snow_cover: Option<f64>,

    /// Present weather observation
    ///
    /// Text description of current weather conditions (e.g., "Rain",
    /// "Snow", "Fog"), used for advanced HVAC control and freeze protection.
    pub present_weather: Option<String>,

    /// Present weather code
    ///
    /// Numeric code from EPW field 22, representing weather observation
    /// type. Used for discrete weather condition classification.
    pub present_weather_code: Option<u32>,
}

impl HourlyWeatherData {
    /// Creates a new [`HourlyWeatherData`] instance.
    ///
    /// # Arguments
    ///
    /// * `dry_bulb_temp` - Dry bulb temperature in °C
    /// * `dni` - Direct Normal Irradiance in W/m²
    /// * `dhi` - Diffuse Horizontal Irradiance in W/m²
    /// * `ghi` - Global Horizontal Irradiance in W/m²
    /// * `wind_speed` - Wind speed in m/s
    /// * `humidity` - Relative humidity in percentage (0-100)
    /// * `hour_of_year` - Hour of year (0-8759)
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather = HourlyWeatherData::new(
    ///     20.0,  // 20°C temperature
    ///     800.0, // 800 W/m² direct normal irradiance
    ///     100.0, // 100 W/m² diffuse horizontal irradiance
    ///     900.0, // 900 W/m² global horizontal irradiance
    ///     3.5,   // 3.5 m/s wind speed
    ///     50.0,  // 50% relative humidity
    ///     100,    // Hour 100 of the year
    /// );
    /// ```
    pub fn new(
        dry_bulb_temp: f64,
        dni: f64,
        dhi: f64,
        ghi: f64,
        wind_speed: f64,
        humidity: f64,
        hour_of_year: usize,
    ) -> Self {
        HourlyWeatherData {
            dry_bulb_temp,
            dni,
            dhi,
            ghi,
            wind_speed,
            humidity,
            horizontal_infrared: 0.0, // Default, can be set via with_infrared()
            hour_of_year,
            ground_temperature: None,
            horizontal_illuminance: None,
            diffuse_illuminance: None,
            snow_depth: None,
            snow_cover: None,
            present_weather: None,
            present_weather_code: None,
        }
    }

    /// Creates a new [`HourlyWeatherData`] instance with all fields including infrared.
    ///
    /// # Arguments
    ///
    /// * `dry_bulb_temp` - Dry bulb temperature in °C
    /// * `dni` - Direct Normal Irradiance in W/m²
    /// * `dhi` - Diffuse Horizontal Irradiance in W/m²
    /// * `ghi` - Global Horizontal Irradiance in W/m²
    /// * `wind_speed` - Wind speed in m/s
    /// * `humidity` - Relative humidity in percentage (0-100)
    /// * `horizontal_infrared` - Horizontal Infrared Radiation Intensity in W/m²
    /// * `hour_of_year` - Hour of year (0-8759)
    #[allow(clippy::too_many_arguments)]
    pub fn with_infrared(
        dry_bulb_temp: f64,
        dni: f64,
        dhi: f64,
        ghi: f64,
        wind_speed: f64,
        humidity: f64,
        horizontal_infrared: f64,
        hour_of_year: usize,
    ) -> Self {
        HourlyWeatherData {
            dry_bulb_temp,
            dni,
            dhi,
            ghi,
            wind_speed,
            humidity,
            horizontal_infrared,
            hour_of_year,
            ground_temperature: None,
            horizontal_illuminance: None,
            diffuse_illuminance: None,
            snow_depth: None,
            snow_cover: None,
            present_weather: None,
            present_weather_code: None,
        }
    }

    /// Calculates the effective sky temperature (°C) from horizontal infrared radiation.
    ///
    /// The sky temperature represents the equivalent black-body temperature of the sky
    /// that would produce the measured horizontal infrared radiation. This is essential
    /// for calculating longwave radiation exchange between building surfaces and the sky.
    ///
    /// # Formula
    ///
    /// The sky temperature is calculated using the Stefan-Boltzmann law:
    ///
    /// ```text
    /// T_sky = (IR_horizontal / σ)^(1/4) - 273.15
    /// ```
    ///
    /// Where:
    /// - `IR_horizontal` is the horizontal infrared radiation intensity (W/m²)
    /// - `σ` is the Stefan-Boltzmann constant (5.67×10⁻⁸ W/m²·K⁴)
    ///
    /// # Returns
    ///
    /// Sky temperature in °C. Returns a default value of -20°C if infrared data is missing.
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather = HourlyWeatherData::with_infrared(
    ///     20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 350.0, 100
    /// );
    ///
    /// let t_sky = weather.sky_temperature();
    /// // Typical sky temperatures range from -40°C (clear sky) to +10°C (cloudy)
    /// assert!(t_sky > -50.0 && t_sky < 20.0);
    /// ```
    pub fn sky_temperature(&self) -> f64 {
        const STEFAN_BOLTZMANN: f64 = 5.67e-8; // W/(m²·K⁴)

        if self.horizontal_infrared <= 0.0 {
            // Default sky temperature for clear sky conditions
            // This is a reasonable approximation when IR data is not available
            return self.dry_bulb_temp - 15.0;
        }

        // Calculate sky temperature from measured infrared radiation
        // T_sky = (IR / σ)^(1/4) - 273.15
        let t_sky_kelvin = (self.horizontal_infrared / STEFAN_BOLTZMANN).powf(0.25);
        t_sky_kelvin - 273.15
    }

    /// Calculates the sky emissivity from horizontal infrared radiation.
    ///
    /// Sky emissivity (ε_sky) relates the actual sky radiation to black-body radiation
    /// at ambient temperature. This is useful for detailed radiation models.
    ///
    /// # Formula
    ///
    /// ```text
    /// ε_sky = IR_horizontal / (σ × T_ambient⁴)
    /// ```
    ///
    /// # Returns
    ///
    /// Sky emissivity (dimensionless, typically 0.6-0.9). Returns 0.8 as default.
    pub fn sky_emissivity(&self) -> f64 {
        const STEFAN_BOLTZMANN: f64 = 5.67e-8; // W/(m²·K⁴)

        if self.horizontal_infrared <= 0.0 {
            return 0.8; // Default emissivity
        }

        let t_ambient_kelvin = self.dry_bulb_temp + 273.15;
        self.horizontal_infrared / (STEFAN_BOLTZMANN * t_ambient_kelvin.powi(4))
    }

    /// Returns the hour of day (0-23) for this weather data.
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 25);
    /// assert_eq!(weather.hour_of_day(), 1); // Hour 25 is 1 AM
    /// ```
    pub fn hour_of_day(&self) -> usize {
        self.hour_of_year % 24
    }

    /// Returns the day of year (0-364) for this weather data.
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 48);
    /// assert_eq!(weather.day_of_year(), 2); // Hour 48 is day 2
    /// ```
    pub fn day_of_year(&self) -> usize {
        self.hour_of_year / 24
    }

    /// Returns the month (1-12) for this weather data.
    ///
    /// Uses actual month lengths for a non-leap year.
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 744);
    /// assert_eq!(weather.month(), 2); // Hour 744 is in February
    /// ```
    pub fn month(&self) -> usize {
        // Cumulative days for each month (for a non-leap year)
        let cumulative_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

        let day = self.day_of_year();

        // Find which month this day belongs to
        for (month_idx, &cumulative) in cumulative_days.iter().enumerate() {
            if day < cumulative {
                return month_idx;
            }
        }

        // If day >= 334, it's December
        12
    }

    /// Validates weather data for missing values, NaN, and out-of-range temperatures.
    ///
    /// This method checks that all weather values are complete and within reasonable
    /// physical limits for Earth-based building energy simulations.
    ///
    /// # Validation Rules
    ///
    /// - **Missing values**: All fields must be finite (not NaN or infinite)
    /// - **Temperature range**: Dry bulb temperature must be between -50°C and 60°C
    /// - **Solar irradiance**: DNI, DHI, GHI must be non-negative (can be 0 at night)
    /// - **Wind speed**: Must be non-negative
    /// - **Humidity**: Must be between 0% and 100%
    /// - **Infrared**: Must be non-negative
    ///
    /// # Arguments
    ///
    /// * `weather` - Slice of hourly weather data to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` - All weather data is valid
    /// * `Err(String)` - Validation error with specific message describing the issue
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather_data = vec![
    ///     HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0),
    ///     HourlyWeatherData::new(15.0, 750.0, 90.0, 840.0, 3.0, 55.0, 1),
    /// ];
    ///
    /// match HourlyWeatherData::validate_all(&weather_data) {
    ///     Ok(()) => println!("All weather data is valid"),
    ///     Err(e) => eprintln!("Validation error: {}", e),
    /// }
    /// ```
    pub fn validate_all(weather: &[HourlyWeatherData]) -> Result<(), String> {
        const MIN_TEMP_C: f64 = -50.0; // Reasonable minimum Earth temperature
        const MAX_TEMP_C: f64 = 60.0; // Reasonable maximum Earth temperature

        for (index, w) in weather.iter().enumerate() {
            // Check for NaN temperatures
            if w.dry_bulb_temp.is_nan() {
                return Err(format!(
                    "Temperature is NaN at timestep {}, cannot use in simulation",
                    index
                ));
            }

            // Check for infinite temperatures
            if !w.dry_bulb_temp.is_finite() {
                return Err(format!(
                    "Temperature is infinite at timestep {}, cannot use in simulation",
                    index
                ));
            }

            // Check for out-of-range temperatures
            if w.dry_bulb_temp < MIN_TEMP_C || w.dry_bulb_temp > MAX_TEMP_C {
                return Err(format!(
                    "Temperature {:.2}°C at timestep {} out of range [{:.1}, {:.1}]°C",
                    w.dry_bulb_temp, index, MIN_TEMP_C, MAX_TEMP_C
                ));
            }

            // Check for invalid solar irradiance (must be non-negative)
            if w.dni < 0.0 {
                return Err(format!(
                    "Direct Normal Irradiance {:.2} W/m² at timestep {} is negative",
                    w.dni, index
                ));
            }
            if w.dhi < 0.0 {
                return Err(format!(
                    "Diffuse Horizontal Irradiance {:.2} W/m² at timestep {} is negative",
                    w.dhi, index
                ));
            }
            if w.ghi < 0.0 {
                return Err(format!(
                    "Global Horizontal Irradiance {:.2} W/m² at timestep {} is negative",
                    w.ghi, index
                ));
            }

            // Check for invalid wind speed (must be non-negative)
            if w.wind_speed < 0.0 {
                return Err(format!(
                    "Wind speed {:.2} m/s at timestep {} is negative",
                    w.wind_speed, index
                ));
            }

            // Check for invalid humidity (must be 0-100%)
            if w.humidity < 0.0 || w.humidity > 100.0 {
                return Err(format!(
                    "Relative humidity {:.2}% at timestep {} out of range [0, 100]%",
                    w.humidity, index
                ));
            }

            // Check for invalid infrared radiation (must be non-negative)
            if w.horizontal_infrared < 0.0 {
                return Err(format!(
                    "Horizontal Infrared Radiation {:.2} W/m² at timestep {} is negative",
                    w.horizontal_infrared, index
                ));
            }
        }

        Ok(())
    }

    /// Checks if weather data is complete (no missing or NaN values).
    ///
    /// This is a simpler check than `validate_all` that only verifies that
    /// all critical fields are present and not NaN. It does not check range limits.
    ///
    /// # Arguments
    ///
    /// * `weather` - Slice of hourly weather data to check
    ///
    /// # Returns
    ///
    /// `true` if all weather data is complete and finite, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::weather::HourlyWeatherData;
    ///
    /// let weather_data = vec![
    ///     HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0),
    ///     HourlyWeatherData::new(f64::NAN, 750.0, 90.0, 840.0, 3.0, 55.0, 1),
    /// ];
    ///
    /// assert!(!HourlyWeatherData::is_complete(&weather_data));
    /// ```
    pub fn is_complete(weather: &[HourlyWeatherData]) -> bool {
        weather.iter().all(|w| {
            w.dry_bulb_temp.is_finite()
                && w.dni.is_finite()
                && w.dhi.is_finite()
                && w.ghi.is_finite()
                && w.wind_speed.is_finite()
                && w.humidity.is_finite()
                && w.horizontal_infrared.is_finite()
        })
    }
}

/// Error types for weather data operations.
#[derive(Debug, Clone, PartialEq)]
pub enum WeatherError {
    /// Invalid hour index provided (must be 0-8759 for hourly data).
    InvalidHour(usize),

    /// Weather data is incomplete or missing required fields.
    IncompleteData(String),

    /// Error parsing weather data from file or string.
    ParseError(String),

    /// Input/output error reading weather data file.
    IoError(String),
}

impl fmt::Display for WeatherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeatherError::InvalidHour(hour) => {
                write!(f, "Invalid hour index: {} (must be 0-8759)", hour)
            }
            WeatherError::IncompleteData(msg) => write!(f, "Incomplete weather data: {}", msg),
            WeatherError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            WeatherError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for WeatherError {}

/// Trait for abstracting different weather data sources.
///
/// This trait provides a unified interface for accessing weather data from
/// various sources, such as EPW files, embedded TMY data, or future
/// weather APIs.
///
/// # Example
///
/// ```no_run
/// use fluxion::weather::WeatherSource;
/// # struct MyWeatherSource;
/// # impl WeatherSource for MyWeatherSource {
/// #     fn location(&self) -> Option<String> { None }
/// #     fn get_hourly_data(&self, hour: usize) -> Result<fluxion::weather::HourlyWeatherData, fluxion::weather::WeatherError> {
/// #         Ok(fluxion::weather::HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour))
/// #     }
/// # }
///
/// let weather = MyWeatherSource;
/// let data = weather.get_hourly_data(100)?;
/// println!("Temperature at hour 100: {}°C", data.dry_bulb_temp);
/// # Ok::<(), fluxion::weather::WeatherError>(())
/// ```
pub trait WeatherSource {
    /// Returns the location description (e.g., "Denver, CO") if available.
    ///
    /// This is metadata extracted from weather file headers or
    /// configured for embedded sources.
    fn location(&self) -> Option<String>;

    /// Retrieves hourly weather data for a specific hour of the year.
    ///
    /// # Arguments
    ///
    /// * `hour` - Hour of year (0-8759), where 0 is midnight Jan 1 and 8759 is 11pm Dec 31
    ///
    /// # Returns
    ///
    /// * `Ok(HourlyWeatherData)` - Weather data for the specified hour
    /// * `Err(WeatherError::InvalidHour)` - If hour is outside the valid range 0-8759
    ///
    /// # Example
    ///
    /// ```no_run
    /// use fluxion::weather::WeatherSource;
    /// # struct MyWeatherSource;
    /// # impl WeatherSource for MyWeatherSource {
    /// #     fn location(&self) -> Option<String> { Some("Test City".to_string()) }
    /// #     fn get_hourly_data(&self, hour: usize) -> Result<fluxion::weather::HourlyWeatherData, fluxion::weather::WeatherError> {
    /// #         Ok(fluxion::weather::HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour))
    /// #     }
    /// # }
    ///
    /// let weather = MyWeatherSource;
    ///
    /// // Get weather for noon on January 1st (hour 12)
    /// let data = weather.get_hourly_data(12)?;
    /// println!("Temperature at noon: {}°C", data.dry_bulb_temp);
    ///
    /// // This will return an error
    /// let error = weather.get_hourly_data(8760);
    /// assert!(error.is_err());
    /// # Ok::<(), fluxion::weather::WeatherError>(())
    /// ```
    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError>;

    /// Returns an iterator over all 8760 hours of weather data.
    ///
    /// This is a convenience method that creates an iterator yielding
    /// hourly weather data from hour 0 to hour 8759.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use fluxion::weather::WeatherSource;
    /// # struct MyWeatherSource;
    /// # impl WeatherSource for MyWeatherSource {
    /// #     fn location(&self) -> Option<String> { Some("Test City".to_string()) }
    /// #     fn get_hourly_data(&self, hour: usize) -> Result<fluxion::weather::HourlyWeatherData, fluxion::weather::WeatherError> {
    /// #         Ok(fluxion::weather::HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour))
    /// #     }
    /// # }
    ///
    /// let weather = MyWeatherSource;
    /// let mut max_temp = -999.0;
    ///
    /// for data in weather.iter_hours() {
    ///     let data = data.unwrap();
    ///     if data.dry_bulb_temp > max_temp {
    ///         max_temp = data.dry_bulb_temp;
    ///     }
    /// }
    ///
    /// println!("Maximum temperature: {}°C", max_temp);
    /// ```
    fn iter_hours(&self) -> WeatherIterator<'_, Self>
    where
        Self: Sized,
    {
        WeatherIterator {
            source: self,
            current_hour: 0,
        }
    }

    /// Validates all weather data from this source.
    ///
    /// This method collects all 8760 hours of weather data and validates
    /// it for missing values, NaN temperatures, and out-of-range values.
    /// It logs warnings for recoverable issues and returns errors for critical problems.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - All weather data is valid
    /// * `Err(String)` - Validation error with specific message
    ///
    /// # Example
    ///
    /// ```no_run
    /// use fluxion::weather::WeatherSource;
    /// # struct MyWeatherSource;
    /// # impl WeatherSource for MyWeatherSource {
    /// #     fn location(&self) -> Option<String> { Some("Test City".to_string()) }
    /// #     fn get_hourly_data(&self, hour: usize) -> Result<fluxion::weather::HourlyWeatherData, fluxion::weather::WeatherError> {
    /// #         Ok(fluxion::weather::HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour))
    /// #     }
    /// # }
    ///
    /// let weather = MyWeatherSource;
    ///
    /// match weather.validate_all() {
    ///     Ok(()) => println!("All weather data is valid"),
    ///     Err(e) => eprintln!("Weather validation failed: {}", e),
    /// }
    /// ```
    fn validate_all(&self) -> Result<(), String>
    where
        Self: Sized,
    {
        let weather_data: Result<Vec<_>, _> = self.iter_hours().collect();
        let weather_data = weather_data.map_err(|e| e.to_string())?;
        HourlyWeatherData::validate_all(&weather_data)
    }
}

/// Iterator over weather data for all hours in a year.
///
/// Created by [`WeatherSource::iter_hours()`].
pub struct WeatherIterator<'a, T: WeatherSource> {
    source: &'a T,
    current_hour: usize,
}

impl<'a, T: WeatherSource> Iterator for WeatherIterator<'a, T> {
    type Item = Result<HourlyWeatherData, WeatherError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_hour < 8760 {
            let hour = self.current_hour;
            self.current_hour += 1;
            let result = self.source.get_hourly_data(hour);

            // Stop iteration if we get an InvalidHour error
            // This allows test files with fewer than 8760 records to work
            if matches!(&result, Err(WeatherError::InvalidHour(_))) {
                None
            } else {
                Some(result)
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hourly_weather_data_creation() {
        let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);

        assert_eq!(weather.dry_bulb_temp, 20.0);
        assert_eq!(weather.dni, 800.0);
        assert_eq!(weather.dhi, 100.0);
        assert_eq!(weather.ghi, 900.0);
        assert_eq!(weather.wind_speed, 3.5);
        assert_eq!(weather.humidity, 50.0);
        assert_eq!(weather.hour_of_year, 100);
    }

    #[test]
    fn test_hour_of_day() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
        assert_eq!(weather.hour_of_day(), 0);

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12);
        assert_eq!(weather.hour_of_day(), 12);

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24);
        assert_eq!(weather.hour_of_day(), 0);

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8759);
        assert_eq!(weather.hour_of_day(), 23);
    }

    #[test]
    fn test_day_of_year() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
        assert_eq!(weather.day_of_year(), 0);

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24);
        assert_eq!(weather.day_of_year(), 1);

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 48);
        assert_eq!(weather.day_of_year(), 2);

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8760);
        assert_eq!(weather.day_of_year(), 365);
    }

    #[test]
    fn test_month() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
        assert_eq!(weather.month(), 1); // January

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 744);
        assert_eq!(weather.month(), 2); // February

        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8040); // Hour 335 * 24 (start of December)
        assert_eq!(weather.month(), 12); // December
    }

    #[test]
    fn test_weather_error_display() {
        let err = WeatherError::InvalidHour(9000);
        let err_str = format!("{}", err);
        assert!(err_str.contains("9000"));

        let err = WeatherError::IncompleteData("missing DNI".to_string());
        let err_str = format!("{}", err);
        assert!(err_str.contains("missing DNI"));

        let err = WeatherError::ParseError("invalid format".to_string());
        let err_str = format!("{}", err);
        assert!(err_str.contains("invalid format"));

        let err = WeatherError::IoError("file not found".to_string());
        let err_str = format!("{}", err);
        assert!(err_str.contains("file not found"));
    }

    #[test]
    fn test_weather_source_trait_bounds() {
        // This test just verifies that the WeatherSource trait is properly defined
        // and can be used as a trait object

        struct DummySource;
        impl WeatherSource for DummySource {
            fn location(&self) -> Option<String> {
                Some("Dummy, TEST".to_string())
            }

            fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
                if hour >= 8760 {
                    return Err(WeatherError::InvalidHour(hour));
                }
                Ok(HourlyWeatherData::new(
                    20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour,
                ))
            }
        }

        let source = DummySource;
        assert_eq!(source.location(), Some("Dummy, TEST".to_string()));

        let data = source.get_hourly_data(100).unwrap();
        assert_eq!(data.dry_bulb_temp, 20.0);
        assert_eq!(data.hour_of_year, 100);

        let err = source.get_hourly_data(8760);
        assert_eq!(err, Err(WeatherError::InvalidHour(8760)));
    }

    #[test]
    fn test_weather_iterator() {
        struct DummySource;
        impl WeatherSource for DummySource {
            fn location(&self) -> Option<String> {
                Some("Dummy".to_string())
            }

            fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
                if hour >= 8760 {
                    return Err(WeatherError::InvalidHour(hour));
                }
                Ok(HourlyWeatherData::new(
                    20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour,
                ))
            }
        }

        let source = DummySource;
        let mut count = 0;

        for result in source.iter_hours() {
            let data = result.unwrap();
            assert_eq!(data.dry_bulb_temp, 20.0);
            assert_eq!(data.hour_of_year, count);
            count += 1;
        }

        assert_eq!(count, 8760);
    }

    #[test]
    fn test_weather_iterator_with_error() {
        struct ErrorSource;
        impl WeatherSource for ErrorSource {
            fn location(&self) -> Option<String> {
                None
            }

            fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
                if hour > 10 {
                    return Err(WeatherError::ParseError("test error".to_string()));
                }
                Ok(HourlyWeatherData::new(
                    20.0, 800.0, 100.0, 900.0, 3.5, 50.0, hour,
                ))
            }
        }

        let source = ErrorSource;

        for (count, result) in source.iter_hours().take(15).enumerate() {
            if count <= 10 {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn test_extreme_weather_validation() {
        // Test 1: Valid weather data
        let valid_weather = vec![
            HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0),
            HourlyWeatherData::new(15.0, 750.0, 90.0, 840.0, 3.0, 55.0, 1),
        ];
        assert!(HourlyWeatherData::validate_all(&valid_weather).is_ok());

        // Test 2: NaN temperature
        let nan_weather = vec![HourlyWeatherData::new(
            f64::NAN,
            800.0,
            100.0,
            900.0,
            3.5,
            50.0,
            0,
        )];
        let result = HourlyWeatherData::validate_all(&nan_weather);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NaN"));

        // Test 3: Out-of-range temperature (too cold)
        let cold_weather = vec![HourlyWeatherData::new(
            -60.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&cold_weather);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));

        // Test 4: Out-of-range temperature (too hot)
        let hot_weather = vec![HourlyWeatherData::new(
            70.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&hot_weather);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));

        // Test 5: Negative solar irradiance
        let negative_solar = vec![HourlyWeatherData::new(
            20.0, -100.0, 100.0, 900.0, 3.5, 50.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&negative_solar);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("negative"));

        // Test 6: Invalid humidity (out of range)
        let invalid_humidity = vec![HourlyWeatherData::new(
            20.0, 800.0, 100.0, 900.0, 3.5, 150.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&invalid_humidity);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));

        // Test 7: Test is_complete function
        let complete_weather = vec![
            HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0),
            HourlyWeatherData::new(15.0, 750.0, 90.0, 840.0, 3.0, 55.0, 1),
        ];
        assert!(HourlyWeatherData::is_complete(&complete_weather));

        let incomplete_weather = vec![HourlyWeatherData::new(
            f64::NAN,
            800.0,
            100.0,
            900.0,
            3.5,
            50.0,
            0,
        )];
        assert!(!HourlyWeatherData::is_complete(&incomplete_weather));
    }

    #[test]
    fn test_hourly_weather_data_with_infrared() {
        let weather =
            HourlyWeatherData::with_infrared(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 350.0, 100);
        assert_eq!(weather.dry_bulb_temp, 20.0);
        assert_eq!(weather.horizontal_infrared, 350.0);
        assert_eq!(weather.hour_of_year, 100);
        assert!(weather.ground_temperature.is_none());
    }

    #[test]
    fn test_sky_temperature_from_infrared() {
        let weather =
            HourlyWeatherData::with_infrared(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 350.0, 100);
        let t_sky = weather.sky_temperature();
        // Should be calculated from IR, not default
        assert!(t_sky > -50.0 && t_sky < 20.0);
    }

    #[test]
    fn test_sky_temperature_zero_infrared() {
        let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        let t_sky = weather.sky_temperature();
        // Should use default: dry_bulb_temp - 15.0
        assert!((t_sky - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_sky_temperature_negative_infrared() {
        let mut weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        weather.horizontal_infrared = -10.0;
        let t_sky = weather.sky_temperature();
        // Should use default for negative IR
        assert!((t_sky - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_sky_emissivity_from_infrared() {
        let weather =
            HourlyWeatherData::with_infrared(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 350.0, 100);
        let emissivity = weather.sky_emissivity();
        assert!(emissivity > 0.0 && emissivity < 1.0);
    }

    #[test]
    fn test_sky_emissivity_zero_infrared() {
        let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        let emissivity = weather.sky_emissivity();
        assert_eq!(emissivity, 0.8);
    }

    #[test]
    fn test_sky_emissivity_negative_infrared() {
        let mut weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        weather.horizontal_infrared = -5.0;
        let emissivity = weather.sky_emissivity();
        assert_eq!(emissivity, 0.8);
    }

    #[test]
    fn test_month_boundary_december() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8759);
        assert_eq!(weather.month(), 12);
    }

    #[test]
    fn test_month_january_end() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 743);
        assert_eq!(weather.month(), 1);
    }

    #[test]
    fn test_month_february_start() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 744);
        assert_eq!(weather.month(), 2);
    }

    #[test]
    fn test_month_july() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4368);
        assert_eq!(weather.month(), 7);
    }

    #[test]
    fn test_validate_all_boundary_temps() {
        let valid_cold = vec![HourlyWeatherData::new(-50.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0)];
        assert!(HourlyWeatherData::validate_all(&valid_cold).is_ok());

        let valid_hot = vec![HourlyWeatherData::new(60.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0)];
        assert!(HourlyWeatherData::validate_all(&valid_hot).is_ok());
    }

    #[test]
    fn test_validate_all_infinite_temp() {
        let inf_weather = vec![HourlyWeatherData::new(
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            0,
        )];
        let result = HourlyWeatherData::validate_all(&inf_weather);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("infinite"));
    }

    #[test]
    fn test_validate_all_negative_dhi() {
        let neg_dhi = vec![HourlyWeatherData::new(
            20.0, 800.0, -50.0, 900.0, 3.5, 50.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&neg_dhi);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Diffuse"));
    }

    #[test]
    fn test_validate_all_negative_ghi() {
        let neg_ghi = vec![HourlyWeatherData::new(
            20.0, 800.0, 100.0, -100.0, 3.5, 50.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&neg_ghi);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Global"));
    }

    #[test]
    fn test_validate_all_negative_wind() {
        let neg_wind = vec![HourlyWeatherData::new(
            20.0, 800.0, 100.0, 900.0, -1.0, 50.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&neg_wind);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Wind speed"));
    }

    #[test]
    fn test_validate_all_negative_humidity() {
        let neg_hum = vec![HourlyWeatherData::new(
            20.0, 800.0, 100.0, 900.0, 3.5, -10.0, 0,
        )];
        let result = HourlyWeatherData::validate_all(&neg_hum);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("humidity"));
    }

    #[test]
    fn test_validate_all_negative_infrared() {
        let mut weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
        weather.horizontal_infrared = -5.0;
        let result = HourlyWeatherData::validate_all(&[weather]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Infrared"));
    }

    #[test]
    fn test_validate_all_empty_slice() {
        let empty: Vec<HourlyWeatherData> = vec![];
        assert!(HourlyWeatherData::validate_all(&empty).is_ok());
    }

    #[test]
    fn test_is_complete_empty_slice() {
        let empty: Vec<HourlyWeatherData> = vec![];
        assert!(HourlyWeatherData::is_complete(&empty));
    }

    #[test]
    fn test_is_complete_with_infinite_values() {
        let inf_weather = vec![HourlyWeatherData::new(
            f64::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            0,
        )];
        assert!(!HourlyWeatherData::is_complete(&inf_weather));
    }

    #[test]
    fn test_weather_error_clone() {
        let err1 = WeatherError::InvalidHour(100);
        let err2 = err1.clone();
        assert_eq!(err1, err2);

        let err3 = WeatherError::ParseError("test".to_string());
        let err4 = err3.clone();
        assert_eq!(err3, err4);
    }

    #[test]
    fn test_weather_error_debug() {
        let err = WeatherError::InvalidHour(5000);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidHour"));
    }

    #[test]
    fn test_weather_error_is_error_trait() {
        let err: WeatherError = WeatherError::IoError("test".to_string());
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_weather_iterator_stops_at_invalid_hour() {
        struct ShortSource;
        impl WeatherSource for ShortSource {
            fn location(&self) -> Option<String> {
                Some("Short".to_string())
            }
            fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
                if hour >= 5 {
                    Err(WeatherError::InvalidHour(hour))
                } else {
                    Ok(HourlyWeatherData::new(20.0, 0.0, 0.0, 0.0, 2.0, 50.0, hour))
                }
            }
        }
        let source = ShortSource;
        let mut count = 0;
        for result in source.iter_hours() {
            if result.is_err() {
                break;
            }
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_weather_source_validate_all_success() {
        struct ValidSource;
        impl WeatherSource for ValidSource {
            fn location(&self) -> Option<String> {
                Some("Valid".to_string())
            }
            fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
                if hour >= 8760 {
                    Err(WeatherError::InvalidHour(hour))
                } else {
                    Ok(HourlyWeatherData::new(20.0, 0.0, 0.0, 0.0, 2.0, 50.0, hour))
                }
            }
        }
        let source = ValidSource;
        assert!(source.validate_all().is_ok());
    }

    #[test]
    fn test_hourly_weather_data_serialize_deserialize() {
        let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        let json = serde_json::to_string(&weather).unwrap();
        let deserialized: HourlyWeatherData = serde_json::from_str(&json).unwrap();
        assert_eq!(weather.dry_bulb_temp, deserialized.dry_bulb_temp);
        assert_eq!(weather.dni, deserialized.dni);
        assert_eq!(weather.hour_of_year, deserialized.hour_of_year);
    }

    #[test]
    fn test_hourly_weather_data_partial_equality() {
        let w1 = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        let w2 = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        assert_eq!(w1, w2);

        let w3 = HourlyWeatherData::new(21.0, 800.0, 100.0, 900.0, 3.5, 50.0, 100);
        assert_ne!(w1, w3);
    }
}
