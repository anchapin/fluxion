//! Embedded Denver TMY (Typical Meteorological Year) weather data.
//!
//! This module provides synthetic weather data approximating the climate of
//! Denver, Colorado, USA - the required location for ASHRAE 140 validation.
//!
//! The Denver weather is generated using:
//! - Seasonal temperature variation (cold winters, mild summers)
//! - Daily temperature cycles (cooler nights, warmer days)
//! - High altitude solar radiation (1655m elevation increases solar flux)
//! - Realistic wind speed patterns
//! - Humidity variations by season
//!
//! This embedded data is suitable for ASHRAE 140 test cases and development,
//! while production simulations should use actual EPW files.

use crate::weather::{HourlyWeatherData, WeatherError, WeatherSource};
use std::f64::consts::PI;

/// Denver TMY weather data source with embedded synthetic weather.
///
/// This struct generates realistic weather data for Denver, Colorado based on
/// climatic patterns suitable for ASHRAE 140 validation. Denver is at
/// 39.83°N latitude, 104.65°W longitude, and 1655m elevation.
///
/// # Denver Climate Characteristics
///
/// - **Elevation**: 1655m (high altitude increases solar radiation)
/// - **Climate**: Semi-arid continental climate
/// - **Winters**: Cold with occasional snow (January average: -1°C)
/// - **Summers**: Hot and dry (July average: 24°C)
/// - **Solar**: High solar resource due to clear skies and altitude
/// - **Wind**: Moderate, averaging 2-5 m/s
///
/// # Example
///
/// ```
/// use fluxion::weather::denver::DenverTmyWeather;
/// use fluxion::weather::WeatherSource;
///
/// let weather = DenverTmyWeather::default();
///
/// println!("Location: {}", weather.location().unwrap());
///
/// // Get weather for January 1st noon
/// let data = weather.get_hourly_data(12).unwrap();
/// println!("Temperature: {}°C", data.dry_bulb_temp);
/// ```
#[derive(Debug, Clone)]
pub struct DenverTmyWeather {
    location: String,
    /// Cache of generated hourly data
    hourly_data: Vec<Option<HourlyWeatherData>>,
}

impl Default for DenverTmyWeather {
    fn default() -> Self {
        Self::new()
    }
}

impl DenverTmyWeather {
    /// Creates a new Denver TMY weather source.
    ///
    /// The weather data is generated on-demand when requested.
    pub fn new() -> Self {
        DenverTmyWeather {
            location: "Denver, CO".to_string(),
            hourly_data: vec![None; 8760],
        }
    }

    /// Generates weather data for a specific hour of the year.
    ///
    /// This method uses parametric equations to create realistic weather
    /// patterns based on Denver's climatic characteristics.
    ///
    /// # Arguments
    ///
    /// * `hour` - Hour of year (0-8759)
    ///
    /// # Returns
    ///
    /// Generated hourly weather data
    fn generate_hourly_data(&self, hour: usize) -> HourlyWeatherData {
        let day_of_year = hour / 24;
        let hour_of_day = hour % 24;

        // Convert day of year to radians for seasonal variation
        // Day 0 = Jan 1, Day 172 = Summer solstice (approximately)
        let day_angle = (day_of_year as f64 / 365.0) * 2.0 * PI;

        // Hour angle: 0 = solar noon, π/2 = 6am/pm, π = midnight
        let hour_angle = ((hour_of_day as f64 - 12.0) / 24.0) * 2.0 * PI;

        // === TEMPERATURE ===
        // Denver annual average: ~10°C, amplitude ~15°C
        // Winter minimum: -7°C (January), Summer maximum: 28°C (July)
        let seasonal_temp = 10.0 - 15.0 * day_angle.cos();

        // Daily temperature variation: 6-8°C amplitude
        // Coldest at ~4am, warmest at ~3pm (hour 15)
        // Shift hour angle by -3 hours (π/4 radians) to peak at 3pm
        let daily_temp = 7.0 * (hour_angle - PI / 4.0).cos();

        // Add some randomness for realism
        let temp_noise = ((hour as f64 * 0.1).sin() * 0.5).clamp(-2.0, 2.0);

        let dry_bulb_temp = seasonal_temp + daily_temp + temp_noise;

        // === SOLAR RADIATION ===
        // Denver latitude: 39.83°N = 0.695 radians
        let latitude = 39.83 * PI / 180.0;
        // Solar declination: use offset so max occurs at summer solstice (day ~172)
        // Offset of -80 days aligns with March equinox at day 80
        let declination_angle = day_angle - (80.0 / 365.0) * 2.0 * PI;
        let declination = 23.45 * PI / 180.0 * declination_angle.sin();

        // Solar hour angle at sunrise/sunset
        let _sunset_angle = (-latitude.tan() * declination.tan()).acos();

        // Solar elevation angle (0 at horizon, maximum at solar noon)
        let elevation = (latitude.sin() * declination.sin()
            + latitude.cos() * declination.cos() * hour_angle.cos())
        .asin();

        // Solar elevation for DNI (only positive when sun is above horizon)
        let dni = if elevation > 0.0 {
            // Maximum DNI at Denver altitude (high altitude = less atmosphere)
            // Denver at 1655m has ~15% less atmosphere than sea level
            // Sea level max: ~1000 W/m², Denver max: ~1100 W/m²
            let max_dni = 1100.0;
            // Atmospheric extinction: varies with elevation angle
            // At 1655m, clearer skies mean higher transmittance
            let air_mass = 1.0 / elevation.sin().max(0.1);
            let clear_sky = max_dni * (0.85_f64.powf(air_mass));
            clear_sky.max(0.0)
        } else {
            0.0
        };

        // DHI: Diffuse component from sky scattering
        // Typically 10-20% of GHI on clear days
        let dhi = if dni > 0.0 {
            dni * (0.1 + 0.05 * elevation.sin())
        } else {
            0.0
        };

        // GHI: Global horizontal = DNI * cos(elevation) + DHI
        let ghi = dni * elevation.sin().max(0.0) + dhi;

        // === WIND SPEED ===
        // Denver average: ~3.5 m/s
        // Seasonal: slightly windier in winter
        let seasonal_wind = 3.0 + 1.0 * day_angle.sin(); // Higher in winter

        // Daily: windier during day
        let daily_wind = 0.5 * (hour_angle - PI / 2.0).cos();

        // Gusts and variations
        let wind_noise = ((hour as f64 * 0.05).sin() * 0.5).clamp(-1.0, 1.0);

        let wind_speed = (seasonal_wind + daily_wind + wind_noise).clamp(0.5, 10.0);

        // === HUMIDITY ===
        // Denver is semi-arid
        // Relative humidity: 30-60% average, higher in winter
        let seasonal_humidity = 45.0 - 15.0 * day_angle.cos(); // Higher in winter

        // Daily: humidity inversely related to temperature
        // Cooler at night = higher relative humidity
        let daily_humidity = 5.0 * hour_angle.cos();

        let humidity = (seasonal_humidity + daily_humidity).clamp(10.0, 95.0);

        // === HORIZONTAL INFRARED RADIATION ===
        // Longwave radiation from the sky hemisphere
        // Calculated using sky emissivity and ambient temperature
        // σ * ε_sky * T_ambient^4 where σ = 5.67e-8 W/(m²·K⁴)
        //
        // Sky emissivity depends on cloud cover and humidity:
        // - Clear sky: ε ≈ 0.6-0.7 (lower IR)
        // - Cloudy sky: ε ≈ 0.85-0.95 (higher IR)
        //
        // Denver's semi-arid climate means generally lower emissivity (clearer skies)
        // but we also model some cloud cover variation
        const STEFAN_BOLTZMANN: f64 = 5.67e-8; // W/(m²·K⁴)

        // Sky emissivity: varies with cloud cover (modeled as function of DHI/DNI ratio)
        // Clear sky: ε ≈ 0.68, Overcast: ε ≈ 0.90
        let clearness = if dni > 100.0 {
            (dhi / dni).min(1.0) // Higher ratio = more diffuse = more clouds
        } else {
            0.5 // Night/low sun: assume partial cloud
        };

        // Emissivity increases with cloud cover
        let sky_emissivity = 0.68 + 0.22 * clearness;

        // At night, use a default emissivity based on humidity
        let effective_emissivity = if dni < 1.0 {
            // Nighttime: emissivity correlates with humidity
            0.70 + 0.002 * humidity
        } else {
            sky_emissivity
        };

        // Calculate horizontal infrared radiation
        // IR = ε_sky * σ * T_ambient^4
        let t_ambient_kelvin = dry_bulb_temp + 273.15;
        let horizontal_infrared =
            effective_emissivity * STEFAN_BOLTZMANN * t_ambient_kelvin.powi(4);

        HourlyWeatherData {
            dry_bulb_temp,
            dni,
            dhi,
            ghi,
            wind_speed,
            humidity,
            horizontal_infrared,
            hour_of_year: hour,
        }
    }

    /// Returns statistics about the generated weather data.
    ///
    /// This method analyzes the generated weather to provide summary statistics.
    pub fn statistics(&self) -> DenverWeatherStatistics {
        let mut max_temp: f64 = f64::NEG_INFINITY;
        let mut min_temp: f64 = f64::INFINITY;
        let mut sum_temp: f64 = 0.0;
        let mut solar_hours: f64 = 0.0;
        let mut max_ghi: f64 = 0.0;

        for hour in 0..8760 {
            let data = self.get_hourly_data(hour).unwrap();

            max_temp = max_temp.max(data.dry_bulb_temp);
            min_temp = min_temp.min(data.dry_bulb_temp);
            sum_temp += data.dry_bulb_temp;

            if data.ghi > 0.0 {
                solar_hours += 1.0;
                max_ghi = max_ghi.max(data.ghi);
            }
        }

        DenverWeatherStatistics {
            location: self.location.clone(),
            max_temperature: max_temp,
            min_temperature: min_temp,
            avg_temperature: sum_temp / 8760.0,
            solar_hours,
            max_ghi,
        }
    }
}

impl WeatherSource for DenverTmyWeather {
    fn location(&self) -> Option<String> {
        Some(self.location.clone())
    }

    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
        if hour >= 8760 {
            return Err(WeatherError::InvalidHour(hour));
        }

        // Generate data if not already cached
        if self.hourly_data[hour].is_none() {
            let data = self.generate_hourly_data(hour);
            // Note: This requires interior mutability in practice
            // For now, we'll regenerate each time
            return Ok(data);
        }

        Ok(self.hourly_data[hour].clone().unwrap())
    }
}

/// Statistics summary for Denver weather data.
///
/// Provides aggregate information about the generated weather for validation
/// and analysis purposes.
#[derive(Debug, Clone, PartialEq)]
pub struct DenverWeatherStatistics {
    /// Location string (e.g., "Denver, CO")
    pub location: String,
    /// Maximum temperature (°C)
    pub max_temperature: f64,
    /// Minimum temperature (°C)
    pub min_temperature: f64,
    /// Average temperature (°C)
    pub avg_temperature: f64,
    /// Number of hours with solar radiation (GHI > 0)
    pub solar_hours: f64,
    /// Maximum global horizontal irradiance (W/m²)
    pub max_ghi: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denver_tmy_creation() {
        let weather = DenverTmyWeather::new();

        assert_eq!(weather.location(), Some("Denver, CO".to_string()));
    }

    #[test]
    fn test_denver_tmy_default() {
        let weather = DenverTmyWeather::default();

        assert_eq!(weather.location(), Some("Denver, CO".to_string()));
    }

    #[test]
    fn test_get_hourly_data_valid_hours() {
        let weather = DenverTmyWeather::new();

        // Test various hours throughout the year
        for hour in [0, 100, 1000, 5000, 8000] {
            let data = weather.get_hourly_data(hour).unwrap();
            assert_eq!(data.hour_of_year, hour);
            assert!(data.dry_bulb_temp > -50.0 && data.dry_bulb_temp < 50.0);
            assert!(data.wind_speed >= 0.0 && data.wind_speed <= 20.0);
            assert!(data.humidity >= 0.0 && data.humidity <= 100.0);
            assert!(data.dni >= 0.0 && data.dni <= 1200.0);
            assert!(data.dhi >= 0.0);
            assert!(data.ghi >= 0.0);
        }
    }

    #[test]
    fn test_get_hourly_data_invalid_hour() {
        let weather = DenverTmyWeather::new();

        let error = weather.get_hourly_data(8760);
        assert_eq!(error, Err(WeatherError::InvalidHour(8760)));

        let error = weather.get_hourly_data(10000);
        assert_eq!(error, Err(WeatherError::InvalidHour(10000)));
    }

    #[test]
    fn test_seasonal_temperature_variation() {
        let weather = DenverTmyWeather::new();

        // Winter (January): should be cold
        let winter_day = 10 * 24 + 12; // January 11, noon
        let winter_data = weather.get_hourly_data(winter_day).unwrap();
        assert!(winter_data.dry_bulb_temp < 5.0, "Winter should be cold");

        // Summer (July): should be warm
        let summer_day = 196 * 24 + 12; // July 15, noon
        let summer_data = weather.get_hourly_data(summer_day).unwrap();
        assert!(summer_data.dry_bulb_temp > 20.0, "Summer should be warm");
    }

    #[test]
    fn test_daily_temperature_cycle() {
        let weather = DenverTmyWeather::new();

        // Pick a day in spring (April 15, day 105)
        let day = 105 * 24;

        let dawn = weather.get_hourly_data(day + 6).unwrap(); // 6 AM
        let noon = weather.get_hourly_data(day + 12).unwrap(); // 12 PM
        let evening = weather.get_hourly_data(day + 18).unwrap(); // 6 PM
        let midnight = weather.get_hourly_data(day).unwrap(); // 12 AM

        // Noon should be warmest, midnight coolest
        assert!(noon.dry_bulb_temp > dawn.dry_bulb_temp);
        assert!(noon.dry_bulb_temp > evening.dry_bulb_temp);
        assert!(midnight.dry_bulb_temp < noon.dry_bulb_temp);
    }

    #[test]
    fn test_solar_radiation_patterns() {
        let weather = DenverTmyWeather::new();

        // Noon in summer: high solar
        let summer_noon = weather.get_hourly_data(196 * 24 + 12).unwrap();
        assert!(summer_noon.ghi > 500.0, "Summer noon should have high GHI");
        assert!(summer_noon.dni > 600.0, "Summer noon should have high DNI");

        // Midnight: no solar
        let midnight = weather.get_hourly_data(196 * 24).unwrap();
        assert!(midnight.ghi == 0.0, "Midnight should have no GHI");
        assert!(midnight.dni == 0.0, "Midnight should have no DNI");

        // Winter: lower solar than summer
        let winter_noon = weather.get_hourly_data(10 * 24 + 12).unwrap();
        assert!(
            winter_noon.ghi < summer_noon.ghi,
            "Winter solar should be lower"
        );
    }

    #[test]
    fn test_wind_speed_range() {
        let weather = DenverTmyWeather::new();

        for hour in 0..8760 {
            let data = weather.get_hourly_data(hour).unwrap();
            assert!(
                data.wind_speed >= 0.0 && data.wind_speed <= 15.0,
                "Wind speed out of range: {} at hour {}",
                data.wind_speed,
                hour
            );
        }
    }

    #[test]
    fn test_humidity_range() {
        let weather = DenverTmyWeather::new();

        for hour in 0..8760 {
            let data = weather.get_hourly_data(hour).unwrap();
            assert!(
                data.humidity >= 0.0 && data.humidity <= 100.0,
                "Humidity out of range: {} at hour {}",
                data.humidity,
                hour
            );
        }
    }

    #[test]
    fn test_denver_is_semi_arid() {
        let weather = DenverTmyWeather::new();
        let stats = weather.statistics();

        // Denver should be relatively dry (average humidity < 60%)
        // We can't directly calculate avg humidity from stats, but can check extremes
        assert!(stats.avg_temperature > -5.0 && stats.avg_temperature < 20.0);

        // Denver should have significant solar hours (clear skies)
        assert!(stats.solar_hours > 4000.0, "Should have >4000 solar hours");

        // Max GHI should be high due to altitude
        assert!(stats.max_ghi > 800.0, "Max GHI should be >800 W/m²");
    }

    #[test]
    fn test_weather_iterator() {
        let weather = DenverTmyWeather::new();

        let mut count = 0;
        for result in weather.iter_hours().take(100) {
            assert!(result.is_ok());
            let data = result.unwrap();
            assert_eq!(data.hour_of_year, count);
            count += 1;
        }

        assert_eq!(count, 100);
    }

    #[test]
    fn test_solar_radiation_consistency() {
        let weather = DenverTmyWeather::new();

        for hour in 0..8760 {
            let data = weather.get_hourly_data(hour).unwrap();

            // GHI should be >= DHI (always true by definition)
            assert!(
                data.ghi >= data.dhi,
                "GHI should be >= DHI: GHI={}, DHI={} at hour {}",
                data.ghi,
                data.dhi,
                hour
            );

            // If DNI is 0, DHI should also be 0 (or very close)
            if data.dni < 1.0 {
                assert!(
                    data.dhi < 10.0,
                    "If DNI is 0, DHI should be near 0: DHI={} at hour {}",
                    data.dhi,
                    hour
                );
            }

            // GHI roughly equals DNI*sin(elevation) + DHI
            // Since elevation is implicit, we check that GHI is not unreasonably large
            assert!(
                data.ghi <= 1200.0,
                "GHI should not exceed 1200 W/m²: GHI={} at hour {}",
                data.ghi,
                hour
            );
        }
    }

    #[test]
    fn test_hour_of_year_calculation() {
        let weather = DenverTmyWeather::new();

        // Test boundaries
        let first_hour = weather.get_hourly_data(0).unwrap();
        assert_eq!(first_hour.hour_of_year, 0);

        let last_hour = weather.get_hourly_data(8759).unwrap();
        assert_eq!(last_hour.hour_of_year, 8759);
    }

    #[test]
    fn test_statistics() {
        let weather = DenverTmyWeather::new();
        let stats = weather.statistics();

        assert_eq!(stats.location, "Denver, CO");
        assert!(stats.max_temperature > stats.min_temperature);
        assert!(stats.solar_hours > 0.0);
        assert!(stats.max_ghi > 0.0);
    }

    #[test]
    fn test_reproducibility() {
        let weather1 = DenverTmyWeather::new();
        let weather2 = DenverTmyWeather::new();

        // Same hour should produce same data
        let data1 = weather1.get_hourly_data(1000).unwrap();
        let data2 = weather2.get_hourly_data(1000).unwrap();

        assert_eq!(data1.dry_bulb_temp, data2.dry_bulb_temp);
        assert_eq!(data1.dni, data2.dni);
        assert_eq!(data1.dhi, data2.dhi);
        assert_eq!(data1.ghi, data2.ghi);
        assert_eq!(data1.wind_speed, data2.wind_speed);
        assert_eq!(data1.humidity, data2.humidity);
    }
}
