//! Embedded Minneapolis TMY (Typical Meteorological Year) weather data.
//!
//! This module provides synthetic weather data approximating the climate of
//! Minneapolis, Minnesota, USA - representing Climate Zone 7/8 (very cold).
//!
//! The Minneapolis weather is generated using:
//! - Very cold winters (subarctic conditions)
//! - Warm summers
//! - Moderate humidity
//! - Moderate solar radiation
//! - Moderate wind speeds
//!
//! This embedded data is suitable for ASHRAE 140 test cases and development,
//! while production simulations should use actual EPW files.

use crate::weather::{HourlyWeatherData, WeatherError, WeatherSource};
use std::f64::consts::PI;

/// Minneapolis TMY weather data source with embedded synthetic weather.
///
/// This struct generates realistic weather data for Minneapolis, Minnesota based on
/// climatic patterns suitable for ASHRAE 140 validation. Minneapolis is at
/// 44.88°N latitude, 93.22°W longitude, and 255m elevation.
///
/// # Minneapolis Climate Characteristics
///
/// - **Elevation**: 255m
/// - **Climate**: Cold continental / very cold winters
/// - **Winters**: Very cold (January average: -10°C)
/// - **Summers**: Warm (July average: 23°C)
/// - **Solar**: Moderate (higher in summer due to longer days)
/// - **Wind**: Moderate, averaging 3-5 m/s
/// - **Humidity**: Moderate (50-80% RH)
///
/// # Example
///
/// ```
/// use fluxion_core::weather::minneapolis::MinneapolisTmyWeather;
/// use fluxion_core::weather::WeatherSource;
///
/// let weather = MinneapolisTmyWeather::default();
///
/// println!("Location: {}", weather.location().unwrap());
///
/// // Get weather for January 1st noon
/// let data = weather.get_hourly_data(12).unwrap();
/// println!("Temperature: {}°C", data.dry_bulb_temp);
/// ```
#[derive(Debug, Clone)]
pub struct MinneapolisTmyWeather {
    location: String,
}

impl Default for MinneapolisTmyWeather {
    fn default() -> Self {
        Self::new()
    }
}

impl MinneapolisTmyWeather {
    pub fn new() -> Self {
        MinneapolisTmyWeather {
            location: "Minneapolis, MN".to_string(),
        }
    }

    fn generate_hourly_data(&self, hour: usize) -> HourlyWeatherData {
        let day_of_year = hour / 24;
        let hour_of_day = hour % 24;

        let day_angle = (day_of_year as f64 / 365.0) * 2.0 * PI;
        let hour_angle = ((hour_of_day as f64 - 12.0) / 24.0) * 2.0 * PI;

        // === TEMPERATURE ===
        // Minneapolis annual average: ~8°C, amplitude ~23°C
        // Winter minimum: ~-20°C (January), Summer maximum: ~30°C (July)
        let seasonal_temp = 8.0 - 23.0 * day_angle.cos();

        // Daily temperature variation: 8-12°C amplitude
        // Peak at ~3pm, minimum at ~5am
        let daily_temp = 10.0 * (hour_angle - PI / 4.0).cos();

        // Small noise for realism
        let temp_noise = ((hour as f64 * 0.1).sin() * 1.0).clamp(-3.0, 3.0);

        let dry_bulb_temp = seasonal_temp + daily_temp + temp_noise;

        // === SOLAR RADIATION ===
        // Minneapolis latitude: 44.88°N = 0.783 radians
        let latitude = 44.88 * PI / 180.0;
        let declination_angle = day_angle - (80.0 / 365.0) * 2.0 * PI;
        let declination = 23.45 * PI / 180.0 * declination_angle.sin();

        let elevation = (latitude.sin() * declination.sin()
            + latitude.cos() * declination.cos() * hour_angle.cos())
        .asin();

        let dni = if elevation > 0.0 {
            // Minneapolis at 255m - slightly less than Denver altitude
            let max_dni = 1050.0;
            let air_mass = 1.0 / elevation.sin().max(0.1);
            let clear_sky = max_dni * (0.82_f64.powf(air_mass));
            clear_sky.max(0.0)
        } else {
            0.0
        };

        // DHI: Moderate diffuse fraction
        let dhi = if dni > 0.0 {
            dni * (0.18 + 0.08 * elevation.sin())
        } else {
            0.0
        };

        let ghi = dni * elevation.sin().max(0.0) + dhi;

        // === WIND SPEED ===
        // Minneapolis average: ~4 m/s
        // Windier in winter
        let seasonal_wind = 4.0 + 1.5 * day_angle.sin();

        // Daily: windier during day
        let daily_wind = 0.5 * (hour_angle - PI / 2.0).cos();

        let wind_noise = ((hour as f64 * 0.05).sin() * 0.5).clamp(-1.0, 1.0);

        let wind_speed = (seasonal_wind + daily_wind + wind_noise).clamp(0.5, 12.0);

        // === HUMIDITY ===
        // Minneapolis: moderate humidity year-round
        // RH: 50-80% average, higher in summer
        let seasonal_humidity = 65.0 - 15.0 * day_angle.cos();

        // Daily: humidity inversely related to temperature
        let daily_humidity = 8.0 * hour_angle.cos();

        let humidity = (seasonal_humidity + daily_humidity).clamp(30.0, 95.0);

        // === HORIZONTAL INFRARED RADIATION ===
        const STEFAN_BOLTZMANN: f64 = 5.67e-8;

        let clearness = if dni > 100.0 {
            (dhi / dni).min(1.0)
        } else {
            0.5
        };

        // Minneapolis: moderate emissivity
        let sky_emissivity = 0.70 + 0.18 * clearness;

        let effective_emissivity = if dni < 1.0 {
            0.72 + 0.0015 * humidity
        } else {
            sky_emissivity
        };

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
            ground_temperature: None,
            horizontal_illuminance: None,
            diffuse_illuminance: None,
            snow_depth: None,
            snow_cover: None,
            present_weather: None,
            present_weather_code: None,
        }
    }

    pub fn statistics(&self) -> MinneapolisWeatherStatistics {
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
            }
            max_ghi = max_ghi.max(data.ghi);
        }

        MinneapolisWeatherStatistics {
            location: self.location.clone(),
            max_temperature: max_temp,
            min_temperature: min_temp,
            avg_temperature: sum_temp / 8760.0,
            solar_hours,
            max_ghi,
        }
    }
}

impl WeatherSource for MinneapolisTmyWeather {
    fn location(&self) -> Option<String> {
        Some(self.location.clone())
    }

    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
        if hour >= 8760 {
            return Err(WeatherError::InvalidHour(hour));
        }

        Ok(self.generate_hourly_data(hour))
    }
}

#[derive(Debug, Clone)]
pub struct MinneapolisWeatherStatistics {
    pub location: String,
    pub max_temperature: f64,
    pub min_temperature: f64,
    pub avg_temperature: f64,
    pub solar_hours: f64,
    pub max_ghi: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minneapolis_tmy_creation() {
        let weather = MinneapolisTmyWeather::new();
        assert_eq!(weather.location(), Some("Minneapolis, MN".to_string()));
    }

    #[test]
    fn test_minneapolis_tmy_default() {
        let weather = MinneapolisTmyWeather::default();
        assert_eq!(weather.location(), Some("Minneapolis, MN".to_string()));
    }

    #[test]
    fn test_get_hourly_data_valid_hours() {
        let weather = MinneapolisTmyWeather::new();

        // Test a few hours across the year
        for hour in [0, 1000, 3000, 5000, 7000, 8759] {
            let result = weather.get_hourly_data(hour);
            assert!(result.is_ok(), "Hour {} should be valid", hour);
        }
    }

    #[test]
    fn test_get_hourly_data_invalid_hour() {
        let weather = MinneapolisTmyWeather::new();
        let result = weather.get_hourly_data(8760);
        assert!(result.is_err());
    }

    #[test]
    fn test_minneapolis_is_cold_in_winter() {
        let weather = MinneapolisTmyWeather::new();
        let data_winter = weather.get_hourly_data(200).unwrap(); // Winter

        // Minneapolis winter should be cold
        assert!(
            data_winter.dry_bulb_temp < 5.0,
            "Minneapolis winter temp {} should be < 5°C",
            data_winter.dry_bulb_temp
        );
    }

    #[test]
    fn test_minneapolis_is_warm_in_summer() {
        let weather = MinneapolisTmyWeather::new();
        let data_summer = weather.get_hourly_data(4344).unwrap(); // Summer

        // Minneapolis summer should be warm
        assert!(
            data_summer.dry_bulb_temp > 15.0,
            "Minneapolis summer temp {} should be > 15°C",
            data_summer.dry_bulb_temp
        );
    }

    #[test]
    fn test_statistics() {
        let weather = MinneapolisTmyWeather::new();
        let stats = weather.statistics();

        assert_eq!(stats.location, "Minneapolis, MN");
        assert!(stats.max_temperature > stats.min_temperature);
        assert!(stats.min_temperature < 0.0); // Very cold winters
        assert!(stats.max_temperature > 20.0); // Warm summers
        assert!(stats.solar_hours > 4000.0);
        assert!(stats.max_ghi > 800.0);
    }
}
