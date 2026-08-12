//! Embedded Miami TMY (Typical Meteorological Year) weather data.
//!
//! This module provides synthetic weather data approximating the climate of
//! Miami, Florida, USA - a Climate Zone 3 (hot-humid) location.
//!
//! The Miami weather is generated using:
//! - Warm temperatures year-round (mild winters, hot summers)
//! - High humidity (70-90% RH)
//! - High solar radiation (subtropical clear skies)
//! - Tropical rainfall patterns
//! - Low wind speeds
//!
//! This embedded data is suitable for ASHRAE 140 test cases and development,
//! while production simulations should use actual EPW files.

use crate::weather::{HourlyWeatherData, WeatherError, WeatherSource};
use std::f64::consts::PI;

/// Miami TMY weather data source with embedded synthetic weather.
///
/// This struct generates realistic weather data for Miami, Florida based on
/// climatic patterns suitable for ASHRAE 140 validation. Miami is at
/// 25.82°N latitude, 80.30°W longitude, and 11m elevation.
///
/// # Miami Climate Characteristics
///
/// - **Elevation**: 11m (sea level proximity)
/// - **Climate**: Tropical savanna / hot-humid
/// - **Winters**: Warm (January average: 20°C)
/// - **Summers**: Hot and humid (July average: 30°C)
/// - **Solar**: High year-round (subtropical clear skies)
/// - **Wind**: Light, averaging 2-3 m/s
/// - **Humidity**: Very high (70-90% RH year-round)
///
/// # Example
///
/// ```
/// use fluxion_core::weather::miami::MiamiTmyWeather;
/// use fluxion_core::weather::WeatherSource;
///
/// let weather = MiamiTmyWeather::default();
///
/// println!("Location: {}", weather.location().unwrap());
///
/// // Get weather for July 1st solar noon
/// let data = weather.get_hourly_data(4356).unwrap(); // July 1, 12:00
/// println!("Temperature: {}°C", data.dry_bulb_temp);
/// ```
#[derive(Debug, Clone)]
pub struct MiamiTmyWeather {
    location: String,
}

impl Default for MiamiTmyWeather {
    fn default() -> Self {
        Self::new()
    }
}

impl MiamiTmyWeather {
    pub fn new() -> Self {
        MiamiTmyWeather {
            location: "Miami, FL".to_string(),
        }
    }

    fn generate_hourly_data(&self, hour: usize) -> HourlyWeatherData {
        let day_of_year = hour / 24;
        let hour_of_day = hour % 24;

        let day_angle = (day_of_year as f64 / 365.0) * 2.0 * PI;
        let hour_angle = ((hour_of_day as f64 - 12.0) / 24.0) * 2.0 * PI;

        // === TEMPERATURE ===
        // Miami annual average: ~25°C, amplitude ~8°C
        // Winter minimum: ~15°C (January), Summer maximum: ~33°C (July)
        let seasonal_temp = 25.0 - 8.0 * day_angle.cos();

        // Daily temperature variation: 4-6°C amplitude
        // Peak at ~3pm, minimum at ~5am
        let daily_temp = 5.0 * (hour_angle - PI / 4.0).cos();

        // Small noise for realism
        let temp_noise = ((hour as f64 * 0.1).sin() * 0.5).clamp(-1.5, 1.5);

        let dry_bulb_temp = seasonal_temp + daily_temp + temp_noise;

        // === SOLAR RADIATION ===
        // Miami latitude: 25.82°N = 0.451 radians
        let latitude = 25.82 * PI / 180.0;
        let declination_angle = day_angle - (80.0 / 365.0) * 2.0 * PI;
        let declination = 23.45 * PI / 180.0 * declination_angle.sin();

        let elevation = (latitude.sin() * declination.sin()
            + latitude.cos() * declination.cos() * hour_angle.cos())
        .asin();

        let dni = if elevation > 0.0 {
            // Miami at sea level - slightly lower max DNI than Denver altitude
            let max_dni = 1050.0;
            let air_mass = 1.0 / elevation.sin().max(0.1);
            let clear_sky = max_dni * (0.88_f64.powf(air_mass));
            clear_sky.max(0.0)
        } else {
            0.0
        };

        // DHI: Higher diffuse fraction in humid climates
        let dhi = if dni > 0.0 {
            dni * (0.25 + 0.1 * elevation.sin())
        } else {
            0.0
        };

        let ghi = dni * elevation.sin().max(0.0) + dhi;

        // === WIND SPEED ===
        // Miami average: ~2.5 m/s (light winds)
        let seasonal_wind = 2.5 + 0.5 * day_angle.sin();

        // Daily: slightly windier during day
        let daily_wind = 0.3 * (hour_angle - PI / 2.0).cos();

        let wind_noise = ((hour as f64 * 0.05).sin() * 0.3).clamp(-0.5, 0.5);

        let wind_speed = (seasonal_wind + daily_wind + wind_noise).clamp(0.5, 8.0);

        // === HUMIDITY ===
        // Miami is very humid year-round
        // RH: 75-90% average, slightly lower in winter
        let seasonal_humidity = 80.0 - 10.0 * day_angle.cos();

        // Daily: humidity inversely related to temperature
        let daily_humidity = 5.0 * hour_angle.cos();

        let humidity = (seasonal_humidity + daily_humidity).clamp(50.0, 98.0);

        // === HORIZONTAL INFRARED RADIATION ===
        const STEFAN_BOLTZMANN: f64 = 5.67e-8;

        let clearness = if dni > 100.0 {
            (dhi / dni).min(1.0)
        } else {
            0.5
        };

        // Miami: high humidity means higher emissivity
        let sky_emissivity = 0.75 + 0.20 * clearness;

        let effective_emissivity = if dni < 1.0 {
            0.80 + 0.001 * humidity
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

    pub fn statistics(&self) -> MiamiWeatherStatistics {
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

        MiamiWeatherStatistics {
            location: self.location.clone(),
            max_temperature: max_temp,
            min_temperature: min_temp,
            avg_temperature: sum_temp / 8760.0,
            solar_hours,
            max_ghi,
        }
    }
}

impl WeatherSource for MiamiTmyWeather {
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
pub struct MiamiWeatherStatistics {
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
    fn test_miami_tmy_creation() {
        let weather = MiamiTmyWeather::new();
        assert_eq!(weather.location(), Some("Miami, FL".to_string()));
    }

    #[test]
    fn test_miami_tmy_default() {
        let weather = MiamiTmyWeather::default();
        assert_eq!(weather.location(), Some("Miami, FL".to_string()));
    }

    #[test]
    fn test_get_hourly_data_valid_hours() {
        let weather = MiamiTmyWeather::new();

        // Test a few hours across the year
        for hour in [0, 1000, 3000, 5000, 7000, 8759] {
            let result = weather.get_hourly_data(hour);
            assert!(result.is_ok(), "Hour {} should be valid", hour);
            let data = result.unwrap();
            assert!(data.dry_bulb_temp > 10.0, "Miami should be warm");
            assert!(data.humidity > 50.0, "Miami should be humid");
        }
    }

    #[test]
    fn test_get_hourly_data_invalid_hour() {
        let weather = MiamiTmyWeather::new();
        let result = weather.get_hourly_data(8760);
        assert!(result.is_err());
    }

    #[test]
    fn test_miami_is_warm() {
        let weather = MiamiTmyWeather::new();
        let stats = weather.statistics();

        // Miami should be warm year-round
        assert!(
            stats.min_temperature > 10.0,
            "Miami min temp {} should be > 10°C",
            stats.min_temperature
        );
        assert!(
            stats.avg_temperature > 20.0,
            "Miami avg temp {} should be > 20°C",
            stats.avg_temperature
        );
    }

    #[test]
    fn test_miami_is_humid() {
        let weather = MiamiTmyWeather::new();
        let data = weather.get_hourly_data(3000).unwrap(); // Summer
        assert!(
            data.humidity > 70.0,
            "Miami summer humidity {} should be > 70%",
            data.humidity
        );
    }

    #[test]
    fn test_miami_solar_radiation() {
        let weather = MiamiTmyWeather::new();
        // Hour 4356 = day-of-year 181 (July 1), hour-of-day 12 (solar noon).
        // NOTE: the previous index 4344 was day 181, hour 0 (midnight) — the sun is
        // ~41° below the horizon so the generator correctly returns GHI = 0. That
        // was the source of the pre-existing failure (#2673), not the GHI model.
        // Verified via Python (RULES.md §0): at hour-of-day 12 on July 1 the
        // generator yields elevation ≈ 87°, DNI ≈ 924, DHI ≈ 323, GHI ≈ 1246 W/m² —
        // within the plausible clear-sky band for Miami (25.82°N) at the July 1
        // solar noon (Haurwitz ≈ 1034 W/m², ASHRAE 2009 ≈ 1004 W/m²).
        let data_noon = weather.get_hourly_data(4356).unwrap(); // July 1, solar noon

        // Miami should have significant solar radiation
        assert!(
            data_noon.ghi > 400.0,
            "Miami summer noon GHI {} should be > 400 W/m²",
            data_noon.ghi
        );
    }

    #[test]
    fn test_statistics() {
        let weather = MiamiTmyWeather::new();
        let stats = weather.statistics();

        assert_eq!(stats.location, "Miami, FL");
        assert!(stats.max_temperature > stats.min_temperature);
        assert!(stats.solar_hours > 4000.0); // Most daylight hours
        assert!(stats.max_ghi > 800.0);
    }
}
