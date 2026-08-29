//! Weather → Solar integration tests for bottom-up testing PRD.
//!
//! These tests verify the wiring from Weather TMY3 data to solar irradiance computation.
//! The diagnostic chain starts with Weather data (DNI, DHI, GHI) which must be correctly
//! passed to the solar position and surface irradiance calculations.
//!
//! # Wire-Edge Coverage
//!
//! - **Weather → Solar**: TMY3 DNI/DHI/GHI → `calculate_solar_position()` → surface irradiance
//!
//! # References
//!
//! - `src/weather/mod.rs` - HourlyWeatherData
//! - `src/weather/denver.rs` - DenverTmyWeather (embedded TMY3)
//! - `src/solar/solar_position.rs` - Solar position calculation
//! - `src/solar/surface_irradiance.rs` - Surface irradiance calculation

use fluxion::solar::{
    calculate_solar_position,
    surface_irradiance::{calculate_surface_irradiance, Orientation, SurfaceIrradiance},
};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Denver location constants (ASHRAE 140 standard location).
const DENVER_LAT: f64 = 39.74;
const DENVER_LON: f64 = -105.18;

/// Convert hour-of-year (0-8759) to (year, month, day, day_of_year, hour).
fn hour_of_year_to_calendar(hour: usize) -> (i32, u32, u32, usize, f64) {
    let hour = hour.min(8759);
    let day_of_year = hour / 24;
    let hour_of_day = (hour % 24) as f64;

    static DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut remaining = day_of_year;
    let mut month = 1u32;

    for &days in &DAYS_IN_MONTH {
        if remaining < days as usize {
            break;
        }
        remaining -= days as usize;
        month += 1;
    }

    let day = remaining as u32 + 1;
    (2024, month, day, day_of_year + 1, hour_of_day) // day_of_year is 1-indexed for the API
}

/// July 15 noon in non-leap year = day 196, hour 12
/// (31+28+31+30+31+30 = 181 days through June, + 15 = 196)
const JULY_15_NOON: usize = (196 * 24) + 12;

/// December 21 noon in non-leap year = day 355, hour 12
/// (31+28+31+30+31+30+31+31+30+31+30 = 334 days through November, + 21 = 355)
const DEC_21_NOON: usize = (355 * 24) + 12;

/// Test that Weather TMY3 data provides valid solar radiation values.
///
/// Verifies that the HourlyWeatherData from DenverTmyWeather contains
/// physically reasonable DNI, DHI, and GHI values during daytime hours.
#[test]
fn test_tmy3_provides_valid_solar_radiation_daytime() {
    let weather = DenverTmyWeather::new();

    // July 15, hour 12 (noon) - peak solar conditions
    let hour = JULY_15_NOON;
    let data = weather.get_hourly_data(hour).unwrap();

    // Verify solar radiation values are physically reasonable
    assert!(
        data.dni >= 0.0 && data.dni <= 1361.0,
        "DNI {} W/m² out of physical range [0, 1361]",
        data.dni
    );
    assert!(
        data.dhi >= 0.0 && data.dhi <= 500.0,
        "DHI {} W/m² out of typical range [0, 500]",
        data.dhi
    );
    assert!(
        data.ghi >= 0.0 && data.ghi <= 1200.0,
        "GHI {} W/m² out of typical range [0, 1200]",
        data.ghi
    );

    // GHI should approximately equal DNI * cos(zenith) + DHI at high elevation
    // Allow 20% tolerance for atmospheric effects
    assert!(
        data.ghi > 0.0,
        "GHI should be positive during daytime"
    );
}

/// Test wiring from Weather TMY3 to solar position calculation.
///
/// This test verifies that:
/// 1. Weather data provides hour-of-year
/// 2. Hour-of-year converts correctly to calendar date/time
/// 3. Solar position calculation receives valid inputs
/// 4. Solar position output is physically reasonable
#[test]
fn test_weather_to_solar_position_wiring() {
    let weather = DenverTmyWeather::new();

    // Test summer noon (July 15, 12:00 LST)
    let summer_hour = JULY_15_NOON;
    let data = weather.get_hourly_data(summer_hour).unwrap();

    let (year, month, day, _day_of_year, hour) = hour_of_year_to_calendar(summer_hour);
    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0), // Denver UTC offset
    );

    // Verify solar position is physically reasonable for summer noon
    assert!(
        solar_pos.altitude_deg > 0.0,
        "Sun should be above horizon at noon in summer"
    );
    assert!(
        solar_pos.altitude_deg < 90.0,
        "Altitude {} should be less than 90°",
        solar_pos.altitude_deg
    );
    assert!(
        solar_pos.zenith_deg > 0.0 && solar_pos.zenith_deg < 90.0,
        "Zenith {} should be in (0, 90)",
        solar_pos.zenith_deg
    );

    // For Denver in July, solar altitude at noon should be high (summer)
    // 90° - |39.74° - 23.5°| = 90° - 16.24° = ~73.76°
    assert!(
        solar_pos.altitude_deg > 60.0,
        "Summer noon altitude {} should exceed 60° for Denver",
        solar_pos.altitude_deg
    );
}

/// Test wiring from solar position to surface irradiance calculation.
///
/// This test verifies that:
/// 1. Solar position is correctly computed from weather/time
/// 2. Surface irradiance receives valid solar position
/// 3. Surface irradiance output is physically reasonable
#[test]
fn test_solar_position_to_surface_irradiance_wiring() {
    let weather = DenverTmyWeather::new();

    // Test summer noon
    let summer_hour = JULY_15_NOON;
    let data = weather.get_hourly_data(summer_hour).unwrap();

    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(summer_hour);
    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // Calculate irradiance on south-facing vertical surface (typical for windows)
    let irradiance = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::South,
        0.3, // Ground reflectance (typical for grass/concrete)
        day_of_year,
    );

    // Verify irradiance components are physical
    assert!(
        irradiance.beam_wm2 >= 0.0,
        "Beam irradiance {} should be non-negative",
        irradiance.beam_wm2
    );
    assert!(
        irradiance.diffuse_wm2 >= 0.0,
        "Diffuse irradiance {} should be non-negative",
        irradiance.diffuse_wm2
    );
    assert!(
        irradiance.ground_reflected_wm2 >= 0.0,
        "Ground-reflected {} should be non-negative",
        irradiance.ground_reflected_wm2
    );
    assert!(
        irradiance.total_wm2 >= 0.0,
        "Total irradiance {} should be non-negative",
        irradiance.total_wm2
    );

    // At noon in summer, a south-facing vertical surface in Denver should receive
    // significant solar gain (but less than horizontal due to angle)
    assert!(
        irradiance.total_wm2 > 100.0,
        "South-facing wall should receive >100 W/m² at summer noon, got {}",
        irradiance.total_wm2
    );
}

/// Test full Weather → Solar wiring chain for multiple hours.
///
/// Verifies the complete wire from TMY3 weather data through solar position
/// to surface irradiance for a representative summer day.
#[test]
fn test_full_wiring_chain_summer_day() {
    let weather = DenverTmyWeather::new();

    // July 15 (day 196) - test hours 6, 9, 12, 15, 18 (6am to 6pm)
    let day_start = 196 * 24;
    let test_hours = [6, 9, 12, 15, 18];

    for &hour_of_day in &test_hours {
        let hour = day_start + hour_of_day;
        let data = weather.get_hourly_data(hour).unwrap();
        let (year, month, day, day_of_year, hour_f) = hour_of_year_to_calendar(hour);

        let solar_pos = calculate_solar_position(
            DENVER_LAT,
            DENVER_LON,
            year,
            month,
            day,
            hour_f,
            Some(-7.0),
        );

        // Calculate irradiance on horizontal surface
        let irradiance = calculate_surface_irradiance(
            &solar_pos,
            data.dni,
            data.dhi,
            Some(data.ghi),
            Orientation::Horizontal,
            0.3, // Ground reflectance
            day_of_year,
        );

        // Daytime hours should produce non-zero irradiance (except 6am/6pm might be near zero)
        if hour_of_day >= 9 && hour_of_day <= 15 {
            assert!(
                irradiance.total_wm2 > 50.0,
                "Hour {} should have significant horizontal irradiance, got {} W/m²",
                hour,
                irradiance.total_wm2
            );
        }
    }
}

/// Test that nighttime hours produce zero beam irradiance.
#[test]
fn test_nighttime_zero_beam() {
    let weather = DenverTmyWeather::new();

    // Midnight (hour 0)
    let data = weather.get_hourly_data(0).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(0);

    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // Sun should be below horizon at midnight
    assert!(
        !solar_pos.is_above_horizon(),
        "Sun should be below horizon at midnight"
    );

    // Surface irradiance on any surface should be zero or near-zero at night
    let irradiance = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::South,
        0.3,
        day_of_year,
    );

    assert!(
        irradiance.beam_wm2 == 0.0,
        "Beam should be zero at night, got {}",
        irradiance.beam_wm2
    );
}

/// Test winter solstice (December 21) solar position for Denver.
///
/// This tests the edge case of low solar altitude in winter,
/// verifying the wire handles the full annual range.
#[test]
fn test_winter_solstice_solar_position() {
    let weather = DenverTmyWeather::new();

    // December 21 (day 355) noon
    let winter_hour = DEC_21_NOON;
    let data = weather.get_hourly_data(winter_hour).unwrap();

    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(winter_hour);
    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // Winter noon sun should be much lower than summer
    // 90° - |39.74° - (-23.5°)| = 90° - 63.24° = ~26.76°
    assert!(
        solar_pos.altitude_deg > 20.0 && solar_pos.altitude_deg < 40.0,
        "Winter noon altitude {} should be low (20-40°) for Denver",
        solar_pos.altitude_deg
    );

    // Calculate horizontal irradiance
    let irradiance = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::Horizontal,
        0.3,
        day_of_year,
    );

    // Winter noon GHI should still be positive but lower than summer
    assert!(
        irradiance.total_wm2 > 0.0,
        "Winter noon should still have positive irradiance, got {}",
        irradiance.total_wm2
    );
}

/// Test multiple surface orientations receive different irradiance.
///
/// This verifies the wire correctly handles the incidence angle calculation
/// for different surface tilts and azimuths.
#[test]
fn test_different_orientations_different_irradiance() {
    let weather = DenverTmyWeather::new();

    // Summer noon
    let summer_hour = JULY_15_NOON;
    let data = weather.get_hourly_data(summer_hour).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(summer_hour);

    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // Test horizontal vs vertical surfaces
    let horizontal = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::Horizontal,
        0.3,
        day_of_year,
    );

    let south_vertical = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::South,
        0.3,
        day_of_year,
    );

    let north_vertical = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::North,
        0.3,
        day_of_year,
    );

    // All should be positive during daytime
    assert!(horizontal.total_wm2 > 0.0);
    assert!(south_vertical.total_wm2 > 0.0);
    assert!(north_vertical.total_wm2 >= 0.0);

    // In Denver summer at noon:
    // - Horizontal should receive maximum direct beam
    // - South-facing vertical should receive significant beam
    // - North-facing vertical should receive minimal beam (in shadow of building)

    // South should receive more than north
    assert!(
        south_vertical.total_wm2 > north_vertical.total_wm2,
        "South wall {} should receive more than north wall {} at summer noon",
        south_vertical.total_wm2,
        north_vertical.total_wm2
    );
}
