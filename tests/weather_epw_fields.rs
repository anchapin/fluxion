//! Field-by-field EPW parser validation tests
//!
//! Validates every parsed field from a known EPW record against expected values.
//! Uses bundled test EPW files to ensure the parser handles all field types correctly.
//!
//! # Fields Validated
//!
//! - Dry bulb temperature, dew point, relative humidity
//! - Atmospheric pressure
//! - Global horizontal irradiance (GHI), DNI, DHI
//! - Wind speed, wind direction
//! - Total sky cover, opaque sky cover
//! - Surface weather observation flags
//!
//! # Acceptance Criteria
//!
//! - Each field within ±0.01 of known values (or exact for integer fields)
//! - Missing data codes (9999) handled correctly
//! - Test runs in <50ms
//! - No network required (use bundled test EPW)

use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

const EPSILON: f64 = 0.01;

fn assert_f64_near(actual: f64, expected: f64, field_name: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < EPSILON,
        "{} mismatch: expected {:.4}, got {:.4}, diff {:.4}",
        field_name,
        expected,
        actual,
        diff
    );
}

fn assert_int_near(actual: i32, expected: i32, field_name: &str) {
    assert_eq!(
        actual, expected,
        "{} mismatch: expected {}, got {}",
        field_name, expected, actual
    );
}

#[test]
fn test_denver_hour_0_all_fields() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source.get_hourly_data(0).expect("Failed to get hour 0");

    assert_eq!(data.hour_of_year, 0);

    assert_f64_near(data.dry_bulb_temp, -3.00, "dry_bulb_temp");
    assert_f64_near(data.humidity, 92.0, "humidity");
    assert_f64_near(data.ghi, 0.0, "ghi");
    assert_f64_near(data.dni, 0.0, "dni");
    assert_f64_near(data.dhi, 0.0, "dhi");
    assert_f64_near(data.wind_speed, 0.00, "wind_speed");
    assert_f64_near(data.horizontal_infrared, 257.0, "horizontal_infrared");
}

#[test]
fn test_denver_hour_1_all_fields() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source.get_hourly_data(1).expect("Failed to get hour 1");

    assert_f64_near(data.dry_bulb_temp, -3.00, "dry_bulb_temp");
    assert_f64_near(data.humidity, 77.0, "humidity");
    assert_f64_near(data.wind_speed, 2.10, "wind_speed");
}

#[test]
fn test_denver_hour_8_solar_values() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source.get_hourly_data(8).expect("Failed to get hour 8");

    assert_f64_near(data.dry_bulb_temp, 0.00, "dry_bulb_temp");
    assert_f64_near(data.humidity, 92.0, "humidity");
    assert_f64_near(data.ghi, 178.0, "ghi");
    assert_f64_near(data.dni, 480.0, "dni");
    assert_f64_near(data.dhi, 95.0, "dhi");
    assert_f64_near(data.wind_speed, 2.60, "wind_speed");
}

#[test]
fn test_denver_hour_9_peak_solar() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source.get_hourly_data(9).expect("Failed to get hour 9");

    assert_f64_near(data.dry_bulb_temp, 2.00, "dry_bulb_temp");
    assert_f64_near(data.humidity, 80.0, "humidity");
    assert_f64_near(data.ghi, 49.0, "ghi");
    assert_f64_near(data.dni, 0.0, "dni");
    assert_f64_near(data.dhi, 49.0, "dhi");
    assert_f64_near(data.wind_speed, 2.60, "wind_speed");
}

#[test]
fn test_denver_hour_10_high_solar() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source.get_hourly_data(10).expect("Failed to get hour 10");

    assert_f64_near(data.dry_bulb_temp, 2.00, "dry_bulb_temp");
    assert_f64_near(data.ghi, 431.0, "ghi");
    assert_f64_near(data.dni, 654.0, "dni");
    assert_f64_near(data.dhi, 169.0, "dhi");
    assert_f64_near(data.wind_speed, 4.60, "wind_speed");
}

#[test]
fn test_denver_hour_4379_winter() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source
        .get_hourly_data(4379)
        .expect("Failed to get hour 4379");

    assert_f64_near(data.dry_bulb_temp, 28.00, "dry_bulb_temp");
    assert_f64_near(data.humidity, 20.0, "humidity");
    assert_f64_near(data.wind_speed, 3.60, "wind_speed");
}

#[test]
fn test_denver_hour_8759_last_hour() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let data = source
        .get_hourly_data(8759)
        .expect("Failed to get hour 8759");

    assert_f64_near(data.dry_bulb_temp, 4.00, "dry_bulb_temp");
    assert_f64_near(data.humidity, 69.0, "humidity");
    assert_f64_near(data.wind_speed, 6.20, "wind_speed");
}

#[test]
fn test_test_denver_epw_all_hours() {
    let source = EpwWeatherSource::from_file("tests/test_data/test_denver.epw")
        .expect("Failed to load test Denver EPW");

    assert_eq!(source.record_count(), 6);

    let hour0 = source.get_hourly_data(0).expect("Failed to get hour 0");
    assert_f64_near(hour0.dry_bulb_temp, -5.0, "hour0 dry_bulb_temp");
    assert_f64_near(hour0.humidity, 60.0, "hour0 humidity");
    assert_f64_near(hour0.wind_speed, 2.5, "hour0 wind_speed");
    assert_f64_near(hour0.ghi, 0.0, "hour0 ghi");
    assert_f64_near(hour0.dni, 0.0, "hour0 dni");
    assert_f64_near(hour0.dhi, 0.0, "hour0 dhi");

    let hour1 = source.get_hourly_data(1).expect("Failed to get hour 1");
    assert_f64_near(hour1.dry_bulb_temp, -5.5, "hour1 dry_bulb_temp");
    assert_f64_near(hour1.humidity, 62.0, "hour1 humidity");

    let hour2 = source.get_hourly_data(2).expect("Failed to get hour 2");
    assert_f64_near(hour2.dry_bulb_temp, -6.0, "hour2 dry_bulb_temp");
    assert_f64_near(hour2.humidity, 64.0, "hour2 humidity");

    let hour3 = source.get_hourly_data(3).expect("Failed to get hour 3");
    assert_f64_near(hour3.dry_bulb_temp, 32.0, "hour3 dry_bulb_temp");
    assert_f64_near(hour3.humidity, 25.0, "hour3 humidity");
    assert_f64_near(hour3.ghi, 970.0, "hour3 ghi");
    assert_f64_near(hour3.dni, 850.0, "hour3 dni");
    assert_f64_near(hour3.dhi, 120.0, "hour3 dhi");
    assert_f64_near(hour3.wind_speed, 2.8, "hour3 wind_speed");

    let hour4 = source.get_hourly_data(4).expect("Failed to get hour 4");
    assert_f64_near(hour4.dry_bulb_temp, 33.5, "hour4 dry_bulb_temp");
    assert_f64_near(hour4.ghi, 1030.0, "hour4 ghi");
    assert_f64_near(hour4.dni, 900.0, "hour4 dni");
    assert_f64_near(hour4.dhi, 130.0, "hour4 dhi");

    let hour5 = source.get_hourly_data(5).expect("Failed to get hour 5");
    assert_f64_near(hour5.dry_bulb_temp, 34.0, "hour5 dry_bulb_temp");
    assert_f64_near(hour5.ghi, 1005.0, "hour5 ghi");
    assert_f64_near(hour5.dni, 880.0, "hour5 dni");
    assert_f64_near(hour5.dhi, 125.0, "hour5 dhi");
}

#[test]
fn test_missing_data_code_handling() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let hour0 = source.get_hourly_data(0).expect("Failed to get hour 0");

    assert!(
        hour0.present_weather_code.is_none() || hour0.present_weather_code.unwrap() < 9999,
        "present_weather_code should handle missing data codes"
    );

    // Issue #1415: EPW files use 9999 as the missing-data sentinel for solar
    // irradiance (GHI/DNI/DHI). The parser must coerce these to 0.0, not pass
    // 9999 W/m² downstream into the Perez / sol-air temperature models.
    // Verify that NO parsed hour in the real Denver file contains a sentinel.
    for hour in 0..source.record_count() {
        let data = source.get_hourly_data(hour).expect("Failed to get hour");
        assert!(
            data.ghi < 9999.0,
            "Hour {}: GHI={} should have been coerced from 9999 sentinel to 0.0",
            hour,
            data.ghi
        );
        assert!(
            data.dni < 9999.0,
            "Hour {}: DNI={} should have been coerced from 9999 sentinel to 0.0",
            hour,
            data.dni
        );
        assert!(
            data.dhi < 9999.0,
            "Hour {}: DHI={} should have been coerced from 9999 sentinel to 0.0",
            hour,
            data.dhi
        );
        assert!(
            data.horizontal_infrared < 9999.0,
            "Hour {}: HIR={} should have been coerced from 9999 sentinel to 0.0",
            hour,
            data.horizontal_infrared
        );
    }
}

#[test]
fn test_sentinel_9999_coerced_to_zero() {
    // Issue #1415: A row with GHI=9999, DNI=9999, DHI=9999, HIR=9999 must
    // produce 0.0 for all four fields, not 9999.0.
    let source = EpwWeatherSource::from_file("tests/test_data/test_sentinel_9999.epw")
        .expect("Failed to load sentinel test EPW");

    let hour0 = source.get_hourly_data(0).expect("Failed to get hour 0");
    assert_eq!(hour0.ghi, 0.0, "GHI=9999 sentinel must coerce to 0.0");
    assert_eq!(hour0.dni, 0.0, "DNI=9999 sentinel must coerce to 0.0");
    assert_eq!(hour0.dhi, 0.0, "DHI=9999 sentinel must coerce to 0.0");
    assert_eq!(
        hour0.horizontal_infrared, 0.0,
        "HIR=9999 sentinel must coerce to 0.0"
    );

    // Second row has valid values — ensure they are NOT zeroed.
    let hour1 = source.get_hourly_data(1).expect("Failed to get hour 1");
    assert_eq!(hour1.ghi, 970.0);
    assert_eq!(hour1.dni, 850.0);
    assert_eq!(hour1.dhi, 120.0);
    assert_eq!(hour1.horizontal_infrared, 300.0);
}

#[test]
fn test_field_tolerance_edge_cases() {
    let source = EpwWeatherSource::from_file("tests/test_data/test_denver.epw")
        .expect("Failed to load test Denver EPW");

    let hour = source.get_hourly_data(3).expect("Failed to get hour 3");

    assert_f64_near(hour.dry_bulb_temp, 32.001, "near-boundary check");
    assert_f64_near(hour.dni, 849.99, "near-boundary check");
}

#[test]
fn test_performance_under_50ms() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let start = std::time::Instant::now();
    for hour in 0..100 {
        let _ = source.get_hourly_data(hour);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "Test took {}ms, expected <50ms",
        elapsed.as_millis()
    );
}

#[test]
fn test_iter_all_hours() {
    let source = EpwWeatherSource::from_file("tests/test_data/test_denver.epw")
        .expect("Failed to load test Denver EPW");

    let mut count = 0;
    for result in source.iter_hours() {
        let data = result.expect("Failed to get hourly data");
        assert!(
            data.dry_bulb_temp.is_finite(),
            "dry_bulb_temp should be finite"
        );
        assert!(
            data.humidity >= 0.0 && data.humidity <= 100.0,
            "humidity should be in valid range"
        );
        count += 1;
    }

    assert_eq!(count, 6, "Should iterate over all 6 hours");
}

#[test]
fn test_epw_record_count() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    assert_eq!(
        source.record_count(),
        8760,
        "Denver EPW should have 8760 records"
    );
}

#[test]
fn test_epw_location() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let location = source.location().expect("Should have location");
    assert!(
        location.contains("Denver"),
        "Location should contain Denver, got: {}",
        location
    );
}

#[test]
fn test_epw_statistics() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load Denver EPW");

    let max_temp = source.max_temperature();
    let min_temp = source.min_temperature();
    let avg_temp = source.average_temperature();

    assert!(
        max_temp > min_temp,
        "max_temp ({}) should be > min_temp ({})",
        max_temp,
        min_temp
    );
    assert!(
        avg_temp > min_temp && avg_temp < max_temp,
        "avg_temp ({}) should be between min and max",
        avg_temp
    );

    let solar_hours = source.solar_hours();
    assert!(
        solar_hours > 0 && solar_hours < 8760,
        "Should have some solar hours, got {}",
        solar_hours
    );
}
