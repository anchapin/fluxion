//! Weather module isolation tests: EPW parser vs EnergyPlus reference data.
//!
//! Validates that the Weather module correctly parses EPW files and produces
//! hourly weather data that matches EnergyPlus output within 1% tolerance.
//!
//! # Reference Data
//!
//! - Source: Denver TMY3 EPW (USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw)
//! - Generated: Using same EPW file with validated field indices (Issue #829 fix)
//! - Location: tests/reference_data/weather/denver_tmy3_reference.csv
//!
//! # What This Tests
//!
//! 1. **EPW Parsing**: Field indices match canonical EPW v3 format
//! 2. **WeatherSource Trait**: EpwWeatherSource implements WeatherSource correctly
//! 3. **Hourly Data**: All 8760 hours parsed with correct values
//! 4. **Psychrometrics**: Humidity ratio calculations match ASHRAE within 1%
//!
//! # Acceptance Criteria (Issue #1011)
//!
//! - [x] WeatherSource trait defined and implemented
//! - [x] EPW parser produces correct field values
//! - [x] All 8760 hourly records parsed without error
//! - [x] Temperature within 1% of E+ reference
//! - [x] Solar radiation (DNI, DHI, GHI) within 1% of E+ reference
//! - [x] Wind speed within 1% of E+ reference
//! - [x] Humidity ratio within 1% of psychrometric calculation
//! - [x] Tests pass: cargo test weather --quiet

use std::fs;
use std::path::Path;
use std::time::Instant;

use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::psychrometrics::{
    calculate_humidity_ratio, PsychrometricCalculations, STANDARD_ATMOSPHERIC_PRESSURE_Pa,
};
use fluxion::weather::{HourlyWeatherData, WeatherSource};

/// Reference data row from the EnergyPlus CSV output.
///
/// CSV columns: hour, dry_bulb_temp_c, humidity_rh_pct, dni_wm2, dhi_wm2, ghi_wm2, wind_speed_ms, humidity_ratio_kgkg
#[derive(Debug, Clone)]
struct ReferenceRow {
    #[allow(dead_code)]
    hour: usize,
    dry_bulb_temp_c: f64,
    humidity_rh_pct: f64,
    dni_wm2: f64,
    dhi_wm2: f64,
    ghi_wm2: f64,
    wind_speed_ms: f64,
    humidity_ratio_kgkg: f64,
}

/// Tolerance for percentage comparisons: 1%
const TOLERANCE_PCT: f64 = 1.0;

/// Load the E+ reference CSV file.
fn load_reference_data() -> Vec<ReferenceRow> {
    let path = Path::new("tests/reference_data/weather/denver_tmy3_reference.csv");
    let content =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));

    let mut rows = Vec::with_capacity(8760);
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 8 {
            continue;
        }
        if parts[0].contains("hour") {
            continue; // Skip header
        }
        rows.push(ReferenceRow {
            hour: parts[0].parse::<usize>().expect("valid hour"),
            dry_bulb_temp_c: parts[1].parse::<f64>().expect("valid dry_bulb_temp"),
            humidity_rh_pct: parts[2].parse::<f64>().expect("valid humidity"),
            dni_wm2: parts[3].parse::<f64>().expect("valid dni"),
            dhi_wm2: parts[4].parse::<f64>().expect("valid dhi"),
            ghi_wm2: parts[5].parse::<f64>().expect("valid ghi"),
            wind_speed_ms: parts[6].parse::<f64>().expect("valid wind_speed"),
            humidity_ratio_kgkg: parts[7].parse::<f64>().expect("valid humidity_ratio"),
        });
    }
    assert_eq!(rows.len(), 8760, "Expected exactly 8760 reference rows");
    rows
}

/// Calculate relative error percentage.
fn rel_error_pct(observed: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-10 {
        if observed.abs() < 1e-10 {
            return 0.0;
        }
        return 100.0;
    }
    ((observed - expected) / expected.abs() * 100.0).abs()
}

/// Check if value is within tolerance of expected value.
fn within_tolerance(observed: f64, expected: f64, tolerance_pct: f64) -> bool {
    rel_error_pct(observed, expected) <= tolerance_pct
}

// =============================================================================
// WEATHERSOURCE TRAIT TESTS
// =============================================================================

#[test]
fn test_weather_source_trait_implementation() {
    // Verify EpwWeatherSource implements WeatherSource trait
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Test location()
    let location = source.location();
    assert!(location.is_some(), "Location should be Some");
    assert!(
        location.unwrap().contains("Denver"),
        "Location should contain 'Denver'"
    );

    // Test get_hourly_data()
    let data = source.get_hourly_data(0).unwrap();
    assert_eq!(data.hour_of_year, 0, "Hour of year should be 0");

    // Test iter_hours() - should iterate all 8760 hours
    let count = source.iter_hours().count();
    assert_eq!(count, 8760, "Should iterate all 8760 hours");
}

#[test]
fn test_weather_source_invalid_hour() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Test out-of-range hour
    let result = source.get_hourly_data(8760);
    assert!(result.is_err(), "Hour 8760 should be invalid");

    // Test another out-of-range value
    let result = source.get_hourly_data(10000);
    assert!(result.is_err(), "Hour 10000 should be invalid");
}

// =============================================================================
// EPW PARSING TESTS - 1% TOLERANCE
// =============================================================================

#[test]
fn test_epw_parsing_full_year_8760_records() {
    let reference = load_reference_data();
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    assert_eq!(
        source.record_count(),
        8760,
        "Should parse exactly 8760 hourly records"
    );

    // Spot check first, middle, and last hours
    let check_hours = [0, 1, 2, 100, 500, 1000, 4000, 5000, 8000, 8759];

    for &hour in &check_hours {
        let data = source.get_hourly_data(hour).unwrap();
        let ref_row = &reference[hour];

        // Temperature
        assert!(
            within_tolerance(data.dry_bulb_temp, ref_row.dry_bulb_temp_c, TOLERANCE_PCT),
            "Hour {}: dry_bulb_temp {} != {} (ref)",
            hour,
            data.dry_bulb_temp,
            ref_row.dry_bulb_temp_c
        );

        // Humidity
        assert!(
            within_tolerance(data.humidity, ref_row.humidity_rh_pct, TOLERANCE_PCT),
            "Hour {}: humidity {} != {} (ref)",
            hour,
            data.humidity,
            ref_row.humidity_rh_pct
        );

        // Wind speed
        assert!(
            within_tolerance(data.wind_speed, ref_row.wind_speed_ms, TOLERANCE_PCT),
            "Hour {}: wind_speed {} != {} (ref)",
            hour,
            data.wind_speed,
            ref_row.wind_speed_ms
        );
    }
}

#[test]
fn test_epw_solar_radiation_parsing() {
    let reference = load_reference_data();
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Test hours with significant solar radiation (daytime hours)
    let solar_hours: Vec<usize> = (0..8760).filter(|&h| reference[h].ghi_wm2 > 10.0).collect();

    assert!(
        solar_hours.len() > 4000,
        "Should have >4000 daytime hours, got {}",
        solar_hours.len()
    );

    // Check a sample of daytime hours
    let sample_size = 100.min(solar_hours.len());
    let step = solar_hours.len() / sample_size;

    for i in (0..solar_hours.len()).step_by(step).take(sample_size) {
        let hour = solar_hours[i];
        let data = source.get_hourly_data(hour).unwrap();
        let ref_row = &reference[hour];

        // DNI
        assert!(
            within_tolerance(data.dni, ref_row.dni_wm2, TOLERANCE_PCT),
            "Hour {}: DNI {} != {} (ref)",
            hour,
            data.dni,
            ref_row.dni_wm2
        );

        // DHI
        assert!(
            within_tolerance(data.dhi, ref_row.dhi_wm2, TOLERANCE_PCT),
            "Hour {}: DHI {} != {} (ref)",
            hour,
            data.dhi,
            ref_row.dhi_wm2
        );

        // GHI
        assert!(
            within_tolerance(data.ghi, ref_row.ghi_wm2, TOLERANCE_PCT),
            "Hour {}: GHI {} != {} (ref)",
            hour,
            data.ghi,
            ref_row.ghi_wm2
        );
    }
}

#[test]
fn test_epw_temperature_range() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    let max_temp = source.max_temperature();
    let min_temp = source.min_temperature();
    let avg_temp = source.average_temperature();

    // Denver TMY3 temperature ranges
    assert!(
        max_temp > 30.0 && max_temp < 45.0,
        "Max temp should be 30-45°C, got {}°C",
        max_temp
    );
    assert!(
        min_temp < 0.0 && min_temp > -30.0,
        "Min temp should be -30-0°C, got {}°C",
        min_temp
    );
    assert!(
        avg_temp > 5.0 && avg_temp < 15.0,
        "Avg temp should be 5-15°C, got {}°C",
        avg_temp
    );
}

#[test]
fn test_epw_solar_hours_count() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();
    let solar_hours = source.solar_hours();

    // Denver should have significant solar hours (GHI > 0)
    assert!(
        solar_hours > 4000 && solar_hours <= 8760,
        "Solar hours should be 4000-8760, got {}",
        solar_hours
    );
}

// =============================================================================
// PSYCHROMETRICS TESTS - 1% TOLERANCE
// =============================================================================

#[test]
fn test_humidity_ratio_from_weather_data() {
    let reference = load_reference_data();
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Test a sample of hours across the year
    let sample_hours: Vec<usize> = (0..8760)
        .step_by(100) // Every 100th hour
        .collect();

    for &hour in &sample_hours {
        let data = source.get_hourly_data(hour).unwrap();
        let ref_row = &reference[hour];

        // Calculate humidity ratio from weather data
        let calculated_hr = calculate_humidity_ratio(
            data.dry_bulb_temp,
            data.humidity,
            STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        );

        // Compare with reference
        assert!(
            within_tolerance(calculated_hr, ref_row.humidity_ratio_kgkg, TOLERANCE_PCT),
            "Hour {}: humidity_ratio {} != {} (ref)",
            hour,
            calculated_hr,
            ref_row.humidity_ratio_kgkg
        );
    }
}

#[test]
fn test_humidity_ratio_calculation_accuracy() {
    // ASHRAE Handbook reference values at standard atmospheric pressure
    // 25°C, 50% RH => ω ≈ 0.0099 kg/kg
    let hr = calculate_humidity_ratio(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
    let expected = 0.0099;
    assert!(
        within_tolerance(hr, expected, TOLERANCE_PCT),
        "humidity_ratio(25°C, 50%) = {} kg/kg, expected ≈ {} kg/kg",
        hr,
        expected
    );

    // 30°C, 80% RH => ω ≈ 0.0217 kg/kg
    let hr = calculate_humidity_ratio(30.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
    let expected = 0.0217;
    assert!(
        within_tolerance(hr, expected, TOLERANCE_PCT),
        "humidity_ratio(30°C, 80%) = {} kg/kg, expected ≈ {} kg/kg",
        hr,
        expected
    );
}

#[test]
fn test_humidity_ratio_psychrometric_consistency() {
    // Verify that humidity ratio calculated from weather data is consistent
    // with the psychrometric formula
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Test all hours for consistency
    let mut max_error = 0.0;
    let mut max_error_hour = 0;

    for hour in 0..8760 {
        let data = source.get_hourly_data(hour).unwrap();
        let calculated_hr = calculate_humidity_ratio(
            data.dry_bulb_temp,
            data.humidity,
            STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        );

        // Use HourlyWeatherData trait method
        let trait_hr = data.humidity_ratio();

        let error = (calculated_hr - trait_hr).abs();
        if error > max_error {
            max_error = error;
            max_error_hour = hour;
        }
    }

    assert!(
        max_error < 1e-10,
        "Max humidity ratio error {} at hour {} too large",
        max_error,
        max_error_hour
    );
}

// =============================================================================
// HOURLYWEATHERDATA VALIDATION TESTS
// =============================================================================

#[test]
fn test_hourly_weather_data_validate_all() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Collect all weather data
    let weather_data: Vec<HourlyWeatherData> = source
        .iter_hours()
        .map(|r| r.expect("valid weather data"))
        .collect();

    // Validate all records
    let result = HourlyWeatherData::validate_all(&weather_data);
    assert!(
        result.is_ok(),
        "All weather data should be valid: {:?}",
        result
    );
}

#[test]
fn test_hourly_weather_data_is_complete() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    let weather_data: Vec<HourlyWeatherData> = source
        .iter_hours()
        .map(|r| r.expect("valid weather data"))
        .collect();

    assert!(
        HourlyWeatherData::is_complete(&weather_data),
        "All weather data should be complete"
    );
}

#[test]
fn test_hourly_weather_data_time_properties() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Test hour 0: January 1, 00:00
    let data = source.get_hourly_data(0).unwrap();
    assert_eq!(data.hour_of_day(), 0, "Hour 0 should be midnight");
    assert_eq!(data.day_of_year(), 0, "Hour 0 should be day 0");
    assert_eq!(data.month(), 1, "Hour 0 should be January");

    // Test hour 12: January 1, 12:00
    let data = source.get_hourly_data(12).unwrap();
    assert_eq!(data.hour_of_day(), 12, "Hour 12 should be noon");
    assert_eq!(data.day_of_year(), 0, "Hour 12 should still be day 0");

    // Test hour 744: February 1, 00:00
    let data = source.get_hourly_data(744).unwrap();
    assert_eq!(data.hour_of_day(), 0, "Hour 744 should be midnight");
    assert_eq!(data.day_of_year(), 31, "Hour 744 should be day 31");
    assert_eq!(data.month(), 2, "Hour 744 should be February");

    // Test hour 8759: December 31, 23:00
    let data = source.get_hourly_data(8759).unwrap();
    assert_eq!(data.hour_of_day(), 23, "Hour 8759 should be 23:00");
    assert_eq!(data.day_of_year(), 364, "Hour 8759 should be day 364");
    assert_eq!(data.month(), 12, "Hour 8759 should be December");
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_epw_nighttime_hours_have_zero_solar() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Find nighttime hours (0-5 and 20-23) in winter months
    let winter_night_hours: Vec<usize> = (0..8760)
        .filter(|&h| {
            let hour_of_day = h % 24;
            let day_of_year = h / 24;
            (hour_of_day < 6 || hour_of_day > 19) && day_of_year < 59 // Jan-Feb
        })
        .collect();

    // Check that nighttime hours have zero or near-zero solar
    for &hour in &winter_night_hours {
        let data = source.get_hourly_data(hour).unwrap();
        assert!(
            data.dni < 1.0 && data.dhi < 1.0 && data.ghi < 1.0,
            "Hour {} (nighttime winter) should have zero solar: DNI={}, DHI={}, GHI={}",
            hour,
            data.dni,
            data.dhi,
            data.ghi
        );
    }
}

#[test]
fn test_epw_daytime_hours_have_solar() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Find midday hours (10-14) in summer months
    let summer_midday_hours: Vec<usize> = (0..8760)
        .filter(|&h| {
            let hour_of_day = h % 24;
            let day_of_year = h / 24;
            (hour_of_day >= 10 && hour_of_day <= 14) && day_of_year >= 151 && day_of_year <= 243
            // Jun-Aug
        })
        .collect();

    // Most summer midday hours should have significant solar
    let solar_count = summer_midday_hours
        .iter()
        .filter(|&&h| {
            let data = source.get_hourly_data(h).unwrap();
            data.ghi > 100.0
        })
        .count();

    assert!(
        solar_count > summer_midday_hours.len() / 2,
        "Most summer midday hours should have GHI > 100 W/m²"
    );
}

#[test]
fn test_epw_wind_speed_non_negative() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    for hour in 0..8760 {
        let data = source.get_hourly_data(hour).unwrap();
        assert!(
            data.wind_speed >= 0.0,
            "Hour {}: wind_speed {} should be non-negative",
            hour,
            data.wind_speed
        );
    }
}

#[test]
fn test_epw_humidity_in_valid_range() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    for hour in 0..8760 {
        let data = source.get_hourly_data(hour).unwrap();
        assert!(
            data.humidity >= 0.0 && data.humidity <= 100.0,
            "Hour {}: humidity {} should be 0-100%",
            hour,
            data.humidity
        );
    }
}

// =============================================================================
// PERFORMANCE TEST
// =============================================================================

#[test]
fn test_epw_parsing_performance_under_500ms() {
    let start = Instant::now();
    let _source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "EPW parsing took {}ms (>500ms)",
        elapsed.as_millis()
    );
}

#[test]
fn test_hourly_data_access_performance_under_100ms() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    let start = Instant::now();
    for hour in 0..8760 {
        let _ = source.get_hourly_data(hour).unwrap();
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "8760 hourly accesses took {}ms (>100ms)",
        elapsed.as_millis()
    );
}

// =============================================================================
// STATISTICS TESTS
// =============================================================================

#[test]
fn test_epw_statistics_consistency() {
    let source = EpwWeatherSource::from_file("tests/test_data/denver.epw").unwrap();

    // Calculate statistics manually
    let mut sum_temp = 0.0;
    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;
    let mut solar_hours = 0;

    for hour in 0..8760 {
        let data = source.get_hourly_data(hour).unwrap();
        sum_temp += data.dry_bulb_temp;
        min_temp = min_temp.min(data.dry_bulb_temp);
        max_temp = max_temp.max(data.dry_bulb_temp);
        if data.ghi > 0.0 {
            solar_hours += 1;
        }
    }

    let avg_temp = sum_temp / 8760.0;

    // Compare with source methods
    assert!(
        (source.average_temperature() - avg_temp).abs() < 0.01,
        "Average temperature mismatch"
    );
    assert!(
        (source.min_temperature() - min_temp).abs() < 0.01,
        "Min temperature mismatch"
    );
    assert!(
        (source.max_temperature() - max_temp).abs() < 0.01,
        "Max temperature mismatch"
    );
    assert_eq!(
        source.solar_hours(),
        solar_hours,
        "Solar hours count mismatch"
    );
}
