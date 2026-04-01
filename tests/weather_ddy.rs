//! Design Day (DDY) weather file parser tests for src/weather/ddy.rs
//!
//! Tests pure logic functions that don't require external files.

use fluxion::weather::ddy::{generate_design_day_hours, DesignDaySource, DesignDaySpec};

#[test]
fn test_design_day_source_new() {
    let ddy = DesignDaySource::new();
    assert!(ddy.location.is_none());
    assert!(ddy.heating_design().is_none());
    assert!(ddy.cooling_design().is_none());
}

#[test]
fn test_days_in_month_indirect() {
    // Test days_in_month indirectly through generate_design_day_hours
    // January design day should have correct hour_of_year calculations
    let spec = DesignDaySpec {
        name: "January".to_string(),
        month: 1,
        day_of_month: 15,
        max_temp: 0.0,
        temp_range: 5.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };
    let hours = generate_design_day_hours(&spec);
    assert_eq!(hours.len(), 24);
    // January 15 = day 14 (0-indexed), hour 0 = 14*24 = 336
    assert_eq!(hours[0].hour_of_year, 336);
}

#[test]
fn test_cumulative_days_indirect() {
    // Test cumulative_days indirectly through hour_of_year calculation
    let july_spec = DesignDaySpec {
        name: "July".to_string(),
        month: 7,
        day_of_month: 21,
        max_temp: 35.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };
    let hours = generate_design_day_hours(&july_spec);
    // July 21 = day 201 (0-indexed), hour 0 = 201*24 = 4824
    assert_eq!(hours[0].hour_of_year, 4824);
}

#[test]
fn test_generate_heating_design_day() {
    let spec = DesignDaySpec {
        name: "Heating Design".to_string(),
        month: 12,
        day_of_month: 21,
        max_temp: -18.6,
        temp_range: 0.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: Some(-18.6),
        humidity_type: Some("Wetbulb".to_string()),
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    assert_eq!(hours.len(), 24);

    // Heating design should have cold temperatures
    let avg_temp = hours.iter().map(|h| h.dry_bulb_temp).sum::<f64>() / 24.0;
    assert!(avg_temp < -15.0, "Heating design should be cold");

    // Heating design should have no solar
    assert!(
        hours.iter().all(|h| h.dni == 0.0),
        "Heating design should have no solar"
    );
}

#[test]
fn test_generate_cooling_design_day() {
    let spec = DesignDaySpec {
        name: "Cooling Design".to_string(),
        month: 7,
        day_of_month: 21,
        max_temp: 35.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: Some(18.0),
        humidity_type: Some("Wetbulb".to_string()),
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    assert_eq!(hours.len(), 24);

    // Cooling design should have hot temperatures
    let max_design_temp = hours
        .iter()
        .map(|h| h.dry_bulb_temp)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    assert!(
        (max_design_temp - 35.0).abs() < 0.1,
        "Cooling design max temp should match spec"
    );

    // Cooling design should have solar (at least midday)
    let has_solar = hours.iter().any(|h| h.dni > 0.0 || h.ghi > 0.0);
    assert!(has_solar, "Cooling design should have solar");
}

#[test]
fn test_generate_design_day_temperature_variation() {
    let spec = DesignDaySpec {
        name: "Test".to_string(),
        month: 6,
        day_of_month: 15,
        max_temp: 30.0,
        temp_range: 15.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    // Temperature should vary throughout the day
    let min_temp = hours
        .iter()
        .map(|h| h.dry_bulb_temp)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_temp = hours
        .iter()
        .map(|h| h.dry_bulb_temp)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    // Range should be approximately equal to temp_range
    let actual_range = max_temp - min_temp;
    assert!(
        (actual_range - 15.0).abs() < 1.0,
        "Temperature range should be close to spec"
    );
}

#[test]
fn test_generate_design_day_solar_pattern() {
    let spec = DesignDaySpec {
        name: "Test".to_string(),
        month: 6,
        day_of_month: 15,
        max_temp: 30.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    // Solar should peak around midday
    let midday_solar = hours[12].dni;
    let morning_solar = hours[6].dni;
    let evening_solar = hours[18].dni;
    let night_solar = hours[0].dni;

    assert!(
        midday_solar > morning_solar,
        "Midday solar should be higher than morning"
    );
    assert!(
        midday_solar > evening_solar,
        "Midday solar should be higher than evening"
    );
    assert!(night_solar == 0.0, "Night solar should be zero");
}

#[test]
fn test_design_day_spec_clone() {
    let spec = DesignDaySpec {
        name: "Test".to_string(),
        month: 6,
        day_of_month: 15,
        max_temp: 30.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: Some(18.0),
        humidity_type: Some("Wetbulb".to_string()),
        humidity_ratio: Some(0.01),
        enthalpy: Some(50000.0),
    };

    let cloned = spec.clone();
    assert_eq!(cloned.name, spec.name);
    assert_eq!(cloned.month, spec.month);
    assert_eq!(cloned.day_of_month, spec.day_of_month);
    assert_eq!(cloned.max_temp, spec.max_temp);
    assert_eq!(cloned.temp_range, spec.temp_range);
    assert_eq!(cloned.day_type, spec.day_type);
    assert_eq!(cloned.wetbulb, spec.wetbulb);
    assert_eq!(cloned.humidity_type, spec.humidity_type);
    assert_eq!(cloned.humidity_ratio, spec.humidity_ratio);
    assert_eq!(cloned.enthalpy, spec.enthalpy);
}

#[test]
fn test_design_day_spec_debug() {
    let spec = DesignDaySpec {
        name: "Test".to_string(),
        month: 6,
        day_of_month: 15,
        max_temp: 30.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let debug_str = format!("{:?}", spec);
    assert!(debug_str.contains("Test"));
    assert!(debug_str.contains("SummerDesignDay"));
}

#[test]
fn test_design_day_source_debug() {
    let ddy = DesignDaySource::new();
    let debug_str = format!("{:?}", ddy);
    assert!(debug_str.contains("DesignDaySource"));
}

#[test]
fn test_generate_design_day_winter_heating_type() {
    let spec = DesignDaySpec {
        name: "Winter Htg".to_string(),
        month: 1,
        day_of_month: 15,
        max_temp: -10.0,
        temp_range: 5.0,
        day_type: "Htg".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    // Should have no solar (heating design)
    assert!(hours.iter().all(|h| h.dni == 0.0));
    assert!(hours.iter().all(|h| h.ghi == 0.0));
}

#[test]
fn test_generate_design_day_summer_cooling_type() {
    let spec = DesignDaySpec {
        name: "Summer Clg".to_string(),
        month: 7,
        day_of_month: 15,
        max_temp: 35.0,
        temp_range: 10.0,
        day_type: "Clg".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    // Should have solar (cooling design)
    let has_solar = hours.iter().any(|h| h.dni > 0.0);
    assert!(has_solar);
}

#[test]
fn test_generate_design_day_hourly_data_fields() {
    let spec = DesignDaySpec {
        name: "Test".to_string(),
        month: 6,
        day_of_month: 15,
        max_temp: 30.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    for hour in &hours {
        // Wind speed should be default (2.0 m/s)
        assert!((hour.wind_speed - 2.0).abs() < 0.01);
        // Humidity should be default (50%)
        assert!((hour.humidity - 50.0).abs() < 0.01);
    }
}

#[test]
fn test_generate_design_day_hour_of_year() {
    let spec = DesignDaySpec {
        name: "January".to_string(),
        month: 1,
        day_of_month: 15,
        max_temp: 0.0,
        temp_range: 5.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    // January 15 = day 14 (0-indexed)
    // Hour 0 should be 14 * 24 + 0 = 336
    assert_eq!(hours[0].hour_of_year, 336);
    // Hour 23 should be 14 * 24 + 23 = 359
    assert_eq!(hours[23].hour_of_year, 359);
}

#[test]
fn test_generate_design_day_hour_of_year_july() {
    let spec = DesignDaySpec {
        name: "July".to_string(),
        month: 7,
        day_of_month: 21,
        max_temp: 35.0,
        temp_range: 10.0,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };

    let hours = generate_design_day_hours(&spec);

    // July 21 = day 31+28+31+30+31+30+20 = 201 (0-indexed)
    // Hour 0 should be 201 * 24 = 4824
    assert_eq!(hours[0].hour_of_year, 4824);
}
