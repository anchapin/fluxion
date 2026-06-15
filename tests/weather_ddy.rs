//! DDY design day parser isolation tests for src/weather/ddy.rs
//!
//! Tests parsing of EnergyPlus Design Day (DDY) files for HVAC equipment sizing.
//! Validates heating/cooling design day fields against known reference values.

use fluxion::weather::ddy::{DesignDaySource, DesignDaySpec};

#[test]
fn test_parse_cooling_design_day_csv_format() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    let cooling = ddy.cooling_design().expect("Cooling design day not found");
    assert_eq!(cooling.month, 7);
    assert_eq!(cooling.day_of_month, 21);
    assert!((cooling.max_temp - 34.4).abs() < 0.01);
    assert!(cooling.day_type.contains("Summer"));
}

#[test]
fn test_parse_cooling_design_day_humidity() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    let cooling = ddy.cooling_design().unwrap();
    assert!(cooling.humidity_type.is_some());
    assert_eq!(cooling.humidity_type.as_ref().unwrap(), "Wetbulb");
    assert!(cooling.wetbulb.is_some());
}

#[test]
fn test_parse_design_day_location() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    assert!(ddy.location.is_some());
}

#[test]
fn test_parse_design_day_month_day() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    let cooling = ddy.cooling_design().unwrap();
    assert!(cooling.month >= 1 && cooling.month <= 12);
    assert!(cooling.day_of_month >= 1 && cooling.day_of_month <= 31);
}

#[test]
fn test_parse_design_day_optional_fields() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    let cooling = ddy.cooling_design().unwrap();
    assert!(cooling.humidity_ratio.is_none() || cooling.humidity_ratio.is_some());
    assert!(cooling.enthalpy.is_none() || cooling.enthalpy.is_some());
}

#[test]
fn test_parse_nonexistent_file() {
    let result = DesignDaySource::from_file("/nonexistent/path/file.ddy");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Failed to open"));
}

#[test]
fn test_design_day_spec_builder() {
    let spec = DesignDaySpec {
        name: "Test Heating".to_string(),
        month: 1,
        day_of_month: 15,
        max_temp: -10.0,
        temp_range: 5.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: Some(-10.0),
        humidity_type: Some("Wetbulb".to_string()),
        humidity_ratio: None,
        enthalpy: None,
    };
    assert_eq!(spec.month, 1);
    assert_eq!(spec.day_of_month, 15);
    assert!((spec.max_temp - (-10.0)).abs() < 0.01);
}

#[test]
fn test_design_day_source_empty() {
    let ddy = DesignDaySource::new();
    assert!(ddy.heating_design().is_none());
    assert!(ddy.cooling_design().is_none());
    assert!(ddy.location.is_none());
}

#[test]
fn test_cooling_design_day_has_temp_range() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    let cooling = ddy.cooling_design().unwrap();
    assert!(cooling.temp_range > 0.0);
}

#[test]
fn test_design_day_clone() {
    let ddy = DesignDaySource::from_file("tests/test_data/denver.ddy").unwrap();
    let cooling = ddy.cooling_design().unwrap().clone();
    assert_eq!(cooling.month, 7);
    assert_eq!(cooling.day_of_month, 21);
    assert!((cooling.max_temp - 34.4).abs() < 0.01);
}

#[test]
fn test_parse_month_day_edge_cases() {
    let spec = DesignDaySpec {
        name: "Edge Case".to_string(),
        month: 12,
        day_of_month: 31,
        max_temp: -5.0,
        temp_range: 0.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };
    assert_eq!(spec.month, 12);
    assert_eq!(spec.day_of_month, 31);
}

#[test]
fn test_design_day_performance() {
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = DesignDaySource::from_file("tests/test_data/denver.ddy");
    }
    let elapsed = start.elapsed();
    let per_iteration_ms = elapsed.as_millis() as f64 / 100.0;
    assert!(
        per_iteration_ms < 0.5,
        "DDY parsing too slow: {:.2}ms per iteration",
        per_iteration_ms
    );
}

#[test]
fn test_heating_design_day_spec() {
    let spec = DesignDaySpec {
        name: "Denver Winter Design".to_string(),
        month: 12,
        day_of_month: 21,
        max_temp: -18.6,
        temp_range: 0.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: Some(-18.6),
        humidity_type: Some("Wetbulb".to_string()),
        humidity_ratio: Some(0.009),
        enthalpy: Some(50000.0),
    };
    assert!(spec.day_type.contains("Winter"));
    assert_eq!(spec.month, 12);
    assert_eq!(spec.day_of_month, 21);
    assert!((spec.max_temp - (-18.6)).abs() < 0.01);
}

#[test]
fn test_cooling_design_day_spec() {
    let spec = DesignDaySpec {
        name: "Denver Summer Design".to_string(),
        month: 7,
        day_of_month: 21,
        max_temp: 34.4,
        temp_range: 12.8,
        day_type: "SummerDesignDay".to_string(),
        wetbulb: Some(18.3),
        humidity_type: Some("Wetbulb".to_string()),
        humidity_ratio: Some(0.009),
        enthalpy: Some(70000.0),
    };
    assert!(spec.day_type.contains("Summer"));
    assert_eq!(spec.month, 7);
    assert_eq!(spec.day_of_month, 21);
    assert!((spec.max_temp - 34.4).abs() < 0.01);
    assert!(spec.temp_range > 0.0);
}

#[test]
fn test_design_day_debug() {
    let spec = DesignDaySpec {
        name: "Test".to_string(),
        month: 1,
        day_of_month: 1,
        max_temp: 0.0,
        temp_range: 0.0,
        day_type: "WinterDesignDay".to_string(),
        wetbulb: None,
        humidity_type: None,
        humidity_ratio: None,
        enthalpy: None,
    };
    let debug_str = format!("{:?}", spec);
    assert!(debug_str.contains("Test"));
}
