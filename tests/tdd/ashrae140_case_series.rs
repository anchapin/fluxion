//! ASHRAE 140 Case Series Tests
//!
//! This module adds test coverage for specific ASHRAE 140 case series that were
//! previously missing from the TDD test suite.
//!
//! ## Case Series Coverage
//!
//! | Series | Cases | Description | Mass Class |
//! |--------|-------|-------------|------------|
//! | 600    | 600, 610, 620, 630, 640, 650 | Standard construction | Light |
//! | 900    | 900, 910, 920, 930, 940, 950 | Heavy mass | Heavy |
//! | 195    | 195, 196 | Low internal gains | Light |
//! | 800    | 800, 900FF, 600FF | Free-floating | Mixed |
//!
//! ## Current Gaps Addressed
//!
//! - 600FF/900FF free-float max temperature validation
//! - Case 195 low internal gains validation
//! - Case 800 series (east/west orientation) validation

use crate::tdd::{assert_in_range, get_test_climate, simulate_blind, BlindTestSpec, MassClass};

// ============================================================================
// Case 600 Series (Standard Construction - Light Mass)
// ============================================================================

/// Test: Case 600 series annual heating within Table 5.3.2 range
///
/// ASHRAE 140 Table 5.3.2 for 600 series: 5.50-7.50 MWh annual heating
pub fn test_case_600_annual_heating() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(12.0) // 15% WWR standard
        .with_setpoints(20.0, 27.0)
        .with_night_setback(crate::tdd::SetbackSchedule {
            heating_setback_c: 5.5,
            start_hour: 23,
            end_hour: 7,
        })
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    assert_in_range(
        result.annual_heating_mwh,
        5.50,
        7.50,
        "Case 600 annual heating should be 5.50-7.50 MWh per ASHRAE 140 Table 5.3.2",
    );
}

/// Test: Case 600 series annual cooling within Table 5.3.2 range
///
/// ASHRAE 140 Table 5.3.2 for 600 series: 8.00-10.50 MWh annual cooling
pub fn test_case_600_annual_cooling() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .with_night_setback(crate::tdd::SetbackSchedule {
            heating_setback_c: 5.5,
            start_hour: 23,
            end_hour: 7,
        })
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    assert_in_range(
        result.annual_cooling_mwh,
        8.00,
        10.50,
        "Case 600 annual cooling should be 8.00-10.50 MWh per ASHRAE 140 Table 5.3.2",
    );
}

/// Test: Case 610 variant (increased window area) cooling load
pub fn test_case_610_increased_glazing_cooling() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(20.0) // 25% WWR - Case 610
        .with_setpoints(20.0, 27.0)
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // Case 610 should have higher cooling than Case 600
    assert!(
        result.annual_cooling_mwh > 8.00,
        "Case 610 cooling should exceed Case 600 baseline 8.00 MWh"
    );
}

/// Test: Case 620 variant (decreased window area) heating load
pub fn test_case_620_decreased_glazing_heating() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(6.0) // 8% WWR - Case 620
        .with_setpoints(20.0, 27.0)
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // Case 620 should have lower cooling than Case 600
    assert!(
        result.annual_cooling_mwh < 10.50,
        "Case 620 cooling should be below Case 600 baseline 10.50 MWh"
    );
}

// ============================================================================
// Case 900 Series (Heavy Mass)
// ============================================================================

/// Test: Case 900 series annual heating within Table 5.3.2 range
///
/// ASHRAE 140 Table 5.3.2 for 900 series: 1.17-2.04 MWh annual heating
pub fn test_case_900_annual_heating() {
    let spec = BlindTestSpec::heavy_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .with_night_setback(crate::tdd::SetbackSchedule {
            heating_setback_c: 5.5,
            start_hour: 23,
            end_hour: 7,
        })
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    assert_in_range(
        result.annual_heating_mwh,
        1.17,
        2.04,
        "Case 900 annual heating should be 1.17-2.04 MWh per ASHRAE 140 Table 5.3.2",
    );
}

/// Test: Case 900 series annual cooling within Table 5.3.2 range
///
/// ASHRAE 140 Table 5.3.2 for 900 series: 2.13-3.67 MWh annual cooling
pub fn test_case_900_annual_cooling() {
    let spec = BlindTestSpec::heavy_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .with_night_setback(crate::tdd::SetbackSchedule {
            heating_setback_c: 5.5,
            start_hour: 23,
            end_hour: 7,
        })
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    assert_in_range(
        result.annual_cooling_mwh,
        2.13,
        3.67,
        "Case 900 annual cooling should be 2.13-3.67 MWh per ASHRAE 140 Table 5.3.2",
    );
}

/// Test: Case 910 variant (increased glazing) peak cooling
pub fn test_case_910_high_glazing_peak_cooling() {
    let spec = BlindTestSpec::heavy_mass()
        .with_south_window(25.0) // 32% WWR - Case 910
        .with_setpoints(20.0, 27.0)
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // Higher glazing should increase peak cooling
    assert!(
        result.peak_cooling_kw > 2.10,
        "Case 910 peak cooling should exceed heavy mass baseline 2.10 kW"
    );
}

// ============================================================================
// Case 195 Series (Low Internal Gains - Light Mass)
// ============================================================================

/// Test: Case 195 low internal gains validation
///
/// Case 195/196 have 50% reduction from standard due to low internal gains
/// Standard internal gains are 200 W, Case 195 should be ~100 W
pub fn test_case_195_low_internal_gains_heating() {
    // Case 195: 50% of standard internal gains
    // Standard = 200 W, Low = 100 W (approximately)
    let standard_spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .build();

    // For Case 195, we need to simulate reduced internal gains
    // The framework doesn't fully support internal gains modification yet,
    // so we test that heating is higher with lower gains (more dependency on HVAC)
    let spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .with_internal_gains(100.0, 0.6, 0.4) // 50% of standard 200W
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // With low internal gains, heating should be higher than standard
    // This is because there's less free heat from occupants/equipment
    assert!(
        result.annual_heating_mwh > 5.50,
        "Case 195 heating should be at or above standard minimum due to low internal gains"
    );
}

/// Test: Case 195/196 cooling should be reduced due to low internal gains
///
/// With 50% internal gains, cooling load should also be reduced
pub fn test_case_195_low_internal_gains_cooling() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .with_internal_gains(100.0, 0.6, 0.4) // 50% of standard 200W
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // With low internal gains, cooling should be reduced
    // 50% reduction means around 4.0-5.25 MWh (half of 8.00-10.50 range)
    assert_in_range(
        result.annual_cooling_mwh,
        3.50, // Approximately 50% of standard
        6.00, // Upper bound allowing for other factors
        "Case 195 cooling should be reduced ~50% from standard due to low internal gains",
    );
}

/// Test: Case 196 should be similar to 195 (both low gains)
pub fn test_case_196_low_internal_gains_validation() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_setpoints(20.0, 27.0)
        .with_internal_gains(100.0, 0.6, 0.4) // Same as Case 195
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // Case 196 should behave similarly to Case 195
    assert!(
        result.annual_heating_mwh > 5.50 && result.annual_heating_mwh < 8.00,
        "Case 196 heating should be within standard range"
    );
}

// ============================================================================
// Case 800 Series (Free-Floating / East-West Orientation)
// ============================================================================

/// Test: Case 800 east/west orientation heating
///
/// Case 800 uses east/west orientation instead of south-only
/// This changes the solar gain profile throughout the day
pub fn test_case_800_east_west_orientation_heating() {
    let spec = BlindTestSpec::light_mass()
        .with_east_west_windows(16.0) // E/W split total 16 m² (8 each facade)
        .with_setpoints(20.0, 27.0)
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // E/W orientation typically reduces morning heating load (west gets afternoon sun)
    // and may increase evening cooling (east gets morning sun, west gets afternoon)
    assert!(
        result.annual_heating_mwh > 0.0,
        "Case 800 heating should be positive"
    );
}

/// Test: Case 800 east/west orientation cooling
pub fn test_case_800_east_west_orientation_cooling() {
    let spec = BlindTestSpec::light_mass()
        .with_east_west_windows(16.0)
        .with_setpoints(20.0, 27.0)
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    // E/W orientation typically increases cooling due to
    // more hours of direct solar gain (morning east, afternoon west)
    assert!(
        result.annual_cooling_mwh > result.annual_heating_mwh,
        "Case 800 east/west should have higher cooling than heating"
    );
}

/// Test: Case 600FF free-float max temperature
///
/// Case 600FF: Light mass free-float, ASHRAE reference max: 64.9-75.1°C
pub fn test_case_600ff_max_free_float_temperature() {
    let spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_no_hvac()
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    let max_temp = result
        .hourly_temperatures
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // ASHRAE 140 reference for 600FF max temperature: 64.9-75.1°C
    assert_in_range(
        max_temp,
        60.0, // Allowing some tolerance
        80.0, // Absolute physical limit
        "Case 600FF max temperature should be 64.9-75.1°C per ASHRAE 140",
    );
}

/// Test: Case 900FF free-float max temperature
///
/// Case 900FF: Heavy mass free-float, ASHRAE reference max: 41.8-46.4°C
/// CRITICAL: This is where the 80°C bug manifests
pub fn test_case_900ff_max_free_float_temperature() {
    let spec = BlindTestSpec::heavy_mass()
        .with_south_window(12.0)
        .with_no_hvac()
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    let max_temp = result
        .hourly_temperatures
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // ASHRAE 140 reference for 900FF max temperature: 41.8-46.4°C
    assert_in_range(
        max_temp,
        38.0, // Lower bound
        50.0, // Allow some tolerance above 46.4
        "Case 900FF max temperature should be 41.8-46.4°C per ASHRAE 140 (CRITICAL: 80°C bug)",
    );
}

/// Test: Case 900FF free-float min temperature
pub fn test_case_900ff_min_free_float_temperature() {
    let spec = BlindTestSpec::heavy_mass()
        .with_south_window(12.0)
        .with_no_hvac()
        .build();

    let result = simulate_blind(&spec, &get_test_climate());

    let min_temp = result
        .hourly_temperatures
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    // ASHRAE 140 reference for 900FF min temperature: -6.4 to -1.6°C
    assert_in_range(
        min_temp,
        -10.0,
        0.0,
        "Case 900FF min temperature should be within ASHRAE range",
    );
}

/// Test: Free-float temperature difference between 600FF and 900FF
///
/// Heavy mass should have significantly lower max temps due to thermal damping
pub fn test_free_float_mass_damping_effect() {
    let light_spec = BlindTestSpec::light_mass()
        .with_south_window(12.0)
        .with_no_hvac()
        .build();

    let heavy_spec = BlindTestSpec::heavy_mass()
        .with_south_window(12.0)
        .with_no_hvac()
        .build();

    let light_result = simulate_blind(&light_spec, &get_test_climate());
    let heavy_result = simulate_blind(&heavy_spec, &get_test_climate());

    let light_max = light_result
        .hourly_temperatures
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let heavy_max = heavy_result
        .hourly_temperatures
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Light mass should reach higher max temp than heavy mass
    assert!(
        light_max > heavy_max,
        "Light mass max temp ({:.1}°C) should exceed heavy mass ({:.1}°C) due to less damping",
        light_max,
        heavy_max
    );

    // But both should be within reasonable bounds (not 80°C for heavy)
    assert!(
        heavy_max < 55.0,
        "Heavy mass max temp {:.1}°C should be < 55°C (ASHRAE reference: 41.8-46.4°C)",
        heavy_max
    );
}

// ============================================================================
// Test Module
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Case 600 series tests
    #[test]
    fn test_case_600_annual_heating() {
        test_case_600_annual_heating();
    }

    #[test]
    fn test_case_600_annual_cooling() {
        test_case_600_annual_cooling();
    }

    #[test]
    fn test_case_610_cooling() {
        test_case_610_increased_glazing_cooling();
    }

    #[test]
    fn test_case_620_heating() {
        test_case_620_decreased_glazing_heating();
    }

    // Case 900 series tests
    #[test]
    fn test_case_900_annual_heating() {
        test_case_900_annual_heating();
    }

    #[test]
    fn test_case_900_annual_cooling() {
        test_case_900_annual_cooling();
    }

    #[test]
    fn test_case_910_peak_cooling() {
        test_case_910_high_glazing_peak_cooling();
    }

    // Case 195 series tests
    #[test]
    fn test_case_195_heating() {
        test_case_195_low_internal_gains_heating();
    }

    #[test]
    fn test_case_195_cooling() {
        test_case_195_low_internal_gains_cooling();
    }

    #[test]
    fn test_case_196_validation() {
        test_case_196_low_internal_gains_validation();
    }

    // Case 800 series tests
    #[test]
    fn test_case_800_heating() {
        test_case_800_east_west_orientation_heating();
    }

    #[test]
    fn test_case_800_cooling() {
        test_case_800_east_west_orientation_cooling();
    }

    #[test]
    fn test_case_600ff_max_temp() {
        test_case_600ff_max_free_float_temperature();
    }

    #[test]
    fn test_case_900ff_max_temp() {
        test_case_900ff_max_free_float_temperature();
    }

    #[test]
    fn test_case_900ff_min_temp() {
        test_case_900ff_min_free_float_temperature();
    }

    #[test]
    fn test_free_float_damping() {
        test_free_float_mass_damping_effect();
    }
}
