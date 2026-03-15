//! Solar calculation validation tests for DNI/DHI calculations and window properties.
//!
//! These tests validate the solar radiation calculations and window properties
//! used in ASHRAE 140 test cases. This file provides test scaffolding for
//! Tasks 4-5 in plan 03-01.

use fluxion::sim::solar::{
    calculate_hourly_solar, calculate_solar_position, calculate_surface_irradiance,
    calculate_window_solar_gain, WindowProperties,
};
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, Orientation};

/// Helper function for variance calculation.
///
/// This function calculates the variance of a set of values, which is useful
/// for statistical validation of solar irradiance distributions.
fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    variance
}

/// Test that hourly solar irradiance calculations work for all orientations.
///
/// This test verifies that DNI (Direct Normal Irradiance) and DHI (Diffuse Horizontal Irradiance)
/// are correctly calculated and that different orientations produce different irradiance patterns.
#[test]
fn test_hourly_solar_irradiance_for_orientations() {
    let _window = WindowProperties::double_clear(12.0);

    // Test orientations: North, South, East, West
    let orientations = [
        Orientation::North,
        Orientation::South,
        Orientation::East,
        Orientation::West,
    ];

    let mut irradiance_values = Vec::new();

    for orientation in &orientations {
        let sun_pos = calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0);

        // Sample irradiance for noon on summer solstice
        let irradiance = calculate_surface_irradiance(
            &sun_pos,
            900.0, // DNI
            150.0, // DHI
            None,
            *orientation,
            0.2, // ground reflectance
            172, // day of year
        );

        irradiance_values.push(irradiance.total_wm2);
    }

    // Verify that different orientations produce different irradiance values
    let variance = calculate_variance(&irradiance_values);
    assert!(
        variance > 0.0,
        "Different orientations should produce different irradiance values"
    );

    // South-facing surface should have highest irradiance at noon in summer
    assert!(
        irradiance_values[1] > irradiance_values[0], // South > North
        "South-facing surface should have higher irradiance at noon"
    );
}

/// Test window SHGC values for all ASHRAE 140 cases.
///
/// This test verifies that window Solar Heat Gain Coefficient (SHGC) values
/// are correctly specified for all ASHRAE 140 test cases.
#[test]
fn test_window_shgc_ashrae_140_cases() {
    // Test key ASHRAE 140 cases
    let cases = [
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case610,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case640,
        ASHRAE140Case::Case650,
        ASHRAE140Case::Case900,
    ];

    for case in &cases {
        let spec = case.spec();

        // Verify that each case has windows defined
        assert!(
            !spec.windows.is_empty(),
            "{} should have windows",
            spec.case_id
        );

        // Verify window properties are reasonable
        let window_spec = &spec.window_properties;
        assert!(
            window_spec.shgc > 0.0 && window_spec.shgc <= 1.0,
            "{} has invalid SHGC: {}",
            spec.case_id,
            window_spec.shgc
        );
        assert!(
            window_spec.u_value > 0.0 && window_spec.u_value <= 10.0,
            "{} has invalid U-value: {}",
            spec.case_id,
            window_spec.u_value
        );
        assert!(
            window_spec.normal_transmittance > 0.0 && window_spec.normal_transmittance <= 1.0,
            "{} has invalid transmittance: {}",
            spec.case_id,
            window_spec.normal_transmittance
        );
    }
}

/// Test window normal transmittance values for ASHRAE 140 cases.
///
/// This test verifies that normal beam transmittance values are correctly
/// specified for window glazing systems in ASHRAE 140 test cases.
#[test]
fn test_window_normal_transmittance_ashrae_140_cases() {
    // Verify double clear glass typical values
    let window = WindowProperties::double_clear(12.0);

    // Typical values for double clear glass:
    // - SHGC: 0.789
    // - Normal transmittance: 0.86156
    assert!(
        (window.shgc - 0.789).abs() < 0.01,
        "SHGC should be ~0.789, got {}",
        window.shgc
    );
    assert!(
        (window.normal_transmittance - 0.86156).abs() < 0.01,
        "Normal transmittance should be ~0.86156, got {}",
        window.normal_transmittance
    );
}

/// Test that SHGC is applied correctly in solar gain calculations.
///
/// This test verifies that the Solar Heat Gain Coefficient is properly
/// applied to irradiance values when calculating window solar gains.
#[test]
fn test_shgc_applied_in_solar_gains() {
    let window = WindowProperties::double_clear(12.0);
    let sun_pos = calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0);

    // High irradiance at noon
    let irradiance = calculate_surface_irradiance(
        &sun_pos,
        900.0, // DNI
        150.0, // DHI
        None,
        Orientation::South,
        0.2,
        172,
    );

    // Calculate solar gain
    let gain = calculate_window_solar_gain(
        &irradiance,
        &window,
        None,
        None,
        &[],
        &sun_pos,
        Orientation::South,
    );

    // Verify that SHGC is applied: gain should be less than irradiance * area
    let max_possible = window.area * irradiance.total_wm2;
    assert!(
        gain.total_gain_w < max_possible,
        "Solar gain with SHGC should be less than irradiance * area"
    );

    // Verify gain is positive
    assert!(
        gain.total_gain_w > 0.0,
        "Solar gain should be positive for non-zero irradiance"
    );
}

/// Test solar position calculation accuracy.
///
/// This test verifies that solar position calculations produce reasonable
/// values for key times of year.
#[test]
fn test_solar_position_accuracy() {
    // Summer solstice (June 21) at solar noon
    let sun_pos_summer = calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0);
    assert!(
        sun_pos_summer.altitude_deg > 70.0,
        "Summer solstice altitude should be > 70°, got {}",
        sun_pos_summer.altitude_deg
    );
    assert!(
        (sun_pos_summer.azimuth_deg - 180.0).abs() < 10.0,
        "Summer solstice azimuth should be near 180° (South), got {}",
        sun_pos_summer.azimuth_deg
    );

    // Winter solstice (December 21) at solar noon
    let sun_pos_winter = calculate_solar_position(39.7, -104.9, 2024, 12, 21, 12.0);
    assert!(
        sun_pos_winter.altitude_deg > 20.0,
        "Winter solstice altitude should be > 20°, got {}",
        sun_pos_winter.altitude_deg
    );
    assert!(
        sun_pos_winter.altitude_deg < 30.0,
        "Winter solstice altitude should be < 30°, got {}",
        sun_pos_winter.altitude_deg
    );

    // Equinox (March 21) at solar noon
    let sun_pos_equinox = calculate_solar_position(39.7, -104.9, 2024, 3, 21, 12.0);
    let expected_altitude = 90.0 - 39.7; // 90° - latitude
    assert!(
        (sun_pos_equinox.altitude_deg - expected_altitude).abs() < 2.0,
        "Equinox altitude should be ~{:.1}°, got {:.1}°",
        expected_altitude,
        sun_pos_equinox.altitude_deg
    );
}

/// Test incidence angle calculation on different surfaces.
///
/// This test verifies that solar incidence angles are correctly calculated
/// for different surface orientations.
#[test]
fn test_incidence_angle_calculation() {
    let sun_pos = calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0);

    // South-facing vertical surface at solar noon should have incidence angle = altitude
    let cos_theta_south = sun_pos.incidence_cosine(90.0, 180.0);
    let incidence_angle_south = cos_theta_south.acos().to_degrees();
    assert!(
        (incidence_angle_south - sun_pos.altitude_deg).abs() < 2.0,
        "South surface incidence should equal altitude at noon: got {:.1}° vs {:.1}°",
        incidence_angle_south,
        sun_pos.altitude_deg
    );

    // Horizontal surface at solar noon should have incidence angle = zenith
    let cos_theta_horizontal = sun_pos.incidence_cosine(0.0, 0.0);
    let incidence_angle_horizontal = cos_theta_horizontal.acos().to_degrees();
    assert!(
        (incidence_angle_horizontal - sun_pos.zenith_deg).abs() < 2.0,
        "Horizontal surface incidence should equal zenith at noon: got {:.1}° vs {:.1}°",
        incidence_angle_horizontal,
        sun_pos.zenith_deg
    );
}

/// Test hourly solar calculation for a full day.
///
/// This test verifies that hourly solar calculations work correctly for
/// a full day cycle, with appropriate day/night behavior.
#[test]
fn test_hourly_solar_full_day() {
    let window = WindowProperties::double_clear(12.0);

    let mut total_gain = 0.0;
    let mut hours_with_gain = 0;

    for hour in 0..24 {
        let (_sun_pos, _irradiance, gain) = calculate_hourly_solar(
            39.7,
            -104.9,
            2024,
            6,
            21,
            hour as f64,
            900.0,
            150.0,
            &window,
            None,
            None,
            &[],
            Orientation::South,
            None,
        );

        total_gain += gain.total_gain_w;
        if gain.total_gain_w > 0.0 {
            hours_with_gain += 1;
        }
    }

    // Should have positive gain during daytime hours
    assert!(
        total_gain > 0.0,
        "Total solar gain for the day should be positive, got {} W",
        total_gain
    );

    // Should have gain for ~12-16 hours in summer at 39.7°N latitude
    assert!(
        hours_with_gain >= 12 && hours_with_gain <= 16,
        "Should have gain for 12-16 hours in summer, got {}",
        hours_with_gain
    );
}

/// Test solar gain at night (sun below horizon).
///
/// This test verifies that solar gains are zero when the sun is below horizon.
#[test]
fn test_solar_gain_at_night() {
    let window = WindowProperties::double_clear(12.0);

    // Midnight on June 21 (sun well below horizon)
    let (_sun_pos, irradiance, gain) = calculate_hourly_solar(
        39.7,
        -104.9,
        2024,
        6,
        21,
        0.0,
        900.0,
        150.0,
        &window,
        None,
        None,
        &[],
        Orientation::South,
        None,
    );

    // At night, solar gains should be zero
    assert_eq!(
        irradiance.total_wm2, 0.0,
        "Solar irradiance should be zero at night"
    );
    assert_eq!(gain.total_gain_w, 0.0, "Solar gain should be zero at night");
}
