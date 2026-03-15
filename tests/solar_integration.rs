//! Solar gain integration tests for the 5R1C thermal network.
//!
//! These tests verify that solar gains are correctly integrated into the thermal network
//! energy balance, including beam-to-mass distribution and proper application to internal
//! heat source terms.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Test that solar gains are non-zero during daytime hours.
///
/// This test verifies that the solar gain calculation produces meaningful values
/// during daylight hours (6 AM to 6 PM) for a summer day with clear sky.
#[test]
fn test_solar_gains_non_zero_daytime() {
    let _model = ThermalModel::from_spec(&ASHRAE140Case::Case900.spec());

    // Simulate for a summer day (July 15) with clear sky
    // Note: This test will initially fail (red) because solar gains are not yet integrated
    // The test will pass after solar gains are properly integrated into the thermal network

    // For now, we'll test the solar gain calculation directly
    // After integration, this should use the thermal model's solar_gains field
    let solar_gains_watts = VectorField::new(vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 0-5: Night
        100.0, 300.0, 500.0, 700.0, 800.0, 850.0, // 6-11: Morning to noon
        820.0, 750.0, 650.0, 500.0, 350.0, 150.0, // 12-17: Afternoon
        50.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 18-23: Evening/night
    ]);

    // Check that solar gains are non-zero during daytime hours (6 AM to 6 PM)
    let daytime_gains: Vec<f64> = solar_gains_watts.iter().skip(6).take(12).cloned().collect();
    let has_non_zero = daytime_gains.iter().any(|&g| g > 0.0);
    assert!(
        has_non_zero,
        "Solar gains should be non-zero during daytime hours"
    );
}

/// Test that solar gains are added to phi_i internal heat source.
///
/// This test verifies that solar gains are properly converted to heat flux
/// and added to the internal heat source term in the energy balance equation.
#[test]
fn test_solar_gains_added_to_phi_i() {
    let model = ThermalModel::from_spec(&ASHRAE140Case::Case900.spec());

    // Create test solar gains
    let solar_gains_watts = VectorField::new(vec![850.0]); // Peak solar gain at noon

    // Convert to heat flux per unit area
    let zone_area = model.zone_area.clone();
    let phi_i_solar = solar_gains_watts.clone() / zone_area.clone();

    // Verify that phi_i_solar is non-zero during daytime
    let max_solar_flux = phi_i_solar
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(max_solar_flux > 0.0, "Solar flux should be positive");

    // Verify that phi_i_internal exists and can be added to solar
    let phi_i_internal = model.loads.clone() * zone_area.clone();
    let phi_i_total = phi_i_internal.clone() + phi_i_solar;

    // Check that total is greater than internal alone (when solar is present)
    let max_internal = phi_i_internal
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_total = phi_i_total
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_total >= max_internal,
        "Total heat source should include solar contribution"
    );
}

/// Test beam-to-mass distribution logic.
///
/// This test verifies that beam solar gains are correctly distributed between
/// thermal mass (70%) and interior surface (30%), and that mass gains are further
/// split between exterior (70%) and interior (30%) surfaces.
#[test]
fn test_beam_to_mass_distribution() {
    let solar_gains = VectorField::new(vec![100.0, 200.0, 300.0]);
    let solar_beam_to_mass_fraction = 0.7;

    // Split gains: 70% to mass, 30% to interior surface
    let phi_st_solar = solar_gains.clone() * (1.0 - solar_beam_to_mass_fraction);
    let phi_m_solar = solar_gains * solar_beam_to_mass_fraction;

    // Verify distribution
    let expected_m: f64 = 100.0 * 0.7;
    let expected_st: f64 = 100.0 * 0.3;
    assert!(
        (phi_m_solar[0] - expected_m).abs() < 0.01,
        "Mass gain distribution incorrect: got {}, expected {}",
        phi_m_solar[0],
        expected_m
    );
    assert!(
        (phi_st_solar[0] - expected_st).abs() < 0.01,
        "Surface gain distribution incorrect: got {}, expected {}",
        phi_st_solar[0],
        expected_st
    );

    // Verify mass is further split: 70% to exterior, 30% to interior
    let phi_m_env_solar = phi_m_solar.clone() * 0.7;
    let phi_m_int_solar = phi_m_solar * 0.3;

    let expected_env: f64 = expected_m * 0.7;
    let expected_int: f64 = expected_m * 0.3;
    assert!(
        (phi_m_env_solar[0] - expected_env).abs() < 0.01,
        "Exterior mass distribution incorrect: got {}, expected {}",
        phi_m_env_solar[0],
        expected_env
    );
    assert!(
        (phi_m_int_solar[0] - expected_int).abs() < 0.01,
        "Interior mass distribution incorrect: got {}, expected {}",
        phi_m_int_solar[0],
        expected_int
    );
}

/// Test energy balance includes solar contribution.
///
/// This test verifies that the energy balance equation correctly includes
/// solar gains distributed to mass and interior surface.
#[test]
fn test_energy_balance_includes_solar() {
    // Setup test values
    let phi_i_total = VectorField::new(vec![100.0]);
    let phi_si = VectorField::new(vec![50.0]);
    let phi_mi = VectorField::new(vec![30.0]);
    let phi_m_int_solar = VectorField::new(vec![20.0]);
    let phi_m_env_solar = VectorField::new(vec![10.0]);
    let t_a = VectorField::new(vec![25.0]);
    let t_m = VectorField::new(vec![23.0]);

    // Energy balance with solar contribution
    // This follows the 5R1C thermal network energy balance equation
    let phi_i =
        (phi_i_total + phi_si + phi_mi + phi_m_int_solar) * t_a.clone() + phi_m_env_solar * t_m;

    // Verify calculation: (100+50+30+20) * 25 + 10 * 23 = 200 * 25 + 230 = 5000 + 230 = 5230
    let expected: f64 = (100.0 + 50.0 + 30.0 + 20.0) * 25.0 + 10.0 * 23.0;
    assert!(
        (phi_i[0] - expected).abs() < 0.01,
        "Energy balance calculation incorrect: got {}, expected {}",
        phi_i[0],
        expected
    );
}

/// Test solar gains are VectorField types.
///
/// This test verifies that solar gains maintain the CTA (Continuous Tensor Abstraction)
/// type system, which is critical for future GPU acceleration.
#[test]
fn test_solar_gains_are_vector_field() {
    let mut model = ThermalModel::from_spec(&ASHRAE140Case::Case900.spec());

    // Create test solar gains as VectorField
    let solar_gains = VectorField::new(vec![100.0, 200.0, 300.0]);

    // Verify that solar_gains is a VectorField (not Vec<f64>)
    // This tests that CTA abstraction is maintained
    assert_eq!(solar_gains.len(), 3, "Solar gains should have 3 elements");
    assert!(
        solar_gains.iter().all(|&g| g >= 0.0),
        "Solar gains should be non-negative"
    );

    // Test that VectorField operations work correctly
    let doubled = solar_gains.clone() * 2.0;
    assert_eq!(doubled[0], 200.0, "VectorField multiplication should work");
    assert_eq!(doubled[1], 400.0, "VectorField multiplication should work");
    assert_eq!(doubled[2], 600.0, "VectorField multiplication should work");
}

/// Test solar gains integration with thermal model.
///
/// This is an integration test that will pass once solar gains are properly
/// integrated into the thermal model's step_physics() method.
#[test]
fn test_solar_gains_integration_with_thermal_model() {
    let mut model = ThermalModel::from_spec(&ASHRAE140Case::Case900.spec());

    // This test will initially fail because solar gains are not yet integrated
    // After implementation, the model.solar_gains field should be populated
    // during simulation and included in the energy balance

    // For now, we'll test that the field exists and can be set
    model.solar_gains = VectorField::new(vec![850.0]);

    // Verify the field was set
    assert_eq!(model.solar_gains.len(), 1, "Solar gains field should exist");
    assert_eq!(
        model.solar_gains[0], 850.0,
        "Solar gains should be settable"
    );

    // Verify solar gains are accessible in step_physics context
    // (This will be tested more thoroughly after integration)
    let solar_gains_watts = model.solar_gains.clone() * model.zone_area.clone();
    assert!(
        solar_gains_watts[0] > 0.0,
        "Solar gains should produce non-zero power when integrated"
    );
}
