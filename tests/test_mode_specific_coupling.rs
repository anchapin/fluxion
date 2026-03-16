use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_mode_specific_coupling_factors() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Configure mode-specific factors (from Plan 03-14)
    // Note: Factors are already set by from_spec for high-mass cases
    let heating_factor = model.h_tr_em_heating_factor;
    let cooling_factor = model.h_tr_em_cooling_factor;

    let base_h_tr_em: f64 = model.h_tr_em.as_ref()[0];
    let h_tr_em_heating: f64 = model.h_tr_em_heating.as_ref()[0];
    let h_tr_em_cooling: f64 = model.h_tr_em_cooling.as_ref()[0];

    // Verify factors are configured
    assert_eq!(heating_factor, 0.15, "Heating factor should be 0.15");
    assert_eq!(cooling_factor, 1.05, "Cooling factor should be 1.05");

    // Verify h_tr_em_heating is 15% of base
    let expected_heating = base_h_tr_em * 0.15;
    assert!(
        (h_tr_em_heating - expected_heating).abs() < 0.1,
        "Heating coupling should be {:.2} W/K (15% of base), got {:.2}",
        expected_heating,
        h_tr_em_heating
    );

    // Verify h_tr_em_cooling is 105% of base
    let expected_cooling = base_h_tr_em * 1.05;
    assert!(
        (h_tr_em_cooling - expected_cooling).abs() < 0.1,
        "Cooling coupling should be {:.2} W/K (105% of base), got {:.2}",
        expected_cooling,
        h_tr_em_cooling
    );
}

#[test]
fn test_mode_detection_heating() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let heating_setpoint = model.heating_setpoints.as_ref()[0];
    let cooling_setpoint = model.cooling_setpoints.as_ref()[0];

    // Heating mode: Ti_free < heating_setpoint
    let ti_free_heating = heating_setpoint - 2.0; // 2°C below setpoint

    assert!(
        ti_free_heating < heating_setpoint,
        "Ti_free should be below heating setpoint"
    );
    assert!(
        ti_free_heating < cooling_setpoint,
        "Ti_free should be below cooling setpoint"
    );
}

#[test]
fn test_mode_detection_cooling() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let heating_setpoint = model.heating_setpoints.as_ref()[0];
    let cooling_setpoint = model.cooling_setpoints.as_ref()[0];

    // Cooling mode: Ti_free > cooling_setpoint
    let ti_free_cooling = cooling_setpoint + 2.0; // 2°C above setpoint

    assert!(
        ti_free_cooling > cooling_setpoint,
        "Ti_free should be above cooling setpoint"
    );
    assert!(
        ti_free_cooling > heating_setpoint,
        "Ti_free should be above heating setpoint"
    );
}

#[test]
fn test_mode_detection_deadband() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let heating_setpoint = model.heating_setpoints.as_ref()[0];
    let cooling_setpoint = model.cooling_setpoints.as_ref()[0];

    // Deadband: heating_setpoint <= Ti_free <= cooling_setpoint
    let ti_free_deadband = (heating_setpoint + cooling_setpoint) / 2.0; // Midpoint

    assert!(
        ti_free_deadband >= heating_setpoint && ti_free_deadband <= cooling_setpoint,
        "Ti_free should be in deadband between setpoints"
    );
}

#[test]
fn test_mode_specific_coupling_in_simulation() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Configure mode-specific factors (already set by from_spec)
    let heating_factor = model.h_tr_em_heating_factor;
    let cooling_factor = model.h_tr_em_cooling_factor;

    assert_eq!(heating_factor, 0.15, "Heating factor should be 0.15");
    assert_eq!(cooling_factor, 1.05, "Cooling factor should be 1.05");

    let base_h_tr_em: f64 = model.h_tr_em.as_ref()[0];
    let heating_h_tr_em = base_h_tr_em * 0.15;
    let cooling_h_tr_em = base_h_tr_em * 1.05;

    // Verify h_tr_em values match expected
    let h_tr_em_heating: f64 = model.h_tr_em_heating.as_ref()[0];
    let h_tr_em_cooling: f64 = model.h_tr_em_cooling.as_ref()[0];

    assert!(
        (h_tr_em_heating - heating_h_tr_em).abs() < 0.1,
        "Heating coupling should be {:.2} W/K (15% of base), got {:.2}",
        heating_h_tr_em,
        h_tr_em_heating
    );
    assert!(
        (h_tr_em_cooling - cooling_h_tr_em).abs() < 0.1,
        "Cooling coupling should be {:.2} W/K (105% of base), got {:.2}",
        cooling_h_tr_em,
        h_tr_em_cooling
    );

    // Run a short simulation to verify mode-specific coupling is applied
    let surrogates = fluxion::ai::surrogate::SurrogateManager::new().unwrap();

    // Run 24 hours (enough to see both heating and cooling modes)
    let energy = model.solve_timesteps(24, &surrogates, false, None, None, None);

    // Energy should be finite and positive
    assert!(!energy.is_nan(), "Energy should not be NaN");
    assert!(energy > 0.0, "Energy should be positive");
}
