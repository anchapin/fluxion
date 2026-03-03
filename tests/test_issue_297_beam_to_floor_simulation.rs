//! Simulation-level test for Issue #297: Geometric Solar Distribution (Beam-to-Floor Logic)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::{ThermalModel, ThermalModelType};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_beam_to_floor_simulation_impact() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Configure model for 5R1C
    model.thermal_model_type = ThermalModelType::FiveROneC;

    // Set 90% beam-to-floor fraction
    model.solar_beam_to_mass_fraction = 0.9;

    // Reduce coupling to mass to make the test more sensitive
    // If coupling is high, heat spreads quickly regardless of where it enters
    model.h_tr_ms = VectorField::from_scalar(10.0, 1);
    model.update_optimization_cache();

    // Define a situation with high beam solar gain
    let zone_area = model.zone_area.as_ref()[0];
    let beam_gain_watts = 1000.0;
    let diffuse_gain_watts = 0.0;

    // Manually set gains (normalized by area as expected by engine)
    model.solar_gains =
        VectorField::from_scalar((beam_gain_watts + diffuse_gain_watts) / zone_area, 1);

    // Run a single step physics
    // We want to see how much energy ends up in the mass node (Tm) vs surface node
    // In 5R1C, phi_m goes to the mass node and directly affects Tm update.

    let initial_tm = model.mass_temperatures.as_ref()[0];
    let outdoor_temp = 20.0;
    model.step_physics(0, outdoor_temp);

    let final_tm = model.mass_temperatures.as_ref()[0];
    let tm_increase_high_fraction = final_tm - initial_tm;

    // Now repeat with low beam-to-floor fraction (e.g. 0.1)
    let mut model_low = ThermalModel::<VectorField>::from_spec(&spec);
    model_low.thermal_model_type = ThermalModelType::FiveROneC;
    model_low.solar_beam_to_mass_fraction = 0.1;
    model_low.h_tr_ms = VectorField::from_scalar(10.0, 1);
    model_low.update_optimization_cache();

    model_low.solar_gains =
        VectorField::from_scalar((beam_gain_watts + diffuse_gain_watts) / zone_area, 1);

    model_low.step_physics(0, outdoor_temp);
    let final_tm_low = model_low.mass_temperatures.as_ref()[0];
    let tm_increase_low_fraction = final_tm_low - initial_tm;

    println!(
        "TM Increase (90% fraction): {:.4} K",
        tm_increase_high_fraction
    );
    println!(
        "TM Increase (10% fraction): {:.4} K",
        tm_increase_low_fraction
    );

    // With 90% fraction, the mass should heat up significantly more
    assert!(
        tm_increase_high_fraction > tm_increase_low_fraction * 2.0,
        "Mass temperature should increase much more when beam-to-floor fraction is high"
    );
}

#[test]
fn test_6r2c_beam_to_floor_simulation_impact() {
    let spec = ASHRAE140Case::Case900.spec(); // Use Case 900 for 6R2C (high mass)
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    model.thermal_model_type = ThermalModelType::SixRTwoC;
    model.solar_beam_to_mass_fraction = 0.9;

    let zone_area = model.zone_area.as_ref()[0];
    let beam_gain_watts = 2000.0;

    model.solar_gains = VectorField::from_scalar(beam_gain_watts / zone_area, 1);

    let initial_tm_env = model.envelope_mass_temperatures.as_ref()[0];
    let initial_tm_int = model.internal_mass_temperatures.as_ref()[0];

    model.step_physics(0, 20.0);

    let final_tm_env = model.envelope_mass_temperatures.as_ref()[0];
    let final_tm_int = model.internal_mass_temperatures.as_ref()[0];

    let increase_env = final_tm_env - initial_tm_env;
    let increase_int = final_tm_int - initial_tm_int;

    println!("6R2C Envelope Mass Increase: {:.4} K", increase_env);
    println!("6R2C Internal Mass Increase: {:.4} K", increase_int);

    // In 6R2C, beam radiation is split between envelope (70%) and internal (30%) mass
    assert!(increase_env > 0.0, "Envelope mass should heat up");
    assert!(increase_int > 0.0, "Internal mass should heat up");
}
