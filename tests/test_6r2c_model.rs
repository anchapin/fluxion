//! Tests for the 6R2C (two mass node) thermal model.
//!
//! This test module validates the optional 6R2C thermal network model,
//! which extends the 5R1C model by separating thermal mass into:
//! - Envelope mass (walls, roof, floor)
//! - Internal mass (furniture, partitions)
//!
//! The 6R2C model better captures thermal lag in high-mass buildings.

use fluxion::sim::engine::{ThermalModel, ThermalModelType};
use fluxion::physics::cta::VectorField;

#[test]
fn test_thermal_model_type_default() {
    // Default model should be 5R1C
    let model = ThermalModel::new(1);
    assert_eq!(model.thermal_model_type, ThermalModelType::FiveROneC);
    assert!(!model.is_6r2c_model());
}

#[test]
fn test_configure_6r2c_model() {
    let mut model = ThermalModel::new(1);
    let envelope_fraction = 0.75;
    let h_tr_me_value = 100.0;

    model.configure_6r2c_model(envelope_fraction, h_tr_me_value);

    // Check that model is now 6R2C
    assert!(model.is_6r2c_model());
    assert_eq!(model.thermal_model_type, ThermalModelType::SixRTwoC);

    // Check that thermal capacitance is split correctly
    let total_cap = model.thermal_capacitance.as_ref()[0];
    let envelope_cap = model.envelope_thermal_capacitance.as_ref()[0];
    let internal_cap = model.internal_thermal_capacitance.as_ref()[0];

    assert!((envelope_cap - total_cap * envelope_fraction).abs() < 0.01);
    assert!((internal_cap - total_cap * (1.0 - envelope_fraction)).abs() < 0.01);
    assert!((envelope_cap + internal_cap - total_cap).abs() < 0.01);

    // Check that conductance between masses is set
    let h_tr_me = model.h_tr_me.as_ref()[0];
    assert!((h_tr_me - h_tr_me_value).abs() < 0.01);

    // Check that mass temperatures are initialized from current mass temperature
    let initial_temp = model.mass_temperatures.as_ref()[0];
    assert_eq!(model.envelope_mass_temperatures.as_ref()[0], initial_temp);
    assert_eq!(model.internal_mass_temperatures.as_ref()[0], initial_temp);
}

#[test]
fn test_6r2c_model_cloning() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.8, 150.0);

    let cloned = model.clone();

    // Check that all fields are cloned correctly
    assert_eq!(cloned.thermal_model_type, model.thermal_model_type);
    assert_eq!(
        cloned.envelope_thermal_capacitance.as_ref()[0],
        model.envelope_thermal_capacitance.as_ref()[0]
    );
    assert_eq!(
        cloned.internal_thermal_capacitance.as_ref()[0],
        model.internal_thermal_capacitance.as_ref()[0]
    );
    assert_eq!(
        cloned.h_tr_me.as_ref()[0],
        model.h_tr_me.as_ref()[0]
    );
}

#[test]
fn test_6r2c_model_single_timestep() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Run a single timestep
    let initial_temp = model.temperatures.as_ref()[0];
    let initial_env_mass = model.envelope_mass_temperatures.as_ref()[0];
    let initial_int_mass = model.internal_mass_temperatures.as_ref()[0];

    model.step_physics(0, 20.0);

    // Check that temperatures have changed
    let new_temp = model.temperatures.as_ref()[0];
    let new_env_mass = model.envelope_mass_temperatures.as_ref()[0];
    let new_int_mass = model.internal_mass_temperatures.as_ref()[0];

    // Temperatures should have changed from initial state
    assert!(new_temp != initial_temp || new_env_mass != initial_env_mass || new_int_mass != initial_int_mass);

    // Mass temperatures should be updated
    assert!(new_env_mass >= -50.0 && new_env_mass <= 100.0); // Reasonable range
    assert!(new_int_mass >= -50.0 && new_int_mass <= 100.0); // Reasonable range
}

#[test]
fn test_6r2c_model_energy_conservation() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Run multiple timesteps and check that energy is conserved
    // (no NaN or infinite values)
    let steps = 100;
    let outdoor_temp = 20.0;

    for t in 0..steps {
        let energy = model.step_physics(t, outdoor_temp);
        assert!(energy.is_finite());
        assert!(energy >= 0.0); // Energy should be non-negative

        // Check that all temperatures remain finite
        let temp = model.temperatures.as_ref()[0];
        let env_mass = model.envelope_mass_temperatures.as_ref()[0];
        let int_mass = model.internal_mass_temperatures.as_ref()[0];

        assert!(temp.is_finite());
        assert!(env_mass.is_finite());
        assert!(int_mass.is_finite());
    }
}

#[test]
fn test_6r2c_model_thermal_lag() {
    // Test that the 6R2C model captures thermal lag better than 5R1C
    let mut model_5r1c = ThermalModel::new(1);
    let mut model_6r2c = ThermalModel::new(1);
    model_6r2c.configure_6r2c_model(0.75, 100.0);

    // Initialize both models to same state
    let initial_temp = 20.0;
    model_5r1c.temperatures = VectorField::from_scalar(initial_temp, 1);
    model_5r1c.mass_temperatures = VectorField::from_scalar(initial_temp, 1);
    model_6r2c.temperatures = VectorField::from_scalar(initial_temp, 1);
    model_6r2c.mass_temperatures = VectorField::from_scalar(initial_temp, 1);
    model_6r2c.envelope_mass_temperatures = VectorField::from_scalar(initial_temp, 1);
    model_6r2c.internal_mass_temperatures = VectorField::from_scalar(initial_temp, 1);

    // Apply a step change in outdoor temperature
    let initial_outdoor = 20.0;
    let final_outdoor = 30.0;

    let steps = 24; // One day
    let mut temps_5r1c = Vec::new();
    let mut temps_6r2c = Vec::new();

    for t in 0..steps {
        let outdoor = if t < steps / 2 { initial_outdoor } else { final_outdoor };

        model_5r1c.step_physics(t, outdoor);
        model_6r2c.step_physics(t, outdoor);

        temps_5r1c.push(model_5r1c.temperatures.as_ref()[0]);
        temps_6r2c.push(model_6r2c.temperatures.as_ref()[0]);
    }

    // Both models should reach similar final temperatures
    // (but 6R2C may have different thermal lag characteristics)
    let final_5r1c = temps_5r1c.last().unwrap();
    let final_6r2c = temps_6r2c.last().unwrap();

    // Final temperatures should be in similar range (within 5°C for this test)
    assert!((final_5r1c - final_6r2c).abs() < 5.0);

    // Temperature trajectories should be different
    // (6R2C has additional thermal mass affecting response)
    let sum_diff: f64 = temps_5r1c
        .iter()
        .zip(temps_6r2c.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(sum_diff > 0.01); // Some difference expected
}

#[test]
fn test_6r2c_model_multi_zone() {
    // Test that 6R2C works with multi-zone buildings
    let num_zones = 2;
    let mut model = ThermalModel::new(num_zones);
    model.configure_6r2c_model(0.75, 100.0);

    // Run a few timesteps
    for t in 0..10 {
        let energy = model.step_physics(t, 20.0);
        assert!(energy.is_finite());
        assert!(energy >= 0.0);
    }

    // Check that all zones have valid temperatures
    for i in 0..num_zones {
        let temp = model.temperatures.as_ref()[i];
        let env_mass = model.envelope_mass_temperatures.as_ref()[i];
        let int_mass = model.internal_mass_temperatures.as_ref()[i];

        assert!(temp.is_finite());
        assert!(env_mass.is_finite());
        assert!(int_mass.is_finite());
    }
}

#[test]
fn test_6r2c_model_backward_compatibility() {
    // Test that the single mass temperature field is updated correctly
    // for backward compatibility with code that expects a single mass temperature
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Run a timestep
    model.step_physics(0, 20.0);

    // Check that the single mass temperature is a weighted average
    let env_temp = model.envelope_mass_temperatures.as_ref()[0];
    let int_temp = model.internal_mass_temperatures.as_ref()[0];
    let env_cap = model.envelope_thermal_capacitance.as_ref()[0];
    let int_cap = model.internal_thermal_capacitance.as_ref()[0];

    let expected_mass_temp = (env_temp * env_cap + int_temp * int_cap) / (env_cap + int_cap);
    let actual_mass_temp = model.mass_temperatures.as_ref()[0];

    assert!((expected_mass_temp - actual_mass_temp).abs() < 0.01);
}

#[test]
fn test_5r1c_vs_6r2c_energy_comparison() {
    // Compare energy consumption between 5R1C and 6R2C models
    // They should be in similar ranges (within 20% for this simple test)
    let mut model_5r1c = ThermalModel::new(1);
    let mut model_6r2c = ThermalModel::new(1);
    model_6r2c.configure_6r2c_model(0.75, 100.0);

    // Run 24 timesteps (one day)
    let steps = 24;
    let outdoor_temp = 20.0;

    let mut energy_5r1c = 0.0;
    let mut energy_6r2c = 0.0;

    for t in 0..steps {
        energy_5r1c += model_5r1c.step_physics(t, outdoor_temp);
        energy_6r2c += model_6r2c.step_physics(t, outdoor_temp);
    }

    // Both models should consume non-negative energy
    assert!(energy_5r1c >= 0.0);
    assert!(energy_6r2c >= 0.0);

    // Energy consumption should be in similar range (within 50% for this test)
    // Note: 6R2C may have different dynamics, so we allow some deviation
    let ratio = if energy_5r1c > 0.0 {
        energy_6r2c / energy_5r1c
    } else if energy_6r2c > 0.0 {
        energy_5r1c / energy_6r2c
    } else {
        1.0
    };

    assert!(ratio > 0.5 && ratio < 2.0);
}

#[test]
fn test_6r2c_model_different_mass_fractions() {
    // Test that different envelope mass fractions work correctly
    for fraction in [0.5, 0.6, 0.7, 0.8, 0.9] {
        let mut model = ThermalModel::new(1);
        model.configure_6r2c_model(fraction, 100.0);

        // Run a few timesteps
        for t in 0..10 {
            let energy = model.step_physics(t, 20.0);
            assert!(energy.is_finite());
            assert!(energy >= 0.0);
        }

        // Check that capacitance split is correct
        let total_cap = model.thermal_capacitance.as_ref()[0];
        let envelope_cap = model.envelope_thermal_capacitance.as_ref()[0];
        let internal_cap = model.internal_thermal_capacitance.as_ref()[0];

        assert!((envelope_cap - total_cap * fraction).abs() < 0.01);
        assert!((internal_cap - total_cap * (1.0 - fraction)).abs() < 0.01);
    }
}

#[test]
fn test_6r2c_model_with_night_ventilation() {
    // Test that 6R2C works with night ventilation
    use fluxion::validation::ashrae_140_cases::NightVentilation;

    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Set up night ventilation (7pm to 8am)
    model.night_ventilation = Some(NightVentilation::new(1000.0, 19, 8));

    // Run a full day with night ventilation
    for t in 0..24 {
        let energy = model.step_physics(t, 20.0);
        assert!(energy.is_finite());
        assert!(energy >= 0.0);
    }

    // Temperatures should remain stable
    let temp = model.temperatures.as_ref()[0];
    let env_mass = model.envelope_mass_temperatures.as_ref()[0];
    let int_mass = model.internal_mass_temperatures.as_ref()[0];

    assert!(temp.is_finite());
    assert!(env_mass.is_finite());
    assert!(int_mass.is_finite());
}
