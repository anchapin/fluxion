//! Example demonstrating the 6R2C thermal model usage.

use fluxion::sim::engine::{ThermalModel, ThermalModelType};

fn main() {
    println!("=== 6R2C Thermal Model Demonstration ===\n");

    // Test 1: Default is 5R1C
    let model_5r1c = ThermalModel::new(1);
    assert_eq!(model_5r1c.thermal_model_type, ThermalModelType::FiveROneC);
    println!("Test 1 PASSED: Default model is 5R1C");

    // Test 2: Configure 6R2C
    let mut model_6r2c = ThermalModel::new(1);
    model_6r2c.configure_6r2c_model(0.75, 100.0);
    assert_eq!(model_6r2c.thermal_model_type, ThermalModelType::SixRTwoC);
    println!("Test 2 PASSED: 6R2C model configured");

    // Test 3: Check capacitance split
    let total_cap = model_6r2c.thermal_capacitance.as_ref()[0];
    let env_cap = model_6r2c.envelope_thermal_capacitance.as_ref()[0];
    let int_cap = model_6r2c.internal_thermal_capacitance.as_ref()[0];

    assert!((env_cap - total_cap * 0.75).abs() < 0.01);
    assert!((int_cap - total_cap * 0.25).abs() < 0.01);
    println!(
        "Test 3 PASSED: Capacitance split correctly (env: {:.2} J/K, int: {:.2} J/K)",
        env_cap, int_cap
    );

    // Test 4: Run simulation
    println!("\nRunning 100 timesteps...");
    for t in 0..100 {
        model_6r2c.step_physics(t, 20.0);
    }
    println!("Test 4 PASSED: 100 timesteps simulated successfully");

    // Test 5: Check temperatures are reasonable
    let temp = model_6r2c.temperatures.as_ref()[0];
    let env_mass = model_6r2c.envelope_mass_temperatures.as_ref()[0];
    let int_mass = model_6r2c.internal_mass_temperatures.as_ref()[0];

    assert!(temp.is_finite() && temp > -50.0 && temp < 100.0);
    assert!(env_mass.is_finite() && env_mass > -50.0 && env_mass < 100.0);
    assert!(int_mass.is_finite() && int_mass > -50.0 && int_mass < 100.0);
    println!("Test 5 PASSED: Temperatures are reasonable");
    println!("  Indoor temp: {:.2}°C", temp);
    println!("  Envelope mass: {:.2}°C", env_mass);
    println!("  Internal mass: {:.2}°C", int_mass);

    println!("\n=== All tests PASSED! ===");
}
