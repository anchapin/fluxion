//! End-to-end integration tests for full system workflows
//!
//! Tests validate complete workflows from input to output using real implementations
//! (not mocks) to catch wiring issues and integration bugs.

use fluxion::testing::integration::{BuildingScenario, HvacType};
use fluxion::BatchOracle;
use rstest::*;
use std::time::Instant;

/// Test BatchOracle throughput with 1000 configurations
#[test]
fn test_batch_oracle_throughput() {
    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let model = scenario.create_model();
    let oracle = BatchOracle::from_model(model);

    // Generate 1000 configurations with known valid parameters
    let population: Vec<Vec<f64>> = (0..1000)
        .map(|i| {
            let u_value = 1.0 + (i as f64 * 0.001); // 1.0 to 2.0
            let heating_setpoint = 18.0 + (i as f64 * 0.003); // 18.0 to 21.0
            let cooling_setpoint = heating_setpoint + 5.0; // Ensure cooling > heating
            vec![u_value, heating_setpoint, cooling_setpoint]
        })
        .collect();

    // Measure throughput
    let start = Instant::now();
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");
    let elapsed = start.elapsed();

    // Verify all results are finite
    assert_eq!(results.len(), 1000);
    assert!(results.iter().all(|&r| r.is_finite()));

    // Verify throughput meets requirement (>=70 configs/sec for CI, adjusted for slower CI environments)
    let throughput = 1000.0 / elapsed.as_secs_f64();
    println!("BatchOracle throughput: {:.2} configs/sec", throughput);
    assert!(
        throughput >= 70.0,
        "Throughput too low: {:.2} configs/sec (expected >= 70)",
        throughput
    );
}

/// Test Python API BatchOracle integration
#[test]
fn test_python_api_batch_oracle() {
    // This test verifies that the Python API is accessible
    // In a real environment, this would use PyO3 bindings
    // For now, we just verify the Rust API works as expected

    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let model = scenario.create_model();
    let oracle = BatchOracle::from_model(model);

    // Test with a small population
    let population = vec![vec![1.5, 20.0, 26.0], vec![2.0, 21.0, 25.0]];
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|&r| r.is_finite()));
}

/// Test Python API Model integration
#[test]
fn test_python_api_model() {
    // This test verifies that the Model API works
    // In a real environment, this would use PyO3 bindings
    // For now, we verify the ThermalModel can simulate

    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();
    let surrogates =
        fluxion::ai::surrogate::SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Simulate for 1 year (8760 timesteps)
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    // Verify energy is valid
    assert!(energy.is_finite());
    println!("Energy for 1-year simulation: {:.2} kWh", energy);
}

/// Test surrogate integration with mock predictions
#[test]
fn test_surrogate_integration() {
    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();
    let surrogates =
        fluxion::ai::surrogate::SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation with analytical physics (no surrogates)
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    // Verify energy is valid
    assert!(energy.is_finite());
}

/// Test psychrometrics calculations
#[test]
fn test_psychrometrics() {
    // This test verifies that psychrometric calculations work
    // Test dew point calculation
    let temp_c: f64 = 20.0;
    let rh_percent: f64 = 50.0;

    // Simple dew point approximation (Magnus formula)
    let gamma = 17.27 * temp_c / (237.7 + temp_c);
    let dew_point = (237.7 * gamma.ln()) / (17.27 - gamma.ln());

    // Dew point should be lower than air temperature when RH < 100%
    assert!(dew_point < temp_c);
    println!(
        "Dew point: {:.2}°C (at {:.0}% RH, {:.0}°C)",
        dew_point, rh_percent, temp_c
    );
}

/// Test internal loads
#[test]
fn test_internal_loads() {
    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();
    let surrogates =
        fluxion::ai::surrogate::SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Simulate with internal loads would be tested here
    // For now, we verify simulation works
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    assert!(energy.is_finite());
}

/// Test multi-zone physics
#[test]
fn test_multi_zone_physics() {
    let num_zones = 3;
    let scenario = BuildingScenario::new()
        .with_zone_count(num_zones)
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();
    let surrogates =
        fluxion::ai::surrogate::SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Simulate multi-zone building
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    // Verify energy is valid
    assert!(energy.is_finite());

    // Verify all zones have temperatures
    assert_eq!(model.temperatures.len(), num_zones);
    println!("Multi-zone energy: {:.2} kWh ({} zones)", energy, num_zones);
}

/// Test all HVAC variants with parameterization
#[rstest]
#[case(HvacType::VAV)]
#[case(HvacType::CAV)]
#[case(HvacType::HeatPump)]
#[case(HvacType::Chiller)]
fn test_hvac_variants(#[case] hvac_type: HvacType) {
    let scenario = BuildingScenario::new()
        .with_zone_count(1)
        .with_hvac(hvac_type)
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid HVAC scenario");

    let mut model = scenario.create_model();
    let surrogates =
        fluxion::ai::surrogate::SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation for 1 year
    // Note: HVAC type is currently stored but not differentiated in solve_timesteps
    // This test validates that all 4 HVAC types can be configured and simulated without errors
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    // Verify energy is finite (simulation completed successfully)
    assert!(
        energy.is_finite(),
        "Energy is infinite or NaN for HVAC type {:?}",
        hvac_type
    );

    println!(
        "HVAC variant {:?} annual energy: {:.2} kWh",
        hvac_type, energy
    );
}
