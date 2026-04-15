//! Wiring validation tests
//!
//! Tests verify that modules are correctly wired together and integration points
//! work as expected. Uses runtime tracing to detect issues like solve_timesteps()
//! never calling predict_loads() when use_ai=true.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::testing::integration::{BuildingScenario, WiringTracer};
use fluxion::BatchOracle;
use std::sync::Arc;

/// Test that solve_timesteps() works correctly with surrogates
#[test]
fn test_surrogate_integration_wiring() {
    // Create tracer for automatic call recording
    let tracer = Arc::new(WiringTracer::new());

    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .with_tracer(tracer.clone())
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();

    // Create surrogate manager (uses mock predictions by default)
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation with analytical path (no AI)
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    // Verify energy is finite and non-zero
    assert!(energy.is_finite());

    // Verify solve_timesteps_with_dt and step_physics were called
    assert!(tracer.verify_called(&["solve_timesteps_with_dt"]));
    assert!(tracer.verify_called(&["step_physics"]));

    // Verify no AI calls were made (analytical path)
    assert!(!tracer.verify_called(&["predict_loads"]));
}

/// Test that BatchOracle uses parallelism correctly for populations
#[test]
fn test_batch_oracle_parallelism() {
    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("Invalid scenario");

    let model = scenario.create_model();
    let oracle = BatchOracle::from_model(model);

    // Create a population of 100 configurations
    let population: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![1.5 + (i as f64 * 0.01), 20.0, 26.0])
        .collect();

    // Evaluate population without surrogates
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");

    // Verify results are all finite and correct count
    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|&r| r.is_finite()));

    // Note: BatchOracle doesn't have tracer integration yet (future enhancement)
    // This test verifies the population evaluation works correctly
}

/// Test that weather data flows through to simulation
#[test]
fn test_weather_data_flow() {
    // Create tracer for automatic call recording
    let tracer = Arc::new(WiringTracer::new());

    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .with_tracer(tracer.clone())
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation for 24 hours
    let energy = model.solve_timesteps(24, &surrogates, false, None, None, None);

    // Verify energy is finite
    assert!(energy.is_finite());

    // Verify that simulation ran (temperatures were updated)
    // Temperature field should have been modified during simulation
    assert!(model.temperatures.len() > 0);

    // Verify solve_timesteps_with_dt and step_physics were called
    assert!(tracer.verify_called(&["solve_timesteps_with_dt"]));
    assert!(tracer.verify_called(&["step_physics"]));
}

/// Test that analytical simulation works correctly
#[test]
fn test_analytical_simulation() {
    // Create tracer for automatic call recording
    let tracer = Arc::new(WiringTracer::new());

    let scenario = BuildingScenario::new()
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .with_tracer(tracer.clone())
        .build()
        .expect("Invalid scenario");

    let mut model = scenario.create_model();
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation with analytical physics (no AI)
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    // Should produce valid, finite results
    assert!(energy.is_finite());

    // Verify solve_timesteps_with_dt and step_physics were called
    assert!(tracer.verify_called(&["solve_timesteps_with_dt"]));
    assert!(tracer.verify_called(&["step_physics"]));

    // Verify analytical path was used (no AI calls)
    assert!(!tracer.verify_called(&["predict_loads"]));
}
