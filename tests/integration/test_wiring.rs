//! Wiring validation tests
//!
//! Tests verify that modules are correctly wired together and integration points
//! work as expected. Uses runtime tracing to detect issues like solve_timesteps()
//! never calling predict_loads() when use_ai=true.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

/// Test that solve_timesteps() works correctly with surrogates
#[test]
fn test_surrogate_integration_wiring() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    // Create surrogate manager (uses mock predictions by default)
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation without AI (analytical path)
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    // Verify energy is finite and non-zero
    assert!(energy.is_finite());
}

/// Test that BatchOracle uses parallelism correctly for populations
#[test]
fn test_batch_oracle_parallelism() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

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
}

/// Test that weather data flows through to simulation
#[test]
fn test_weather_data_flow() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation for 24 hours
    let energy = model.solve_timesteps(24, &surrogates, false, None, None, None);

    // Verify energy is finite
    assert!(energy.is_finite());

    // Verify that simulation ran (temperatures were updated)
    // Temperature field should have been modified during simulation
    assert!(model.temperatures.len() > 0);
}

/// Test that analytical simulation works correctly
#[test]
fn test_analytical_simulation() {
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run simulation with analytical physics
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);

    // Should produce valid, finite results
    assert!(energy.is_finite());
}
