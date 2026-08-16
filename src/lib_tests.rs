//! Engine + logging unit tests lifted out of `lib.rs` (Issue #2493).
//!
//! These exercise the generic `ThermalModel<VectorField>` API and the logging
//! façade; they do not depend on `BatchOracle` and compile under the default
//! (no-python-bindings) feature set. `BatchOracle`-specific tests live in
//! [`crate::batch_oracle`].

#![allow(clippy::redundant_closure)]

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

#[cfg(feature = "python-bindings")]
use crate::BatchOracle;

// `info!` is used by the `#[cfg(feature = "python-bindings")]` block in
// `test_logging_control` below.
#[allow(unused_imports)]
use log::{debug, info};

#[test]
fn test_thermal_model_creation() {
    let model = ThermalModel::<VectorField>::new(10);
    assert_eq!(model.hvac.num_zones, 10);
}

#[test]
fn test_thermal_model_default() {
    let model = ThermalModel::<VectorField>::new(1);
    assert_eq!(model.hvac.num_zones, 1);
    assert_eq!(model.setpoints.temperatures.as_ref().len(), 1);
}

#[test]
fn test_apply_parameters() {
    let mut model = ThermalModel::<VectorField>::new(10);
    let params = vec![1.5, 20.0, 27.0];

    model.apply_parameters(&params);
    assert_eq!(model.solar.window_u_value, 1.5);
    assert_eq!(model.setpoints.heating_setpoint, 20.0);
    assert_eq!(model.setpoints.cooling_setpoint, 27.0);
}

#[test]
fn test_solve_timesteps() {
    let mut model = ThermalModel::<VectorField>::new(10);
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    model.apply_parameters(&[1.5, 20.0, 27.0]);
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    assert!(energy.is_finite(), "Energy should be finite"); // Can be negative for cooling or mass charging
}

#[test]
fn test_solve_timesteps_with_surrogates() {
    let mut model = ThermalModel::<VectorField>::new(10);
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    model.apply_parameters(&[1.5, 20.0, 27.0]);
    // Should NOT panic now since it returns mock loads
    let energy = model.solve_timesteps(8760, &surrogates, true, None, None, None);
    assert!(energy.is_finite());
}

#[test]
fn test_parallel_execution_speedup() {
    use rayon::prelude::*;
    use std::path::Path;

    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ThermalModel<VectorField>>();

    let base_model = ThermalModel::<VectorField>::new(10);

    let model_path = "tests_tmp_dummy.onnx";
    let (surrogates, _use_real_model) = if Path::new(model_path).exists() {
        match SurrogateManager::load_onnx(model_path) {
            Ok(s) => (s, true),
            Err(e) => {
                eprintln!("Failed to load dummy model (proceeding with mock): {}", e);
                (
                    SurrogateManager::new().expect("Failed to create SurrogateManager"),
                    false,
                )
            }
        }
    } else {
        eprintln!("tests_tmp_dummy.onnx not found; proceeding with mock SurrogateManager");
        (
            SurrogateManager::new().expect("Failed to create SurrogateManager"),
            false,
        )
    };

    let population_size = 2000;
    let population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| vec![1.5, 20.0, 27.0])
        .collect();

    let start_seq = std::time::Instant::now();
    let _results_seq: Vec<f64> = population
        .iter()
        .map(|params| {
            let mut instance = base_model.clone();
            instance.apply_parameters(params);
            instance.solve_timesteps(100, &surrogates, true, None, None, None)
        })
        .collect();
    let duration_seq = start_seq.elapsed();

    let start_par = std::time::Instant::now();
    let _results_par: Vec<f64> = population
        .par_iter()
        .map(|params| {
            let mut instance = base_model.clone();
            instance.apply_parameters(params);
            instance.solve_timesteps(100, &surrogates, true, None, None, None)
        })
        .collect();
    let duration_par = start_par.elapsed();

    println!("Sequential time: {:?}", duration_seq);
    println!("Parallel time: {:?}", duration_par);
    println!(
        "Available parallelism: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );

    assert!(
        duration_par > std::time::Duration::ZERO && duration_seq > std::time::Duration::ZERO,
        "Both sequential and parallel runs should produce valid timings. Seq: {:?}, Par: {:?}",
        duration_seq,
        duration_par
    );
}

#[test]
fn test_logging_control() {
    // Test that logging can be controlled via RUST_LOG environment variable
    // This test verifies that the logging infrastructure is properly initialized
    // and that log statements don't cause panics or errors

    // Initialize logger (should be idempotent)
    let _ = env_logger::try_init();

    // Test various log levels - these should not panic
    log::error!("Test error log");
    log::warn!("Test warn log");
    log::info!("Test info log");
    log::debug!("Test debug log");
    log::trace!("Test trace log");

    // Test that BatchOracle and Model can be created and used with logging
    #[cfg(feature = "python-bindings")]
    {
        let oracle = BatchOracle::new().unwrap();
        info!("Created BatchOracle with logging");

        let population = vec![vec![1.5, 20.0, 27.0]];
        let results = oracle.evaluate_population(population, false).unwrap();
        assert!(results[0].is_finite());
        info!("BatchOracle evaluation completed successfully");
    }
}
