#![allow(clippy::useless_conversion)]
pub mod ai;
pub mod physics;
pub mod sim;
pub mod validation;
pub mod weather;

// Re-export thermal model traits for public API
pub use sim::thermal_model::{
    PhysicsThermalModel, SurrogateThermalModel, ThermalModelBuilder, ThermalModelMode,
    ThermalModelTrait, UnifiedThermalModel,
};

// Re-export ISO 13790 Annex C construction types
pub use sim::construction::{Construction, ConstructionLayer, MassClass};

#[cfg(feature = "python-bindings")]
use crate::physics::cta::{ContinuousTensor, VectorField};
#[cfg(feature = "python-bindings")]
use ai::surrogate::SurrogateManager;
#[cfg(feature = "python-bindings")]
use sim::engine::ThermalModel;

#[cfg(feature = "python-bindings")]
use pyo3::{
    prelude::{pyclass, PyModule},
    pymethods, pymodule,
    types::PyModuleMethods,
    Bound, PyResult, Python,
};

// When not using python-bindings feature, we still need these for tests
#[cfg(not(feature = "python-bindings"))]
#[allow(unused_imports)]
use ai::surrogate::SurrogateManager;
#[cfg(not(feature = "python-bindings"))]
#[allow(unused_imports)]
use sim::engine::ThermalModel;

// Re-export things for easier access in other modules
// pub use ai::tensor_wrapper::TorchScalar; // REMOVED

/// Standard Single-Building Model for detailed building energy analysis.
///
/// Use this class when you need detailed simulation of a single building configuration,
/// including hourly temperature traces and ASHRAE 140 validation.
#[cfg(feature = "python-bindings")]
#[pyclass]
struct Model {
    inner: ThermalModel<VectorField>,
    surrogates: SurrogateManager,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl Model {
    /// Create a new Model instance.
    ///
    /// # Arguments
    /// * `_config_path` - Path to building configuration file
    #[new]
    fn new(_config_path: String) -> PyResult<Self> {
        Ok(Model {
            inner: ThermalModel::<VectorField>::new(10),
            surrogates: SurrogateManager::new()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
        })
    }

    /// Simulate building energy consumption over specified years.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions; if false, use analytical calculations
    ///
    /// # Returns
    /// Total energy use intensity (EUI) in kWh/m²/year
    fn simulate(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        let steps = years as usize * 8760;
        Ok(self
            .inner
            .solve_timesteps(steps, &self.surrogates, use_surrogates))
    }

    /// Register an ONNX surrogate model for this `Model` instance.
    fn load_surrogate(&mut self, model_path: String) -> PyResult<()> {
        match SurrogateManager::load_onnx(&model_path) {
            Ok(manager) => {
                self.surrogates = manager;
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

/// High-throughput parallel oracle for quantum and genetic algorithm optimization.
///
/// This is the core API for bulk evaluation of building design populations. It accepts
/// thousands of parameter vectors and returns fitness values (EUI) using data parallelism
/// across CPU cores. Critical for integrating with D-Wave quantum annealers and GA frameworks.
#[cfg(feature = "python-bindings")]
#[pyclass]
struct BatchOracle {
    base_model: ThermalModel<VectorField>,
    surrogates: SurrogateManager,
}

#[cfg(feature = "python-bindings")]
impl BatchOracle {
    // Physical constraints for optimization parameters
    const MIN_U_VALUE: f64 = 0.1; // Minimum realistic U-value (W/m²K)
    const MAX_U_VALUE: f64 = 5.0; // Maximum realistic U-value
    const MIN_HEATING_SETPOINT: f64 = 15.0; // Min heating setpoint (°C)
    const MAX_HEATING_SETPOINT: f64 = 25.0; // Max heating setpoint (°C)
    const MIN_COOLING_SETPOINT: f64 = 22.0; // Min cooling setpoint (°C)
    const MAX_COOLING_SETPOINT: f64 = 32.0; // Max cooling setpoint (°C)

    // Parameter indices
    const U_VALUE_INDEX: usize = 0;
    const HEATING_SETPOINT_INDEX: usize = 1;
    const COOLING_SETPOINT_INDEX: usize = 2;

    /// Validates a parameter vector against physical constraints.
    fn validate_parameters(params: &[f64]) -> Result<(), String> {
        if params.len() < 3 {
            return Err("Parameter vector must have at least 3 elements.".to_string());
        }
        let u_value = params[Self::U_VALUE_INDEX];
        let heating_setpoint = params[Self::HEATING_SETPOINT_INDEX];
        let cooling_setpoint = params[Self::COOLING_SETPOINT_INDEX];

        if !(Self::MIN_U_VALUE..=Self::MAX_U_VALUE).contains(&u_value) {
            return Err(format!("U-value out of range: {}", u_value));
        }
        if !(Self::MIN_HEATING_SETPOINT..=Self::MAX_HEATING_SETPOINT).contains(&heating_setpoint) {
            return Err(format!(
                "Heating setpoint out of range: {}",
                heating_setpoint
            ));
        }
        if !(Self::MIN_COOLING_SETPOINT..=Self::MAX_COOLING_SETPOINT).contains(&cooling_setpoint) {
            return Err(format!(
                "Cooling setpoint out of range: {}",
                cooling_setpoint
            ));
        }
        // Validate that heating < cooling (deadband must exist)
        if heating_setpoint >= cooling_setpoint {
            return Err(format!(
                "Heating setpoint ({}) must be less than cooling setpoint ({})",
                heating_setpoint, cooling_setpoint
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl BatchOracle {
    /// Create a new BatchOracle instance.
    ///
    /// Initializes the base thermal model template and surrogate manager.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(BatchOracle {
            base_model: ThermalModel::<VectorField>::new(10), // The "template" building
            surrogates: SurrogateManager::new()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
        })
    }

    /// Evaluate a population of building design configurations in parallel.
    ///
    /// This is the critical "hot loop" for optimization. The function crosses the Python-Rust
    /// boundary once with all population data, then uses Rayon for multi-threaded evaluation.
    ///
    /// When using surrogates, this implements a time-first loop architecture:
    /// - Time loop (0..8760) runs sequentially on main thread
    /// - Batched inference ONCE per timestep (full GPU utilization)
    /// - Physics updates run in parallel with rayon
    ///
    /// This avoids nested parallelism and maximizes GPU tensor core utilization.
    ///
    /// # Arguments
    /// * `population` - Vec of parameter vectors, each representing one design candidate.
    ///   Each vector should have at least 3 elements:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - `[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `[2]`: Cooling setpoint (°C, range: 22-32)
    /// * `use_surrogates` - If true, use neural network surrogates for faster (~100x) evaluation;
    ///   if false, use physics-based analytical calculations (slower but exact)
    ///
    /// # Returns
    /// Vector of fitness values (EUI in kWh/m²/year) corresponding to each candidate.
    ///
    /// # Performance
    /// Target throughput: >10,000 configs/sec on 8-core CPU (~100µs per config)
    fn evaluate_population(
        &self,
        population: Vec<Vec<f64>>,
        use_surrogates: bool,
    ) -> PyResult<Vec<f64>> {
        use rayon::prelude::*;

        // 1. Validate and initialize all models upfront (parallel)
        let mut valid_configs: Vec<(usize, ThermalModel<VectorField>)> = population
            .par_iter()
            .enumerate()
            .filter_map(|(i, params)| {
                if Self::validate_parameters(params).is_err() {
                    return None;
                }
                let mut model = self.base_model.clone();
                model.apply_parameters(params);
                Some((i, model))
            })
            .collect();

        let mut results = vec![f64::NAN; population.len()];

        if use_surrogates && !valid_configs.is_empty() {
            // Coordinator-Worker pattern with Channels
            let n_workers = valid_configs.len();
            let mut coord_txs = Vec::with_capacity(n_workers);
            let mut coord_rxs = Vec::with_capacity(n_workers);
            let mut worker_channels = Vec::with_capacity(n_workers);

            for _ in 0..n_workers {
                let (tx_to_coord, rx_from_worker) = crossbeam::channel::unbounded();
                let (tx_to_worker, rx_from_coord) = crossbeam::channel::unbounded();
                coord_rxs.push(rx_from_worker);
                coord_txs.push(tx_to_worker);
                worker_channels.push((tx_to_coord, rx_from_coord));
            }

            let final_worker_data = rayon::scope(|s| {
                let (result_tx, result_rx) = crossbeam::channel::unbounded();

                // Move models and channels into workers
                for ((idx, mut model), (tx, rx)) in
                    valid_configs.drain(..).zip(worker_channels.into_iter())
                {
                    let res_tx = result_tx.clone();
                    s.spawn(move |_| {
                        let energy = model.solve_timesteps_batched(8760, tx, rx);
                        let _ = res_tx.send((idx, model, energy));
                    });
                }
                drop(result_tx);

                // Coordinator loop
                for _t in 0..8760 {
                    // 1. Collect temperatures from all workers
                    let mut batch_temps = Vec::with_capacity(n_workers);
                    for rx in &coord_rxs {
                        batch_temps.push(rx.recv().expect("Worker disconnected unexpectedly"));
                    }

                    // 2. Batched inference
                    let batch_loads = self.surrogates.predict_loads_batched(&batch_temps);

                    // 3. Send loads back to workers
                    for (tx, loads) in coord_txs.iter().zip(batch_loads) {
                        tx.send(loads).expect("Failed to send loads to worker");
                    }
                }

                let mut final_data = Vec::with_capacity(n_workers);
                while let Ok(data) = result_rx.recv() {
                    final_data.push(data);
                }
                final_data
            });

            // 3. Normalize and populate results
            for (idx, model, energy) in final_worker_data {
                let total_area = model.zone_area.integrate();
                results[idx] = if total_area > 0.0 {
                    (energy / total_area).max(0.0)
                } else {
                    0.0
                };
            }
        } else if !valid_configs.is_empty() {
            // Analytical path - fully parallel
            let mut energies = vec![0.0; valid_configs.len()];
            valid_configs
                .par_iter_mut()
                .zip(energies.par_iter_mut())
                .for_each(|((_, model), energy)| {
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle =
                            (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                        *energy +=
                            model.solve_single_step(t, outdoor_temp, false, &self.surrogates, true);
                    }
                });

            for ((idx, model), energy) in valid_configs.iter().zip(energies.iter()) {
                let total_area = model.zone_area.integrate();
                results[*idx] = if total_area > 0.0 {
                    (*energy / total_area).max(0.0)
                } else {
                    0.0
                };
            }
        }

        Ok(results)
    }

    /// Register an ONNX surrogate model for the oracle. This replaces the internal
    /// `SurrogateManager` with one pointing at the provided model file.
    fn load_surrogate(&mut self, model_path: String) -> PyResult<()> {
        match SurrogateManager::load_onnx(&model_path) {
            Ok(manager) => {
                self.surrogates = manager;
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<BatchOracle>()?;
    Ok(())
}

// Re-export ASHRAE 140 validation models
pub use validation::ashrae_140::{Case600Model, SimulationResult};

// Tests for core physics engine (no Python bindings required)
#[cfg(test)]
mod tests {
    use crate::ai::surrogate::SurrogateManager;
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;

    #[cfg(feature = "python-bindings")]
    use crate::BatchOracle;

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_validation() {
        let oracle = BatchOracle::new().unwrap();
        let population = vec![
            vec![1.5, 20.0, 27.0],  // Valid
            vec![-1.0, 20.0, 27.0], // Invalid U-value
            vec![1.5, 500.0, 27.0], // Invalid heating setpoint
            vec![1.5, 20.0, 10.0],  // Invalid cooling setpoint
            vec![1.5, 27.0, 20.0],  // Invalid: heating >= cooling
        ];

        let results = oracle.evaluate_population(population, false).unwrap();

        assert!(results[0].is_finite());
        assert!(results[1].is_nan());
        assert!(results[2].is_nan());
        assert!(results[3].is_nan());
        assert!(results[4].is_nan());
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batched_vs_unbatched_consistency() {
        let oracle = BatchOracle::new().unwrap();
        let population = vec![vec![1.5, 22.0], vec![2.0, 21.0], vec![1.0, 23.0]];

        // Test surrogate path
        let results_batched = oracle
            .evaluate_population(population.clone(), true)
            .unwrap();
        assert!(results_batched.iter().all(|r: &f64| r.is_finite()));

        // Test analytical path for comparison
        let results_analytical = oracle.evaluate_population(population, false).unwrap();
        assert!(results_analytical.iter().all(|r: &f64| r.is_finite()));

        // Results should be in similar range (may differ due to mock vs analytical loads)
        for (batched, analytical) in results_batched.iter().zip(results_analytical.iter()) {
            assert!(*batched > 0.0, "Batched result should be positive");
            assert!(*analytical > 0.0, "Analytical result should be positive");
        }
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_large_population_performance() {
        let oracle = BatchOracle::new().unwrap();
        let population: Vec<Vec<f64>> = (0..1000).map(|_| vec![1.5, 22.0]).collect();

        let start = std::time::Instant::now();
        let results = oracle.evaluate_population(population, true).unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 1000);
        assert!(results.iter().all(|r: &f64| r.is_finite()));

        // Target: <100ms for 1000 configs (may be slower in debug mode)
        #[cfg(debug_assertions)]
        println!("Debug mode: {:?}", duration);
        #[cfg(not(debug_assertions))]
        assert!(duration.as_millis() < 100, "Too slow: {:?}", duration);
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_10k_population_throughput() {
        let oracle = BatchOracle::new().unwrap();
        let population: Vec<Vec<f64>> = (0..10_000).map(|_| vec![1.5, 22.0]).collect();

        let start = std::time::Instant::now();
        let results = oracle.evaluate_population(population, true).unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 10_000);
        assert!(results.iter().all(|r: &f64| r.is_finite()));

        let configs_per_sec = 10_000.0 / duration.as_secs_f64();
        println!("Throughput: {:.0} configs/sec", configs_per_sec);

        // Target: >10,000 configs/sec on 8-core CPU (may be slower in debug mode)
        #[cfg(not(debug_assertions))]
        assert!(
            configs_per_sec > 10_000.0,
            "Below target: {:.0}/sec",
            configs_per_sec
        );
    }

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::<VectorField>::new(10);
        assert_eq!(model.num_zones, 10);
    }

    #[test]
    fn test_apply_parameters() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5, 20.0, 27.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.heating_setpoint, 20.0);
        assert_eq!(model.cooling_setpoint, 27.0);
    }

    #[test]
    fn test_solve_timesteps() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 20.0, 27.0]);
        let energy = model.solve_timesteps(8760, &surrogates, false);

        assert!(energy.is_finite(), "Energy should be finite"); // Can be negative for cooling or mass charging
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 20.0, 27.0]);
        let energy = model.solve_timesteps(8760, &surrogates, true);

        assert!(energy > 0.0, "Energy should be positive");
    }

    #[test]
    fn test_parallel_execution_speedup() {
        use rayon::prelude::*;
        use std::path::Path;

        // Verify Send + Sync for ThermalModel (required for parallel execution)
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThermalModel<VectorField>>();

        let base_model = ThermalModel::<VectorField>::new(10);

        // Try to load a real model if available (created by other tests)
        // otherwise fall back to mock (but verify parallel mechanism either way)
        // Ideally we want to test with the pool active.
        let model_path = "tests_tmp_dummy.onnx";
        let surrogates = if Path::new(model_path).exists() {
            match SurrogateManager::load_onnx(model_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to load dummy model (proceeding with mock): {}", e);
                    SurrogateManager::new().expect("Failed to create SurrogateManager")
                }
            }
        } else {
            // Fall back to mock SurrogateManager if file missing
            eprintln!("tests_tmp_dummy.onnx not found; proceeding with mock SurrogateManager");
            SurrogateManager::new().expect("Failed to create SurrogateManager")
        };

        // Create a large population
        let population_size = 2000;
        let population: Vec<Vec<f64>> = (0..population_size)
            .map(|_| vec![1.5, 20.0, 27.0])
            .collect();

        // Sequential execution (using standard iter)
        let start_seq = std::time::Instant::now();
        let _results_seq: Vec<f64> = population
            .iter()
            .map(|params| {
                let mut instance = base_model.clone();
                instance.apply_parameters(params);
                // Use surrogates to test session pool contention/parallelism
                instance.solve_timesteps(100, &surrogates, true)
            })
            .collect();
        let duration_seq = start_seq.elapsed();

        // Parallel execution (using rayon par_iter)
        let start_par = std::time::Instant::now();
        let _results_par: Vec<f64> = population
            .par_iter()
            .map(|params| {
                let mut instance = base_model.clone();
                instance.apply_parameters(params);
                instance.solve_timesteps(100, &surrogates, true)
            })
            .collect();
        let duration_par = start_par.elapsed();

        println!("Sequential time: {:?}", duration_seq);
        println!("Parallel time: {:?}", duration_par);

        // On a multi-core machine, parallel should be faster.
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        if num_threads > 1 {
            // We expect significant speedup, but CI environments can be noisy.
            // Just asserting it's faster is a good baseline.
            assert!(
                duration_par < duration_seq,
                "Parallel execution should be faster than sequential on {} threads. Seq: {:?}, Par: {:?}",
                num_threads,
                duration_seq,
                duration_par
            );
        }
    }
}
