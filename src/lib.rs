pub mod ai;
pub mod physics;
pub mod sim;

#[cfg(feature = "python-bindings")]
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "python-bindings")]
use rayon::prelude::ParallelIterator;

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
    inner: ThermalModel,
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
            inner: ThermalModel::new(10),
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
    base_model: ThermalModel,
    surrogates: SurrogateManager,
}

#[cfg(feature = "python-bindings")]
impl BatchOracle {
    // Physical constraints for optimization parameters
    const MIN_U_VALUE: f64 = 0.1; // Minimum realistic U-value (W/m²K)
    const MAX_U_VALUE: f64 = 5.0; // Maximum realistic U-value
    const MIN_SETPOINT: f64 = 15.0; // Min HVAC setpoint (°C)
    const MAX_SETPOINT: f64 = 30.0; // Max HVAC setpoint (°C)

    // Parameter indices
    const U_VALUE_INDEX: usize = 0;
    const SETPOINT_INDEX: usize = 1;

    /// Validates a parameter vector against physical constraints.
    fn validate_parameters(params: &[f64]) -> Result<(), String> {
        if params.len() < 2 {
            return Err("Parameter vector must have at least 2 elements.".to_string());
        }
        let u_value = params[Self::U_VALUE_INDEX];
        let setpoint = params[Self::SETPOINT_INDEX];

        if u_value < Self::MIN_U_VALUE || u_value > Self::MAX_U_VALUE {
            return Err(format!("U-value out of range: {}", u_value));
        }
        if setpoint < Self::MIN_SETPOINT || setpoint > Self::MAX_SETPOINT {
            return Err(format!("Setpoint out of range: {}", setpoint));
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
            base_model: ThermalModel::new(10), // The "template" building
            surrogates: SurrogateManager::new()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
        })
    }

    /// Evaluate a population of building design configurations in parallel.
    ///
    /// This is the critical "hot loop" for optimization. The function crosses the Python-Rust
    /// boundary once with all population data, then uses Rayon for multi-threaded evaluation.
    ///
    /// # Arguments
    /// * `population` - Vec of parameter vectors, each representing one design candidate.
    ///   Each vector should have at least 2 elements:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.5-3.0)
    ///   - `[1]`: HVAC setpoint (°C, range: 19-24)
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
        // 1. Cross the Python-Rust boundary ONCE with all data.

        // 2. Use Rayon to parallelize the simulation of each candidate.
        // This scales linearly with CPU cores.
        let results: Vec<f64> = population
            .par_iter()
            .map(|params| {
                if Self::validate_parameters(params).is_err() {
                    return f64::NAN;
                }

                // Light clone of the base model state
                let mut instance = self.base_model.clone();

                // Apply the specific genes (window ratio, setpoints, etc.)
                instance.apply_parameters(params);

                // Run the physics engine
                instance.solve_timesteps(8760, &self.surrogates, use_surrogates)
            })
            .collect();

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

// Tests for core physics engine (no Python bindings required)
#[cfg(test)]
mod tests {
    use crate::ai::surrogate::SurrogateManager;
    use crate::sim::engine::ThermalModel;

    #[cfg(feature = "python-bindings")]
    use crate::BatchOracle;

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_validation() {
        let oracle = BatchOracle::new().unwrap();
        let population = vec![
            vec![1.5, 22.0],  // Valid
            vec![-1.0, 22.0], // Invalid U-value
            vec![1.5, 500.0], // Invalid setpoint
            vec![0.05, 22.0], // Invalid U-value (too low)
            vec![1.5, 10.0],  // Invalid setpoint (too low)
        ];

        let results = oracle.evaluate_population(population, false).unwrap();

        assert!(results[0].is_finite());
        assert!(results[1].is_nan());
        assert!(results[2].is_nan());
        assert!(results[3].is_nan());
        assert!(results[4].is_nan());
    }

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::new(10);
        assert_eq!(model.num_zones, 10);
    }

    #[test]
    fn test_apply_parameters() {
        let mut model = ThermalModel::new(10);
        let params = vec![1.5, 22.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }

    #[test]
    fn test_solve_timesteps() {
        let mut model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 21.0]);
        let energy = model.solve_timesteps(8760, &surrogates, false);

        assert!(energy > 0.0, "Energy should be positive");
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let mut model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 21.0]);
        let energy = model.solve_timesteps(8760, &surrogates, true);

        assert!(energy > 0.0, "Energy should be positive");
    }

    #[test]
    fn test_parallel_execution_speedup() {
        use rayon::prelude::*;
        use std::path::Path;

        // Verify Send + Sync for ThermalModel (required for parallel execution)
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThermalModel>();

        let base_model = ThermalModel::new(10);

        // Try to load a real model if available (created by other tests)
        // otherwise fall back to mock (but verify parallel mechanism either way)
        // Ideally we want to test with the pool active.
        let model_path = "tests_tmp_dummy.onnx";
        let surrogates = if Path::new(model_path).exists() {
            SurrogateManager::load_onnx(model_path).expect("Failed to load dummy model")
        } else {
            // Even with mock, the code path goes through predict_loads.
            // But if model_loaded is false, it returns early.
            // We need model_loaded=true to test the pool.
            // If no file, we can't easily test the pool without mocking Session.
            // But we know tests_tmp_dummy.onnx exists in this env.
            panic!("tests_tmp_dummy.onnx not found. Run generation script or ai tests first.");
        };

        // Create a large population
        let population_size = 2000;
        let population: Vec<Vec<f64>> = (0..population_size).map(|_| vec![1.5, 22.0]).collect();

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
