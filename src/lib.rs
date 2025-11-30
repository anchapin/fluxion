pub mod ai;
pub mod physics;
pub mod sim;
pub mod validation;


#[cfg(feature = "python-bindings")]
use crate::physics::cta::{ContinuousTensor, VectorField};
#[cfg(feature = "python-bindings")]
use crate::physics::nd_array::NDArrayField;
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

#[cfg(feature = "python-bindings")]
#[derive(Clone)]
enum BackendModel {
    Vector(Box<ThermalModel<VectorField>>),
    NDArray(Box<ThermalModel<NDArrayField>>),
}

#[cfg(feature = "python-bindings")]
impl BackendModel {
    fn apply_parameters(&mut self, params: &[f64]) {
        match self {
            BackendModel::Vector(m) => m.as_mut().apply_parameters(params),
            BackendModel::NDArray(m) => m.as_mut().apply_parameters(params),
        }
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_ai: bool,
    ) -> f64 {
        match self {
            BackendModel::Vector(m) => m.as_mut().solve_timesteps(steps, surrogates, use_ai),
            BackendModel::NDArray(m) => m.as_mut().solve_timesteps(steps, surrogates, use_ai),
        }
    }

    fn set_loads(&mut self, loads: &[f64]) {
        match self {
            BackendModel::Vector(m) => m.loads = m.temperatures.new_with_data(loads.to_vec()),
            BackendModel::NDArray(m) => m.loads = m.temperatures.new_with_data(loads.to_vec()),
        }
    }

    fn step_physics(&mut self, outdoor_temp: f64) -> f64 {
        match self {
            BackendModel::Vector(m) => m.step_physics(outdoor_temp),
            BackendModel::NDArray(m) => m.step_physics(outdoor_temp),
        }
    }

    fn calc_analytical_loads(&mut self, timestep: usize, use_analytical_gains: bool) {
        match self {
            BackendModel::Vector(m) => m.calc_analytical_loads(timestep, use_analytical_gains),
            BackendModel::NDArray(m) => m.calc_analytical_loads(timestep, use_analytical_gains),
        }
    }

    fn get_total_area(&self) -> f64 {
        match self {
            BackendModel::Vector(m) => m.zone_area.integrate(),
            BackendModel::NDArray(m) => m.zone_area.integrate(),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl AsRef<[f64]> for BackendModel {
    fn as_ref(&self) -> &[f64] {
        match self {
            BackendModel::Vector(m) => m.temperatures.as_slice(),
            BackendModel::NDArray(m) => m.temperatures.as_slice(),
        }
    }
}

/// Standard Single-Building Model for detailed building energy analysis.
///
/// Use this class when you need detailed simulation of a single building configuration,
/// including hourly temperature traces and ASHRAE 140 validation.
#[cfg(feature = "python-bindings")]
#[pyclass]
struct Model {
    inner: BackendModel,
    surrogates: SurrogateManager,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl Model {
    /// Create a new Model instance.
    ///
    /// # Arguments
    /// * `_config_path` - Path to building configuration file
    /// * `shape` - Optional shape for NDArray backend (e.g., [10, 10]). If None, uses Vector backend with 10 zones.
    #[new]
    #[pyo3(signature = (_config_path, shape = None))]
    fn new(_config_path: String, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let inner = if let Some(s) = shape {
            BackendModel::NDArray(Box::new(
                ThermalModel::<NDArrayField>::new_ndarray_with_shape(s),
            ))
        } else {
            BackendModel::Vector(Box::new(ThermalModel::<VectorField>::new(10)))
        };
        Ok(Model {
            inner,
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
    base_model: BackendModel,
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

        if !(Self::MIN_U_VALUE..=Self::MAX_U_VALUE).contains(&u_value) {
            return Err(format!("U-value out of range: {}", u_value));
        }
        if !(Self::MIN_SETPOINT..=Self::MAX_SETPOINT).contains(&setpoint) {
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
    ///
    /// # Arguments
    /// * `shape` - Optional shape for NDArray backend.
    #[new]
    #[pyo3(signature = (shape = None))]
    fn new(shape: Option<Vec<usize>>) -> PyResult<Self> {
        let base_model = if let Some(s) = shape {
            BackendModel::NDArray(Box::new(
                ThermalModel::<NDArrayField>::new_ndarray_with_shape(s),
            ))
        } else {
            BackendModel::Vector(Box::new(ThermalModel::<VectorField>::new(10)))
        };
        Ok(BatchOracle {
            base_model,
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
        use rayon::prelude::*;

        // 1. Cross the Python-Rust boundary ONCE with all data.

        // 2. Initialize all models in parallel
        let mut active_instances: Vec<(usize, BackendModel)> = population
            .par_iter()
            .enumerate()
            .filter_map(|(i, params)| {
                if Self::validate_parameters(params).is_err() {
                    return None;
                }
                let mut instance = self.base_model.clone();
                instance.apply_parameters(params);
                Some((i, instance))
            })
            .collect();

        // 3. Prepare accumulators for energy
        let mut active_energies = vec![0.0; active_instances.len()];

        // 4. Time loop (0..8760) - "Outer Loop" Pattern for Batched Inference
        for t in 0..8760 {
            let hour_of_day = t % 24;
            let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
            let outdoor_temp = 10.0 + 10.0 * daily_cycle;

            if use_surrogates {
                // Collect inputs for the entire population
                let batch_inputs: Vec<&[f64]> = active_instances
                    .iter()
                    .map(|(_, m)| m.as_ref())
                    .collect();

                // Run batched inference once
                let batch_loads = self.surrogates.predict_loads_batched(&batch_inputs);

                // Update physics in parallel
                active_instances
                    .par_iter_mut()
                    .zip(active_energies.par_iter_mut())
                    .zip(batch_loads.par_iter())
                    .for_each(|(((_, model), energy), loads)| {
                        model.set_loads(loads);
                        *energy += model.step_physics(outdoor_temp);
                    });
            } else {
                // Analytical loads + Physics in parallel
                active_instances
                    .par_iter_mut()
                    .zip(active_energies.par_iter_mut())
                    .for_each(|((_, model), energy)| {
                        model.calc_analytical_loads(t, true);
                        *energy += model.step_physics(outdoor_temp);
                    });
            }
        }

        // 5. Finalize results (Calculate EUI)
        let mut final_results = vec![f64::NAN; population.len()];

        for ((idx, model), energy_kwh) in active_instances.iter().zip(active_energies.iter()) {
            let area = model.get_total_area();
            if area > 0.0 {
                final_results[*idx] = energy_kwh / area;
            } else {
                final_results[*idx] = 0.0;
            }
        }

        Ok(final_results)
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
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;

    #[cfg(feature = "python-bindings")]
    use crate::BatchOracle;

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_validation() {
        let oracle = BatchOracle::new(None).unwrap();
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

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_surrogates() {
        let oracle = BatchOracle::new(None).unwrap();
        let population = vec![
            vec![1.5, 22.0],
            vec![2.0, 21.0],
        ];

        // This exercises the batched loop path with mock surrogates
        let results = oracle.evaluate_population(population, true).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_finite());
        assert!(results[0] > 0.0);
        assert!(results[1].is_finite());
        assert!(results[1] > 0.0);
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_model_ndarray_backend() {
        use crate::Model;
        // Instantiate model with NDArray backend (10x10 grid)
        let mut model = Model::new("config.json".to_string(), Some(vec![10, 10])).unwrap();

        // Run simulation
        let result = model.simulate(1, false).unwrap();

        // Verify result is valid
        assert!(result > 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::<VectorField>::new(10);
        assert_eq!(model.num_zones, 10);
    }

    #[test]
    fn test_apply_parameters() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5, 22.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }

    #[test]
    fn test_solve_timesteps() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 21.0]);
        let energy = model.solve_timesteps(8760, &surrogates, false);

        assert!(energy > 0.0, "Energy should be positive");
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let mut model = ThermalModel::<VectorField>::new(10);
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
