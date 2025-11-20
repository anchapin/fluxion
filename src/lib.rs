use rayon::prelude::*;

pub mod ai;
pub mod sim;

#[cfg(feature = "python-bindings")]
use ai::surrogate::SurrogateManager;
#[cfg(feature = "python-bindings")]
use sim::engine::ThermalModel;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

// When not using python-bindings feature, we still need these for tests
#[cfg(not(feature = "python-bindings"))]
use ai::surrogate::SurrogateManager;
#[cfg(not(feature = "python-bindings"))]
use sim::engine::ThermalModel;

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
    use super::*;

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

        model.apply_parameters(&vec![1.5, 21.0]);
        let energy = model.solve_timesteps(8760, &surrogates, false);

        assert!(energy > 0.0, "Energy should be positive");
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let mut model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&vec![1.5, 21.0]);
        let energy = model.solve_timesteps(8760, &surrogates, true);

        assert!(energy > 0.0, "Energy should be positive");
    }
}
