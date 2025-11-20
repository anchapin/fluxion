use pyo3::prelude::*;
use rayon::prelude::*;

pub mod sim;
pub mod ai;

use sim::engine::ThermalModel;
use ai::surrogate::SurrogateManager;

#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<BatchOracle>()?;
    Ok(())
}

/// Standard Single-Building Model for detailed building energy analysis.
/// 
/// Use this class when you need detailed simulation of a single building configuration,
/// including hourly temperature traces and ASHRAE 140 validation.
#[pyclass]
struct Model {
    inner: ThermalModel,
    surrogates: SurrogateManager,
}

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
            surrogates: SurrogateManager::new()?,
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
        Ok(self.inner.solve_timesteps(steps, &self.surrogates, use_surrogates))
    }
}

/// High-throughput parallel oracle for quantum and genetic algorithm optimization.
/// 
/// This is the core API for bulk evaluation of building design populations. It accepts
/// thousands of parameter vectors and returns fitness values (EUI) using data parallelism
/// across CPU cores. Critical for integrating with D-Wave quantum annealers and GA frameworks.
#[pyclass]
struct BatchOracle {
    base_model: ThermalModel,
    surrogates: SurrogateManager,
}

#[pymethods]
impl BatchOracle {
    /// Create a new BatchOracle instance.
    /// 
    /// Initializes the base thermal model template and surrogate manager.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(BatchOracle {
            base_model: ThermalModel::new(10), // The "template" building
            surrogates: SurrogateManager::new()?,
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
    fn evaluate_population(&self, population: Vec<Vec<f64>>, use_surrogates: bool) -> PyResult<Vec<f64>> {
        // 1. Cross the Python-Rust boundary ONCE with all data.
        
        // 2. Use Rayon to parallelize the simulation of each candidate.
        // This scales linearly with CPU cores.
        let results: Vec<f64> = population.par_iter()
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
