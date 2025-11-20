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

/// Standard Single-Building Model (for detailed analysis)
#[pyclass]
struct Model {
    inner: ThermalModel,
    surrogates: SurrogateManager,
}

#[pymethods]
impl Model {
    #[new]
    fn new(_config_path: String) -> PyResult<Self> {
        Ok(Model {
            inner: ThermalModel::new(10),
            surrogates: SurrogateManager::new()?,
        })
    }

    fn simulate(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        let steps = years as usize * 8760;
        Ok(self.inner.solve_timesteps(steps, &self.surrogates, use_surrogates))
    }
}

/// High-Throughput Oracle for Quantum/ML Optimization Loops.
/// Evaluates thousands of configurations in parallel.
#[pyclass]
struct BatchOracle {
    base_model: ThermalModel,
    surrogates: SurrogateManager,
}

#[pymethods]
impl BatchOracle {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(BatchOracle {
            base_model: ThermalModel::new(10), // The "template" building
            surrogates: SurrogateManager::new()?,
        })
    }

    /// The "Hot Loop" function.
    /// Accepts a list of parameter vectors (population).
    /// Returns a list of fitness values (EUI).
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
