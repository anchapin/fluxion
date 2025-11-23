use pyo3::prelude::*;

mod ai;
pub mod physics;
mod sim;

use sim::engine::ThermalModel;

#[pyclass]
pub struct BatchOracle {
    base_model: ThermalModel,
}

#[pymethods]
impl BatchOracle {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(BatchOracle {
            base_model: ThermalModel::new(10),
        })
    }

    fn evaluate_population(
        &self,
        population: Vec<Vec<f64>>,
        use_surrogates: bool,
    ) -> PyResult<Vec<f64>> {
        let results = population
            .iter()
            .map(|params| {
                let mut model = self.base_model.clone();
                model.apply_parameters(params);
                model.solve_timesteps(8760, use_surrogates)
            })
            .collect();
        Ok(results)
    }
}

#[pyclass]
pub struct Model {
    model: ThermalModel,
}

#[pymethods]
impl Model {
    #[new]
    fn new(num_zones: usize) -> Self {
        Model {
            model: ThermalModel::new(num_zones),
        }
    }

    fn simulate(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        let total_steps = (years as usize) * 8760;
        Ok(self.model.solve_timesteps(total_steps, use_surrogates))
    }
}

#[pymodule]
fn fluxion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BatchOracle>()?;
    m.add_class::<Model>()?;
    Ok(())
}
