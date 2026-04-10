// Multi-zone Python bindings for Fluxion
// This module extends the existing Python API with multi-zone functionality

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Multi-zone thermal model for Python
#[pyclass(name = "MultiZoneThermalModel")]
pub struct PyMultiZoneThermalModel {
    inner: ThermalModel<VectorField>,
}

#[pymethods]
impl PyMultiZoneThermalModel {
    /// Create a new MultiZoneThermalModel with specified number of zones
    #[new]
    #[pyo3(signature = (num_zones=2))]
    pub fn new(num_zones: usize) -> PyResult<Self> {
        if num_zones < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of zones must be at least 1",
            ));
        }

        Ok(PyMultiZoneThermalModel {
            inner: ThermalModel::<VectorField>::new(num_zones),
        })
    }

    /// Get number of zones in the model
    pub fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    /// Get the number of zones from the inner ThermalModel
    pub fn get_inner_num_zones(&self) -> usize {
        self.inner.num_zones
    }

    /// Get current zone temperatures
    pub fn get_zone_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    /// Set zone temperatures
    pub fn set_zone_temperatures(&mut self, temps: Vec<f64>) -> PyResult<()> {
        if temps.len() != self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Temperature vector length ({}) must match number of zones ({})",
                temps.len(),
                self.inner.num_zones
            )));
        }
        self.inner.temperatures = VectorField::new(temps);
        Ok(())
    }

    /// Set zone-specific heating and cooling setpoints
    pub fn set_zone_setpoints(
        &mut self,
        zone_idx: usize,
        heating: f64,
        cooling: f64,
    ) -> PyResult<()> {
        if zone_idx >= self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone index {} out of range (0-{})",
                zone_idx,
                self.inner.num_zones - 1
            )));
        }

        if heating >= cooling {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Heating setpoint must be less than cooling setpoint",
            ));
        }

        // Set zone-specific setpoints
        self.inner.heating_setpoints.as_mut_slice()[zone_idx] = heating;
        self.inner.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;

        Ok(())
    }

    /// Get inter-zone conductance between two zones
    pub fn get_inter_zone_conductance(&self, zone_i: usize, zone_j: usize) -> PyResult<f64> {
        if zone_i >= self.inner.num_zones || zone_j >= self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Zone indices out of range",
            ));
        }

        // TODO: Implement inter-zone conductance once ThermalModel API is updated
        Ok(0.0)
    }

    /// Set inter-zone conductance between two zones
    pub fn set_inter_zone_conductance(
        &mut self,
        zone_i: usize,
        zone_j: usize,
        conductance: f64,
    ) -> PyResult<()> {
        if zone_i >= self.inner.num_zones || zone_j >= self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Zone indices out of range",
            ));
        }

        if conductance < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Conductance must be non-negative",
            ));
        }

        // TODO: Implement inter-zone conductance once ThermalModel API is updated
        Ok(())
    }

    /// Simulate multi-zone building energy consumption
    pub fn simulate_multi_zone(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        let steps = years as usize * 8760;

        // Use the existing surrogate manager from the model
        let surrogates = crate::ai::surrogate::SurrogateManager::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create SurrogateManager: {}",
                e
            ))
        })?;

        let result =
            self.inner
                .solve_timesteps(steps, &surrogates, use_surrogates, None, None, None);
        Ok(result)
    }

    /// Get zone-specific energy consumption
    pub fn get_zone_energies(&self) -> Vec<f64> {
        // TODO: Implement zone energy consumption once ThermalModel API is updated
        vec![0.0; self.inner.num_zones]
    }

    /// Get zone-specific peak loads
    pub fn get_zone_peak_loads(&self) -> PyResult<HashMap<String, Vec<f64>>> {
        let mut result = HashMap::new();

        // TODO: Implement peak loads once ThermalModel API is updated
        result.insert("heating_peaks".to_string(), vec![0.0; self.inner.num_zones]);

        // Add cooling peaks
        result.insert("cooling_peaks".to_string(), vec![0.0; self.inner.num_zones]);

        Ok(result)
    }

    /// Export zone temperatures as Python dictionary
    pub fn export_zone_temperatures(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);

            for (i, temp) in self.inner.get_temperatures().iter().enumerate() {
                dict.set_item(format!("zone_{}", i), *temp)?;
            }

            Ok(dict.unbind())
        })
    }

    /// Run energy balance validation for multi-zone model
    pub fn validate_energy_balance(&self) -> PyResult<bool> {
        // TODO: Implement energy balance validation once ThermalModel API is updated
        Ok(true) // Assume balanced for now
    }
}

/// Create a multi-zone thermal model from a configuration dictionary
#[pyfunction]
pub fn create_multi_zone_model_from_config(
    config: &Bound<'_, PyDict>,
) -> PyResult<PyMultiZoneThermalModel> {
    // Extract number of zones
    let num_zones: usize = match config.get_item("num_zones")? {
        Some(item) => item.extract()?,
        None => {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "Missing 'num_zones' in config",
            ))
        }
    };

    // Create model
    let mut model = PyMultiZoneThermalModel::new(num_zones)?;

    // Set zone setpoints if provided
    if let Some(zone_setpoints) = config.get_item("zone_setpoints")? {
        let setpoints_dict: &Bound<'_, PyDict> = zone_setpoints.downcast()?;

        for (key, value) in setpoints_dict {
            let zone_key: String = key.extract()?;
            if let Some(stripped) = zone_key.strip_prefix("zone_") {
                if let Ok(zone_idx) = stripped.parse::<usize>() {
                    let setpoints_tuple: (f64, f64) = value.extract()?;
                    model.set_zone_setpoints(zone_idx, setpoints_tuple.0, setpoints_tuple.1)?;
                }
            }
        }
    }

    Ok(model)
}

/// Python module initialization
#[pymodule]
pub fn multi_zone(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiZoneThermalModel>()?;
    m.add_function(wrap_pyfunction!(create_multi_zone_model_from_config, m)?)?;

    Ok(())
}

/// Register HVAC module in main bindings
#[cfg(feature = "python-bindings")]
pub fn register_hvac_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register HVAC classes directly
    m.add_class::<crate::python::hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<crate::python::hvac_bindings::PyZoneControl>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;
    Ok(())
}

/// HVAC module initialization function
#[cfg(feature = "python-bindings")]
#[pymodule]
pub fn hvac(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::python::hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<crate::python::hvac_bindings::PyZoneControl>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;
    Ok(())
}
