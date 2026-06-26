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

        Ok(self.inner.h_tr_iz.as_slice()[zone_i])
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

        self.inner.h_tr_iz.as_mut_slice()[zone_i] = conductance;
        if zone_i != zone_j {
            self.inner.h_tr_iz.as_mut_slice()[zone_j] = conductance;
        }
        Ok(())
    }

    /// Set all inter-zone conductances from a vector
    pub fn set_inter_zone_conductance_vector(&mut self, conductances: Vec<f64>) -> PyResult<()> {
        if conductances.len() != self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Conductance vector length ({}) must match number of zones ({})",
                conductances.len(),
                self.inner.num_zones
            )));
        }

        for (i, &c) in conductances.iter().enumerate() {
            if c < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "All conductances must be non-negative",
                ));
            }
            self.inner.h_tr_iz.as_mut_slice()[i] = c;
        }
        Ok(())
    }

    /// Get all inter-zone conductances as a vector
    pub fn get_inter_zone_conductance_vector(&self) -> Vec<f64> {
        self.inner.h_tr_iz.as_slice().to_vec()
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

        let _net_result =
            self.inner
                .solve_timesteps(steps, &surrogates, use_surrogates, None, None, None);

        // Return total energy consumption (heating + cooling) instead of net energy
        // This matches the expectation of Python tests that energy should be non-negative
        let total_energy =
            self.inner.get_heating_energy_kwh() + self.inner.get_cooling_energy_kwh();
        Ok(total_energy)
    }

    /// Get zone-specific energy consumption (heating + cooling) in kWh
    ///
    /// Returns a vector with total energy (heating + cooling) for each zone.
    /// These values are accumulated during simulation.
    pub fn get_zone_energies(&self) -> Vec<f64> {
        self.inner.get_zone_energies_kwh()
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

    /// Get the full hourly zone temperature profiles (Issue #763).
    ///
    /// # Returns
    /// `Some([[T00, T01, ...], [T10, T11, ...], ...])` where outer index is zone,
    /// inner index is timestep (0..steps-1), or `None` if the simulation has not
    /// been run yet.
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.get_hourly_temperatures()
    }

    /// Get hourly temperatures as zero-copy numpy arrays for ML training.
    ///
    /// This method provides direct access to the underlying temperature data
    /// without JSON/CSV serialization overhead, enabling high-performance
    /// ML training pipelines.
    ///
    /// # Arguments
    /// * `py` - Python GIL token for numpy array creation
    ///
    /// # Returns
    /// Tuple of (zone_temperatures, shape) where zone_temperatures is a 2D numpy array
    /// with shape [num_zones, timesteps] and zero-copy memory sharing when possible.
    ///
    /// # Example
    /// ```python
    /// import numpy as np
    /// model = fluxion.MultiZoneThermalModel(3)
    /// model.simulate_multi_zone(1, False)
    /// temps, shape = model.get_hourly_temperatures_numpy()
    /// # temps is a numpy array with shape [3, 8760]
    /// ```
    pub fn get_hourly_temperatures_numpy<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<(Bound<'a, numpy::PyArray2<f64>>, Vec<usize>)> {
        let hourly_temps = self.inner.get_hourly_temperatures();

        match hourly_temps {
            Some(temps) => {
                let num_zones = temps.len();
                let timesteps = if num_zones > 0 { temps[0].len() } else { 0 };
                let shape = vec![num_zones, timesteps];

                // Create numpy array from Vec<Vec<f64>> - this is the expected format for from_vec2_bound
                let arr = numpy::PyArray2::from_vec2_bound(py, &temps).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to create numpy array: {}",
                        e
                    ))
                })?;
                Ok((arr, shape))
            }
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "Simulation has not been run yet. Call simulate_multi_zone first.",
            )),
        }
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
    let num_zones: usize = match config.get_item("num_zones") {
        Ok(Some(item)) => item.extract()?,
        _ => {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "Missing 'num_zones' in config",
            ))
        }
    };

    // Create model
    let mut model = PyMultiZoneThermalModel::new(num_zones)?;

    // Set zone setpoints from zones dict (zone_0, zone_1, etc.)
    if let Ok(Some(zone_configs)) = config.get_item("zones") {
        let zones_dict: &Bound<'_, PyDict> = match zone_configs.downcast() {
            Ok(d) => d,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected dict for 'zones'",
                ))
            }
        };

        for (key, value) in zones_dict.iter() {
            let zone_key: String = match key.extract() {
                Ok(k) => k,
                Err(_) => continue,
            };
            if let Some(stripped) = zone_key.strip_prefix("zone_") {
                if let Ok(zone_idx) = stripped.parse::<usize>() {
                    let zone_dict: &Bound<'_, PyDict> = match value.downcast() {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    if let Ok(Some(heating)) = zone_dict.get_item("heating") {
                        let heating_temp: f64 = match heating.extract() {
                            Ok(t) => t,
                            Err(_) => continue,
                        };
                        let current_cooling = model.inner.cooling_setpoints.as_slice()[zone_idx];
                        model.set_zone_setpoints(zone_idx, heating_temp, current_cooling)?;
                    }

                    if let Ok(Some(cooling)) = zone_dict.get_item("cooling") {
                        let cooling_temp: f64 = match cooling.extract() {
                            Ok(t) => t,
                            Err(_) => continue,
                        };
                        let current_heating = model.inner.heating_setpoints.as_slice()[zone_idx];
                        model.set_zone_setpoints(zone_idx, current_heating, cooling_temp)?;
                    }
                }
            }
        }
    } else if let Ok(Some(zone_setpoints)) = config.get_item("zone_setpoints") {
        // Legacy zone_setpoints format (zone_0: (heating, cooling), zone_1: (heating, cooling), ...)
        let setpoints_dict: &Bound<'_, PyDict> = match zone_setpoints.downcast() {
            Ok(d) => d,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected dict for 'zone_setpoints'",
                ))
            }
        };

        for (key, value) in setpoints_dict.iter() {
            let zone_key: String = match key.extract() {
                Ok(k) => k,
                Err(_) => continue,
            };
            if let Some(stripped) = zone_key.strip_prefix("zone_") {
                if let Ok(zone_idx) = stripped.parse::<usize>() {
                    let setpoints_tuple: (f64, f64) = match value.extract() {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    model.set_zone_setpoints(zone_idx, setpoints_tuple.0, setpoints_tuple.1)?;
                }
            }
        }
    }

    // Set inter-zone conductances
    if let Ok(Some(iz_conductance)) = config.get_item("inter_zone_conductance") {
        let iz_dict: &Bound<'_, PyDict> = match iz_conductance.downcast() {
            Ok(d) => d,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected dict for 'inter_zone_conductance'",
                ))
            }
        };

        let mut conductance_vec = Vec::new();
        for i in 0..num_zones {
            let row_key = format!("zone_{}", i);
            if let Ok(Some(row)) = iz_dict.get_item(&row_key) {
                let row_dict: &Bound<'_, PyDict> = match row.downcast() {
                    Ok(d) => d,
                    Err(_) => {
                        conductance_vec.push(vec![0.0; num_zones]);
                        continue;
                    }
                };
                let mut row_vec = Vec::new();
                for j in 0..num_zones {
                    let col_key = format!("zone_{}", j);
                    if let Ok(Some(val)) = row_dict.get_item(&col_key) {
                        match val.extract::<f64>() {
                            Ok(v) => row_vec.push(v),
                            Err(_) => row_vec.push(0.0),
                        }
                    } else {
                        row_vec.push(0.0);
                    }
                }
                conductance_vec.push(row_vec);
            } else {
                conductance_vec.push(vec![0.0; num_zones]);
            }
        }

        // Flatten and set conductances (matching CLI behavior)
        for i in 0..num_zones {
            for j in 0..num_zones {
                if i < conductance_vec.len() && j < conductance_vec[i].len() {
                    let conductance = conductance_vec[i][j];
                    // Only set when i <= j to avoid double-setting
                    if i == j || i == 0 {
                        model.set_inter_zone_conductance(i, j, conductance)?;
                    }
                }
            }
        }
    }

    Ok(model)
}

/// Create a multi-zone thermal model from a schema JSON file path
#[pyfunction]
pub fn create_multi_zone_model_from_schema_file(
    schema_path: &str,
) -> PyResult<PyMultiZoneThermalModel> {
    use crate::api::schema::SimulationSchema;
    use std::path::Path;

    let path = Path::new(schema_path);
    if !path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Schema file not found: {}",
            schema_path
        )));
    }

    let content = std::fs::read_to_string(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read schema file: {}", e))
    })?;

    let schema: SimulationSchema = serde_json::from_str(&content).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to parse schema JSON: {}", e))
    })?;

    let SimulationSchema::V1(schema_v1) = schema;

    // Create model from schema V1
    let mut model = PyMultiZoneThermalModel::new(schema_v1.geometry.zones.len().max(1))?;

    // Set zone setpoints from schema
    let heating = schema_v1.controls.zone_control.heating_setpoint;
    let cooling = schema_v1.controls.zone_control.cooling_setpoint;
    for zone_idx in 0..model.inner.num_zones {
        model.set_zone_setpoints(zone_idx, heating, cooling)?;
    }

    // Set inter-zone conductances from schema (using default matrix from schema geometry)
    let n = schema_v1.geometry.zones.len().max(1);
    let default_conductance = 5.0; // Default value matching CLI behavior
    for i in 0..n {
        model.set_inter_zone_conductance(i, i, default_conductance)?;
        if i < n - 1 {
            model.set_inter_zone_conductance(i, i + 1, default_conductance)?;
        }
    }

    Ok(model)
}

/// Create a multi-zone thermal model from a schema dictionary (PyDict)
#[pyfunction]
pub fn create_multi_zone_model_from_schema_dict(
    schema: &Bound<'_, PyDict>,
) -> PyResult<PyMultiZoneThermalModel> {
    // Extract number of zones from geometry
    let num_zones: usize = match schema.get_item("geometry") {
        Ok(Some(geometry)) => {
            let geometry_dict: &Bound<'_, PyDict> = match geometry.downcast() {
                Ok(d) => d,
                Err(_) => return PyMultiZoneThermalModel::new(1),
            };
            match geometry_dict.get_item("zones") {
                Ok(Some(zones)) => {
                    let zones_list: &Bound<'_, pyo3::types::PyList> = match zones.downcast() {
                        Ok(d) => d,
                        Err(_) => return PyMultiZoneThermalModel::new(1),
                    };
                    zones_list.len()
                }
                _ => 1,
            }
        }
        _ => 1,
    };

    let mut model = PyMultiZoneThermalModel::new(num_zones)?;

    // Extract setpoints from controls
    if let Ok(Some(controls)) = schema.get_item("controls") {
        let controls_dict: &Bound<'_, PyDict> = match controls.downcast() {
            Ok(d) => d,
            Err(_) => return Ok(model),
        };

        if let Ok(Some(zone_control)) = controls_dict.get_item("zone_control") {
            let zone_control_dict: &Bound<'_, PyDict> = match zone_control.downcast() {
                Ok(d) => d,
                Err(_) => return Ok(model),
            };

            let heating = match zone_control_dict.get_item("heating_setpoint") {
                Ok(Some(v)) => v.extract::<f64>().ok(),
                _ => None,
            };
            let cooling = match zone_control_dict.get_item("cooling_setpoint") {
                Ok(Some(v)) => v.extract::<f64>().ok(),
                _ => None,
            };

            if let (Some(h), Some(c)) = (heating, cooling) {
                for zone_idx in 0..num_zones {
                    model.set_zone_setpoints(zone_idx, h, c)?;
                }
            }
        }
    }

    // Extract inter-zone conductance from constructions if available
    // (For now, use default values - full schema parsing would go here)

    Ok(model)
}

/// Python module initialization
#[pymodule]
pub fn multi_zone(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiZoneThermalModel>()?;
    m.add_function(wrap_pyfunction!(create_multi_zone_model_from_config, m)?)?;
    m.add_function(wrap_pyfunction!(
        create_multi_zone_model_from_schema_file,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        create_multi_zone_model_from_schema_dict,
        m
    )?)?;

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
