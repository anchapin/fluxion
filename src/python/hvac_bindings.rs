//! Python bindings for HVAC functionality
//!
//! This module provides PyO3 bindings for zone setpoints and control functionality

use crate::hvac::zone_control::{HVACStatus, ZoneControl};
use crate::hvac::zone_setpoints::ZoneSetpoints;
use crate::physics::cta::VectorField;
use crate::python::bindings::PyMultiZoneThermalModel;
use crate::thermal::ThermalModel;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

/// Python wrapper for ZoneSetpoints
#[pyclass(name = "ZoneSetpoints")]
pub struct PyZoneSetpoints {
    inner: ZoneSetpoints,
}

#[pymethods]
impl PyZoneSetpoints {
    /// Create new ZoneSetpoints with specified number of zones
    #[new]
    pub fn new(num_zones: usize) -> PyResult<Self> {
        if num_zones < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of zones must be at least 1",
            ));
        }

        Ok(PyZoneSetpoints {
            inner: ZoneSetpoints::new(num_zones),
        })
    }

    /// Set heating setpoint for a zone
    pub fn set_heating_setpoint(&mut self, zone_id: usize, temp: f64) -> PyResult<()> {
        self.inner
            .set_heating_setpoint(zone_id, temp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Set cooling setpoint for a zone
    pub fn set_cooling_setpoint(&mut self, zone_id: usize, temp: f64) -> PyResult<()> {
        self.inner
            .set_cooling_setpoint(zone_id, temp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Set deadband for a zone
    pub fn set_deadband(&mut self, zone_id: usize, deadband: f64) -> PyResult<()> {
        self.inner
            .set_deadband(zone_id, deadband)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Get heating setpoint for a zone
    pub fn get_heating_setpoint(&self, zone_id: usize) -> PyResult<f64> {
        if zone_id >= self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone ID {} out of range (0-{})",
                zone_id,
                self.inner.num_zones - 1
            )));
        }
        Ok(self.inner.get_heating_setpoint(zone_id))
    }

    /// Get cooling setpoint for a zone
    pub fn get_cooling_setpoint(&self, zone_id: usize) -> PyResult<f64> {
        if zone_id >= self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone ID {} out of range (0-{})",
                zone_id,
                self.inner.num_zones - 1
            )));
        }
        Ok(self.inner.get_cooling_setpoint(zone_id))
    }

    /// Get deadband for a zone
    pub fn get_deadband(&self, zone_id: usize) -> PyResult<f64> {
        if zone_id >= self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone ID {} out of range (0-{})",
                zone_id,
                self.inner.num_zones - 1
            )));
        }
        Ok(self.inner.get_deadband(zone_id))
    }

    /// Validate all setpoints
    pub fn validate(&self) -> PyResult<()> {
        self.inner
            .validate_setpoints()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Get number of zones
    pub fn num_zones(&self) -> usize {
        self.inner.num_zones
    }
}

/// Python wrapper for ZoneControl
#[pyclass(name = "ZoneControl")]
pub struct PyZoneControl {
    inner: Arc<Mutex<ZoneControl>>,
}

#[pymethods]
impl PyZoneControl {
    /// Create new ZoneControl from thermal model and setpoints
    #[new]
    pub fn new(
        thermal_model: &PyMultiZoneThermalModel,
        setpoints: &PyZoneSetpoints,
    ) -> PyResult<Self> {
        let thermal_model_arc = Arc::new(thermal_model.inner.clone());
        let setpoints_copy = ZoneSetpoints::new(setpoints.inner.num_zones);

        // Copy setpoints from Python wrapper to Rust struct
        for zone_id in 0..setpoints.inner.num_zones {
            let heating = setpoints.get_heating_setpoint(zone_id)?;
            let cooling = setpoints.get_cooling_setpoint(zone_id)?;
            let deadband = setpoints.get_deadband(zone_id)?;

            setpoints_copy
                .set_heating_setpoint(zone_id, heating)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            setpoints_copy
                .set_cooling_setpoint(zone_id, cooling)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            setpoints_copy
                .set_deadband(zone_id, deadband)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        }

        let zone_control = ZoneControl::new(thermal_model_arc, setpoints_copy);

        Ok(PyZoneControl {
            inner: Arc::new(Mutex::new(zone_control)),
        })
    }

    /// Update HVAC controls with current temperatures
    pub fn update_controls(&self, temperatures: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut control = self.inner.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to lock ZoneControl: {}", e))
        })?;

        let temp_vector = VectorField::new(temperatures.clone());
        let energy_input = control.update_zone_controls(&temp_vector);

        Ok(energy_input.as_slice().to_vec())
    }

    /// Get HVAC status for a zone
    pub fn get_zone_status(&self, zone_id: usize) -> PyResult<String> {
        let control = self.inner.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to lock ZoneControl: {}", e))
        })?;

        let status = control.get_zone_hvac_status(zone_id);

        let status_str = match status {
            HVACStatus::Heating => "heating",
            HVACStatus::Cooling => "cooling",
            HVACStatus::Off => "off",
        };

        Ok(status_str.to_string())
    }

    /// Get energy input for a zone
    pub fn get_energy_input(&self, zone_id: usize, current_temp: f64) -> PyResult<f64> {
        let control = self.inner.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to lock ZoneControl: {}", e))
        })?;

        let status = control.get_zone_hvac_status(zone_id);
        let energy = control.calculate_energy_input(zone_id, current_temp, &status);

        Ok(energy)
    }
}

/// Create ZoneSetpoints from configuration dictionary
#[pyfunction]
pub fn create_zone_setpoints(config: &Bound<'_, PyDict>) -> PyResult<PyZoneSetpoints> {
    let num_zones: usize = config
        .get_item("num_zones")
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'num_zones' in config"))?
        .extract()?;

    let mut setpoints = PyZoneSetpoints::new(num_zones)?;

    // Set zone setpoints if provided
    if let Ok(zone_configs) = config.get_item("zones") {
        let zones_dict: &Bound<'_, PyDict> = zone_configs.downcast()?;

        for (key, value) in zones_dict {
            let zone_key: String = key.extract()?;
            if zone_key.starts_with("zone_") {
                if let Ok(zone_idx) = zone_key[5..].parse::<usize>() {
                    let zone_dict: &Bound<'_, PyDict> = value.downcast()?;

                    if let Ok(heating) = zone_dict.get_item("heating") {
                        let heating_temp: f64 = heating.extract()?;
                        setpoints.set_heating_setpoint(zone_idx, heating_temp)?;
                    }

                    if let Ok(cooling) = zone_dict.get_item("cooling") {
                        let cooling_temp: f64 = cooling.extract()?;
                        setpoints.set_cooling_setpoint(zone_idx, cooling_temp)?;
                    }

                    if let Ok(deadband) = zone_dict.get_item("deadband") {
                        let deadband_val: f64 = deadband.extract()?;
                        setpoints.set_deadband(zone_idx, deadband_val)?;
                    }
                }
            }
        }
    }

    setpoints.validate()?;
    Ok(setpoints)
}

/// Python module initialization
#[pymodule]
pub fn hvac(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZoneSetpoints>()?;
    m.add_class::<PyZoneControl>()?;
    m.add_function(wrap_pyfunction!(create_zone_setpoints, m)?)?;

    Ok(())
}
