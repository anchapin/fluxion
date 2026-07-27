//! Python bindings for HVAC functionality
//!
//! This module provides PyO3 bindings for zone setpoints, control, schedule
//! functionality, and deep HVAC equipment configuration (Issue #1797):
//! equipment types (chiller, boiler, heat pump, VAV terminal, CAV system),
//! system-type / operating-mode enums, and the detailed airside
//! [`VavTerminalUnit`] with its control and performance structs.

use crate::physics::cta::VectorField;
use crate::python::bindings::PyMultiZoneThermalModel;
use crate::sim::hvac::cooling_coil::CoolingCoil;
use crate::sim::hvac::equipment::{Boiler, Chiller, HVACMode, VariableCapacityEquipment};
use crate::sim::hvac::heating_coil::{HeatingCoilComponent, HeatingCoilControl};
use crate::sim::hvac::vav_terminal::{
    VavOperatingMode, VavTerminal, VavTerminalControl, VavTerminalPerformance, VavTerminalUnit,
};
use crate::sim::hvac::zones::schedule::{DailySchedule, HVACSchedule, ScheduleType};
use crate::sim::hvac::zones::zone_control::{HVACStatus, ZoneControl};
use crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints;
use crate::sim::hvac::{CAVSystem, HVACSystemType, HeatPump, HeatPumpMode, VAVTerminal};
use crate::thermal::thermal_model::ThermalModel;
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
        if zone_id >= self.inner.num_zones() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone ID {} out of range (0-{})",
                zone_id,
                self.inner.num_zones() - 1
            )));
        }
        Ok(self.inner.get_heating_setpoint(zone_id))
    }

    /// Get cooling setpoint for a zone
    pub fn get_cooling_setpoint(&self, zone_id: usize) -> PyResult<f64> {
        if zone_id >= self.inner.num_zones() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone ID {} out of range (0-{})",
                zone_id,
                self.inner.num_zones() - 1
            )));
        }
        Ok(self.inner.get_cooling_setpoint(zone_id))
    }

    /// Get deadband for a zone
    pub fn get_deadband(&self, zone_id: usize) -> PyResult<f64> {
        if zone_id >= self.inner.num_zones() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone ID {} out of range (0-{})",
                zone_id,
                self.inner.num_zones() - 1
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
        self.inner.num_zones()
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
        let num_zones = thermal_model.get_inner_num_zones();
        let thermal_model_arc = Arc::new(ThermalModel::new(num_zones, 20.0));
        let mut setpoints_copy = ZoneSetpoints::new(setpoints.num_zones());

        // Copy setpoints from Python wrapper to Rust struct
        for zone_id in 0..setpoints.num_zones() {
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
        let mut control = self.inner.lock().map_err(|e| {
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
    let num_zones_value = match config.get_item("num_zones")? {
        Some(value) => value,
        None => {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "Missing 'num_zones' in config",
            ))
        }
    };
    let num_zones: usize = num_zones_value.extract()?;

    let mut setpoints = PyZoneSetpoints::new(num_zones)?;

    // Set zone setpoints if provided
    if let Ok(Some(zone_configs)) = config.get_item("zones") {
        let zones_dict: &Bound<'_, PyDict> = zone_configs
            .downcast()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected dict for 'zones'"))?;

        for (key, value) in zones_dict {
            let zone_key: String = key.extract()?;
            if let Some(stripped) = zone_key.strip_prefix("zone_") {
                if let Ok(zone_idx) = stripped.parse::<usize>() {
                    let zone_dict: &Bound<'_, PyDict> = value.downcast()?;

                    if let Ok(Some(heating)) = zone_dict.get_item("heating") {
                        let heating_temp: f64 = heating.extract()?;
                        setpoints.set_heating_setpoint(zone_idx, heating_temp)?;
                    }

                    if let Ok(Some(cooling)) = zone_dict.get_item("cooling") {
                        let cooling_temp: f64 = cooling.extract()?;
                        setpoints.set_cooling_setpoint(zone_idx, cooling_temp)?;
                    }

                    if let Ok(Some(deadband)) = zone_dict.get_item("deadband") {
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

/// Python wrapper for DailySchedule
#[pyclass(name = "DailySchedule")]
pub struct PyDailySchedule {
    inner: DailySchedule,
}

#[pymethods]
impl PyDailySchedule {
    #[new]
    pub fn new(name: String, schedule_type: String) -> PyResult<Self> {
        let schedule_type = match schedule_type.as_str() {
            "Constant" => ScheduleType::Constant,
            "DailyCycle" => ScheduleType::DailyCycle,
            "Weekly" => ScheduleType::Weekly,
            "Custom" => ScheduleType::Custom,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid schedule type. Use: Constant, DailyCycle, Weekly, or Custom",
                ))
            }
        };

        let mut schedule = match schedule_type {
            ScheduleType::Weekly => DailySchedule::weekly(name),
            _ => DailySchedule::new(),
        };
        schedule.schedule_type = schedule_type;

        Ok(PyDailySchedule { inner: schedule })
    }

    pub fn set_hour(&mut self, hour: usize, value: f64) -> PyResult<()> {
        self.inner.set_hour(hour, value);
        Ok(())
    }

    pub fn fill_range(&mut self, start_hour: usize, end_hour: usize, value: f64) -> PyResult<()> {
        self.inner.fill_range(start_hour, end_hour, value);
        Ok(())
    }

    pub fn value(&self, hour: usize) -> f64 {
        self.inner.value(hour)
    }

    #[staticmethod]
    pub fn constant(value: f64) -> Self {
        PyDailySchedule {
            inner: DailySchedule::constant(value),
        }
    }
}

/// Python wrapper for HVACSchedule
#[pyclass(name = "HVACSchedule")]
pub struct PyHVACSchedule {
    inner: HVACSchedule,
}

#[pymethods]
impl PyHVACSchedule {
    #[new]
    pub fn new() -> Self {
        PyHVACSchedule {
            inner: HVACSchedule::new(),
        }
    }

    #[staticmethod]
    pub fn constant_schedule(heating_sp: f64, cooling_sp: f64) -> Self {
        PyHVACSchedule {
            inner: HVACSchedule::constant_schedule(heating_sp, cooling_sp),
        }
    }

    #[staticmethod]
    pub fn setback_schedule(
        day_heat: f64,
        night_heat: f64,
        cool_sp: f64,
        night_start: usize,
        night_end: usize,
    ) -> Self {
        PyHVACSchedule {
            inner: HVACSchedule::setback_schedule(
                day_heat,
                night_heat,
                cool_sp,
                night_start,
                night_end,
            ),
        }
    }

    #[staticmethod]
    pub fn with_operating_hours(
        heating_sp: f64,
        cooling_sp: f64,
        start_hour: usize,
        end_hour: usize,
    ) -> Self {
        PyHVACSchedule {
            inner: HVACSchedule::with_operating_hours(heating_sp, cooling_sp, start_hour, end_hour),
        }
    }

    #[staticmethod]
    pub fn free_floating() -> Self {
        PyHVACSchedule {
            inner: HVACSchedule::free_floating(),
        }
    }

    pub fn is_free_floating(&self) -> bool {
        self.inner.is_free_floating()
    }

    pub fn heating_setpoint(&self, hour: usize) -> f64 {
        self.inner.heating_setpoint(hour)
    }

    pub fn cooling_setpoint(&self, hour: usize) -> f64 {
        self.inner.cooling_setpoint(hour)
    }

    pub fn get_heating_schedule(&self) -> PyDailySchedule {
        PyDailySchedule {
            inner: self.inner.heating.clone(),
        }
    }

    pub fn get_cooling_schedule(&self) -> PyDailySchedule {
        PyDailySchedule {
            inner: self.inner.cooling.clone(),
        }
    }
}

impl Default for PyHVACSchedule {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Deep HVAC configuration (Issue #1797)
//
// The classes below expose HVAC equipment types, system-type / operating-mode
// enums, and the detailed airside `VavTerminalUnit` so Python users can build
// and inspect complete VAV/CAV/heat-pump/chiller/boiler systems from Python.
//
// Memory model: every Python wrapper holds an **owned** Rust value (no
// references back into a parent model). Getters clone primitives out; setters
// clone primitives in. This matches the snapshot/owned-value model documented
// in `model_bindings.rs`.
// =============================================================================

/// HVAC system type.
///
/// Mirrors [`crate::sim::hvac::HVACSystemType`]. Exposed as a Python enum so
/// users can compare with `==` (e.g. `system.system_type == HVACSystemType.VAV`).
#[pyclass(name = "HVACSystemType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyHVACSystemType {
    Simple,
    VAV,
    CAV,
    HeatPump,
    Ideal,
}

impl From<HVACSystemType> for PyHVACSystemType {
    fn from(t: HVACSystemType) -> Self {
        match t {
            HVACSystemType::Simple => PyHVACSystemType::Simple,
            HVACSystemType::VAV => PyHVACSystemType::VAV,
            HVACSystemType::CAV => PyHVACSystemType::CAV,
            HVACSystemType::HeatPump => PyHVACSystemType::HeatPump,
            HVACSystemType::Ideal => PyHVACSystemType::Ideal,
        }
    }
}

impl From<PyHVACSystemType> for HVACSystemType {
    fn from(t: PyHVACSystemType) -> Self {
        match t {
            PyHVACSystemType::Simple => HVACSystemType::Simple,
            PyHVACSystemType::VAV => HVACSystemType::VAV,
            PyHVACSystemType::CAV => HVACSystemType::CAV,
            PyHVACSystemType::HeatPump => HVACSystemType::HeatPump,
            PyHVACSystemType::Ideal => HVACSystemType::Ideal,
        }
    }
}

/// HVAC operating mode (heating / cooling / off).
///
/// Mirrors [`crate::sim::hvac::equipment::HVACMode`].
#[pyclass(name = "HVACMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyHVACMode {
    Heating,
    Cooling,
    Off,
}

impl From<HVACMode> for PyHVACMode {
    fn from(m: HVACMode) -> Self {
        match m {
            HVACMode::Heating => PyHVACMode::Heating,
            HVACMode::Cooling => PyHVACMode::Cooling,
            HVACMode::Off => PyHVACMode::Off,
        }
    }
}

impl From<PyHVACMode> for HVACMode {
    fn from(m: PyHVACMode) -> Self {
        match m {
            PyHVACMode::Heating => HVACMode::Heating,
            PyHVACMode::Cooling => HVACMode::Cooling,
            PyHVACMode::Off => HVACMode::Off,
        }
    }
}

/// Heat-pump operating mode.
///
/// Mirrors [`crate::sim::hvac::HeatPumpMode`].
#[pyclass(name = "HeatPumpMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyHeatPumpMode {
    Heating,
    Cooling,
    Off,
}

impl From<HeatPumpMode> for PyHeatPumpMode {
    fn from(m: HeatPumpMode) -> Self {
        match m {
            HeatPumpMode::Heating => PyHeatPumpMode::Heating,
            HeatPumpMode::Cooling => PyHeatPumpMode::Cooling,
            HeatPumpMode::Off => PyHeatPumpMode::Off,
        }
    }
}

impl From<PyHeatPumpMode> for HeatPumpMode {
    fn from(m: PyHeatPumpMode) -> Self {
        match m {
            PyHeatPumpMode::Heating => HeatPumpMode::Heating,
            PyHeatPumpMode::Cooling => HeatPumpMode::Cooling,
            PyHeatPumpMode::Off => HeatPumpMode::Off,
        }
    }
}

/// Operating mode of a VAV terminal unit.
///
/// Mirrors [`crate::sim::hvac::vav_terminal::VavOperatingMode`].
#[pyclass(name = "VavOperatingMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyVavOperatingMode {
    Cooling,
    Heating,
    Deadband,
}

impl From<VavOperatingMode> for PyVavOperatingMode {
    fn from(m: VavOperatingMode) -> Self {
        match m {
            VavOperatingMode::Cooling => PyVavOperatingMode::Cooling,
            VavOperatingMode::Heating => PyVavOperatingMode::Heating,
            VavOperatingMode::Deadband => PyVavOperatingMode::Deadband,
        }
    }
}

impl From<PyVavOperatingMode> for VavOperatingMode {
    fn from(m: PyVavOperatingMode) -> Self {
        match m {
            PyVavOperatingMode::Cooling => VavOperatingMode::Cooling,
            PyVavOperatingMode::Heating => VavOperatingMode::Heating,
            PyVavOperatingMode::Deadband => VavOperatingMode::Deadband,
        }
    }
}

// -----------------------------------------------------------------------------
// Equipment types
// -----------------------------------------------------------------------------

/// Chiller equipment model (cooling-only, polynomial efficiency curves).
///
/// Python wrapper for [`crate::sim::hvac::equipment::Chiller`].
#[pyclass(name = "Chiller")]
pub struct PyChiller {
    pub(crate) inner: Chiller,
}

#[pymethods]
impl PyChiller {
    /// Create a new chiller.
    ///
    /// Args:
    ///     id: Equipment identifier.
    ///     cooling_capacity: Rated cooling capacity [W].
    ///     cooling_cop: Rated cooling COP at design conditions.
    ///     design_temp: Design outdoor temperature for cooling [°C].
    #[new]
    pub fn new(id: String, cooling_capacity: f64, cooling_cop: f64, design_temp: f64) -> Self {
        PyChiller {
            inner: Chiller::new(id, cooling_capacity, cooling_cop, design_temp),
        }
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn cooling_capacity(&self) -> f64 {
        self.inner.cooling_capacity
    }

    #[getter]
    pub fn cooling_cop(&self) -> f64 {
        self.inner.cooling_cop
    }

    #[getter]
    pub fn design_temp(&self) -> f64 {
        self.inner.design_temp
    }

    /// Calculate actual capacity [W] at a part-load ratio and outdoor temp.
    pub fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.inner.calculate_capacity(plr, outdoor_temp)
    }

    /// Calculate power consumption [W] for a load and outdoor temp in cooling mode.
    pub fn calculate_power(&self, load: f64, outdoor_temp: f64) -> f64 {
        self.inner
            .calculate_power(load, outdoor_temp, HVACMode::Cooling)
    }

    /// Current part-load ratio (0.0 to 1.0).
    #[getter]
    pub fn current_plr(&self) -> f64 {
        self.inner.current_plr()
    }
}

/// Boiler equipment model (heating-only, polynomial efficiency curves).
///
/// Python wrapper for [`crate::sim::hvac::equipment::Boiler`].
#[pyclass(name = "Boiler")]
pub struct PyBoiler {
    pub(crate) inner: Boiler,
}

#[pymethods]
impl PyBoiler {
    /// Create a new boiler.
    ///
    /// Args:
    ///     id: Equipment identifier.
    ///     heating_capacity: Rated heating capacity [W].
    ///     efficiency: Rated efficiency (AFUE, 0.0 to 1.0).
    ///     design_temp: Design outdoor temperature for heating [°C].
    #[new]
    pub fn new(id: String, heating_capacity: f64, efficiency: f64, design_temp: f64) -> Self {
        PyBoiler {
            inner: Boiler::new(id, heating_capacity, efficiency, design_temp),
        }
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn heating_capacity(&self) -> f64 {
        self.inner.heating_capacity
    }

    #[getter]
    pub fn efficiency(&self) -> f64 {
        self.inner.efficiency
    }

    #[getter]
    pub fn design_temp(&self) -> f64 {
        self.inner.design_temp
    }

    /// Calculate actual capacity [W] at a part-load ratio and outdoor temp.
    pub fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.inner.calculate_capacity(plr, outdoor_temp)
    }

    /// Calculate power consumption [W] for a load and outdoor temp in heating mode.
    pub fn calculate_power(&self, load: f64, outdoor_temp: f64) -> f64 {
        self.inner
            .calculate_power(load, outdoor_temp, HVACMode::Heating)
    }

    #[getter]
    pub fn current_plr(&self) -> f64 {
        self.inner.current_plr()
    }
}

/// Heat-pump system with COP curves.
///
/// Python wrapper for [`crate::sim::hvac::HeatPump`]. Exposes capacity, COP,
/// and power calculations plus mode selection from setpoints.
#[pyclass(name = "HeatPump")]
pub struct PyHeatPump {
    pub(crate) inner: HeatPump,
}

#[pymethods]
impl PyHeatPump {
    /// Create a new heat pump.
    ///
    /// Args:
    ///     id: System identifier.
    ///     heating_capacity: Rated heating capacity [W].
    ///     cooling_capacity: Rated cooling capacity [W].
    ///     heating_cop: Rated heating COP at design conditions.
    ///     cooling_cop: Rated cooling COP (EER) at design conditions.
    #[new]
    pub fn new(
        id: String,
        heating_capacity: f64,
        cooling_capacity: f64,
        heating_cop: f64,
        cooling_cop: f64,
    ) -> Self {
        PyHeatPump {
            inner: HeatPump::new(
                id,
                heating_capacity,
                cooling_capacity,
                heating_cop,
                cooling_cop,
            ),
        }
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn heating_capacity(&self) -> f64 {
        self.inner.heating_capacity
    }

    #[getter]
    pub fn cooling_capacity(&self) -> f64 {
        self.inner.cooling_capacity
    }

    #[getter]
    pub fn heating_cop(&self) -> f64 {
        self.inner.heating_cop
    }

    #[getter]
    pub fn cooling_cop(&self) -> f64 {
        self.inner.cooling_cop
    }

    /// Current operating mode ("heating", "cooling", or "off").
    #[getter]
    pub fn mode(&self) -> String {
        match self.inner.mode {
            HeatPumpMode::Heating => "heating",
            HeatPumpMode::Cooling => "cooling",
            HeatPumpMode::Off => "off",
        }
        .to_string()
    }

    /// Actual heating COP at a given outdoor temperature.
    pub fn heating_cop_at_temperature(&self, outdoor_temp: f64) -> f64 {
        self.inner.heating_cop_at_temperature(outdoor_temp)
    }

    /// Actual cooling COP at a given outdoor temperature.
    pub fn cooling_cop_at_temperature(&self, outdoor_temp: f64) -> f64 {
        self.inner.cooling_cop_at_temperature(outdoor_temp)
    }

    /// Heating power consumption [W] at a given outdoor temperature.
    pub fn heating_power(&self, outdoor_temp: f64) -> f64 {
        self.inner.heating_power(outdoor_temp)
    }

    /// Cooling power consumption [W] at a given outdoor temperature.
    pub fn cooling_power(&self, outdoor_temp: f64) -> f64 {
        self.inner.cooling_power(outdoor_temp)
    }

    /// Set the operating mode from zone temperature and setpoints.
    pub fn set_mode(&mut self, zone_temp: f64, heating_sp: f64, cooling_sp: f64) {
        self.inner.set_mode(zone_temp, heating_sp, cooling_sp);
    }
}

/// VAV terminal unit (high-level model with reheat coil).
///
/// Python wrapper for [`crate::sim::hvac::VAVTerminal`]. This is the
/// simplified envelope-level model; for the detailed airside terminal see
/// [`PyVavTerminalUnit`].
#[pyclass(name = "VAVTerminal")]
pub struct PyVAVTerminal {
    pub(crate) inner: VAVTerminal,
}

#[pymethods]
impl PyVAVTerminal {
    /// Create a new VAV terminal unit.
    ///
    /// Args:
    ///     id: Terminal unit identifier.
    ///     zone_id: Index of the zone served by this terminal.
    ///     max_airflow: Maximum air flow rate [m³/s].
    #[new]
    pub fn new(id: String, zone_id: usize, max_airflow: f64) -> Self {
        PyVAVTerminal {
            inner: VAVTerminal::new(id, zone_id, max_airflow),
        }
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn zone_id(&self) -> usize {
        self.inner.zone_id
    }

    #[getter]
    pub fn max_airflow(&self) -> f64 {
        self.inner.max_airflow
    }

    #[getter]
    pub fn min_airflow(&self) -> f64 {
        self.inner.min_airflow
    }

    #[getter]
    pub fn reheat_capacity(&self) -> f64 {
        self.inner.reheat_capacity
    }

    /// Set the minimum air flow rate [m³/s].
    pub fn set_min_airflow(&mut self, min_airflow: f64) {
        self.inner.min_airflow = min_airflow;
    }

    /// Set the reheat coil capacity [W].
    pub fn set_reheat_capacity(&mut self, reheat_capacity: f64) {
        self.inner.reheat_capacity = reheat_capacity;
    }

    /// Calculate heating demand from the reheat coil [W].
    pub fn reheat_demand(&self, supply_temp: f64, zone_temp: f64) -> f64 {
        self.inner.reheat_demand(supply_temp, zone_temp)
    }
}

/// CAV (Constant Air Volume) system.
///
/// Python wrapper for [`crate::sim::hvac::CAVSystem`].
#[pyclass(name = "CAVSystem")]
pub struct PyCAVSystem {
    pub(crate) inner: CAVSystem,
}

#[pymethods]
impl PyCAVSystem {
    /// Create a new CAV system.
    ///
    /// Args:
    ///     id: System identifier.
    ///     design_airflow: Design air flow rate [m³/s].
    #[new]
    pub fn new(id: String, design_airflow: f64) -> Self {
        PyCAVSystem {
            inner: CAVSystem::new(id, design_airflow),
        }
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn design_airflow(&self) -> f64 {
        self.inner.design_airflow
    }

    #[getter]
    pub fn fan_power(&self) -> f64 {
        self.inner.fan_power
    }

    #[getter]
    pub fn fan_efficiency(&self) -> f64 {
        self.inner.fan_efficiency
    }

    #[getter]
    pub fn heating_capacity(&self) -> f64 {
        self.inner.heating_capacity
    }

    #[getter]
    pub fn cooling_capacity(&self) -> f64 {
        self.inner.cooling_capacity
    }

    pub fn set_fan_power(&mut self, fan_power: f64) {
        self.inner.fan_power = fan_power;
    }

    pub fn set_fan_efficiency(&mut self, fan_efficiency: f64) {
        self.inner.fan_efficiency = fan_efficiency;
    }

    pub fn set_heating_capacity(&mut self, heating_capacity: f64) {
        self.inner.heating_capacity = heating_capacity;
    }

    pub fn set_cooling_capacity(&mut self, cooling_capacity: f64) {
        self.inner.cooling_capacity = cooling_capacity;
    }

    /// Fan power consumption [W] (accounting for efficiency).
    pub fn fan_power_consumption(&self) -> f64 {
        self.inner.fan_power_consumption()
    }
}

// -----------------------------------------------------------------------------
// Detailed airside VAV terminal unit (Issue #1797 / T2.5)
// -----------------------------------------------------------------------------

/// Reference VAV terminal unit with fan, cooling coil, and optional reheat.
///
/// Python wrapper for [`crate::sim::hvac::vav_terminal::VavTerminalUnit`].
/// This is the detailed airside model that composes a fan, a cooling coil,
/// and an optional reheat coil into a single zone terminal.
///
/// Example — build a VAV system from Python:
///
/// ```python
/// from fluxion import VavTerminalUnit, VavOperatingMode
///
/// terminal = VavTerminalUnit(
///     id="VAV-1",
///     zone_id=0,
///     max_airflow=0.5,
///     cooling_capacity=8000.0,
///     reheat_capacity=5000.0,
/// )
/// assert terminal.max_airflow == 0.5
/// assert terminal.has_reheat
/// ```
#[pyclass(name = "VavTerminalUnit")]
pub struct PyVavTerminalUnit {
    pub(crate) inner: VavTerminalUnit,
}

#[pymethods]
impl PyVavTerminalUnit {
    /// Create a detailed VAV terminal unit with auto-sized fan and coils.
    ///
    /// Args:
    ///     id: Terminal-unit identifier.
    ///     zone_id: Index of the zone served by this terminal.
    ///     max_airflow: Maximum (design) volumetric airflow [m³/s].
    ///     cooling_capacity: Rated total cooling capacity [W].
    ///     reheat_capacity: Rated reheat capacity [W]. Use 0.0 for a
    ///         cooling-only terminal (no reheat coil).
    #[new]
    #[pyo3(signature = (id, zone_id, max_airflow, cooling_capacity, reheat_capacity=0.0))]
    pub fn new(
        id: String,
        zone_id: usize,
        max_airflow: f64,
        cooling_capacity: f64,
        reheat_capacity: f64,
    ) -> PyResult<Self> {
        if max_airflow <= 0.0 || !max_airflow.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_airflow must be finite and positive",
            ));
        }
        if cooling_capacity <= 0.0 || !cooling_capacity.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cooling_capacity must be finite and positive",
            ));
        }

        // Design dry-air mass flow from volumetric flow at standard density.
        let design_mass_flow = max_airflow * crate::sim::hvac::fan::STANDARD_AIR_DENSITY_KG_PER_M3;

        let cooling_coil = CoolingCoil::new(
            format!("{id}-CC"),
            cooling_capacity,
            0.75, // Typical sensible heat ratio
            0.15, // Typical bypass factor
            10.0, // Apparatus dew point [°C]
            design_mass_flow,
        );

        let reheat_coil = if reheat_capacity > 0.0 {
            Some(HeatingCoilComponent::new(
                format!("{id}-HC"),
                reheat_capacity,
                design_mass_flow,
            ))
        } else {
            None
        };

        let unit = VavTerminalUnit::new(id, zone_id, max_airflow, cooling_coil, reheat_coil);

        Ok(PyVavTerminalUnit { inner: unit })
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn zone_id(&self) -> usize {
        self.inner.zone_id
    }

    /// Maximum volumetric airflow [m³/s].
    #[getter]
    pub fn max_airflow(&self) -> f64 {
        self.inner.max_airflow_m3_per_s()
    }

    /// Minimum volumetric airflow [m³/s].
    #[getter]
    pub fn min_airflow(&self) -> f64 {
        self.inner.min_airflow_m3_per_s()
    }

    /// Minimum airflow as a fraction of maximum ∈ [0, 1].
    #[getter]
    pub fn min_airflow_ratio(&self) -> f64 {
        self.inner.min_airflow_ratio
    }

    /// Override the minimum airflow ratio (turndown fraction).
    pub fn set_min_airflow_ratio(&mut self, ratio: f64) {
        self.inner = self.inner.clone().with_min_airflow_ratio(ratio);
    }

    /// Rated total cooling capacity of the cooling coil [W].
    #[getter]
    pub fn rated_cooling_capacity(&self) -> f64 {
        self.inner.rated_cooling_capacity_w()
    }

    /// Rated reheat capacity [W] (0.0 when no reheat coil is present).
    #[getter]
    pub fn rated_reheat_capacity(&self) -> f64 {
        self.inner.rated_reheat_capacity_w()
    }

    /// Whether the terminal is equipped with a reheat coil.
    #[getter]
    pub fn has_reheat(&self) -> bool {
        self.inner.has_reheat()
    }

    /// Last persisted damper position ∈ [0, 1].
    #[getter]
    pub fn current_damper_position(&self) -> f64 {
        self.inner.current_damper_position()
    }
}

/// Control signal for a [`PyVavTerminalUnit`].
///
/// Encapsulates the damper position and coil on/off states. Mirrors
/// [`crate::sim::hvac::vav_terminal::VavTerminalControl`].
#[pyclass(name = "VavTerminalControl")]
#[derive(Clone)]
pub struct PyVavTerminalControl {
    pub(crate) inner: VavTerminalControl,
}

#[pymethods]
impl PyVavTerminalControl {
    /// Build a control signal.
    ///
    /// Args:
    ///     damper_position: Damper position ∈ [0, 1]. `0` = minimum airflow,
    ///         `1` = maximum.
    ///     cooling_active: Whether the cooling coil is active at full effectiveness.
    ///     reheat_setpoint: Supply-air temperature setpoint for the reheat coil
    ///         [°C]. Use `None` (the default) to turn the reheat coil off.
    #[new]
    #[pyo3(signature = (damper_position, cooling_active=false, reheat_setpoint=None))]
    pub fn new(damper_position: f64, cooling_active: bool, reheat_setpoint: Option<f64>) -> Self {
        let reheat = reheat_setpoint.map(HeatingCoilControl::LeavingTempSetpoint);
        PyVavTerminalControl {
            inner: VavTerminalControl {
                damper_position,
                cooling_active,
                reheat,
            },
        }
    }

    /// Cooling-mode control: cooling coil active at the given damper position.
    #[staticmethod]
    pub fn cooling(damper_position: f64) -> Self {
        PyVavTerminalControl {
            inner: VavTerminalControl::cooling(damper_position),
        }
    }

    /// Heating (reheat) mode: damper at minimum, reheat driving to the setpoint.
    #[staticmethod]
    pub fn heating(supply_setpoint_c: f64) -> Self {
        PyVavTerminalControl {
            inner: VavTerminalControl::heating(supply_setpoint_c),
        }
    }

    /// Deadband: damper at minimum, all coils off.
    #[staticmethod]
    pub fn deadband() -> Self {
        PyVavTerminalControl {
            inner: VavTerminalControl::deadband(),
        }
    }

    #[getter]
    pub fn damper_position(&self) -> f64 {
        self.inner.damper_position
    }

    #[getter]
    pub fn cooling_active(&self) -> bool {
        self.inner.cooling_active
    }

    /// Resolved operating mode.
    #[getter]
    pub fn mode(&self) -> PyVavOperatingMode {
        self.inner.mode().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "VavTerminalControl(damper_position={}, cooling_active={}, mode={})",
            self.inner.damper_position,
            self.inner.cooling_active,
            match self.inner.mode() {
                VavOperatingMode::Cooling => "Cooling",
                VavOperatingMode::Heating => "Heating",
                VavOperatingMode::Deadband => "Deadband",
            }
        )
    }
}

/// Performance result of a VAV terminal calculation.
///
/// Mirrors [`crate::sim::hvac::vav_terminal::VavTerminalPerformance`]. Only
/// the key scalar capacities and flow rates are exposed; the full moist-air
/// supply state is summarized as a supply dry-bulb temperature for
/// Python-friendliness.
#[pyclass(name = "VavTerminalPerformance")]
pub struct PyVavTerminalPerformance {
    mode: VavOperatingMode,
    damper_position: f64,
    fan_speed_fraction: f64,
    volumetric_flow_m3_per_s: f64,
    dry_air_mass_flow_kg_per_s: f64,
    cooling_total_capacity_w: f64,
    cooling_sensible_capacity_w: f64,
    cooling_latent_capacity_w: f64,
    cooling_shr: f64,
    reheat_capacity_w: f64,
    fan_shaft_power_w: f64,
    fan_motor_power_w: f64,
    fan_heat_w: f64,
    condensate_rate_kg_per_s: f64,
    supply_dry_bulb_c: f64,
}

impl From<VavTerminalPerformance> for PyVavTerminalPerformance {
    fn from(p: VavTerminalPerformance) -> Self {
        let supply_dry_bulb_c = p.supply_air.dry_bulb_c;
        PyVavTerminalPerformance {
            mode: p.mode,
            damper_position: p.damper_position,
            fan_speed_fraction: p.fan_speed_fraction,
            volumetric_flow_m3_per_s: p.volumetric_flow_m3_per_s,
            dry_air_mass_flow_kg_per_s: p.dry_air_mass_flow_kg_per_s,
            cooling_total_capacity_w: p.cooling_total_capacity_w,
            cooling_sensible_capacity_w: p.cooling_sensible_capacity_w,
            cooling_latent_capacity_w: p.cooling_latent_capacity_w,
            cooling_shr: p.cooling_shr,
            reheat_capacity_w: p.reheat_capacity_w,
            fan_shaft_power_w: p.fan_shaft_power_w,
            fan_motor_power_w: p.fan_motor_power_w,
            fan_heat_w: p.fan_heat_w,
            condensate_rate_kg_per_s: p.condensate_rate_kg_per_s,
            supply_dry_bulb_c,
        }
    }
}

#[pymethods]
impl PyVavTerminalPerformance {
    #[getter]
    pub fn mode(&self) -> PyVavOperatingMode {
        self.mode.into()
    }

    #[getter]
    pub fn damper_position(&self) -> f64 {
        self.damper_position
    }

    #[getter]
    pub fn fan_speed_fraction(&self) -> f64 {
        self.fan_speed_fraction
    }

    #[getter]
    pub fn volumetric_flow_m3_per_s(&self) -> f64 {
        self.volumetric_flow_m3_per_s
    }

    #[getter]
    pub fn dry_air_mass_flow_kg_per_s(&self) -> f64 {
        self.dry_air_mass_flow_kg_per_s
    }

    #[getter]
    pub fn cooling_total_capacity_w(&self) -> f64 {
        self.cooling_total_capacity_w
    }

    #[getter]
    pub fn cooling_sensible_capacity_w(&self) -> f64 {
        self.cooling_sensible_capacity_w
    }

    #[getter]
    pub fn cooling_latent_capacity_w(&self) -> f64 {
        self.cooling_latent_capacity_w
    }

    #[getter]
    pub fn cooling_shr(&self) -> f64 {
        self.cooling_shr
    }

    #[getter]
    pub fn reheat_capacity_w(&self) -> f64 {
        self.reheat_capacity_w
    }

    #[getter]
    pub fn fan_shaft_power_w(&self) -> f64 {
        self.fan_shaft_power_w
    }

    #[getter]
    pub fn fan_motor_power_w(&self) -> f64 {
        self.fan_motor_power_w
    }

    #[getter]
    pub fn fan_heat_w(&self) -> f64 {
        self.fan_heat_w
    }

    #[getter]
    pub fn condensate_rate_kg_per_s(&self) -> f64 {
        self.condensate_rate_kg_per_s
    }

    /// Supply-air dry-bulb temperature [°C].
    #[getter]
    pub fn supply_dry_bulb_c(&self) -> f64 {
        self.supply_dry_bulb_c
    }
}

/// Compute the full terminal performance for the given entering-air state.
///
/// This is a free function (not a method) because it takes the entering-air
/// psychrometric state plus a density, which is awkward to express as a
/// method argument in Python.
///
/// Args:
///     terminal: The VAV terminal unit to evaluate.
///     entering_dry_bulb_c: Entering-air dry-bulb temperature [°C].
///     entering_humidity_ratio: Entering-air humidity ratio [kg/kg dry air].
///     air_density_kg_per_m3: Air density [kg/m³].
///     control: The control signal (damper + coil states).
#[pyfunction]
pub fn compute_vav_terminal_performance(
    terminal: &PyVavTerminalUnit,
    entering_dry_bulb_c: f64,
    entering_humidity_ratio: f64,
    air_density_kg_per_m3: f64,
    control: &PyVavTerminalControl,
) -> PyResult<PyVavTerminalPerformance> {
    use crate::sim::hvac::airside_state::MoistAirState;

    let entering =
        MoistAirState::from_humidity_ratio(entering_dry_bulb_c, entering_humidity_ratio, 101_325.0)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid entering-air state: {e}"))
            })?;

    let perf = terminal
        .inner
        .compute_terminal_performance(&entering, air_density_kg_per_m3, &control.inner)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Terminal performance error: {e}"))
        })?;

    Ok(perf.into())
}

// Python module initialization - classes are registered in main fluxion module
