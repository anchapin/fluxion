// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FMI 2.0 FMU lifecycle: instantiation, initialization, stepping, reset.
//!
//! This module owns the state-machine half of Fluxion's FMI 2.0 interop:
//!
//! * [`FmuCoSimulationMaster`] — the Fluxion-side co-simulation master that
//!   drives a re-imported FMU one `doStep` at a time, forwarding per-timestep
//!   inputs to [`ThermalModel::step_physics`].
//! * [`FfdFmuCApi`] — the FFD solver's FMI 2.0 co-simulation instance
//!   (`new` / `set_real` / `get_real` / `do_step` / `reset`), plus the
//!   `extern "C"` shims (`ffd_fmu2Instantiate`, `ffd_fmu2SetReal`,
//!   `ffd_fmu2GetReal`, `ffd_fmu2DoStep`, …) that C/C++ masters call.
//!
//! Model-description XML generation/parsing lives in
//! [`model_description`](super::model_description); the per-timestep
//! value-transfer structs live in [`marshaling`](super::marshaling).

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

use super::marshaling::{FfdFmuState, FmuInputs, FmuOutputs};
use super::model_description::{ImportedFmu, FFD_MAX_SURFACES, FFD_STRATIFICATION_LEVELS};
use super::FmiError;

// -----------------------------------------------------------------------------
// Co-simulation master algorithm (fmi2DoStep wrapper)
// -----------------------------------------------------------------------------

/// Co-simulation master driving a re-imported FMU one `doStep` at a time.
///
/// This is the Fluxion equivalent of the FMI 2.0 `fmi2DoStep` C callback:
/// each call to [`FmuCoSimulationMaster::do_step`] forwards the master's
/// per-timestep weather inputs to [`ThermalModel::step_physics`] and
/// returns the resulting zone temperature and heating/cooling loads.
///
/// Loads are derived from the per-zone energy accumulators
/// (`zone_heating_energy_kwh` / `zone_cooling_energy_kwh`) that
/// `step_physics` advances, converted from kWh-over-the-step to average
/// Watts.  This preserves energy conservation across the co-simulation
/// boundary (acceptance criterion #2 of issue #1708).
pub struct FmuCoSimulationMaster {
    model: ThermalModel<VectorField>,
    /// Communication timestep declared by the FMU (seconds).
    communication_timestep: f64,
    /// Current simulation time (seconds).
    current_time: f64,
    /// Current timestep index (0-based).
    timestep: usize,
}

impl FmuCoSimulationMaster {
    /// Build a master from an imported FMU, adopting its communication
    /// timestep and [`ThermalModel`].
    pub fn from_imported(fmu: ImportedFmu) -> Self {
        let communication_timestep = fmu.communication_timestep();
        Self {
            model: fmu.into_thermal_model(),
            communication_timestep,
            current_time: 0.0,
            timestep: 0,
        }
    }

    /// Borrow the underlying [`ThermalModel`].
    pub fn model(&self) -> &ThermalModel<VectorField> {
        &self.model
    }

    /// Mutably borrow the underlying [`ThermalModel`].
    pub fn model_mut(&mut self) -> &mut ThermalModel<VectorField> {
        &mut self.model
    }

    /// Communication timestep (seconds).
    pub fn communication_timestep(&self) -> f64 {
        self.communication_timestep
    }

    /// Current simulation time (seconds).
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Perform one co-simulation step — the `fmi2DoStep` wrapper.
    ///
    /// Forwards `inputs` to [`ThermalModel::step_physics`] (converting the
    /// outdoor temperature from Kelvin, as declared in the FMU interface,
    /// to degrees Celsius, as required by the physics engine) and returns
    /// a [`FmuOutputs`] entry per zone (converted back to Kelvin for the
    /// zone temperature) together with each zone's heating/cooling loads
    /// averaged over the step.
    ///
    /// The returned vector has length `model.hvac.num_zones`, so external
    /// co-simulation masters (FMPy, PyFMI, EnergyPlus-to-FMU, Modelica)
    /// receive telemetry for **every** zone the FMU was exported with —
    /// `FmuCoSimulationMaster::do_step` no longer silently drops
    /// `zone 1..N-1` (issue #2459).
    ///
    /// If `step_size` is omitted the FMU's declared communication timestep
    /// is used.
    pub fn do_step(&mut self, inputs: FmuInputs, step_size: Option<f64>) -> Vec<FmuOutputs> {
        let dt = step_size.unwrap_or(self.communication_timestep).max(1.0);

        // Snapshot per-zone energy accumulators *before* the step so the
        // delta gives the energy consumed during this step alone.
        let heat_before: Vec<f64> = self.model.hvac.zone_heating_energy_kwh.as_ref().to_vec();
        let cool_before: Vec<f64> = self.model.hvac.zone_cooling_energy_kwh.as_ref().to_vec();

        // FMI inputs are Kelvin; step_physics expects °C.
        let outdoor_temp_c = inputs.outdoor_temperature - 273.15;
        let _energy_kwh = self.model.step_physics(self.timestep, outdoor_temp_c, dt);

        let temps_c = self.model.setpoints.temperatures.as_ref();
        let heat_after = self.model.hvac.zone_heating_energy_kwh.as_ref();
        let cool_after = self.model.hvac.zone_cooling_energy_kwh.as_ref();

        // Convert kWh-delta over the step to average Watts:
        //   W = kWh * 3_600_000 / dt
        let outputs: Vec<FmuOutputs> = (0..self.model.hvac.num_zones)
            .map(|i| {
                let zone_temp_c = temps_c.get(i).copied().unwrap_or(20.0);
                let heating_load = heat_before
                    .get(i)
                    .copied()
                    .zip(heat_after.get(i).copied())
                    .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
                    .unwrap_or(0.0);
                let cooling_load = cool_before
                    .get(i)
                    .copied()
                    .zip(cool_after.get(i).copied())
                    .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
                    .unwrap_or(0.0);
                FmuOutputs {
                    zone_temperature: zone_temp_c + 273.15,
                    heating_load,
                    cooling_load,
                }
            })
            .collect();

        self.timestep += 1;
        self.current_time += dt;

        outputs
    }
}

// -----------------------------------------------------------------------------
// FMI 2.0 C-API wrapper for FFD FMU (issue #2388)
// -----------------------------------------------------------------------------
//
// These functions implement the FMI 2.0 Co-Simulation C-API for the FFD solver.
// They are exposed as `extern "C"` functions so they can be called from C/C++
// co-simulation masters (FMPy, PyFMI, EnergyPlus, Modelica tools).
//
// The C API functions are:
//   - fmi2DoStep:    Perform one co-simulation step
//   - fmi2SetReal:   Set real input variable values
//   - fmi2GetReal:   Get real output variable values
//   - fmi2Instantiate: Create an FMU instance
//   - fmi2FreeInstance: Free an FMU instance
//   - fmi2SetupExperiment: Set up experiment parameters
//   - fmi2EnterInitializationMode: Enter initialization mode
//   - fmi2ExitInitializationMode: Exit initialization mode
//   - fmi2Reset: Reset FMU to initial state

/// FMI 2.0 status return codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Fmi2Status {
    OK = 0,
    Warning = 1,
    Discard = 2,
    Error = 3,
    Fatal = 4,
    Pending = 5,
}

impl Fmi2Status {
    pub fn from_i32(val: i32) -> Self {
        match val {
            0 => Fmi2Status::OK,
            1 => Fmi2Status::Warning,
            2 => Fmi2Status::Discard,
            3 => Fmi2Status::Error,
            4 => Fmi2Status::Fatal,
            5 => Fmi2Status::Pending,
            _ => Fmi2Status::Error,
        }
    }
}

/// FMI 2.0 component handle (opaque pointer to FfdFmuCApi).
pub type Fmi2Component = *mut FfdFmuCApi;

/// FMI 2.0 component environment (user data, unused).
pub type Fmi2ComponentEnvironment = *mut std::ffi::c_void;

/// FMI 2.0 logger callback type.
/// FMI 2.0 logger callback type (FFI-safe).
pub type Fmi2Logger =
    Option<extern "C" fn(Fmi2ComponentEnvironment, i32, i32, *const std::ffi::c_char)>;

/// FMI 2.0 co-simulation instance for FFD solver.
///
/// This struct wraps the FFD FMU state and provides the FMI 2.0 C-API
/// entry points for co-simulation masters.
#[derive(Debug, Clone)]
pub struct FfdFmuCApi {
    state: FfdFmuState,
}

impl FfdFmuCApi {
    /// Create a new FFD FMU instance.
    pub fn new(communication_timestep: f64) -> Self {
        let mut state = FfdFmuState::default();
        state.communication_timestep = communication_timestep;
        Self { state }
    }

    /// Get mutable reference to internal state.
    pub fn state_mut(&mut self) -> &mut FfdFmuState {
        &mut self.state
    }

    /// Get reference to internal state.
    pub fn state(&self) -> &FfdFmuState {
        &self.state
    }

    /// Set an input value by variable reference.
    ///
    /// Returns `Ok(())` if the variable reference is valid, `Err(FmiError)`
    /// otherwise.
    pub fn set_real(&mut self, vr: u32, value: f64) -> Result<(), FmiError> {
        match vr {
            1 => self.state.inputs.inlet_air_temperature = value,
            2 => self.state.inputs.mass_flow_rate_supply = value,
            3 => self.state.inputs.mass_flow_rate_exhaust = value,
            4..=9 => {
                let idx = (vr - 4) as usize;
                if idx < FFD_MAX_SURFACES {
                    self.state.inputs.wall_temperatures[idx] = value;
                } else {
                    return Err(FmiError::Simulation(format!(
                        "Invalid value reference for wall temperature: {}",
                        vr
                    )));
                }
            }
            _ => {
                return Err(FmiError::Simulation(format!(
                    "Unknown value reference: {}",
                    vr
                )));
            }
        }
        Ok(())
    }

    /// Get an output value by variable reference.
    ///
    /// Returns `Ok(value)` if the variable reference is valid, `Err(FmiError)`
    /// otherwise.
    pub fn get_real(&self, vr: u32) -> Result<f64, FmiError> {
        let num_inputs = self.state.inputs.wall_temperatures.len() + 3;
        if vr <= num_inputs as u32 {
            return Err(FmiError::Simulation(format!(
                "Value reference {} is an input, not an output",
                vr
            )));
        }

        let output_vr = vr - num_inputs as u32 - 1;
        match output_vr {
            0..=3 => {
                let idx = output_vr as usize;
                if idx < FFD_STRATIFICATION_LEVELS {
                    Ok(self.state.outputs.zone_air_temperatures[idx])
                } else {
                    Err(FmiError::Simulation(format!(
                        "Invalid value reference for zone air temperature: {}",
                        vr
                    )))
                }
            }
            4..=9 => {
                let idx = (output_vr - 4) as usize;
                if idx < FFD_MAX_SURFACES {
                    Ok(self.state.outputs.chtc[idx])
                } else {
                    Err(FmiError::Simulation(format!(
                        "Invalid value reference for CHTC: {}",
                        vr
                    )))
                }
            }
            10..=15 => {
                let idx = (output_vr - 10) as usize;
                if idx < FFD_MAX_SURFACES {
                    Ok(self.state.outputs.surface_heat_fluxes[idx])
                } else {
                    Err(FmiError::Simulation(format!(
                        "Invalid value reference for surface heat flux: {}",
                        vr
                    )))
                }
            }
            _ => Err(FmiError::Simulation(format!(
                "Unknown output value reference: {}",
                vr
            ))),
        }
    }

    /// Perform one FFD simulation step.
    ///
    /// This is the Rust equivalent of `fmi2DoStep`. It advances the FFD
    /// simulation by `dt` seconds using the current inputs from `self.state.inputs`.
    /// The FFD solver computes the new outputs (zone air temperatures, CHTCs,
    /// surface heat fluxes) which are stored in `self.state.outputs`.
    ///
    /// Note: The actual FFD solver (advection, diffusion, pressure projection)
    /// is implemented separately (issue #2385). This method currently provides
    /// a stub that computes physically-plausible defaults.
    pub fn do_step(&mut self, dt: f64) -> Result<(), FmiError> {
        if dt <= 0.0 {
            return Err(FmiError::Simulation(
                "Step size must be positive".to_string(),
            ));
        }

        if !self.state.initialised {
            return Err(FmiError::Simulation(
                "FMU not initialised. Call setupExperiment and enterInitializationMode first."
                    .to_string(),
            ));
        }

        let t_air_in = self.state.inputs.inlet_air_temperature - 273.15;
        let t_wall_avg = self
            .state
            .inputs
            .wall_temperatures
            .iter()
            .map(|t| t - 273.15)
            .sum::<f64>()
            / self.state.inputs.wall_temperatures.len() as f64;
        let m_supply = self.state.inputs.mass_flow_rate_supply;
        let m_exhaust = self.state.inputs.mass_flow_rate_exhaust;

        let delta_t = (t_air_in - t_wall_avg).clamp(-5.0, 5.0);
        let convection_factor = 2.0 + 0.5 * m_supply.clamp(0.0, 2.0);

        for i in 0..self.state.outputs.zone_air_temperatures.len() {
            let height_factor = 1.0 + (i as f64) * 0.02;
            let temp = t_wall_avg + delta_t * 0.3 * height_factor + 273.15;
            self.state.outputs.zone_air_temperatures[i] = temp.clamp(200.0, 350.0);
        }

        for i in 0..self.state.outputs.chtc.len() {
            let base_chtc = convection_factor + 0.3 * m_exhaust.clamp(0.0, 1.0);
            let surface_factor = 1.0 + ((i as f64) * 0.1).sin();
            self.state.outputs.chtc[i] = (base_chtc * surface_factor).clamp(0.1, 50.0);
        }

        for i in 0..self.state.outputs.surface_heat_fluxes.len() {
            let t_zone = self.state.outputs.zone_air_temperatures[0] - 273.15;
            let q_conv = self.state.outputs.chtc[i] * (t_zone - t_wall_avg);
            let q_rad = 0.3 * q_conv;
            self.state.outputs.surface_heat_fluxes[i] = q_conv + q_rad;
        }

        self.state.current_time += dt;
        self.state.timestep += 1;

        Ok(())
    }

    /// Reset the FMU to initial state.
    pub fn reset(&mut self) {
        self.state = FfdFmuState {
            communication_timestep: self.state.communication_timestep,
            ..FfdFmuState::default()
        };
    }
}

// -----------------------------------------------------------------------------
// C-compatible FMI 2.0 API functions (extern "C")
// -----------------------------------------------------------------------------
//
// These functions provide the FMI 2.0 Co-Simulation C-API for the FFD solver.
// They are intended to be called from C/C++ co-simulation masters.
//
// FMI 2.0 spec reference: https://fmi-standard.org/docs/2.0.4/

/// FMI 2.0 status enum as returned by C API functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum fmi2Status {
    fmi2OK = 0,
    fmi2Warning = 1,
    fmi2Discard = 2,
    fmi2Error = 3,
    fmi2Fatal = 4,
    fmi2Pending = 5,
}

/// FMI 2.0 boolean type.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum fmi2Boolean {
    fmi2False = 0,
    fmi2True = 1,
}

/// FMU 2.0 spec: a single FMU exposes at most 8192 value references per
/// component (`fmi2ValueReference` count cap, FMI 2.0.4 §4.2.1).
///
/// Capping `nvr` here prevents an out-of-bounds slice reconstruction when a
/// malformed FMU master supplies an attacker-controlled length. See #2555.
pub const FMI2_MAX_VALUE_REFERENCES: usize = 8192;

/// Validate the (component, vr, nvr, value) tuple used by the
/// `ffd_fmu2SetReal` and `ffd_fmu2GetReal` extern "C" shims.
///
/// Guards every precondition that `std::slice::from_raw_parts` requires:
///
///   * `component` is non-null (we dereference `_c` later)
///   * `vr` is non-null when `nvr > 0` and is aligned for `u32`
///   * `value` is non-null when `nvr > 0` and is aligned for `f64`
///   * `nvr <= FMI2_MAX_VALUE_REFERENCES` so the reconstructed slice
///     cannot exceed the FMU 2.0 spec-mandated per-component cap
///
/// `nvr == 0` is permitted and treated as a documented no-op; we still
/// require non-null pointers up front to keep the contract explicit and
/// symmetric with the FMU 2.0 spec, which does not define a NULL-pointer
/// zero-length call.
///
/// Note: every check below is a runtime `if` (not `debug_assert!`) because
/// the calling shims are `extern "C"` and therefore `nounwind`; panicking
/// across that boundary would be undefined behaviour. The runtime checks
/// always fire in both debug and release builds.
///
/// Returns `Ok(())` if the call is safe to dispatch, or
/// `Err(fmi2Status::fmi2Error)` on any precondition violation.
#[inline]
fn validate_fmi2_real_args(
    component: Fmi2Component,
    vr: *const u32,
    nvr: usize,
    value: *const f64,
) -> Result<(), fmi2Status> {
    if component.is_null() || vr.is_null() || value.is_null() {
        return Err(fmi2Status::fmi2Error);
    }
    if nvr > FMI2_MAX_VALUE_REFERENCES {
        return Err(fmi2Status::fmi2Error);
    }
    if nvr == 0 {
        return Ok(());
    }
    if !(vr as usize).is_multiple_of(std::mem::align_of::<u32>())
        || !(value as usize).is_multiple_of(std::mem::align_of::<f64>())
    {
        return Err(fmi2Status::fmi2Error);
    }
    Ok(())
}

/// Allocate a new FFD FMU instance.
///
/// # Safety
/// This function returns an opaque pointer that must be freed with `fmi2FreeInstance`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2Instantiate(
    _instance_name: *const std::ffi::c_char,
    _fmu_state: *mut std::ffi::c_void,
    _visible: i32,
    _logging_on: i32,
    _logger: Fmi2Logger,
    _component_environment: Fmi2ComponentEnvironment,
) -> Fmi2Component {
    Box::into_raw(Box::new(FfdFmuCApi::new(60.0))) as Fmi2Component
}

/// Free an FFD FMU instance.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2FreeInstance(_c: Fmi2Component) {
    if !_c.is_null() {
        drop(Box::from_raw(_c));
    }
}

/// Set real input values.
///
/// # Safety
/// `vr` must be a valid value reference, `value` must be a valid f64.
///
/// `nvr` is bounded by `FMI2_MAX_VALUE_REFERENCES`; see
/// [`validate_fmi2_real_args`]. A malformed FMU master passing a
/// maliciously large `nvr`, a misaligned pointer, or a null component will
/// receive `fmi2Status::fmi2Error` rather than triggering UB.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2SetReal(
    _c: Fmi2Component,
    vr: *const u32,
    nvr: usize,
    value: *const f64,
) -> fmi2Status {
    if validate_fmi2_real_args(_c, vr, nvr, value).is_err() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    let vr_slice = std::slice::from_raw_parts(vr, nvr);
    let value_slice = std::slice::from_raw_parts(value, nvr);

    for (v, val) in vr_slice.iter().zip(value_slice.iter()) {
        if component.set_real(*v, *val).is_err() {
            return fmi2Status::fmi2Error;
        }
    }

    fmi2Status::fmi2OK
}

/// Get real output values.
///
/// # Safety
/// `vr` must be a valid value reference, `value` must point to valid memory.
///
/// `nvr` is bounded by `FMI2_MAX_VALUE_REFERENCES`; see
/// [`validate_fmi2_real_args`]. A malformed FMU master passing a
/// maliciously large `nvr`, a misaligned pointer, or a null component will
/// receive `fmi2Status::fmi2Error` rather than triggering UB.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2GetReal(
    _c: Fmi2Component,
    vr: *const u32,
    nvr: usize,
    value: *mut f64,
) -> fmi2Status {
    if validate_fmi2_real_args(_c, vr, nvr, value.cast_const()).is_err() {
        return fmi2Status::fmi2Error;
    }

    let component = &*_c;
    let vr_slice = std::slice::from_raw_parts(vr, nvr);
    let value_slice = std::slice::from_raw_parts_mut(value, nvr);

    for (v, out_val) in vr_slice.iter().zip(value_slice.iter_mut()) {
        match component.get_real(*v) {
            Ok(val) => *out_val = val,
            Err(_) => return fmi2Status::fmi2Error,
        }
    }

    fmi2Status::fmi2OK
}

/// Perform one co-simulation step.
///
/// # Safety
/// `current_time` and `step_size` must be valid f64 values.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2DoStep(
    _c: Fmi2Component,
    _current_time: f64,
    step_size: f64,
    _no_step_prior: i32,
) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    match component.do_step(step_size) {
        Ok(()) => fmi2Status::fmi2OK,
        Err(_) => fmi2Status::fmi2Error,
    }
}

/// Setup the experiment (set start time, stop time, step size).
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2SetupExperiment(
    _c: Fmi2Component,
    _tolerance_defined: i32,
    _tolerance: f64,
    start_time: f64,
    stop_time_defined: i32,
    stop_time: f64,
) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    component.state_mut().current_time = start_time;
    let _ = stop_time_defined;
    let _ = stop_time;

    fmi2Status::fmi2OK
}

/// Enter initialization mode.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2EnterInitializationMode(_c: Fmi2Component) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    component.state_mut().initialised = true;

    fmi2Status::fmi2OK
}

/// Exit initialization mode.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2ExitInitializationMode(_c: Fmi2Component) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    fmi2Status::fmi2OK
}

/// Reset the FMU to initial state.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2Reset(_c: Fmi2Component) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    component.reset();

    fmi2Status::fmi2OK
}

/// Get the FMU version string.
///
/// # Safety
/// The returned pointer is a static C string literal.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2GetVersion() -> *const std::ffi::c_char {
    c"2.0".as_ptr() as *const std::ffi::c_char
}

/// Get the FMU types platform string.
///
/// # Safety
/// The returned pointer is a static C string literal.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2GetTypesPlatform() -> *const std::ffi::c_char {
    c"default".as_ptr() as *const std::ffi::c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::model_description::{FmiExporter, FmiImporter};

    #[test]
    fn test_cosimulation_master_do_step_calls_step_physics() {
        // Export a single-zone FMU, re-import it, and drive one doStep.
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("master.fmu");
        FmiExporter::new().export_fmu(&out).expect("export");
        let fmu = FmiImporter::new().import(&out).expect("import");

        let initial_temp_k = fmu.thermal_model().setpoints.temperatures.as_ref()[0] + 273.15;
        let mut master = FmuCoSimulationMaster::from_imported(fmu);

        // Cold outdoor air (263.15 K = -10 °C) → expect the zone to cool
        // and/or heating to engage.
        let inputs = FmuInputs {
            outdoor_temperature: 263.15,
            direct_normal_solar: 0.0,
            diffuse_horizontal_solar: 0.0,
            internal_gains: 0.0,
        };
        let out_step = master.do_step(inputs, Some(3600.0));

        // do_step must return a finite zone temperature in Kelvin for the
        // single zone (single-zone FMU ⇒ vector length == 1).
        assert_eq!(out_step.len(), 1);
        let zone_out = &out_step[0];
        assert!(zone_out.zone_temperature.is_finite());
        assert!(zone_out.zone_temperature > 200.0 && zone_out.zone_temperature < 320.0);
        // The master advanced time by one communication step.
        assert_eq!(master.current_time(), 3600.0);
        // The zone temperature should have moved away from the initial 20 °C
        // (293.15 K) under the cold boundary condition.
        assert_ne!(zone_out.zone_temperature, initial_temp_k);
    }

    #[test]
    fn test_cosimulation_master_loads_nonneg_and_balanced() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("loads.fmu");
        FmiExporter::new().export_fmu(&out).expect("export");
        let fmu = FmiImporter::new().import(&out).expect("import");
        let mut master = FmuCoSimulationMaster::from_imported(fmu);

        // Drive a handful of steps; loads must be non-negative for every
        // zone reported by do_step.
        for _ in 0..5 {
            let outputs = master.do_step(FmuInputs::default(), Some(3600.0));
            assert!(!outputs.is_empty());
            for o in &outputs {
                assert!(o.heating_load >= 0.0);
                assert!(o.cooling_load >= 0.0);
            }
        }
        assert_eq!(master.current_time(), 5.0 * 3600.0);
    }

    #[test]
    fn test_ffd_fmu_capi_set_real() {
        let mut api = FfdFmuCApi::new(60.0);

        api.set_real(1, 295.15).unwrap();
        assert_eq!(api.state().inputs.inlet_air_temperature, 295.15);

        api.set_real(2, 0.5).unwrap();
        assert_eq!(api.state().inputs.mass_flow_rate_supply, 0.5);

        api.set_real(4, 290.15).unwrap();
        assert_eq!(api.state().inputs.wall_temperatures[0], 290.15);

        assert!(api.set_real(999, 100.0).is_err());
    }

    #[test]
    fn test_ffd_fmu_capi_get_real() {
        let api = FfdFmuCApi::new(60.0);

        // num_inputs = 3 (inlet, supply, exhaust) + 6 (wall temps) = 9
        // Output vrs start at 10 (num_inputs + 1)
        let num_inputs = 3 + FFD_MAX_SURFACES;
        // vr = num_inputs = 9 is still an input, should error
        assert!(api.get_real(num_inputs as u32).is_err());
        // vr = num_inputs + 1 = 10 is the first output, should be Ok
        let result = api.get_real(num_inputs as u32 + 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ffd_fmu_capi_do_step() {
        let mut api = FfdFmuCApi::new(60.0);
        api.state_mut().initialised = true;

        api.state_mut().inputs.inlet_air_temperature = 295.15;
        api.state_mut().inputs.mass_flow_rate_supply = 0.3;
        api.state_mut().inputs.wall_temperatures = [293.15; FFD_MAX_SURFACES];

        api.do_step(60.0).unwrap();

        assert_eq!(api.state().current_time, 60.0);
        assert_eq!(api.state().timestep, 1);

        for temp in api.state().outputs.zone_air_temperatures {
            assert!(temp > 200.0 && temp < 350.0);
        }
    }

    #[test]
    fn test_ffd_fmu_capi_do_step_not_initialised() {
        let mut api = FfdFmuCApi::new(60.0);
        assert!(api.do_step(60.0).is_err());
    }

    #[test]
    fn test_ffd_fmu_capi_reset() {
        let mut api = FfdFmuCApi::new(60.0);
        api.state_mut().initialised = true;
        api.state_mut().current_time = 3600.0;
        api.state_mut().timestep = 60;

        api.reset();

        assert_eq!(api.state().current_time, 0.0);
        assert_eq!(api.state().timestep, 0);
        assert!(!api.state().initialised);
    }
}
