// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Per-timestep value marshaling between Rust and FMI C types.
//!
//! This module owns the plain data structs that cross the FMI boundary each
//! communication step:
//!
//! * [`FmuInputs`] / [`FmuOutputs`] — per-zone weather inputs and simulated
//!   outputs for the building-energy FMU interface (SI units, as declared in
//!   the generated `modelDescription.xml`).
//! * [`FfdFmuInputs`] / [`FfdFmuOutputs`] / [`FfdFmuState`] — BES→FFD inputs,
//!   FFD→BES outputs, and the FFD FMU's co-simulation state snapshot.
//!
//! These structs carry no behavior; the XML describing them is generated in
//! [`model_description`](super::model_description) and the stepping logic
//! that consumes them lives in [`lifecycle`](super::lifecycle).

use super::model_description::{FFD_MAX_SURFACES, FFD_STRATIFICATION_LEVELS};

/// Per-timestep FMI inputs for a single zone, in the units declared by the
/// Fluxion FMU interface (SI units: Kelvin, W/m², W).
#[derive(Debug, Clone, Copy)]
pub struct FmuInputs {
    /// Outdoor dry-bulb temperature (K).
    pub outdoor_temperature: f64,
    /// Direct normal solar irradiance (W/m²).
    pub direct_normal_solar: f64,
    /// Diffuse horizontal solar irradiance (W/m²).
    pub diffuse_horizontal_solar: f64,
    /// Internal heat gains (W).
    pub internal_gains: f64,
}

impl Default for FmuInputs {
    fn default() -> Self {
        // Matches the `<Real start=…>` defaults emitted by the exporter.
        Self {
            outdoor_temperature: 280.0,
            direct_normal_solar: 0.0,
            diffuse_horizontal_solar: 0.0,
            internal_gains: 0.0,
        }
    }
}

/// Per-timestep FMI outputs for a single zone, in the units declared by the
/// Fluxion FMU interface.
#[derive(Debug, Clone, Copy, Default)]
pub struct FmuOutputs {
    /// Zone air temperature (K).
    pub zone_temperature: f64,
    /// Heating load over the step (W, non-negative).
    pub heating_load: f64,
    /// Cooling load over the step (W, non-negative).
    pub cooling_load: f64,
}

/// FFD FMU input data per timestep (BES → FFD).
#[derive(Debug, Clone, Copy)]
pub struct FfdFmuInputs {
    /// Inlet/supply air temperature (K).
    pub inlet_air_temperature: f64,
    /// HVAC supply air mass flow rate (kg/s).
    pub mass_flow_rate_supply: f64,
    /// HVAC exhaust air mass flow rate (kg/s).
    pub mass_flow_rate_exhaust: f64,
    /// Wall temperatures at zone boundaries (K), indexed by surface.
    pub wall_temperatures: [f64; FFD_MAX_SURFACES],
}

impl Default for FfdFmuInputs {
    fn default() -> Self {
        Self {
            inlet_air_temperature: 293.15,
            mass_flow_rate_supply: 0.0,
            mass_flow_rate_exhaust: 0.0,
            wall_temperatures: [293.15; FFD_MAX_SURFACES],
        }
    }
}

/// FFD FMU output data per timestep (FFD → BES).
#[derive(Debug, Clone, Copy, Default)]
pub struct FfdFmuOutputs {
    /// Stratified zone air temperatures at different heights (K).
    pub zone_air_temperatures: [f64; FFD_STRATIFICATION_LEVELS],
    /// Convective heat transfer coefficients per surface (W/m²K).
    pub chtc: [f64; FFD_MAX_SURFACES],
    /// Surface heat fluxes per surface (W/m²).
    pub surface_heat_fluxes: [f64; FFD_MAX_SURFACES],
}

/// FFD FMU state for co-simulation.
#[derive(Debug, Clone)]
pub struct FfdFmuState {
    /// Current simulation time (s).
    pub current_time: f64,
    /// Current timestep index.
    pub timestep: usize,
    /// Communication timestep (s).
    pub communication_timestep: f64,
    /// Inputs from BES.
    pub inputs: FfdFmuInputs,
    /// Outputs from FFD solver.
    pub outputs: FfdFmuOutputs,
    /// Whether the FMU has been initialised.
    pub initialised: bool,
}

impl Default for FfdFmuState {
    fn default() -> Self {
        Self {
            current_time: 0.0,
            timestep: 0,
            communication_timestep: 60.0,
            inputs: FfdFmuInputs::default(),
            outputs: FfdFmuOutputs::default(),
            initialised: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffd_fmu_inputs_default() {
        let inputs = FfdFmuInputs::default();
        assert_eq!(inputs.inlet_air_temperature, 293.15);
        assert_eq!(inputs.mass_flow_rate_supply, 0.0);
        assert_eq!(inputs.mass_flow_rate_exhaust, 0.0);
        for t in inputs.wall_temperatures {
            assert_eq!(t, 293.15);
        }
    }

    #[test]
    fn test_ffd_fmu_outputs_default() {
        let outputs = FfdFmuOutputs::default();
        for temp in outputs.zone_air_temperatures {
            assert_eq!(temp, 0.0);
        }
        for chtc in outputs.chtc {
            assert_eq!(chtc, 0.0);
        }
        for flux in outputs.surface_heat_fluxes {
            assert_eq!(flux, 0.0);
        }
    }
}
