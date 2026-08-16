//! Mass state — thermal mass, capacitances, surface temps, mass energies, mass conductances.
//!
//! Extracted from `ThermalModelData` (Issue #2878). Owns every per-zone mass /
//! capacitance / surface-temperature tensor (5R1C lumped mass, 6R2C envelope
//! vs internal split, 9R4C ceiling/floor/partition split, partitioned LW
//! surface temps, air-node ODE state) plus the mass-energy bookkeeping fields
//! and the few `h_tr_*` conductance vectors that are conceptually part of
//! the mass-side thermal network rather than the boundary-side network.

use crate::physics::cta::ContinuousTensor;
use fluxion_core::multi_node::MultiNodeThermalMass;

pub struct MassState<T: ContinuousTensor<f64>> {
    // Lumped mass + capacitances (5R1C).
    pub mass_temperatures: T,
    pub thermal_capacitance: T,
    /// Air-node thermal capacitance C_air = ρ_air · cp_air · V_zone  [J/K] — Issue #1522.
    pub air_thermal_capacitance: T,
    /// Independent air-node temperature state for the 5R1C model — Issue #1585.
    pub air_temperatures: T,
    /// Issue #2339 — number of sub-steps for the air-node ODE update.
    pub sub_hour_air_node_steps: u32,

    /// Issue #1860 — first-order low-pass solar-lag state.
    pub solar_lag: T,
    /// Issue #1860 — independent interior wall-surface ODE state (5R1C).
    pub wall_surface_temperatures: T,
    /// Issue #2890 — partitioned LW interior surface temperatures.
    pub surface_temp_floor: T,
    /// See [`Self::surface_temp_floor`].
    pub surface_temp_ceiling: T,
    /// See [`Self::surface_temp_floor`].
    pub surface_temp_wall: T,

    // 6R2C envelope vs internal split.
    pub envelope_mass_temperatures: T,
    pub internal_mass_temperatures: T,
    pub envelope_thermal_capacitance: T,
    pub internal_thermal_capacitance: T,
    /// Conductance between envelope and internal mass.
    pub h_tr_me: T,

    // 8R3C / 9R4C per-mass-node states.
    pub ceiling_mass_temperatures: Option<T>,
    pub floor_mass_temperatures: Option<T>,
    pub partition_mass_temperatures: Option<T>,
    pub ceiling_thermal_capacitance: Option<T>,
    pub floor_thermal_capacitance: Option<T>,
    pub partition_thermal_capacitance: Option<T>,
    pub h_tr_ceiling: Option<T>,
    pub h_tr_floor_mass: Option<T>,
    pub h_tr_partition: Option<T>,

    // Energy bookkeeping for the 6R2C path (Issue #272, #274, #275, #432).
    pub previous_mass_temperatures: T,
    pub mass_energy_change_cumulative: f64,
    pub envelope_mass_energy_change_cumulative: f64,
    pub internal_mass_energy_change_cumulative: f64,

    /// Multi-node thermal mass state for the 9R4C model.
    pub multi_node_thermal_mass: Option<MultiNodeThermalMass>,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for MassState<T> {
    fn clone(&self) -> Self {
        Self {
            mass_temperatures: self.mass_temperatures.clone(),
            thermal_capacitance: self.thermal_capacitance.clone(),
            air_thermal_capacitance: self.air_thermal_capacitance.clone(),
            air_temperatures: self.air_temperatures.clone(),
            sub_hour_air_node_steps: self.sub_hour_air_node_steps,

            solar_lag: self.solar_lag.clone(),
            wall_surface_temperatures: self.wall_surface_temperatures.clone(),
            surface_temp_floor: self.surface_temp_floor.clone(),
            surface_temp_ceiling: self.surface_temp_ceiling.clone(),
            surface_temp_wall: self.surface_temp_wall.clone(),

            envelope_mass_temperatures: self.envelope_mass_temperatures.clone(),
            internal_mass_temperatures: self.internal_mass_temperatures.clone(),
            envelope_thermal_capacitance: self.envelope_thermal_capacitance.clone(),
            internal_thermal_capacitance: self.internal_thermal_capacitance.clone(),
            h_tr_me: self.h_tr_me.clone(),

            ceiling_mass_temperatures: self.ceiling_mass_temperatures.clone(),
            floor_mass_temperatures: self.floor_mass_temperatures.clone(),
            partition_mass_temperatures: self.partition_mass_temperatures.clone(),
            ceiling_thermal_capacitance: self.ceiling_thermal_capacitance.clone(),
            floor_thermal_capacitance: self.floor_thermal_capacitance.clone(),
            partition_thermal_capacitance: self.partition_thermal_capacitance.clone(),
            h_tr_ceiling: self.h_tr_ceiling.clone(),
            h_tr_floor_mass: self.h_tr_floor_mass.clone(),
            h_tr_partition: self.h_tr_partition.clone(),

            previous_mass_temperatures: self.previous_mass_temperatures.clone(),
            mass_energy_change_cumulative: self.mass_energy_change_cumulative,
            envelope_mass_energy_change_cumulative: self.envelope_mass_energy_change_cumulative,
            internal_mass_energy_change_cumulative: self.internal_mass_energy_change_cumulative,

            multi_node_thermal_mass: self.multi_node_thermal_mass.clone(),
        }
    }
}
