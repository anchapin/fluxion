//! SoA (Structure of Arrays) storage for HVAC equipment ECS.
//!
//! This module provides the core ECS storage using Structure of Arrays layout.
//! All component arrays are contiguous `Vec<f64>` for cache efficiency and
//! SIMD-friendly access patterns.
//!
//! # SoA Layout
//!
//! Instead of storing entities as structs (AoS):
//! ```ignore
//! struct Entity { temperature: f64, pressure: f64, ... }  // Array of Structs
//! ```
//!
//! We store component fields in separate arrays (SoA):
//! ```ignore
//! temperatures: Vec<f64>,  // All temperatures
//! pressures: Vec<f64>,      // All pressures
//! mass_flowrates: Vec<f64>, // All mass flow rates
//! enthalpies: Vec<f64>,     // All enthalpies
//! ```
//!
//! This enables:
//! - Zero-copy iteration over component arrays
//! - Better cache utilization for component-iteration workloads
//! - SIMD-friendly contiguous memory access

use crate::ecs::components::{ControlSignal, EquipmentParameters, PhysicalState};
use crate::ecs::entity::{EquipmentEntity, EquipmentKind};

/// SoA storage for HVAC equipment ECS.
///
/// Stores all component data in contiguous arrays for efficient iteration.
#[derive(Clone, Debug)]
pub struct EquipmentWorld {
    /// Entity kind for each entity
    kinds: Vec<EquipmentKind>,

    // PhysicalState components (SoA)
    temperatures: Vec<f64>,
    pressures: Vec<f64>,
    mass_flowrates: Vec<f64>,
    enthalpies: Vec<f64>,

    // EquipmentParameters components (SoA)
    rated_capacities: Vec<f64>,
    efficiencies: Vec<f64>,
    nominal_flowrates: Vec<f64>,
    control_types: Vec<f64>,

    // ControlSignal components (SoA)
    setpoints: Vec<f64>,
    positions: Vec<f64>,
    on_offs: Vec<f64>,

    // Output arrays for system results (heat_transfer outputs)
    heat_transfer_outputs: Vec<f64>,
}

impl EquipmentWorld {
    /// Create a new empty equipment world.
    pub fn new() -> Self {
        Self::with_capacity(64)
    }

    /// Create a new equipment world with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            kinds: Vec::with_capacity(capacity),

            temperatures: Vec::with_capacity(capacity),
            pressures: Vec::with_capacity(capacity),
            mass_flowrates: Vec::with_capacity(capacity),
            enthalpies: Vec::with_capacity(capacity),

            rated_capacities: Vec::with_capacity(capacity),
            efficiencies: Vec::with_capacity(capacity),
            nominal_flowrates: Vec::with_capacity(capacity),
            control_types: Vec::with_capacity(capacity),

            setpoints: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            on_offs: Vec::with_capacity(capacity),

            heat_transfer_outputs: Vec::with_capacity(capacity),
        }
    }

    /// Get the total number of entities.
    pub fn entity_count(&self) -> usize {
        self.kinds.len()
    }

    /// Spawn a new entity of the given kind.
    /// Returns the new entity ID.
    pub fn spawn(&mut self, kind: EquipmentKind) -> EquipmentEntity {
        let index = self.kinds.len() as u64;

        self.kinds.push(kind);

        // Initialize physical state with defaults
        let state = PhysicalState::default_for(kind);
        self.temperatures.push(state.temperature);
        self.pressures.push(state.pressure);
        self.mass_flowrates.push(state.mass_flowrate);
        self.enthalpies.push(state.enthalpy);

        // Initialize parameters with defaults
        let params = match kind {
            EquipmentKind::Chiller => EquipmentParameters::chiller(100_000.0, 5.0),
            EquipmentKind::Boiler => EquipmentParameters::boiler(50_000.0, 0.9),
            EquipmentKind::Pump => EquipmentParameters::pump(0.5, 100_000.0, 5000.0),
            EquipmentKind::VavBox => EquipmentParameters::vav_box(5000.0, 0.05),
            _ => EquipmentParameters::default(),
        };
        self.rated_capacities.push(params.rated_capacity);
        self.efficiencies.push(params.efficiency);
        self.nominal_flowrates.push(params.nominal_flowrate);
        self.control_types.push(params.control_type);

        // Initialize control signal
        let ctrl = ControlSignal::default();
        self.setpoints.push(ctrl.setpoint);
        self.positions.push(ctrl.position);
        self.on_offs.push(ctrl.on_off);

        // Initialize output arrays
        self.heat_transfer_outputs.push(0.0);

        EquipmentEntity::new(index)
    }

    /// Get entity kind.
    pub fn get_kind(&self, entity: EquipmentEntity) -> EquipmentKind {
        self.kinds[entity.index() as usize]
    }

    // Physical state accessors
    pub fn get_temperature(&self, entity: EquipmentEntity) -> f64 {
        self.temperatures[entity.index() as usize]
    }

    pub fn set_temperature(&mut self, entity: EquipmentEntity, value: f64) {
        self.temperatures[entity.index() as usize] = value;
    }

    pub fn get_pressure(&self, entity: EquipmentEntity) -> f64 {
        self.pressures[entity.index() as usize]
    }

    pub fn set_pressure(&mut self, entity: EquipmentEntity, value: f64) {
        self.pressures[entity.index() as usize] = value;
    }

    pub fn get_mass_flowrate(&self, entity: EquipmentEntity) -> f64 {
        self.mass_flowrates[entity.index() as usize]
    }

    pub fn set_mass_flowrate(&mut self, entity: EquipmentEntity, value: f64) {
        self.mass_flowrates[entity.index() as usize] = value;
    }

    pub fn get_enthalpy(&self, entity: EquipmentEntity) -> f64 {
        self.enthalpies[entity.index() as usize]
    }

    pub fn set_enthalpy(&mut self, entity: EquipmentEntity, value: f64) {
        self.enthalpies[entity.index() as usize] = value;
    }

    // Equipment parameters accessors
    pub fn get_rated_capacity(&self, entity: EquipmentEntity) -> f64 {
        self.rated_capacities[entity.index() as usize]
    }

    pub fn set_rated_capacity(&mut self, entity: EquipmentEntity, value: f64) {
        self.rated_capacities[entity.index() as usize] = value;
    }

    pub fn get_efficiency(&self, entity: EquipmentEntity) -> f64 {
        self.efficiencies[entity.index() as usize]
    }

    pub fn set_efficiency(&mut self, entity: EquipmentEntity, value: f64) {
        self.efficiencies[entity.index() as usize] = value;
    }

    pub fn get_nominal_flowrate(&self, entity: EquipmentEntity) -> f64 {
        self.nominal_flowrates[entity.index() as usize]
    }

    pub fn set_nominal_flowrate(&mut self, entity: EquipmentEntity, value: f64) {
        self.nominal_flowrates[entity.index() as usize] = value;
    }

    // Control signal accessors
    pub fn get_setpoint(&self, entity: EquipmentEntity) -> f64 {
        self.setpoints[entity.index() as usize]
    }

    pub fn set_setpoint(&mut self, entity: EquipmentEntity, value: f64) {
        self.setpoints[entity.index() as usize] = value;
    }

    pub fn get_position(&self, entity: EquipmentEntity) -> f64 {
        self.positions[entity.index() as usize]
    }

    pub fn set_position(&mut self, entity: EquipmentEntity, value: f64) {
        self.positions[entity.index() as usize] = value.clamp(0.0, 1.0);
    }

    pub fn get_on_off(&self, entity: EquipmentEntity) -> bool {
        self.on_offs[entity.index() as usize] > 0.5
    }

    pub fn set_on_off(&mut self, entity: EquipmentEntity, value: bool) {
        self.on_offs[entity.index() as usize] = if value { 1.0 } else { 0.0 };
    }

    // Output accessor
    pub fn get_heat_transfer_output(&self, entity: EquipmentEntity) -> f64 {
        self.heat_transfer_outputs[entity.index() as usize]
    }

    /// Get mutable slice to temperatures for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn temperatures_mut(&mut self) -> &mut [f64] {
        &mut self.temperatures
    }

    /// Get const slice to temperatures for zero-copy iteration.
    pub fn temperatures_slice(&self) -> &[f64] {
        &self.temperatures
    }

    /// Get mutable slice to enthalpies for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn enthalpies_mut(&mut self) -> &mut [f64] {
        &mut self.enthalpies
    }

    /// Get const slice to enthalpies for zero-copy iteration.
    pub fn enthalpies_slice(&self) -> &[f64] {
        &self.enthalpies
    }

    /// Get mutable slice to mass flowrates for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mass_flowrates_mut(&mut self) -> &mut [f64] {
        &mut self.mass_flowrates
    }

    /// Get const slice to mass flowrates for zero-copy iteration.
    pub fn mass_flowrates_slice(&self) -> &[f64] {
        &self.mass_flowrates
    }

    /// Get mutable slice to rated capacities for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn rated_capacities_mut(&mut self) -> &mut [f64] {
        &mut self.rated_capacities
    }

    /// Get const slice to rated capacities for zero-copy iteration.
    pub fn rated_capacities_slice(&self) -> &[f64] {
        &self.rated_capacities
    }

    /// Get mutable slice to efficiencies for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn efficiencies_mut(&mut self) -> &mut [f64] {
        &mut self.efficiencies
    }

    /// Get const slice to efficiencies for zero-copy iteration.
    pub fn efficiencies_slice(&self) -> &[f64] {
        &self.efficiencies
    }

    /// Get const slice to nominal flowrates for zero-copy iteration.
    pub fn nominal_flowrates_slice(&self) -> &[f64] {
        &self.nominal_flowrates
    }

    /// Get const slice to on_offs for zero-copy iteration.
    pub fn on_offs_slice(&self) -> &[f64] {
        &self.on_offs
    }

    /// Get mutable slice to positions for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn positions_mut(&mut self) -> &mut [f64] {
        &mut self.positions
    }

    /// Get const slice to positions for zero-copy iteration.
    pub fn positions_slice(&self) -> &[f64] {
        &self.positions
    }

    /// Get mutable slice to setpoints for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn setpoints_mut(&mut self) -> &mut [f64] {
        &mut self.setpoints
    }

    /// Get const slice to setpoints for zero-copy iteration.
    pub fn setpoints_slice(&self) -> &[f64] {
        &self.setpoints
    }

    /// Get mutable slice to on_offs for zero-copy iteration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_offs_mut(&mut self) -> &mut [f64] {
        &mut self.on_offs
    }

    /// Get mutable slice to heat transfer outputs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn heat_transfer_outputs_mut(&mut self) -> &mut [f64] {
        &mut self.heat_transfer_outputs
    }

    /// Get the kinds slice for iteration.
    pub fn kinds_slice(&self) -> &[EquipmentKind] {
        &self.kinds
    }

    /// Run mass balance system.
    ///
    /// Applies mass conservation: mass_flowrate_out = mass_flowrate_in for steady state.
    /// This is a simple pass that ensures mass flow is non-negative.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_mass_balance(&mut self) {
        use crate::ecs::systems::MassBalanceSystem;
        MassBalanceSystem::run(self);
    }

    /// Run heat transfer system.
    ///
    /// Computes heat transfer based on equipment kind:
    /// - Chillers: Q = m_dot * (h_evap - h_cond) with COP correction
    /// - Boilers: Q = m_dot * c_p * (T_return - T_enter) / eta
    /// - VAV boxes: Q = m_dot * c_p * (T_supply - T_inlet)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_heat_transfer(&mut self) {
        use crate::ecs::systems::HeatTransferSystem;
        HeatTransferSystem::run(self);
    }

    /// Run control loop system.
    ///
    /// Updates actuator positions based on setpoint tracking error.
    /// Uses simple proportional control: u = Kp * e, where e = setpoint - measured.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_control_loop(&mut self) {
        use crate::ecs::systems::ControlLoopSystem;
        ControlLoopSystem::run(self);
    }
}

impl Default for EquipmentWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_world_has_zero_entities() {
        let w = EquipmentWorld::new();
        assert_eq!(w.entity_count(), 0);
        assert!(w.kinds_slice().is_empty());
        assert!(w.temperatures_slice().is_empty());
        assert!(w.enthalpies_slice().is_empty());
    }

    #[test]
    fn with_capacity_allocates_for_preallocation() {
        let w = EquipmentWorld::with_capacity(128);
        assert_eq!(w.entity_count(), 0);
        // Capacity is a hint, not exposed, but a fresh world should still be empty.
        assert!(w.temperatures_slice().is_empty());
        assert!(w.enthalpies_slice().is_empty());
        assert!(w.mass_flowrates_slice().is_empty());
        assert!(w.rated_capacities_slice().is_empty());
        assert!(w.efficiencies_slice().is_empty());
        assert!(w.nominal_flowrates_slice().is_empty());
        assert!(w.positions_slice().is_empty());
        assert!(w.setpoints_slice().is_empty());
        assert!(w.on_offs_slice().is_empty());
        assert!(w.kinds_slice().is_empty());
    }

    #[test]
    fn spawn_increments_entity_count() {
        let mut w = EquipmentWorld::new();
        let a = w.spawn(EquipmentKind::Chiller);
        let b = w.spawn(EquipmentKind::Boiler);
        let c = w.spawn(EquipmentKind::Pump);
        assert_eq!(w.entity_count(), 3);
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(c.index(), 2);
        assert_eq!(w.get_kind(a), EquipmentKind::Chiller);
        assert_eq!(w.get_kind(b), EquipmentKind::Boiler);
        assert_eq!(w.get_kind(c), EquipmentKind::Pump);
    }

    #[test]
    fn physical_state_round_trip() {
        let mut w = EquipmentWorld::new();
        let e = w.spawn(EquipmentKind::Chiller);
        w.set_temperature(e, 7.5);
        w.set_pressure(e, 250_000.0);
        w.set_mass_flowrate(e, 1.25);
        w.set_enthalpy(e, 42_000.0);
        assert_eq!(w.get_temperature(e), 7.5);
        assert_eq!(w.get_pressure(e), 250_000.0);
        assert_eq!(w.get_mass_flowrate(e), 1.25);
        assert_eq!(w.get_enthalpy(e), 42_000.0);
    }

    #[test]
    fn equipment_parameters_round_trip() {
        let mut w = EquipmentWorld::new();
        let e = w.spawn(EquipmentKind::Pump);
        w.set_rated_capacity(e, 7_500.0);
        w.set_efficiency(e, 0.92);
        w.set_nominal_flowrate(e, 0.75);
        assert_eq!(w.get_rated_capacity(e), 7_500.0);
        assert_eq!(w.get_efficiency(e), 0.92);
        assert_eq!(w.get_nominal_flowrate(e), 0.75);
    }

    #[test]
    fn control_signal_round_trip() {
        let mut w = EquipmentWorld::new();
        let e = w.spawn(EquipmentKind::VavBox);
        w.set_setpoint(e, 22.5);
        w.set_position(e, 0.65);
        w.set_on_off(e, true);
        assert_eq!(w.get_setpoint(e), 22.5);
        assert_eq!(w.get_position(e), 0.65);
        assert!(w.get_on_off(e));
        w.set_on_off(e, false);
        assert!(!w.get_on_off(e));
    }

    #[test]
    fn set_position_clamps_to_unit_interval() {
        let mut w = EquipmentWorld::new();
        let e = w.spawn(EquipmentKind::Damper);
        w.set_position(e, 2.5);
        assert_eq!(w.get_position(e), 1.0, "position above 1.0 must clamp");
        w.set_position(e, -0.5);
        assert_eq!(w.get_position(e), 0.0, "position below 0.0 must clamp");
        w.set_position(e, 0.42);
        assert_eq!(w.get_position(e), 0.42, "in-range value must pass through");
    }

    #[test]
    fn set_on_off_encodes_as_f64() {
        let mut w = EquipmentWorld::new();
        let e = w.spawn(EquipmentKind::Fan);
        w.set_on_off(e, true);
        assert_eq!(w.on_offs_slice()[0], 1.0);
        w.set_on_off(e, false);
        assert_eq!(w.on_offs_slice()[0], 0.0);
        // get_on_off uses the > 0.5 threshold, so the boundary cases map
        // exactly to the boolean getter.
        assert!(!w.get_on_off(e));
        w.set_on_off(e, true);
        assert!(w.get_on_off(e));
        // Even when read back-to-back, the storage round-trips the bool->f64 mapping.
        for _ in 0..5 {
            w.set_on_off(e, true);
            assert!(w.get_on_off(e));
            w.set_on_off(e, false);
            assert!(!w.get_on_off(e));
        }
    }

    #[test]
    fn soa_slices_match_entity_count() {
        let mut w = EquipmentWorld::new();
        for _ in 0..4 {
            w.spawn(EquipmentKind::Chiller);
        }
        let n = w.entity_count();
        assert_eq!(w.temperatures_slice().len(), n);
        assert_eq!(w.enthalpies_slice().len(), n);
        assert_eq!(w.mass_flowrates_slice().len(), n);
        assert_eq!(w.rated_capacities_slice().len(), n);
        assert_eq!(w.efficiencies_slice().len(), n);
        assert_eq!(w.nominal_flowrates_slice().len(), n);
        assert_eq!(w.positions_slice().len(), n);
        assert_eq!(w.setpoints_slice().len(), n);
        assert_eq!(w.on_offs_slice().len(), n);
        assert_eq!(w.kinds_slice().len(), n);
    }

    #[test]
    fn kind_specific_default_initialization() {
        let mut w = EquipmentWorld::new();
        let chiller = w.spawn(EquipmentKind::Chiller);
        let boiler = w.spawn(EquipmentKind::Boiler);
        let pump = w.spawn(EquipmentKind::Pump);
        // Chiller and boiler defaults differ in temperature.
        assert!(w.get_temperature(chiller) < w.get_temperature(boiler));
        // Pump default nominal flowrate is set per-kind (0.5).
        assert!(w.get_nominal_flowrate(pump) > 0.0);
    }
}
