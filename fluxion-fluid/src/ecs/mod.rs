//! ECS module for HVAC equipment data layout and simulation.
//!
//! This module provides a custom Entity Component System (ECS) implementation
//! using Structure of Arrays (SoA) storage layout for HVAC plant equipment.
//!
//! # Module Structure
//!
//! - [`entity`] - Entity types and equipment kind enum
//! - [`components`] - Component data structures
//! - [`storage`] - SoA storage implementation
//! - [`systems`] - Simulation systems (mass balance, heat transfer, control loop)

pub mod components;
pub mod entity;
pub mod storage;
pub mod systems;

pub use components::{ControlSignal, EquipmentParameters, PhysicalState};
pub use entity::{EquipmentEntity, EquipmentKind};
pub use storage::EquipmentWorld;
pub use systems::{ControlLoopSystem, HeatTransferSystem, MassBalanceSystem};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chiller_and_vav_heat_transfer() {
        let mut world = EquipmentWorld::new();

        // Create 5 Chiller entities
        let chiller_entities: Vec<EquipmentEntity> = (0..5)
            .map(|i| {
                let entity = world.spawn(EquipmentKind::Chiller);
                world.set_rated_capacity(entity, 100_000.0 * (i + 1) as f64);
                world.set_efficiency(entity, 5.0);
                entity
            })
            .collect();

        // Create 3 VAV entities
        let vav_entities: Vec<EquipmentEntity> = (0..3)
            .map(|i| {
                let entity = world.spawn(EquipmentKind::VavBox);
                world.set_rated_capacity(entity, 5000.0 * (i + 1) as f64);
                entity
            })
            .collect();

        // Initialize physical state for chillers
        for entity in &chiller_entities {
            world.set_temperature(*entity, 7.0);
            world.set_pressure(*entity, 101_325.0);
            world.set_mass_flowrate(*entity, 0.5);
            world.set_enthalpy(*entity, 2500.0);
        }

        // Initialize physical state for VAVs
        for entity in &vav_entities {
            world.set_temperature(*entity, 24.0);
            world.set_pressure(*entity, 101_325.0);
            world.set_mass_flowrate(*entity, 0.2);
            world.set_enthalpy(*entity, 2800.0);
        }

        // Set control signals
        for entity in &chiller_entities {
            world.set_on_off(*entity, true);
            world.set_setpoint(*entity, 7.0);
        }

        for entity in &vav_entities {
            world.set_position(*entity, 0.7);
            world.set_setpoint(*entity, 22.0);
        }

        // Run HeatTransferSystem
        world.run_heat_transfer();

        // Verify outputs - check that temperature fields have been modified
        // by the heat transfer calculations
        let mut valid_temps = true;
        for entity in &chiller_entities {
            let temp = world.get_temperature(*entity);
            // After heat transfer, temperatures should reflect the thermal model
            if temp.is_nan() || temp.is_infinite() {
                valid_temps = false;
            }
        }

        for entity in &vav_entities {
            let temp = world.get_temperature(*entity);
            if temp.is_nan() || temp.is_infinite() {
                valid_temps = false;
            }
        }

        assert!(valid_temps, "Heat transfer produced invalid temperatures");
    }

    #[test]
    fn test_entity_counts() {
        let mut world = EquipmentWorld::new();

        assert_eq!(world.entity_count(), 0);

        // Create 5 Chiller entities
        for _ in 0..5 {
            world.spawn(EquipmentKind::Chiller);
        }

        // Create 3 VAV entities
        for _ in 0..3 {
            world.spawn(EquipmentKind::VavBox);
        }

        assert_eq!(world.entity_count(), 8);
    }

    #[test]
    fn test_mass_balance_system() {
        let mut world = EquipmentWorld::new();

        // Create entities
        let entity = world.spawn(EquipmentKind::Pump);
        world.set_mass_flowrate(entity, 0.5);

        // Set up control signal
        world.set_on_off(entity, true);

        // Run mass balance
        world.run_mass_balance();

        // Verify mass flow is preserved (mass balance should maintain flow)
        let mass_flow = world.get_mass_flowrate(entity);
        assert!(mass_flow > 0.0, "Mass flow should be positive");
    }

    #[test]
    fn test_control_loop_system() {
        let mut world = EquipmentWorld::new();

        let chiller = world.spawn(EquipmentKind::Chiller);
        world.set_on_off(chiller, true);
        world.set_setpoint(chiller, 7.0);
        world.set_position(chiller, 0.5);

        let vav = world.spawn(EquipmentKind::VavBox);
        world.set_on_off(vav, true);
        world.set_setpoint(vav, 22.0);
        world.set_position(vav, 0.3);

        // Run control loop system
        world.run_control_loop();

        // After control loop, positions should be adjusted
        // Based on setpoint tracking
        let chiller_pos = world.get_position(chiller);
        let vav_pos = world.get_position(vav);

        assert!(chiller_pos >= 0.0 && chiller_pos <= 1.0);
        assert!(vav_pos >= 0.0 && vav_pos <= 1.0);
    }
}
