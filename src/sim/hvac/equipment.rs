//! HVAC Equipment Models — Re-export shim
//!
//! Issue #4157: The equipment types (`AnyEquipment`, `VariableCapacityEquipment`,
//! `HVACMode`, `Chiller`, `Boiler`, `VAVTerminal`, `CAVSystem`, `HeatPump`, etc.)
//! have been moved to `fluxion_core::hvac_equipment` to break the
//! `validation ↔ sim` dependency cycle driven by `CaseSpec`'s `hvac_equipment` field.
//!
//! This file re-exports everything from `fluxion_core::hvac_equipment` so that
//! existing `crate::sim::hvac::equipment::*` paths continue to resolve unchanged.
//! The main crate retains full equipment types with psychrometric state in
//! `src/sim/hvac/mod.rs` for Python/NAPI bindings and advanced use cases.

// Re-export all types from the leaf crate
pub use fluxion_core::hvac_equipment::*;
