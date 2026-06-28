//! Multi-Node Thermal Model Data Structures (Phase 6)
//!
//! # Crate split (issue #1349 — Phase 2)
//!
//! As of #1349, the implementation lives in `fluxion_core::multi_node` (the
//! workspace leaf crate). This file is a thin re-export shim so existing
//! `crate::sim::multi_node_thermal::*` and `fluxion::sim::multi_node_thermal::*`
//! paths keep working without call-site edits.
//!
//! All types (`ThermalMassNode`, `MultiNodeThermalMass`, `MultiNodeModelType`,
//! `MassAirCouplingMode`) are defined in `fluxion_core::multi_node`.

#[doc(inline)]
pub use fluxion_core::multi_node::*;