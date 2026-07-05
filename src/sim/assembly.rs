//! Building Assembly Module
//!
//! # Crate split (issue #1349 — Phase 2)
//!
//! As of #1349, the implementation lives in `fluxion_core::assembly` (the
//! workspace leaf crate). This file is a thin re-export shim so existing
//! `crate::sim::assembly::*` and `fluxion::sim::assembly::*` paths keep working
//! without call-site edits.
//!
//! All types (`BuildingAssembly`, `AssemblyBuilder`, `MaterialLayer`,
//! `ConcreteMaterial`, `InsulationMaterial`, `GypsumMaterial`, `BrickMaterial`,
//! `MaterialYAML`, `LayerYAML`, `AssemblyYAML`, `AssemblyError`,
//! `ThermalMassClassification`, `load_materials`, `load_assemblies`) are defined
//! in `fluxion_core::assembly`.

#[doc(inline)]
pub use fluxion_core::assembly::*;
