//! Per-Surface Conduction Solver for Multi-Node Thermal Model
//!
//! # Crate split (Issue #2462 — Phase 2 of the crate split)
//!
//! As of #2462, the implementation lives in `fluxion_core::per_surface_conduction`
//! (the workspace leaf crate) so the `physics ↔ sim` cycle documented in
//! `docs/mutation_testing_crate_split.md` §"Phase 2" can close.
//!
//! This file is a thin re-export shim so existing
//! `crate::sim::per_surface_conduction::*` and `fluxion::sim::per_surface_conduction::*`
//! paths keep working without call-site edits.
//!
//! All types (`SurfaceKind`, `MassNode`, `SurfaceNode`, `PerSurfaceConductionSolver`)
//! are defined in `fluxion_core::per_surface_conduction`.

#[doc(inline)]
pub use fluxion_core::per_surface_conduction::*;
