//! Thin re-export shim for the canonical `ThermalModel`.
//!
//! Issue #2876: this module previously defined a small, lightweight
//! `ThermalModel` struct that lived in parallel with the canonical
//! `crate::sim::thermal_model_core::ThermalModel<VectorField>`. The two
//! types shared a name but had different APIs (different field layout,
//! different constructor signature, different method set), creating a
//! parallel-types drift hazard: a future PR adding `pub mod thermal_model;`
//! to a fresh namespace would silently shadow the canonical type and break
//! Python/NAPI/CLI consumers.
//!
//! This file now resolves to a single canonical `ThermalModel` via a
//! concrete `pub type` alias to `crate::sim::thermal_model_core::ThermalModel<VectorField>`.
//! Callers that previously constructed `ThermalModel::new(num_zones, temp)`
//! should switch to `ThermalModel::new(num_zones)` — the canonical constructor
//! defaults zone temperatures to 20 °C, matching the most common caller intent.

pub type ThermalModel =
    crate::sim::thermal_model_core::ThermalModel<crate::physics::cta::VectorField>;
