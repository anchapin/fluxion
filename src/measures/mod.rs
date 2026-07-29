// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Hybrid Measure Approach — Phase 1 (Declarative Deltas).
//!
//! This module is the in-memory, zero-Python-GIL building-model mutation
//! substrate described in Issue #1811. It supports massive Monte Carlo
//! data generation by applying declarative JSON Patches (RFC 6902) to a
//! [`model::FluxionModel`] across thousands of `rayon` threads.
//!
//! # Module layout
//!
//! - [`model::FluxionModel`] — the building model that Deltas mutate.
//!   Includes ASHRAE 140 Case 600 and Case 900 reference constructors.
//! - [`error::DeltaError`] — typed error variants. **Never panics on
//!   user-supplied data.**
//! - [`json_patch::apply_delta`] — the entry point: takes a
//!   `&mut FluxionModel` and a `&json_patch::Patch` and applies the
//!   patch atomically (or leaves the model unchanged on failure).
//!
//! # Module location rationale
//!
//! This module lives in the main `fluxion` crate (not `fluxion-core`)
//! on purpose. The cycle-breaking rule (see `AGENTS.md`) forbids
//! `fluxion-core` from importing anything from `sim/`, `physics/`,
//! `ai/`, or `validation/`, and as more Delta-related types land here
//! (M2, M4, M6) the entire measures sub-system stays self-contained.
//!
//! # Example
//!
//! ```
//! use fluxion::measures::json_patch::apply_delta;
//! use fluxion::measures::model::FluxionModel;
//! use json_patch::Patch;
//! use serde_json::json;
//!
//! let mut model = FluxionModel::ashrae_140_case_600();
//! let patch: Patch = serde_json::from_value(json!([
//!     { "op": "replace", "path": "/zones/zone_1/volume", "value": 200.0 }
//! ])).unwrap();
//!
//! apply_delta(&mut model, &patch).unwrap();
//! assert_eq!(model.zones["zone_1"].volume, 200.0);
//! ```

pub mod error;
pub mod json_patch;
pub mod model;
pub mod provenance;

// Convenience re-exports so callers can write
// `use fluxion::measures::{FluxionModel, apply_delta, DeltaError};`
pub use error::DeltaError;
pub use json_patch::{apply_delta, apply_delta_with_name};
pub use model::{
    AssemblySpec, ConstructionSpec, FluxionModel, MaterialLayer, ZoneSpec, MEASURES_SCHEMA_VERSION,
};
pub use provenance::{digest_of_patch, logical_timestamp, AppliedDelta, DeltaSource};
