//! Python bindings for Fluxion Model interior structs (Issue #1812).
//!
//! This module is a thin facade: the implementation is decomposed into
//! focused submodules, and every public item is re-exported here so all
//! `crate::python::model_bindings::X` paths — and the `#[pymodule]`
//! registrations in `src/lib.rs` — are unchanged.
//!
//! - [`model`] — the [`Model`](model::Model) pyclass, the snapshot pyclasses
//!   (`PyOrientation`, `PyShadingDevice`, `PyMaterial`, `PySurface`, `PyZone`)
//!   and the model<->snapshot helpers,
//! - [`hvac`] — the [`PyHVACSystem`](hvac::PyHVACSystem) snapshot pyclass and
//!   the HVAC model<->snapshot helpers,
//! - [`batch`] — the batch-evaluation logic behind `Model`'s
//!   `get_parameter_bounds` / `validate_parameters_py` entrypoints.
//!
//! See [`model`] for the PyO3 lifetime / ownership story.

mod batch;
mod hvac;
mod model;

pub use hvac::{apply_hvac_system_to_model, hvac_system_from_model, PyHVACSystem};
pub(crate) use model::populate_default_model_physics;
pub use model::{
    all_surfaces_from_model, all_zones_from_model, reshape_surfaces_for_model, surface_to_wall,
    zone_from_model, Model, PyMaterial, PyOrientation, PyShadingDevice, PyShadingType, PySurface,
    PyZone,
};
