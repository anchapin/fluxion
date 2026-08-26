//! Fluxion: Rust-based Building Energy Modeling (BEM) engine
//!
//! Neuro-Symbolic hybrid architecture combining physics-based thermal networks with AI surrogates.
//! Designed for high-throughput evaluation of building design configurations (10,000+ configs/sec).
//!
//! # Architecture
//! - **BatchOracle**: High-throughput parallel evaluation for optimization loops
//! - **Model**: Single-building detailed analysis for validation and inspection
//! - **ThermalModel**: ISO 13790-compliant 5R1C/6R2C thermal network using CTA
//! - **SurrogateManager**: AI surrogate models for fast load prediction (ONNX Runtime)
//!
//! # Python API
//! ```python,ignore
//! from fluxion import BatchOracle, Model
//!
//! # Batch evaluation for optimization
//! oracle = BatchOracle()
//! results = oracle.evaluate_population([[1.5, 20.0, 22.0]], False)
//!
//! # Single building simulation
//! model = Model.from_case("600")
//! eui = model.simulate(years=1, use_surrogates=False)
//! ```
//!
//! # Performance
//! - Throughput: 10,000+ configurations/second on 8-core CPU
//! - Latency: <100ms for single configuration (8760 timesteps)
//! - Memory: Minimal allocations via CTA buffer reuse
//!
//! # Validation
//! - ASHRAE Standard 140 compliant (18/18 cases passing)
//! - Multi-reference validation (EnergyPlus, ESP-r, TRNSYS)
//! - Free-floating temperature validation (10/10 cases passing)
//!
//! # Modules
//! - [`sim::engine`] - ThermalModel and physics engine
//! - [`physics::cta`] - Continuous Tensor Abstraction
//! - [`ai::surrogate`] - ONNX-based surrogate models
//! - [`validation::ashrae_140_validator`] - ASHRAE 140 validation
//! - [`api`] - Python bindings and error types
//!
//! See [`BatchOracle`] and [`Model`] for Python API details.
//! See docs/API_REFERENCE.md for complete API documentation.

#![allow(clippy::useless_conversion)]
#![allow(nonstandard_style)]
#![allow(clippy::useless_vec)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::clone_on_ref_ptr)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::unnecessary_to_owned)]
#![allow(clippy::len_zero)]
#![allow(clippy::comparison_to_empty)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::expect_used)]
#![allow(clippy::derive_ord_xor_partial_ord)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::use_self)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::vec_init_then_push)]

// =============================================================================
// Module declarations
// =============================================================================

pub mod ai;
pub mod analysis;
pub mod api;
pub mod cli;
pub mod interop;
pub mod io;
pub mod measures;
pub mod napi;
pub mod orchestration;
pub mod performance;
pub mod physics;
#[cfg(feature = "python-bindings")]
pub mod python;
pub mod quantum;
pub mod sim;
pub mod solar;
pub mod testing;
pub mod thermal;
pub mod twin;
pub mod util;
pub mod validation;

// Issue #2493: BatchOracle core (struct + physics `evaluate_population` hot
// loop) was extracted here from `lib.rs`. The PyO3 surface (`#[pymethods]`)
// lives in [`python::batch_oracle_bindings`]. NOT feature-gated — the type is
// part of the public Rust API and is consumed by `analysis`, `bin/fluxion`,
// the NAPI bindings, and many integration tests without `python-bindings`.
pub mod batch_oracle;

// Engine + logging unit tests lifted out of `lib.rs` (Issue #2493).
#[cfg(test)]
mod lib_tests;

// =============================================================================
// Re-exports (fluxion-core leaf modules + sim traits)
// =============================================================================

// #1255: `weather` now lives in the `fluxion-core` workspace crate (a dependency
// leaf). Re-export it so all existing `crate::weather::...` paths resolve unchanged.
pub use fluxion_core::weather;

// #1349 (Phase 2 crate split): `assembly` and `multi_node` were moved from
// `src/sim/` into `fluxion-core` to break the physics<->sim dependency cycle.
// The original `src/sim/assembly.rs` and `src/sim/multi_node_thermal.rs` files
// are now thin re-export shims, so existing `crate::sim::assembly::*` and
// `crate::sim::multi_node_thermal::*` paths still resolve. Top-level re-exports
// here make `crate::assembly::*` and `crate::multi_node::*` work too.
pub use fluxion_core::{assembly, multi_node};

// #1441 (Phase 2 cycle break, continued): ASHRAE-140 leaf data types
// (Orientation, WindowArea, ConstructionType, ShadingType, ShadingDevice,
// GlassType, WindowSpec, InternalLoads, HvacSchedule, NightVentilation,
// BuildingType, GeometrySpec, ConductanceReferences) were moved from
// `src/validation/ashrae_140_cases.rs` into `fluxion_core::ashrae_cases` to
// break the `sim ↔ validation` cycle. The validation module re-exports each
// type at its original path, so `fluxion::validation::ashrae_140_cases::*`
// paths still resolve unchanged. This top-level re-export makes
// `fluxion::ashrae_cases::Orientation` work too.
pub use fluxion_core::ashrae_cases;

// #2462 (Phase 2 cycle break, continued): the remaining `physics ↔ sim`
// cycle edges documented in ARCHITECTURE.md §"Remaining cycles" were broken
// by hoisting `ConstructionLayer` + helpers into `fluxion_core::construction`,
// `PerSurfaceConductionSolver` + `SurfaceKind` (+ `MassNode`, `SurfaceNode`)
// into `fluxion_core::per_surface_conduction`, and the Stefan–Boltzmann
// constant out of `sim::sky_radiation` into the new
// `fluxion_core::physics_constants` leaf. The corresponding sim files
// (`src/sim/construction.rs`, `src/sim/per_surface_conduction.rs`,
// `src/sim/sky_radiation.rs`) keep the old paths alive via `pub use`
// re-exports, so `crate::sim::construction::*`, `crate::sim::per_surface_conduction::*`,
// and `crate::sim::sky_radiation::STEFAN_BOLTZMANN` all still resolve. The
// top-level re-exports below make `fluxion::construction::*`,
// `fluxion::per_surface_conduction::*`, and `fluxion::physics_constants::STEFAN_BOLTZMANN`
// work too.
pub use fluxion_core::{construction, per_surface_conduction, physics_constants};

// #2527 (DoS hardening): parser size/depth/repetition limits. Re-exported so
// `fluxion-mcp` / CLI / `BatchOracle` callers can pass
// `fluxion::parser_limits::ParserLimits::cli_default()` to the `_with_limits`
// parser entry points, and so the strict HTTP default is available as
// `ParserLimits::http_default()`.
pub use fluxion_core::parser_limits;

// Re-export thermal model traits for public API
pub use sim::surface_flux_provider::{
    MockSurfaceHeatFluxProvider, PhysicsSurfaceFluxProvider, SurfaceHeatFluxProvider,
};
pub use sim::thermal_model::{
    HybridRouting, HybridThermalModel, PhysicsThermalModel, SurrogateThermalModel,
    ThermalModelBuilder, ThermalModelMode, ThermalModelTrait, UnifiedThermalModel,
    ZoneComfortMetrics,
};
pub use sim::thermal_model_mock::MockThermalModel;

// Re-export ISO 13790 Annex C construction types
// Issue #2462: `sim::construction` is now a thin re-export shim over
// `fluxion_core::construction`; the re-export below keeps the historical
// `fluxion::sim::construction::ConstructionLayer` path alive. The explicit
// top-level `fluxion::construction::{Construction, ConstructionLayer, MassClass}`
// re-export is added in the `#2462` block above.
pub use sim::construction::{Construction, ConstructionLayer, MassClass};

// Re-export utility tariff types for financial cost tracking
pub use sim::utility_tariff::{CostAccumulator, DemandAccumulator, TouPeriod, UtilityTariff};

// Issue #2493: BatchOracle core (struct + analytical/surrogate
// `evaluate_population`) lives in [`batch_oracle`]. Re-exported at the crate
// root so the historical `fluxion::BatchOracle` / `crate::BatchOracle` Rust
// API paths — used by `analysis`, `bin/fluxion`, the NAPI layer, and many
// integration tests — resolve unchanged.
pub use batch_oracle::BatchOracle;

// Re-export ASHRAE 140 validation models
pub use validation::ashrae140::high_mass;

// =============================================================================
// Python bindings entrypoint
// =============================================================================
//
// `Model`, `BatchOracle`'s `#[pymethods]`, `ParameterBounds`, and the
// construction / geometry `#[pyclass]` types now live in [`python`] (Issue
// #2493). The crate root keeps only the `#[pymodule]` registration so it
// stays a thin wiring shim. Python-visible class names are unchanged because
// they are set via `#[pyclass(name = "...")]` (or default to the Rust struct
// name), so the published wheel's `fluxion.Model` / `fluxion.BatchOracle` /
// `fluxion.BuildingParameters` / etc. import paths remain identical.

// Issue #2528: re-export the FFI crates so integration tests under
// `tests/` (which link against `fluxion` as an external crate) can construct
// numpy arrays and acquire the GIL without their own `numpy`/`pyo3`
// `[dev-dependencies]` entry. Default-feature builds are unaffected: this
// `pub use` is feature-gated exactly like the imports below.
#[cfg(feature = "python-bindings")]
pub use {numpy, pyo3};

#[cfg(feature = "python-bindings")]
use crate::api::error::{FluxionErrorPy, SimulationError, SurrogateError, ValidationError};
#[cfg(feature = "python-bindings")]
use crate::api::parameters::BuildingParameters;
#[cfg(feature = "python-bindings")]
use pyo3::{
    prelude::{pymodule, PyModule},
    types::PyModuleMethods,
    Bound, PyResult, Python,
};

/// Initialize the `fluxion` Python module.
///
/// Wires up the `#[pyclass]` types defined in [`python`] and the crate-level
/// error/parameter types. The class registration order is preserved exactly
/// from the pre-#2493 `lib.rs` so the public Python API (`fluxion.Model`,
/// `fluxion.BatchOracle`, `fluxion.ParameterBounds`,
/// `fluxion.BuildingParameters`, `fluxion.VectorField`, …) is bit-for-bit
/// backwards-compatible.
#[cfg(feature = "python-bindings")]
#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Issue #2528: install the PyO3-aware panic hook before registering any
    // #[pyfunction]. A panic inside a #[pyfunction] (e.g. from the
    // `unsafe { array.as_slice() }` / `RawArrayView::from_shape_ptr` blocks)
    // would otherwise abort the host interpreter / leak source paths via the
    // default `std::panic` hook. Idempotent — safe across re-exports.
    crate::python::panic_hook::install();

    // Register custom exception types
    m.add("FluxionError", _py.get_type::<FluxionErrorPy>())?;
    m.add("ValidationError", _py.get_type::<ValidationError>())?;
    m.add("SurrogateError", _py.get_type::<SurrogateError>())?;
    m.add("SimulationError", _py.get_type::<SimulationError>())?;

    // Top-level classes. `Model` now lives in `python::model_bindings`; the
    // construction / vector-field / geometry types in
    // `python::construction_bindings`; `ParameterBounds` in
    // `python::batch_oracle_bindings`. `BatchOracle` is re-exported at the
    // crate root from `batch_oracle`, so `crate::BatchOracle` resolves to the
    // `#[pyclass]` struct unchanged.
    m.add_class::<python::model_bindings::Model>()?;
    m.add_class::<crate::BatchOracle>()?;
    m.add_class::<python::batch_oracle_bindings::ParameterBounds>()?;
    m.add_class::<BuildingParameters>()?;
    m.add_class::<python::construction_bindings::PyVectorField>()?;
    m.add_class::<python::construction_bindings::PyConstruction>()?;
    m.add_class::<python::construction_bindings::PyConstructionLayer>()?;
    m.add_class::<python::construction_bindings::PyMassClass>()?;
    m.add_class::<python::construction_bindings::PySurfaceType>()?;
    m.add_class::<python::construction_bindings::PyWallSurface>()?;
    m.add_class::<python::construction_bindings::PyGeometryTensor>()?;

    // Register multi-zone module
    python::multi_zone(_py, m)?;

    // Register HVAC classes directly in main module for now
    m.add_class::<python::hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<python::hvac_bindings::PyZoneControl>()?;
    m.add_class::<python::hvac_bindings::PyDailySchedule>()?;
    m.add_class::<python::hvac_bindings::PyHVACSchedule>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;

    // Deep HVAC configuration (Issue #1797): system-type / mode enums,
    // equipment types, and the detailed airside VAV terminal unit.
    m.add_class::<python::hvac_bindings::PyHVACSystemType>()?;
    m.add_class::<python::hvac_bindings::PyHVACMode>()?;
    m.add_class::<python::hvac_bindings::PyHeatPumpMode>()?;
    m.add_class::<python::hvac_bindings::PyVavOperatingMode>()?;
    m.add_class::<python::hvac_bindings::PyChiller>()?;
    m.add_class::<python::hvac_bindings::PyBoiler>()?;
    m.add_class::<python::hvac_bindings::PyHeatPump>()?;
    m.add_class::<python::hvac_bindings::PyVAVTerminal>()?;
    m.add_class::<python::hvac_bindings::PyCAVSystem>()?;
    m.add_class::<python::hvac_bindings::PyVavTerminalUnit>()?;
    m.add_class::<python::hvac_bindings::PyVavTerminalControl>()?;
    m.add_class::<python::hvac_bindings::PyVavTerminalPerformance>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        python::hvac_bindings::compute_vav_terminal_performance,
        m
    )?)?;

    m.add_class::<python::osm_bindings::PyOsmReader>()?;
    m.add_class::<python::osm_bindings::PyOsmWriter>()?;
    m.add_function(pyo3::wrap_pyfunction!(python::osm_bindings::import_osm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(python::osm_bindings::export_osm, m)?)?;

    // Register 9R4C Multi-Node Solver classes
    m.add_class::<python::multi_node_bindings::PyThermalMassNode>()?;
    m.add_class::<python::multi_node_bindings::PyMultiNodeThermalMass>()?;
    m.add_class::<python::multi_node_bindings::PyMassAirCouplingMode>()?;
    m.add_class::<python::multi_node_bindings::PySurfaceExteriorTemperatures>()?;
    m.add_class::<python::multi_node_bindings::PyMultiNodeSolver>()?;

    // Register FluxionModel interior struct bindings (Issue #1812).
    m.add_class::<python::model_bindings::PyOrientation>()?;
    m.add_class::<python::model_bindings::PyShadingType>()?;
    m.add_class::<python::model_bindings::PyShadingDevice>()?;
    m.add_class::<python::model_bindings::PyMaterial>()?;
    m.add_class::<python::model_bindings::PySurface>()?;
    m.add_class::<python::model_bindings::PyZone>()?;
    m.add_class::<python::model_bindings::PyHVACSystem>()?;

    Ok(())
}
