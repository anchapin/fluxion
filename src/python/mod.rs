// Python module exports for Fluxion
// This module re-exports all Python bindings including multi-zone functionality

// `osm_bindings` and `bindings` both define a top-level `export_osm` helper that
// wraps the same underlying `interop::osm` function (different Python-facing
// variants). Glob-re-exporting both is intentional and the dupe is benign —
// at the Python layer only `osm_bindings::export_osm` is registered (see
// `add_function!(osm_bindings::export_osm, m)` below). To avoid splitting the
// glob re-exports into per-item aliases (a much larger refactor with no
// behaviour change) we silence `ambiguous_glob_reexports` for the two
// conflicting imports and document the intent here.
#[allow(ambiguous_glob_reexports)]
#[cfg(feature = "python-bindings")]
pub use bindings::*;

#[allow(ambiguous_glob_reexports)]
#[cfg(feature = "python-bindings")]
pub use osm_bindings::*;

pub mod bindings;
#[cfg(feature = "python-bindings")]
pub mod hvac_bindings;
#[cfg(feature = "python-bindings")]
pub mod model_bindings;
// Issue #2528: panic-safety hook + unsafe-site shape validation.
#[cfg(feature = "python-bindings")]
pub mod multi_node_bindings;
#[cfg(feature = "python-bindings")]
pub mod osm_bindings;
#[cfg(feature = "python-bindings")]
pub mod panic_hook;

#[cfg(feature = "python-bindings")]
pub use hvac_bindings::*;
#[cfg(feature = "python-bindings")]
pub use model_bindings::*;
#[cfg(feature = "python-bindings")]
pub use multi_node_bindings::*;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Initialize the Python module
#[cfg(feature = "python-bindings")]
#[pymodule]
pub fn fluxion_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Issue #2528: same panic hook as the main `fluxion` module. Idempotent.
    crate::python::panic_hook::install();

    // Import and initialize the main fluxion module
    let _fluxion_module = PyModule::import(_py, "fluxion")?;

    // Re-export multi-zone functionality
    m.add_wrapped(pyo3::wrap_pymodule!(bindings::multi_zone))?;

    // Register HVAC classes directly
    m.add_class::<hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<hvac_bindings::PyZoneControl>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        hvac_bindings::create_zone_setpoints,
        m
    )?)?;

    m.add_class::<osm_bindings::PyOsmReader>()?;
    m.add_class::<osm_bindings::PyOsmWriter>()?;
    m.add_function(pyo3::wrap_pyfunction!(osm_bindings::import_osm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(osm_bindings::export_osm, m)?)?;

    // Register 9R4C Multi-Node Solver classes
    m.add_class::<multi_node_bindings::PyThermalMassNode>()?;
    m.add_class::<multi_node_bindings::PyMultiNodeThermalMass>()?;
    m.add_class::<multi_node_bindings::PyMassAirCouplingMode>()?;
    m.add_class::<multi_node_bindings::PySurfaceExteriorTemperatures>()?;
    m.add_class::<multi_node_bindings::PyMultiNodeSolver>()?;

    // Register FluxionModel interior struct bindings (Issue #1812).
    m.add_class::<model_bindings::PyOrientation>()?;
    m.add_class::<model_bindings::PyShadingType>()?;
    m.add_class::<model_bindings::PyShadingDevice>()?;
    m.add_class::<model_bindings::PyMaterial>()?;
    m.add_class::<model_bindings::PySurface>()?;
    m.add_class::<model_bindings::PyZone>()?;
    m.add_class::<model_bindings::PyHVACSystem>()?;

    Ok(())
}
