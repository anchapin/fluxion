// Python module exports for Fluxion
// This module re-exports all Python bindings including multi-zone functionality

pub mod bindings;
#[cfg(feature = "python-bindings")]
pub mod hvac_bindings;
#[cfg(feature = "python-bindings")]
pub mod multi_node_bindings;
#[cfg(feature = "python-bindings")]
pub mod osm_bindings;

#[cfg(feature = "python-bindings")]
pub use hvac_bindings::*;
#[cfg(feature = "python-bindings")]
pub use multi_node_bindings::*;
#[cfg(feature = "python-bindings")]
pub use osm_bindings::*;

#[cfg(feature = "python-bindings")]
pub use bindings::*;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Initialize the Python module
#[cfg(feature = "python-bindings")]
#[pymodule]
pub fn fluxion_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Import and initialize the main fluxion module
    let _fluxion_module = PyModule::import_bound(_py, "fluxion")?;

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

    Ok(())
}
