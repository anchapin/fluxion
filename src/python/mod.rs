// Python module exports for Fluxion
// This module re-exports all Python bindings including multi-zone functionality

pub mod bindings;
// TODO: Enable HVAC bindings once ThermalModel API is resolved
// #[cfg(feature = "python-bindings")]
// pub mod hvac_bindings;

#[cfg(feature = "python-bindings")]
pub use bindings::*;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Initialize the Python module
#[cfg(feature = "python-bindings")]
#[pymodule]
pub fn fluxion_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Import and initialize the main fluxion module
    let fluxion_module = PyModule::import_bound(_py, "fluxion")?;

    // Re-export multi-zone functionality
    m.add_wrapped(pyo3::wrap_pymodule!(bindings::multi_zone))?;

    Ok(())
}
