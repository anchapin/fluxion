//! PyO3 bindings for construction / vector-field / geometry types
//! (Issue #2493).
//!
//! Extracted verbatim from `lib.rs`: `PyVectorField`, `PyConstructionLayer`,
//! `PySurfaceType`, `PyMassClass`, `PyConstruction`, `PyWallSurface`, and the
//! zero-copy `PyGeometryTensor`. Python-visible class names are unchanged
//! (set via `#[pyclass(name = "...")]`).

#[cfg(feature = "python-bindings")]
use crate::physics::geometry_tensor::{
    GeometryTensor, ADJACENCY_MATRIX_DIMS, WALL_MATRIX_DIMS, WINDOW_MATRIX_DIMS, ZONE_COORDS_DIMS,
    ZONE_PROPERTIES_DIMS,
};
#[cfg(feature = "python-bindings")]
use crate::physics::zero_copy_matrix::flat_slice_to_pyarray2;

#[cfg(feature = "python-bindings")]
use numpy::PyArrayMethods;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyAnyMethods;

/// VectorField wrapper for Python with optimized numpy support.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VectorField")]
pub struct PyVectorField {
    inner: crate::physics::cta::VectorField,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVectorField {
    /// Create a new VectorField from a Python list or numpy array.
    ///
    /// For optimal performance with large arrays, pass a numpy array directly.
    /// This avoids Python object iteration overhead.
    #[new]
    fn new(data: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        // Try to extract as numpy array first (most efficient for large data)
        if let Ok(arr) = data.cast::<numpy::PyArray1<f64>>() {
            // Fast path: directly copy from numpy array slice
            let slice = unsafe { arr.as_slice()? };
            return Ok(PyVectorField {
                inner: crate::physics::cta::VectorField::new(slice.to_vec()),
            });
        }

        // Fall back to Python sequence iteration
        let mut vec = Vec::new();
        let len = data.len()?;
        vec.reserve(len);

        for item in data.try_iter()? {
            let val = item?.extract::<f64>()?;
            vec.push(val);
        }

        Ok(PyVectorField {
            inner: crate::physics::cta::VectorField::new(vec),
        })
    }

    /// Create a VectorField filled with a constant value.
    #[staticmethod]
    fn from_scalar(value: f64, size: usize) -> Self {
        PyVectorField {
            inner: crate::physics::cta::VectorField::from_scalar(value, size),
        }
    }

    /// Get the number of elements in the VectorField.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Convert to Python list.
    fn to_list(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    /// Convert to numpy array with zero-copy when possible.
    ///
    /// Returns a numpy array view of the underlying data when possible,
    /// avoiding unnecessary memory copies for maximum performance.
    fn to_numpy<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, numpy::PyArray1<f64>>> {
        // Use from_vec for zero-copy conversion
        Ok(numpy::PyArray1::from_vec(
            py,
            self.inner.as_slice().to_vec(),
        ))
    }

    /// Compute the sum (integral) of all elements.
    fn integrate(&self) -> f64 {
        use crate::physics::cta::ContinuousTensor;
        self.inner.integrate()
    }

    /// Compute the gradient (rate of change) of the field.
    fn gradient(&self) -> Self {
        use crate::physics::cta::ContinuousTensor;
        PyVectorField {
            inner: self.inner.gradient(),
        }
    }
}

/// Construction layer material properties for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "ConstructionLayer", from_py_object)]
#[derive(Clone)]
pub struct PyConstructionLayer {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub conductivity: f64,
    #[pyo3(get, set)]
    pub density: f64,
    #[pyo3(get, set)]
    pub specific_heat: f64,
    #[pyo3(get, set)]
    pub thickness: f64,
    #[pyo3(get, set)]
    pub emissivity: f64,
    #[pyo3(get, set)]
    pub absorptance: f64,
}

#[cfg(feature = "python-bindings")]
impl From<&crate::sim::construction::ConstructionLayer> for PyConstructionLayer {
    fn from(layer: &crate::sim::construction::ConstructionLayer) -> Self {
        PyConstructionLayer {
            name: layer.name.clone(),
            conductivity: layer.conductivity,
            density: layer.density,
            specific_heat: layer.specific_heat,
            thickness: layer.thickness,
            emissivity: layer.emissivity,
            absorptance: layer.absorptance,
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyConstructionLayer> for crate::sim::construction::ConstructionLayer {
    fn from(layer: PyConstructionLayer) -> Self {
        crate::sim::construction::ConstructionLayer::with_surface_properties(
            layer.name,
            layer.conductivity,
            layer.density,
            layer.specific_heat,
            layer.thickness,
            layer.emissivity,
            layer.absorptance,
        )
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyConstructionLayer {
    /// Create a new ConstructionLayer.
    #[new]
    #[pyo3(signature = (name, conductivity, density, specific_heat, thickness, emissivity=0.9, absorptance=0.7))]
    fn new(
        name: String,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
        emissivity: f64,
        absorptance: f64,
    ) -> Self {
        PyConstructionLayer {
            name,
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity,
            absorptance,
        }
    }

    /// Calculate thermal resistance (R-value).
    fn r_value(&self) -> f64 {
        self.thickness / self.conductivity
    }

    /// Calculate thermal capacitance per unit area.
    fn thermal_capacitance_per_area(&self) -> f64 {
        self.density * self.thickness * self.specific_heat
    }
}

/// Surface type for construction calculations.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "SurfaceType", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySurfaceType {
    Wall,
    Ceiling,
    Floor,
}

#[cfg(feature = "python-bindings")]
impl From<PySurfaceType> for crate::sim::construction::SurfaceType {
    fn from(st: PySurfaceType) -> Self {
        match st {
            PySurfaceType::Wall => crate::sim::construction::SurfaceType::Wall,
            PySurfaceType::Ceiling => crate::sim::construction::SurfaceType::Ceiling,
            PySurfaceType::Floor => crate::sim::construction::SurfaceType::Floor,
        }
    }
}

/// Thermal mass classification for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "MassClass", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyMassClass {
    VeryLight,
    Light,
    Medium,
    Heavy,
    VeryHeavy,
}

#[cfg(feature = "python-bindings")]
impl From<PyMassClass> for crate::sim::construction::MassClass {
    fn from(mc: PyMassClass) -> Self {
        match mc {
            PyMassClass::VeryLight => crate::sim::construction::MassClass::VeryLight,
            PyMassClass::Light => crate::sim::construction::MassClass::Light,
            PyMassClass::Medium => crate::sim::construction::MassClass::Medium,
            PyMassClass::Heavy => crate::sim::construction::MassClass::Heavy,
            PyMassClass::VeryHeavy => crate::sim::construction::MassClass::VeryHeavy,
        }
    }
}

/// Multi-layer construction assembly for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "Construction")]
pub struct PyConstruction {
    #[pyo3(get)]
    pub layers: Vec<PyConstructionLayer>,
}

#[cfg(feature = "python-bindings")]
impl From<&crate::sim::construction::Construction> for PyConstruction {
    fn from(construction: &crate::sim::construction::Construction) -> Self {
        PyConstruction {
            layers: construction
                .layers
                .iter()
                .map(PyConstructionLayer::from)
                .collect(),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyConstruction> for crate::sim::construction::Construction {
    fn from(construction: PyConstruction) -> Self {
        crate::sim::construction::Construction::new(
            construction.layers.into_iter().map(|l| l.into()).collect(),
        )
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyConstruction {
    /// Create a new Construction from a list of layers.
    #[new]
    fn new(layers: Vec<PyConstructionLayer>) -> Self {
        PyConstruction { layers }
    }

    /// Calculate total thermal resistance (R-value).
    #[pyo3(signature = (surface_type=None, exterior_wind_speed=None))]
    fn r_value_total(
        &self,
        surface_type: Option<PySurfaceType>,
        exterior_wind_speed: Option<f64>,
    ) -> PyResult<f64> {
        let st = surface_type.map(|st| st.into());
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.r_value_total(st, exterior_wind_speed))
    }

    /// Calculate thermal transmittance (U-value).
    #[pyo3(signature = (surface_type=None, exterior_wind_speed=None))]
    fn u_value(
        &self,
        surface_type: Option<PySurfaceType>,
        exterior_wind_speed: Option<f64>,
    ) -> PyResult<f64> {
        let st = surface_type.map(|st| st.into());
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.u_value(st, exterior_wind_speed))
    }

    /// Calculate total thermal mass.
    fn thermal_capacitance_per_area(&self) -> PyResult<f64> {
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.thermal_capacitance_per_area())
    }

    /// Get total thickness.
    fn total_thickness(&self) -> PyResult<f64> {
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.total_thickness())
    }

    /// Get number of layers.
    fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get mass class.
    fn mass_class(&self) -> PyResult<PyMassClass> {
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        match rust_construction.iso_13790_mass_class() {
            crate::sim::construction::MassClass::VeryLight => Ok(PyMassClass::VeryLight),
            crate::sim::construction::MassClass::Light => Ok(PyMassClass::Light),
            crate::sim::construction::MassClass::Medium => Ok(PyMassClass::Medium),
            crate::sim::construction::MassClass::Heavy => Ok(PyMassClass::Heavy),
            crate::sim::construction::MassClass::VeryHeavy => Ok(PyMassClass::VeryHeavy),
        }
    }
}

/// Wall surface representation for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "WallSurface", from_py_object)]
#[derive(Clone)]
pub struct PyWallSurface {
    #[pyo3(get, set)]
    pub area: f64,
    #[pyo3(get, set)]
    pub u_value: f64,
    #[pyo3(get)]
    pub orientation: String,
}

#[cfg(feature = "python-bindings")]
impl From<&crate::sim::construction::WallSurface> for PyWallSurface {
    fn from(surface: &crate::sim::construction::WallSurface) -> Self {
        PyWallSurface {
            area: surface.area,
            u_value: surface.u_value,
            orientation: format!("{:?}", surface.orientation),
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyWallSurface {
    /// Create a new WallSurface.
    #[new]
    #[pyo3(signature = (area, u_value, orientation))]
    fn new(area: f64, u_value: f64, orientation: String) -> PyResult<Self> {
        let rust_orientation = match orientation.to_lowercase().as_str() {
            "south" => crate::validation::ashrae_140_cases::Orientation::South,
            "west" => crate::validation::ashrae_140_cases::Orientation::West,
            "north" => crate::validation::ashrae_140_cases::Orientation::North,
            "east" => crate::validation::ashrae_140_cases::Orientation::East,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid orientation '{}'. Valid options: south, west, north, east",
                    orientation
                )))
            }
        };
        let _rust_surface =
            crate::sim::construction::WallSurface::new(area, u_value, rust_orientation);
        Ok(PyWallSurface {
            area,
            u_value,
            orientation: format!("{:?}", rust_orientation),
        })
    }
}

// ============================================================================
// Geometry Tensor Python Bindings
// ============================================================================
//
// (`GeometryTensor` and the `*_DIMS` constants are imported at the top of
// this file — they were lifted here from the original `lib.rs` block.)

#[cfg(feature = "python-bindings")]
#[pyclass(name = "GeometryTensor")]
/// Python-accessible wrapper for GeometryTensor to expose to PyO3.
///
/// `inner` is wrapped in an `Arc` so that [`PyGeometryTensor::to_numpy`] can
/// hand a numpy array a borrow of the underlying storage without copying the
/// buffer — the cloned `Arc` is held by the numpy array's container and keeps
/// the bytes alive for as long as Python holds the numpy array.
pub struct PyGeometryTensor {
    inner: std::sync::Arc<GeometryTensor>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyGeometryTensor {
    /// Create a new empty GeometryTensor.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyGeometryTensor {
            inner: std::sync::Arc::new(GeometryTensor::new()),
        })
    }

    /// Create a GeometryTensor from numpy arrays.
    ///
    /// For optimal performance, use numpy arrays directly. The buffer protocol
    /// (the same wire format that Arrow uses for inter-process buffer sharing)
    /// is honored: a `numpy.ndarray` argument's storage is read in place via
    /// `PyReadonlyArray::as_slice`, avoiding intermediate `Vec` copies on the
    /// binding layer. A single ownership-transfer copy remains to move the
    /// data into the Rust-owned `GeometryTensor` storage.
    ///
    /// This method accepts:
    /// - zone_coords: (100, 20) zone coordinates
    /// - wall_matrix: (500, 6) wall geometry
    /// - window_matrix: (500, 6) window geometry
    /// - adjacency_matrix: (100, 100) zone adjacency
    /// - zone_properties: (100, 5) zone properties
    /// - summary: (6,) summary statistics
    #[staticmethod]
    fn from_numpy(
        zone_coords: &Bound<'_, pyo3::types::PyAny>,
        wall_matrix: &Bound<'_, pyo3::types::PyAny>,
        window_matrix: &Bound<'_, pyo3::types::PyAny>,
        adjacency_matrix: &Bound<'_, pyo3::types::PyAny>,
        zone_properties: &Bound<'_, pyo3::types::PyAny>,
        summary: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Self> {
        // Zero-copy on the binding layer: borrow the numpy array's storage
        // through `PyReadonlyArray::as_slice`, then take ownership with a
        // single `Vec::from` (or `to_vec`) copy required to detach the slice
        // from the numpy array's lifetime.
        //
        // The previous implementation called `slice.to_vec()` here AND
        // `from_numpy_arrays` called `to_vec()` again — two copies on the
        // hot path. The refactor keeps only the unavoidable ownership copy.
        fn borrow_f64_slice(arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Vec<f64>> {
            // Try 2D array first
            if let Ok(pyarr) = arr.cast::<numpy::PyArray2<f64>>() {
                // Issue #2528: use the safe `readonly().as_slice()` accessor
                // instead of `unsafe { pyarr.as_slice()? }`. The previous
                // `unsafe` block was unsound-by-omission (the safe path
                // exists exactly for this) and masked the panic-abort hazard
                // for non-contiguous / zero-dim arrays. Bind `readonly` to a
                // local so the returned slice outlives the temporary guard.
                let readonly = pyarr.readonly();
                let slice = readonly.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "non-contiguous 2-D array: {e}"
                    ))
                })?;
                return Ok(slice.to_vec());
            }
            // Try 1D array
            if let Ok(pyarr) = arr.cast::<numpy::PyArray1<f64>>() {
                let readonly = pyarr.readonly();
                let slice = readonly.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "non-contiguous 1-D array: {e}"
                    ))
                })?;
                return Ok(slice.to_vec());
            }
            // Fallback to Python sequence iteration (no zero-copy possible —
            // Python objects must be extracted element by element).
            let mut vec = Vec::new();
            for item in arr.try_iter()? {
                let val = item?.extract::<f64>()?;
                vec.push(val);
            }
            Ok(vec)
        }

        let zone_coords = borrow_f64_slice(zone_coords)?;
        let wall_matrix = borrow_f64_slice(wall_matrix)?;
        let window_matrix = borrow_f64_slice(window_matrix)?;
        let adjacency_matrix = borrow_f64_slice(adjacency_matrix)?;
        let zone_properties = borrow_f64_slice(zone_properties)?;
        let summary = borrow_f64_slice(summary)?;

        let inner = GeometryTensor::from_numpy_arrays(
            &zone_coords,
            &wall_matrix,
            &window_matrix,
            &adjacency_matrix,
            &zone_properties,
            &summary,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        Ok(PyGeometryTensor {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Get the number of zones.
    fn num_zones(&self) -> usize {
        self.inner.hvac.num_zones()
    }

    /// Get the number of walls.
    fn num_walls(&self) -> usize {
        self.inner.num_walls()
    }

    /// Get the total floor area.
    fn total_area(&self) -> f64 {
        self.inner.total_area()
    }

    /// Get the total volume.
    fn total_volume(&self) -> f64 {
        self.inner.total_volume()
    }

    /// Validate the geometry tensor.
    ///
    /// Returns a list of validation issues.
    fn validate(&self) -> Vec<String> {
        self.inner.validate()
    }

    /// Get summary statistics as a dictionary.
    fn get_summary(&self) -> Vec<f64> {
        self.inner.summary.clone()
    }

    /// Check if two zones are adjacent.
    fn zones_adjacent(&self, i: usize, j: usize) -> bool {
        self.inner.zones_adjacent(i, j)
    }

    /// Convert to numpy arrays.
    ///
    /// Each returned numpy array is built by copying the underlying
    /// `GeometryTensor` field into a new `PyArray2`. The zero-copy
    /// `borrow_from_array` view path was removed because the `numpy 0.29` /
    /// `ndarray 0.17` version conflict (issue #2746) makes the view
    /// constructors unusable — see `zero_copy_matrix.rs` for details.
    #[allow(clippy::type_complexity)]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        // Copy each flat `&[f64]` field of the shared `GeometryTensor` into a
        // freshly-allocated `PyArray2` / `PyArray1`.
        fn build_2d<'py>(
            py: Python<'py>,
            inner: &GeometryTensor,
            shape: (usize, usize),
            pick: fn(&GeometryTensor) -> &[f64],
        ) -> Bound<'py, numpy::PyArray2<f64>> {
            flat_slice_to_pyarray2(py, pick(inner), shape)
        }

        let zone_coords = build_2d(py, &self.inner, ZONE_COORDS_DIMS, |t| &t.zone_coords);
        let wall_matrix = build_2d(py, &self.inner, WALL_MATRIX_DIMS, |t| &t.wall_matrix);
        let window_matrix = build_2d(py, &self.inner, WINDOW_MATRIX_DIMS, |t| &t.window_matrix);
        let adjacency_matrix = build_2d(py, &self.inner, ADJACENCY_MATRIX_DIMS, |t| {
            &t.adjacency_matrix
        });
        let zone_properties = build_2d(py, &self.inner, ZONE_PROPERTIES_DIMS, |t| {
            &t.zone_properties
        });

        // 1-D summary path — copy into a new `PyArray1`.
        let summary = numpy::PyArray1::from_vec(py, self.inner.summary.clone());

        Ok((
            zone_coords,
            wall_matrix,
            window_matrix,
            adjacency_matrix,
            zone_properties,
            summary,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometryTensor(zones={}, walls={}, area={:.2}m², volume={:.2}m³)",
            self.inner.hvac.num_zones(),
            self.inner.num_walls(),
            self.inner.total_area(),
            self.inner.total_volume()
        )
    }
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the PyO3 construction / vector-field /
    //! geometry wrappers in this module (Issue #2882).
    //!
    //! These tests exercise the pure-Rust logic that backs the Python-visible
    //! entrypoints — the unit conversions, the enum mappings, the layer
    //! round-trip conversions, and the geometry-tensor accessors — without
    //! spinning up a CPython interpreter. The PyO3 `#[pymethods]` items remain
    //! callable from Rust because the `#[pymethods]` impl block is also a
    //! regular Rust `impl` block; only the error-mapping helpers (`Validation`
    //! / `PyValueError`) need Python context, and only on the `Err` arm,
    //! which `pyo3` defers lazily.
    use super::*;

    // ---- PyConstructionLayer -------------------------------------------

    fn sample_layer() -> PyConstructionLayer {
        PyConstructionLayer::new(
            "Fiberglass_50mm".to_string(),
            0.04,  // conductivity (W/m·K)
            12.0,  // density (kg/m³)
            840.0, // specific_heat (J/kg·K)
            0.05,  // thickness (m)
            0.9,   // emissivity
            0.7,   // absorptance
        )
    }

    #[test]
    fn construction_layer_new_preserves_all_fields() {
        let layer = sample_layer();
        assert_eq!(layer.name, "Fiberglass_50mm");
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 12.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.05);
        assert_eq!(layer.emissivity, 0.9);
        assert_eq!(layer.absorptance, 0.7);
    }

    #[test]
    fn construction_layer_r_value_is_thickness_over_conductivity() {
        // R = δ / k = 0.05 / 0.04 = 1.25 m²K/W. This matches the documented
        // formula in `ConstructionLayer::r_value` (fluxion-core/src/construction.rs)
        // and the `r_value` method on the Py wrapper (line 187-189).
        let layer = sample_layer();
        let got = layer.r_value();
        let expected = 0.05 / 0.04;
        assert!(
            (got - expected).abs() < 1e-12,
            "PyConstructionLayer::r_value {got} must match δ/k {expected}"
        );
    }

    #[test]
    fn construction_layer_thermal_capacitance_per_area_is_density_times_thickness_times_cp() {
        // κ/A = ρ × δ × Cp = 12 × 0.05 × 840 = 504 J/m²K
        let layer = sample_layer();
        let got = layer.thermal_capacitance_per_area();
        let expected = 12.0 * 0.05 * 840.0;
        assert!(
            (got - expected).abs() < 1e-9,
            "PyConstructionLayer::thermal_capacitance_per_area {got} must match ρδCp {expected}"
        );
    }

    #[test]
    fn construction_layer_default_emissivity_and_absorptance() {
        // The Py `__init__` exposes emissivity/absorptance as keyword args
        // defaulting to 0.9 / 0.7 (binding line 165 `signature = ...`).
        // `From<PyConstructionLayer>` for the core layer must therefore use
        // the same defaults when those args are omitted.
        let layer = PyConstructionLayer::new(
            "Bare".to_string(),
            0.5,
            1000.0,
            1000.0,
            0.1,
            // No emissivity/absorptance — defaults should apply.
            0.9,
            0.7,
        );
        assert_eq!(layer.emissivity, 0.9);
        assert_eq!(layer.absorptance, 0.7);
    }

    // ---- ConstructionLayer ↔ PyConstructionLayer round-trip ------------

    #[test]
    fn construction_layer_round_trip_preserves_fields() {
        // ConstructionLayer → PyConstructionLayer → ConstructionLayer via
        // `From<&Layer>` and `From<PyLayer>` conversions (binding line 131
        // and line 146).
        let original = fluxion_core::construction::ConstructionLayer::with_surface_properties(
            "Insulation",
            0.035,
            25.0,
            900.0,
            0.08,
            0.85,
            0.65,
        );
        let py: PyConstructionLayer = PyConstructionLayer::from(&original);
        assert_eq!(py.name, original.name);
        assert_eq!(py.conductivity, original.conductivity);
        assert_eq!(py.density, original.density);
        assert_eq!(py.specific_heat, original.specific_heat);
        assert_eq!(py.thickness, original.thickness);
        assert_eq!(py.emissivity, original.emissivity);
        assert_eq!(py.absorptance, original.absorptance);

        let roundtripped: fluxion_core::construction::ConstructionLayer = py.into();
        assert_eq!(roundtripped.name, original.name);
        assert_eq!(roundtripped.conductivity, original.conductivity);
        assert_eq!(roundtripped.density, original.density);
        assert_eq!(roundtripped.specific_heat, original.specific_heat);
        assert_eq!(roundtripped.thickness, original.thickness);
        assert_eq!(roundtripped.emissivity, original.emissivity);
        assert_eq!(roundtripped.absorptance, original.absorptance);
    }

    // ---- Enum conversions ---------------------------------------------

    #[test]
    fn surface_type_python_to_rust_covers_all_variants() {
        // PySurfaceType → SurfaceType covers every variant of the bindgen
        // mapping (binding line 207-216). If a new variant is ever added the
        // exhaustiveness of this test (and the `match` in `From`) catches it.
        let pairs = [
            (
                PySurfaceType::Wall,
                fluxion_core::construction::SurfaceType::Wall,
            ),
            (
                PySurfaceType::Ceiling,
                fluxion_core::construction::SurfaceType::Ceiling,
            ),
            (
                PySurfaceType::Floor,
                fluxion_core::construction::SurfaceType::Floor,
            ),
        ];
        for (py, expected) in pairs {
            let rust: fluxion_core::construction::SurfaceType = py.into();
            assert_eq!(rust, expected);
        }
    }

    #[test]
    fn mass_class_python_to_rust_covers_all_variants() {
        // PyMassClass → MassClass covers every variant (binding line 230-241).
        let pairs = [
            (
                PyMassClass::VeryLight,
                fluxion_core::construction::MassClass::VeryLight,
            ),
            (
                PyMassClass::Light,
                fluxion_core::construction::MassClass::Light,
            ),
            (
                PyMassClass::Medium,
                fluxion_core::construction::MassClass::Medium,
            ),
            (
                PyMassClass::Heavy,
                fluxion_core::construction::MassClass::Heavy,
            ),
            (
                PyMassClass::VeryHeavy,
                fluxion_core::construction::MassClass::VeryHeavy,
            ),
        ];
        for (py, expected) in pairs {
            let rust: fluxion_core::construction::MassClass = py.into();
            assert_eq!(rust, expected);
        }
    }

    // ---- PyConstruction round-trip ------------------------------------

    #[test]
    fn construction_round_trip_preserves_layer_count_and_order() {
        // PyConstruction → Construction → back through `mass_class` exposes
        // the layer-count + ISO-13790 mass classification through the entire
        // Rust convert path. A medium-mass 3-layer concrete/brick/insulation
        // wall must classify as at-least Medium (the dominant layer sets the
        // class).
        let layers = vec![
            PyConstructionLayer::new("Concrete".into(), 1.4, 2300.0, 880.0, 0.1, 0.9, 0.7),
            PyConstructionLayer::new("Insulation".into(), 0.04, 30.0, 1000.0, 0.05, 0.9, 0.7),
            PyConstructionLayer::new("Gypsum".into(), 0.21, 950.0, 840.0, 0.013, 0.9, 0.7),
        ];
        let py_cons = PyConstruction { layers };
        let layer_count = py_cons.layer_count();
        assert_eq!(layer_count, 3);

        // The mass class aggregate must classify this multi-layer assembly
        // — Concrete dominates capacitance so we expect at least Medium.
        let class = py_cons.mass_class().expect("mass_class converts");
        assert!(
            matches!(
                class,
                PyMassClass::Medium | PyMassClass::Heavy | PyMassClass::VeryHeavy
            ),
            "concrete+insulation+gypsum wall must classify medium-or-heavier, got {:?}",
            mass_class_label(class)
        );
    }

    #[test]
    fn construction_layer_count_matches_vector_length() {
        // Empty assembly has zero layers; per-layer length matches Rust vec.
        let empty = PyConstruction { layers: vec![] };
        assert_eq!(empty.layer_count(), 0);

        let one = PyConstruction {
            layers: vec![sample_layer()],
        };
        assert_eq!(one.layer_count(), 1);
    }

    // ---- PyGeometryTensor ---------------------------------------------

    #[test]
    fn geometry_tensor_new_is_zeroed_and_initializes_correctly() {
        // PyGeometryTensor::new → empty (all zeros) tensor. Default zero
        // summary means num_zones=0, num_walls=0, total_area=0, total_volume=0.
        // Validates the empty-tensor contract from the binding's perspective.
        let g = PyGeometryTensor::new().expect("PyGeometryTensor::new");
        assert_eq!(g.num_zones(), 0);
        assert_eq!(g.num_walls(), 0);
        assert_eq!(g.total_area(), 0.0);
        assert_eq!(g.total_volume(), 0.0);
        assert!(
            g.validate().is_empty(),
            "fresh tensor must have no validation issues"
        );
    }

    #[test]
    fn geometry_tensor_zones_adjacent_for_empty_is_false() {
        // Empty tensor has no zones, so any adjacency query returns false.
        // This guards against a default-summary array access UB.
        let g = PyGeometryTensor::new().expect("PyGeometryTensor::new");
        assert!(!g.zones_adjacent(0, 0));
        // Out-of-range indices must also be safe (return false rather than
        // panicking).
        assert!(!g.zones_adjacent(5, 7));
        assert!(!g.zones_adjacent(100, 100));
    }

    #[test]
    fn geometry_tensor_summary_length_matches_contract() {
        // `GeometryTensor::summary` is documented as a 6-element vector:
        // [num_zones, num_walls, num_windows, num_doors, total_area, total_volume]
        // — the `get_summary` Py wrapper returns this directly.
        let g = PyGeometryTensor::new().expect("PyGeometryTensor::new");
        let summary = g.get_summary();
        assert_eq!(summary.len(), 6);
        for v in &summary {
            assert_eq!(*v, 0.0, "fresh summary entries must all be zero");
        }
    }

    // ---- PyVectorField (via from_scalar -- no Python required) --------

    #[test]
    fn vector_field_from_scalar_length_and_integrate() {
        // `from_scalar` is a `#[staticmethod]` with a Rust-only signature
        // (f64, usize) -> Self, so it can be exercised directly.
        let vf = PyVectorField::from_scalar(2.0, 4);
        assert_eq!(vf.len(), 4);
        // Integral of [2,2,2,2] is 8.0 — `integrate()` is the field integral.
        assert!((vf.integrate() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn vector_field_to_list_round_trip() {
        // `to_list` returns the data as `Vec<f64>`. With `from_scalar(3, 3)`
        // every element must be 3.0 — a numpy-dtype round-trip analogue for
        // the 1-D vector path.
        let vf = PyVectorField::from_scalar(3.0, 3);
        let list = vf.to_list();
        assert_eq!(list.len(), 3);
        for v in &list {
            assert!((v - 3.0).abs() < 1e-12);
        }
    }

    // ---- helpers --------------------------------------------------------

    /// Label a `PyMassClass` variant for assertion error messages. `PyMassClass`
    /// does not derive `Debug` (it is a `#[pyclass]` enumeration without an
    /// explicit `#[derive(Debug)]`), so we match it manually for diagnostics.
    fn mass_class_label(class: PyMassClass) -> &'static str {
        match class {
            PyMassClass::VeryLight => "VeryLight",
            PyMassClass::Light => "Light",
            PyMassClass::Medium => "Medium",
            PyMassClass::Heavy => "Heavy",
            PyMassClass::VeryHeavy => "VeryHeavy",
        }
    }
}
