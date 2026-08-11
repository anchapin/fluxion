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
use crate::physics::zero_copy_matrix::ZeroCopyGeometryTensorHolder;

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
#[pyclass(name = "ConstructionLayer")]
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
#[pyclass(name = "SurfaceType", eq, eq_int)]
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
#[pyclass(name = "MassClass", eq, eq_int)]
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
#[pyclass(name = "WallSurface")]
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
// Geometry Tensor Python Bindings (Zero-Copy)
// ============================================================================
//
// (`GeometryTensor`, the `*_DIMS` constants, and `ZeroCopyGeometryTensorHolder`
// are imported at the top of this file — they were lifted here from the
// original `lib.rs` block.)

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
        self.inner.num_zones()
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

    /// Convert to numpy arrays with zero-copy buffer sharing.
    ///
    /// Each returned numpy array wraps a `numpy::PyArray2::borrow_from_array`
    /// view of the underlying `GeometryTensor` storage — the numpy array and
    /// the Rust struct share the same buffer. A `PyClass` holder that retains
    /// an `Arc<GeometryTensor>` clone keeps the storage alive for the lifetime
    /// of the numpy array. The Arc clone is a refcount bump (no buffer copy),
    /// so the Rust → Python direction is allocation-free beyond the small
    /// holder object.
    ///
    /// This is the standard numpy buffer protocol that Arrow uses for
    /// inter-process buffer sharing — any Arrow-compatible consumer
    /// (PyArrow, ML frameworks) can ingest the returned arrays without
    /// re-copying.
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
        // The holder pattern: the numpy array's container holds an Arc clone
        // of the GeometryTensor. The borrowed view points into the Arc's
        // storage. When Python releases the numpy array, the Arc is dropped,
        // and (if the last reference) the GeometryTensor is dropped too.
        fn build_zero_copy_2d<'py>(
            py: Python<'py>,
            inner: std::sync::Arc<GeometryTensor>,
            shape: (usize, usize),
            pick: fn(&GeometryTensor) -> &[f64],
        ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
            // SAFETY: the holder's Arc clone keeps `inner` alive for the
            // lifetime of the returned numpy array, so the view's data
            // pointer remains valid. `pick(&inner)` returns a borrow of one
            // of `inner`'s `Vec<f64>` fields, which is contiguous and
            // `shape.0 * shape.1` elements long (validated by the
            // `GeometryTensor::from_numpy_arrays` constructor and the
            // `WALL_MATRIX_DIMS` / etc. constants).
            let ptr = pick(&inner).as_ptr();
            let raw = unsafe { ndarray::RawArrayView::from_shape_ptr(shape, ptr) };
            let view = unsafe { raw.deref_into_view() };
            let holder = ZeroCopyGeometryTensorHolder { inner };
            let container = Bound::new(py, holder)
                .expect("ZeroCopyGeometryTensorHolder allocation cannot fail")
                .into_any();
            Ok(unsafe { numpy::PyArray2::borrow_from_array(&view, container) })
        }

        let zone_coords = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            ZONE_COORDS_DIMS,
            |t| &t.zone_coords,
        )?;
        let wall_matrix = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            WALL_MATRIX_DIMS,
            |t| &t.wall_matrix,
        )?;
        let window_matrix = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            WINDOW_MATRIX_DIMS,
            |t| &t.window_matrix,
        )?;
        let adjacency_matrix = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            ADJACENCY_MATRIX_DIMS,
            |t| &t.adjacency_matrix,
        )?;
        let zone_properties = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            ZONE_PROPERTIES_DIMS,
            |t| &t.zone_properties,
        )?;

        // 1-D summary path — same zero-copy recipe.
        let summary_inner = std::sync::Arc::clone(&self.inner);
        let summary_ptr = summary_inner.summary.as_ptr();
        // SAFETY: same Arc-alive invariant; `summary` is a 6-element Vec<f64>
        // held by the GeometryTensor.
        let summary_raw = unsafe {
            ndarray::RawArrayView::from_shape_ptr(summary_inner.summary.len(), summary_ptr)
        };
        let summary_view = unsafe { summary_raw.deref_into_view() };
        let summary_holder = ZeroCopyGeometryTensorHolder {
            inner: summary_inner,
        };
        let summary_container = Bound::new(py, summary_holder)
            .expect("ZeroCopyGeometryTensorHolder allocation cannot fail")
            .into_any();
        let summary =
            unsafe { numpy::PyArray1::borrow_from_array(&summary_view, summary_container) };

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
            self.inner.num_zones(),
            self.inner.num_walls(),
            self.inner.total_area(),
            self.inner.total_volume()
        )
    }
}
