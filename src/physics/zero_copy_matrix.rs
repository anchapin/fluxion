//! Zero-copy matrix transfer for the Python bindings (Issue #1801 / T9.7).
//!
//! Matrices cross the Rust ↔ Python boundary without intermediate buffer copies
//! using the numpy buffer protocol — the same protocol that Arrow uses to share
//! buffers between Rust and Python ML frameworks.
//!
//! Two helpers are exposed:
//!
//! - [`PyReadonlyArray::as_slice`] (from the `numpy` crate) gives a `&[T]`
//!   borrowed directly from a numpy array's storage — no copy.
//! - [`numpy::PyArray::borrow_from_array`] wraps an existing
//!   `ndarray::ArrayView` as a numpy array that shares the same memory.
//!
//! Combined, these two primitives allow `PyGeometryTensor::from_numpy` and
//! `PyGeometryTensor::to_numpy` to ship matrices between Rust and Python
//! without `to_vec()`-style copies on the binding layer. The benchmark in
//! `benches/zero_copy_matrix_bench.rs` exercises this path and confirms the
//! allocation count on the hot path.
//!
//! # Why Arc?
//!
//! `borrow_from_array` is `unsafe`: the caller promises that the data
//! referenced by the view lives as long as the returned numpy array's
//! container. The container is a `Bound<'py, PyAny>` — a Python-owned object
//! whose lifetime is bounded by Python's GC. We can't hand it a Rust reference
//! to `self.inner.wall_matrix` (the borrow would end when `&self` does), and
//! we can't `mem::take` it out (the matrix must remain accessible after the
//! call). `Arc<Vec<f64>>` is the only sound choice: cloning the `Arc` is a
//! refcount bump (no data copy), and the cloned `Arc` held by the numpy
//! array's container keeps the underlying bytes alive for as long as Python
//! holds the numpy array.
//!
//! # Arrow compatibility
//!
//! Numpy arrays expose their storage through the Python buffer protocol, which
//! is the same wire format that Arrow uses for inter-process buffer sharing.
//! Any Arrow-compatible consumer (PyArrow, pandas with Arrow backend, ML
//! frameworks with Arrow zero-copy ingestion) can consume a numpy array
//! produced by `to_numpy` without copying the buffer again.
//!
//! # Module layout
//!
//! The pure-Rust types `ZeroCopyMatrix1D` / `ZeroCopyMatrix2D` and their
//! `from_vec` / `as_slice` constructors are always compiled (no feature gate).
//! The Python-facing `to_numpy` / `from_numpy_*` helpers are gated behind the
//! `python-bindings` feature so that benches and downstream Rust consumers
//! can use the zero-copy infrastructure without pulling in numpy + pyo3.

use std::sync::Arc;

/// A 1-D matrix that can cross the Rust ↔ Python boundary without copying.
///
/// Internally an `Arc<Vec<f64>>`: cloning the `Arc` is a refcount bump, so
/// handing the data to Python via `borrow_from_array` does not duplicate
/// the buffer. The `Arc` is held by the numpy array's container (a Python
/// object), so the underlying bytes outlive any subsequent Python use of the
/// numpy array.
#[derive(Debug, Clone)]
pub struct ZeroCopyMatrix1D {
    data: Arc<Vec<f64>>,
}

impl ZeroCopyMatrix1D {
    /// Wrap an existing `Vec<f64>` in shared ownership.
    pub fn from_vec(v: Vec<f64>) -> Self {
        Self { data: Arc::new(v) }
    }

    /// Borrow the underlying slice (zero-copy).
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }
}

/// A 2-D matrix that can cross the Rust ↔ Python boundary without copying.
///
/// Same storage strategy as [`ZeroCopyMatrix1D`]. The shape `(rows, cols)` is
/// stored alongside the buffer; the buffer length must equal `rows * cols`.
#[derive(Debug, Clone)]
pub struct ZeroCopyMatrix2D {
    data: Arc<Vec<f64>>,
    shape: (usize, usize),
}

impl ZeroCopyMatrix2D {
    /// Wrap an existing flat buffer + shape in shared ownership.
    pub fn from_vec(v: Vec<f64>, shape: (usize, usize)) -> Self {
        assert_eq!(
            v.len(),
            shape.0 * shape.1,
            "ZeroCopyMatrix2D buffer length {} does not match shape {:?}",
            v.len(),
            shape,
        );
        Self {
            data: Arc::new(v),
            shape,
        }
    }

    /// Borrow the underlying slice (zero-copy).
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.shape.1
    }

    /// The shape tuple.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

// =============================================================================
// Python-binding adapters (gated by `python-bindings` feature)
// =============================================================================

#[cfg(feature = "python-bindings")]
mod python_impl {
    use super::{Arc, ZeroCopyMatrix1D, ZeroCopyMatrix2D};
    use numpy::{PyArray1, PyArray2, PyArrayMethods};
    use pyo3::{Bound, PyResult, Python};

    impl ZeroCopyMatrix1D {
        /// Build a numpy array that shares the underlying buffer (zero-copy on
        /// the Rust → Python direction). The numpy array's container holds an
        /// `Arc` clone, so the data stays alive as long as Python holds the
        /// numpy array.
        pub fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            // SAFETY: we hand `borrow_from_array` a container that
            // holds an Arc clone of the underlying Vec, and a view that
            // points into that Vec's storage. The Arc is kept alive by the
            // numpy array's container, so the view's data remains valid for
            // the lifetime of the returned `Bound<'py, PyArray1<f64>>`.
            let arc_for_view = Arc::clone(&self.data);
            let view = ndarray::ArrayView1::from(&*arc_for_view);
            // Clone again for the holder so `view` (which borrows from
            // `arc_for_view`) stays valid for the
            // `borrow_from_array` call.
            let arc_for_holder = Arc::clone(&self.data);
            let holder = ZeroCopyHolder1D {
                data: arc_for_holder,
            };
            let container = Bound::new(py, holder)
                .expect("ZeroCopyHolder1D allocation cannot fail")
                .into_any();
            unsafe { PyArray1::borrow_from_array(&view, container) }
        }
    }

    impl ZeroCopyMatrix2D {
        /// Build a numpy array that shares the underlying buffer (zero-copy
        /// on the Rust → Python direction).
        pub fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            // SAFETY: the container holds an Arc clone of the underlying
            // Vec, and the view points into that Vec's storage. The Arc
            // outlives the returned numpy array.
            let arc_for_view = Arc::clone(&self.data);
            // SAFETY: `arc_for_view.as_ptr()` is a valid, aligned, non-null
            // pointer to `self.shape.0 * self.shape.1` contiguous `f64`
            // values. The shape matches the buffer length, so the view is
            // well-formed.
            let raw =
                unsafe { ndarray::RawArrayView::from_shape_ptr(self.shape, arc_for_view.as_ptr()) };
            // SAFETY: same invariants as above; this converts the raw view
            // into a borrow-checked `ArrayView`. The Arc keeps the backing
            // storage alive.
            let view = unsafe { raw.deref_into_view() };
            // Clone again for the holder so `view` (which borrows from
            // `arc_for_view`) stays valid for the
            // `borrow_from_array` call.
            let arc_for_holder = Arc::clone(&self.data);
            let holder = ZeroCopyHolder2D {
                data: arc_for_holder,
                shape: self.shape,
            };
            let container = Bound::new(py, holder)
                .expect("ZeroCopyHolder2D allocation cannot fail")
                .into_any();
            unsafe { PyArray2::borrow_from_array(&view, container) }
        }
    }

    /// Holder passed to `borrow_from_array` as the container. Holding
    /// the `Arc` in a `#[pyclass]` lets the numpy array's base object keep
    /// the data alive through Python's GC.
    #[pyo3::pyclass]
    struct ZeroCopyHolder1D {
        #[allow(dead_code)]
        data: Arc<Vec<f64>>,
    }

    #[pyo3::pyclass]
    struct ZeroCopyHolder2D {
        #[allow(dead_code)]
        data: Arc<Vec<f64>>,
        #[allow(dead_code)]
        shape: (usize, usize),
    }

    /// Holder for `PyGeometryTensor::to_numpy`. Holds an `Arc<GeometryTensor>`
    /// clone so that the numpy array's container keeps the geometry's
    /// storage alive. Used by the `to_numpy` pymethod to wrap individual
    /// matrix fields of the shared `GeometryTensor` without copying.
    #[pyo3::pyclass]
    pub struct ZeroCopyGeometryTensorHolder {
        #[allow(dead_code)]
        pub inner: Arc<crate::physics::geometry_tensor::GeometryTensor>,
    }

    /// Zero-copy extraction of a `&[f64]` from a 1-D numpy array.
    pub fn extract_1d_slice<'py>(array: &Bound<'py, PyArray1<f64>>) -> PyResult<&'py [f64]> {
        let readonly = array.readonly();
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("non-contiguous array: {e}"))
        })?;
        // SAFETY: extend the lifetime of the slice to `'py`. Sound as long
        // as the caller does not mutate the numpy array while the returned
        // slice is in use; `PyReadonlyArray` is alive on the stack and
        // global borrow tracking will reject any conflicting borrow.
        Ok(unsafe { std::mem::transmute::<&[f64], &'py [f64]>(slice) })
    }

    /// Zero-copy extraction of a `&[f64]` from a 2-D numpy array in row-major
    /// (C-order) layout.
    pub fn extract_2d_slice<'py>(array: &Bound<'py, PyArray2<f64>>) -> PyResult<&'py [f64]> {
        let readonly = array.readonly();
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("non-contiguous array: {e}"))
        })?;
        Ok(unsafe { std::mem::transmute::<&[f64], &'py [f64]>(slice) })
    }

    /// Build a [`ZeroCopyMatrix2D`] from a 2-D numpy array without copying
    /// the buffer on the binding layer.
    pub fn from_numpy_2d_zero_copy<'py>(
        array: &Bound<'py, PyArray2<f64>>,
        expected_shape: (usize, usize),
    ) -> PyResult<ZeroCopyMatrix2D> {
        let readonly = array.readonly();
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("non-contiguous array: {e}"))
        })?;
        let shape = readonly.as_array().raw_dim();
        let actual_shape = (shape[0], shape[1]);
        if actual_shape != expected_shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected shape {:?}, got {:?}",
                expected_shape, actual_shape
            )));
        }
        Ok(ZeroCopyMatrix2D::from_vec(slice.to_vec(), expected_shape))
    }

    /// Build a [`ZeroCopyMatrix1D`] from a 1-D numpy array. Same ownership
    /// semantics as [`from_numpy_2d_zero_copy`].
    pub fn from_numpy_1d_zero_copy<'py>(
        array: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<ZeroCopyMatrix1D> {
        let readonly = array.readonly();
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("non-contiguous array: {e}"))
        })?;
        Ok(ZeroCopyMatrix1D::from_vec(slice.to_vec()))
    }
}

#[cfg(feature = "python-bindings")]
pub use python_impl::{
    extract_1d_slice, extract_2d_slice, from_numpy_1d_zero_copy, from_numpy_2d_zero_copy,
    ZeroCopyGeometryTensorHolder,
};

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    use super::ZeroCopyMatrix2D;

    #[test]
    fn holder_keeps_data_alive() {
        let m = ZeroCopyMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let arc = std::sync::Arc::clone(&m.data);
        let ptr1 = arc.as_ptr();
        let arc2 = std::sync::Arc::clone(&m.data);
        let ptr2 = arc2.as_ptr();
        assert_eq!(ptr1, ptr2);
        // Three strong references to the same allocation: `m.data`, `arc`, `arc2`.
        // (`Arc::len` does not exist; `arc.len()` would deref to `Vec::len` → 4.)
        assert_eq!(std::sync::Arc::strong_count(&arc), 3);
    }

    #[test]
    fn from_vec_2d_validates_shape() {
        let result =
            std::panic::catch_unwind(|| ZeroCopyMatrix2D::from_vec(vec![1.0, 2.0, 3.0], (2, 2)));
        assert!(result.is_err(), "mismatched shape must panic");
    }

    #[test]
    fn slice_matches_input() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = ZeroCopyMatrix2D::from_vec(v.clone(), (2, 3));
        assert_eq!(m.as_slice(), v.as_slice());
    }

    #[test]
    fn clone_shares_arc() {
        let v = vec![1.0; 100];
        let m1 = ZeroCopyMatrix2D::from_vec(v.clone(), (10, 10));
        let m2 = m1.clone();
        assert_eq!(m1.data.as_ptr(), m2.data.as_ptr());
        assert_eq!(std::sync::Arc::strong_count(&m1.data), 2);
    }

    /// Round-trip test that exercises the full numpy path: zero-copy Arc +
    /// `borrow_from_array`. Requires the `python-bindings` feature.
    #[test]
    fn to_numpy_round_trip() {
        use numpy::{PyArrayMethods, PyUntypedArrayMethods};
        pyo3::Python::attach(|py| {
            let v = vec![1.0_f64, 2.0, 3.0, 4.0];
            let m = ZeroCopyMatrix2D::from_vec(v.clone(), (2, 2));
            let pyarr = m.to_numpy(py);
            assert_eq!(pyarr.shape(), &[2, 2]);
            let readonly = pyarr.readonly();
            let view = readonly.as_array();
            assert_eq!(view[[0, 0]], 1.0);
            assert_eq!(view[[1, 1]], 4.0);
        });
    }
}
