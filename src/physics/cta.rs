use std::ops::{Add, AddAssign, Div, Index, Mul, Sub};

#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArrayMethods};
#[cfg(feature = "python-bindings")]
use pyo3::{prelude::*, PyResult};

/// The Continuous Tensor Abstraction (CTA) trait.
///
/// This trait defines the interface for tensor operations in Fluxion, abstracting
/// over underlying storage (CPU Vec, GPU Buffer, etc.) and dimensionality.
///
/// It is designed to be generic over the element type `T` and to support
/// common operations in physics simulations like element-wise maps, reductions,
/// and differential operators.
pub trait ContinuousTensor<T>:
    // Basic arithmetic operations
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    // Required for many internal operations
    + Sized
    + Clone
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + AddAssign + Default,
{
    /// Applies a function element-wise to the tensor.
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T;

    /// Combines two tensors element-wise using a binary function.
    fn zip_with<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(T, T) -> T;

    /// Reduces the tensor to a single value using a binary function.
    fn reduce<F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, T) -> T;

    /// Computes the integral of the tensor field.
    /// The exact meaning of "integral" depends on the tensor's dimensionality.
    /// For a 1D VectorField, this is equivalent to a sum.
    fn integrate(&self) -> T;

    /// Computes the gradient of the tensor field.
    /// The result is a new tensor representing the rate of change.
    /// The exact implementation will vary (e.g., finite differences for grids).
    fn gradient(&self) -> Self;

    /// Creates a new tensor of the same shape and size, filled with a constant value.
    fn constant_like(&self, value: T) -> Self;

    /// Computes the element-wise minimum of two tensors.
    fn elementwise_min(&self, other: &Self) -> Self;

    /// Computes the element-wise maximum of two tensors.
    fn elementwise_max(&self, other: &Self) -> Self;
}

/// A basic CPU-based implementation of ContinuousTensor using Vec<f64>.
/// Represents a 1D Tensor (Vector).
#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
#[derive(Debug, Clone, PartialEq)]
pub struct VectorField {
    data: Vec<f64>,
}

impl VectorField {
    /// Create a new VectorField from a vector of data.
    pub fn new(data: Vec<f64>) -> Self {
        VectorField { data }
    }

    /// Create a new VectorField with all elements initialized to a scalar value.
    pub fn from_scalar(value: f64, size: usize) -> Self {
        VectorField {
            data: vec![value; size],
        }
    }

    /// Get the number of elements in the field.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the field has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the underlying data slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Get a mutable reference to the underlying data slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Return an iterator over the field elements.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }
}

impl Index<usize> for VectorField {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl Add for VectorField {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        // Optimization: reuse self buffer
        assert_eq!(self.len(), rhs.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
        self
    }
}

impl Sub for VectorField {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        // Optimization: reuse self buffer
        assert_eq!(self.len(), rhs.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b;
        }
        self
    }
}

impl Mul for VectorField {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
        // Optimization: reuse self buffer
        assert_eq!(self.len(), rhs.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a *= b;
        }
        self
    }
}

impl Div for VectorField {
    type Output = Self;
    fn div(mut self, rhs: Self) -> Self {
        // Optimization: reuse self buffer
        assert_eq!(self.len(), rhs.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a /= b;
        }
        self
    }
}

impl ContinuousTensor<f64> for VectorField {
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        VectorField {
            data: self.data.iter().copied().map(f).collect(),
        }
    }

    fn zip_with<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        VectorField {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    fn reduce<F>(&self, init: f64, f: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        self.data.iter().copied().fold(init, f)
    }

    fn integrate(&self) -> f64 {
        // For a 1D discrete field with unit spacing, the integral is the sum of elements.
        self.data.iter().sum()
    }

    fn gradient(&self) -> Self {
        // Central differences for interior points, forward/backward for boundaries
        let n = self.data.len();
        if n == 0 {
            return VectorField::new(vec![]);
        }
        if n == 1 {
            return VectorField::from_scalar(0.0, 1);
        }

        let mut grad_data = vec![0.0; n];
        // Forward difference for first element
        grad_data[0] = self.data[1] - self.data[0];
        // Central differences for interior
        for (grad, window) in grad_data[1..n - 1].iter_mut().zip(self.data.windows(3)) {
            *grad = 0.5 * (window[2] - window[0]);
        }
        // Backward difference for last element
        grad_data[n - 1] = self.data[n - 1] - self.data[n - 2];
        VectorField::new(grad_data)
    }

    fn constant_like(&self, value: f64) -> Self {
        VectorField::from_scalar(value, self.len())
    }

    fn elementwise_min(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a.min(b))
    }

    fn elementwise_max(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a.max(b))
    }
}

// Convenience implementations for Scalar <-> Tensor operations
impl Mul<f64> for VectorField {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        // Optimization: reuse self buffer
        for x in &mut self.data {
            *x *= rhs;
        }
        self
    }
}

impl Div<f64> for VectorField {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self {
        // Optimization: reuse self buffer
        for x in &mut self.data {
            *x /= rhs;
        }
        self
    }
}

impl AsRef<[f64]> for VectorField {
    fn as_ref(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(feature = "python-bindings")]
impl VectorField {
    /// Create a VectorField from a numpy array.
    ///
    /// This method converts a numpy array to a VectorField by copying the data.
    pub fn from_numpy_array<'py>(
        _py: Python<'py>,
        array: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<Self> {
        let numpy_array = array.downcast::<PyArray1<f64>>()?;
        let slice = unsafe { numpy_array.as_slice()? };
        Ok(VectorField::new(slice.to_vec()))
    }

    /// Convert this VectorField to a numpy array.
    ///
    /// Returns a PyArray1 that borrows the data from this VectorField.
    /// The Python array will be valid only as long as the VectorField exists.
    pub fn to_numpy_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.data)
    }

    /// Validate that a Python object is a valid numpy array for conversion.
    ///
    /// Checks that the object is a numpy array with dtype float64 and 1-dimensional.
    pub fn validate_numpy_array<'py>(obj: &Bound<'py, pyo3::PyAny>) -> PyResult<()> {
        let _numpy_array = obj.downcast::<PyArray1<f64>>()?;
        Ok(())
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl VectorField {
    #[new]
    fn new_py(data: Vec<f64>) -> Self {
        VectorField::new(data)
    }

    fn integrate(&self) -> f64 {
        ContinuousTensor::integrate(self)
    }

    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyArray1::from_slice_bound(py, &self.data).into_py(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorField(len={}, data=[{}])",
            self.data.len(),
            self.data
                .iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn __str__(&self) -> String {
        format!(
            "[{}]",
            self.data
                .iter()
                .map(|x| format!("{:.2}", x))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_field_ops() {
        let v1 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let v2 = VectorField::new(vec![4.0, 5.0, 6.0]);

        let sum = v1.clone() + v2.clone();
        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);

        let prod = v1.clone() * v2.clone();
        assert_eq!(prod.data, vec![4.0, 10.0, 18.0]);

        let scaled = v1 * 2.0;
        assert_eq!(scaled.data, vec![2.0, 4.0, 6.0]);
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_vector_field_to_numpy() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let vf = VectorField::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let numpy_array = vf.to_numpy_array(py);

            assert_eq!(numpy_array.len().unwrap(), 5);
            let slice = unsafe { numpy_array.as_slice().unwrap() };
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        });
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_vector_field_from_numpy() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let numpy_array = PyArray1::from_slice_bound(py, &data);
            let vf = VectorField::from_numpy_array(py, &numpy_array).unwrap();

            assert_eq!(vf.len(), 5);
            assert_eq!(vf.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        });
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_vector_field_roundtrip() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let original = VectorField::new(vec![1.5, 2.5, 3.5, 4.5, 5.5]);

            let numpy_array = original.to_numpy_array(py);
            let recovered = VectorField::from_numpy_array(py, &numpy_array).unwrap();

            assert_eq!(original.len(), recovered.len());
            assert_eq!(original.as_slice(), recovered.as_slice());
        });
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_numpy_validation() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let data = vec![1.0, 2.0, 3.0];
            let array = PyArray1::from_slice_bound(py, &data);
            assert!(
                VectorField::validate_numpy_array(&array).is_ok(),
                "Valid array should pass validation"
            );

            let empty_data: Vec<f64> = vec![];
            let empty_array = PyArray1::from_slice_bound(py, &empty_data);
            assert!(
                VectorField::validate_numpy_array(&empty_array).is_ok(),
                "Empty array should be valid"
            );
        });
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_empty_vector_field_conversion() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let vf = VectorField::new(vec![]);
            let numpy_array = vf.to_numpy_array(py);
            assert_eq!(numpy_array.len().unwrap(), 0);

            let recovered = VectorField::from_numpy_array(py, &numpy_array).unwrap();
            assert!(recovered.is_empty());
        });
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_large_vector_field_conversion() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let large_data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
            let vf = VectorField::new(large_data.clone());
            let numpy_array = vf.to_numpy_array(py);
            assert_eq!(numpy_array.len().unwrap(), 10000);

            let recovered = VectorField::from_numpy_array(py, &numpy_array).unwrap();
            assert_eq!(vf.as_slice(), recovered.as_slice());
        });
    }

    #[test]
    fn test_is_empty() {
        let v_empty = VectorField::new(vec![]);
        assert!(v_empty.is_empty());
        assert_eq!(v_empty.len(), 0);

        let v_non_empty = VectorField::new(vec![1.0]);
        assert!(!v_non_empty.is_empty());
        assert_eq!(v_non_empty.len(), 1);
    }

    #[test]
    fn test_integrate() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.integrate(), 10.0);
    }

    #[test]
    fn test_gradient() {
        let v = VectorField::new(vec![1.0, 2.0, 4.0, 7.0]);
        let grad = v.gradient();
        assert_eq!(grad.as_slice(), &[1.0, 1.5, 2.5, 3.0]);
    }

    #[test]
    fn test_in_place_arithmetic() {
        // Verify that operations reuse memory (check pointer equality would be hard in safe Rust,
        // but we can verify results are correct and "mut" is working as expected)

        // Add
        let mut v1 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let v2 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let ptr_before = v1.as_slice().as_ptr();
        v1 = v1 + v2;
        let ptr_after = v1.as_slice().as_ptr();
        assert_eq!(v1.as_slice(), &[11.0, 22.0, 33.0]);
        assert_eq!(ptr_before, ptr_after, "Add should reuse allocation of LHS");

        // Sub
        let mut v3 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let v4 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let ptr_before = v3.as_slice().as_ptr();
        v3 = v3 - v4;
        let ptr_after = v3.as_slice().as_ptr();
        assert_eq!(v3.as_slice(), &[9.0, 18.0, 27.0]);
        assert_eq!(ptr_before, ptr_after, "Sub should reuse allocation of LHS");

        // Mul scalar
        let mut v5 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let ptr_before = v5.as_slice().as_ptr();
        v5 = v5 * 2.0;
        let ptr_after = v5.as_slice().as_ptr();
        assert_eq!(v5.as_slice(), &[2.0, 4.0, 6.0]);
        assert_eq!(ptr_before, ptr_after, "Mul<f64> should reuse allocation");
    }
}
