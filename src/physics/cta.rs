use std::convert::AsMut;
use std::ops::{Add, AddAssign, Div, Index, Mul, Sub};

#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArrayMethods};

#[cfg(feature = "python-bindings")]
use pyo3::{pymethods, Bound, IntoPy, PyAny, PyObject, PyResult, Python};

#[cfg(feature = "python-bindings")]
use pyo3::types::PyAnyMethods;

/// Trait for continuous tensor operations.
///
/// Defines common operations for continuous field representations.
/// Implemented by VectorField for CPU-based operations.
/// Can be extended for GPU-based implementations (e.g., CUDA tensors).
///
/// # Required Methods
/// - `new()`: Create from data
/// - `map()`: Apply function element-wise
/// - `zip_with()`: Combine two tensors element-wise
/// - `reduce()`: Reduce tensor to single value
/// - `integrate()`: Compute spatial integral
/// - `gradient()`: Compute spatial gradient
/// - `constant_like()`: Create tensor of same shape filled with constant
/// - In-place operations: `add_assign`, `sub_assign`, `mul_assign`, `div_assign`
///
/// # Example
/// ```rust
/// use fluxion::physics::cta::{VectorField, ContinuousTensor};
///
/// let field: VectorField = VectorField::new(vec![1.0, 2.0, 3.0]);
/// let grad = field.gradient();
/// let integral = field.integrate();
///
/// // Element-wise operations
/// let doubled = field.map(|x| x * 2.0);
/// let min = field.elementwise_min(&doubled);
/// ```
///
/// # Performance
/// Operations are designed to be vectorizable and thread-safe.
/// Implementations should reuse buffers where possible to minimize allocations.
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

    /// Adds another tensor to this one in-place (element-wise).
    fn add_assign(&mut self, other: &Self);

    /// Subtracts another tensor from this one in-place (element-wise).
    fn sub_assign(&mut self, other: &Self);

    /// Multiplies this tensor by another in-place (element-wise).
    fn mul_assign(&mut self, other: &Self);

    /// Divides this tensor by another in-place (element-wise).
    fn div_assign(&mut self, other: &Self);
}

/// Continuous scalar field representation using CTA operations.
///
/// Provides unified API for tensor-like operations used by the physics engine.
/// Abstracts vector operations to enable future GPU acceleration.
/// Implements the ContinuousTensor trait for element-wise operations.
///
/// # Architecture
/// - Element-wise operations (+, -, *, /)
/// - Gradient and integration methods for spatial derivatives
/// - Supports 1D vectors (time series, spatial arrays)
/// - Zero-copy conversion to/from NumPy arrays (Python bindings)
///
/// # Usage
/// ```rust
/// use fluxion::physics::cta::VectorField;
///
/// let v = VectorField::new(vec![1.0, 2.0, 3.0]);
/// let g = v.gradient();
/// let integral = v.integrate();
///
/// // Element-wise arithmetic
/// let sum = v.clone() + VectorField::new(vec![4.0, 5.0, 6.0]);
/// let scaled = v * 2.0;
/// ```
///
/// # Performance
/// - Operations are element-wise for vectorization
/// - Future: GPU acceleration via backend abstraction
/// - Current: CPU-based with SIMD optimization
/// - Buffer reuse in arithmetic operations to minimize allocations
///
/// # Thread Safety
/// VectorField is Clone and Send, enabling parallel evaluation.
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

    /// Apply a function element-wise in-place, modifying the field.
    ///
    /// This is an internal optimization helper for future allocation reduction.
    /// For cases where a `map` operation would create a new vector, this method
    /// can be used to reuse the existing allocation.
    ///
    /// # Example
    /// ```rust
    /// let mut v = VectorField::new(vec![1.0, 2.0, 3.0]);
    /// v.map_in_place(|x| x * 2.0);
    /// assert_eq!(v.as_slice(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn map_in_place<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for x in &mut self.data {
            *x = f(*x);
        }
    }
}

impl AsMut<[f64]> for VectorField {
    fn as_mut(&mut self) -> &mut [f64] {
        &mut self.data
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
        // Optimized: manual loop avoids slice allocations from .windows(3), improving cache locality
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
        // Central differences for interior points - manual index access eliminates slice overhead
        for i in 1..n - 1 {
            grad_data[i] = 0.5 * (self.data[i + 1] - self.data[i - 1]);
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

    fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    fn sub_assign(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= *b;
        }
    }

    fn mul_assign(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a *= *b;
        }
    }

    fn div_assign(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a /= *b;
        }
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
#[allow(unexpected_cfgs)]
impl VectorField {
    /// Convert to a numpy array (zero-copy borrow).
    pub fn to_numpy_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.data)
    }

    /// Create from a numpy array.
    #[allow(unused_unsafe)]
    pub fn from_numpy_array<'py>(_py: Python<'py>, array: &Bound<'py, PyAny>) -> PyResult<Self> {
        let numpy_array = array.downcast::<PyArray1<f64>>()?;
        let slice = unsafe { numpy_array.as_slice()? };
        Ok(VectorField::new(slice.to_vec()))
    }
}

#[cfg(feature = "python-bindings")]
#[allow(unexpected_cfgs)]
#[pymethods]
impl VectorField {
    #[new]
    fn new_py(data: Vec<f64>) -> Self {
        VectorField::new(data)
    }

    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.to_numpy_array(py).into_py(py))
    }

    fn integrate(&self) -> f64 {
        ContinuousTensor::integrate(self)
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorField(len={}, data=[{}])",
            self.data.len(),
            self.data
                .iter()
                .take(5)
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

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_vector_field_numpy_conversion() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let original = VectorField::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

            let numpy_array = original.to_numpy_array(py);
            let recovered = VectorField::from_numpy_array(py, &numpy_array).unwrap();

            assert_eq!(original.len(), recovered.len());
            assert_eq!(original.as_slice(), recovered.as_slice());
        });
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_empty_vector_field_numpy() {
        pyo3::prepare_freethreaded_python();

        pyo3::Python::with_gil(|py| {
            let empty = VectorField::new(vec![]);
            let numpy_array = empty.to_numpy_array(py);
            let recovered = VectorField::from_numpy_array(py, &numpy_array).unwrap();

            assert!(recovered.is_empty());
        });
    }

    #[test]
    fn test_gradient_from_cta() {
        // Helper function implementing the old windows(3)-based gradient
        // Used as reference to ensure optimized implementation produces identical results
        fn gradient_old(data: &[f64]) -> Vec<f64> {
            let n = data.len();
            if n == 0 {
                return vec![];
            }
            if n == 1 {
                return vec![0.0];
            }

            let mut grad_data = vec![0.0; n];
            // Forward difference for first element
            grad_data[0] = data[1] - data[0];
            // Central differences for interior points using windows(3)
            for (grad, window) in grad_data[1..n - 1].iter_mut().zip(data.windows(3)) {
                *grad = 0.5 * (window[2] - window[0]);
            }
            // Backward difference for last element
            grad_data[n - 1] = data[n - 1] - data[n - 2];
            grad_data
        }

        let test_cases = vec![
            vec![1.0, 2.0, 4.0, 7.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![5.0],
            vec![],
            vec![1.5, 2.5, 3.5, 4.5, 5.5],
            vec![10.0, -5.0, 0.0, 7.5],
        ];

        for data in test_cases {
            let v = VectorField::new(data.clone());
            let grad = v.gradient();
            let expected = gradient_old(&data);
            assert_eq!(
                grad.as_slice(),
                expected.as_slice(),
                "Gradient mismatch for input {:?}",
                data
            );
        }
    }

    #[test]
    fn test_subtraction() {
        let v1 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let v2 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let diff = v1 - v2;
        assert_eq!(diff.data, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_division() {
        let v1 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let v2 = VectorField::new(vec![2.0, 4.0, 6.0]);
        let quotient = v1 / v2;
        assert_eq!(quotient.data, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_scalar_division() {
        let v = VectorField::new(vec![10.0, 20.0, 30.0]);
        let divided = v / 2.0;
        assert_eq!(divided.data, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_map() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0]);
        let squared = v.map(|x| x * x);
        assert_eq!(squared.data, vec![1.0, 4.0, 9.0]);

        let negative = v.map(|x| -x);
        assert_eq!(negative.data, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_zip_with() {
        let v1 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let v2 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let combined = v1.zip_with(&v2, |a, b| a + b * 2.0);
        assert_eq!(combined.data, vec![21.0, 42.0, 63.0]);
    }

    #[test]
    fn test_reduce_sum() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0, 4.0]);
        let sum = v.reduce(0.0, |acc, x| acc + x);
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_reduce_max() {
        let v = VectorField::new(vec![1.0, 5.0, 3.0, 9.0, 2.0]);
        let max_val = v.reduce(f64::NEG_INFINITY, |acc, x| acc.max(x));
        assert_eq!(max_val, 9.0);
    }

    #[test]
    fn test_reduce_product() {
        let v = VectorField::new(vec![2.0, 3.0, 4.0]);
        let product = v.reduce(1.0, |acc, x| acc * x);
        assert_eq!(product, 24.0);
    }

    #[test]
    fn test_constant_like() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0, 4.0]);
        let constant = v.constant_like(7.5);
        assert_eq!(constant.data, vec![7.5; 4]);
        assert_eq!(constant.len(), 4);
    }

    #[test]
    fn test_elementwise_min() {
        let v1 = VectorField::new(vec![5.0, 2.0, 8.0, 1.0]);
        let v2 = VectorField::new(vec![3.0, 7.0, 4.0, 9.0]);
        let result = v1.elementwise_min(&v2);
        assert_eq!(result.data, vec![3.0, 2.0, 4.0, 1.0]);
    }

    #[test]
    fn test_elementwise_max() {
        let v1 = VectorField::new(vec![5.0, 2.0, 8.0, 1.0]);
        let v2 = VectorField::new(vec![3.0, 7.0, 4.0, 9.0]);
        let result = v1.elementwise_max(&v2);
        assert_eq!(result.data, vec![5.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_div_assign() {
        let mut v1 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let v2 = VectorField::new(vec![2.0, 4.0, 6.0]);
        v1.div_assign(&v2);
        assert_eq!(v1.data, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_index() {
        let v = VectorField::new(vec![10.0, 20.0, 30.0]);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 20.0);
        assert_eq!(v[2], 30.0);
    }

    #[test]
    fn test_clone() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0]);
        let cloned = v.clone();
        assert_eq!(cloned.data, v.data);
    }

    #[test]
    fn test_debug() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0]);
        let debug_str = format!("{:?}", v);
        assert!(debug_str.contains("VectorField"));
    }

    #[test]
    fn test_as_slice() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_large_vector_operations() {
        let large: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let v = VectorField::new(large);

        let doubled = v.clone() * 2.0;
        for i in 0..1000 {
            assert_eq!(doubled[i], i as f64 * 2.0);
        }

        let summed = v + doubled.clone();
        for i in 0..1000 {
            assert_eq!(summed[i], i as f64 * 3.0);
        }
    }

    #[test]
    fn test_gradient_constant() {
        let v = VectorField::new(vec![5.0; 5]);
        let grad = v.gradient();
        assert_eq!(grad.data, vec![0.0; 5]);
    }

    #[test]
    fn test_gradient_linear_increasing() {
        let v = VectorField::new(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let grad = v.gradient();
        assert_eq!(grad.data, vec![1.0; 5]);
    }

    #[test]
    fn test_gradient_single_element() {
        let v = VectorField::new(vec![5.0]);
        let grad = v.gradient();
        assert_eq!(grad.data, vec![0.0]);
    }

    #[test]
    fn test_gradient_empty() {
        let v = VectorField::new(vec![]);
        let grad = v.gradient();
        assert!(grad.is_empty());
    }

    #[test]
    fn test_integrate_constant() {
        let v = VectorField::new(vec![3.0; 5]);
        let integral = v.integrate();
        assert_eq!(integral, 15.0); // 3.0 * 5 = sum
    }

    #[test]
    fn test_integrate_linear() {
        let v = VectorField::new(vec![0.0, 1.0, 2.0, 3.0]);
        let integral = v.integrate();
        // Integration is just sum for 1D
        assert_eq!(integral, 6.0); // 0 + 1 + 2 + 3
    }

    #[test]
    fn test_integrate_single() {
        let v = VectorField::new(vec![5.0]);
        let integral = v.integrate();
        assert_eq!(integral, 5.0);
    }

    #[test]
    fn test_integrate_empty() {
        let v = VectorField::new(vec![]);
        let integral = v.integrate();
        assert_eq!(integral, 0.0);
    }

    #[test]
    fn test_in_place_operations_chaining() {
        let mut v1 = VectorField::new(vec![10.0, 20.0, 30.0]);
        let v2 = VectorField::new(vec![1.0, 2.0, 3.0]);
        let v3 = VectorField::new(vec![100.0, 200.0, 300.0]);

        v1.add_assign(&v2);
        v1.mul_assign(&v3);
        v1.sub_assign(&v2);

        // (10 + 1) * 100 - 1 = 1099
        // (20 + 2) * 200 - 2 = 4398
        // (30 + 3) * 300 - 3 = 9897
        assert_eq!(v1[0], 1099.0);
        assert_eq!(v1[1], 4398.0);
        assert_eq!(v1[2], 9897.0);
    }

    #[test]
    fn test_operations_with_negative_values() {
        let v1 = VectorField::new(vec![-5.0, -10.0, -15.0]);
        let v2 = VectorField::new(vec![2.0, 5.0, 3.0]);

        let sum = v1.clone() + v2.clone();
        assert_eq!(sum.data, vec![-3.0, -5.0, -12.0]);

        let prod = v1.clone() * v2.clone();
        assert_eq!(prod.data, vec![-10.0, -50.0, -45.0]);

        let scaled = v2 * -2.0;
        assert_eq!(scaled.data, vec![-4.0, -10.0, -6.0]);
    }

    #[test]
    fn test_map_complex_function() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = v.map(|x| x * x + x / 2.0);
        // 1*1 + 1/2 = 1.5
        // 2*2 + 2/2 = 5.0
        // 3*3 + 3/2 = 10.5
        // 4*4 + 4/2 = 18.0
        assert_eq!(result.data, vec![1.5, 5.0, 10.5, 18.0]);
    }

    #[test]
    fn test_reduce_with_zero_init() {
        let v = VectorField::new(vec![5.0, 10.0, 15.0]);
        let sum = v.reduce(0.0, |acc, x| acc + x);
        assert_eq!(sum, 30.0);
    }
}
