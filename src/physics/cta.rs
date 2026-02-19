use std::ops::{Add, AddAssign, Div, Index, Mul, Sub};

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
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
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
    fn add(self, rhs: Self) -> Self {
        self.zip_with(&rhs, |a, b| a + b)
    }
}

impl Sub for VectorField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.zip_with(&rhs, |a, b| a - b)
    }
}

impl Mul for VectorField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.zip_with(&rhs, |a, b| a * b)
    }
}

impl Div for VectorField {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.zip_with(&rhs, |a, b| a / b)
    }
}

impl<'a> Add<&'a VectorField> for VectorField {
    type Output = VectorField;
    fn add(self, rhs: &'a VectorField) -> VectorField {
        self.zip_with(rhs, |a, b| a + b)
    }
}

impl<'a> Sub<&'a VectorField> for VectorField {
    type Output = VectorField;
    fn sub(self, rhs: &'a VectorField) -> VectorField {
        self.zip_with(rhs, |a, b| a - b)
    }
}

impl<'a> Mul<&'a VectorField> for VectorField {
    type Output = VectorField;
    fn mul(self, rhs: &'a VectorField) -> VectorField {
        self.zip_with(rhs, |a, b| a * b)
    }
}

impl<'a> Div<&'a VectorField> for VectorField {
    type Output = VectorField;
    fn div(self, rhs: &'a VectorField) -> VectorField {
        self.zip_with(rhs, |a, b| a / b)
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
    fn mul(self, rhs: f64) -> Self {
        self.map(|x| x * rhs)
    }
}

impl Div<f64> for VectorField {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self.map(|x| x / rhs)
    }
}

impl AsRef<[f64]> for VectorField {
    fn as_ref(&self) -> &[f64] {
        &self.data
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
}
