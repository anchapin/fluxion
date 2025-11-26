use std::ops::{Add, Div, Index, Mul, Sub};

/// The Continuous Tensor Abstraction (CTA) trait.
///
/// This trait defines the interface for tensor operations in Fluxion, abstracting
/// over underlying storage (CPU Vec, GPU Buffer, etc.) and dimensionality.
pub trait ContinuousTensor<T>:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Sized + Clone
{
    /// Apply a function element-wise.
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T;

    /// Combine two tensors element-wise with a function.
    fn zip_with<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(T, T) -> T;

    /// Reduce the tensor to a single value.
    fn reduce<F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, T) -> T;

    /// Create a new tensor of the same shape with a constant value.
    fn constant_like(&self, value: T) -> Self;
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

    fn constant_like(&self, value: f64) -> Self {
        VectorField::from_scalar(value, self.len())
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
}
