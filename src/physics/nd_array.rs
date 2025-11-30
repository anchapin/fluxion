use crate::physics::cta::ContinuousTensor;
use ndarray::{ArrayD, IxDyn};
use std::ops::{Add, Div, Index, Mul, Sub};

/// N-dimensional tensor backed by ndarray for CTA.
#[derive(Debug, Clone, PartialEq)]
pub struct NDArrayField {
    arr: ArrayD<f64>,
}

impl NDArrayField {
    /// Create an NDArrayField from a shape and flat data vector.
    pub fn from_shape_vec(shape: Vec<usize>, data: Vec<f64>) -> Self {
        let arr =
            ArrayD::from_shape_vec(IxDyn(&shape), data).expect("Shape and data length mismatch");
        NDArrayField { arr }
    }

    /// Return the shape of the underlying ndarray as a Vec<usize>.
    pub fn shape(&self) -> Vec<usize> {
        self.arr.shape().to_vec()
    }

    /// Number of elements in the flattened array.
    pub fn len(&self) -> usize {
        self.arr.len()
    }

    /// Get a contiguous slice of the flattened data.
    pub fn as_slice(&self) -> &[f64] {
        self.arr.as_slice().expect("Array not contiguous")
    }

    /// True if array has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Index<usize> for NDArrayField {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.as_slice()[idx]
    }
}

impl Add for NDArrayField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let shape = self.shape();
        let v: Vec<f64> = self
            .as_slice()
            .iter()
            .zip(rhs.as_slice().iter())
            .map(|(a, b)| a + b)
            .collect();
        NDArrayField::from_shape_vec(shape, v)
    }
}
impl Sub for NDArrayField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let shape = self.shape();
        let v: Vec<f64> = self
            .as_slice()
            .iter()
            .zip(rhs.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        NDArrayField::from_shape_vec(shape, v)
    }
}
impl Mul for NDArrayField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let shape = self.shape();
        let v: Vec<f64> = self
            .as_slice()
            .iter()
            .zip(rhs.as_slice().iter())
            .map(|(a, b)| a * b)
            .collect();
        NDArrayField::from_shape_vec(shape, v)
    }
}
impl Div for NDArrayField {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let shape = self.shape();
        let v: Vec<f64> = self
            .as_slice()
            .iter()
            .zip(rhs.as_slice().iter())
            .map(|(a, b)| a / b)
            .collect();
        NDArrayField::from_shape_vec(shape, v)
    }
}

// Scalar multiplication/division implementations
impl Mul<f64> for NDArrayField {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        NDArrayField::from_shape_vec(
            self.shape(),
            self.as_slice().iter().map(|x| x * rhs).collect(),
        )
    }
}
impl Div<f64> for NDArrayField {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        NDArrayField::from_shape_vec(
            self.shape(),
            self.as_slice().iter().map(|x| x / rhs).collect(),
        )
    }
}

impl ContinuousTensor<f64> for NDArrayField {
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let v: Vec<f64> = self.as_slice().iter().copied().map(f).collect();
        NDArrayField::from_shape_vec(self.shape(), v)
    }

    fn zip_with<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        // Note: For stricter shape checking, we should assert_eq!(self.shape(), other.shape());
        let v: Vec<f64> = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        NDArrayField::from_shape_vec(self.shape(), v)
    }

    fn reduce<F>(&self, init: f64, f: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        self.as_slice().iter().copied().fold(init, f)
    }

    fn integrate(&self) -> f64 {
        // Trapezoidal on flattened data
        let s = self.as_slice();
        let n = s.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return s[0];
        }
        let mut sum = 0.0;
        for i in 0..n - 1 {
            sum += 0.5 * (s[i] + s[i + 1]);
        }
        sum
    }

    fn gradient(&self) -> Self {
        let s = self.as_slice();
        let n = s.len();
        if n == 0 {
            return NDArrayField::from_shape_vec(vec![0], vec![]);
        }
        if n == 1 {
            return NDArrayField::from_shape_vec(vec![1], vec![0.0]);
        }
        let mut g = vec![0.0; n];
        g[0] = s[1] - s[0];
        for i in 1..n - 1 {
            g[i] = 0.5 * (s[i + 1] - s[i - 1]);
        }
        g[n - 1] = s[n - 1] - s[n - 2];
        NDArrayField::from_shape_vec(self.shape(), g)
    }

    fn constant_like(&self, value: f64) -> Self {
        NDArrayField::from_shape_vec(self.shape(), vec![value; self.len()])
    }

    fn new_with_data(&self, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), self.len(), "Data length mismatch");
        NDArrayField::from_shape_vec(self.shape(), data)
    }
}

impl From<crate::physics::cta::VectorField> for NDArrayField {
    fn from(v: crate::physics::cta::VectorField) -> Self {
        NDArrayField::from_shape_vec(vec![v.len()], v.as_slice().to_vec())
    }
}

impl AsRef<[f64]> for NDArrayField {
    fn as_ref(&self) -> &[f64] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;

    #[test]
    fn test_ndarray_from_vector() {
        let v = VectorField::new(vec![1.0, 2.0, 3.0]);
        let n: NDArrayField = v.into();
        assert_eq!(n.len(), 3);
        assert_eq!(n.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ndarray_map_preserves_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = NDArrayField::from_shape_vec(vec![2, 3], data);
        assert_eq!(tensor.shape(), vec![2, 3]);
        let mapped = tensor.map(|x| x * 2.0);
        assert_eq!(mapped.shape(), vec![2, 3]);
    }

    #[test]
    fn test_ndarray_add_preserves_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = NDArrayField::from_shape_vec(vec![2, 3], data.clone());
        let t2 = NDArrayField::from_shape_vec(vec![2, 3], data);
        let sum = t1 + t2;
        assert_eq!(sum.shape(), vec![2, 3]);
    }
}
