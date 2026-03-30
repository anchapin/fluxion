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

    /// Return the shape of the underlying ndarray as a `Vec<usize>`.
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

    fn elementwise_min(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a.min(b))
    }

    fn elementwise_max(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a.max(b))
    }

    fn add_assign(&mut self, other: &Self) {
        self.arr = &self.arr + &other.arr;
    }

    fn sub_assign(&mut self, other: &Self) {
        self.arr = &self.arr - &other.arr;
    }

    fn mul_assign(&mut self, other: &Self) {
        self.arr = &self.arr * &other.arr;
    }

    fn div_assign(&mut self, other: &Self) {
        self.arr = &self.arr / &other.arr;
    }
}

impl From<crate::physics::cta::VectorField> for NDArrayField {
    fn from(v: crate::physics::cta::VectorField) -> Self {
        NDArrayField::from_shape_vec(vec![v.len()], v.as_slice().to_vec())
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

    #[test]
    fn test_ndarray_add_values() {
        let t1 = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![4.0, 5.0, 6.0]);
        let sum = t1 + t2;
        assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_ndarray_sub_preserves_shape() {
        let data = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t1 = NDArrayField::from_shape_vec(vec![2, 3], data.clone());
        let t2 = NDArrayField::from_shape_vec(vec![2, 3], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let diff = t1 - t2;
        assert_eq!(diff.shape(), vec![2, 3]);
        assert_eq!(diff.as_slice(), &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_ndarray_mul_preserves_shape() {
        let t1 = NDArrayField::from_shape_vec(vec![3], vec![2.0, 3.0, 4.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![3.0, 2.0, 5.0]);
        let product = t1 * t2;
        assert_eq!(product.shape(), vec![3]);
        assert_eq!(product.as_slice(), &[6.0, 6.0, 20.0]);
    }

    #[test]
    fn test_ndarray_div_preserves_shape() {
        let t1 = NDArrayField::from_shape_vec(vec![3], vec![10.0, 12.0, 16.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![2.0, 3.0, 4.0]);
        let quotient = t1 / t2;
        assert_eq!(quotient.shape(), vec![3]);
        assert_eq!(quotient.as_slice(), &[5.0, 4.0, 4.0]);
    }

    #[test]
    fn test_ndarray_scalar_mul() {
        let t = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let scaled = t * 2.5;
        assert_eq!(scaled.shape(), vec![3]);
        assert_eq!(scaled.as_slice(), &[2.5, 5.0, 7.5]);
    }

    #[test]
    fn test_ndarray_scalar_div() {
        let t = NDArrayField::from_shape_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let divided = t / 5.0;
        assert_eq!(divided.shape(), vec![4]);
        assert_eq!(divided.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_ndarray_as_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = NDArrayField::from_shape_vec(vec![2, 2], data.clone());
        assert_eq!(t.as_slice(), data.as_slice());
    }

    #[test]
    fn test_ndarray_is_empty_false() {
        let t = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_ndarray_index() {
        let t = NDArrayField::from_shape_vec(vec![3], vec![10.0, 20.0, 30.0]);
        assert_eq!(t[0], 10.0);
        assert_eq!(t[1], 20.0);
        assert_eq!(t[2], 30.0);
    }

    #[test]
    fn test_ndarray_zip_with() {
        let t1 = NDArrayField::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = NDArrayField::from_shape_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let result = t1.zip_with(&t2, |a, b| a + b * 2.0);
        assert_eq!(result.as_slice(), &[21.0, 42.0, 63.0, 84.0]);
    }

    #[test]
    fn test_ndarray_zip_with_mismatched_length_panics() {
        let t1 = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let t2 = NDArrayField::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let _ = std::panic::catch_unwind(|| {
            let _ = t1.zip_with(&t2, |a, b| a + b);
        });
    }

    #[test]
    fn test_ndarray_reduce_sum() {
        let t = NDArrayField::from_shape_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let sum = t.reduce(0.0, |acc, x| acc + x);
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_ndarray_reduce_product() {
        let t = NDArrayField::from_shape_vec(vec![4], vec![2.0, 3.0, 4.0, 5.0]);
        let product = t.reduce(1.0, |acc, x| acc * x);
        assert_eq!(product, 120.0);
    }

    #[test]
    fn test_ndarray_reduce_max() {
        let t = NDArrayField::from_shape_vec(vec![5], vec![3.0, 1.0, 7.0, 2.0, 5.0]);
        let max_val = t.reduce(f64::NEG_INFINITY, |acc, x| acc.max(x));
        assert_eq!(max_val, 7.0);
    }

    #[test]
    fn test_ndarray_integrate_constant() {
        let t = NDArrayField::from_shape_vec(vec![5], vec![2.0; 5]);
        let integral = t.integrate();
        // Trapezoidal: (2+2)/2 * 4 = 8.0
        assert_eq!(integral, 8.0);
    }

    #[test]
    fn test_ndarray_integrate_linear() {
        let t = NDArrayField::from_shape_vec(vec![4], vec![0.0, 1.0, 2.0, 3.0]);
        let integral = t.integrate();
        // Trapezoidal: 0.5*(0+1) + 0.5*(1+2) + 0.5*(2+3) = 0.5 + 1.5 + 2.5 = 4.5
        assert!((integral - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_ndarray_integrate_single_element() {
        let t = NDArrayField::from_shape_vec(vec![1], vec![5.0]);
        let integral = t.integrate();
        assert_eq!(integral, 5.0);
    }

    #[test]
    fn test_ndarray_integrate_empty() {
        let t = NDArrayField::from_shape_vec(vec![0], vec![]);
        let integral = t.integrate();
        assert_eq!(integral, 0.0);
    }

    #[test]
    fn test_ndarray_gradient_linear() {
        let t = NDArrayField::from_shape_vec(vec![4], vec![0.0, 1.0, 2.0, 3.0]);
        let g = t.gradient();
        // Linear sequence should have constant gradient of 1
        assert_eq!(g.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_ndarray_gradient_constant() {
        let t = NDArrayField::from_shape_vec(vec![4], vec![5.0; 4]);
        let g = t.gradient();
        assert_eq!(g.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ndarray_gradient_single_element() {
        let t = NDArrayField::from_shape_vec(vec![1], vec![5.0]);
        let g = t.gradient();
        assert_eq!(g.as_slice(), &[0.0]);
    }

    #[test]
    fn test_ndarray_gradient_empty() {
        let t = NDArrayField::from_shape_vec(vec![0], vec![]);
        let g = t.gradient();
        assert_eq!(g.shape(), vec![0]);
        assert!(g.is_empty());
    }

    #[test]
    fn test_ndarray_constant_like() {
        let t = NDArrayField::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let constant = t.constant_like(7.5);
        assert_eq!(constant.shape(), vec![2, 3]);
        assert_eq!(constant.as_slice(), &[7.5; 6]);
    }

    #[test]
    fn test_ndarray_elementwise_min() {
        let t1 = NDArrayField::from_shape_vec(vec![4], vec![5.0, 2.0, 8.0, 1.0]);
        let t2 = NDArrayField::from_shape_vec(vec![4], vec![3.0, 7.0, 4.0, 9.0]);
        let result = t1.elementwise_min(&t2);
        assert_eq!(result.as_slice(), &[3.0, 2.0, 4.0, 1.0]);
    }

    #[test]
    fn test_ndarray_elementwise_max() {
        let t1 = NDArrayField::from_shape_vec(vec![4], vec![5.0, 2.0, 8.0, 1.0]);
        let t2 = NDArrayField::from_shape_vec(vec![4], vec![3.0, 7.0, 4.0, 9.0]);
        let result = t1.elementwise_max(&t2);
        assert_eq!(result.as_slice(), &[5.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_ndarray_add_assign() {
        let mut t1 = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![4.0, 5.0, 6.0]);
        t1.add_assign(&t2);
        assert_eq!(t1.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_ndarray_sub_assign() {
        let mut t1 = NDArrayField::from_shape_vec(vec![3], vec![5.0, 7.0, 9.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![4.0, 5.0, 6.0]);
        t1.sub_assign(&t2);
        assert_eq!(t1.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ndarray_mul_assign() {
        let mut t1 = NDArrayField::from_shape_vec(vec![3], vec![2.0, 3.0, 4.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![3.0, 2.0, 5.0]);
        t1.mul_assign(&t2);
        assert_eq!(t1.as_slice(), &[6.0, 6.0, 20.0]);
    }

    #[test]
    fn test_ndarray_div_assign() {
        let mut t1 = NDArrayField::from_shape_vec(vec![3], vec![10.0, 12.0, 16.0]);
        let t2 = NDArrayField::from_shape_vec(vec![3], vec![2.0, 3.0, 4.0]);
        t1.div_assign(&t2);
        assert_eq!(t1.as_slice(), &[5.0, 4.0, 4.0]);
    }

    #[test]
    fn test_ndarray_operations_preserve_multidimensional_shape() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let t1 = NDArrayField::from_shape_vec(vec![2, 3], data1);
        let t2 = NDArrayField::from_shape_vec(vec![2, 3], data2);

        let sum = t1.clone() + t2.clone();
        let diff = t1.clone() - t2.clone();
        let prod = t1.clone() * t2.clone();
        let quot = t1 / t2;

        assert_eq!(sum.shape(), vec![2, 3]);
        assert_eq!(diff.shape(), vec![2, 3]);
        assert_eq!(prod.shape(), vec![2, 3]);
        assert_eq!(quot.shape(), vec![2, 3]);
    }

    #[test]
    fn test_ndarray_clone_and_equality() {
        let t1 = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let t2 = t1.clone();
        assert_eq!(t1, t2);
        assert!(!std::ptr::eq(
            t1.as_slice() as *const _,
            t2.as_slice() as *const _
        ));
    }

    #[test]
    fn test_ndarray_debug_formatting() {
        let t = NDArrayField::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let debug_str = format!("{:?}", t);
        assert!(debug_str.contains("NDArrayField"));
    }
}
