use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

/// Defines a continuous scalar field over a 2D domain.
pub trait ContinuousField<T>
where
    T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
{
    /// Evaluates the field at a given (u, v) coordinate.
    fn at(&self, u: f64, v: f64) -> T;

    /// Computes the double integral of the field over a rectangular region.
    fn integrate(&self, min_u: f64, max_u: f64, min_v: f64, max_v: f64) -> T {
        let steps = 100;
        let du = (max_u - min_u) / steps as f64;
        let dv = (max_v - min_v) / steps as f64;
        let mut sum = T::zero();

        for i in 0..steps {
            for j in 0..steps {
                let u = min_u + (i as f64 + 0.5) * du;
                let v = min_v + (j as f64 + 0.5) * dv;
                sum += self.at(u, v) * (du * dv);
            }
        }
        sum
    }
}

/// A simple implementation of ContinuousField representing a constant value over the domain.
pub struct ConstantField<T> {
    pub value: T,
}

impl<T> ContinuousField<T> for ConstantField<T>
where
    T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
{
    fn at(&self, _u: f64, _v: f64) -> T {
        self.value.clone()
    }

    fn integrate(&self, min_u: f64, max_u: f64, min_v: f64, max_v: f64) -> T {
        self.value.clone() * ((max_u - min_u) * (max_v - min_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LinearField {
        slope_u: f64,
        slope_v: f64,
        intercept: f64,
    }

    impl ContinuousField<f64> for LinearField {
        fn at(&self, u: f64, v: f64) -> f64 {
            self.slope_u * u + self.slope_v * v + self.intercept
        }
    }

    #[test]
    fn test_constant_field_integration() {
        let field = ConstantField { value: 5.0 };
        let result = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_field_integration() {
        let field = LinearField {
            slope_u: 1.0,
            slope_v: 1.0,
            intercept: 0.0,
        };
        let result = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!((result - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_integration_bounds() {
        let field = ConstantField { value: 2.0 };
        let result = field.integrate(0.0, 2.0, 0.0, 2.0);
        assert!((result - 8.0).abs() < 1e-6);
    }
}
