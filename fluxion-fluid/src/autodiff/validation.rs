//! Jacobian validation module for HVAC components.
//!
//! Provides utilities for verifying analytical Jacobians against finite-difference
//! approximations and gradient descent optimization tests.

use nalgebra::DMatrix;

const FINITE_DIFF_EPSILON: f64 = 1e-6;
const RELATIVE_TOLERANCE: f64 = 1e-4;

pub fn finite_diff_epsilon() -> f64 {
    FINITE_DIFF_EPSILON
}

pub fn relative_tolerance() -> f64 {
    RELATIVE_TOLERANCE
}

pub fn compute_finite_diff_jacobian<F>(f: F, x: &[f64], epsilon: f64) -> DMatrix<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let fx = f(x);
    let m = fx.len();
    let mut jac = DMatrix::zeros(m, n);

    for j in 0..n {
        let mut x_plus = x.to_vec();
        x_plus[j] += epsilon;
        let f_plus = f(&x_plus);

        for i in 0..m {
            jac[(i, j)] = (f_plus[i] - fx[i]) / epsilon;
        }
    }

    jac
}

pub fn relative_error(a: f64, b: f64) -> f64 {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let denom = abs_a.max(abs_b);
    if denom < 1e-10 {
        (a - b).abs() / 1e-10
    } else {
        (a - b).abs() / denom
    }
}

pub fn verify_jacobian_entries(analytical: &DMatrix<f64>, finite_diff: &DMatrix<f64>) -> bool {
    if analytical.nrows() != finite_diff.nrows() || analytical.ncols() != finite_diff.ncols() {
        return false;
    }

    for i in 0..analytical.nrows() {
        for j in 0..analytical.ncols() {
            if relative_error(analytical[(i, j)], finite_diff[(i, j)]) > RELATIVE_TOLERANCE {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_error_identical() {
        assert!(relative_error(1.0, 1.0) < 1e-10);
    }

    #[test]
    fn test_relative_error_small_diff() {
        let err = relative_error(1.0, 1.0 + 1e-5);
        assert!((err - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_finite_diff_jacobian_linear() {
        // f(x) = [2*x0 + x1, x0 - x1]
        // J = [[2, 1], [1, -1]]
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1], x[0] - x[1]];
        let x = [1.0, 2.0];
        let jac = compute_finite_diff_jacobian(f, &x, 1e-6);

        assert!((jac[(0, 0)] - 2.0).abs() < 1e-4);
        assert!((jac[(0, 1)] - 1.0).abs() < 1e-4);
        assert!((jac[(1, 0)] - 1.0).abs() < 1e-4);
        assert!((jac[(1, 1)] - (-1.0)).abs() < 1e-4);
    }
}
