//! DifferentiableComponent trait and implementations for HVAC equipment.
//!
//! This module provides the core trait for analytical Jacobian computation
//! and implementations for all major HVAC component types.

use nalgebra::{DMatrix, DVector};

const EPSILON: f64 = 1e-10;

pub use super::validation::{finite_diff_epsilon, relative_error, verify_jacobian_entries};

/// Differentiable HVAC component trait for analytical Jacobian computation.
///
/// This trait enables reverse-mode automatic differentiation for Model Predictive
/// Control (MPC) and setpoint optimization by exposing exact derivative matrices.
///
/// # Type Parameters
///
/// * `Input` - Input variables (control signals, upstream conditions)
/// * `Output` - Output variables (capacity, energy consumption, outlet conditions)
/// * `State` - Internal state variables
///
/// # Jacobian Matrices
///
/// The trait provides two Jacobian matrices:
///
/// - `jacobian_input`: $\\frac{\\partial \\text{Outputs}}{\\partial \\text{Inputs}})$
///   - Shape: `(n_outputs, n_inputs)`
/// - `jacobian_state`: $\\frac{\\partial \\text{Outputs}}{\\partial \\text{States}})$
///   - Shape: `(n_outputs, n_states)`
///
/// # Accuracy Verification
///
/// All implementations verify analytical Jacobians against finite-difference
/// approximations with $\\epsilon = 10^{-6}$ and relative tolerance $10^{-4}$.
pub trait DifferentiableComponent: Send + Sync {
    type Input;
    type Output;
    type State;

    fn evaluate(&self, input: &Self::Input, state: &Self::State) -> Self::Output;

    fn jacobian_input(&self, input: &Self::Input, state: &Self::State) -> DMatrix<f64>;

    fn jacobian_state(&self, input: &Self::Input, state: &Self::State) -> DMatrix<f64>;

    fn num_inputs(&self) -> usize;

    fn num_outputs(&self) -> usize;

    fn num_states(&self) -> usize;
}

/// Compute finite-difference Jacobian approximation for validation.
///
/// Uses forward differences: $\\frac{f(x + \\epsilon) - f(x)}{\\epsilon}$
pub fn finite_diff_jacobian<F>(f: F, x: &[f64], epsilon: f64) -> DMatrix<f64>
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

/// Relative difference between two values, handling near-zero cases.
#[inline]
pub fn relative_diff(a: f64, b: f64) -> f64 {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let max_ab = abs_a.max(abs_b);
    let diff = (a - b).abs();
    // When both values are very small, treat them as essentially equal
    if abs_a <= EPSILON && abs_b <= EPSILON {
        0.0
    } else if max_ab < EPSILON {
        diff / EPSILON
    } else {
        diff / max_ab
    }
}

/// Gradient descent optimizer using analytical Jacobians.
///
/// Demonstrates MPC control using the analytical Jacobian for a VAV box
/// supply air temperature controller.
pub struct GradientDescentOptimizer {
    pub learning_rate: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for GradientDescentOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            tolerance: 1e-3,
            max_iterations: 10,
        }
    }
}

impl GradientDescentOptimizer {
    pub fn new(learning_rate: f64, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            tolerance,
            max_iterations,
        }
    }

    pub fn optimize<C>(
        &self,
        component: &C,
        target: &[f64],
        input: &mut [f64],
        state: &[f64],
    ) -> usize
    where
        C: DifferentiableComponent<Input = Vec<f64>, Output = Vec<f64>, State = Vec<f64>>,
    {
        let mut iteration = 0;

        for _ in 0..self.max_iterations {
            // Convert slices to Vec for trait calls
            let input_vec = input.to_vec();
            let state_vec = state.to_vec();
            let output = component.evaluate(&input_vec, &state_vec);

            // Compute error vector
            let error: Vec<f64> = output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| o - t)
                .collect();

            // Compute gradient using Jacobian for squared error:
            // E = 0.5 * sum(error_i^2), dE/dx = sum(error_i * d(error_i)/dx) = J^T * error
            let jacobian = component.jacobian_input(&input_vec, &state_vec);
            let error_vec = DVector::from_vec(error);
            let gradient = jacobian.transpose() * &error_vec;

            // Check convergence on error norm
            let error_norm = error_vec.norm();
            if error_norm < self.tolerance {
                break;
            }

            // Update: input = input - learning_rate * gradient
            for i in 0..input.len() {
                input[i] -= self.learning_rate * gradient[i];
            }

            iteration += 1;
        }

        iteration
    }
}

/// Optimize control input using gradient descent with analytical Jacobian.
pub fn optimize_with_gradient_descent<C>(
    component: &C,
    target: &[f64],
    input: &mut [f64],
    state: &[f64],
    learning_rate: f64,
    tolerance: f64,
    max_iterations: usize,
) -> usize
where
    C: DifferentiableComponent<Input = Vec<f64>, Output = Vec<f64>, State = Vec<f64>>,
{
    let optimizer = GradientDescentOptimizer::new(learning_rate, tolerance, max_iterations);
    optimizer.optimize(component, target, input, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_diff_near_zero() {
        assert!(relative_diff(0.0, 0.0) < 1e-10);
        assert!(relative_diff(1e-10, 0.0) < 1e-2);
    }

    #[test]
    fn test_relative_diff_normal() {
        assert!((relative_diff(1.0, 1.0 + 1e-5) - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_finite_diff_jacobian_linear() {
        // f(x) = [2*x0 + x1, x0 - x1]
        // J = [[2, 1], [1, -1]]
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1], x[0] - x[1]];
        let x = [1.0, 2.0];
        let jac = finite_diff_jacobian(f, &x, 1e-6);

        assert!((jac[(0, 0)] - 2.0).abs() < 1e-4);
        assert!((jac[(0, 1)] - 1.0).abs() < 1e-4);
        assert!((jac[(1, 0)] - 1.0).abs() < 1e-4);
        assert!((jac[(1, 1)] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_gradient_descent_converges() {
        // The optimizer minimizes E = 0.5 * ||output - target||^2
        // dE/dx = J^T * (output - target)
        //
        // For output = 2*x, target = 0:
        // E = 0.5 * (2*x)^2 = 2*x^2
        // dE/dx = 4*x
        //
        // But the optimizer computes: grad = J^T * error = 2 * (2*x) = 4*x
        // This is CORRECT for minimizing E = 0.5 * ||output - target||^2
        //
        // With lr = 0.1: x_new = x - 0.1 * 4*x = x * (1 - 0.4) = x * 0.6
        // From x=10: after 14 steps, x ≈ 0.004
        // error_norm = 2*x ≈ 0.008 (still > 1e-3)
        // After 20 steps: x ≈ 0.0004, error_norm ≈ 0.0008 (< 1e-3) ✓
        struct LinearSystem;
        impl DifferentiableComponent for LinearSystem {
            type Input = Vec<f64>;
            type Output = Vec<f64>;
            type State = Vec<f64>;

            fn evaluate(&self, input: &Self::Input, _state: &Self::State) -> Self::Output {
                vec![2.0 * input[0]]
            }

            fn jacobian_input(&self, _input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![2.0])
            }

            fn jacobian_state(&self, _input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
                DMatrix::zeros(1, 0)
            }

            fn num_inputs(&self) -> usize {
                1
            }
            fn num_outputs(&self) -> usize {
                1
            }
            fn num_states(&self) -> usize {
                0
            }
        }

        let component = LinearSystem;
        let mut input = vec![10.0];
        let target = vec![0.0];
        let state = vec![];

        // With lr=0.1 and 25 iterations, should converge to error < 1e-3
        let iterations =
            optimize_with_gradient_descent(&component, &target, &mut input, &state, 0.1, 1e-3, 25);

        let error_norm = (2.0 * input[0]).abs();
        assert!(iterations < 25, "took {} iterations", iterations);
        assert!(
            error_norm < 1e-3,
            "error_norm = {} after {} iterations",
            error_norm,
            iterations
        );
    }
}
