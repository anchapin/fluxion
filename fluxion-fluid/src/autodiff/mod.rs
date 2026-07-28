//! Automatic differentiation module for HVAC component Jacobians.
//!
//! This module provides the [`DifferentiableComponent`] trait for exposing
//! exact analytical Jacobian matrices from HVAC equipment, enabling reverse-mode
//! automatic differentiation for Model Predictive Control (MPC) and setpoint
//! optimization.
//!
//! ## Design
//!
//! All HVAC equipment nodes (chillers, boilers, coils, VAV boxes, pumps) implement
//! the [`DifferentiableComponent`] trait that returns:
//!
//! - `jacobian_input`: partial Outputs / partial Inputs
//! - `jacobian_state`: partial Outputs / partial States
//!
//! ## Accuracy Verification
//!
//! All analytical Jacobians are verified against finite-difference approximations
//! with epsilon = 1e-6 and relative tolerance 1e-4.
//!
//! ## No Heap Allocation
//!
//! Jacobian computation uses pre-allocated arrays via nalgebra's stack-allocated
//! matrices for zero heap allocation in the hot path.

pub mod component;
pub mod validation;

pub use component::{
    finite_diff_jacobian, optimize_with_gradient_descent, relative_diff, DifferentiableComponent,
    GradientDescentOptimizer,
};
pub use validation::{finite_diff_epsilon, relative_error, verify_jacobian_entries};
