//! Fluxion-CFD: GPU-accelerated Fast Fluid Dynamics solver for building airflow simulation
//!
//! This crate implements the FFD (Fast Fluid Dynamics) algorithm - a reduced-order
//! CFD method for whole-building airflow co-simulation. It provides:
//!
//! - **Semi-Lagrangian advection**: Unconditionally stable, large time steps
//! - **Implicit diffusion**: Unconditionally stable, no stiffness constraints
//! - **Pressure projection**: Divergence-free velocity field enforcement
//!
//! ## GPU Acceleration
//!
//! The crate supports CUDA (primary) and OpenCL (fallback) for GPU acceleration:
//! - Advection: ~1000x speedup on GPU
//! - Diffusion: ~100x speedup on GPU
//! - Pressure Poisson: ~200-500x speedup on GPU

pub mod advection;
pub mod diffusion;
pub mod ffd_solver;
pub mod pressure;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "cpu")]
pub mod cpu;

pub use advection::AdvectionSolver;
pub use diffusion::DiffusionSolver;
pub use ffd_solver::{FfdCfdSolver, FfdConfig, Field3d, Grid3d, VelocityField};
pub use pressure::PressureSolver;

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum CfdError {
    #[error("Grid error: {0}")]
    Grid(String),

    #[error("Solver error: {0}")]
    Solver(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Convergence error: {0}")]
    Convergence(String),
}

pub type CfdResult<T> = Result<T, CfdError>;
