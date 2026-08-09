//! Fluxion-CFD: Fast Fluid Dynamics solver for building airflow simulation
//!
//! This crate implements the FFD (Fast Fluid Dynamics) algorithm - a reduced-order
//! CFD method for whole-building airflow co-simulation. It provides:
//!
//! - **Semi-Lagrangian advection**: Unconditionally stable, large time steps
//! - **Implicit diffusion**: Unconditionally stable, no stiffness constraints
//! - **Pressure projection**: Divergence-free velocity field enforcement
//!
//! ## GPU Acceleration (Issue #2386 / #2456)
//!
//! The crate is wired for CUDA (primary) and OpenCL (fallback) GPU
//! acceleration behind `--features cuda` / `--features opencl`. The
//! architecture contract is in place (`GpuBackend` enum,
//! `get_available_backend` / `supports_gpu` accessors, the `gpu` module
//! stub) but **the actual CUDA / OpenCL kernels are not yet
//! implemented**. CUDA kernel authoring is tracked as a follow-up
//! issue (out of scope for #2456).
//!
//! The numbers below are **target** annotations, not measured values.
//! They were claimed when issue #2386 closed but cannot be falsified
//! against the current code path (the prior criterion bench bodies
//! returned `black_box(32 * 32 * 32)` — integer multiplication cost —
//! so the Performance Dashboard received `0.0 ns/iter` for the FFD
//! track). The CPU baseline that the GPU port must beat is now
//! measurable via `cargo bench -p fluxion-cfd --bench ffd_bench`.
//!
//! - Advection: ~1000× speedup on GPU (**target**, unmeasured)
//! - Diffusion: ~100× speedup on GPU (**target**, unmeasured)
//! - Pressure Poisson: ~200–500× speedup on GPU (**target**, unmeasured)
//!
//! ## CPU Baseline (Issue #2456 First Step)
//!
//! The [`cpu`] shim exposes a GPU-portable API surface (straight-line
//! Rust ports of the top-level loops). The shim is bit-identical to
//! the top-level implementations on a 32³ grid (enforced by
//! `fluxion-cfd/tests/cpu_gpu_parity.rs`) so a future CUDA / OpenCL
//! port that mirrors the shim's loop structure has a provable
//! reference.

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
