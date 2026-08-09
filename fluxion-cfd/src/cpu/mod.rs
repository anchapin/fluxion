//! CPU implementations for FFD core routines (issue #2456 First Step).
//!
//! Each module is a GPU-portable Rust port of the top-level algorithm
//! in [`crate::advection`] / [`crate::diffusion`] / [`crate::pressure`].
//! Bit-identity with the top-level is enforced by
//! `fluxion-cfd/tests/cpu_gpu_parity.rs` on a 32³ grid.

pub mod advect;
pub mod diffuse;
pub mod poisson;

pub use advect::CpuAdvectSolver;
pub use diffuse::CpuDiffuseSolver;
pub use poisson::CpuPoissonSolver;
