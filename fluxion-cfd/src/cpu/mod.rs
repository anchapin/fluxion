//! CPU implementations for FFD core routines

pub mod advect;
pub mod diffuse;
pub mod poisson;

pub use advect::CpuAdvectSolver;
pub use diffuse::CpuDiffuseSolver;
pub use poisson::CpuPoissonSolver;
