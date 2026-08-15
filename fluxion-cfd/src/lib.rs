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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfd_error_grid_display_includes_context() {
        let e = CfdError::Grid("nx must be positive".into());
        let s = format!("{e}");
        assert!(
            s.contains("Grid"),
            "display should mention 'Grid' (got {s:?})"
        );
        assert!(
            s.contains("nx must be positive"),
            "display should include the context message (got {s:?})"
        );
    }

    #[test]
    fn cfd_error_solver_display_includes_context() {
        let e = CfdError::Solver("CG did not converge".into());
        let s = format!("{e}");
        assert!(
            s.contains("Solver"),
            "display should mention 'Solver' (got {s:?})"
        );
        assert!(s.contains("CG did not converge"));
    }

    #[test]
    fn cfd_error_gpu_display_includes_context() {
        let e = CfdError::Gpu("cuda kernel launch failed".into());
        let s = format!("{e}");
        assert!(
            s.contains("GPU"),
            "display should mention 'GPU' (got {s:?})"
        );
        assert!(s.contains("cuda kernel launch failed"));
    }

    #[test]
    fn cfd_error_invalid_parameter_display_includes_context() {
        let e = CfdError::InvalidParameter("dt must be > 0".into());
        let s = format!("{e}");
        assert!(
            s.contains("Invalid parameter"),
            "display should mention 'Invalid parameter' (got {s:?})"
        );
        assert!(s.contains("dt must be > 0"));
    }

    #[test]
    fn cfd_error_convergence_display_includes_context() {
        let e = CfdError::Convergence("Poisson solve exceeded 1000 iters".into());
        let s = format!("{e}");
        assert!(
            s.contains("Convergence"),
            "display should mention 'Convergence' (got {s:?})"
        );
        assert!(s.contains("Poisson solve exceeded 1000 iters"));
    }

    #[test]
    fn cfd_error_variants_are_distinct() {
        let errors = [
            CfdError::Grid(String::new()),
            CfdError::Solver(String::new()),
            CfdError::Gpu(String::new()),
            CfdError::InvalidParameter(String::new()),
            CfdError::Convergence(String::new()),
        ];
        let formatted: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
        for (i, a) in formatted.iter().enumerate() {
            for (j, b) in formatted.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "variants {i} and {j} must format differently");
                }
            }
        }
    }

    #[test]
    fn cfd_result_ok_unwraps_to_value() {
        let r: CfdResult<i32> = Ok(42);
        match r {
            Ok(v) => assert_eq!(v, 42),
            Err(_) => panic!("expected Ok variant"),
        }
    }

    #[test]
    fn cfd_result_err_propagates() {
        let r: CfdResult<i32> = Err(CfdError::Grid("nope".into()));
        assert!(r.is_err());
    }

    #[test]
    fn public_reexports_are_constructible() {
        let _ = AdvectionSolver::new();
        let _ = DiffusionSolver::default();
        let _ = PressureSolver::default();
        let cfg = FfdConfig {
            nx: 4,
            ny: 4,
            nz: 4,
            ..FfdConfig::default()
        };
        let _ = FfdCfdSolver::new(cfg).expect("FfdCfdSolver::new must succeed on a 4x4x4 grid");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_dispatch_reports_cpu_backend_by_default() {
        use crate::gpu::{get_available_backend, supports_gpu, GpuBackend};
        assert_eq!(get_available_backend(), GpuBackend::CPU);
        assert!(!supports_gpu());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_config_default_uses_cpu_backend() {
        use crate::gpu::{GpuBackend, GpuConfig};
        let cfg = GpuConfig::default();
        assert_eq!(cfg.backend, GpuBackend::CPU);
    }
}
