//! CPU implementation of the implicit diffusion solver (issue #2456).
//!
//! Wraps the top-level [`crate::diffusion::DiffusionSolver`] CG iteration
//! behind the GPU-portable [`CpuDiffuseSolver`] API surface. The GPU
//! kernel author will mirror this entry-point shape when adding
//! CUDA / OpenCL implementations behind `--features cuda` /
//! `--features opencl`.
//!
//! Per RULES.md, this implementation must remain bit-identical to the
//! top-level on the same input. The bit-identity invariant is enforced
//! by `fluxion-cfd/tests/cpu_gpu_parity.rs`.

use crate::{CfdResult, DiffusionSolver, Field3d, Grid3d};

#[allow(dead_code)]
pub struct CpuDiffuseSolver {
    inner: DiffusionSolver,
}

impl CpuDiffuseSolver {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            inner: DiffusionSolver::new(max_iter, tolerance),
        }
    }

    /// Implicit-diffusion entry point that takes the precomputed
    /// `alpha = dt * nu` coefficient (matches the top-level
    /// `DiffusionSolver::solve_implicit` contract).
    ///
    /// The top-level `step(grid, dt, nu, scalar)` contract computes
    /// `alpha = dt * nu` internally and calls `solve_implicit(grid,
    /// alpha, scalar)`. We invoke the same `step` with `nu = 1.0` and
    /// `dt = alpha` so that `alpha_inner = alpha * 1.0 = alpha` —
    /// bit-identical arithmetic, no parameter tuning (RULES.md).
    pub fn solve(&self, grid: &Grid3d, alpha: f64, scalar: &mut Field3d) -> CfdResult<()> {
        self.inner.step(grid, alpha, 1.0, scalar)
    }

    /// GPU-portable convenience: takes `dt` and `nu` separately so the
    /// caller does not have to pre-multiply.
    pub fn diffuse_scalar(
        &self,
        grid: &Grid3d,
        dt: f64,
        nu: f64,
        scalar: &mut Field3d,
    ) -> CfdResult<()> {
        let alpha = dt * nu;
        self.solve(grid, alpha, scalar)
    }
}

impl Default for CpuDiffuseSolver {
    fn default() -> Self {
        Self::new(1000, 1e-8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression gate: the prior stub `solve` / `diffuse_scalar`
    /// returned `Ok(())` without mutating `scalar`. This test fails if
    /// the solver ever regresses to a no-op — it asserts the field
    /// actually changes after one diffusion step.
    #[test]
    fn diffuse_scalar_is_not_a_noop() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        // Initial field: 1.0 interior, 0.0 boundary. After one diffusion
        // step the interior must decrease (or at least change) as
        // heat bleeds toward the boundary.
        let mut scalar = Field3d::filled(8, 8, 8, 1.0);
        for j in 0..8 {
            for k in 0..8 {
                scalar.data[8 * (j + 8 * k)] = 0.0;
                scalar.data[7 + 8 * (j + 8 * k)] = 0.0;
            }
        }
        let pre_sum: f64 = scalar.data.iter().sum();
        CpuDiffuseSolver::default()
            .diffuse_scalar(&grid, 0.01, 1.0, &mut scalar)
            .expect("diffuse_scalar should succeed");
        let post_sum: f64 = scalar.data.iter().sum();
        assert!(
            (pre_sum - post_sum).abs() > 0.0,
            "diffusion step must change the field (pre_sum={pre_sum}, post_sum={post_sum})"
        );
    }

    /// `solve(grid, alpha, scalar)` must produce the same numerical
    /// result as `diffuse_scalar(grid, alpha, 1.0, scalar)`. This is
    /// the GPU-portable / dt-nu-portable equivalence that the GPU
    /// kernel author relies on.
    #[test]
    fn solve_matches_diffuse_scalar_for_nu_one() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let alpha = 0.01;

        let mut a = Field3d::filled(8, 8, 8, 2.0);
        let mut b = Field3d::filled(8, 8, 8, 2.0);
        CpuDiffuseSolver::default()
            .solve(&grid, alpha, &mut a)
            .unwrap();
        CpuDiffuseSolver::default()
            .diffuse_scalar(&grid, alpha, 1.0, &mut b)
            .unwrap();

        for (idx, (&x, &y)) in a.data.iter().zip(b.data.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-12,
                "solve vs diffuse_scalar mismatch at {idx}: {x} vs {y}"
            );
        }
    }
}
