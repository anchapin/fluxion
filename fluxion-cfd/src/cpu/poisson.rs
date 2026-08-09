//! CPU implementation of the pressure-Poisson solver (issue #2456).
//!
//! Wraps the top-level [`crate::pressure::PressureSolver`] projection
//! pipeline (divergence → CG Poisson → gradient correction) behind the
//! GPU-portable [`CpuPoissonSolver`] API surface. The GPU kernel author
//! will mirror this entry-point shape when adding CUDA / OpenCL
//! implementations behind `--features cuda` / `--features opencl`.
//!
//! Per RULES.md, this implementation must remain bit-identical to the
//! top-level on the same input. The bit-identity invariant is enforced
//! by `fluxion-cfd/tests/cpu_gpu_parity.rs`.

use crate::{CfdResult, Grid3d, PressureSolver, VelocityField};

#[allow(dead_code)]
pub struct CpuPoissonSolver {
    inner: PressureSolver,
}

impl CpuPoissonSolver {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            inner: PressureSolver::new(max_iter, tolerance),
        }
    }

    /// Pressure projection: `u_new = u_star - (dt/2) * ∇p` where `p` is
    /// the solution of `∇²p = ∇·u_star`. Delegates to the top-level
    /// `PressureSolver::project` so the divergence + CG-Poisson +
    /// gradient-correction sequence runs in the documented order.
    pub fn project(&self, grid: &Grid3d, dt: f64, velocity: &mut VelocityField) -> CfdResult<()> {
        self.inner.project(grid, dt, velocity)
    }

    /// GPU-portable accessor for the post-projection divergence residual
    /// (`||∇·u_new||₂`). The GPU port must return the same shape.
    pub fn compute_residual(&self, grid: &Grid3d, velocity: &VelocityField) -> CfdResult<f64> {
        self.inner.compute_residual(grid, velocity)
    }
}

impl Default for CpuPoissonSolver {
    fn default() -> Self {
        Self::new(1000, 1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression gate: the prior stub `project` returned `Ok(())`
    /// without touching `velocity`. This test fails if the solver ever
    /// regresses to a no-op — it asserts that a velocity field with
    /// non-zero divergence is actually modified after projection (the
    /// gradient-of-pressure correction subtracts a non-trivial amount).
    ///
    /// The input is a linear-in-x profile (`u = i * dx`), whose central
    /// divergence is `u_x = 1/dx = 10` — non-zero in the interior, so
    /// the projection has work to do.
    #[test]
    fn project_is_not_a_noop() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut velocity = VelocityField::zeros(8, 8, 8);
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    let idx = i + 8 * (j + 8 * k);
                    velocity.u.data[idx] = i as f64 * grid.dx; // u = x, divergence = 10
                }
            }
        }
        let pre_u_sum: f64 = velocity.u.data.iter().sum();
        CpuPoissonSolver::default()
            .project(&grid, 0.001, &mut velocity)
            .expect("project should succeed");
        let post_u_sum: f64 = velocity.u.data.iter().sum();
        assert!(
            (pre_u_sum - post_u_sum).abs() > 0.0,
            "projection must change u (pre_sum={pre_u_sum}, post_sum={post_u_sum})"
        );
    }

    /// `compute_residual` must return a non-negative scalar and must
    /// actually reflect the input velocity field. The stub returned
    /// `Ok(0.0)` regardless of input, so this test catches a regression
    /// to the stub by asserting the residual is non-zero for a velocity
    /// field with non-zero divergence (linear-in-x `u = x`).
    #[test]
    fn compute_residual_reflects_input_velocity() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut velocity = VelocityField::zeros(8, 8, 8);
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    let idx = i + 8 * (j + 8 * k);
                    velocity.u.data[idx] = i as f64 * grid.dx; // u = x, divergence = 10
                }
            }
        }
        let residual = CpuPoissonSolver::default()
            .compute_residual(&grid, &velocity)
            .expect("compute_residual should succeed");
        assert!(
            residual >= 0.0,
            "residual must be non-negative (got {residual})"
        );
        assert!(
            residual > 0.0,
            "residual must reflect the non-zero divergence (got 0.0)"
        );
    }
}
