//! CPU implementation of pressure Poisson solver

use crate::{CfdResult, Grid3d, VelocityField};

#[allow(dead_code)]
pub struct CpuPoissonSolver {
    max_iter: usize,
    tolerance: f64,
}

impl CpuPoissonSolver {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            max_iter,
            tolerance,
        }
    }

    pub fn project(&self, grid: &Grid3d, dt: f64, velocity: &mut VelocityField) -> CfdResult<()> {
        let _ = (grid, dt, velocity);
        Ok(())
    }

    pub fn compute_residual(&self, grid: &Grid3d, velocity: &VelocityField) -> CfdResult<f64> {
        let _ = (grid, velocity);
        Ok(0.0)
    }
}

impl Default for CpuPoissonSolver {
    fn default() -> Self {
        Self::new(1000, 1e-6)
    }
}
