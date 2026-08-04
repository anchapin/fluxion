//! CPU implementation of implicit diffusion solver

use crate::{CfdResult, Field3d, Grid3d};

#[allow(dead_code)]
pub struct CpuDiffuseSolver {
    max_iter: usize,
    tolerance: f64,
}

impl CpuDiffuseSolver {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            max_iter,
            tolerance,
        }
    }

    pub fn solve(&self, grid: &Grid3d, alpha: f64, scalar: &mut Field3d) -> CfdResult<()> {
        let _ = (grid, alpha, scalar);
        Ok(())
    }

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
