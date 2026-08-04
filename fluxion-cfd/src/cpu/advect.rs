//! CPU implementation of semi-Lagrangian advection

use crate::{CfdResult, Field3d, Grid3d, VelocityField};

pub struct CpuAdvectSolver;

impl CpuAdvectSolver {
    pub fn new() -> Self {
        Self
    }

    pub fn advect_scalar(
        &self,
        grid: &Grid3d,
        dt: f64,
        velocity: &VelocityField,
        scalar: &Field3d,
        result: &mut Field3d,
    ) -> CfdResult<()> {
        let _ = (grid, dt, velocity, scalar, result);
        Ok(())
    }
}

impl Default for CpuAdvectSolver {
    fn default() -> Self {
        Self::new()
    }
}
