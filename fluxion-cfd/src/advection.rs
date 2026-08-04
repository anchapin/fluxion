//! Semi-Lagrangian advection solver for FFD
//!
//! This module implements the advection step using semi-Lagrangian backtrace.
//! The method is unconditionally stable, allowing large time steps.

use crate::{CfdResult, Field3d, Grid3d, VelocityField};

#[allow(dead_code)]
pub struct AdvectionSolver {
    order: InterpolationOrder,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum InterpolationOrder {
    #[default]
    Linear,
    Cubic,
}

impl AdvectionSolver {
    pub fn new() -> Self {
        Self {
            order: InterpolationOrder::Linear,
        }
    }

    pub fn with_order(order: InterpolationOrder) -> Self {
        Self { order }
    }

    pub fn step(
        &self,
        grid: &Grid3d,
        dt: f64,
        velocity: &VelocityField,
        scalar: &mut Field3d,
    ) -> CfdResult<()> {
        let mut temp = scalar.clone();
        self.advect_scalar(grid, dt, velocity, scalar, &mut temp)?;
        scalar.copy_from(&temp)?;
        Ok(())
    }

    pub fn advect_velocity(
        &self,
        grid: &Grid3d,
        dt: f64,
        velocity: &VelocityField,
        u_star: &mut Field3d,
        v_star: &mut Field3d,
        w_star: &mut Field3d,
    ) -> CfdResult<()> {
        let mut temp_u = u_star.clone();
        let mut temp_v = v_star.clone();
        let mut temp_w = w_star.clone();
        self.backtrace(grid, dt, velocity, u_star, &mut temp_u)?;
        self.backtrace(grid, dt, velocity, v_star, &mut temp_v)?;
        self.backtrace(grid, dt, velocity, w_star, &mut temp_w)?;
        u_star.copy_from(&temp_u)?;
        v_star.copy_from(&temp_v)?;
        w_star.copy_from(&temp_w)?;
        Ok(())
    }

    fn advect_scalar(
        &self,
        grid: &Grid3d,
        dt: f64,
        velocity: &VelocityField,
        scalar: &Field3d,
        result: &mut Field3d,
    ) -> CfdResult<()> {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = grid.linear_index(i, j, k).unwrap();
                    let u = velocity.u.data[idx];
                    let v = velocity.v.data[idx];
                    let w = velocity.w.data[idx];
                    let x0 = i as f64 - dt * u / grid.dx;
                    let y0 = j as f64 - dt * v / grid.dy;
                    let z0 = k as f64 - dt * w / grid.dz;
                    let value = self.interpolate_scalar(scalar, grid, x0, y0, z0);
                    result.data[idx] = value;
                }
            }
        }
        Ok(())
    }

    fn backtrace(
        &self,
        grid: &Grid3d,
        dt: f64,
        velocity: &VelocityField,
        field: &Field3d,
        result: &mut Field3d,
    ) -> CfdResult<()> {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = grid.linear_index(i, j, k).unwrap();
                    let u = velocity.u.data[idx];
                    let v = velocity.v.data[idx];
                    let w = velocity.w.data[idx];
                    let x0 = i as f64 - dt * u / grid.dx;
                    let y0 = j as f64 - dt * v / grid.dy;
                    let z0 = k as f64 - dt * w / grid.dz;
                    let value = self.interpolate_scalar(field, grid, x0, y0, z0);
                    result.data[idx] = value;
                }
            }
        }
        Ok(())
    }

    fn interpolate_scalar(&self, field: &Field3d, grid: &Grid3d, x: f64, y: f64, z: f64) -> f64 {
        let x = x.clamp(0.0, (grid.nx - 1) as f64);
        let y = y.clamp(0.0, (grid.ny - 1) as f64);
        let z = z.clamp(0.0, (grid.nz - 1) as f64);
        let i0 = x.floor() as usize;
        let j0 = y.floor() as usize;
        let k0 = z.floor() as usize;
        let i1 = (i0 + 1).min(grid.nx - 1);
        let j1 = (j0 + 1).min(grid.ny - 1);
        let k1 = (k0 + 1).min(grid.nz - 1);
        let fx = x - i0 as f64;
        let fy = y - j0 as f64;
        let fz = z - k0 as f64;
        let c000 = field.get(i0, j0, k0).unwrap_or(0.0);
        let c100 = field.get(i1, j0, k0).unwrap_or(0.0);
        let c010 = field.get(i0, j1, k0).unwrap_or(0.0);
        let c110 = field.get(i1, j1, k0).unwrap_or(0.0);
        let c001 = field.get(i0, j0, k1).unwrap_or(0.0);
        let c101 = field.get(i1, j0, k1).unwrap_or(0.0);
        let c011 = field.get(i0, j1, k1).unwrap_or(0.0);
        let c111 = field.get(i1, j1, k1).unwrap_or(0.0);
        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }
}

impl Default for AdvectionSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_advection_solver_creation() {
        let solver = AdvectionSolver::new();
        assert!(matches!(solver.order, InterpolationOrder::Linear));
    }

    #[test]
    fn test_interpolate_scalar_constant_field() {
        let solver = AdvectionSolver::new();
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let field = Field3d::filled(8, 8, 8, 1.0);
        let value = solver.interpolate_scalar(&field, &grid, 3.5, 3.5, 3.5);
        assert!(approx_eq(value, 1.0, 1e-10));
    }
}
