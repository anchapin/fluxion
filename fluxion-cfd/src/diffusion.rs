//! Implicit diffusion solver for FFD
//!
//! This module implements the diffusion step using an implicit scheme.

use crate::{CfdResult, Field3d, Grid3d, VelocityField};

pub struct DiffusionSolver {
    max_iter: usize,
    tolerance: f64,
}

impl DiffusionSolver {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            max_iter,
            tolerance,
        }
    }

    pub fn with_config(max_iter: usize, tolerance: f64) -> Self {
        Self::new(max_iter, tolerance)
    }

    pub fn step(&self, grid: &Grid3d, dt: f64, nu: f64, scalar: &mut Field3d) -> CfdResult<()> {
        let alpha = dt * nu;
        self.solve_implicit(grid, alpha, scalar)?;
        Ok(())
    }

    pub fn diffuse_velocity(
        &self,
        grid: &Grid3d,
        dt: f64,
        nu: f64,
        velocity: &mut VelocityField,
    ) -> CfdResult<()> {
        let alpha = dt * nu;
        self.solve_implicit(grid, alpha, &mut velocity.u)?;
        self.solve_implicit(grid, alpha, &mut velocity.v)?;
        self.solve_implicit(grid, alpha, &mut velocity.w)?;
        Ok(())
    }

    fn solve_implicit(&self, grid: &Grid3d, alpha: f64, scalar: &mut Field3d) -> CfdResult<()> {
        let _nx = grid.nx;
        let _ny = grid.ny;
        let _nz = grid.nz;
        let _dx2 = grid.dx * grid.dx;
        let _dy2 = grid.dy * grid.dy;
        let _dz2 = grid.dz * grid.dz;
        let mut r = scalar.data.clone();
        let mut p = scalar.data.clone();
        let mut ap = vec![0.0; scalar.data.len()];
        let rsold = self.dot(&r, &r);
        if rsold.sqrt() < self.tolerance {
            return Ok(());
        }
        for _iter in 0..self.max_iter {
            self.apply_laplacian(grid, alpha, &p, &mut ap)?;
            let alpha_k = self.dot(&p, &ap);
            if alpha_k.abs() < 1e-15 {
                break;
            }
            let alpha_val = rsold / alpha_k;
            for i in 0..scalar.data.len() {
                r[i] -= alpha_val * ap[i];
                scalar.data[i] += alpha_val * p[i];
            }
            let rsnew = self.dot(&r, &r);
            if rsnew.sqrt() < self.tolerance {
                break;
            }
            let beta = rsnew / rsold;
            for i in 0..p.len() {
                p[i] = r[i] + beta * p[i];
            }
        }
        Ok(())
    }

    fn apply_laplacian(
        &self,
        grid: &Grid3d,
        alpha: f64,
        u: &[f64],
        result: &mut [f64],
    ) -> CfdResult<()> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let dz2 = grid.dz * grid.dz;
        let ax = alpha / dx2;
        let ay = alpha / dy2;
        let az = alpha / dz2;
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + nx * (j + ny * k);
                    let center = u[idx];
                    let left = u[idx - 1];
                    let right = u[idx + 1];
                    let down = u[idx - nx];
                    let up = u[idx + nx];
                    let back = u[idx - nx * ny];
                    let front = u[idx + nx * ny];
                    let laplacian = (left - 2.0 * center + right) * ax
                        + (down - 2.0 * center + up) * ay
                        + (back - 2.0 * center + front) * az;
                    result[idx] = u[idx] - laplacian;
                }
            }
        }
        for k in 0..nz {
            for j in 0..ny {
                let idx0 = nx * (j + ny * k);
                result[idx0] = u[idx0];
                let idxn = (nx - 1) + nx * (j + ny * k);
                result[idxn] = u[idxn];
            }
        }
        for k in 0..nz {
            for i in 0..nx {
                let idx0 = i + nx * ny * k;
                result[idx0] = u[idx0];
                let idxn = i + nx * ((ny - 1) + ny * k);
                result[idxn] = u[idxn];
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let idx0 = i + nx * j;
                result[idx0] = u[idx0];
                let idxn = i + nx * (j + ny * (nz - 1));
                result[idxn] = u[idxn];
            }
        }
        Ok(())
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

impl Default for DiffusionSolver {
    fn default() -> Self {
        Self::new(1000, 1e-8)
    }
}
