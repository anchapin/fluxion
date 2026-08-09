//! Pressure Poisson solver for FFD

use crate::{CfdResult, Field3d, Grid3d, VelocityField};

pub struct PressureSolver {
    max_iter: usize,
    tolerance: f64,
}

impl PressureSolver {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            max_iter,
            tolerance,
        }
    }

    pub fn with_config(max_iter: usize, tolerance: f64) -> Self {
        Self::new(max_iter, tolerance)
    }

    pub fn project(&self, grid: &Grid3d, dt: f64, velocity: &mut VelocityField) -> CfdResult<()> {
        let mut divergence = self.compute_divergence(grid, velocity)?;
        self.solve_poisson(grid, &mut divergence)?;
        self.apply_gradient(grid, dt, &divergence, velocity)?;
        self.enforce_divergence_free(grid, velocity)?;
        Ok(())
    }

    fn compute_divergence(&self, grid: &Grid3d, velocity: &VelocityField) -> CfdResult<Field3d> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        let mut div = Field3d::zeros(nx, ny, nz);
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + nx * (j + ny * k);
                    let u_left = velocity.u.data[idx - 1];
                    let u_right = velocity.u.data[idx + 1];
                    let v_down = velocity.v.data[idx - nx];
                    let v_up = velocity.v.data[idx + nx];
                    let w_back = velocity.w.data[idx - nx * ny];
                    let w_front = velocity.w.data[idx + nx * ny];
                    div.data[idx] = (u_right - u_left) / (2.0 * dx)
                        + (v_up - v_down) / (2.0 * dy)
                        + (w_front - w_back) / (2.0 * dz);
                }
            }
        }
        Ok(div)
    }

    fn solve_poisson(&self, grid: &Grid3d, divergence: &mut Field3d) -> CfdResult<()> {
        let _nx = grid.nx;
        let _ny = grid.ny;
        let _nz = grid.nz;
        let _dx2 = grid.dx * grid.dx;
        let _dy2 = grid.dy * grid.dy;
        let _dz2 = grid.dz * grid.dz;
        let mut r = divergence.data.clone();
        let mut p = divergence.data.clone();
        let mut ap = vec![0.0; divergence.data.len()];
        let mut rsold = self.dot(&r, &r);
        if rsold.sqrt() < self.tolerance {
            return Ok(());
        }
        for _iter in 0..self.max_iter {
            self.apply_poisson_operator(grid, &p, &mut ap)?;
            let alpha_k = self.dot(&p, &ap);
            if alpha_k.abs() < 1e-15 {
                break;
            }
            let alpha_val = rsold / alpha_k;
            for i in 0..divergence.data.len() {
                r[i] -= alpha_val * ap[i];
                divergence.data[i] += alpha_val * p[i];
            }
            let rsnew = self.dot(&r, &r);
            if rsnew.sqrt() < self.tolerance {
                break;
            }
            let beta = rsnew / rsold;
            for i in 0..p.len() {
                p[i] = r[i] + beta * p[i];
            }
            // Issue #2456: standard CG requires `rsold ← rsnew` at the end of
            // each iteration. Same fix as `DiffusionSolver::solve_implicit` —
            // without it, `alpha_val = rsold/alpha_k` blows up and the Poisson
            // solve produces NaN on non-uniform divergence fields.
            rsold = rsnew;
        }
        Ok(())
    }

    fn apply_poisson_operator(
        &self,
        grid: &Grid3d,
        p: &[f64],
        result: &mut [f64],
    ) -> CfdResult<()> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let dz2 = grid.dz * grid.dz;
        let coef = 1.0 / (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2);
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + nx * (j + ny * k);
                    let left = p[idx - 1];
                    let right = p[idx + 1];
                    let down = p[idx - nx];
                    let up = p[idx + nx];
                    let back = p[idx - nx * ny];
                    let front = p[idx + nx * ny];
                    let center = p[idx];
                    result[idx] = coef
                        * ((left + right) / dx2 + (down + up) / dy2 + (back + front) / dz2
                            - center * (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2));
                }
            }
        }
        for k in 0..nz {
            for j in 0..ny {
                let idx0 = nx * (j + ny * k);
                result[idx0] = 0.0;
                let idxn = (nx - 1) + nx * (j + ny * k);
                result[idxn] = 0.0;
            }
        }
        for k in 0..nz {
            for i in 0..nx {
                let idx0 = i + nx * ny * k;
                result[idx0] = 0.0;
                let idxn = i + nx * ((ny - 1) + ny * k);
                result[idxn] = 0.0;
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let idx0 = i + nx * j;
                result[idx0] = 0.0;
                let idxn = i + nx * (j + ny * (nz - 1));
                result[idxn] = 0.0;
            }
        }
        Ok(())
    }

    fn apply_gradient(
        &self,
        grid: &Grid3d,
        dt: f64,
        pressure: &Field3d,
        velocity: &mut VelocityField,
    ) -> CfdResult<()> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        let half_dt = 0.5 * dt;
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + nx * (j + ny * k);
                    let p_left = pressure.data[idx - 1];
                    let p_right = pressure.data[idx + 1];
                    let p_down = pressure.data[idx - nx];
                    let p_up = pressure.data[idx + nx];
                    let p_back = pressure.data[idx - nx * ny];
                    let p_front = pressure.data[idx + nx * ny];
                    velocity.u.data[idx] -= half_dt * (p_right - p_left) / dx;
                    velocity.v.data[idx] -= half_dt * (p_up - p_down) / dy;
                    velocity.w.data[idx] -= half_dt * (p_front - p_back) / dz;
                }
            }
        }
        Ok(())
    }

    fn enforce_divergence_free(
        &self,
        grid: &Grid3d,
        velocity: &mut VelocityField,
    ) -> CfdResult<()> {
        velocity.apply_boundary_velocity(
            grid,
            &mut velocity.u.clone(),
            &mut velocity.v.clone(),
            &mut velocity.w.clone(),
        );
        Ok(())
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn compute_residual(&self, grid: &Grid3d, velocity: &VelocityField) -> CfdResult<f64> {
        let div = self.compute_divergence(grid, velocity)?;
        let norm: f64 = div.data.iter().map(|&v| v * v).sum();
        Ok(norm.sqrt())
    }
}

impl Default for PressureSolver {
    fn default() -> Self {
        Self::new(1000, 1e-6)
    }
}
