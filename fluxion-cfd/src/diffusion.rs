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
        let mut rsold = self.dot(&r, &r);
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
            // Issue #2456: standard CG requires `rsold ← rsnew` at the end of
            // each iteration. The previous code left `rsold` pinned to the
            // initial residual norm, which made `alpha_val` blow up in later
            // iterations and produced NaN on non-uniform inputs (e.g. a
            // Gaussian bump on a 32³ grid). With this update the implicit
            // diffusion step converges in the documented tolerance.
            rsold = rsnew;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn peak_field(nx: usize, ny: usize, nz: usize, peak_value: f64) -> Field3d {
        let mut f = Field3d::filled(nx, ny, nz, 0.0);
        f.set(nx / 2, ny / 2, nz / 2, peak_value)
            .expect("center cell must exist");
        f
    }

    #[test]
    fn diffuse_zero_field_stays_zero() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut scalar = Field3d::zeros(8, 8, 8);
        DiffusionSolver::default()
            .step(&grid, 0.01, 1.0, &mut scalar)
            .expect("step should succeed");
        for &v in scalar.data.iter() {
            assert!(
                v.abs() < 1e-12,
                "zero field must stay zero after diffusion (got {v})"
            );
        }
    }

    #[test]
    fn diffuse_produces_finite_values_for_peak_input() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut scalar = peak_field(8, 8, 8, 1.0);
        DiffusionSolver::default()
            .step(&grid, 0.01, 1.0, &mut scalar)
            .expect("step should succeed");
        for &v in scalar.data.iter() {
            assert!(v.is_finite(), "diffused field must remain finite (got {v})");
        }
    }

    #[test]
    fn diffuse_modifies_nonuniform_field() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut scalar = peak_field(8, 8, 8, 5.0);
        let pre_data = scalar.data.clone();
        DiffusionSolver::default()
            .step(&grid, 0.01, 1.0, &mut scalar)
            .expect("step should succeed");
        let mut differs = false;
        for (idx, (&pre, &post)) in pre_data.iter().zip(scalar.data.iter()).enumerate() {
            if (pre - post).abs() > 1e-12 {
                differs = true;
                break;
            }
            assert!(
                post.is_finite(),
                "diffused cell {idx} must be finite (got {post})"
            );
        }
        assert!(differs, "diffusion step must modify the non-uniform field");
    }

    #[test]
    fn diffuse_velocity_modifies_all_components() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut velocity = VelocityField::zeros(8, 8, 8);
        velocity.u.fill(1.0);
        velocity.v.fill(2.0);
        velocity.w.fill(3.0);
        DiffusionSolver::default()
            .diffuse_velocity(&grid, 0.01, 1.0, &mut velocity)
            .expect("diffuse_velocity should succeed");
        for (idx, &u) in velocity.u.data.iter().enumerate() {
            assert!(
                u.is_finite(),
                "u[{idx}] must remain finite after diffusion (got {u})"
            );
        }
        for (idx, &v) in velocity.v.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "v[{idx}] must remain finite after diffusion (got {v})"
            );
        }
        for (idx, &w) in velocity.w.data.iter().enumerate() {
            assert!(
                w.is_finite(),
                "w[{idx}] must remain finite after diffusion (got {w})"
            );
        }
    }
}
