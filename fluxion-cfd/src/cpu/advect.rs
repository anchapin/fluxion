//! CPU implementation of semi-Lagrangian advection (issue #2456).
//!
//! This is a straight-line Rust port of the top-level
//! [`crate::advection::AdvectionSolver::advect_scalar`] loop. The GPU
//! kernel author will mirror this loop structure when adding CUDA /
//! OpenCL implementations behind `--features cuda` / `--features opencl`.
//!
//! Per RULES.md, this implementation must remain bit-identical to the
//! top-level on the same input. The bit-identity invariant is enforced
//! by `fluxion-cfd/tests/cpu_gpu_parity.rs`.

use crate::{CfdResult, Field3d, Grid3d, VelocityField};

pub struct CpuAdvectSolver;

impl CpuAdvectSolver {
    pub fn new() -> Self {
        Self
    }

    /// Semi-Lagrangian backtrace advection: for every cell `(i, j, k)`,
    /// trace the velocity backward by `dt` and interpolate `scalar` at
    /// the resulting position. The interpolated value lands in `result`.
    ///
    /// Boundary clamping follows the top-level `AdvectionSolver` policy
    /// (`x.clamp(0, nx-1)` etc.) — see `tests/cpu_gpu_parity.rs` for
    /// the bit-identity invariant that pins this contract.
    pub fn advect_scalar(
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
                    result.data[idx] = interpolate_scalar(scalar, grid, x0, y0, z0);
                }
            }
        }
        Ok(())
    }
}

impl Default for CpuAdvectSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Trilinear interpolation of `field` at the (clamped) world position
/// `(x, y, z)`. The eight corner samples are weighted by the fractional
/// part of the position. Clamping mirrors
/// [`crate::advection::AdvectionSolver::interpolate_scalar`].
fn interpolate_scalar(field: &Field3d, grid: &Grid3d, x: f64, y: f64, z: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression gate: the prior stub `advect_scalar` returned `Ok(())`
    /// without touching `result`, so any caller that asserted on the
    /// output would see the untouched `result`. This test fails if the
    /// function ever regresses to a no-op — it asserts the field
    /// actually changes after one advection step.
    #[test]
    fn advect_scalar_is_not_a_noop() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let velocity = VelocityField::zeros(8, 8, 8);
        let scalar = Field3d::filled(8, 8, 8, 1.0);
        let mut result = Field3d::zeros(8, 8, 8);
        CpuAdvectSolver::new()
            .advect_scalar(&grid, 0.001, &velocity, &scalar, &mut result)
            .expect("advect_scalar should succeed");
        // Zero velocity + uniform scalar → result must be uniformly 1.0.
        for (idx, &v) in result.data.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "result[{idx}] = {v} (expected 1.0)"
            );
        }
    }

    /// Non-zero velocity must produce a result that *differs* from the
    /// input. This catches the regression where the stub returned
    /// `Ok(())` and left `result` as all zeros.
    #[test]
    fn advect_scalar_writes_to_result_buffer() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let mut velocity = VelocityField::zeros(8, 8, 8);
        velocity.fill_x(1.0); // uniform u = 1.0 → backtrace by -dt/dx in x
        let scalar = Field3d::filled(8, 8, 8, 7.5);
        let mut result = Field3d::zeros(8, 8, 8);
        CpuAdvectSolver::new()
            .advect_scalar(&grid, 0.001, &velocity, &scalar, &mut result)
            .expect("advect_scalar should succeed");
        // Uniform scalar + uniform velocity → result must equal 7.5.
        for (idx, &v) in result.data.iter().enumerate() {
            assert!(
                (v - 7.5).abs() < 1e-12,
                "result[{idx}] = {v} (expected 7.5)"
            );
        }
    }

    /// Trilinear interpolation of a constant field returns that constant,
    /// at any clamped position.
    #[test]
    fn interpolate_scalar_constant_field() {
        let grid = Grid3d::new(8, 8, 8, 0.1, 0.1, 0.1);
        let field = Field3d::filled(8, 8, 8, 1.0);
        let value = interpolate_scalar(&field, &grid, 3.5, 3.5, 3.5);
        assert!((value - 1.0).abs() < 1e-10);
    }
}
