//! Fast Fluid Dynamics solver core types

use crate::CfdResult;

#[derive(Debug, Clone)]
pub struct FfdConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,
    pub nu: f64,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Default for FfdConfig {
    fn default() -> Self {
        Self {
            nx: 32,
            ny: 32,
            nz: 32,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            dt: 0.001,
            nu: 1.0e-5,
            max_iter: 1000,
            tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Grid3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub volume: f64,
}

impl Grid3d {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        let volume = (nx as f64) * dx * (ny as f64) * dy * (nz as f64) * dz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            volume,
        }
    }

    pub fn num_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return None;
        }
        Some(i + self.nx * (j + self.ny * k))
    }

    pub fn validate(&self) -> CfdResult<()> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(crate::CfdError::Grid(
                "Grid dimensions must be positive".into(),
            ));
        }
        if self.dx <= 0.0 || self.dy <= 0.0 || self.dz <= 0.0 {
            return Err(crate::CfdError::Grid(
                "Grid spacing must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Field3d {
    pub data: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Field3d {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            data: vec![0.0; nx * ny * nz],
            nx,
            ny,
            nz,
        }
    }

    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        Self::new(nx, ny, nz)
    }

    pub fn filled(nx: usize, ny: usize, nz: usize, value: f64) -> Self {
        Self {
            data: vec![value; nx * ny * nz],
            nx,
            ny,
            nz,
        }
    }

    pub fn num_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn index(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return None;
        }
        Some(i + self.nx * (j + self.ny * k))
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> Option<f64> {
        self.index(i, j, k).map(|idx| self.data[idx])
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f64) -> Option<()> {
        self.index(i, j, k).map(|idx| self.data[idx] = value)
    }

    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    pub fn scale(&mut self, alpha: f64) {
        for v in &mut self.data {
            *v *= alpha;
        }
    }

    pub fn add(&mut self, other: &Field3d) -> CfdResult<()> {
        if self.num_cells() != other.num_cells() {
            return Err(crate::CfdError::Solver("Field dimensions mismatch".into()));
        }
        for (i, v) in self.data.iter_mut().enumerate() {
            *v += other.data[i];
        }
        Ok(())
    }

    pub fn copy_from(&mut self, other: &Field3d) -> CfdResult<()> {
        if self.num_cells() != other.num_cells() {
            return Err(crate::CfdError::Solver("Field dimensions mismatch".into()));
        }
        self.data.copy_from_slice(&other.data);
        Ok(())
    }

    pub fn apply_boundary_pressure(&mut self, grid: &Grid3d) {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                if let Some(idx) = self.index(0, j, k) {
                    self.data[idx] = self.data[self.index(1, j, k).unwrap()];
                }
                if let Some(idx) = self.index(grid.nx - 1, j, k) {
                    self.data[idx] = self.data[self.index(grid.nx - 2, j, k).unwrap()];
                }
            }
        }
        for i in 0..grid.nx {
            for k in 0..grid.nz {
                if let Some(idx) = self.index(i, 0, k) {
                    self.data[idx] = self.data[self.index(i, 1, k).unwrap()];
                }
                if let Some(idx) = self.index(i, grid.ny - 1, k) {
                    self.data[idx] = self.data[self.index(i, grid.ny - 2, k).unwrap()];
                }
            }
        }
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                if let Some(idx) = self.index(i, j, 0) {
                    self.data[idx] = self.data[self.index(i, j, 1).unwrap()];
                }
                if let Some(idx) = self.index(i, j, grid.nz - 1) {
                    self.data[idx] = self.data[self.index(i, j, grid.nz - 2).unwrap()];
                }
            }
        }
    }

    pub fn apply_boundary_velocity(&mut self, grid: &Grid3d) {
        let _ = grid;
    }
}

#[derive(Debug, Clone)]
pub struct VelocityField {
    pub u: Field3d,
    pub v: Field3d,
    pub w: Field3d,
}

impl VelocityField {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            u: Field3d::new(nx, ny, nz),
            v: Field3d::new(nx, ny, nz),
            w: Field3d::new(nx, ny, nz),
        }
    }

    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            u: Field3d::zeros(nx, ny, nz),
            v: Field3d::zeros(nx, ny, nz),
            w: Field3d::zeros(nx, ny, nz),
        }
    }

    pub fn num_cells(&self) -> usize {
        self.u.num_cells()
    }

    pub fn fill(&mut self, value: f64) {
        self.u.fill(value);
        self.v.fill(value);
        self.w.fill(value);
    }

    /// Fill only the x-component (u) of the velocity field with a constant value.
    pub fn fill_x(&mut self, value: f64) {
        self.u.fill(value);
    }

    /// Fill only the y-component (v) of the velocity field with a constant value.
    pub fn fill_y(&mut self, value: f64) {
        self.v.fill(value);
    }

    /// Fill only the z-component (w) of the velocity field with a constant value.
    pub fn fill_z(&mut self, value: f64) {
        self.w.fill(value);
    }

    pub fn apply_boundary_velocity(
        &mut self,
        grid: &Grid3d,
        _u: &mut Field3d,
        _v: &mut Field3d,
        _w: &mut Field3d,
    ) {
        let _ = grid;
    }
}

pub struct FfdCfdSolver {
    config: FfdConfig,
    grid: Grid3d,
    advection: crate::AdvectionSolver,
    diffusion: crate::DiffusionSolver,
    pressure: crate::PressureSolver,
    velocity: VelocityField,
    pressure_field: Field3d,
}

impl FfdCfdSolver {
    pub fn new(config: FfdConfig) -> CfdResult<Self> {
        let nx = config.nx;
        let ny = config.ny;
        let nz = config.nz;
        let grid = Grid3d::new(nx, ny, nz, config.dx, config.dy, config.dz);
        grid.validate()?;
        Ok(Self {
            config,
            grid,
            advection: crate::AdvectionSolver::new(),
            diffusion: crate::DiffusionSolver::new(1000, 1e-8),
            pressure: crate::PressureSolver::new(1000, 1e-6),
            velocity: VelocityField::zeros(nx, ny, nz),
            pressure_field: Field3d::zeros(nx, ny, nz),
        })
    }

    pub fn step(&mut self, dt: f64) -> CfdResult<()> {
        self.advection.advect_velocity(
            &self.grid,
            dt,
            &self.velocity,
            &mut self.velocity.u.clone(),
            &mut self.velocity.v.clone(),
            &mut self.velocity.w.clone(),
        )?;
        self.diffusion
            .diffuse_velocity(&self.grid, dt, self.config.nu, &mut self.velocity)?;
        self.pressure.project(&self.grid, dt, &mut self.velocity)?;
        Ok(())
    }

    pub fn velocity(&self) -> &VelocityField {
        &self.velocity
    }

    pub fn pressure(&self) -> &Field3d {
        &self.pressure_field
    }

    /// Read-only access to the configuration used to construct this solver.
    pub fn config(&self) -> &FfdConfig {
        &self.config
    }

    /// Read-only access to the underlying computational grid.
    pub fn grid(&self) -> &Grid3d {
        &self.grid
    }

    /// Set the velocity field to a uniform value across the whole domain.
    ///
    /// This is used by the loose-coupling adapter (`fluxion::sim::ffd_cfd_adapter`)
    /// to translate `BesToFfdBoundaryConditions` (e.g. wind pressure, HVAC supply
    /// flow) into an inlet velocity field for the FFD solver. With the current
    /// FFD API, there is no separate "boundary" type, so the simplest faithful
    /// translation is to fill the domain with a uniform velocity before the
    /// first step and let advection/diffusion/pressure evolve it from there.
    pub fn fill_velocity(&mut self, u: f64, v: f64, w: f64) {
        self.velocity.fill_x(u);
        self.velocity.fill_y(v);
        self.velocity.fill_z(w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> FfdConfig {
        FfdConfig {
            nx: 4,
            ny: 4,
            nz: 4,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            dt: 0.001,
            nu: 1.0e-5,
            max_iter: 50,
            tolerance: 1.0e-6,
        }
    }

    #[test]
    fn ffd_config_default_is_reasonable() {
        let cfg = FfdConfig::default();
        assert!(cfg.nx > 0 && cfg.ny > 0 && cfg.nz > 0);
        assert!(cfg.dx > 0.0 && cfg.dy > 0.0 && cfg.dz > 0.0);
        assert!(cfg.dt > 0.0);
        assert!(cfg.nu > 0.0);
        assert!(cfg.tolerance > 0.0);
        assert!(cfg.max_iter > 0);
    }

    #[test]
    fn grid3d_volume_and_num_cells() {
        let g = Grid3d::new(2, 3, 4, 0.5, 0.5, 0.5);
        assert_eq!(g.num_cells(), 24);
        assert!((g.volume - 2.0 * 3.0 * 4.0 * 0.125).abs() < 1.0e-12);
    }

    #[test]
    fn grid3d_validation_rejects_zero_dim() {
        let g = Grid3d::new(0, 1, 1, 0.1, 0.1, 0.1);
        assert!(g.validate().is_err());
        let g = Grid3d::new(1, 0, 1, 0.1, 0.1, 0.1);
        assert!(g.validate().is_err());
        let g = Grid3d::new(1, 1, 0, 0.1, 0.1, 0.1);
        assert!(g.validate().is_err());
    }

    #[test]
    fn grid3d_validation_rejects_zero_spacing() {
        let g = Grid3d::new(1, 1, 1, 0.0, 0.1, 0.1);
        assert!(g.validate().is_err());
        let g = Grid3d::new(1, 1, 1, 0.1, 0.0, 0.1);
        assert!(g.validate().is_err());
        let g = Grid3d::new(1, 1, 1, 0.1, 0.1, 0.0);
        assert!(g.validate().is_err());
    }

    #[test]
    fn grid3d_validation_accepts_positive_grid() {
        let g = Grid3d::new(2, 2, 2, 0.1, 0.1, 0.1);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn grid3d_linear_index_and_bounds() {
        let g = Grid3d::new(4, 5, 6, 0.1, 0.1, 0.1);
        assert_eq!(g.linear_index(0, 0, 0), Some(0));
        assert_eq!(g.linear_index(3, 4, 5), Some(3 + 4 * (4 + 5 * 5)));
        assert_eq!(g.linear_index(4, 0, 0), None);
        assert_eq!(g.linear_index(0, 5, 0), None);
        assert_eq!(g.linear_index(0, 0, 6), None);
    }

    #[test]
    fn field3d_set_and_get_roundtrip() {
        let mut f = Field3d::zeros(4, 4, 4);
        assert_eq!(f.get(1, 2, 3), Some(0.0));
        f.set(1, 2, 3, 7.5).unwrap();
        assert_eq!(f.get(1, 2, 3), Some(7.5));
    }

    #[test]
    fn field3d_out_of_bounds_returns_none() {
        let mut f = Field3d::zeros(4, 4, 4);
        assert_eq!(f.set(4, 0, 0, 1.0), None);
        assert_eq!(f.set(0, 4, 0, 1.0), None);
        assert_eq!(f.set(0, 0, 4, 1.0), None);
        assert_eq!(f.get(4, 0, 0), None);
    }

    #[test]
    fn field3d_fill_and_scale() {
        let mut f = Field3d::zeros(3, 3, 3);
        f.fill(2.0);
        f.scale(3.0);
        for &v in f.data.iter() {
            assert!((v - 6.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn field3d_add_mismatched_dimensions_errors() {
        let mut a = Field3d::zeros(3, 3, 3);
        let b = Field3d::zeros(4, 4, 4);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn field3d_copy_from_mismatched_dimensions_errors() {
        let mut a = Field3d::zeros(3, 3, 3);
        let b = Field3d::zeros(4, 4, 4);
        assert!(a.copy_from(&b).is_err());
    }

    #[test]
    fn field3d_add_same_dimensions_succeeds() {
        let mut a = Field3d::filled(2, 2, 2, 1.0);
        let b = Field3d::filled(2, 2, 2, 2.0);
        a.add(&b).expect("add should succeed");
        for &v in a.data.iter() {
            assert!((v - 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn velocity_field_fill_components_are_independent() {
        let mut v = VelocityField::zeros(2, 2, 2);
        v.fill_x(1.0);
        v.fill_y(2.0);
        v.fill_z(3.0);
        for &u in v.u.data.iter() {
            assert!((u - 1.0).abs() < 1.0e-12);
        }
        for &vv in v.v.data.iter() {
            assert!((vv - 2.0).abs() < 1.0e-12);
        }
        for &w in v.w.data.iter() {
            assert!((w - 3.0).abs() < 1.0e-12);
        }
        v.fill(7.0);
        for &u in v.u.data.iter() {
            assert!((u - 7.0).abs() < 1.0e-12);
        }
        for &vv in v.v.data.iter() {
            assert!((vv - 7.0).abs() < 1.0e-12);
        }
        for &w in v.w.data.iter() {
            assert!((w - 7.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn ffd_solver_new_returns_ok_for_valid_config() {
        let s = FfdCfdSolver::new(small_config()).expect("new should succeed");
        assert_eq!(s.grid().nx, 4);
        assert_eq!(s.grid().ny, 4);
        assert_eq!(s.grid().nz, 4);
        assert_eq!(s.velocity().num_cells(), 4 * 4 * 4);
        assert_eq!(s.pressure().num_cells(), 4 * 4 * 4);
    }

    #[test]
    fn ffd_solver_new_rejects_zero_dimension() {
        let cfg = FfdConfig {
            nx: 0,
            ..small_config()
        };
        assert!(FfdCfdSolver::new(cfg).is_err());
    }

    #[test]
    fn ffd_solver_accessors_return_consistent_data() {
        let s = FfdCfdSolver::new(small_config()).expect("new should succeed");
        assert_eq!(s.config().nx, 4);
        assert_eq!(s.config().dt, 0.001);
        assert_eq!(s.grid().dx, 0.1);
        assert_eq!(s.velocity().u.nx, 4);
    }

    #[test]
    fn ffd_solver_fill_velocity_sets_components() {
        let mut s = FfdCfdSolver::new(small_config()).expect("new should succeed");
        s.fill_velocity(1.0, 2.0, 3.0);
        for &u in s.velocity().u.data.iter() {
            assert!((u - 1.0).abs() < 1.0e-12);
        }
        for &v in s.velocity().v.data.iter() {
            assert!((v - 2.0).abs() < 1.0e-12);
        }
        for &w in s.velocity().w.data.iter() {
            assert!((w - 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn ffd_solver_step_runs_on_small_grid() {
        let mut s = FfdCfdSolver::new(small_config()).expect("new should succeed");
        s.fill_velocity(1.0, 0.0, 0.0);
        s.step(0.001).expect("step should succeed");
        for &u in s.velocity().u.data.iter() {
            assert!(u.is_finite(), "u must remain finite after step (got {u})");
        }
        for &v in s.velocity().v.data.iter() {
            assert!(v.is_finite(), "v must remain finite after step (got {v})");
        }
        for &w in s.velocity().w.data.iter() {
            assert!(w.is_finite(), "w must remain finite after step (got {w})");
        }
    }
}
