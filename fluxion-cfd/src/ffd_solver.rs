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
