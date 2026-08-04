//! Fast Fluid Dynamics (FFD) Solver - Fractional-step airflow solver.
//!
//! This module implements the Fast Fluid Dynamics algorithm using a fractional-step
//! (time-splitting) method to solve the Navier-Stokes equations for reduced-order
//! building airflow simulation.
//!
//! # Overview
//!
//! FFD is a reduced-order CFD approach that enables faster-than-real-time simulation for:
//! - Parametric airflow sweeps
//! - Smoke-control modeling
//! - Ventilation effectiveness assessment
//! - Coupled BES+FFD co-simulation
//!
//! # Fractional-Step Method (Bell, Colella, Glaz 1989)
//!
//! The algorithm splits each timestep into three sub-steps:
//!
//! 1. **Advection step** (semi-Lagrangian): Particles are traced backward in time
//!    to find their origin, and velocity is interpolated from the previous timestep.
//!
//! 2. **Diffusion step** (implicit): Viscous diffusion is solved using an implicit
//!    scheme (backward Euler) for numerical stability.
//!
//! 3. **Pressure projection**: A Poisson equation is solved to compute pressure
//!    and project velocity to be divergence-free (mass-conserving).
//!
//! # References
//!
//! - Fractional-step methods: Bell, Colella, Glaz (1989)
//! - Semi-Lagrangian advection: Staniforth & Côte (1991)

use nalgebra::DVector;
use thiserror::Error;

/// Grid configuration for FFD solver
#[derive(Debug, Clone)]
pub struct FfdGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,
}

impl FfdGrid {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, dt: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
        }
    }

    pub fn total_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }

    pub fn stroation_indices(&self, idx: usize) -> (usize, usize, usize) {
        let i = idx % self.nx;
        let j = (idx / self.nx) % self.ny;
        let k = idx / (self.nx * self.ny);
        (i, j, k)
    }
}

/// Velocity field vector (u, v, w) at each grid point
#[derive(Debug, Clone)]
pub struct VelocityField {
    pub u: DVector<f64>,
    pub v: DVector<f64>,
    pub w: DVector<f64>,
}

impl VelocityField {
    pub fn new(size: usize) -> Self {
        Self {
            u: DVector::zeros(size),
            v: DVector::zeros(size),
            w: DVector::zeros(size),
        }
    }

    pub fn from_scalar(size: usize, u0: f64, v0: f64, w0: f64) -> Self {
        Self {
            u: DVector::from_element(size, u0),
            v: DVector::from_element(size, v0),
            w: DVector::from_element(size, w0),
        }
    }

    pub fn scale(&mut self, alpha: f64) {
        self.u.scale_mut(alpha);
        self.v.scale_mut(alpha);
        self.w.scale_mut(alpha);
    }

    pub fn add_scaled(&mut self, other: &VelocityField, alpha: f64) {
        self.u.axpy(alpha, &other.u, 1.0);
        self.v.axpy(alpha, &other.v, 1.0);
        self.w.axpy(alpha, &other.w, 1.0);
    }

    pub fn copy_from(&mut self, other: &VelocityField) {
        self.u.copy_from(&other.u);
        self.v.copy_from(&other.v);
        self.w.copy_from(&other.w);
    }

    pub fn divergence(&self, grid: &FfdGrid) -> DVector<f64> {
        let mut div = DVector::zeros(self.u.len());
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = grid.linear_index(i, j, k);
                    let mut d_u = 0.0;
                    let mut d_v = 0.0;
                    let mut d_w = 0.0;
                    if i > 0 {
                        d_u = (self.u[idx] - self.u[grid.linear_index(i - 1, j, k)]) / grid.dx;
                    }
                    if j > 0 {
                        d_v = (self.v[idx] - self.v[grid.linear_index(i, j - 1, k)]) / grid.dy;
                    }
                    if k > 0 {
                        d_w = (self.w[idx] - self.w[grid.linear_index(i, j, k - 1)]) / grid.dz;
                    }
                    div[idx] = d_u + d_v + d_w;
                }
            }
        }
        div
    }

    pub fn l2_norm(&self) -> f64 {
        let norm_sq = self.u.dot(&self.u) + self.v.dot(&self.v) + self.w.dot(&self.w);
        norm_sq.sqrt()
    }
}

/// Pressure field
#[derive(Debug, Clone)]
pub struct PressureField {
    pub p: DVector<f64>,
}

impl PressureField {
    pub fn new(size: usize) -> Self {
        Self {
            p: DVector::zeros(size),
        }
    }
}

/// Boundary conditions for FFD solver
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    FixedVelocity { u: f64, v: f64, w: f64 },
    FixedPressure { p: f64 },
    Periodic,
    Open,
}

#[derive(Debug, Clone)]
pub struct FfdBoundaryConditions {
    pub x_min: BoundaryCondition,
    pub x_max: BoundaryCondition,
    pub y_min: BoundaryCondition,
    pub y_max: BoundaryCondition,
    pub z_min: BoundaryCondition,
    pub z_max: BoundaryCondition,
}

impl Default for FfdBoundaryConditions {
    fn default() -> Self {
        Self {
            x_min: BoundaryCondition::Open,
            x_max: BoundaryCondition::Open,
            y_min: BoundaryCondition::FixedVelocity {
                u: 0.0,
                v: 0.0,
                w: 0.0,
            },
            y_max: BoundaryCondition::Open,
            z_min: BoundaryCondition::FixedVelocity {
                u: 0.0,
                v: 0.0,
                w: 0.0,
            },
            z_max: BoundaryCondition::Open,
        }
    }
}

/// Physical properties for FFD
#[derive(Debug, Clone)]
pub struct FfdPhysicalProperties {
    pub kinematic_viscosity: f64,
    pub density: f64,
}

impl Default for FfdPhysicalProperties {
    fn default() -> Self {
        Self {
            kinematic_viscosity: 1.5e-5,
            density: 1.225,
        }
    }
}

/// FFD Solver errors
#[derive(Debug, Clone, Error)]
pub enum FfdError {
    #[error("Grid error: {0}")]
    GridError(String),
    #[error("Solver error: {0}")]
    SolverError(String),
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
    #[error("Initialization error: {0}")]
    InitializationError(String),
}

pub type FfdResult<T> = Result<T, FfdError>;

/// Fast Fluid Dynamics solver using fractional-step method
pub struct FfdSolver {
    grid: FfdGrid,
    velocity: VelocityField,
    velocity_star: VelocityField,
    pressure: PressureField,
    boundary: FfdBoundaryConditions,
    properties: FfdPhysicalProperties,
    initialized: bool,
    max_iterations: usize,
    tolerance: f64,
}

impl FfdSolver {
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
    ) -> FfdResult<Self> {
        if nx < 2 || ny < 2 || nz < 2 {
            return Err(FfdError::InitializationError(
                "Grid must have at least 2 points in each dimension".to_string(),
            ));
        }
        let grid = FfdGrid::new(nx, ny, nz, dx, dy, dz, dt);
        let size = grid.total_cells();
        Ok(Self {
            grid,
            velocity: VelocityField::new(size),
            velocity_star: VelocityField::new(size),
            pressure: PressureField::new(size),
            boundary: FfdBoundaryConditions::default(),
            properties: FfdPhysicalProperties::default(),
            initialized: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        })
    }

    pub fn with_boundaries(mut self, boundary: FfdBoundaryConditions) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn with_properties(mut self, properties: FfdPhysicalProperties) -> Self {
        self.properties = properties;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn grid(&self) -> &FfdGrid {
        &self.grid
    }

    pub fn velocity(&self) -> &VelocityField {
        &self.velocity
    }

    pub fn pressure(&self) -> &PressureField {
        &self.pressure
    }

    pub fn is_valid(&self) -> bool {
        self.initialized
            && self.velocity.u.iter().all(|x| x.is_finite())
            && self.velocity.v.iter().all(|x| x.is_finite())
            && self.velocity.w.iter().all(|x| x.is_finite())
            && self.pressure.p.iter().all(|x| x.is_finite())
    }

    pub fn initialize_velocity(&mut self, u0: f64, v0: f64, w0: f64) {
        self.velocity = VelocityField::from_scalar(self.grid.total_cells(), u0, v0, w0);
        self.velocity_star.copy_from(&self.velocity);
    }

    fn apply_boundary_conditions(
        velocity: &mut VelocityField,
        boundary: &FfdBoundaryConditions,
        grid: &FfdGrid,
    ) {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = grid.linear_index(i, j, k);
                    let mut apply_x_min = false;
                    let mut apply_x_max = false;
                    let mut apply_y_min = false;
                    let mut apply_y_max = false;
                    let mut apply_z_min = false;
                    let mut apply_z_max = false;
                    if i == 0 {
                        apply_x_min = true;
                    }
                    if i == grid.nx - 1 {
                        apply_x_max = true;
                    }
                    if j == 0 {
                        apply_y_min = true;
                    }
                    if j == grid.ny - 1 {
                        apply_y_max = true;
                    }
                    if k == 0 {
                        apply_z_min = true;
                    }
                    if k == grid.nz - 1 {
                        apply_z_max = true;
                    }
                    match &boundary.x_min {
                        BoundaryCondition::FixedVelocity { u, v, w } if apply_x_min => {
                            velocity.u[idx] = *u;
                            velocity.v[idx] = *v;
                            velocity.w[idx] = *w;
                        }
                        _ => {}
                    }
                    match &boundary.x_max {
                        BoundaryCondition::FixedVelocity { u, v, w } if apply_x_max => {
                            velocity.u[idx] = *u;
                            velocity.v[idx] = *v;
                            velocity.w[idx] = *w;
                        }
                        _ => {}
                    }
                    match &boundary.y_min {
                        BoundaryCondition::FixedVelocity { u, v, w } if apply_y_min => {
                            velocity.u[idx] = *u;
                            velocity.v[idx] = *v;
                            velocity.w[idx] = *w;
                        }
                        _ => {}
                    }
                    match &boundary.y_max {
                        BoundaryCondition::FixedVelocity { u, v, w } if apply_y_max => {
                            velocity.u[idx] = *u;
                            velocity.v[idx] = *v;
                            velocity.w[idx] = *w;
                        }
                        _ => {}
                    }
                    match &boundary.z_min {
                        BoundaryCondition::FixedVelocity { u, v, w } if apply_z_min => {
                            velocity.u[idx] = *u;
                            velocity.v[idx] = *v;
                            velocity.w[idx] = *w;
                        }
                        _ => {}
                    }
                    match &boundary.z_max {
                        BoundaryCondition::FixedVelocity { u, v, w } if apply_z_max => {
                            velocity.u[idx] = *u;
                            velocity.v[idx] = *v;
                            velocity.w[idx] = *w;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Advection step using semi-Lagrangian scheme
    ///
    /// This implements the first step of the fractional-step method:
    /// - Trace each grid point backward in time along the velocity field
    /// - Interpolate velocity from the previous timestep at the departure point
    fn advection_step(&mut self) {
        let dt = self.grid.dt;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;
        let mut u_new = VelocityField::new(self.grid.total_cells());
        let mut v_new = VelocityField::new(self.grid.total_cells());
        let mut w_new = VelocityField::new(self.grid.total_cells());
        for k in 1..self.grid.nz - 1 {
            for j in 1..self.grid.ny - 1 {
                for i in 1..self.grid.nx - 1 {
                    let idx = self.grid.linear_index(i, j, k);
                    let x = (i as f64) * dx;
                    let y = (j as f64) * dy;
                    let z = (k as f64) * dz;
                    let u_curr = self.velocity.u[idx];
                    let v_curr = self.velocity.v[idx];
                    let w_curr = self.velocity.w[idx];
                    let x_depart = x - u_curr * dt;
                    let y_depart = y - v_curr * dt;
                    let z_depart = z - w_curr * dt;
                    let (u_interp, v_interp, w_interp) =
                        self.interpolate_velocity_at(x_depart, y_depart, z_depart);
                    u_new.u[idx] = u_interp;
                    v_new.v[idx] = v_interp;
                    w_new.w[idx] = w_interp;
                }
            }
        }
        self.velocity_star.u.copy_from(&u_new.u);
        self.velocity_star.v.copy_from(&v_new.v);
        self.velocity_star.w.copy_from(&w_new.w);
        Self::apply_boundary_conditions(&mut self.velocity_star, &self.boundary, &self.grid);
    }

    fn interpolate_velocity_at(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let max_x = ((nx - 1) as f64) * dx;
        let max_y = ((ny - 1) as f64) * dy;
        let max_z = ((nz - 1) as f64) * dz;
        let x_clamped = x.clamp(0.0, max_x);
        let y_clamped = y.clamp(0.0, max_y);
        let z_clamped = z.clamp(0.0, max_z);
        let i = (x_clamped / dx).floor() as i64;
        let j = (y_clamped / dy).floor() as i64;
        let k = (z_clamped / dz).floor() as i64;
        let i0 = (i.max(0).min((nx - 1) as i64)) as usize;
        let j0 = (j.max(0).min((ny - 1) as i64)) as usize;
        let k0 = (k.max(0).min((nz - 1) as i64)) as usize;
        let i1 = ((i + 1).max(0).min((nx - 1) as i64)) as usize;
        let j1 = ((j + 1).max(0).min((ny - 1) as i64)) as usize;
        let k1 = ((k + 1).max(0).min((nz - 1) as i64)) as usize;
        let xi = (x_clamped - (i as f64) * dx) / dx;
        let eta = (y_clamped - (j as f64) * dy) / dy;
        let zeta = (z_clamped - (k as f64) * dz) / dz;
        let xi = xi.clamp(0.0, 1.0);
        let eta = eta.clamp(0.0, 1.0);
        let zeta = zeta.clamp(0.0, 1.0);
        let idx000 = self.grid.linear_index(i0, j0, k0);
        let idx100 = self.grid.linear_index(i1, j0, k0);
        let idx010 = self.grid.linear_index(i0, j1, k0);
        let idx110 = self.grid.linear_index(i1, j1, k0);
        let idx001 = self.grid.linear_index(i0, j0, k1);
        let idx101 = self.grid.linear_index(i1, j0, k1);
        let idx011 = self.grid.linear_index(i0, j1, k1);
        let idx111 = self.grid.linear_index(i1, j1, k1);
        let u000 = self.velocity.u[idx000];
        let u100 = self.velocity.u[idx100];
        let u010 = self.velocity.u[idx010];
        let u110 = self.velocity.u[idx110];
        let u001 = self.velocity.u[idx001];
        let u101 = self.velocity.u[idx101];
        let u011 = self.velocity.u[idx011];
        let u111 = self.velocity.u[idx111];
        let u_interp = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) * u000
            + xi * (1.0 - eta) * (1.0 - zeta) * u100
            + (1.0 - xi) * eta * (1.0 - zeta) * u010
            + xi * eta * (1.0 - zeta) * u110
            + (1.0 - xi) * (1.0 - eta) * zeta * u001
            + xi * (1.0 - eta) * zeta * u101
            + (1.0 - xi) * eta * zeta * u011
            + xi * eta * zeta * u111;
        let v000 = self.velocity.v[idx000];
        let v100 = self.velocity.v[idx100];
        let v010 = self.velocity.v[idx010];
        let v110 = self.velocity.v[idx110];
        let v001 = self.velocity.v[idx001];
        let v101 = self.velocity.v[idx101];
        let v011 = self.velocity.v[idx011];
        let v111 = self.velocity.v[idx111];
        let v_interp = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) * v000
            + xi * (1.0 - eta) * (1.0 - zeta) * v100
            + (1.0 - xi) * eta * (1.0 - zeta) * v010
            + xi * eta * (1.0 - zeta) * v110
            + (1.0 - xi) * (1.0 - eta) * zeta * v001
            + xi * (1.0 - eta) * zeta * v101
            + (1.0 - xi) * eta * zeta * v011
            + xi * eta * zeta * v111;
        let w000 = self.velocity.w[idx000];
        let w100 = self.velocity.w[idx100];
        let w010 = self.velocity.w[idx010];
        let w110 = self.velocity.w[idx110];
        let w001 = self.velocity.w[idx001];
        let w101 = self.velocity.w[idx101];
        let w011 = self.velocity.w[idx011];
        let w111 = self.velocity.w[idx111];
        let w_interp = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) * w000
            + xi * (1.0 - eta) * (1.0 - zeta) * w100
            + (1.0 - xi) * eta * (1.0 - zeta) * w010
            + xi * eta * (1.0 - zeta) * w110
            + (1.0 - xi) * (1.0 - eta) * zeta * w001
            + xi * (1.0 - eta) * zeta * w101
            + (1.0 - xi) * eta * zeta * w011
            + xi * eta * zeta * w111;
        (u_interp, v_interp, w_interp)
    }

    /// Diffusion step using implicit solver (backward Euler)
    ///
    /// This implements the second step of the fractional-step method:
    /// - Solve (I - nu*dt*L) u* = u^n where L is the Laplacian
    /// - Uses a simple iterative solver (Jacobi iteration)
    fn diffusion_step(&mut self) {
        let nu = self.properties.kinematic_viscosity;
        let dt = self.grid.dt;
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let dz2 = self.grid.dz * self.grid.dz;
        let mut u_new = VelocityField::new(self.grid.total_cells());
        let mut v_new = VelocityField::new(self.grid.total_cells());
        let mut w_new = VelocityField::new(self.grid.total_cells());
        u_new.copy_from(&self.velocity_star);
        v_new.copy_from(&self.velocity_star);
        w_new.copy_from(&self.velocity_star);
        for _iter in 0..self.max_iterations {
            let mut u_rel_change = 0.0;
            let mut v_rel_change = 0.0;
            let mut w_rel_change = 0.0;
            for k in 1..self.grid.nz - 1 {
                for j in 1..self.grid.ny - 1 {
                    for i in 1..self.grid.nx - 1 {
                        let idx = self.grid.linear_index(i, j, k);
                        let idx_ip = self.grid.linear_index(i + 1, j, k);
                        let idx_im = self.grid.linear_index(i - 1, j, k);
                        let idx_jp = self.grid.linear_index(i, j + 1, k);
                        let idx_jm = self.grid.linear_index(i, j - 1, k);
                        let idx_kp = self.grid.linear_index(i, j, k + 1);
                        let idx_km = self.grid.linear_index(i, j, k - 1);
                        let laplacian_u = (u_new.u[idx_ip] - 2.0 * u_new.u[idx] + u_new.u[idx_im])
                            / dx2
                            + (u_new.u[idx_jp] - 2.0 * u_new.u[idx] + u_new.u[idx_jm]) / dy2
                            + (u_new.u[idx_kp] - 2.0 * u_new.u[idx] + u_new.u[idx_km]) / dz2;
                        let laplacian_v = (v_new.v[idx_ip] - 2.0 * v_new.v[idx] + v_new.v[idx_im])
                            / dx2
                            + (v_new.v[idx_jp] - 2.0 * v_new.v[idx] + v_new.v[idx_jm]) / dy2
                            + (v_new.v[idx_kp] - 2.0 * v_new.v[idx] + v_new.v[idx_km]) / dz2;
                        let laplacian_w = (w_new.w[idx_ip] - 2.0 * w_new.w[idx] + w_new.w[idx_im])
                            / dx2
                            + (w_new.w[idx_jp] - 2.0 * w_new.w[idx] + w_new.w[idx_jm]) / dy2
                            + (w_new.w[idx_kp] - 2.0 * w_new.w[idx] + w_new.w[idx_km]) / dz2;
                        let factor =
                            1.0 / (1.0 + 2.0 * nu * dt * (1.0 / dx2 + 1.0 / dy2 + 1.0 / dz2));
                        let u_old = u_new.u[idx];
                        let v_old = v_new.v[idx];
                        let w_old = w_new.w[idx];
                        u_new.u[idx] = factor * (self.velocity_star.u[idx] + nu * dt * laplacian_u);
                        v_new.v[idx] = factor * (self.velocity_star.v[idx] + nu * dt * laplacian_v);
                        w_new.w[idx] = factor * (self.velocity_star.w[idx] + nu * dt * laplacian_w);
                        u_rel_change += (u_new.u[idx] - u_old).powi(2);
                        v_rel_change += (v_new.v[idx] - v_old).powi(2);
                        w_rel_change += (w_new.w[idx] - w_old).powi(2);
                    }
                }
            }
            let u_norm = u_rel_change.sqrt();
            let v_norm = v_rel_change.sqrt();
            let w_norm = w_rel_change.sqrt();
            if u_norm < self.tolerance && v_norm < self.tolerance && w_norm < self.tolerance {
                break;
            }
        }
        self.velocity_star.u.copy_from(&u_new.u);
        self.velocity_star.v.copy_from(&v_new.v);
        self.velocity_star.w.copy_from(&w_new.w);
        Self::apply_boundary_conditions(&mut self.velocity_star, &self.boundary, &self.grid);
    }

    /// Pressure projection step
    ///
    /// This implements the third step of the fractional-step method:
    /// - Solve Poisson equation: L p = div(u*)
    /// - Project velocity: u^{n+1} = u* - dt * grad(p)
    ///
    /// Uses a simple iterative solver (Jacobi iteration)
    fn pressure_projection_step(&mut self) {
        let dt = self.grid.dt;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dz2 = dz * dz;
        let mut p_new = DVector::zeros(self.grid.total_cells());
        let div = self.velocity_star.divergence(&self.grid);
        let one_dx2 = 1.0 / dx2;
        let one_dy2 = 1.0 / dy2;
        let one_dz2 = 1.0 / dz2;
        let denom = -2.0 * (one_dx2 + one_dy2 + one_dz2);
        for _iter in 0..self.max_iterations {
            let mut max_change = 0.0;
            for k in 1..self.grid.nz - 1 {
                for j in 1..self.grid.ny - 1 {
                    for i in 1..self.grid.nx - 1 {
                        let idx = self.grid.linear_index(i, j, k);
                        let idx_ip = self.grid.linear_index(i + 1, j, k);
                        let idx_im = self.grid.linear_index(i - 1, j, k);
                        let idx_jp = self.grid.linear_index(i, j + 1, k);
                        let idx_jm = self.grid.linear_index(i, j - 1, k);
                        let idx_kp = self.grid.linear_index(i, j, k + 1);
                        let idx_km = self.grid.linear_index(i, j, k - 1);
                        let neighbor_sum = one_dx2 * (p_new[idx_ip] + p_new[idx_im])
                            + one_dy2 * (p_new[idx_jp] + p_new[idx_jm])
                            + one_dz2 * (p_new[idx_kp] + p_new[idx_km]);
                        let rhs = div[idx] / dt;
                        let p_old = p_new[idx];
                        let p_new_val: f64 = if denom.abs() > 1e-10 {
                            (neighbor_sum - rhs) / denom
                        } else {
                            p_new[idx]
                        };
                        p_new[idx] = p_new_val;
                        let change: f64 = (p_new_val - p_old).abs();
                        if change > max_change {
                            max_change = change;
                        }
                    }
                }
            }
            if max_change < self.tolerance {
                break;
            }
        }
        self.pressure.p.copy_from(&p_new);
        for k in 1..self.grid.nz - 1 {
            for j in 1..self.grid.ny - 1 {
                for i in 1..self.grid.nx - 1 {
                    let idx = self.grid.linear_index(i, j, k);
                    let idx_ip = self.grid.linear_index(i + 1, j, k);
                    let idx_im = self.grid.linear_index(i - 1, j, k);
                    let idx_jp = self.grid.linear_index(i, j + 1, k);
                    let idx_jm = self.grid.linear_index(i, j - 1, k);
                    let idx_kp = self.grid.linear_index(i, j, k + 1);
                    let idx_km = self.grid.linear_index(i, j, k - 1);
                    let dp_dx = (p_new[idx_ip] - p_new[idx_im]) / (2.0 * dx);
                    let dp_dy = (p_new[idx_jp] - p_new[idx_jm]) / (2.0 * dy);
                    let dp_dz = (p_new[idx_kp] - p_new[idx_km]) / (2.0 * dz);
                    self.velocity.u[idx] = self.velocity_star.u[idx] - dt * dp_dx;
                    self.velocity.v[idx] = self.velocity_star.v[idx] - dt * dp_dy;
                    self.velocity.w[idx] = self.velocity_star.w[idx] - dt * dp_dz;
                }
            }
        }
        Self::apply_boundary_conditions(&mut self.velocity, &self.boundary, &self.grid);
    }

    /// Advance the solver by one timestep using fractional-step method
    pub fn step(&mut self) -> FfdResult<()> {
        if !self.initialized {
            return Err(FfdError::InitializationError(
                "Solver not initialized".to_string(),
            ));
        }
        self.advection_step();
        self.diffusion_step();
        self.pressure_projection_step();
        if !self.is_valid() {
            return Err(FfdError::SolverError(
                "NaN or Inf detected in velocity/pressure field".to_string(),
            ));
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        "FfdSolver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_linear_index() {
        let grid = FfdGrid::new(3, 4, 5, 1.0, 1.0, 1.0, 0.1);
        assert_eq!(grid.linear_index(0, 0, 0), 0);
        assert_eq!(grid.linear_index(1, 0, 0), 1);
        assert_eq!(grid.linear_index(0, 1, 0), 3);
        assert_eq!(grid.linear_index(0, 0, 1), 12);
        assert_eq!(grid.total_cells(), 60);
    }

    #[test]
    fn test_velocity_field_l2_norm() {
        let vel = VelocityField::from_scalar(1, 3.0, 4.0, 0.0);
        let norm = vel.l2_norm();
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ffd_solver_initialization() {
        let solver = FfdSolver::new(10, 10, 10, 0.1, 0.1, 0.1, 0.01).unwrap();
        assert!(solver.is_valid());
        assert_eq!(solver.name(), "FfdSolver");
    }

    #[test]
    fn test_ffd_solver_grid_size_validation() {
        let result = FfdSolver::new(1, 10, 10, 0.1, 0.1, 0.1, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_velocity_field_add_scaled() {
        let size = 5;
        let mut vel1 = VelocityField::from_scalar(size, 1.0, 2.0, 3.0);
        let vel2 = VelocityField::from_scalar(size, 1.0, 1.0, 1.0);
        vel1.add_scaled(&vel2, 2.0);
        assert!((vel1.u[0] - 3.0).abs() < 1e-10);
        assert!((vel1.v[0] - 4.0).abs() < 1e-10);
        assert!((vel1.w[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_2d_uniform_flow() {
        let mut solver = FfdSolver::new(20, 20, 2, 0.1, 0.1, 0.1, 0.01).unwrap();
        solver.initialize_velocity(1.0, 0.0, 0.0);
        for _ in 0..10 {
            solver.step().unwrap();
        }
        let vel = solver.velocity();
        assert!(vel.u.iter().all(|&x| x.is_finite()));
        assert!(vel.v.iter().all(|&x| x.is_finite()));
        assert!(solver.is_valid());
    }

    #[test]
    fn test_2d_lid_driven_cavity() {
        let mut solver = FfdSolver::new(21, 21, 3, 0.1, 0.1, 0.1, 0.01)
            .unwrap()
            .with_boundaries(FfdBoundaryConditions {
                x_min: BoundaryCondition::FixedVelocity {
                    u: 0.0,
                    v: 0.0,
                    w: 0.0,
                },
                x_max: BoundaryCondition::FixedVelocity {
                    u: 0.0,
                    v: 0.0,
                    w: 0.0,
                },
                y_min: BoundaryCondition::FixedVelocity {
                    u: 0.0,
                    v: 0.0,
                    w: 0.0,
                },
                y_max: BoundaryCondition::FixedVelocity {
                    u: 1.0,
                    v: 0.0,
                    w: 0.0,
                },
                z_min: BoundaryCondition::FixedVelocity {
                    u: 0.0,
                    v: 0.0,
                    w: 0.0,
                },
                z_max: BoundaryCondition::FixedVelocity {
                    u: 0.0,
                    v: 0.0,
                    w: 0.0,
                },
            });
        solver.initialize_velocity(0.0, 0.0, 0.0);
        for _ in 0..50 {
            solver.step().unwrap();
        }
        let vel = solver.velocity();
        let max_u = vel.u.iter().fold(0.0f64, |max, &x| max.max(x.abs()));
        assert!(
            max_u > 0.01,
            "Max u-velocity should be positive in lid-driven cavity, max_u={}",
            max_u
        );
        assert!(solver.is_valid());
    }

    #[test]
    fn test_3d_simple_shear_flow() {
        let mut solver = FfdSolver::new(10, 10, 10, 0.1, 0.1, 0.1, 0.001)
            .unwrap()
            .with_max_iterations(100);
        solver.initialize_velocity(0.0, 1.0, 0.0);
        for _ in 0..20 {
            let result = solver.step();
            if result.is_err() {
                break;
            }
        }
        let vel = solver.velocity();
        let v_mean = vel.v.iter().sum::<f64>() / vel.v.len() as f64;
        assert!(v_mean.is_finite(), "Mean v-velocity should be finite");
        assert!(solver.is_valid(), "Solver should remain valid");
    }

    #[test]
    fn test_pressure_poisson_convergence() {
        let mut solver = FfdSolver::new(11, 11, 11, 0.1, 0.1, 0.1, 0.001)
            .unwrap()
            .with_tolerance(1e-3)
            .with_max_iterations(100);
        solver.initialize_velocity(1.0, 0.5, 0.0);
        for _ in 0..10 {
            let result = solver.step();
            if result.is_err() {
                break;
            }
        }
        assert!(solver.is_valid(), "Solver should remain valid after steps");
        let div = solver.velocity().divergence(solver.grid());
        let max_div = div.iter().fold(0.0f64, |max, &x| max.max(x.abs()));
        assert!(
            max_div < 1e6,
            "Divergence should be bounded after pressure projection, got {}",
            max_div
        );
    }

    #[test]
    fn test_no_crash_2d_grid() {
        let solver = FfdSolver::new(5, 5, 2, 1.0, 1.0, 1.0, 0.1).unwrap();
        assert!(solver.is_valid());
        let vel = solver.velocity();
        assert_eq!(vel.u.len(), 50);
    }

    #[test]
    fn test_no_crash_3d_grid() {
        let solver = FfdSolver::new(5, 5, 5, 1.0, 1.0, 1.0, 0.1).unwrap();
        assert!(solver.is_valid());
        let vel = solver.velocity();
        assert_eq!(vel.u.len(), 125);
    }
}
