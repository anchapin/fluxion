//! Implicit Finite Difference solver for 1D heat conduction.
//!
//! This module implements the Backward Time, Central Space (BTCS) scheme
//! for solving the heat equation with Robin boundary conditions.
//!
//! # Overview
//!
//! The `ImplicitFDSolver` assembles and solves the tridiagonal linear system
//! that arises from implicit discretization of the heat equation:
//!
//! ```text
//! -Fo·T_{i-1}^{n+1} + (1+2Fo)·T_i^{n+1} - Fo·T_{i+1}^{n+1} = T_i^n
//! ```
//!
//! where Fo = α·Δt/Δx² is the Fourier number.
//!
//! The system is solved efficiently using the Thomas algorithm (TDMA) in O(n) operations.
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::fd_discretization::{MaterialLayer, WallDiscretization};
//! use fluxion::physics::fd_solver::{ImplicitFDSolver, SurfaceBC};
//!
//! // Create discretization
//! let layers = vec![MaterialLayer::new("Concrete", 0.200, 1.4, 2300.0, 880.0)];
//! let disc = WallDiscretization::from_layers(&layers, 20);
//!
//! // Create solver
//! let mut solver = ImplicitFDSolver::new(disc, 20.0); // Initial T = 20°C
//!
//! // Define boundary conditions
//! let interior_bc = SurfaceBC::new_interior(8.0, 21.0); // h=8 W/m²K, T_zone=21°C
//! let exterior_bc = SurfaceBC::new_exterior(25.0, 5.0, 0.0); // h=25, T_out=5°C
//!
//! // Advance by one hour
//! solver.step(3600.0, &interior_bc, &exterior_bc);
//!
//! // Sample temperature at a node (returns Vec of all node temps via
//! // repeated `temperature_at`; there is no `temperatures()` bulk accessor).
//! let t_first_node = solver.temperature_at(0);
//! ```

use crate::physics::fd_discretization::WallDiscretization;
use std::fmt;

/// Surface boundary condition (Robin type).
///
/// Represents convective + radiative heat transfer at a surface:
///
/// ```text
/// q = h·(T_surface - T_fluid) + q_external
/// ```
#[derive(Debug, Clone)]
pub struct SurfaceBC {
    /// Combined convective/radiative coefficient [W/m²·K].
    pub h: f64,
    /// Fluid temperature (zone air or sol-air) [°C].
    pub t_fluid: f64,
    /// External heat flux (solar, etc.) [W/m²].
    pub q_external: f64,
}

impl SurfaceBC {
    /// Create interior surface BC.
    ///
    /// # Arguments
    ///
    /// * `h_conv` - Convective heat transfer coefficient [W/m²·K] (typical: 8)
    /// * `t_zone` - Zone air temperature [°C]
    ///
    /// # Returns
    ///
    /// Interior BC with zero external flux.
    pub fn new_interior(h_conv: f64, t_zone: f64) -> Self {
        Self {
            h: h_conv,
            t_fluid: t_zone,
            q_external: 0.0,
        }
    }

    /// Create exterior surface BC with sol-air temperature.
    ///
    /// # Arguments
    ///
    /// * `h_combined` - Combined convective coefficient [W/m²·K] (typical: 25)
    /// * `t_sol_air` - Sol-air temperature [°C]
    /// * `q_solar_direct` - Direct solar flux [W/m²] (already included in sol-air if using standard definition)
    ///
    /// # Returns
    ///
    /// Exterior BC.
    pub fn new_exterior(h_combined: f64, t_sol_air: f64, q_solar_direct: f64) -> Self {
        Self {
            h: h_combined,
            t_fluid: t_sol_air,
            q_external: q_solar_direct,
        }
    }

    /// Create BC with explicit radiative component.
    ///
    /// # Arguments
    ///
    /// * `h_conv` - Convective coefficient [W/m²·K]
    /// * `h_rad` - Radiative coefficient [W/m²·K]
    /// * `t_fluid` - Reference temperature [°C]
    /// * `q_solar` - Solar flux [W/m²]
    pub fn new_combined(h_conv: f64, h_rad: f64, t_fluid: f64, q_solar: f64) -> Self {
        Self {
            h: h_conv + h_rad,
            t_fluid,
            q_external: q_solar,
        }
    }
}

/// Tridiagonal matrix coefficients for the implicit system.
#[derive(Debug, Clone)]
struct TridiagonalSystem {
    /// Lower diagonal (A coefficients), length = n-1.
    lower: Vec<f64>,
    /// Main diagonal (B coefficients), length = n.
    main: Vec<f64>,
    /// Upper diagonal (C coefficients), length = n-1.
    upper: Vec<f64>,
    /// Right-hand side (D values), length = n.
    rhs: Vec<f64>,
}

impl TridiagonalSystem {
    /// Create new system with n equations.
    fn new(n: usize) -> Self {
        Self {
            lower: vec![0.0; n - 1],
            main: vec![0.0; n],
            upper: vec![0.0; n - 1],
            rhs: vec![0.0; n],
        }
    }
}

/// Time-integration scheme applied to the semi-discrete FD conduction
/// system (Issue #3980). The spatial discretization is unchanged; only the
/// temporal weighting of the discrete Laplacian differs.
///
/// - [`TimeIntegrationScheme::BackwardEuler`]: θ=1, 1st order, L-stable —
///   the historical scheme, kept as the baseline and as the first-step
///   bootstrap for multi-step schemes.
/// - [`TimeIntegrationScheme::Bdf2`]: 2nd-order backward differentiation
///   with variable-step support, L-stable for stiff walls. Production
///   default since Issue #3980.
/// - [`TimeIntegrationScheme::CrankNicolson`]: θ=½ trapezoidal rule, 2nd
///   order, A-stable but NOT L-stable — can oscillate for large Fourier
///   numbers (light walls at coarse steps). Provided for accuracy-vs-cost
///   characterization (Issue #3980 sweep), not the production default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeIntegrationScheme {
    /// Fully implicit Euler (1st order). Historical behaviour.
    BackwardEuler,
    /// Second-order backward differentiation formula (2-step BDF).
    #[default]
    Bdf2,
    /// Trapezoidal/Crank-Nicolson (θ=½, 2nd order, not L-stable).
    CrankNicolson,
}

/// Implicit finite difference solver for 1D heat conduction.
///
/// # Fields
///
/// * `discretization` - Wall spatial discretization
/// * `temperatures` - Current temperature at each node [°C]
/// * `dt` - Current timestep [s]
/// * `scheme` - Time-integration scheme (Issue #3980)
///
/// # Example
/// ```rust,ignore
/// // See the module-level # Example block above for the full construction
/// // (MaterialLayer -> WallDiscretization -> ImplicitFDSolver + SurfaceBC).
/// let disc = WallDiscretization::from_layers(&layers, 20);
/// let mut solver = ImplicitFDSolver::new(disc, 20.0);
///
/// for _ in 0..24 {
///     solver.step(3600.0, &interior_bc, &exterior_bc);
/// }
/// ```
pub struct ImplicitFDSolver {
    /// Wall spatial discretization.
    pub discretization: WallDiscretization,
    /// Temperature at each node [°C].
    pub temperatures: Vec<f64>,
    /// Current timestep [s].
    pub dt: f64,
    /// Cached Fourier numbers for each node.
    /// Per-node dt-scaled conductance weights of the conservative
    /// control-volume network (Issue #3981): `conductance_left[i]` couples
    /// node i to node i-1, `conductance_right[i]` to node i+1, both already
    /// divided by the node heat capacity C_i = rho_i*cp_i*dx_i. For a
    /// uniform single layer these reduce to the classic Fourier number
    /// Fo = alpha*dt/dx^2; for multi-layer walls each side carries the true
    /// series resistance 1/((dx_i/2)/k_i + (dx_{i+-1}/2)/k_{i+-1}) so the
    /// steady state solves the exact layer resistance network.
    conductance_left: Vec<f64>,
    conductance_right: Vec<f64>,
    /// Time-integration scheme (Issue #3980).
    scheme: TimeIntegrationScheme,
    /// `T^{n-1}` history for BDF2 (`None` until the first step completes).
    previous_temperatures: Option<Vec<f64>>,
    /// dt of the previous accepted step (variable-step BDF2 ratio).
    previous_dt: Option<f64>,
    /// Boundary conditions of the previous step for the Crank-Nicolson
    /// explicit source term (`None` until the first step completes).
    previous_interior_bc: Option<SurfaceBC>,
    previous_exterior_bc: Option<SurfaceBC>,
}

impl ImplicitFDSolver {
    /// Create new solver with uniform initial temperature.
    ///
    /// # Arguments
    ///
    /// * `discretization` - Wall discretization
    /// * `initial_temp` - Initial temperature for all nodes [°C]
    ///
    /// # Returns
    ///
    /// Solver ready for time stepping.
    pub fn new(discretization: WallDiscretization, initial_temp: f64) -> Self {
        let n = discretization.total_nodes;
        Self {
            discretization,
            temperatures: vec![initial_temp; n],
            dt: 3600.0, // Default 1 hour
            conductance_left: vec![0.0; n],
            conductance_right: vec![0.0; n],
            scheme: TimeIntegrationScheme::default(),
            previous_temperatures: None,
            previous_dt: None,
            previous_interior_bc: None,
            previous_exterior_bc: None,
        }
    }

    /// Create a solver with an explicit time-integration scheme (Issue #3980).
    ///
    /// # Arguments
    ///
    /// * `discretization` - Wall discretization
    /// * `initial_temp` - Initial temperature for all nodes [°C]
    /// * `scheme` - Time-integration scheme (see [`TimeIntegrationScheme`])
    pub fn with_scheme(
        discretization: WallDiscretization,
        initial_temp: f64,
        scheme: TimeIntegrationScheme,
    ) -> Self {
        let mut solver = Self::new(discretization, initial_temp);
        solver.scheme = scheme;
        solver
    }

    /// Create a solver with an explicit scheme and a non-uniform initial
    /// temperature profile (validation seeds eigenmodes / measured profiles;
    /// production paths use uniform initial temperatures).
    ///
    /// # Arguments
    ///
    /// * `discretization` - Wall discretization
    /// * `initial_temperatures` - Per-node initial temperature [°C]; length
    ///   must equal `discretization.total_nodes`
    /// * `scheme` - Time-integration scheme (see [`TimeIntegrationScheme`])
    ///
    /// # Panics
    ///
    /// Panics if the profile length does not match the node count.
    pub fn with_scheme_and_temperatures(
        discretization: WallDiscretization,
        initial_temperatures: Vec<f64>,
        scheme: TimeIntegrationScheme,
    ) -> Self {
        assert_eq!(
            initial_temperatures.len(),
            discretization.total_nodes,
            "initial temperature profile must match node count"
        );
        let mut solver = Self::new(discretization, 0.0);
        solver.temperatures = initial_temperatures;
        solver.scheme = scheme;
        solver
    }

    /// Create solver with temperature gradient.
    ///
    /// # Arguments
    ///
    /// * `discretization` - Wall discretization
    /// * `t_interior` - Interior surface temperature [°C]
    /// * `t_exterior` - Exterior surface temperature [°C]
    ///
    /// # Returns
    ///
    /// Solver with linear initial temperature profile.
    pub fn with_gradient(
        discretization: WallDiscretization,
        t_interior: f64,
        t_exterior: f64,
    ) -> Self {
        let n = discretization.total_nodes;
        let mut temperatures = Vec::with_capacity(n);

        for i in 0..n {
            let frac = discretization.node_positions[i] / discretization.total_thickness;
            temperatures.push(t_interior + frac * (t_exterior - t_interior));
        }

        Self {
            discretization,
            temperatures,
            dt: 3600.0,
            conductance_left: vec![0.0; n],
            conductance_right: vec![0.0; n],
            scheme: TimeIntegrationScheme::default(),
            previous_temperatures: None,
            previous_dt: None,
            previous_interior_bc: None,
            previous_exterior_bc: None,
        }
    }

    /// Time-integration scheme in use (Issue #3980).
    pub fn scheme(&self) -> TimeIntegrationScheme {
        self.scheme
    }

    /// Calculate Fourier number Fo = α·Δt/Δx² for each node.
    /// Recompute the conservative conductance weights for the current
    /// timestep (Issue #3981). Each weight is dt*G/C with G the series
    /// conductance between adjacent node centers (half-cell each side) and
    /// C the node heat capacity — the standard finite-volume form (Patankar).
    /// Unlike the former uniform-grid Fo = alpha*dt/dx^2, this represents
    /// layer interfaces exactly: at steady state the node network solves the
    /// true series resistance of the wall.
    fn update_conductance_weights(&mut self, dt: f64) {
        let n = self.discretization.total_nodes;
        let k = &self.discretization.conductivity;
        let dx = &self.discretization.node_volumes;
        let rho = &self.discretization.density;
        let cp = &self.discretization.specific_heat;
        for i in 0..n {
            let c_i = rho[i] * cp[i] * dx[i];
            if i > 0 {
                let g_left = 1.0 / ((dx[i] / 2.0) / k[i] + (dx[i - 1] / 2.0) / k[i - 1]);
                self.conductance_left[i] = dt * g_left / c_i;
            }
            if i < n - 1 {
                let g_right = 1.0 / ((dx[i] / 2.0) / k[i] + (dx[i + 1] / 2.0) / k[i + 1]);
                self.conductance_right[i] = dt * g_right / c_i;
            }
        }
    }

    /// Boundary (Robin) weights for one film: returns (dt*G_b/C_node,
    /// dt*(G_b/h)/C_node) where G_b = 1/(1/h + half-cell conduction). The
    /// second weight scales an externally absorbed surface flux q so that
    /// the face balance h*(T_air - T_s) + q = (T_s - T_node)/R_half is
    /// satisfied exactly at steady state.
    fn boundary_weights(&self, h: f64, idx: usize, dt: f64) -> (f64, f64) {
        let k = &self.discretization.conductivity;
        let dx = &self.discretization.node_volumes;
        let rho = &self.discretization.density;
        let cp = &self.discretization.specific_heat;
        let c_node = rho[idx] * cp[idx] * dx[idx];
        let g_b = 1.0 / (1.0 / h + (dx[idx] / 2.0) / k[idx]);
        (dt * g_b / c_node, dt * (g_b / h) / c_node)
    }

    /// Per-step time-integration weights for the assembled system.
    ///
    /// Unified form (mass normalized to 1):
    /// `T^{n+1} + w·S(T^{n+1}) = hist`, where `S` is the Fo-scaled discrete
    /// Laplacian including Robin boundary terms. The spatial operator is
    /// IDENTICAL for all schemes (Issue #3980 keeps the discretization);
    /// only `w` and the history RHS differ:
    ///
    /// * BackwardEuler: `w = 1`, `hist = T^n`
    /// * Bdf2 (constant dt): `w = 2/3`, `hist = (2·T^n − ½·T^{n-1})·(2/3)`
    ///
    /// Variable-step BDF2 derivation (ρ = dt/dt_prev): the Lagrange
    /// derivative through (t_{n+1}, t_n, t_{n-1}) evaluated at t_{n+1} gives
    /// `(1+2ρ)/(1+ρ)·T^{n+1} − S(T^{n+1}) = (1+ρ)·T^n − ρ²/(1+ρ)·T^{n-1}`;
    /// normalizing by mass yields the pair below. At ρ = 1 this reproduces
    /// [`crate::physics::bdf_engine::coefficients::BDF2`]
    /// (α = [½, −2, 3/2], β = 2/3).
    fn scheme_weights(
        &self,
        dt: f64,
        interior_bc: &SurfaceBC,
        exterior_bc: &SurfaceBC,
    ) -> (f64, Vec<f64>) {
        match self.scheme {
            TimeIntegrationScheme::BackwardEuler => (1.0, self.temperatures.clone()),
            TimeIntegrationScheme::CrankNicolson => {
                // θ = ½: rhs = T^n + (1-θ)·S(T^n, BCs at t_n). The explicit
                // source term uses the PREVIOUS step's boundary values; on
                // the first step they are unavailable and the current values
                // stand in (exact for constant BCs).
                let w = 0.5;
                let int_old = self.previous_interior_bc.as_ref().unwrap_or(interior_bc);
                let ext_old = self.previous_exterior_bc.as_ref().unwrap_or(exterior_bc);
                let s = self.explicit_laplacian(&self.temperatures, int_old, ext_old);
                let mut rhs = self.temperatures.clone();
                for (r, si) in rhs.iter_mut().zip(s.iter()) {
                    *r += (1.0 - w) * si;
                }
                (w, rhs)
            }
            TimeIntegrationScheme::Bdf2 => {
                match (&self.previous_temperatures, self.previous_dt) {
                    (Some(prev), Some(prev_dt)) if prev_dt > 0.0 => {
                        let rho = dt / prev_dt;
                        let alpha0 = (1.0 + 2.0 * rho) / (1.0 + rho);
                        let w = 1.0 / alpha0;
                        let hist: Vec<f64> = self
                            .temperatures
                            .iter()
                            .zip(prev.iter())
                            .map(|(&tn, &tn1)| {
                                ((1.0 + rho) * tn - (rho * rho / (1.0 + rho)) * tn1) / alpha0
                            })
                            .collect();
                        (w, hist)
                    }
                    // First step: backward-Euler bootstrap (a single 1st-order
                    // step; global order stays 2).
                    _ => (1.0, self.temperatures.clone()),
                }
            }
        }
    }

    /// Explicit evaluation of the conservative spatial operator S (dt-scaled
    /// by node heat capacity) — the explicit operator of the θ-scheme. Row
    /// structure mirrors the implicit assembly exactly so the two remain
    /// consistent. Interior rows use the per-side conductance weights
    /// (Issue #3981); boundary rows use the film+half-cell Robin conductance
    /// with the absorbed surface source split.
    fn explicit_laplacian(
        &self,
        temps: &[f64],
        interior: &SurfaceBC,
        exterior: &SurfaceBC,
    ) -> Vec<f64> {
        let n = self.discretization.total_nodes;
        let dt = self.dt;
        let mut s = vec![0.0; n];
        for i in 1..n - 1 {
            s[i] = self.conductance_left[i] * (temps[i - 1] - temps[i])
                + self.conductance_right[i] * (temps[i + 1] - temps[i]);
        }
        // Interior boundary row: film through the boundary half-cell.
        {
            let (a_b, q_w) = self.boundary_weights(interior.h, 0, dt);
            s[0] = self.conductance_right[0] * (temps[1] - temps[0])
                + a_b * (interior.t_fluid - temps[0])
                + q_w * interior.q_external;
        }
        // Exterior boundary row: film through the boundary half-cell.
        {
            let (a_b, q_w) = self.boundary_weights(exterior.h, n - 1, dt);
            s[n - 1] = self.conductance_left[n - 1] * (temps[n - 2] - temps[n - 1])
                + a_b * (exterior.t_fluid - temps[n - 1])
                + q_w * exterior.q_external;
        }
        s
    }

    /// Assemble tridiagonal system for implicit scheme.
    fn assemble_system(&self, w: f64, hist: &[f64]) -> TridiagonalSystem {
        let n = self.discretization.total_nodes;
        let mut sys = TridiagonalSystem::new(n);

        // Conservative per-side conductances (Issue #3981): row i carries
        // -w·aL·T_{i-1} + (1 + w·(aL+aR))·T_i - w·aR·T_{i+1}. Boundary rows
        // (0 and n-1) are finalized by apply_interior_bc/apply_exterior_bc.
        for (i, hist_i) in hist.iter().enumerate() {
            sys.main[i] = 1.0 + w * (self.conductance_left[i] + self.conductance_right[i]);
            if i > 0 {
                sys.lower[i - 1] = -w * self.conductance_left[i];
            }
            if i < n - 1 {
                sys.upper[i] = -w * self.conductance_right[i];
            }

            // RHS: scheme history (T^n for backward Euler, BDF2 combination
            // for second-order schemes)
            sys.rhs[i] = *hist_i;
        }

        sys
    }

    /// Apply interior surface boundary condition (Robin BC).
    ///
    /// Conservative half-cell form (Issue #3981): the film resistance and the
    /// boundary half-cell act in series between the zone air and node 0:
    /// ```text
    /// G_b = 1 / (1/h + (dx_0/2)/k_0)
    /// C_0/dt·(T_0^{n+1} - hist) = w·[G_b·(T_zone - T_0) + (G_b/h)·q + G_{01}·(T_1 - T_0)]
    /// ```
    /// At steady state this row realizes exactly 1/h + half-cell in series
    /// with the interior network, for any layer arrangement.
    fn apply_interior_bc(
        &mut self,
        sys: &mut TridiagonalSystem,
        bc: &SurfaceBC,
        w: f64,
        hist: &[f64],
    ) {
        let (a_b, q_w) = self.boundary_weights(bc.h, 0, self.dt);
        sys.main[0] = 1.0 + w * (self.conductance_right[0] + a_b);
        sys.upper[0] = -w * self.conductance_right[0];
        sys.rhs[0] = hist[0] + w * (a_b * bc.t_fluid + q_w * bc.q_external);
    }

    /// Apply exterior surface boundary condition (Robin BC) — conservative
    /// half-cell form, mirror of `apply_interior_bc` on node n-1.
    fn apply_exterior_bc(
        &mut self,
        sys: &mut TridiagonalSystem,
        bc: &SurfaceBC,
        w: f64,
        hist: &[f64],
    ) {
        let n = self.discretization.total_nodes;
        let (a_b, q_w) = self.boundary_weights(bc.h, n - 1, self.dt);
        sys.main[n - 1] = 1.0 + w * (self.conductance_left[n - 1] + a_b);
        sys.lower[n - 2] = -w * self.conductance_left[n - 1];
        sys.rhs[n - 1] = hist[n - 1] + w * (a_b * bc.t_fluid + q_w * bc.q_external);
    }

    /// Solve tridiagonal system using Thomas algorithm (TDMA).
    ///
    /// # Arguments
    ///
    /// * `sys` - Tridiagonal system to solve
    ///
    /// # Returns
    ///
    /// Solution vector T^{n+1}.
    fn thomas_algorithm(sys: &TridiagonalSystem) -> Vec<f64> {
        let n = sys.main.len();
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];

        // Forward sweep
        c_prime[0] = sys.upper[0] / sys.main[0];
        d_prime[0] = sys.rhs[0] / sys.main[0];

        for i in 1..n {
            let denom = sys.main[i] - sys.lower[i - 1] * c_prime[i - 1];
            if i < n - 1 {
                c_prime[i] = sys.upper[i] / denom;
            }
            d_prime[i] = (sys.rhs[i] - sys.lower[i - 1] * d_prime[i - 1]) / denom;
        }

        // Back substitution
        let mut x = vec![0.0; n];
        x[n - 1] = d_prime[n - 1];

        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        x
    }

    /// Advance solution by one timestep.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep duration [s]
    /// * `interior_bc` - Interior surface boundary condition
    /// * `exterior_bc` - Exterior surface boundary condition
    ///
    /// # Returns
    ///
    /// New temperature vector after timestep.
    pub fn step(&mut self, dt: f64, interior_bc: &SurfaceBC, exterior_bc: &SurfaceBC) -> Vec<f64> {
        // Update conservative conductance weights for new timestep (Issue #3981)
        self.update_conductance_weights(dt);
        self.dt = dt;

        // Time-integration weights (Issue #3980): scheme-dependent implicit
        // weight `w` and history RHS (BE: T^n; CN: T^n + explicit Laplacian;
        // BDF2: 2-step combination).
        let (w, hist) = self.scheme_weights(dt, interior_bc, exterior_bc);

        // Assemble tridiagonal system
        let mut sys = self.assemble_system(w, &hist);

        // Apply boundary conditions
        self.apply_interior_bc(&mut sys, interior_bc, w, &hist);
        self.apply_exterior_bc(&mut sys, exterior_bc, w, &hist);

        // Solve system
        let new_temps = Self::thomas_algorithm(&sys);
        // Update state: T^n becomes the T^{n-1} history for multi-step
        // schemes; dt drives the variable-step BDF2 ratio; the CN explicit
        // source term keeps the boundary values of this step.
        self.previous_temperatures =
            Some(std::mem::replace(&mut self.temperatures, new_temps.clone()));
        self.previous_dt = Some(dt);
        self.previous_interior_bc = Some(interior_bc.clone());
        self.previous_exterior_bc = Some(exterior_bc.clone());

        new_temps
    }

    /// Get current temperature at a specific node.
    #[inline]
    pub fn temperature_at(&self, node_idx: usize) -> Option<f64> {
        self.temperatures.get(node_idx).copied()
    }

    /// Get interior surface temperature (node 0).
    #[inline]
    pub fn interior_surface_temp(&self) -> f64 {
        self.temperatures[0]
    }

    /// Get exterior surface temperature (last node).
    #[inline]
    pub fn exterior_surface_temp(&self) -> f64 {
        self.temperatures[self.temperatures.len() - 1]
    }

    /// Calculate heat flux at interior surface [W/m²], positive from zone
    /// air INTO the wall.
    ///
    /// Consistent with the conservative boundary row (Issue #3981): the film
    /// and the boundary half-cell act in series between the zone air and
    /// node 0, so the face flux is G_b·(T_zone - T_0) with
    /// G_b = 1/(1/h + (dx_0/2)/k_0). Node 0 sits a half-cell inside the
    /// wall, not on the face, so using the bare film h against T_0 would
    /// bias the flux and break steady-state energy conservation across the
    /// wall. At steady state this telescopes with the interior network to
    /// the exact series-resistance flux.
    pub fn interior_heat_flux(&self, h_interior: f64, t_zone: f64) -> f64 {
        let k = self.discretization.conductivity[0];
        let dx = self.discretization.node_volumes[0];
        let g_b = 1.0 / (1.0 / h_interior + (dx / 2.0) / k);
        g_b * (t_zone - self.temperatures[0])
    }

    /// Calculate heat flux at exterior surface [W/m²], positive from
    /// ambient INTO the wall — mirror of `interior_heat_flux` on node n-1.
    pub fn exterior_heat_flux(&self, h_exterior: f64, t_sol_air: f64) -> f64 {
        let n = self.discretization.total_nodes;
        let k = self.discretization.conductivity[n - 1];
        let dx = self.discretization.node_volumes[n - 1];
        let g_b = 1.0 / (1.0 / h_exterior + (dx / 2.0) / k);
        g_b * (t_sol_air - self.temperatures[n - 1])
    }

    /// Calculate total energy stored in wall [J/m²].
    pub fn stored_energy(&self, reference_temp: f64) -> f64 {
        let mut energy = 0.0;

        for i in 0..self.discretization.total_nodes {
            let mass = self.discretization.density[i] * self.discretization.node_volumes[i];
            let cp = self.discretization.specific_heat[i];
            energy += mass * cp * (self.temperatures[i] - reference_temp);
        }

        energy
    }

    /// Check energy balance over timestep.
    ///
    /// # Returns
    ///
    /// Energy balance error [J/m²]: E_in - E_out - ΔE_stored
    pub fn energy_balance_error(
        &self,
        prev_energy: f64,
        q_interior: f64,
        q_exterior: f64,
        dt: f64,
    ) -> f64 {
        let current_energy = self.stored_energy(0.0);
        let delta_stored = current_energy - prev_energy;

        // Energy in from exterior, out to interior
        let e_in = q_exterior * dt;
        let e_out = q_interior * dt;

        e_in - e_out - delta_stored
    }
}

impl fmt::Display for ImplicitFDSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FD Solver State:")?;
        writeln!(f, "  Timestep: {:.1} s", self.dt)?;
        writeln!(f, "  Nodes: {}", self.discretization.total_nodes)?;
        writeln!(f, "  T_interior: {:.2}°C", self.interior_surface_temp())?;
        writeln!(f, "  T_exterior: {:.2}°C", self.exterior_surface_temp())?;

        // Show temperature profile (every 5th node)
        writeln!(f, "  Temperature profile:")?;
        for i in (0..self.temperatures.len()).step_by(5.max(self.temperatures.len() / 10)) {
            writeln!(
                f,
                "    Node {:3} (x={:.3}m): {:.2}°C",
                i, self.discretization.node_positions[i], self.temperatures[i]
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fd_discretization::MaterialLayer;

    /// Create simple homogeneous wall for testing.
    fn concrete_wall(thickness: f64, nodes: usize) -> WallDiscretization {
        let layers = vec![MaterialLayer::new(
            "Concrete", thickness, 1.4, 2300.0, 880.0,
        )];
        WallDiscretization::from_layers(&layers, nodes)
    }

    #[test]
    fn test_default_scheme_is_bdf2() {
        // Issue #3980: the production default is the 2nd-order BDF2 scheme;
        // BE remains available explicitly for baselines and cross-checks.
        let solver = ImplicitFDSolver::new(concrete_wall(0.2, 20), 20.0);
        assert_eq!(solver.scheme(), TimeIntegrationScheme::Bdf2);
    }

    #[test]
    fn test_steady_state_conduction() {
        // 200mm concrete wall, T_interior=20°C, T_exterior=0°C
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::with_gradient(disc.clone(), 20.0, 0.0);

        // Apply steady BCs with very high h to approximate fixed temperature
        let interior_bc = SurfaceBC::new_interior(1e9, 20.0); // Extremely high h → fixed T
        let exterior_bc = SurfaceBC::new_exterior(1e9, 0.0, 0.0);

        // Run to steady state (200 hours for thick wall)
        for _ in 0..200 {
            solver.step(3600.0, &interior_bc, &exterior_bc);
        }

        // Check linear temperature profile (allow 5% tolerance)
        for i in 0..solver.temperatures.len() {
            let x_frac = disc.node_positions[i] / disc.total_thickness;
            let t_expected = 20.0 - x_frac * 20.0;
            assert!(
                (solver.temperatures[i] - t_expected).abs() < 1.0
                    || (solver.temperatures[i] - t_expected).abs() / 20.0 < 0.05,
                "Node {}: T={:.2}, expected {:.2}",
                i,
                solver.temperatures[i],
                t_expected
            );
        }
    }

    #[test]
    fn test_transient_step_response() {
        // Semi-infinite solid approximation: sudden surface temp change
        let disc = concrete_wall(0.500, 50); // Thick wall
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        // Suddenly change surface to 0°C
        let interior_bc = SurfaceBC::new_interior(1e9, 0.0);
        let exterior_bc = SurfaceBC::new_exterior(1e9, 20.0, 0.0);

        // After 1 hour, check penetration depth
        solver.step(3600.0, &interior_bc, &exterior_bc);

        // Temperature change should be localized near surface
        assert!(
            solver.temperatures[0] < 10.0,
            "Surface should cool significantly, got {:.2}",
            solver.temperatures[0]
        );
        assert!(
            solver.temperatures[solver.temperatures.len() - 1] > 15.0,
            "Far end should stay warm, got {:.2}",
            solver.temperatures[solver.temperatures.len() - 1]
        );
    }

    #[test]
    fn test_energy_conservation() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc.clone(), 20.0);

        let interior_bc = SurfaceBC::new_interior(8.0, 21.0);
        let exterior_bc = SurfaceBC::new_exterior(25.0, 5.0, 0.0);

        let mut prev_energy = solver.stored_energy(0.0);

        // Run for 10 hours, checking energy balance each step
        for _hour in 0..10 {
            let q_ext_before = solver.exterior_heat_flux(25.0, 5.0);
            let q_int_before = solver.interior_heat_flux(8.0, 21.0);

            solver.step(3600.0, &interior_bc, &exterior_bc);

            let q_ext_after = solver.exterior_heat_flux(25.0, 5.0);
            let q_int_after = solver.interior_heat_flux(8.0, 21.0);

            // Average flux during timestep
            let q_ext_avg = (q_ext_before + q_ext_after) / 2.0;
            let q_int_avg = (q_int_before + q_int_after) / 2.0;

            let current_energy = solver.stored_energy(0.0);
            let delta_stored = current_energy - prev_energy;

            // Energy in from exterior, out to interior
            let e_in = q_ext_avg * 3600.0;
            let e_out = q_int_avg * 3600.0;

            let error = (e_in - e_out - delta_stored).abs();

            // Energy balance should be maintained (within 50% for first-order method with large dt)
            let total_flow = e_in.abs().max(e_out.abs()).max(delta_stored.abs());
            assert!(
                error < total_flow * 0.5 || error < 500000.0,
                "Energy balance error = {:.2} J/m²",
                error
            );

            prev_energy = current_energy;
        }
    }

    #[test]
    fn test_conductance_weight_calculation() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        solver.update_conductance_weights(3600.0);

        // For a uniform layer the per-side weights equal the classic Fourier
        // number Fo = alpha*dt/dx^2: concrete alpha ~= 6.9e-7 m2/s, dx = 0.01,
        // dt = 3600 -> Fo ~= 0.025. Interior nodes must realize this exactly.
        let fo_classic = 6.9e-7_f64 * 3600.0 / (0.01_f64 * 0.01);
        for i in 1..solver.discretization.total_nodes - 1 {
            let a_left = solver.conductance_left[i];
            let a_right = solver.conductance_right[i];
            assert!(
                (a_left - fo_classic).abs() / fo_classic < 0.02,
                "a_left = {a_left:.6} vs classic Fo {fo_classic:.6}"
            );
            assert!(
                (a_right - fo_classic).abs() / fo_classic < 0.02,
                "a_right = {a_right:.6} vs classic Fo {fo_classic:.6}"
            );
        }
    }

    #[test]
    fn test_multilayer_conductance_weights_realize_series_resistance() {
        // Two layers with a 20:1 conductivity contrast: the steady network
        // must realize the exact series resistance (Issue #3981 regression
        // guard for the former uniform-stencil defect).
        use super::super::fd_discretization::MaterialLayer;
        let layers = vec![
            MaterialLayer::new("brick", 0.1, 0.81, 1920.0, 790.0),
            MaterialLayer::new("eps", 0.08, 0.04, 25.0, 1400.0),
        ];
        let disc = WallDiscretization::from_layers(&layers, 30);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);
        solver.update_conductance_weights(3600.0);
        // Sum of 1/(a_i)·(C_i/dt) over the chain reconstructs total R between
        // the two boundary node centers; verify via steady flux instead:
        // drive with constant BCs and compare realized U to the analytical
        // series resistance (films included).
        for _ in 0..60 * 24 {
            let bc_int = SurfaceBC::new_interior(8.3, 20.0);
            let bc_ext = SurfaceBC::new_exterior(18.3, 28.0, 0.0);
            solver.step(3600.0, &bc_int, &bc_ext);
        }
        let q_through =
            0.5 * (solver.exterior_heat_flux(18.3, 28.0) - solver.interior_heat_flux(8.3, 20.0));
        let u_true = 1.0 / (1.0 / 8.3 + 0.1 / 0.81 + 0.08 / 0.04 + 1.0 / 18.3);
        assert!(
            (q_through / 8.0 / u_true - 1.0).abs() < 0.005,
            "realized U {:.6} vs analytical {:.6}",
            q_through / 8.0,
            u_true
        );
    }

    #[test]
    fn test_thomas_algorithm_correctness() {
        // Test with known tridiagonal system
        let sys = TridiagonalSystem {
            lower: vec![-1.0, -1.0, -1.0],
            main: vec![4.0, 4.0, 4.0, 4.0],
            upper: vec![-1.0, -1.0, -1.0],
            rhs: vec![5.0, 2.0, 2.0, 3.0],
        };

        let x = ImplicitFDSolver::thomas_algorithm(&sys);

        // Verify A·x = b
        assert!((4.0 * x[0] - x[1] - 5.0).abs() < 1e-10);
        assert!((-x[0] + 4.0 * x[1] - x[2] - 2.0).abs() < 1e-10);
        assert!((-x[1] + 4.0 * x[2] - x[3] - 2.0).abs() < 1e-10);
        assert!((-x[2] + 4.0 * x[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_case_900_wall_simulation() {
        // Case 900 high-mass wall
        let layers = vec![
            MaterialLayer::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            MaterialLayer::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
            MaterialLayer::new("Insulation", 0.050, 0.04, 50.0, 840.0),
            MaterialLayer::new("Brick", 0.100, 0.81, 1920.0, 790.0),
        ];

        let disc = WallDiscretization::from_layers(&layers, 10);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        // Diurnal cycle simulation (24 hours)
        for hour in 0..24 {
            let t_out = 10.0 + 5.0 * ((hour as f64 - 6.0) * std::f64::consts::PI / 12.0).sin();
            let t_sol_air = t_out + 3.0; // Small solar gain

            let interior_bc = SurfaceBC::new_interior(8.0, 20.0);
            let exterior_bc = SurfaceBC::new_exterior(25.0, t_sol_air, 0.0);

            solver.step(3600.0, &interior_bc, &exterior_bc);
        }

        // Check that temperatures are physically reasonable (0-50°C range)
        for t in &solver.temperatures {
            assert!(
                *t > -10.0 && *t < 60.0,
                "T = {:.2}°C outside reasonable range",
                t
            );
        }

        // Check that insulation layer shows temperature drop (nodes 20-29)
        let t_concrete = solver.temperatures[15]; // In concrete
        let t_brick = solver.temperatures[35]; // In brick
                                               // Insulation should cause temperature gradient
        assert!(
            (t_concrete - t_brick).abs() > 0.1,
            "Insulation should create temperature gradient"
        );
    }

    // === Phase 3: Additional coverage tests ===

    #[test]
    fn test_surface_bc_new_combined() {
        let bc = SurfaceBC::new_combined(8.0, 5.0, 20.0, 100.0);
        assert_eq!(bc.h, 13.0); // 8 + 5
        assert_eq!(bc.t_fluid, 20.0);
        assert_eq!(bc.q_external, 100.0);
    }

    #[test]
    fn test_surface_bc_new_interior() {
        let bc = SurfaceBC::new_interior(8.0, 21.0);
        assert_eq!(bc.h, 8.0);
        assert_eq!(bc.t_fluid, 21.0);
        assert_eq!(bc.q_external, 0.0);
    }

    #[test]
    fn test_surface_bc_new_exterior() {
        let bc = SurfaceBC::new_exterior(25.0, 5.0, 300.0);
        assert_eq!(bc.h, 25.0);
        assert_eq!(bc.t_fluid, 5.0);
        assert_eq!(bc.q_external, 300.0);
    }

    #[test]
    fn test_temperature_at_valid() {
        let disc = concrete_wall(0.200, 20);
        let solver = ImplicitFDSolver::new(disc, 20.0);

        // Valid node indices
        for i in 0..20 {
            let temp = solver.temperature_at(i);
            assert!(temp.is_some());
            assert_eq!(temp.unwrap(), 20.0);
        }
    }

    #[test]
    fn test_temperature_at_invalid() {
        let disc = concrete_wall(0.200, 20);
        let solver = ImplicitFDSolver::new(disc, 20.0);

        // Invalid node indices
        assert!(solver.temperature_at(999).is_none());
        assert!(solver.temperature_at(20).is_none());
    }

    #[test]
    fn test_interior_surface_temp() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 25.0);

        assert_eq!(solver.interior_surface_temp(), 25.0);

        // Change interior temperature
        solver.temperatures[0] = 30.0;
        assert_eq!(solver.interior_surface_temp(), 30.0);
    }

    #[test]
    fn test_exterior_surface_temp() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 15.0);

        assert_eq!(solver.exterior_surface_temp(), 15.0);

        // Change exterior temperature
        let n = solver.temperatures.len() - 1;
        solver.temperatures[n] = 10.0;
        assert_eq!(solver.exterior_surface_temp(), 10.0);
    }

    #[test]
    fn test_interior_heat_flux() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        // Zero flux when temperatures equal
        let flux = solver.interior_heat_flux(8.0, 20.0);
        assert_eq!(flux, 0.0);

        // Positive flux into zone. Conservative convention (Issue #3981):
        // the film acts through the boundary half-cell, G_b = 1/(1/h + dx/2k).
        // concrete_wall(0.2, 20): dx = 0.01, k = 1.7 -> G_b = 1/(1/8 + 0.005/1.7).
        solver.temperatures[0] = 18.0;
        let flux = solver.interior_heat_flux(8.0, 20.0);
        assert!(flux > 0.0); // Heat flowing into zone
        let g_b = 1.0 / (1.0 / 8.0 + 0.005 / 1.4);
        assert!((flux - g_b * (20.0 - 18.0)).abs() < 1e-12);
    }

    #[test]
    fn test_exterior_heat_flux() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        // Zero flux when temperatures equal
        let flux = solver.exterior_heat_flux(25.0, 20.0);
        assert_eq!(flux, 0.0);

        // Positive flux into wall. Conservative convention (Issue #3981):
        // film through the boundary half-cell, G_b = 1/(1/h + dx/2k).
        let n = solver.temperatures.len() - 1;
        solver.temperatures[n] = 15.0;
        let flux = solver.exterior_heat_flux(25.0, 20.0);
        assert!(flux > 0.0); // Heat flowing into wall
        let g_b = 1.0 / (1.0 / 25.0 + 0.005 / 1.4);
        assert!((flux - g_b * (20.0 - 15.0)).abs() < 1e-12);
    }

    #[test]
    fn test_stored_energy_zero_reference() {
        let disc = concrete_wall(0.200, 20);
        let solver = ImplicitFDSolver::new(disc, 20.0);

        // With 0°C reference, should have positive energy
        let energy = solver.stored_energy(0.0);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_stored_energy_same_reference() {
        let disc = concrete_wall(0.200, 20);
        let solver = ImplicitFDSolver::new(disc, 20.0);

        // With same reference, energy should be zero
        let energy = solver.stored_energy(20.0);
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_energy_balance_error() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        let prev_energy = solver.stored_energy(0.0);
        let q_int = 10.0;
        let q_ext = 20.0;
        let dt = 3600.0;

        // Make one step
        let interior_bc = SurfaceBC::new_interior(8.0, 21.0);
        let exterior_bc = SurfaceBC::new_exterior(25.0, 5.0, 0.0);
        solver.step(dt, &interior_bc, &exterior_bc);

        let error = solver.energy_balance_error(prev_energy, q_int, q_ext, dt);
        // Error should be finite
        assert!(error.is_finite());
    }

    #[test]
    fn test_step_various_timesteps() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        let interior_bc = SurfaceBC::new_interior(8.0, 21.0);
        let exterior_bc = SurfaceBC::new_exterior(25.0, 5.0, 0.0);

        // Test various timestep sizes
        for dt in [300.0, 600.0, 1800.0, 3600.0, 7200.0] {
            let result = solver.step(dt, &interior_bc, &exterior_bc);
            assert_eq!(result.len(), 20);
            // All temperatures should be finite
            for t in &result {
                assert!(t.is_finite());
            }
        }
    }

    #[test]
    fn test_step_with_solar_flux() {
        let disc = concrete_wall(0.200, 20);
        let mut solver = ImplicitFDSolver::new(disc, 20.0);

        // Apply solar flux to exterior
        let interior_bc = SurfaceBC::new_interior(8.0, 20.0);
        let exterior_bc = SurfaceBC::new_exterior(25.0, 5.0, 500.0); // 500 W/m² solar

        let temps_before = solver.temperatures.clone();
        solver.step(3600.0, &interior_bc, &exterior_bc);
        let temps_after = solver.temperatures.clone();

        // Exterior should warm up due to solar
        let n = temps_before.len() - 1;
        assert!(temps_after[n] > temps_before[n]);
    }

    #[test]
    fn test_with_gradient() {
        let disc = concrete_wall(0.200, 20);
        let solver = ImplicitFDSolver::with_gradient(disc.clone(), 25.0, 5.0);

        // Gradient should be monotonic (interior > middle > exterior)
        assert!(solver.temperatures[0] > solver.temperatures[10]);
        assert!(solver.temperatures[10] > solver.temperatures[19]);

        // First and last should be close to specified values
        assert!((solver.temperatures[0] - 25.0).abs() < 2.0);
        assert!((solver.temperatures[19] - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_tridiagonal_system_new() {
        let sys = TridiagonalSystem::new(5);
        assert_eq!(sys.lower.len(), 4);
        assert_eq!(sys.main.len(), 5);
        assert_eq!(sys.upper.len(), 4);
        assert_eq!(sys.rhs.len(), 5);

        // All should be zero
        for i in 0..5 {
            assert_eq!(sys.main[i], 0.0);
            if i < 4 {
                assert_eq!(sys.lower[i], 0.0);
                assert_eq!(sys.upper[i], 0.0);
            }
            assert_eq!(sys.rhs[i], 0.0);
        }
    }

    #[test]
    fn test_solver_display() {
        let disc = concrete_wall(0.200, 20);
        let solver = ImplicitFDSolver::new(disc, 20.0);

        let display_str = format!("{}", solver);
        assert!(display_str.contains("FD Solver State"));
        assert!(display_str.contains("Timestep"));
        assert!(display_str.contains("Nodes"));
        assert!(display_str.contains("T_interior"));
        assert!(display_str.contains("T_exterior"));
        assert!(display_str.contains("Temperature profile"));
    }

    #[test]
    fn test_surface_bc_debug_clone() {
        let bc = SurfaceBC::new_interior(8.0, 21.0);
        let debug_str = format!("{:?}", bc);
        assert!(debug_str.contains("SurfaceBC"));

        let cloned = bc.clone();
        assert_eq!(cloned.h, bc.h);
        assert_eq!(cloned.t_fluid, bc.t_fluid);
        assert_eq!(cloned.q_external, bc.q_external);
    }
}
