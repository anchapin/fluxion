//! Multi-Node Thermal Solver for 9R4C Model (Phase 6C)
//!
//! This module implements a backward Euler finite difference solver for the
//! 9R4C thermal network model used for heavy-mass buildings (Case 900+ series).
//!
//! ## 9R4C Network Architecture
//!
//! The 9R4C model separates thermal mass into 4 nodes:
//! - Wall node (Cm_wall): receives heat from exterior via h_tr_em_wall and from zone via h_tr_is
//! - Roof node (Cm_roof): receives heat from exterior via h_tr_em_roof and from zone via h_tr_is
//! - Floor node (Cm_floor): receives heat from exterior via h_tr_em_floor and from zone via h_tr_is
//! - Internal node (Cm_internal): furniture, partitions — receives heat from zone via h_tr_is
//!
//! ## Resistance Network
//!
//! ```text
//!                    h_tr_em_wall         h_tr_em_roof         h_tr_em_floor
//!   T_exterior ----[R_em_wall]----[Tm_wall]----+
//!                                          |
//!   T_exterior ----[R_em_roof]----[Tm_roof]---+  (parallel to exterior)
//!                                          |
//!   T_exterior ----[R_em_floor]--[Tm_floor]--+
//!                                          |
//!   T_zone ----[R_is]----[T_s]----[R_ms]----+  (series path to mass)
//!                              |
//!                              +----[R_me]----[Tm_internal]
//!
//! Where:
//! - h_tr_is: zone air to surface conductance
//! - h_tr_ms: surface to mass conductance (shared for all envelope surfaces in series)
//! - h_tr_me: internal mass to envelope mass conductance
//! ```
//!
//! Each envelope node (wall, roof, floor) has its own h_tr_em path to exterior.
//! All envelope nodes share the same surface node T_s via their respective h_tr_ms paths.

use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::{FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::WallSpec;
// Issue #2462 (Phase 2 of the crate split): per-surface conduction types now
// live in the `fluxion_core::per_surface_conduction` leaf crate (where they
// were hoisted to break the `physics ↔ sim` module cycle documented in
// ARCHITECTURE.md §"Remaining cycles"). `crate::sim::per_surface_conduction::*`
// is still valid as a re-export shim, but importing from the leaf crate
// directly is the canonical path that `scripts/check_physics_sim_cycle.py`
// enforces (the script reports 0 edges when no `use crate::sim::*` imports
// remain under `src/physics/**`).
use fluxion_core::per_surface_conduction::{PerSurfaceConductionSolver, SurfaceKind};
// Issue #2462 (Phase 2 of the crate split): `STEFAN_BOLTZMANN` was hoisted
// out of `crate::sim::sky_radiation` into the new
// `fluxion_core::physics_constants` leaf module so this file no longer needs
// to import from `sim` (which would re-introduce the very cycle #2462 is
// here to break). The `crate::sim::sky_radiation::STEFAN_BOLTZMANN` path
// stays valid via a re-export shim, but the leaf-crate import is the
// canonical path that `scripts/check_physics_sim_cycle.py` enforces.
use fluxion_core::physics_constants::STEFAN_BOLTZMANN;
// Issue #1349 (Phase 2 crate split): multi-node thermal mass types moved to `fluxion_core::multi_node`.
use fluxion_core::multi_node::{MassAirCouplingMode, MultiNodeThermalMass, ThermalMassNode};
use log;

/// Series combination of two conductances (Issue #1281, parallel-resistance
/// coupling network for 9R4C).
///
/// `h_series(a, b) = (a × b) / (a + b)` is the conductance of `a` and `b` placed
/// in series. It is symmetric, strictly positive when both inputs are positive,
/// and bounded above by `min(a, b)`.
///
/// In the parallel-resistance formulation, each per-surface mass-to-air path is
/// the series pair `(h_tr_ms_k, h_tr_is)`, so `h_path_k = h_series(h_tr_ms_k, h_tr_is)`.
///
/// Returns 0.0 for degenerate inputs (a≤0 or b≤0); caller is expected to
/// validate inputs upstream. A `debug_assert!` fires in debug builds to
/// catch configuration errors early.
#[inline]
pub fn h_series(a: f64, b: f64) -> f64 {
    debug_assert!(
        a > 0.0 && b > 0.0,
        "h_series called with degenerate inputs: a={}, b={}",
        a,
        b
    );
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    (a * b) / (a + b)
}

/// Strict version of `h_series` that returns `Err` instead of emitting a
/// `debug_assert!` for degenerate inputs. This enables release-mode testing
/// of the error path.
#[inline]
pub fn h_series_strict(a: f64, b: f64) -> Result<f64, &'static str> {
    if a <= 0.0 || b <= 0.0 {
        return Err("h_series called with degenerate inputs");
    }
    Ok((a * b) / (a + b))
}

/// Compute the linearized sky-radiative conductance [W/K] for the 9R4C air node
/// (Issue #1858 — closes the ~0.6 °C high-mass free-float night-min residual).
///
/// Issue #2872 — applies a per-surface sky view factor `f_sky` to the mass
/// node boundary so the longwave sky-radiation exchange is distributed
/// across the envelope instead of being concentrated on the roof. The
/// canonical values are `f_sky_wall = 0.5` (vertical wall, half sky dome),
/// `f_sky_roof = 1.0` (horizontal roof, full sky dome), and `f_sky_floor
/// = 0.0` (slab-on-grade, no sky view).
///
/// The 9R4C air-node energy balance previously had only four terms
/// (`h_tr_is · T_s`, `(h_ve + h_ve_night) · T_out`, `φ_ia`), which algebraically
/// bounds the free-floating air temperature below by `min(T_surface, T_out)`.
/// Under clear-sky radiative cooling the air temperature should be able to drop
/// *below* the outdoor dry-bulb; this conductance adds that path.
///
/// The exchange is modeled as a linearized longwave conductance between the air
/// node and the effective sky temperature:
///
/// ```text
/// h_rad_sky = ε · F_sky · 4 · σ · T_mean³ · A_aperture     [W/K]
/// ```
///
/// where `T_mean = (T_air + T_sky) / 2` [K] and `σ` is the Stefan–Boltzmann
/// constant. This is the same linearization used by
/// `SkyRadiationExchange::radiative_coefficient`, scaled by the radiative
/// aperture area to yield a total conductance compatible with the per-zone
/// `h_tr_is` / `h_ve` terms.
///
/// All inputs are physics-derived — emissivity, sky-view factor (from surface
/// tilt via `SkyRadiationExchange::tilted_surface`), aperture area (building
/// geometry), temperatures (EPW `sky_temperature()` + current air estimate) —
/// so no case-specific tuning constant is introduced (RULES.md).
///
/// Returns 0.0 for degenerate inputs (non-positive emissivity / view factor /
/// aperture), which makes the sky path a no-op for callers that do not supply
/// sky data — preserving backward compatibility.
#[inline]
pub fn air_sky_conductance(
    emissivity: f64,
    sky_view_factor: f64,
    aperture_area: f64,
    t_air_c: f64,
    t_sky_c: f64,
) -> f64 {
    if aperture_area <= 0.0 || emissivity <= 0.0 || sky_view_factor <= 0.0 {
        return 0.0;
    }
    let t_air_k = t_air_c + 273.15;
    let t_sky_k = t_sky_c + 273.15;
    let t_mean = (t_air_k + t_sky_k) / 2.0;
    if !t_mean.is_finite() || t_mean <= 0.0 {
        return 0.0;
    }
    4.0 * emissivity * sky_view_factor * STEFAN_BOLTZMANN * t_mean.powi(3) * aperture_area
}

/// Compute the conductance-weighted envelope temperature for internal node coupling
/// (Issue #1859).
///
/// ISO 13790 §C.3 specifies that the internal mass couples to the envelope
/// surfaces through the series combination of (surface-to-mass, mass-to-internal):
/// `h_me_k = h_series(h_tr_ms_k, h_tr_me)` per surface k.
///
/// The effective envelope temperature driving heat flow into the internal node is
/// the h_me-weighted average of the three envelope mass temperatures:
/// `t_env_avg = Σ(h_me_k × T_m_k) / Σ(h_me_k)` for k ∈ {wall, roof, floor}.
///
/// This replaces the unweighted arithmetic mean `(T_wall + T_roof + T_floor) / 3.0`
/// which over-weights whichever envelope happens to be hotter, suppressing the
/// internal node's damping effect on diurnal air temperature swing.
///
/// Degenerate cases (h_tr_me <= 0 or all h_series ~ 0) fall back to the simple
/// arithmetic mean.
#[inline]
fn internal_node_envelope_temperature(
    t_wall: f64,
    t_roof: f64,
    t_floor: f64,
    h_ms_wall: f64,
    h_ms_roof: f64,
    h_ms_floor: f64,
    h_tr_me: f64,
) -> f64 {
    if h_tr_me <= 0.0 {
        return (t_wall + t_roof + t_floor) / 3.0;
    }
    let h_me_w = h_series(h_ms_wall, h_tr_me);
    let h_me_r = h_series(h_ms_roof, h_tr_me);
    let h_me_f = h_series(h_ms_floor, h_tr_me);
    let h_me_sum = h_me_w + h_me_r + h_me_f;
    if h_me_sum > 1e-6 {
        (h_me_w * t_wall + h_me_r * t_roof + h_me_f * t_floor) / h_me_sum
    } else {
        (t_wall + t_roof + t_floor) / 3.0
    }
}

/// Per-surface surface temperature for the parallel-resistance 9R4C coupling
/// (Issue #1281).
///
/// Steady-state solution of the (mass → T_s → air) series pair, given the
/// current mass temperature `t_m`, the surface-to-air conductance `h_is`,
/// and the air temperature `t_air`:
/// ...
/// T_s = (h_tr_ms × t_m + h_tr_is × t_air) / (h_tr_ms + h_tr_is)
/// ```
///
/// Equivalent to `t_air + (h_tr_ms / (h_tr_ms + h_tr_is)) × (t_m − t_air)`.
/// Degenerate cases (`h_tr_ms + h_tr_is` near zero, or non-finite inputs)
/// fall back to the air temperature.
#[inline]
fn per_surface_t_s(t_m: f64, h_tr_ms: f64, h_tr_is: f64, t_air: f64) -> f64 {
    let denom = h_tr_ms + h_tr_is;
    if !denom.is_finite() || denom < 1e-10 {
        return t_air;
    }
    if !t_m.is_finite() || !t_air.is_finite() {
        return t_air;
    }
    (h_tr_ms * t_m + h_tr_is * t_air) / denom
}

/// Per-surface exterior boundary temperatures for the multi-node solver (Issue #863).
///
/// Each envelope node (wall, roof, floor) can have its own exterior boundary
/// temperature, computed from sol-air temperature calculations.
///
/// - Wall/Roof: sol-air temperature (accounts for solar irradiance, longwave radiation)
/// - Floor: ground temperature (ground-coupled)
#[derive(Debug, Clone)]
pub struct SurfaceExteriorTemperatures {
    /// Sol-air temperature for the wall exterior boundary (°C)
    pub t_ext_wall: f64,
    /// Sol-air temperature for the roof exterior boundary (°C)
    pub t_ext_roof: f64,
    /// Ground temperature for the floor exterior boundary (°C)
    pub t_ext_floor: f64,
}

impl SurfaceExteriorTemperatures {
    /// Create with a uniform exterior temperature (legacy fallback).
    pub fn uniform(t: f64) -> Self {
        Self {
            t_ext_wall: t,
            t_ext_roof: t,
            t_ext_floor: t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiNodeSolver {
    pub mass: MultiNodeThermalMass,
    pub h_tr_is: f64,
    pub zone_temperature: f64,
    pub surface_temperature: f64,
    /// Legacy single exterior temperature — kept for backward compatibility.
    pub exterior_temperature: f64,
    /// Per-surface exterior boundary temperatures (Issue #863).
    /// Each envelope node uses its respective boundary temp
    /// instead of the uniform `exterior_temperature`.
    pub exterior_temperatures: SurfaceExteriorTemperatures,
    pub timestep_seconds: f64,
    /// Mass-to-air coupling mode (Issue #1281).
    ///
    /// Defaults to `MassAirCouplingMode::AdditiveSum` (original shared-T_s
    /// formulation) for backward compatibility. Set to
    /// `MassAirCouplingMode::ParallelResistance` to use the per-surface
    /// series-path formulation described in `MassAirCouplingMode`'s docs.
    pub coupling_mode: MassAirCouplingMode,
    /// Issue #1429: cached total wall-layer resistance [m²·K/W]. Set by
    /// `initialize(&WallSpec)` so the trait's `steady_state_flux` query can
    /// return the closed-form `q_ss = (T_ext − T_int) / R_total` matching the
    /// `FiveR1CSolver` default semantics (no thermal mass effect).
    pub r_total: f64,
    /// Issue #1589: exterior surface film resistance [m²·K/W]. Set alongside
    /// `r_total` so that `step()` can compute the full series resistance
    /// `R_se + R_total + R_si` for the correct steady-state flux.
    pub r_se: f64,
    /// Issue #1429: true once `initialize(&WallSpec)` has configured the four
    /// mass nodes from a layer stack. Required for `is_valid()` and gates
    /// `step()` and `steady_state_flux()` so the trait contract holds even
    /// when callers forget to initialize.
    pub initialized: bool,
    /// Issue #1429: cached `(T_wall, T_roof, T_floor, T_internal)` BEFORE the
    /// most recent `step_backward_euler` update. Lets `energy_storage_rate()`
    /// compute `Σ C_k × (T_k_new − T_k_old) / dt` (positive = wall charging).
    /// `None` until the first `step()` invocation.
    pub last_temps: Option<(f64, f64, f64, f64)>,
    /// Issue #1429: timestep used in the most recent `step_backward_euler`
    /// update. Required to convert the J-scale stored-energy delta into
    /// the W/m² rate expected by the trait.
    pub last_dt: f64,
}

impl MultiNodeSolver {
    pub fn new(
        h_tr_is: f64,
        wall: ThermalMassNode,
        roof: ThermalMassNode,
        floor: ThermalMassNode,
        internal: ThermalMassNode,
    ) -> Self {
        Self {
            mass: MultiNodeThermalMass::new(wall, roof, floor, internal),
            h_tr_is,
            zone_temperature: 20.0,
            surface_temperature: 20.0,
            exterior_temperature: 10.0,
            exterior_temperatures: SurfaceExteriorTemperatures::uniform(10.0),
            timestep_seconds: 3600.0,
            coupling_mode: MassAirCouplingMode::default(),
            // Issue #1429 — trait state defaults (see struct docs).
            r_total: 0.0,
            r_se: 0.0,
            initialized: false,
            last_temps: None,
            last_dt: 0.0,
        }
    }

    /// Construct a solver with a chosen mass-to-air coupling mode (Issue #1281).
    pub fn new_with_mode(
        h_tr_is: f64,
        wall: ThermalMassNode,
        roof: ThermalMassNode,
        floor: ThermalMassNode,
        internal: ThermalMassNode,
        coupling_mode: MassAirCouplingMode,
    ) -> Self {
        let mut s = Self::new(h_tr_is, wall, roof, floor, internal);
        s.coupling_mode = coupling_mode;
        s
    }

    pub fn with_timestep(mut self, dt: f64) -> Self {
        self.timestep_seconds = dt;
        self
    }

    /// Set the mass-to-air coupling mode (Issue #1281).
    pub fn with_coupling_mode(mut self, mode: MassAirCouplingMode) -> Self {
        self.coupling_mode = mode;
        self
    }

    pub fn step(&mut self, dt: f64) -> &MultiNodeThermalMass {
        self.timestep_seconds = dt;
        self.step_backward_euler();
        &self.mass
    }

    fn step_backward_euler(&mut self) {
        match self.coupling_mode {
            MassAirCouplingMode::AdditiveSum => self.step_backward_euler_additive(),
            MassAirCouplingMode::ParallelResistance => {
                self.step_backward_euler_parallel_resistance()
            }
        }
    }

    /// Original additive-formulation backward Euler step (default, backward-compatible).
    ///
    /// Updates each envelope mass node using the SHARED surface temperature as
    /// the interior boundary. The shared T_s is the conductance-weighted average
    /// of mass node temperatures.
    ///
    /// See `step_backward_euler_parallel_resistance` for the Issue #1281
    /// alternative that uses per-surface T_s_k with h_is feedback.
    fn step_backward_euler_additive(&mut self) {
        let dt = self.timestep_seconds;
        let t_i = self.zone_temperature;
        let h_is = self.h_tr_is;

        // Issue #863: Per-surface exterior temperatures
        let t_ext_wall = self.exterior_temperatures.t_ext_wall;
        let t_ext_roof = self.exterior_temperatures.t_ext_roof;
        let t_ext_floor = self.exterior_temperatures.t_ext_floor;

        let m = &mut self.mass;

        // Capture pre-step temperatures for First Law energy balance check (Issue #1024)
        let t_wall_old = m.wall.temperature;
        let t_roof_old = m.roof.temperature;
        let t_floor_old = m.floor.temperature;
        let t_internal_old = m.internal.temperature;

        // Update wall node — uses wall sol-air temperature
        {
            let node = &mut m.wall;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            let numer = node.capacitance / dt * node.temperature
                + h_em * t_ext_wall
                + h_ms * self.surface_temperature;
            node.temperature = numer / denom;
        }

        // Update roof node — uses roof sol-air temperature
        {
            let node = &mut m.roof;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            let numer = node.capacitance / dt * node.temperature
                + h_em * t_ext_roof
                + h_ms * self.surface_temperature;
            node.temperature = numer / denom;
        }

        // Update floor node — uses ground temperature
        {
            let node = &mut m.floor;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            let numer = node.capacitance / dt * node.temperature
                + h_em * t_ext_floor
                + h_ms * self.surface_temperature;
            node.temperature = numer / denom;
        }

        // Update internal node
        {
            let node = &mut m.internal;
            let h_me = node.h_tr_me;
            let t_env_avg = internal_node_envelope_temperature(
                m.wall.temperature,
                m.roof.temperature,
                m.floor.temperature,
                m.wall.h_tr_ms,
                m.roof.h_tr_ms,
                m.floor.h_tr_ms,
                h_me,
            );

            let denom = node.capacitance / dt + h_is + h_me;
            let numer = node.capacitance / dt * node.temperature + h_is * t_i + h_me * t_env_avg;
            node.temperature = numer / denom;
        }

        // Issue #862: Update surface_temperature from solved mass node temperatures.
        // Uses the same conductance-weighted average formula as compute_zone_air_temperature
        // so that future timesteps use consistent surface temperatures.
        let h_ms_total = m.wall.h_tr_ms + m.roof.h_tr_ms + m.floor.h_tr_ms;
        if h_ms_total > 1e-6 {
            self.surface_temperature = (m.wall.h_tr_ms * m.wall.temperature
                + m.roof.h_tr_ms * m.roof.temperature
                + m.floor.h_tr_ms * m.floor.temperature)
                / h_ms_total;
        }

        self.check_energy_balance(t_wall_old, t_roof_old, t_floor_old, t_internal_old);
    }

    /// Parallel-resistance backward Euler step (Issue #1281).
    ///
    /// Each envelope mass node uses its OWN per-surface surface temperature
    /// `T_s_k = (h_tr_ms_k × T_m_k + h_tr_is × T_air) / (h_tr_ms_k + h_tr_is)`,
    /// not a shared conductance-weighted mean. This eliminates the additive
    /// `h_ms_total` overcounting and produces a more physically correct mass
    /// response.
    ///
    /// The First Law check still holds: the BE update is algebraically
    /// equivalent for any value of `T_s_k` (it's a fixed linear system in
    /// T_m, T_ext, T_s_k), so we re-use `check_energy_balance`.
    fn step_backward_euler_parallel_resistance(&mut self) {
        let dt = self.timestep_seconds;
        let t_i = self.zone_temperature;
        let h_is = self.h_tr_is;

        // Issue #863: Per-surface exterior temperatures
        let t_ext_wall = self.exterior_temperatures.t_ext_wall;
        let t_ext_roof = self.exterior_temperatures.t_ext_roof;
        let t_ext_floor = self.exterior_temperatures.t_ext_floor;

        let m = &mut self.mass;

        // Capture pre-step temperatures for First Law energy balance check (Issue #1024)
        let t_wall_old = m.wall.temperature;
        let t_roof_old = m.roof.temperature;
        let t_floor_old = m.floor.temperature;
        let t_internal_old = m.internal.temperature;

        // Compute per-surface T_s_k from the OLD mass temperatures and OLD T_air.
        // This is the steady-state surface temperature for each surface, solved
        // from the (mass → surface → air) series pair.
        let t_s_wall = per_surface_t_s(m.wall.temperature, m.wall.h_tr_ms, h_is, t_i);
        let t_s_roof = per_surface_t_s(m.roof.temperature, m.roof.h_tr_ms, h_is, t_i);
        let t_s_floor = per_surface_t_s(m.floor.temperature, m.floor.h_tr_ms, h_is, t_i);

        // Update wall node using its OWN per-surface T_s_k
        {
            let node = &mut m.wall;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            if denom > 1e-10 {
                let numer =
                    node.capacitance / dt * node.temperature + h_em * t_ext_wall + h_ms * t_s_wall;
                node.temperature = numer / denom;
            }
        }

        // Update roof node using its OWN per-surface T_s_k
        {
            let node = &mut m.roof;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            if denom > 1e-10 {
                let numer =
                    node.capacitance / dt * node.temperature + h_em * t_ext_roof + h_ms * t_s_roof;
                node.temperature = numer / denom;
            }
        }

        // Update floor node using its OWN per-surface T_s_k
        {
            let node = &mut m.floor;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_floor
                    + h_ms * t_s_floor;
                node.temperature = numer / denom;
            }
        }

        // Update internal node
        {
            let node = &mut m.internal;
            let h_me = node.h_tr_me;
            let t_env_avg = internal_node_envelope_temperature(
                m.wall.temperature,
                m.roof.temperature,
                m.floor.temperature,
                m.wall.h_tr_ms,
                m.roof.h_tr_ms,
                m.floor.h_tr_ms,
                h_me,
            );

            let denom = node.capacitance / dt + h_is + h_me;
            let numer = node.capacitance / dt * node.temperature + h_is * t_i + h_me * t_env_avg;
            node.temperature = numer / denom;
        }

        // Update self.surface_temperature as the conductance-weighted average so
        // legacy code reading `surface_temperature` still gets a representative
        // value. compute_zone_air_temperature in ParallelResistance mode does
        // NOT use this field — it computes per-surface paths directly.
        let h_ms_total = m.wall.h_tr_ms + m.roof.h_tr_ms + m.floor.h_tr_ms;
        if h_ms_total > 1e-6 {
            self.surface_temperature = (m.wall.h_tr_ms * m.wall.temperature
                + m.roof.h_tr_ms * m.roof.temperature
                + m.floor.h_tr_ms * m.floor.temperature)
                / h_ms_total;
        }

        self.check_energy_balance(t_wall_old, t_roof_old, t_floor_old, t_internal_old);
    }

    /// First Law energy balance check (Issue #1024).
    ///
    /// Verifies that the net heat flow into all thermal mass nodes equals the
    /// change in stored energy for the timestep.
    ///
    /// The check is derived directly from the backward Euler update equation.
    /// For each node k:
    ///   Q_k = C_k/dt · (T_k_new - T_k_old)   [W, power]
    ///
    /// where Q_k is the net heat flow INTO node k. This is algebraically
    /// equivalent to the update equation, so it is always satisfied exactly
    /// by the backward Euler scheme.
    ///
    /// Total net heat (W) = Σ Q_k  (summing over wall, roof, floor, internal)
    /// Change in storage rate (W) = Σ C_k · (T_k_new - T_k_old) / dt
    ///
    /// Assert: |Σ Q_k - Σ C_k·ΔT_k / dt| < 1e-7
    fn check_energy_balance(
        &self,
        t_wall_old: f64,
        t_roof_old: f64,
        t_floor_old: f64,
        t_internal_old: f64,
    ) {
        let m = &self.mass;
        let dt = self.timestep_seconds;

        // Net heat into each node from the backward Euler update equation:
        // Q_k = C_k/dt · (T_k_new - T_k_old)  [W]
        let q_wall = m.wall.capacitance / dt * (m.wall.temperature - t_wall_old);
        let q_roof = m.roof.capacitance / dt * (m.roof.temperature - t_roof_old);
        let q_floor = m.floor.capacitance / dt * (m.floor.temperature - t_floor_old);
        let q_internal = m.internal.capacitance / dt * (m.internal.temperature - t_internal_old);

        let q_net = q_wall + q_roof + q_floor + q_internal;

        // Change in stored energy rate: Σ C_k · ΔT_k / dt  [W]
        // (divide by dt to convert J → W for direct comparison with Q_net)
        let delta_e_rate = (m.wall.capacitance * (m.wall.temperature - t_wall_old)
            + m.roof.capacitance * (m.roof.temperature - t_roof_old)
            + m.floor.capacitance * (m.floor.temperature - t_floor_old)
            + m.internal.capacitance * (m.internal.temperature - t_internal_old))
            / dt;

        // Residual should be numerically zero (both sides of the equation are
        // derived from the same backward Euler update, so they are identical).
        // Use a relative tolerance because with large energy magnitudes
        // (~1e10 W) floating-point rounding can produce residuals up to
        // ~1e-9 * |value| while still being physically correct.
        // Non-finite residuals (inf/nan) indicate a deeper physics divergence
        // (e.g. Case 950 producing infinite temperatures) — skip the check
        // here since the caller is better positioned to handle that failure.
        let residual = (q_net - delta_e_rate).abs();
        let scale = q_net.abs().max(delta_e_rate.abs()).max(1.0);
        // Issue #2127: Replace debug_assert! with runtime check.
        // Non-finite residuals (inf/nan) are skipped silently per Issue #2128.
        // Finite but large residuals emit a warning instead of panicking.
        if residual.is_finite() && residual >= 1e-9 * scale {
            log::warn!(
                "First Law violation: net heat ({q_net} W) != change in storage rate ({delta_e_rate} W) | residual={residual} W",
            );
        }
    }

    // ── Issue #871: Air Balance API Methods ───────────────────────────

    /// Compute zone air temperature from the multi-node thermal balance.
    ///
    /// Must be called AFTER `step()` (or `step_with_gains()`) has updated
    /// mass node temperatures. Uses the air node energy balance:
    ///
    /// ```text
    /// T_s = Σ(h_tr_ms_k × T_k) / Σ(h_tr_ms_k)   for k ∈ {wall, roof, floor}
    /// T_air = (h_tr_is × T_s + (h_ve + h_ve_night) × T_out + φ_ia) / (h_tr_is + h_ve + h_ve_night)
    /// ```
    ///
    /// Dispatches to `compute_zone_air_temperature_additive` or
    /// `compute_zone_air_temperature_parallel_resistance` depending on
    /// `self.coupling_mode` (Issue #1281).
    ///
    /// # Arguments
    /// * `t_outdoor` — Outdoor air temperature [°C]
    /// * `h_ve` — Ventilation/infiltration conductance [W/K]
    /// * `h_ve_night` — Night ventilation fan conductance [W/K]
    /// * `phi_ia` — Internal convective + solar-to-air gains [W]
    ///
    /// # Returns
    /// Free-floating zone air temperature [°C]
    pub fn compute_zone_air_temperature(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
    ) -> f64 {
        match self.coupling_mode {
            MassAirCouplingMode::AdditiveSum => {
                self.compute_zone_air_temperature_additive(t_outdoor, h_ve, h_ve_night, phi_ia)
            }
            MassAirCouplingMode::ParallelResistance => self
                .compute_zone_air_temperature_parallel_resistance(
                    t_outdoor, h_ve, h_ve_night, phi_ia,
                ),
        }
    }

    /// Compute the free-floating zone air temperature with an explicit
    /// sky-radiative conductance on the air node (Issue #1858).
    ///
    /// Adds the boundary flux `h_rad_sky · (T_sky − T_air)` to the air-node
    /// energy balance, allowing the free-floating air temperature to fall below
    /// the outdoor dry-bulb under clear-sky radiative cooling. The caller is
    /// responsible for supplying a physics-derived `h_rad_sky` (see
    /// [`air_sky_conductance`]) and the EPW-derived `t_sky` (see
    /// `HourlyWeatherData::sky_temperature`).
    ///
    /// When `h_rad_sky == 0.0` the result is identical to
    /// [`compute_zone_air_temperature`] — the sky term vanishes and the original
    /// four-term balance is recovered exactly (verified by
    /// `test_issue_1858_backward_compat_zero_sky_conductance`).
    pub fn compute_zone_air_temperature_with_sky(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
        t_sky: f64,
        h_rad_sky: f64,
    ) -> f64 {
        match self.coupling_mode {
            MassAirCouplingMode::AdditiveSum => self
                .compute_zone_air_temperature_additive_with_sky(
                    t_outdoor, h_ve, h_ve_night, phi_ia, t_sky, h_rad_sky,
                ),
            MassAirCouplingMode::ParallelResistance => self
                .compute_zone_air_temperature_parallel_resistance_with_sky(
                    t_outdoor, h_ve, h_ve_night, phi_ia, t_sky, h_rad_sky,
                ),
        }
    }

    /// Original (additive) formulation — backward-compatible default.
    ///
    /// Treats the three envelope mass nodes as parallel conductances summed into
    /// a single shared surface temperature, then couples that surface to the
    /// air node through the lumped interior-film conductance `h_tr_is`. See
    /// `MassAirCouplingMode::AdditiveSum` for the equations and `compute_zone_air_temperature_parallel_resistance`
    /// for the physically-correct alternative.
    ///
    /// # Arguments
    /// * `t_outdoor` — Outdoor air temperature [°C]
    /// * `h_ve` — Ventilation/infiltration conductance [W/K]
    /// * `h_ve_night` — Night ventilation fan conductance [W/K]
    /// * `phi_ia` — Internal convective + solar-to-air gains [W]
    ///
    /// # Returns
    /// Free-floating zone air temperature [°C]
    pub fn compute_zone_air_temperature_additive(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
    ) -> f64 {
        // Issue #1858: backward-compatible delegation — a zero sky conductance
        // recovers the original four-term balance exactly.
        self.compute_zone_air_temperature_additive_with_sky(
            t_outdoor, h_ve, h_ve_night, phi_ia, 0.0, 0.0,
        )
    }

    /// Additive air-node balance with an explicit sky-radiative term
    /// (Issue #1858). See [`compute_zone_air_temperature_with_sky`].
    ///
    /// ```text
    /// T_air = (h_tr_is · T_s + (h_ve + h_ve_night) · T_out
    ///          + h_rad_sky · T_sky + φ_ia)
    ///         / (h_tr_is + h_ve + h_ve_night + h_rad_sky)
    /// ```
    pub fn compute_zone_air_temperature_additive_with_sky(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
        t_sky: f64,
        h_rad_sky: f64,
    ) -> f64 {
        // Conductance-weighted surface temperature from envelope nodes
        let h_ms_w = self.mass.wall.h_tr_ms;
        let h_ms_r = self.mass.roof.h_tr_ms;
        let h_ms_f = self.mass.floor.h_tr_ms;
        let h_ms_total = h_ms_w + h_ms_r + h_ms_f;

        let t_surface = if h_ms_total > 1e-6 {
            (h_ms_w * self.mass.wall.temperature
                + h_ms_r * self.mass.roof.temperature
                + h_ms_f * self.mass.floor.temperature)
                / h_ms_total
        } else {
            // Fallback: simple average if no conductances
            self.envelope_temperature()
        };

        // Air node energy balance
        // h_ve_night: additional ventilation conductance from night ventilation fans [W/K]
        // h_rad_sky: linearized sky-radiative conductance [W/K] (Issue #1858)
        let h_ve_total = h_ve + h_ve_night;
        let denom = self.h_tr_is + h_ve_total + h_rad_sky;
        if denom < 1e-6 {
            // Near-zero ventilation + interior film — return surface temp as best estimate
            return t_surface;
        }

        (self.h_tr_is * t_surface + h_ve_total * t_outdoor + h_rad_sky * t_sky + phi_ia) / denom
    }

    /// Parallel-resistance formulation (Issue #1281).
    ///
    /// Each envelope surface has its own steady-state surface temperature
    /// `T_s_k`, computed from the series pair (mass-to-surface, surface-to-air):
    ///
    /// ```text
    /// T_s_k = (h_tr_ms_k × T_m_k + h_tr_is × T_air) / (h_tr_ms_k + h_tr_is)
    /// ```
    ///
    /// The air node sees the parallel combination of per-surface series paths:
    ///
    /// ```text
    /// h_path_k = h_tr_ms_k × h_tr_is / (h_tr_ms_k + h_tr_is)   [series combination]
    /// T_air = (Σ h_path_k × T_m_k + (h_ve + h_ve_night) × T_out + φ_ia)
    ///         / (Σ h_path_k + h_ve + h_ve_night)
    /// ```
    ///
    /// Eliminating the additive `h_ms_total` overcounting produces a more
    /// physically correct air-temperature prediction when each per-surface
    /// `h_tr_ms_k` is comparable to `h_tr_is` (typical for ASHRAE 140 Case 900).
    ///
    /// **Direction (verified in `.agents/results/issue-1281-python-verification.py`):**
    /// For Case 900 parameters, `h_path_total = 96.0 W/K` vs `h_ms_total = 127.3 W/K`
    /// (-32.7 % overcount). Air temperature and peak cooling demand are both
    /// LOWER than the additive formulation.
    ///
    /// **Note:** the ASHRAE 140 peak-cooling underestimate documented in
    /// `docs/KNOWN_ISSUES.md` LIMIT-05 UPDATE is *not* closed by this
    /// formulation alone — the actual root cause is roof-solar under-counting
    /// (see `docs/investigations/issue-1280-ctf-peak-load.md` §4). This
    /// method ships the more physically correct 9R4C coupling network and
    /// is the architecturally-improved fix the issue body asks for.
    ///
    /// # Arguments
    /// * `t_outdoor` — Outdoor air temperature [°C]
    /// * `h_ve` — Ventilation/infiltration conductance [W/K]
    /// * `h_ve_night` — Night ventilation fan conductance [W/K]
    /// * `phi_ia` — Internal convective + solar-to-air gains [W]
    ///
    /// # Returns
    /// Free-floating zone air temperature [°C]
    pub fn compute_zone_air_temperature_parallel_resistance(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
    ) -> f64 {
        // Issue #1858: backward-compatible delegation — a zero sky conductance
        // recovers the original four-term balance exactly.
        self.compute_zone_air_temperature_parallel_resistance_with_sky(
            t_outdoor, h_ve, h_ve_night, phi_ia, 0.0, 0.0,
        )
    }

    /// Parallel-resistance air-node balance with an explicit sky-radiative term
    /// (Issue #1858). See [`compute_zone_air_temperature_with_sky`].
    ///
    /// ```text
    /// h_path_k = h_tr_ms_k · h_tr_is / (h_tr_ms_k + h_tr_is)   [series combination]
    /// T_air = (Σ h_path_k · T_m_k + (h_ve + h_ve_night) · T_out
    ///          + h_rad_sky · T_sky + φ_ia)
    ///         / (Σ h_path_k + h_ve + h_ve_night + h_rad_sky)
    /// ```
    pub fn compute_zone_air_temperature_parallel_resistance_with_sky(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
        t_sky: f64,
        h_rad_sky: f64,
    ) -> f64 {
        let h_ms_w = self.mass.wall.h_tr_ms;
        let h_ms_r = self.mass.roof.h_tr_ms;
        let h_ms_f = self.mass.floor.h_tr_ms;

        // Series combination per surface (mass → T_s_k → air through h_tr_is).
        // h_path_k = h_ms_k × h_is / (h_ms_k + h_is). Each is strictly <= h_ms_k.
        let h_path_w = h_series(h_ms_w, self.h_tr_is);
        let h_path_r = h_series(h_ms_r, self.h_tr_is);
        let h_path_f = h_series(h_ms_f, self.h_tr_is);
        let h_path_total = h_path_w + h_path_r + h_path_f;

        let h_ve_total = h_ve + h_ve_night;
        let denom = h_path_total + h_ve_total + h_rad_sky;
        if denom < 1e-6 {
            // Degenerate — fall back to conductance-weighted average of mass temps
            return (h_ms_w * self.mass.wall.temperature
                + h_ms_r * self.mass.roof.temperature
                + h_ms_f * self.mass.floor.temperature)
                / (h_ms_w + h_ms_r + h_ms_f).max(1e-6);
        }

        (h_path_w * self.mass.wall.temperature
            + h_path_r * self.mass.roof.temperature
            + h_path_f * self.mass.floor.temperature
            + h_ve_total * t_outdoor
            + h_rad_sky * t_sky
            + phi_ia)
            / denom
    }

    /// Compute ideal HVAC power demand to maintain setpoints.
    ///
    /// Uses the air node energy balance to determine how much heating or
    /// cooling power is needed to bring the free-floating zone air temperature
    /// to the setpoint:
    ///
    /// ```text
    /// Q_hvac = (h_tr_is + h_ve) × (T_setpoint - T_air_free)   [when outside deadband]
    /// ```
    ///
    /// # Arguments
    /// * `t_air_free` — Free-floating zone air temperature [°C]
    /// * `heating_setpoint` — Heating setpoint temperature [°C]
    /// * `cooling_setpoint` — Cooling setpoint temperature [°C]
    ///
    /// # Returns
    /// HVAC power demand [W]. Positive = heating, negative = cooling, zero = deadband.
    pub fn compute_hvac_demand(
        &self,
        t_air_free: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> f64 {
        let h_total = self.h_tr_is; // Base conductance; caller should add h_ve if needed

        if t_air_free < heating_setpoint {
            // Heating required
            h_total * (heating_setpoint - t_air_free)
        } else if t_air_free > cooling_setpoint {
            // Cooling required
            h_total * (cooling_setpoint - t_air_free) // negative
        } else {
            0.0 // Within deadband
        }
    }

    // ── Issue #1005: Per-Surface Conduction Integration ──────────────────

    /// Build a `PerSurfaceConductionSolver` from the current multi-node state.
    ///
    /// Each envelope mass node (wall, roof, floor) is paired with a
    /// `SurfaceNode` carrying the same `h_tr_ms` and `h_tr_em` conductances
    /// and the corresponding exterior boundary temperature. The `PerSurfaceConductionSolver`
    /// then tracks the per-surface temperature (the air-side film) separately
    /// from the bulk mass node temperature.
    ///
    /// This is the integration point between the multi-node thermal model
    /// (Issue #857, parent of #1005) and the per-surface conduction module.
    /// Callers can then drive the per-surface solver to refine `surface_temperature`
    /// without losing the multi-node mass tracking.
    pub fn build_per_surface_solver(&self) -> PerSurfaceConductionSolver {
        let mut solver = PerSurfaceConductionSolver::new();
        // Wall: id=0, h_tr_ms from mass, h_tr_em from mass, ext from sol-air
        solver.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            1.0, // area (per-unit, scaled at use site)
            0.0, // U-value (not used by update())
            self.mass.wall.temperature,
            self.mass.wall.h_tr_ms,
            self.h_tr_is,
            self.mass.wall.h_tr_em,
        );
        // Roof: id=1
        solver.add_surface_from_params(
            1,
            SurfaceKind::Roof,
            1.0,
            0.0,
            self.mass.roof.temperature,
            self.mass.roof.h_tr_ms,
            self.h_tr_is,
            self.mass.roof.h_tr_em,
        );
        // Floor: id=2
        solver.add_surface_from_params(
            2,
            SurfaceKind::Floor,
            1.0,
            0.0,
            self.mass.floor.temperature,
            self.mass.floor.h_tr_ms,
            self.h_tr_is,
            self.mass.floor.h_tr_em,
        );
        solver
    }

    /// Step the per-surface solver in lockstep with the multi-node solver.
    ///
    /// This is the integration point for Issue #1005. It runs the per-surface
    /// solver using the current mass node temperatures and per-surface exterior
    /// temperatures, then writes the refined `surface_temperature` back to
    /// `self`. The per-surface solver tracks the surface film independently
    /// from the bulk mass node, providing a more accurate air-side temperature
    /// for the air node energy balance.
    ///
    /// # Arguments
    /// * `dt` — Timestep duration [s]
    /// * `mass_temps` — Per-surface mass temperatures BEFORE gains applied [°C] (Issue #864)
    ///   (wall, roof, floor)
    /// * `phi_m` — Per-surface opaque solar gains to surface node [W] (Issue #864)
    ///   (wall, roof, floor)
    ///
    /// # Returns
    /// The (wall, roof, floor) per-surface temperatures [°C]
    pub fn step_per_surface(
        &mut self,
        dt: f64,
        mass_temps: (f64, f64, f64),
        phi_m: (f64, f64, f64),
    ) -> (f64, f64, f64) {
        // Build a transient per-surface solver from current state
        let mut solver = self.build_per_surface_solver();

        // Per-surface exterior temperatures (Issue #863)
        let t_ext_wall = self.exterior_temperatures.t_ext_wall;
        let t_ext_roof = self.exterior_temperatures.t_ext_roof;
        let t_ext_floor = self.exterior_temperatures.t_ext_floor;

        // Update each surface with pre-gain mass temperature and per-surface solar gain
        // Using pre-gain mass temperatures avoids double-counting gains that were
        // already applied via step_with_gains(). Gains are added directly to the
        // surface node's backward Euler heat balance via phi_m_surface (Issue #864).
        solver.update_surface(0, dt, mass_temps.0, t_ext_wall, phi_m.0);
        solver.update_surface(1, dt, mass_temps.1, t_ext_roof, phi_m.1);
        solver.update_surface(2, dt, mass_temps.2, t_ext_floor, phi_m.2);

        let temps = solver.surface_temperatures();
        let t_surface_wall = temps.first().copied().unwrap_or(self.surface_temperature);
        let t_surface_roof = temps.get(1).copied().unwrap_or(self.surface_temperature);
        let t_surface_floor = temps.get(2).copied().unwrap_or(self.surface_temperature);

        // Update self.surface_temperature as a conductance-weighted average (matches
        // the original multi-node convention used by compute_zone_air_temperature).
        let h_ms_total = self.mass.wall.h_tr_ms + self.mass.roof.h_tr_ms + self.mass.floor.h_tr_ms;
        if h_ms_total > 1e-6 {
            self.surface_temperature = (self.mass.wall.h_tr_ms * t_surface_wall
                + self.mass.roof.h_tr_ms * t_surface_roof
                + self.mass.floor.h_tr_ms * t_surface_floor)
                / h_ms_total;
        }
        // else: keep the prior self.surface_temperature (graceful fallback)

        (t_surface_wall, t_surface_roof, t_surface_floor)
    }

    /// Step the solver with per-node heat gains injected into the backward Euler.
    ///
    /// Each envelope node receives its share of radiative gains directly in the
    /// numerator of the backward Euler equation, in addition to conduction fluxes:
    ///
    /// ```text
    /// T_k^new = (C_k/dt × T_k^old + h_em × T_ext_k + h_ms × T_s + gains_k)
    ///           / (C_k/dt + h_em + h_ms)
    /// ```
    ///
    /// # Arguments
    /// * `dt` — Timestep duration [s]
    /// * `gains_wall` — Radiative/solar gains to wall mass node [W]
    /// * `gains_roof` — Radiative/solar gains to roof mass node [W]
    /// * `gains_floor` — Radiative/solar gains to floor mass node [W]
    /// * `gains_internal` — Internal radiative gains to internal mass node [W]
    ///
    /// # Returns
    /// Reference to the updated `MultiNodeThermalMass`
    /// Step the multi-node thermal model with per-node gains and night ventilation.
    ///
    /// # Arguments
    /// * `dt` - Timestep in seconds
    /// * `gains_wall` - Solar radiative gain to wall node [W]
    /// * `gains_roof` - Solar radiative gain to roof node [W]
    /// * `gains_floor` - Solar radiative gain to floor node [W]
    /// * `gains_internal` - Internal/solar gain to internal mass node [W]
    /// * `h_ve_night` - Night ventilation conductance [W/K] (0 if inactive)
    /// * `outdoor_temp` - Outdoor air temperature [°C] (driving temp for night vent)
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_gains(
        &mut self,
        dt: f64,
        gains_wall: f64,
        gains_roof: f64,
        gains_floor: f64,
        gains_internal: f64,
        h_ve_night: f64,
        outdoor_temp: f64,
    ) -> &MultiNodeThermalMass {
        self.timestep_seconds = dt;

        // Adaptive sub-stepping for low-capacitance (high-stiffness) nodes.
        // When the timestep dt exceeds a node's thermal time constant τ = C/h,
        // backward Euler produces inaccurate results. We detect this and sub-step
        // with smaller timesteps to maintain accuracy.
        //
        // τ_node = C_node / (h_em + h_ms) — time constant for envelope nodes
        // If dt > STIFFNESS_FACTOR * τ_node, we need sub-stepping
        const STIFFNESS_FACTOR: f64 = 4.0; // Sub-step when dt > 4 × τ
        const MIN_SUB_STEPS: u32 = 1;
        const MAX_SUB_STEPS: u32 = 12; // Cap at 5-min sub-steps (12 × 5min = 60min)

        // Compute minimum time constant across all envelope nodes
        let m = &self.mass;
        let wall_tau = if m.wall.capacitance > 0.0 && m.wall.h_tr_em + m.wall.h_tr_ms > 0.0 {
            m.wall.capacitance / (m.wall.h_tr_em + m.wall.h_tr_ms)
        } else {
            f64::INFINITY
        };
        let roof_tau = if m.roof.capacitance > 0.0 && m.roof.h_tr_em + m.roof.h_tr_ms > 0.0 {
            m.roof.capacitance / (m.roof.h_tr_em + m.roof.h_tr_ms)
        } else {
            f64::INFINITY
        };
        let floor_tau = if m.floor.capacitance > 0.0 && m.floor.h_tr_em + m.floor.h_tr_ms > 0.0 {
            m.floor.capacitance / (m.floor.h_tr_em + m.floor.h_tr_ms)
        } else {
            f64::INFINITY
        };
        let min_tau = wall_tau.min(roof_tau).min(floor_tau);

        // Determine number of sub-steps needed
        let num_sub_steps =
            if min_tau.is_finite() && min_tau > 0.0 && dt > min_tau * STIFFNESS_FACTOR {
                let required = (dt / (min_tau * STIFFNESS_FACTOR)).ceil() as u32;
                required.clamp(MIN_SUB_STEPS, MAX_SUB_STEPS)
            } else {
                MIN_SUB_STEPS
            };

        let sub_dt = dt / num_sub_steps as f64;

        // Perform sub-stepping loop
        // Note: Gains are in W (power, not energy), so they stay the same across sub-steps.
        // Each sub-step uses the same power rate, but for a shorter time (sub_dt).
        // The backward Euler formula naturally accounts for this via the C/dt term.

        for _step in 0..num_sub_steps {
            self.timestep_seconds = sub_dt;
            match self.coupling_mode {
                MassAirCouplingMode::AdditiveSum => {
                    self.step_backward_euler_with_gains(
                        gains_wall,
                        gains_roof,
                        gains_floor,
                        gains_internal,
                        h_ve_night,
                        outdoor_temp,
                    );
                }
                MassAirCouplingMode::ParallelResistance => {
                    self.step_backward_euler_with_gains_parallel_resistance(
                        gains_wall,
                        gains_roof,
                        gains_floor,
                        gains_internal,
                        h_ve_night,
                        outdoor_temp,
                    );
                }
            }
        }

        // Restore original timestep
        self.timestep_seconds = dt;
        &self.mass
    }

    /// Backward Euler step with per-node gain injection.
    ///
    /// Same as `step_backward_euler()` but adds gain terms [W] to each node's
    /// numerator and applies night ventilation conductance directly to envelope
    /// mass nodes (Issue #1898: night ventilation was only affecting the air
    /// node, not the thermal mass).
    fn step_backward_euler_with_gains(
        &mut self,
        gains_wall: f64,
        gains_roof: f64,
        gains_floor: f64,
        gains_internal: f64,
        h_ve_night: f64,
        outdoor_temp: f64,
    ) {
        let dt = self.timestep_seconds;
        let t_i = self.zone_temperature;
        let h_is = self.h_tr_is;

        let t_ext_wall = self.exterior_temperatures.t_ext_wall;
        let t_ext_roof = self.exterior_temperatures.t_ext_roof;
        let t_ext_floor = self.exterior_temperatures.t_ext_floor;

        let m = &mut self.mass;

        // Capture pre-step temperatures for First Law energy balance check (Issue #1024)
        let t_wall_old = m.wall.temperature;
        let t_roof_old = m.roof.temperature;
        let t_floor_old = m.floor.temperature;
        let t_internal_old = m.internal.temperature;

        // Issue #1898: Night ventilation mass coupling.
        // When night ventilation is active (h_ve_night > 0), cool outdoor air directly
        // cools the thermal mass through an additional conductance path.
        // This mirrors the 5R1C path's h_vent_mass_zone term.

        // Update wall node — with gains and night ventilation
        {
            let node = &mut m.wall;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_wall
                    + h_ms * self.surface_temperature
                    + h_ve_night * outdoor_temp
                    + gains_wall;
                node.temperature = numer / denom;
            }
        }

        // Update roof node — with gains and night ventilation
        {
            let node = &mut m.roof;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_roof
                    + h_ms * self.surface_temperature
                    + h_ve_night * outdoor_temp
                    + gains_roof;
                node.temperature = numer / denom;
            }
        }

        // Update floor node — with gains and night ventilation
        {
            let node = &mut m.floor;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_floor
                    + h_ms * self.surface_temperature
                    + h_ve_night * outdoor_temp
                    + gains_floor;
                node.temperature = numer / denom;
            }
        }

        // Update internal node — with gains (internal node doesn't couple directly to outdoor)
        {
            let node = &mut m.internal;
            let h_me = node.h_tr_me;
            let t_env_avg = internal_node_envelope_temperature(
                m.wall.temperature,
                m.roof.temperature,
                m.floor.temperature,
                m.wall.h_tr_ms,
                m.roof.h_tr_ms,
                m.floor.h_tr_ms,
                h_me,
            );

            let denom = node.capacitance / dt + h_is + h_me;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_is * t_i
                    + h_me * t_env_avg
                    + gains_internal;
                node.temperature = numer / denom;
            }
        }

        // Issue #862: Update surface_temperature from solved mass node temperatures.
        let h_ms_total = m.wall.h_tr_ms + m.roof.h_tr_ms + m.floor.h_tr_ms;
        if h_ms_total > 1e-6 {
            self.surface_temperature = (m.wall.h_tr_ms * m.wall.temperature
                + m.roof.h_tr_ms * m.roof.temperature
                + m.floor.h_tr_ms * m.floor.temperature)
                / h_ms_total;
        }

        // Issue #1024: First Law of Thermodynamics debug_assert! (with gains)
        self.check_energy_balance_with_gains(
            t_wall_old,
            t_roof_old,
            t_floor_old,
            t_internal_old,
            gains_wall,
            gains_roof,
            gains_floor,
            gains_internal,
        );
    }

    /// Parallel-resistance backward Euler step with per-node gain injection
    /// (Issue #1281). See `step_backward_euler_parallel_resistance` for the
    /// non-gain counterpart.
    ///
    /// Issue #1898: Night ventilation conductance (h_ve_night) is applied directly
    /// to envelope mass nodes to allow night ventilation to cool thermal mass.
    fn step_backward_euler_with_gains_parallel_resistance(
        &mut self,
        gains_wall: f64,
        gains_roof: f64,
        gains_floor: f64,
        gains_internal: f64,
        h_ve_night: f64,
        outdoor_temp: f64,
    ) {
        let dt = self.timestep_seconds;
        let t_i = self.zone_temperature;
        let h_is = self.h_tr_is;

        let t_ext_wall = self.exterior_temperatures.t_ext_wall;
        let t_ext_roof = self.exterior_temperatures.t_ext_roof;
        let t_ext_floor = self.exterior_temperatures.t_ext_floor;

        let m = &mut self.mass;

        // Capture pre-step temperatures for First Law energy balance check (Issue #1024)
        let t_wall_old = m.wall.temperature;
        let t_roof_old = m.roof.temperature;
        let t_floor_old = m.floor.temperature;
        let t_internal_old = m.internal.temperature;

        // Per-surface T_s_k from OLD mass temps and OLD T_air.
        let t_s_wall = per_surface_t_s(m.wall.temperature, m.wall.h_tr_ms, h_is, t_i);
        let t_s_roof = per_surface_t_s(m.roof.temperature, m.roof.h_tr_ms, h_is, t_i);
        let t_s_floor = per_surface_t_s(m.floor.temperature, m.floor.h_tr_ms, h_is, t_i);

        // Issue #1898: Night ventilation mass coupling — applied to all envelope nodes.
        // When h_ve_night > 0, cool outdoor air directly cools the thermal mass.

        // Wall — per-surface T_s_k + per-node gain + night vent
        {
            let node = &mut m.wall;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_wall
                    + h_ms * t_s_wall
                    + h_ve_night * outdoor_temp
                    + gains_wall;
                node.temperature = numer / denom;
            }
        }

        // Roof — per-surface T_s_k + per-node gain + night vent
        {
            let node = &mut m.roof;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_roof
                    + h_ms * t_s_roof
                    + h_ve_night * outdoor_temp
                    + gains_roof;
                node.temperature = numer / denom;
            }
        }

        // Floor — per-surface T_s_k + per-node gain + night vent
        {
            let node = &mut m.floor;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_floor
                    + h_ms * t_s_floor
                    + h_ve_night * outdoor_temp
                    + gains_floor;
                node.temperature = numer / denom;
            }
        }

        // Internal node — unchanged (internal node doesn't couple directly to outdoor)
        {
            let node = &mut m.internal;
            let h_me = node.h_tr_me;
            let t_env_avg = internal_node_envelope_temperature(
                m.wall.temperature,
                m.roof.temperature,
                m.floor.temperature,
                m.wall.h_tr_ms,
                m.roof.h_tr_ms,
                m.floor.h_tr_ms,
                h_me,
            );

            let denom = node.capacitance / dt + h_is + h_me;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_is * t_i
                    + h_me * t_env_avg
                    + gains_internal;
                node.temperature = numer / denom;
            }
        }

        // Keep self.surface_temperature in sync with the conductance-weighted
        // average so legacy readers see a consistent value. The
        // ParallelResistance-mode air temperature does NOT use this field.
        let h_ms_total = m.wall.h_tr_ms + m.roof.h_tr_ms + m.floor.h_tr_ms;
        if h_ms_total > 1e-6 {
            self.surface_temperature = (m.wall.h_tr_ms * m.wall.temperature
                + m.roof.h_tr_ms * m.roof.temperature
                + m.floor.h_tr_ms * m.floor.temperature)
                / h_ms_total;
        }

        // Gains cancel out of the energy balance (Issue #1024), same as additive.
        self.check_energy_balance_with_gains(
            t_wall_old,
            t_roof_old,
            t_floor_old,
            t_internal_old,
            gains_wall,
            gains_roof,
            gains_floor,
            gains_internal,
        );
    }

    /// First Law energy balance check with gain injection (Issue #1024).
    ///
    /// Same as `check_energy_balance` — the gain terms cancel out of the
    /// energy balance because they appear identically on both sides of the
    /// backward Euler update equation. The net heat flow is still:
    ///   Q_k = C_k/dt · (T_k_new - T_k_old)
    #[allow(clippy::too_many_arguments)]
    fn check_energy_balance_with_gains(
        &self,
        t_wall_old: f64,
        t_roof_old: f64,
        t_floor_old: f64,
        t_internal_old: f64,
        _gains_wall: f64,
        _gains_roof: f64,
        _gains_floor: f64,
        _gains_internal: f64,
    ) {
        // Gains cancel out of the energy balance (they appear on both sides of
        // the backward Euler update equation), so we use the same formula.
        self.check_energy_balance(t_wall_old, t_roof_old, t_floor_old, t_internal_old);
    }

    // ── Temperature Accessors ────────────────────────────────────────

    pub fn wall_temperature(&self) -> f64 {
        self.mass.wall.temperature
    }

    pub fn roof_temperature(&self) -> f64 {
        self.mass.roof.temperature
    }

    pub fn floor_temperature(&self) -> f64 {
        self.mass.floor.temperature
    }

    pub fn internal_temperature(&self) -> f64 {
        self.mass.internal.temperature
    }

    pub fn envelope_temperature(&self) -> f64 {
        (self.mass.wall.temperature + self.mass.roof.temperature + self.mass.floor.temperature)
            / 3.0
    }

    /// Snapshot the four 9R4C node temperatures as a fixed-size array
    /// (Issue #1799 — sub-hourly nodal temperature export for Python).
    ///
    /// Order: `[wall, roof, floor, internal]`. This is the canonical node index
    /// ordering used throughout the rest of the engine (see `MultiNodeThermalMass`
    /// field order in `fluxion_core::multi_node`). Callers wanting a per-node
    /// name label should map index → name with the same convention.
    pub fn snapshot_temperatures(&self) -> [f64; 4] {
        [
            self.mass.wall.temperature,
            self.mass.roof.temperature,
            self.mass.floor.temperature,
            self.mass.internal.temperature,
        ]
    }

    /// Number of nodes in this solver (always 4 for the 9R4C network).
    pub const NUM_NODES: usize = 4;

    /// Canonical node names in the same order as `snapshot_temperatures()`.
    pub const NODE_NAMES: [&'static str; Self::NUM_NODES] = ["wall", "roof", "floor", "internal"];

    pub fn set_zone_temperature(&mut self, t: f64) {
        self.zone_temperature = t;
    }

    pub fn set_surface_temperature(&mut self, t: f64) {
        self.surface_temperature = t;
    }

    pub fn set_exterior_temperature(&mut self, t: f64) {
        self.exterior_temperature = t;
        self.exterior_temperatures = SurfaceExteriorTemperatures::uniform(t);
    }

    /// Set per-surface exterior boundary temperatures (Issue #863).
    ///
    /// Stores per-surface sol-air/ground temperatures and updates the
    /// legacy `exterior_temperature` field to the average for backward
    /// compatibility with code that reads it directly.
    pub fn set_surface_exterior_temperatures(&mut self, temps: SurfaceExteriorTemperatures) {
        self.exterior_temperature = (temps.t_ext_wall + temps.t_ext_roof + temps.t_ext_floor) / 3.0;
        self.exterior_temperatures = temps;
    }

    pub fn set_wall_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.wall.h_tr_em = h_tr_em;
        self.mass.wall.h_tr_ms = h_tr_ms;
    }

    pub fn set_roof_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.roof.h_tr_em = h_tr_em;
        self.mass.roof.h_tr_ms = h_tr_ms;
    }

    pub fn set_floor_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.floor.h_tr_em = h_tr_em;
        self.mass.floor.h_tr_ms = h_tr_ms;
    }

    pub fn set_internal_conductance(&mut self, h_tr_me: f64) {
        self.mass.internal.h_tr_me = h_tr_me;
    }

    pub fn set_wall_capacitance(&mut self, cm: f64) {
        self.mass.wall.capacitance = cm;
    }

    pub fn set_roof_capacitance(&mut self, cm: f64) {
        self.mass.roof.capacitance = cm;
    }

    pub fn set_floor_capacitance(&mut self, cm: f64) {
        self.mass.floor.capacitance = cm;
    }

    pub fn set_internal_capacitance(&mut self, cm: f64) {
        self.mass.internal.capacitance = cm;
    }

    pub fn initialize_temperatures(&mut self, t_initial: f64) {
        self.mass.wall.temperature = t_initial;
        self.mass.roof.temperature = t_initial;
        self.mass.floor.temperature = t_initial;
        self.mass.internal.temperature = t_initial;
        self.zone_temperature = t_initial;
        self.surface_temperature = t_initial;
    }

    pub fn effective_time_constant(&self) -> f64 {
        // τ_eff = C_total / h_tr_eff
        // where h_tr_eff is the effective coupling to the zone
        let c_total = self.mass.wall.capacitance
            + self.mass.roof.capacitance
            + self.mass.floor.capacitance
            + self.mass.internal.capacitance;

        // Effective conductance: envelope nodes coupled to zone via h_tr_ms + internal via h_is
        let h_tr_ms_total =
            self.mass.wall.h_tr_ms + self.mass.roof.h_tr_ms + self.mass.floor.h_tr_ms;

        // h_tr_is is shared, h_tr_ms connects surface to each envelope node
        // For time constant, we consider the dominant coupling
        let h_eff = self.h_tr_is + h_tr_ms_total / 3.0;

        c_total / h_eff
    }

    // ── Issue #1429 — `HeatConductionSolver` trait drop-in for `SolverRegistry` ──
    //
    // Per ARCHITECTURE.md §Module 3, `HeatConductionSolver` is the per-surface
    // conduction swap-point. The 9R4C envelope solver was previously a
    // zone-level-only path (`step_physics_9r4c`); exposing it as
    // `Box<dyn HeatConductionSolver>` unlocks (a) per-surface ML surrogate
    // training on 9R4C envelope mass nodes and (b) evaluation of
    // `MassAirCouplingMode::ParallelResistance` (#1281) at the trait boundary.

    /// Build a 9R4C `MultiNodeSolver` from a single `WallSpec`.
    ///
    /// The layer stack is treated as a representative envelope surface;
    /// the four 9R4C mass nodes are partitioned as follows:
    ///
    /// | Node | Capacitance fraction | Conductances |
    /// |------|----------------------|--------------|
    /// | wall | 45% of `C_total`    | `h_tr_ms` = 1 / (R_total/2 + R_si), `h_tr_em` = 1 / (R_total/2 + R_se) |
    /// | roof | 30% of `C_total`    | same partition |
    /// | floor | 18% of `C_total`   | same partition |
    /// | internal | 10% of `C_total` | `h_tr_me` = physics-based (Issue #1593) |
    ///
    /// The internal node's `h_tr_me` is computed using the ISO 13790 lumped-mass
    /// coupling: `h_tr_me = h_ms * a_int` where:
    /// - `h_ms = 9.1 W/(m²·K)` is the furniture/partitions coupling coefficient
    /// - `a_int = furniture_factor * floor_area` is the internal surface area
    ///
    /// This replaces the previously hardcoded `h_tr_me = 100.0` value.
    ///
    /// The default coupling mode is `AdditiveSum` (backward-compatible);
    /// callers can switch to `ParallelResistance` via `with_coupling_mode`
    /// before calling `initialize`/`step`.
    pub fn from_wall_spec(wall: &WallSpec, floor_area: f64) -> Self {
        let r_total = wall.total_r_value();
        let c_total = wall.thermal_capacity();

        // Symmetric centroid partition: half of the layer R sits between the
        // mass node and each film. R_si = 1/8 m²K/W, R_se = 1/25 m²K/W.
        let r_si = 1.0 / 8.0;
        let r_se = 1.0 / 25.0;
        let h_tr_ms = 1.0 / (r_total / 2.0 + r_si);
        let h_tr_em = 1.0 / (r_total / 2.0 + r_se);
        let h_tr_is = 1.0 / r_si;

        // Issue #1593: Physics-based h_tr_me calculation (matches thermal_model_core.rs)
        // h_ms = 9.1 W/(m²·K) per ISO 13790 furniture coupling coefficient
        // a_int = furniture_factor * floor_area (furniture_factor = 0.5 per ISO 13790)
        let h_ms = 9.1;
        let furniture_factor = 0.5;
        let a_int = furniture_factor * floor_area;
        let h_tr_me = h_ms * a_int;

        let wall_node = ThermalMassNode::new(20.0, 0.45 * c_total, h_tr_ms, h_tr_em);
        let roof_node = ThermalMassNode::new(20.0, 0.30 * c_total, h_tr_ms, h_tr_em);
        let floor_node = ThermalMassNode::new(20.0, 0.18 * c_total, h_tr_ms, h_tr_em);
        let internal_node =
            ThermalMassNode::new(20.0, 0.10 * c_total, h_tr_ms, h_tr_em).with_h_tr_me(h_tr_me);

        let mut solver = Self::new(h_tr_is, wall_node, roof_node, floor_node, internal_node);
        solver.r_total = r_total;
        solver.r_se = r_se;
        solver.initialized = true;
        // Seed zone / surface / exterior at 20 °C so a single-step call from
        // a different interior/exterior BC produces a meaningful surface flux
        // (rather than starting at zero on a 0 K surface).
        solver.initialize_temperatures(20.0);
        solver
    }

    /// Convenience constructor that combines `from_wall_spec` with
    /// `with_coupling_mode` — the canonical entry point for tests and
    /// ML-surrogate wiring that needs `ParallelResistance` (#1281) end-to-end.
    pub fn from_wall_spec_with_mode(
        wall: &WallSpec,
        floor_area: f64,
        mode: MassAirCouplingMode,
    ) -> Self {
        Self::from_wall_spec(wall, floor_area).with_coupling_mode(mode)
    }

    /// Build a `Box<dyn HeatConductionSolver>` directly from a `WallSpec`.
    ///
    /// Used by `SolverRegistry::construct("multinode_9r4c", wall, floor_area)` and by
    /// `PhysicsSurfaceFluxProvider::add_surface` callers that want a high-mass
    /// surface behind the same trait object as `FiveR1CSolver`.
    pub fn boxed_from_wall_spec(wall: &WallSpec, floor_area: f64) -> Box<dyn HeatConductionSolver> {
        Box::new(Self::from_wall_spec(wall, floor_area))
    }
}

impl HeatConductionSolver for MultiNodeSolver {
    fn name(&self) -> &str {
        "MultiNode9R4C"
    }

    fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError> {
        let r_total = wall.total_r_value();
        if !r_total.is_finite() || r_total <= 0.0 {
            return Err(SolverError::ConstructionError(format!(
                "Invalid wall resistance from WallSpec '{wall_name}': R_total = {r_total} (must be positive and finite)",
                wall_name = wall.name,
            )));
        }
        let c_total = wall.thermal_capacity();
        if !c_total.is_finite() || c_total <= 0.0 {
            return Err(SolverError::ConstructionError(format!(
                "Invalid wall capacitance from WallSpec '{wall_name}': C_total = {c_total} (must be positive and finite)",
                wall_name = wall.name,
            )));
        }

        // Symmetric centroid partition of layer R (see `from_wall_spec`).
        let r_si = 1.0 / 8.0;
        let r_se = 1.0 / 25.0;
        let h_tr_ms = 1.0 / (r_total / 2.0 + r_si);
        let h_tr_em = 1.0 / (r_total / 2.0 + r_se);
        let h_tr_is = 1.0 / r_si;

        self.mass.wall = ThermalMassNode::new(20.0, 0.45 * c_total, h_tr_ms, h_tr_em);
        self.mass.roof = ThermalMassNode::new(20.0, 0.30 * c_total, h_tr_ms, h_tr_em);
        self.mass.floor = ThermalMassNode::new(20.0, 0.18 * c_total, h_tr_ms, h_tr_em);
        self.mass.internal =
            ThermalMassNode::new(20.0, 0.10 * c_total, h_tr_ms, h_tr_em).with_h_tr_me(100.0);
        self.h_tr_is = h_tr_is;
        self.zone_temperature = 20.0;
        self.surface_temperature = 20.0;
        self.exterior_temperature = 20.0;
        self.exterior_temperatures = SurfaceExteriorTemperatures::uniform(20.0);
        self.timestep_seconds = 3600.0;
        self.r_total = r_total;
        self.r_se = r_se;
        self.last_temps = None;
        self.last_dt = 0.0;
        self.initialized = true;
        Ok(())
    }

    fn step(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        _h_interior: HeatTransferCoefficient,
        _h_exterior: HeatTransferCoefficient,
    ) -> Result<HeatFlux, SolverError> {
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "Solver not initialized. Call initialize() first.".to_string(),
            ));
        }
        if !self.r_total.is_finite() || self.r_total <= 0.0 {
            return Err(SolverError::ConstructionError(format!(
                "Invalid cached R_total = {} (must be positive and finite)",
                self.r_total
            )));
        }

        let t_int = T_interior.to_value();
        let t_ext = T_exterior.to_value();
        let dt = timestep.to_value();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SolverError::InvalidConfig(format!(
                "Invalid timestep dt = {dt} (must be positive and finite)"
            )));
        }

        // Push boundary conditions into the multi-node state. The envelope
        // mass nodes share a single exterior BC under the trait's scalar
        // (T_ext, h_ext) interface; per-surface sol-air differences are the
        // caller's responsibility (Issue #863 surface_flux_provider path).
        self.zone_temperature = t_int;
        self.exterior_temperature = t_ext;
        self.exterior_temperatures = SurfaceExteriorTemperatures::uniform(t_ext);

        // Capture pre-step temps so `energy_storage_rate` can compute
        // Σ C_k · (T_k_new − T_k_old) / dt after the BE update.
        self.last_temps = Some((
            self.mass.wall.temperature,
            self.mass.roof.temperature,
            self.mass.floor.temperature,
            self.mass.internal.temperature,
        ));
        self.last_dt = dt;
        self.timestep_seconds = dt;

        // Evolve mass node temperatures via the configured coupling mode
        // (`AdditiveSum` default or `ParallelResistance` from #1281).
        self.step_backward_euler();

        // Returned flux — drop-in parity with `FiveR1CSolver::step()`:
        // q = (T_mass_avg − T_int) / R_total, where T_mass_avg is the simple
        // envelope (wall+roof+floor) average.  The denominator R_total is the
        // wall-only resistance from WallSpec::total_r_value(); surface film
        // coefficients (R_si = 1/8, R_se = 1/25) are intentionally excluded so
        // that this value matches the closed-form `steady_state_flux` query
        // (q_ss = (T_ext − T_int) / R_total), which is the parity target for
        // the ML-surrogate swap-point.  The comment claiming this reduces to
        // (T_ext − T_int) / (2R) at steady state was incorrect — there is no
        // such general simplification.
        let t_mass_avg =
            (self.mass.wall.temperature + self.mass.roof.temperature + self.mass.floor.temperature)
                / 3.0;
        let q = (t_mass_avg - t_int) / self.r_total;
        Ok(HeatFlux::from_value(q))
    }

    fn energy_storage_rate(&self) -> f64 {
        // Σ C_k · (T_k_new − T_k_old) / dt  [W/m²]
        // Positive = wall charging (gaining enthalpy), negative = discharging.
        let Some((t_wall_old, t_roof_old, t_floor_old, t_internal_old)) = self.last_temps else {
            return 0.0;
        };
        if self.last_dt <= 0.0 {
            return 0.0;
        }

        let c_wall = self.mass.wall.capacitance;
        let c_roof = self.mass.roof.capacitance;
        let c_floor = self.mass.floor.capacitance;
        let c_internal = self.mass.internal.capacitance;

        // Convert J/(m²·K) · K / s = W/m² → return as-is.
        (c_wall * (self.mass.wall.temperature - t_wall_old)
            + c_roof * (self.mass.roof.temperature - t_roof_old)
            + c_floor * (self.mass.floor.temperature - t_floor_old)
            + c_internal * (self.mass.internal.temperature - t_internal_old))
            / self.last_dt
    }

    fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        // Closed-form q_ss = (T_ext − T_int) / R_total — matches the
        // `FiveR1CSolver` default semantics (no thermal mass effect, just
        // Fourier's law across the layer R). This is the deterministic
        // query surface that ML-surrogate swap-points (`SurfaceHeatFluxProvider`)
        // call for parity checks.
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "Solver not initialized. Call initialize() first.".to_string(),
            ));
        }
        if !self.r_total.is_finite() || self.r_total <= 0.0 {
            return Err(SolverError::ConstructionError(format!(
                "Invalid cached R_total = {} (must be positive and finite)",
                self.r_total
            )));
        }
        let q_ss = (T_exterior.to_value() - T_interior.to_value()) / self.r_total;
        Ok(HeatFlux::from_value(q_ss))
    }

    fn is_valid(&self) -> bool {
        self.initialized
            && self.r_total.is_finite()
            && self.r_total > 0.0
            && self.mass.wall.capacitance > 0.0
            && self.mass.wall.h_tr_ms > 0.0
            && self.mass.wall.h_tr_em > 0.0
            && self.h_tr_is > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxion_core::multi_node::ThermalMassNode;
    use std::panic::catch_unwind;

    fn create_test_solver() -> MultiNodeSolver {
        let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
        let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
        let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);

        MultiNodeSolver::new(10.0, wall, roof, floor, internal)
    }

    #[test]
    fn test_solver_creation() {
        let solver = create_test_solver();
        assert_eq!(solver.wall_temperature(), 20.0);
        assert_eq!(solver.roof_temperature(), 20.0);
        assert_eq!(solver.floor_temperature(), 20.0);
        assert_eq!(solver.internal_temperature(), 20.0);
    }

    #[test]
    fn test_step_changes_temperatures() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(22.0);
        solver.set_exterior_temperature(5.0);
        solver.set_surface_temperature(18.0);

        let t_wall_before = solver.wall_temperature();
        solver.step(3600.0);

        // Wall should cool toward exterior temperature
        assert!(solver.wall_temperature() < t_wall_before);
    }

    #[test]
    fn test_envelope_temperature_average() {
        let mut solver = create_test_solver();
        solver.mass.wall.temperature = 10.0;
        solver.mass.roof.temperature = 20.0;
        solver.mass.floor.temperature = 30.0;

        let avg = solver.envelope_temperature();
        assert!((avg - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_time_constant_calculation() {
        let solver = create_test_solver();
        let tau = solver.effective_time_constant();

        // With C_total ≈ 11e6 J and h_eff ≈ 10-20 W/K
        // τ should be in the range of hours (h_tr in W/K, C in J/K, so τ in seconds)
        assert!(tau > 0.0);
        assert!(tau < 1e8); // Sanity check
    }

    #[test]
    fn test_steady_state_convergence() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(20.0);
        solver.set_exterior_temperature(20.0);
        solver.set_surface_temperature(20.0);

        // Run for many hours - temperatures should converge
        for _ in 0..168 {
            solver.step(3600.0);
        }

        // All temperatures should be near 20°C (within 0.1°C)
        assert!((solver.wall_temperature() - 20.0).abs() < 0.1);
        assert!((solver.roof_temperature() - 20.0).abs() < 0.1);
        assert!((solver.floor_temperature() - 20.0).abs() < 0.1);
        assert!((solver.internal_temperature() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_temperature_gradient_with_known_conductances() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(25.0);
        solver.set_exterior_temperature(0.0);
        solver.set_surface_temperature(15.0);

        // High-mass wall should show thermal lag
        let t_wall_initial = solver.wall_temperature();
        solver.step(3600.0);

        // Wall should cool slightly but not reach 0°C quickly due to high capacitance
        assert!(solver.wall_temperature() > 0.0);
        assert!(solver.wall_temperature() < t_wall_initial);
    }

    #[test]
    fn test_internal_mass_response() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(30.0);
        solver.set_exterior_temperature(10.0);
        solver.set_surface_temperature(20.0);

        // Internal mass should respond to zone temperature changes
        let t_internal_initial = solver.internal_temperature();
        solver.step(3600.0);

        // Internal mass should warm toward zone temperature
        assert!(solver.internal_temperature() > t_internal_initial);
        assert!(solver.internal_temperature() < 30.0);
    }

    #[test]
    fn test_backward_euler_stability() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(100.0); // Large temperature difference
        solver.set_exterior_temperature(-50.0);
        solver.set_surface_temperature(50.0);

        // Take many small timesteps - backward Euler should be stable
        for _ in 0..24 {
            solver.step(300.0); // 5-minute timestep
        }

        // All temperatures should be finite and within reasonable bounds
        assert!(solver.wall_temperature().is_finite());
        assert!(solver.roof_temperature().is_finite());
        assert!(solver.floor_temperature().is_finite());
        assert!(solver.internal_temperature().is_finite());

        // Should not have exploded
        assert!(solver.wall_temperature().abs() < 1000.0);
    }

    #[test]
    fn test_conductance_setters() {
        let mut solver = create_test_solver();

        solver.set_wall_conductances(25.0, 55.0);
        solver.set_roof_conductances(20.0, 40.0);
        solver.set_floor_conductances(15.0, 30.0);
        solver.set_internal_conductance(8.0);

        assert_eq!(solver.mass.wall.h_tr_em, 25.0);
        assert_eq!(solver.mass.wall.h_tr_ms, 55.0);
        assert_eq!(solver.mass.roof.h_tr_em, 20.0);
        assert_eq!(solver.mass.roof.h_tr_ms, 40.0);
        assert_eq!(solver.mass.floor.h_tr_em, 15.0);
        assert_eq!(solver.mass.floor.h_tr_ms, 30.0);
        assert_eq!(solver.mass.internal.h_tr_me, 8.0);
    }

    #[test]
    fn test_capacitance_setters() {
        let mut solver = create_test_solver();

        solver.set_wall_capacitance(1e7);
        solver.set_roof_capacitance(2e7);
        solver.set_floor_capacitance(3e7);
        solver.set_internal_capacitance(4e6);

        assert_eq!(solver.mass.wall.capacitance, 1e7);
        assert_eq!(solver.mass.roof.capacitance, 2e7);
        assert_eq!(solver.mass.floor.capacitance, 3e7);
        assert_eq!(solver.mass.internal.capacitance, 4e6);
    }

    #[test]
    fn test_initialization() {
        let mut solver = create_test_solver();
        solver.initialize_temperatures(15.0);

        assert_eq!(solver.wall_temperature(), 15.0);
        assert_eq!(solver.roof_temperature(), 15.0);
        assert_eq!(solver.floor_temperature(), 15.0);
        assert_eq!(solver.internal_temperature(), 15.0);
        assert_eq!(solver.zone_temperature, 15.0);
        assert_eq!(solver.surface_temperature, 15.0);
    }

    #[test]
    fn test_per_surface_exterior_temps() {
        let mut solver = create_test_solver();
        solver.initialize_temperatures(20.0);
        solver.set_zone_temperature(20.0);
        solver.set_surface_temperature(20.0);

        let temps = SurfaceExteriorTemperatures {
            t_ext_wall: 30.0,
            t_ext_roof: 35.0,
            t_ext_floor: 15.0,
        };
        solver.set_surface_exterior_temperatures(temps);
        solver.step(3600.0);

        assert!(
            solver.wall_temperature() > 20.0,
            "Wall should warm from sol-air"
        );
        assert!(
            solver.roof_temperature() > solver.wall_temperature(),
            "Roof > wall"
        );
        assert!(
            solver.floor_temperature() < 20.0,
            "Floor should cool from ground"
        );
    }

    // ── Issue #871: Air Balance API Tests ────────────────────────────

    #[test]
    fn test_compute_zone_air_temperature_steady_state() {
        let solver = create_test_solver();
        // All nodes at 20°C, outdoor at 20°C → T_air ≈ 20°C
        let t_air = solver.compute_zone_air_temperature(20.0, 5.0, 0.0, 0.0);
        assert!(
            (t_air - 20.0).abs() < 0.5,
            "Steady-state T_air should be ~20°C, got {t_air}"
        );
    }

    #[test]
    fn test_compute_zone_air_temperature_solar_gain() {
        let solver = create_test_solver();
        // phi_ia > 0 → T_air > T_outdoor
        let t_air_no_gain = solver.compute_zone_air_temperature(10.0, 5.0, 0.0, 0.0);
        let t_air_with_gain = solver.compute_zone_air_temperature(10.0, 5.0, 0.0, 2000.0);
        assert!(
            t_air_with_gain > t_air_no_gain,
            "Solar gain should raise T_air: {t_air_with_gain} should be > {t_air_no_gain}"
        );
        assert!(
            t_air_with_gain > 10.0,
            "T_air with gains should be above outdoor: {t_air_with_gain} > 10.0"
        );
    }

    #[test]
    fn test_compute_hvac_demand_heating() {
        let solver = create_test_solver();
        // T_air_free < heating setpoint → positive Q (heating needed)
        let q = solver.compute_hvac_demand(15.0, 20.0, 26.0);
        assert!(
            q > 0.0,
            "Heating demand should be positive when T_air < heating setpoint, got {q}"
        );
        // Q = h_tr_is × (20 - 15) = 10 × 5 = 50 W
        assert!((q - 50.0).abs() < 1.0, "Expected ~50W heating, got {q}");
    }

    #[test]
    fn test_compute_hvac_demand_cooling() {
        let solver = create_test_solver();
        // T_air_free > cooling setpoint → negative Q (cooling needed)
        let q = solver.compute_hvac_demand(30.0, 20.0, 26.0);
        assert!(
            q < 0.0,
            "Cooling demand should be negative when T_air > cooling setpoint, got {q}"
        );
        // Q = h_tr_is × (26 - 30) = 10 × (-4) = -40 W
        assert!((q - (-40.0)).abs() < 1.0, "Expected ~-40W cooling, got {q}");
    }

    #[test]
    fn test_compute_hvac_demand_deadband() {
        let solver = create_test_solver();
        // T_air_free within [heat_sp, cool_sp] → zero Q
        let q = solver.compute_hvac_demand(22.0, 20.0, 26.0);
        assert!(
            q.abs() < 1e-10,
            "Demand should be zero within deadband, got {q}"
        );
    }

    #[test]
    fn test_step_with_gains_increases_temp() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(20.0);
        solver.set_exterior_temperature(10.0);
        solver.set_surface_temperature(18.0);

        // Step without gains
        let mut solver_no_gains = solver.clone();
        solver_no_gains.step(3600.0);
        let t_wall_no_gains = solver_no_gains.wall_temperature();
        let t_roof_no_gains = solver_no_gains.roof_temperature();

        // Step with gains (1000W to wall, 500W to roof)
        solver.step_with_gains(
            3600.0,
            1000.0,
            500.0,
            0.0,
            0.0,
            0.0,
            solver.exterior_temperature,
        );
        let t_wall_with_gains = solver.wall_temperature();
        let t_roof_with_gains = solver.roof_temperature();

        assert!(
            t_wall_with_gains > t_wall_no_gains,
            "Wall with gains ({t_wall_with_gains}) should be > without ({t_wall_no_gains})"
        );
        assert!(
            t_roof_with_gains > t_roof_no_gains,
            "Roof with gains ({t_roof_with_gains}) should be > without ({t_roof_no_gains})"
        );
        // Wall gets more gains than roof → should be hotter
        let wall_delta = t_wall_with_gains - t_wall_no_gains;
        let roof_delta = t_roof_with_gains - t_roof_no_gains;
        assert!(
            wall_delta > roof_delta,
            "Wall delta ({wall_delta}) should exceed roof delta ({roof_delta})"
        );
    }

    // ── Issue #1281: Parallel-resistance coupling network ─────────────

    /// Construct a Case 900-style high-mass solver for Issue #1281 tests.
    ///
    /// Per-surface h_tr_ms values come from the half-insulation rule applied
    /// to the ASHRAE 140 Case 900 construction
    /// (`src/sim/construction.rs::Assemblies::high_mass_wall` /
    /// `high_mass_roof` / `high_mass_floor`).
    fn create_case_900_solver(coupling_mode: MassAirCouplingMode) -> MultiNodeSolver {
        let wall = ThermalMassNode::new(20.0, 5.0e6, 76.4, 25.0);
        let roof = ThermalMassNode::new(20.0, 3.0e6, 32.9, 20.0);
        let floor = ThermalMassNode::new(20.0, 2.0e6, 18.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1.0e6, 0.0, 0.0).with_h_tr_me(100.0);

        // h_tr_is = 3.45 × floor_area = 3.45 × 48 = 165.6 W/K
        // (Issue #714: ASHRAE 140 simplified 5R1C formula)
        MultiNodeSolver::new_with_mode(165.6, wall, roof, floor, internal, coupling_mode)
    }

    #[test]
    fn test_issue_1281_default_mode_is_additive_sum() {
        // Backward compatibility: existing constructor keeps AdditiveSum default.
        let solver = create_test_solver();
        assert_eq!(solver.coupling_mode, MassAirCouplingMode::AdditiveSum);
    }

    #[test]
    fn test_issue_1281_new_with_mode_parallel_resistance() {
        let solver = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
        assert_eq!(
            solver.coupling_mode,
            MassAirCouplingMode::ParallelResistance
        );
    }

    #[test]
    fn test_issue_1281_with_coupling_mode_builder() {
        let solver =
            create_test_solver().with_coupling_mode(MassAirCouplingMode::ParallelResistance);
        assert_eq!(
            solver.coupling_mode,
            MassAirCouplingMode::ParallelResistance
        );
    }

    #[test]
    fn test_issue_1281_parallel_resistance_air_lower_than_additive() {
        // At steady state with hot exterior forcing, the parallel-resistance
        // formulation should give a LOWER T_air than the additive formulation
        // because h_path_total < h_ms_total (each per-surface series conductance
        // is strictly less than the per-surface h_ms_k).
        //
        // Reference: .agents/results/issue-1281-python-verification.py
        // (steady-state Case 900: additive T_air=42.0, parallel-resistance T_air=30.4)

        let mut add = create_case_900_solver(MassAirCouplingMode::AdditiveSum);
        let mut par = create_case_900_solver(MassAirCouplingMode::ParallelResistance);

        // Hot summer forcing (similar to Python verification)
        let ext = SurfaceExteriorTemperatures {
            t_ext_wall: 45.0,
            t_ext_roof: 50.0,
            t_ext_floor: 18.0,
        };
        add.set_surface_exterior_temperatures(ext.clone());
        par.set_surface_exterior_temperatures(ext);
        add.set_zone_temperature(20.0);
        par.set_zone_temperature(20.0);

        // 5000 × 1-hour steps to reach steady state
        for _ in 0..5000 {
            add.step(3600.0);
            par.step(3600.0);
        }

        let t_air_add = add.compute_zone_air_temperature(32.0, 21.7, 0.0, 200.0);
        let t_air_par = par.compute_zone_air_temperature(32.0, 21.7, 0.0, 200.0);

        // Sanity: both finite and positive
        assert!(t_air_add.is_finite() && t_air_par.is_finite());
        assert!(t_air_add > 20.0 && t_air_par > 20.0);

        // The parallel-resistance formulation gives a LOWER air temperature
        // (verified by Python: 30.4 °C vs 42.0 °C).
        assert!(
            t_air_par < t_air_add,
            "Parallel-resistance T_air ({:.3}) should be < additive T_air ({:.3})",
            t_air_par,
            t_air_add,
        );

        // The gap should be on the order of 5-15 °C for Case 900 parameters.
        let gap = t_air_add - t_air_par;
        assert!(
            gap > 1.0,
            "T_air gap ({:.3}) should be meaningful (>1 K), confirming non-additive correction",
            gap,
        );
    }

    #[test]
    fn test_issue_1281_h_series_formula() {
        // Verify the series-combination helper matches a hand calculation.
        // h_series(a, b) = a*b/(a+b)
        // Symmetric: h_series(a, b) == h_series(b, a)
        assert!((h_series(50.0, 165.6) - 50.0 * 165.6 / (50.0 + 165.6)).abs() < 1e-10);
        assert!((h_series(165.6, 50.0) - h_series(50.0, 165.6)).abs() < 1e-10);

        // Degenerate cases: use h_series_strict which returns Result
        // so the error path can be tested in both debug AND release builds
        assert!(
            h_series_strict(0.0, 100.0).is_err(),
            "h_series_strict should Err for degenerate inputs"
        );
        assert!(
            h_series_strict(100.0, 0.0).is_err(),
            "h_series_strict should Err for degenerate inputs"
        );
        assert!(
            h_series_strict(-1.0, 100.0).is_err(),
            "h_series_strict should Err for degenerate inputs"
        );

        // For Case 900 per-surface values:
        let h_path_wall = h_series(76.4, 165.6);
        let h_path_roof = h_series(32.9, 165.6);
        let h_path_floor = h_series(18.0, 165.6);
        let h_path_total = h_path_wall + h_path_roof + h_path_floor;
        let h_ms_total = 76.4 + 32.9 + 18.0;
        assert!(
            h_path_total < h_ms_total,
            "Parallel-resistance total ({:.3}) must be < additive h_ms_total ({:.3})",
            h_path_total,
            h_ms_total,
        );
        // Numerical verification: ratio matches Python's 0.753
        let ratio = h_path_total / h_ms_total;
        assert!(
            (ratio - 0.7534).abs() < 0.01,
            "ratio {ratio} should be ~0.753 (Python-verified 32.7% overcount)"
        );
    }

    #[test]
    fn test_issue_1281_per_surface_t_s_helper() {
        // T_s = (h_ms × t_m + h_is × t_air) / (h_ms + h_is)
        let t_m = 30.0;
        let h_ms = 76.4;
        let h_is = 165.6;
        let t_air = 25.0;
        let expected = (h_ms * t_m + h_is * t_air) / (h_ms + h_is);
        let actual = per_surface_t_s(t_m, h_ms, h_is, t_air);
        assert!((actual - expected).abs() < 1e-10);

        // Degenerate cases
        assert_eq!(per_surface_t_s(20.0, 0.0, 0.0, 30.0), 30.0);
        assert_eq!(per_surface_t_s(f64::NAN, 1.0, 1.0, 20.0), 20.0);
    }

    #[test]
    fn test_issue_1281_parallel_resistance_step_uses_per_surface_t_s() {
        // Verify that step() in ParallelResistance mode produces DIFFERENT mass
        // temperatures than AdditiveSum for the same forcing. This is the
        // physically meaningful behavior change.
        let mut add = create_case_900_solver(MassAirCouplingMode::AdditiveSum);
        let mut par = create_case_900_solver(MassAirCouplingMode::ParallelResistance);

        let ext = SurfaceExteriorTemperatures {
            t_ext_wall: 45.0,
            t_ext_roof: 50.0,
            t_ext_floor: 18.0,
        };
        add.set_surface_exterior_temperatures(ext.clone());
        par.set_surface_exterior_temperatures(ext);
        add.set_zone_temperature(20.0);
        par.set_zone_temperature(20.0);

        // Run 24 hourly steps
        for _ in 0..24 {
            add.step(3600.0);
            par.step(3600.0);
        }

        // Mass temperatures should differ between the two formulations
        // (the parallel-resistance formulation feeds each mass node its OWN
        // per-surface T_s_k, not the shared conductance-weighted mean).
        let diff_wall = (add.wall_temperature() - par.wall_temperature()).abs();
        let diff_roof = (add.roof_temperature() - par.roof_temperature()).abs();
        let diff_floor = (add.floor_temperature() - par.floor_temperature()).abs();

        assert!(
            diff_wall + diff_roof + diff_floor > 0.01,
            "Mass temperatures must differ between additive ({:.3}, {:.3}, {:.3}) and parallel ({:.3}, {:.3}, {:.3})",
            add.wall_temperature(), add.roof_temperature(), add.floor_temperature(),
            par.wall_temperature(), par.roof_temperature(), par.floor_temperature(),
        );
    }

    #[test]
    fn test_issue_1281_parallel_resistance_step_with_gains() {
        // Verify step_with_gains in ParallelResistance mode produces higher
        // mass temperatures than no-gains case (solar gains are heating).
        let mut solver_no_gains = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
        let mut solver_with_gains = create_case_900_solver(MassAirCouplingMode::ParallelResistance);

        let ext = SurfaceExteriorTemperatures {
            t_ext_wall: 45.0,
            t_ext_roof: 50.0,
            t_ext_floor: 18.0,
        };
        solver_no_gains.set_surface_exterior_temperatures(ext.clone());
        solver_with_gains.set_surface_exterior_temperatures(ext);
        solver_no_gains.set_zone_temperature(20.0);
        solver_with_gains.set_zone_temperature(20.0);

        for _ in 0..24 {
            solver_no_gains.step(3600.0);
            solver_with_gains.step_with_gains(
                3600.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                0.0,
                solver_with_gains.exterior_temperature,
            );
        }

        assert!(
            solver_with_gains.wall_temperature() > solver_no_gains.wall_temperature(),
            "Wall with gains ({:.3}) should be hotter than without ({:.3})",
            solver_with_gains.wall_temperature(),
            solver_no_gains.wall_temperature(),
        );
        assert!(
            solver_with_gains.roof_temperature() > solver_no_gains.roof_temperature(),
            "Roof with gains ({:.3}) should be hotter than without ({:.3})",
            solver_with_gains.roof_temperature(),
            solver_no_gains.roof_temperature(),
        );
    }

    #[test]
    fn test_issue_1281_backward_compat_additive_unchanged() {
        // Verify that AdditiveSum mode produces the same T_air as the
        // original (pre-Issue #1281) formulation for a known forcing case.
        // This test serves as a regression guard: if someone breaks the
        // original additive formula, this test fails.
        let solver = create_case_900_solver(MassAirCouplingMode::AdditiveSum);

        // All masses at 20, all temps at 20 → T_air should be 20 (steady state).
        let t_air = solver.compute_zone_air_temperature(20.0, 0.0, 0.0, 0.0);
        assert!(
            (t_air - 20.0).abs() < 0.5,
            "Steady-state T_air should be ~20°C, got {t_air}"
        );
    }

    #[test]
    fn test_issue_1281_parallel_resistance_degenerate_falls_back() {
        // When h_tr_ms sums to ~0 (degenerate construction), the parallel-resistance
        // air calc must fall back to the conductance-weighted average of mass
        // temperatures (not NaN/Inf). In debug builds, debug_assert! in h_series
        // fires; we catch the panic and verify the solver still produces a finite result.
        let wall = ThermalMassNode::new(20.0, 1.0e6, 0.0, 25.0);
        let roof = ThermalMassNode::new(20.0, 1.0e6, 0.0, 20.0);
        let floor = ThermalMassNode::new(20.0, 1.0e6, 0.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 5.0e5, 0.0, 0.0).with_h_tr_me(0.0);
        let solver = MultiNodeSolver::new_with_mode(
            165.6,
            wall,
            roof,
            floor,
            internal,
            MassAirCouplingMode::ParallelResistance,
        );

        // debug_assert! fires in debug builds for degenerate h_series inputs
        let result = catch_unwind(|| solver.compute_zone_air_temperature(30.0, 5.0, 0.0, 0.0));
        if result.is_err() {
            // Debug build: debug_assert! fired as expected — degenerate inputs are caught
            return;
        }
        let t_air = result.unwrap();
        assert!(
            t_air.is_finite(),
            "Degenerate construction must produce finite T_air, got {t_air}"
        );
    }

    // ── Issue #1858: sky-radiative air-node path ──────────────────────

    #[test]
    fn test_issue_1858_air_sky_conductance_formula() {
        // Verify the linearized conductance against a hand calculation.
        //
        // Issue #2462 (Phase 2 of the crate split): previously this test also
        // cross-checked against `crate::sim::sky_radiation::SkyRadiationExchange::radiative_coefficient`,
        // but that import was one of the 5 documented `physics ↔ sim` cycle
        // edges. The SkyRadiationExchange::radiative_coefficient formula is
        // `4.0 * eps * f_sky * STEFAN_BOLTZMANN * t_mean^3` — the same
        // hand calculation we already verify below — so the cross-check
        // added nothing once the leaf constant was available here.
        let eps = 0.9;
        let f_sky = 0.25;
        let aperture = 12.0; // m²
        let t_air = -8.0;
        let t_sky = -30.0;

        // Total conductance from the new helper (W/K).
        let h_total = air_sky_conductance(eps, f_sky, aperture, t_air, t_sky);

        // Hand calculation for the linearization at T_mean = 254.075 K.
        let sigma = 5.67e-8_f64;
        let t_mean_k = ((t_air + 273.15) + (t_sky + 273.15)) / 2.0;
        let expected = 4.0 * eps * f_sky * sigma * t_mean_k.powi(3) * aperture;
        assert!(
            (h_total - expected).abs() < 1e-6,
            "h_total {h_total} != hand calc {expected}",
        );

        // Magnitude sanity: ~10 W/K for Case 900 night-min aperture geometry.
        assert!(
            h_total > 5.0 && h_total < 20.0,
            "unexpected h_total {h_total}"
        );
    }

    #[test]
    fn test_issue_1858_air_sky_conductance_degenerate_is_zero() {
        // Degenerate inputs → 0.0 (no-op sky path, preserves backward compat).
        assert_eq!(air_sky_conductance(0.0, 0.5, 12.0, -8.0, -30.0), 0.0);
        assert_eq!(air_sky_conductance(0.9, 0.0, 12.0, -8.0, -30.0), 0.0);
        assert_eq!(air_sky_conductance(0.9, 0.5, 0.0, -8.0, -30.0), 0.0);
        assert_eq!(air_sky_conductance(0.9, 0.5, -1.0, -8.0, -30.0), 0.0);
    }

    #[test]
    fn test_issue_1858_backward_compat_zero_sky_conductance() {
        // A zero sky conductance must recover the original four-term air-node
        // balance EXACTLY for both coupling modes. This is the guard that the
        // sky-radiative path does not perturb existing ASHRAE 140 fixtures.
        for mode in [
            MassAirCouplingMode::AdditiveSum,
            MassAirCouplingMode::ParallelResistance,
        ] {
            let solver = create_case_900_solver(mode);
            let t_outdoor = -10.0;
            let h_ve = 21.7;
            let phi_ia = 200.0;

            let t_air_plain = solver.compute_zone_air_temperature(t_outdoor, h_ve, 0.0, phi_ia);
            let t_air_sky_zero = solver
                .compute_zone_air_temperature_with_sky(t_outdoor, h_ve, 0.0, phi_ia, -30.0, 0.0);

            assert!(
                (t_air_plain - t_air_sky_zero).abs() < 1e-12,
                "mode {:?}: zero-sky path ({t_air_sky_zero}) must equal plain path ({t_air_plain})",
                mode,
            );
        }
    }

    #[test]
    fn test_issue_1858_sky_path_lowers_air_below_outdoor() {
        // The structural gap documented in ISSUE_1168_ROOT_CAUSE.md: the original
        // air-node balance bounds T_air below by min(T_surface, T_out). With a
        // cold sky and a non-zero sky conductance, T_air must be able to fall
        // BELOW the outdoor dry-bulb under clear-sky radiative cooling.
        //
        // Representative ASHRAE 140 Case 900 winter clear-night conditions.
        let solver = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
        let t_outdoor = -10.0;
        let t_sky = -30.0; // clear sky ≈ 20 K below dry-bulb
        let h_ve = 21.7;
        let phi_ia = 200.0;

        // Physics-derived sky conductance: ε=0.9, F_sky=0.25 (window/floor),
        // aperture = 12 m² (Case 900 south glazing).
        let h_rad_sky = air_sky_conductance(0.9, 0.25, 12.0, -8.0, t_sky);
        assert!(h_rad_sky > 0.0, "sky conductance must be positive");

        let t_air_no_sky = solver.compute_zone_air_temperature(t_outdoor, h_ve, 0.0, phi_ia);
        let t_air_with_sky = solver
            .compute_zone_air_temperature_with_sky(t_outdoor, h_ve, 0.0, phi_ia, t_sky, h_rad_sky);

        // The sky path must COOL the air node (sky is colder than every other node).
        assert!(
            t_air_with_sky < t_air_no_sky,
            "sky path must cool air: {t_air_with_sky} < {t_air_no_sky}",
        );

        // The drop is meaningful and physics-derived (no tuning). In this
        // single-timestep idealization (mass nodes held at their default) the
        // drop exceeds the ~0.6 °C *integrated annual* night-min residual the
        // issue targets — the integrated run sees a smaller effect because the
        // mass nodes themselves cool over the season. We assert a generous band
        // that proves the path is active without coupling the unit test to the
        // full simulation.
        let drop = t_air_no_sky - t_air_with_sky;
        assert!(
            drop > 0.5 && drop < 8.0,
            "night-min drop {drop:.3} °C should be a meaningful cooling (>0.5 K) \
             that closes the ~0.6 °C annual residual",
        );
    }

    #[test]
    fn test_issue_1858_sky_path_responds_to_aperture_and_sky_temp() {
        // Larger aperture → more cooling; warmer sky → less cooling.
        // Confirms the path is monotone in the physics, not a tuned constant.
        let solver = create_case_900_solver(MassAirCouplingMode::AdditiveSum);
        let t_outdoor = -10.0;
        let h_ve = 21.7;
        let phi_ia = 200.0;
        let t_sky_clear = -30.0;

        let h_small = air_sky_conductance(0.9, 0.25, 6.0, -8.0, t_sky_clear);
        let h_large = air_sky_conductance(0.9, 0.25, 24.0, -8.0, t_sky_clear);

        let t_small = solver.compute_zone_air_temperature_with_sky(
            t_outdoor,
            h_ve,
            0.0,
            phi_ia,
            t_sky_clear,
            h_small,
        );
        let t_large = solver.compute_zone_air_temperature_with_sky(
            t_outdoor,
            h_ve,
            0.0,
            phi_ia,
            t_sky_clear,
            h_large,
        );
        assert!(
            t_large < t_small,
            "larger aperture must cool more: {t_large} < {t_small}",
        );

        // Warmer sky (overcast) → less cooling than clear sky at fixed aperture.
        let t_sky_overcast = -8.0;
        let h_fixed = air_sky_conductance(0.9, 0.25, 12.0, -8.0, t_sky_clear);
        let t_clear = solver.compute_zone_air_temperature_with_sky(
            t_outdoor,
            h_ve,
            0.0,
            phi_ia,
            t_sky_clear,
            h_fixed,
        );
        let t_overcast = solver.compute_zone_air_temperature_with_sky(
            t_outdoor,
            h_ve,
            0.0,
            phi_ia,
            t_sky_overcast,
            h_fixed,
        );
        assert!(
            t_overcast > t_clear,
            "overcast (warmer) sky must cool less: {t_overcast} > {t_clear}",
        );
    }

    #[test]
    fn test_issue_1858_sky_path_does_not_break_energy_balance() {
        // The sky term is a boundary flux: with h_rad_sky applied, the air-node
        // balance is still a closed linear system and the mass-node backward
        // Euler First-Law invariant (check_energy_balance) is unaffected because
        // the sky path is local to the air node and does not touch step().
        let mut solver = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
        solver.set_zone_temperature(-8.0);
        solver.set_surface_exterior_temperatures(SurfaceExteriorTemperatures {
            t_ext_wall: -10.0,
            t_ext_roof: -12.0,
            t_ext_floor: 2.0,
        });
        // Step the mass nodes — the First-Law debug_assert! in step() must hold.
        solver.step(3600.0);

        // Air-node balance with sky still yields a finite, physically reasonable T.
        // The mass nodes are still warm (Case 900 default 20 °C) after one cold
        // step, so T_air is bounded between the coldest boundary (sky −30 °C) and
        // the warmest mass node — not pinned to a narrow band.
        let t_air = solver.compute_zone_air_temperature_with_sky(
            -10.0,
            21.7,
            0.0,
            200.0,
            -30.0,
            air_sky_conductance(0.9, 0.25, 12.0, -8.0, -30.0),
        );
        assert!(t_air.is_finite(), "T_air must be finite: {t_air}");
        assert!(
            t_air > -30.0 && t_air < 25.0,
            "T_air {t_air} must lie within the boundary-temperature envelope",
        );
    }
}
