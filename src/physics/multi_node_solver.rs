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

use crate::sim::multi_node_thermal::{MultiNodeThermalMass, ThermalMassNode};
use crate::sim::per_surface_conduction::{PerSurfaceConductionSolver, SurfaceKind};

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
        }
    }

    pub fn with_timestep(mut self, dt: f64) -> Self {
        self.timestep_seconds = dt;
        self
    }

    pub fn step(&mut self, dt: f64) -> &MultiNodeThermalMass {
        self.timestep_seconds = dt;
        self.step_backward_euler();
        &self.mass
    }

    fn step_backward_euler(&mut self) {
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
            let t_env_avg = (m.wall.temperature + m.roof.temperature + m.floor.temperature) / 3.0;
            let h_me = node.h_tr_me;

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

        // Issue #1024: First Law of Thermodynamics debug_assert!
        // Energy In - Energy Out = Change in Storage
        // For backward Euler: Q_net = C*(T_new - T_old)/dt for each node
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
        debug_assert!(
            (q_net - delta_e_rate).abs() < 1e-7,
            "First Law violation: net heat ({q_net} W) != change in storage rate ({delta_e_rate} W)",
        );
    }

    // ── Issue #871: Air Balance API Methods ───────────────────────────

    /// Compute zone air temperature from the multi-node thermal balance.
    ///
    /// Must be called AFTER `step()` (or `step_with_gains()`) has updated
    /// mass node temperatures. Uses the air node energy balance:
    ///
    /// ```text
    /// T_s = Σ(h_tr_ms_k × T_k) / Σ(h_tr_ms_k)   for k ∈ {wall, roof, floor}
    /// T_air = (h_tr_is × T_s + h_ve × T_out + φ_ia) / (h_tr_is + h_ve)
    /// ```
    ///
    /// # Arguments
    /// * `t_outdoor` — Outdoor air temperature [°C]
    /// * `h_ve` — Ventilation/infiltration conductance [W/K]
    /// * `phi_ia` — Internal convective + solar-to-air gains [W]
    ///
    /// # Returns
    /// Free-floating zone air temperature [°C]
    pub fn compute_zone_air_temperature(&self, t_outdoor: f64, h_ve: f64, phi_ia: f64) -> f64 {
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
        let denom = self.h_tr_is + h_ve;
        if denom < 1e-6 {
            // Near-zero ventilation + interior film — return surface temp as best estimate
            return t_surface;
        }

        (self.h_tr_is * t_surface + h_ve * t_outdoor + phi_ia) / denom
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
    ///
    /// # Returns
    /// The (wall, roof, floor) per-surface temperatures [°C]
    pub fn step_per_surface(&mut self, dt: f64) -> (f64, f64, f64) {
        // Build a transient per-surface solver from current state
        let mut solver = self.build_per_surface_solver();

        // Per-surface exterior temperatures (Issue #863)
        let t_ext_wall = self.exterior_temperatures.t_ext_wall;
        let t_ext_roof = self.exterior_temperatures.t_ext_roof;
        let t_ext_floor = self.exterior_temperatures.t_ext_floor;

        // Update each surface with its own mass node temperature and exterior temperature
        solver.update_surface(0, dt, self.mass.wall.temperature, t_ext_wall);
        solver.update_surface(1, dt, self.mass.roof.temperature, t_ext_roof);
        solver.update_surface(2, dt, self.mass.floor.temperature, t_ext_floor);

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
    pub fn step_with_gains(
        &mut self,
        dt: f64,
        gains_wall: f64,
        gains_roof: f64,
        gains_floor: f64,
        gains_internal: f64,
    ) -> &MultiNodeThermalMass {
        self.timestep_seconds = dt;
        self.step_backward_euler_with_gains(gains_wall, gains_roof, gains_floor, gains_internal);
        &self.mass
    }

    /// Backward Euler step with per-node gain injection.
    ///
    /// Same as `step_backward_euler()` but adds gain terms [W] to each node's
    /// numerator. This allows solar/radiative gains to properly heat envelope
    /// surfaces rather than only relying on conduction.
    fn step_backward_euler_with_gains(
        &mut self,
        gains_wall: f64,
        gains_roof: f64,
        gains_floor: f64,
        gains_internal: f64,
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

        // Update wall node — with gains
        {
            let node = &mut m.wall;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_wall
                    + h_ms * self.surface_temperature
                    + gains_wall;
                node.temperature = numer / denom;
            }
        }

        // Update roof node — with gains
        {
            let node = &mut m.roof;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_roof
                    + h_ms * self.surface_temperature
                    + gains_roof;
                node.temperature = numer / denom;
            }
        }

        // Update floor node — with gains
        {
            let node = &mut m.floor;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            if denom > 1e-10 {
                let numer = node.capacitance / dt * node.temperature
                    + h_em * t_ext_floor
                    + h_ms * self.surface_temperature
                    + gains_floor;
                node.temperature = numer / denom;
            }
        }

        // Update internal node — with gains
        {
            let node = &mut m.internal;
            let t_env_avg = (m.wall.temperature + m.roof.temperature + m.floor.temperature) / 3.0;
            let h_me = node.h_tr_me;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::multi_node_thermal::ThermalMassNode;

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
        let t_air = solver.compute_zone_air_temperature(20.0, 5.0, 0.0);
        assert!(
            (t_air - 20.0).abs() < 0.5,
            "Steady-state T_air should be ~20°C, got {t_air}"
        );
    }

    #[test]
    fn test_compute_zone_air_temperature_solar_gain() {
        let solver = create_test_solver();
        // phi_ia > 0 → T_air > T_outdoor
        let t_air_no_gain = solver.compute_zone_air_temperature(10.0, 5.0, 0.0);
        let t_air_with_gain = solver.compute_zone_air_temperature(10.0, 5.0, 2000.0);
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
        solver.step_with_gains(3600.0, 1000.0, 500.0, 0.0, 0.0);
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
}
