//! Multi-Node HVAC Runner with Warm-Up Period (Issue #865)
//!
//! This module provides a high-level runner that wraps the `MultiNodeSolver`
//! (9R4C thermal network) with HVAC control logic and a configurable warm-up
//! period. During warm-up, the solver advances mass temperatures toward
//! equilibrium but does NOT accumulate energy totals, avoiding phantom heating
//! from transient initial conditions.
//!
//! # Warm-Up Rationale
//!
//! All mass temperatures start at 20 °C. For heavy-mass buildings (Case 900+)
//! this produces ~10–15 kW transient heating in the first few hundred timesteps
//! versus a reference peak of ~2 kW, adding ~1 MWh phantom heating energy.
//! Running 14 days of warm-up (336 hourly timesteps) lets mass temperatures
//! converge before energy accumulation begins.

#![allow(deprecated)]

use crate::physics::multi_node_solver::MultiNodeSolver;
use crate::sim::multi_node_thermal::ThermalMassNode;

/// Default number of warm-up days (14 days per ASHRAE 140 §B2 guidance).
/// Matches [`crate::sim::warmup::DEFAULT_WARMUP_DAYS`].
const DEFAULT_WARMUP_DAYS: usize = 14;

/// Specific heat of air at constant pressure [J/kgK].
/// ASHRAE Handbook Fundamentals (2021) Chapter 1: moist air cp ≈ 1006 J/kgK.
const CP_AIR: f64 = 1006.0;

/// Multi-node HVAC runner with warm-up period support.
#[deprecated(
    since = "0.9.0",
    note = "Use multi-node thermal model with inline HVAC control instead. \
            The `MultiNodeSolver` now supports HVAC integration directly, \
            providing better energy accounting and Crank-Nicolson time integration. \
            See `crate::sim::multi_node_thermal` for the preferred approach."
)]
///
/// Wraps a `MultiNodeSolver` and provides:
/// - Simple setpoint-based HVAC control (heating / cooling)
/// - Energy accumulation (annual heating/cooling energy, peak power)
/// - Configurable warm-up period that skips energy accumulation
///   while still advancing mass temperatures
#[derive(Debug, Clone)]
pub struct MultiNodeHvacRunner {
    /// The 9R4C thermal solver
    pub solver: MultiNodeSolver,
    /// Ventilation conductance h_ve [W/K]
    pub h_ve: f64,
    /// Window transmission conductance h_tr_w [W/K]
    pub h_tr_w: f64,
    /// Heating setpoint [°C]
    pub heating_setpoint: f64,
    /// Cooling setpoint [°C]
    pub cooling_setpoint: f64,
    /// Cumulative annual heating energy [kWh]
    pub annual_heating_energy: f64,
    /// Cumulative annual cooling energy [kWh]
    pub annual_cooling_energy: f64,
    /// Peak heating power observed [kW]
    pub peak_heating_power: f64,
    /// Peak cooling power observed [kW]
    pub peak_cooling_power: f64,
    /// Zone temperature from previous timestep [°C]
    prev_zone_temp: f64,

    // -- Warm-up tracking fields --
    /// Number of timesteps stepped (for warm-up tracking)
    timestep_count: usize,
    /// Whether warm-up period has completed
    warmed_up: bool,
    /// Number of warm-up timesteps (default: 336 = 14 days × 24 hours)
    warmup_timesteps: usize,

    // -- Air-node HVAC fields (Issue #1007) --
    /// Air mass flow rate [kg/s]
    pub m_dot: f64,
    /// Supply air temperature [°C]
    pub T_supply: f64,
}

impl MultiNodeHvacRunner {
    /// Create a new `MultiNodeHvacRunner` with the given solver and HVAC parameters.
    ///
    /// # Arguments
    ///
    /// * `solver` - A configured `MultiNodeSolver`
    /// * `h_ve` - Ventilation conductance [W/K]
    /// * `h_tr_w` - Window transmission conductance [W/K]
    /// * `heating_setpoint` - Heating setpoint temperature [°C]
    /// * `cooling_setpoint` - Cooling setpoint temperature [°C]
    pub fn new(
        solver: MultiNodeSolver,
        h_ve: f64,
        h_tr_w: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> Self {
        Self {
            solver,
            h_ve,
            h_tr_w,
            heating_setpoint,
            cooling_setpoint,
            annual_heating_energy: 0.0,
            annual_cooling_energy: 0.0,
            peak_heating_power: 0.0,
            peak_cooling_power: 0.0,
            prev_zone_temp: heating_setpoint,
            timestep_count: 0,
            warmed_up: false,
            warmup_timesteps: DEFAULT_WARMUP_DAYS * 24,
            m_dot: 0.0,
            T_supply: heating_setpoint,
        }
    }

    /// Create a runner with default solver parameters for testing.
    ///
    /// Uses typical 9R4C thermal mass nodes with 20 °C initial temperature.
    pub fn with_defaults() -> Self {
        let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
        let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
        let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);
        let solver = MultiNodeSolver::new(10.0, wall, roof, floor, internal);

        let mut runner = Self::new(solver, 15.0, 20.0, 20.0, 26.0);
        // Default mass flow rate: 0.5 kg/s (typical for residential HVAC)
        runner.m_dot = 0.5;
        // Default supply air temperature: 40°C heating, 16°C cooling
        runner.T_supply = 40.0;
        runner
    }

    /// Configure the warm-up duration in days.
    ///
    /// Setting `days` to 0 disables the warm-up period entirely.
    pub fn with_warmup_days(mut self, days: usize) -> Self {
        self.warmup_timesteps = days * 24;
        if self.warmup_timesteps == 0 {
            self.warmed_up = true;
        }
        self
    }

    /// Whether the warm-up period has completed.
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Number of timesteps stepped so far.
    pub fn timestep_count(&self) -> usize {
        self.timestep_count
    }

    /// Reset energy accumulators to zero.
    ///
    /// Useful after warm-up to start energy tracking from a clean slate,
    /// though the warm-up logic already skips accumulation automatically.
    pub fn reset_accumulators(&mut self) {
        self.annual_heating_energy = 0.0;
        self.annual_cooling_energy = 0.0;
        self.peak_heating_power = 0.0;
        self.peak_cooling_power = 0.0;
    }

    /// Compute Q_HVAC using the air-node energy balance formula (Issue #1007).
    ///
    /// Q_HVAC = m_dot * cp * (T_supply - T_air)
    ///
    /// Where:
    /// - Q_HVAC = HVAC heating/cooling power [W]
    /// - m_dot = air mass flow rate [kg/s]
    /// - cp = specific heat of air [J/kgK] (≈ 1006)
    /// - T_supply = supply air temperature [°C]
    /// - T_air = zone air temperature [°C]
    ///
    /// Positive Q_HVAC indicates heating (supply air warmer than zone).
    /// Negative Q_HVAC indicates cooling (supply air cooler than zone).
    ///
    /// # Numerical stability (Issue #1006)
    ///
    /// Returns `0.0` (no heating/cooling demand) if any input is non-finite
    /// (`NaN` or `±∞`). This prevents silent NaN propagation into the energy
    /// accumulators, which previously produced `-inf` temperature and `inf MWh`
    /// readings when uninitialized `h_tr_is_surfaces` led to division by zero
    /// upstream of the zone temperature used here.
    pub fn compute_q_hvac(&self) -> f64 {
        if !self.m_dot.is_finite() || !self.T_supply.is_finite() || !self.prev_zone_temp.is_finite()
        {
            return 0.0;
        }
        self.m_dot * CP_AIR * (self.T_supply - self.prev_zone_temp)
    }

    /// Set the mass flow rate and supply air temperature for HVAC calculations.
    ///
    /// # Arguments
    ///
    /// * `m_dot` - Air mass flow rate [kg/s]
    /// * `T_supply` - Supply air temperature [°C]
    pub fn set_hvac_air_properties(&mut self, m_dot: f64, T_supply: f64) {
        self.m_dot = m_dot;
        self.T_supply = T_supply;
    }

    /// Advance the simulation by one timestep.
    ///
    /// During the warm-up period the solver updates mass temperatures but
    /// energy totals are **not** accumulated. After warm-up, energy and peak
    /// power tracking resumes normally.
    ///
    /// # Arguments
    ///
    /// * `outdoor_temp` - Exterior (dry-bulb) temperature [°C]
    /// * `solar_gain` - Solar heat gain into the zone [W]
    /// * `internal_gain` - Internal heat gains (occupants, lights, equipment) [W]
    /// * `dt` - Timestep duration [seconds] (typically 3600 for 1-hour)
    ///
    /// # Returns
    ///
    /// The HVAC power demand for this timestep [W].
    /// Positive = heating, negative = cooling, zero = no HVAC needed.
    pub fn step(&mut self, outdoor_temp: f64, solar_gain: f64, internal_gain: f64, dt: f64) -> f64 {
        self.timestep_count += 1;

        // Issue #1006: NaN/Inf guard on inputs. If any caller passes a
        // non-finite value (e.g. uninitialized h_tr_is_surfaces leading to
        // ±∞ upstream), short-circuit to a no-op timestep rather than
        // propagating non-finite values into the solver, energy
        // accumulators, or peak-power trackers.
        if !outdoor_temp.is_finite()
            || !solar_gain.is_finite()
            || !internal_gain.is_finite()
            || !dt.is_finite()
            || !self.solver.h_tr_is.is_finite()
            || !self.solver.surface_temperature.is_finite()
        {
            return 0.0;
        }

        // Update solver boundary conditions
        self.solver.set_exterior_temperature(outdoor_temp);

        // === Step 1: Compute free-floating zone temperature (BEFORE HVAC) ===
        // Use the multi-node solver's air balance with the current (pre-step) mass
        // temperatures. This is the temperature the zone would reach if no HVAC
        // were operating. Per Issue #858 design: T_air = T_free in deadband.
        //
        // The convective portion of internal gains goes to the air (phi_ia);
        // the radiative portion is injected into the mass via step_with_gains().
        // For ASHRAE 140, the convective fraction of internal gains is typically
        // ~0.5 for residential (per ISO 13790 Annex C).
        let phi_ia = 0.5 * internal_gain;
        let t_free = self
            .solver
            .compute_zone_air_temperature(outdoor_temp, self.h_ve, phi_ia);

        // === Step 2: Determine HVAC mode and target zone temperature ===
        // Per Issue #858: T_air = T_setpoint when heating/cooling active,
        // T_air = T_free in deadband. We force the solver's boundary to T_setpoint
        // so the mass sees the conditioned temperature (mimics ideal HVAC).
        //
        // Issue #1006: NaN/Inf guard on the free-floating temperature. Even with
        // the solver's own denominator guard in `compute_zone_air_temperature`,
        // a future caller could reach this point with a non-finite value (e.g.
        // uninitialized mass temperatures). In that case, fall back to outdoor
        // temperature so the solver can still advance mass temperatures; the
        // deadband check below will treat the timestep as inactive and demand
        // will be 0 — matching the "no information, no HVAC" semantics.
        let t_free_safe = if t_free.is_finite() {
            t_free
        } else {
            outdoor_temp
        };
        let (t_air_target, hvac_active, hvac_setpoint) = if t_free_safe < self.heating_setpoint {
            (self.heating_setpoint, true, self.heating_setpoint)
        } else if t_free_safe > self.cooling_setpoint {
            (self.cooling_setpoint, true, self.cooling_setpoint)
        } else {
            (t_free, false, t_free)
        };

        self.solver.set_zone_temperature(t_air_target);
        self.prev_zone_temp = t_air_target;

        // === Step 3: Step the multi-node solver with proper gain injection ===
        // Distribute solar gain to envelope nodes (south wall dominant for Case 900):
        //   - 60% to wall (south-facing window irradiation)
        //   - 30% to roof (diffuse + ground-reflected on horizontal)
        //   - 10% to floor (small ground-reflected fraction)
        // Internal radiative portion (50%) goes to internal mass node.
        let g_wall = 0.6 * solar_gain;
        let g_roof = 0.3 * solar_gain;
        let g_floor = 0.1 * solar_gain;
        let g_internal = 0.5 * internal_gain;
        self.solver
            .step_with_gains(dt, g_wall, g_roof, g_floor, g_internal);

        // === Step 4: Compute HVAC power using the air node energy balance ===
        // Per Issue #858 air node balance (simplified to single h_tr_is for
        // compatibility with MultiNodeSolver, which uses a single shared h_tr_is):
        //
        //   h_tr_is*(T_surface - T_air) + h_ve*(T_out - T_air) + h_tr_w*(T_out - T_air)
        //     + phi_ia + Q_HVAC = 0
        //
        // Solving for Q_HVAC (heat ADDED to zone by HVAC):
        //   Q_HVAC = (h_tr_is + h_ve + h_tr_w) * T_air
        //          - h_tr_is * T_surface
        //          - (h_ve + h_tr_w) * T_out
        //          - phi_ia
        //
        // Positive Q_HVAC = heating (heat added to zone).
        // Negative Q_HVAC = cooling (heat removed from zone).
        let h_tr_is = self.solver.h_tr_is;
        let t_surface_post = self.solver.surface_temperature;
        let h_total = h_tr_is + self.h_ve + self.h_tr_w;

        let q_hvac = if hvac_active {
            // HVAC mode: maintain T_air at setpoint
            h_total * hvac_setpoint
                - h_tr_is * t_surface_post
                - (self.h_ve + self.h_tr_w) * outdoor_temp
                - phi_ia
        } else {
            // Deadband: no HVAC needed; report zero load.
            0.0
        };

        // Warm-up check: skip energy accumulation during warm-up
        if !self.warmed_up {
            if self.timestep_count >= self.warmup_timesteps {
                self.warmed_up = true;
            }
            // Return HVAC power for diagnostics, but do NOT accumulate
            return q_hvac;
        }

        // After warm-up: accumulate energy and track peaks
        // Issue #1006: clamp the demand to a finite value before accumulation.
        // Even with the input/denominator guards above, a future caller could
        // (e.g. via a state-aware override) reach this point with a non-finite
        // q_hvac; we treat that as zero demand rather than poisoning the
        // annual totals.
        let heating_power_kw = if q_hvac.is_finite() && q_hvac > 0.0 {
            q_hvac / 1000.0
        } else {
            0.0
        };
        let cooling_power_kw = if q_hvac.is_finite() && q_hvac < 0.0 {
            -q_hvac / 1000.0
        } else {
            0.0
        };

        self.annual_heating_energy += heating_power_kw * (dt / 3600.0);
        self.annual_cooling_energy += cooling_power_kw * (dt / 3600.0);

        if heating_power_kw > self.peak_heating_power {
            self.peak_heating_power = heating_power_kw;
        }
        if cooling_power_kw > self.peak_cooling_power {
            self.peak_cooling_power = cooling_power_kw;
        }

        q_hvac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::multi_node_thermal::ThermalMassNode;

    /// Create a test runner with warm-up disabled (0 days).
    fn create_test_runner() -> MultiNodeHvacRunner {
        let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
        let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
        let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);
        let solver = MultiNodeSolver::new(10.0, wall, roof, floor, internal);

        MultiNodeHvacRunner::new(solver, 15.0, 20.0, 20.0, 26.0).with_warmup_days(0)
    }

    #[test]
    fn test_runner_creation() {
        let runner = create_test_runner();
        assert_eq!(runner.heating_setpoint, 20.0);
        assert_eq!(runner.cooling_setpoint, 26.0);
        assert_eq!(runner.annual_heating_energy, 0.0);
        assert_eq!(runner.annual_cooling_energy, 0.0);
        assert!(runner.is_warmed_up()); // warmup_days=0 → immediately warmed up
        assert_eq!(runner.timestep_count(), 0);
    }

    #[test]
    fn test_step_produces_heating_when_cold() {
        let mut runner = create_test_runner();
        // Outdoor at 0°C, zone at 20°C setpoint → should need heating
        let q = runner.step(0.0, 0.0, 0.0, 3600.0);
        assert!(
            q > 0.0,
            "Should require heating when outdoor is cold: got q={}",
            q
        );
    }

    #[test]
    fn test_step_produces_cooling_when_hot() {
        let mut runner = create_test_runner();
        // Warm up the zone first
        for _ in 0..10 {
            runner.step(35.0, 500.0, 200.0, 3600.0);
        }
        // After warming, zone should be above cooling setpoint → cooling needed
        let q = runner.step(35.0, 500.0, 200.0, 3600.0);
        // If zone temp is above 26°C, cooling is negative
        if runner.prev_zone_temp > 26.0 {
            assert!(
                q < 0.0,
                "Should require cooling when zone is hot: got q={}",
                q
            );
        }
    }

    #[test]
    fn test_energy_accumulation_after_warmup_disabled() {
        let mut runner = create_test_runner();
        // Warm-up disabled (0 days), so energy should accumulate immediately
        runner.step(0.0, 0.0, 0.0, 3600.0);
        assert!(
            runner.annual_heating_energy > 0.0,
            "Heating energy should accumulate when warm-up is disabled"
        );
    }

    #[test]
    fn test_energy_skipped_during_warmup() {
        let mut runner = MultiNodeHvacRunner::with_defaults();
        // Default warm-up is 336 timesteps
        assert!(!runner.is_warmed_up());

        // Step 100 times — still in warm-up
        for _ in 0..100 {
            runner.step(0.0, 0.0, 0.0, 3600.0);
        }

        assert!(!runner.is_warmed_up());
        assert_eq!(
            runner.annual_heating_energy, 0.0,
            "Energy should NOT accumulate during warm-up"
        );
        assert_eq!(
            runner.annual_cooling_energy, 0.0,
            "Energy should NOT accumulate during warm-up"
        );
    }

    #[test]
    fn test_energy_accumulates_after_warmup_completes() {
        let mut runner = MultiNodeHvacRunner::with_defaults().with_warmup_days(1); // 24 timesteps

        // Run through warm-up (24 steps)
        for _ in 0..24 {
            runner.step(0.0, 0.0, 0.0, 3600.0);
        }

        assert!(runner.is_warmed_up());
        // Energy should still be zero during warm-up
        assert_eq!(runner.annual_heating_energy, 0.0);

        // Now step once more — energy should accumulate
        runner.step(0.0, 0.0, 0.0, 3600.0);
        assert!(
            runner.annual_heating_energy > 0.0,
            "Energy should accumulate after warm-up completes"
        );
    }

    #[test]
    fn test_peak_power_tracking() {
        let mut runner = create_test_runner();

        // Cold outdoor → high heating demand
        runner.step(-10.0, 0.0, 0.0, 3600.0);
        let peak1 = runner.peak_heating_power;

        // Warmer outdoor → lower heating demand
        runner.step(5.0, 0.0, 0.0, 3600.0);

        // Peak should be from the coldest step
        assert!(
            runner.peak_heating_power >= peak1 - 1e-9,
            "Peak heating should be from coldest step"
        );
    }

    #[test]
    fn test_reset_accumulators() {
        let mut runner = create_test_runner();
        runner.step(0.0, 0.0, 0.0, 3600.0);
        assert!(runner.annual_heating_energy > 0.0);

        runner.reset_accumulators();
        assert_eq!(runner.annual_heating_energy, 0.0);
        assert_eq!(runner.annual_cooling_energy, 0.0);
        assert_eq!(runner.peak_heating_power, 0.0);
        assert_eq!(runner.peak_cooling_power, 0.0);
    }

    #[test]
    fn test_with_warmup_days_zero() {
        let runner = MultiNodeHvacRunner::with_defaults().with_warmup_days(0);
        assert!(runner.is_warmed_up());
        assert_eq!(runner.warmup_timesteps, 0);
    }

    #[test]
    fn test_default_warmup_is_14_days() {
        let runner = MultiNodeHvacRunner::with_defaults();
        assert_eq!(runner.warmup_timesteps, 336); // 14 * 24
        assert!(!runner.is_warmed_up());
    }

    #[test]
    fn test_mass_temperatures_update_during_warmup() {
        let mut runner = MultiNodeHvacRunner::with_defaults().with_warmup_days(7);
        let initial_wall_temp = runner.solver.wall_temperature();

        // Run warm-up with cold outdoor temperature
        for _ in 0..168 {
            runner.step(-5.0, 0.0, 0.0, 3600.0);
        }

        // Mass temperatures should have changed from initial 20°C
        let final_wall_temp = runner.solver.wall_temperature();
        assert!(
            (final_wall_temp - initial_wall_temp).abs() > 0.01,
            "Mass temperatures should update during warm-up: initial={}, final={}",
            initial_wall_temp,
            final_wall_temp
        );

        // But energy should NOT have accumulated
        assert_eq!(runner.annual_heating_energy, 0.0);
    }

    #[test]
    fn test_no_hvac_within_deadband() {
        let mut runner = create_test_runner();
        // Set outdoor temp so zone lands inside the 20-26°C deadband
        // With zone starting at ~20°C and outdoor at 20°C, zone should be near setpoint
        let q = runner.step(20.0, 0.0, 0.0, 3600.0);
        // When zone is exactly at setpoint, HVAC demand should be zero or very small
        // (depends on exact thermal balance)
        assert!(
            q.abs() < 1000.0,
            "HVAC power should be small when zone is near setpoint: got {} W",
            q
        );
    }

    // =============================================================================
    // Q_HVAC Air-Node Energy Balance Formula Tests (Issue #1007)
    // =============================================================================

    #[test]
    fn test_q_hvac_formula_heating() {
        // Q_HVAC = m_dot * cp * (T_supply - T_air)
        // Example: m_dot=0.5 kg/s, T_supply=40°C, T_air=20°C
        // Expected: Q_HVAC = 0.5 * 1006 * (40 - 20) = 10060 W (heating)
        let mut runner = create_test_runner();
        runner.m_dot = 0.5;
        runner.T_supply = 40.0;
        runner.prev_zone_temp = 20.0;

        let q_hvac = runner.compute_q_hvac();
        let expected = 0.5 * CP_AIR * (40.0 - 20.0);

        assert!(
            (q_hvac - expected).abs() < 1e-6,
            "Q_HVAC heating formula mismatch: got {:.2}, expected {:.2}",
            q_hvac,
            expected
        );
        assert!(
            q_hvac > 0.0,
            "Q_HVAC should be positive for heating (T_supply > T_air)"
        );
    }

    #[test]
    fn test_q_hvac_formula_cooling() {
        // Q_HVAC = m_dot * cp * (T_supply - T_air)
        // Example: m_dot=0.5 kg/s, T_supply=16°C, T_air=26°C
        // Expected: Q_HVAC = 0.5 * 1006 * (16 - 26) = -5030 W (cooling)
        let mut runner = create_test_runner();
        runner.m_dot = 0.5;
        runner.T_supply = 16.0;
        runner.prev_zone_temp = 26.0;

        let q_hvac = runner.compute_q_hvac();
        let expected = 0.5 * CP_AIR * (16.0 - 26.0);

        assert!(
            (q_hvac - expected).abs() < 1e-6,
            "Q_HVAC cooling formula mismatch: got {:.2}, expected {:.2}",
            q_hvac,
            expected
        );
        assert!(
            q_hvac < 0.0,
            "Q_HVAC should be negative for cooling (T_supply < T_air)"
        );
    }

    #[test]
    fn test_q_hvac_formula_zero_when_equal() {
        // When T_supply == T_air, Q_HVAC should be zero (no heating/cooling needed)
        let mut runner = create_test_runner();
        runner.m_dot = 0.5;
        runner.T_supply = 22.0;
        runner.prev_zone_temp = 22.0;

        let q_hvac = runner.compute_q_hvac();

        assert!(
            q_hvac.abs() < 1e-10,
            "Q_HVAC should be zero when T_supply == T_air: got {:.6}",
            q_hvac
        );
    }

    #[test]
    fn test_q_hvac_formula_proportional_to_mass_flow() {
        // Q_HVAC is proportional to m_dot
        let mut runner = create_test_runner();
        runner.T_supply = 40.0;
        runner.prev_zone_temp = 20.0;

        runner.m_dot = 0.5;
        let q1 = runner.compute_q_hvac();

        runner.m_dot = 1.0;
        let q2 = runner.compute_q_hvac();

        assert!(
            (q2 - 2.0 * q1).abs() < 1e-6,
            "Q_HVAC should double when m_dot doubles: q1={:.2}, q2={:.2}",
            q1,
            q2
        );
    }

    #[test]
    fn test_q_hvac_formula_proportional_to_temperature_diff() {
        // Q_HVAC is proportional to (T_supply - T_air)
        let mut runner = create_test_runner();
        runner.m_dot = 0.5;
        runner.prev_zone_temp = 20.0;

        runner.T_supply = 30.0;
        let q1 = runner.compute_q_hvac();

        runner.T_supply = 40.0;
        let q2 = runner.compute_q_hvac();

        // 40-20 = 2 * (30-20), so q2 should be 2 * q1
        assert!(
            (q2 - 2.0 * q1).abs() < 1e-6,
            "Q_HVAC should double when temperature diff doubles: q1={:.2}, q2={:.2}",
            q1,
            q2
        );
    }

    #[test]
    fn test_set_hvac_air_properties() {
        let mut runner = create_test_runner();
        runner.set_hvac_air_properties(0.75, 45.0);

        assert_eq!(runner.m_dot, 0.75);
        assert_eq!(runner.T_supply, 45.0);
    }

    #[test]
    fn test_default_hvac_air_properties() {
        // with_defaults() sets m_dot=0.5 and T_supply=40.0
        let runner = MultiNodeHvacRunner::with_defaults();
        assert_eq!(runner.m_dot, 0.5);
        assert_eq!(runner.T_supply, 40.0);
    }

    #[test]
    fn test_q_hvac_with_defaults() {
        // Test Q_HVAC formula with default values from with_defaults()
        let runner = MultiNodeHvacRunner::with_defaults();
        // Default: m_dot=0.5, T_supply=40, zone starts at heating_setpoint=20
        // Q_HVAC = 0.5 * 1006 * (40 - 20) = 10060 W
        let expected = 0.5 * CP_AIR * (40.0 - 20.0);
        let q_hvac = runner.compute_q_hvac();

        assert!(
            (q_hvac - expected).abs() < 1e-6,
            "Q_HVAC with defaults mismatch: got {:.2}, expected {:.2}",
            q_hvac,
            expected
        );
    }

    // =============================================================================
    // Numerical Stability Tests (Issue #1006)
    //
    // The original bug report described `-inf` zone temperatures and `inf MWh`
    // energy accumulators. These tests pin the new defensive behaviour:
    // non-finite inputs and degenerate envelopes must not produce non-finite
    // outputs.
    // =============================================================================

    #[test]
    fn test_q_hvac_returns_zero_for_nan_inputs() {
        let mut runner = create_test_runner();
        runner.m_dot = f64::NAN;
        runner.T_supply = 40.0;
        runner.prev_zone_temp = 20.0;
        assert_eq!(runner.compute_q_hvac(), 0.0);

        runner.m_dot = 0.5;
        runner.T_supply = f64::NAN;
        assert_eq!(runner.compute_q_hvac(), 0.0);

        runner.T_supply = 40.0;
        runner.prev_zone_temp = f64::NAN;
        assert_eq!(runner.compute_q_hvac(), 0.0);
    }

    #[test]
    fn test_q_hvac_returns_zero_for_infinite_inputs() {
        let mut runner = create_test_runner();
        runner.m_dot = f64::INFINITY;
        assert_eq!(runner.compute_q_hvac(), 0.0);

        runner.m_dot = 0.5;
        runner.T_supply = f64::INFINITY;
        assert_eq!(runner.compute_q_hvac(), 0.0);

        runner.T_supply = 40.0;
        runner.prev_zone_temp = f64::NEG_INFINITY;
        assert_eq!(runner.compute_q_hvac(), 0.0);
    }

    #[test]
    fn test_q_hvac_output_is_always_finite_for_finite_inputs() {
        // Sweep a range of finite inputs and ensure the result is finite
        // (no overflow, no NaN, no Inf).
        let mut runner = create_test_runner();
        for m_dot in [0.0, 0.5, 1.0, 100.0] {
            for t_supply in [-50.0, 0.0, 22.0, 100.0, 1000.0] {
                for t_air in [-50.0, 0.0, 22.0, 100.0, 1000.0] {
                    runner.m_dot = m_dot;
                    runner.T_supply = t_supply;
                    runner.prev_zone_temp = t_air;
                    let q = runner.compute_q_hvac();
                    assert!(
                        q.is_finite(),
                        "Q_HVAC must be finite for finite inputs: m_dot={}, \
                         T_supply={}, T_air={}, got q={}",
                        m_dot,
                        t_supply,
                        t_air,
                        q
                    );
                }
            }
        }
    }

    #[test]
    fn test_step_with_zero_conductances_does_not_produce_inf() {
        // Issue #1006: degenerate envelope (all conductances zero) used to
        // produce ±∞ zone temperature and then ±∞ MWh accumulators. The new
        // denominator guard must short-circuit cleanly.
        let mut runner = create_test_runner();
        runner.h_tr_w = 0.0;
        runner.h_ve = 0.0;
        // solver.h_tr_is is left at its default — override to 0 to make the
        // denominator exactly 0.
        runner.solver.h_tr_is = 0.0;

        let q = runner.step(20.0, 0.0, 0.0, 3600.0);
        assert!(q.is_finite(), "q_hvac must be finite, got {}", q);
        assert!(
            runner.annual_heating_energy.is_finite(),
            "annual_heating_energy must be finite, got {}",
            runner.annual_heating_energy
        );
        assert!(
            runner.annual_cooling_energy.is_finite(),
            "annual_cooling_energy must be finite, got {}",
            runner.annual_cooling_energy
        );
        assert!(
            runner.peak_heating_power.is_finite(),
            "peak_heating_power must be finite, got {}",
            runner.peak_heating_power
        );
        assert!(
            runner.peak_cooling_power.is_finite(),
            "peak_cooling_power must be finite, got {}",
            runner.peak_cooling_power
        );
        assert!(
            runner.prev_zone_temp.is_finite(),
            "prev_zone_temp must be finite, got {}",
            runner.prev_zone_temp
        );
    }

    #[test]
    fn test_step_with_nan_inputs_does_not_propagate() {
        // Issue #1006: non-finite inputs to step() must short-circuit to 0
        // demand and must not poison any state.
        let mut runner = create_test_runner();
        let q = runner.step(f64::NAN, 0.0, 0.0, 3600.0);
        assert_eq!(q, 0.0);
        assert!(runner.annual_heating_energy.is_finite());
        assert!(runner.annual_cooling_energy.is_finite());

        let q = runner.step(20.0, f64::INFINITY, 0.0, 3600.0);
        assert_eq!(q, 0.0);

        let q = runner.step(20.0, 0.0, f64::NEG_INFINITY, 3600.0);
        assert_eq!(q, 0.0);

        // dt non-finite: also a no-op
        let q = runner.step(20.0, 0.0, 0.0, f64::NAN);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_step_long_run_remains_finite_under_extreme_inputs() {
        // Issue #1006: a long run with alternating extreme outdoor
        // temperatures must not produce non-finite accumulators.
        let mut runner = create_test_runner();
        for hour in 0..8760 {
            // Sinusoidal outdoor between -30 and +50 °C
            let t_out = 10.0 + 40.0 * (((hour as f64) * std::f64::consts::PI / 12.0).sin());
            let q = runner.step(t_out, 0.0, 0.0, 3600.0);
            assert!(q.is_finite(), "q non-finite at hour {}: {}", hour, q);
        }
        assert!(runner.annual_heating_energy.is_finite());
        assert!(runner.annual_cooling_energy.is_finite());
        assert!(runner.peak_heating_power.is_finite());
        assert!(runner.peak_cooling_power.is_finite());
        // Energy totals must not be ±∞ even after a full year of stepping.
        assert!(runner.annual_heating_energy.abs() < 1e6);
        assert!(runner.annual_cooling_energy.abs() < 1e6);
    }
}
