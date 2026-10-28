//! Mass state — thermal mass, capacitances, surface temps, mass energies, mass conductances.
//!
//! Extracted from `ThermalModelData` (Issue #2878). Owns every per-zone mass /
//! capacitance / surface-temperature tensor (5R1C lumped mass, 6R2C envelope
//! vs internal split, 9R4C ceiling/floor/partition split, partitioned LW
//! surface temps, air-node ODE state) plus the mass-energy bookkeeping fields
//! and the few `h_tr_*` conductance vectors that are conceptually part of
//! the mass-side thermal network rather than the boundary-side network.

use super::ContinuousTensor;
use fluxion_core::multi_node::MultiNodeThermalMass;

pub struct MassState<T: ContinuousTensor<f64>> {
    // Lumped mass + capacitances (5R1C).
    pub mass_temperatures: T,
    pub thermal_capacitance: T,
    /// Air-node thermal capacitance C_air = ρ_air · cp_air · V_zone  [J/K] — Issue #1522.
    pub air_thermal_capacitance: T,
    /// Independent air-node temperature state for the 5R1C model — Issue #1585.
    pub air_temperatures: T,
    /// Issue #2339 — number of sub-steps for the air-node ODE update.
    pub sub_hour_air_node_steps: u32,

    /// Issue #1860 — first-order low-pass solar-lag state.
    pub solar_lag: T,
    /// Issue #1860 — independent interior wall-surface ODE state (5R1C).
    pub wall_surface_temperatures: T,
    /// Issue #2890 — partitioned LW interior surface temperatures.
    pub surface_temp_floor: T,
    /// See [`Self::surface_temp_floor`].
    pub surface_temp_ceiling: T,
    /// See [`Self::surface_temp_floor`].
    pub surface_temp_wall: T,

    // 6R2C envelope vs internal split.
    pub envelope_mass_temperatures: T,
    pub internal_mass_temperatures: T,
    pub envelope_thermal_capacitance: T,
    pub internal_thermal_capacitance: T,
    /// Conductance between envelope and internal mass.
    pub h_tr_me: T,

    // 8R3C / 9R4C per-mass-node states.
    pub ceiling_mass_temperatures: Option<T>,
    pub floor_mass_temperatures: Option<T>,
    pub partition_mass_temperatures: Option<T>,
    pub ceiling_thermal_capacitance: Option<T>,
    pub floor_thermal_capacitance: Option<T>,
    pub partition_thermal_capacitance: Option<T>,
    pub h_tr_ceiling: Option<T>,
    pub h_tr_floor_mass: Option<T>,
    pub h_tr_partition: Option<T>,

    // Energy bookkeeping for the 6R2C path (Issue #272, #274, #275, #432).
    pub previous_mass_temperatures: T,
    pub mass_energy_change_cumulative: f64,
    pub envelope_mass_energy_change_cumulative: f64,
    pub internal_mass_energy_change_cumulative: f64,

    /// Multi-node thermal mass state for the 9R4C model.
    pub multi_node_thermal_mass: Option<MultiNodeThermalMass>,

    /// [R70 HARNESS — scratch, env-gated (loop round 70, #3983 D-b). Dynamic
    /// per-class interior surface-node state `t_s_c` (wall/roof/floor × zone,
    /// flat 3·num_zones) dynamic per-class interior surface-node state `t_s_c`
    /// (wall/roof/floor × zone). D5 default since loop round 71 (#3983 §57):
    /// the surface node is advanced first (backward Euler, consuming its own
    /// previous state) before the mass node, so `phi_st_c` acquires a step of
    /// real dynamics. `None` until the first step initializes it algebraically.]
    pub r70_surface_node_temperatures: Option<T>,

    /// Previous-step per-class interior surface-node temperatures (loop
    /// round 73, #1522/#4504): the `t_s_old` snapshot the t_s-first backward
    /// Euler advance consumed, persisted alongside the advanced state so the
    /// strict-energy invariant checker can recompute the surface-node
    /// identity exactly. Write-only for the engine (never read back by the
    /// integrator); `None` before the first step.
    pub r70_surface_node_temperatures_prev: Option<T>,

    /// [R71 — sub-hour driver resolution, default since loop round 73
    /// (#1522; env-gated harness `FLX_R71_AIR_CAP` in round 72). Previous-hour per-zone air-node ODE
    /// drivers — the numerator WITHOUT the night-vent term
    /// (`num_tm + num_rest`) and the denominator — persisted at the end of
    /// each step so the sub-step loop can linearly ramp the drivers across
    /// the hour instead of freezing them at their top-of-hour values
    /// (dt/τ_air ≈ 3.6 makes the frozen-driver steady state the dominant
    /// air-node response; #1585 gave the node true C_air but not
    /// sub-hour drivers). Zero-initialized; the first step detects zero and
    /// seeds with the current values (no ramp on step 1).]
    pub r71_air_driver_num_prev: T,
    pub r71_air_driver_den_prev: T,

    /// [R90 HARNESS — §LIMIT-40 storage-release split discriminator
    /// (`FLX_R90_PHIST_RAMP`): previous-hour gain drives (phi_st, phi_m),
    /// persisted at the end of each step so the sub-step loop can ramp the
    /// mass-node CN drive and the R77 lag-filter input across the hour at
    /// the R71 midpoint fraction instead of freezing both at their
    /// top-of-hour values. Zero-initialized; the first step detects zero
    /// and seeds with the current values (no ramp on step 1, mirroring the
    /// R71 zero-prev seed convention). The flag-off default never reads
    /// these fields, so unconditional persistence keeps that build
    /// bit-identical.]
    pub r90_phi_st_prev: T,
    pub r90_phi_m_prev: T,

    /// [R75 HARNESS — #3983 free-float structural question, round 75.
    /// `FLX_R75_VENT_RAMP`: previous-hour night-vent numerator term
    /// (`h_ve_night · T_outdoor`, 0 when the vent was inactive) and a
    /// first-step seed flag. The default applies the binary vent term at
    /// its current-hour value inside the sub-step loop (spec-correct: the
    /// 140 night-vent schedule holds for the whole hour, §5.2.1.10 /
    /// Table 5-160); the arm interpolates it across the existing sub-steps
    /// at the R71 midpoint fraction, resolving the 07:00 / 18:00 schedule
    /// transitions intra-hour. Harness arm: the default build never reads
    /// these fields (bit-identical).]
    pub r75_vent_term_prev: T,
    pub r75_vent_has_prev: bool,

    /// [R76 HARNESS — #3983 sub-hour mass-state resolution, round 76.
    /// `FLX_R76_MASS_SUBSTEPS`: previous-hour per-zone sol-air driving
    /// temperature and a first-step seed flag, persisted at the end of each
    /// step so the sub-stepped mass advance can linearly ramp the exterior
    /// driving term across the hour (same convention as the R71 driver
    /// ramp). Harness arm: the default build never reads these fields
    /// (bit-identical).]
    pub r76_sol_air_prev: T,
    pub r76_has_prev: bool,

    /// [Round 78 (#4512) — pooled t_s-first advance workspaces.] Reusable
    /// `t_s_old` / `t_s_new` buffers for the D5 per-class surface-node
    /// advance (mass_state only; `None` until the first step). The D5
    /// default flip (2e51d3ff, round 71) rebuilt both per zone per step
    /// (`to_vec` + `clone`) — ~2 heap allocations per timestep that
    /// breached the dhat BatchOracle allocation budget (#4512) while the
    /// Performance Dashboard dhat job was skipped. Arithmetic unchanged:
    /// the buffers carry the same values the per-step allocations did
    /// (bit-identical). Engine-private scratch; never read back before
    /// being overwritten.
    pub r70_ts_old_workspace: Option<Vec<f64>>,
    pub r70_ts_new_workspace: Option<Vec<f64>>,

    /// [Issue #4555 — thermostat sensor node (TRV sensor temperature).]
    /// Optional bank of grey-box sensor nodes (see
    /// [`crate::sim::sensor_node::SensorNodeBank`]). The nodes are
    /// *observational*: they read the zone air temperature and the radiator
    /// activity and write only their own state, so they cannot change the
    /// zone energy balance. `None` (the default) means no sensor node is
    /// attached and the physics path is untouched.
    pub sensor_nodes: Option<crate::sim::sensor_node::SensorNodeBank>,

    /// [Round 78 — coupled-scheme checker record (#3983 §64,
    /// checker-lags-integrator, #4504 class).] Per zone, the linear
    /// coefficients the coupled single-N mass advance consumed this step,
    /// `[a_0, b_0, …, a_{N−1}, b_{N−1}, denom_final]` (odd length ≥ 3;
    /// `denom_final` in W/K scales the composite residual to W). The
    /// hourly mass identity no longer matches a mass node advanced by the
    /// coupled dt/N-mapped sub-step scheme, so `invariant_checker` mirrors
    /// the composite from this record instead. Write-only for the engine;
    /// `None` on any step that did not run the coupled scheme.
    pub r77_mass_substep_coeffs: Option<Vec<Vec<f64>>>,

    /// [Round 105 — air-node affine record (§LIMIT-19/#1344 checker
    /// air-node mirror; plan:
    /// docs/implementation-plans/checker-air-node-mirror-plan.md §4).]
    /// Per zone, the affine coefficients the coupled single-N AIR-node
    /// advance consumed this step, five entries per sub-step,
    /// `[D_0, M_0, W_0, den_0, h3od_0, …, h3od_{N−1}]`, pushed in
    /// lockstep with `r77_mass_substep_coeffs` (equal per-zone sub-step
    /// counts; `len % 5 == 0`). `D` is the consumed relax factor,
    /// `M`/`W` the mass-refresh/load response coefficients
    /// `(1−D)·coef/den` (`coef` read on the branch that produced it:
    /// R76 refresh, R71 ramp, R93/R94 lag-route share with the R88
    /// scaling), `den` the ramped denominator actually consumed,
    /// `h3od` the mass advance's air-route response coefficient —
    /// `Σ_c w_cap·h3_c/den_c` on the per-class branch, `h_tr3_arg_i/denom_m`
    /// lumped — the gain from the `t_i_sub` response to the mass
    /// response (the lumped conductance ratio is NOT the per-class
    /// route; the record carries the consumed composite).
    /// Write-only for the engine — the zero-delta checker replay
    /// consumes committed states (the #3297 residual-zero property is
    /// untouched); the checker's air-route mirror consumes the record in
    /// a later increment. `None` on any step that did not run the
    /// coupled scheme on the default integrator route: the R74
    /// ramped-forcing and R90 filtered-lag archaeology arms have map
    /// shapes this tuple cannot express and keep the committed-air
    /// convention (no record).
    pub r77_air_substep_coeffs: Option<Vec<Vec<f64>>>,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for MassState<T> {
    fn clone(&self) -> Self {
        Self {
            mass_temperatures: self.mass_temperatures.clone(),
            thermal_capacitance: self.thermal_capacitance.clone(),
            air_thermal_capacitance: self.air_thermal_capacitance.clone(),
            air_temperatures: self.air_temperatures.clone(),
            sub_hour_air_node_steps: self.sub_hour_air_node_steps,

            solar_lag: self.solar_lag.clone(),
            wall_surface_temperatures: self.wall_surface_temperatures.clone(),
            surface_temp_floor: self.surface_temp_floor.clone(),
            surface_temp_ceiling: self.surface_temp_ceiling.clone(),
            surface_temp_wall: self.surface_temp_wall.clone(),

            envelope_mass_temperatures: self.envelope_mass_temperatures.clone(),
            internal_mass_temperatures: self.internal_mass_temperatures.clone(),
            envelope_thermal_capacitance: self.envelope_thermal_capacitance.clone(),
            internal_thermal_capacitance: self.internal_thermal_capacitance.clone(),
            h_tr_me: self.h_tr_me.clone(),

            ceiling_mass_temperatures: self.ceiling_mass_temperatures.clone(),
            floor_mass_temperatures: self.floor_mass_temperatures.clone(),
            partition_mass_temperatures: self.partition_mass_temperatures.clone(),
            ceiling_thermal_capacitance: self.ceiling_thermal_capacitance.clone(),
            floor_thermal_capacitance: self.floor_thermal_capacitance.clone(),
            partition_thermal_capacitance: self.partition_thermal_capacitance.clone(),
            h_tr_ceiling: self.h_tr_ceiling.clone(),
            h_tr_floor_mass: self.h_tr_floor_mass.clone(),
            h_tr_partition: self.h_tr_partition.clone(),

            previous_mass_temperatures: self.previous_mass_temperatures.clone(),
            mass_energy_change_cumulative: self.mass_energy_change_cumulative,
            envelope_mass_energy_change_cumulative: self.envelope_mass_energy_change_cumulative,
            internal_mass_energy_change_cumulative: self.internal_mass_energy_change_cumulative,

            multi_node_thermal_mass: self.multi_node_thermal_mass.clone(),
            r70_surface_node_temperatures: self.r70_surface_node_temperatures.clone(),
            r70_surface_node_temperatures_prev: self.r70_surface_node_temperatures_prev.clone(),
            r71_air_driver_num_prev: self.r71_air_driver_num_prev.clone(),
            r71_air_driver_den_prev: self.r71_air_driver_den_prev.clone(),
            r90_phi_st_prev: self.r90_phi_st_prev.clone(),
            r90_phi_m_prev: self.r90_phi_m_prev.clone(),
            r75_vent_term_prev: self.r75_vent_term_prev.clone(),
            r75_vent_has_prev: self.r75_vent_has_prev,
            r76_sol_air_prev: self.r76_sol_air_prev.clone(),
            r76_has_prev: self.r76_has_prev,
            r70_ts_old_workspace: self.r70_ts_old_workspace.clone(),
            r70_ts_new_workspace: self.r70_ts_new_workspace.clone(),
            r77_mass_substep_coeffs: self.r77_mass_substep_coeffs.clone(),
            r77_air_substep_coeffs: self.r77_air_substep_coeffs.clone(),
            sensor_nodes: self.sensor_nodes.clone(),
        }
    }
}
