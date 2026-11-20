//! The teacher path's air-loop assembly — Issue #3984, the wiring slice.
//!
//! # What this module is
//!
//! [`TeacherAirLoop`] assembles a [`PressureDrivenAirLoop`] from the
//! quantities a zone spec actually carries (heating/cooling capacities)
//! under **stated, fixed design rules**, and steps it against the teacher
//! envelope's own air state. It is the production air-side route of the
//! gauge (teacher) path: the loop's thermostat — with real dead-band
//! hysteresis on *both* sides of the band — decides the mode, the VAV
//! damper position is solved so the **pressure-driven network** delivers
//! the load-bearing flow, and the ideal sensible coil's duty is the
//! metered HVAC energy of the step.
//!
//! # Composition contract (what the loop owns vs what the envelope keeps)
//!
//! * **The loop owns the conditioning**: mode command (thermostat
//!   hysteresis), supply flow (solved from the loop pressure balance for
//!   the commanded damper — never imposed), and the sensible coil duty.
//! * **The envelope keeps the ventilation**: this model's mechanical
//!   ventilation (the night-ventilation schedule and the hybrid
//!   scheduled-ventilation slot) is *zone-level air exchange* — it enters
//!   the zone balance through the envelope's ventilation conductance
//!   `h_ve`, exactly as on the §C.3 route this module replaces. The loop's
//!   outdoor-air mixing box is **not** attached here (the loop API supports
//!   a mixer-less ring), so the ventilation can never be double-counted
//!   and the envelope's `h_ve` plumbing keeps its §C.3 semantics. The
//!   mixer itself stays validated by the §11 engine-run harness.
//! * **The ledger and the closed-form propagation are unchanged**: the
//!   metered duty is the zone-side net (coil duty; the supply-air exchange
//!   against the zone return nets to it in a mixer-less ring), and the
//!   propagated air temperature is the ISO §C.3 closed form
//!   `t_final = t_free + q/h_coeff` — the same semantics the §C.3 route
//!   used, now driven by the loop's delivered duty instead of an imposed
//!   ideal demand.
//!
//! # Design rules (stated once, tested as identities)
//!
//! * Design supply flow: the larger of the heating capacity flow and the
//!   cooling capacity flow, at the design supply ΔTs below.
//! * Design supply ΔT: **20 K** heating (conventional warm-air design) and
//!   **10 K** cooling (conventional cold-air design).
//! * Fan: shut-off static **500 Pa** (conventional low-pressure duct
//!   system), rated at the design flow with a 3/4·static rise, so the
//!   curve coefficient is `static/(4·Q_d²)`.
//! * Network resistance at the full-open damper: supply duct 25 %, return
//!   duct 25 %, damper 50 % — so the solved flow at a fully open damper is
//!   *exactly* the design flow (identity-tested).
//! * Idle damper position: solved for **25 % of the design flow** — the
//!   conventional minimum-position recirculation. At idle the coil is off
//!   and the supply is the return, so the idle flow is thermally neutral;
//!   its only role is keeping the fan and network in a defined state.
//! * Per-step VAV damper command: the position whose pressure-balance
//!   solve delivers `max(load flow, 25 % floor)`, where the load flow is
//!   the load estimate divided by `cp·ΔT_design` — the flow modulates with
//!   the load and the fan never stops (stated floor).
//! * Thermostat mapping: the arms **engage at the §C.3 setpoints** (heat
//!   at/below `h_sp`, cool at/above `c_sp` — exactly where the §C.3 route
//!   switched) and release one
//!   [`TEACHER_THERMOSTAT_HYSTERESIS_K`] (0.5 K, the conventional
//!   thermostat dead band) into the band, so the controlled variable
//!   dwells inside `(h_sp + δ, c_sp − δ)` — real, non-ideal control per
//!   the issue's acceptance criteria. The module's own convention centers
//!   its dead band on a single setpoint; `refresh_step_inputs` maps the
//!   two §C.3 setpoints onto it exactly.
//! * The conditioned coil is an **ideal sensible coil with discharge-air
//!   control**: the delivered duty is the teacher controller's load
//!   estimate (the envelope's own air-node conductance times the setpoint
//!   error), clamped by the flow the pressure balance actually solves at
//!   the design ΔT (flow limit) and then by the spec capacity — the same
//!   clamp semantics the §C.3 route applied.
//!
//! # What this module is not
//!
//! * It does **not** integrate the loop's own `ZoneAirNode`: the zone
//!   state belongs to the gauge envelope (the loop's standalone zone-node
//!   composition remains the §11/module unit-test harness). The loop's
//!   thermostat, damper, pressure solve and coil semantics are what the
//!   teacher consumes.
//! * It is **sensible-only**: the gauge path carries no humidity state, so
//!   no latent duty is metered here. The loop's latent capability stays
//!   validated by the §11 engine-run harness.
//! * Fan electric power is telemetry only; the thermal ledger books the
//!   delivered air-side duty, exactly as the §C.3 route it replaces did.

use super::fan::STANDARD_AIR_DENSITY_KG_PER_M3;
use super::pressure_air_loop::{
    AirLoopElement, AirLoopError, FanCurve, PressureDrivenAirLoop, ThermostatState, ZoneAirNode,
    ZoneThermostat,
};

/// Design heating supply ΔT [K] — conventional warm-air design value.
pub const TEACHER_HEATING_SUPPLY_DELTA_T_K: f64 = 20.0;
/// Design cooling supply ΔT [K] — conventional cold-air design value.
pub const TEACHER_COOLING_SUPPLY_DELTA_T_K: f64 = 10.0;
/// Fan shut-off static pressure [Pa] — conventional low-pressure duct system.
pub const TEACHER_FAN_STATIC_PRESSURE_PA: f64 = 500.0;
/// Rated fan pressure rise at the design flow [Pa] — 3/4 of the static, so
/// the curve coefficient is `static/(4·Q_d²)`.
const TEACHER_FAN_RATED_RISE_PA: f64 = 0.75 * TEACHER_FAN_STATIC_PRESSURE_PA;
/// Idle damper target flow as a fraction of the design flow — the
/// conventional minimum-position recirculation (thermally neutral: at idle
/// the coil is off and the supply is the return).
const TEACHER_IDLE_FLOW_FRACTION: f64 = 0.25;
/// Thermostat release hysteresis [K] — the conventional thermostat dead
/// band. The arms ENGAGE at the §C.3 setpoints (the comfort-band edges)
/// and RELEASE one hysteresis width into the band, so the controlled
/// variable dwells inside `(h_sp + δ, c_sp − δ)` — real (non-ideal)
/// control, per the issue's acceptance criteria.
pub const TEACHER_THERMOSTAT_HYSTERESIS_K: f64 = 0.5;
/// Dry-air specific heat [J/kg·K] — the loop module's own literal.
const CP_DRY_AIR_J_PER_KG_K: f64 = super::pressure_air_loop::DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K;

/// The spec quantities the teacher assembly is sized from.
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherAirLoopConfig {
    /// Spec heating capacity [W] (the §C.3 clamp value).
    pub heating_capacity_w: f64,
    /// Spec cooling capacity [W] (the §C.3 clamp value).
    pub cooling_capacity_w: f64,
}

/// One conditioned teacher step, produced by [`TeacherAirLoop::step_conditioned`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeacherAirLoopStep {
    /// Signed energy delivered to the zone this step [kWh] — positive
    /// heating, negative cooling, exactly the ledger sign convention the
    /// §C.3 route used.
    pub energy_kwh: f64,
    /// Signed delivered sensible duty [W] (`energy_kwh·3_600_000/dt`).
    pub delivered_duty_w: f64,
    /// Flow solved from the loop pressure balance this step [m³/s].
    pub solved_flow_m3_per_s: f64,
    /// Fan air power at the solved flow [W] (telemetry).
    pub fan_air_power_w: f64,
    /// Thermostat state after this step's update.
    pub thermostat_state: ThermostatState,
    /// Thermostat actuations so far (the hysteresis audit trail).
    pub actuations: u64,
}

/// The teacher path's air loop: a [`PressureDrivenAirLoop`] assembled under
/// the module's stated design rules, stepped against the gauge envelope's
/// air state. See the module documentation for the composition contract.
#[derive(Debug, Clone)]
pub struct TeacherAirLoop {
    air_loop: PressureDrivenAirLoop,
    design_supply_flow_m3_per_s: f64,
    idle_damper_position: f64,
    heating_capacity_w: f64,
    cooling_capacity_w: f64,
    /// The TRUE §C.3 heating setpoint [°C] — the load estimate targets it,
    /// never the thermostat's hysteresis-shifted effective setpoint.
    heating_setpoint_c: f64,
    /// The TRUE §C.3 cooling setpoint [°C].
    cooling_setpoint_c: f64,
    last_thermostat_state: ThermostatState,
}

impl TeacherAirLoop {
    /// Assemble the teacher loop under the stated design rules.
    pub fn from_config(config: TeacherAirLoopConfig) -> Result<Self, AirLoopError> {
        for (field, value) in [
            ("heating_capacity_w", config.heating_capacity_w),
            ("cooling_capacity_w", config.cooling_capacity_w),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(AirLoopError::InvalidParameter {
                    field,
                    requirement: "finite and >= 0",
                    value,
                });
            }
        }
        if config.heating_capacity_w <= 0.0 && config.cooling_capacity_w <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "heating_capacity_w, cooling_capacity_w",
                requirement:
                    "at least one capacity must be > 0 (the route is not dispatched for unconditioned zones)",
                value: 0.0,
            });
        }

        let heat_flow = if config.heating_capacity_w > 0.0 {
            config.heating_capacity_w
                / (STANDARD_AIR_DENSITY_KG_PER_M3
                    * CP_DRY_AIR_J_PER_KG_K
                    * TEACHER_HEATING_SUPPLY_DELTA_T_K)
        } else {
            0.0
        };
        let cool_flow = if config.cooling_capacity_w > 0.0 {
            config.cooling_capacity_w
                / (STANDARD_AIR_DENSITY_KG_PER_M3
                    * CP_DRY_AIR_J_PER_KG_K
                    * TEACHER_COOLING_SUPPLY_DELTA_T_K)
        } else {
            0.0
        };
        let design_flow = heat_flow.max(cool_flow);
        if !design_flow.is_finite() || design_flow <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "design_supply_flow",
                requirement: "finite and > 0 (at least one capacity must be > 0)",
                value: design_flow,
            });
        }

        // Fan: 500 Pa shut-off, rated at the design flow with a 3/4 static
        // rise → curve coefficient static/(4·Q_d²).
        let curve = FanCurve::from_rated_point(
            TEACHER_FAN_STATIC_PRESSURE_PA,
            design_flow,
            TEACHER_FAN_RATED_RISE_PA,
        )?;
        // Total resistance at the fully open damper: the rated rise must be
        // absorbed by the ring, so R_total = rated_rise / Q_d². Split: 25 %
        // supply duct, 25 % return duct, 50 % damper (open).
        let r_total_open = TEACHER_FAN_RATED_RISE_PA / (design_flow * design_flow);
        let r_supply_duct = 0.25 * r_total_open;
        let r_return_duct = 0.25 * r_total_open;
        let r_damper_open = 0.5 * r_total_open;
        let idle_damper_position = Self::damper_position_for_flow(
            r_damper_open,
            r_total_open,
            &curve,
            TEACHER_IDLE_FLOW_FRACTION * design_flow,
        );

        let network = super::pressure_air_loop::AirLoopNetwork::new(
            vec![
                "outdoor".to_string(),
                "fan_discharge".to_string(),
                "zone".to_string(),
                "return".to_string(),
            ],
            vec![
                AirLoopElement::Fan {
                    id: "supply_fan".to_string(),
                    curve,
                    speed_fraction: 1.0,
                },
                AirLoopElement::Duct {
                    id: "supply_duct".to_string(),
                    resistance_pa_s2_per_m6: r_supply_duct,
                },
                AirLoopElement::Damper {
                    id: "zone_damper".to_string(),
                    open_resistance_pa_s2_per_m6: r_damper_open,
                    position: 1.0,
                },
                AirLoopElement::Duct {
                    id: "return_duct".to_string(),
                    resistance_pa_s2_per_m6: r_return_duct,
                },
            ],
        )?;

        // Placeholders — refreshed per step from the live setpoints before
        // any command is taken (the thermostat is never updated without a
        // `refresh_step_inputs` first).
        let thermostat = ZoneThermostat::with_heating_setpoints(
            24.0,
            20.0,
            4.0,
            1.0,
            1.0,
            idle_damper_position,
        )?;
        // The loop's own zone node is NOT integrated on the teacher path
        // (the gauge envelope is the zone); it only carries the placeholder
        // the composition contract requires.
        let zone = ZoneAirNode::new(20.0, 1.0, 0.0, 0.0, 0.0, 20.0, 20.0)?;
        let air_loop = PressureDrivenAirLoop::new(network, thermostat, zone, "zone_damper")?;

        Ok(Self {
            air_loop,
            design_supply_flow_m3_per_s: design_flow,
            idle_damper_position,
            heating_capacity_w: config.heating_capacity_w,
            cooling_capacity_w: config.cooling_capacity_w,
            heating_setpoint_c: 20.0,
            cooling_setpoint_c: 24.0,
            last_thermostat_state: ThermostatState::Idle,
        })
    }

    /// Damper position whose pressure-balance solve delivers the requested
    /// flow (the VAV command inversion).
    ///
    /// The ring at damper position `p` carries `R_open/p²` plus the fixed
    /// ducts; the operating point closes `φ²·P_static = (c + ΣR)·Q²`, so
    /// for the requested flow `Q`:
    /// `p = sqrt(R_open / (P_static/Q² − c − R_ducts))`.
    fn damper_position_for_flow(
        r_damper_open: f64,
        r_total_open: f64,
        curve: &FanCurve,
        requested_flow_m3_per_s: f64,
    ) -> f64 {
        if !requested_flow_m3_per_s.is_finite() || requested_flow_m3_per_s <= 0.0 {
            return 0.0;
        }
        let r_needed = curve.static_pressure_pa
            / (requested_flow_m3_per_s * requested_flow_m3_per_s)
            - curve.curve_coefficient_pa_s2_per_m6;
        let r_damper_needed = r_needed - (r_total_open - r_damper_open);
        if r_damper_needed <= r_damper_open {
            // The requested flow is at or above the full-open operating
            // point: the damper stays fully open (any shortfall below the
            // requested flow is fan-limited, which the solved flow then
            // reports honestly).
            return 1.0;
        }
        let position = (r_damper_open / r_damper_needed).sqrt();
        position.clamp(0.0, 1.0)
    }

    /// The design supply flow [m³/s] (the full-open operating point).
    pub fn design_supply_flow_m3_per_s(&self) -> f64 {
        self.design_supply_flow_m3_per_s
    }

    /// The idle damper position (the 25 % recirculation solve).
    pub fn idle_damper_position(&self) -> f64 {
        self.idle_damper_position
    }

    /// The thermostat actuations so far (the hysteresis audit trail).
    pub fn actuations(&self) -> u64 {
        self.air_loop.thermostat().actuations()
    }

    /// The thermostat state after the most recent step.
    pub fn last_state(&self) -> ThermostatState {
        self.last_thermostat_state
    }

    /// Refresh the per-step scheduled setpoints. The ventilation stays
    /// envelope-owned on this path, so the setpoint mapping is the only
    /// per-step refresh. See the doc comment on the mapping semantics.
    pub fn refresh_step_inputs(
        &mut self,
        heating_setpoint_c: f64,
        cooling_setpoint_c: f64,
    ) -> Result<(), AirLoopError> {
        if !heating_setpoint_c.is_finite() || !cooling_setpoint_c.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "setpoints",
                requirement: "finite",
                value: heating_setpoint_c.min(cooling_setpoint_c),
            });
        }
        let hysteresis = TEACHER_THERMOSTAT_HYSTERESIS_K;
        // Module convention: engage at s + half, release at s − half.
        // Cooling must engage at c_sp → s = c_sp − hysteresis/2 (release
        // then lands at c_sp − hysteresis). Heating must engage at h_sp →
        // s = h_sp + hysteresis/2 (release at h_sp + hysteresis).
        let cooling_effective = cooling_setpoint_c - 0.5 * hysteresis;
        let heating_effective = heating_setpoint_c + 0.5 * hysteresis;
        self.heating_setpoint_c = heating_setpoint_c;
        self.cooling_setpoint_c = cooling_setpoint_c;
        self.air_loop.thermostat_mut().set_setpoints(
            cooling_effective,
            Some(heating_effective),
            hysteresis,
        )
    }

    /// Step one conditioned hour against the teacher envelope's free-float
    /// final air temperature.
    ///
    /// `h_coeff_w_per_k` is the envelope's own air-node conductance — the
    /// same coefficient the dispatcher's closed-form propagation uses, so
    /// the estimate the coil modulates to and the ledger's propagation
    /// share one consistent quantity.
    pub fn step_conditioned(
        &mut self,
        t_air_free_final_c: f64,
        h_coeff_w_per_k: f64,
        dt_s: f64,
    ) -> Result<TeacherAirLoopStep, AirLoopError> {
        if !t_air_free_final_c.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "t_air_free_final_c",
                requirement: "finite",
                value: t_air_free_final_c,
            });
        }
        if !h_coeff_w_per_k.is_finite() || h_coeff_w_per_k < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "h_coeff_w_per_k",
                requirement: "finite and >= 0",
                value: h_coeff_w_per_k,
            });
        }
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "dt_s",
                requirement: "finite and > 0",
                value: dt_s,
            });
        }
        // The thermostat reads the envelope's air state (not the loop's
        // placeholder zone node).
        let command = {
            let thermostat = self.air_loop.thermostat_mut();
            thermostat.update(t_air_free_final_c)
        };
        self.last_thermostat_state = command.state;

        // The delivered duty of this step and the flow it is carried by:
        // * Idle — the coil is off and the supply is the return: zero duty,
        //   the damper holds the minimum-recirculation position.
        // * Heating/Cooling — the VAV command solves the damper position
        //   for `max(load flow, idle floor)`, the pressure balance delivers
        //   the flow, and the coil duty is the load estimate clamped by the
        //   flow limit (the solved flow at the design ΔT) and the spec
        //   capacity — the §C.3 clamp semantics, now flow-aware.
        let (damper_position, delivered_duty_w) = match command.state {
            ThermostatState::Idle => (self.idle_damper_position, 0.0),
            ThermostatState::Heating => {
                // The estimate targets the TRUE §C.3 setpoint — never the
                // thermostat's hysteresis-shifted effective value.
                let estimate =
                    h_coeff_w_per_k * (self.heating_setpoint_c - t_air_free_final_c).max(0.0);
                let load_flow =
                    estimate / (CP_DRY_AIR_J_PER_KG_K * TEACHER_HEATING_SUPPLY_DELTA_T_K);
                let target_flow = load_flow
                    .max(TEACHER_IDLE_FLOW_FRACTION * self.design_supply_flow_m3_per_s)
                    .min(self.design_supply_flow_m3_per_s);
                let damper = self.damper_position_for_current_resistances(target_flow);
                let flow_limit = self.solve_flow_limit_w(damper, TEACHER_HEATING_SUPPLY_DELTA_T_K);
                let duty = estimate
                    .min(flow_limit)
                    .min(self.heating_capacity_w.max(0.0));
                (damper, duty.max(0.0))
            }
            ThermostatState::Cooling => {
                let estimate =
                    -h_coeff_w_per_k * (t_air_free_final_c - self.cooling_setpoint_c).max(0.0);
                let load_flow =
                    estimate.abs() / (CP_DRY_AIR_J_PER_KG_K * TEACHER_COOLING_SUPPLY_DELTA_T_K);
                let target_flow = load_flow
                    .max(TEACHER_IDLE_FLOW_FRACTION * self.design_supply_flow_m3_per_s)
                    .min(self.design_supply_flow_m3_per_s);
                let damper = self.damper_position_for_current_resistances(target_flow);
                let flow_limit = self.solve_flow_limit_w(damper, TEACHER_COOLING_SUPPLY_DELTA_T_K);
                let duty = estimate
                    .max(-flow_limit)
                    .max(-self.cooling_capacity_w.max(0.0));
                (damper, duty.min(0.0))
            }
        };
        self.air_loop.command_damper(damper_position)?;
        let solution = self
            .air_loop
            .network_mut()
            .solve_with_return_humidity_ratio(0.0)?;
        let energy_kwh = delivered_duty_w * dt_s / 3_600_000.0;
        Ok(TeacherAirLoopStep {
            energy_kwh,
            delivered_duty_w,
            solved_flow_m3_per_s: solution.flow_m3_per_s,
            fan_air_power_w: solution.fan_air_power_w,
            thermostat_state: command.state,
            actuations: self.air_loop.thermostat().actuations(),
        })
    }

    /// The damper position for a requested flow, from the assembly's own
    /// stored resistances.
    fn damper_position_for_current_resistances(&self, requested_flow_m3_per_s: f64) -> f64 {
        let (r_damper_open, r_total_open) = self.resistances();
        Self::damper_position_for_flow(
            r_damper_open,
            r_total_open,
            &self.fan_curve(),
            requested_flow_m3_per_s,
        )
    }

    /// The duty the solved flow carries at the given supply ΔT [W].
    ///
    /// Runs the actual pressure-balance solve at the commanded damper
    /// position, so the flow limit is the *solved* flow's carrying
    /// capacity — fan-limited hours report honestly through it.
    fn solve_flow_limit_w(&self, damper_position: f64, supply_delta_t_k: f64) -> f64 {
        let mut probe = self.air_loop.clone();
        probe
            .command_damper(damper_position)
            .and_then(|()| probe.network_mut().solve_with_return_humidity_ratio(0.0))
            .map(|solution| {
                STANDARD_AIR_DENSITY_KG_PER_M3
                    * solution.flow_m3_per_s
                    * CP_DRY_AIR_J_PER_KG_K
                    * supply_delta_t_k
            })
            .unwrap_or(0.0)
    }

    fn fan_curve(&self) -> FanCurve {
        match self.air_loop.network().elements().first() {
            Some(AirLoopElement::Fan { curve, .. }) => curve.clone(),
            _ => unreachable!("the teacher loop's first element is the fan"),
        }
    }

    fn resistances(&self) -> (f64, f64) {
        let mut damper_open = 0.0;
        let mut ducts = 0.0;
        for element in self.air_loop.network().elements() {
            match element {
                AirLoopElement::Duct {
                    resistance_pa_s2_per_m6,
                    ..
                } => ducts += resistance_pa_s2_per_m6,
                AirLoopElement::Damper {
                    open_resistance_pa_s2_per_m6,
                    ..
                } => damper_open = *open_resistance_pa_s2_per_m6,
                AirLoopElement::Fan { .. } => {}
            }
        }
        (damper_open, damper_open + ducts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference teacher assembly: 6 kW heating / 3 kW cooling — whose
    /// capacity flows at the design ΔTs are 0.248756… and 0.248756… m³/s
    /// respectively (the 20 K / 10 K ratio), so the design flow is shared.
    fn reference_config() -> TeacherAirLoopConfig {
        TeacherAirLoopConfig {
            heating_capacity_w: 6000.0,
            cooling_capacity_w: 3000.0,
        }
    }

    fn reference_loop() -> TeacherAirLoop {
        TeacherAirLoop::from_config(reference_config()).expect("the reference loop assembles")
    }

    /// The design flow of the reference config, closed-form.
    fn reference_design_flow() -> f64 {
        let rho = STANDARD_AIR_DENSITY_KG_PER_M3;
        let cp = CP_DRY_AIR_J_PER_KG_K;
        let heat: f64 = 6000.0 / (rho * cp * 20.0);
        let cool = 3000.0 / (rho * cp * 10.0);
        heat.max(cool)
    }

    #[test]
    fn design_flow_is_the_max_of_the_capacity_flows() {
        let loop_ = reference_loop();
        let expected = reference_design_flow();
        assert!(
            (loop_.design_supply_flow_m3_per_s() - expected).abs() < 1.0e-12,
            "design flow {} != closed form {}",
            loop_.design_supply_flow_m3_per_s(),
            expected
        );
    }

    #[test]
    fn full_open_damper_solves_exactly_the_design_flow() {
        // A deep heating call clamps at the flow limit ṁ_d·cp·ΔT_h — which
        // equals the heating capacity only if the solved full-open flow IS
        // the design flow. The two sizing rules close on each other.
        let mut loop_ = reference_loop();
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        let step = loop_
            .step_conditioned(0.0, 5.0e5, 3600.0)
            .expect("the step solves");
        let expected = 6000.0; // heat cap == ṁ_d·cp·20 K by the sizing rule
        assert!(
            (step.delivered_duty_w - expected).abs() < 1.0e-6,
            "delivered {} != design-flow capacity {}",
            step.delivered_duty_w,
            expected
        );
    }

    #[test]
    fn unclamped_heating_step_delivers_exactly_the_load_estimate() {
        let mut loop_ = reference_loop();
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        // h = 50 W/K, error 10 K → 500 W, far below every clamp.
        let step = loop_
            .step_conditioned(10.0, 50.0, 3600.0)
            .expect("the step solves");
        assert!(
            (step.delivered_duty_w - 500.0).abs() < 1.0e-9,
            "delivered {} != estimate 500",
            step.delivered_duty_w
        );
        assert_eq!(step.thermostat_state, ThermostatState::Heating);
    }

    #[test]
    fn spec_capacity_clamps_before_the_flow_limit_when_tighter() {
        let mut loop_ = TeacherAirLoop::from_config(TeacherAirLoopConfig {
            heating_capacity_w: 4000.0, // tighter than ṁ_d·cp·20 K
            cooling_capacity_w: 3000.0,
        })
        .expect("the loop assembles");
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        let step = loop_
            .step_conditioned(0.0, 5.0e5, 3600.0)
            .expect("the step solves");
        assert!(
            (step.delivered_duty_w - 4000.0).abs() < 1.0e-6,
            "delivered {} != cap 4000",
            step.delivered_duty_w
        );
    }

    #[test]
    fn idle_step_delivers_zero_duty_and_keeps_the_fan_defined() {
        let mut loop_ = reference_loop();
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        // Setpoints 20/24 with the zone at 22: inside the band → Idle.
        let step = loop_
            .step_conditioned(22.0, 50.0, 3600.0)
            .expect("the step solves");
        assert_eq!(step.thermostat_state, ThermostatState::Idle);
        assert_eq!(step.delivered_duty_w, 0.0);
        assert_eq!(step.energy_kwh, 0.0);
        // The fan still runs at the minimum-recirculation position: a
        // defined, nonzero solved flow (thermally neutral — supply =
        // return at idle).
        assert!(step.solved_flow_m3_per_s > 0.0);
        assert!(
            (step.solved_flow_m3_per_s
                - TEACHER_IDLE_FLOW_FRACTION * loop_.design_supply_flow_m3_per_s())
            .abs()
                < 1.0e-9,
            "idle flow {} != the 25 % recirculation solve {}",
            step.solved_flow_m3_per_s,
            TEACHER_IDLE_FLOW_FRACTION * loop_.design_supply_flow_m3_per_s()
        );
    }

    #[test]
    fn cooling_step_delivers_the_negative_estimate_with_the_cooling_clamps() {
        let mut loop_ = reference_loop();
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        // Zone at 30 °C, h = 50 W/K, c_sp 24 → estimate −300 W.
        let step = loop_
            .step_conditioned(30.0, 50.0, 3600.0)
            .expect("the step solves");
        assert_eq!(step.thermostat_state, ThermostatState::Cooling);
        assert!((step.delivered_duty_w + 300.0).abs() < 1.0e-9);
        // Capacity clamp: a deep cooling call clamps at −3 kW.
        let step = loop_
            .step_conditioned(60.0, 5.0e5, 3600.0)
            .expect("the step solves");
        assert!(
            (step.delivered_duty_w + 3000.0).abs() < 1.0e-6,
            "delivered {} != −cap 3000",
            step.delivered_duty_w
        );
    }

    #[test]
    fn energy_matches_duty_times_dt() {
        let mut loop_ = reference_loop();
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        let step = loop_
            .step_conditioned(10.0, 50.0, 3600.0)
            .expect("the step solves");
        assert!((step.energy_kwh - step.delivered_duty_w * 3600.0 / 3_600_000.0).abs() < 1.0e-15);
    }

    #[test]
    fn the_command_series_shows_dead_band_dwell() {
        let mut loop_ = reference_loop();
        loop_
            .refresh_step_inputs(20.0, 24.0)
            .expect("the refresh is valid");
        // Heating engages AT the heating setpoint (the §C.3 entry edge —
        // here the module's centered convention is mapped so the engage
        // edge is h_sp = 20 °C) and HOLDS through the release hysteresis
        // (engages at 17 °C, held at 20.3 < 20 + δ): a stateless edge
        // comparison would re-decide every hour.
        loop_.step_conditioned(17.0, 50.0, 3600.0).expect("step");
        assert_eq!(loop_.last_state(), ThermostatState::Heating);
        assert_eq!(loop_.actuations(), 1);
        loop_.step_conditioned(20.3, 50.0, 3600.0).expect("step");
        assert_eq!(loop_.last_state(), ThermostatState::Heating);
        loop_.step_conditioned(20.6, 50.0, 3600.0).expect("step");
        assert_eq!(loop_.last_state(), ThermostatState::Idle);
        assert_eq!(loop_.actuations(), 2);
        // The cooling arm engages at c_sp = 24 (not half a band above it).
        loop_.step_conditioned(24.2, 50.0, 3600.0).expect("step");
        assert_eq!(loop_.last_state(), ThermostatState::Cooling);
        assert_eq!(loop_.actuations(), 3);
        // Cooling releases one hysteresis width into the band.
        loop_.step_conditioned(24.0, 50.0, 3600.0).expect("step");
        assert_eq!(loop_.last_state(), ThermostatState::Cooling);
        loop_.step_conditioned(23.4, 50.0, 3600.0).expect("step");
        assert_eq!(loop_.last_state(), ThermostatState::Idle);
        assert_eq!(loop_.actuations(), 4);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(TeacherAirLoop::from_config(TeacherAirLoopConfig {
            heating_capacity_w: -1.0,
            cooling_capacity_w: 3000.0,
        })
        .is_err());
        // Zero capacity on both sides: the route has nothing to dispatch.
        assert!(TeacherAirLoop::from_config(TeacherAirLoopConfig {
            heating_capacity_w: 0.0,
            cooling_capacity_w: 0.0,
        })
        .is_err());
    }
}
