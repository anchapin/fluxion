//! HVAC controller module
//!
//! HVAC mode state machine and IdealHVACController implementation.

/// Determines whether HVAC is actively controlling temperature or just tracking it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HvacSystemMode {
    /// Normal HVAC operation with heating/cooling based on setpoints
    #[default]
    Controlled,
    /// Free-floating mode: no HVAC, track temperatures only
    /// Used for ASHRAE 140 FF cases (600FF, 900FF, 650FF, 950FF)
    FreeFloat,
}

/// HVAC operation mode for dual setpoint control.
///
/// The HVAC system operates in three modes based on zone temperature:
/// - `Heating`: Zone temperature is below heating setpoint
/// - `Cooling`: Zone temperature is above cooling setpoint
/// - `Off`: Zone temperature is within the deadband (between heating and cooling setpoints)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HVACMode {
    Heating,
    Cooling,
    Off,
}

/// Ideal HVAC controller with deadband and staging support.
///
/// This controller implements ASHRAE 140 compliant HVAC control with:
/// - Dual setpoint control (heating and cooling)
/// - Deadband between heating and cooling setpoints
/// - Optional staging for multi-stage systems
/// - Proportional control near setpoints to prevent cycling
#[derive(Clone, Debug)]
pub struct IdealHVACController {
    /// Heating setpoint (°C)
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Deadband tolerance (°C) - prevents rapid cycling near setpoints
    pub deadband_tolerance: f64,
    /// Number of heating stages (1 = single stage, 2+ = multi-stage)
    pub heating_stages: u8,
    /// Number of cooling stages (1 = single stage, 2+ = multi-stage)
    pub cooling_stages: u8,
    /// Maximum heating capacity per stage (W)
    pub heating_capacity_per_stage: f64,
    /// Maximum cooling capacity per stage (W)
    pub cooling_capacity_per_stage: f64,
}

impl IdealHVACController {
    /// Creates a new ideal HVAC controller with specified setpoints.
    pub fn new(heating_setpoint: f64, cooling_setpoint: f64) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            heating_stages: 1,
            cooling_stages: 1,
            heating_capacity_per_stage: 100_000.0,
            cooling_capacity_per_stage: 100_000.0,
        }
    }

    /// Creates a controller with staging support.
    pub fn with_stages(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        heating_stages: u8,
        cooling_stages: u8,
        heating_capacity_per_stage: f64,
        cooling_capacity_per_stage: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            heating_stages,
            cooling_stages,
            heating_capacity_per_stage,
            cooling_capacity_per_stage,
        }
    }

    /// Returns the current HVAC mode based on zone temperature.
    pub fn determine_mode(&self, zone_temp: f64) -> HVACMode {
        let heating_threshold = self.heating_setpoint - self.deadband_tolerance;
        let cooling_threshold = self.cooling_setpoint + self.deadband_tolerance;

        if zone_temp < heating_threshold {
            HVACMode::Heating
        } else if zone_temp > cooling_threshold {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        }
    }

    /// Calculates the required HVAC power (W) to maintain setpoint.
    pub fn calculate_power(&self, _zone_temp: f64, free_float_temp: f64, sensitivity: f64) -> f64 {
        let mode = self.determine_mode(free_float_temp);

        match mode {
            HVACMode::Heating => {
                let target_temp = self.heating_setpoint + self.deadband_tolerance;
                let temp_deficit = target_temp - free_float_temp;
                let power_needed = temp_deficit / sensitivity;
                let max_power = self.heating_capacity_per_stage * self.heating_stages as f64;
                power_needed.clamp(0.0, max_power)
            }
            HVACMode::Cooling => {
                let target_temp = self.cooling_setpoint - self.deadband_tolerance;
                let temp_excess = free_float_temp - target_temp;
                let power_needed = temp_excess / sensitivity;
                let max_power = self.cooling_capacity_per_stage * self.cooling_stages as f64;
                (-power_needed).clamp(-max_power, 0.0)
            }
            HVACMode::Off => 0.0,
        }
    }

    /// Returns the number of active heating stages for the given power output.
    pub fn active_heating_stages(&self, power_watts: f64) -> u8 {
        if power_watts <= 0.0 || self.heating_stages == 0 {
            return 0;
        }
        let stages_needed = (power_watts / self.heating_capacity_per_stage).ceil() as u8;
        stages_needed.min(self.heating_stages)
    }

    /// Returns the number of active cooling stages for the given power output.
    pub fn active_cooling_stages(&self, power_watts: f64) -> u8 {
        if power_watts >= 0.0 || self.cooling_stages == 0 {
            return 0;
        }
        let stages_needed = (power_watts.abs() / self.cooling_capacity_per_stage).ceil() as u8;
        stages_needed.min(self.cooling_stages)
    }

    /// Validates that the setpoints form a valid deadband.
    pub fn validate(&self) -> Result<(), String> {
        let deadband = self.cooling_setpoint - self.heating_setpoint;
        if deadband < 2.0 * self.deadband_tolerance {
            return Err(format!(
                "Invalid deadband: cooling setpoint ({:.1}°C) must be at least {:.1}°C above heating setpoint ({:.1}°C)",
                self.cooling_setpoint,
                2.0 * self.deadband_tolerance,
                self.heating_setpoint
            ));
        }
        Ok(())
    }
}

impl Default for IdealHVACController {
    fn default() -> Self {
        Self::new(20.0, 27.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Floating-point comparisons in this module deal with simple arithmetic
    // (multiply / divide / ceil) on the order of 1e2..1e5, so a 1e-9 epsilon
    // is well below any meaningful precision.
    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // ==================== Construction & Defaults ====================

    #[test]
    fn test_new_preserves_setpoints_and_default_deadband() {
        let ctrl = IdealHVACController::new(20.0, 27.0);
        assert!(approx_eq(ctrl.heating_setpoint, 20.0));
        assert!(approx_eq(ctrl.cooling_setpoint, 27.0));
        // new() must seed the ASHRAE 140-style defaults documented in the struct.
        assert!(approx_eq(ctrl.deadband_tolerance, 0.5));
        assert_eq!(ctrl.heating_stages, 1);
        assert_eq!(ctrl.cooling_stages, 1);
        assert!(approx_eq(ctrl.heating_capacity_per_stage, 100_000.0));
        assert!(approx_eq(ctrl.cooling_capacity_per_stage, 100_000.0));
    }

    #[test]
    fn test_default_matches_new_with_ashrae_setpoints() {
        // Default uses the canonical ASHRAE 140 setpoints (20°C heat / 27°C cool).
        let via_default = IdealHVACController::default();
        let via_new = IdealHVACController::new(20.0, 27.0);
        assert!(approx_eq(
            via_default.heating_setpoint,
            via_new.heating_setpoint
        ));
        assert!(approx_eq(
            via_default.cooling_setpoint,
            via_new.cooling_setpoint
        ));
        assert!(approx_eq(
            via_default.deadband_tolerance,
            via_new.deadband_tolerance
        ));
    }

    #[test]
    fn test_with_stages_capacity_multiplication() {
        // with_stages wires every field through; capacity fields must round-trip
        // exactly so that max power = per_stage * stages holds downstream.
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 3, 2, 10_000.0, 10_000.0);
        assert_eq!(ctrl.heating_stages, 3);
        assert_eq!(ctrl.cooling_stages, 2);
        assert!(approx_eq(ctrl.heating_capacity_per_stage, 10_000.0));
        assert!(approx_eq(ctrl.cooling_capacity_per_stage, 10_000.0));
        // Total installed heating = 3 * 10000 = 30000 W; cooling = 2 * 10000 = 20000 W.
        assert!(approx_eq(
            ctrl.heating_capacity_per_stage * ctrl.heating_stages as f64,
            30_000.0
        ));
        assert!(approx_eq(
            ctrl.cooling_capacity_per_stage * ctrl.cooling_stages as f64,
            20_000.0
        ));
    }

    #[test]
    fn test_clone_preserves_state() {
        // Clone is derived; the controller is shared across rayon population iters
        // in BatchOracle, so a clone must be byte-for-byte equivalent.
        let ctrl = IdealHVACController::with_stages(18.0, 26.0, 2, 3, 5_000.0, 7_500.0);
        let clone = ctrl.clone();
        assert!(approx_eq(clone.heating_setpoint, ctrl.heating_setpoint));
        assert!(approx_eq(clone.cooling_setpoint, ctrl.cooling_setpoint));
        assert!(approx_eq(clone.deadband_tolerance, ctrl.deadband_tolerance));
        assert_eq!(clone.heating_stages, ctrl.heating_stages);
        assert_eq!(clone.cooling_stages, ctrl.cooling_stages);
        assert!(approx_eq(
            clone.heating_capacity_per_stage,
            ctrl.heating_capacity_per_stage
        ));
        assert!(approx_eq(
            clone.cooling_capacity_per_stage,
            ctrl.cooling_capacity_per_stage
        ));
    }

    // ==================== determine_mode boundary cases ====================
    //
    // determine_mode uses strict `<` (heating) and strict `>` (cooling):
    //   heating_threshold = heating_sp - deadband   (19.5°C at defaults)
    //   cooling_threshold = cooling_sp + deadband   (27.5°C at defaults)
    // A reading exactly on either threshold must fall through to Off — this
    // is the deadband-as-hysteresis contract that prevents rapid cycling.

    #[test]
    fn test_determine_mode_heating_below_threshold() {
        let ctrl = IdealHVACController::default(); // h_thresh = 19.5
        assert_eq!(ctrl.determine_mode(19.499), HVACMode::Heating);
        assert_eq!(ctrl.determine_mode(10.0), HVACMode::Heating);
    }

    #[test]
    fn test_determine_mode_at_heating_threshold_is_off() {
        let ctrl = IdealHVACController::default();
        // zone_temp == heating_sp - deadband must NOT trigger heating (strict <).
        assert_eq!(ctrl.determine_mode(19.5), HVACMode::Off);
    }

    #[test]
    fn test_determine_mode_cooling_above_threshold() {
        let ctrl = IdealHVACController::default(); // c_thresh = 27.5
        assert_eq!(ctrl.determine_mode(27.5 + 1e-3), HVACMode::Cooling);
        assert_eq!(ctrl.determine_mode(35.0), HVACMode::Cooling);
    }

    #[test]
    fn test_determine_mode_at_cooling_threshold_is_off() {
        let ctrl = IdealHVACController::default();
        // zone_temp == cooling_sp + deadband must NOT trigger cooling (strict >).
        assert_eq!(ctrl.determine_mode(27.5), HVACMode::Off);
    }

    #[test]
    fn test_determine_mode_deadband_center_is_off() {
        // Center of the deadband [19.5, 27.5] is firmly in the no-action zone.
        let ctrl = IdealHVACController::default();
        assert_eq!(ctrl.determine_mode(23.0), HVACMode::Off);
        assert_eq!(ctrl.determine_mode(19.51), HVACMode::Off);
        assert_eq!(ctrl.determine_mode(27.49), HVACMode::Off);
    }

    #[test]
    fn test_determine_mode_respects_custom_deadband() {
        // A wider deadband widens the Off region — verify the controller
        // honors a non-default tolerance end-to-end.
        let mut ctrl = IdealHVACController::new(20.0, 24.0);
        ctrl.deadband_tolerance = 1.0;
        // heating_threshold = 19.0, cooling_threshold = 25.0
        assert_eq!(ctrl.determine_mode(19.0), HVACMode::Off);
        assert_eq!(ctrl.determine_mode(18.999), HVACMode::Heating);
        assert_eq!(ctrl.determine_mode(25.0), HVACMode::Off);
        assert_eq!(ctrl.determine_mode(25.001), HVACMode::Cooling);
    }

    // ==================== calculate_power (heating) ====================

    #[test]
    fn test_calculate_power_heating_proportional() {
        // free_float below heating_threshold => proportional heat injection.
        // target = heating_sp + deadband = 20.5°C
        // deficit = 20.5 - 15.0 = 5.5 K
        // power  = 5.5 / 0.01 (sensitivity) = 550 W (well under the 100 kW cap).
        let ctrl = IdealHVACController::default();
        let p = ctrl.calculate_power(0.0, 15.0, 0.01);
        assert!(approx_eq(p, 550.0));
        assert!(p > 0.0, "heating power must be positive");
    }

    #[test]
    fn test_calculate_power_heating_clamped_to_max_capacity() {
        // Huge deficit / tiny sensitivity => demand exceeds installed capacity
        // and must clamp to per_stage * stages.
        let ctrl = IdealHVACController::default(); // 1 stage * 100_000 W
        let p = ctrl.calculate_power(0.0, -10.0, 0.0001);
        assert!(approx_eq(p, 100_000.0));
    }

    #[test]
    fn test_calculate_power_in_deadband_is_zero() {
        // Inside the deadband the controller is Off — no energy injected.
        // This is also the in-controller equivalent of a free-float skip:
        // when there is no thermal error signal, output is identically zero.
        let ctrl = IdealHVACController::default();
        assert!(approx_eq(ctrl.calculate_power(0.0, 23.0, 0.01), 0.0));
        assert!(approx_eq(ctrl.calculate_power(0.0, 19.5, 0.01), 0.0));
        assert!(approx_eq(ctrl.calculate_power(0.0, 27.5, 0.01), 0.0));
    }

    // ==================== calculate_power (cooling) ====================
    //
    // Cooling power is returned as a *negative* watt value (heat extraction);
    // calculate_power therefore clamps into [-max, 0.0].

    #[test]
    fn test_calculate_power_cooling_proportional() {
        // target = cooling_sp - deadband = 26.5°C
        // excess = 30.0 - 26.5 = 3.5 K
        // power  = -(3.5 / 0.01) = -350 W.
        let ctrl = IdealHVACController::default();
        let p = ctrl.calculate_power(0.0, 30.0, 0.01);
        assert!(approx_eq(p, -350.0));
        assert!(p < 0.0, "cooling power must be negative (extraction)");
    }

    #[test]
    fn test_calculate_power_cooling_clamped_to_max_capacity() {
        let ctrl = IdealHVACController::default();
        let p = ctrl.calculate_power(0.0, 50.0, 0.0001);
        assert!(approx_eq(p, -100_000.0));
    }

    #[test]
    fn test_calculate_power_respects_multi_stage_capacity() {
        // with_stages: 3 heat stages * 10 kW = 30 kW; 2 cool stages * 10 kW = 20 kW.
        // Demand above 30 kW (heat) / 20 kW (cool) must clamp to the staged total,
        // not the single-stage per_stage value.
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 3, 2, 10_000.0, 10_000.0);

        // Heating: deficit 5.5 / 0.00005 = 110_000 W demand → clamp to 30_000.
        let p_h = ctrl.calculate_power(0.0, 15.0, 0.00005);
        assert!(approx_eq(p_h, 30_000.0));

        // Cooling: excess 3.5 / 0.00002 = 175_000 W demand → clamp to -20_000.
        let p_c = ctrl.calculate_power(0.0, 30.0, 0.00002);
        assert!(approx_eq(p_c, -20_000.0));

        // And below the cap, proportional control must still hold.
        // Heating: 5.5 / 0.001 = 5_500 W (under 30_000 cap).
        assert!(approx_eq(ctrl.calculate_power(0.0, 15.0, 0.001), 5_500.0));
        // Cooling: 3.5 / 0.001 = 3_500 W (under 20_000 cap).
        assert!(approx_eq(ctrl.calculate_power(0.0, 30.0, 0.001), -3_500.0));
    }

    // ==================== active_heating_stages ====================

    #[test]
    fn test_active_heating_stages_zero_for_non_positive_power() {
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 2, 2, 10_000.0, 10_000.0);
        assert_eq!(ctrl.active_heating_stages(0.0), 0);
        assert_eq!(ctrl.active_heating_stages(-5.0), 0);
    }

    #[test]
    fn test_active_heating_stages_ceil_rounding_and_cap() {
        // cap_per_stage = 10_000 W, 2 stages installed.
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 2, 2, 10_000.0, 10_000.0);
        // 1 W → ceil(0.0001) = 1 stage.
        assert_eq!(ctrl.active_heating_stages(1.0), 1);
        // Exactly one stage's worth → still 1 (ceil(1.0) = 1).
        assert_eq!(ctrl.active_heating_stages(10_000.0), 1);
        // One watt over one stage → 2 (ceil(1.0001) = 2).
        assert_eq!(ctrl.active_heating_stages(10_001.0), 2);
        // Demand exceeding all installed stages saturates at the stage count.
        assert_eq!(ctrl.active_heating_stages(25_000.0), 2);
        assert_eq!(ctrl.active_heating_stages(100_000.0), 2);
    }

    #[test]
    fn test_active_heating_stages_zero_stages_installed() {
        // Defensive: if no stages are installed, no stage can activate —
        // independent of the requested power.
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 0, 1, 10_000.0, 10_000.0);
        assert_eq!(ctrl.active_heating_stages(5_000.0), 0);
    }

    // ==================== active_cooling_stages ====================
    //
    // Note the sign convention: cooling power is <= 0; positive power
    // is treated as "not cooling" and returns 0 stages.

    #[test]
    fn test_active_cooling_stages_zero_for_non_negative_power() {
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 2, 2, 10_000.0, 10_000.0);
        assert_eq!(ctrl.active_cooling_stages(0.0), 0);
        assert_eq!(ctrl.active_cooling_stages(5.0), 0);
    }

    #[test]
    fn test_active_cooling_stages_ceil_rounding_and_cap() {
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 2, 2, 10_000.0, 10_000.0);
        assert_eq!(ctrl.active_cooling_stages(-1.0), 1);
        assert_eq!(ctrl.active_cooling_stages(-10_000.0), 1);
        assert_eq!(ctrl.active_cooling_stages(-10_001.0), 2);
        assert_eq!(ctrl.active_cooling_stages(-25_000.0), 2);
        assert_eq!(ctrl.active_cooling_stages(-100_000.0), 2);
    }

    #[test]
    fn test_active_cooling_stages_zero_stages_installed() {
        let ctrl = IdealHVACController::with_stages(20.0, 27.0, 1, 0, 10_000.0, 10_000.0);
        assert_eq!(ctrl.active_cooling_stages(-5_000.0), 0);
    }

    // ==================== validate() ====================

    #[test]
    fn test_validate_accepts_default_config() {
        // Default deadband = 27 - 20 = 7 K >> 2 * 0.5 = 1 K minimum.
        assert!(IdealHVACController::default().validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_deadband_below_two_tolerances() {
        // cooling_sp - heating_sp = 0.5 K < 2 * 0.5 K = 1 K → must reject.
        let ctrl = IdealHVACController::new(20.0, 20.5);
        let err = ctrl.validate().unwrap_err();
        // Error message should mention both setpoints so operators can diagnose.
        let msg = err.to_lowercase();
        assert!(
            msg.contains("deadband"),
            "error must name the deadband: {err}"
        );
        assert!(
            msg.contains("20.0"),
            "error must cite heating setpoint: {err}"
        );
        assert!(
            msg.contains("20.5"),
            "error must cite cooling setpoint: {err}"
        );
    }

    #[test]
    fn test_validate_accepts_exact_minimum_deadband() {
        // deadband == 2 * tolerance is the boundary; the check is `<`, so
        // equality must pass (no off-by-one regression in the threshold).
        let ctrl = IdealHVACController::new(20.0, 21.0); // 1.0 == 2 * 0.5
        assert!(ctrl.validate().is_ok());
    }

    // ==================== HvacSystemMode (free-float skip path) ====================

    #[test]
    fn test_hvac_system_mode_default_is_controlled() {
        // Callers (thermal_model_core, engine) branch on this enum to decide
        // whether to invoke calculate_power at all. Default must be Controlled
        // so that adding a controller to a model actually controls it.
        assert_eq!(HvacSystemMode::default(), HvacSystemMode::Controlled);
    }

    #[test]
    fn test_free_float_mode_skips_controlled_power() {
        // The FreeFloat skip path lives at the caller, not in the controller,
        // but its correctness depends on (a) FreeFloat != Controlled and
        // (b) calculate_power returning exactly 0 W when there is no thermal
        // error. Asserting both properties pins the contract the callers
        // rely on for ASHRAE 140 cases 600FF / 650FF / 900FF / 950FF.
        assert_ne!(HvacSystemMode::FreeFloat, HvacSystemMode::Controlled);

        // Simulate the caller's skip: in FreeFloat the controller is never
        // invoked, so the zone sees zero injected/extracted power regardless
        // of how far it drifts from setpoint.
        let mode = HvacSystemMode::FreeFloat;
        let ctrl = IdealHVACController::default();
        let free_float_temp = 15.0; // would normally demand heating
        let injected = match mode {
            HvacSystemMode::Controlled => ctrl.calculate_power(0.0, free_float_temp, 0.01),
            HvacSystemMode::FreeFloat => 0.0,
        };
        assert!(
            approx_eq(injected, 0.0),
            "FreeFloat must inject zero power, got {injected}"
        );

        // Sanity: the same conditions under Controlled would have heated.
        assert!(
            ctrl.calculate_power(0.0, free_float_temp, 0.01) > 0.0,
            "Controlled mode should be actively heating at 15°C"
        );
    }
}
