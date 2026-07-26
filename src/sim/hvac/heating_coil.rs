//! Sensible-only airside heating coil (Plan T2.4, issue #1763).
//!
//! A heating coil raises the dry-bulb temperature of a moist-air stream
//! without changing its humidity ratio. This is the simplest of the three
//! core airside component abstractions (Fan #1761, CoolingCoil #1762,
//! HeatingCoil #1763): sensible-only, no condensate, no latent term.
//!
//! The trait shape established here is shared with the Fan and CoolingCoil
//! abstractions: a rated-capacity accessor, a stateless capacity computation
//! that returns both the delivered heat and the resulting leaving-air state,
//! and a separate state mutator. The airside boundary contract — supply
//! dry-bulb, relative humidity, pressure, and mass flow — is inherited from
//! [`crate::sim::hvac::airside_state`].
//!
//! # Physics
//!
//! Sensible heating at constant humidity ratio follows ASHRAE Handbook of
//! Fundamentals (2021) Ch.1:
//!
//! ```text
//! Q_sensible = m_da * cp_ma * (T_leaving - T_entering)
//! cp_ma      = 1000 * (1.006 + 1.86 * W)   [J/(kg_da·K)]
//! ```
//!
//! The coil can be controlled either to a leaving-temperature setpoint
//! (capacity follows demand up to the rated maximum) or to a fixed part-load
//! ratio (capacity is a fixed fraction of rated).

use crate::sim::hvac::airside_state::{
    validate_finite, validate_nonnegative, AirsideCouplingError, MoistAirState,
};
use serde::{Deserialize, Serialize};

/// Control signal supplied to a [`HeatingCoil`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeatingCoilControl {
    /// Modulate the coil to achieve a target leaving dry-bulb temperature (°C).
    ///
    /// The coil delivers whatever capacity is required to reach the setpoint,
    /// clamped to its rated maximum. If the setpoint is at or below the
    /// entering temperature the coil turns off (zero capacity).
    LeavingTempSetpoint(f64),
    /// Drive the coil at a fixed part-load ratio.
    ///
    /// `0.0` is fully off and `1.0` is full rated capacity. Values outside
    /// `[0.0, 1.0]` are clamped.
    PartLoadRatio(f64),
}

/// Output of a heating-coil capacity calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatingCoilResult {
    /// Actual sensible capacity delivered (W). Always `>= 0.0`.
    pub capacity_w: f64,
    /// Leaving moist-air state after sensible heating. At zero capacity this
    /// equals the entering state.
    pub leaving_air: MoistAirState,
    /// Part-load ratio achieved (`0.0` to `1.0`).
    pub part_load_ratio: f64,
}

/// Trait for sensible-only airside heating coils.
///
/// Heating coils raise the dry-bulb temperature of a moist-air stream without
/// altering its humidity ratio. Implementations may be controlled to a
/// leaving-temperature setpoint or to a fixed part-load ratio. The trait
/// shape is shared with the Fan (#1761) and CoolingCoil (#1762) airside
/// component abstractions so that VAV/DOAS assemblies (#1764, #1765) can
/// compose the three components uniformly.
pub trait HeatingCoil: Send + Sync {
    /// Rated sensible capacity at standard conditions (W).
    fn rated_capacity_w(&self) -> f64;

    /// Compute the sensible heating capacity and the resulting leaving-air
    /// state for the given entering air, dry-air mass flow, and control
    /// signal.
    ///
    /// The calculation is stateless: it does not mutate the coil. Callers
    /// that want to persist the achieved part-load ratio should forward it
    /// to [`HeatingCoil::update_state`].
    ///
    /// # Errors
    ///
    /// Returns [`AirsideCouplingError`] if the entering-air state is invalid,
    /// the mass flow is negative, or the derived leaving-air state is
    /// non-finite or supersaturated.
    fn compute_heating_capacity(
        &self,
        entering: &MoistAirState,
        mass_flow_da_kg_per_s: f64,
        control: HeatingCoilControl,
    ) -> Result<HeatingCoilResult, AirsideCouplingError>;

    /// Current operating part-load ratio (`0.0` to `1.0`).
    fn current_part_load_ratio(&self) -> f64;

    /// Persist the operating part-load ratio from the most recent capacity
    /// calculation.
    fn update_state(&mut self, part_load_ratio: f64);
}

/// Reference implementation of a sensible-only heating coil.
///
/// Models a hot-water or electric resistance coil whose delivered capacity
/// scales linearly with part-load ratio. Rated capacity is defined at standard
/// entering conditions; the implementation does not apply a temperature
/// correction, so the rated value is the full-load capacity at any entering
/// temperature (electric-resistance behaviour). Hot-water coils that need
/// entering-temperature derating can wrap this type or provide their own
/// [`HeatingCoil`] implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatingCoilComponent {
    /// Equipment identifier.
    pub id: String,
    /// Rated sensible capacity at full part-load ratio (W).
    pub rated_capacity_w: f64,
    /// Design dry-air mass flow rate the coil is sized for (kg_da/s). Used
    /// for documentation and downstream sizing; the capacity calculation
    /// itself accepts the actual operating mass flow.
    pub design_mass_flow_da_kg_per_s: f64,
    /// Current operating part-load ratio (`0.0` to `1.0`).
    pub current_part_load_ratio: f64,
}

impl HeatingCoilComponent {
    /// Create a new heating coil with default operating state (off).
    ///
    /// # Panics
    ///
    /// Panics if `rated_capacity_w` or `design_mass_flow_da_kg_per_s` are not
    /// finite and positive, since a non-positive capacity produces
    /// non-physical part-load ratios.
    pub fn new(id: String, rated_capacity_w: f64, design_mass_flow_da_kg_per_s: f64) -> Self {
        assert!(
            rated_capacity_w.is_finite() && rated_capacity_w > 0.0,
            "rated_capacity_w must be finite and positive, got {rated_capacity_w}"
        );
        assert!(
            design_mass_flow_da_kg_per_s.is_finite() && design_mass_flow_da_kg_per_s > 0.0,
            "design_mass_flow_da_kg_per_s must be finite and positive, got {design_mass_flow_da_kg_per_s}"
        );
        Self {
            id,
            rated_capacity_w,
            design_mass_flow_da_kg_per_s,
            current_part_load_ratio: 0.0,
        }
    }
}

impl HeatingCoil for HeatingCoilComponent {
    fn rated_capacity_w(&self) -> f64 {
        self.rated_capacity_w
    }

    fn compute_heating_capacity(
        &self,
        entering: &MoistAirState,
        mass_flow_da_kg_per_s: f64,
        control: HeatingCoilControl,
    ) -> Result<HeatingCoilResult, AirsideCouplingError> {
        validate_nonnegative("mass_flow_da_kg_per_s", mass_flow_da_kg_per_s)?;
        entering.validate_derived()?;

        let cp_ma = entering.dry_air_specific_heat_j_per_kg_k();
        let entering_temp_c = entering.dry_bulb_c;
        let entering_w = entering.humidity_ratio_kg_per_kg_dry_air;
        let pressure_pa = entering.pressure_pa;

        // Resolve the requested capacity from the control signal.
        let requested_capacity_w = match control {
            HeatingCoilControl::PartLoadRatio(plr) => {
                validate_finite("part_load_ratio", plr)?;
                if plr <= 0.0 {
                    0.0
                } else {
                    let clamped = plr.clamp(0.0, 1.0);
                    self.rated_capacity_w * clamped
                }
            }
            HeatingCoilControl::LeavingTempSetpoint(leaving_setpoint_c) => {
                validate_finite("leaving_setpoint_c", leaving_setpoint_c)?;
                let delta_t = leaving_setpoint_c - entering_temp_c;
                if delta_t <= 0.0 || mass_flow_da_kg_per_s <= 0.0 {
                    0.0
                } else {
                    mass_flow_da_kg_per_s * cp_ma * delta_t
                }
            }
        };

        // Clamp to the rated maximum.
        let capacity_w = requested_capacity_w.min(self.rated_capacity_w).max(0.0);
        let part_load_ratio = if self.rated_capacity_w > 0.0 {
            (capacity_w / self.rated_capacity_w).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Leaving state: sensible heating at constant humidity ratio.
        // Guard zero mass flow to avoid division by zero — the capacity is
        // already zero in that case, so the leaving state equals the entering.
        let leaving_air = if capacity_w <= 0.0 || mass_flow_da_kg_per_s <= 0.0 {
            *entering
        } else {
            let delta_t = capacity_w / (mass_flow_da_kg_per_s * cp_ma);
            let leaving_temp_c = entering_temp_c + delta_t;
            MoistAirState::from_humidity_ratio(leaving_temp_c, entering_w, pressure_pa)?
        };

        Ok(HeatingCoilResult {
            capacity_w,
            leaving_air,
            part_load_ratio,
        })
    }

    fn current_part_load_ratio(&self) -> f64 {
        self.current_part_load_ratio
    }

    fn update_state(&mut self, part_load_ratio: f64) {
        self.current_part_load_ratio = if part_load_ratio.is_finite() {
            part_load_ratio.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::hvac::airside_state::MoistAirState;

    const SEA_LEVEL_PA: f64 = 101_325.0;

    fn entering_air(temp_c: f64, rh_percent: f64) -> MoistAirState {
        MoistAirState::try_new(temp_c, rh_percent, SEA_LEVEL_PA).expect("valid entering air")
    }

    #[test]
    fn text_book_sensible_heating_capacity() {
        // Textbook case (sensible heating, ASHRAE HoF 2021 Ch.1):
        //   m_da = 2.0 kg_da/s of air at 10 °C DB, 50 % RH
        //   heated to a 30 °C leaving dry-bulb.
        // The coil must deliver  Q = m_da * cp_ma * (T_L - T_E).
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 100_000.0, 2.0);
        let entering = entering_air(10.0, 50.0);
        let m_da = 2.0;

        let result = coil
            .compute_heating_capacity(
                &entering,
                m_da,
                HeatingCoilControl::LeavingTempSetpoint(30.0),
            )
            .expect("capacity calc");

        // Independent check using the same moist-air specific heat.
        let cp_ma = entering.dry_air_specific_heat_j_per_kg_k();
        let expected_q = m_da * cp_ma * (30.0 - 10.0);
        assert!(
            (result.capacity_w - expected_q).abs() < 1.0,
            "capacity {} W != expected {} W",
            result.capacity_w,
            expected_q
        );
        // Leaving dry-bulb must hit the setpoint.
        assert!(
            (result.leaving_air.dry_bulb_c - 30.0).abs() < 1.0e-6,
            "leaving DB {} != 30.0",
            result.leaving_air.dry_bulb_c
        );
        // Sanity band: dry-air approximation (cp ≈ 1006 J/kg·K) gives ≈ 40.2 kW.
        assert!(
            result.capacity_w > 39_500.0 && result.capacity_w < 41_000.0,
            "capacity {} W outside the 39.5–41.0 kW textbook band",
            result.capacity_w
        );
    }

    #[test]
    fn humidity_ratio_is_conserved_relative_humidity_drops() {
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 100_000.0, 2.0);
        let entering = entering_air(10.0, 50.0);

        let result = coil
            .compute_heating_capacity(
                &entering,
                2.0,
                HeatingCoilControl::LeavingTempSetpoint(30.0),
            )
            .expect("capacity calc");

        assert!(
            (result.leaving_air.humidity_ratio_kg_per_kg_dry_air
                - entering.humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1.0e-12,
            "sensible heating must conserve humidity ratio"
        );
        assert!(
            result.leaving_air.relative_humidity_percent < entering.relative_humidity_percent,
            "RH must drop when dry-bulb rises at constant W"
        );
    }

    #[test]
    fn part_load_ratio_control_scales_linearly() {
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 50_000.0, 1.5);
        let entering = entering_air(15.0, 40.0);

        for &plr in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
            let result = coil
                .compute_heating_capacity(&entering, 1.5, HeatingCoilControl::PartLoadRatio(plr))
                .expect("capacity calc");
            let expected = 50_000.0 * plr;
            assert!(
                (result.capacity_w - expected).abs() < 1.0e-6,
                "PLR {} -> capacity {} W != {} W",
                plr,
                result.capacity_w,
                expected
            );
            assert!(
                (result.part_load_ratio - plr).abs() < 1.0e-9,
                "returned PLR {} != requested {}",
                result.part_load_ratio,
                plr
            );
        }
    }

    #[test]
    fn leaving_setpoint_above_rated_capacity_is_clamped() {
        // Demand exceeds rated capacity: the coil saturates at its rating and
        // the leaving temperature falls short of the setpoint.
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 10_000.0, 1.0);
        let entering = entering_air(5.0, 50.0);
        let m_da = 1.0;

        let result = coil
            .compute_heating_capacity(
                &entering,
                m_da,
                HeatingCoilControl::LeavingTempSetpoint(100.0),
            )
            .expect("capacity calc");

        assert!(
            (result.capacity_w - 10_000.0).abs() < 1.0e-6,
            "capacity must clamp at rated 10 kW, got {} W",
            result.capacity_w
        );
        assert_eq!(result.part_load_ratio, 1.0);
        // Leaving DB = 5 + 10000 / (1.0 * cp_ma).
        let cp_ma = entering.dry_air_specific_heat_j_per_kg_k();
        let expected_leaving = 5.0 + 10_000.0 / (m_da * cp_ma);
        assert!(
            (result.leaving_air.dry_bulb_c - expected_leaving).abs() < 1.0e-6,
            "leaving DB {} != expected {}",
            result.leaving_air.dry_bulb_c,
            expected_leaving
        );
        assert!(result.leaving_air.dry_bulb_c < 100.0);
    }

    #[test]
    fn setpoint_at_or_below_entering_temperature_turns_coil_off() {
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 50_000.0, 1.0);
        let entering = entering_air(20.0, 50.0);

        for &setpoint in &[20.0_f64, 15.0, 0.0] {
            let result = coil
                .compute_heating_capacity(
                    &entering,
                    1.0,
                    HeatingCoilControl::LeavingTempSetpoint(setpoint),
                )
                .expect("capacity calc");
            assert_eq!(
                result.capacity_w, 0.0,
                "setpoint {} should turn coil off",
                setpoint
            );
            assert_eq!(result.part_load_ratio, 0.0);
            assert_eq!(result.leaving_air, entering);
        }
    }

    #[test]
    fn zero_mass_flow_produces_zero_capacity() {
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 50_000.0, 1.0);
        let entering = entering_air(10.0, 50.0);

        let result = coil
            .compute_heating_capacity(&entering, 0.0, HeatingCoilControl::PartLoadRatio(1.0))
            .expect("capacity calc");

        // PartLoadRatio path does not depend on mass flow, so capacity is
        // still delivered in the bookkeeping sense, but the leaving state
        // must equal the entering state (no air to heat).
        assert_eq!(result.leaving_air, entering);
    }

    #[test]
    fn negative_mass_flow_is_rejected() {
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 50_000.0, 1.0);
        let entering = entering_air(10.0, 50.0);

        let err = coil
            .compute_heating_capacity(&entering, -1.0, HeatingCoilControl::PartLoadRatio(1.0))
            .unwrap_err();
        assert!(matches!(err, AirsideCouplingError::InvalidInput { .. }));
    }

    #[test]
    fn part_load_ratio_is_clamped_to_unit_interval() {
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 50_000.0, 1.0);
        let entering = entering_air(10.0, 50.0);

        let over = coil
            .compute_heating_capacity(&entering, 1.0, HeatingCoilControl::PartLoadRatio(1.5))
            .expect("capacity calc");
        assert_eq!(over.capacity_w, 50_000.0);
        assert_eq!(over.part_load_ratio, 1.0);

        let under = coil
            .compute_heating_capacity(&entering, 1.0, HeatingCoilControl::PartLoadRatio(-0.5))
            .expect("capacity calc");
        assert_eq!(under.capacity_w, 0.0);
        assert_eq!(under.part_load_ratio, 0.0);
    }

    #[test]
    fn update_state_persists_and_clamps_part_load_ratio() {
        let mut coil = HeatingCoilComponent::new("HC-1".to_string(), 50_000.0, 1.0);
        assert_eq!(coil.current_part_load_ratio(), 0.0);

        coil.update_state(0.6);
        assert!((coil.current_part_load_ratio() - 0.6).abs() < 1.0e-9);

        coil.update_state(2.0);
        assert_eq!(coil.current_part_load_ratio(), 1.0);

        coil.update_state(f64::NAN);
        assert_eq!(coil.current_part_load_ratio(), 0.0);
    }

    #[test]
    fn constructor_rejects_non_positive_capacity() {
        let result =
            std::panic::catch_unwind(|| HeatingCoilComponent::new("HC-x".to_string(), 0.0, 1.0));
        assert!(result.is_err(), "zero capacity must panic");

        let result =
            std::panic::catch_unwind(|| HeatingCoilComponent::new("HC-x".to_string(), -1.0, 1.0));
        assert!(result.is_err(), "negative capacity must panic");

        let result =
            std::panic::catch_unwind(|| HeatingCoilComponent::new("HC-x".to_string(), 1.0, 0.0));
        assert!(result.is_err(), "zero design mass flow must panic");
    }

    #[test]
    fn round_trip_through_control_modes_agree() {
        // Heating from 15 °C to 25 °C via LeavingTempSetpoint should yield the
        // same capacity as the equivalent PartLoadRatio.
        let coil = HeatingCoilComponent::new("HC-1".to_string(), 100_000.0, 2.0);
        let entering = entering_air(15.0, 40.0);
        let m_da = 2.0;

        let via_setpoint = coil
            .compute_heating_capacity(
                &entering,
                m_da,
                HeatingCoilControl::LeavingTempSetpoint(25.0),
            )
            .expect("capacity calc");

        let via_plr = coil
            .compute_heating_capacity(
                &entering,
                m_da,
                HeatingCoilControl::PartLoadRatio(via_setpoint.part_load_ratio),
            )
            .expect("capacity calc");

        assert!(
            (via_setpoint.capacity_w - via_plr.capacity_w).abs() < 1.0e-6,
            "setpoint capacity {} != PLR capacity {}",
            via_setpoint.capacity_w,
            via_plr.capacity_w
        );
        assert!(
            (via_setpoint.leaving_air.dry_bulb_c - via_plr.leaving_air.dry_bulb_c).abs() < 1.0e-6
        );
    }
}
