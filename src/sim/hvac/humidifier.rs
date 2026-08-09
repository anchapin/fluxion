//! Adiabatic airside humidifier component (Issue #2464, Plan T2.6 supplement).
//!
//! Mirrors the [`crate::sim::hvac::heating_coil::HeatingCoil`] trait shape so the
//! airside coupling layer (`AirsideEnvelopeCoupler`, #1767) and downstream
//! assemblies (DOAS #1765, VAV/Cav terminal units) can compose a humidifier
//! uniformly with fans, cooling coils, and heating coils.
//!
//! ## Physical model
//!
//! An adiabatic humidifier adds moisture to the airstream **without external
//! heat input** (precedent: `EnergyPlus Humidifier:Steam:Adiabatic`). The
//! process is isothermal at the rated condition in the simplified EnergyPlus
//! model: the supply humidity ratio rises at constant dry-bulb. The latent
//! energy associated with the added water appears as a **positive supply
//! latent load** on the zone air balance:
//!
//! ```text
//! Q_lat  = ṁ_da · h_fg · (W_out − W_in)         [W]
//! ṁ_h2o = ṁ_da · (W_out − W_in)                [kg_water/s]
//! ```
//!
//! with `h_fg = 2 501 kJ/kg` (latent heat of vaporization at 0 °C — ASHRAE
//! HoF 2021 Ch.1 reference value).
//!
//! ## Capacity clamping
//!
//! The requested moisture addition is clamped against the rated moisture
//! rate. If the requested rate exceeds the rating, the leaving state is
//! written at the rated rate (capacity-limited) and
//! [`HumidifierResult::part_load_ratio`] reflects the clamp. The caller
//! should track the part-load ratio for downstream sizing.
//!
//! ## Leaving state
//!
//! The leaving humidity ratio is the **minimum** of the requested target and
//! the saturation humidity ratio at the entering dry-bulb (so the resulting
//! state is always physical). When `target_humidity_ratio > w_sat(T_db)` the
//! air would be supersaturated, which is rejected by
//! [`MoistAirState::from_humidity_ratio`]; in that case the humidifier
//! saturates the air and returns the saturation ratio. For typical DOAS
//! setpoints (`target_dew_point ≤ supply_dry_bulb`) this is never reached.

use crate::sim::hvac::airside_state::{
    validate_nonnegative, validate_positive, AirsideCouplingError, MoistAirState,
};
use fluxion_core::weather::psychrometrics::calculate_humidity_ratio;
use serde::{Deserialize, Serialize};

/// Latent heat of vaporization of water at 0 °C [J/kg] — ASHRAE HoF 2021
/// Ch.1 Eq. 32 reference. The 20 °C value (≈ 2 450 kJ/kg) used by the issue
/// spec is the temperature-corrected form; we use the 0 °C reference here so
/// the airside humidity-ratio ↔ latent-enthalpy identity round-trips with
/// the psychrometric library's `calculate_enthalpy` (which uses 2 501 kJ/kg).
const H_FG_0C_J_PER_KG: f64 = 2_501_000.0;

/// Control signal supplied to a [`Humidifier`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HumidifierControl {
    /// Drive the humidifier to a fixed moisture addition rate [kg_water/s].
    MoistureRate(f64),
    /// Drive the humidifier to a target leaving humidity ratio [kg/kg_da].
    /// The actual rate is `(W_target − W_entering) · �_da`, clamped to the
    /// rated moisture rate.
    TargetHumidityRatio(f64),
}

/// Output of a humidifier capacity calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumidifierResult {
    /// Latent capacity delivered to the airstream [W] (positive quantity —
    /// humidification adds energy to the air).
    pub capacity_w: f64,
    /// Leaving moist-air state with elevated humidity ratio at constant
    /// dry-bulb (ideal adiabatic humidifier).
    pub leaving_air: MoistAirState,
    /// Moisture addition rate achieved [kg_water/s].
    pub moisture_rate_kg_per_s: f64,
    /// Part-load ratio achieved (`0.0` to `1.0`).
    pub part_load_ratio: f64,
}

/// Trait for adiabatic airside humidifiers.
pub trait Humidifier: Send + Sync {
    /// Rated moisture addition rate [kg_water/s] at full part-load ratio.
    fn rated_moisture_kg_per_s(&self) -> f64;

    /// Compute the moisture addition and the resulting leaving-air state for
    /// the given entering air, dry-air mass flow, and control signal.
    ///
    /// Stateless — does not mutate the humidifier. Callers that want to
    /// persist the achieved part-load ratio should forward it to
    /// [`Humidifier::update_state`].
    fn compute_humidification_capacity(
        &self,
        entering: &MoistAirState,
        mass_flow_da_kg_per_s: f64,
        control: HumidifierControl,
    ) -> Result<HumidifierResult, AirsideCouplingError>;

    /// Current operating part-load ratio (`0.0` to `1.0`).
    fn current_part_load_ratio(&self) -> f64;

    /// Persist the operating part-load ratio from the most recent capacity
    /// calculation.
    fn update_state(&mut self, part_load_ratio: f64);
}

/// Reference implementation of an adiabatic humidifier.
///
/// Models an isothermal steam humidifier (no temperature change to the air at
/// the rated condition — the standard EnergyPlus `Humidifier:Steam:Adiabatic`
/// simplification). The rated moisture rate is the full-load water addition
/// rate at any entering condition; the implementation does not derate for
/// entering temperature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumidifierComponent {
    /// Equipment identifier.
    pub id: String,
    /// Rated moisture addition rate at full part-load ratio [kg_water/s].
    pub rated_moisture_kg_per_s: f64,
    /// Design dry-air mass flow rate the humidifier is sized for [kg_da/s].
    /// Used for documentation and downstream sizing; the capacity calculation
    /// accepts the actual operating mass flow.
    pub design_mass_flow_da_kg_per_s: f64,
    /// Current operating part-load ratio (`0.0` to `1.0`).
    pub current_part_load_ratio: f64,
}

impl HumidifierComponent {
    /// Create a new adiabatic humidifier with default operating state (off).
    ///
    /// # Panics
    ///
    /// Panics if `rated_moisture_kg_per_s` or `design_mass_flow_da_kg_per_s`
    /// are not finite and positive, since a non-positive rating produces
    /// non-physical part-load ratios.
    pub fn new(
        id: String,
        rated_moisture_kg_per_s: f64,
        design_mass_flow_da_kg_per_s: f64,
    ) -> Self {
        assert!(
            rated_moisture_kg_per_s.is_finite() && rated_moisture_kg_per_s > 0.0,
            "rated_moisture_kg_per_s must be finite and positive, got {rated_moisture_kg_per_s}"
        );
        assert!(
            design_mass_flow_da_kg_per_s.is_finite() && design_mass_flow_da_kg_per_s > 0.0,
            "design_mass_flow_da_kg_per_s must be finite and positive, got {design_mass_flow_da_kg_per_s}"
        );
        Self {
            id,
            rated_moisture_kg_per_s,
            design_mass_flow_da_kg_per_s,
            current_part_load_ratio: 0.0,
        }
    }
}

impl Humidifier for HumidifierComponent {
    fn rated_moisture_kg_per_s(&self) -> f64 {
        self.rated_moisture_kg_per_s
    }

    fn compute_humidification_capacity(
        &self,
        entering: &MoistAirState,
        mass_flow_da_kg_per_s: f64,
        control: HumidifierControl,
    ) -> Result<HumidifierResult, AirsideCouplingError> {
        validate_nonnegative("mass_flow_da_kg_per_s", mass_flow_da_kg_per_s)?;
        entering.validate_derived()?;

        let w_in = entering.humidity_ratio_kg_per_kg_dry_air;
        let pressure_pa = entering.pressure_pa;

        // Resolve the requested moisture rate from the control signal.
        let requested_rate_kg_per_s = match control {
            HumidifierControl::MoistureRate(rate) => {
                validate_nonnegative("moisture_rate_kg_per_s", rate)?;
                rate
            }
            HumidifierControl::TargetHumidityRatio(w_target) => {
                validate_nonnegative("target_humidity_ratio_kg_per_kg", w_target)?;
                if mass_flow_da_kg_per_s <= 0.0 {
                    0.0
                } else {
                    let delta_w = w_target - w_in;
                    if delta_w <= 0.0 {
                        0.0
                    } else {
                        mass_flow_da_kg_per_s * delta_w
                    }
                }
            }
        };

        // Clamp to the rated maximum.
        let delivered_rate_kg_per_s = requested_rate_kg_per_s
            .min(self.rated_moisture_kg_per_s)
            .max(0.0);
        let part_load_ratio = if self.rated_moisture_kg_per_s > 0.0 {
            (delivered_rate_kg_per_s / self.rated_moisture_kg_per_s).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Leaving state: ideal adiabatic → constant dry-bulb, raised humidity ratio.
        // Guard zero mass flow: the rate is already zero in that case, so the
        // leaving state equals the entering state.
        let (leaving_air, capacity_w) = if delivered_rate_kg_per_s <= 0.0
            || mass_flow_da_kg_per_s <= 0.0
        {
            (*entering, 0.0)
        } else {
            let w_out = w_in + delivered_rate_kg_per_s / mass_flow_da_kg_per_s;
            // Saturate against the dry-bulb. If the requested ratio would
            // supersaturate (only reachable when the caller passes a
            // physically impossible target), cap at saturation.
            let w_sat = calculate_humidity_ratio(entering.dry_bulb_c, 100.0, pressure_pa);
            let w_out_capped = w_out.min(w_sat);
            let leaving =
                MoistAirState::from_humidity_ratio(entering.dry_bulb_c, w_out_capped, pressure_pa)?;
            // Capacity is the latent heat of vaporization of the added water.
            let capacity_w = delivered_rate_kg_per_s * H_FG_0C_J_PER_KG;
            (leaving, capacity_w)
        };

        Ok(HumidifierResult {
            capacity_w,
            leaving_air,
            moisture_rate_kg_per_s: delivered_rate_kg_per_s,
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

#[allow(dead_code)]
fn _assert_validate_positive_used(v: f64) -> Result<(), AirsideCouplingError> {
    validate_positive("unused", v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxion_core::weather::psychrometrics::saturation_vapor_pressure;

    const SEA_LEVEL_PA: f64 = 101_325.0;

    fn entering_air(temp_c: f64, rh_percent: f64) -> MoistAirState {
        MoistAirState::try_new(temp_c, rh_percent, SEA_LEVEL_PA).expect("valid entering air")
    }

    #[test]
    fn text_book_adiabatic_humidifier_capacity() {
        // Post-reheat DOAS supply air at 18 °C / 30 % RH (cold-dry winter
        // case) enters the humidifier with a target leaving humidity ratio
        // equal to `w_sat(10 °C)` ≈ 7.6e-3 kg/kg. With m_da ≈ 1.8 kg/s the
        // moisture rate is `(W_target − W_in) · m_da`.
        let humidifier = HumidifierComponent::new("HUM-1".to_string(), 0.050, 2.0);
        let entering = entering_air(18.0, 30.0);
        let m_da = 1.8;

        let w_target = calculate_humidity_ratio(10.0, 100.0, SEA_LEVEL_PA);
        let result = humidifier
            .compute_humidification_capacity(
                &entering,
                m_da,
                HumidifierControl::TargetHumidityRatio(w_target),
            )
            .expect("humidifier capacity");

        let w_in = entering.humidity_ratio_kg_per_kg_dry_air;
        let expected_rate = m_da * (w_target - w_in);
        assert!(
            (result.moisture_rate_kg_per_s - expected_rate).abs() < 1e-9,
            "rate {} vs expected {}",
            result.moisture_rate_kg_per_s,
            expected_rate
        );
        assert!(result.moisture_rate_kg_per_s > 0.0);
        // Capacity: Q_lat = ṁ_h2o · h_fg (using the 0 °C ASHRAE reference value).
        let expected_capacity = expected_rate * H_FG_0C_J_PER_KG;
        assert!(
            (result.capacity_w - expected_capacity).abs() < 1.0,
            "capacity {} W vs expected {} W",
            result.capacity_w,
            expected_capacity
        );
        // Leaving state is at constant dry-bulb (ideal adiabatic).
        assert!(
            (result.leaving_air.dry_bulb_c - entering.dry_bulb_c).abs() < 1e-9,
            "ideal adiabatic leaves dry-bulb unchanged"
        );
        // Leaving humidity ratio approaches the target.
        assert!(
            (result.leaving_air.humidity_ratio_kg_per_kg_dry_air - w_target).abs() < 1e-9,
            "leaving humidity ratio {} vs target {}",
            result.leaving_air.humidity_ratio_kg_per_kg_dry_air,
            w_target
        );
    }

    #[test]
    fn capacity_clamped_to_rated_when_target_exceeds_rating() {
        // Demand a very high moisture rate that exceeds the rated 0.005 kg/s.
        let humidifier = HumidifierComponent::new("HUM-2".to_string(), 0.005, 2.0);
        let entering = entering_air(20.0, 10.0);
        let m_da = 1.8;

        let result = humidifier
            .compute_humidification_capacity(
                &entering,
                m_da,
                HumidifierControl::MoistureRate(0.050),
            )
            .expect("humidifier capacity");

        assert!(
            (result.moisture_rate_kg_per_s - 0.005).abs() < 1e-12,
            "rate {} must clamp to rated 0.005",
            result.moisture_rate_kg_per_s
        );
        assert!((result.part_load_ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_rate_returns_entering_state() {
        let humidifier = HumidifierComponent::new("HUM-3".to_string(), 0.050, 2.0);
        let entering = entering_air(20.0, 50.0);
        let result = humidifier
            .compute_humidification_capacity(&entering, 1.8, HumidifierControl::MoistureRate(0.0))
            .expect("humidifier capacity");
        assert_eq!(result.moisture_rate_kg_per_s, 0.0);
        assert_eq!(result.capacity_w, 0.0);
        assert_eq!(result.leaving_air, entering);
    }

    #[test]
    fn leaving_air_caps_at_saturation_when_target_supersaturates() {
        // Physically impossible target humidity ratio above saturation at the
        // entering dry-bulb must be clamped, not panicked.
        let humidifier = HumidifierComponent::new("HUM-4".to_string(), 1.0, 2.0);
        let entering = entering_air(10.0, 50.0);
        let w_sat = saturation_vapor_pressure(10.0);
        // A target well above saturation — should cap at w_sat.
        let result = humidifier
            .compute_humidification_capacity(
                &entering,
                1.0,
                HumidifierControl::TargetHumidityRatio(w_sat + 1.0),
            )
            .expect("humidifier capacity");
        // Leaving humidity ratio should be at (or below) saturation at the
        // entering dry-bulb. We check the reconstructed partial vapor pressure.
        assert!(result.leaving_air.partial_vapor_pressure_pa <= w_sat + 1e-3);
        assert!(result.leaving_air.relative_humidity_percent <= 100.0 + 1e-6);
    }
}
