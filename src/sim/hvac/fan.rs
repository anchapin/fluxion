//! Airside fan component (issue #1761, plan T2.2).
//!
//! Defines the [`Fan`] trait and [`FanComponent`] struct that establish the
//! airside-component abstraction shape the cooling/heating coils and VAV/DOAS
//! units will follow. This is a **pure component**: it owns the fan power and
//! flow model only and does **not** couple to the envelope solver.
//!
//! ## Physics
//!
//! All quantities are SI. The component stores rated (design) values and
//! derives off-design performance from the fan affinity laws
//! (ASHRAE Handbook—Fundamentals, Ch. 21; Kreider et al.,
//! *Heating and Cooling of Buildings*), assuming the operating point tracks
//! the rated system curve at a constant efficiency:
//!
//! | Quantity | Scaling with speed fraction `φ` ∈ [0, 1] |
//! |----------|-------------------------------------------|
//! | Volumetric flow `Q̇` | `φ · Q̇_rated` (fan law 1) |
//! | Pressure rise `ΔP` | `φ² · ΔP_rated` (fan law 2) |
//! | Air power `Q̇·ΔP` | `φ³ · Q̇_rated·ΔP_rated` (fan law 3) |
//!
//! At a given speed the volumetric flow is set by the impeller and is
//! independent of density, while pressure scales linearly with the actual
//! air density relative to the design density:
//!
//! - `ΔP(φ, ρ) = φ² · ΔP_rated · (ρ / ρ_design)`
//! - `ṁ(φ, ρ) = ρ · Q̇(φ)` (dry-air mass flow)
//! - `P_air(φ, ρ) = Q̇(φ) · ΔP(φ, ρ)` (power imparted to the air)
//! - `P_shaft(φ, ρ) = P_air / η_total` (brake / shaft power)
//! - `P_motor(φ, ρ) = P_shaft / η_motor` (electrical input power)

use serde::{Deserialize, Serialize};

/// Standard air density used as the default design condition [kg/m³].
///
/// ASHRAE standard air: dry-bulb 20 °C, 101.325 kPa. Matches the
/// `1.2 kg/m³` convention used throughout the airside coupling layer.
pub const STANDARD_AIR_DENSITY_KG_PER_M3: f64 = 1.2;

/// Trait for airside fan components.
///
/// Establishes the component-interface shape that the cooling/heating coils
/// and VAV/DOAS terminal units will follow (issue #1761). Implementations
/// translate a speed command into volumetric flow, pressure rise, mass flow,
/// and power demand without coupling to the zone thermal solver.
///
/// All methods take a `speed_fraction` ∈ [0, 1] (1.0 = full design speed) and
/// the actual air density `air_density_kg_per_m3` so that density-corrected
/// pressure and mass flow can be resolved by the caller.
pub trait Fan: Send + Sync {
    /// Volumetric flow rate at the given speed fraction [m³/s].
    fn volumetric_flow(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64;

    /// Dry-air mass flow rate at the given speed fraction [kg/s].
    fn mass_flow_rate(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64;

    /// Total pressure rise across the fan at the given speed fraction [Pa].
    fn pressure_rise(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64;

    /// Air power (power imparted to the air) at the given speed fraction [W].
    fn air_power(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64;

    /// Shaft (brake) power drawn by the fan at the given speed fraction [W].
    fn shaft_power(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64;

    /// Rated volumetric flow at full design speed [m³/s].
    fn rated_volumetric_flow(&self) -> f64;

    /// Rated total pressure rise at full design speed [Pa].
    fn rated_pressure_rise(&self) -> f64;

    /// Total (fan) efficiency at rated conditions, dimensionless ∈ (0, 1].
    fn rated_efficiency(&self) -> f64;
}

/// Centrifugal/axial fan model parameterised by rated performance.
///
/// Stores the four design-point quantities required by the issue acceptance
/// criteria — rated flow, pressure rise, total efficiency, and a power model
/// (fan-law cubed scaling plus an optional motor-efficiency stage) — and
/// derives off-design operation from the fan affinity laws.
///
/// `total_efficiency` is the overall fan (impeller) efficiency used to convert
/// air power into shaft power; `motor_efficiency` is the drive-motor
/// efficiency used to convert shaft power into electrical input power. When
/// motor losses are not modelled, set `motor_efficiency = 1.0` so that
/// [`FanComponent::motor_power`] equals [`FanComponent::shaft_power`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanComponent {
    /// Equipment identifier.
    pub id: String,
    /// Rated volumetric flow rate at full design speed [m³/s].
    pub rated_volumetric_flow_m3_per_s: f64,
    /// Rated total pressure rise at full design speed [Pa].
    pub rated_pressure_rise_pa: f64,
    /// Total fan (impeller) efficiency, dimensionless ∈ (0, 1].
    pub total_efficiency: f64,
    /// Motor/drive efficiency, dimensionless ∈ (0, 1]. Use `1.0` to ignore.
    pub motor_efficiency: f64,
    /// Air density the rated point was specified at [kg/m³].
    pub design_air_density_kg_per_m3: f64,
}

impl FanComponent {
    /// Create a fan with a given rated point and a unity motor efficiency.
    ///
    /// `design_air_density_kg_per_m3` defaults to standard air
    /// ([`STANDARD_AIR_DENSITY_KG_PER_M3`]) when not provided.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        rated_volumetric_flow_m3_per_s: f64,
        rated_pressure_rise_pa: f64,
        total_efficiency: f64,
    ) -> Self {
        Self::with_motor(
            id,
            rated_volumetric_flow_m3_per_s,
            rated_pressure_rise_pa,
            total_efficiency,
            1.0,
            STANDARD_AIR_DENSITY_KG_PER_M3,
        )
    }

    /// Create a fan with an explicit motor efficiency and design density.
    #[allow(clippy::too_many_arguments)]
    pub fn with_motor(
        id: String,
        rated_volumetric_flow_m3_per_s: f64,
        rated_pressure_rise_pa: f64,
        total_efficiency: f64,
        motor_efficiency: f64,
        design_air_density_kg_per_m3: f64,
    ) -> Self {
        Self {
            id,
            rated_volumetric_flow_m3_per_s,
            rated_pressure_rise_pa,
            total_efficiency,
            motor_efficiency,
            design_air_density_kg_per_m3,
        }
    }

    /// Motor electrical input power at the given speed fraction [W].
    ///
    /// This extends the power model beyond [`Fan::shaft_power`] by applying the
    /// motor/drive efficiency. When `motor_efficiency == 1.0` it collapses to
    /// the shaft power.
    pub fn motor_power(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64 {
        let shaft = self.shaft_power(speed_fraction, air_density_kg_per_m3);
        let eta_motor = if self.motor_efficiency > 0.0 {
            self.motor_efficiency
        } else {
            1.0
        };
        shaft / eta_motor
    }

    /// Density correction factor applied to pressure (actual / design).
    #[inline]
    fn density_factor(&self, air_density_kg_per_m3: f64) -> f64 {
        if self.design_air_density_kg_per_m3 > 0.0 {
            air_density_kg_per_m3 / self.design_air_density_kg_per_m3
        } else {
            1.0
        }
    }
}

impl Fan for FanComponent {
    fn volumetric_flow(&self, speed_fraction: f64, _air_density_kg_per_m3: f64) -> f64 {
        let phi = speed_fraction.clamp(0.0, 1.0);
        phi * self.rated_volumetric_flow_m3_per_s
    }

    fn mass_flow_rate(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64 {
        air_density_kg_per_m3 * self.volumetric_flow(speed_fraction, air_density_kg_per_m3)
    }

    fn pressure_rise(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64 {
        let phi = speed_fraction.clamp(0.0, 1.0);
        phi * phi * self.rated_pressure_rise_pa * self.density_factor(air_density_kg_per_m3)
    }

    fn air_power(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64 {
        self.volumetric_flow(speed_fraction, air_density_kg_per_m3)
            * self.pressure_rise(speed_fraction, air_density_kg_per_m3)
    }

    fn shaft_power(&self, speed_fraction: f64, air_density_kg_per_m3: f64) -> f64 {
        let air = self.air_power(speed_fraction, air_density_kg_per_m3);
        let eta = if self.total_efficiency > 0.0 {
            self.total_efficiency
        } else {
            1.0
        };
        air / eta
    }

    fn rated_volumetric_flow(&self) -> f64 {
        self.rated_volumetric_flow_m3_per_s
    }

    fn rated_pressure_rise(&self) -> f64 {
        self.rated_pressure_rise_pa
    }

    fn rated_efficiency(&self) -> f64 {
        self.total_efficiency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook fan validation against the fan affinity laws.
    ///
    /// Reference case: a fan rated at 2.0 m³/s against a 500 Pa pressure rise
    /// with a total efficiency of 0.70 moving standard air (ρ = 1.2 kg/m³).
    /// Hand calculation:
    /// - air power (full speed) = Q̇ · ΔP = 2.0 · 500 = 1000 W
    /// - shaft power (full speed) = 1000 / 0.70 = 1428.571… W
    /// - mass flow (full speed) = ρ · Q̇ = 1.2 · 2.0 = 2.4 kg/s
    #[test]
    fn fan_law_full_speed_textbook_case() {
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        let rho = STANDARD_AIR_DENSITY_KG_PER_M3;

        assert_eq!(fan.rated_volumetric_flow(), 2.0);
        assert_eq!(fan.rated_pressure_rise(), 500.0);
        assert!((fan.rated_efficiency() - 0.70).abs() < 1.0e-12);

        // Flow & pressure at full speed equal the rated values.
        assert!((fan.volumetric_flow(1.0, rho) - 2.0).abs() < 1.0e-12);
        assert!((fan.pressure_rise(1.0, rho) - 500.0).abs() < 1.0e-12);

        // Air power: Q̇·ΔP = 1000 W.
        assert!((fan.air_power(1.0, rho) - 1000.0).abs() < 1.0e-9);
        // Shaft power: 1000 / 0.70.
        assert!((fan.shaft_power(1.0, rho) - 1428.571_428_571_4).abs() < 1.0e-6);
        // Mass flow: ρ·Q̇ = 2.4 kg/s.
        assert!((fan.mass_flow_rate(1.0, rho) - 2.4).abs() < 1.0e-12);
    }

    /// Fan law 1 (Q̇ ∝ φ), law 2 (ΔP ∝ φ²) and law 3 (P ∝ φ³) at 50 % speed.
    #[test]
    fn fan_law_half_speed_cubed_power() {
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        let rho = STANDARD_AIR_DENSITY_KG_PER_M3;

        // Law 1: flow halves.
        assert!((fan.volumetric_flow(0.5, rho) - 1.0).abs() < 1.0e-12);
        // Law 2: pressure falls to a quarter (0.5² · 500 = 125 Pa).
        assert!((fan.pressure_rise(0.5, rho) - 125.0).abs() < 1.0e-9);
        // Law 3: air power falls to an eighth (0.5³ · 1000 = 125 W).
        assert!((fan.air_power(0.5, rho) - 125.0).abs() < 1.0e-9);
        // Shaft power tracks the cubed ratio (125 / 0.70 = 178.571… W).
        assert!((fan.shaft_power(0.5, rho) - 178.571_428_571_4).abs() < 1.0e-6);

        // The shaft-power ratio between half and full speed must equal φ³.
        let full = fan.shaft_power(1.0, rho);
        let half = fan.shaft_power(0.5, rho);
        assert!((half / full - 0.125).abs() < 1.0e-9);
    }

    /// Zero speed → zero flow, pressure and power.
    #[test]
    fn zero_speed_produces_zero_output() {
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        let rho = STANDARD_AIR_DENSITY_KG_PER_M3;
        assert_eq!(fan.volumetric_flow(0.0, rho), 0.0);
        assert_eq!(fan.pressure_rise(0.0, rho), 0.0);
        assert_eq!(fan.air_power(0.0, rho), 0.0);
        assert_eq!(fan.shaft_power(0.0, rho), 0.0);
        assert_eq!(fan.mass_flow_rate(0.0, rho), 0.0);
    }

    /// Out-of-range speed fractions clamp to the valid [0, 1] domain.
    #[test]
    fn speed_fraction_clamps_to_valid_range() {
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        let rho = STANDARD_AIR_DENSITY_KG_PER_M3;
        // Negative speed clamps to zero.
        assert_eq!(fan.volumetric_flow(-0.5, rho), 0.0);
        // Speed above 1.0 clamps to full speed.
        assert!((fan.volumetric_flow(1.5, rho) - 2.0).abs() < 1.0e-12);
        assert!((fan.pressure_rise(1.5, rho) - 500.0).abs() < 1.0e-9);
    }

    /// Mass flow scales linearly with air density at fixed speed.
    #[test]
    fn mass_flow_scales_with_density() {
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        // At sea-level standard air (1.2) vs. a higher density (1.4).
        let m_std = fan.mass_flow_rate(1.0, 1.2);
        let m_dense = fan.mass_flow_rate(1.0, 1.4);
        assert!((m_std - 2.4).abs() < 1.0e-12);
        assert!((m_dense - 2.8).abs() < 1.0e-12);
        // Volumetric flow is density-independent (set by the impeller).
        assert!((fan.volumetric_flow(1.0, 1.4) - 2.0).abs() < 1.0e-12);
    }

    /// Pressure rise scales with the density ratio relative to the design point.
    #[test]
    fn pressure_rise_scales_with_density_ratio() {
        // Rated at 1.2 kg/m³; operate at 1.0 kg/m³ → pressure falls by 1/1.2.
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        let p_at_1p0 = fan.pressure_rise(1.0, 1.0);
        assert!((p_at_1p0 - 500.0 * (1.0 / 1.2)).abs() < 1.0e-9);
    }

    /// Motor-power stage applies the drive efficiency on top of shaft power.
    #[test]
    fn motor_power_applies_motor_efficiency() {
        let fan = FanComponent::with_motor("FAN-1".into(), 2.0, 500.0, 0.70, 0.90, 1.2);
        let rho = 1.2;
        let shaft = fan.shaft_power(1.0, rho);
        let motor = fan.motor_power(1.0, rho);
        assert!((motor - shaft / 0.90).abs() < 1.0e-6);
        assert!((motor - 1428.571_428_571_4 / 0.90).abs() < 1.0e-5);
    }

    /// The default constructor collapses motor power to shaft power.
    #[test]
    fn default_motor_efficiency_is_unity() {
        let fan = FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70);
        let rho = 1.2;
        assert_eq!(fan.motor_power(1.0, rho), fan.shaft_power(1.0, rho));
    }

    /// The struct is `Clone` and round-trips through Serde.
    #[test]
    fn clone_and_serde_round_trip() {
        let fan = FanComponent::with_motor("FAN-1".into(), 2.0, 500.0, 0.70, 0.92, 1.18);
        let cloned = fan.clone();
        assert_eq!(fan.id, cloned.id);
        assert!((fan.rated_pressure_rise() - cloned.rated_pressure_rise()).abs() < 1.0e-12);

        let json = serde_json::to_string(&fan).expect("serialize");
        let back: FanComponent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(fan.id, back.id);
        assert!((fan.total_efficiency - back.total_efficiency).abs() < 1.0e-12);
        assert!((fan.motor_efficiency - back.motor_efficiency).abs() < 1.0e-12);
    }

    /// The trait can be used through a `dyn Fan` reference.
    #[test]
    fn trait_object_dispatch() {
        let fan: Box<dyn Fan> = Box::new(FanComponent::new("FAN-1".into(), 2.0, 500.0, 0.70));
        let rho = 1.2;
        assert!((fan.volumetric_flow(1.0, rho) - 2.0).abs() < 1.0e-12);
        assert!((fan.shaft_power(1.0, rho) - 1428.571_428_571_4).abs() < 1.0e-6);
        assert!((fan.mass_flow_rate(1.0, rho) - 2.4).abs() < 1.0e-12);
    }
}
