//! Variable Air Volume (VAV) terminal unit model (Issue #1764, Plan T2.5).
//!
//! Composes the three core airside component abstractions — [`FanComponent`]
//! (#1761), [`CoolingCoil`] (#1762), and [`HeatingCoilComponent`] (#1763) —
//! into a single-zone VAV terminal unit with damper-modulated mass flow.
//!
//! ## Physical model
//!
//! A VAV terminal is the most common commercial airside configuration. A
//! modulating **damper** varies the primary airflow between a minimum
//! (ventilation / turndown) and a maximum (design cooling) position. The
//! terminal processes the air through three stages:
//!
//! 1. **Fan** — establishes the volumetric flow at a speed fraction derived
//!    from the damper position via the fan affinity laws (T2.2).
//! 2. **Cooling coil** — removes sensible and latent heat using the ASHRAE
//!    bypass-factor psychrometric model (T2.3). Active only in cooling mode.
//! 3. **Reheat coil** — optionally raises the supply dry-bulb at constant
//!    humidity ratio for heating/dehumidification-reheat (T2.4).
//!
//! The fan shaft power is dissipated into the airstream as **fan heat**,
//! raising the dry-bulb between the fan and the coils.
//!
//! ### Damper-to-speed mapping
//!
//! The damper position `d ∈ [0, 1]` maps linearly to a fan speed fraction
//! bounded by the minimum airflow ratio `r_min`:
//!
//! ```text
//! φ(d) = r_min + d · (1 − r_min)
//! ```
//!
//! At `d = 0` the terminal delivers `r_min · Q̇_max` (minimum ventilation);
//! at `d = 1` it delivers the full rated flow `Q̇_max`.
//!
//! ### Operating modes
//!
//! | Mode | Damper | Cooling coil | Reheat coil |
//! |------|--------|-------------|-------------|
//! | Cooling | modulated [min, max] | active (full capacity) | off |
//! | Heating | minimum | off | active (setpoint or PLR) |
//! | Deadband | minimum | off | off |
//!
//! In cooling mode the damper modulates airflow to track the zone load; the
//! coil capacity scales naturally with mass flow. In heating mode the damper
//! closes to minimum and the reheat coil warms the minimum-flow air.

use crate::sim::hvac::airside_state::{
    validate_finite, validate_nonnegative, AirsideCouplingError, MoistAirState,
};
use crate::sim::hvac::cooling_coil::{CoilPerformance, CoolingCoil, CoolingCoilBehavior};
use crate::sim::hvac::fan::{Fan, FanComponent};
use crate::sim::hvac::heating_coil::{HeatingCoil, HeatingCoilComponent, HeatingCoilControl};
use crate::sim::hvac::part_load_curves::FanPowerCurve;
use serde::{Deserialize, Serialize};

/// Operating mode of a VAV terminal unit.
///
/// Determines which coils are active and how the damper is interpreted. The
/// zone controller selects the mode based on the zone temperature relative to
/// its deadband setpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VavOperatingMode {
    /// Cooling: damper modulates airflow; cooling coil active; reheat off.
    Cooling,
    /// Heating: damper at minimum; cooling off; reheat coil active.
    Heating,
    /// Deadband: damper at minimum; all coils off (ventilation only).
    Deadband,
}

/// Control signal supplied to a [`VavTerminalUnit`].
///
/// Encapsulates the damper position and the on/off state of each coil so the
/// calling code (zone controller) can express a complete operating point in a
/// single value.
#[derive(Debug, Clone, PartialEq)]
pub struct VavTerminalControl {
    /// Damper position ∈ [0, 1]. `0` = minimum airflow, `1` = maximum.
    pub damper_position: f64,
    /// Whether the cooling coil is active at full effectiveness.
    pub cooling_active: bool,
    /// Reheat coil control. `None` turns the reheat off; `Some` delegates to
    /// the heating coil's own [`HeatingCoilControl`].
    pub reheat: Option<HeatingCoilControl>,
}

impl VavTerminalControl {
    /// Cooling mode: cooling coil active, damper at the given position, no
    /// reheat.
    pub fn cooling(damper_position: f64) -> Self {
        Self {
            damper_position,
            cooling_active: true,
            reheat: None,
        }
    }

    /// Heating (reheat) mode: damper at minimum (0.0), cooling off, reheat
    /// coil driving toward `supply_setpoint_c`.
    pub fn heating(supply_setpoint_c: f64) -> Self {
        Self {
            damper_position: 0.0,
            cooling_active: false,
            reheat: Some(HeatingCoilControl::LeavingTempSetpoint(supply_setpoint_c)),
        }
    }

    /// Deadband: damper at minimum, all coils off.
    pub fn deadband() -> Self {
        Self {
            damper_position: 0.0,
            cooling_active: false,
            reheat: None,
        }
    }

    /// Resolve the operating mode from the active controls.
    pub fn mode(&self) -> VavOperatingMode {
        if self.cooling_active {
            VavOperatingMode::Cooling
        } else if self.reheat.is_some() {
            VavOperatingMode::Heating
        } else {
            VavOperatingMode::Deadband
        }
    }
}

/// Full performance result of a VAV terminal calculation.
///
/// All capacity values are **positive** quantities: cooling capacity is the
/// heat removed from the air, reheat capacity is the heat added, and fan heat
/// is the shaft power dissipated into the airstream.
#[derive(Debug, Clone, PartialEq)]
pub struct VavTerminalPerformance {
    /// Operating mode that produced this result.
    pub mode: VavOperatingMode,
    /// Damper position used [0, 1].
    pub damper_position: f64,
    /// Fan speed fraction resolved from the damper position [0, 1].
    pub fan_speed_fraction: f64,
    /// Supply-air state delivered to the zone.
    pub supply_air: MoistAirState,
    /// Volumetric flow rate at the supply-air density [m³/s].
    pub volumetric_flow_m3_per_s: f64,
    /// Dry-air mass flow rate [kg/s].
    pub dry_air_mass_flow_kg_per_s: f64,
    /// Total cooling capacity from the cooling coil [W] (0 if inactive).
    pub cooling_total_capacity_w: f64,
    /// Sensible cooling capacity [W].
    pub cooling_sensible_capacity_w: f64,
    /// Latent cooling capacity [W].
    pub cooling_latent_capacity_w: f64,
    /// Sensible heat ratio of the cooling coil (sensible / total).
    pub cooling_shr: f64,
    /// Reheat (heating) capacity delivered [W] (0 if inactive or absent).
    pub reheat_capacity_w: f64,
    /// Fan shaft power [W].
    pub fan_shaft_power_w: f64,
    /// Fan motor electrical input power [W].
    pub fan_motor_power_w: f64,
    /// Fan heat added to the airstream [W] (equals shaft power).
    pub fan_heat_w: f64,
    /// Condensate removal rate from the cooling coil [kg/s].
    pub condensate_rate_kg_per_s: f64,
}

/// Trait for VAV terminal units.
///
/// Establishes the composition interface that the airside coupling layer
/// (`AirsideEnvelopeCoupler`, #1767) and BESTEST comparative cases (T1.5) will
/// use. Implementations own a fan, cooling coil, and optional reheat coil and
/// translate a [`VavTerminalControl`] into a complete [`VavTerminalPerformance`]
/// without coupling to the zone thermal solver.
pub trait VavTerminal: Send + Sync {
    /// Maximum (design) volumetric airflow [m³/s].
    fn max_airflow_m3_per_s(&self) -> f64;

    /// Minimum volumetric airflow [m³/s] (turndown limit).
    fn min_airflow_m3_per_s(&self) -> f64;

    /// Minimum airflow as a fraction of maximum ∈ [0, 1].
    fn min_airflow_ratio(&self) -> f64;

    /// Last persisted damper position ∈ [0, 1].
    fn current_damper_position(&self) -> f64;

    /// Rated total cooling capacity of the cooling coil [W].
    fn rated_cooling_capacity_w(&self) -> f64;

    /// Rated reheat capacity [W]. Returns 0.0 when no reheat coil is present.
    fn rated_reheat_capacity_w(&self) -> f64;

    /// Whether the terminal is equipped with a reheat coil.
    fn has_reheat(&self) -> bool;

    /// Compute the full terminal performance for the given entering-air state,
    /// air density, and control signal.
    ///
    /// The calculation is **stateless**: it does not mutate the terminal.
    /// Callers that want to persist the damper position should forward it to
    /// [`VavTerminal::update_state`].
    ///
    /// # Errors
    ///
    /// Returns [`AirsideCouplingError`] if any input is non-finite, negative,
    /// or produces a non-physical (supersaturated) leaving-air state.
    fn compute_terminal_performance(
        &self,
        entering: &MoistAirState,
        air_density_kg_per_m3: f64,
        control: &VavTerminalControl,
    ) -> Result<VavTerminalPerformance, AirsideCouplingError>;

    /// Persist the operating damper position from the most recent performance
    /// calculation.
    fn update_state(&mut self, damper_position: f64);
}

/// Reference implementation of a VAV terminal unit with reheat.
///
/// Owns a [`FanComponent`], [`CoolingCoil`], and an optional
/// [`HeatingCoilComponent`]. The fan establishes flow from the damper-derived
/// speed fraction; the cooling coil removes sensible and latent load; the
/// reheat coil (when present) raises the supply temperature for heating mode.
///
/// Fan shaft power is computed from the [`FanPowerCurve`] field
/// (`fan_power_curve`) rather than the raw cubed affinity law: the curve
/// multiplies the rated shaft power at the actual air density by the
/// flow-ratio coefficient, capturing ASHRAE 90.1-2016 §6.5.3.1.1 fan-power
/// allowance behaviour including static-pressure-reset (SPR) compensation.
/// See issue #2465 for the SPR compensation rationale.
///
/// All three sub-components are concrete types so that the terminal is
/// `Clone + Serialize + Deserialize` without trait-object indirection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VavTerminalUnit {
    /// Terminal-unit identifier.
    pub id: String,
    /// Index of the zone served by this terminal.
    pub zone_id: usize,
    /// Supply fan.
    pub fan: FanComponent,
    /// Cooling coil.
    pub cooling_coil: CoolingCoil,
    /// Optional reheat (heating) coil. `None` for cooling-only terminals.
    pub reheat_coil: Option<HeatingCoilComponent>,
    /// Minimum airflow as a fraction of the fan rated flow ∈ [0, 1].
    pub min_airflow_ratio: f64,
    /// Last persisted damper position ∈ [0, 1].
    pub current_damper_position: f64,
    /// Fan-power curve applied to the rated shaft power at the actual air
    /// density. Defaults to `FanPowerCurve::with_spr_compensation()` (the
    /// ASHRAE 90.1-2016 SPR-compensated polynomial). Overridable via
    /// [`VavTerminalUnit::with_fan_power_curve`].
    ///
    /// `#[serde(default)]` keeps backward compatibility with terminals
    /// serialized before issue #2465 (which encoded only the affinity-law
    /// path); missing field deserialises to the SPR-compensated default.
    #[serde(default = "default_spr_fan_power_curve")]
    pub fan_power_curve: FanPowerCurve,
}

/// Helper: returns the canonical SPR-compensated curve for `#[serde(default)]`.
fn default_spr_fan_power_curve() -> FanPowerCurve {
    FanPowerCurve::with_spr_compensation()
}

impl VavTerminalUnit {
    /// Create a VAV terminal with an auto-sized fan and the given coils.
    ///
    /// The fan is sized for `max_airflow_m3_per_s` at 500 Pa total pressure
    /// rise and 70 % total efficiency (standard commercial VAV fan). Override
    /// with [`VavTerminalUnit::with_fan`] for a custom fan curve.
    ///
    /// The minimum airflow ratio defaults to 0.30 (30 % turndown), a typical
    /// VAV box minimum-stop. Override with
    /// [`VavTerminalUnit::with_min_airflow_ratio`].
    ///
    /// The fan-power curve defaults to
    /// [`FanPowerCurve::with_spr_compensation`] (ASHRAE 90.1-2016
    /// SPR-compensated polynomial). Override with
    /// [`VavTerminalUnit::with_fan_power_curve`] to use a non-SPR curve or a
    /// custom coefficient set (issue #2465).
    pub fn new(
        id: String,
        zone_id: usize,
        max_airflow_m3_per_s: f64,
        cooling_coil: CoolingCoil,
        reheat_coil: Option<HeatingCoilComponent>,
    ) -> Self {
        let fan = FanComponent::new(format!("{id}-FAN"), max_airflow_m3_per_s, 500.0, 0.70);
        Self {
            id,
            zone_id,
            fan,
            cooling_coil,
            reheat_coil,
            min_airflow_ratio: 0.30,
            current_damper_position: 0.0,
            fan_power_curve: FanPowerCurve::with_spr_compensation(),
        }
    }

    /// Override the auto-sized fan with a custom [`FanComponent`].
    pub fn with_fan(mut self, fan: FanComponent) -> Self {
        self.fan = fan;
        self
    }

    /// Override the minimum airflow ratio (turndown fraction).
    pub fn with_min_airflow_ratio(mut self, ratio: f64) -> Self {
        self.min_airflow_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Override the fan-power curve (issue #2465).
    ///
    /// Pass [`FanPowerCurve::with_spr_compensation`] (the default) for the
    /// ASHRAE 90.1-2016 SPR-compensated polynomial, [`FanPowerCurve::new`]
    /// for the non-SPR quadratic curve, or a custom coefficient set via
    /// [`FanPowerCurve::with_coeffs`].
    pub fn with_fan_power_curve(mut self, curve: FanPowerCurve) -> Self {
        self.fan_power_curve = curve;
        self
    }

    /// Maximum volumetric airflow — the fan rated flow [m³/s].
    pub fn max_airflow_m3_per_s(&self) -> f64 {
        self.fan.rated_volumetric_flow()
    }

    /// Minimum volumetric airflow [m³/s].
    pub fn min_airflow_m3_per_s(&self) -> f64 {
        self.max_airflow_m3_per_s() * self.min_airflow_ratio
    }

    /// Map a damper position to a fan speed fraction.
    ///
    /// `d = 0` → `r_min`, `d = 1` → `1.0`. Values are clamped to `[0, 1]`.
    fn speed_fraction(&self, damper_position: f64) -> f64 {
        let d = damper_position.clamp(0.0, 1.0);
        self.min_airflow_ratio + d * (1.0 - self.min_airflow_ratio)
    }
}

impl VavTerminal for VavTerminalUnit {
    fn max_airflow_m3_per_s(&self) -> f64 {
        self.max_airflow_m3_per_s()
    }

    fn min_airflow_m3_per_s(&self) -> f64 {
        self.min_airflow_m3_per_s()
    }

    fn min_airflow_ratio(&self) -> f64 {
        self.min_airflow_ratio
    }

    fn current_damper_position(&self) -> f64 {
        self.current_damper_position
    }

    fn rated_cooling_capacity_w(&self) -> f64 {
        self.cooling_coil.rated_total_capacity()
    }

    fn rated_reheat_capacity_w(&self) -> f64 {
        self.reheat_coil
            .as_ref()
            .map(|c| c.rated_capacity_w())
            .unwrap_or(0.0)
    }

    fn has_reheat(&self) -> bool {
        self.reheat_coil.is_some()
    }

    fn compute_terminal_performance(
        &self,
        entering: &MoistAirState,
        air_density_kg_per_m3: f64,
        control: &VavTerminalControl,
    ) -> Result<VavTerminalPerformance, AirsideCouplingError> {
        entering.validate_derived()?;
        validate_finite("damper_position", control.damper_position)?;
        validate_nonnegative("air_density_kg_per_m3", air_density_kg_per_m3)?;

        let mode = control.mode();

        // ---- 1. Fan: damper → speed → flow, mass flow, power ---------------
        let speed_fraction = self.speed_fraction(control.damper_position);
        let volumetric_flow = self
            .fan
            .volumetric_flow(speed_fraction, air_density_kg_per_m3);
        let moist_mass_flow = self
            .fan
            .mass_flow_rate(speed_fraction, air_density_kg_per_m3);

        // Shaft power: rated power at the actual air density × the
        // `fan_power_curve` power ratio. Replaces the raw cubed affinity law
        // with the ASHRAE 90.1-2016 SPR-compensated polynomial (issue #2465).
        // At φ = 1.0 the ratio is 1.0 so full-speed operation is unchanged.
        let rated_shaft_power = self.fan.shaft_power(1.0, air_density_kg_per_m3);
        let power_ratio = self.fan_power_curve.power_ratio_at(speed_fraction);
        let shaft_power = rated_shaft_power * power_ratio;
        let rated_motor_power = self.fan.motor_power(1.0, air_density_kg_per_m3);
        let motor_power = rated_motor_power * power_ratio;

        // Dry-air mass flow for coil calculations.
        let w = entering.humidity_ratio_kg_per_kg_dry_air;
        let dry_air_mass_flow = if (1.0 + w) > 0.0 {
            moist_mass_flow / (1.0 + w)
        } else {
            0.0
        };

        // ---- 2. Fan heat → post-fan air state ------------------------------
        let cp_ma_j = entering.dry_air_specific_heat_j_per_kg_k();
        let fan_heat_w = shaft_power;

        let post_fan = if dry_air_mass_flow > 0.0 && fan_heat_w > 0.0 {
            let delta_t = fan_heat_w / (dry_air_mass_flow * cp_ma_j);
            MoistAirState::from_humidity_ratio(
                entering.dry_bulb_c + delta_t,
                w,
                entering.pressure_pa,
            )?
        } else {
            *entering
        };

        // ---- 3. Cooling coil (active only in Cooling mode) -----------------
        let (cooling_perf, post_cooling): (CoilPerformance, MoistAirState) =
            if control.cooling_active && dry_air_mass_flow > 0.0 {
                let perf = self
                    .cooling_coil
                    .compute_cooling_capacity(&post_fan, dry_air_mass_flow)?;
                let leaving = perf.leaving_air;
                (perf, leaving)
            } else {
                (zero_coil_performance(post_fan), post_fan)
            };

        // ---- 4. Reheat coil (active only in Heating mode) ------------------
        let (supply_air, reheat_capacity_w) = match (&control.reheat, &self.reheat_coil) {
            (Some(reheat_control), Some(coil)) if dry_air_mass_flow > 0.0 => {
                let result = coil.compute_heating_capacity(
                    &post_cooling,
                    dry_air_mass_flow,
                    *reheat_control,
                )?;
                (result.leaving_air, result.capacity_w)
            }
            _ => (post_cooling, 0.0),
        };

        Ok(VavTerminalPerformance {
            mode,
            damper_position: control.damper_position.clamp(0.0, 1.0),
            fan_speed_fraction: speed_fraction,
            supply_air,
            volumetric_flow_m3_per_s: volumetric_flow,
            dry_air_mass_flow_kg_per_s: dry_air_mass_flow,
            cooling_total_capacity_w: cooling_perf.total_capacity_w,
            cooling_sensible_capacity_w: cooling_perf.sensible_capacity_w,
            cooling_latent_capacity_w: cooling_perf.latent_capacity_w,
            cooling_shr: cooling_perf.shr,
            reheat_capacity_w,
            fan_shaft_power_w: shaft_power,
            fan_motor_power_w: motor_power,
            fan_heat_w,
            condensate_rate_kg_per_s: cooling_perf.condensate_rate_kg_per_s,
        })
    }

    fn update_state(&mut self, damper_position: f64) {
        self.current_damper_position = if damper_position.is_finite() {
            damper_position.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

/// Zero-capacity cooling result (coil off), leaving air unchanged.
fn zero_coil_performance(leaving_air: MoistAirState) -> CoilPerformance {
    CoilPerformance {
        total_capacity_w: 0.0,
        sensible_capacity_w: 0.0,
        latent_capacity_w: 0.0,
        shr: 0.0,
        leaving_air,
        condensate_rate_kg_per_s: 0.0,
    }
}

/// Helper trait for tests: override a deadband control into a cooling control
/// at a given damper position. (Defined only in cfg(test).)
#[cfg(test)]
trait CoolingOverride {
    fn cooling_override(self, damper_position: f64) -> Self;
}

#[cfg(test)]
impl CoolingOverride for VavTerminalControl {
    fn cooling_override(self, damper_position: f64) -> Self {
        VavTerminalControl::cooling(damper_position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::hvac::fan::STANDARD_AIR_DENSITY_KG_PER_M3;

    const SEA_LEVEL_PA: f64 = 101_325.0;

    fn entering_air(temp_c: f64, rh_percent: f64) -> MoistAirState {
        MoistAirState::try_new(temp_c, rh_percent, SEA_LEVEL_PA).expect("valid entering air")
    }

    /// Standard test terminal: 2.0 m³/s max flow, 30 kW cooling coil, 10 kW
    /// reheat coil, 30 % minimum airflow ratio.
    fn test_terminal() -> VavTerminalUnit {
        let cooling = CoolingCoil::new(
            "CC-1".to_string(),
            30_000.0, // 30 kW rated total
            0.75,     // rated SHR
            0.15,     // bypass factor
            10.0,     // ADP 10 °C
            2.0,      // design mass flow
        );
        let reheat = HeatingCoilComponent::new("HC-1".to_string(), 10_000.0, 2.0);
        VavTerminalUnit::new("VAV-1".to_string(), 0, 2.0, cooling, Some(reheat))
    }

    /// Terminal without a reheat coil (cooling-only VAV box).
    fn cooling_only_terminal() -> VavTerminalUnit {
        let cooling = CoolingCoil::new("CC-2".to_string(), 20_000.0, 0.75, 0.15, 10.0, 2.0);
        VavTerminalUnit::new("VAV-2".to_string(), 1, 2.0, cooling, None)
    }

    // -----------------------------------------------------------------------
    // Constructor & accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn constructor_sets_defaults() {
        let terminal = test_terminal();
        assert_eq!(terminal.id, "VAV-1");
        assert_eq!(terminal.zone_id, 0);
        assert!((terminal.max_airflow_m3_per_s() - 2.0).abs() < 1e-12);
        assert!((terminal.min_airflow_m3_per_s() - 0.6).abs() < 1e-12); // 30% of 2.0
        assert!((terminal.min_airflow_ratio - 0.30).abs() < 1e-12);
        assert_eq!(terminal.current_damper_position, 0.0);
        assert!(terminal.has_reheat());
        assert!((terminal.rated_cooling_capacity_w() - 30_000.0).abs() < 1e-9);
        assert!((terminal.rated_reheat_capacity_w() - 10_000.0).abs() < 1e-9);
        // Default fan-power curve is SPR-compensated (issue #2465).
        assert!(
            (terminal.fan_power_curve.power_ratio_at(0.30) - 0.1729).abs() < 1e-3,
            "default VAV fan-power curve should be SPR-compensated; \
             power_ratio_at(0.30) = {}",
            terminal.fan_power_curve.power_ratio_at(0.30)
        );
    }

    #[test]
    fn cooling_only_terminal_has_no_reheat() {
        let terminal = cooling_only_terminal();
        assert!(!terminal.has_reheat());
        assert_eq!(terminal.rated_reheat_capacity_w(), 0.0);
    }

    #[test]
    fn with_fan_overrides_auto_sized_fan() {
        let terminal = test_terminal();
        let custom = terminal.with_fan(FanComponent::with_motor(
            "custom".into(),
            3.0,
            750.0,
            0.80,
            0.92,
            STANDARD_AIR_DENSITY_KG_PER_M3,
        ));
        assert!((custom.max_airflow_m3_per_s() - 3.0).abs() < 1e-12);
        assert!((custom.fan.rated_pressure_rise() - 750.0).abs() < 1e-9);
    }

    #[test]
    fn with_min_airflow_ratio_clamps() {
        let terminal = test_terminal()
            .with_min_airflow_ratio(0.5)
            .with_min_airflow_ratio(-1.0);
        assert!((terminal.min_airflow_ratio - 0.0).abs() < 1e-12);

        let terminal = test_terminal().with_min_airflow_ratio(2.0);
        assert!((terminal.min_airflow_ratio - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Control mode resolution
    // -----------------------------------------------------------------------

    #[test]
    fn control_mode_resolution() {
        assert_eq!(
            VavTerminalControl::cooling(0.5).mode(),
            VavOperatingMode::Cooling
        );
        assert_eq!(
            VavTerminalControl::heating(30.0).mode(),
            VavOperatingMode::Heating
        );
        assert_eq!(
            VavTerminalControl::deadband().mode(),
            VavOperatingMode::Deadband
        );
    }

    #[test]
    fn heating_control_sets_damper_to_minimum() {
        let ctrl = VavTerminalControl::heating(30.0);
        assert_eq!(ctrl.damper_position, 0.0);
        assert!(!ctrl.cooling_active);
        assert!(ctrl.reheat.is_some());
    }

    // -----------------------------------------------------------------------
    // Damper → fan speed mapping
    // -----------------------------------------------------------------------

    #[test]
    fn damper_maps_to_speed_fraction() {
        let terminal = test_terminal();
        // min_ratio = 0.30
        assert!((terminal.speed_fraction(0.0) - 0.30).abs() < 1e-12);
        assert!((terminal.speed_fraction(1.0) - 1.00).abs() < 1e-12);
        assert!((terminal.speed_fraction(0.5) - 0.65).abs() < 1e-12); // 0.3 + 0.5*0.7
    }

    #[test]
    fn damper_clamps_outside_unit_interval() {
        let terminal = test_terminal();
        assert!((terminal.speed_fraction(-0.5) - 0.30).abs() < 1e-12); // clamps to 0
        assert!((terminal.speed_fraction(1.5) - 1.00).abs() < 1e-12); // clamps to 1
    }

    // -----------------------------------------------------------------------
    // Airflow modulation tests
    // -----------------------------------------------------------------------

    #[test]
    fn airflow_scales_with_damper_position() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf_min = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
            .unwrap();
        let perf_max = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .unwrap();

        // At damper=0 (deadband), flow = 30% of rated = 0.6 m³/s.
        assert!(
            (perf_min.volumetric_flow_m3_per_s - 0.6).abs() < 0.01,
            "min flow {} vs 0.6",
            perf_min.volumetric_flow_m3_per_s
        );
        // At damper=1, flow = rated = 2.0 m³/s.
        assert!(
            (perf_max.volumetric_flow_m3_per_s - 2.0).abs() < 0.01,
            "max flow {} vs 2.0",
            perf_max.volumetric_flow_m3_per_s
        );
        // Max flow must exceed min flow.
        assert!(perf_max.volumetric_flow_m3_per_s > perf_min.volumetric_flow_m3_per_s);
    }

    #[test]
    fn dry_air_mass_flow_proportional_to_volumetric_flow() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf_half = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(0.5))
            .unwrap();
        let perf_full = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .unwrap();

        // Half damper → 0.65 speed → 0.65 of full mass flow.
        let ratio = perf_half.dry_air_mass_flow_kg_per_s / perf_full.dry_air_mass_flow_kg_per_s;
        assert!(
            (ratio - 0.65).abs() < 0.01,
            "mass flow ratio {ratio} vs 0.65"
        );
    }

    // -----------------------------------------------------------------------
    // Cooling mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn cooling_mode_produces_cooled_supply_air() {
        let terminal = test_terminal();
        let entering = entering_air(30.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .unwrap();

        assert_eq!(perf.mode, VavOperatingMode::Cooling);
        assert!(
            perf.cooling_total_capacity_w > 0.0,
            "should have cooling capacity"
        );
        assert!(perf.cooling_sensible_capacity_w > 0.0);
        assert!(perf.cooling_latent_capacity_w > 0.0);
        assert!(
            perf.cooling_sensible_capacity_w + perf.cooling_latent_capacity_w
                - perf.cooling_total_capacity_w
                < 1.0,
            "sensible + latent must equal total"
        );
        // Supply air must be cooler than the entering air (plus fan heat).
        assert!(
            perf.supply_air.dry_bulb_c < entering.dry_bulb_c,
            "supply {} must be below entering {}",
            perf.supply_air.dry_bulb_c,
            entering.dry_bulb_c
        );
        // Condensate should be removed.
        assert!(perf.condensate_rate_kg_per_s > 0.0);
        assert_eq!(perf.reheat_capacity_w, 0.0);
    }

    #[test]
    fn cooling_capacity_scales_with_damper() {
        let terminal = test_terminal();
        let entering = entering_air(30.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf_half = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(0.5))
            .unwrap();
        let perf_full = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .unwrap();

        // More airflow → more capacity.
        assert!(
            perf_full.cooling_total_capacity_w > perf_half.cooling_total_capacity_w,
            "full {} should exceed half {}",
            perf_full.cooling_total_capacity_w,
            perf_half.cooling_total_capacity_w
        );
    }

    // -----------------------------------------------------------------------
    // Heating / reheat mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn reheat_mode_raises_supply_temperature() {
        let terminal = test_terminal();
        // Entering air already cooled (e.g. primary air at 13 °C).
        let entering = entering_air(13.0, 90.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::heating(25.0))
            .unwrap();

        assert_eq!(perf.mode, VavOperatingMode::Heating);
        assert!(perf.reheat_capacity_w > 0.0, "should deliver reheat");
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        // Supply should be near the 25 °C setpoint (within rated capacity).
        assert!(
            (perf.supply_air.dry_bulb_c - 25.0).abs() < 0.5,
            "supply {} should approach 25 °C",
            perf.supply_air.dry_bulb_c
        );
        // Damper at minimum.
        assert!(
            (perf.volumetric_flow_m3_per_s - 0.6).abs() < 0.01,
            "reheat should run at min flow"
        );
    }

    #[test]
    fn reheat_setpoint_above_rated_capacity_is_clamped() {
        let terminal = test_terminal();
        let entering = entering_air(13.0, 90.0);
        let rho = entering.density_kg_per_m3;

        // Demand a huge setpoint — reheat clamps at rated 10 kW.
        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::heating(100.0))
            .unwrap();

        assert!(
            (perf.reheat_capacity_w - 10_000.0).abs() < 1.0,
            "reheat {} must clamp at rated 10 kW",
            perf.reheat_capacity_w
        );
        assert!(perf.supply_air.dry_bulb_c < 100.0);
    }

    #[test]
    fn no_reheat_coil_ignores_reheat_control() {
        let terminal = cooling_only_terminal();
        let entering = entering_air(13.0, 90.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::heating(25.0))
            .unwrap();

        // No reheat coil → reheat capacity is 0 despite the heating control.
        assert_eq!(perf.reheat_capacity_w, 0.0);
        assert_eq!(perf.mode, VavOperatingMode::Heating);
    }

    // -----------------------------------------------------------------------
    // Deadband mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn deadband_mode_delivers_minimum_flow_no_coils() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
            .unwrap();

        assert_eq!(perf.mode, VavOperatingMode::Deadband);
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert_eq!(perf.reheat_capacity_w, 0.0);
        assert!(
            (perf.volumetric_flow_m3_per_s - 0.6).abs() < 0.01,
            "deadband flow should be minimum"
        );
    }

    // -----------------------------------------------------------------------
    // Fan heat tests
    // -----------------------------------------------------------------------

    #[test]
    fn fan_heat_raises_supply_in_deadband() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
            .unwrap();

        // In deadband the only heat source is the fan, so supply > entering.
        assert!(perf.fan_heat_w > 0.0);
        assert!(
            perf.supply_air.dry_bulb_c > entering.dry_bulb_c,
            "supply {} should exceed entering {} due to fan heat",
            perf.supply_air.dry_bulb_c,
            entering.dry_bulb_c
        );
    }

    /// VAV terminal fan power follows the SPR-compensated polynomial, not the
    /// raw cubed affinity law (issue #2465).
    ///
    /// At damper position `d = 1.0` → φ = 1.0; at `d = 0.5` → φ = 0.65 (because
    /// the standard minimum airflow ratio is 0.30). The SPR-compensated
    /// polynomial `P/P_r = 0.395·φ + 0.605·φ²` yields a ratio of 0.5124 at
    /// φ = 0.65, *not* the cubed affinity value of 0.2746. The 1 % tolerance
    /// is tight enough to catch the regression but loose enough to absorb
    /// floating-point rounding.
    #[test]
    fn fan_power_matches_spr_compensated_curve() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf_full = terminal
            .compute_terminal_performance(
                &entering,
                rho,
                &VavTerminalControl::deadband().cooling_override(1.0),
            )
            .unwrap();

        // Use two damper positions to verify SPR-compensated scaling.
        let perf_half = terminal
            .compute_terminal_performance(
                &entering,
                rho,
                &VavTerminalControl {
                    damper_position: 0.5,
                    cooling_active: false,
                    reheat: None,
                },
            )
            .unwrap();

        let phi_full = perf_full.fan_speed_fraction;
        let phi_half = perf_half.fan_speed_fraction;
        // Expected ratio from the SPR-compensated curve, NOT cubed affinity.
        let expected_ratio = terminal.fan_power_curve.power_ratio_at(phi_half)
            / terminal.fan_power_curve.power_ratio_at(phi_full);
        let actual_ratio = perf_half.fan_shaft_power_w / perf_full.fan_shaft_power_w;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "shaft power ratio {actual_ratio} vs SPR-compensated expected {expected_ratio} \
             (cubed affinity would give {})",
            (phi_half / phi_full).powi(3)
        );
    }

    /// The terminal's default SPR-compensated curve validates at the standard
    /// load points (re-uses the part-load curve unit-test guarantee).
    #[test]
    fn fan_power_curve_validates_at_load_points() {
        let terminal = test_terminal();
        use crate::sim::hvac::part_load_curves::PartLoadCurve;
        assert!(terminal.fan_power_curve.validate_at_load_points());

        // Also confirm the SPR-compensated curve gives the documented
        // ASHRAE 90.1-2016 reference values at the VAV part-load range.
        let curve = &terminal.fan_power_curve;
        assert!(
            (curve.power_ratio_at(0.30) - 0.1729).abs() < 1e-3,
            "SPR power ratio at φ=0.30 = {}, expected ~0.1729",
            curve.power_ratio_at(0.30)
        );
        assert!(
            (curve.power_ratio_at(1.00) - 1.0).abs() < 1e-9,
            "SPR power ratio at φ=1.00 = {}, expected 1.0",
            curve.power_ratio_at(1.00)
        );
    }

    /// A terminal configured with a non-SPR curve falls back to the standard
    /// quadratic coefficients and yields a different (lower) part-load ratio
    /// than the SPR-compensated default.
    #[test]
    fn with_fan_power_curve_overrides_spr_default() {
        let terminal = test_terminal().with_fan_power_curve(FanPowerCurve::new());
        // Non-SPR curve at φ=0.30 uses the quadratic coefficients
        // (b=0.51830, c=0.48170) → 0.51830*0.30 + 0.48170*0.09 = 0.19884.
        let ratio = terminal.fan_power_curve.power_ratio_at(0.30);
        assert!(
            (ratio - 0.19884).abs() < 1e-3,
            "non-SPR curve at φ=0.30 = {ratio}, expected ~0.19884 (quadratic)",
        );
        // Must differ from the SPR-compensated default (~0.1729).
        let spr_ratio = FanPowerCurve::with_spr_compensation().power_ratio_at(0.30);
        assert!(
            (ratio - spr_ratio).abs() > 1e-3,
            "non-SPR ratio {ratio} should differ from SPR ratio {spr_ratio}",
        );
    }

    // -----------------------------------------------------------------------
    // Energy balance
    // -----------------------------------------------------------------------

    #[test]
    fn energy_balance_cooling_minus_reheat_plus_fan_heat() {
        let terminal = test_terminal();
        let entering = entering_air(30.0, 60.0);
        let rho = entering.density_kg_per_m3;

        // Simultaneous cooling + reheat (dehumidification-reheat scenario).
        let perf = terminal
            .compute_terminal_performance(
                &entering,
                rho,
                &VavTerminalControl {
                    damper_position: 1.0,
                    cooling_active: true,
                    reheat: Some(HeatingCoilControl::LeavingTempSetpoint(20.0)),
                },
            )
            .unwrap();

        // Net enthalpy change of the air stream must equal:
        //   -cooling_total + reheat + fan_heat
        // Use full moist-air enthalpy (not just sensible ΔT) since cooling
        // removes latent heat via condensation.
        let h_entering = entering.enthalpy_kj_per_kg_dry_air;
        let h_supply = perf.supply_air.enthalpy_kj_per_kg_dry_air;
        let expected_delta_h =
            -perf.cooling_total_capacity_w + perf.reheat_capacity_w + perf.fan_heat_w;
        let actual_delta_h = perf.dry_air_mass_flow_kg_per_s * (h_supply - h_entering) * 1000.0;

        // Exact to within 0.5 % (psychrometric state reconstruction rounding).
        assert!(
            (actual_delta_h - expected_delta_h).abs() / expected_delta_h.abs().max(1.0) < 0.005,
            "energy balance: actual {actual_delta_h} W vs expected {expected_delta_h} W"
        );
    }

    /// Regression test for issue #2465 — VAV terminal fan power must track
    /// the ASHRAE 90.1-2016 SPR-compensated polynomial across the typical
    /// VAV part-load range. Compares the fluxion `fan_shaft_power_w` value at
    /// each damper position to the EnergyPlus reference ratio (proportional
    /// to rated shaft power). At full speed (φ = 1) both models agree exactly;
    /// at φ = 0.3 the SPR-compensated value is **6.4×** the raw cubed
    /// affinity value, eliminating the 44 % mean fan-energy error vs
    /// EnergyPlus reference documented in the issue.
    #[test]
    fn fan_power_matches_energyplus_reference_curve_issue_2465() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        // Damper positions to sweep across φ ∈ [0.30, 1.00] (the
        // min_airflow_ratio = 0.30 maps d=0.0 to φ=0.30).
        let damper_positions = [0.0_f64, 0.25, 0.5, 0.75, 1.0];

        // Reference: full speed shaft power at actual density.
        let rated_shaft_power = terminal.fan.shaft_power(1.0, rho);
        assert!(rated_shaft_power > 0.0);

        // Compute the SPR-compensated reference curve at each φ.
        for &d in &damper_positions {
            let phi = 0.30 + d * 0.70;
            let perf = terminal
                .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(d))
                .unwrap();
            let fluxion_ratio = perf.fan_shaft_power_w / rated_shaft_power;
            let reference_ratio = terminal.fan_power_curve.power_ratio_at(phi);
            assert!(
                (fluxion_ratio - reference_ratio).abs() < 1e-9,
                "damper={d}, φ={phi:.3}: fluxion={fluxion_ratio:.4} vs SPR={reference_ratio:.4}"
            );

            // EnergyPlus reference: ASHRAE 90.1-2016 SPR-compensated polynomial.
            // Verify the fluxion ratio matches within 1 % of the EnergyPlus
            // reference at the VAV min-airflow setpoint (φ = 0.30), where the
            // raw cubed affinity under-predicts by 84 %.
            if (phi - 0.30).abs() < 1e-9 {
                let raw_cubic = phi * phi * phi;
                let eplus_expected = 0.395 * phi + 0.605 * phi * phi;
                assert!(
                    (fluxion_ratio - eplus_expected).abs() < 1e-9,
                    "VAV min setpoint: fluxion {fluxion_ratio} vs EnergyPlus {eplus_expected}"
                );
                // Confirm the under-prediction of the raw affinity law vs
                // the new SPR-compensated value at this setpoint.
                assert!(
                    raw_cubic < 0.05,
                    "raw affinity at φ=0.30 should be ~0.027 (got {raw_cubic:.4})"
                );
                assert!(
                    eplus_expected > 0.15,
                    "SPR at φ=0.30 should be ~0.173 (got {eplus_expected:.4})"
                );
                assert!(
                    eplus_expected / raw_cubic > 5.0,
                    "SPR/raw ratio at φ=0.30 should be ≥6.4× (got {:.1}×)",
                    eplus_expected / raw_cubic
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn zero_minimum_ratio_allows_zero_flow() {
        let terminal = test_terminal().with_min_airflow_ratio(0.0);
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
            .unwrap();

        assert!(
            perf.volumetric_flow_m3_per_s.abs() < 1e-9,
            "zero min ratio + damper 0 → zero flow"
        );
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert_eq!(perf.fan_heat_w, 0.0);
        assert_eq!(perf.supply_air, entering);
    }

    #[test]
    fn negative_density_is_rejected() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let err = terminal
            .compute_terminal_performance(&entering, -1.0, &VavTerminalControl::deadband())
            .unwrap_err();
        assert!(matches!(err, AirsideCouplingError::InvalidInput { .. }));
    }

    #[test]
    fn update_state_persists_and_clamps_damper() {
        let mut terminal = test_terminal();
        assert_eq!(terminal.current_damper_position(), 0.0);

        terminal.update_state(0.7);
        assert!((terminal.current_damper_position() - 0.7).abs() < 1e-9);

        terminal.update_state(2.0);
        assert_eq!(terminal.current_damper_position(), 1.0);

        terminal.update_state(f64::NAN);
        assert_eq!(terminal.current_damper_position(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Serde & Clone
    // -----------------------------------------------------------------------

    #[test]
    fn clone_and_serde_round_trip() {
        let terminal = test_terminal();
        let cloned = terminal.clone();
        assert_eq!(terminal.id, cloned.id);
        assert!(
            (terminal.rated_cooling_capacity_w() - cloned.rated_cooling_capacity_w()).abs() < 1e-9
        );

        let json = serde_json::to_string(&terminal).expect("serialize");
        let back: VavTerminalUnit = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(terminal.id, back.id);
        assert!((terminal.max_airflow_m3_per_s() - back.max_airflow_m3_per_s()).abs() < 1e-9);
        assert!(back.has_reheat());
    }

    // -----------------------------------------------------------------------
    // Trait-object dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn trait_object_dispatch() {
        let terminal: Box<dyn VavTerminal> = Box::new(test_terminal());
        assert!((terminal.max_airflow_m3_per_s() - 2.0).abs() < 1e-9);
        assert!((terminal.min_airflow_m3_per_s() - 0.6).abs() < 1e-9);
        assert!(terminal.has_reheat());

        let entering = entering_air(30.0, 50.0);
        let rho = entering.density_kg_per_m3;
        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .unwrap();
        assert!(perf.cooling_total_capacity_w > 0.0);
    }

    // -----------------------------------------------------------------------
    // Integration: full cooling-to-heating mode transition
    // -----------------------------------------------------------------------

    #[test]
    fn mode_transition_cooling_then_reheat() {
        let terminal = test_terminal();
        let entering = entering_air(26.0, 55.0);
        let rho = entering.density_kg_per_m3;

        // Simulate a zone cooling demand → full damper, cooling on.
        let cooling_perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .unwrap();
        assert!(cooling_perf.supply_air.dry_bulb_c < entering.dry_bulb_c);
        assert!(cooling_perf.cooling_total_capacity_w > 0.0);

        // Now the zone is satisfied and needs heating → damper to min, reheat on.
        let primary = MoistAirState::try_new(13.0, 90.0, SEA_LEVEL_PA).unwrap();
        let rho_primary = primary.density_kg_per_m3;
        let heating_perf = terminal
            .compute_terminal_performance(&primary, rho_primary, &VavTerminalControl::heating(22.0))
            .unwrap();
        assert!(heating_perf.reheat_capacity_w > 0.0);
        assert!(
            (heating_perf.supply_air.dry_bulb_c - 22.0).abs() < 0.5,
            "reheat supply {} should approach 22 °C",
            heating_perf.supply_air.dry_bulb_c
        );
        assert_eq!(heating_perf.cooling_total_capacity_w, 0.0);
    }
}
