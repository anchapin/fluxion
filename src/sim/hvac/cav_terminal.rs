//! Constant Air Volume (CAV) terminal unit model (Issue #1903).
//!
//! A CAV terminal provides a fixed, constant airflow to a zone regardless of
//! cooling or heating demand. The airflow rate is set at design conditions and
//! does not modulate like a VAV terminal.
//!
//! ## Physical model
//!
//! A CAV terminal is the simplest airside configuration. Unlike VAV, there is
//! no damper to modulate - the supply fan runs at a constant speed and delivers
//! a fixed volumetric flow rate. The terminal processes air through stages:
//!
//! 1. **Fan** — establishes the volumetric flow at rated speed (constant).
//! 2. **Cooling coil** — removes sensible and latent heat using the ASHRAE
//!    bypass-factor psychrometric model. Active only in cooling mode.
//! 3. **Heating coil** — raises the supply dry-bulb for heating. Active
//!    only in heating mode.
//!
//! ### Operating modes
//!
//! | Mode | Airflow | Cooling coil | Heating coil |
//! |------|---------|-------------|---------------|
//! | Cooling | constant (rated) | active (full capacity) | off |
//! | Heating | constant (rated) | off | active (setpoint or PLR) |
//! | Deadband | constant (minimum) | off | off |
//!
//! ## EnergyPlus mapping
//!
//! This corresponds to `AirTerminal:SingleDuct:Uncontrolled` in EnergyPlus.

use crate::sim::hvac::airside_state::{validate_nonnegative, AirsideCouplingError, MoistAirState};
use crate::sim::hvac::cooling_coil::{CoilPerformance, CoolingCoil, CoolingCoilBehavior};
use crate::sim::hvac::fan::{Fan, FanComponent};
use crate::sim::hvac::heating_coil::{HeatingCoil, HeatingCoilComponent, HeatingCoilControl};
use crate::sim::hvac::part_load_curves::FanPowerCurve;
use serde::{Deserialize, Serialize};

/// Operating mode of a CAV terminal unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CavOperatingMode {
    /// Cooling: constant airflow; cooling coil active; heating off.
    Cooling,
    /// Heating: constant airflow; cooling off; heating coil active.
    Heating,
    /// Deadband: constant minimum airflow; all coils off (ventilation only).
    Deadband,
}

/// Control signal supplied to a [`CavTerminalUnit`].
///
/// For CAV, the control is simpler than VAV - only the heating coil
/// setpoint needs to be specified (or None to turn it off).
#[derive(Debug, Clone, PartialEq)]
pub struct CavTerminalControl {
    /// Whether the cooling coil is active at full effectiveness.
    pub cooling_active: bool,
    /// Heating coil control. `None` turns the heating off.
    pub heating: Option<HeatingCoilControl>,
}

impl CavTerminalControl {
    /// Cooling mode: cooling coil active, constant airflow.
    pub fn cooling() -> Self {
        Self {
            cooling_active: true,
            heating: None,
        }
    }

    /// Heating mode: heating coil driving toward `supply_setpoint_c`, constant airflow.
    pub fn heating(supply_setpoint_c: f64) -> Self {
        Self {
            cooling_active: false,
            heating: Some(HeatingCoilControl::LeavingTempSetpoint(supply_setpoint_c)),
        }
    }

    /// Deadband: all coils off, constant minimum airflow.
    pub fn deadband() -> Self {
        Self {
            cooling_active: false,
            heating: None,
        }
    }

    /// Resolve the operating mode from the active controls.
    pub fn mode(&self) -> CavOperatingMode {
        if self.cooling_active {
            CavOperatingMode::Cooling
        } else if self.heating.is_some() {
            CavOperatingMode::Heating
        } else {
            CavOperatingMode::Deadband
        }
    }
}

/// Full performance result of a CAV terminal calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct CavTerminalPerformance {
    /// Operating mode that produced this result.
    pub mode: CavOperatingMode,
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
    /// Heating capacity delivered [W] (0 if inactive or absent).
    pub heating_capacity_w: f64,
    /// Fan shaft power [W].
    pub fan_shaft_power_w: f64,
    /// Fan motor electrical input power [W].
    pub fan_motor_power_w: f64,
    /// Fan heat added to the airstream [W] (equals shaft power).
    pub fan_heat_w: f64,
    /// Condensate removal rate from the cooling coil [kg/s].
    pub condensate_rate_kg_per_s: f64,
}

impl Default for CavTerminalPerformance {
    fn default() -> Self {
        Self {
            mode: CavOperatingMode::Deadband,
            supply_air: MoistAirState::try_new(20.0, 50.0, 101325.0).unwrap(),
            volumetric_flow_m3_per_s: 0.0,
            dry_air_mass_flow_kg_per_s: 0.0,
            cooling_total_capacity_w: 0.0,
            cooling_sensible_capacity_w: 0.0,
            cooling_latent_capacity_w: 0.0,
            cooling_shr: 0.0,
            heating_capacity_w: 0.0,
            fan_shaft_power_w: 0.0,
            fan_motor_power_w: 0.0,
            fan_heat_w: 0.0,
            condensate_rate_kg_per_s: 0.0,
        }
    }
}

/// Trait for CAV terminal units.
pub trait CavTerminal: Send + Sync {
    /// Maximum (design) volumetric airflow [m³/s].
    fn max_airflow_m3_per_s(&self) -> f64;

    /// Minimum volumetric airflow [m³/s] (for deadband/ventilation mode).
    fn min_airflow_m3_per_s(&self) -> f64;

    /// Rated total cooling capacity of the cooling coil [W].
    fn rated_cooling_capacity_w(&self) -> f64;

    /// Rated heating capacity [W]. Returns 0.0 when no heating coil is present.
    fn rated_heating_capacity_w(&self) -> f64;

    /// Whether the terminal is equipped with a heating coil.
    fn has_heating(&self) -> bool;

    /// Compute the full terminal performance for the given entering-air state,
    /// air density, and control signal.
    fn compute_terminal_performance(
        &self,
        entering: &MoistAirState,
        air_density_kg_per_m3: f64,
        control: &CavTerminalControl,
    ) -> Result<CavTerminalPerformance, AirsideCouplingError>;
}

/// Reference implementation of a CAV terminal unit with optional heating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CavTerminalUnit {
    /// Terminal-unit identifier.
    pub id: String,
    /// Index of the zone served by this terminal.
    pub zone_id: usize,
    /// Supply fan.
    pub fan: FanComponent,
    /// Cooling coil.
    pub cooling_coil: CoolingCoil,
    /// Optional heating coil. `None` for cooling-only terminals.
    pub heating_coil: Option<HeatingCoilComponent>,
    /// Airflow fraction in deadband mode (0.0 to 1.0). Default 0.5 (50% of rated).
    pub deadband_airflow_fraction: f64,
    /// Fan-power curve applied to the rated shaft power at the actual air
    /// density. Defaults to [`FanPowerCurve::new`] — the non-SPR quadratic
    /// polynomial — because CAV boxes run at constant volume and do not
    /// exhibit static-pressure-reset savings (issue #2465). Overridable via
    /// [`CavTerminalUnit::with_fan_power_curve`].
    ///
    /// `#[serde(default)]` keeps backward compatibility with terminals
    /// serialized before issue #2465; missing field deserialises to the
    /// non-SPR default (equivalent to the existing CAV behaviour at full
    /// speed where the curve's power ratio is exactly 1.0).
    #[serde(default = "default_cav_fan_power_curve")]
    pub fan_power_curve: FanPowerCurve,
}

/// Helper: returns the canonical non-SPR curve for `#[serde(default)]`.
fn default_cav_fan_power_curve() -> FanPowerCurve {
    FanPowerCurve::new()
}

impl CavTerminalUnit {
    /// Create a new CAV terminal with an auto-sized fan and the given coils.
    ///
    /// The fan is sized for `max_airflow_m3_per_s` at 500 Pa total pressure
    /// rise and 70% total efficiency (standard commercial fan).
    ///
    /// The fan-power curve defaults to [`FanPowerCurve::new`] (the non-SPR
    /// quadratic polynomial — appropriate for constant-volume systems). At
    /// φ = 1.0 the ratio is 1.0, matching the existing full-speed behaviour.
    /// Override with [`CavTerminalUnit::with_fan_power_curve`] for SPR
    /// compensation (issue #2465).
    pub fn new(
        id: String,
        zone_id: usize,
        max_airflow_m3_per_s: f64,
        cooling_coil: CoolingCoil,
        heating_coil: Option<HeatingCoilComponent>,
    ) -> Self {
        let fan = FanComponent::new(format!("{id}-FAN"), max_airflow_m3_per_s, 500.0, 0.70);
        Self {
            id,
            zone_id,
            fan,
            cooling_coil,
            heating_coil,
            deadband_airflow_fraction: 0.5,
            fan_power_curve: FanPowerCurve::new(),
        }
    }

    /// Override the auto-sized fan with a custom [`FanComponent`].
    pub fn with_fan(mut self, fan: FanComponent) -> Self {
        self.fan = fan;
        self
    }

    /// Set the deadband airflow fraction (fraction of rated flow in deadband).
    pub fn with_deadband_fraction(mut self, fraction: f64) -> Self {
        self.deadband_airflow_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Override the fan-power curve (issue #2465).
    ///
    /// Pass [`FanPowerCurve::new`] (the default) for the standard non-SPR
    /// quadratic curve, [`FanPowerCurve::with_spr_compensation`] for the
    /// SPR-compensated polynomial, or a custom coefficient set via
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
        self.max_airflow_m3_per_s() * self.deadband_airflow_fraction
    }
}

impl CavTerminal for CavTerminalUnit {
    fn max_airflow_m3_per_s(&self) -> f64 {
        self.max_airflow_m3_per_s()
    }

    fn min_airflow_m3_per_s(&self) -> f64 {
        self.min_airflow_m3_per_s()
    }

    fn rated_cooling_capacity_w(&self) -> f64 {
        self.cooling_coil.rated_total_capacity()
    }

    fn rated_heating_capacity_w(&self) -> f64 {
        self.heating_coil
            .as_ref()
            .map(|c| c.rated_capacity_w())
            .unwrap_or(0.0)
    }

    fn has_heating(&self) -> bool {
        self.heating_coil.is_some()
    }

    fn compute_terminal_performance(
        &self,
        entering: &MoistAirState,
        air_density_kg_per_m3: f64,
        control: &CavTerminalControl,
    ) -> Result<CavTerminalPerformance, AirsideCouplingError> {
        entering.validate_derived()?;
        validate_nonnegative("air_density_kg_per_m3", air_density_kg_per_m3)?;

        let mode = control.mode();

        // Determine fan speed fraction based on mode
        // In cooling/heating: full speed; in deadband: reduced speed
        let fan_speed_fraction = match mode {
            CavOperatingMode::Deadband => self.deadband_airflow_fraction,
            _ => 1.0,
        };

        // ---- 1. Fan: constant speed → flow, mass flow, power ---------------
        let volumetric_flow = self
            .fan
            .volumetric_flow(fan_speed_fraction, air_density_kg_per_m3);
        let moist_mass_flow = self
            .fan
            .mass_flow_rate(fan_speed_fraction, air_density_kg_per_m3);

        // Shaft power: rated power at the actual air density × the
        // `fan_power_curve` power ratio. Replaces the raw cubed affinity law
        // with the standard ASHRAE quadratic polynomial (issue #2465).
        // At φ = 1.0 the non-SPR curve gives a ratio of 1.0, so full-speed
        // operation is unchanged.
        let rated_shaft_power = self.fan.shaft_power(1.0, air_density_kg_per_m3);
        let power_ratio = self.fan_power_curve.power_ratio_at(fan_speed_fraction);
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
                (zero_cooling_performance(post_fan), post_fan)
            };

        // ---- 4. Heating coil (active only in Heating mode) ----------------
        let (supply_air, heating_capacity_w) = match (&control.heating, &self.heating_coil) {
            (Some(heating_control), Some(coil)) if dry_air_mass_flow > 0.0 => {
                let result = coil.compute_heating_capacity(
                    &post_cooling,
                    dry_air_mass_flow,
                    *heating_control,
                )?;
                (result.leaving_air, result.capacity_w)
            }
            _ => (post_cooling, 0.0),
        };

        Ok(CavTerminalPerformance {
            mode,
            supply_air,
            volumetric_flow_m3_per_s: volumetric_flow,
            dry_air_mass_flow_kg_per_s: dry_air_mass_flow,
            cooling_total_capacity_w: cooling_perf.total_capacity_w,
            cooling_sensible_capacity_w: cooling_perf.sensible_capacity_w,
            cooling_latent_capacity_w: cooling_perf.latent_capacity_w,
            cooling_shr: cooling_perf.shr,
            heating_capacity_w,
            fan_shaft_power_w: shaft_power,
            fan_motor_power_w: motor_power,
            fan_heat_w,
            condensate_rate_kg_per_s: cooling_perf.condensate_rate_kg_per_s,
        })
    }
}

fn zero_cooling_performance(leaving_air: MoistAirState) -> CoilPerformance {
    CoilPerformance {
        total_capacity_w: 0.0,
        sensible_capacity_w: 0.0,
        latent_capacity_w: 0.0,
        shr: 0.0,
        leaving_air,
        condensate_rate_kg_per_s: 0.0,
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
    /// heating coil, 50% deadband airflow.
    fn test_terminal() -> CavTerminalUnit {
        let cooling = CoolingCoil::new(
            "CC-1".to_string(),
            30_000.0, // 30 kW rated total
            0.75,     // rated SHR
            0.15,     // bypass factor
            10.0,     // ADP 10 °C
            2.0,      // design mass flow
        );
        let heating = HeatingCoilComponent::new("HC-1".to_string(), 10_000.0, 2.0);
        CavTerminalUnit::new("CAV-1".to_string(), 0, 2.0, cooling, Some(heating))
    }

    /// Cooling-only terminal (no heating coil).
    fn cooling_only_terminal() -> CavTerminalUnit {
        let cooling = CoolingCoil::new("CC-2".to_string(), 20_000.0, 0.75, 0.15, 10.0, 2.0);
        CavTerminalUnit::new("CAV-2".to_string(), 1, 2.0, cooling, None)
    }

    // -----------------------------------------------------------------------
    // Constructor & accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn constructor_sets_defaults() {
        let terminal = test_terminal();
        assert_eq!(terminal.id, "CAV-1");
        assert_eq!(terminal.zone_id, 0);
        assert!((terminal.max_airflow_m3_per_s() - 2.0).abs() < 1e-12);
        assert!((terminal.min_airflow_m3_per_s() - 1.0).abs() < 1e-12); // 50% of 2.0
        assert!((terminal.deadband_airflow_fraction - 0.50).abs() < 1e-12);
        assert!(terminal.has_heating());
        assert!((terminal.rated_cooling_capacity_w() - 30_000.0).abs() < 1e-9);
        assert!((terminal.rated_heating_capacity_w() - 10_000.0).abs() < 1e-9);
        // Default fan-power curve is the non-SPR quadratic (issue #2465).
        // At φ = 1.0 (cooling/heating) and φ = 0.5 (deadband) the values
        // come from `0.5183·φ + 0.4817·φ²`.
        let ratio_full = terminal.fan_power_curve.power_ratio_at(1.0);
        let ratio_half = terminal.fan_power_curve.power_ratio_at(0.5);
        assert!((ratio_full - 1.0).abs() < 1e-9);
        assert!(
            (ratio_half - 0.5_f64.mul_add(0.5183, 0.4817 * 0.25)).abs() < 1e-9,
            "non-SPR curve at φ=0.5 = {ratio_half}"
        );
    }

    #[test]
    fn cooling_only_terminal_has_no_heating() {
        let terminal = cooling_only_terminal();
        assert!(!terminal.has_heating());
        assert_eq!(terminal.rated_heating_capacity_w(), 0.0);
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
    fn with_deadband_fraction_clamps() {
        let terminal = test_terminal()
            .with_deadband_fraction(0.3)
            .with_deadband_fraction(-1.0);
        assert!((terminal.deadband_airflow_fraction - 0.0).abs() < 1e-12);

        let terminal = test_terminal().with_deadband_fraction(2.0);
        assert!((terminal.deadband_airflow_fraction - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Control mode resolution
    // -----------------------------------------------------------------------

    #[test]
    fn control_mode_resolution() {
        assert_eq!(
            CavTerminalControl::cooling().mode(),
            CavOperatingMode::Cooling
        );
        assert_eq!(
            CavTerminalControl::heating(30.0).mode(),
            CavOperatingMode::Heating
        );
        assert_eq!(
            CavTerminalControl::deadband().mode(),
            CavOperatingMode::Deadband
        );
    }

    #[test]
    fn heating_control_sets_no_cooling() {
        let ctrl = CavTerminalControl::heating(30.0);
        assert!(!ctrl.cooling_active);
        assert!(ctrl.heating.is_some());
    }

    // -----------------------------------------------------------------------
    // Airflow tests
    // -----------------------------------------------------------------------

    #[test]
    fn airflow_is_constant_in_cooling_mode() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf_cooling = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::cooling())
            .unwrap();
        let perf_deadband = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::deadband())
            .unwrap();

        // In cooling mode, flow should be at rated maximum (2.0 m³/s)
        assert!(
            (perf_cooling.volumetric_flow_m3_per_s - 2.0).abs() < 0.01,
            "cooling flow {} should be rated max",
            perf_cooling.volumetric_flow_m3_per_s
        );

        // In deadband mode, flow should be at deadband fraction (1.0 m³/s = 50% of 2.0)
        assert!(
            (perf_deadband.volumetric_flow_m3_per_s - 1.0).abs() < 0.01,
            "deadband flow {} should be at deadband fraction",
            perf_deadband.volumetric_flow_m3_per_s
        );
    }

    #[test]
    fn airflow_is_constant_in_heating_mode() {
        let terminal = test_terminal();
        let entering = entering_air(20.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::heating(35.0))
            .unwrap();

        // In heating mode, flow should be at rated maximum (2.0 m³/s)
        assert!(
            (perf.volumetric_flow_m3_per_s - 2.0).abs() < 0.01,
            "heating flow {} should be rated max",
            perf.volumetric_flow_m3_per_s
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
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::cooling())
            .unwrap();

        assert_eq!(perf.mode, CavOperatingMode::Cooling);
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
        assert_eq!(perf.heating_capacity_w, 0.0);
    }

    // -----------------------------------------------------------------------
    // Heating mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn heating_mode_raises_supply_temperature() {
        let terminal = test_terminal();
        // Entering air at cold supply temperature (e.g., outdoor air in winter)
        let entering = entering_air(10.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::heating(14.0))
            .unwrap();

        assert_eq!(perf.mode, CavOperatingMode::Heating);
        assert!(perf.heating_capacity_w > 0.0, "should deliver heating");
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        // With 10kW heating and 2 m³/s flow starting from 10°C, we can raise
        // temperature by about 4-5°C (including fan heat), so ~14°C is achievable.
        assert!(
            perf.supply_air.dry_bulb_c > entering.dry_bulb_c,
            "supply {} should be warmer than entering {}",
            perf.supply_air.dry_bulb_c,
            entering.dry_bulb_c
        );
        assert!(
            (perf.supply_air.dry_bulb_c - 14.0).abs() < 1.0,
            "supply {} should approach 14 °C",
            perf.supply_air.dry_bulb_c
        );
    }

    #[test]
    fn no_heating_coil_ignores_heating_control() {
        let terminal = cooling_only_terminal();
        let entering = entering_air(20.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::heating(35.0))
            .unwrap();

        // No heating coil → heating capacity is 0 despite the heating control.
        assert_eq!(perf.heating_capacity_w, 0.0);
        assert_eq!(perf.mode, CavOperatingMode::Heating);
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
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::deadband())
            .unwrap();

        assert_eq!(perf.mode, CavOperatingMode::Deadband);
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert_eq!(perf.heating_capacity_w, 0.0);
        assert!(
            (perf.volumetric_flow_m3_per_s - 1.0).abs() < 0.01,
            "deadband flow should be at deadband fraction"
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
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::deadband())
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

    // -----------------------------------------------------------------------
    // Energy balance
    // -----------------------------------------------------------------------

    #[test]
    fn energy_balance_cooling_minus_heating_plus_fan_heat() {
        let terminal = test_terminal();
        let entering = entering_air(30.0, 60.0);
        let rho = entering.density_kg_per_m3;

        // Simultaneous cooling + heating (reheat scenario).
        let perf = terminal
            .compute_terminal_performance(
                &entering,
                rho,
                &CavTerminalControl {
                    cooling_active: true,
                    heating: Some(HeatingCoilControl::LeavingTempSetpoint(20.0)),
                },
            )
            .unwrap();

        // Net enthalpy change of the air stream must equal:
        //   -cooling_total + heating + fan_heat
        let h_entering = entering.enthalpy_kj_per_kg_dry_air;
        let h_supply = perf.supply_air.enthalpy_kj_per_kg_dry_air;
        let expected_delta_h =
            -perf.cooling_total_capacity_w + perf.heating_capacity_w + perf.fan_heat_w;
        let actual_delta_h = perf.dry_air_mass_flow_kg_per_s * (h_supply - h_entering) * 1000.0;

        // Exact to within 0.5% (psychrometric state reconstruction rounding).
        assert!(
            (actual_delta_h - expected_delta_h).abs() / expected_delta_h.abs().max(1.0) < 0.005,
            "energy balance: actual {actual_delta_h} W vs expected {expected_delta_h} W"
        );
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn zero_deadband_fraction_allows_zero_flow_in_deadband() {
        let terminal = test_terminal().with_deadband_fraction(0.0);
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        let perf = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::deadband())
            .unwrap();

        assert!(
            perf.volumetric_flow_m3_per_s.abs() < 1e-9,
            "zero deadband fraction → zero flow in deadband"
        );
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert_eq!(perf.fan_heat_w, 0.0);
    }

    #[test]
    fn negative_density_is_rejected() {
        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let err = terminal
            .compute_terminal_performance(&entering, -1.0, &CavTerminalControl::deadband())
            .unwrap_err();
        assert!(matches!(err, AirsideCouplingError::InvalidInput { .. }));
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
        let back: CavTerminalUnit = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(terminal.id, back.id);
        assert!((terminal.max_airflow_m3_per_s() - back.max_airflow_m3_per_s()).abs() < 1e-9);
        assert!(back.has_heating());
    }

    // -----------------------------------------------------------------------
    // Trait-object dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn trait_object_dispatch() {
        let terminal: Box<dyn CavTerminal> = Box::new(test_terminal());
        assert!((terminal.max_airflow_m3_per_s() - 2.0).abs() < 1e-9);
        assert!((terminal.min_airflow_m3_per_s() - 1.0).abs() < 1e-9);
        assert!(terminal.has_heating());

        let entering = entering_air(30.0, 50.0);
        let rho = entering.density_kg_per_m3;
        let perf = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::cooling())
            .unwrap();
        assert!(perf.cooling_total_capacity_w > 0.0);
    }

    /// Regression test for issue #2465 — CAV terminal fan power is routed
    /// through `FanPowerCurve::new()` (non-SPR). At full speed (φ = 1) the
    /// curve returns a power ratio of exactly 1.0; in deadband mode
    /// (φ = `deadband_airflow_fraction`) it returns a non-zero value via the
    /// quadratic polynomial — matching ASHRAE's standard fan-power model
    /// and eliminating the raw cubed affinity under-prediction.
    #[test]
    fn fan_power_matches_non_spr_curve_issue_2465() {
        use crate::sim::hvac::part_load_curves::PartLoadCurve;

        let terminal = test_terminal();
        let entering = entering_air(24.0, 50.0);
        let rho = entering.density_kg_per_m3;

        // Full-speed (cooling) → φ = 1.0 → ratio = 1.0 → rated shaft power.
        let rated_shaft_power = terminal.fan.shaft_power(1.0, rho);
        let perf_cooling = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::cooling())
            .unwrap();
        let cooling_ratio = perf_cooling.fan_shaft_power_w / rated_shaft_power;
        let cooling_expected = terminal.fan_power_curve.power_ratio_at(1.0);
        assert!(
            (cooling_ratio - cooling_expected).abs() < 1e-9,
            "cooling (φ=1): fluxion {cooling_ratio} vs curve {cooling_expected}"
        );

        // Deadband mode → φ = deadband_airflow_fraction = 0.5.
        let perf_deadband = terminal
            .compute_terminal_performance(&entering, rho, &CavTerminalControl::deadband())
            .unwrap();
        let deadband_ratio = perf_deadband.fan_shaft_power_w / rated_shaft_power;
        let deadband_expected = terminal.fan_power_curve.power_ratio_at(0.5);
        assert!(
            (deadband_ratio - deadband_expected).abs() < 1e-9,
            "deadband (φ=0.5): fluxion {deadband_ratio} vs curve {deadband_expected}"
        );

        // The non-SPR curve yields ~0.38 at φ = 0.5, which is the ASHRAE
        // constant-volume fan-power behaviour at half flow — substantially
        // larger than the raw cubed affinity value of 0.125, but still below
        // the SPR-compensated value (issue #2465 root-cause summary).
        assert!(deadband_expected > 0.35 && deadband_expected < 0.40);

        // Curve validates at the standard load points.
        assert!(terminal.fan_power_curve.validate_at_load_points());
    }
}
