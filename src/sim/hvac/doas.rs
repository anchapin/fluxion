//! Dedicated Outdoor Air System (DOAS) model (Issue #1765, Plan T2.6).
//!
//! A DOAS is the standard commercial configuration for **decoupled**
//! ventilation: a constant-volume unit conditions **100 % outdoor air** to a
//! fixed dew-point and a neutral supply dry-bulb, delivering ventilation
//! independently of the zone sensible or latent load. A separate sensible-only
//! zone system (typically a VAV terminal, #1764) then handles the zone
//! sensible load. Because the DOAS supply dew-point is held constant, the zone
//! latent load is removed at the DOAS rather than at the zone terminal — this
//! is the *latent + sensible decoupling* acceptance criterion.
//!
//! ## Physical model
//!
//! The DOAS composes the four airside component abstractions —
//! [`FanComponent`] (#1761), [`CoolingCoil`] (#1762),
//! [`HeatingCoilComponent`] (#1763), and optionally a
//! [`HumidifierComponent`](crate::sim::hvac::humidifier::HumidifierComponent)
//! (#2464) — but drives them with a **dew-point target** rather than a
//! zone-temperature target. The conditioning sequence is:
//!
//! 1. **Supply fan** — constant-volume, full design speed. Establishes the
//!    ventilation mass flow and dissipates shaft power as fan heat into the
//!    100 % outdoor airstream.
//! 2. **Cooling / dehumidification** — when the outdoor-air dew-point exceeds
//!    the target, the DOAS cools and dehumidifies to a leaving state whose
//!    humidity ratio equals the saturation humidity ratio at the target
//!    dew-point (`w_target = w_sat(T_dp,target)`). The leaving air is driven
//!    toward saturation at the target dew-point; because dew-point is a pure
//!    function of humidity ratio, the **leaving dew-point equals the target
//!    regardless of entering-air humidity** (the decoupling guarantee),
//!    provided the rated cooling capacity is not exceeded.
//! 3. **Neutral reheat** — a sensible-only reheat coil raises the dry-bulb to
//!    the neutral supply setpoint (typically 17–19 °C) at constant humidity
//!    ratio. This restores sensible capacity removed by the cooling coil
//!    without re-introducing moisture.
//! 4. **Winter humidification** (Issue #2464, optional) — when the DOAS is
//!    equipped with a humidifier and the post-reheat humidity ratio is below
//!    `w_target`, an adiabatic humidifier raises the leaving humidity ratio
//!    to `w_target` at constant dry-bulb (ideal-adiabatic simplification,
//!    matching EnergyPlus `Humidifier:Steam:Adiabatic`). This restores the
//!    ASHRAE 62.1 §6.4 minimum indoor humidity guidance in cold-dry climates.
//!
//! ### Mode selection
//!
//! The operating mode is resolved from the outdoor-air state relative to the
//! two setpoints (target dew-point and supply dry-bulb):
//!
//! | Outdoor-air condition | Mode | Active coils |
//!|-----------------------|------|--------------|
//! | dew-pt > target | `CoolingDehumidification` | cooling + reheat |
//! | dew-pt ≤ target, db < supply db | `HeatingOnly` | reheat |
//! | dew-pt ≤ target, db > supply db | `SensibleCooling` | cooling (sensible) |
//! | dew-pt ≤ target, db ≈ supply db | `Ventilation` | none (fan only) |
//! | `active = false` | `Off` | none |
//!
//! ### Capacity clamping
//!
//! If the cooling capacity required to reach the target dew-point exceeds the
//! rated capacity of the cooling coil, the leaving state is interpolated along
//! the psychrometric line from the entering air to the target saturated state
//! by the fraction `f = rated / required`. The achieved dew-point then exceeds
//! the target and [`DoasPerformance::target_dew_point_met`] is `false`.
//!
//! ### Scope / known limitations
//!
//! - **Winter humidification (Issue #2464).** When the DOAS is equipped with
//!   an optional [`HumidifierComponent`](crate::sim::hvac::humidifier::HumidifierComponent)
//!   and the post-reheat humidity ratio is below `w_sat(target_dew_point)` —
//!   i.e., outdoor air is drier than the target dew-point — the DOAS engages
//!   an adiabatic humidifier stage that drives the leaving humidity ratio to
//!   the target. This restores the ASHRAE 62.1 §6.4 minimum indoor humidity
//!   guidance in cold-dry climates (4–6 months/yr in ASHRAE 169 climate zones
//!   5B, 6A, 7, 8). The latent heat `Q_lat = ṁ_h2o · h_fg` is delivered to
//!   the airstream and credited by [`airside_coupling`](crate::sim::hvac::airside_coupling)
//!   via `supply_latent_heat_w`. When the humidifier is `None` (the default)
//!   or the outdoor air is already at/above the target, behavior is
//!   identical to the pre-#2464 implementation.
//! - **Blow-through fan placement.** Fan heat is applied to the outdoor air
//!   before the coils, matching the [`VavTerminalUnit`] convention.
//!
//! ## Composability with VAV
//!
//! A [`DoasPerformance`] resolves to the same [`MoistAirState`] + volume-flow
//! boundary used by [`VavTerminalPerformance`](crate::sim::hvac::vav_terminal),
//! so a DOAS supply can feed a VAV terminal's entering-air argument directly,
//! with the VAV damper/reheat handling only zone-sensible load.

use crate::sim::hvac::airside_state::{
    validate_finite, validate_nonnegative, validate_positive, AirsideCouplingError, MoistAirState,
};
use crate::sim::hvac::cooling_coil::{CoolingCoil, CoolingCoilBehavior};
use crate::sim::hvac::fan::{Fan, FanComponent};
use crate::sim::hvac::heating_coil::{HeatingCoil, HeatingCoilComponent, HeatingCoilControl};
use crate::sim::hvac::humidifier::{Humidifier, HumidifierComponent, HumidifierControl};
use fluxion_core::weather::psychrometrics::{calculate_dew_point, calculate_humidity_ratio};
use serde::{Deserialize, Serialize};

/// Specific heat of dry air, kJ/(kg·K) — ASHRAE HoF Ch.1.
const CP_DRY_AIR_KJ_PER_KG_K: f64 = 1.006;
/// Specific heat of water vapor, kJ/(kg·K) — ASHRAE HoF Ch.1.
const CP_WATER_VAPOR_KJ_PER_KG_K: f64 = 1.86;
/// Deadband half-width on the supply dry-bulb comparison [°C].
const SUPPLY_DB_DEADBAND_C: f64 = 0.05;

/// Operating mode of a DOAS unit.
///
/// Resolved from the outdoor-air state relative to the target dew-point and
/// the neutral supply dry-bulb setpoint. See the module docs for the mode
/// selection table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoasMode {
    /// Cooling + dehumidification: outdoor dew-point above target.
    /// Cooling coil drives leaving humidity ratio to the target; reheat coil
    /// restores the dry-bulb to the neutral supply setpoint.
    CoolingDehumidification,
    /// Sensible cooling only: outdoor dew-point at/below target but dry-bulb
    /// above the supply setpoint. Cools at constant humidity ratio.
    SensibleCooling,
    /// Sensible heating only: outdoor dew-point at/below target and dry-bulb
    /// below the supply setpoint. Reheat coil raises dry-bulb; no
    /// dehumidification.
    HeatingOnly,
    /// Ventilation: outdoor air already neutral (dew-pt ≤ target, dry-bulb
    /// within deadband of supply setpoint). Fan runs; no coils active.
    Ventilation,
    /// Unit off: fan off, no conditioning, supply equals outdoor air.
    Off,
}

/// Control signal supplied to a [`Doas`] unit.
///
/// A DOAS is controlled by two fixed setpoints — the target leaving dew-point
/// (latent) and the neutral supply dry-bulb (sensible) — plus an on/off flag.
/// There is no zone-coupled term: the DOAS tracks its own setpoints, not the
/// zone load, which is the basis of the sensible/latent decoupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoasControl {
    /// Target leaving dew-point [°C]. The DOAS holds the supply dew-point at
    /// this value whenever the cooling coil has sufficient capacity. Typical
    /// neutral-air dew-points are 7–11 °C (≈ 45–50 % RH at 20 °C zone temp).
    pub target_dew_point_c: f64,
    /// Neutral supply dry-bulb setpoint [°C]. Post-reheat leaving temperature.
    /// Typical DOAS neutral supply is 16–19 °C.
    pub supply_dry_bulb_c: f64,
    /// Whether the DOAS fan and conditioning are active. When `false` the unit
    /// is off and the supply equals the outdoor air at zero flow.
    pub active: bool,
}

impl DoasControl {
    /// Build a control for an active DOAS at the given setpoints.
    pub fn active(target_dew_point_c: f64, supply_dry_bulb_c: f64) -> Self {
        Self {
            target_dew_point_c,
            supply_dry_bulb_c,
            active: true,
        }
    }

    /// Build an off control (fan and coils inactive).
    pub fn off() -> Self {
        Self {
            target_dew_point_c: 10.0,
            supply_dry_bulb_c: 18.0,
            active: false,
        }
    }
}

/// Full performance result of a DOAS calculation.
///
/// All capacity values are **positive** quantities reported as for
/// [`crate::sim::hvac::vav_terminal::VavTerminalPerformance`]: cooling is heat
/// removed from the air, reheat is heat added, fan heat is shaft power
/// dissipated into the airstream, and humidifier is latent heat delivered to
/// the air.
#[derive(Debug, Clone, PartialEq)]
pub struct DoasPerformance {
    /// Operating mode that produced this result.
    pub mode: DoasMode,
    /// Supply-air state delivered to the zone (or downstream VAV terminal).
    pub supply_air: MoistAirState,
    /// Volumetric flow rate at the supply-air density [m³/s]. Constant for an
    /// active DOAS (full fan speed); zero when off.
    pub volumetric_flow_m3_per_s: f64,
    /// Dry-air mass flow rate [kg/s].
    pub dry_air_mass_flow_kg_per_s: f64,
    /// Total cooling capacity (sensible + latent) [W] (0 if no cooling).
    pub cooling_total_capacity_w: f64,
    /// Sensible cooling capacity [W].
    pub cooling_sensible_capacity_w: f64,
    /// Latent cooling capacity [W] (dehumidification load).
    pub cooling_latent_capacity_w: f64,
    /// Sensible heat ratio of the cooling stage (sensible / total).
    pub cooling_shr: f64,
    /// Neutral reheat capacity delivered [W] (0 if no reheat).
    pub reheat_capacity_w: f64,
    /// Adiabatic humidifier capacity delivered [W] (latent heat added to
    /// the airstream, 0 if no humidifier or no winter humidification active).
    /// Issue #2464.
    pub humidifier_capacity_w: f64,
    /// Moisture addition rate from the humidifier [kg_water/s].
    /// Zero when no humidifier or no winter humidification active.
    /// Issue #2464.
    pub humidifier_moisture_rate_kg_per_s: f64,
    /// Whether the winter humidification stage was active this step.
    /// `true` only when the humidifier ran (post-reheat humidity ratio was
    /// below `w_sat(target_dew_point)` and a humidifier was present).
    /// Issue #2464.
    pub humidifier_active: bool,
    /// Fan shaft power [W].
    pub fan_shaft_power_w: f64,
    /// Fan motor electrical input power [W].
    pub fan_motor_power_w: f64,
    /// Fan heat added to the airstream [W] (equals shaft power).
    pub fan_heat_w: f64,
    /// Condensate removal rate from the cooling coil [kg/s].
    pub condensate_rate_kg_per_s: f64,
    /// Dew-point actually achieved at the supply [°C].
    pub supply_dew_point_c: f64,
    /// Whether the target dew-point was met (`true` when the cooling coil had
    /// sufficient capacity to reach `target_dew_point_c`, **or** when the
    /// humidifier raised the leaving humidity ratio to `w_sat(target_dew_point)`
    /// in cold-dry conditions). Always `true` in
    /// `SensibleCooling` / `HeatingOnly` without humidifier / `Ventilation` /
    /// `Off` modes.
    pub target_dew_point_met: bool,
}

impl DoasPerformance {
    /// Net total heat delivered to the zone [W].
    ///
    /// `−cooling_total + reheat + fan_heat`. Negative values mean the DOAS is
    /// a net cooling source (typical in dehumidification mode); positive means
    /// net heating (winter heating-only mode).
    pub fn net_zone_heat_w(&self) -> f64 {
        -self.cooling_total_capacity_w + self.reheat_capacity_w + self.fan_heat_w
    }
}

/// Trait for Dedicated Outdoor Air System units.
///
/// Establishes the composition interface mirrored from
/// [`crate::sim::hvac::vav_terminal::VavTerminal`] so that the airside coupling
/// layer (`AirsideEnvelopeCoupler`, #1767) can treat a DOAS supply uniformly.
/// Implementations own a fan, cooling coil, and optional reheat coil and
/// translate a [`DoasControl`] into a complete [`DoasPerformance`] without
/// coupling to the zone thermal solver.
pub trait Doas: Send + Sync {
    /// Constant outdoor-air volumetric flow rate at full fan speed [m³/s].
    fn outdoor_air_flow_m3_per_s(&self) -> f64;

    /// Rated total cooling capacity of the cooling coil [W].
    fn rated_cooling_capacity_w(&self) -> f64;

    /// Rated reheat capacity [W]. Returns 0.0 when no reheat coil is present.
    fn rated_reheat_capacity_w(&self) -> f64;

    /// Whether the unit is equipped with a reheat coil.
    fn has_reheat(&self) -> bool;

    /// Compute the full DOAS performance for the given outdoor-air state, air
    /// density, and control signal.
    ///
    /// The calculation is **stateless**: it does not mutate the unit. Callers
    /// that want to persist the resolved mode should forward it to
    /// [`Doas::update_state`].
    ///
    /// # Errors
    ///
    /// Returns [`AirsideCouplingError`] if any input is non-finite, negative,
    /// or produces a non-physical (supersaturated) leaving-air state.
    fn compute_doas_performance(
        &self,
        outdoor_air: &MoistAirState,
        air_density_kg_per_m3: f64,
        control: &DoasControl,
    ) -> Result<DoasPerformance, AirsideCouplingError>;

    /// Persist the operating mode from the most recent performance calculation.
    fn update_state(&mut self, mode: DoasMode);
}

/// Reference implementation of a constant-volume DOAS.
///
/// Owns a [`FanComponent`] (always run at full design speed), a [`CoolingCoil`]
/// (dehumidification + sensible cooling), an optional
/// [`HeatingCoilComponent`] (neutral reheat), and an optional
/// [`HumidifierComponent`] (winter humidification — Issue #2464). The cooling
/// coil's rated capacity clamps the achievable dehumidification; its bypass
/// factor and apparatus dew point are retained for the rated-capacity
/// reference, but the controlled DOAS overrides the leaving-air humidity
/// ratio to the target dew-point whenever capacity permits (see module docs).
///
/// All sub-components are concrete types so the unit is
/// `Clone + Serialize + Deserialize` without trait-object indirection, matching
/// [`crate::sim::hvac::vav_terminal::VavTerminalUnit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoasUnit {
    /// Unit identifier.
    pub id: String,
    /// Constant-volume supply fan.
    pub fan: FanComponent,
    /// Cooling / dehumidification coil.
    pub cooling_coil: CoolingCoil,
    /// Optional neutral reheat coil. `None` for cooling-only DOAS (rare; most
    /// DOAS reheat to a neutral supply temperature).
    pub reheat_coil: Option<HeatingCoilComponent>,
    /// Optional adiabatic humidifier for winter humidification (Issue #2464).
    /// Engaged when the post-reheat humidity ratio is below
    /// `w_sat(target_dew_point)` to satisfy ASHRAE 62.1 §6.4 minimum indoor
    /// humidity guidance in cold-dry climates.
    pub humidifier: Option<HumidifierComponent>,
    /// Last persisted operating mode.
    pub current_mode: DoasMode,
}

impl DoasUnit {
    /// Create a DOAS with an auto-sized fan and the given coils.
    ///
    /// The fan is sized for `design_outdoor_air_flow_m3_per_s` at 750 Pa total
    /// pressure rise (typical DOAS external static, higher than a zone VAV fan)
    /// and 70 % total efficiency. Override with [`DoasUnit::with_fan`] for a
    /// custom fan.
    ///
    /// Pass `None` for the humidifier to disable winter humidification (the
    /// pre-#2464 behavior). Pass `Some(...)` to enable the stage; the
    /// humidifier activates automatically when the post-reheat humidity ratio
    /// is below `w_sat(target_dew_point)`.
    pub fn new(
        id: String,
        design_outdoor_air_flow_m3_per_s: f64,
        cooling_coil: CoolingCoil,
        reheat_coil: Option<HeatingCoilComponent>,
        humidifier: Option<HumidifierComponent>,
    ) -> Self {
        let fan = FanComponent::new(
            format!("{id}-FAN"),
            design_outdoor_air_flow_m3_per_s,
            750.0,
            0.70,
        );
        Self {
            id,
            fan,
            cooling_coil,
            reheat_coil,
            humidifier,
            current_mode: DoasMode::Off,
        }
    }

    /// Override the auto-sized fan with a custom [`FanComponent`].
    pub fn with_fan(mut self, fan: FanComponent) -> Self {
        self.fan = fan;
        self
    }

    /// Equip the DOAS with an adiabatic humidifier (Issue #2464). Use this
    /// to add a humidifier after construction without rebuilding the unit.
    pub fn with_humidifier(mut self, humidifier: HumidifierComponent) -> Self {
        self.humidifier = Some(humidifier);
        self
    }

    /// Whether the unit is equipped with a winter humidifier.
    pub fn has_humidifier(&self) -> bool {
        self.humidifier.is_some()
    }

    /// Constant outdoor-air volumetric flow rate — the fan rated flow [m³/s].
    pub fn outdoor_air_flow_m3_per_s(&self) -> f64 {
        self.fan.rated_volumetric_flow()
    }

    /// Resolve the operating mode from the entering-air state and control.
    ///
    /// Pure function: does not depend on the unit's coils, only on the
    /// setpoints and the outdoor-air psychrometrics. Public so tests can
    /// verify the mode-selection logic in isolation.
    pub fn resolve_mode(
        outdoor_air: &MoistAirState,
        control: &DoasControl,
    ) -> Result<DoasMode, AirsideCouplingError> {
        validate_finite("target_dew_point_c", control.target_dew_point_c)?;
        validate_finite("supply_dry_bulb_c", control.supply_dry_bulb_c)?;
        outdoor_air.validate_derived()?;

        if !control.active {
            return Ok(DoasMode::Off);
        }

        let oa_dp = dew_point_of(outdoor_air);
        let target_dp = control.target_dew_point_c;

        if oa_dp > target_dp {
            Ok(DoasMode::CoolingDehumidification)
        } else {
            // dew-point satisfied; decide on sensible conditioning
            let db_diff = outdoor_air.dry_bulb_c - control.supply_dry_bulb_c;
            if db_diff < -SUPPLY_DB_DEADBAND_C {
                Ok(DoasMode::HeatingOnly)
            } else if db_diff > SUPPLY_DB_DEADBAND_C {
                Ok(DoasMode::SensibleCooling)
            } else {
                Ok(DoasMode::Ventilation)
            }
        }
    }
}

impl Doas for DoasUnit {
    fn outdoor_air_flow_m3_per_s(&self) -> f64 {
        self.outdoor_air_flow_m3_per_s()
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

    fn compute_doas_performance(
        &self,
        outdoor_air: &MoistAirState,
        air_density_kg_per_m3: f64,
        control: &DoasControl,
    ) -> Result<DoasPerformance, AirsideCouplingError> {
        outdoor_air.validate_derived()?;
        validate_nonnegative("air_density_kg_per_m3", air_density_kg_per_m3)?;
        validate_finite("target_dew_point_c", control.target_dew_point_c)?;
        validate_finite("supply_dry_bulb_c", control.supply_dry_bulb_c)?;

        let mode = DoasUnit::resolve_mode(outdoor_air, control)?;
        let pressure_pa = outdoor_air.pressure_pa;

        // ---- Off: no flow, no coils ----------------------------------------
        if mode == DoasMode::Off {
            return Ok(DoasPerformance {
                mode,
                supply_air: *outdoor_air,
                volumetric_flow_m3_per_s: 0.0,
                dry_air_mass_flow_kg_per_s: 0.0,
                cooling_total_capacity_w: 0.0,
                cooling_sensible_capacity_w: 0.0,
                cooling_latent_capacity_w: 0.0,
                cooling_shr: 0.0,
                reheat_capacity_w: 0.0,
                humidifier_capacity_w: 0.0,
                humidifier_moisture_rate_kg_per_s: 0.0,
                humidifier_active: false,
                fan_shaft_power_w: 0.0,
                fan_motor_power_w: 0.0,
                fan_heat_w: 0.0,
                condensate_rate_kg_per_s: 0.0,
                supply_dew_point_c: dew_point_of(outdoor_air),
                target_dew_point_met: true,
            });
        }

        // ---- 1. Fan: constant full speed → flow, mass flow, power ---------
        let speed_fraction = 1.0; // DOAS is constant-volume at design speed.
        let volumetric_flow = self
            .fan
            .volumetric_flow(speed_fraction, air_density_kg_per_m3);
        let moist_mass_flow = self
            .fan
            .mass_flow_rate(speed_fraction, air_density_kg_per_m3);
        let shaft_power = self.fan.shaft_power(speed_fraction, air_density_kg_per_m3);
        let motor_power = self.fan.motor_power(speed_fraction, air_density_kg_per_m3);

        let w_oa = outdoor_air.humidity_ratio_kg_per_kg_dry_air;
        let dry_air_mass_flow = if (1.0 + w_oa) > 0.0 {
            moist_mass_flow / (1.0 + w_oa)
        } else {
            0.0
        };

        // ---- 2. Fan heat → post-fan state ---------------------------------
        let cp_ma_j = outdoor_air.dry_air_specific_heat_j_per_kg_k();
        let fan_heat_w = shaft_power;
        let post_fan = if dry_air_mass_flow > 0.0 && fan_heat_w > 0.0 {
            let delta_t = fan_heat_w / (dry_air_mass_flow * cp_ma_j);
            MoistAirState::from_humidity_ratio(outdoor_air.dry_bulb_c + delta_t, w_oa, pressure_pa)?
        } else {
            *outdoor_air
        };

        // ---- 3. Mode-specific conditioning --------------------------------
        let (
            cooling_total,
            cooling_sensible,
            cooling_latent,
            cooling_shr,
            condensate,
            post_cooling,
            target_met,
        ) = match mode {
            DoasMode::CoolingDehumidification => {
                cool_dehumidify(&post_fan, dry_air_mass_flow, control, &self.cooling_coil)?
            }
            DoasMode::SensibleCooling => {
                let supply_db = control.supply_dry_bulb_c;
                sensible_cool_to(&post_fan, dry_air_mass_flow, supply_db, pressure_pa)?
            }
            // No cooling in these modes; post-cooling equals post-fan.
            DoasMode::HeatingOnly | DoasMode::Ventilation | DoasMode::Off => zero_cooling(post_fan),
        };

        // ---- 4. Neutral reheat (sensible) ---------------------------------
        // Reheat brings the dry-bulb up to the supply setpoint at constant
        // humidity ratio. Active when the post-cooling dry-bulb is below the
        // setpoint and a reheat coil is present.
        let (supply_air, reheat_capacity_w) =
            match (&self.reheat_coil, mode, dry_air_mass_flow > 0.0) {
                (Some(coil), _, true) if post_cooling.dry_bulb_c < control.supply_dry_bulb_c => {
                    let result = coil.compute_heating_capacity(
                        &post_cooling,
                        dry_air_mass_flow,
                        HeatingCoilControl::LeavingTempSetpoint(control.supply_dry_bulb_c),
                    )?;
                    (result.leaving_air, result.capacity_w)
                }
                _ => (post_cooling, 0.0),
            };

        // ---- 5. Winter humidification (Issue #2464) -----------------------
        // Engages the humidifier when the post-reheat humidity ratio is below
        // `w_sat(target_dew_point)` — i.e., outdoor air is drier than the
        // target dew-point setpoint. The humidifier drives the leaving humidity
        // ratio to `w_sat(target_dew_point)`, restoring the ASHRAE 62.1 §6.4
        // minimum indoor humidity guidance in cold-dry climates (4–6 months/yr
        // in ASHRAE 169 climate zones 5B, 6A, 7, 8). Latent heat is delivered
        // to the airstream as `Q_lat = �_h2o · h_fg`; the airside coupling
        // layer (`airside_coupling.rs`) credits this via `supply_latent_heat_w`.
        //
        // The comparison uses a 1 %-of-target epsilon to absorb round-trip
        // drift between the cooling coil's `from_humidity_ratio` write and
        // the same expression in `w_target_saturation` (both go through the
        // psychrometric library's saturation table, which can introduce
        // sub-microscopic round-off).
        let w_target_saturation =
            calculate_humidity_ratio(control.target_dew_point_c, 100.0, pressure_pa);
        let w_supply = supply_air.humidity_ratio_kg_per_kg_dry_air;
        let w_engage_epsilon = w_target_saturation * 1.0e-4;
        let needs_humidification =
            dry_air_mass_flow > 0.0 && w_supply < w_target_saturation - w_engage_epsilon;
        let (
            supply_air,
            humidifier_capacity_w,
            humidifier_moisture_rate_kg_per_s,
            humidifier_active,
            target_met_after_humidifier,
        ) = match (&self.humidifier, needs_humidification) {
            (Some(humidifier), true) => {
                let result = humidifier.compute_humidification_capacity(
                    &supply_air,
                    dry_air_mass_flow,
                    HumidifierControl::TargetHumidityRatio(w_target_saturation),
                )?;
                // After the humidifier, the leaving humidity ratio is at the
                // target saturation (subject to the saturation guard inside
                // the humidifier when the reheat coil is capacity-limited);
                // the dew-point target is met when the post-humidifier
                // humidity ratio is at or above the target.
                let dp_met = result.leaving_air.humidity_ratio_kg_per_kg_dry_air
                    >= w_target_saturation - w_engage_epsilon;
                (
                    result.leaving_air,
                    result.capacity_w,
                    result.moisture_rate_kg_per_s,
                    true,
                    target_met && dp_met,
                )
            }
            _ => (supply_air, 0.0, 0.0, false, target_met),
        };

        Ok(DoasPerformance {
            mode,
            supply_air,
            volumetric_flow_m3_per_s: volumetric_flow,
            dry_air_mass_flow_kg_per_s: dry_air_mass_flow,
            cooling_total_capacity_w: cooling_total,
            cooling_sensible_capacity_w: cooling_sensible,
            cooling_latent_capacity_w: cooling_latent,
            cooling_shr,
            reheat_capacity_w,
            humidifier_capacity_w,
            humidifier_moisture_rate_kg_per_s,
            humidifier_active,
            fan_shaft_power_w: shaft_power,
            fan_motor_power_w: motor_power,
            fan_heat_w,
            condensate_rate_kg_per_s: condensate,
            supply_dew_point_c: dew_point_of(&supply_air),
            target_dew_point_met: target_met_after_humidifier,
        })
    }

    fn update_state(&mut self, mode: DoasMode) {
        self.current_mode = mode;
    }
}

// ---------------------------------------------------------------------------
// Stage helpers
// ---------------------------------------------------------------------------

/// Type alias for the cooling-stage result tuple.
type CoolingStage = (f64, f64, f64, f64, f64, MoistAirState, bool);

/// Cooling + dehumidification to the target dew-point.
///
/// Drives the leaving-air humidity ratio to `w_sat(target_dp)` so that the
/// leaving dew-point equals the target. If the required cooling exceeds the
/// coil rated capacity, the leaving state is interpolated along the
/// entering→target enthalpy path by `f = rated / required`; the achieved
/// dew-point then exceeds the target and the met-flag is `false`. The clamp is
/// **enthalpy-exact**: the leaving enthalpy is set so the delivered capacity
/// equals the rated capacity to numerical precision.
///
/// Capacities follow ASHRAE HoF (2021) Ch.1:
/// `q_total = ṁ_da·(h_in − h_out)·1000`, `q_sens = ṁ_da·c_p,ma·(T_in − T_out)·1000`,
/// `q_lat = q_total − q_sens`, `ṁ_cond = ṁ_da·(w_in − w_out)`.
fn cool_dehumidify(
    entering: &MoistAirState,
    dry_air_mass_flow: f64,
    control: &DoasControl,
    coil: &CoolingCoil,
) -> Result<CoolingStage, AirsideCouplingError> {
    let pressure_pa = entering.pressure_pa;
    let target_dp = control.target_dew_point_c;

    // Target leaving humidity ratio = saturation ratio at the target dew-point.
    let w_target = calculate_humidity_ratio(target_dp, 100.0, pressure_pa);
    // Target leaving dry-bulb: drive toward saturation at the target dew-point
    // (the minimum-energy state achieving w_target). A real coil leaves air at
    // ~95–99 % RH near the ADP; using the saturated state is the conservative
    // max-dehumidification reference and makes the dew-point guarantee exact.
    let t_target = target_dp;
    let h_in = entering.enthalpy_kj_per_kg_dry_air;
    let h_target = enthalpy_from_state(t_target, w_target);
    let w_in = entering.humidity_ratio_kg_per_kg_dry_air;
    let required_cooling = (dry_air_mass_flow * (h_in - h_target) * 1000.0).max(0.0);
    let rated = coil.rated_total_capacity();

    // Resolve the leaving state.
    let (leaving, target_met) = if required_cooling <= rated || required_cooling <= 0.0 {
        // Full dehumidification: leaving air at the target saturated state.
        let leaving = MoistAirState::from_humidity_ratio(t_target, w_target, pressure_pa)?;
        (leaving, true)
    } else {
        // Capacity-limited: enthalpy-exact clamp along the entering→target path.
        // f = rated / required makes the delivered total capacity equal `rated`.
        let f = (rated / required_cooling).clamp(0.0, 1.0);
        let h_out = h_in - f * (h_in - h_target);
        let w_out = w_in + f * (w_target - w_in);
        // Solve T_out from h = 1.006·T + w·(2501 + 1.86·T):
        //   T = (h − 2501·w) / (1.006 + 1.86·w).
        let denom = CP_DRY_AIR_KJ_PER_KG_K + CP_WATER_VAPOR_KJ_PER_KG_K * w_out;
        let t_out = if denom > 0.0 {
            (h_out - 2501.0 * w_out) / denom
        } else {
            entering.dry_bulb_c
        };
        let leaving = MoistAirState::from_humidity_ratio(t_out, w_out, pressure_pa)?;
        (leaving, false)
    };

    let t_out = leaving.dry_bulb_c;
    let w_out = leaving.humidity_ratio_kg_per_kg_dry_air;
    let h_out = leaving.enthalpy_kj_per_kg_dry_air;
    let cp_ma = CP_DRY_AIR_KJ_PER_KG_K + CP_WATER_VAPOR_KJ_PER_KG_K * w_in;

    let total = (dry_air_mass_flow * (h_in - h_out) * 1000.0).max(0.0);
    let sensible = (dry_air_mass_flow * cp_ma * (entering.dry_bulb_c - t_out) * 1000.0).max(0.0);
    let latent = (total - sensible).max(0.0);
    let shr = if total > 0.0 {
        (sensible / total).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let condensate = (dry_air_mass_flow * (w_in - w_out)).max(0.0);

    Ok((
        total, sensible, latent, shr, condensate, leaving, target_met,
    ))
}

/// Sensible-only cooling at constant humidity ratio toward a target dry-bulb.
///
/// Used when the outdoor dew-point is already at/below target but the dry-bulb
/// exceeds the supply setpoint. No condensate, no latent term. Capacity is not
/// clamped against the cooling coil here (sensible-only cooling demand is
/// small relative to the dehumidification case); the caller-selected coil
/// sizing bounds it in practice.
fn sensible_cool_to(
    entering: &MoistAirState,
    dry_air_mass_flow: f64,
    target_db_c: f64,
    pressure_pa: f64,
) -> Result<CoolingStage, AirsideCouplingError> {
    let w = entering.humidity_ratio_kg_per_kg_dry_air;
    let cp_ma = CP_DRY_AIR_KJ_PER_KG_K + CP_WATER_VAPOR_KJ_PER_KG_K * w;

    let (leaving, total, sensible) = if entering.dry_bulb_c > target_db_c && dry_air_mass_flow > 0.0
    {
        let leaving = MoistAirState::from_humidity_ratio(target_db_c, w, pressure_pa)?;
        let total = dry_air_mass_flow * cp_ma * (entering.dry_bulb_c - target_db_c) * 1000.0;
        (leaving, total, total)
    } else {
        (*entering, 0.0, 0.0)
    };

    Ok((
        total,
        sensible,
        0.0, // no latent term in sensible-only cooling
        if total > 0.0 { 1.0 } else { 0.0 },
        0.0, // no condensate
        leaving,
        true, // dew-point target trivially met (no dehumidification attempted)
    ))
}

/// Zero-capacity cooling result (coil off), leaving air unchanged.
fn zero_cooling(leaving_air: MoistAirState) -> CoolingStage {
    (0.0, 0.0, 0.0, 0.0, 0.0, leaving_air, true)
}

// ---------------------------------------------------------------------------
// Psychrometric helpers
// ---------------------------------------------------------------------------

/// Enthalpy of moist air [kJ/kg_da] from dry-bulb and humidity ratio.
///
/// `h = 1.006·T + w·(2501 + 1.86·T)` — ASHRAE HoF (2021) Ch.1.
fn enthalpy_from_state(dry_bulb_c: f64, humidity_ratio: f64) -> f64 {
    CP_DRY_AIR_KJ_PER_KG_K * dry_bulb_c
        + humidity_ratio * (2501.0 + CP_WATER_VAPOR_KJ_PER_KG_K * dry_bulb_c)
}

/// Dew-point of a [`MoistAirState`] [°C].
///
/// Reconstructs the relative humidity from the humidity ratio and delegates to
/// `fluxion_core::weather::psychrometrics::calculate_dew_point`.
fn dew_point_of(state: &MoistAirState) -> f64 {
    // The MoistAirState already carries relative_humidity_percent; use it directly.
    calculate_dew_point(
        state.dry_bulb_c,
        state.relative_humidity_percent,
        state.pressure_pa,
    )
}

// Suppress unused-import warning for `validate_positive` when the feature set
// does not exercise it; kept for API symmetry with sibling modules.
#[allow(dead_code)]
fn _assert_validate_positive_used(v: f64) -> Result<(), AirsideCouplingError> {
    validate_positive("unused", v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::hvac::fan::STANDARD_AIR_DENSITY_KG_PER_M3;

    const SEA_LEVEL_PA: f64 = 101_325.0;

    fn outdoor_air(temp_c: f64, rh_percent: f64) -> MoistAirState {
        MoistAirState::try_new(temp_c, rh_percent, SEA_LEVEL_PA).expect("valid outdoor air")
    }

    /// Standard test DOAS: 1.5 m³/s outdoor air (≈1.8 kg_da/s), 150 kW cooling
    /// coil, 35 kW reheat coil. Sized to dehumidify up to 35 °C/80 % RH OA to a
    /// 10 °C dew-point and to reheat the cold coil leaving air to an 18 °C
    /// neutral supply. Target dew-point 10 °C, neutral supply 18 °C.
    /// No humidifier — preserves the pre-#2464 behavior.
    fn test_doas() -> DoasUnit {
        let cooling = CoolingCoil::new(
            "CC-DOAS".to_string(),
            150_000.0, // 150 kW rated total
            0.50,      // rated SHR (latent-dominated, typical DOAS)
            0.10,      // bypass factor
            10.0,      // ADP 10 °C
            1.8,       // design mass flow
        );
        let reheat = HeatingCoilComponent::new("HC-DOAS".to_string(), 35_000.0, 1.8);
        DoasUnit::new("DOAS-1".to_string(), 1.5, cooling, Some(reheat), None)
    }

    /// Test DOAS with a winter humidifier (Issue #2464). Sized for the same
    /// airflow and reheat but with a 0.050 kg_water/s adiabatic humidifier —
    /// well above the ≈ 4.16e-3 kg/s required for −10 °C / 20 % RH outdoor air
    /// targeting a 10 °C dew-point.
    fn test_doas_with_humidifier() -> DoasUnit {
        let cooling = CoolingCoil::new("CC-DOAS".to_string(), 150_000.0, 0.50, 0.10, 10.0, 1.8);
        let reheat = HeatingCoilComponent::new("HC-DOAS".to_string(), 35_000.0, 1.8);
        let humidifier = HumidifierComponent::new("HUM-DOAS".to_string(), 0.050, 1.8);
        DoasUnit::new(
            "DOAS-1H".to_string(),
            1.5,
            cooling,
            Some(reheat),
            Some(humidifier),
        )
    }

    /// DOAS without a reheat coil (cooling-only — atypical but valid).
    fn cooling_only_doas() -> DoasUnit {
        let cooling = CoolingCoil::new("CC-2".to_string(), 120_000.0, 0.50, 0.10, 10.0, 1.8);
        DoasUnit::new("DOAS-2".to_string(), 1.5, cooling, None, None)
    }

    // -----------------------------------------------------------------------
    // Constructor & accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn constructor_sets_defaults() {
        let doas = test_doas();
        assert_eq!(doas.id, "DOAS-1");
        assert!((doas.outdoor_air_flow_m3_per_s() - 1.5).abs() < 1e-12);
        assert!(doas.has_reheat());
        assert!((doas.rated_cooling_capacity_w() - 150_000.0).abs() < 1e-9);
        assert!((doas.rated_reheat_capacity_w() - 35_000.0).abs() < 1e-9);
        assert_eq!(doas.current_mode, DoasMode::Off);
    }

    #[test]
    fn cooling_only_doas_has_no_reheat() {
        let doas = cooling_only_doas();
        assert!(!doas.has_reheat());
        assert_eq!(doas.rated_reheat_capacity_w(), 0.0);
    }

    #[test]
    fn with_fan_overrides_auto_sized_fan() {
        let doas = test_doas().with_fan(FanComponent::with_motor(
            "custom".into(),
            3.0,
            900.0,
            0.75,
            0.92,
            STANDARD_AIR_DENSITY_KG_PER_M3,
        ));
        assert!((doas.outdoor_air_flow_m3_per_s() - 3.0).abs() < 1e-12);
        assert!((doas.fan.rated_pressure_rise() - 900.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Mode resolution
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_mode_off_when_inactive() {
        let oa = outdoor_air(32.0, 60.0);
        let mode = DoasUnit::resolve_mode(&oa, &DoasControl::off()).unwrap();
        assert_eq!(mode, DoasMode::Off);
    }

    #[test]
    fn resolve_mode_cooling_dehumidification_when_oa_humid() {
        // OA 32 °C / 60 % RH → dew-point ≈ 23 °C > 10 °C target.
        let oa = outdoor_air(32.0, 60.0);
        let mode = DoasUnit::resolve_mode(&oa, &DoasControl::active(10.0, 18.0)).unwrap();
        assert_eq!(mode, DoasMode::CoolingDehumidification);
    }

    #[test]
    fn resolve_mode_heating_only_when_oa_cold_and_dry() {
        // OA 2 °C / 30 % RH → dew-point well below 10 °C, db < 18 °C.
        let oa = outdoor_air(2.0, 30.0);
        let mode = DoasUnit::resolve_mode(&oa, &DoasControl::active(10.0, 18.0)).unwrap();
        assert_eq!(mode, DoasMode::HeatingOnly);
    }

    #[test]
    fn resolve_mode_sensible_cooling_when_oa_warm_but_dry() {
        // OA 30 °C / 15 % RH → dew-point well below 10 °C, db > 18 °C.
        let oa = outdoor_air(30.0, 15.0);
        let mode = DoasUnit::resolve_mode(&oa, &DoasControl::active(10.0, 18.0)).unwrap();
        assert_eq!(mode, DoasMode::SensibleCooling);
    }

    #[test]
    fn resolve_mode_ventilation_when_oa_already_neutral() {
        // OA 18 °C / 50 % RH → dew-point ≈ 7.4 °C ≤ 10 °C, db ≈ 18 °C.
        let oa = outdoor_air(18.0, 50.0);
        let mode = DoasUnit::resolve_mode(&oa, &DoasControl::active(10.0, 18.0)).unwrap();
        assert_eq!(mode, DoasMode::Ventilation);
    }

    // -----------------------------------------------------------------------
    // Constant-volume behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn flow_is_constant_regardless_of_load() {
        let doas = test_doas();
        let oa_hot = outdoor_air(35.0, 70.0);
        let oa_mild = outdoor_air(18.0, 50.0);
        let rho_hot = oa_hot.density_kg_per_m3;
        let rho_mild = oa_mild.density_kg_per_m3;

        let perf_hot = doas
            .compute_doas_performance(&oa_hot, rho_hot, &DoasControl::active(10.0, 18.0))
            .unwrap();
        let perf_mild = doas
            .compute_doas_performance(&oa_mild, rho_mild, &DoasControl::active(10.0, 18.0))
            .unwrap();

        // Constant volume: flow independent of OA condition (rated flow = 1.5).
        assert!(
            (perf_hot.volumetric_flow_m3_per_s - 1.5).abs() < 0.01,
            "hot flow {} vs 1.5",
            perf_hot.volumetric_flow_m3_per_s
        );
        assert!(
            (perf_mild.volumetric_flow_m3_per_s - 1.5).abs() < 0.01,
            "mild flow {} vs 1.5",
            perf_mild.volumetric_flow_m3_per_s
        );
    }

    // -----------------------------------------------------------------------
    // Dehumidification mode: the decoupling guarantee
    // -----------------------------------------------------------------------

    #[test]
    fn dehumidification_holds_fixed_dew_point_across_oa_humidity() {
        // KEY DECOUPLING TEST: supply dew-point must equal the target (10 °C)
        // across a range of outdoor humidities, because the DOAS dehumidifies
        // to a fixed w_target independent of the entering-air state.
        let doas = test_doas();
        let target_dp = 10.0;
        let control = DoasControl::active(target_dp, 18.0);

        for (t_oa, rh_oa) in [(32.0, 40.0), (32.0, 60.0), (35.0, 80.0), (28.0, 90.0)] {
            let oa = outdoor_air(t_oa, rh_oa);
            let rho = oa.density_kg_per_m3;
            let perf = doas.compute_doas_performance(&oa, rho, &control).unwrap();

            assert_eq!(
                perf.mode,
                DoasMode::CoolingDehumidification,
                "OA {t_oa}/{rh_oa} should be dehumidification mode"
            );
            assert!(
                perf.target_dew_point_met,
                "OA {t_oa}/{rh_oa}: target dew-point should be met (capacity sufficient)"
            );
            // Supply dew-point must match the target exactly (reheat preserves w).
            assert!(
                (perf.supply_dew_point_c - target_dp).abs() < 0.05,
                "OA {t_oa}/{rh_oa}: supply dew-pt {} vs target {target_dp}",
                perf.supply_dew_point_c
            );
        }
    }

    #[test]
    fn dehumidification_removes_condensate_and_dominates_latent() {
        let doas = test_doas();
        let oa = outdoor_air(32.0, 60.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert_eq!(perf.mode, DoasMode::CoolingDehumidification);
        assert!(perf.cooling_total_capacity_w > 0.0);
        assert!(perf.cooling_latent_capacity_w > 0.0);
        assert!(perf.condensate_rate_kg_per_s > 0.0);
        // DOAS is latent-dominated (SHR < 0.6 for deep dehumidification).
        assert!(
            perf.cooling_shr < 0.6,
            "SHR {} should be latent-dominated",
            perf.cooling_shr
        );
        // Supply humidity ratio must be below outdoor-air ratio.
        assert!(
            perf.supply_air.humidity_ratio_kg_per_kg_dry_air < oa.humidity_ratio_kg_per_kg_dry_air,
            "supply must be drier than OA"
        );
        // Sensible + latent must equal total.
        assert!(
            (perf.cooling_sensible_capacity_w + perf.cooling_latent_capacity_w
                - perf.cooling_total_capacity_w)
                .abs()
                < 1.0,
            "sensible + latent must equal total"
        );
    }

    #[test]
    fn dehumidification_reheats_to_neutral_supply_db() {
        let doas = test_doas();
        let oa = outdoor_air(32.0, 60.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert!(perf.reheat_capacity_w > 0.0, "reheat should be active");
        // Supply dry-bulb must be near the 18 °C neutral setpoint.
        assert!(
            (perf.supply_air.dry_bulb_c - 18.0).abs() < 0.5,
            "supply db {} should approach 18 °C",
            perf.supply_air.dry_bulb_c
        );
    }

    #[test]
    fn capacity_clamp_flags_unmet_dew_point() {
        // Undersized cooling coil: 5 kW cannot dehumidify 32 °C/60 % OA at
        // 1.8 kg/s (required ≈ 88 kW). The DOAS must report
        // target_dew_point_met = false and a supply dew-point above 10 °C.
        // The enthalpy-exact clamp delivers exactly the rated 5 kW.
        let cooling = CoolingCoil::new("tiny".to_string(), 5_000.0, 0.50, 0.10, 10.0, 1.8);
        let reheat = HeatingCoilComponent::new("hc".to_string(), 35_000.0, 1.8);
        let doas = DoasUnit::new("small".to_string(), 1.5, cooling, Some(reheat), None);

        let oa = outdoor_air(32.0, 60.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert!(
            !perf.target_dew_point_met,
            "undersized coil must flag unmet dew-point"
        );
        // Achieved dew-point must be above the target.
        assert!(
            perf.supply_dew_point_c > 10.0,
            "achieved dp {} should exceed target",
            perf.supply_dew_point_c
        );
        // Cooling delivered must equal the rated clamp (enthalpy-exact).
        assert!(
            (perf.cooling_total_capacity_w - 5_000.0).abs() < 1.0e-3,
            "cooling {} must clamp at rated 5 kW",
            perf.cooling_total_capacity_w
        );
    }

    // -----------------------------------------------------------------------
    // Heating-only mode (winter)
    // -----------------------------------------------------------------------

    #[test]
    fn heating_only_raises_db_without_dehumidification() {
        let doas = test_doas();
        let oa = outdoor_air(2.0, 30.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert_eq!(perf.mode, DoasMode::HeatingOnly);
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert!(perf.reheat_capacity_w > 0.0);
        assert_eq!(perf.condensate_rate_kg_per_s, 0.0);
        // Humidity ratio unchanged (no humidification — documented limitation).
        assert!(
            (perf.supply_air.humidity_ratio_kg_per_kg_dry_air
                - oa.humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1e-9,
            "heating-only must not change humidity ratio"
        );
        // Supply dry-bulb near the 18 °C setpoint.
        assert!(
            (perf.supply_air.dry_bulb_c - 18.0).abs() < 0.5,
            "supply db {} should approach 18 °C",
            perf.supply_air.dry_bulb_c
        );
    }

    // -----------------------------------------------------------------------
    // Sensible-cooling & ventilation modes
    // -----------------------------------------------------------------------

    #[test]
    fn sensible_cooling_lowers_db_preserves_humidity() {
        let doas = test_doas();
        // OA 30 °C / 15 % RH → dry, warm.
        let oa = outdoor_air(30.0, 15.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert_eq!(perf.mode, DoasMode::SensibleCooling);
        assert!(perf.cooling_total_capacity_w > 0.0);
        assert_eq!(perf.cooling_latent_capacity_w, 0.0);
        assert_eq!(perf.condensate_rate_kg_per_s, 0.0);
        assert!(
            (perf.cooling_shr - 1.0).abs() < 1e-9,
            "sensible-only SHR = 1"
        );
        // Humidity ratio unchanged.
        assert!(
            (perf.supply_air.humidity_ratio_kg_per_kg_dry_air
                - oa.humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1e-9
        );
        // Supply dry-bulb near the 18 °C setpoint.
        assert!(
            (perf.supply_air.dry_bulb_c - 18.0).abs() < 0.5,
            "supply db {} should approach 18 °C",
            perf.supply_air.dry_bulb_c
        );
    }

    #[test]
    fn ventilation_passes_air_through_with_fan_heat_only() {
        let doas = test_doas();
        let oa = outdoor_air(18.0, 50.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert_eq!(perf.mode, DoasMode::Ventilation);
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert_eq!(perf.reheat_capacity_w, 0.0);
        assert!(perf.fan_heat_w > 0.0);
        // Supply slightly warmer than OA due to fan heat (within deadband of setpoint).
        assert!(
            perf.supply_air.dry_bulb_c >= oa.dry_bulb_c,
            "fan heat must not cool the air"
        );
    }

    // -----------------------------------------------------------------------
    // Off mode
    // -----------------------------------------------------------------------

    #[test]
    fn off_mode_delivers_no_flow_no_conditioning() {
        let doas = test_doas();
        let oa = outdoor_air(32.0, 60.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::off())
            .unwrap();

        assert_eq!(perf.mode, DoasMode::Off);
        assert_eq!(perf.volumetric_flow_m3_per_s, 0.0);
        assert_eq!(perf.dry_air_mass_flow_kg_per_s, 0.0);
        assert_eq!(perf.cooling_total_capacity_w, 0.0);
        assert_eq!(perf.reheat_capacity_w, 0.0);
        assert_eq!(perf.fan_shaft_power_w, 0.0);
        assert_eq!(perf.supply_air, oa);
    }

    // -----------------------------------------------------------------------
    // Fan heat
    // -----------------------------------------------------------------------

    #[test]
    fn fan_heat_raises_db_in_ventilation_mode() {
        let doas = test_doas();
        let oa = outdoor_air(18.0, 50.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        assert!(perf.fan_heat_w > 0.0);
        assert!(
            perf.supply_air.dry_bulb_c > oa.dry_bulb_c,
            "supply {} should exceed OA {} due to fan heat",
            perf.supply_air.dry_bulb_c,
            oa.dry_bulb_c
        );
    }

    // -----------------------------------------------------------------------
    // Energy balance (First Law across the DOAS)
    // -----------------------------------------------------------------------

    #[test]
    fn energy_balance_dehumidification_mode() {
        let doas = test_doas();
        let oa = outdoor_air(32.0, 60.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        // Net enthalpy change of the air stream must equal
        //   -cooling_total + reheat + fan_heat.
        let h_in = oa.enthalpy_kj_per_kg_dry_air;
        let h_out = perf.supply_air.enthalpy_kj_per_kg_dry_air;
        let expected = -perf.cooling_total_capacity_w + perf.reheat_capacity_w + perf.fan_heat_w;
        let actual = perf.dry_air_mass_flow_kg_per_s * (h_out - h_in) * 1000.0;

        assert!(
            (actual - expected).abs() / expected.abs().max(1.0) < 0.005,
            "energy balance: actual {actual} W vs expected {expected} W"
        );
    }

    #[test]
    fn energy_balance_heating_only_mode() {
        let doas = test_doas();
        let oa = outdoor_air(2.0, 30.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();

        let h_in = oa.enthalpy_kj_per_kg_dry_air;
        let h_out = perf.supply_air.enthalpy_kj_per_kg_dry_air;
        let expected = perf.reheat_capacity_w + perf.fan_heat_w; // no cooling
        let actual = perf.dry_air_mass_flow_kg_per_s * (h_out - h_in) * 1000.0;

        assert!(
            (actual - expected).abs() / expected.abs().max(1.0) < 0.005,
            "energy balance: actual {actual} W vs expected {expected} W"
        );
    }

    // -----------------------------------------------------------------------
    // Edge cases & validation
    // -----------------------------------------------------------------------

    #[test]
    fn negative_density_is_rejected() {
        let doas = test_doas();
        let oa = outdoor_air(32.0, 60.0);
        let err = doas
            .compute_doas_performance(&oa, -1.0, &DoasControl::active(10.0, 18.0))
            .unwrap_err();
        assert!(matches!(err, AirsideCouplingError::InvalidInput { .. }));
    }

    #[test]
    fn update_state_persists_mode() {
        let mut doas = test_doas();
        assert_eq!(doas.current_mode, DoasMode::Off);
        doas.update_state(DoasMode::CoolingDehumidification);
        assert_eq!(doas.current_mode, DoasMode::CoolingDehumidification);
        doas.update_state(DoasMode::Ventilation);
        assert_eq!(doas.current_mode, DoasMode::Ventilation);
    }

    // -----------------------------------------------------------------------
    // Composability with VAV: DOAS supply feeds a VAV entering-air argument
    // -----------------------------------------------------------------------

    #[test]
    fn doas_supply_can_feed_vav_entering_air() {
        use crate::sim::hvac::cooling_coil::CoolingCoil as VavCoolingCoil;
        use crate::sim::hvac::heating_coil::HeatingCoilComponent as VavReheat;
        use crate::sim::hvac::vav_terminal::{VavTerminal, VavTerminalControl, VavTerminalUnit};

        // 1. Run the DOAS to produce a neutral, dehumidified supply.
        let doas = test_doas();
        let oa = outdoor_air(32.0, 60.0);
        let rho_oa = oa.density_kg_per_m3;
        let doas_perf = doas
            .compute_doas_performance(&oa, rho_oa, &DoasControl::active(10.0, 18.0))
            .unwrap();
        assert_eq!(doas_perf.mode, DoasMode::CoolingDehumidification);
        // DOAS supply dew-point held at 10 °C, db near 18 °C.
        assert!((doas_perf.supply_dew_point_c - 10.0).abs() < 0.05);

        // 2. Feed the DOAS supply as the entering air to a VAV terminal. The
        //    VAV only needs to handle zone-sensible load; the latent load has
        //    already been removed by the DOAS. This must not error.
        let vav_cooling = VavCoolingCoil::new("VAV-CC".into(), 20_000.0, 0.85, 0.20, 12.0, 2.0);
        let vav_reheat = VavReheat::new("VAV-HC".into(), 8_000.0, 2.0);
        let vav = VavTerminalUnit::new("VAV-1".into(), 0, 2.0, vav_cooling, Some(vav_reheat));

        let rho_supply = doas_perf.supply_air.density_kg_per_m3;
        let vav_perf = vav
            .compute_terminal_performance(
                &doas_perf.supply_air,
                rho_supply,
                &VavTerminalControl::cooling(1.0),
            )
            .expect("DOAS supply must compose as VAV entering air");

        // The VAV sees already-dry air: its condensate removal should be small
        // (latent already handled by the DOAS).
        assert!(vav_perf.cooling_total_capacity_w >= 0.0);
        assert!(
            vav_perf.condensate_rate_kg_per_s < doas_perf.condensate_rate_kg_per_s,
            "VAV condensate {} should be less than DOAS condensate {} (latent decoupled)",
            vav_perf.condensate_rate_kg_per_s,
            doas_perf.condensate_rate_kg_per_s
        );
    }

    // -----------------------------------------------------------------------
    // Serde & Clone
    // -----------------------------------------------------------------------

    #[test]
    fn clone_and_serde_round_trip() {
        let doas = test_doas();
        let cloned = doas.clone();
        assert_eq!(doas.id, cloned.id);
        assert!((doas.rated_cooling_capacity_w() - cloned.rated_cooling_capacity_w()).abs() < 1e-9);

        let json = serde_json::to_string(&doas).expect("serialize");
        let back: DoasUnit = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(doas.id, back.id);
        assert!((doas.outdoor_air_flow_m3_per_s() - back.outdoor_air_flow_m3_per_s()).abs() < 1e-9);
        assert!(back.has_reheat());
    }

    // -----------------------------------------------------------------------
    // Trait-object dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn trait_object_dispatch() {
        let doas: Box<dyn Doas> = Box::new(test_doas());
        assert!((doas.outdoor_air_flow_m3_per_s() - 1.5).abs() < 1e-9);
        assert!(doas.has_reheat());

        let oa = outdoor_air(32.0, 60.0);
        let rho = oa.density_kg_per_m3;
        let perf = doas
            .compute_doas_performance(&oa, rho, &DoasControl::active(10.0, 18.0))
            .unwrap();
        assert!(perf.cooling_total_capacity_w > 0.0);
        assert!((perf.supply_dew_point_c - 10.0).abs() < 0.05);
    }
}
