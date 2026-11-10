//! Pressure-driven equation-based HVAC air loop — Issue #3984, **SLICE 1**.
//!
//! ## What this slice is
//!
//! A self-contained air-loop model in which **airflow is solved from a
//! pressure balance** instead of being *imposed* on the simulation:
//!
//! * [`FanCurve`] — a fan pressure–flow characteristic
//!   `ΔP(Q; φ) = φ²·P_static − c·Q²` (the quadratic fan curve with the
//!   affinity laws applied to the curve, so `Q ∝ φ` and `ΔP ∝ φ²` along a
//!   fixed operating point).
//! * [`AirLoopElement`] — the elements of the duct network: a fan
//!   ([`AirLoopElement::Fan`]), a duct ([`AirLoopElement::Duct`]) and a
//!   modulating damper ([`AirLoopElement::Damper`]). Every non-fan element
//!   is a **quadratic resistance**: `ΔP = R·Q·|Q|`.
//! * [`AirLoopNetwork`] — the closed duct loop (supply path + return path)
//!   and the **network solve**: the loop pressure balance
//!
//!   ```text
//!   F(Q) = ΔP_fan(Q; φ) − Σ_i R_i·Q² = 0
//!   ```
//!
//!   is solved for `Q ≥ 0` with a bracketed Newton iteration (the residual
//!   is continuous and strictly decreasing, so the operating point is
//!   unique). The node pressures are then *recovered* by walking the loop,
//!   and the walk is checked for closure and for node mass balance. The
//!   flow is therefore an output of the pressure balance — no flow constant
//!   is imposed anywhere in this module.
//! * [`ZoneThermostat`] — a thermostat with **real (non-ideal) control
//!   semantics**: a dead band with hysteresis, so the actuator state is
//!   *held* while the controlled variable is inside the band (dead-band
//!   dwell) and the controlled variable **overshoots** the setpoint instead
//!   of converging onto it.
//! * [`ZoneAirNode`] — a first-order zone-air node (backward Euler) used as
//!   the sink of the supply path, so the controller and the loop can be
//!   exercised as a closed dynamic system.
//! * [`PressureDrivenAirLoop`] — the composition of the three: thermostat →
//!   damper position → **pressure-driven network solve** → supply flow →
//!   zone air node.
//! * [`MoistAirStream`] / [`OutdoorAirMixer`] — the **outdoor-air and
//!   mixing capability**: a mixing box combines the outdoor-air stream with
//!   the return stream of the loop (on a dry-air mass basis) and forms the
//!   mixed supply state — temperature *and* humidity ratio — that the supply
//!   path and the zone air node consume.
//! * **Humidity state** — the zone air node carries a humidity ratio and a
//!   latent internal gain, and the loop step integrates the zone moisture
//!   balance (backward Euler, the humidity counterpart of the thermal
//!   balance) so that the latent duty of a dehumidifying coil is a
//!   representable quantity.
//!
//! ## What this slice is **not**
//!
//! * **It is not wired as the default production HVAC path.** The existing
//!   sequential, imposed-flow airside path (`src/sim/hvac/airside_coupling.rs`,
//!   `src/sim/hvac/ideal_loads.rs`) is untouched and remains the production
//!   path. ADR-0017 D5 sequences the HVAC migration behind the teacher
//!   validation suite (#3986); this slice only provides the pressure-driven
//!   core that the teacher will consume.
//! * **It is not part of the DAE residual assembly.** `ThermalModelTrait`
//!   exposes `hvac_power_demand(&mut self, …)` and has no component slot for
//!   an air loop, so there is no clean slot to plug into yet; the module is
//!   deliberately standalone rather than sprawling into the solver.
//! * **Plant / condenser loops are out of scope** (they are water-side, not
//!   air-side, and belong to `src/sim/hvac/plant/`).
//! * **Equipment-catalogue breadth is explicitly out of scope** per the
//!   issue: this slice models *one* fan, *one* duct, *one* damper and *one*
//!   zone. No VAV/CAV/DOAS catalogue, no part-load curves, no coil
//!   performance maps.
//! * **No ASHRAE 140 scoring and no parameter tuning** (RULES.md): the
//!   numbers in the tests are physically-scaled design values, not constants
//!   fitted to a reference case.
//!
//! ## Non-ideal control behaviour (what the tests assert)
//!
//! The thermostat is an on/off device with a dead band of width `b` centred
//! on the cooling setpoint `T_sp`:
//!
//! * it engages cooling when `T > T_sp + b/2` and releases it when
//!   `T < T_sp − b/2` — the classic hysteretic thermostat;
//! * **inside the band the state is held** (dead-band dwell): no actuation
//!   happens while the controlled variable is inside the band;
//! * because the switch can only happen at a discrete timestep and the zone
//!   has thermal mass, the cooling action **overshoots past the setpoint**
//!   (the zone is driven *below* `T_sp` before the coil can release) and the
//!   idle action overshoots above it. The controlled variable therefore
//!   **cycles instead of converging** — the non-ideal behaviour that the
//!   idealised "read a setpoint, return a load" request logic cannot
//!   produce.
//!
//! ## Physics references
//!
//! * Fan curve and affinity laws: ASHRAE Handbook—Fundamentals, Ch. 21
//!   (Fans, Air Performance); Kreider et al., *Heating and Cooling of
//!   Buildings*.
//! * Quadratic duct resistance (`ΔP ∝ Q²`): ASHRAE Handbook—Fundamentals,
//!   Ch. 21 (duct fitting).
//! * Damper flow/position characteristic (`R_eff = R_open / position²`, i.e.
//!   the flow through a fixed pressure drop is proportional to the damper
//!   position): ASHRAE Handbook—Fundamentals, Ch. 21 (dampers).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::fan::STANDARD_AIR_DENSITY_KG_PER_M3;

/// Dry-air specific heat at constant pressure [J/(kg·K)].
///
/// Standard value used by the airside coupling layer for dry-air energy
/// balances.
pub const DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K: f64 = 1005.0;

/// Specific heat of dry air at constant pressure [kJ/(kg·K)].
///
/// ASHRAE Handbook—Fundamentals (2021), Ch. 1 (Psychrometrics). The
/// psychrometric value (1.006 kJ/(kg·K)) differs slightly from the
/// dry-air energy-balance value used by the thermal path
/// ([`DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K`]); the psychrometric one is used
/// only for the moist-air enthalpy of the mixing balance.
pub const SPECIFIC_HEAT_DRY_AIR_KJ_PER_KG_K: f64 = 1.006;

/// Specific heat of water vapour at constant pressure [kJ/(kg·K)].
///
/// ASHRAE Handbook—Fundamentals (2021), Ch. 1 (Psychrometrics).
pub const SPECIFIC_HEAT_WATER_VAPOR_KJ_PER_KG_K: f64 = 1.86;

/// Latent heat of vaporization of water at 0 °C [kJ/kg].
///
/// ASHRAE Handbook—Fundamentals (2021), Ch. 1 (Psychrometrics).
pub const LATENT_HEAT_VAPORIZATION_0C_KJ_PER_KG: f64 = 2501.0;

/// Latent heat of vaporization of water at 0 °C [J/kg].
///
/// The same latent heat as [`LATENT_HEAT_VAPORIZATION_0C_KJ_PER_KG`], in the
/// unit system of the zone energy balance; it converts a latent heat load
/// [W] into a water-vapour mass flow [kg/s].
pub const LATENT_HEAT_VAPORIZATION_0C_J_PER_KG: f64 = 2_501_000.0;

/// Absolute tolerance on the loop pressure-balance residual [Pa].
///
/// The Newton iteration on `F(Q)` converges to machine precision on the
/// reference configurations; the acceptance tolerance is deliberately tight
/// so a regression in the solve cannot hide behind a loose bound.
pub const LOOP_RESIDUAL_TOLERANCE_PA: f64 = 1.0e-9;

/// Maximum number of Newton iterations used by the loop solve.
const MAX_SOLVE_ITERATIONS: usize = 200;

/// Default (total) fan efficiency, `shaft power / air power` [-].
///
/// The default of 1.0 makes the shaft power equal to the air power, so the
/// default **preserves the behaviour of the existing API**: the air power
/// reported by [`AirLoopSolution::fan_air_power_w`] is unchanged, and the
/// shaft power and the fan temperature rise derived from it are the air
/// power and the air temperature rise of the pre-existing model.
pub const DEFAULT_FAN_EFFICIENCY: f64 = 1.0;

/// Serde default for [`FanCurve::efficiency`].
fn default_fan_efficiency() -> f64 {
    DEFAULT_FAN_EFFICIENCY
}

/// Air density used to convert a volumetric flow into a mass flow
/// [kg/m³].
///
/// `moist_air_density_kg_per_m3` is the density of the humid-air state at
/// the point of interest **when a humidity state is available**. The
/// humidity model itself belongs to split A (issue #4705) and is *not*
/// part of this module, so the only thing this helper needs from a
/// humidity state is its density; the **standard dry-air density is the
/// fallback** and is what runs when no humidity state is supplied (i.e. on
/// this base).
///
/// A non-finite or non-positive density is treated as "no humidity state
/// available" and falls back to the standard density.
pub fn air_density_kg_per_m3(moist_air_density_kg_per_m3: Option<f64>) -> f64 {
    match moist_air_density_kg_per_m3 {
        Some(density) if density.is_finite() && density > 0.0 => density,
        _ => STANDARD_AIR_DENSITY_KG_PER_M3,
    }
}

/// Relative bracket width at which the safeguarded Newton iteration stops.
const BRACKET_RELATIVE_TOLERANCE: f64 = 1.0e-13;

/// Errors raised by the pressure-driven air loop.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AirLoopError {
    /// A construction-time parameter was outside its admissible range.
    #[error("air-loop parameter `{field}` must be {requirement}, got {value}")]
    InvalidParameter {
        /// Name of the offending parameter.
        field: &'static str,
        /// Human-readable admissibility requirement.
        requirement: &'static str,
        /// The rejected value.
        value: f64,
    },
    /// The duct network is not a usable closed loop.
    #[error("air-loop network topology is invalid: {reason}")]
    InvalidTopology {
        /// Why the topology was rejected.
        reason: &'static str,
    },
    /// The loop pressure balance could not be solved to the required tolerance.
    #[error("air-loop solve did not converge: residual {residual_pa:.6e} Pa after {iterations} iterations")]
    ConvergenceFailure {
        /// Residual of the loop pressure balance at the last iterate [Pa].
        residual_pa: f64,
        /// Number of iterations performed.
        iterations: usize,
    },
    /// A derived solution quantity was not finite.
    #[error("air-loop solution field `{field}` is not finite")]
    NonFiniteResult {
        /// Name of the offending field.
        field: &'static str,
    },
}

/// Quadratic fan pressure–flow characteristic.
///
/// The full-speed curve is the parabola `ΔP(Q) = P_static − c·Q²`, where
/// `P_static` is the shut-off pressure and `c` the curve coefficient
/// [Pa·s²/m⁶]. The affinity laws are applied *to the curve* (ASHRAE
/// Handbook—Fundamentals, Ch. 21):
///
/// ```text
/// ΔP(Q; φ) = φ² · ΔP(Q/φ) = φ² · P_static − c · Q²
/// ```
///
/// so at a fixed system the flow follows fan law 1 (`Q ∝ φ`) and the
/// pressure rise fan law 2 (`ΔP ∝ φ²`), while the *shape* of the curve
/// (the coefficient `c`) is speed-independent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FanCurve {
    /// Shut-off (static) pressure at full speed [Pa].
    pub static_pressure_pa: f64,
    /// Curve coefficient `c` in `ΔP = static_pressure − c·Q²` [Pa·s²/m⁶].
    pub curve_coefficient_pa_s2_per_m6: f64,
    /// Total fan efficiency, `shaft power / air power` [-].
    ///
    /// The default is [`DEFAULT_FAN_EFFICIENCY`] (1.0), which makes the
    /// shaft power identical to the air power and therefore leaves every
    /// pre-existing result unchanged. A realistic value (e.g. 0.70) is what
    /// the shaft-power and fan-temperature-rise reporting uses.
    #[serde(default = "default_fan_efficiency")]
    pub efficiency: f64,
}

impl FanCurve {
    /// Construct a fan curve from its shut-off pressure and curve coefficient.
    pub fn new(
        static_pressure_pa: f64,
        curve_coefficient_pa_s2_per_m6: f64,
    ) -> Result<Self, AirLoopError> {
        if !static_pressure_pa.is_finite() || static_pressure_pa <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "static_pressure_pa",
                requirement: "finite and > 0",
                value: static_pressure_pa,
            });
        }
        if !curve_coefficient_pa_s2_per_m6.is_finite() || curve_coefficient_pa_s2_per_m6 <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "curve_coefficient_pa_s2_per_m6",
                requirement: "finite and > 0",
                value: curve_coefficient_pa_s2_per_m6,
            });
        }
        Ok(Self {
            static_pressure_pa,
            curve_coefficient_pa_s2_per_m6,
            efficiency: DEFAULT_FAN_EFFICIENCY,
        })
    }

    /// Construct a fan curve with an explicit fan efficiency.
    ///
    /// The efficiency is the total fan efficiency, `shaft power / air
    /// power` [-]; it must lie in `(0, 1]`.
    pub fn with_efficiency(
        static_pressure_pa: f64,
        curve_coefficient_pa_s2_per_m6: f64,
        efficiency: f64,
    ) -> Result<Self, AirLoopError> {
        let mut curve = Self::new(static_pressure_pa, curve_coefficient_pa_s2_per_m6)?;
        curve.set_efficiency(efficiency)?;
        Ok(curve)
    }

    /// Set the fan efficiency, `shaft power / air power` [-].
    pub fn set_efficiency(&mut self, efficiency: f64) -> Result<(), AirLoopError> {
        if !efficiency.is_finite() || efficiency <= 0.0 || efficiency > 1.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "efficiency",
                requirement: "finite and within (0, 1]",
                value: efficiency,
            });
        }
        self.efficiency = efficiency;
        Ok(())
    }

    /// The fan efficiency, `shaft power / air power` [-].
    pub fn efficiency(&self) -> f64 {
        self.efficiency
    }

    /// Shaft power drawn by the fan for a given air power [W].
    ///
    /// `shaft power = air power / efficiency`. With the default efficiency
    /// of 1.0 the shaft power equals the air power, which is the
    /// pre-existing behaviour.
    pub fn shaft_power_w(&self, air_power_w: f64) -> f64 {
        if !air_power_w.is_finite() {
            return f64::NAN;
        }
        air_power_w / self.efficiency
    }

    /// Fan air temperature rise implied by a shaft power [K].
    ///
    /// `ΔT = shaft power / (ṁ·c_p)`. The mass flow must be positive; a
    /// non-positive mass flow carries no energy and reports a zero rise.
    pub fn temperature_rise_k(&self, shaft_power_w: f64, mass_flow_kg_per_s: f64) -> f64 {
        if !shaft_power_w.is_finite() || !mass_flow_kg_per_s.is_finite() {
            return f64::NAN;
        }
        if mass_flow_kg_per_s <= 0.0 {
            return 0.0;
        }
        shaft_power_w / (mass_flow_kg_per_s * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K)
    }

    /// Fan air temperature rise implied by an air power [K].
    ///
    /// Convenience wrapper: the air power is first converted to the shaft
    /// power with [`FanCurve::shaft_power_w`], then the rise is computed
    /// from the shaft power.
    pub fn air_temperature_rise_k(&self, air_power_w: f64, mass_flow_kg_per_s: f64) -> f64 {
        self.temperature_rise_k(self.shaft_power_w(air_power_w), mass_flow_kg_per_s)
    }

    /// Construct a fan curve from a rated operating point at full speed.
    ///
    /// The curve passes through `(rated_flow_m3_per_s, rated_pressure_rise_pa)`
    /// and reaches `static_pressure_pa` at zero flow, i.e.
    /// `c = (P_static − ΔP_rated) / Q_rated²`.
    pub fn from_rated_point(
        static_pressure_pa: f64,
        rated_flow_m3_per_s: f64,
        rated_pressure_rise_pa: f64,
    ) -> Result<Self, AirLoopError> {
        if !rated_flow_m3_per_s.is_finite() || rated_flow_m3_per_s <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "rated_flow_m3_per_s",
                requirement: "finite and > 0",
                value: rated_flow_m3_per_s,
            });
        }
        if !rated_pressure_rise_pa.is_finite() || rated_pressure_rise_pa < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "rated_pressure_rise_pa",
                requirement: "finite and >= 0",
                value: rated_pressure_rise_pa,
            });
        }
        let coefficient = (static_pressure_pa - rated_pressure_rise_pa)
            / (rated_flow_m3_per_s * rated_flow_m3_per_s);
        Self::new(static_pressure_pa, coefficient)
    }

    /// Pressure rise produced by the fan at `flow_m3_per_s` and `speed_fraction`.
    ///
    /// `speed_fraction` is the fan speed relative to design speed, clamped to
    /// `[0, 1]`. A stopped fan (`speed_fraction == 0`) produces no rise.
    ///
    /// The returned value is the *available* rise; it becomes zero at the
    /// zero-pressure flow [`FanCurve::zero_pressure_flow_m3_per_s`], beyond
    /// which the flow is not sustainable by this fan.
    pub fn pressure_rise_pa(&self, flow_m3_per_s: f64, speed_fraction: f64) -> f64 {
        let phi = speed_fraction.clamp(0.0, 1.0);
        if phi <= 0.0 {
            return 0.0;
        }
        let phi2 = phi * phi;
        phi2 * self.static_pressure_pa
            - self.curve_coefficient_pa_s2_per_m6 * flow_m3_per_s * flow_m3_per_s
    }

    /// Flow at which the fan can no longer produce any pressure rise [m³/s].
    pub fn zero_pressure_flow_m3_per_s(&self, speed_fraction: f64) -> f64 {
        let phi = speed_fraction.clamp(0.0, 1.0);
        if phi <= 0.0 || self.curve_coefficient_pa_s2_per_m6 <= 0.0 {
            return 0.0;
        }
        phi * (self.static_pressure_pa / self.curve_coefficient_pa_s2_per_m6).sqrt()
    }

    /// Inverse of [`FanCurve::pressure_rise_pa`]: the flow that produces
    /// `pressure_rise_pa` at `speed_fraction` [m³/s].
    ///
    /// Returns `0.0` when the requested rise exceeds the fan's shut-off
    /// pressure (the fan cannot push that hard and no flow is driven).
    pub fn flow_for_pressure_rise_m3_per_s(
        &self,
        pressure_rise_pa: f64,
        speed_fraction: f64,
    ) -> f64 {
        let phi = speed_fraction.clamp(0.0, 1.0);
        if phi <= 0.0 {
            return 0.0;
        }
        let head = phi * phi * self.static_pressure_pa;
        if pressure_rise_pa >= head {
            return 0.0;
        }
        let coefficient = self.curve_coefficient_pa_s2_per_m6;
        if coefficient <= 0.0 {
            return 0.0;
        }
        ((head - pressure_rise_pa) / coefficient).sqrt()
    }
}

/// A moist-air stream: dry-bulb temperature plus humidity ratio.
///
/// The stream state is the minimum the air loop needs to carry the humidity
/// of the airstream: the dry-bulb temperature [°C] and the humidity ratio
/// `W` (mass of water vapour per mass of dry air) [kg/kg]. The
/// psychrometric enthalpy of the stream follows from the two
/// (ASHRAE Handbook—Fundamentals, 2021, Ch. 1):
///
/// ```text
/// h = c_pa·T + W·(h_fg,0 + c_pv·T)      [kJ/kg dry air]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoistAirStream {
    /// Dry-bulb temperature [°C].
    pub temperature_c: f64,
    /// Humidity ratio [kg water vapour/kg dry air].
    pub humidity_ratio_kg_per_kg_dry_air: f64,
}

impl MoistAirStream {
    /// Construct a moist-air stream.
    ///
    /// Non-finite temperatures and negative humidity ratios are rejected.
    pub fn new(
        temperature_c: f64,
        humidity_ratio_kg_per_kg_dry_air: f64,
    ) -> Result<Self, AirLoopError> {
        if !temperature_c.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "temperature_c",
                requirement: "finite",
                value: temperature_c,
            });
        }
        if !humidity_ratio_kg_per_kg_dry_air.is_finite() || humidity_ratio_kg_per_kg_dry_air < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "humidity_ratio_kg_per_kg_dry_air",
                requirement: "finite and >= 0",
                value: humidity_ratio_kg_per_kg_dry_air,
            });
        }
        Ok(Self {
            temperature_c,
            humidity_ratio_kg_per_kg_dry_air,
        })
    }

    /// Specific enthalpy of the stream [kJ/kg dry air].
    pub fn enthalpy_kj_per_kg_dry_air(&self) -> f64 {
        SPECIFIC_HEAT_DRY_AIR_KJ_PER_KG_K * self.temperature_c
            + self.humidity_ratio_kg_per_kg_dry_air
                * (LATENT_HEAT_VAPORIZATION_0C_KJ_PER_KG
                    + SPECIFIC_HEAT_WATER_VAPOR_KJ_PER_KG_K * self.temperature_c)
    }

    /// Mix two streams on a dry-air mass basis.
    ///
    /// `outdoor_air_fraction` is the fraction of the mixed dry-air mass
    /// flow drawn from `outdoor_air`; the remainder comes from
    /// `return_air`. The moisture balance is exact
    /// (`W_mix = f·W_oa + (1 − f)·W_ret`) and the enthalpy balance
    /// (`h_mix = f·h_oa + (1 − f)·h_ret`) is closed on the mixed humidity
    /// ratio:
    ///
    /// ```text
    /// T_mix = (h_mix − h_fg,0·W_mix) / (c_pa + c_pv·W_mix)
    /// ```
    pub fn mix(
        outdoor_air: &Self,
        return_air: &Self,
        outdoor_air_fraction: f64,
    ) -> Result<Self, AirLoopError> {
        if !outdoor_air_fraction.is_finite() || !(0.0..=1.0).contains(&outdoor_air_fraction) {
            return Err(AirLoopError::InvalidParameter {
                field: "outdoor_air_fraction",
                requirement: "finite and within [0, 1]",
                value: outdoor_air_fraction,
            });
        }
        let fraction = outdoor_air_fraction;
        let humidity_ratio = fraction * outdoor_air.humidity_ratio_kg_per_kg_dry_air
            + (1.0 - fraction) * return_air.humidity_ratio_kg_per_kg_dry_air;
        let enthalpy = fraction * outdoor_air.enthalpy_kj_per_kg_dry_air()
            + (1.0 - fraction) * return_air.enthalpy_kj_per_kg_dry_air();
        let temperature = (enthalpy - LATENT_HEAT_VAPORIZATION_0C_KJ_PER_KG * humidity_ratio)
            / (SPECIFIC_HEAT_DRY_AIR_KJ_PER_KG_K
                + SPECIFIC_HEAT_WATER_VAPOR_KJ_PER_KG_K * humidity_ratio);
        Self::new(temperature, humidity_ratio)
    }

    /// Mix two streams given their volumetric flow rates [m³/s].
    ///
    /// The dry-air mass-flow split is taken from the volumetric split — the
    /// same constant-density approximation the zone air node uses for its
    /// dry-air mass — so the result is the fraction-based mixture of the two
    /// streams. Flows must be finite and non-negative; if both flows are
    /// zero the mixture is the return stream.
    pub fn mix_by_flow(
        outdoor_air: &Self,
        return_air: &Self,
        outdoor_air_flow_m3_per_s: f64,
        return_air_flow_m3_per_s: f64,
    ) -> Result<Self, AirLoopError> {
        for (field, value) in [
            ("outdoor_air_flow_m3_per_s", outdoor_air_flow_m3_per_s),
            ("return_air_flow_m3_per_s", return_air_flow_m3_per_s),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(AirLoopError::InvalidParameter {
                    field,
                    requirement: "finite and >= 0",
                    value,
                });
            }
        }
        let total_flow = outdoor_air_flow_m3_per_s + return_air_flow_m3_per_s;
        let fraction = if total_flow > 0.0 {
            outdoor_air_flow_m3_per_s / total_flow
        } else {
            0.0
        };
        Self::mix(outdoor_air, return_air, fraction)
    }
}

/// An outdoor-air mixing box: the outdoor-air inlet of the air path.
///
/// The mixer combines the outdoor-air stream with the return stream of the
/// loop and produces the **mixed supply air** — temperature *and* humidity
/// ratio — that the supply path and the zone air node consume. It is
/// attached to the loop with
/// [`AirLoopNetwork::with_outdoor_air_mixer`]; without it the loop keeps
/// its historical behaviour (the supply stream is the return stream).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutdoorAirMixer {
    /// Fraction of the mixed supply stream drawn from the outdoor air
    /// [0..1].
    pub outdoor_air_fraction: f64,
    /// State of the outdoor-air stream.
    pub outdoor_air: MoistAirStream,
}

impl OutdoorAirMixer {
    /// Construct an outdoor-air mixing box.
    ///
    /// The outdoor-air fraction must be finite and within `[0, 1]`.
    pub fn new(
        outdoor_air: MoistAirStream,
        outdoor_air_fraction: f64,
    ) -> Result<Self, AirLoopError> {
        if !outdoor_air_fraction.is_finite() || !(0.0..=1.0).contains(&outdoor_air_fraction) {
            return Err(AirLoopError::InvalidParameter {
                field: "outdoor_air_fraction",
                requirement: "finite and within [0, 1]",
                value: outdoor_air_fraction,
            });
        }
        Ok(Self {
            outdoor_air_fraction,
            outdoor_air,
        })
    }

    /// Mix the outdoor air with the return stream.
    ///
    /// The result is the mixed supply state (temperature and humidity
    /// ratio) of the mixing box.
    pub fn mix(&self, return_air: &MoistAirStream) -> Result<MoistAirStream, AirLoopError> {
        MoistAirStream::mix(&self.outdoor_air, return_air, self.outdoor_air_fraction)
    }
}

/// One element of the duct network.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AirLoopElement {
    /// The fan: the pressure source of the loop.
    Fan {
        /// Element identifier.
        id: String,
        /// Fan pressure–flow characteristic.
        curve: FanCurve,
        /// Fan speed relative to design speed, clamped to `[0, 1]`.
        speed_fraction: f64,
    },
    /// A duct run: a fixed quadratic resistance.
    Duct {
        /// Element identifier.
        id: String,
        /// Quadratic resistance coefficient [Pa·s²/m⁶].
        resistance_pa_s2_per_m6: f64,
    },
    /// A modulating damper: a quadratic resistance whose coefficient depends
    /// on the damper position.
    ///
    /// The effective resistance is `R_open / position²`, i.e. at a fixed
    /// pressure drop the flow through the damper is proportional to its
    /// position (ASHRAE Handbook—Fundamentals, Ch. 21). A fully shut damper
    /// (`position == 0`) blocks the path.
    Damper {
        /// Element identifier.
        id: String,
        /// Quadratic resistance coefficient at full open [Pa·s²/m⁶].
        open_resistance_pa_s2_per_m6: f64,
        /// Damper position, `0` (shut) to `1` (fully open).
        position: f64,
    },
}

impl AirLoopElement {
    /// Identifier of the element.
    pub fn id(&self) -> &str {
        match self {
            AirLoopElement::Fan { id, .. }
            | AirLoopElement::Duct { id, .. }
            | AirLoopElement::Damper { id, .. } => id,
        }
    }

    /// Effective quadratic resistance of the element [Pa·s²/m⁶].
    ///
    /// The fan contributes no series loss (its pressure rise is handled
    /// separately); a shut damper reports an infinite resistance.
    pub fn resistance_pa_s2_per_m6(&self) -> f64 {
        match self {
            AirLoopElement::Fan { .. } => 0.0,
            AirLoopElement::Duct {
                resistance_pa_s2_per_m6,
                ..
            } => *resistance_pa_s2_per_m6,
            AirLoopElement::Damper {
                open_resistance_pa_s2_per_m6,
                position,
                ..
            } => damper_resistance(*open_resistance_pa_s2_per_m6, *position),
        }
    }

    /// Pressure change across the element for a path flow of `flow_m3_per_s`.
    ///
    /// Positive values are a pressure *rise* (the fan), negative values are
    /// a pressure *drop* (the quadratic resistances). The flow is taken
    /// positive along the path direction.
    pub fn pressure_change_pa(&self, flow_m3_per_s: f64) -> f64 {
        match self {
            AirLoopElement::Fan {
                curve,
                speed_fraction,
                ..
            } => curve.pressure_rise_pa(flow_m3_per_s, *speed_fraction),
            AirLoopElement::Duct { .. } | AirLoopElement::Damper { .. } => {
                let resistance = self.resistance_pa_s2_per_m6();
                if !resistance.is_finite() {
                    // A shut damper carries no flow, hence no pressure drop.
                    return 0.0;
                }
                -resistance * flow_m3_per_s * flow_m3_per_s
            }
        }
    }

    /// Flow implied by the pressure change across the element [m³/s].
    ///
    /// This is the inverse of [`AirLoopElement::pressure_change_pa`] and is
    /// used to check that the recovered node pressures are consistent with
    /// the solved flow.
    pub fn implied_flow_m3_per_s(&self, pressure_change_pa: f64) -> f64 {
        match self {
            AirLoopElement::Fan {
                curve,
                speed_fraction,
                ..
            } => curve.flow_for_pressure_rise_m3_per_s(pressure_change_pa, *speed_fraction),
            AirLoopElement::Duct { .. } | AirLoopElement::Damper { .. } => {
                let resistance = self.resistance_pa_s2_per_m6();
                if !resistance_is_usable(&resistance) {
                    return 0.0;
                }
                let magnitude = (pressure_change_pa.abs() / resistance).sqrt();
                // A resistance drops the pressure along the flow direction.
                if pressure_change_pa < 0.0 {
                    magnitude
                } else {
                    -magnitude
                }
            }
        }
    }
}

/// `true` when a resistance coefficient can carry flow.
fn resistance_is_usable(resistance: &f64) -> bool {
    resistance.is_finite() && *resistance > 0.0
}

/// Effective resistance of a damper at `position` [Pa·s²/m⁶].
///
/// `R_eff = R_open / position²`; a shut damper (`position <= 0`) reports an
/// infinite resistance.
fn damper_resistance(open_resistance_pa_s2_per_m6: f64, position: f64) -> f64 {
    if position <= 0.0 {
        return f64::INFINITY;
    }
    let position = position.min(1.0);
    open_resistance_pa_s2_per_m6 / (position * position)
}

/// The solved state of the air loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AirLoopSolution {
    /// The solved path flow [m³/s], positive in the supply direction.
    pub flow_m3_per_s: f64,
    /// Gauge pressure at each node [Pa]; node 0 is the reference node.
    pub node_pressures_pa: Vec<f64>,
    /// Flow through each element, implied by the recovered pressures [m³/s].
    pub element_flows_m3_per_s: Vec<f64>,
    /// Pressure change across each element [Pa] (rise for the fan).
    pub element_pressure_changes_pa: Vec<f64>,
    /// Pressure rise produced by the fan at the operating point [Pa].
    pub fan_pressure_rise_pa: f64,
    /// Air power delivered to the airstream by the fan [W].
    pub fan_air_power_w: f64,
    /// Shaft power drawn by the fan [W].
    ///
    /// `shaft power = air power / fan efficiency`. With the default
    /// efficiency of 1.0 this equals [`AirLoopSolution::fan_air_power_w`].
    #[serde(default)]
    pub fan_shaft_power_w: f64,
    /// Fan air temperature rise implied by the shaft power [K].
    ///
    /// `ΔT = shaft power / (ṁ·c_p)`, with the mass flow evaluated at the
    /// standard air density (no humidity state is available on this path).
    #[serde(default)]
    pub fan_temperature_rise_k: f64,
    /// Residual of the loop pressure balance at the operating point [Pa].
    pub loop_residual_pa: f64,
    /// Residual of the pressure walk around the closed loop [Pa].
    pub pressure_closure_residual_pa: f64,
    /// Largest node mass-balance residual `|inflow − outflow|` [m³/s].
    pub node_mass_balance_residual_m3_per_s: f64,
    /// Humidity ratio of the air delivered to the zone supply path
    /// [kg water vapour/kg dry air].
    ///
    /// `0.0` when the loop was solved without a return-stream humidity
    /// ratio ([`AirLoopNetwork::solve`]). When a return humidity ratio is
    /// supplied ([`AirLoopNetwork::solve_with_return_humidity_ratio`]) the
    /// value is the mixed supply humidity ratio if an outdoor-air mixing
    /// box is configured, and the return humidity ratio otherwise.
    #[serde(default)]
    pub supply_humidity_ratio_kg_per_kg_dry_air: f64,
}

impl AirLoopSolution {
    /// `true` when every reported quantity is finite.
    pub fn is_finite(&self) -> bool {
        self.flow_m3_per_s.is_finite()
            && self.fan_pressure_rise_pa.is_finite()
            && self.fan_air_power_w.is_finite()
            && self.fan_shaft_power_w.is_finite()
            && self.fan_temperature_rise_k.is_finite()
            && self.loop_residual_pa.is_finite()
            && self.pressure_closure_residual_pa.is_finite()
            && self.node_mass_balance_residual_m3_per_s.is_finite()
            && self.supply_humidity_ratio_kg_per_kg_dry_air.is_finite()
            && self.node_pressures_pa.iter().all(|p| p.is_finite())
            && self.element_flows_m3_per_s.iter().all(|q| q.is_finite())
            && self
                .element_pressure_changes_pa
                .iter()
                .all(|dp| dp.is_finite())
    }
}

/// The solved state of a loop driven in **specified-flow** mode.
///
/// In this mode the design mass flow and the fan pressure rise are *given*
/// (the QAS shape) instead of being solved from a fitted duct resistance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecifiedFlowSolution {
    /// The prescribed design mass flow [kg/s].
    pub mass_flow_kg_per_s: f64,
    /// The design volumetric flow at the reporting density [m³/s].
    pub volumetric_flow_m3_per_s: f64,
    /// The density used for the mass/volumetric conversion [kg/m³].
    pub density_kg_per_m3: f64,
    /// The prescribed fan pressure rise [Pa].
    pub pressure_rise_pa: f64,
    /// Air power delivered to the airstream by the fan [W].
    pub fan_air_power_w: f64,
    /// Shaft power drawn by the fan [W].
    pub fan_shaft_power_w: f64,
    /// Fan air temperature rise implied by the shaft power [K].
    pub fan_temperature_rise_k: f64,
}

impl SpecifiedFlowSolution {
    /// `true` when every reported quantity is finite.
    pub fn is_finite(&self) -> bool {
        self.mass_flow_kg_per_s.is_finite()
            && self.volumetric_flow_m3_per_s.is_finite()
            && self.density_kg_per_m3.is_finite()
            && self.pressure_rise_pa.is_finite()
            && self.fan_air_power_w.is_finite()
            && self.fan_shaft_power_w.is_finite()
            && self.fan_temperature_rise_k.is_finite()
    }
}

/// A loop driven in **specified-flow** mode.
///
/// The design mass flow and the fan pressure rise are prescribed; **no duct
/// resistance is fitted and none is needed**, because the operating point is
/// given rather than solved. The mode reports the powers and the fan
/// temperature rise implied by the specified operating point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpecifiedFlowLoop {
    fan_curve: FanCurve,
    mass_flow_kg_per_s: f64,
    pressure_rise_pa: f64,
}

impl SpecifiedFlowLoop {
    /// Build a specified-flow loop from a fan curve and the design point.
    ///
    /// `mass_flow_kg_per_s` must be finite and positive and
    /// `pressure_rise_pa` must be finite and non-negative.
    pub fn new(
        fan_curve: FanCurve,
        mass_flow_kg_per_s: f64,
        pressure_rise_pa: f64,
    ) -> Result<Self, AirLoopError> {
        if !mass_flow_kg_per_s.is_finite() || mass_flow_kg_per_s <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "mass_flow_kg_per_s",
                requirement: "finite and > 0",
                value: mass_flow_kg_per_s,
            });
        }
        if !pressure_rise_pa.is_finite() || pressure_rise_pa < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "pressure_rise_pa",
                requirement: "finite and >= 0",
                value: pressure_rise_pa,
            });
        }
        Ok(Self {
            fan_curve,
            mass_flow_kg_per_s,
            pressure_rise_pa,
        })
    }

    /// The fan curve of the loop.
    pub fn fan_curve(&self) -> &FanCurve {
        &self.fan_curve
    }

    /// The prescribed design mass flow [kg/s].
    pub fn mass_flow_kg_per_s(&self) -> f64 {
        self.mass_flow_kg_per_s
    }

    /// The prescribed fan pressure rise [Pa].
    pub fn pressure_rise_pa(&self) -> f64 {
        self.pressure_rise_pa
    }

    /// Solve the specified-flow operating point.
    ///
    /// `moist_air_density_kg_per_m3` is the density of the humid-air state
    /// at the point of interest when one is available; the standard
    /// dry-air density is the fallback (see [`air_density_kg_per_m3`]).
    ///
    /// The prescribed mass flow is reproduced exactly: it is an input of the
    /// mode, not a solved quantity.
    pub fn solve(
        &self,
        moist_air_density_kg_per_m3: Option<f64>,
    ) -> Result<SpecifiedFlowSolution, AirLoopError> {
        let density = air_density_kg_per_m3(moist_air_density_kg_per_m3);
        let volumetric_flow_m3_per_s = self.mass_flow_kg_per_s / density;
        let fan_air_power_w = volumetric_flow_m3_per_s * self.pressure_rise_pa;
        let fan_shaft_power_w = self.fan_curve.shaft_power_w(fan_air_power_w);
        let fan_temperature_rise_k = self
            .fan_curve
            .temperature_rise_k(fan_shaft_power_w, self.mass_flow_kg_per_s);

        let solution = SpecifiedFlowSolution {
            mass_flow_kg_per_s: self.mass_flow_kg_per_s,
            volumetric_flow_m3_per_s,
            density_kg_per_m3: density,
            pressure_rise_pa: self.pressure_rise_pa,
            fan_air_power_w,
            fan_shaft_power_w,
            fan_temperature_rise_k,
        };
        if !solution.is_finite() {
            return Err(AirLoopError::NonFiniteResult {
                field: "specified_flow_solution",
            });
        }
        Ok(solution)
    }
}

/// A closed duct loop whose airflow is solved from the pressure balance.
///
/// The loop is described as an ordered list of nodes and an ordered list of
/// elements; element `i` connects node `i` to node `(i + 1) mod n`, i.e. the
/// network is a ring: the supply path (fan → duct → damper) followed by the
/// return path back to the reference node. Node 0 is the reference node
/// (ambient) and its gauge pressure is fixed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AirLoopNetwork {
    nodes: Vec<String>,
    elements: Vec<AirLoopElement>,
    reference_pressure_pa: f64,
    outdoor_air_mixer: Option<OutdoorAirMixer>,
}

impl AirLoopNetwork {
    /// Build a closed loop from its nodes and elements.
    ///
    /// The node and element counts must match (a ring) and the loop must
    /// contain exactly one fan, which is the pressure source of the loop.
    pub fn new(nodes: Vec<String>, elements: Vec<AirLoopElement>) -> Result<Self, AirLoopError> {
        Self::with_reference_pressure(nodes, elements, 0.0)
    }

    /// Build a closed loop with an explicit reference (ambient) pressure.
    pub fn with_reference_pressure(
        nodes: Vec<String>,
        elements: Vec<AirLoopElement>,
        reference_pressure_pa: f64,
    ) -> Result<Self, AirLoopError> {
        if nodes.len() < 2 {
            return Err(AirLoopError::InvalidTopology {
                reason: "a loop needs at least two nodes",
            });
        }
        if nodes.len() != elements.len() {
            return Err(AirLoopError::InvalidTopology {
                reason: "a closed ring needs exactly one element per node",
            });
        }
        if !reference_pressure_pa.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "reference_pressure_pa",
                requirement: "finite",
                value: reference_pressure_pa,
            });
        }
        let fan_count = elements
            .iter()
            .filter(|element| matches!(element, AirLoopElement::Fan { .. }))
            .count();
        if fan_count != 1 {
            return Err(AirLoopError::InvalidTopology {
                reason: "a loop needs exactly one fan (the pressure source)",
            });
        }
        for element in &elements {
            match element {
                AirLoopElement::Fan { speed_fraction, .. } => {
                    if !speed_fraction.is_finite() {
                        return Err(AirLoopError::InvalidParameter {
                            field: "speed_fraction",
                            requirement: "finite",
                            value: *speed_fraction,
                        });
                    }
                }
                AirLoopElement::Duct {
                    resistance_pa_s2_per_m6,
                    ..
                } => {
                    if !resistance_pa_s2_per_m6.is_finite() || *resistance_pa_s2_per_m6 <= 0.0 {
                        return Err(AirLoopError::InvalidParameter {
                            field: "resistance_pa_s2_per_m6",
                            requirement: "finite and > 0",
                            value: *resistance_pa_s2_per_m6,
                        });
                    }
                }
                AirLoopElement::Damper {
                    open_resistance_pa_s2_per_m6,
                    position,
                    ..
                } => {
                    if !open_resistance_pa_s2_per_m6.is_finite()
                        || *open_resistance_pa_s2_per_m6 <= 0.0
                    {
                        return Err(AirLoopError::InvalidParameter {
                            field: "open_resistance_pa_s2_per_m6",
                            requirement: "finite and > 0",
                            value: *open_resistance_pa_s2_per_m6,
                        });
                    }
                    if !position.is_finite() || !(0.0..=1.0).contains(position) {
                        return Err(AirLoopError::InvalidParameter {
                            field: "position",
                            requirement: "finite and within [0, 1]",
                            value: *position,
                        });
                    }
                }
            }
        }
        Ok(Self {
            nodes,
            elements,
            reference_pressure_pa,
            outdoor_air_mixer: None,
        })
    }

    /// Attach an outdoor-air mixing box to the loop.
    ///
    /// The mixing box forms the mixed supply air (the outdoor-air stream
    /// combined with the return stream of the loop) that the supply path
    /// delivers to the zone. Without a mixing box the loop keeps its
    /// historical behaviour: the supply stream is the return stream.
    pub fn with_outdoor_air_mixer(mut self, mixer: OutdoorAirMixer) -> Self {
        self.outdoor_air_mixer = Some(mixer);
        self
    }

    /// The outdoor-air mixing box of the loop, if one is configured.
    pub fn outdoor_air_mixer(&self) -> Option<&OutdoorAirMixer> {
        self.outdoor_air_mixer.as_ref()
    }

    /// Node identifiers, in path order (node 0 is the reference node).
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// Elements, in path order.
    pub fn elements(&self) -> &[AirLoopElement] {
        &self.elements
    }

    /// Index of the element with the given identifier.
    pub fn element_index_by_id(&self, id: &str) -> Option<usize> {
        self.elements.iter().position(|element| element.id() == id)
    }

    /// Set the position of a damper element.
    pub fn set_damper_position(&mut self, index: usize, position: f64) -> Result<(), AirLoopError> {
        if !position.is_finite() || !(0.0..=1.0).contains(&position) {
            return Err(AirLoopError::InvalidParameter {
                field: "position",
                requirement: "finite and within [0, 1]",
                value: position,
            });
        }
        match self.elements.get_mut(index) {
            Some(AirLoopElement::Damper {
                position: current, ..
            }) => {
                *current = position;
                Ok(())
            }
            Some(_) => Err(AirLoopError::InvalidTopology {
                reason: "the addressed element is not a damper",
            }),
            None => Err(AirLoopError::InvalidTopology {
                reason: "the addressed element does not exist",
            }),
        }
    }

    /// Set the speed fraction of the fan element.
    pub fn set_fan_speed_fraction(
        &mut self,
        index: usize,
        speed_fraction: f64,
    ) -> Result<(), AirLoopError> {
        if !speed_fraction.is_finite() || !(0.0..=1.0).contains(&speed_fraction) {
            return Err(AirLoopError::InvalidParameter {
                field: "speed_fraction",
                requirement: "finite and within [0, 1]",
                value: speed_fraction,
            });
        }
        match self.elements.get_mut(index) {
            Some(AirLoopElement::Fan {
                speed_fraction: current,
                ..
            }) => {
                *current = speed_fraction;
                Ok(())
            }
            Some(_) => Err(AirLoopError::InvalidTopology {
                reason: "the addressed element is not the fan",
            }),
            None => Err(AirLoopError::InvalidTopology {
                reason: "the addressed element does not exist",
            }),
        }
    }

    /// Build the **specified-flow** view of this loop.
    ///
    /// The design mass flow and the fan pressure rise are prescribed, so the
    /// duct resistances of the network are *not* used and none has to be
    /// fitted. The fan curve (and therefore the fan efficiency) is taken
    /// from the fan element of the network.
    pub fn specified_flow_loop(
        &self,
        mass_flow_kg_per_s: f64,
        pressure_rise_pa: f64,
    ) -> Result<SpecifiedFlowLoop, AirLoopError> {
        let fan_curve = self
            .elements
            .iter()
            .find_map(|element| match element {
                AirLoopElement::Fan { curve, .. } => Some(*curve),
                _ => None,
            })
            .expect("the topology is validated to contain exactly one fan");
        SpecifiedFlowLoop::new(fan_curve, mass_flow_kg_per_s, pressure_rise_pa)
    }

    /// Total series resistance of the loop [Pa·s²/m⁶].
    ///
    /// `f64::INFINITY` when the path is fully blocked (a shut damper).
    pub fn total_series_resistance_pa_s2_per_m6(&self) -> f64 {
        let mut total = 0.0;
        for element in &self.elements {
            let resistance = element.resistance_pa_s2_per_m6();
            if !resistance.is_finite() {
                return f64::INFINITY;
            }
            total += resistance;
        }
        total
    }

    /// Residual of the loop pressure balance for a candidate flow [Pa].
    ///
    /// `F(Q) = ΔP_fan(Q; φ) − Σ_i R_i·Q²`. The operating point is the root
    /// of this function.
    pub fn loop_pressure_residual_pa(&self, flow_m3_per_s: f64) -> f64 {
        self.elements
            .iter()
            .map(|element| element.pressure_change_pa(flow_m3_per_s))
            .sum()
    }

    /// Solve the loop pressure balance and recover the node pressures.
    ///
    /// The flow is the root of `F(Q) = 0` found by a bracketed Newton
    /// iteration; `F` is continuous and strictly decreasing for `Q > 0`
    /// (`F(0) = ΔP_fan(0) > 0`, `F(Q) → −∞` as `Q → ∞`), so the root is
    /// unique. The node pressures are then recovered by walking the loop
    /// from the reference node, and the walk is verified for closure and for
    /// node mass balance.
    pub fn solve(&self) -> Result<AirLoopSolution, AirLoopError> {
        let node_count = self.nodes.len();
        let flow = self.solve_loop_flow()?;

        // Recover the node pressures by walking the loop.
        let mut node_pressures_pa = vec![0.0; node_count];
        node_pressures_pa[0] = self.reference_pressure_pa;
        let mut element_pressure_changes_pa = Vec::with_capacity(node_count);
        for (index, element) in self.elements.iter().enumerate() {
            let change = element.pressure_change_pa(flow);
            element_pressure_changes_pa.push(change);
            node_pressures_pa[(index + 1) % node_count] = node_pressures_pa[index] + change;
        }

        // The flow implied by the recovered pressures must agree with the
        // solved flow; this is the consistency (mass-balance) check.
        let mut element_flows_m3_per_s = Vec::with_capacity(node_count);
        for (index, element) in self.elements.iter().enumerate() {
            let change = element_pressure_changes_pa[index];
            element_flows_m3_per_s.push(element.implied_flow_m3_per_s(change));
        }
        let mut node_mass_balance_residual_m3_per_s: f64 = 0.0;
        for node in 0..node_count {
            let inflow = element_flows_m3_per_s[(node + node_count - 1) % node_count];
            let outflow = element_flows_m3_per_s[node];
            let residual = (inflow - outflow).abs();
            if residual > node_mass_balance_residual_m3_per_s {
                node_mass_balance_residual_m3_per_s = residual;
            }
        }

        let fan_index = self
            .elements
            .iter()
            .position(|element| matches!(element, AirLoopElement::Fan { .. }))
            .expect("the topology is validated to contain exactly one fan");
        let fan_pressure_rise_pa = element_pressure_changes_pa[fan_index];
        let closure = element_pressure_changes_pa.iter().sum::<f64>();

        // Air power, shaft power and the fan temperature rise. The air
        // power is the flow times the pressure rise (unchanged behaviour);
        // the shaft power divides it by the fan efficiency of the fan
        // element (1.0 by default, so the shaft power equals the air power
        // on the pre-existing path).
        let fan_air_power_w = flow * fan_pressure_rise_pa;
        let fan_curve = match &self.elements[fan_index] {
            AirLoopElement::Fan { curve, .. } => *curve,
            _ => unreachable!("the fan index points at the fan element"),
        };
        let fan_shaft_power_w = fan_curve.shaft_power_w(fan_air_power_w);
        let fan_mass_flow_kg_per_s = air_density_kg_per_m3(None) * flow;
        let fan_temperature_rise_k =
            fan_curve.temperature_rise_k(fan_shaft_power_w, fan_mass_flow_kg_per_s);

        let solution = AirLoopSolution {
            flow_m3_per_s: flow,
            node_pressures_pa,
            element_flows_m3_per_s,
            element_pressure_changes_pa,
            fan_pressure_rise_pa,
            fan_air_power_w,
            fan_shaft_power_w,
            fan_temperature_rise_k,
            loop_residual_pa: self.loop_pressure_residual_pa(flow),
            pressure_closure_residual_pa: closure.abs(),
            node_mass_balance_residual_m3_per_s,
            supply_humidity_ratio_kg_per_kg_dry_air: 0.0,
        };

        if !solution.is_finite() {
            return Err(AirLoopError::NonFiniteResult { field: "solution" });
        }
        Ok(solution)
    }

    /// Solve the loop balance and report the humidity ratio of the supply air.
    ///
    /// `return_humidity_ratio_kg_per_kg_dry_air` is the humidity ratio of
    /// the return stream of the loop (the zone air is the return stream of
    /// the supply path). With an outdoor-air mixing box configured, the
    /// reported supply humidity ratio is the exact moisture balance of the
    /// mixture, `W_supply = f·W_oa + (1 − f)·W_return`; without a mixing box
    /// it is the return humidity ratio itself.
    pub fn solve_with_return_humidity_ratio(
        &self,
        return_humidity_ratio_kg_per_kg_dry_air: f64,
    ) -> Result<AirLoopSolution, AirLoopError> {
        if !return_humidity_ratio_kg_per_kg_dry_air.is_finite()
            || return_humidity_ratio_kg_per_kg_dry_air < 0.0
        {
            return Err(AirLoopError::InvalidParameter {
                field: "return_humidity_ratio_kg_per_kg_dry_air",
                requirement: "finite and >= 0",
                value: return_humidity_ratio_kg_per_kg_dry_air,
            });
        }
        let mut solution = self.solve()?;
        solution.supply_humidity_ratio_kg_per_kg_dry_air = match &self.outdoor_air_mixer {
            Some(mixer) => {
                let fraction = mixer.outdoor_air_fraction;
                fraction * mixer.outdoor_air.humidity_ratio_kg_per_kg_dry_air
                    + (1.0 - fraction) * return_humidity_ratio_kg_per_kg_dry_air
            }
            None => return_humidity_ratio_kg_per_kg_dry_air,
        };
        Ok(solution)
    }

    /// Root-find the loop pressure balance for the path flow.
    fn solve_loop_flow(&self) -> Result<f64, AirLoopError> {
        let total_resistance = self.total_series_resistance_pa_s2_per_m6();
        if !total_resistance.is_finite() {
            // The path is blocked (shut damper): no flow can be driven.
            return Ok(0.0);
        }
        let fan_running = self.elements.iter().any(|element| match element {
            AirLoopElement::Fan { speed_fraction, .. } => *speed_fraction > 0.0,
            _ => false,
        });
        if !fan_running {
            return Ok(0.0);
        }

        // Bracket the root: F(0) >= 0 and F(hi) < 0.
        let mut low = 0.0;
        let mut high = 1.0;
        while self.loop_pressure_residual_pa(high) > 0.0 {
            high *= 2.0;
            if high > 1.0e6 {
                return Err(AirLoopError::ConvergenceFailure {
                    residual_pa: self.loop_pressure_residual_pa(high),
                    iterations: 0,
                });
            }
        }

        // Safeguarded Newton iteration (the derivative of F is analytic:
        // dF/dQ = -2·c·Q - 2·ΣR·Q).
        let mut flow = 0.5 * (low + high);
        let mut iterations = 0;
        for _ in 0..MAX_SOLVE_ITERATIONS {
            iterations += 1;
            let residual = self.loop_pressure_residual_pa(flow);
            if residual.abs() <= LOOP_RESIDUAL_TOLERANCE_PA {
                break;
            }
            if high - low <= BRACKET_RELATIVE_TOLERANCE * high.max(1.0) {
                break;
            }
            let derivative = self.loop_pressure_residual_derivative(flow);
            let mut candidate = if derivative.abs() > f64::MIN_POSITIVE {
                flow - residual / derivative
            } else {
                f64::NAN
            };
            if !candidate.is_finite() || candidate <= low || candidate >= high {
                candidate = 0.5 * (low + high);
            }
            if residual > 0.0 {
                low = flow;
            } else {
                high = flow;
            }
            flow = candidate;
        }

        let residual = self.loop_pressure_residual_pa(flow);
        if residual.abs() > 1.0e-6 {
            return Err(AirLoopError::ConvergenceFailure {
                residual_pa: residual,
                iterations,
            });
        }
        Ok(flow)
    }

    /// Analytic derivative `dF/dQ` of the loop pressure balance.
    fn loop_pressure_residual_derivative(&self, flow_m3_per_s: f64) -> f64 {
        let mut derivative = 0.0;
        for element in &self.elements {
            derivative += match element {
                AirLoopElement::Fan {
                    curve,
                    speed_fraction,
                    ..
                } => {
                    let _ = speed_fraction;
                    -2.0 * curve.curve_coefficient_pa_s2_per_m6 * flow_m3_per_s
                }
                AirLoopElement::Duct { .. } | AirLoopElement::Damper { .. } => {
                    let resistance = element.resistance_pa_s2_per_m6();
                    if !resistance.is_finite() {
                        0.0
                    } else {
                        -2.0 * resistance * flow_m3_per_s
                    }
                }
            };
        }
        derivative
    }
}

/// Operating state of the zone thermostat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermostatState {
    /// The cooling coil is off; the supply path carries ventilation only.
    Idle,
    /// The cooling coil is on.
    Cooling,
}

/// The actuator command produced by the thermostat.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermostatCommand {
    /// Thermostat state after the update.
    pub state: ThermostatState,
    /// Commanded damper position, `0` (shut) to `1` (fully open).
    pub damper_position: f64,
    /// Whether the cooling coil is active.
    pub coil_on: bool,
}

/// Hysteretic (on/off) zone thermostat with a dead band.
///
/// This is a **real** control device, not the idealised "read a setpoint,
/// return a load" request logic used by the production sequential path:
///
/// * the dead band of width `dead_band_c` is centred on the cooling
///   setpoint;
/// * cooling is engaged when the zone temperature rises above
///   `cooling_setpoint_c + dead_band_c / 2` and released when it falls
///   below `cooling_setpoint_c − dead_band_c / 2` — the hysteresis is what
///   makes the device non-ideal;
/// * **inside the band the state is held** (dead-band dwell): the actuator
///   command does not change, so nothing is actuated while the controlled
///   variable sits in the band;
/// * while cooling, the damper is driven to `cooling_damper_position` and
///   the coil runs; while idle, the damper drops to `idle_damper_position`
///   (the minimum ventilation flow) and the coil is off.
///
/// Because the switch can only occur at a discrete timestep and the zone
/// has thermal mass, the cooling action overshoots past the setpoint: the
/// controlled variable cycles around the setpoint instead of converging
/// onto it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneThermostat {
    /// Cooling setpoint [°C].
    pub cooling_setpoint_c: f64,
    /// Total width of the dead band [K].
    pub dead_band_c: f64,
    /// Damper position commanded while cooling.
    pub cooling_damper_position: f64,
    /// Damper position commanded while idle (minimum ventilation).
    pub idle_damper_position: f64,
    state: ThermostatState,
    actuations: u64,
}

impl ZoneThermostat {
    /// Construct a thermostat.
    pub fn new(
        cooling_setpoint_c: f64,
        dead_band_c: f64,
        cooling_damper_position: f64,
        idle_damper_position: f64,
    ) -> Result<Self, AirLoopError> {
        if !cooling_setpoint_c.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "cooling_setpoint_c",
                requirement: "finite",
                value: cooling_setpoint_c,
            });
        }
        if !dead_band_c.is_finite() || dead_band_c < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "dead_band_c",
                requirement: "finite and >= 0",
                value: dead_band_c,
            });
        }
        for (field, value) in [
            ("cooling_damper_position", cooling_damper_position),
            ("idle_damper_position", idle_damper_position),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(AirLoopError::InvalidParameter {
                    field,
                    requirement: "finite and within [0, 1]",
                    value,
                });
            }
        }
        Ok(Self {
            cooling_setpoint_c,
            dead_band_c,
            cooling_damper_position,
            idle_damper_position,
            state: ThermostatState::Idle,
            actuations: 0,
        })
    }

    /// Current thermostat state.
    pub fn state(&self) -> ThermostatState {
        self.state
    }

    /// Number of state changes (actuations) since construction.
    pub fn actuations(&self) -> u64 {
        self.actuations
    }

    /// Update the thermostat with the current zone temperature and return
    /// the actuator command.
    pub fn update(&mut self, zone_temperature_c: f64) -> ThermostatCommand {
        if !zone_temperature_c.is_finite() {
            // A non-finite sensor reading cannot justify a state change:
            // hold the current state (fail safe, no actuation).
            return self.command();
        }
        let half_band = 0.5 * self.dead_band_c;
        let previous = self.state;
        match self.state {
            ThermostatState::Idle => {
                if zone_temperature_c > self.cooling_setpoint_c + half_band {
                    self.state = ThermostatState::Cooling;
                }
            }
            ThermostatState::Cooling => {
                if zone_temperature_c < self.cooling_setpoint_c - half_band {
                    self.state = ThermostatState::Idle;
                }
            }
        }
        if self.state != previous {
            self.actuations += 1;
        }
        self.command()
    }

    /// The actuator command implied by the current state.
    pub fn command(&self) -> ThermostatCommand {
        let (damper_position, coil_on) = match self.state {
            ThermostatState::Idle => (self.idle_damper_position, false),
            ThermostatState::Cooling => (self.cooling_damper_position, true),
        };
        ThermostatCommand {
            state: self.state,
            damper_position,
            coil_on,
        }
    }
}

/// First-order zone air node (backward Euler).
///
/// The zone air is a single well-mixed node with a thermal capacitance
/// `ρ·V·c_p`, coupled to the outdoor air through the envelope conductance,
/// to the internal gains, and to the supply air stream through the
/// (pressure-driven) supply flow. This is the sink of the supply path used
/// by [`PressureDrivenAirLoop`]; it is *not* a replacement for the
/// multi-node envelope solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneAirNode {
    /// Current zone air temperature [°C].
    pub temperature_c: f64,
    /// Zone air volume [m³].
    pub volume_m3: f64,
    /// Envelope conductance between the zone and the outdoor air [W/K].
    pub envelope_conductance_w_per_k: f64,
    /// Outdoor air temperature [°C].
    pub outdoor_temperature_c: f64,
    /// Internal heat gain [W].
    pub internal_gain_w: f64,
    /// Supply air temperature delivered by the coil when it is running [°C].
    pub conditioned_supply_temperature_c: f64,
    /// Supply air temperature when the coil is off (unconditioned air) [°C].
    pub unconditioned_supply_temperature_c: f64,
    /// Zone air humidity ratio [kg water vapour/kg dry air].
    ///
    /// Defaults to `0.0` (a dry zone); the humidity state is only
    /// transported when the loop is stepped with a humidity-carrying supply
    /// stream (see [`ZoneAirNode::step_humidity`]).
    #[serde(default)]
    pub humidity_ratio_kg_per_kg_dry_air: f64,
    /// Latent internal heat gain released into the zone air [W].
    ///
    /// A positive value adds water vapour to the zone air (occupants,
    /// appliances, wet surfaces): the latent heat is converted into a
    /// water-vapour mass flow with the latent heat of vaporization at 0 °C
    /// ([`LATENT_HEAT_VAPORIZATION_0C_J_PER_KG`]). A negative value removes
    /// moisture (an internal dehumidifier).
    #[serde(default)]
    pub latent_internal_gain_w: f64,
    /// Humidity ratio of the air leaving the cooling coil
    /// [kg water vapour/kg dry air], when the coil dehumidifies.
    ///
    /// `None` (the default) means the coil is *sensible only*: the supply
    /// air keeps the humidity ratio of the mixed supply stream. When it is
    /// set, the coil removes the moisture between the mixed stream and this
    /// value, and the corresponding latent duty is reported as the coil
    /// latent load of the loop step.
    #[serde(default)]
    pub conditioned_supply_humidity_ratio: Option<f64>,
}

impl ZoneAirNode {
    /// Construct a zone air node.
    pub fn new(
        temperature_c: f64,
        volume_m3: f64,
        envelope_conductance_w_per_k: f64,
        outdoor_temperature_c: f64,
        internal_gain_w: f64,
        conditioned_supply_temperature_c: f64,
        unconditioned_supply_temperature_c: f64,
    ) -> Result<Self, AirLoopError> {
        if !volume_m3.is_finite() || volume_m3 <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "volume_m3",
                requirement: "finite and > 0",
                value: volume_m3,
            });
        }
        if !temperature_c.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "temperature_c",
                requirement: "finite",
                value: temperature_c,
            });
        }
        if !envelope_conductance_w_per_k.is_finite() || envelope_conductance_w_per_k < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "envelope_conductance_w_per_k",
                requirement: "finite and >= 0",
                value: envelope_conductance_w_per_k,
            });
        }
        Ok(Self {
            temperature_c,
            volume_m3,
            envelope_conductance_w_per_k,
            outdoor_temperature_c,
            internal_gain_w,
            conditioned_supply_temperature_c,
            unconditioned_supply_temperature_c,
            humidity_ratio_kg_per_kg_dry_air: 0.0,
            latent_internal_gain_w: 0.0,
            conditioned_supply_humidity_ratio: None,
        })
    }

    /// Set the humidity ratio of the air leaving the cooling coil.
    ///
    /// The humidity ratio must be finite and non-negative.
    pub fn with_conditioned_supply_humidity_ratio(
        mut self,
        humidity_ratio_kg_per_kg_dry_air: f64,
    ) -> Result<Self, AirLoopError> {
        if !humidity_ratio_kg_per_kg_dry_air.is_finite() || humidity_ratio_kg_per_kg_dry_air < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "conditioned_supply_humidity_ratio",
                requirement: "finite and >= 0",
                value: humidity_ratio_kg_per_kg_dry_air,
            });
        }
        self.conditioned_supply_humidity_ratio = Some(humidity_ratio_kg_per_kg_dry_air);
        Ok(self)
    }

    /// Set the zone air humidity ratio.
    ///
    /// The humidity ratio must be finite and non-negative.
    pub fn with_humidity_ratio(
        mut self,
        humidity_ratio_kg_per_kg_dry_air: f64,
    ) -> Result<Self, AirLoopError> {
        if !humidity_ratio_kg_per_kg_dry_air.is_finite() || humidity_ratio_kg_per_kg_dry_air < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "humidity_ratio_kg_per_kg_dry_air",
                requirement: "finite and >= 0",
                value: humidity_ratio_kg_per_kg_dry_air,
            });
        }
        self.humidity_ratio_kg_per_kg_dry_air = humidity_ratio_kg_per_kg_dry_air;
        Ok(self)
    }

    /// Set the latent internal heat gain of the zone.
    ///
    /// The latent gain must be finite; a positive value adds water vapour
    /// to the zone air, a negative value removes it.
    pub fn with_latent_internal_gain(
        mut self,
        latent_internal_gain_w: f64,
    ) -> Result<Self, AirLoopError> {
        if !latent_internal_gain_w.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "latent_internal_gain_w",
                requirement: "finite",
                value: latent_internal_gain_w,
            });
        }
        self.latent_internal_gain_w = latent_internal_gain_w;
        Ok(self)
    }

    /// Mass of dry air in the zone [kg].
    ///
    /// The constant-density approximation of the thermal balance is used
    /// here as well, so the moisture balance and the energy balance see the
    /// same dry-air mass.
    pub fn dry_air_mass_kg(&self) -> f64 {
        STANDARD_AIR_DENSITY_KG_PER_M3 * self.volume_m3
    }

    /// Mass of water vapour currently held in the zone air [kg].
    pub fn water_vapour_mass_kg(&self) -> f64 {
        self.dry_air_mass_kg() * self.humidity_ratio_kg_per_kg_dry_air
    }

    /// Water-vapour mass flow of the latent internal gain [kg/s].
    pub fn latent_water_vapour_mass_flow_kg_per_s(&self) -> f64 {
        self.latent_internal_gain_w / LATENT_HEAT_VAPORIZATION_0C_J_PER_KG
    }

    /// Heat capacity of the zone air node [J/K].
    pub fn heat_capacity_j_per_k(&self) -> f64 {
        STANDARD_AIR_DENSITY_KG_PER_M3 * self.volume_m3 * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K
    }

    /// Mass flow of the supply stream [kg/s].
    ///
    /// `moist_air_density_kg_per_m3` is the density of the humid-air state
    /// at the supply condition when one is available; the standard dry-air
    /// density is the fallback (see [`air_density_kg_per_m3`]).
    pub fn supply_mass_flow_kg_per_s(
        &self,
        supply_flow_m3_per_s: f64,
        moist_air_density_kg_per_m3: Option<f64>,
    ) -> f64 {
        air_density_kg_per_m3(moist_air_density_kg_per_m3) * supply_flow_m3_per_s
    }

    /// Inverse zone-node query: the supply air temperature required to hold
    /// the zone at `target_temperature_c` while the supply stream removes
    /// `load_w` from the zone.
    ///
    /// The forward node step takes the supply temperature as an input and
    /// returns the zone temperature; this query is its algebraic inverse:
    ///
    /// ```text
    /// ṁ·c_p·(t_supply − t_zone) = −q   ⟹   t_supply = t_zone − q / (ṁ·c_p)
    /// ```
    ///
    /// `load_w` is the sensible load removed from the zone by the supply
    /// stream (positive for a cooling load), `supply_flow_m3_per_s` is the
    /// supply volumetric flow and `moist_air_density_kg_per_m3` is the
    /// density of the humid-air state when one is available (the standard
    /// dry-air density is the fallback).
    pub fn supply_temperature_for_target(
        &self,
        target_temperature_c: f64,
        load_w: f64,
        supply_flow_m3_per_s: f64,
        moist_air_density_kg_per_m3: Option<f64>,
    ) -> Result<f64, AirLoopError> {
        if !target_temperature_c.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "target_temperature_c",
                requirement: "finite",
                value: target_temperature_c,
            });
        }
        if !load_w.is_finite() {
            return Err(AirLoopError::InvalidParameter {
                field: "load_w",
                requirement: "finite",
                value: load_w,
            });
        }
        if !supply_flow_m3_per_s.is_finite() || supply_flow_m3_per_s <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "supply_flow_m3_per_s",
                requirement: "finite and > 0",
                value: supply_flow_m3_per_s,
            });
        }
        let mass_flow_capacitance = self
            .supply_mass_flow_kg_per_s(supply_flow_m3_per_s, moist_air_density_kg_per_m3)
            * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K;
        if mass_flow_capacitance <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "supply_flow_m3_per_s",
                requirement: "finite and > 0",
                value: supply_flow_m3_per_s,
            });
        }
        Ok(target_temperature_c - load_w / mass_flow_capacitance)
    }

    /// Advance the zone air node by `dt_s` with the given supply stream.
    ///
    /// Returns the new zone air temperature [°C]. The integration is
    /// backward Euler (unconditionally stable for the linear balance).
    pub fn step(
        &mut self,
        dt_s: f64,
        supply_flow_m3_per_s: f64,
        supply_air_temperature_c: f64,
    ) -> f64 {
        let capacitance = self.heat_capacity_j_per_k();
        let mass_flow_capacitance = STANDARD_AIR_DENSITY_KG_PER_M3
            * supply_flow_m3_per_s
            * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K;
        let denominator =
            capacitance / dt_s + self.envelope_conductance_w_per_k + mass_flow_capacitance;
        if denominator <= 0.0 {
            return self.temperature_c;
        }
        let numerator = capacitance / dt_s * self.temperature_c
            + self.internal_gain_w
            + self.envelope_conductance_w_per_k * self.outdoor_temperature_c
            + mass_flow_capacitance * supply_air_temperature_c;
        self.temperature_c = numerator / denominator;
        self.temperature_c
    }

    /// Advance the zone humidity state by `dt_s` for the given supply stream.
    ///
    /// The zone moisture balance is the humidity counterpart of the thermal
    /// balance of [`ZoneAirNode::step`] and is integrated with the same
    /// backward-Euler scheme:
    ///
    /// ```text
    /// m_da·(W_new − W_old)/dt = ṁ_sup·(W_sup − W_new) + W_latent/h_fg
    /// ```
    ///
    /// where `m_da` is the dry-air mass of the zone, `ṁ_sup` the dry-air
    /// mass flow of the supply stream, `W_sup` the humidity ratio of the
    /// supply air and `W_latent/h_fg` the water-vapour mass flow of the
    /// latent internal gain. Returns the new zone humidity ratio [kg/kg].
    pub fn step_humidity(
        &mut self,
        dt_s: f64,
        supply_flow_m3_per_s: f64,
        supply_humidity_ratio_kg_per_kg_dry_air: f64,
    ) -> Result<f64, AirLoopError> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "dt_s",
                requirement: "finite and > 0",
                value: dt_s,
            });
        }
        if !supply_flow_m3_per_s.is_finite() || supply_flow_m3_per_s < 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "supply_flow_m3_per_s",
                requirement: "finite and >= 0",
                value: supply_flow_m3_per_s,
            });
        }
        if !supply_humidity_ratio_kg_per_kg_dry_air.is_finite()
            || supply_humidity_ratio_kg_per_kg_dry_air < 0.0
        {
            return Err(AirLoopError::InvalidParameter {
                field: "supply_humidity_ratio_kg_per_kg_dry_air",
                requirement: "finite and >= 0",
                value: supply_humidity_ratio_kg_per_kg_dry_air,
            });
        }
        let dry_air_mass = self.dry_air_mass_kg();
        let supply_dry_air_mass_flow = STANDARD_AIR_DENSITY_KG_PER_M3 * supply_flow_m3_per_s;
        let denominator = dry_air_mass / dt_s + supply_dry_air_mass_flow;
        if denominator <= 0.0 {
            return Ok(self.humidity_ratio_kg_per_kg_dry_air);
        }
        let numerator = dry_air_mass / dt_s * self.humidity_ratio_kg_per_kg_dry_air
            + supply_dry_air_mass_flow * supply_humidity_ratio_kg_per_kg_dry_air
            + self.latent_water_vapour_mass_flow_kg_per_s();
        self.humidity_ratio_kg_per_kg_dry_air = numerator / denominator;
        Ok(self.humidity_ratio_kg_per_kg_dry_air)
    }
}

/// State of the air loop after one coupled step.
#[derive(Debug, Clone, PartialEq)]
pub struct AirLoopStepResult {
    /// Zone air temperature after the step [°C].
    pub zone_temperature_c: f64,
    /// Supply flow solved from the pressure balance [m³/s].
    pub supply_flow_m3_per_s: f64,
    /// Supply mass flow [kg/s].
    pub supply_mass_flow_kg_per_s: f64,
    /// Commanded damper position after the step.
    pub damper_position: f64,
    /// Whether the cooling coil is running.
    pub coil_on: bool,
    /// Thermostat state after the step.
    pub thermostat_state: ThermostatState,
    /// Number of thermostat actuations so far.
    pub actuation_count: u64,
    /// The pressure-driven solution of this step.
    pub solution: AirLoopSolution,
    /// Zone air humidity ratio after the step [kg water vapour/kg dry air].
    pub zone_humidity_ratio_kg_per_kg_dry_air: f64,
    /// Humidity ratio of the air delivered to the zone in this step
    /// [kg water vapour/kg dry air].
    ///
    /// This is the humidity ratio of the mixed supply stream after the
    /// coil (a dehumidifying coil lowers it).
    pub supply_humidity_ratio_kg_per_kg_dry_air: f64,
    /// The mixed outdoor-air/return supply state, when the loop has an
    /// outdoor-air mixing box configured.
    pub mixed_supply_state: Option<MoistAirStream>,
    /// Latent load of the supply path [W].
    ///
    /// Positive when the supply air is drier than the zone air (the latent
    /// duty the coil has to remove); zero when the coil is off.
    pub coil_latent_load_w: f64,
}

/// One zone served by a pressure-driven air loop.
///
/// The coupling is: **thermostat → damper position → pressure-driven network
/// solve → supply flow → zone air node**. The flow is never imposed; it is
/// the solution of the loop pressure balance for the commanded damper
/// position.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureDrivenAirLoop {
    network: AirLoopNetwork,
    thermostat: ZoneThermostat,
    zone: ZoneAirNode,
    damper_index: usize,
    last_solution: Option<AirLoopSolution>,
}

impl PressureDrivenAirLoop {
    /// Assemble the loop from its parts.
    ///
    /// `damper_id` identifies the damper element the thermostat drives.
    pub fn new(
        network: AirLoopNetwork,
        thermostat: ZoneThermostat,
        zone: ZoneAirNode,
        damper_id: &str,
    ) -> Result<Self, AirLoopError> {
        let damper_index =
            network
                .element_index_by_id(damper_id)
                .ok_or(AirLoopError::InvalidTopology {
                    reason: "the network has no element with the requested damper id",
                })?;
        if !matches!(
            network.elements()[damper_index],
            AirLoopElement::Damper { .. }
        ) {
            return Err(AirLoopError::InvalidTopology {
                reason: "the addressed element is not a damper",
            });
        }
        Ok(Self {
            network,
            thermostat,
            zone,
            damper_index,
            last_solution: None,
        })
    }

    /// The duct network.
    pub fn network(&self) -> &AirLoopNetwork {
        &self.network
    }

    /// The thermostat.
    pub fn thermostat(&self) -> &ZoneThermostat {
        &self.thermostat
    }

    /// The zone air node.
    pub fn zone(&self) -> &ZoneAirNode {
        &self.zone
    }

    /// Mutable access to the zone air node (e.g. to apply a load step).
    pub fn zone_mut(&mut self) -> &mut ZoneAirNode {
        &mut self.zone
    }

    /// The solution of the most recent step.
    pub fn last_solution(&self) -> Option<&AirLoopSolution> {
        self.last_solution.as_ref()
    }

    /// Advance the coupled loop by `dt_s`.
    ///
    /// The thermostat is updated first (with the zone temperature at the
    /// beginning of the step), the commanded damper position is applied to
    /// the network, the network is solved for the pressure balance, and the
    /// resulting supply flow is then integrated into the zone air node.
    pub fn step(&mut self, dt_s: f64) -> Result<AirLoopStepResult, AirLoopError> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(AirLoopError::InvalidParameter {
                field: "dt_s",
                requirement: "finite and > 0",
                value: dt_s,
            });
        }
        let command = self.thermostat.update(self.zone.temperature_c);
        self.network
            .set_damper_position(self.damper_index, command.damper_position)?;
        // The return stream of the loop is the zone air; the outdoor-air
        // mixing box combines it with the outdoor air to form the mixed
        // supply state.
        let return_stream = MoistAirStream::new(
            self.zone.temperature_c,
            self.zone.humidity_ratio_kg_per_kg_dry_air,
        )?;
        let mixed_supply_state = match self.network.outdoor_air_mixer() {
            Some(mixer) => Some(mixer.mix(&return_stream)?),
            None => None,
        };
        let solution = self
            .network
            .solve_with_return_humidity_ratio(self.zone.humidity_ratio_kg_per_kg_dry_air)?;
        let supply_temperature = if command.coil_on {
            self.zone.conditioned_supply_temperature_c
        } else {
            self.zone.unconditioned_supply_temperature_c
        };
        let zone_temperature = self
            .zone
            .step(dt_s, solution.flow_m3_per_s, supply_temperature);
        let supply_mass_flow_kg_per_s = STANDARD_AIR_DENSITY_KG_PER_M3 * solution.flow_m3_per_s;
        // The humidity ratio of the mixed supply stream (the outdoor-air /
        // return mixture, or the return stream itself when the loop has no
        // mixing box).
        let mixed_humidity_ratio = solution.supply_humidity_ratio_kg_per_kg_dry_air;
        // A dehumidifying coil delivers drier air than the mixed stream.
        let supply_humidity_ratio = if command.coil_on {
            self.zone
                .conditioned_supply_humidity_ratio
                .unwrap_or(mixed_humidity_ratio)
        } else {
            mixed_humidity_ratio
        };
        let zone_humidity_ratio =
            self.zone
                .step_humidity(dt_s, solution.flow_m3_per_s, supply_humidity_ratio)?;
        // The coil-side latent duty: the moisture the (running) coil takes
        // out of the mixed supply stream.
        let coil_latent_load_w = if command.coil_on {
            supply_mass_flow_kg_per_s
                * (mixed_humidity_ratio - supply_humidity_ratio)
                * LATENT_HEAT_VAPORIZATION_0C_J_PER_KG
        } else {
            0.0
        };
        self.last_solution = Some(solution.clone());
        Ok(AirLoopStepResult {
            zone_temperature_c: zone_temperature,
            supply_flow_m3_per_s: solution.flow_m3_per_s,
            supply_mass_flow_kg_per_s,
            damper_position: command.damper_position,
            coil_on: command.coil_on,
            thermostat_state: command.state,
            actuation_count: self.thermostat.actuations(),
            solution,
            zone_humidity_ratio_kg_per_kg_dry_air: zone_humidity_ratio,
            supply_humidity_ratio_kg_per_kg_dry_air: supply_humidity_ratio,
            mixed_supply_state,
            coil_latent_load_w,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference fan: shut-off pressure 300 Pa, curve coefficient 600 Pa·s²/m⁶
    /// (i.e. 150 Pa rise at 0.5 m³/s).
    fn reference_curve() -> FanCurve {
        FanCurve::new(300.0, 600.0).expect("reference fan curve is valid")
    }

    /// The reference loop: ambient → fan → duct → terminal damper → zone →
    /// return duct → ambient.
    fn reference_network() -> AirLoopNetwork {
        AirLoopNetwork::new(
            vec![
                "ambient".to_string(),
                "fan_discharge".to_string(),
                "duct_exit".to_string(),
                "zone".to_string(),
            ],
            vec![
                AirLoopElement::Fan {
                    id: "supply_fan".to_string(),
                    curve: reference_curve(),
                    speed_fraction: 1.0,
                },
                AirLoopElement::Duct {
                    id: "supply_duct".to_string(),
                    resistance_pa_s2_per_m6: 400.0,
                },
                AirLoopElement::Damper {
                    id: "terminal_damper".to_string(),
                    open_resistance_pa_s2_per_m6: 200.0,
                    position: 1.0,
                },
                AirLoopElement::Duct {
                    id: "return_duct".to_string(),
                    resistance_pa_s2_per_m6: 200.0,
                },
            ],
        )
        .expect("reference network topology is valid")
    }

    /// Closed-form operating point of the reference loop, used as an
    /// independent cross-check of the numerical solve:
    /// `φ²·P_static = (c + ΣR)·Q²`.
    fn closed_form_flow(speed_fraction: f64, total_resistance: f64) -> f64 {
        speed_fraction * (300.0 / (600.0 + total_resistance)).sqrt()
    }

    #[test]
    fn fan_curve_reproduces_the_rated_point_and_the_affinity_laws() {
        let curve = FanCurve::from_rated_point(300.0, 0.5, 150.0).expect("rated point is valid");
        assert!((curve.curve_coefficient_pa_s2_per_m6 - 600.0).abs() < 1.0e-12);
        // The rated point is reproduced exactly.
        assert!((curve.pressure_rise_pa(0.5, 1.0) - 150.0).abs() < 1.0e-12);
        // Fan law 2 along the affinity-scaled curve: at half speed the flow
        // halves and the pressure rise quarters (evaluated at the
        // affinity-scaled flow).
        let half_speed_flow = 0.25;
        assert!((curve.pressure_rise_pa(half_speed_flow, 0.5) - 0.25 * 150.0).abs() < 1.0e-12);
        // A stopped fan produces no pressure rise.
        assert!(curve.pressure_rise_pa(0.25, 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn fan_curve_inverts_the_pressure_rise() {
        let curve = reference_curve();
        for speed in [0.4, 0.7, 1.0] {
            for flow in [0.05, 0.2, 0.4] {
                let rise = curve.pressure_rise_pa(flow, speed);
                let recovered = curve.flow_for_pressure_rise_m3_per_s(rise, speed);
                assert!((recovered - flow).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn operating_point_satisfies_the_fan_curve_and_the_resistances() {
        let network = reference_network();
        let solution = network.solve().expect("the reference loop solves");

        // The loop pressure balance holds at the operating point.
        assert!(solution.loop_residual_pa.abs() <= LOOP_RESIDUAL_TOLERANCE_PA);
        // The pressure walk closes on the reference node.
        assert!(solution.pressure_closure_residual_pa <= 1.0e-9);
        // The fan rise equals the sum of the duct/damper pressure drops.
        let drops: f64 = solution
            .element_pressure_changes_pa
            .iter()
            .skip(1)
            .sum::<f64>();
        assert!((solution.fan_pressure_rise_pa + drops).abs() < 1.0e-9);
        // The solved flow reproduces the closed-form operating point of the
        // series loop (independent cross-check of the solve).
        let expected = closed_form_flow(1.0, network.total_series_resistance_pa_s2_per_m6());
        assert!((solution.flow_m3_per_s - expected).abs() < 1.0e-9);
    }

    #[test]
    fn higher_damper_resistance_reduces_the_solved_flow() {
        let mut network = reference_network();
        let damper_index = network
            .element_index_by_id("terminal_damper")
            .expect("the damper exists");
        let mut previous = f64::INFINITY;
        for position in [1.0, 0.8, 0.5, 0.25, 0.1] {
            network
                .set_damper_position(damper_index, position)
                .expect("the damper position is valid");
            let solution = network.solve().expect("the loop solves");
            assert!(solution.flow_m3_per_s < previous);
            assert!(solution.flow_m3_per_s >= 0.0);
            previous = solution.flow_m3_per_s;
        }
    }

    #[test]
    fn higher_duct_resistance_reduces_the_solved_flow() {
        let mut previous = f64::INFINITY;
        for resistance in [100.0, 400.0, 900.0, 1600.0] {
            let network = AirLoopNetwork::new(
                vec![
                    "ambient".to_string(),
                    "fan_discharge".to_string(),
                    "duct_exit".to_string(),
                    "zone".to_string(),
                ],
                vec![
                    AirLoopElement::Fan {
                        id: "supply_fan".to_string(),
                        curve: reference_curve(),
                        speed_fraction: 1.0,
                    },
                    AirLoopElement::Duct {
                        id: "supply_duct".to_string(),
                        resistance_pa_s2_per_m6: resistance,
                    },
                    AirLoopElement::Damper {
                        id: "terminal_damper".to_string(),
                        open_resistance_pa_s2_per_m6: 200.0,
                        position: 1.0,
                    },
                    AirLoopElement::Duct {
                        id: "return_duct".to_string(),
                        resistance_pa_s2_per_m6: 200.0,
                    },
                ],
            )
            .expect("the loop topology is valid");
            let solution = network.solve().expect("the loop solves");
            assert!(solution.flow_m3_per_s < previous);
            previous = solution.flow_m3_per_s;
        }
    }

    #[test]
    fn higher_fan_speed_raises_the_solved_flow() {
        let mut network = reference_network();
        let fan_index = network
            .element_index_by_id("supply_fan")
            .expect("the fan exists");
        let mut previous = -1.0;
        for speed in [0.3, 0.5, 0.75, 1.0] {
            network
                .set_fan_speed_fraction(fan_index, speed)
                .expect("the fan speed is valid");
            let solution = network.solve().expect("the loop solves");
            assert!(solution.flow_m3_per_s > previous);
            // Fan law 1: the flow follows the speed fraction linearly.
            let expected = closed_form_flow(speed, network.total_series_resistance_pa_s2_per_m6());
            assert!((solution.flow_m3_per_s - expected).abs() < 1.0e-9);
            previous = solution.flow_m3_per_s;
        }
    }

    #[test]
    fn solved_flow_is_finite_nonnegative_and_mass_balanced() {
        let network = reference_network();
        let solution = network.solve().expect("the reference loop solves");

        assert!(solution.is_finite());
        assert!(solution.flow_m3_per_s.is_finite());
        assert!(solution.flow_m3_per_s >= 0.0);
        assert!(solution.fan_air_power_w >= 0.0);
        // Every element carries the same flow: inflow equals outflow at
        // every node.
        for flow in &solution.element_flows_m3_per_s {
            assert!((flow - solution.flow_m3_per_s).abs() < 1.0e-9);
        }
        assert!(solution.node_mass_balance_residual_m3_per_s < 1.0e-9);
        // The zone node sits slightly above ambient: the fan pressurises the
        // zone against the return-path resistance.
        let zone_pressure = solution.node_pressures_pa[3];
        assert!(zone_pressure > 0.0);
    }

    #[test]
    fn a_shut_damper_blocks_the_loop() {
        let mut network = reference_network();
        let damper_index = network
            .element_index_by_id("terminal_damper")
            .expect("the damper exists");
        network
            .set_damper_position(damper_index, 0.0)
            .expect("the damper position is valid");
        let solution = network.solve().expect("the blocked loop still solves");
        assert!(solution.flow_m3_per_s.abs() < 1.0e-12);
        // The blocked fan develops its shut-off pressure at the discharge.
        assert!((solution.fan_pressure_rise_pa - 300.0).abs() < 1.0e-9);
    }

    #[test]
    fn thermostat_holds_its_state_inside_the_dead_band() {
        let mut thermostat =
            ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        // Inside the band: no actuation, the command does not change.
        for temperature in [22.4, 22.2, 22.0, 21.8, 21.6] {
            let command = thermostat.update(temperature);
            assert_eq!(command.state, ThermostatState::Idle);
            assert!(!command.coil_on);
            assert!((command.damper_position - 0.2).abs() < 1.0e-12);
            assert_eq!(thermostat.actuations(), 0);
        }
        // Above the upper edge the coil engages (one actuation).
        let command = thermostat.update(22.6);
        assert_eq!(command.state, ThermostatState::Cooling);
        assert!(command.coil_on);
        assert!((command.damper_position - 1.0).abs() < 1.0e-12);
        assert_eq!(thermostat.actuations(), 1);
        // Below the lower edge the coil releases (second actuation).
        let command = thermostat.update(21.4);
        assert_eq!(command.state, ThermostatState::Idle);
        assert_eq!(thermostat.actuations(), 2);
    }

    #[test]
    fn thermostat_rejects_invalid_parameters() {
        assert!(ZoneThermostat::new(f64::NAN, 1.0, 1.0, 0.2).is_err());
        assert!(ZoneThermostat::new(22.0, -1.0, 1.0, 0.2).is_err());
        assert!(ZoneThermostat::new(22.0, 1.0, 1.5, 0.2).is_err());
    }

    /// The reference zone: 300 m³, UA = 200 W/K, outdoors at 35 °C.
    fn reference_zone() -> ZoneAirNode {
        ZoneAirNode::new(21.0, 300.0, 200.0, 35.0, 500.0, 13.0, 35.0)
            .expect("the reference zone is valid")
    }

    #[test]
    fn zone_step_conserves_energy() {
        let mut zone = reference_zone();
        let dt = 60.0;
        let flow = 0.4;
        let supply_temperature = 13.0;
        let temperature_before = zone.temperature_c;
        let temperature_after = zone.step(dt, flow, supply_temperature);
        // The change of the zone internal energy equals the net heat flow
        // over the step (backward Euler on a linear balance).
        let capacitance = zone.heat_capacity_j_per_k();
        let mass_flow_capacitance =
            STANDARD_AIR_DENSITY_KG_PER_M3 * flow * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K;
        let heat_in = zone.internal_gain_w
            + zone.envelope_conductance_w_per_k * (zone.outdoor_temperature_c - temperature_after)
            + mass_flow_capacitance * (supply_temperature - temperature_after);
        let stored = capacitance * (temperature_after - temperature_before);
        assert!((stored - heat_in * dt).abs() < 1.0e-6);
    }

    #[test]
    fn step_response_overshoots_the_setpoint_and_cycles() {
        // Reference closed loop: hysteretic thermostat + pressure-driven
        // network + first-order zone.
        let network = reference_network();
        let thermostat = ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        let mut air_loop =
            PressureDrivenAirLoop::new(network, thermostat, reference_zone(), "terminal_damper")
                .expect("the air loop assembles");

        let dt = 60.0;
        let steps = 6 * 60; // six hours
        let step_at = 30; // the load step happens after half an hour
                          // (T_before, T_after, actuations, coil_on)
        let mut trace: Vec<(f64, f64, u64, bool)> = Vec::with_capacity(steps);
        for index in 0..steps {
            // Load step: the internal gain triples after half an hour.
            air_loop.zone_mut().internal_gain_w = if index < step_at { 500.0 } else { 1500.0 };
            let temperature_before = air_loop.zone().temperature_c;
            let result = air_loop.step(dt).expect("the loop steps");
            trace.push((
                temperature_before,
                result.zone_temperature_c,
                result.actuation_count,
                result.coil_on,
            ));
        }

        let setpoint = 22.0;
        let dead_band = 1.0;
        let post_step: Vec<f64> = trace[step_at..].iter().map(|row| row.1).collect();
        let minimum = post_step.iter().copied().fold(f64::INFINITY, f64::min);

        // (1) OVERSHOOT: the cooling action drives the zone below the
        //     setpoint by more than a quarter of the dead band — the
        //     controlled variable does not settle on the setpoint.
        assert!(
            minimum < setpoint - 0.25 * dead_band,
            "expected an overshoot below the setpoint, got minimum {minimum}"
        );
        // (2) The response does not converge: the peak-to-peak amplitude of
        //     the last hour is a sizeable fraction of the dead band (the
        //     hysteretic limit cycle).
        let last_hour: Vec<f64> = trace[trace.len() - 60..].iter().map(|row| row.1).collect();
        let last_min = last_hour.iter().copied().fold(f64::INFINITY, f64::min);
        let last_max = last_hour.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            last_max - last_min >= 0.5 * dead_band,
            "expected a limit cycle of at least half the dead band, got {}",
            last_max - last_min
        );
        // (3) DEAD-BAND DWELL: the thermostat never actuates while the zone
        //     temperature is inside the dead band.
        let mut actuations_inside_band = 0;
        let mut previous_actuations = 0;
        for (temperature_before, _, actuations, _) in &trace {
            if (temperature_before - setpoint).abs() < 0.5 * dead_band
                && *actuations > previous_actuations
            {
                actuations_inside_band += 1;
            }
            previous_actuations = *actuations;
        }
        assert_eq!(
            actuations_inside_band, 0,
            "the thermostat must not actuate inside the dead band"
        );
        // (4) The load step is answered by the closed loop: the cooling duty
        //     cycle rises (the loop works harder) while the zone
        //     temperature stays regulated around the setpoint.
        let duty_before =
            trace[..step_at].iter().filter(|row| row.3).count() as f64 / step_at as f64;
        let duty_after =
            trace[step_at..].iter().filter(|row| row.3).count() as f64 / (steps - step_at) as f64;
        assert!(
            duty_after > duty_before,
            "expected a higher cooling duty after the load step, got {duty_before} -> {duty_after}"
        );
        // The loop did cycle: more than one actuation per hour on average.
        assert!(trace.last().expect("the trace is not empty").2 > 10);
    }

    #[test]
    fn step_response_flow_is_solved_from_the_pressure_balance() {
        let network = reference_network();
        let thermostat = ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        let mut air_loop =
            PressureDrivenAirLoop::new(network, thermostat, reference_zone(), "terminal_damper")
                .expect("the air loop assembles");

        let mut cooling_flow: Option<f64> = None;
        let mut idle_flow: Option<f64> = None;
        for _ in 0..120 {
            let result = air_loop.step(60.0).expect("the loop steps");
            // The supply flow is the solution of the loop pressure balance
            // for the commanded damper position, not an imposed constant.
            let total_resistance = air_loop.network().total_series_resistance_pa_s2_per_m6();
            let expected = closed_form_flow(1.0, total_resistance);
            assert!((result.supply_flow_m3_per_s - expected).abs() < 1.0e-9);
            if result.coil_on {
                cooling_flow = Some(result.supply_flow_m3_per_s);
            } else {
                idle_flow = Some(result.supply_flow_m3_per_s);
            }
        }
        // Both operating states were visited and they differ: the damper
        // position changes the solved flow.
        let cooling = cooling_flow.expect("the cooling state was visited");
        let idle = idle_flow.expect("the idle state was visited");
        assert!(cooling > idle);
    }

    #[test]
    fn fan_temperature_rise_matches_the_analytic_value_at_a_stated_efficiency() {
        // A fan stated at 70 % total efficiency (the QAS convention).
        let efficiency = 0.70;
        let curve =
            FanCurve::with_efficiency(300.0, 600.0, efficiency).expect("the fan curve is valid");
        assert_eq!(curve.efficiency(), efficiency);

        let flow_m3_per_s = 0.5;
        let pressure_rise_pa = 150.0;
        let mass_flow_kg_per_s = 0.60;

        // The analytic chain: air power -> shaft power -> temperature rise.
        let air_power_w = flow_m3_per_s * pressure_rise_pa;
        let shaft_power_w = air_power_w / efficiency;
        let expected_rise_k =
            shaft_power_w / (mass_flow_kg_per_s * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K);

        assert!((curve.shaft_power_w(air_power_w) - shaft_power_w).abs() < 1.0e-12);
        assert!(
            (curve.temperature_rise_k(shaft_power_w, mass_flow_kg_per_s) - expected_rise_k).abs()
                < 1.0e-12
        );
        assert!(
            (curve.air_temperature_rise_k(air_power_w, mass_flow_kg_per_s) - expected_rise_k).abs()
                < 1.0e-12
        );
        // The value itself: 107.14 W of shaft power over 0.6 kg/s of air.
        assert!(expected_rise_k > 0.17 && expected_rise_k < 0.19);
    }

    #[test]
    fn the_default_fan_efficiency_preserves_the_pre_existing_air_power_results() {
        let curve = reference_curve();
        assert_eq!(curve.efficiency(), DEFAULT_FAN_EFFICIENCY);

        let network = reference_network();
        let solution = network.solve().expect("the reference loop solves");

        // The air power is untouched ...
        assert!(
            (solution.fan_air_power_w - solution.flow_m3_per_s * solution.fan_pressure_rise_pa)
                .abs()
                < 1.0e-12
        );
        // ... and the shaft power equals it at the default efficiency.
        assert_eq!(solution.fan_shaft_power_w, solution.fan_air_power_w);
        let mass_flow_kg_per_s = STANDARD_AIR_DENSITY_KG_PER_M3 * solution.flow_m3_per_s;
        let expected_rise_k =
            solution.fan_shaft_power_w / (mass_flow_kg_per_s * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K);
        assert!((solution.fan_temperature_rise_k - expected_rise_k).abs() < 1.0e-12);
    }

    #[test]
    fn specified_flow_mode_reproduces_the_design_mass_flow_without_a_fitted_resistance() {
        let efficiency = 0.70;
        let curve =
            FanCurve::with_efficiency(300.0, 600.0, efficiency).expect("the fan curve is valid");
        let design_mass_flow_kg_per_s = 0.60;
        let design_pressure_rise_pa = 250.0;

        // The specified-flow mode carries no duct resistance at all: the
        // design flow and the design pressure rise are given.
        let specified =
            SpecifiedFlowLoop::new(curve, design_mass_flow_kg_per_s, design_pressure_rise_pa)
                .expect("the specified-flow loop is valid");
        let solution = specified
            .solve(None)
            .expect("the specified-flow loop solves");

        // The design mass flow is reproduced exactly.
        assert_eq!(solution.mass_flow_kg_per_s, design_mass_flow_kg_per_s);
        assert_eq!(solution.pressure_rise_pa, design_pressure_rise_pa);
        assert!(solution.is_finite());

        // No humidity state is available on this base, so the standard
        // density is the fallback that runs.
        assert_eq!(solution.density_kg_per_m3, STANDARD_AIR_DENSITY_KG_PER_M3);
        assert_eq!(
            solution.volumetric_flow_m3_per_s,
            design_mass_flow_kg_per_s / STANDARD_AIR_DENSITY_KG_PER_M3
        );

        let expected_air_power_w = solution.volumetric_flow_m3_per_s * design_pressure_rise_pa;
        assert!((solution.fan_air_power_w - expected_air_power_w).abs() < 1.0e-12);
        let expected_shaft_power_w = expected_air_power_w / efficiency;
        assert!((solution.fan_shaft_power_w - expected_shaft_power_w).abs() < 1.0e-12);
        let expected_rise_k =
            expected_shaft_power_w / (design_mass_flow_kg_per_s * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K);
        assert!((solution.fan_temperature_rise_k - expected_rise_k).abs() < 1.0e-12);

        // The same mode reached from a network ignores the duct resistances
        // of that network entirely: the specified flow is not fitted.
        let network = reference_network();
        let from_network = network
            .specified_flow_loop(design_mass_flow_kg_per_s, design_pressure_rise_pa)
            .expect("the network yields a specified-flow loop");
        let network_solution = from_network.solve(None).expect("the loop solves");
        assert_eq!(
            network_solution.mass_flow_kg_per_s,
            design_mass_flow_kg_per_s
        );
        // The efficiency-independent operating point is identical too (the
        // reference network's fan carries the default efficiency 1.0, so its
        // shaft power and rise legitimately differ from the 0.70 loop above).
        assert_eq!(
            network_solution.volumetric_flow_m3_per_s,
            solution.volumetric_flow_m3_per_s
        );
        assert_eq!(network_solution.fan_air_power_w, solution.fan_air_power_w);
        assert_eq!(
            network_solution.fan_shaft_power_w,
            network_solution.fan_air_power_w
        );
    }

    #[test]
    fn inverse_zone_node_query_round_trips_the_forward_step() {
        let mut zone = reference_zone();
        let dt = 60.0;
        let supply_flow_m3_per_s = 0.4;
        let supply_temperature_c = 13.0;

        // Forward step: supply temperature in, zone temperature out.
        let temperature_before = zone.temperature_c;
        let temperature_after = zone.step(dt, supply_flow_m3_per_s, supply_temperature_c);

        // The load the supply stream removed from the zone, read off the
        // forward step's energy balance (an independent quantity):
        //   C/Δt·(T_new − T_old) = G·(T_out − T_new) + q_int + ṁ·c_p·(t_s − T_new)
        let capacitance = zone.heat_capacity_j_per_k();
        let mass_flow_capacitance = STANDARD_AIR_DENSITY_KG_PER_M3
            * supply_flow_m3_per_s
            * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K;
        // (The stream's removed load is the negative of the heat the stream
        // added, ṁ·c_p·(t_s − T_new) — the sign the inverse query's contract
        // takes: `t_supply = t_zone − q_removed / (ṁ·c_p)`.)
        let load_w = -(capacitance / dt * (temperature_after - temperature_before)
            - zone.envelope_conductance_w_per_k * (zone.outdoor_temperature_c - temperature_after)
            - zone.internal_gain_w);

        // Inverse query: the supply temperature that holds the zone at the
        // temperature the forward step produced.
        let required = zone
            .supply_temperature_for_target(temperature_after, load_w, supply_flow_m3_per_s, None)
            .expect("the inverse query is valid");
        assert!(
            (required - supply_temperature_c).abs() < 1.0e-9,
            "expected the supply temperature {supply_temperature_c}, got {required}"
        );

        // A loadless query returns the target temperature itself.
        let unchanged = zone
            .supply_temperature_for_target(temperature_after, 0.0, supply_flow_m3_per_s, None)
            .expect("the inverse query is valid");
        assert!((unchanged - temperature_after).abs() < 1.0e-12);
    }

    /// A loop with an outdoor-air mixing box: the outdoor air (12 °C,
    /// `W = 0.004 kg/kg`) is mixed with the return air at 25 % of the
    /// supply dry-air mass flow.
    fn mixing_network() -> AirLoopNetwork {
        let outdoor_air = MoistAirStream::new(12.0, 0.004).expect("the outdoor-air state is valid");
        let mixer = OutdoorAirMixer::new(outdoor_air, 0.25).expect("the mixing box is valid");
        reference_network().with_outdoor_air_mixer(mixer)
    }

    /// A zone that sits above the thermostat band, so the coil runs and the
    /// terminal damper is fully open (`Q = √(300/(600 + 800)) m³/s`).
    fn cooling_zone() -> ZoneAirNode {
        ZoneAirNode::new(23.0, 300.0, 200.0, 35.0, 500.0, 13.0, 35.0)
            .expect("the cooling zone is valid")
    }

    #[test]
    fn outdoor_air_mixer_forms_the_analytic_mixed_state() {
        // Outdoor air 10 °C / 0.003 kg/kg, return air 24 °C / 0.009 kg/kg,
        // mixed at 30 % of the outdoor air.
        let outdoor_air = MoistAirStream::new(10.0, 0.003).expect("the outdoor air is valid");
        let return_air = MoistAirStream::new(24.0, 0.009).expect("the return air is valid");
        let mixer = OutdoorAirMixer::new(outdoor_air, 0.3).expect("the mixing box is valid");

        let mixed = mixer.mix(&return_air).expect("the mixture is formed");

        // Analytic moisture balance: W_mix = 0.3·0.003 + 0.7·0.009 = 0.0072.
        assert!((mixed.humidity_ratio_kg_per_kg_dry_air - 0.0072).abs() < 1.0e-12);
        // Analytic enthalpy balance:
        //   h_oa  = 1.006·10 + 0.003·(2501 + 1.86·10) = 17.6188 kJ/kg
        //   h_ret = 1.006·24 + 0.009·(2501 + 1.86·24) = 47.05476 kJ/kg
        //   h_mix = 0.3·h_oa + 0.7·h_ret = 38.223972 kJ/kg
        //   T_mix = (h_mix − 2501·W_mix)/(1.006 + 1.86·W_mix) = 19.832186244349572 °C
        let expected_temperature_c = 19.832_186_244_349_572;
        assert!(
            (mixed.temperature_c - expected_temperature_c).abs() < 1.0e-12,
            "expected {expected_temperature_c} °C, got {}",
            mixed.temperature_c
        );
        // The mixture lies between the two streams in temperature.
        assert!(mixed.temperature_c > outdoor_air.temperature_c);
        assert!(mixed.temperature_c < return_air.temperature_c);

        // The end points of the mixing range reproduce the streams.
        let all_outdoor = OutdoorAirMixer::new(outdoor_air, 1.0)
            .expect("the mixing box is valid")
            .mix(&return_air)
            .expect("the mixture is formed");
        assert!((all_outdoor.temperature_c - outdoor_air.temperature_c).abs() < 1.0e-12);
        assert!(
            (all_outdoor.humidity_ratio_kg_per_kg_dry_air
                - outdoor_air.humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1.0e-12
        );
        let all_return = OutdoorAirMixer::new(outdoor_air, 0.0)
            .expect("the mixing box is valid")
            .mix(&return_air)
            .expect("the mixture is formed");
        assert!((all_return.temperature_c - return_air.temperature_c).abs() < 1.0e-12);
        assert!(
            (all_return.humidity_ratio_kg_per_kg_dry_air
                - return_air.humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1.0e-12
        );

        // A flow-based mixture is the fraction-based mixture of the same
        // split (constant-density dry-air mass approximation).
        let by_flow = MoistAirStream::mix_by_flow(&outdoor_air, &return_air, 0.3, 0.7)
            .expect("the mixture is formed");
        assert!((by_flow.temperature_c - mixed.temperature_c).abs() < 1.0e-12);
        assert!(
            (by_flow.humidity_ratio_kg_per_kg_dry_air - mixed.humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn zone_humidity_state_evolves_and_closes_the_moisture_balance() {
        // The zone starts at 23 °C (above the thermostat band), so the coil
        // runs and the terminal damper is fully open.
        let thermostat = ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        let zone = cooling_zone()
            .with_humidity_ratio(0.009)
            .expect("the humidity ratio is valid")
            .with_latent_internal_gain(2000.0)
            .expect("the latent gain is valid");
        let mut air_loop =
            PressureDrivenAirLoop::new(mixing_network(), thermostat, zone, "terminal_damper")
                .expect("the air loop assembles");

        let dt = 60.0;
        let humidity_before = air_loop.zone().humidity_ratio_kg_per_kg_dry_air;
        let result = air_loop.step(dt).expect("the loop steps");

        // The supply stream is the outdoor-air/return mixture:
        // W_supply = 0.25·0.004 + 0.75·0.009 = 0.00775 kg/kg.
        assert!(
            (result.supply_humidity_ratio_kg_per_kg_dry_air - 0.00775).abs() < 1.0e-12,
            "expected 0.00775 kg/kg, got {}",
            result.supply_humidity_ratio_kg_per_kg_dry_air
        );
        // The mixed supply state carries both the temperature and the
        // humidity ratio of the mixture.
        let mixed = result
            .mixed_supply_state
            .expect("the loop reports the mixed supply state");
        assert!((mixed.humidity_ratio_kg_per_kg_dry_air - 0.00775).abs() < 1.0e-12);
        assert!(mixed.temperature_c > 12.0 && mixed.temperature_c < 23.0);

        // Analytic backward-Euler solution of the zone moisture balance:
        //   m_da = 1.2·300 = 360 kg, ṁ_sup = 1.2·Q, Q = √(300/(600+800))
        //   W_new = (m_da/dt·W_old + ṁ_sup·W_sup + W_latent/h_fg)/(m_da/dt + ṁ_sup)
        //        = 0.009016065163706656 kg/kg.
        assert!(
            (result.zone_humidity_ratio_kg_per_kg_dry_air - 0.009_016_065_163_706_656).abs()
                < 1.0e-12,
            "expected 0.009016065163706656 kg/kg, got {}",
            result.zone_humidity_ratio_kg_per_kg_dry_air
        );

        // The moisture balance closes: the accumulation of water vapour in
        // the zone equals the moisture carried in by the supply stream plus
        // the latent internal gain, minus the moisture carried out.
        let dry_air_mass = air_loop.zone().dry_air_mass_kg();
        let supply_dry_air_mass_flow = STANDARD_AIR_DENSITY_KG_PER_M3 * result.supply_flow_m3_per_s;
        let latent_mass_flow = 2000.0 / LATENT_HEAT_VAPORIZATION_0C_J_PER_KG;
        let accumulation =
            dry_air_mass * (result.zone_humidity_ratio_kg_per_kg_dry_air - humidity_before) / dt;
        let net = supply_dry_air_mass_flow * result.supply_humidity_ratio_kg_per_kg_dry_air
            - supply_dry_air_mass_flow * result.zone_humidity_ratio_kg_per_kg_dry_air
            + latent_mass_flow;
        assert!(
            (accumulation - net).abs() < 1.0e-12,
            "moisture balance residual {} kg/s",
            (accumulation - net).abs()
        );

        // The zone humidity state is carried by the node and by the zone
        // object of the loop.
        assert!(
            (air_loop.zone().humidity_ratio_kg_per_kg_dry_air
                - result.zone_humidity_ratio_kg_per_kg_dry_air)
                .abs()
                < 1.0e-15
        );
    }

    #[test]
    fn latent_internal_gain_moistens_the_zone_without_outdoor_air() {
        // Without a mixing box the supply stream is the return stream, so
        // the only moisture source is the latent internal gain.
        let thermostat = ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        let zone = reference_zone()
            .with_humidity_ratio(0.009)
            .expect("the humidity ratio is valid")
            .with_latent_internal_gain(2000.0)
            .expect("the latent gain is valid");
        let mut air_loop =
            PressureDrivenAirLoop::new(reference_network(), thermostat, zone, "terminal_damper")
                .expect("the air loop assembles");

        let dt = 60.0;
        let humidity_before = air_loop.zone().humidity_ratio_kg_per_kg_dry_air;
        let result = air_loop.step(dt).expect("the loop steps");

        // The supply stream carries the humidity of the return stream.
        assert!((result.supply_humidity_ratio_kg_per_kg_dry_air - humidity_before).abs() < 1.0e-15);
        // The latent gain raises the zone humidity:
        //   W_new = (m_da/dt·W_old + ṁ·W_old + W_latent/h_fg)/(m_da/dt + ṁ)
        //        = 0.009127663572925932 kg/kg.
        assert!(
            (result.zone_humidity_ratio_kg_per_kg_dry_air - 0.009_127_663_572_925_932).abs()
                < 1.0e-12,
            "expected 0.009127663572925932 kg/kg, got {}",
            result.zone_humidity_ratio_kg_per_kg_dry_air
        );
        assert!(result.zone_humidity_ratio_kg_per_kg_dry_air > humidity_before);
    }

    #[test]
    fn a_dehumidifying_coil_reports_a_coil_side_latent_load() {
        let thermostat = ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        let zone = cooling_zone()
            .with_humidity_ratio(0.009)
            .expect("the humidity ratio is valid")
            .with_conditioned_supply_humidity_ratio(0.005)
            .expect("the coil humidity ratio is valid");
        let mut air_loop =
            PressureDrivenAirLoop::new(mixing_network(), thermostat, zone, "terminal_damper")
                .expect("the air loop assembles");

        let result = air_loop.step(60.0).expect("the loop steps");

        // The coil delivers drier air than the mixed stream.
        assert!((result.supply_humidity_ratio_kg_per_kg_dry_air - 0.005).abs() < 1.0e-12);
        // The coil-side latent duty is the moisture it removes from the
        // mixed stream: ṁ·(W_mixed − W_supply)·h_fg.
        let expected_latent_load_w = STANDARD_AIR_DENSITY_KG_PER_M3
            * result.supply_flow_m3_per_s
            * (0.00775 - 0.005)
            * LATENT_HEAT_VAPORIZATION_0C_J_PER_KG;
        assert!(
            (result.coil_latent_load_w - expected_latent_load_w).abs() < 1.0e-9,
            "expected {expected_latent_load_w} W, got {}",
            result.coil_latent_load_w
        );
        assert!(result.coil_latent_load_w > 0.0);
    }

    #[test]
    fn a_loop_without_a_mixing_box_leaves_the_zone_humidity_unchanged() {
        // The historical behaviour is preserved: without a mixing box and
        // without a latent gain the zone humidity state does not move.
        let thermostat = ZoneThermostat::new(22.0, 1.0, 1.0, 0.2).expect("the thermostat is valid");
        let zone = reference_zone()
            .with_humidity_ratio(0.009)
            .expect("the humidity ratio is valid");
        let mut air_loop =
            PressureDrivenAirLoop::new(reference_network(), thermostat, zone, "terminal_damper")
                .expect("the air loop assembles");

        let result = air_loop.step(60.0).expect("the loop steps");
        assert!(
            (result.zone_humidity_ratio_kg_per_kg_dry_air - 0.009).abs() < 1.0e-15,
            "expected the zone humidity to be unchanged, got {}",
            result.zone_humidity_ratio_kg_per_kg_dry_air
        );
        assert!(result.mixed_supply_state.is_none());
        assert!(result.coil_latent_load_w == 0.0);
    }
}
