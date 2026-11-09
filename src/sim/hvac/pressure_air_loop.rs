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

/// Absolute tolerance on the loop pressure-balance residual [Pa].
///
/// The Newton iteration on `F(Q)` converges to machine precision on the
/// reference configurations; the acceptance tolerance is deliberately tight
/// so a regression in the solve cannot hide behind a loose bound.
pub const LOOP_RESIDUAL_TOLERANCE_PA: f64 = 1.0e-9;

/// Maximum number of Newton iterations used by the loop solve.
const MAX_SOLVE_ITERATIONS: usize = 200;

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
        })
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
    /// Residual of the loop pressure balance at the operating point [Pa].
    pub loop_residual_pa: f64,
    /// Residual of the pressure walk around the closed loop [Pa].
    pub pressure_closure_residual_pa: f64,
    /// Largest node mass-balance residual `|inflow − outflow|` [m³/s].
    pub node_mass_balance_residual_m3_per_s: f64,
}

impl AirLoopSolution {
    /// `true` when every reported quantity is finite.
    pub fn is_finite(&self) -> bool {
        self.flow_m3_per_s.is_finite()
            && self.fan_pressure_rise_pa.is_finite()
            && self.fan_air_power_w.is_finite()
            && self.loop_residual_pa.is_finite()
            && self.pressure_closure_residual_pa.is_finite()
            && self.node_mass_balance_residual_m3_per_s.is_finite()
            && self.node_pressures_pa.iter().all(|p| p.is_finite())
            && self.element_flows_m3_per_s.iter().all(|q| q.is_finite())
            && self
                .element_pressure_changes_pa
                .iter()
                .all(|dp| dp.is_finite())
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
        })
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

        let solution = AirLoopSolution {
            flow_m3_per_s: flow,
            node_pressures_pa,
            element_flows_m3_per_s,
            element_pressure_changes_pa,
            fan_pressure_rise_pa,
            fan_air_power_w: flow * fan_pressure_rise_pa,
            loop_residual_pa: self.loop_pressure_residual_pa(flow),
            pressure_closure_residual_pa: closure.abs(),
            node_mass_balance_residual_m3_per_s,
        };

        if !solution.is_finite() {
            return Err(AirLoopError::NonFiniteResult { field: "solution" });
        }
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
        })
    }

    /// Heat capacity of the zone air node [J/K].
    pub fn heat_capacity_j_per_k(&self) -> f64 {
        STANDARD_AIR_DENSITY_KG_PER_M3 * self.volume_m3 * DRY_AIR_SPECIFIC_HEAT_J_PER_KG_K
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
        let solution = self.network.solve()?;
        let supply_temperature = if command.coil_on {
            self.zone.conditioned_supply_temperature_c
        } else {
            self.zone.unconditioned_supply_temperature_c
        };
        let zone_temperature = self
            .zone
            .step(dt_s, solution.flow_m3_per_s, supply_temperature);
        let supply_mass_flow_kg_per_s = STANDARD_AIR_DENSITY_KG_PER_M3 * solution.flow_m3_per_s;
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
}
