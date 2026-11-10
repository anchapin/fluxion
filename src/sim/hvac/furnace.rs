//! Fuel-fired forced-air furnace equipment layer — the §10 (space-heating
//! equipment performance, "HE" cases) component of fluxion's ASHRAE 140
//! validation work.
//!
//! This is the **equipment layer** a future §10 harness will dispatch
//! through: given a heating load and a timestep, a [`Furnace`] reports the
//! heat it delivered, the fuel it burned, the electrical energy its fans and
//! its pilot drew, and keeps the cumulative fuel / fan / pilot energy that a
//! §10 report would need.
//!
//! ## Scope and status (read before using)
//!
//! * **Standalone.** Like the §9 DX layer
//!   ([`super::dx_performance_map`]), this module is a self-contained
//!   component in `src/sim/hvac/`. It is **not** wired into any simulation
//!   path, control loop, Python binding, or schema, and it changes **no**
//!   existing simulation output.
//! * **Not a §10 case generator, and not a §10 scorer.** There are no HE
//!   cases, no acceptance bands, no reference values and no scoring code
//!   here.
//! * **No field mapping.** The committed results-CSV contract
//!   (`data/reference/ashrae140/bestest_gsr_csv_contract.json`) states that
//!   the §10 reporting columns (7 fields) report what a machine *consumed*
//!   and are out of scope until the equipment layer exists. This module
//!   creates the equipment layer; it deliberately does **not** implement the
//!   reporting contract and does **not** name or map any reporting field —
//!   the field names are not fixed by the committed contract.
//!
//! ## Blocked on
//!
//! The full #4272 acceptance (HE cases generated, runnable and deterministic;
//! fuel consumption and cumulative fan energy scored against the standard's
//! references, including the Addendum a tables) is **blocked** on:
//!
//! 1. **#4285** — the normative HE case data and reference tables live in the
//!    licensed ASHRAE 140-2023 supplemental materials, which are not in this
//!    repository.
//! 2. **#3984** — the equation-based air loop (PR #4597) has not landed on
//!    `develop`, so no §10 harness can yet drive this layer through a real
//!    air path.
//!
//! This slice therefore delivers only the layer's structure and energy
//! accounting, with every case-specific parameter left as an input.
//!
//! ## What this is NOT
//!
//! * **Not the hydronic [`super::equipment::Boiler`].** A §10 furnace is a
//!   forced-air, fuel-fired, atmospheric/condensing unit: its efficiency
//!   degrades with part load, it has a supply (blower) fan, an induced-draft
//!   fan and a pilot/ignition circuit, and its fan heat is *not* part of the
//!   delivered heat. The generic hydronic `Boiler` plant model is a
//!   different machine and is left untouched by this module.
//! * **Not a fan model.** The repo's [`super::fan::Fan`] /
//!   [`super::fan::FanComponent`] model *airside* fan physics (flow,
//!   pressure rise, affinity laws, motor efficiency). A furnace's blower and
//!   induced-draft fan are only needed here as **power draw**, so this
//!   module takes explicit electrical power inputs instead of duplicating
//!   fan physics. Reusing `FanComponent` would require a flow and pressure
//!   pair per fan plus an air density, none of which the §10 energy
//!   accounting consumes; the power inputs are therefore the honest
//!   interface for this layer.
//!
//! ## Model
//!
//! All quantities are SI (W, s, kWh).
//!
//! ### Part-load ratio
//!
//! ```text
//! PLR = load_w / rated_capacity_w        (0 ≤ PLR ≤ 1)
//! ```
//!
//! The rated capacity is the furnace's heating capacity at the design
//! conditions; the load is the sensible heat the caller asks the furnace to
//! deliver over the timestep. A load above the rated capacity is an input
//! error (see *Input handling*), never a silently clamped one.
//!
//! ### Efficiency
//!
//! ```text
//! η(PLR) = rated_efficiency                              (PartLoadEfficiency::Constant)
//! η(PLR) = a + b·PLR + c·PLR² + d·PLR³                   (PartLoadEfficiency::Curve)
//! ```
//!
//! The curve form reuses the repository's existing curve machinery
//! ([`super::efficiency_curves::EfficiencyCurve`]) and evaluates its PLR
//! polynomial only (`EfficiencyCurve::evaluate_polynomial`). The curve's
//! *temperature-degradation* stage (`cop_at`) is deliberately **not** used:
//! this layer has no outdoor-temperature input, and a furnace's seasonal
//! efficiency variation is a §10 case property, not a property of the
//! machine.
//!
//! The committed default coefficient set
//! ([`super::efficiency_curves::default_ahri_coefficients`]) contains no
//! furnace-appropriate curve — its `boiler` entry is the *hydronic* boiler
//! placeholder (and the whole set is documented in-tree as placeholder
//! values pending #3942) — so the furnace's part-load behaviour is an
//! explicit constructor input rather than a borrowed curve.
//!
//! ### Energy-accounting identities
//!
//! Per timestep of duration `Δt`:
//!
//! ```text
//! delivered_heat_w = load_w                                  (the call for heat is met)
//! fuel_input_w     = delivered_heat_w / η(PLR)   if PLR > 0, else 0
//! supply_fan_w     = supply_fan_power_w         if the burner is firing, else 0
//! draft_fan_w      = draft_fan_power_w          if the burner is firing, else 0
//! total_fan_w      = supply_fan_w + draft_fan_w
//! pilot_w          = pilot_power_w              per the pilot mode (see below)
//! ```
//!
//! and the accumulators (kWh, `hours = Δt / 3600`):
//!
//! ```text
//! cumulative_fuel_kwh         += fuel_input_w        · hours / 1000
//! cumulative_supply_fan_kwh  += supply_fan_w        · hours / 1000
//! cumulative_draft_fan_kwh   += draft_fan_w         · hours / 1000
//! cumulative_fan_kwh          = cumulative_supply_fan_kwh + cumulative_draft_fan_kwh
//! cumulative_pilot_kwh        += pilot_w             · hours / 1000
//! ```
//!
//! Invariants the unit tests pin:
//!
//! * `fuel_input_w == delivered_heat_w / η(PLR)` whenever `PLR > 0`, and
//!   `fuel_input_w == 0` at `PLR == 0`.
//! * `cumulative_fan_kwh == cumulative_supply_fan_kwh + cumulative_draft_fan_kwh`
//!   at every step (the two fan streams are reported separately *and*
//!   summed).
//! * Each accumulator equals the sum of its per-step contributions (up to
//!   floating-point tolerance).
//! * Pilot / standby energy is accounted **separately** from fuel and is
//!   never folded into "efficiency": `η` describes the burner only.
//!
//! ### Fan and pilot conventions
//!
//! * **Fan runs with the call for heat.** The blower and the induced-draft
//!   fan run whenever the burner is firing (`load_w > 0`), including at low
//!   part load; they are off when there is no call for heat. This is the
//!   standard forced-air behaviour (the blower is what moves the heat) and
//!   it means fan energy is present even while the burner cycles at part
//!   load.
//! * **Pilot mode** is an explicit switch:
//!   * [`PilotMode::Standing`] — a standing pilot is lit continuously, so
//!     pilot power is drawn on *every* timestep, including timesteps with
//!     no call for heat.
//!   * [`PilotMode::Intermittent`] — the burner is lit only when there is a
//!     call for heat, so pilot/ignition power is drawn only while firing.
//!
//! ### Input handling
//!
//! One uniform policy: **invalid inputs are rejected with a typed error**
//! ([`FurnaceError`]); nothing is silently clamped and no non-finite value
//! ever enters the energy accounting.
//!
//! | Input | Behaviour |
//! |-------|-----------|
//! | `load_w` negative | [`FurnaceError::NegativeLoad`] |
//! | `load_w` above the rated capacity | [`FurnaceError::LoadExceedsRatedCapacity`] |
//! | `load_w` NaN / infinite | [`FurnaceError::NonFiniteLoad`] |
//! | `timestep_s` negative | [`FurnaceError::NegativeTimestep`] |
//! | `timestep_s` NaN / infinite | [`FurnaceError::NonFiniteTimestep`] |
//! | non-positive rated capacity / efficiency outside (0, 1] / negative or non-finite fan or pilot power | [`FurnaceError::InvalidConfiguration`] |
//! | efficiency curve evaluating outside (0, 1] at the operating PLR | [`FurnaceError::InvalidEfficiency`] |
//!
//! ### Parameter provenance
//!
//! **Every** parameter is a constructor input of [`Furnace::new`]:
//! rated capacity, rated efficiency, supply-fan power, draft-fan power,
//! pilot power and the pilot mode. No numeric default is baked into this
//! module, because no furnace parameter is derivable from this repository:
//! the §10 case specifications (the furnace description that the normative
//! HE cases prescribe) live in the licensed ASHRAE 140-2023 supplemental
//! materials tracked by **#4285** and are deliberately not reproduced,
//! approximated or guessed here.
//!
//! Consequently **no value in this module is an ASHRAE 140 §10 value**, and
//! no test in this module asserts a normative number: every expectation is
//! hand-computed from the module's own documented formulas.
//!
//! ## What remains (blocked)
//!
//! * HE case generation, and the §10 / Addendum a reference data — blocked
//!   on **#4285** (licensed supplemental materials).
//! * Wiring this layer into a simulation / air loop — blocked on
//!   **#3984** (PR #4597).
//! * §10 scoring and the results-CSV reporting field mapping — follow the
//!   layer, and are out of scope here.

use serde::{Deserialize, Serialize};

use super::efficiency_curves::EfficiencyCurve;

/// Seconds per hour — the conversion used by the accumulators.
const SECONDS_PER_HOUR: f64 = 3600.0;

/// How the furnace's pilot / ignition circuit behaves.
///
/// The two modes differ *only* in when pilot power is drawn; they never
/// change the fuel accounting for a call for heat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotMode {
    /// A standing pilot stays lit whenever the furnace is available, so
    /// pilot power is drawn on every timestep — including timesteps with no
    /// call for heat.
    Standing,
    /// Intermittent ignition: the burner is lit only when there is a call
    /// for heat, so pilot / ignition power is drawn only while the burner
    /// is firing.
    Intermittent,
}

/// How the furnace's efficiency varies with part-load ratio.
///
/// The furnace's *rated* efficiency is always the constructor's input; this
/// enum only says how the efficiency at the operating part-load ratio is
/// derived from it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartLoadEfficiency {
    /// The rated efficiency applies at every part-load ratio (no part-load
    /// degradation).
    Constant,
    /// The efficiency follows the PLR polynomial of the repository's
    /// existing curve machinery ([`EfficiencyCurve`]): `a + b·PLR + c·PLR² +
    /// d·PLR³`. Only the PLR polynomial is evaluated — the curve's
    /// temperature-degradation stage is not used, because this layer has no
    /// outdoor-temperature input.
    Curve(EfficiencyCurve),
}

/// Errors returned by the furnace layer.
///
/// Every invalid input is rejected explicitly; see the *Input handling*
/// section of the module header.
#[derive(Debug, Clone, PartialEq)]
pub enum FurnaceError {
    /// The requested heating load was negative.
    NegativeLoad,
    /// The requested heating load exceeded the furnace's rated capacity.
    LoadExceedsRatedCapacity {
        /// The rejected load [W].
        load_w: f64,
        /// The furnace's rated heating capacity [W].
        rated_capacity_w: f64,
    },
    /// The requested heating load was NaN or infinite.
    NonFiniteLoad,
    /// The timestep duration was negative.
    NegativeTimestep,
    /// The timestep duration was NaN or infinite.
    NonFiniteTimestep,
    /// The furnace was configured with an unusable parameter (for example a
    /// non-positive rated capacity, or a rated efficiency outside (0, 1]).
    InvalidConfiguration {
        /// The offending parameter's name.
        parameter: String,
        /// The offending parameter's value.
        value: f64,
    },
    /// The part-load efficiency curve evaluated to a value outside (0, 1] at
    /// the operating part-load ratio.
    InvalidEfficiency {
        /// The part-load ratio at which the curve was evaluated.
        plr: f64,
        /// The efficiency the curve produced there.
        efficiency: f64,
    },
}

impl std::fmt::Display for FurnaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FurnaceError::NegativeLoad => write!(f, "heating load is negative"),
            FurnaceError::LoadExceedsRatedCapacity {
                load_w,
                rated_capacity_w,
            } => write!(
                f,
                "heating load {load_w} W exceeds the rated capacity {rated_capacity_w} W"
            ),
            FurnaceError::NonFiniteLoad => write!(f, "heating load is not finite"),
            FurnaceError::NegativeTimestep => write!(f, "timestep duration is negative"),
            FurnaceError::NonFiniteTimestep => write!(f, "timestep duration is not finite"),
            FurnaceError::InvalidConfiguration { parameter, value } => {
                write!(f, "invalid furnace configuration: {parameter} = {value}")
            }
            FurnaceError::InvalidEfficiency { plr, efficiency } => write!(
                f,
                "part-load efficiency {efficiency} at PLR {plr} is outside (0, 1]"
            ),
        }
    }
}

impl std::error::Error for FurnaceError {}

/// Cumulative energy accumulated by a [`Furnace`] over successive timesteps.
///
/// The accumulator names are deliberately generic: they describe *what the
/// machine consumed*, not any reporting field of the §10 results contract
/// (whose field names are not fixed by the committed contract — see the
/// module header).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct FurnaceEnergyTotals {
    /// Cumulative fuel energy consumed for the heating load [kWh].
    pub cumulative_fuel_kwh: f64,
    /// Cumulative electrical energy of the supply (blower) fan [kWh].
    pub cumulative_supply_fan_kwh: f64,
    /// Cumulative electrical energy of the induced-draft fan [kWh].
    pub cumulative_draft_fan_kwh: f64,
    /// Cumulative electrical energy of all fans (supply + draft) [kWh].
    ///
    /// Always equal to `cumulative_supply_fan_kwh +
    /// cumulative_draft_fan_kwh`.
    pub cumulative_fan_kwh: f64,
    /// Cumulative electrical energy of the pilot / ignition circuit [kWh].
    pub cumulative_pilot_kwh: f64,
}

impl FurnaceEnergyTotals {
    /// Zero every accumulator.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The result of evaluating the furnace for one timestep.
///
/// All powers are time-averaged over the timestep.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FurnaceEvaluation {
    /// Heat delivered to the caller over the timestep [W].
    pub delivered_heat_w: f64,
    /// Fuel energy input for the delivered heat over the timestep [W].
    ///
    /// Zero when there is no call for heat.
    pub fuel_input_w: f64,
    /// Supply (blower) fan electrical power draw [W].
    pub supply_fan_power_w: f64,
    /// Induced-draft fan electrical power draw [W].
    pub draft_fan_power_w: f64,
    /// Total fan electrical power draw (supply + draft) [W].
    pub total_fan_power_w: f64,
    /// Pilot / ignition electrical power draw [W].
    pub pilot_power_w: f64,
    /// The operating part-load ratio `load_w / rated_capacity_w`.
    pub plr: f64,
    /// The burner efficiency actually used at that part-load ratio.
    pub efficiency: f64,
    /// Whether the burner was firing (i.e. there was a call for heat).
    pub burner_firing: bool,
}

/// A fuel-fired forced-air furnace.
///
/// The furnace is a pure accounting component: it has no state beyond its
/// cumulative energy totals, and it is not connected to any simulation
/// loop. Every physical parameter is a constructor input (see the module
/// header's *Parameter provenance* section).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Furnace {
    /// Equipment identifier.
    pub id: String,
    /// Rated heating capacity at the design conditions [W].
    pub rated_capacity_w: f64,
    /// Rated burner efficiency at the design conditions (AFUE-style, (0, 1]).
    pub rated_efficiency: f64,
    /// How the efficiency varies with part-load ratio.
    pub part_load_efficiency: PartLoadEfficiency,
    /// Supply (blower) fan electrical power while firing [W].
    pub supply_fan_power_w: f64,
    /// Induced-draft fan electrical power while firing [W].
    pub draft_fan_power_w: f64,
    /// Pilot / ignition electrical power [W].
    pub pilot_power_w: f64,
    /// Whether the pilot is standing or intermittent.
    pub pilot_mode: PilotMode,
    /// Cumulative energy accumulated over the timesteps evaluated so far.
    pub energy: FurnaceEnergyTotals,
}

impl Furnace {
    /// Create a furnace from its physical parameters.
    ///
    /// No numeric default is provided for any parameter: the furnace
    /// description used by the §10 cases lives in the licensed ASHRAE
    /// 140-2023 supplemental materials (#4285), so every value is supplied
    /// by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        rated_capacity_w: f64,
        rated_efficiency: f64,
        supply_fan_power_w: f64,
        draft_fan_power_w: f64,
        pilot_power_w: f64,
        pilot_mode: PilotMode,
    ) -> Self {
        Self {
            id,
            rated_capacity_w,
            rated_efficiency,
            part_load_efficiency: PartLoadEfficiency::Constant,
            supply_fan_power_w,
            draft_fan_power_w,
            pilot_power_w,
            pilot_mode,
            energy: FurnaceEnergyTotals::default(),
        }
    }

    /// Replace the part-load efficiency description (constant by default).
    pub fn with_part_load_efficiency(mut self, part_load_efficiency: PartLoadEfficiency) -> Self {
        self.part_load_efficiency = part_load_efficiency;
        self
    }

    /// The cumulative energy totals accumulated so far.
    pub fn energy_totals(&self) -> &FurnaceEnergyTotals {
        &self.energy
    }

    /// Zero every cumulative energy accumulator.
    pub fn reset_accumulators(&mut self) {
        self.energy.reset();
    }

    /// The burner efficiency at a given part-load ratio.
    ///
    /// `PartLoadEfficiency::Constant` returns the rated efficiency;
    /// `PartLoadEfficiency::Curve` evaluates the curve's PLR polynomial.
    /// An efficiency outside (0, 1] is rejected as
    /// [`FurnaceError::InvalidEfficiency`].
    pub fn efficiency_at(&self, plr: f64) -> Result<f64, FurnaceError> {
        let efficiency = match &self.part_load_efficiency {
            PartLoadEfficiency::Constant => self.rated_efficiency,
            PartLoadEfficiency::Curve(curve) => curve.evaluate_polynomial(plr),
        };
        if !efficiency.is_finite() || efficiency <= 0.0 || efficiency > 1.0 {
            return Err(FurnaceError::InvalidEfficiency { plr, efficiency });
        }
        Ok(efficiency)
    }

    /// Evaluate the furnace for one timestep and accumulate the energy.
    ///
    /// `load_w` is the sensible heat the caller asks the furnace to deliver
    /// over the timestep; `timestep_s` is the timestep duration in seconds.
    /// The returned powers are time-averages over the timestep, and the
    /// cumulative totals in [`Furnace::energy_totals`] are updated in place.
    ///
    /// See the module header for the energy-accounting identities and for
    /// the input-handling policy.
    pub fn evaluate(
        &mut self,
        load_w: f64,
        timestep_s: f64,
    ) -> Result<FurnaceEvaluation, FurnaceError> {
        let evaluation = self.compute(load_w)?;
        self.accumulate(&evaluation, timestep_s)?;
        Ok(evaluation)
    }

    /// Evaluate the furnace for one timestep *without* accumulating energy.
    ///
    /// Useful for probes and for tests that want the per-step numbers
    /// without touching the accumulators.
    pub fn preview(&self, load_w: f64) -> Result<FurnaceEvaluation, FurnaceError> {
        self.compute(load_w)
    }

    /// Compute the per-step evaluation (no accumulation).
    fn compute(&self, load_w: f64) -> Result<FurnaceEvaluation, FurnaceError> {
        if !load_w.is_finite() {
            return Err(FurnaceError::NonFiniteLoad);
        }
        if load_w < 0.0 {
            return Err(FurnaceError::NegativeLoad);
        }
        if !self.rated_capacity_w.is_finite() || self.rated_capacity_w <= 0.0 {
            return Err(FurnaceError::InvalidConfiguration {
                parameter: "rated_capacity_w".to_string(),
                value: self.rated_capacity_w,
            });
        }
        if load_w > self.rated_capacity_w {
            return Err(FurnaceError::LoadExceedsRatedCapacity {
                load_w,
                rated_capacity_w: self.rated_capacity_w,
            });
        }
        if !self.rated_efficiency.is_finite()
            || self.rated_efficiency <= 0.0
            || self.rated_efficiency > 1.0
        {
            return Err(FurnaceError::InvalidConfiguration {
                parameter: "rated_efficiency".to_string(),
                value: self.rated_efficiency,
            });
        }
        // Fan and pilot powers are trusted constructor inputs only in the
        // sense that the caller supplies them; a negative or non-finite
        // power must never enter the energy accounting.
        for (parameter, value) in [
            ("supply_fan_power_w", self.supply_fan_power_w),
            ("draft_fan_power_w", self.draft_fan_power_w),
            ("pilot_power_w", self.pilot_power_w),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(FurnaceError::InvalidConfiguration {
                    parameter: parameter.to_string(),
                    value,
                });
            }
        }

        let plr = load_w / self.rated_capacity_w;
        let efficiency = self.efficiency_at(plr)?;
        let burner_firing = load_w > 0.0;

        // Fuel is zero when there is no call for heat; the pilot circuit is
        // accounted separately and never folded into the efficiency.
        let fuel_input_w = if burner_firing {
            load_w / efficiency
        } else {
            0.0
        };

        // The blower and the induced-draft fan run with the call for heat.
        let supply_fan_power_w = if burner_firing {
            self.supply_fan_power_w
        } else {
            0.0
        };
        let draft_fan_power_w = if burner_firing {
            self.draft_fan_power_w
        } else {
            0.0
        };

        let pilot_power_w = match self.pilot_mode {
            PilotMode::Standing => self.pilot_power_w,
            PilotMode::Intermittent => {
                if burner_firing {
                    self.pilot_power_w
                } else {
                    0.0
                }
            }
        };

        Ok(FurnaceEvaluation {
            delivered_heat_w: load_w,
            fuel_input_w,
            supply_fan_power_w,
            draft_fan_power_w,
            total_fan_power_w: supply_fan_power_w + draft_fan_power_w,
            pilot_power_w,
            plr,
            efficiency,
            burner_firing,
        })
    }

    /// Add one timestep's evaluation to the cumulative energy totals.
    fn accumulate(
        &mut self,
        evaluation: &FurnaceEvaluation,
        timestep_s: f64,
    ) -> Result<(), FurnaceError> {
        if !timestep_s.is_finite() {
            return Err(FurnaceError::NonFiniteTimestep);
        }
        if timestep_s < 0.0 {
            return Err(FurnaceError::NegativeTimestep);
        }
        let hours = timestep_s / SECONDS_PER_HOUR;
        self.energy.cumulative_fuel_kwh += evaluation.fuel_input_w * hours / 1000.0;
        self.energy.cumulative_supply_fan_kwh += evaluation.supply_fan_power_w * hours / 1000.0;
        self.energy.cumulative_draft_fan_kwh += evaluation.draft_fan_power_w * hours / 1000.0;
        self.energy.cumulative_fan_kwh =
            self.energy.cumulative_supply_fan_kwh + self.energy.cumulative_draft_fan_kwh;
        self.energy.cumulative_pilot_kwh += evaluation.pilot_power_w * hours / 1000.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A furnace with round numbers so every expectation below can be
    /// hand-computed from the module's documented formulas.
    fn sample_furnace() -> Furnace {
        Furnace::new(
            "test-furnace".to_string(),
            10_000.0, // rated capacity [W]
            0.80,     // rated efficiency
            300.0,    // supply fan [W]
            100.0,    // draft fan [W]
            10.0,     // pilot [W]
            PilotMode::Standing,
        )
    }

    /// A curve whose PLR polynomial is `0.75 + 0.10·PLR`.
    fn linear_curve() -> PartLoadEfficiency {
        PartLoadEfficiency::Curve(EfficiencyCurve::new([0.75, 0.10, 0.0, 0.0], 0.0, 0.0))
    }

    #[test]
    fn full_load_fuel_is_delivered_over_rated_efficiency() {
        let mut furnace = sample_furnace();
        let step = furnace.evaluate(10_000.0, 3600.0).expect("valid step");

        // PLR = 10000 / 10000 = 1, η = 0.80 (constant part-load behaviour).
        assert_eq!(step.plr, 1.0);
        assert_eq!(step.efficiency, 0.80);
        // Delivered heat is the requested load.
        assert_eq!(step.delivered_heat_w, 10_000.0);
        // Fuel = delivered / η = 10000 / 0.8 = 12500 W.
        assert!((step.fuel_input_w - 12_500.0).abs() < 1e-9);
        // Fans run with the call for heat: 300 + 100 = 400 W.
        assert_eq!(step.supply_fan_power_w, 300.0);
        assert_eq!(step.draft_fan_power_w, 100.0);
        assert_eq!(step.total_fan_power_w, 400.0);
        // Standing pilot draws continuously.
        assert_eq!(step.pilot_power_w, 10.0);

        // One hour at 12500 W is 12.5 kWh of fuel; 400 W of fans is
        // 0.4 kWh; 10 W of pilot is 0.01 kWh.
        let totals = *furnace.energy_totals();
        assert!((totals.cumulative_fuel_kwh - 12.5).abs() < 1e-9);
        assert!((totals.cumulative_supply_fan_kwh - 0.3).abs() < 1e-9);
        assert!((totals.cumulative_draft_fan_kwh - 0.1).abs() < 1e-9);
        assert!((totals.cumulative_fan_kwh - 0.4).abs() < 1e-9);
        assert!((totals.cumulative_pilot_kwh - 0.01).abs() < 1e-9);
    }

    #[test]
    fn zero_load_has_no_fuel_but_still_draws_the_standing_pilot() {
        let mut furnace = sample_furnace();
        let step = furnace.evaluate(0.0, 3600.0).expect("valid step");

        assert_eq!(step.plr, 0.0);
        assert_eq!(step.delivered_heat_w, 0.0);
        // No call for heat → no fuel, and the fans are off.
        assert_eq!(step.fuel_input_w, 0.0);
        assert_eq!(step.supply_fan_power_w, 0.0);
        assert_eq!(step.draft_fan_power_w, 0.0);
        assert_eq!(step.total_fan_power_w, 0.0);
        // The standing pilot keeps burning.
        assert_eq!(step.pilot_power_w, 10.0);

        let totals = *furnace.energy_totals();
        assert_eq!(totals.cumulative_fuel_kwh, 0.0);
        assert_eq!(totals.cumulative_fan_kwh, 0.0);
        assert!((totals.cumulative_pilot_kwh - 0.01).abs() < 1e-9);
    }

    #[test]
    fn part_load_efficiency_follows_the_supplied_curve() {
        let mut furnace = sample_furnace().with_part_load_efficiency(linear_curve());
        let step = furnace.evaluate(5_000.0, 3600.0).expect("valid step");

        // PLR = 5000 / 10000 = 0.5; η = 0.75 + 0.10 * 0.5 = 0.80.
        assert!((step.plr - 0.5).abs() < 1e-12);
        assert!((step.efficiency - 0.80).abs() < 1e-12);
        // Fuel = 5000 / 0.8 = 6250 W.
        assert!((step.fuel_input_w - 6_250.0).abs() < 1e-9);
        // Delivered heat scales with the call for heat.
        assert!((step.delivered_heat_w - 5_000.0).abs() < 1e-9);
        // 6250 W for one hour is 6.25 kWh.
        assert!((furnace.energy_totals().cumulative_fuel_kwh - 6.25).abs() < 1e-9);
    }

    #[test]
    fn fan_energy_is_reported_separately_and_summed() {
        let mut furnace = sample_furnace();
        let step = furnace.evaluate(2_500.0, 1800.0).expect("valid step");

        // The burner fires at PLR = 0.25, so the fans run.
        assert!(step.burner_firing);
        assert!((step.plr - 0.25).abs() < 1e-12);
        assert_eq!(step.supply_fan_power_w, 300.0);
        assert_eq!(step.draft_fan_power_w, 100.0);
        assert_eq!(step.total_fan_power_w, 400.0);

        let totals = *furnace.energy_totals();
        // Half an hour: 300 W → 0.15 kWh, 100 W → 0.05 kWh.
        assert!((totals.cumulative_supply_fan_kwh - 0.15).abs() < 1e-9);
        assert!((totals.cumulative_draft_fan_kwh - 0.05).abs() < 1e-9);
        // The cumulative fan energy is the sum of the two streams.
        assert!((totals.cumulative_fan_kwh - 0.20).abs() < 1e-9);
        assert!(
            (totals.cumulative_fan_kwh
                - (totals.cumulative_supply_fan_kwh + totals.cumulative_draft_fan_kwh))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn standing_pilot_accumulates_at_all_times_while_intermittent_only_when_firing() {
        let mut standing = sample_furnace();
        standing.evaluate(0.0, 3600.0).expect("valid step");
        standing.evaluate(4_000.0, 3600.0).expect("valid step");
        let standing_totals = *standing.energy_totals();
        // Two hours of pilot at 10 W = 0.02 kWh.
        assert!((standing_totals.cumulative_pilot_kwh - 0.02).abs() < 1e-9);

        let mut intermittent = Furnace::new(
            "test-furnace".to_string(),
            10_000.0,
            0.80,
            300.0,
            100.0,
            10.0,
            PilotMode::Intermittent,
        );
        let idle = intermittent.evaluate(0.0, 3600.0).expect("valid step");
        assert_eq!(idle.pilot_power_w, 0.0);
        let firing = intermittent.evaluate(4_000.0, 3600.0).expect("valid step");
        assert_eq!(firing.pilot_power_w, 10.0);
        let intermittent_totals = *intermittent.energy_totals();
        // Only the firing hour contributes pilot energy.
        assert!((intermittent_totals.cumulative_pilot_kwh - 0.01).abs() < 1e-9);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let mut furnace = sample_furnace();

        assert_eq!(
            furnace.evaluate(-1.0, 3600.0).unwrap_err(),
            FurnaceError::NegativeLoad
        );
        assert_eq!(
            furnace.evaluate(f64::NAN, 3600.0).unwrap_err(),
            FurnaceError::NonFiniteLoad
        );
        assert_eq!(
            furnace.evaluate(f64::INFINITY, 3600.0).unwrap_err(),
            FurnaceError::NonFiniteLoad
        );
        assert_eq!(
            furnace.evaluate(10_001.0, 3600.0).unwrap_err(),
            FurnaceError::LoadExceedsRatedCapacity {
                load_w: 10_001.0,
                rated_capacity_w: 10_000.0,
            }
        );
        assert_eq!(
            furnace.evaluate(1_000.0, -1.0).unwrap_err(),
            FurnaceError::NegativeTimestep
        );
        assert_eq!(
            furnace.evaluate(1_000.0, f64::NAN).unwrap_err(),
            FurnaceError::NonFiniteTimestep
        );

        // A rejected step leaves the accumulators untouched.
        let totals = *furnace.energy_totals();
        assert_eq!(totals.cumulative_fuel_kwh, 0.0);
        assert_eq!(totals.cumulative_fan_kwh, 0.0);
        assert_eq!(totals.cumulative_pilot_kwh, 0.0);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let mut zero_capacity = sample_furnace();
        zero_capacity.rated_capacity_w = 0.0;
        assert!(matches!(
            zero_capacity.evaluate(0.0, 3600.0),
            Err(FurnaceError::InvalidConfiguration { .. })
        ));

        let mut bad_efficiency = sample_furnace();
        bad_efficiency.rated_efficiency = 1.5;
        assert!(matches!(
            bad_efficiency.evaluate(0.0, 3600.0),
            Err(FurnaceError::InvalidConfiguration { .. })
        ));

        // A curve that evaluates above 1.0 is rejected at the operating PLR.
        let mut bad_curve = sample_furnace().with_part_load_efficiency(PartLoadEfficiency::Curve(
            EfficiencyCurve::new([0.90, 0.40, 0.0, 0.0], 0.0, 0.0),
        ));
        assert!(matches!(
            bad_curve.evaluate(5_000.0, 3600.0),
            Err(FurnaceError::InvalidEfficiency { .. })
        ));

        // Negative or non-finite fan / pilot powers are rejected, so no
        // non-finite value can enter the energy accounting through them.
        let mut bad_fan = sample_furnace();
        bad_fan.supply_fan_power_w = -50.0;
        assert!(matches!(
            bad_fan.evaluate(1_000.0, 3600.0),
            Err(FurnaceError::InvalidConfiguration { .. })
        ));
        let mut bad_pilot = sample_furnace();
        bad_pilot.pilot_power_w = f64::NAN;
        assert!(matches!(
            bad_pilot.evaluate(1_000.0, 3600.0),
            Err(FurnaceError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn cumulative_energy_matches_the_sum_of_the_per_step_contributions() {
        let mut furnace = sample_furnace().with_part_load_efficiency(linear_curve());
        let steps: [(f64, f64); 4] = [
            (0.0, 3600.0),
            (7_500.0, 1800.0),
            (2_500.0, 900.0),
            (10_000.0, 3600.0),
        ];

        let mut fuel_kwh = 0.0;
        let mut supply_kwh = 0.0;
        let mut draft_kwh = 0.0;
        let mut pilot_kwh = 0.0;
        for (load_w, timestep_s) in steps {
            let step = furnace.evaluate(load_w, timestep_s).expect("valid step");
            let hours = timestep_s / 3600.0;
            fuel_kwh += step.fuel_input_w * hours / 1000.0;
            supply_kwh += step.supply_fan_power_w * hours / 1000.0;
            draft_kwh += step.draft_fan_power_w * hours / 1000.0;
            pilot_kwh += step.pilot_power_w * hours / 1000.0;
        }

        let totals = *furnace.energy_totals();
        assert!((totals.cumulative_fuel_kwh - fuel_kwh).abs() < 1e-9);
        assert!((totals.cumulative_supply_fan_kwh - supply_kwh).abs() < 1e-9);
        assert!((totals.cumulative_draft_fan_kwh - draft_kwh).abs() < 1e-9);
        assert!((totals.cumulative_fan_kwh - (supply_kwh + draft_kwh)).abs() < 1e-9);
        assert!((totals.cumulative_pilot_kwh - pilot_kwh).abs() < 1e-9);

        // Reset clears every accumulator.
        furnace.reset_accumulators();
        let reset_totals = *furnace.energy_totals();
        assert_eq!(reset_totals.cumulative_fuel_kwh, 0.0);
        assert_eq!(reset_totals.cumulative_fan_kwh, 0.0);
        assert_eq!(reset_totals.cumulative_pilot_kwh, 0.0);
    }

    #[test]
    fn preview_does_not_accumulate() {
        let furnace = sample_furnace();
        let step = furnace.preview(10_000.0).expect("valid step");
        assert!((step.fuel_input_w - 12_500.0).abs() < 1e-9);
        let totals = *furnace.energy_totals();
        assert_eq!(totals.cumulative_fuel_kwh, 0.0);
        assert_eq!(totals.cumulative_fan_kwh, 0.0);
    }

    #[test]
    fn fuel_identity_holds_for_every_step_of_a_run() {
        let mut furnace = sample_furnace().with_part_load_efficiency(linear_curve());
        for load_w in [0.0, 1_000.0, 4_321.0, 10_000.0] {
            let step = furnace.evaluate(load_w, 600.0).expect("valid step");
            if step.plr > 0.0 {
                assert!(
                    (step.fuel_input_w - step.delivered_heat_w / step.efficiency).abs() < 1e-9,
                    "fuel identity broken at load {load_w}"
                );
            } else {
                assert_eq!(step.fuel_input_w, 0.0);
            }
        }
    }
}
