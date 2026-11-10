//! ASHRAE 140-2023 §11 air-side HVAC engine-run harness — Issue #4274.
//!
//! # What this module does
//!
//! The §11 *targets* exist: `fluxion-core/src/qas_airside.rs` is the QASv2
//! port (the merged quasi-analytical solution of ASHRAE RP-865, Yuill and
//! Haberl 2002, published in Standard 140-2023 Informative Annex B17 §B17.3.2)
//! and `data/reference/ashrae140/section11_qas_targets.json` carries its
//! published outputs for cases 1–6 × FC/SZ/CAV/VAV. What did **not** exist is
//! any run of *fluxion's own* air-side model against those targets — the gap
//! named in `docs/compliance/section_coverage.md` ("Analytical targets
//! implemented, test set not. fluxion's own air-side model is not yet run
//! against those targets.").
//!
//! This module closes it:
//!
//! 1. Every QAS case (1–6) × system (FC, SZ, CAV, VAV) — all 24 cells — is
//!    dispatched through fluxion's own air-side components
//!    (`fluxion::sim::hvac::CoolingCoil`, `HeatingCoilComponent`,
//!    `FanComponent`) and fluxion-core psychrometrics, in the QAS
//!    configuration (economizer off, no return-duct heat gain).
//! 2. Every quantity the QAS targets publish for that cell is scored
//!    against the committed target value. Rows that fall outside the
//!    comparison tolerance are recorded as **known-fail rows with the real
//!    numbers** (engine value, target, gap) — never hidden.
//! 3. The recorded rows live in
//!    `data/section11_engine_run_results.json`, which this module verifies
//!    against the live engine run on **every** test execution, so a recorded
//!    number cannot go stale.
//!
//! Nothing here fits a band to the engine output: the QAS targets are point
//! values, and the comparison tolerances below are declared up front as
//! engineering comparison tolerances for an independently written model (they
//! are not derived from the observed differences).
//!
//! # Case 2 caveat (carried verbatim)
//!
//! > the workbook's Test Case Inputs table publishes no dew point for case 2,
//! > only a wet bulb of 19.756 degF; the dew point here is derived via the
//! > sub-freezing adiabatic saturation relation, so case 2 is provisional
//! > until confirmed against the workbook
//!
//! That caveat is reproduced verbatim from
//! `data/reference/ashrae140/section11_qas_targets.json` (`_case_2_caveat`)
//! and is attached to every case-2 row of the recorded results file.
//!
//! # Step-0 mapping: what the QAS publishes vs. what fluxion can produce
//!
//! | QAS quantity (unit) | FC | SZ | CAV | VAV | fluxion API that produces it |
//! |---|---|---|---|---|---|
//! | `heating_coil` (Btu/h) | ✓ | ✓ | – | – | `HeatingCoilComponent::compute_heating_capacity` (`LeavingTempSetpoint`) |
//! | `preheat_coil` (Btu/h) | – | – | ✓ | ✓ | `HeatingCoilComponent::compute_heating_capacity` (`LeavingTempSetpoint`) |
//! | `zone1_reheat`, `zone2_reheat`, `total_reheat` (Btu/h) | – | – | ✓ | ✓ | `HeatingCoilComponent::compute_heating_capacity` (`LeavingTempSetpoint`) |
//! | `total_heat` (Btu/h) | – | – | ✓ | ✓ | preheat + reheat coils (sum of the two above) |
//! | `cooling_sensible`, `cooling_latent`, `cooling_total` (Btu/h) | ✓ | ✓ | ✓ | ✓ | `CoolingCoil::compute_cooling_capacity` (bypass-factor model) |
//! | `mixed_air_temp` (degF) | ✓ | ✓ | ✓ | ✓ | *no dedicated fluxion mixing component exists* — harness-level dry-air mass / enthalpy balance over `MoistAirState` (see below) |
//! | `supply_air_temp` (degF) | ✓ | ✓ | ✓ | ✓ | coil leaving state + `FanComponent::shaft_power` fan air-temperature rise |
//! | `return_air_temp` (degF) | – | ✓ | – | – | `FanComponent::shaft_power` (return-fan rise over the zone state) |
//! | `zone1_supply_temp`, `zone2_supply_temp` (degF) | – | – | ✓ | ✓ | zone load balance over the engine's supply mass flow + `MoistAirState` enthalpy |
//! | `supply_mass_flow` (lbm/min) | ✓ | ✓ | – | – | `FanComponent::mass_flow_rate` / `MoistAirState` density |
//!
//! Component-by-component, from the Step-0 survey of `src/sim/hvac/`:
//!
//! * `cooling_coil.rs` — `CoolingCoil` (bypass-factor psychrometrics) is the
//!   only fluxion component that produces a **latent** air-side load, so it is
//!   the cooling-coil path for all four systems. Its leaving-air state is the
//!   bypass-factor interpolation toward the apparatus dew point, so a coil
//!   with `bypass_factor = 0` leaves the air saturated at the ADP — the same
//!   ideal-coil leaving state the QAS assumes.
//! * `heating_coil.rs` — `HeatingCoilComponent` is a sensible-only coil
//!   controlled to a leaving dry-bulb setpoint; it is the heating-coil,
//!   preheat-coil and zone-reheat path (the QAS coils are sensible-only too:
//!   the zone latent load is met by the zone air, not by the coil).
//! * `fan.rs` — `FanComponent` gives flow, pressure rise and shaft power but
//!   **no air temperature rise**. The QAS workbook's fan temperature rise is
//!   `shaft power / (ṁ·cp)` (its `WORK2HEAT` constant carries the same
//!   convention, and fluxion's own `cav_terminal.rs` documents "fan heat
//!   added to the airstream (equals shaft power)"), so the harness derives
//!   the rise from the fan's **shaft** power — the engine's own convention.
//! * `economizer.rs` — only a control predicate
//!   (`is_economizer_active`) and a free-cooling capacity estimate; the QAS
//!   cases are run with the economizer **off**, so this module is not on the
//!   dispatch path (documented, not silently skipped).
//! * `cav_terminal.rs` / `vav_terminal.rs` — terminal units that own their
//!   **own** cooling coil, fan and (optional) heating coil. The QAS CAV/VAV
//!   topology is a central air-handling unit (preheat coil + cooling coil +
//!   supply fan) with a separate per-zone reheat coil, so the terminal-unit
//!   components do not match that topology and are not used for the §11
//!   dispatch; the reheat coils are `HeatingCoilComponent`s.
//! * `airside_state.rs` — `MoistAirState` is the state type that carries the
//!   engine's psychrometrics (humidity ratio, enthalpy, density) and is the
//!   currency of the dispatch.
//! * `airside_coupling.rs` — the 9R4C envelope coupler; it is a
//!   *transient* coupling to the envelope solver and produces no §11 steady
//!   state, so it is not on the dispatch path.
//!
//! fluxion has **no** §11-shaped airside system object (no air-mixing
//! component, no §11 system assembly). The harness therefore assembles the §11
//! topology from fluxion's own components and fluxion-core psychrometrics,
//! and the **system-level glue** (outdoor/return air mixing, flow splits, zone
//! load balances) is harness code — it is flagged as such on every affected
//! row (`method` field). The quantities that fluxion's components decide
//! (coil loads, coil leaving states, fan rises) are the engine's own.
//!
//! # Comparison tolerances (declared up front, not fitted)
//!
//! * loads (Btu/h): `|engine − target| ≤ max(5 % of |target|, 20 Btu/h)`;
//! * temperatures (degF): `|engine − target| ≤ 1.0 degF`;
//! * supply mass flow (lbm/min): `|engine − target| ≤ 2 % of |target|`.
//!
//! These are engineering comparison tolerances for an independently written
//! model; they are not bands fitted to the observed differences and they are
//! not the Annex B example-result statistics (which the standard expressly
//! disclaims as acceptance criteria).
//!
//! # Statuses
//!
//! * `pass` — the engine value is inside the comparison tolerance;
//! * `known_fail` — the engine value is outside it; the row carries the real
//!   engine value, the target and the gap;
//! * `not_runnable` — the quantity cannot be produced by the engine's API;
//!   the row quotes the exact API gap.
//!
//! # Re-recording the results file
//!
//! ```text
//! FLUXION_SECTION11_WRITE_RESULTS=tests/validation/hvac_equipment_rated_performance/data/section11_engine_run_results.json \
//!   cargo test --test hvac_equipment_rated_performance section11
//! ```
//!
//! The re-record path is explicit and opt-in; the ordinary test run always
//! compares the live engine run against the committed file.
//!
//! # Issue #4665 — the `PressureDrivenAirLoop` dispatch was attempted and blocked
//!
//! Issue #4665 asked for the four §11 systems to be dispatched through the
//! engine's real pressure-driven air loop
//! (`src/sim/hvac/pressure_air_loop.rs`) instead of the harness-side
//! mixing / flow-split / zone-balance code documented above.
//!
//! The attempt was made against the **existing public API only** (the brief
//! forbids extending it) and it was **stopped**, because that API cannot
//! express any of the four §11 configurations. The gaps, with the API
//! evidence, are:
//!
//! | §11 quantity | harness-side function | air-loop dispatch | why it does not exist |
//! |---|---|---|---|
//! | `mixed_air_temp` (all 24 cells) | `mix_streams` | none | `AirLoopElement` has only `Fan` / `Duct` / `Damper` (`pressure_air_loop.rs:282`): the loop has no outdoor-air inlet and no mixing element, and `AirLoopNetwork` is a single closed ring (one path, one solved flow), so the outdoor/return mixture every §11 coil sees cannot be formed |
//! | `cooling_latent` (all 24 cells) | zone humidity-ratio balance | none | the air-loop module carries no humidity at all (`grep -i humid src/sim/hvac/pressure_air_loop.rs` → 0 hits): `ZoneAirNode` (`:981`) has temperatures, a volume, an envelope conductance and a scalar internal gain, and `AirLoopSolution` (`:416`) has flow/pressure/power only — the zone latent load (`r1_ql`, `r2_ql`) and the return-air humidity ratio have no representation |
//! | CAV/VAV `supply_mass_flow`, `zone1_supply_temp`, `zone2_supply_temp`, `zone1_reheat`, `zone2_reheat`, `total_reheat`, `total_heat` | `dispatch_two_zone` flow split and zone balances | none | `PressureDrivenAirLoop` holds one `ZoneAirNode`, one `ZoneThermostat` and one damper (`:1102`); the QAS CAV/VAV topology is a central AHU whose single supply flow **splits** between two zones, with a second (return) fan downstream of the merge — the ring topology accepts exactly one fan (`:505`) and has no branch nodes |
//! | VAV flow rule | `load_flow.max(m_exh)` | none | the damper position is commanded by the hysteretic on/off `ZoneThermostat` (`:863`) from two fixed positions; the QAS VAV rule (the minimum flow that meets the zone sensible load at the AHU discharge temperature, floored by the exhaust flow) is not commandable through the loop |
//! | fan air-temperature rise | `fan_temperature_rise` | partial | `AirLoopSolution::fan_air_power_w` is the **air** power (flow × pressure rise, `:713`); the QAS workbook's convention is shaft power / (ṁ·cp) at 70 % fan efficiency, and `FanCurve` carries no efficiency — a harness-side efficiency factor would be needed |
//! | `supply_mass_flow` (FC/SZ) | design flow × density | partial | the loop *solves* the flow from a quadratic fan curve plus series resistances, while the QAS *specifies* the flow and a constant pressure rise; reproducing the specified flow would mean choosing a duct resistance to hit it (a fitted value), and `AirLoopStepResult::supply_mass_flow_kg_per_s` uses the constant `STANDARD_AIR_DENSITY_KG_PER_M3 = 1.2` rather than the moist-air density at the coil leaving state |
//! | `zone1_supply_temp` / `zone2_supply_temp` | `t_zone - q/(ṁ·cp)` | none | `ZoneAirNode` takes the supply temperature as an *input* and returns the zone temperature after a transient step; the inverse the QAS asks for (the supply temperature that holds the given zone temperature under the given load) is not exposed, so it would have to be inverted in harness code — the very zone balance the issue asks to remove |
//!
//! The full mapping, the evidence and the smallest API extension that would
//! unblock the dispatch are recorded in `WORKER_REPORT.md`. Until that API
//! exists the harness keeps its documented system-level glue, and every
//! affected row keeps saying so in its `method` field; nothing here is
//! hidden.
//!
//! # Determinism
//!
//! No weather file, no randomness, no wall-clock dependence: every input is
//! the committed case input and every output is a pure function of it.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use fluxion::sim::hvac::{
    CoolingCoil, CoolingCoilBehavior, Fan, FanComponent, HeatingCoil, HeatingCoilComponent,
    HeatingCoilControl, MoistAirState,
};
use fluxion_core::qas_airside::{Ambient, TEST_CASES};
use fluxion_core::weather::psychrometrics::{
    calculate_dew_point, calculate_humidity_ratio, latent_heat_vaporization,
    partial_vapor_pressure, saturation_humidity_ratio, saturation_vapor_pressure,
    STANDARD_ATMOSPHERIC_PRESSURE_Pa,
};

// ---------------------------------------------------------------------------
// Committed artefacts
// ---------------------------------------------------------------------------

/// The committed quasi-analytical targets (read-only reference data).
const SECTION11_TARGETS_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/reference/ashrae140/section11_qas_targets.json"
));
/// The recorded per-cell engine-run rows (the known-fail record).
const SECTION11_RESULTS_JSON: &str = include_str!("data/section11_engine_run_results.json");

/// Opt-in path for re-recording the results file (see the module docs).
const WRITE_RESULTS_ENV: &str = "FLUXION_SECTION11_WRITE_RESULTS";

/// Relative tolerance for the live-vs-recorded comparison (the recorded file
/// stores values rounded to 1e-6).
const RECORDED_RTOL: f64 = 1e-6;
/// Absolute tolerance for the live-vs-recorded comparison.
const RECORDED_ATOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Unit conversions (explicit at the boundary)
// ---------------------------------------------------------------------------

/// 1 W = 3.412141633 Btu/h (exact conversion: 1 Btu/h = 0.2930710702 W).
const BTUH_PER_W: f64 = 3.412_141_633;
/// 1 Btu/h = 0.2930710702 W.
const W_PER_BTUH: f64 = 0.293_071_070_2;
/// Temperature difference: 1 K = 1.8 degF.
const F_PER_C: f64 = 1.8;
/// 1 cfm = 0.000471947443 m³/s.
const M3_S_PER_CFM: f64 = 0.000_471_947_443;
/// 1 lbm = 0.45359237 kg.
const KG_PER_LBM: f64 = 0.453_592_37;
/// 1 in. w.g. = 249.0889 Pa.
const PA_PER_IN_WG: f64 = 249.088_9;

// ---------------------------------------------------------------------------
// QAS configuration constants (QASv2 INPUTS sheet; see
// `fluxion-core/src/qas_airside.rs` — reference data, read-only)
// ---------------------------------------------------------------------------

/// Supply-fan pressure rise (in. w.g.).
const SFPR_IN_WG: f64 = 2.0;
/// Return-fan pressure rise (in. w.g.).
const RFPR_IN_WG: f64 = 1.0;
/// Design fan efficiency (fraction).
const DFE: f64 = 0.70;
/// Supply-air setpoint for CAV/VAV (degF).
const SASP_F: f64 = 55.0;
/// Preheat setpoint for CAV/VAV (degF).
const PHSP_F: f64 = 45.0;
/// Nominal supply cfm for CAV/VAV.
const NSCFM: f64 = 1300.0;
/// Nominal return cfm for CAV/VAV (the QAS workbook's return-fan rating).
const NRCFM: f64 = 800.0;

// ---------------------------------------------------------------------------
// Comparison tolerances (declared up front — see the module docs)
// ---------------------------------------------------------------------------

/// Load comparison tolerance (relative).
const LOAD_RTOL: f64 = 0.05;
/// Load comparison tolerance (absolute floor, Btu/h).
const LOAD_ATOL_BTUH: f64 = 20.0;
/// Temperature comparison tolerance (degF).
const TEMP_ATOL_F: f64 = 1.0;
/// Supply mass-flow comparison tolerance (relative).
const MASSFLOW_RTOL: f64 = 0.02;

/// The note attached to the rows of a cell whose cooling coil runs on air
/// drier than saturation at the coil leaving temperature (the QAS's
/// sensible-only coil case).
const DRY_COIL_NOTE: &str = "the QAS coil leaves the humidity ratio unchanged on air that is \
                            drier than saturation at the coil leaving temperature (a                             sensible-only coil), while fluxion's CoolingCoil always leaves the air \
                            at the apparatus dew point, so the latent load and the downstream                             humidity differ for this cell";

/// Maximum fixed-point iterations for the §11 steady-state solve.
const MAX_ITER: usize = 200;
/// Fixed-point convergence tolerance.
const SOLVE_TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Case inputs and the engine dispatch
// ---------------------------------------------------------------------------

/// The four §11 system models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum System {
    Fc,
    Sz,
    Cav,
    Vav,
}

impl System {
    /// The key used by the QAS targets file.
    fn key(self) -> &'static str {
        match self {
            System::Fc => "FC",
            System::Sz => "SZ",
            System::Cav => "CAV",
            System::Vav => "VAV",
        }
    }

    /// Every system, in the order the targets file uses.
    const ALL: [System; 4] = [System::Fc, System::Sz, System::Cav, System::Vav];
}

/// The unit of a published QAS quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Load,
    Temp,
    MassFlow,
}

impl Unit {
    fn label(self) -> &'static str {
        match self {
            Unit::Load => "Btu/h",
            Unit::Temp => "degF",
            Unit::MassFlow => "lbm/min",
        }
    }
}

/// One zone of a §11 case, in SI units.
#[derive(Debug, Clone, Copy)]
struct ZoneSpec {
    /// Zone dry-bulb temperature (degC).
    t_c: f64,
    /// Zone sensible load (W; positive = cooling load).
    q_sensible_w: f64,
    /// Zone latent load (W; positive = moisture added to the zone air).
    q_latent_w: f64,
    /// Design supply volumetric flow (m³/s).
    v_design_m3_s: f64,
    /// Exhaust (outdoor) volumetric flow (m³/s).
    v_exhaust_m3_s: f64,
}

/// One §11 case, in SI units.
#[derive(Debug, Clone, Copy)]
struct CaseSpec {
    case: usize,
    /// Outdoor dry-bulb (degC).
    oadb_c: f64,
    /// Outdoor dew point (degC).
    oadp_c: f64,
    /// True when the dew point is the derived (provisional) case-2 value.
    oadp_derived: bool,
    /// Zone 1.
    zone1: ZoneSpec,
    /// Zone 2 (CAV/VAV only).
    zone2: Option<ZoneSpec>,
}

/// Build the SI case specification from the committed QAS case inputs.
fn case_spec(case: usize) -> CaseSpec {
    let inputs = TEST_CASES[case - 1];
    // `Ambient::new` resolves the outdoor state, including the *derived*
    // case-2 dew point (the case publishes a wet bulb only) and the flag that
    // says so.
    let ambient = Ambient::new(case).expect("QAS ambient for case");
    assert!(
        ambient.oadp_derived == inputs.oadp.is_none(),
        "case {case}: dew-point provenance disagrees between the case inputs and the ambient"
    );
    let zone = |t_f: f64, qs: f64, ql: f64, cfm: f64, exh: f64| ZoneSpec {
        t_c: f_to_c(t_f),
        q_sensible_w: qs * W_PER_BTUH,
        q_latent_w: ql * W_PER_BTUH,
        v_design_m3_s: cfm * M3_S_PER_CFM,
        v_exhaust_m3_s: exh * M3_S_PER_CFM,
    };
    CaseSpec {
        case,
        oadb_c: f_to_c(inputs.oadb),
        oadp_c: f_to_c(ambient.oadp),
        oadp_derived: ambient.oadp_derived,
        zone1: zone(
            inputs.r1_t,
            inputs.r1_qs,
            inputs.r1_ql,
            inputs.r1_cfm,
            inputs.r1_exh,
        ),
        zone2: Some(zone(
            inputs.r2_t,
            inputs.r2_qs,
            inputs.r2_ql,
            inputs.r2_cfm,
            inputs.r2_exh,
        )),
    }
}

fn f_to_c(f: f64) -> f64 {
    (f - 32.0) / F_PER_C
}

fn c_to_f(c: f64) -> f64 {
    c * F_PER_C + 32.0
}

// ---------------------------------------------------------------------------
// Psychrometric helpers over the engine's own state type
// ---------------------------------------------------------------------------

/// Specific heat of moist air (J/(kg·K)), as the engine computes it
/// (`src/sim/hvac/airside_state.rs`: `1000·(1.006 + 1.86·ω)`).
fn cp_ma(w: f64) -> f64 {
    1000.0 * (1.006 + 1.86 * w)
}

/// Dry-air density (kg/m³) of a moist-air state.
fn dry_air_density(state: &MoistAirState) -> f64 {
    state.density_kg_per_m3 / (1.0 + state.humidity_ratio_kg_per_kg_dry_air)
}

/// Build an engine `MoistAirState` from a dry-bulb temperature and a humidity
/// ratio, through the engine's own psychrometric relations.
fn state_from_w(t_c: f64, w: f64) -> MoistAirState {
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
    let pv = partial_vapor_pressure(w, p);
    let rh = (100.0 * pv / saturation_vapor_pressure(t_c)).clamp(0.0, 100.0);
    MoistAirState::try_new(t_c, rh, p).expect("engine moist-air state")
}

/// Outdoor-air humidity ratio implied by the case's dry bulb and dew point.
fn outdoor_humidity_ratio(oadb_c: f64, oadp_c: f64) -> f64 {
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
    let rh = 100.0 * saturation_vapor_pressure(oadp_c) / saturation_vapor_pressure(oadb_c);
    calculate_humidity_ratio(oadb_c, rh.clamp(0.0, 100.0), p)
}

/// Mix dry-air streams by mass and enthalpy (the harness-level air mixer;
/// fluxion has no dedicated air-mixing component — see the module docs).
fn mix_streams(streams: &[(f64, MoistAirState)]) -> MoistAirState {
    // Zero-mass streams (a zone that carries no return air, e.g. the VAV
    // minimum-flow case) drop out of the balance.
    let streams: Vec<(f64, MoistAirState)> =
        streams.iter().copied().filter(|(m, _)| *m > 0.0).collect();
    let m_total: f64 = streams.iter().map(|(m, _)| *m).sum();
    assert!(m_total > 0.0, "cannot mix a zero-mass stream set");
    let h: f64 = streams
        .iter()
        .map(|(m, s)| m * s.enthalpy_kj_per_kg_dry_air)
        .sum::<f64>()
        / m_total;
    let w: f64 = streams
        .iter()
        .map(|(m, s)| m * s.humidity_ratio_kg_per_kg_dry_air)
        .sum::<f64>()
        / m_total;
    // Invert the engine's enthalpy relation h = 1.006·T + ω·(2501 + 1.86·T)
    // (`fluxion-core/src/weather/psychrometrics.rs::calculate_enthalpy`).
    let t_c = (h - w * 2501.0) / (1.006 + 1.86 * w);
    state_from_w(t_c, w)
}

/// Fan air-temperature rise (degC) from the engine's `FanComponent`.
///
/// fluxion's fan is a flow/power component with no air-temperature-rise
/// output; the QAS workbook's fan rise is `shaft power / (ṁ·cp)`, which is
/// also fluxion's own convention (`cav_terminal.rs`: "fan heat added to the
/// airstream (equals shaft power)"). The fan's rated density is set to the
/// operating dry-air density because the QAS workbook specifies a *constant*
/// fan pressure rise independent of density.
fn fan_temperature_rise(fan: &FanComponent, phi: f64, rho_da: f64, m_dot: f64, w: f64) -> f64 {
    let shaft_w = fan.shaft_power(phi, rho_da);
    shaft_w / (m_dot * cp_ma(w))
}

// ---------------------------------------------------------------------------
// Coil dispatch through fluxion's own components
// ---------------------------------------------------------------------------

/// A cooling-coil run through `CoolingCoil::compute_cooling_capacity`.
///
/// `bypass_factor = 0` and `apparatus_dew_point = leaving temperature` give
/// the ideal coil the QAS assumes (leaving air saturated at the leaving
/// temperature).
///
/// The QAS coil's leaving humidity ratio is `min(saturated at the leaving
/// temperature, entering humidity ratio)`: on air that is *drier* than
/// saturation at the leaving temperature the workbook's coil removes no
/// moisture at all (a sensible-only coil). fluxion's coil always leaves the
/// air at the ADP state, so it cannot represent that case: it leaves saturated
/// air (and reports a latent load where the workbook reports none). The
/// harness runs fluxion's coil as it is and flags the difference on the
/// affected rows — an engine-model difference, not a numerical one, recorded
/// rather than hidden.
struct CoolingCoilRun {
    sensible_w: f64,
    latent_w: f64,
    total_w: f64,
    leaving: MoistAirState,
    /// True when the entering air is drier than saturation at the leaving
    /// temperature, i.e. the QAS's sensible-only coil case.
    dry_coil_case: bool,
}

fn run_cooling_coil(entering: &MoistAirState, m_dot: f64, t_leaving_c: f64) -> CoolingCoilRun {
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
    let w_in = entering.humidity_ratio_kg_per_kg_dry_air;
    let w_sat_leaving = saturation_humidity_ratio(t_leaving_c, p);
    let dry_coil_case = w_in < w_sat_leaving;
    // Coil configuration. Normally the ideal coil: bypass factor 0, apparatus
    // dew point at the leaving temperature (leaving air saturated there).
    //
    // On air that is drier than saturation at the leaving temperature the
    // QAS coil removes no moisture (its leaving humidity ratio is
    // `min(saturated at the leaving temperature, entering humidity ratio)`):
    // the coil cools sensibly only. fluxion's bypass-factor coil reaches the
    // ADP state, so the equivalent configuration is an apparatus dew point at
    // the entering-air dew point with the bypass factor solved for the
    // leaving temperature: the leaving state is then the requested
    // temperature with the humidity ratio unchanged, and the latent load is
    // zero — the same leaving state and loads the QAS coil produces.
    let (bypass, adp_c) = if dry_coil_case {
        let t_dp_c =
            calculate_dew_point(entering.dry_bulb_c, entering.relative_humidity_percent, p);
        let span = entering.dry_bulb_c - t_dp_c;
        assert!(span > 0.0, "entering air is already saturated");
        let bypass = ((t_leaving_c - t_dp_c) / span).clamp(0.0, 0.999);
        (bypass, t_dp_c)
    } else {
        (0.0, t_leaving_c)
    };
    let coil = CoolingCoil::new(
        "§11 cooling coil".to_string(),
        f64::MAX / 4.0,
        1.0,
        bypass,
        adp_c,
        m_dot,
    );
    let perf = coil
        .compute_cooling_capacity(entering, m_dot)
        .expect("cooling coil capacity");
    assert!(
        (perf.leaving_air.dry_bulb_c - t_leaving_c).abs() < 1e-6,
        "the cooling coil must leave at the requested temperature"
    );
    CoolingCoilRun {
        sensible_w: perf.sensible_capacity_w,
        latent_w: perf.latent_capacity_w,
        total_w: perf.total_capacity_w,
        leaving: perf.leaving_air,
        dry_coil_case,
    }
}

/// A heating-coil run through `HeatingCoilComponent::compute_heating_capacity`.
///
/// The coils are instantiated with an effectively unlimited rated capacity so
/// that the steady-state dispatch is demand-driven: the §11 case inputs carry
/// no coil size, and the QAS workbook's coils are likewise unconstrained.
fn run_heating_coil(
    entering: &MoistAirState,
    m_dot: f64,
    t_leaving_c: f64,
) -> (f64, MoistAirState) {
    let coil = HeatingCoilComponent::new(
        "§11 heating coil".to_string(),
        f64::MAX / 4.0,
        m_dot.max(1e-6),
    );
    let res = coil
        .compute_heating_capacity(
            entering,
            m_dot,
            HeatingCoilControl::LeavingTempSetpoint(t_leaving_c),
        )
        .expect("heating coil capacity");
    (res.capacity_w, res.leaving_air)
}

// ---------------------------------------------------------------------------
// Engine results per system
// ---------------------------------------------------------------------------

/// The engine's own air-side result for one case × system.
#[derive(Debug, Clone, Default)]
struct SystemResult {
    heating_coil_w: f64,
    preheat_coil_w: Option<f64>,
    zone1_reheat_w: Option<f64>,
    zone2_reheat_w: Option<f64>,
    cooling_sensible_w: f64,
    cooling_latent_w: f64,
    cooling_total_w: f64,
    mixed_air_c: f64,
    supply_air_c: f64,
    return_air_c: Option<f64>,
    zone1_supply_c: Option<f64>,
    zone2_supply_c: Option<f64>,
    supply_mass_flow_kg_s: f64,
    /// A note about an engine-model difference that affects this cell's rows.
    coil_note: Option<String>,
}

// ---------------------------------------------------------------------------
// FC / SZ dispatch (four-pipe fan coil and single-zone air handler)
// ---------------------------------------------------------------------------

/// Dispatch the four-pipe fan coil (FC) or the single-zone air handler (SZ).
///
/// `with_return_fan` selects the SZ topology (a return fan raises the return
/// air temperature before the outdoor/return mixing). The economizer is off
/// (the QAS as-delivered configuration) and there is no return-duct heat gain.
fn dispatch_fc_sz(spec: &CaseSpec, with_return_fan: bool) -> SystemResult {
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
    let zone = spec.zone1;
    let w_out = outdoor_humidity_ratio(spec.oadb_c, spec.oadp_c);
    let outdoor = state_from_w(spec.oadb_c, w_out);

    let mut t_coil_c = zone.t_c;
    let mut w_supply = w_out;
    let mut coil_note: Option<String> = None;

    let mut result = SystemResult::default();
    for _ in 0..MAX_ITER {
        // --- supply (coil leaving) state and dry-air mass flow ------------
        let leaving = state_from_w(t_coil_c, w_supply);
        let rho_da_leaving = dry_air_density(&leaving);
        let m_total = rho_da_leaving * zone.v_design_m3_s;
        let cp_supply = cp_ma(w_supply);

        // --- zone humidity ratio from the zone latent load ----------------
        let h_fg = latent_heat_vaporization(zone.t_c);
        let w_zone = w_supply + zone.q_latent_w / (m_total * h_fg);

        // --- return air (and the SZ return fan) ---------------------------
        let mut ret = state_from_w(zone.t_c, w_zone);
        // The QAS workbook expresses the exhaust (outdoor) airflow at the
        // *return* conditions (`roemf = r1_exh / rosv` in
        // `fluxion-core/src/qas_airside.rs`), and the return flow is the
        // supply mass flow less the exhaust mass flow. The harness mirrors
        // that convention so that the comparison is between models and not
        // between configurations.
        let rho_da_ret = dry_air_density(&ret);
        let m_exhaust = rho_da_ret * zone.v_exhaust_m3_s;
        let m_ret = m_total - m_exhaust;
        if with_return_fan {
            let v_return = zone.v_design_m3_s - zone.v_exhaust_m3_s;
            let fan = FanComponent::with_motor(
                "§11 return fan".to_string(),
                v_return,
                RFPR_IN_WG * PA_PER_IN_WG,
                DFE,
                1.0,
                rho_da_ret,
            );
            // The return-fan flow ratio is the return volumetric flow (at the
            // return conditions) over the fan's rated flow.
            let phi = (m_ret / rho_da_ret) / v_return;
            let d_t = fan_temperature_rise(
                &fan,
                phi,
                rho_da_ret,
                m_ret,
                ret.humidity_ratio_kg_per_kg_dry_air,
            );
            ret = state_from_w(ret.dry_bulb_c + d_t, ret.humidity_ratio_kg_per_kg_dry_air);
        }

        // --- outdoor / return mixing (harness-level balance) --------------
        let m_out = m_exhaust;
        let mixed = mix_streams(&[(m_out, outdoor), (m_ret, ret)]);

        // --- supply fan and the zone supply temperature -------------------
        let fan = FanComponent::with_motor(
            "§11 supply fan".to_string(),
            zone.v_design_m3_s,
            SFPR_IN_WG * PA_PER_IN_WG,
            DFE,
            1.0,
            rho_da_leaving,
        );
        let d_t_fan = fan_temperature_rise(&fan, 1.0, rho_da_leaving, m_total, w_supply);
        // The zone supply temperature that meets the zone sensible load
        // (harness-level load balance over the engine's mass flow).
        let t_zone_supply_c = zone.t_c - zone.q_sensible_w / (m_total * cp_supply);
        let t_coil_new = t_zone_supply_c - d_t_fan;

        // --- the coil ------------------------------------------------------
        let (heating_w, cooling, w_supply_new) = if t_coil_new >= mixed.dry_bulb_c {
            // Heating: the sensible-only coil brings the mixed air up to the
            // supply temperature; the humidity ratio is unchanged.
            let (capacity_w, leaving_air) = run_heating_coil(&mixed, m_total, t_coil_new);
            (
                capacity_w,
                None,
                leaving_air.humidity_ratio_kg_per_kg_dry_air,
            )
        } else {
            let run = run_cooling_coil(&mixed, m_total, t_coil_new);
            let w_leaving = run.leaving.humidity_ratio_kg_per_kg_dry_air;
            coil_note = run.dry_coil_case.then(|| DRY_COIL_NOTE.to_string());
            (0.0, Some(run), w_leaving)
        };

        result = SystemResult {
            heating_coil_w: heating_w,
            preheat_coil_w: None,
            zone1_reheat_w: None,
            zone2_reheat_w: None,
            cooling_sensible_w: cooling.as_ref().map_or(0.0, |c| c.sensible_w),
            cooling_latent_w: cooling.as_ref().map_or(0.0, |c| c.latent_w),
            cooling_total_w: cooling.as_ref().map_or(0.0, |c| c.total_w),
            mixed_air_c: mixed.dry_bulb_c,
            supply_air_c: t_zone_supply_c,
            return_air_c: if with_return_fan {
                Some(ret.dry_bulb_c)
            } else {
                None
            },
            zone1_supply_c: None,
            zone2_supply_c: None,
            supply_mass_flow_kg_s: m_total,
            coil_note: coil_note.clone(),
        };

        let converged =
            (t_coil_new - t_coil_c).abs() < SOLVE_TOL && (w_supply_new - w_supply).abs() < 1e-12;
        t_coil_c = t_coil_new;
        w_supply = w_supply_new;
        if converged {
            break;
        }
    }
    let _ = p;
    result
}

// ---------------------------------------------------------------------------
// CAV / VAV dispatch (constant- and variable-volume reheat)
// ---------------------------------------------------------------------------

/// Dispatch the constant-volume (CAV) or variable-volume (VAV) reheat
/// system: a central air-handling unit (preheat coil, cooling coil, supply
/// fan) serving two zones, each with its own reheat coil.
fn dispatch_two_zone(spec: &CaseSpec, variable_volume: bool) -> SystemResult {
    let zone1 = spec.zone1;
    let zone2 = spec.zone2.expect("two-zone case");
    let w_out = outdoor_humidity_ratio(spec.oadb_c, spec.oadp_c);
    let outdoor = state_from_w(spec.oadb_c, w_out);

    let phsp_c = f_to_c(PHSP_F);
    let sasp_c = f_to_c(SASP_F);
    let v_total_design = NSCFM * M3_S_PER_CFM;

    let mut t_coil_c = phsp_c;
    let mut w_supply = w_out;
    // Previous-iteration values, used where the §11 solve is circular (the
    // zone humidity ratio needs the zone flow, and the VAV flow rule needs
    // the AHU discharge temperature).
    let mut m1_prev = zone1.v_design_m3_s * 1.2;
    let mut m2_prev = zone2.v_design_m3_s * 1.2;
    let mut t_discharge_prev = sasp_c;
    let mut coil_note: Option<String> = None;

    let mut result = SystemResult::default();
    for _ in 0..MAX_ITER {
        // --- AHU leaving state and mass flow ------------------------------
        let leaving = state_from_w(t_coil_c, w_supply);
        let rho_da_leaving = dry_air_density(&leaving);
        let cp_supply = cp_ma(w_supply);

        // --- zone humidity ratios from the zone latent loads ----------------
        let h_fg1 = latent_heat_vaporization(zone1.t_c);
        let h_fg2 = latent_heat_vaporization(zone2.t_c);
        let w_zone1 = w_supply + zone1.q_latent_w / (m1_prev * h_fg1);
        let w_zone2 = w_supply + zone2.q_latent_w / (m2_prev * h_fg2);

        // --- zone return states and the exhaust flows -----------------------
        // As in the FC/SZ dispatch, the QAS workbook expresses the exhaust
        // airflow at the return conditions.
        let ret1 = state_from_w(zone1.t_c, w_zone1);
        let ret2 = state_from_w(zone2.t_c, w_zone2);
        let m_exh1 = dry_air_density(&ret1) * zone1.v_exhaust_m3_s;
        let m_exh2 = dry_air_density(&ret2) * zone2.v_exhaust_m3_s;

        // --- zone mass flows ----------------------------------------------
        // CAV: the fixed design volume per zone, at the AHU leaving density.
        // VAV: the minimum flow that meets the zone sensible load at the AHU
        // discharge temperature, floored by the exhaust flow (the QAS
        // workbook's VAV flow rule).
        let m1 = if variable_volume {
            let load_flow = zone1.q_sensible_w / (cp_supply * (zone1.t_c - t_discharge_prev));
            if load_flow.is_finite() {
                load_flow.max(m_exh1)
            } else {
                m_exh1
            }
        } else {
            rho_da_leaving * zone1.v_design_m3_s
        };
        let m2 = if variable_volume {
            let load_flow = zone2.q_sensible_w / (cp_supply * (zone2.t_c - t_discharge_prev));
            if load_flow.is_finite() {
                load_flow.max(m_exh2)
            } else {
                m_exh2
            }
        } else {
            rho_da_leaving * zone2.v_design_m3_s
        };
        let m_total = m1 + m2;

        // --- AHU supply fan rise and the discharge temperature -------------
        let fan = FanComponent::with_motor(
            "§11 AHU supply fan".to_string(),
            v_total_design,
            SFPR_IN_WG * PA_PER_IN_WG,
            DFE,
            1.0,
            rho_da_leaving,
        );
        let phi = (m_total / rho_da_leaving) / v_total_design;
        let d_t_fan = fan_temperature_rise(&fan, phi, rho_da_leaving, m_total, w_supply);
        let t_discharge_c = t_coil_c + d_t_fan;

        // --- outdoor / return mixing (harness-level balance) ---------------
        // The CAV/VAV systems of the QAS workbook have a **return fan** (1 in.
        // w.g., 800 cfm nominal, 70 % efficient) downstream of the zone return
        // mixing: the mixed return air is raised by the fan's air-temperature
        // rise before it meets the outdoor air.
        let m_ret1 = m1 - m_exh1;
        let m_ret2 = m2 - m_exh2;
        let m_ret_total = m_ret1 + m_ret2;
        let mut ret_mixed = if m_ret_total > 0.0 {
            mix_streams(&[(m_ret1, ret1), (m_ret2, ret2)])
        } else {
            ret1
        };
        if m_ret_total > 0.0 {
            let v_rated_return = NRCFM * M3_S_PER_CFM;
            let rho_da_ret_mix = dry_air_density(&ret_mixed);
            let fan = FanComponent::with_motor(
                "§11 return fan".to_string(),
                v_rated_return,
                RFPR_IN_WG * PA_PER_IN_WG,
                DFE,
                1.0,
                rho_da_ret_mix,
            );
            let phi = (m_ret_total / rho_da_ret_mix) / v_rated_return;
            let d_t = fan_temperature_rise(
                &fan,
                phi,
                rho_da_ret_mix,
                m_ret_total,
                ret_mixed.humidity_ratio_kg_per_kg_dry_air,
            );
            ret_mixed = state_from_w(
                ret_mixed.dry_bulb_c + d_t,
                ret_mixed.humidity_ratio_kg_per_kg_dry_air,
            );
        }
        let m_out = (m_exh1 + m_exh2).max(m_total - m_ret_total);
        let mixed = mix_streams(&[(m_out, outdoor), (m_ret_total, ret_mixed)]);

        if std::env::var("FLUXION_SECTION11_DEBUG").is_ok() {
            let oash = cp_ma(outdoor.humidity_ratio_kg_per_kg_dry_air);
            let rash = cp_ma(ret1.humidity_ratio_kg_per_kg_dry_air);
            let rtsh = cp_ma(ret2.humidity_ratio_kg_per_kg_dry_air);
            let mrt =
                (m_ret1 * zone1.t_c * rash + m_ret2 * zone2.t_c * rtsh) / (m_ret1 + m_ret2) / rash;
            let w_mix = (m_ret1 * ret1.humidity_ratio_kg_per_kg_dry_air
                + m_ret2 * ret2.humidity_ratio_kg_per_kg_dry_air
                + m_out * outdoor.humidity_ratio_kg_per_kg_dry_air)
                / m_total;
            let mash = cp_ma(w_mix);
            let mat_qas = (m_out * outdoor.dry_bulb_c * oash + (m_total - m_out) * mrt * rash)
                / (m_total * mash);
            println!(
                "DBG two-zone: oadb={}F m_out={} m_ret1={} m_ret2={} m_total={} T_mix={}F T_mix_qas={}F w_mix={}",
                c_to_f(spec.oadb_c), m_out, m_ret1, m_ret2, m_total, c_to_f(mixed.dry_bulb_c), c_to_f(mat_qas), w_mix
            );
        }
        // --- preheat coil ---------------------------------------------------
        let t_preheat_c = mixed.dry_bulb_c.max(phsp_c);
        let (preheat_w, preheated) = run_heating_coil(&mixed, m_total, t_preheat_c);

        // --- cooling coil ----------------------------------------------------
        let t_coil_new = t_preheat_c.min(sasp_c - d_t_fan);
        let (cooling, w_supply_new) = if t_coil_new >= t_preheat_c {
            // No cooling: the AHU supply temperature is already below the
            // cooling coil's leaving temperature (QAS: `tacc = min(tahc, …)`).
            (None, preheated.humidity_ratio_kg_per_kg_dry_air)
        } else {
            let run = run_cooling_coil(&preheated, m_total, t_coil_new);
            let w_leaving = run.leaving.humidity_ratio_kg_per_kg_dry_air;
            coil_note = run.dry_coil_case.then(|| DRY_COIL_NOTE.to_string());
            (Some(run), w_leaving)
        };

        // --- zone supply temperatures and reheat coils ----------------------
        let t_zone1_supply_c = zone1.t_c - zone1.q_sensible_w / (m1 * cp_supply);
        let t_zone2_supply_c = zone2.t_c - zone2.q_sensible_w / (m2 * cp_supply);
        let discharge_state = state_from_w(t_discharge_c, w_supply);
        let (reheat1_w, _) = run_heating_coil(&discharge_state, m1, t_zone1_supply_c);
        let (reheat2_w, _) = run_heating_coil(&discharge_state, m2, t_zone2_supply_c);

        result = SystemResult {
            heating_coil_w: 0.0,
            preheat_coil_w: Some(preheat_w),
            zone1_reheat_w: Some(reheat1_w),
            zone2_reheat_w: Some(reheat2_w),
            cooling_sensible_w: cooling.as_ref().map_or(0.0, |c| c.sensible_w),
            cooling_latent_w: cooling.as_ref().map_or(0.0, |c| c.latent_w),
            cooling_total_w: cooling.as_ref().map_or(0.0, |c| c.total_w),
            mixed_air_c: mixed.dry_bulb_c,
            supply_air_c: t_discharge_c,
            return_air_c: None,
            zone1_supply_c: Some(t_zone1_supply_c),
            zone2_supply_c: Some(t_zone2_supply_c),
            supply_mass_flow_kg_s: m_total,
            coil_note: coil_note.clone(),
        };

        let converged =
            (t_coil_new - t_coil_c).abs() < SOLVE_TOL && (w_supply_new - w_supply).abs() < 1e-12;
        t_coil_c = t_coil_new;
        w_supply = w_supply_new;
        m1_prev = m1;
        m2_prev = m2;
        t_discharge_prev = t_discharge_c;
        if converged {
            break;
        }
    }
    result
}

/// Dispatch one case × system through the engine's own air-side components.
fn dispatch(case: usize, system: System) -> SystemResult {
    let spec = case_spec(case);
    match system {
        System::Fc => dispatch_fc_sz(&spec, false),
        System::Sz => dispatch_fc_sz(&spec, true),
        System::Cav => dispatch_two_zone(&spec, false),
        System::Vav => dispatch_two_zone(&spec, true),
    }
}

// ---------------------------------------------------------------------------
// Scoring against the committed QAS targets
// ---------------------------------------------------------------------------

/// The quantities the QAS targets publish for each system, with their units.
fn published_quantities(system: System) -> Vec<(&'static str, Unit)> {
    let loads = [
        ("cooling_sensible", Unit::Load),
        ("cooling_latent", Unit::Load),
        ("cooling_total", Unit::Load),
    ];
    let mut out: Vec<(&'static str, Unit)> = Vec::new();
    match system {
        System::Fc => {
            out.push(("heating_coil", Unit::Load));
            out.extend(loads);
            out.push(("mixed_air_temp", Unit::Temp));
            out.push(("supply_air_temp", Unit::Temp));
            out.push(("supply_mass_flow", Unit::MassFlow));
        }
        System::Sz => {
            out.push(("heating_coil", Unit::Load));
            out.extend(loads);
            out.push(("mixed_air_temp", Unit::Temp));
            out.push(("return_air_temp", Unit::Temp));
            out.push(("supply_air_temp", Unit::Temp));
            out.push(("supply_mass_flow", Unit::MassFlow));
        }
        System::Cav | System::Vav => {
            out.push(("preheat_coil", Unit::Load));
            out.push(("zone1_reheat", Unit::Load));
            out.push(("zone2_reheat", Unit::Load));
            out.push(("total_reheat", Unit::Load));
            out.push(("total_heat", Unit::Load));
            out.extend(loads);
            out.push(("mixed_air_temp", Unit::Temp));
            out.push(("supply_air_temp", Unit::Temp));
            out.push(("zone1_supply_temp", Unit::Temp));
            out.push(("zone2_supply_temp", Unit::Temp));
        }
    }
    out
}

/// The engine's value for a published quantity, with the method that produced
/// it. `None` means the engine's API cannot produce the quantity — the caller
/// records the row as not-runnable with the quoted gap.
fn engine_value(
    system: System,
    result: &SystemResult,
    quantity: &str,
) -> Option<(f64, &'static str)> {
    let load = |w: f64| (w * BTUH_PER_W, "engine coil component (W → Btu/h)");
    match (system, quantity) {
        (System::Fc | System::Sz, "heating_coil") => Some(load(result.heating_coil_w)),
        (System::Cav | System::Vav, "preheat_coil") => Some(load(result.preheat_coil_w?)),
        (System::Cav | System::Vav, "zone1_reheat") => Some(load(result.zone1_reheat_w?)),
        (System::Cav | System::Vav, "zone2_reheat") => Some(load(result.zone2_reheat_w?)),
        (System::Cav | System::Vav, "total_reheat") => {
            Some(load(result.zone1_reheat_w? + result.zone2_reheat_w?))
        }
        (System::Cav | System::Vav, "total_heat") => Some(load(
            result.preheat_coil_w? + result.zone1_reheat_w? + result.zone2_reheat_w?,
        )),
        (_, "cooling_sensible") => Some(load(result.cooling_sensible_w)),
        (_, "cooling_latent") => Some(load(result.cooling_latent_w)),
        (_, "cooling_total") => Some(load(result.cooling_total_w)),
        (_, "mixed_air_temp") => Some((
            c_to_f(result.mixed_air_c),
            "harness-level dry-air mass/enthalpy mixing balance over the engine's MoistAirState \
             (fluxion has no air-mixing component)",
        )),
        (_, "supply_air_temp") => Some((
            c_to_f(result.supply_air_c),
            "engine coil leaving state + FanComponent shaft-power fan rise; the zone supply \
             temperature is the engine's load balance",
        )),
        (System::Sz, "return_air_temp") => Some((
            c_to_f(result.return_air_c?),
            "FanComponent shaft-power return-fan rise over the engine's zone state",
        )),
        (System::Cav | System::Vav, "zone1_supply_temp") => Some((
            c_to_f(result.zone1_supply_c?),
            "harness-level zone load balance over the engine's supply mass flow and MoistAirState \
             enthalpy",
        )),
        (System::Cav | System::Vav, "zone2_supply_temp") => Some((
            c_to_f(result.zone2_supply_c?),
            "harness-level zone load balance over the engine's supply mass flow and MoistAirState \
             enthalpy",
        )),
        (System::Fc | System::Sz, "supply_mass_flow") => Some((
            result.supply_mass_flow_kg_s / KG_PER_LBM * 60.0,
            "engine supply-flow state (FanComponent flow × MoistAirState dry-air density)",
        )),
        _ => None,
    }
}

/// The comparison tolerance for a quantity.
fn tolerance(unit: Unit, target: f64) -> f64 {
    match unit {
        Unit::Load => (LOAD_RTOL * target.abs()).max(LOAD_ATOL_BTUH),
        Unit::Temp => TEMP_ATOL_F,
        Unit::MassFlow => MASSFLOW_RTOL * target.abs(),
    }
}

/// Score one published quantity against its committed target.
fn score_row(
    system: System,
    result: &SystemResult,
    quantity: &str,
    unit: Unit,
    target: f64,
) -> Value {
    let mut row = Map::new();
    row.insert("quantity".into(), json!(quantity));
    row.insert("unit".into(), json!(unit.label()));
    row.insert("target".into(), json!(round6(target)));
    match engine_value(system, result, quantity) {
        Some((value, method)) => {
            let tol = tolerance(unit, target);
            let gap = value - target;
            let inside = gap.abs() <= tol;
            row.insert("engine_value".into(), json!(round6(value)));
            row.insert("gap".into(), json!(round6(gap)));
            row.insert("tolerance".into(), json!(round6(tol)));
            row.insert(
                "gap_pct_of_target".into(),
                json!(round6(if target.abs() > 0.0 {
                    gap / target.abs() * 100.0
                } else {
                    f64::INFINITY
                })),
            );
            row.insert(
                "status".into(),
                json!(if inside { "pass" } else { "known_fail" }),
            );
            let method = match &result.coil_note {
                Some(note) => format!("{method} | {note}"),
                None => method.to_string(),
            };
            row.insert("method".into(), json!(method));
            row.insert("reason".into(), json!(null));
        }
        None => {
            row.insert("engine_value".into(), json!(null));
            row.insert("gap".into(), json!(null));
            row.insert("tolerance".into(), json!(null));
            row.insert("gap_pct_of_target".into(), json!(null));
            row.insert("status".into(), json!("not_runnable"));
            row.insert("method".into(), json!(null));
            row.insert(
                "reason".into(),
                json!(format!(
                    "no fluxion API produces `{quantity}` for the {} system: the air-side \
                     component set in `src/sim/hvac/` (cooling_coil.rs, heating_coil.rs, fan.rs, \
                     cav_terminal.rs, vav_terminal.rs, economizer.rs, airside_coupling.rs, \
                     airside_state.rs) exposes no symbol that can deliver this quantity from the \
                     §11 case inputs",
                    system.key()
                )),
            );
        }
    }
    Value::Object(row)
}

// ---------------------------------------------------------------------------
// The recorded results file
// ---------------------------------------------------------------------------

fn round6(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    (x * 1e6).round() / 1e6
}

fn targets_json() -> Value {
    serde_json::from_str(SECTION11_TARGETS_JSON).expect("parse §11 QAS targets")
}

/// The whole §11 engine run: every case × system, every published quantity.
fn build_results() -> Value {
    let targets = targets_json();
    let case2_caveat = targets["_case_2_caveat"]
        .as_str()
        .expect("case-2 caveat")
        .to_string();

    let mut cases: Vec<Value> = Vec::new();
    for case in 1..=6usize {
        let case_block = &targets["cases"][case.to_string()];
        assert!(
            !case_block.is_null(),
            "the §11 targets file has no case {case}"
        );
        let spec = case_spec(case);
        assert_eq!(
            spec.case, case,
            "the case specification must carry its case number"
        );
        let mut systems: Vec<Value> = Vec::new();
        for system in System::ALL {
            let result = dispatch(case, system);
            let target_block = &case_block[system.key()];
            let mut rows: Vec<Value> = Vec::new();
            for (quantity, unit) in published_quantities(system) {
                let target = target_block[quantity]
                    .as_f64()
                    .unwrap_or_else(|| panic!("case {case} {} has no `{quantity}`", system.key()));
                let mut row = score_row(system, &result, quantity, unit, target);
                if case == 2 {
                    let obj = row.as_object_mut().expect("row object");
                    obj.insert("case_2_caveat".into(), json!(case2_caveat));
                }
                rows.push(row);
            }
            systems.push(json!({
                "system": system.key(),
                "rows": rows,
            }));
        }
        cases.push(json!({
            "case": case,
            "oadb_F": round6(c_to_f(spec.oadb_c)),
            "oadp_F": round6(c_to_f(spec.oadp_c)),
            "oadp_derived": spec.oadp_derived,
            "systems": systems,
        }));
    }

    let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
    for case in &cases {
        for system in case["systems"].as_array().expect("systems array") {
            for row in system["rows"].as_array().expect("rows array") {
                let status = row["status"].as_str().expect("status");
                *counts.entry(status).or_insert(0) += 1;
            }
        }
    }

    json!({
        "_doc": "Recorded ASHRAE 140-2023 Section 11 (QASv2 quasi-analytical targets) engine-run \
                 results for fluxion's own air-side model (Issue #4274). GENERATED by the §11 \
                 engine-run harness \
                 (tests/validation/hvac_equipment_rated_performance/section11_engine_run.rs) — \
                 re-record with FLUXION_SECTION11_WRITE_RESULTS; never hand-edit.",
        "_targets": "data/reference/ashrae140/section11_qas_targets.json (QASv2.xlsm, merged \
                     quasi-analytical solution, ASHRAE RP-865; economizer off, zero return-duct \
                     heat gain)",
        "_method": "Each case × system is dispatched through fluxion's own air-side components \
                    (CoolingCoil, HeatingCoilComponent, FanComponent) and fluxion-core \
                    psychrometrics in the QAS configuration; the system-level glue (outdoor/return \
                    mixing, flow splits, zone load balances) is harness code and is flagged on \
                    every affected row. The QAS targets are point values, not bands: no band is \
                    fitted to the engine output.",
        "_tolerances": {
            "load_Btu_per_h": format!("max({LOAD_RTOL} of |target|, {LOAD_ATOL_BTUH} Btu/h)"),
            "temp_degF": TEMP_ATOL_F,
            "supply_mass_flow_lbm_per_min": format!("{MASSFLOW_RTOL} of |target|"),
        },
        "_status_legend": {
            "pass": "the engine value is inside the declared comparison tolerance",
            "known_fail": "the engine value is outside the comparison tolerance — recorded with \
                           the real engine value, target and gap",
            "not_runnable": "the quantity cannot be produced by fluxion's air-side API; the row \
                             quotes the API gap",
        },
        "_case_2_caveat": case2_caveat,
        "_row_counts": counts,
        "cases": cases,
    })
}

/// Deep JSON comparison with numeric tolerance (the §9 harness's pattern).
fn json_diff(live: &Value, recorded: &Value, path: &str) -> Vec<String> {
    let mut diffs = Vec::new();
    match (live, recorded) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().unwrap_or(f64::NAN),
                b.as_f64().unwrap_or(f64::NAN),
            );
            let tol = RECORDED_ATOL + RECORDED_RTOL * a.abs().max(b.abs());
            // NaN compares unequal to every tolerance: a non-finite live or
            // recorded value is a diff, not a match.
            let within_tolerance = (a - b).abs() <= tol;
            if !within_tolerance {
                diffs.push(format!("{path}: live {a} vs recorded {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => diffs.extend(json_diff(x, y, &format!("{path}.{key}"))),
                    (Some(_), None) => {
                        diffs.push(format!("{path}.{key}: missing in recorded file"))
                    }
                    (None, Some(_)) => diffs.push(format!("{path}.{key}: missing in live run")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                diffs.push(format!(
                    "{path}: live has {} entries, recorded has {}",
                    a.len(),
                    b.len()
                ));
            } else {
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    diffs.extend(json_diff(x, y, &format!("{path}[{i}]")));
                }
            }
        }
        (x, y) => {
            if x != y {
                diffs.push(format!("{path}: live {x} vs recorded {y}"));
            }
        }
    }
    diffs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The unit conversions at the QAS/engine boundary are the ones the QAS
/// workbook and the ASHRAE Handbook use.
#[test]
fn test_section11_unit_conversions() {
    assert!((BTUH_PER_W - 1.0 / W_PER_BTUH).abs() < 1e-9);
    assert!((1.0 * W_PER_BTUH * BTUH_PER_W - 1.0).abs() < 1e-9);
    // 1 kW = 3412.141633 Btu/h.
    assert!((1000.0 * BTUH_PER_W - 3412.141633).abs() < 1e-6);
    // Temperature: 0 degF = -17.7778 degC, 100 degF = 37.7778 degC.
    assert!((f_to_c(32.0)).abs() < 1e-12);
    assert!((f_to_c(212.0) - 100.0).abs() < 1e-12);
    assert!((c_to_f(f_to_c(70.7)) - 70.7).abs() < 1e-9);
    assert!((c_to_f(21.5) - 70.7).abs() < 1e-9);
    // 600 cfm = 0.28317 m³/s.
    assert!((600.0 * M3_S_PER_CFM - 0.283_168_466).abs() < 1e-9);
    // 1 lbm/min = 0.00755987 kg/s.
    assert!((KG_PER_LBM / 60.0 - 0.007_559_873).abs() < 1e-9);
    // 1 in. w.g. = 249.0889 Pa.
    assert!((PA_PER_IN_WG - 249.0889).abs() < 1e-9);
}

/// The harness reads the committed case inputs, and the derived case-2 dew
/// point matches the committed targets file.
#[test]
fn test_section11_case_inputs_match_committed_targets() {
    let targets = targets_json();
    for case in 1..=6usize {
        let spec = case_spec(case);
        let block = &targets["cases"][case.to_string()];
        assert!(
            (spec.oadb_c - f_to_c(block["oadb"].as_f64().expect("oadb"))).abs() < 1e-9,
            "case {case}: outdoor dry bulb disagrees with the committed targets"
        );
        assert!(
            (spec.oadp_c - f_to_c(block["oadp"].as_f64().expect("oadp"))).abs() < 1e-6,
            "case {case}: outdoor dew point disagrees with the committed targets"
        );
        assert_eq!(
            spec.oadp_derived,
            block["oadp_derived"].as_bool().expect("oadp_derived"),
            "case {case}: dew-point provenance disagrees with the committed targets"
        );
    }
}

/// The QAS case-2 dew point is the derived (provisional) one, and the caveat
/// is carried verbatim.
#[test]
fn test_section11_case_two_caveat_is_carried() {
    let targets = targets_json();
    let caveat = targets["_case_2_caveat"].as_str().expect("caveat");
    assert!(
        caveat.contains("provisional"),
        "the committed case-2 caveat must be carried verbatim"
    );
    let spec = case_spec(2);
    assert!(
        spec.oadp_derived,
        "case 2 publishes a wet bulb only: its dew point is the derived one"
    );
    for case in 1..=6usize {
        assert_eq!(
            case_spec(case).oadp_derived,
            case == 2,
            "only case 2 has a derived dew point"
        );
    }
}

/// Every published quantity of every case × system cell is scored against
/// the committed target — no cell and no quantity is silently omitted.
#[test]
fn test_section11_every_cell_and_quantity_is_accounted_for() {
    let targets = targets_json();
    let live = build_results();
    let cases = live["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 6, "all six §11 cases must be dispatched");
    for case in cases {
        let case_no = case["case"].as_u64().expect("case");
        let systems = case["systems"].as_array().expect("systems array");
        assert_eq!(
            systems.len(),
            4,
            "case {case_no}: all four systems must be dispatched"
        );
        for system in systems {
            let key = system["system"].as_str().expect("system");
            let target_block = &targets["cases"][case_no.to_string()][key];
            let target_keys: Vec<&String> = target_block
                .as_object()
                .expect("target object")
                .keys()
                .filter(|k| k.as_str() != "system")
                .collect();
            let rows = system["rows"].as_array().expect("rows array");
            assert_eq!(
                rows.len(),
                target_keys.len(),
                "case {case_no} {key}: every published target quantity must be scored"
            );
            for row in rows {
                let quantity = row["quantity"].as_str().expect("quantity");
                assert!(
                    target_keys.iter().any(|k| k.as_str() == quantity),
                    "case {case_no} {key}: row `{quantity}` is not a published target quantity"
                );
                let status = row["status"].as_str().expect("status");
                assert!(
                    matches!(status, "pass" | "known_fail" | "not_runnable"),
                    "case {case_no} {key} {quantity}: unexpected status {status}"
                );
                match status {
                    "pass" | "known_fail" => {
                        let value = row["engine_value"].as_f64().expect("engine value");
                        let target = row["target"].as_f64().expect("target");
                        let tol = row["tolerance"].as_f64().expect("tolerance");
                        assert!(
                            value.is_finite(),
                            "case {case_no} {key}: non-finite engine value"
                        );
                        if status == "pass" {
                            assert!(
                                (value - target).abs() <= tol,
                                "case {case_no} {key} {quantity}: a pass row must sit inside the \
                                 tolerance"
                            );
                        } else {
                            assert!(
                                (value - target).abs() > tol,
                                "case {case_no} {key} {quantity}: a known_fail row must sit \
                                 outside the tolerance"
                            );
                        }
                    }
                    "not_runnable" => {
                        let reason = row["reason"].as_str().expect("reason");
                        assert!(
                            reason.len() > 40,
                            "case {case_no} {key} {quantity}: a not-runnable row must quote the \
                             API gap"
                        );
                        assert!(
                            row["engine_value"].is_null(),
                            "case {case_no} {key} {quantity}: a not-runnable row must not carry an \
                             engine value"
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Every case-2 row carries the provisional-dew-point caveat verbatim.
#[test]
fn test_section11_case_two_rows_carry_the_caveat() {
    let caveat = targets_json()["_case_2_caveat"]
        .as_str()
        .expect("caveat")
        .to_string();
    let live = build_results();
    for case in live["cases"].as_array().expect("cases array") {
        let case_no = case["case"].as_u64().expect("case");
        for system in case["systems"].as_array().expect("systems array") {
            for row in system["rows"].as_array().expect("rows array") {
                if case_no == 2 {
                    assert_eq!(
                        row["case_2_caveat"].as_str(),
                        Some(caveat.as_str()),
                        "case 2 {} {}: the provisional-dew-point caveat must be carried",
                        system["system"].as_str().unwrap_or("?"),
                        row["quantity"].as_str().unwrap_or("?")
                    );
                } else {
                    assert!(
                        row.get("case_2_caveat").is_none(),
                        "case {case_no}: the case-2 caveat must not appear on other cases"
                    );
                }
            }
        }
    }
}

/// The dispatch is deterministic: two independent runs produce byte-identical
/// results.
#[test]
fn test_section11_engine_run_is_deterministic() {
    let first = build_results();
    let second = build_results();
    assert_eq!(
        serde_json::to_string(&first).expect("serialize"),
        serde_json::to_string(&second).expect("serialize"),
        "the §11 engine run must be deterministic"
    );
}

/// The recorded results file matches the live engine run — every recorded
/// number is re-measured, so a known-fail row cannot go stale silently.
#[test]
fn test_section11_recorded_results_match_live_engine_run() {
    let live = build_results();
    if let Ok(path) = std::env::var(WRITE_RESULTS_ENV) {
        let text = serde_json::to_string_pretty(&live).expect("serialize") + "\n";
        std::fs::write(&path, text).expect("write §11 engine-run results");
        println!("re-recorded §11 engine-run results to {path}");
        return;
    }
    let recorded: Value = serde_json::from_str(SECTION11_RESULTS_JSON)
        .expect("parse recorded §11 engine-run results");
    let diffs = json_diff(&live, &recorded, "results");
    assert!(
        diffs.is_empty(),
        "recorded §11 engine-run results differ from the live run (re-record with \
         FLUXION_SECTION11_WRITE_RESULTS):\n  {}",
        diffs.join("\n  ")
    );
}

/// The case-1 anchors published by the QASv2 workbook are reproduced by the
/// engine dispatch (they are the workbook values the QAS port is checked
/// against, and they are the one place where the *published* target and the
/// engine's own model can be compared without any modelling freedom).
#[test]
fn test_section11_case_one_anchor_is_reproduced_by_the_engine_dispatch() {
    let targets = targets_json();
    let case1 = &targets["cases"]["1"];
    let fc = dispatch(1, System::Fc);
    let heating = fc.heating_coil_w * BTUH_PER_W;
    let target = case1["FC"]["heating_coil"].as_f64().expect("target");
    assert!(
        (heating - target).abs() <= LOAD_RTOL * target.abs(),
        "case 1 FC heating coil: engine {heating} Btu/h vs the published QASv2 anchor {target} \
         Btu/h"
    );
}

/// The engine's fan rise convention is documented and the fan components are
/// used with the QAS workbook's pressure rises (2 in. w.g. supply, 1 in. w.g.
/// return) and design efficiency (70 %).
#[test]
fn test_section11_fan_configuration_matches_the_qas_workbook() {
    assert!((SFPR_IN_WG - 2.0).abs() < 1e-12);
    assert!((RFPR_IN_WG - 1.0).abs() < 1e-12);
    assert!((DFE - 0.70).abs() < 1e-12);
    assert!((SASP_F - 55.0).abs() < 1e-12);
    assert!((PHSP_F - 45.0).abs() < 1e-12);
    assert!((NSCFM - 1300.0).abs() < 1e-12);
}
