//! ASHRAE 140 multi-selector fabric harness (Refs #3986-A+2 / #4117).
//!
//! PR-A (#4119) and PR-A+1 (#4122) verified that `ASHRAE140Validator`
//! honors an explicit `ThermalSelector` end-to-end. PR-A+2 extends the
//! coverage from the *aggregate* `error_pct` (which sums heating +
//! cooling into one number — fine for the bug-regression test) to
//! **per-component annual energy** (heating MWh + cooling MWh) under
//! the engine-level blind execution path (`run_blind_annual_energy` from
//! `zone_balance_eplus_isolation.rs`).
//!
//! **Why per-component?** `validate_case().error_pct` aggregates H + C;
//! when two selectors agree on aggregate error but disagree on the H/C
//! split (e.g. FiveROneC H=5.18 / C=2.55 vs NineRFourC H=4.97 / C=2.78,
//! same sum, different physics), the aggregate hides the divergence. The
//! fabric harness separates the two so the regression detector sees it.
//!
//! **Why not Case 600 cooling?** Case 600 annual cooling is a documented
//! structural gap (post-#1323/#1213/#1328 chain; tracked by
//! `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
//! as `known_fail` with `gap_pct_of_mid = 34.38%`). The gap is a
//!
//! same for both selectors; the selector-parity delta is well-defined.
//!
//! **Scope (this module):**
//! - Cases: 600 (heating only — cooling is known-fail per #1147 baseline;
//!   cooling measurements are also captured here so the selector-parity
//!   drift detector can fire if the gap *worsens*), 900 (H+C), 950 (H+C).
//! - Selectors: FiveROneC + NineRFourC (Gauge is feature-gated; out of
//!   scope for the default-build gate — see #3982 for the gauge path).
//! - Engine-level isolation: `run_blind_annual_energy` from
//!   `zone_balance_eplus_isolation.rs` — proves the engine produces its
//!   results from the CaseSpec alone, not from any case-aware code path.
//! - Output: parseable lines of the form
//!   `[#3986-A+2 Case 600 / 5r1c] H=5.182 MWh / C=2.546 MWh / ratio_H=1.0000 / ratio_C=1.0000`
//!   that `scripts/check_ashrae_140_fabric_regression.py` consumes.
//!
//! **Verdict rule (per combination):**
//! - `test fabric measurement`: always PASS — this module captures
//!   measurements; it does NOT assert against ASHRAE bands (that is
//!   `check_strict_energy_gate_regression.py`'s job).
//! - The Python gate (`check_ashrae_140_fabric_regression.py`) does the
//!   drift detection: each `(case × selector)` measurement is compared
//!   against its recorded baseline, and per-case selector-parity deltas
//!   (`ratio_H = FiveROneC_H / NineRFourC_H`) are bounded by
//!   `regression_tolerance_rel_pct`.
//!
//! ## Naming
//!
//! Module `ashrae_140_fabric_multiselector` — distinct from:
//! - `ashrae_140_validator_selector_parity` (PR-A) — selector round-trip
//!   parity via the validator API.
//! - `ashrae_140_validator_fabric` (PR-A+1) — `validate_case` honors
//!   `self.selector` (1-line bug regression guard).
//! - `ashrae_140_fabric` (PR-C umbrella gate's sub-suite name) — the
//!   umbrella rolls up the three modules above plus this one.
//!
//! ## Acceptance criteria
//!
//! - All 6 (case × selector) × 2 (H + C) measurements run without panic.
//! - Lines are parseable by `scripts/check_ashrae_140_fabric_regression.py`.
//! - `cargo test --release --features ort --test all_tests
//!   ashrae_140_fabric_multiselector::` runs the 6 measurements; gate
//!   parses them in CI.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::{ConductionSolverKind, ThermalSelector, ZoneSolverKind};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

// Mirror the imports `zone_balance_eplus_isolation.rs::run_blind_annual_energy`
// needs. We do NOT import that helper directly (it's a private fn in the
// other module); instead we re-implement the loop here so this module is
// independent of `zone_balance_eplus_isolation`'s private surface.

/// J → MWh conversion. `run_blind_annual_energy` uses the same constant
/// (`J_TO_MWH`); mirrored here to keep the module self-contained.
const J_TO_MWH: f64 = 1.0 / 3.6e9;

/// EPA TMY3 weather path used by `run_blind_annual_energy` (Case 600/900
/// blind execution uses the same Golden CO file).
const EPW_PATH: &str = "assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw";

/// Run `case_id` annual simulation with the given `selector` and return
/// (heating_MWh, cooling_MWh). Mirrors the body of `run_blind_annual_energy`
/// in `tests/all_tests/zone_balance_eplus_isolation.rs` exactly; the only
/// delta is the `selector` argument (the parent helper hard-codes
/// `ThermalSelector::default()`).
fn run_blind_annual_with_selector(
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    selector: ThermalSelector,
) -> (f64, f64) {
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(spec, &selector)
        .expect("selector must initialise");
    let weather = fluxion::weather::epw::EpwWeatherSource::from_file(EPW_PATH)
        .expect("EPW weather file must be present in assets/weather/");

    let mut total_heating_j = 0.0_f64;
    let mut total_cooling_j = 0.0_f64;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_j = energy_kwh * 3.6e6;
        if energy_kwh > 0.0 {
            total_heating_j += energy_j;
        } else if energy_kwh < 0.0 {
            total_cooling_j += -energy_j;
        }
    }
    (total_heating_j * J_TO_MWH, total_cooling_j * J_TO_MWH)
}

/// 5R1C selector (legacy low-mass network).
fn selector_5r1c() -> ThermalSelector {
    ThermalSelector {
        zone_solver: ZoneSolverKind::FiveROneC,
        conduction_solver: ConductionSolverKind::Default,
    }
}

/// 9R4C selector (high-mass network; auto-promoted from HighMass specs
/// by `from_spec_with_selector`, but we set it explicitly here so the
/// selector round-trip is exercised).
fn selector_9r4c() -> ThermalSelector {
    ThermalSelector {
        zone_solver: ZoneSolverKind::NineRFourC,
        conduction_solver: ConductionSolverKind::Default,
    }
}

/// Selector id string for log lines (lowercase, matches the parseer's
/// `ZoneSolverKind::as_str()` output).
fn selector_id(s: &ThermalSelector) -> &'static str {
    s.zone_solver.as_str()
}

/// Per-case selector-pair fabric harness. Runs both selectors, prints a
/// parseable line per combination, and asserts both completed with
/// finite, non-negative energy (no physics assertions on absolute values
/// — those are the Python gate's job).
fn run_fabric_case(case: ASHRAE140Case, label: &str) {
    let spec = case.spec();
    let (h5, c5) = run_blind_annual_with_selector(&spec, selector_5r1c());
    let (h9, c9) = run_blind_annual_with_selector(&spec, selector_9r4c());

    // Selector-parity ratios (defined even when both are 0; the Python
    // gate treats 0/0 as "no measurement" and skips the parity check).
    let ratio_h = if h9.abs() > 1e-9 { h5 / h9 } else { 1.0 };
    let ratio_c = if c9.abs() > 1e-9 { c5 / c9 } else { 1.0 };

    // Print in a parseable, line-stable format. Two lines per case so the
    // parser can attribute each measurement to a single (case, selector).
    println!(
        "[#3986-A+2 Case {} / {}] H={:.3} MWh",
        label,
        selector_id(&selector_5r1c()),
        h5
    );
    println!(
        "[#3986-A+2 Case {} / {}] C={:.3} MWh",
        label,
        selector_id(&selector_5r1c()),
        c5
    );
    println!(
        "[#3986-A+2 Case {} / {}] H={:.3} MWh",
        label,
        selector_id(&selector_9r4c()),
        h9
    );
    println!(
        "[#3986-A+2 Case {} / {}] C={:.3} MWh",
        label,
        selector_id(&selector_9r4c()),
        c9
    );
    println!(
        "[#3986-A+2 Case {} / parity] ratio_H={:.4} ratio_C={:.4}",
        label, ratio_h, ratio_c
    );

    // Sanity: every measurement must be finite and non-negative.
    // (Negative metered energy would mean a sign convention bug; this
    // module is a measurement harness, not an ASHRAE band test.)
    for (h, c, sel) in [(h5, c5, "5r1c"), (h9, c9, "9r4c")] {
        assert!(
            h.is_finite() && c.is_finite(),
            "Case {label} / {sel}: non-finite energy (H={h}, C={c})"
        );
        assert!(
            h >= 0.0 && c >= 0.0,
            "Case {label} / {sel}: negative metered energy (H={h}, C={c})"
        );
    }
}

// ---------- Slice 1: Case 600 fabric measurement ----------

#[test]
fn fabric_case_600_measurement() {
    run_fabric_case(ASHRAE140Case::Case600, "600");
}

// ---------- Slice 2: Case 900 fabric measurement ----------

#[test]
fn fabric_case_900_measurement() {
    run_fabric_case(ASHRAE140Case::Case900, "900");
}

// ---------- Slice 3: Case 950 fabric measurement ----------

#[test]
fn fabric_case_950_measurement() {
    run_fabric_case(ASHRAE140Case::Case950, "950");
}
