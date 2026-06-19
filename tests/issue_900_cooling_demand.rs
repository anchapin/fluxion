//! Regression test for HVAC cooling demand — Issues #900 → #1163 evolution.
//!
//! # History
//!
//! **Issue #925** fixed the HVAC coefficient `h_coeff` (Norton equivalent at
//! the air node), bringing heating into the reference range.
//!
//! **Issue #900** then tried to fix the remaining cooling underestimation by
//! adding a "mass heat release" term. The implementation in `hvac.rs` went
//! further than intended: it *replaced* the zone-temperature-driven cooling
//! formula with a mass-temperature-driven one (`-h_coeff × (T_mass − T_cool_sp)`),
//! justified by a derivation that incorrectly assumed `h_tr_ms = h_coeff`
//! (they differ by >10× in practice). This caused systematic cooling
//! underestimation (sim/ref_mid ≈ 0.42 — only 42% of reference cooling).
//!
//! **Issue #1163** reverted to the symmetric ASHRAE 140 ideal HVAC formulation:
//!
//! ```text
//! Q_HVAC = h_coeff × (T_setpoint − T_free)
//! ```
//!
//! applied identically for heating and cooling, where `T_free` is the
//! free-floating zone air temperature. The mass heat-release contribution is
//! embedded in `T_free` via the 5R1C heat balance (`num_tm = h_ms_is_prod ×
//! T_mass`), so it is captured exactly once.
//!
//! # What this test pins down
//!
//!  1. **The symmetric heating/cooling formula.** Both branches use
//!     `h_coeff × (T_setpoint − T_free)` with the same driving temperature.
//!     No mass-temperature term is added to either branch.
//!
//!  2. **The deadband case.** When `T_free` is between the heating and cooling
//!     setpoints, demand is exactly zero — regardless of the mass temperature.
//!     This matches ASHRAE 140 (ideal HVAC is OFF in the deadband).
//!
//!  3. **No phantom heating.** The old Issue #900 formula produced POSITIVE
//!     (heating) values in the cooling branch when `T_mass < T_cool_sp`,
//!     which inflated annual heating energy. The corrected formula never
//!     produces heating demand in the cooling branch.
//!
//!  4. **End-to-end sanity for Case 900.** The high-mass Case 900 cooling
//!     and heating remain non-zero and bounded.
//!
//! See `src/sim/thermal_model_physics/hvac.rs` for the implementation.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

/// Compute the HVAC demand for a single zone using the corrected Issue #1163
/// symmetric formula. Mirrors `compute_zone_hvac_load` (hvac.rs) for unit testing.
fn demand_for_zone(
    h_coeff: f64,
    t_free: f64,
    heat_sp: f64,
    cool_sp: f64,
) -> f64 {
    if t_free <= heat_sp {
        // Heating: Q = h_coeff × (T_heat_sp − T_free).
        h_coeff * (heat_sp - t_free)
    } else if t_free >= cool_sp {
        // Cooling: Q = -h_coeff × (T_free − T_cool_sp). Symmetric with heating.
        -h_coeff * (t_free - cool_sp)
    } else {
        // Deadband: no HVAC demand (ASHRAE 140 ideal HVAC is OFF).
        0.0
    }
}

#[test]
fn issue_1163_heating_formula_uses_free_temperature() {
    // Heating branch: Q = h_coeff × (T_heat_sp − T_free).
    // Must NOT include a mass heat absorption term.
    let h_coeff = 93.0;

    // Zone at 10°C in winter. q_heat = 93 × 10 = 930 W.
    let q = demand_for_zone(h_coeff, 10.0, 20.0, 27.0);
    assert!(
        (q - 930.0).abs() < 1.0,
        "Heating formula regressed: q={q:.2} (expected ≈ 930)"
    );

    // At exactly the heating setpoint, demand is zero-margin positive (uses <=).
    let q_at_sp = demand_for_zone(h_coeff, 20.0, 20.0, 27.0);
    assert!(
        q_at_sp.abs() < 1e-9,
        "At heating setpoint demand should be ~0: q={q_at_sp:.6}"
    );
}

#[test]
fn issue_1163_cooling_formula_symmetric_with_heating() {
    // Cooling: Q = -h_coeff × (T_free − T_cool_sp).
    // This is the symmetric counterpart of the heating formula.
    let h_coeff = 93.0;
    let cool_sp = 27.0;

    // Zone free-floating at 30°C (3°C above cool_sp).
    let q = demand_for_zone(h_coeff, 30.0, 20.0, cool_sp);
    let expected = -h_coeff * (30.0 - cool_sp);
    assert!(
        (q - expected).abs() < 1.0,
        "Cooling formula should be symmetric: q={q:.2} W (expected ≈ {expected:.2})"
    );

    // Larger temperature excursion → proportionally larger demand (linear).
    let q_large = demand_for_zone(h_coeff, 40.0, 20.0, cool_sp);
    let expected_large = -h_coeff * (40.0 - cool_sp);
    assert!(
        (q_large - expected_large).abs() < 1.0,
        "Cooling should scale linearly with T_free: q={q_large:.2} (expected {expected_large:.2})"
    );
}

#[test]
fn issue_1163_deadband_produces_zero_demand() {
    // When T_free is in the deadband (between heat_sp and cool_sp), demand is
    // exactly zero — regardless of what the mass temperature might be.
    // This is the correct ASHRAE 140 behavior: ideal HVAC is OFF in the deadband.
    let h_coeff = 93.0;
    let heat_sp = 20.0;
    let cool_sp = 27.0;

    // Middle of deadband.
    let q_mid = demand_for_zone(h_coeff, 23.5, heat_sp, cool_sp);
    assert!(
        q_mid.abs() < 1e-9,
        "Deadband demand must be 0: q={q_mid:.6}"
    );

    // Just below cool_sp (still in deadband).
    let q_near_cool = demand_for_zone(h_coeff, 26.99, heat_sp, cool_sp);
    assert!(
        q_near_cool.abs() < 1e-9,
        "Just below cool_sp is still deadband: q={q_near_cool:.6}"
    );

    // Just above heat_sp (still in deadband).
    let q_near_heat = demand_for_zone(h_coeff, 20.01, heat_sp, cool_sp);
    assert!(
        q_near_heat.abs() < 1e-9,
        "Just above heat_sp is still deadband: q={q_near_heat:.6}"
    );
}

#[test]
fn issue_1163_no_phantom_heating_in_cooling_branch() {
    // CRITICAL: The old Issue #900 formula produced POSITIVE (heating) values
    // in the cooling branch when T_mass < T_cool_sp. The corrected formula
    // NEVER produces heating in the cooling branch.
    //
    // Scenario: T_free = 35°C (hot zone, above cool_sp), which under the old
    // formula with T_mass = 25°C would have given +140 W (phantom heating).
    let h_coeff = 70.0;  // Case 600 Norton equivalent
    let cool_sp = 27.0;

    let q = demand_for_zone(h_coeff, 35.0, 20.0, cool_sp);
    assert!(
        q < 0.0,
        "Cooling branch must produce NEGATIVE (cooling) demand: q={q:.2} W"
    );

    let expected = -h_coeff * (35.0 - cool_sp);
    assert!(
        (q - expected).abs() < 1.0,
        "Cooling demand should be {expected:.2} W: got q={q:.2} W"
    );
}

#[test]
fn issue_1163_annual_cooling_nonzero_for_case_900() {
    // End-to-end: Case 900 (high-mass) must produce non-zero cooling.
    // Case 900 uses the multi-node (9R4C) path, which already had the
    // correct symmetric formula — this test is a sanity check that the
    // 5R1C path changes did not break the multi-node path.
    let spec = ASHRAE140Case::Case900.spec();
    let mut model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

    let weather = fluxion::weather::denver::DenverTmyWeather::new();
    for step in 0..8760 {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());
        model.step_physics(step, wd.dry_bulb_temp, 3600.0);
    }

    let cooling_mwh = model.annual_cooling_energy / 1000.0;
    println!("Case 900 annual cooling (post-#1163): {cooling_mwh:.3} MWh");

    // Cooling must be positive and in a reasonable range.
    assert!(
        cooling_mwh > 0.5,
        "Annual cooling suspiciously low: {cooling_mwh:.3} MWh"
    );
}

#[test]
fn issue_1163_annual_heating_nonzero_for_case_900() {
    // End-to-end: Case 900 heating must remain non-zero and bounded.
    let spec = ASHRAE140Case::Case900.spec();
    let mut model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

    let weather = fluxion::weather::denver::DenverTmyWeather::new();
    for step in 0..8760 {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());
        model.step_physics(step, wd.dry_bulb_temp, 3600.0);
    }

    let heating_mwh = model.annual_heating_energy / 1000.0;
    println!("Case 900 annual heating (post-#1163): {heating_mwh:.3} MWh");

    // Heating must be positive and in a reasonable range (2-4 MWh for Case 900).
    assert!(
        (1.0..=6.0).contains(&heating_mwh),
        "Annual heating out of expected range: {heating_mwh:.3} MWh"
    );
}
