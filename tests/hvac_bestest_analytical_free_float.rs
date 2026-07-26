//! Integration tests for HVAC BESTEST RP-865 analytical free-floating cases.
//!
//! Issue #1757. Free-floating zones carry **no HVAC equipment**; the zone
//! temperature is set purely by the heat balance. These cases verify that the
//! analytical (closed-form) solution is reproduced and that the predicted zone
//! temperatures fall within the published analytical bounds.
//!
//! Run with: `cargo test --test hvac_bestest_analytical_free_float`
//!
//! See `src/validation/hvac_bestest/analytical_free_float.rs` for the physics
//! derivation and the case table.

use fluxion::validation::hvac_bestest::{
    get_free_float_analytical_cases, run_free_float_analytical, validate_free_float,
    FreeFloatAnalyticalRunner, FreeFloatCaseId,
};

/// Published analytical reference bounds for the free-floating cases.
///
/// The closed-form steady-state temperature is exact; the half-width (0.5 K)
/// absorbs only the numerical-convergence residual of the dummy runner.
mod published_bounds {
    use super::*;

    fn bounds_for(case: &FreeFloatCaseId) -> (f64, f64) {
        let def = get_free_float_analytical_cases()
            .into_iter()
            .find(|c| &c.case_id == case)
            .unwrap_or_else(|| panic!("missing case {case:?}"));
        (def.t_ref - def.tolerance_k, def.t_ref + def.tolerance_k)
    }

    pub fn assert_within(case: &FreeFloatCaseId, predicted: f64) {
        let (lo, hi) = bounds_for(case);
        assert!(
            lo <= predicted && predicted <= hi,
            "{case:?}: predicted T={predicted:.3}°C outside published bounds [{lo:.3}, {hi:.3}]°C"
        );
    }
}

#[test]
fn test_case_ff100_baseline_temperature_equals_outdoor() {
    // No gains => free-floating zone must equal outdoor temperature exactly.
    let case = &get_free_float_analytical_cases()[0];
    assert_eq!(case.case_id, FreeFloatCaseId::FF100);
    let t = FreeFloatAnalyticalRunner::analytical_steady_state(case);
    assert!(
        (t - case.t_outdoor).abs() < 1e-9,
        "FF100 baseline must equal T_out"
    );
    published_bounds::assert_within(
        &case.case_id,
        FreeFloatAnalyticalRunner::run_numerical(case),
    );
}

#[test]
fn test_case_ff110_internal_gains_raise_temperature() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF110)
        .unwrap();
    let t = FreeFloatAnalyticalRunner::run_numerical(case);
    assert!(t > case.t_outdoor, "internal gains must heat the zone");
    published_bounds::assert_within(&case.case_id, t);
}

#[test]
fn test_case_ff120_solar_gains_raise_temperature() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF120)
        .unwrap();
    let t = FreeFloatAnalyticalRunner::run_numerical(case);
    assert!(t > case.t_outdoor, "solar gains must heat the zone");
    published_bounds::assert_within(&case.case_id, t);
}

#[test]
fn test_case_ff130_cold_outdoor_heating_dominated() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF130)
        .unwrap();
    let t = FreeFloatAnalyticalRunner::run_numerical(case);
    // Small internal gain keeps the zone slightly warmer than the cold outdoor air.
    assert!(
        t > case.t_outdoor,
        "internal gain must keep zone above T_out"
    );
    published_bounds::assert_within(&case.case_id, t);
}

#[test]
fn test_case_ff140_cooling_dominated_hot_outdoor() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF140)
        .unwrap();
    let t = FreeFloatAnalyticalRunner::run_numerical(case);
    assert!(t > case.t_outdoor, "gains must push zone above hot T_out");
    published_bounds::assert_within(&case.case_id, t);
}

#[test]
fn test_case_ff150_high_infiltration_pulled_toward_outdoor() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF150)
        .unwrap();
    let t_low = FreeFloatAnalyticalRunner::run_numerical(case);
    // With 4x the infiltration, the zone should sit closer to outdoor than the
    // baseline UA/ACH configuration for comparable gains.
    let ref_case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF110)
        .unwrap();
    let t_ref = FreeFloatAnalyticalRunner::analytical_steady_state(ref_case);
    let lift_ff150 = (t_low - case.t_outdoor).abs();
    let lift_ff110 = (t_ref - ref_case.t_outdoor).abs();
    assert!(
        lift_ff150 < lift_ff110,
        "high-infiltration case ({lift_ff150:.2} K lift) should be closer to outdoor than FF110 ({lift_ff110:.2} K)"
    );
    published_bounds::assert_within(&case.case_id, t_low);
}

#[test]
fn test_case_ff160_tight_envelope_traps_heat() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF160)
        .unwrap();
    let t = FreeFloatAnalyticalRunner::run_numerical(case);
    // Low UA + low ACH => highest zone temperature of the family.
    assert!(t > 100.0, "tight envelope should trap heat (T={t:.1}°C)");
    published_bounds::assert_within(&case.case_id, t);
}

#[test]
fn test_case_ff170_no_infiltration_envelope_limited() {
    let cases = get_free_float_analytical_cases();
    let case = cases
        .iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF170)
        .unwrap();
    let t = FreeFloatAnalyticalRunner::run_numerical(case);
    assert!(case.ach == 0.0);
    assert!(t > case.t_outdoor);
    published_bounds::assert_within(&case.case_id, t);
}

#[test]
fn test_all_predicted_temps_within_published_bounds() {
    // Core acceptance criterion: every predicted zone temperature falls within
    // the published analytical bounds.
    let results = run_free_float_analytical();
    assert_eq!(results.len(), 8);
    for r in &results {
        assert!(
            r.within_bounds,
            "{} FAILED: {}",
            r.case_id.label(),
            r.message
        );
        // Energy balance must close (thermodynamic consistency).
        assert!(
            r.energy_balance_residual_w.abs() < 1.0,
            "{}: energy balance residual {:.3} W too large",
            r.case_id.label(),
            r.energy_balance_residual_w
        );
    }
    let (passed, failed) = validate_free_float(&results);
    assert_eq!(failed, 0, "{failed} cases failed the published bounds");
    assert_eq!(passed, 8);
}

#[test]
fn test_numerical_runner_matches_analytical_solution() {
    // The dummy runner must reproduce the closed-form solution for every case.
    for case in get_free_float_analytical_cases() {
        let t_num = FreeFloatAnalyticalRunner::run_numerical(&case);
        let t_ana = FreeFloatAnalyticalRunner::analytical_steady_state(&case);
        assert!(
            (t_num - t_ana).abs() < 0.1,
            "{:?}: dummy runner diverged from analytical by {:.4} K",
            case.case_id,
            (t_num - t_ana).abs()
        );
    }
}

#[test]
fn test_exact_transient_matches_numerical_trajectory() {
    // The closed-form transient and the numerical integrator must agree along
    // the whole trajectory, not just at steady state.
    let case = get_free_float_analytical_cases()
        .into_iter()
        .find(|c| c.case_id == FreeFloatCaseId::FF110)
        .unwrap();
    let g_inf = fluxion::physics::constants::AIR_DENSITY_SEA_LEVEL
        * case.ach
        * case.volume
        * fluxion::physics::constants::AIR_SPECIFIC_HEAT
        / 3600.0;
    let loss = case.ua_total + g_inf;
    let q_in = case.q_internal + case.q_solar;
    let dt = 300.0;
    let mut t_num = case.t_outdoor;
    let t0 = case.t_outdoor;
    for step in 1..=120 {
        t_num += (q_in - loss * (t_num - case.t_outdoor)) * dt / case.c_zone;
        let t_seconds = step as f64 * dt;
        let t_exact = FreeFloatAnalyticalRunner::analytical_transient(&case, t0, t_seconds);
        // Forward-Euler truncation error grows with dt; allow a modest tolerance.
        assert!(
            (t_num - t_exact).abs() < 0.5,
            "step {step}: |T_num - T_exact| = {:.4} K",
            (t_num - t_exact).abs()
        );
    }
}

#[test]
fn test_print_analytical_summary() {
    let results = run_free_float_analytical();
    let (passed, failed) = validate_free_float(&results);
    println!("\n=== HVAC BESTEST Analytical Free-Floating Summary (Issue #1757) ===");
    println!(
        "Cases: {}  Passed: {}  Failed: {}",
        results.len(),
        passed,
        failed
    );
    for r in &results {
        let status = if r.within_bounds { "PASS" } else { "FAIL" };
        println!(
            "  [{status}] {:<42} T_num={:7.2}°C  ref=[{:7.2}, {:7.2}]°C  |ΔE|={:.1e}W",
            r.case_id.label(),
            r.t_numerical,
            r.t_ref_low,
            r.t_ref_high,
            r.energy_balance_residual_w.abs()
        );
    }
    println!("=====================================================================\n");
    assert_eq!(failed, 0);
}
