//! Regression test for Issue #827 — hourly free-float temperature profile via
//! the `FreeFloatValidationResult` API.
//!
//! Consumes the in-memory hourly profile (no filesystem touch — the
//! `pr821-diag` CSV is the offline equivalent) and asserts on:
//!   1. The profile is `Some` for FF cases, `None` for non-FF cases.
//!   2. Length is exactly 8760 (annual hourly run).
//!   3. The recorded min/max equal the cumulative min/max from the profile.
//!   4. The series shows real diurnal swing (at least one adjacent-hour
//!      increase AND one decrease) — guards against a future regression that
//!      collapses the profile to a constant.
//!
//! Acceptance criteria from the issue (all met):
//!   - opt-in `hourly_temperatures: Option<Vec<f64>>` populated only for FF
//!   - no allocation for non-FF cases (Option stays `None`)
//!   - regression test consumes the in-memory profile (this file)
//!   - `pr821-diag` CSV remains available as a separate artefact (unchanged)

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::ashrae_140_validator::{validate_ashrae_140, FreeFloatValidationResult};

fn assert_ff_profile_shape(case_id: &str, result: &FreeFloatValidationResult) {
    let v = result
        .hourly_temperatures
        .as_ref()
        .unwrap_or_else(|| panic!("[{case_id}] hourly_temperatures must be Some for FF case"));
    assert_eq!(
        v.len(),
        8760,
        "[{case_id}] hourly_temperatures length must be exactly 8760, got {}",
        v.len()
    );

    let cum_min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let cum_max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (cum_min - result.free_float_min_temp).abs() < 1e-9,
        "[{case_id}] returned min {} must match profile min {}",
        result.free_float_min_temp,
        cum_min
    );
    assert!(
        (cum_max - result.free_float_max_temp).abs() < 1e-9,
        "[{case_id}] returned max {} must match profile max {}",
        result.free_float_max_temp,
        cum_max
    );

    // Diurnal-swing sanity: not a constant series.
    let mut saw_increase = false;
    let mut saw_decrease = false;
    for w in v.windows(2) {
        if w[1] > w[0] {
            saw_increase = true;
        }
        if w[1] < w[0] {
            saw_decrease = true;
        }
        if saw_increase && saw_decrease {
            break;
        }
    }
    assert!(
        saw_increase && saw_decrease,
        "[{case_id}] hourly_temperatures must show real diurnal swing (saw \
         increase = {saw_increase}, decrease = {saw_decrease})"
    );
}

#[test]
fn hourly_profile_populated_for_600ff() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let result = validate_ashrae_140(&spec);
    assert_ff_profile_shape("600FF", &result);
}

#[test]
fn hourly_profile_populated_for_650ff() {
    let spec = ASHRAE140Case::Case650FF.spec();
    let result = validate_ashrae_140(&spec);
    assert_ff_profile_shape("650FF", &result);
}

#[test]
fn hourly_profile_populated_for_900ff() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let result = validate_ashrae_140(&spec);
    assert_ff_profile_shape("900FF", &result);
}

#[test]
fn hourly_profile_none_for_non_ff_case() {
    // Case 600 (HVAC-controlled, not FF) must NOT allocate the per-hour vector.
    let spec = ASHRAE140Case::Case600.spec();
    let result = validate_ashrae_140(&spec);
    assert!(
        result.hourly_temperatures.is_none(),
        "non-FF Case 600: hourly_temperatures must be None (zero-overhead path), \
         got Some(len={})",
        result
            .hourly_temperatures
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0)
    );
}
