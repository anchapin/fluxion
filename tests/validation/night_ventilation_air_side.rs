//! Regression test for Issue #824 — Case 650 air-side night ventilation.
//!
//! Verifies that the air-side night-ventilation conductance added in this PR
//! actually changes Case 650FF behaviour relative to the otherwise-identical
//! Case 600FF (which has no night ventilation).
//!
//! Acceptance per Issue #824:
//!   - 600FF / 650FF zero-HVAC FF guards continue passing (already covered
//!     by tests/ashrae_140_case_600_series.rs::free_float_hvac_guard).
//!   - Night ventilation has a *physically meaningful* effect on 650FF
//!     compared to 600FF (this test).
//!   - Implementation does not restore the legacy h_vent_mass × 0.3 direct
//!     mass-cooling path (verified by code review of step_physics_5r1c).
//!
//! This test does NOT assert the absolute reference band — that is gated by
//! Issue #831 (envelope re-derivation) and is not closable here.

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::ashrae_140_validator::validate_ashrae_140;

#[test]
fn night_ventilation_cools_650ff_below_600ff() {
    let r600 = validate_ashrae_140(&ASHRAE140Case::Case600FF.spec());
    let r650 = validate_ashrae_140(&ASHRAE140Case::Case650FF.spec());

    // Both cases share envelope, geometry, and weather. The only difference
    // is Case 650FF's 18:00–07:00 fan at 1703.16 m³/h. Air-side ventilation
    // through the fan path must produce a measurable cooling effect on both
    // peak (summer) and trough (winter) air temperatures relative to 600FF.
    let max_drop = r600.free_float_max_temp - r650.free_float_max_temp;
    let min_drop = r600.free_float_min_temp - r650.free_float_min_temp;

    eprintln!(
        "[#824] 600FF max={:.3}°C, 650FF max={:.3}°C (drop {:.3}°C)",
        r600.free_float_max_temp, r650.free_float_max_temp, max_drop
    );
    eprintln!(
        "[#824] 600FF min={:.3}°C, 650FF min={:.3}°C (drop {:.3}°C)",
        r600.free_float_min_temp, r650.free_float_min_temp, min_drop
    );

    // Quantitative thresholds chosen well below the observed deltas
    // (max drop ~2.8°C, min drop ~6.9°C at time of writing) so trivial
    // changes to envelope coupling don't tank the test, but a regression that
    // re-zeros the night-vent path will fail loudly.
    assert!(
        max_drop > 1.0,
        "[#824] night-vent must lower 650FF peak air temperature by at least 1.0°C \
         relative to 600FF (got {max_drop:.3}°C)"
    );
    assert!(
        min_drop > 3.0,
        "[#824] night-vent must lower 650FF winter trough air temperature by at least 3.0°C \
         relative to 600FF (got {min_drop:.3}°C; spec is no-heating, fan still runs \
         18:00-07:00, cold air invades)"
    );
}

#[test]
fn night_ventilation_does_not_break_600ff() {
    // Case 600FF has no night ventilation. Any change to the air-side
    // conductance under #824 must leave 600FF behaviour bit-identical to its
    // pre-issue baseline. Pin the observed values so future regressions of
    // the inactive-night-vent code path fail loudly. Tolerance covers
    // unrelated rounding / non-deterministic floor effects.
    let r600 = validate_ashrae_140(&ASHRAE140Case::Case600FF.spec());
    eprintln!(
        "[#824] 600FF baseline: max={:.3}°C, min={:.3}°C",
        r600.free_float_max_temp, r600.free_float_min_temp
    );
    // Pinned observed values for the `validate_ashrae_140` free-function code
    // path — note this enables `model.ctf_primary = true` for FF cases, which
    // is a *different* solver path than `tests/ashrae_140_case_600_series.rs`
    // (the 600-series test file uses the standard 5R1C path and observes
    // 48.28°C / -7.70°C for 600FF). Issue #824's air-side wiring is a no-op
    // for 600FF (night_ventilation = None), so these numbers are unchanged
    // before/after this PR. They will move when #831 closes the envelope gap.
    assert!(
        (r600.free_float_max_temp - 46.25).abs() < 0.5,
        "[#824] 600FF max under validate_ashrae_140 path must be ~46.25°C \
         (got {:.3}°C); pre-#824 baseline (CTF-primary FF path)",
        r600.free_float_max_temp
    );
    assert!(
        (r600.free_float_min_temp - (-14.20)).abs() < 0.5,
        "[#824] 600FF min under validate_ashrae_140 path must be ~-14.20°C \
         (got {:.3}°C)",
        r600.free_float_min_temp
    );
}
