//! Unit tests for the Multi-Zone Cold Start Gate's warm-sample guard
//! (Issue #3685).
//!
//! These tests are intentionally INDEPENDENT of the `ort` feature: the
//! guard logic in `tests/cold_start_guard/mod.rs` is pure std f64 math,
//! so the collapsed-denominator regression proof below runs on any
//! default-features build (`cargo test -p fluxion --test
//! cold_start_guard_test`) without paying the ~30 MB ort transitive
//! build. The ort-gated measurement that CONSUMES these helpers lives
//! in `tests/surrogate_cold_start_test.rs`.
//!
//! ## Regression being pinned (Issue #3685)
//!
//! On 2026-09-11 the Multi-Zone Cold Start Gate failed spuriously on 3
//! PRs (#3670, #3678, #3679): the warm-time measurement intermittently
//! read ~0.002 ms, and dividing the healthy cold time (~0.019–0.041 ms)
//! by that collapsed denominator inflated the cold/warm ratio to 10–16
//! against the CI thresholds. The guard must reject such samples BEFORE
//! they become a denominator — the first test below feeds exactly the
//! observed collapsed readings and asserts no ratio is produced.

#[path = "cold_start_guard/mod.rs"]
mod cold_start_guard;

use cold_start_guard::{
    percentile_sorted, valid_warm_samples, warm_p95_statistic, WARM_EPSILON_MS,
    WARM_MIN_VALID_SAMPLES,
};

/// THE Issue #3685 regression proof: collapsed warm readings (the exact
/// values observed in the failing CI jobs — warm=0.002/0.002/0.003 ms
/// against cold=0.036–0.041 ms, i.e. ratios 14–18) must be REJECTED by
/// the epsilon guard so they can never be divided by.
///
/// With every sample invalid, `warm_p95_statistic` returns `None` — the
/// caller must report the ratio as unmeasurable instead of emitting the
/// bogus ~15× figure that tripped the gate spuriously.
#[test]
fn collapsed_warm_samples_are_rejected_not_used_as_denominator() {
    // Raw warm readings exactly as captured from the failing
    // `Multi-Zone Cold Start Gate` job on 2026-09-12 (develop run
    // 34679814428): warm=0.002, 0.002, 0.003 ms (plus the surrounding
    // artifact band).
    let collapsed: [f64; 9] = [
        0.0016, 0.002, 0.002, 0.0025, 0.003, 0.002, 0.0018, 0.0022, 0.0027,
    ];

    let valid = valid_warm_samples(&collapsed);
    assert!(
        valid.is_empty(),
        "collapsed samples below the {WARM_EPSILON_MS:?} ms epsilon must be \
         rejected, got {valid:?}"
    );
    assert!(
        warm_p95_statistic(&collapsed).is_none(),
        "no denominator may be produced from all-collapsed samples — \
         fabricating one is what inflated the ratio to 10-16 in #3685"
    );

    // Sanity: the pre-guard code WOULD have divided by ~0.002 ms —
    // cold=0.040 / warm=0.002 = 20x, exactly the order of the bogus
    // 14-18x ratios in the failing jobs. The guard makes that path
    // unreachable: no denominator exists at all.
    let cold_ms = 0.040_f64;
    assert_eq!(
        warm_p95_statistic(&collapsed).map(|denom| cold_ms / denom),
        None,
        "guarded ratio must not exist when every warm sample is collapsed"
    );
}

/// p95 over a MIXED pool (collapsed artifacts + honest samples) must be
/// computed from the VALID subset only, so a few artifacts cannot drag
/// the denominator down.
#[test]
fn p95_statistic_excludes_collapsed_artifacts_from_mixed_pool() {
    // 3 collapsed artifacts + 6 honest ~0.020 ms samples.
    let mixed: [f64; 9] = [0.002, 0.0025, 0.002, 0.02, 0.021, 0.022, 0.02, 0.021, 0.022];

    let valid = valid_warm_samples(&mixed);
    assert_eq!(valid.len(), 6, "artifacts below epsilon must be dropped");
    assert!(valid.iter().all(|ms| *ms >= WARM_EPSILON_MS));

    let p95 = warm_p95_statistic(&mixed).expect("6 valid >= min valid samples");
    assert!(
        (p95 - 0.022).abs() < 1e-9,
        "p95 of valid samples should sit in the honest upper tail, got {p95}"
    );

    // The guarded ratio stays sane (~1x) where a raw-minimum denominator
    // would have produced 0.020 / 0.002 = 10x.
    let cold_ms = 0.020;
    let guarded_ratio = cold_ms / p95;
    let raw_ratio = cold_ms / 0.002;
    assert!(
        guarded_ratio < 2.0 && raw_ratio >= 10.0,
        "guard must keep the ratio sane (guarded={guarded_ratio}, raw={raw_ratio})"
    );
}

/// Fail-closed: fewer than `WARM_MIN_VALID_SAMPLES` valid samples →
/// `None` (unmeasurable), never a denominator fabricated from an
/// under-sampled pool. Also pins the epsilon boundary itself (>= vs <).
#[test]
fn under_sampled_pool_is_unmeasurable_and_epsilon_boundary_is_exact() {
    // Only 2 valid samples — below WARM_MIN_VALID_SAMPLES.
    let under: [f64; 4] = [0.002, 0.02, 0.5, 0.002];
    assert_eq!(valid_warm_samples(&under).len(), 2);
    assert_eq!(
        warm_p95_statistic(&under),
        None,
        "under-sampled pool must be unmeasurable (fail-closed)"
    );

    // Exactly WARM_MIN_VALID_SAMPLES valid samples → measurable.
    let exactly: [f64; 5] = [0.01, 0.01, 0.01, 0.01, 0.01];
    assert_eq!(valid_warm_samples(&exactly).len(), WARM_MIN_VALID_SAMPLES);
    assert_eq!(warm_p95_statistic(&exactly), Some(0.01));

    // Epsilon boundary: sample == epsilon is VALID (>=), just below is not.
    let boundary: [f64; 3] = [
        WARM_EPSILON_MS - f64::EPSILON,
        WARM_EPSILON_MS,
        WARM_EPSILON_MS + 1e-6,
    ];
    let valid = valid_warm_samples(&boundary);
    assert_eq!(
        valid,
        vec![WARM_EPSILON_MS, WARM_EPSILON_MS + 1e-6],
        "samples >= epsilon are valid; strictly-below are invalid"
    );

    // NaN / inf readings are invalid regardless of magnitude.
    assert!(valid_warm_samples(&[f64::NAN, f64::INFINITY, 0.02]).len() == 1);

    // Nearest-rank percentile sanity on a known distribution.
    let sorted: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    assert!((percentile_sorted(&sorted, 0.95) - 95.0).abs() < 1e-9);
    assert!((percentile_sorted(&sorted, 0.0) - 1.0).abs() < 1e-9);
    assert!((percentile_sorted(&sorted, 1.0) - 100.0).abs() < 1e-9);
}
