//! Pure statistics helpers for the Multi-Zone Cold Start Gate (Issue #2919).
//!
//! Shared by `tests/surrogate_cold_start_test.rs` (the PR-blocking gate
//! measurement, `#[cfg(feature = "ort")]`) and
//! `tests/cold_start_guard_test.rs` (always-compiled unit tests for the
//! guard logic itself). This module must stay dependency-free (std only)
//! so the guard tests run without the `ort` feature.
//!
//! ## Why these helpers exist (Issue #3685)
//!
//! The gate's cold/warm ratio intermittently collapsed to 10–16× against
//! the CI thresholds because the warm-time measurement on the 193-byte
//! dummy fixture intermittently reads ~0.002–0.003 ms — below any
//! physically-meaningful ORT forward-pass cost. Dividing the (healthy,
//! ~0.019–0.041 ms) cold time by such a collapsed denominator produces a
//! garbage ratio that trips the CI gate spuriously (3 PRs on 2026-09-11:
//! #3670, #3678, #3679, each fixed by a no-op re-run).
//!
//! The hardening contract:
//!
//! 1. **Epsilon guard** — a warm sample below [`WARM_EPSILON_MS`] is an
//!    INVALID sample: it is discarded and never used as a ratio
//!    denominator.
//! 2. **Bounded re-measure** — callers re-measure warm batches until
//!    enough valid samples exist or a bounded attempt count is exhausted
//!    (see `WARM_MAX_BATCH_ATTEMPTS` in the gate test).
//! 3. **Robust statistic** — the denominator is the p95 of the valid
//!    warm samples, not a single reading: collapsed samples are low
//!    outliers and p95 ignores the bottom of the distribution.
//! 4. **Fail-closed** — when fewer than [`WARM_MIN_VALID_SAMPLES`] valid
//!    samples survive, [`warm_p95_statistic`] returns `None` and the
//!    ratio is UNMEASURABLE (never fabricated from invalid data).

/// Warm samples below this floor (milliseconds) are measurement
/// artifacts, not real ORT forward passes (Issue #3685).
///
/// A warm `predict_loads_onnx` call allocates scratch tensors, runs the
/// ORT session, and extracts the output tensor; that work cannot
/// complete in a few microseconds on any supported runner. Readings in
/// the 0.002–0.003 ms band are timer/cache artifacts of the trivial
/// 193-byte CI fixture — exactly the regime where the cold/warm ratio
/// loses meaning. On real models (256 MiB shipped surrogate) warm
/// latency is ~5–10 ms, three orders of magnitude above this floor, so
/// the guard never rejects honest samples where the ratio matters.
pub const WARM_EPSILON_MS: f64 = 0.01;

/// Minimum number of VALID (>= [`WARM_EPSILON_MS`]) warm samples required
/// before a p95 statistic is computed (Issue #3685).
///
/// Below this count the denominator would be dominated by individual
/// sample jitter, so the statistic is `None` (unmeasurable) instead.
pub const WARM_MIN_VALID_SAMPLES: usize = 5;

/// Filter `samples_ms` down to the VALID warm samples (Issue #3685).
///
/// A sample is valid when it is finite and >= [`WARM_EPSILON_MS`].
/// Collapsed artifact readings (e.g. 0.002 ms) are dropped here and can
/// therefore never reach the ratio denominator.
pub fn valid_warm_samples(samples_ms: &[f64]) -> Vec<f64> {
    samples_ms
        .iter()
        .copied()
        .filter(|ms| ms.is_finite() && *ms >= WARM_EPSILON_MS)
        .collect()
}

/// Nearest-rank percentile of an ASCENDING-sorted slice.
///
/// `pct` is in `[0.0, 1.0]` (0.95 = p95). Returns the first element for
/// an empty slice only when `pct <= 0.0`; callers must not pass an empty
/// slice for `pct > 0.0` (the p95 path asserts instead of returning a
/// fabricated value).
pub fn percentile_sorted(sorted_asc: &[f64], pct: f64) -> f64 {
    assert!(!sorted_asc.is_empty(), "percentile of empty slice");
    assert!(
        (0.0..=1.0).contains(&pct),
        "percentile must be in [0, 1], got {pct}"
    );
    let n = sorted_asc.len();
    // Nearest-rank: smallest index whose cumulative rank >= pct * n.
    let rank = ((pct * n as f64).ceil() as usize).clamp(1, n);
    sorted_asc[rank - 1]
}

/// Robust warm-time denominator: p95 over the VALID warm samples
/// (Issue #3685).
///
/// Returns `None` when fewer than [`WARM_MIN_VALID_SAMPLES`] valid
/// samples survive the epsilon guard — the caller must treat the
/// cold/warm ratio as UNMEASURABLE in that case (fail-closed: never
/// divide by a collapsed or under-sampled denominator).
pub fn warm_p95_statistic(samples_ms: &[f64]) -> Option<f64> {
    let mut valid = valid_warm_samples(samples_ms);
    if valid.len() < WARM_MIN_VALID_SAMPLES {
        return None;
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(percentile_sorted(&valid, 0.95))
}
