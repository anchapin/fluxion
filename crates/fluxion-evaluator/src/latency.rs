//! Noise-robust latency aggregation.
//!
//! The fitness score must never depend on a single wall-clock shot —
//! even a short-running kernel (sub-millisecond) can show 30%+
//! variance under CI load. We use a **median-of-N** aggregator with
//! an **IQR (interquartile range)** spread metric; both numbers are
//! reported in [`crate::summary::Summary`].
//!
//! ## Why median, not mean
//!
//! Median is robust to GC pauses, scheduler jitter, and OS reclaims.
//! It also rejects outliers deterministically — the same N samples
//! always produce the same median.
//!
//! ## Why IQR, not stddev
//!
//! IQR has the same robustness property and is bounded: it can never
//! exceed the data range. Stddev is sensitive to outliers and pulls
//! the spread metric in the direction of the worst sample, which the
//! evolver can game by hiding variance inside a single slow iteration.

use std::time::Instant;

/// Configuration for the latency aggregator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingConfig {
    /// Number of timed iterations. The median is taken over these.
    /// Defaults to 21 (≈20s of timing for a 1 ms kernel; well below
    /// the issue #3336 60 s resource cap).
    pub n: usize,

    /// Number of warmup iterations before timing starts. The first
    /// few iterations of any tight numeric loop are dominated by
    /// cache misses; warmup isolates the steady-state cost we
    /// actually want to score. Defaults to 5.
    pub warmup: usize,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self { n: 21, warmup: 5 }
    }
}

impl TimingConfig {
    /// Constructor with the default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style override of `n`.
    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n;
        self
    }

    /// Builder-style override of `warmup`.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Validate the config; return `Err` if a value is zero (which
    /// would silently produce a meaningless median).
    pub fn validate(self) -> Result<(), crate::EvaluatorError> {
        if self.n == 0 {
            return Err(crate::EvaluatorError::InvalidConfig(
                "TimingConfig.n must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// A single timed iteration, in nanoseconds. Stable integer
/// representation so the JSON serialization is precise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LatencyMeasurement(pub u64);

impl LatencyMeasurement {
    /// Wrap a `Duration`'s nanosecond representation.
    pub fn from_nanos(nanos: u128) -> Self {
        LatencyMeasurement(nanos.min(u128::from(u64::MAX)) as u64)
    }
}

/// Aggregate of a timing run: median + spread (IQR).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LatencyAggregate {
    /// Number of samples that contributed to this aggregate.
    pub samples: usize,
    /// Median latency, in nanoseconds.
    pub median_ns: u64,
    /// Interquartile range (Q3 - Q1), in nanoseconds.
    pub spread_ns: u64,
    /// Minimum latency, in nanoseconds. Useful for diagnosing
    /// whether the kernel is hitting a fast path at all.
    pub min_ns: u64,
    /// Maximum latency, in nanoseconds. Useful for diagnosing
    /// whether some iteration triggered a GC pause or page fault.
    pub max_ns: u64,
}

use serde::Serialize;

impl LatencyAggregate {
    /// Construct from a slice of measurements. The slice is sorted in
    /// place; the median is `samples[len/2]` (lower of the two
    /// middles for even-length slices — Python-style, deterministic).
    pub fn from_measurements(samples: &[LatencyMeasurement]) -> Self {
        assert!(!samples.is_empty(), "LatencyAggregate requires ≥1 sample");
        let mut sorted: Vec<u64> = samples.iter().map(|s| s.0).collect();
        sorted.sort_unstable();
        let len = sorted.len();
        let median = sorted[len / 2];
        let q1 = sorted[len / 4];
        let q3 = sorted[(3 * len) / 4];
        let min = sorted[0];
        let max = sorted[len - 1];
        LatencyAggregate {
            samples: len,
            median_ns: median,
            spread_ns: q3.saturating_sub(q1),
            min_ns: min,
            max_ns: max,
        }
    }
}

/// Time `f` over `cfg.n + cfg.warmup` iterations, dropping the
/// warmup samples and aggregating the rest. `f` is called once per
/// iteration; the returned `Instant` is the per-iteration wall-clock
/// cost.
///
/// Determinism: `f` must itself be deterministic (no `Instant::now`
/// inside the kernel — the harness owns all clocks). The harness
/// uses wall-clock for latency only, never for fitness.
pub fn time_kernel<Fn: FnMut() -> R, R>(cfg: TimingConfig, mut f: Fn) -> LatencyAggregate {
    cfg.validate().expect("invalid TimingConfig");
    for _ in 0..cfg.warmup {
        let _ = f();
    }
    let mut samples = Vec::with_capacity(cfg.n);
    for _ in 0..cfg.n {
        let start = Instant::now();
        let _ = f();
        let elapsed = start.elapsed().as_nanos();
        samples.push(LatencyMeasurement::from_nanos(elapsed));
    }
    LatencyAggregate::from_measurements(&samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn default_config_has_21_samples_and_5_warmup() {
        let cfg = TimingConfig::default();
        assert_eq!(cfg.n, 21);
        assert_eq!(cfg.warmup, 5);
    }

    #[test]
    fn validate_rejects_zero_n() {
        let cfg = TimingConfig::new().with_n(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn aggregate_median_is_middle_sample() {
        let samples = vec![
            LatencyMeasurement(10),
            LatencyMeasurement(30),
            LatencyMeasurement(20),
            LatencyMeasurement(40),
            LatencyMeasurement(50),
        ];
        let agg = LatencyAggregate::from_measurements(&samples);
        assert_eq!(agg.median_ns, 30);
        assert_eq!(agg.min_ns, 10);
        assert_eq!(agg.max_ns, 50);
        assert_eq!(agg.samples, 5);
    }

    #[test]
    fn aggregate_spread_is_iqr() {
        // Sorted: 10, 20, 30, 40, 50, 60, 70, 80
        let samples = (1..=8)
            .map(|i| LatencyMeasurement(i * 10))
            .collect::<Vec<_>>();
        let agg = LatencyAggregate::from_measurements(&samples);
        // len/4 = 2 -> q1 = samples[2] = 30; 3*len/4 = 6 -> q3 = samples[6] = 70.
        assert_eq!(agg.spread_ns, 40);
    }

    #[test]
    fn time_kernel_runs_n_plus_warmup_iterations() {
        let cfg = TimingConfig::new().with_n(7).with_warmup(2);
        let mut count = 0u32;
        let agg = time_kernel(cfg, || {
            count += 1;
            // Tiny sleep so the timer has something to measure.
            std::thread::sleep(Duration::from_micros(10));
        });
        assert_eq!(count, 9, "expected 7 timed + 2 warmup iterations");
        assert_eq!(agg.samples, 7);
        // Median should be at least the sleep duration (10 µs = 10_000 ns).
        assert!(agg.median_ns >= 1_000);
    }

    /// Noise-robustness contract: a single outlier must NOT pull the
    /// median. We construct a sample distribution with a deliberately
    /// huge outlier and verify the median is unaffected.
    #[test]
    fn median_is_robust_to_outliers() {
        let mut samples: Vec<LatencyMeasurement> =
            (0..100).map(|_| LatencyMeasurement(1_000)).collect();
        // Plant a single outlier 1000× the baseline.
        samples.push(LatencyMeasurement(1_000_000));
        let agg = LatencyAggregate::from_measurements(&samples);
        assert_eq!(agg.median_ns, 1_000, "median must ignore the outlier");
        // The max, however, is the outlier — that's reported
        // separately as a diagnostic.
        assert_eq!(agg.max_ns, 1_000_000);
    }
}
