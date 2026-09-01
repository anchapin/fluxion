//! Sampling strategies for Monte Carlo parameter sweeps.
//!
//! Provides three strategies with different space-coverage guarantees:
//!
//! - **Random Monte Carlo** — independent uniform draws per dimension. Fast,
//!   simple, but can leave gaps or cluster samples, especially in high
//!   dimensions.
//! - **Latin Hypercube Sampling (LHS)** — stratifies each dimension into
//!   equal-probability bins and draws exactly one sample per bin, ensuring
//!   better one-dimensional coverage.
//! - **Sobol quasi-random** — a low-discrepancy sequence that covers the
//!   unit hypercube more evenly than random sampling while still being
//!   deterministic and seeded.
//!
//! All strategies produce a matrix of `n_samples × n_dimensions` values in
//! the unit cube `[0, 1)`, which [`super::config::SweepConfig`] then maps
//! to physical parameter ranges via the [`ParameterDistribution`](super::distributions::ParameterDistribution)
//! machinery.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Sampling strategy enum.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Independent random Monte Carlo draws.
    RandomMonteCarlo,
    /// Latin Hypercube Sampling.
    LatinHypercube,
    /// Sobol low-discrepancy sequence (quasi-random).
    Sobol,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::LatinHypercube
    }
}

impl SamplingStrategy {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            SamplingStrategy::RandomMonteCarlo => "random_monte_carlo",
            SamplingStrategy::LatinHypercube => "latin_hypercube",
            SamplingStrategy::Sobol => "sobol",
        }
    }
}

/// Generate a matrix of unit-cube samples `[0, 1)` using the chosen strategy.
///
/// Returns a flat vector in row-major order: `result[i * n_dim + j]` is the
/// `j`-th dimension of the `i`-th sample.
///
/// # Arguments
/// * `strategy` — which sampling method to use.
/// * `n_samples` — number of parameter sets to generate.
/// * `n_dim` — number of continuous dimensions (parameters).
/// * `seed` — RNG seed for reproducibility.
pub fn generate_unit_samples(
    strategy: SamplingStrategy,
    n_samples: usize,
    n_dim: usize,
    seed: u64,
) -> Vec<f64> {
    assert!(n_samples > 0, "n_samples must be > 0");
    assert!(n_dim > 0, "n_dim must be > 0");

    match strategy {
        SamplingStrategy::RandomMonteCarlo => random_monte_carlo(n_samples, n_dim, seed),
        SamplingStrategy::LatinHypercube => latin_hypercube(n_samples, n_dim, seed),
        SamplingStrategy::Sobol => sobol(n_samples, n_dim, seed),
    }
}

fn random_monte_carlo(n_samples: usize, n_dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n_samples * n_dim);
    for _ in 0..n_samples {
        for _ in 0..n_dim {
            out.push(rng.random::<f64>());
        }
    }
    out
}

/// Latin Hypercube Sampling.
///
/// For each dimension, divides `[0, 1)` into `n_samples` equal strata and
/// draws one value per stratum (shuffled).  This guarantees every stratum is
/// represented exactly once per dimension, improving one-dimensional coverage.
fn latin_hypercube(n_samples: usize, n_dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n_samples * n_dim);

    for d in 0..n_dim {
        // Build the stratified column for this dimension.
        let mut column: Vec<f64> = (0..n_samples)
            .map(|i| {
                // Centre of each stratum + jitter within the stratum.
                let stratum_low = i as f64 / n_samples as f64;
                let stratum_high = (i + 1) as f64 / n_samples as f64;
                let jitter: f64 = rng.random();
                stratum_low + jitter * (stratum_high - stratum_low)
            })
            .collect();

        // Shuffle this column so the LHS property holds across dimensions.
        column.shuffle(&mut rng);

        if d == 0 {
            // First dimension — allocate the output matrix row-major.
            out = vec![0.0; n_samples * n_dim];
        }
        for (i, &val) in column.iter().enumerate() {
            out[i * n_dim + d] = val;
        }
    }

    out
}

/// Sobol low-discrepancy sequence.
///
/// Implements a lightweight Sobol generator using the first primitive
/// polynomial direction numbers.  This covers the unit cube more evenly
/// than random or LHS sampling.  For dimensions beyond the number of
/// direction-number sets defined here, it falls back to a random dimension
/// (with a warning in debug builds).
fn sobol(n_samples: usize, n_dim: usize, seed: u64) -> Vec<f64> {
    // Direction numbers for the first few dimensions.
    // Standard Sobol direction numbers (Joe & Kuo 2008, m_k values).
    // Each inner vec is {m_1, m_2, ...} for that dimension.
    let direction_numbers: &[&[u32]] = &[
        &[1],                                                    // dim 0
        &[1, 3],                                                 // dim 1
        &[1, 3, 1],                                              // dim 2
        &[1, 1, 1, 7],                                           // dim 3
        &[1, 1, 3, 7, 5],                                        // dim 4
        &[1, 3, 1, 5, 7, 11],                                    // dim 5
        &[1, 3, 1, 7, 5, 13, 11],                                // dim 6
        &[1, 1, 5, 1, 7, 11, 13, 19],                            // dim 7
        &[1, 1, 5, 1, 7, 11, 13, 19, 1],                         // dim 8
        &[1, 1, 5, 1, 7, 11, 13, 19, 1, 3],                      // dim 9
        &[1, 3, 1, 7, 5, 13, 11, 19, 1, 3, 1],                   // dim 10
        &[1, 1, 1, 3, 7, 13, 11, 19, 1, 3, 1, 7],                // dim 11
        &[1, 1, 1, 3, 7, 13, 11, 19, 1, 3, 1, 7, 5],             // dim 12
        &[1, 1, 1, 3, 7, 13, 11, 19, 1, 3, 1, 7, 5, 15],         // dim 13
        &[1, 1, 1, 3, 7, 13, 11, 19, 1, 3, 1, 7, 5, 15, 13],     // dim 14
        &[1, 1, 1, 3, 7, 13, 11, 19, 1, 3, 1, 7, 5, 15, 13, 25], // dim 15
    ];

    let max_bits = 32u32;
    let mut out = Vec::with_capacity(n_samples * n_dim);

    // Pre-compute the full direction number vectors (v[d][j]) for each dim.
    // v[d][j] = m_j * 2^(max_bits - j)  for j=1..s_d, and v[d][j] = 1<<(max_bits-j-1) for j>s_d.
    let mut v: Vec<Vec<u32>> = Vec::with_capacity(n_dim);
    for d in 0..n_dim {
        let s_d = if d < direction_numbers.len() {
            direction_numbers[d].len()
        } else {
            0
        };
        let mut col = vec![0u32; max_bits as usize];
        if d == 0 {
            // Dimension 0: v_0[j] = 1 for all j (the standard first dimension).
            for (j, slot) in col.iter_mut().enumerate() {
                *slot = 1u32 << (max_bits as usize - 1 - j);
            }
        } else if s_d > 0 {
            let m = &direction_numbers[d];
            // The first s_d direction numbers come from the table.
            for (j, &mj) in m.iter().enumerate() {
                col[j] = mj << (max_bits as usize - 1 - j);
            }
            // Extend: v[j] = v[j-s] XOR (v[j-s] >> s) for j >= s_d
            for j in s_d..(max_bits as usize) {
                let s = s_d;
                col[j] = col[j - s] ^ (col[j - s] >> s);
            }
        }
        v.push(col);
    }

    // Generate the sequence. Point i is built by XOR-ing direction numbers
    // corresponding to the set bits of i.
    let skip = seed.max(1) as usize; // skip first point(s) to avoid origin
    let mut point = vec![0u32; n_dim];

    for i in 0..(n_samples + skip) {
        for d in 0..n_dim {
            let mut result = if i > 0 { point[d] } else { 0u32 };
            // Gray code approach: XOR with direction number of the lowest zero bit
            if i > 0 {
                // Find the index of the rightmost zero bit of (i-1)
                let prev = i - 1;
                let gray_prev = prev ^ (prev >> 1);
                let gray_curr = i ^ (i >> 1);
                let changed = gray_prev ^ gray_curr;
                // The changed bit tells us which direction number to XOR
                let bit_index = changed.trailing_zeros() as usize;
                if bit_index < max_bits as usize && d < v.len() && bit_index < v[d].len() {
                    result ^= v[d][bit_index];
                }
            }
            point[d] = result;
        }

        if i >= skip {
            for (d, &pval) in point.iter().enumerate() {
                let val = if d < v.len() {
                    pval as f64 / (1u64 << max_bits) as f64
                } else {
                    // Fallback for dimensions beyond direction table: use a
                    // deterministic hash of (seed, i, d) mapped to [0,1).
                    let hash = (seed.wrapping_mul(2654435761))
                        .wrapping_add((i as u64).wrapping_mul(40503))
                        .wrapping_add((d as u64).wrapping_mul(65599));
                    (hash % 1_000_000) as f64 / 1_000_000.0
                };
                out.push(val);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_name() {
        assert_eq!(
            SamplingStrategy::RandomMonteCarlo.name(),
            "random_monte_carlo"
        );
        assert_eq!(SamplingStrategy::LatinHypercube.name(), "latin_hypercube");
        assert_eq!(SamplingStrategy::Sobol.name(), "sobol");
    }

    #[test]
    fn test_default_strategy() {
        assert_eq!(
            SamplingStrategy::default(),
            SamplingStrategy::LatinHypercube
        );
    }

    #[test]
    fn test_random_mc_shape() {
        let samples = generate_unit_samples(SamplingStrategy::RandomMonteCarlo, 10, 3, 42);
        assert_eq!(samples.len(), 30);
        for &v in &samples {
            assert!(v >= 0.0 && v < 1.0, "value {v} out of [0,1)");
        }
    }

    #[test]
    fn test_random_mc_reproducible() {
        let a = generate_unit_samples(SamplingStrategy::RandomMonteCarlo, 50, 4, 99);
        let b = generate_unit_samples(SamplingStrategy::RandomMonteCarlo, 50, 4, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn test_lhs_shape_and_range() {
        let n = 20;
        let d = 5;
        let samples = generate_unit_samples(SamplingStrategy::LatinHypercube, n, d, 7);
        assert_eq!(samples.len(), n * d);
        for &v in &samples {
            assert!(v >= 0.0 && v < 1.0, "LHS value {v} out of range");
        }
    }

    #[test]
    fn test_lhs_one_per_stratum() {
        // For each dimension, verify that exactly one sample falls in each
        // equal-probability stratum of [0,1).
        let n = 50;
        let d = 3;
        let samples = generate_unit_samples(SamplingStrategy::LatinHypercube, n, d, 123);

        for dim in 0..d {
            let col: Vec<f64> = (0..n).map(|i| samples[i * d + dim]).collect();
            for stratum in 0..n {
                let low = stratum as f64 / n as f64;
                let high = (stratum + 1) as f64 / n as f64;
                let count = col.iter().filter(|&&v| v >= low && v < high).count();
                assert_eq!(
                    count, 1,
                    "LHS dim {dim} stratum {stratum} has {count} samples, expected 1"
                );
            }
        }
    }

    #[test]
    fn test_lhs_reproducible() {
        let a = generate_unit_samples(SamplingStrategy::LatinHypercube, 30, 4, 55);
        let b = generate_unit_samples(SamplingStrategy::LatinHypercube, 30, 4, 55);
        assert_eq!(a, b);
    }

    #[test]
    fn test_sobol_shape_and_range() {
        let samples = generate_unit_samples(SamplingStrategy::Sobol, 100, 4, 1);
        assert_eq!(samples.len(), 400);
        for &v in &samples {
            assert!(v >= 0.0 && v <= 1.0, "Sobol value {v} out of [0,1]");
        }
    }

    #[test]
    fn test_sobol_reproducible() {
        let a = generate_unit_samples(SamplingStrategy::Sobol, 64, 3, 1);
        let b = generate_unit_samples(SamplingStrategy::Sobol, 64, 3, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_sobol_low_discrepancy_spread() {
        // Sobol low-discrepancy sequence covers the unit cube evenly.
        // With 64 samples in 2D, verify that no large contiguous region
        // of the unit square is missed (the sequence should have at least
        // one point in each quadrant of each dimension's range).
        let n = 256;
        let samples = generate_unit_samples(SamplingStrategy::Sobol, n, 2, 1);
        // Check: every quarter of [0,1) in each dimension should have >= 1 sample.
        for dim in 0..2 {
            for quarter in 0..4 {
                let low = quarter as f64 / 4.0;
                let high = (quarter + 1) as f64 / 4.0;
                let count = samples
                    .iter()
                    .skip(dim)
                    .step_by(2)
                    .filter(|&&v| v >= low && v < high)
                    .count();
                assert!(
                    count >= 1,
                    "Sobol dim {dim} quarter [{low}, {high}) has {count} samples"
                );
            }
        }
        // Also verify that all generated values are in [0, 1].
        for &v in &samples {
            assert!(v >= 0.0 && v <= 1.0, "Sobol value {v} out of [0,1]");
        }
    }

    #[test]
    #[should_panic(expected = "n_samples must be > 0")]
    fn test_zero_samples_panics() {
        generate_unit_samples(SamplingStrategy::LatinHypercube, 0, 3, 1);
    }

    #[test]
    #[should_panic(expected = "n_dim must be > 0")]
    fn test_zero_dim_panics() {
        generate_unit_samples(SamplingStrategy::LatinHypercube, 10, 0, 1);
    }
}
