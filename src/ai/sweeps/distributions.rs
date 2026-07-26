//! Parameter distribution types for Monte Carlo parameter sweeps.
//!
//! Each building or environmental parameter in a sweep is described by a
//! distribution from which individual sample values are drawn. This module
//! provides the [`ParameterDistribution`] enum and a [`Choice`] helper for
//! discrete parameters (e.g. climate zone labels).
//!
//! # Distributions
//!
//! | Variant    | Use case                                         |
//! |------------|--------------------------------------------------|
//! | `Uniform`  | Wide parameter ranges where all values are equally plausible |
//! | `Normal`   | Parameters with a known central tendency         |
//! | `LogNormal`| Strictly positive, skewed parameters (e.g. conductivity) |
//!
//! All distributions are sampled via a caller-supplied `rand::Rng`, which
//! guarantees reproducibility when a seeded RNG (e.g. `StdRng::seed_from_u64`)
//! is used.

use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::LogNormal as LogNormalDist;
use rand_distr::Normal as NormalDist;
use rand_distr::NormalError;

/// Statistical distribution for a continuous parameter.
///
/// Used by [`crate::ai::sweeps::config::SweepConfig`] to describe the range
/// and shape of every swept building/environmental parameter.
#[derive(Clone, Debug)]
pub enum ParameterDistribution {
    /// Uniform distribution over `[min, max)`.
    Uniform { min: f64, max: f64 },

    /// Normal (Gaussian) distribution with `mean` and `std_dev`.
    Normal { mean: f64, std_dev: f64 },

    /// Log-normal distribution with parameters `mu` (log-mean) and `sigma`
    /// (log-std-dev).  Samples are strictly positive.
    LogNormal { mu: f64, sigma: f64 },
}

impl ParameterDistribution {
    /// Create a uniform distribution.
    ///
    /// # Panics
    /// Panics in [`Self::validate`] if `min >= max`.
    pub fn uniform(min: f64, max: f64) -> Self {
        ParameterDistribution::Uniform { min, max }
    }

    /// Create a normal distribution.
    pub fn normal(mean: f64, std_dev: f64) -> Self {
        ParameterDistribution::Normal { mean, std_dev }
    }

    /// Create a log-normal distribution.
    pub fn log_normal(mu: f64, sigma: f64) -> Self {
        ParameterDistribution::LogNormal { mu, sigma }
    }

    /// Check that the distribution parameters are valid.
    ///
    /// Returns `Ok(())` when the distribution can produce samples, or an
    /// error message describing the problem.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            ParameterDistribution::Uniform { min, max } => {
                if min >= max {
                    return Err(format!(
                        "Uniform distribution requires min < max, got min={min}, max={max}"
                    ));
                }
                if !min.is_finite() || !max.is_finite() {
                    return Err("Uniform bounds must be finite".to_string());
                }
            }
            ParameterDistribution::Normal { mean, std_dev } => {
                if *std_dev <= 0.0 {
                    return Err(format!("Normal std_dev must be positive, got {std_dev}"));
                }
                if !mean.is_finite() || !std_dev.is_finite() {
                    return Err("Normal parameters must be finite".to_string());
                }
            }
            ParameterDistribution::LogNormal { mu, sigma } => {
                if *sigma <= 0.0 {
                    return Err(format!("LogNormal sigma must be positive, got {sigma}"));
                }
                if !mu.is_finite() || !sigma.is_finite() {
                    return Err("LogNormal parameters must be finite".to_string());
                }
            }
        }
        Ok(())
    }

    /// Draw a single sample from this distribution using the provided RNG.
    ///
    /// Returns the raw sample without any clamping.  Callers that need to
    /// enforce hard bounds (e.g. humidity in `[0, 100]`) should clamp
    /// afterwards.
    ///
    /// # Errors
    /// Returns `Err` if the underlying `rand_distr` distribution could not be
    /// constructed (e.g. invalid parameters).
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<f64, NormalError> {
        match self {
            ParameterDistribution::Uniform { min, max } => {
                // gen_range panics on empty range; validate() already guards
                // against min >= max, but we use a fallback here for safety.
                if min >= max {
                    return Ok(*min);
                }
                Ok(rng.gen_range(*min..*max))
            }
            ParameterDistribution::Normal { mean, std_dev } => {
                let dist = NormalDist::new(*mean, *std_dev)?;
                Ok(dist.sample(rng))
            }
            ParameterDistribution::LogNormal { mu, sigma } => {
                let dist = LogNormalDist::new(*mu, *sigma)?;
                Ok(dist.sample(rng))
            }
        }
    }

    /// Draw a sample clamped to `[low, high]`.
    ///
    /// Useful for distributions that can produce out-of-range values (e.g.
    /// humidity from a Normal distribution).  Returns the raw sample if
    /// `low >= high` (no clamping).
    pub fn sample_clamped<R: Rng>(
        &self,
        rng: &mut R,
        low: f64,
        high: f64,
    ) -> Result<f64, NormalError> {
        let val = self.sample(rng)?;
        if low >= high {
            return Ok(val);
        }
        Ok(val.clamp(low, high))
    }
}

/// Discrete distribution for categorical parameters (e.g. climate zone labels,
/// building types).
#[derive(Clone, Debug)]
pub struct Choice {
    /// The possible values, each sampled with equal probability.
    pub values: Vec<String>,
}

impl Choice {
    /// Create a new categorical distribution.
    pub fn new<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Choice {
            values: values.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Validate that at least one value is present.
    pub fn validate(&self) -> Result<(), String> {
        if self.values.is_empty() {
            return Err("Choice must have at least one value".to_string());
        }
        Ok(())
    }

    /// Draw a single value uniformly at random.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> &str {
        let idx = rng.gen_range(0..self.values.len());
        &self.values[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_uniform_validation() {
        assert!(ParameterDistribution::uniform(0.0, 10.0).validate().is_ok());
        assert!(ParameterDistribution::uniform(10.0, 10.0)
            .validate()
            .is_err());
        assert!(ParameterDistribution::uniform(10.0, 5.0)
            .validate()
            .is_err());
        assert!(ParameterDistribution::uniform(f64::NAN, 5.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_normal_validation() {
        assert!(ParameterDistribution::normal(50.0, 5.0).validate().is_ok());
        assert!(ParameterDistribution::normal(50.0, 0.0).validate().is_err());
        assert!(ParameterDistribution::normal(50.0, -1.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_lognormal_validation() {
        assert!(ParameterDistribution::log_normal(0.0, 1.0)
            .validate()
            .is_ok());
        assert!(ParameterDistribution::log_normal(0.0, 0.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_uniform_sample_in_range() {
        let dist = ParameterDistribution::uniform(0.0, 100.0);
        let mut rng = seeded_rng();
        for _ in 0..1000 {
            let val = dist.sample(&mut rng).unwrap();
            assert!(val >= 0.0 && val < 100.0);
        }
    }

    #[test]
    fn test_normal_sample_reproducible() {
        let dist = ParameterDistribution::normal(50.0, 10.0);
        let mut rng1 = seeded_rng();
        let mut rng2 = seeded_rng();
        for _ in 0..100 {
            assert!(
                (dist.sample(&mut rng1).unwrap() - dist.sample(&mut rng2).unwrap()).abs() < 1e-12
            );
        }
    }

    #[test]
    fn test_lognormal_sample_positive() {
        let dist = ParameterDistribution::log_normal(0.0, 0.5);
        let mut rng = seeded_rng();
        for _ in 0..1000 {
            assert!(dist.sample(&mut rng).unwrap() > 0.0);
        }
    }

    #[test]
    fn test_sample_clamped() {
        let dist = ParameterDistribution::normal(60.0, 50.0);
        let mut rng = seeded_rng();
        for _ in 0..1000 {
            let val = dist.sample_clamped(&mut rng, 0.0, 100.0).unwrap();
            assert!(val >= 0.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_choice_sample() {
        let choice = Choice::new(["4A", "5A", "6A"]);
        assert!(choice.validate().is_ok());
        let mut rng = seeded_rng();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..300 {
            seen.insert(choice.sample(&mut rng).to_string());
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_choice_empty_invalid() {
        let choice = Choice::new::<[&str; 0], &str>([]);
        assert!(choice.validate().is_err());
    }
}
