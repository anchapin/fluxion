//! Monte Carlo parameter sweeps via declarative deltas (Issue #1813).
//!
//! Builds on the deterministic delta engine (`crate::analysis::delta`) by sampling
//! building-model parameters from configurable probability distributions and running
//! a full annual simulation for each draw. This is the **Phase 1 (Declarative
//! Deltas)** piece of the OSimFlow hybrid-measure approach: one base model file
//! plus a lightweight delta file describing the parameter distributions, expanded
//! into N sampled building models and simulated across `rayon` threads.
//!
//! ## Delta file format (YAML or JSON)
//!
//! ```yaml
//! samples: 1000          # number of Monte Carlo draws (default 1000, per #1813)
//! seed: 42               # optional RNG seed for reproducibility
//! warm_up_years: 2       # optional convergence warm-up years (default 2)
//! parameters:
//!   infiltration_ach:
//!     distribution: uniform
//!     min: 0.3
//!     max: 1.5
//!   window_properties.u_value:
//!     distribution: normal
//!     mean: 3.0
//!     std: 0.5
//!   window_properties.shgc:
//!     distribution: triangular
//!     min: 0.4
//!     mode: 0.7
//!     max: 0.9
//! ```
//!
//! The base model is a standalone serialized [`CaseSpec`] file referenced by
//! `--base-model <path>` on the worker entrypoint (see `cli::monte_carlo`).

use crate::analysis::delta::{run_simulation, set_nested, SimulationResult};
use crate::validation::ashrae_140_cases::CaseSpec;
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Default number of Monte Carlo samples (per Issue #1813 acceptance criterion).
pub const DEFAULT_SAMPLES: usize = 1000;

/// Default RNG seed used when the delta file omits `seed`. A fixed default keeps
/// sweeps reproducible out of the box; callers can override per-run.
pub const DEFAULT_SEED: u64 = 0x5EED_1813;

/// Default convergence warm-up years (matches the delta engine default).
pub const DEFAULT_WARM_UP_YEARS: u32 = 2;

/// Supported probability distributions for a swept parameter.
///
/// Each variant maps 1:1 onto a `rand_distr` sampler. Values are sampled in the
/// same units the [`CaseSpec`] field expects (e.g. W/m²K for U-values, ACH for
/// infiltration), so the caller is responsible for choosing physically sensible
/// distribution parameters.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "distribution", rename_all = "lowercase")]
pub enum Distribution {
    /// Continuous uniform on `[min, max]`.
    Uniform { min: f64, max: f64 },
    /// Normal (Gaussian) with given `mean` and `std`.
    Normal { mean: f64, std: f64 },
    /// Lognormal: `mean` and `std` describe the underlying normal in log-space.
    Lognormal { mean: f64, std: f64 },
    /// Triangular on `[min, max]` peaked at `mode`.
    Triangular { min: f64, mode: f64, max: f64 },
    /// Degenerate distribution: always returns `value`. Useful for pinning a
    /// parameter while sweeping others, or for control variates.
    Fixed { value: f64 },
}

impl Distribution {
    /// Draw a single sample using the supplied RNG.
    ///
    /// `Fixed`, `Uniform`, and `Triangular` are infallible. `Normal`/`Lognormal`
    /// can fail if `std <= 0`; that surfaces as an error so callers don't get
    /// silent `NaN` propagation into the physics solver.
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Result<f64> {
        use rand_distr::{
            Distribution as _, LogNormal, Normal, Triangular, Uniform as UniformDistr,
        };
        Ok(match self {
            Distribution::Fixed { value } => *value,
            Distribution::Uniform { min, max } => {
                if max < min {
                    anyhow::bail!("uniform: max ({max}) < min ({min})");
                }
                UniformDistr::new_inclusive(*min, *max).unwrap().sample(rng)
            }
            Distribution::Normal { mean, std } => {
                if *std <= 0.0 {
                    anyhow::bail!("normal: std must be > 0 (got {std})");
                }
                Normal::new(*mean, *std)?.sample(rng)
            }
            Distribution::Lognormal { mean, std } => {
                if *std <= 0.0 {
                    anyhow::bail!("lognormal: std must be > 0 (got {std})");
                }
                LogNormal::new(*mean, *std)?.sample(rng)
            }
            Distribution::Triangular { min, mode, max } => {
                if !(min <= mode && mode <= max) {
                    anyhow::bail!(
                        "triangular: require min ({min}) <= mode ({mode}) <= max ({max})"
                    );
                }
                Triangular::new(*min, *max, *mode)?.sample(rng)
            }
        })
    }
}

/// Declarative Monte Carlo delta specification, loaded from the `--delta-file`.
///
/// The base model is supplied separately (`--base-model`) so the same delta file
/// can be replayed against different base models. Parameter paths use dot notation
/// that matches the serialized [`CaseSpec`] field tree (e.g.
/// `window_properties.u_value`, `infiltration_ach`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MonteCarloDelta {
    /// Number of Monte Carlo samples to draw. Defaults to [`DEFAULT_SAMPLES`].
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// RNG seed for reproducibility. Defaults to [`DEFAULT_SEED`].
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Convergence warm-up years before collecting results. Defaults to
    /// [`DEFAULT_WARM_UP_YEARS`].
    #[serde(default = "default_warm_up_years")]
    pub warm_up_years: u32,
    /// Parameter name → distribution. Names use dot notation against the
    /// serialized `CaseSpec` tree.
    #[serde(default)]
    pub parameters: HashMap<String, Distribution>,
}

fn default_samples() -> usize {
    DEFAULT_SAMPLES
}
fn default_seed() -> u64 {
    DEFAULT_SEED
}
fn default_warm_up_years() -> u32 {
    DEFAULT_WARM_UP_YEARS
}

impl Default for MonteCarloDelta {
    fn default() -> Self {
        MonteCarloDelta {
            samples: DEFAULT_SAMPLES,
            seed: DEFAULT_SEED,
            warm_up_years: DEFAULT_WARM_UP_YEARS,
            parameters: HashMap::new(),
        }
    }
}

impl MonteCarloDelta {
    /// Parse a delta file (YAML or JSON, selected by extension) from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read delta file {}", path.display()))?;
        Self::from_str_ext(&text, path)
    }

    /// Parse from a string with an explicit extension hint (`.yaml`/`.yml` → YAML,
    /// `.json` → JSON, otherwise YAML).
    pub fn from_str_ext(text: &str, path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "json" {
            serde_json::from_str(text)
                .with_context(|| format!("failed to parse JSON delta file {}", path.display()))
        } else {
            serde_yaml::from_str(text)
                .with_context(|| format!("failed to parse YAML delta file {}", path.display()))
        }
    }

    /// Validate internal consistency (samples > 0, distributions well-formed).
    pub fn validate(&self) -> Result<()> {
        use rand::SeedableRng;
        if self.samples == 0 {
            anyhow::bail!("delta file: `samples` must be > 0");
        }
        for (name, dist) in &self.parameters {
            dist.sample(&mut rand::rngs::StdRng::seed_from_u64(0))
                .with_context(|| format!("distribution for parameter '{name}' is invalid"))?;
        }
        Ok(())
    }
}

/// A single drawn sample: the realized value for each parameter path.
#[derive(Debug, Clone, Serialize)]
pub struct ParameterSample {
    /// Monotonic draw index (0-based).
    pub index: usize,
    /// parameter path → sampled value.
    pub values: HashMap<String, f64>,
}

/// Draw `delta.samples` parameter samples using a seeded RNG.
///
/// Deterministic for a given `MonteCarloDelta` (seed + parameter ordering). The
/// parameter iteration order is fixed by sorting the parameter names so the
/// output stream is stable across Rust hash-map randomness.
pub fn sample_parameters(delta: &MonteCarloDelta) -> Result<Vec<ParameterSample>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(delta.seed);
    let mut names: Vec<&String> = delta.parameters.keys().collect();
    names.sort();
    let mut out = Vec::with_capacity(delta.samples);
    for index in 0..delta.samples {
        let mut values = HashMap::with_capacity(names.len());
        for name in &names {
            let dist = &delta.parameters[*name];
            let mut v = dist.sample(&mut rng)?;
            // Guard against NaN/inf leaking into the physics solver.
            if !v.is_finite() {
                v = match dist {
                    Distribution::Fixed { value } => *value,
                    Distribution::Uniform { min, .. } => *min,
                    Distribution::Normal { mean, .. } | Distribution::Lognormal { mean, .. } => {
                        *mean
                    }
                    Distribution::Triangular { mode, .. } => *mode,
                };
            }
            values.insert((*name).clone(), v);
        }
        out.push(ParameterSample { index, values });
    }
    Ok(out)
}

/// Apply a parameter sample to a base [`CaseSpec`] and return the patched spec.
///
/// Reuses the delta engine's `set_nested` so the dot-notation semantics match the
/// existing deterministic delta workflow exactly (deep merge into the serialized
/// `CaseSpec` tree).
pub fn apply_sample(base: &CaseSpec, sample: &ParameterSample) -> Result<CaseSpec> {
    let mut yaml = serde_yaml::to_value(base)
        .context("failed to serialize base CaseSpec to YAML for patching")?;
    for (path, value) in &sample.values {
        set_nested(&mut yaml, path, serde_yaml::to_value(value)?)
            .with_context(|| format!("failed to apply sampled parameter '{path}'"))?;
    }
    serde_yaml::from_value(yaml).context("failed to deserialize patched CaseSpec")
}

/// Result of a single Monte Carlo draw: the inputs and the simulation outputs.
#[derive(Debug, Clone, Serialize)]
pub struct MonteCarloResult {
    /// Draw index (matches [`ParameterSample::index`]).
    pub index: usize,
    /// The sampled parameter values that produced this result.
    pub inputs: HashMap<String, f64>,
    pub annual_heating_mwh: f64,
    pub annual_cooling_mwh: f64,
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    /// Error message if the simulation for this draw failed (kept separate so one
    /// bad draw doesn't abort the whole sweep).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl MonteCarloResult {
    /// Build a successful result from a simulation output.
    pub fn from_sim(index: usize, inputs: HashMap<String, f64>, sim: SimulationResult) -> Self {
        MonteCarloResult {
            index,
            inputs,
            annual_heating_mwh: sim.annual_heating_mwh,
            annual_cooling_mwh: sim.annual_cooling_mwh,
            peak_heating_kw: sim.peak_heating_kw,
            peak_cooling_kw: sim.peak_cooling_kw,
            error: None,
        }
    }
    /// Build a failed result (used by the sequential fallback so a single bad
    /// draw doesn't abort the sweep, mirroring `run_sweep`'s parallel behaviour).
    pub fn from_err(index: usize, inputs: HashMap<String, f64>, err: anyhow::Error) -> Self {
        MonteCarloResult {
            index,
            inputs,
            annual_heating_mwh: f64::NAN,
            annual_cooling_mwh: f64::NAN,
            peak_heating_kw: f64::NAN,
            peak_cooling_kw: f64::NAN,
            error: Some(format!("{err:#}")),
        }
    }
}

/// Summary statistics over the successful draws of a sweep, per output metric.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SweepStatistics {
    pub count: usize,
    pub failures: usize,
    pub heating_mwh_mean: f64,
    pub heating_mwh_std: f64,
    pub cooling_mwh_mean: f64,
    pub cooling_mwh_std: f64,
    pub heating_mwh_p05: f64,
    pub heating_mwh_p95: f64,
    pub cooling_mwh_p05: f64,
    pub cooling_mwh_p95: f64,
}

/// Aggregate a collection of [`MonteCarloResult`] into summary statistics.
///
/// Failed draws (those carrying an `error`) are excluded from the metric
/// statistics but counted in `failures`.
pub fn summarize(results: &[MonteCarloResult]) -> SweepStatistics {
    let ok: Vec<&MonteCarloResult> = results.iter().filter(|r| r.error.is_none()).collect();
    let n = ok.len();
    let mut stats = SweepStatistics {
        count: n,
        failures: results.len().saturating_sub(n),
        ..Default::default()
    };
    if n == 0 {
        return stats;
    }
    let heating: Vec<f64> = ok.iter().map(|r| r.annual_heating_mwh).collect();
    let cooling: Vec<f64> = ok.iter().map(|r| r.annual_cooling_mwh).collect();
    let (hm, hs) = mean_std(&heating);
    let (cm, cs) = mean_std(&cooling);
    stats.heating_mwh_mean = hm;
    stats.heating_mwh_std = hs;
    stats.cooling_mwh_mean = cm;
    stats.cooling_mwh_std = cs;
    stats.heating_mwh_p05 = percentile(&heating, hm, 5.0);
    stats.heating_mwh_p95 = percentile(&heating, hm, 95.0);
    stats.cooling_mwh_p05 = percentile(&cooling, cm, 5.0);
    stats.cooling_mwh_p95 = percentile(&cooling, cm, 95.0);
    stats
}

/// Run a full Monte Carlo sweep: sample → patch → simulate, in parallel.
///
/// Parallelism is at the **sample level only** (per AGENTS.md / batch-oracle
/// guidance): each draw patches its own `CaseSpec` clone and runs an independent
/// 8760-step simulation. The inner simulation loop stays single-threaded to avoid
/// rayon thread-pool exhaustion.
///
/// `collect_hourly` controls whether each draw retains hourly diagnostics
/// (disabled by default — Monte Carlo sweeps aggregate annual metrics, not
/// hourly traces).
pub fn run_sweep(
    base: &CaseSpec,
    delta: &MonteCarloDelta,
    collect_hourly: bool,
) -> Result<Vec<MonteCarloResult>> {
    delta.validate()?;
    let samples = sample_parameters(delta)?;
    let warm_up = delta.warm_up_years;
    let results: Vec<MonteCarloResult> = samples
        .par_iter()
        .map(|sample| match apply_sample(base, sample) {
            Ok(spec) => match run_simulation(&spec, collect_hourly, warm_up) {
                Ok(sim) => MonteCarloResult::from_sim(sample.index, sample.values.clone(), sim),
                Err(e) => MonteCarloResult::from_err(sample.index, sample.values.clone(), e),
            },
            Err(e) => MonteCarloResult::from_err(sample.index, sample.values.clone(), e),
        })
        .collect();
    Ok(results)
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let denom = (n - 1.0).max(1.0);
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / denom;
    (mean, variance.sqrt())
}

/// Linear-interpolation percentile of an unsorted sample (nearest-rank fallback).
///
/// `pct` is in `[0, 100]`. `fallback_mean` is returned for empty inputs so the
/// caller's serialization never emits `NaN` for a struct field.
fn percentile(values: &[f64], fallback_mean: f64, pct: f64) -> f64 {
    if values.is_empty() {
        return fallback_mean;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (pct / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;
    use rand::SeedableRng;

    fn two_param_delta(seed: u64) -> MonteCarloDelta {
        let mut parameters = HashMap::new();
        parameters.insert(
            "infiltration_ach".to_string(),
            Distribution::Uniform { min: 0.3, max: 1.5 },
        );
        parameters.insert(
            "window_properties.u_value".to_string(),
            Distribution::Normal {
                mean: 3.0,
                std: 0.3,
            },
        );
        MonteCarloDelta {
            samples: 50,
            seed,
            warm_up_years: 0,
            parameters,
        }
    }

    #[test]
    fn distribution_fixed_returns_constant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let d = Distribution::Fixed { value: 1.25 };
        for _ in 0..10 {
            assert_eq!(d.sample(&mut rng).unwrap(), 1.25);
        }
    }

    #[test]
    fn distribution_uniform_respects_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let d = Distribution::Uniform {
            min: 0.0,
            max: 10.0,
        };
        for _ in 0..100 {
            let v = d.sample(&mut rng).unwrap();
            assert!((0.0..=10.0).contains(&v), "{v} out of bounds");
        }
    }

    #[test]
    fn distribution_normal_rejects_nonpositive_std() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let d = Distribution::Normal {
            mean: 1.0,
            std: 0.0,
        };
        assert!(d.sample(&mut rng).is_err());
    }

    #[test]
    fn distribution_triangular_rejects_bad_mode() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let d = Distribution::Triangular {
            min: 0.0,
            mode: 5.0, // mode > max
            max: 1.0,
        };
        assert!(d.sample(&mut rng).is_err());
    }

    #[test]
    fn sampling_is_deterministic_for_fixed_seed() {
        let delta = two_param_delta(99);
        let a = sample_parameters(&delta).unwrap();
        let b = sample_parameters(&delta).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.values, y.values, "non-deterministic sampling");
        }
    }

    #[test]
    fn sampling_different_seeds_yield_different_draws() {
        let a = sample_parameters(&two_param_delta(1)).unwrap();
        let b = sample_parameters(&two_param_delta(2)).unwrap();
        assert_eq!(a.len(), b.len());
        let diffs = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.values != y.values)
            .count();
        assert!(diffs > 0, "different seeds produced identical draws");
    }

    #[test]
    fn apply_sample_overrides_base_field() {
        let base = ASHRAE140Case::Case600.spec();
        let sample = ParameterSample {
            index: 0,
            values: HashMap::from([("infiltration_ach".to_string(), 1.234)]),
        };
        let patched = apply_sample(&base, &sample).unwrap();
        assert!(
            (patched.infiltration_ach - 1.234).abs() < 1e-9,
            "patched infiltration_ach = {}",
            patched.infiltration_ach
        );
    }

    #[test]
    fn apply_sample_supports_nested_path() {
        let base = ASHRAE140Case::Case600.spec();
        let sample = ParameterSample {
            index: 0,
            values: HashMap::from([("window_properties.u_value".to_string(), 4.5)]),
        };
        let patched = apply_sample(&base, &sample).unwrap();
        assert!(
            (patched.window_properties.u_value - 4.5).abs() < 1e-9,
            "patched window u_value = {}",
            patched.window_properties.u_value
        );
    }

    #[test]
    fn run_sweep_small_count_runs_all_draws() {
        // 5 draws with 0 warm-up years keeps the test fast while still exercising
        // the full patch → simulate pipeline end-to-end.
        let base = ASHRAE140Case::Case600.spec();
        let delta = two_param_delta(7);
        let results = run_sweep(&base, &delta, false).unwrap();
        assert_eq!(results.len(), delta.samples);
        assert!(results.iter().all(|r| r.error.is_none()));
        // Indices are monotonic 0..samples.
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.annual_heating_mwh.is_finite());
        }
    }

    #[test]
    fn summarize_excludes_failures() {
        let results = vec![
            MonteCarloResult {
                index: 0,
                inputs: HashMap::new(),
                annual_heating_mwh: 5.0,
                annual_cooling_mwh: 3.0,
                peak_heating_kw: 10.0,
                peak_cooling_kw: 15.0,
                error: None,
            },
            MonteCarloResult {
                index: 1,
                inputs: HashMap::new(),
                annual_heating_mwh: 7.0,
                annual_cooling_mwh: 5.0,
                peak_heating_kw: 12.0,
                peak_cooling_kw: 17.0,
                error: None,
            },
            MonteCarloResult {
                index: 2,
                inputs: HashMap::new(),
                annual_heating_mwh: f64::NAN,
                annual_cooling_mwh: f64::NAN,
                peak_heating_kw: f64::NAN,
                peak_cooling_kw: f64::NAN,
                error: Some("sim failed".to_string()),
            },
        ];
        let stats = summarize(&results);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.failures, 1);
        assert!((stats.heating_mwh_mean - 6.0).abs() < 1e-9);
    }

    #[test]
    fn delta_yaml_roundtrip() {
        let delta = two_param_delta(42);
        let yaml = serde_yaml::to_string(&delta).unwrap();
        let parsed: MonteCarloDelta = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.samples, delta.samples);
        assert_eq!(parsed.seed, delta.seed);
        assert_eq!(parsed.parameters.len(), delta.parameters.len());
        parsed.validate().unwrap();
    }

    #[test]
    fn delta_from_file_parses_yaml() {
        let dir = std::env::temp_dir().join("mc1813_yaml");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("delta.yaml");
        let yaml = "\
samples: 10
seed: 7
parameters:
  infiltration_ach:
    distribution: uniform
    min: 0.5
    max: 1.0
";
        std::fs::write(&path, yaml).unwrap();
        let delta = MonteCarloDelta::from_file(&path).unwrap();
        assert_eq!(delta.samples, 10);
        assert_eq!(delta.seed, 7);
        assert_eq!(delta.parameters.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delta_from_file_parses_json() {
        let dir = std::env::temp_dir().join("mc1813_json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("delta.json");
        let json = r#"{
  "samples": 8,
  "seed": 3,
  "parameters": {
    "infiltration_ach": { "distribution": "uniform", "min": 0.4, "max": 0.8 }
  }
}"#;
        std::fs::write(&path, json).unwrap();
        let delta = MonteCarloDelta::from_file(&path).unwrap();
        assert_eq!(delta.samples, 8);
        assert_eq!(delta.seed, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_rejects_zero_samples() {
        let mut delta = MonteCarloDelta::default();
        delta.samples = 0;
        assert!(delta.validate().is_err());
    }
}
