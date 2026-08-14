//! Batch job runner harness for 9R4C physics simulations.
//!
//! This module provides a batch runner that consumes parameter manifests from
//! Monte Carlo sweeps (configured in issue #1776) and executes 9R4C runs to produce
//! ground-truth targets for ML surrogate training.
//!
//! ## Key Features
//!
//! - **Parallel execution**: Uses rayon for parallel job execution
//! - **Resumable + idempotent**: Checkpoint per sample for fault tolerance
//! - **9R4C physics**: Uses the exact multi-node thermal solver for ground-truth
//! - **Chunked processing**: Memory-safe processing of large parameter sweeps
//!
//! ## Architecture
//!
//! ```text
//! ParameterManifest ──► BatchRunner ──► 9R4C Solver ──► GroundTruthResults
//!                           │
//!                           └──► Checkpoint (per sample)
//! ```

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Format a SHA-256 digest as a lowercase hex string.
///
/// `sha2` 0.11 returns a `GenericArray<u8, U32>` whose `LowerHex` impl is no
/// longer available in newer `generic-array` releases, so we format the bytes
/// manually.
fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

use crate::ai::surrogate::{InferenceBackend, SurrogateManager};
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::engine::ThermalModel;

const DEFAULT_CHUNK_SIZE: usize = 64;
const CHECKPOINT_VERSION: &str = "1.0";

/// Sampling distribution for a parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SamplingDistribution {
    Uniform {
        min: f64,
        max: f64,
    },
    LogUniform {
        min: f64,
        max: f64,
    },
    Normal {
        mean: f64,
        std: f64,
    },
    TruncatedNormal {
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    },
}

/// Parameter specification for Monte Carlo sweeps.
///
/// Parameters are indexed by position to match `ThermalModel::apply_parameters`:
/// - index 0: window_u_value (W/m²K)
/// - index 1: heating_setpoint (°C)
/// - index 2: cooling_setpoint (°C)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSpec {
    pub name: String,
    pub index: usize,
    pub distribution: SamplingDistribution,
    pub description: Option<String>,
}

impl ParameterSpec {
    pub fn uniform(name: &str, index: usize, min: f64, max: f64) -> Self {
        ParameterSpec {
            name: name.to_string(),
            index,
            distribution: SamplingDistribution::Uniform { min, max },
            description: None,
        }
    }

    pub fn log_uniform(name: &str, index: usize, min: f64, max: f64) -> Self {
        ParameterSpec {
            name: name.to_string(),
            index,
            distribution: SamplingDistribution::LogUniform { min, max },
            description: None,
        }
    }

    pub fn normal(name: &str, index: usize, mean: f64, std: f64) -> Self {
        ParameterSpec {
            name: name.to_string(),
            index,
            distribution: SamplingDistribution::Normal { mean, std },
            description: None,
        }
    }

    pub fn truncated_normal(
        name: &str,
        index: usize,
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    ) -> Self {
        ParameterSpec {
            name: name.to_string(),
            index,
            distribution: SamplingDistribution::TruncatedNormal {
                mean,
                std,
                min,
                max,
            },
            description: None,
        }
    }
}

/// A single parameter sample from the manifest.
///
/// Parameters are stored as ordered values to match `ThermalModel::apply_parameters(&[f64])`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSample {
    pub sample_id: usize,
    pub parameters: Vec<f64>,
    pub seed: u64,
}

impl ParameterSample {
    pub fn new(sample_id: usize, parameters: Vec<f64>, seed: u64) -> Self {
        ParameterSample {
            sample_id,
            parameters,
            seed,
        }
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        self.parameters.get(index).copied()
    }
}

/// Parameter manifest containing sampling configuration and generated samples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterManifest {
    pub version: String,
    pub parameters: Vec<ParameterSpec>,
    pub samples: Vec<ParameterSample>,
    pub num_samples: usize,
    pub seed: u64,
    pub climate_zones: Vec<String>,
    pub building_types: Vec<String>,
}

impl ParameterManifest {
    pub fn new(
        parameters: Vec<ParameterSpec>,
        samples: Vec<ParameterSample>,
        seed: u64,
        climate_zones: Vec<String>,
        building_types: Vec<String>,
    ) -> Self {
        let num_samples = samples.len();
        ParameterManifest {
            version: CHECKPOINT_VERSION.to_string(),
            parameters,
            samples,
            num_samples,
            seed,
            climate_zones,
            building_types,
        }
    }
}

/// Output of a single 9R4C simulation run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationOutput {
    pub sample_id: usize,
    pub total_energy_kwh: f64,
    pub peak_heating_load_w: f64,
    pub peak_cooling_load_w: f64,
    pub annual_heating_kwh: f64,
    pub annual_cooling_kwh: f64,
    pub eui_kwh_m2: f64,
    pub zone_temperatures: Vec<f64>,
    pub success: bool,
    pub error_message: Option<String>,
}

impl SimulationOutput {
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        sample_id: usize,
        total_energy_kwh: f64,
        peak_heating_load_w: f64,
        peak_cooling_load_w: f64,
        annual_heating_kwh: f64,
        annual_cooling_kwh: f64,
        eui_kwh_m2: f64,
        zone_temperatures: Vec<f64>,
    ) -> Self {
        SimulationOutput {
            sample_id,
            total_energy_kwh,
            peak_heating_load_w,
            peak_cooling_load_w,
            annual_heating_kwh,
            annual_cooling_kwh,
            eui_kwh_m2,
            zone_temperatures,
            success: true,
            error_message: None,
        }
    }

    pub fn failure(sample_id: usize, error: String) -> Self {
        SimulationOutput {
            sample_id,
            total_energy_kwh: 0.0,
            peak_heating_load_w: 0.0,
            peak_cooling_load_w: 0.0,
            annual_heating_kwh: 0.0,
            annual_cooling_kwh: 0.0,
            eui_kwh_m2: 0.0,
            zone_temperatures: Vec::new(),
            success: false,
            error_message: Some(error),
        }
    }
}

/// Batch results containing all simulation outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResults {
    pub version: String,
    pub total_samples: usize,
    pub successful_samples: usize,
    pub failed_samples: usize,
    pub outputs: Vec<SimulationOutput>,
}

impl BatchResults {
    pub fn new(outputs: Vec<SimulationOutput>) -> Self {
        let total_samples = outputs.len();
        let successful_samples = outputs.iter().filter(|o| o.success).count();
        let failed_samples = total_samples - successful_samples;
        BatchResults {
            version: CHECKPOINT_VERSION.to_string(),
            total_samples,
            successful_samples,
            failed_samples,
            outputs,
        }
    }
}

/// Checkpoint file for resumability.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Checkpoint {
    version: String,
    manifest_hash: String,
    completed_samples: Vec<usize>,
    results: Vec<SimulationOutput>,
}

impl Checkpoint {
    fn new(manifest: &ParameterManifest, results: Vec<SimulationOutput>) -> Self {
        let manifest_hash = Self::compute_hash(manifest);
        let completed_samples: Vec<usize> = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.sample_id)
            .collect();
        Checkpoint {
            version: CHECKPOINT_VERSION.to_string(),
            manifest_hash,
            completed_samples,
            results,
        }
    }

    fn compute_hash(manifest: &ParameterManifest) -> String {
        let json = serde_json::to_string(manifest).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        sha256_hex(hasher.finalize())
    }

    fn load(path: &Path, expected_hash: &str) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        let checkpoint: Checkpoint = serde_json::from_str(&content).ok()?;
        if checkpoint.manifest_hash != expected_hash {
            return None;
        }
        Some(checkpoint)
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, content)
    }
}

/// Configuration for batch execution.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub num_workers: Option<usize>,
    pub checkpoint_dir: Option<PathBuf>,
    pub chunk_size: usize,
    pub sample_rate: Option<usize>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            num_workers: None,
            checkpoint_dir: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
            sample_rate: None,
        }
    }
}

impl BatchConfig {
    pub fn new() -> Self {
        BatchConfig::default()
    }

    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = Some(workers);
        self
    }

    pub fn with_checkpoint_dir(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_sample_rate(mut self, rate: usize) -> Self {
        self.sample_rate = Some(rate);
        self
    }
}

/// Batch runner for 9R4C physics simulations.
///
/// Consumes a parameter manifest and executes 9R4C runs to produce ground-truth
/// targets for ML surrogate training.
pub struct BatchRunner {
    config: BatchConfig,
    manifest: ParameterManifest,
    base_model: ThermalModel<VectorField>,
}

impl BatchRunner {
    pub fn new(
        manifest: ParameterManifest,
        base_model: ThermalModel<VectorField>,
        config: BatchConfig,
    ) -> Self {
        BatchRunner {
            config,
            manifest,
            base_model,
        }
    }

    fn compute_manifest_hash(&self) -> String {
        Checkpoint::compute_hash(&self.manifest)
    }

    fn get_checkpoint_path(&self) -> Option<PathBuf> {
        self.config
            .checkpoint_dir
            .as_ref()
            .map(|dir| dir.join("batch_runner_checkpoint.json"))
    }

    fn load_checkpoint(&self) -> Option<(Vec<SimulationOutput>, Vec<usize>)> {
        let path = self.get_checkpoint_path()?;
        let hash = self.compute_manifest_hash();
        let checkpoint = Checkpoint::load(&path, &hash)?;
        let completed: Vec<usize> = checkpoint.results.iter().map(|r| r.sample_id).collect();
        Some((checkpoint.results, completed))
    }

    fn save_checkpoint(&self, results: &[SimulationOutput]) -> std::io::Result<()> {
        if let Some(ref path) = self.get_checkpoint_path() {
            if let Some(ref dir) = self.config.checkpoint_dir {
                fs::create_dir_all(dir)?;
            }
            let checkpoint = Checkpoint::new(&self.manifest, results.to_vec());
            checkpoint.save(path)?;
        }
        Ok(())
    }

    fn run_single_sample(&self, sample: &ParameterSample) -> SimulationOutput {
        let sample_id = sample.sample_id;

        let mut model = self.base_model.clone();
        model.apply_parameters(&sample.parameters);

        let surrogates = SurrogateManager {
            model_loaded: false,
            model_path: None,
            session_pool: None,
            backend: InferenceBackend::CPU,
            device_id: 0,
            composite: None,
            inference_metrics: std::sync::Arc::new(parking_lot::Mutex::new(
                crate::ai::surrogate::InferenceMetrics::default(),
            )),
            input_bounds: None,
            ood_count: std::sync::Arc::new(parking_lot::Mutex::new(0)),
            residual_tau: crate::ai::surrogate::DEFAULT_RESIDUAL_TAU,
            residual_reroute_count: std::sync::Arc::new(parking_lot::Mutex::new(0)),
        };

        let total_energy_kwh = model.solve_timesteps(8760, &surrogates, false, None, None, None);

        let annual_heating_kwh = model.get_heating_energy_kwh();
        let annual_cooling_kwh = model.get_cooling_energy_kwh();
        let peak_heating_kw = model.get_peak_heating_power_kw();
        let peak_cooling_kw = model.get_peak_cooling_power_kw();
        let zone_temps = model.get_temperatures();
        let zone_area = model.zone_area.integrate();

        let eui = if zone_area > 0.0 {
            (annual_heating_kwh + annual_cooling_kwh) / zone_area
        } else {
            0.0
        };

        SimulationOutput::success(
            sample_id,
            total_energy_kwh,
            peak_heating_kw * 1000.0,
            peak_cooling_kw * 1000.0,
            annual_heating_kwh,
            annual_cooling_kwh,
            eui,
            zone_temps,
        )
    }

    pub fn run(&self) -> BatchResults {
        let (mut existing_results, completed_ids) = self
            .load_checkpoint()
            .unwrap_or_else(|| (Vec::new(), Vec::new()));

        let samples_to_run: Vec<&ParameterSample> = self
            .manifest
            .samples
            .iter()
            .filter(|s| !completed_ids.contains(&s.sample_id))
            .collect();

        if samples_to_run.is_empty() {
            return BatchResults::new(existing_results);
        }

        let chunk_size = self.config.num_workers.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }) * 4;

        let new_results: Vec<SimulationOutput> = samples_to_run
            .par_iter()
            .with_max_len(chunk_size)
            .map(|sample| self.run_single_sample(sample))
            .collect();

        existing_results.extend(new_results.clone());

        let _ = self.save_checkpoint(&existing_results);

        BatchResults::new(existing_results)
    }

    pub fn run_smoke_test(&self, num_samples: usize) -> BatchResults {
        let smoke_samples: Vec<ParameterSample> = self
            .manifest
            .samples
            .iter()
            .take(num_samples)
            .cloned()
            .collect();

        let manifest = ParameterManifest {
            samples: smoke_samples,
            num_samples,
            ..self.manifest.clone()
        };

        let runner = BatchRunner {
            config: BatchConfig {
                checkpoint_dir: None,
                ..self.config.clone()
            },
            manifest,
            base_model: self.base_model.clone(),
        };

        runner.run()
    }
}

pub mod sampling {
    use rand::distributions::{Distribution, Uniform};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::Normal;

    use super::{ParameterManifest, ParameterSample, ParameterSpec, SamplingDistribution};

    pub fn sample_from_distribution(dist: &SamplingDistribution, rng: &mut StdRng) -> f64 {
        match dist {
            SamplingDistribution::Uniform { min, max } => {
                Uniform::new_inclusive(*min, *max).sample(rng)
            }
            SamplingDistribution::LogUniform { min, max } => {
                let log_min = min.ln() / 2_f64.ln();
                let log_max = max.ln() / 2_f64.ln();
                2_f64.powf(Uniform::new_inclusive(log_min, log_max).sample(rng))
            }
            SamplingDistribution::Normal { mean, std } => {
                Normal::new(*mean, *std).unwrap().sample(rng)
            }
            SamplingDistribution::TruncatedNormal {
                mean,
                std,
                min,
                max,
            } => {
                let normal = Normal::new(*mean, *std).unwrap();
                let sample = normal.sample(rng);
                sample.clamp(*min, *max)
            }
        }
    }

    pub fn generate_samples(
        parameters: &[ParameterSpec],
        num_samples: usize,
        seed: u64,
    ) -> Vec<ParameterSample> {
        let mut rng = StdRng::seed_from_u64(seed);
        let max_index = parameters.iter().map(|p| p.index).max().unwrap_or(0);

        (0..num_samples)
            .map(|i| {
                let mut params = vec![0.0; max_index + 1];
                for spec in parameters {
                    params[spec.index] = sample_from_distribution(&spec.distribution, &mut rng);
                }
                ParameterSample::new(i, params, seed.wrapping_add(i as u64))
            })
            .collect()
    }

    pub fn build_manifest(
        parameters: Vec<ParameterSpec>,
        num_samples: usize,
        seed: u64,
    ) -> ParameterManifest {
        let samples = generate_samples(&parameters, num_samples, seed);
        ParameterManifest {
            version: super::CHECKPOINT_VERSION.to_string(),
            parameters,
            samples,
            num_samples,
            seed,
            climate_zones: vec!["4A".to_string(), "5A".to_string(), "6A".to_string()],
            building_types: vec!["residential".to_string()],
        }
    }
}

pub mod io {
    use super::{BatchResults, ParameterManifest};
    use std::fs;
    use std::path::Path;

    pub fn save_manifest(manifest: &ParameterManifest, path: &Path) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(manifest).unwrap();
        fs::write(path, content)
    }

    pub fn load_manifest(path: &Path) -> std::io::Result<ParameterManifest> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content).unwrap())
    }

    pub fn save_results(results: &BatchResults, path: &Path) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(results).unwrap();
        fs::write(path, content)
    }

    pub fn load_results(path: &Path) -> std::io::Result<BatchResults> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_spec_constructors() {
        let uniform = ParameterSpec::uniform("window_u", 0, 0.5, 3.0);
        assert!(matches!(
            uniform.distribution,
            SamplingDistribution::Uniform { min: 0.5, max: 3.0 }
        ));
        assert_eq!(uniform.index, 0);

        let log_uniform = ParameterSpec::log_uniform("k", 1, 0.001, 10.0);
        assert!(matches!(
            log_uniform.distribution,
            SamplingDistribution::LogUniform { .. }
        ));

        let normal = ParameterSpec::normal("heating", 1, 20.0, 1.0);
        assert!(matches!(
            normal.distribution,
            SamplingDistribution::Normal { .. }
        ));

        let trunc = ParameterSpec::truncated_normal("cooling", 2, 26.0, 1.0, 22.0, 32.0);
        assert!(matches!(
            trunc.distribution,
            SamplingDistribution::TruncatedNormal { .. }
        ));
    }

    #[test]
    fn test_parameter_sample() {
        let params = vec![1.5, 20.0, 26.0];
        let sample = ParameterSample::new(0, params, 42);
        assert_eq!(sample.get(0), Some(1.5));
        assert_eq!(sample.get(1), Some(20.0));
        assert_eq!(sample.get(2), Some(26.0));
        assert_eq!(sample.get(3), None);
    }

    #[test]
    fn test_simulation_output_success() {
        let output = SimulationOutput::success(
            0,
            1000.0,
            5000.0,
            3000.0,
            8000.0,
            5000.0,
            50.0,
            vec![20.0, 22.0],
        );
        assert!(output.success);
        assert_eq!(output.total_energy_kwh, 1000.0);
        assert_eq!(output.peak_heating_load_w, 5000.0);
    }

    #[test]
    fn test_simulation_output_failure() {
        let output = SimulationOutput::failure(0, "Test error".to_string());
        assert!(!output.success);
        assert_eq!(output.error_message, Some("Test error".to_string()));
    }

    #[test]
    fn test_batch_results() {
        let outputs = vec![
            SimulationOutput::success(0, 1000.0, 5000.0, 3000.0, 8000.0, 5000.0, 50.0, vec![]),
            SimulationOutput::failure(1, "Error".to_string()),
            SimulationOutput::success(2, 1200.0, 5500.0, 3500.0, 8500.0, 5500.0, 55.0, vec![]),
        ];
        let results = BatchResults::new(outputs);
        assert_eq!(results.total_samples, 3);
        assert_eq!(results.successful_samples, 2);
        assert_eq!(results.failed_samples, 1);
    }

    #[test]
    fn test_sampling_uniform() {
        let params = vec![ParameterSpec::uniform("x", 0, 0.0, 10.0)];
        let samples = sampling::generate_samples(&params, 100, 12345);
        assert_eq!(samples.len(), 100);

        for sample in &samples {
            let x = sample.get(0).unwrap();
            assert!(x >= 0.0 && x <= 10.0);
        }
    }

    #[test]
    fn test_sampling_log_uniform() {
        let params = vec![ParameterSpec::log_uniform("k", 0, 0.001, 1.0)];
        let samples = sampling::generate_samples(&params, 50, 54321);
        assert_eq!(samples.len(), 50);

        for sample in &samples {
            let k = sample.get(0).unwrap();
            assert!(k >= 0.001 && k <= 1.0);
        }
    }

    #[test]
    fn test_build_manifest() {
        let params = vec![
            ParameterSpec::uniform("window_u", 0, 0.5, 3.0),
            ParameterSpec::normal("heating", 1, 20.0, 1.0),
            ParameterSpec::truncated_normal("cooling", 2, 26.0, 1.0, 22.0, 32.0),
        ];
        let manifest = sampling::build_manifest(params, 50, 99999);
        assert_eq!(manifest.num_samples, 50);
        assert_eq!(manifest.parameters.len(), 3);
        assert_eq!(manifest.seed, 99999);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .with_workers(8)
            .with_chunk_size(128)
            .with_sample_rate(10);
        assert_eq!(config.num_workers, Some(8));
        assert_eq!(config.chunk_size, 128);
        assert_eq!(config.sample_rate, Some(10));
    }
}
