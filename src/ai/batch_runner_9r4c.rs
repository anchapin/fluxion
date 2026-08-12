//! 9R4C batch job runner harness (Issue #1777, plan key T5.2).
//!
//! Consumes a [`ParameterManifest`] (produced by the Monte Carlo sweep in
//! issue #1776 / `src/ai/sweeps/`) and runs each parameter sample through the
//! **exact 9R4C multi-node physics solver** to produce ground-truth targets for
//! ML surrogate training.
//!
//! ## Acceptance criteria (Issue #1777)
//!
//! 1. **Consumes the parameter manifest and executes 9R4C runs** — see
//!    [`BatchRunner9R4C::run`].
//! 2. **Resumable + idempotent (checkpoint per chunk)** — see
//!    [`HarnessCheckpoint`] and the incremental-save loop inside `run`.
//!    The checkpoint is keyed by a SHA-256 hash of the manifest, so a stale
//!    checkpoint from a different sweep is detected and discarded.
//! 3. **Local small-scale smoke run** — see [`BatchRunner9R4C::run_smoke`].
//!
//! ## Why a separate harness?
//!
//! The generic [`crate::ai::batch_runner::BatchRunner`] accepts any
//! `ThermalModel`, which means a caller could accidentally feed in a 5R1C model.
//! This harness **factory-constructs the 9R4C model** from an ASHRAE 140
//! high-mass [`CaseSpec`] and refuses to run if the solver is not engaged,
//! guaranteeing that every training target originates from the 9R4C solver.

use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ai::batch_runner::{BatchResults, ParameterManifest, ParameterSample, SimulationOutput};
use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::engine::ThermalModel;
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec, ConstructionType};

/// Default number of timesteps for a full annual run (8 760 hours).
pub const DEFAULT_TIMESTEPS: usize = 8_760;

/// Harness / checkpoint format version.
pub const HARNESS_VERSION: &str = "9r4c-1.0";

/// Configuration for the 9R4C batch harness.
#[derive(Clone, Debug)]
pub struct HarnessConfig {
    /// Number of physics timesteps per sample (8 760 = full year).
    pub timesteps_per_sample: usize,
    /// Directory for checkpoint persistence.  When `None` the run is
    /// in-memory only (no resumability).
    pub checkpoint_dir: Option<PathBuf>,
    /// Save a checkpoint after every `checkpoint_every` completed samples.
    pub checkpoint_every: usize,
    /// Parallel worker count.  `None` → rayon default thread pool.
    pub num_workers: Option<usize>,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        HarnessConfig {
            timesteps_per_sample: DEFAULT_TIMESTEPS,
            checkpoint_dir: None,
            checkpoint_every: 16,
            num_workers: None,
        }
    }
}

impl HarnessConfig {
    /// Create a config optimised for fast local smoke runs.
    pub fn smoke() -> Self {
        HarnessConfig {
            timesteps_per_sample: 168, // one week
            checkpoint_dir: None,
            checkpoint_every: 4,
            num_workers: None,
        }
    }

    pub fn with_checkpoint_dir(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    pub fn with_timesteps(mut self, steps: usize) -> Self {
        self.timesteps_per_sample = steps;
        self
    }

    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = Some(workers);
        self
    }
}

/// On-disk checkpoint for resumable batch execution.
///
/// Keyed by a SHA-256 hash of the serialised [`ParameterManifest`] so that a
/// stale checkpoint from a different sweep is detected and ignored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarnessCheckpoint {
    pub version: String,
    pub manifest_hash: String,
    pub completed_sample_ids: Vec<usize>,
    pub results: Vec<SimulationOutput>,
}

impl HarnessCheckpoint {
    fn compute_manifest_hash(manifest: &ParameterManifest) -> String {
        let json = serde_json::to_string(manifest).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn from_results(manifest: &ParameterManifest, results: &[SimulationOutput]) -> Self {
        HarnessCheckpoint {
            version: HARNESS_VERSION.to_string(),
            manifest_hash: Self::compute_manifest_hash(manifest),
            completed_sample_ids: results.iter().map(|r| r.sample_id).collect(),
            results: results.to_vec(),
        }
    }

    /// Load a checkpoint from disk, returning `None` if the file is missing
    /// or the manifest hash does not match (stale checkpoint).
    pub fn load(path: &Path, expected_hash: &str) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        let checkpoint: HarnessCheckpoint = serde_json::from_str(&content).ok()?;
        if checkpoint.manifest_hash != expected_hash {
            return None;
        }
        Some(checkpoint)
    }

    /// Atomically write the checkpoint to disk (write-tmp then rename).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self).unwrap();
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, &content)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Batch job runner harness backed by the 9R4C multi-node physics solver.
///
/// Construct via [`BatchRunner9R4C::from_case_spec`] or
/// [`BatchRunner9R4C::from_ashrae_case`].  The constructor **verifies** that
/// the 9R4C solver is engaged and panics otherwise — this is intentional so
/// that training targets are guaranteed to originate from the correct solver.
pub struct BatchRunner9R4C {
    model: ThermalModel<VectorField>,
    manifest: ParameterManifest,
    config: HarnessConfig,
}

impl BatchRunner9R4C {
    /// Build a harness from an explicit [`CaseSpec`].
    ///
    /// The spec **must** use [`ConstructionType::HighMass`] so that
    /// `from_spec` initialises the per-surface conductances and
    /// [`crate::physics::multi_node_solver::MultiNodeSolver`] instances.
    pub fn from_case_spec(
        spec: &CaseSpec,
        manifest: ParameterManifest,
        config: HarnessConfig,
    ) -> Self {
        assert_eq!(
            spec.construction_type,
            ConstructionType::HighMass,
            "BatchRunner9R4C requires a HighMass CaseSpec to engage the 9R4C solver"
        );

        let model = ThermalModel::<VectorField>::from_spec(spec);

        let runner = BatchRunner9R4C {
            model,
            manifest,
            config,
        };
        runner.verify_9r4c();
        runner
    }

    /// Convenience factory using a built-in ASHRAE 140 case enum.
    ///
    /// `ASHRAE140Case::Case900` and `Case900FF` are the canonical high-mass
    /// cases that produce a 9R4C model.
    pub fn from_ashrae_case(
        case: &ASHRAE140Case,
        manifest: ParameterManifest,
        config: HarnessConfig,
    ) -> Self {
        let spec = case.spec();
        Self::from_case_spec(&spec, manifest, config)
    }

    /// Verify that the 9R4C solver is actually engaged on the base model.
    pub fn verify_9r4c(&self) {
        assert!(
            self.model.is_nine_r4c_model(),
            "ThermalModel is not in 9R4C mode — refusing to generate training targets"
        );
        assert!(
            !self.model.conduction.multi_node_solvers.is_empty(),
            "No MultiNodeSolver instances found — 9R4C solver was not initialised"
        );
    }

    /// Returns `true` if the base model is configured for 9R4C.
    pub fn is_9r4c(&self) -> bool {
        self.model.is_nine_r4c_model() && !self.model.conduction.multi_node_solvers.is_empty()
    }

    /// Number of samples in the manifest.
    pub fn num_samples(&self) -> usize {
        self.manifest.samples.len()
    }

    fn checkpoint_path(&self) -> Option<PathBuf> {
        self.config
            .checkpoint_dir
            .as_ref()
            .map(|dir| dir.join("batch_runner_9r4c_checkpoint.json"))
    }

    fn manifest_hash(&self) -> String {
        HarnessCheckpoint::compute_manifest_hash(&self.manifest)
    }

    fn load_checkpoint(&self) -> Option<HarnessCheckpoint> {
        let path = self.checkpoint_path()?;
        let hash = self.manifest_hash();
        HarnessCheckpoint::load(&path, &hash)
    }

    fn save_checkpoint(&self, results: &[SimulationOutput]) {
        if let Some(ref path) = self.checkpoint_path() {
            let checkpoint = HarnessCheckpoint::from_results(&self.manifest, results);
            if let Err(e) = checkpoint.save(path) {
                log::warn!("Failed to save 9R4C checkpoint: {e}");
            }
        }
    }

    /// Execute a single parameter sample through the 9R4C solver.
    fn run_single_sample(&self, sample: &ParameterSample) -> SimulationOutput {
        let sample_id = sample.sample_id;
        let steps = self.config.timesteps_per_sample;

        let mut model = self.model.clone();
        model.apply_parameters(&sample.parameters);

        let surrogates = match SurrogateManager::new() {
            Ok(m) => m,
            Err(e) => {
                return SimulationOutput::failure(
                    sample_id,
                    format!("Failed to create SurrogateManager: {e}"),
                );
            }
        };

        let total_energy_kwh = model.solve_timesteps(steps, &surrogates, false, None, None, None);

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

    /// Run the full batch with resumable, incremental checkpointing.
    ///
    /// 1. Loads any existing checkpoint and filters out completed samples.
    /// 2. Processes remaining samples in parallel (rayon).
    /// 3. Saves a checkpoint every `config.checkpoint_every` samples so a
    ///    crash loses at most one chunk of work.
    /// 4. Returns the complete [`BatchResults`] (existing + new).
    pub fn run(&self) -> BatchResults {
        let mut results: Vec<SimulationOutput> = self
            .load_checkpoint()
            .map(|cp| cp.results)
            .unwrap_or_default();

        let completed: std::collections::HashSet<usize> =
            results.iter().map(|r| r.sample_id).collect();

        let pending: Vec<&ParameterSample> = self
            .manifest
            .samples
            .iter()
            .filter(|s| !completed.contains(&s.sample_id))
            .collect();

        if pending.is_empty() {
            return BatchResults::new(results);
        }

        let checkpoint_every = self.config.checkpoint_every.max(1);
        let pool = self.config.num_workers.map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("Failed to create rayon thread pool")
        });

        let process_chunk = |chunk: &[&ParameterSample]| -> Vec<SimulationOutput> {
            chunk
                .par_iter()
                .map(|s| self.run_single_sample(s))
                .collect()
        };

        // If a custom thread pool was requested, run inside its scope.
        if let Some(ref pool) = pool {
            pool.install(|| {
                for chunk in pending.chunks(checkpoint_every) {
                    results.extend(process_chunk(chunk));
                    self.save_checkpoint(&results);
                }
            });
        } else {
            for chunk in pending.chunks(checkpoint_every) {
                results.extend(process_chunk(chunk));
                self.save_checkpoint(&results);
            }
        }

        BatchResults::new(results)
    }

    /// Quick local smoke run — validates the end-to-end pipeline before
    /// cloud scale-out.
    ///
    /// Takes only the first `num_samples` from the manifest and uses the
    /// configured `timesteps_per_sample` (typically reduced via
    /// [`HarnessConfig::smoke`]).  No checkpoint is written.
    pub fn run_smoke(&self, num_samples: usize) -> BatchResults {
        let smoke_samples: Vec<ParameterSample> = self
            .manifest
            .samples
            .iter()
            .take(num_samples)
            .cloned()
            .collect();

        let smoke_manifest = ParameterManifest {
            samples: smoke_samples,
            num_samples,
            ..self.manifest.clone()
        };

        let runner = BatchRunner9R4C {
            model: self.model.clone(),
            manifest: smoke_manifest,
            config: HarnessConfig {
                checkpoint_dir: None,
                ..self.config.clone()
            },
        };

        runner.run()
    }
}

/// Serialisation helpers for manifests and results.
pub mod harness_io {
    use super::{BatchResults, HarnessCheckpoint, ParameterManifest};
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

    pub fn save_checkpoint(checkpoint: &HarnessCheckpoint, path: &Path) -> std::io::Result<()> {
        checkpoint.save(path)
    }

    pub fn load_checkpoint(path: &Path) -> std::io::Result<HarnessCheckpoint> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::batch_runner::sampling;
    use crate::ai::batch_runner::{ParameterSpec, SimulationOutput};

    fn build_test_manifest(n: usize) -> ParameterManifest {
        let params = vec![
            ParameterSpec::uniform("window_u_value", 0, 1.0, 3.0),
            ParameterSpec::uniform("heating_setpoint", 1, 18.0, 22.0),
            ParameterSpec::uniform("cooling_setpoint", 2, 24.0, 28.0),
        ];
        sampling::build_manifest(params, n, 42)
    }

    #[test]
    fn test_harness_config_defaults() {
        let cfg = HarnessConfig::default();
        assert_eq!(cfg.timesteps_per_sample, DEFAULT_TIMESTEPS);
        assert_eq!(cfg.checkpoint_every, 16);
        assert!(cfg.checkpoint_dir.is_none());
    }

    #[test]
    fn test_harness_config_smoke() {
        let cfg = HarnessConfig::smoke();
        assert_eq!(cfg.timesteps_per_sample, 168);
        assert_eq!(cfg.checkpoint_every, 4);
    }

    #[test]
    fn test_harness_config_builder() {
        let cfg = HarnessConfig::default().with_timesteps(24).with_workers(2);
        assert_eq!(cfg.timesteps_per_sample, 24);
        assert_eq!(cfg.num_workers, Some(2));
    }

    #[test]
    fn test_checkpoint_hash_stability() {
        let manifest = build_test_manifest(5);
        let hash1 = HarnessCheckpoint::compute_manifest_hash(&manifest);
        let hash2 = HarnessCheckpoint::compute_manifest_hash(&manifest);
        assert_eq!(hash1, hash2, "identical manifests must hash identically");
    }

    #[test]
    fn test_checkpoint_hash_changes_with_manifest() {
        let m1 = build_test_manifest(5);
        let m2 = build_test_manifest(10);
        let h1 = HarnessCheckpoint::compute_manifest_hash(&m1);
        let h2 = HarnessCheckpoint::compute_manifest_hash(&m2);
        assert_ne!(h1, h2, "different manifests must hash differently");
    }

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let manifest = build_test_manifest(3);
        let results = vec![
            SimulationOutput::success(0, 100.0, 5000.0, 3000.0, 60.0, 40.0, 5.0, vec![20.0]),
            SimulationOutput::success(1, 110.0, 5500.0, 3500.0, 66.0, 44.0, 5.5, vec![21.0]),
        ];
        let checkpoint = HarnessCheckpoint::from_results(&manifest, &results);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        checkpoint.save(path).unwrap();

        let hash = HarnessCheckpoint::compute_manifest_hash(&manifest);
        let loaded = HarnessCheckpoint::load(path, &hash).expect("checkpoint must load");
        assert_eq!(loaded.version, HARNESS_VERSION);
        assert_eq!(loaded.results.len(), 2);
        assert_eq!(loaded.completed_sample_ids, vec![0, 1]);
    }

    #[test]
    fn test_checkpoint_rejects_stale_hash() {
        let manifest = build_test_manifest(3);
        let checkpoint = HarnessCheckpoint::from_results(&manifest, &[]);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        checkpoint.save(tmp.path()).unwrap();

        // Wrong hash → should return None.
        let loaded = HarnessCheckpoint::load(tmp.path(), "deadbeef");
        assert!(loaded.is_none(), "stale checkpoint must be rejected");
    }

    #[test]
    fn test_manifest_and_results_serialization_roundtrip() {
        let manifest = build_test_manifest(5);

        let tmp = tempfile::tempdir().unwrap();
        let mpath = tmp.path().join("manifest.json");
        harness_io::save_manifest(&manifest, &mpath).unwrap();
        let loaded = harness_io::load_manifest(&mpath).unwrap();
        assert_eq!(loaded.num_samples, 5);
        assert_eq!(loaded.parameters.len(), 3);

        let results = BatchResults::new(vec![SimulationOutput::success(
            0,
            100.0,
            5000.0,
            3000.0,
            60.0,
            40.0,
            5.0,
            vec![20.0],
        )]);
        let rpath = tmp.path().join("results.json");
        harness_io::save_results(&results, &rpath).unwrap();
        let loaded_results = harness_io::load_results(&rpath).unwrap();
        assert_eq!(loaded_results.total_samples, 1);
        assert_eq!(loaded_results.successful_samples, 1);
    }
}
