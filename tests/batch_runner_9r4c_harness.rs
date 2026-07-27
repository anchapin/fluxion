//! Integration tests for the 9R4C batch job runner harness (Issue #1777).
//!
//! These tests exercise the three acceptance criteria end-to-end:
//! 1. Consumes a parameter manifest and executes 9R4C runs.
//! 2. Resumable + idempotent (checkpoint per chunk).
//! 3. Local small-scale smoke run validates the pipeline.

use std::collections::HashSet;

use fluxion::ai::batch_runner::sampling;
use fluxion::ai::batch_runner::{BatchResults, ParameterManifest, ParameterSpec};
use fluxion::ai::batch_runner_9r4c::{
    harness_io, BatchRunner9R4C, HarnessCheckpoint, HarnessConfig, HARNESS_VERSION,
};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Build a small parameter manifest suitable for fast tests.
fn small_manifest(n: usize) -> ParameterManifest {
    let params = vec![
        ParameterSpec::uniform("window_u_value", 0, 1.0, 3.0),
        ParameterSpec::uniform("heating_setpoint", 1, 18.0, 22.0),
        ParameterSpec::uniform("cooling_setpoint", 2, 24.0, 28.0),
    ];
    sampling::build_manifest(params, n, 12345)
}

/// A harness config with very few timesteps so tests complete quickly.
fn fast_config() -> HarnessConfig {
    HarnessConfig::default().with_timesteps(24).with_workers(2)
}

// ---------------------------------------------------------------------------
// Acceptance criterion 1: consumes manifest and executes 9R4C runs
// ---------------------------------------------------------------------------

#[test]
fn test_9r4c_solver_engaged() {
    let manifest = small_manifest(1);
    let runner =
        BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, fast_config());

    assert!(runner.is_9r4c(), "harness must report 9R4C engagement");
}

#[test]
fn test_full_batch_produces_valid_outputs() {
    let manifest = small_manifest(3);
    let runner =
        BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, fast_config());

    let results: BatchResults = runner.run();

    assert_eq!(results.total_samples, 3, "all samples must be processed");
    assert_eq!(
        results.successful_samples, 3,
        "all samples must succeed in 9R4C mode"
    );
    assert_eq!(results.failed_samples, 0);

    for output in &results.outputs {
        assert!(output.success, "sample {} must succeed", output.sample_id);
        assert!(
            output.total_energy_kwh.is_finite(),
            "sample {} energy must be finite",
            output.sample_id
        );
        assert!(
            output.total_energy_kwh.abs() < 1.0e9,
            "sample {} energy must be in a sane range, got {}",
            output.sample_id,
            output.total_energy_kwh
        );
        assert!(
            !output.zone_temperatures.is_empty(),
            "sample {} must produce zone temperatures",
            output.sample_id
        );
        assert!(
            output.error_message.is_none(),
            "sample {} must have no error",
            output.sample_id
        );
    }
}

#[test]
fn test_all_sample_ids_present() {
    let n = 5;
    let manifest = small_manifest(n);
    let runner =
        BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, fast_config());

    let results = runner.run();

    let ids: HashSet<usize> = results.outputs.iter().map(|o| o.sample_id).collect();
    for i in 0..n {
        assert!(ids.contains(&i), "sample id {i} must be in results");
    }
}

// ---------------------------------------------------------------------------
// Acceptance criterion 2: resumable + idempotent (checkpoint per chunk)
// ---------------------------------------------------------------------------

#[test]
fn test_checkpoint_persistence_and_resume() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint_dir = tmp.path().to_path_buf();

    let manifest = small_manifest(4);
    let config = fast_config().with_checkpoint_dir(checkpoint_dir.clone());

    // First run — processes all 4 samples and writes a checkpoint.
    let runner1 = BatchRunner9R4C::from_ashrae_case(
        &ASHRAE140Case::Case900,
        manifest.clone(),
        config.clone(),
    );
    let results1 = runner1.run();
    assert_eq!(results1.successful_samples, 4);

    // Checkpoint file must exist on disk.
    let cp_path = checkpoint_dir.join("batch_runner_9r4c_checkpoint.json");
    assert!(cp_path.exists(), "checkpoint file must be written");

    // Load checkpoint directly and verify structure.
    let checkpoint = harness_io::load_checkpoint(&cp_path).unwrap();
    assert_eq!(checkpoint.version, HARNESS_VERSION);
    assert_eq!(checkpoint.results.len(), 4);
    assert_eq!(checkpoint.completed_sample_ids.len(), 4);

    // Second run — checkpoint is present, so no samples should be recomputed.
    // Results must be identical.
    let runner2 = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, config);
    let results2 = runner2.run();
    assert_eq!(results2.successful_samples, 4);
    assert_eq!(results2.total_samples, 4);

    // Verify energy values match between runs.
    for (a, b) in results1.outputs.iter().zip(results2.outputs.iter()) {
        let energy_delta = (a.total_energy_kwh - b.total_energy_kwh).abs();
        assert!(
            energy_delta < 1e-6,
            "sample {} energy changed on resume: {} vs {}",
            a.sample_id,
            a.total_energy_kwh,
            b.total_energy_kwh
        );
    }
}

#[test]
fn test_checkpoint_stale_manifest_discarded() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint_dir = tmp.path().to_path_buf();

    // Run with manifest A (4 samples).
    let manifest_a = small_manifest(4);
    let config = fast_config().with_checkpoint_dir(checkpoint_dir.clone());
    let runner_a = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest_a, config);
    runner_a.run();

    let cp_path = checkpoint_dir.join("batch_runner_9r4c_checkpoint.json");
    assert!(cp_path.exists());

    // Run with manifest B (different sample count → different hash).
    // The stale checkpoint must be discarded and all samples re-run.
    let manifest_b = small_manifest(6);
    let config_b = fast_config().with_checkpoint_dir(checkpoint_dir.clone());
    let runner_b = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest_b, config_b);
    let results_b = runner_b.run();
    assert_eq!(
        results_b.total_samples, 6,
        "stale checkpoint must be discarded — all 6 samples must run"
    );
}

#[test]
fn test_idempotency_same_results_on_rerun() {
    let manifest = small_manifest(3);
    let config = fast_config();

    let runner1 = BatchRunner9R4C::from_ashrae_case(
        &ASHRAE140Case::Case900,
        manifest.clone(),
        config.clone(),
    );
    let results1 = runner1.run();

    let runner2 = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, config);
    let results2 = runner2.run();

    assert_eq!(results1.outputs.len(), results2.outputs.len());

    for (a, b) in results1.outputs.iter().zip(results2.outputs.iter()) {
        assert_eq!(a.sample_id, b.sample_id);
        assert!(
            (a.total_energy_kwh - b.total_energy_kwh).abs() < 1e-9,
            "energy must be deterministic for sample {}",
            a.sample_id
        );
        assert!(
            (a.annual_heating_kwh - b.annual_heating_kwh).abs() < 1e-9,
            "heating energy must be deterministic for sample {}",
            a.sample_id
        );
        assert!(
            (a.annual_cooling_kwh - b.annual_cooling_kwh).abs() < 1e-9,
            "cooling energy must be deterministic for sample {}",
            a.sample_id
        );
    }
}

#[test]
fn test_in_memory_run_no_checkpoint_file() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest = small_manifest(2);
    // No checkpoint_dir → in-memory only.
    let config = fast_config();
    let runner = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, config);

    let results = runner.run();
    assert_eq!(results.successful_samples, 2);

    // No checkpoint file should have been created.
    let cp_path = tmp.path().join("batch_runner_9r4c_checkpoint.json");
    assert!(!cp_path.exists());
}

// ---------------------------------------------------------------------------
// Acceptance criterion 3: local small-scale smoke run
// ---------------------------------------------------------------------------

#[test]
fn test_smoke_run_end_to_end() {
    let manifest = small_manifest(8);
    let config = HarnessConfig::smoke().with_workers(2);
    let runner = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, config);

    // Run only the first 3 samples as a smoke test.
    let results = runner.run_smoke(3);

    assert_eq!(results.total_samples, 3);
    assert_eq!(results.successful_samples, 3);
    assert_eq!(results.failed_samples, 0);

    // Every output must have valid, finite metrics.
    for output in &results.outputs {
        assert!(output.success);
        assert!(output.total_energy_kwh.is_finite());
        assert!(output.peak_heating_load_w.is_finite());
        assert!(output.peak_cooling_load_w.is_finite());
        assert!(output.eui_kwh_m2.is_finite());
    }
}

#[test]
fn test_smoke_run_fewer_than_manifest() {
    let manifest = small_manifest(10);
    let config = HarnessConfig::smoke();
    let runner = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, config);

    let results = runner.run_smoke(2);
    assert_eq!(results.total_samples, 2);
    assert!(results.successful_samples >= 1);
}

// ---------------------------------------------------------------------------
// Manifest and results IO round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_manifest_results_disk_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();

    let manifest = small_manifest(4);
    let config = fast_config();
    let runner =
        BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest.clone(), config);
    let results = runner.run();

    // Save manifest + results to disk.
    let mpath = tmp.path().join("manifest.json");
    let rpath = tmp.path().join("results.json");
    harness_io::save_manifest(&manifest, &mpath).unwrap();
    harness_io::save_results(&results, &rpath).unwrap();

    // Reload and verify.
    let loaded_manifest = harness_io::load_manifest(&mpath).unwrap();
    assert_eq!(loaded_manifest.num_samples, 4);
    assert_eq!(loaded_manifest.parameters.len(), 3);

    let loaded_results: BatchResults = harness_io::load_results(&rpath).unwrap();
    assert_eq!(loaded_results.total_samples, 4);
    assert_eq!(
        loaded_results.successful_samples,
        results.successful_samples
    );

    // Spot-check energy values survive the round-trip.
    for (orig, loaded) in results.outputs.iter().zip(loaded_results.outputs.iter()) {
        assert!(
            (orig.total_energy_kwh - loaded.total_energy_kwh).abs() < 1e-9,
            "energy must survive JSON round-trip for sample {}",
            orig.sample_id
        );
    }
}

#[test]
fn test_checkpoint_atomic_write() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint_dir = tmp.path().to_path_buf();

    let manifest = small_manifest(3);
    let config = fast_config().with_checkpoint_dir(checkpoint_dir.clone());
    let runner = BatchRunner9R4C::from_ashrae_case(&ASHRAE140Case::Case900, manifest, config);
    runner.run();

    let cp_path = checkpoint_dir.join("batch_runner_9r4c_checkpoint.json");
    assert!(cp_path.exists(), "final checkpoint must exist");

    // No leftover .tmp file from the atomic write.
    let tmp_path = cp_path.with_extension("tmp");
    assert!(!tmp_path.exists(), "no stale .tmp file should remain");

    // Checkpoint must be valid JSON and loadable.
    let checkpoint: HarnessCheckpoint = harness_io::load_checkpoint(&cp_path).unwrap();
    assert!(!checkpoint.results.is_empty());
    assert!(!checkpoint.manifest_hash.is_empty());
}
