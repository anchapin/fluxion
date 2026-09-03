//! Integration tests for `src/ai/tensor_dataset.rs` (Issue #1778, plan key T5.3).
//!
//! Covers the three acceptance criteria:
//!  1. Schema is defined + versioned (round-trip + manifest persistence).
//!  2. Writer emits sharded tensor files in the agreed format.
//!  3. Schema validation rejects malformed shards.
//!
//! Plus formatting utilities that convert physics solver outputs into the
//! tensor schema.

use fluxion::ai::batch_runner::{
    BatchResults, ParameterManifest, ParameterSample, ParameterSpec, SimulationOutput,
};
use fluxion::ai::tensor_dataset::{
    batch_results_to_samples, parse_shard, read_shard, validate_dataset_dir, validate_shard,
    TensorDType, TensorDatasetError, TensorDatasetManifest, TensorDatasetWriter, TensorFeatureSpec,
    TensorSample, FTDS_FIXED_HEADER_LEN, FTDS_MAGIC, TENSOR_DATASET_SCHEMA_VERSION,
};

use std::fs;
use std::path::PathBuf;

fn feature_names() -> Vec<String> {
    TensorFeatureSpec::defaults()
        .iter()
        .map(|s| s.name.clone())
        .collect()
}

fn target_names() -> Vec<String> {
    TensorSample::DEFAULT_TARGET_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn synthetic_sample(id: usize) -> TensorSample {
    TensorSample {
        sample_id: id,
        inputs: vec![1.5 + id as f64 * 0.1, 20.0, 26.0],
        targets: vec![
            1000.0 + id as f64,
            5000.0,
            3000.0,
            800.0,
            200.0,
            50.0 + id as f64,
        ],
        timeseries: None,
    }
}

fn writer(dir: &std::path::Path, shard_size: usize) -> TensorDatasetWriter {
    TensorDatasetWriter::new(
        dir,
        TensorFeatureSpec::defaults(),
        target_names(),
        /* timeseries_length */ 0,
        shard_size,
    )
    .expect("writer creation")
}

// --------------------------------------------------------------------------
// AC1: Schema is defined + versioned
// --------------------------------------------------------------------------

#[test]
fn schema_version_is_pinned() {
    assert_eq!(TENSOR_DATASET_SCHEMA_VERSION, "1.0.0");
    assert_eq!(FTDS_MAGIC, *b"FTDS");
    // Fixed header layout: magic(4) + semver(6) + dtype(1) + n_samples(8)
    // + n_features(8) + n_targets(8) + has_ts(1) + ts_len(8) + sidecar_len(4).
    assert_eq!(FTDS_FIXED_HEADER_LEN, 48);
}

#[test]
fn manifest_round_trips_through_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    for i in 0..3 {
        w.push(synthetic_sample(i)).unwrap();
    }
    let manifest = w.finish().unwrap();

    let loaded = TensorDatasetManifest::load(dir.path()).expect("manifest loads");
    assert_eq!(loaded.schema_version, manifest.schema_version);
    assert_eq!(loaded.schema_version, TENSOR_DATASET_SCHEMA_VERSION);
    assert_eq!(loaded.dtype, TensorDType::F64);
    assert_eq!(loaded.n_samples_total, 3);
    assert_eq!(loaded.n_input_features, 3);
    assert_eq!(loaded.target_names, target_names());
    assert_eq!(loaded.input_feature_names, feature_names());
    assert!(!loaded.has_timeseries);
    assert_eq!(loaded.timeseries_length, 0);
    assert_eq!(loaded.shards.len(), 1);
}

// --------------------------------------------------------------------------
// AC2: Writer emits sharded tensor files in the agreed format
// --------------------------------------------------------------------------

#[test]
fn writer_emits_multiple_shards() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 2);
    for i in 0..5 {
        w.push(synthetic_sample(i)).unwrap();
    }
    let manifest = w.finish().unwrap();

    // 5 samples / 2 per shard → 3 shards.
    assert_eq!(manifest.shards.len(), 3);
    assert_eq!(manifest.n_samples_total, 5);
    let shard_counts: Vec<usize> = manifest.shards.iter().map(|s| s.n_samples).collect();
    assert_eq!(shard_counts, vec![2, 2, 1]);

    for ref_s in &manifest.shards {
        let path = dir.path().join(&ref_s.path);
        assert!(path.exists(), "shard {} should exist", ref_s.path);
        assert!(ref_s.path.starts_with("shard-"));
        assert!(ref_s.path.ends_with(".ftds"));
        assert_eq!(ref_s.sha256.len(), 64, "sha256 hex string");
    }
}

#[test]
fn shard_round_trip_preserves_payload() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 8);
    let mut original = Vec::new();
    for i in 0..4 {
        let s = synthetic_sample(i);
        original.push(s.clone());
        w.push(s).unwrap();
    }
    let manifest = w.finish().unwrap();

    let shard_path = dir.path().join(&manifest.shards[0].path);
    let shard = read_shard(&shard_path).expect("shard parses");

    assert_eq!(shard.header.n_samples, 4);
    assert_eq!(shard.header.n_input_features, 3);
    assert_eq!(shard.header.n_targets, 6);
    assert_eq!(shard.sample_ids, vec![0, 1, 2, 3]);

    // Inputs row-major [N, F].
    for (i, s) in original.iter().enumerate() {
        for (j, v) in s.inputs.iter().enumerate() {
            let idx = i * 3 + j;
            assert!((shard.inputs[idx] - *v).abs() < 1e-12, "inputs[{i},{j}]");
        }
        for (j, v) in s.targets.iter().enumerate() {
            let idx = i * 6 + j;
            assert!((shard.targets[idx] - *v).abs() < 1e-12, "targets[{i},{j}]");
        }
    }

    // Array views shape-check.
    let inv = shard.inputs_view();
    assert_eq!(inv.shape(), &[4, 3]);
    let tv = shard.targets_view();
    assert_eq!(tv.shape(), &[4, 6]);
}

#[test]
fn shard_file_starts_with_magic_and_version() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    w.push(synthetic_sample(0)).unwrap();
    let manifest = w.finish().unwrap();

    let bytes = fs::read(dir.path().join(&manifest.shards[0].path)).unwrap();
    assert_eq!(&bytes[0..4], &FTDS_MAGIC);
    // dtype byte at offset 10 must be F64.
    assert_eq!(bytes[10], TensorDType::F64.as_byte());
}

// --------------------------------------------------------------------------
// AC3: Schema validation rejects malformed shards
// --------------------------------------------------------------------------

#[test]
fn validator_accepts_well_formed_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 3);
    for i in 0..6 {
        w.push(synthetic_sample(i)).unwrap();
    }
    w.finish().unwrap();

    let report = validate_dataset_dir(dir.path());
    assert!(report.ok, "errors: {:?}", report.errors);
    assert_eq!(report.n_samples, 6);
}

#[test]
fn validator_rejects_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    w.push(synthetic_sample(0)).unwrap();
    let manifest = w.finish().unwrap();

    let path = dir.path().join(&manifest.shards[0].path);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'X'; // corrupt magic
    fs::write(&path, &bytes).unwrap();

    let report = validate_shard(&path, None);
    assert!(!report.ok);
    assert!(report.errors.iter().any(|e| e.contains("magic")));
}

#[test]
fn validator_rejects_truncated_payload() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    w.push(synthetic_sample(0)).unwrap();
    let manifest = w.finish().unwrap();

    let path = dir.path().join(&manifest.shards[0].path);
    let bytes = fs::read(&path).unwrap();
    // Drop 40 bytes off the end (truncate footer + part of ids).
    let truncated = &bytes[..bytes.len() - 40];
    fs::write(&path, truncated).unwrap();

    let report = validate_shard(&path, None);
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|e| e.contains("truncated") || e.contains("Truncated")));
}

#[test]
fn validator_rejects_corrupted_payload_via_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    w.push(synthetic_sample(0)).unwrap();
    let manifest = w.finish().unwrap();

    let path = dir.path().join(&manifest.shards[0].path);
    let mut bytes = fs::read(&path).unwrap();
    // Flip a byte inside the inputs payload (well after the 48-byte header +
    // sidecar JSON). Body offset of inputs is fixed_header + sidecar; pick a
    // safe late-body index.
    let body_idx = bytes.len().saturating_sub(80);
    bytes[body_idx] ^= 0xFF;
    fs::write(&path, &bytes).unwrap();

    let report = validate_shard(&path, None);
    assert!(!report.ok, "expected integrity failure, got: {:?}", report);
    assert!(report
        .errors
        .iter()
        .any(|e| e.contains("sha256") || e.contains("Integrity")));
}

#[test]
fn validator_rejects_manifest_with_missing_shard_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    w.push(synthetic_sample(0)).unwrap();
    let manifest = w.finish().unwrap();

    // Delete the shard but keep the manifest pointing at it.
    fs::remove_file(dir.path().join(&manifest.shards[0].path)).unwrap();

    let report = validate_dataset_dir(dir.path());
    assert!(!report.ok);
    assert!(report.errors.iter().any(|e| e.contains("missing shard")));
}

#[test]
fn validator_rejects_dtype_mismatch_against_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = writer(dir.path(), 4);
    w.push(synthetic_sample(0)).unwrap();
    let manifest = w.finish().unwrap();

    let path = dir.path().join(&manifest.shards[0].path);
    let mut bytes = fs::read(&path).unwrap();
    // Flip dtype byte to F32 — every value below is still f64, so the reader's
    // dtype guard trips before parsing; the manifest-driven mismatch is also
    // detectable. Rewrite the body length is unnecessary: the parser fails.
    bytes[10] = TensorDType::F32.as_byte();
    // The checksum is now stale; patch the footer to match so we isolate the
    // dtype failure rather than an integrity failure.
    recompute_footer(&mut bytes);
    fs::write(&path, &bytes).unwrap();

    let report = validate_shard(&path, Some(&manifest));
    assert!(!report.ok);
    assert!(
        report
            .errors
            .iter()
            .any(|e| e.contains("dtype") || e.contains("version")),
        "errors: {:?}",
        report.errors
    );
}

#[test]
fn validator_rejects_zero_sample_shard() {
    let bytes = craft_empty_shard_bytes();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.ftds");
    fs::write(&path, &bytes).unwrap();
    let report = validate_shard(&path, None);
    assert!(!report.ok);
    assert!(report.errors.iter().any(|e| e.contains("empty shard")));
}

fn recompute_footer(bytes: &mut [u8]) {
    use sha2::{Digest, Sha256};
    if bytes.len() < 32 {
        return;
    }
    let body_len = bytes.len() - 32;
    let mut h = Sha256::new();
    h.update(&bytes[..body_len]);
    let digest = h.finalize();
    bytes[body_len..body_len + 32].copy_from_slice(&digest);
}

fn craft_empty_shard_bytes() -> Vec<u8> {
    // Minimal valid-looking shard header that declares n_samples == 0.
    use sha2::{Digest, Sha256};
    let sidecar = serde_json::json!({
        "schema_version": TENSOR_DATASET_SCHEMA_VERSION,
        "dtype": TensorDType::F64,
        "n_samples": 0usize,
        "n_input_features": 3usize,
        "n_targets": 6usize,
        "has_timeseries": false,
        "timeseries_length": 0usize,
        "input_feature_names": feature_names(),
        "target_names": target_names(),
    });
    let sidecar_bytes = serde_json::to_vec(&sidecar).unwrap();
    let mut fixed = [0u8; FTDS_FIXED_HEADER_LEN];
    fixed[0..4].copy_from_slice(&FTDS_MAGIC);
    fixed[4..6].copy_from_slice(&1u16.to_le_bytes());
    fixed[6..8].copy_from_slice(&0u16.to_le_bytes());
    fixed[8..10].copy_from_slice(&0u16.to_le_bytes());
    fixed[10] = TensorDType::F64.as_byte();
    // n_samples = 0
    fixed[11..19].copy_from_slice(&0u64.to_le_bytes());
    fixed[19..27].copy_from_slice(&3u64.to_le_bytes());
    fixed[27..35].copy_from_slice(&6u64.to_le_bytes());
    fixed[35] = 0;
    fixed[36..44].copy_from_slice(&0u64.to_le_bytes());
    fixed[44..48].copy_from_slice(&(sidecar_bytes.len() as u32).to_le_bytes());

    let mut out = Vec::with_capacity(fixed.len() + sidecar_bytes.len() + 32);
    out.extend_from_slice(&fixed);
    out.extend_from_slice(&sidecar_bytes);
    let body_len = out.len();
    let mut h = Sha256::new();
    h.update(&out);
    out.extend_from_slice(&h.finalize());
    let _ = body_len; // footer covers everything before the 32-byte digest
    out
}

// --------------------------------------------------------------------------
// Formatting utilities
// --------------------------------------------------------------------------

#[test]
fn formatter_maps_successful_simulation() {
    let out = SimulationOutput::success(0, 1234.5, 5000.0, 3000.0, 800.0, 434.5, 50.0, vec![]);
    let sample =
        TensorSample::from_simulation_output(&out, &[1.5, 20.0, 26.0], &feature_names(), 0)
            .expect("formats successful sim");

    assert_eq!(sample.sample_id, 0);
    assert_eq!(sample.inputs, vec![1.5, 20.0, 26.0]);
    assert_eq!(
        sample.targets,
        vec![1234.5, 5000.0, 3000.0, 800.0, 434.5, 50.0]
    );
    assert!(sample.timeseries.is_none());
}

#[test]
fn formatter_maps_zone_temperatures_when_timeseries_enabled() {
    let temps = vec![20.0, 20.5, 21.0];
    let out = SimulationOutput::success(1, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, temps.clone());
    let sample = TensorSample::from_simulation_output(&out, &[1.0, 2.0, 3.0], &feature_names(), 3)
        .expect("formats with timeseries");
    assert_eq!(sample.timeseries.as_deref(), Some(temps.as_slice()));
}

#[test]
fn formatter_rejects_timeseries_arity_mismatch() {
    let out = SimulationOutput::success(0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, vec![20.0, 21.0]); // len 2
    let err = TensorSample::from_simulation_output(&out, &[1.0, 2.0, 3.0], &feature_names(), 3)
        .unwrap_err();
    assert!(matches!(
        err,
        TensorDatasetError::Arity {
            kind: "timeseries",
            ..
        }
    ));
}

#[test]
fn batch_results_extraction_skips_failures_and_missing_params() {
    let parameters = vec![
        ParameterSpec::uniform("window_u_value", 0, 0.5, 3.0),
        ParameterSpec::normal("heating_setpoint", 1, 20.0, 1.0),
        ParameterSpec::truncated_normal("cooling_setpoint", 2, 26.0, 1.0, 22.0, 32.0),
    ];
    // Three samples: 0 ok, 1 failure, 2 ok.
    let samples = vec![
        ParameterSample::new(0, vec![1.5, 20.0, 26.0], 1),
        ParameterSample::new(1, vec![1.6, 19.0, 25.0], 2),
        ParameterSample::new(2, vec![1.7, 21.0, 27.0], 3),
    ];
    let manifest = ParameterManifest::new(
        parameters,
        samples,
        42,
        vec!["4A".into()],
        vec!["residential".into()],
    );
    let results = BatchResults::new(vec![
        SimulationOutput::success(0, 1000.0, 5000.0, 3000.0, 800.0, 200.0, 50.0, vec![]),
        SimulationOutput::failure(1, "solver diverged".to_string()),
        SimulationOutput::success(2, 1100.0, 5100.0, 3100.0, 810.0, 290.0, 55.0, vec![]),
    ]);

    let extracted = batch_results_to_samples(&results, &manifest, &feature_names(), 0);
    assert_eq!(extracted.samples.len(), 2);
    assert_eq!(extracted.samples[0].sample_id, 0);
    assert_eq!(extracted.samples[1].sample_id, 2);
    assert_eq!(extracted.skipped.len(), 1);
    assert_eq!(extracted.skipped[0].0, 1);
}

#[test]
fn end_to_end_writer_validator_through_batch_results() {
    // Wire the formatter → writer → on-disk dataset → validator end to end.
    let parameters = vec![
        ParameterSpec::uniform("window_u_value", 0, 0.5, 3.0),
        ParameterSpec::normal("heating_setpoint", 1, 20.0, 1.0),
        ParameterSpec::truncated_normal("cooling_setpoint", 2, 26.0, 1.0, 22.0, 32.0),
    ];
    let samples: Vec<ParameterSample> = (0..7)
        .map(|i| ParameterSample::new(i, vec![1.0 + i as f64, 20.0, 26.0], i as u64 + 1))
        .collect();
    let manifest = ParameterManifest::new(
        parameters,
        samples,
        99,
        vec!["4A".into(), "5A".into()],
        vec!["residential".into()],
    );
    let results = BatchResults::new(
        (0..7)
            .map(|i| {
                SimulationOutput::success(
                    i,
                    1000.0 + i as f64,
                    5000.0,
                    3000.0,
                    800.0,
                    200.0,
                    50.0,
                    vec![],
                )
            })
            .collect(),
    );

    let dir = tempfile::tempdir().unwrap();
    let mut writer = TensorDatasetWriter::new(
        dir.path(),
        TensorFeatureSpec::defaults(),
        target_names(),
        0,
        /* shard_size */ 3,
    )
    .unwrap();

    let extracted = batch_results_to_samples(&results, &manifest, &feature_names(), 0);
    assert_eq!(extracted.samples.len(), 7);
    for s in extracted.samples {
        writer.push(s).unwrap();
    }
    let manifest_out = writer.finish().unwrap();
    assert_eq!(manifest_out.shards.len(), 3); // ceil(7/3)

    let report = validate_dataset_dir(dir.path());
    assert!(report.ok, "errors: {:?}", report.errors);
    assert_eq!(report.n_samples, 7);

    // Each shard independently parses and matches the manifest.
    let on_disk_manifest = TensorDatasetManifest::load(dir.path()).unwrap();
    let total_in_shards: usize = on_disk_manifest
        .shards
        .iter()
        .map(|s| {
            let shard = read_shard(&dir.path().join(&s.path)).unwrap();
            assert_eq!(shard.header.schema_version, TENSOR_DATASET_SCHEMA_VERSION);
            shard.header.n_samples
        })
        .sum();
    assert_eq!(total_in_shards, 7);
}

#[test]
fn parse_shard_rejects_incompatible_major_version() {
    let bytes = craft_incompatible_v2_shard();
    match parse_shard(&bytes) {
        Err(TensorDatasetError::UnsupportedVersion { file, .. }) => {
            assert!(file.starts_with("2."), "file version was {file}");
        }
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
}

fn craft_incompatible_v2_shard() -> Vec<u8> {
    // Same structure as the empty shard but with major version 2 — reader must
    // refuse to interpret it.
    let mut bytes = craft_empty_shard_bytes();
    bytes[4..6].copy_from_slice(&2u16.to_le_bytes()); // major = 2
    recompute_footer(&mut bytes);
    bytes
}

// Ensure a dataset directory that doesn't even have a manifest fails cleanly.
#[test]
fn validate_dataset_dir_without_manifest_fails() {
    let dir = tempfile::tempdir().unwrap();
    let report = validate_dataset_dir(dir.path());
    assert!(!report.ok);
    assert!(report.errors.iter().any(|e| e.contains("manifest")));
}

// Silence dead-code warnings for helpers only used by ignored/feature paths.
#[allow(dead_code)]
fn _pathbuf_helper(p: &str) -> PathBuf {
    PathBuf::from(p)
}
