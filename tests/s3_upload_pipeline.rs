//! Integration tests for the S3 upload pipeline (Issue #1779).
//!
//! These tests exercise the full upload pipeline — provenance manifest
//! generation, dataset-version prefixing, multipart upload, and resume — using
//! a mock [`S3Transport`] so no real S3 credentials are needed.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use fluxion::ai::batch_runner::sampling;
use fluxion::ai::batch_runner::{BatchResults, ParameterSpec, SimulationOutput};
use fluxion::ai::s3_upload::{
    MultipartUploadState, PartResult, ProvenanceManifest, ProvenanceManifestBuilder, PutResult,
    S3Transport, S3UploadConfig, S3Uploader, DEFAULT_PART_SIZE,
};
use fluxion::ai::tensor_dataset::{
    TensorDatasetWriter, TensorFeatureSpec, TensorSample, TENSOR_DATASET_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};

// =============================================================================
// Mock S3 transport — simulates S3 in memory
// =============================================================================

/// A stored object in the mock S3 backend.
#[derive(Clone, Debug)]
struct MockObject {
    body: Vec<u8>,
    metadata: BTreeMap<String, String>,
}

/// An in-progress multipart upload in the mock backend.
#[derive(Clone, Debug, Default)]
struct MockMultipart {
    metadata: BTreeMap<String, String>,
    parts: std::collections::BTreeMap<u32, Vec<u8>>,
}

/// Mock S3 backend that stores objects in memory.
#[derive(Clone, Default)]
struct MockS3Backend {
    objects: Arc<Mutex<std::collections::BTreeMap<String, MockObject>>>,
    multiparts: Arc<Mutex<std::collections::BTreeMap<(String, String), MockMultipart>>>,
    /// If set, each `create_multipart` call waits this many ms (for testing
    /// resume timing).
    part_delay_ms: u64,
}

/// Mock transport implementing [`S3Transport`].
struct MockS3Transport {
    backend: MockS3Backend,
}

impl MockS3Backend {
    fn new() -> Self {
        MockS3Backend::default()
    }

    fn object_count(&self) -> usize {
        self.objects.lock().unwrap().len()
    }

    fn get_object(&self, key: &str) -> Option<MockObject> {
        self.objects.lock().unwrap().get(key).cloned()
    }

    // Kept as a mock-backend helper for future tests.
    #[allow(dead_code)]
    fn has_object(&self, key: &str) -> bool {
        self.objects.lock().unwrap().contains_key(key)
    }
}

impl MockS3Transport {
    fn new(backend: MockS3Backend) -> Self {
        MockS3Transport { backend }
    }
}

impl S3Transport for MockS3Transport {
    fn put_object(
        &self,
        _bucket: &str,
        key: &str,
        body: &[u8],
        metadata: &BTreeMap<String, String>,
    ) -> fluxion::ai::s3_upload::Result<PutResult> {
        let etag = format!("mock-etag-{}", body.len());
        self.backend.objects.lock().unwrap().insert(
            key.to_string(),
            MockObject {
                body: body.to_vec(),
                metadata: metadata.clone(),
            },
        );
        Ok(PutResult {
            etag,
            key: key.to_string(),
            size: body.len(),
        })
    }

    fn head_object(
        &self,
        _bucket: &str,
        key: &str,
    ) -> fluxion::ai::s3_upload::Result<Option<usize>> {
        let size = self
            .backend
            .objects
            .lock()
            .unwrap()
            .get(key)
            .map(|o| o.body.len());
        Ok(size)
    }

    fn create_multipart(
        &self,
        _bucket: &str,
        key: &str,
        metadata: &BTreeMap<String, String>,
    ) -> fluxion::ai::s3_upload::Result<String> {
        let upload_id = format!("upload-{}", uuid_like(key));
        self.backend.multiparts.lock().unwrap().insert(
            (key.to_string(), upload_id.clone()),
            MockMultipart {
                metadata: metadata.clone(),
                parts: std::collections::BTreeMap::new(),
            },
        );
        Ok(upload_id)
    }

    fn upload_part(
        &self,
        _bucket: &str,
        key: &str,
        upload_id: &str,
        part_number: u32,
        body: &[u8],
    ) -> fluxion::ai::s3_upload::Result<PartResult> {
        if self.backend.part_delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.backend.part_delay_ms));
        }
        let etag = format!("part-etag-{part_number}");
        let mut multiparts = self.backend.multiparts.lock().unwrap();
        let entry = multiparts
            .get_mut(&(key.to_string(), upload_id.to_string()))
            .ok_or_else(|| fluxion::ai::s3_upload::S3UploadError::Multipart {
                key: key.to_string(),
                reason: "upload_id not found".to_string(),
            })?;
        entry.parts.insert(part_number, body.to_vec());
        Ok(PartResult {
            part_number,
            etag,
            size: body.len(),
        })
    }

    fn complete_multipart(
        &self,
        _bucket: &str,
        key: &str,
        upload_id: &str,
        parts: &[PartResult],
    ) -> fluxion::ai::s3_upload::Result<PutResult> {
        let mut multiparts = self.backend.multiparts.lock().unwrap();
        let mp_key = (key.to_string(), upload_id.to_string());
        let mp = multiparts.remove(&mp_key).ok_or_else(|| {
            fluxion::ai::s3_upload::S3UploadError::Multipart {
                key: key.to_string(),
                reason: "upload_id not found on complete".to_string(),
            }
        })?;

        // Reassemble the full body from parts
        let mut full_body = Vec::new();
        for p in parts {
            if let Some(part_data) = mp.parts.get(&p.part_number) {
                full_body.extend_from_slice(part_data);
            } else {
                return Err(fluxion::ai::s3_upload::S3UploadError::Multipart {
                    key: key.to_string(),
                    reason: format!("part {} missing", p.part_number),
                });
            }
        }

        let total_size = full_body.len();
        self.backend.objects.lock().unwrap().insert(
            key.to_string(),
            MockObject {
                body: full_body,
                metadata: mp.metadata,
            },
        );

        Ok(PutResult {
            etag: format!("complete-etag-{total_size}"),
            key: key.to_string(),
            size: total_size,
        })
    }

    fn abort_multipart(
        &self,
        _bucket: &str,
        key: &str,
        upload_id: &str,
    ) -> fluxion::ai::s3_upload::Result<()> {
        self.backend
            .multiparts
            .lock()
            .unwrap()
            .remove(&(key.to_string(), upload_id.to_string()));
        Ok(())
    }
}

/// Simple deterministic pseudo-ID from a string.
fn uuid_like(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// =============================================================================
// Test helpers
// =============================================================================

/// Create a temporary dataset directory with a known number of shards and
/// samples per shard.
fn make_dataset_dir(
    n_shards: usize,
    samples_per_shard: usize,
    timeseries_length: usize,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let target_names: Vec<String> = TensorSample::DEFAULT_TARGET_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut writer = TensorDatasetWriter::new(
        dir.path(),
        TensorFeatureSpec::defaults(),
        target_names,
        timeseries_length,
        samples_per_shard,
    )
    .unwrap();

    let mut sample_id = 0;
    for _shard in 0..n_shards {
        for _s in 0..samples_per_shard {
            let ts = if timeseries_length > 0 {
                Some(
                    (0..timeseries_length)
                        .map(|i| (sample_id + i) as f64)
                        .collect(),
                )
            } else {
                None
            };
            let sample = TensorSample {
                sample_id,
                inputs: vec![1.0 + sample_id as f64 * 0.1, 20.0, 25.0],
                targets: vec![100.0 + sample_id as f64, 5000.0, 3000.0, 60.0, 40.0, 5.0],
                timeseries: ts,
            };
            writer.push(sample).unwrap();
            sample_id += 1;
        }
    }
    writer.finish().unwrap();
    dir
}

/// Build an [`S3UploadConfig`] suitable for tests (mock transport).
fn test_config(state_dir: Option<PathBuf>, part_size: usize) -> S3UploadConfig {
    S3UploadConfig {
        bucket: "test-bucket".to_string(),
        key_prefix: "datasets/ftds".to_string(),
        region: "us-east-1".to_string(),
        credentials: fluxion::ai::s3_upload::AwsCredentials {
            access_key_id: "test-key".to_string(),
            secret_access_key: "test-secret".to_string(),
            session_token: None,
        },
        part_size,
        multipart_threshold: part_size,
        state_dir,
    }
}

/// Build a provenance manifest builder with standard test fields.
fn test_provenance_builder(s3_prefix: &str) -> ProvenanceManifestBuilder {
    ProvenanceManifestBuilder::new("test-bucket", s3_prefix)
        .solver_version("9r4c-1.0")
        .git_sha("a1b2c3d")
        .parameter_seed(42)
        .weather_source("TMY3-4A")
}

// =============================================================================
// Provenance manifest tests
// =============================================================================

// Schema snapshot documenting the on-disk provenance manifest shape; not
// constructed directly in code.
#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
struct ProvenanceSerdeCheck {
    provenance_schema_version: String,
    solver_version: String,
    git_sha: String,
    parameter_seed: u64,
    weather_source: String,
    dataset_hash: String,
}

#[test]
fn provenance_manifest_serializes_all_required_fields() {
    let dir = make_dataset_dir(2, 3, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();

    let prov = test_provenance_builder("datasets/ftds/v1/abc").build(&manifest);

    let json = serde_json::to_string(&prov).unwrap();
    assert!(json.contains("solver_version"));
    assert!(json.contains("git_sha"));
    assert!(json.contains("parameter_seed"));
    assert!(json.contains("weather_source"));
    assert!(json.contains("dataset_hash"));
    assert!(json.contains("n_samples"));
    assert!(json.contains("n_shards"));
    assert!(json.contains("generated_at_utc"));
    assert!(json.contains("provenance_created_at_utc"));
}

#[test]
fn provenance_manifest_roundtrip() {
    let dir = make_dataset_dir(1, 2, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();

    let prov = test_provenance_builder("datasets/ftds/v1/abc").build(&manifest);

    let json = serde_json::to_string(&prov).unwrap();
    let back: ProvenanceManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(back.solver_version, "9r4c-1.0");
    assert_eq!(back.git_sha, "a1b2c3d");
    assert_eq!(back.parameter_seed, 42);
    assert_eq!(back.weather_source, "TMY3-4A");
    assert_eq!(back.n_samples, 2);
    assert_eq!(back.n_shards, 1);
    assert_eq!(back.dataset_schema_version, TENSOR_DATASET_SCHEMA_VERSION);
    assert_eq!(back.dataset_hash, prov.dataset_hash);
}

#[test]
fn provenance_accepts_custom_generation_parameters() {
    let dir = make_dataset_dir(1, 1, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();

    let params = serde_json::json!({
        "timesteps_per_sample": 8760,
        "case": "Case900",
        "construction": "HighMass",
    });

    let prov = test_provenance_builder("p")
        .generation_parameters(params.clone())
        .build(&manifest);

    assert_eq!(prov.generation_parameters, params);
}

#[test]
fn provenance_dataset_hash_differs_for_different_datasets() {
    let dir1 = make_dataset_dir(1, 5, 0);
    let dir2 = make_dataset_dir(1, 10, 0);
    let m1 = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir1.path()).unwrap();
    let m2 = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir2.path()).unwrap();

    let prov1 = test_provenance_builder("p").build(&m1);
    let prov2 = test_provenance_builder("p").build(&m2);

    assert_ne!(prov1.dataset_hash, prov2.dataset_hash);
}

// =============================================================================
// Dataset-version prefixing tests
// =============================================================================

#[test]
fn upload_uses_version_prefixed_keys() {
    let dir = make_dataset_dir(2, 3, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let backend = MockS3Backend::new();
    let config = test_config(None, DEFAULT_PART_SIZE);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // Every key should be under datasets/ftds/v1/<hash>/
    for obj in &report.objects {
        assert!(
            obj.key.starts_with("datasets/ftds/v1/"),
            "key {} is not version-prefixed",
            obj.key
        );
    }

    // The provenance manifest should be at the dataset prefix
    let provenance_key = report
        .objects
        .iter()
        .map(|o| &o.key)
        .find(|k| k.ends_with("/provenance.json"));
    assert!(provenance_key.is_some(), "provenance.json must be uploaded");

    // The tensor manifest should also be uploaded
    let manifest_key = report
        .objects
        .iter()
        .map(|o| &o.key)
        .find(|k| k.ends_with("/manifest.json"));
    assert!(manifest_key.is_some(), "manifest.json must be uploaded");
}

#[test]
fn upload_prefix_includes_schema_major_version() {
    let config = test_config(None, DEFAULT_PART_SIZE);
    // The prefix format is: <key_prefix>/v<major>/<dataset_hash>
    let prefix = config.dataset_prefix("1.2.3", "abc123");
    assert!(prefix.contains("/v1/abc123"));
    assert!(!prefix.contains("/v1.2.3/"));
}

// =============================================================================
// Simple PUT upload tests (small shards)
// =============================================================================

#[test]
fn upload_small_dataset_uses_simple_put() {
    let dir = make_dataset_dir(2, 3, 0); // tiny shards
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let backend = MockS3Backend::new();
    let config = test_config(None, DEFAULT_PART_SIZE);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // No multipart for small shards
    assert_eq!(report.multipart_uploads, 0);
    assert_eq!(report.parts_resumed, 0);

    // 2 shards + manifest + provenance = 4 objects
    assert_eq!(report.objects_uploaded, 4);
    assert!(report.bytes_uploaded > 0);

    // Objects stored in mock S3
    assert_eq!(backend.object_count(), 4);
}

#[test]
fn upload_round_trips_shard_data() {
    let dir = make_dataset_dir(1, 2, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let backend = MockS3Backend::new();
    let config = test_config(None, DEFAULT_PART_SIZE);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // Verify shard data in mock S3 matches local file
    let shard_key = report
        .objects
        .iter()
        .find(|o| o.key.contains("shard-"))
        .map(|o| o.key.clone())
        .unwrap();
    let shard_name = shard_key.split('/').next_back().unwrap();
    let local_bytes = fs::read(dir.path().join(shard_name)).unwrap();
    let remote_obj = backend.get_object(&shard_key).unwrap();
    assert_eq!(remote_obj.body, local_bytes);
}

#[test]
fn upload_attaches_metadata_to_shards() {
    let dir = make_dataset_dir(1, 1, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let backend = MockS3Backend::new();
    let config = test_config(None, DEFAULT_PART_SIZE);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    let shard_key = report
        .objects
        .iter()
        .find(|o| o.key.contains("shard-"))
        .map(|o| o.key.clone())
        .unwrap();
    let obj = backend.get_object(&shard_key).unwrap();
    // The dataset-hash metadata should be attached
    assert!(obj.metadata.contains_key("dataset-hash"));
    assert!(obj.metadata.contains_key("solver-version"));
}

// =============================================================================
// Multipart upload tests (large shards)
// =============================================================================

#[test]
fn upload_large_shard_uses_multipart() {
    // Create a dataset with a large "shard" by writing many samples to a
    // single shard, then lowering the multipart threshold.
    let dir = make_dataset_dir(1, 100, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let backend = MockS3Backend::new();
    // Use a very small part size / threshold so even a modest shard triggers
    // multipart.
    let small_threshold = 512;
    let config = test_config(None, small_threshold);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // At least one object used multipart (the shard)
    assert!(
        report.multipart_uploads >= 1,
        "expected at least 1 multipart upload, got {}",
        report.multipart_uploads
    );

    // The shard was reassembled correctly — verify round-trip
    let shard_key = report
        .objects
        .iter()
        .find(|o| o.key.contains("shard-"))
        .map(|o| o.key.clone())
        .unwrap();
    let shard_name = shard_key.split('/').next_back().unwrap();
    let local_bytes = fs::read(dir.path().join(shard_name)).unwrap();
    let remote_obj = backend.get_object(&shard_key).unwrap();
    assert_eq!(remote_obj.body.len(), local_bytes.len());
    assert_eq!(remote_obj.body, local_bytes);
}

#[test]
fn multipart_reassembles_correctly() {
    let dir = make_dataset_dir(1, 50, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let backend = MockS3Backend::new();
    let config = test_config(None, 256);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // Find the shard object that was uploaded via multipart
    let shard_obj = report
        .objects
        .iter()
        .find(|o| o.used_multipart)
        .expect("at least one object should use multipart");
    assert!(shard_obj.n_parts > 1, "multipart should have > 1 part");

    // Verify the reassembled object matches the original
    let shard_name = shard_obj.key.split('/').next_back().unwrap();
    let local_bytes = fs::read(dir.path().join(shard_name)).unwrap();
    let remote_obj = backend.get_object(&shard_obj.key).unwrap();
    assert_eq!(remote_obj.body, local_bytes);
}

// =============================================================================
// Resume tests
// =============================================================================

#[test]
fn multipart_resume_skips_completed_parts() {
    let dir = make_dataset_dir(1, 80, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();

    let state_dir = tempfile::tempdir().unwrap();

    // First, simulate a partial upload: create the state file as if part 1
    // of a shard was already uploaded.
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);
    let dataset_hash = prov.dataset_hash.clone();
    let s3_prefix = format!("datasets/ftds/v1/{dataset_hash}");

    let shard_name = &manifest.shards[0].path;
    let shard_key = format!("{s3_prefix}/{shard_name}");
    let shard_body = fs::read(dir.path().join(shard_name.as_str())).unwrap();
    let part_size = 512usize;
    let total_parts = shard_body.len().div_ceil(part_size) as u32;

    // Simulate: part 1 was already uploaded
    let state = MultipartUploadState {
        key: shard_key.clone(),
        upload_id: "upload-fake".to_string(),
        total_size: shard_body.len(),
        part_size,
        total_parts,
        completed_parts: vec![PartResult {
            part_number: 1,
            etag: "part-etag-1".to_string(),
            size: part_size,
        }],
        local_path: String::new(),
    };
    state.save(state_dir.path()).unwrap();

    // Now run the full upload — the shard should resume from part 2.
    let backend = MockS3Backend::new();
    // Pre-seed the multipart in the backend so "upload-fake" is known
    {
        let mut mp = backend.multiparts.lock().unwrap();
        mp.insert(
            (shard_key.clone(), "upload-fake".to_string()),
            MockMultipart {
                metadata: BTreeMap::new(),
                parts: std::collections::BTreeMap::from([(1, shard_body[..part_size].to_vec())]),
            },
        );
    }

    let config = test_config(Some(state_dir.path().to_path_buf()), part_size);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // At least one part was resumed
    assert!(
        report.parts_resumed >= 1,
        "expected at least 1 part resumed, got {}",
        report.parts_resumed
    );

    // The state file should be cleaned up after completion
    assert!(
        !state_dir
            .path()
            .join(MultipartUploadState::state_filename(&shard_key))
            .exists(),
        "state file should be removed after successful completion"
    );

    // The reassembled shard should match the original
    let remote_obj = backend.get_object(&shard_key).unwrap();
    assert_eq!(remote_obj.body, shard_body);
}

#[test]
fn upload_is_idempotent_across_runs() {
    // Running upload_dataset twice with the same data produces the same
    // dataset_hash and the same S3 prefix (idempotency).
    let dir = make_dataset_dir(1, 5, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();

    let backend1 = MockS3Backend::new();
    let config1 = test_config(None, DEFAULT_PART_SIZE);
    let prov1 = test_provenance_builder("datasets/ftds").build(&manifest);
    let uploader1 = S3Uploader::with_transport(config1, MockS3Transport::new(backend1));
    let report1 = uploader1.upload_dataset(dir.path(), prov1).unwrap();

    let backend2 = MockS3Backend::new();
    let config2 = test_config(None, DEFAULT_PART_SIZE);
    let prov2 = test_provenance_builder("datasets/ftds").build(&manifest);
    let uploader2 = S3Uploader::with_transport(config2, MockS3Transport::new(backend2));
    let report2 = uploader2.upload_dataset(dir.path(), prov2).unwrap();

    assert_eq!(report1.s3_prefix, report2.s3_prefix);
    assert_eq!(report1.dataset_hash, report2.dataset_hash);
    assert_eq!(report1.objects.len(), report2.objects.len());
}

// =============================================================================
// Error handling tests
// =============================================================================

#[test]
fn upload_fails_without_manifest() {
    let dir = tempfile::tempdir().unwrap(); // empty dir, no manifest.json
    let prov = test_provenance_builder("datasets/ftds").build(
        &fluxion::ai::tensor_dataset::TensorDatasetManifest {
            schema_version: "1.0.0".to_string(),
            created_at_utc: "2024-01-01T00:00:00Z".to_string(),
            dtype: fluxion::ai::tensor_dataset::TensorDType::F64,
            n_samples_total: 0,
            n_input_features: 0,
            input_feature_names: vec![],
            target_names: vec![],
            has_timeseries: false,
            timeseries_length: 0,
            normalization: None,
            shards: vec![],
        },
    );

    let backend = MockS3Backend::new();
    let config = test_config(None, DEFAULT_PART_SIZE);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend));

    let result = uploader.upload_dataset(dir.path(), prov);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        fluxion::ai::s3_upload::S3UploadError::NoManifest
    ));
}

// =============================================================================
// End-to-end: BatchResults → tensor dataset → S3 upload
// =============================================================================

#[test]
fn end_to_end_batch_results_to_s3() {
    // Build a small BatchResults (simulating 9R4C solver output), format into
    // a tensor dataset, then upload to mock S3.
    let params = vec![
        ParameterSpec::uniform("window_u_value", 0, 1.0, 3.0),
        ParameterSpec::uniform("heating_setpoint", 1, 18.0, 22.0),
        ParameterSpec::uniform("cooling_setpoint", 2, 24.0, 28.0),
    ];
    let manifest = sampling::build_manifest(params, 10, 42);

    // Simulate solver outputs
    let outputs: Vec<SimulationOutput> = (0..10)
        .map(|i| {
            SimulationOutput::success(
                i,
                1000.0 + i as f64 * 10.0,
                5000.0,
                3000.0,
                600.0,
                400.0,
                50.0,
                vec![],
            )
        })
        .collect();
    let results = BatchResults::new(outputs);

    // Format into tensor dataset
    let dir = tempfile::tempdir().unwrap();
    let feature_names: Vec<String> = TensorFeatureSpec::defaults()
        .iter()
        .map(|s| s.name.clone())
        .collect();
    let target_names: Vec<String> = TensorSample::DEFAULT_TARGET_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    let extraction = fluxion::ai::tensor_dataset::batch_results_to_samples(
        &results,
        &manifest,
        &feature_names,
        0,
    );
    assert_eq!(extraction.samples.len(), 10);

    let mut writer = TensorDatasetWriter::new(
        dir.path(),
        TensorFeatureSpec::defaults(),
        target_names,
        0,
        5, // 2 shards
    )
    .unwrap();
    for sample in extraction.samples {
        writer.push(sample).unwrap();
    }
    let tensor_manifest = writer.finish().unwrap();
    assert_eq!(tensor_manifest.n_samples_total, 10);

    // Upload to S3
    let prov = test_provenance_builder("datasets/ftds")
        .solver_version("9r4c-1.0")
        .parameter_seed(manifest.seed)
        .weather_source("TMY3-4A")
        .git_sha("test-sha-001")
        .build(&tensor_manifest);

    let backend = MockS3Backend::new();
    let config = test_config(None, DEFAULT_PART_SIZE);
    let uploader = S3Uploader::with_transport(config, MockS3Transport::new(backend.clone()));

    let report = uploader.upload_dataset(dir.path(), prov).unwrap();

    // 2 shards + manifest + provenance
    assert_eq!(report.objects_uploaded, 4);
    assert_eq!(report.provenance.solver_version, "9r4c-1.0");
    assert_eq!(report.provenance.parameter_seed, 42);
    assert_eq!(report.provenance.git_sha, "test-sha-001");
    assert_eq!(report.provenance.weather_source, "TMY3-4A");
    assert_eq!(report.provenance.n_samples, 10);
    assert_eq!(report.provenance.n_shards, 2);
    assert!(!report.dataset_hash.is_empty());
}

#[test]
fn provenance_manifest_written_to_dataset_dir() {
    // Verify that the provenance manifest can be written alongside the dataset
    // for local provenance tracking (even without S3).
    let dir = make_dataset_dir(1, 3, 0);
    let manifest = fluxion::ai::tensor_dataset::TensorDatasetManifest::load(dir.path()).unwrap();
    let prov = test_provenance_builder("datasets/ftds").build(&manifest);

    let prov_path = dir.path().join("provenance.json");
    let bytes = serde_json::to_vec_pretty(&prov).unwrap();
    fs::write(&prov_path, &bytes).unwrap();

    // Read back and verify
    let loaded: serde_json::Value = serde_json::from_slice(&fs::read(&prov_path).unwrap()).unwrap();
    assert_eq!(loaded["solver_version"], "9r4c-1.0");
    assert_eq!(loaded["parameter_seed"], 42);
    assert_eq!(loaded["weather_source"], "TMY3-4A");
}
