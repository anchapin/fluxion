//! Tensor dataset output schema and formatting for ML surrogate training (Issue #1778, plan key T5.3).
//!
//! Defines a versioned, sharded tensor dataset format ("FTDS" — Fluxion Tensor
//! Dataset Shard) that turns physics-solver outputs (`BatchResults` /
//! `SimulationOutput` from [`crate::ai::batch_runner`]) into stable training
//! tensors consumable by the surrogate trainers.
//!
//! ## Design
//!
//! - **Schema versioning**: [`TENSOR_DATASET_SCHEMA_VERSION`] + a serialized
//!   [`TensorDatasetManifest`] that travels alongside the data.
//! - **Tensor layout**: three contiguous row-major arrays per shard:
//!   `inputs [N, F]`, `targets [N, T]`, optional `timeseries [N, L]` (e.g.
//!   zone-temperature traces), where `N` = samples in shard, `F` = input
//!   feature count, `T` = scalar target count, `L` = fixed timeseries length.
//! - **Binary shard format**: magic `b"FTDS"` + versioned fixed header +
//!   JSON sidecar (feature/target names) + raw little-endian `f64` payload +
//!   per-sample `sample_id` array + `SHA-256` footer covering the whole file.
//! - **Custom binary over HDF5/NPZ**: avoids system libraries (HDF5) and
//!   keeps the published crate dependency-light and under the 10 MB crates.io
//!   cap. The format is fully self-describing and forward-compatible via the
//!   semver header.
//! - **Validation**: [`validate_shard`] / [`validate_dataset_dir`] reject
//!   malformed shards (bad magic, version mismatch, truncated payload, NaN/Inf
//!   values, shape/dtype mismatch vs. manifest).
//! - **Formatting utilities**: [`TensorSample::from_simulation_output`] and
//!   [`batch_results_to_samples`] convert physics outputs into the schema.
//!
//! ## Example
//!
//! ```no_run
//! use fluxion::ai::tensor_dataset::{
//!     TensorDatasetWriter, TensorSample, TensorFeatureSpec, TensorDType,
//! };
//! use std::path::Path;
//!
//! let feature_spec = TensorFeatureSpec::defaults();
//! let target_names: Vec<String> = TensorSample::default_target_names()
//!     .iter().map(|s| s.to_string()).collect();
//! let mut writer = TensorDatasetWriter::new(
//!     Path::new("/tmp/dataset"),
//!     feature_spec,
//!     target_names,
//!     /* timeseries_length */ 0,
//!     /* shard_size */ 1024,
//! ).unwrap();
//!
//! // writer.push(sample) ... then:
//! // let manifest = writer.finish().unwrap();
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ai::batch_runner::{BatchResults, ParameterManifest, SimulationOutput};

/// Current schema version of the FTDS format (semver).
///
/// Increment the *major* field for breaking layout changes (magic, header
/// field order, dtype semantics). Minor/patch are additive.
pub const TENSOR_DATASET_SCHEMA_VERSION: &str = "1.0.0";

/// Magic bytes opening every FTDS shard file: ASCII `"FTDS"`.
pub const FTDS_MAGIC: [u8; 4] = *b"FTDS";

/// Fixed byte length of the binary shard header (before the JSON sidecar).
pub const FTDS_FIXED_HEADER_LEN: usize = 48;

/// Default shard size (samples per shard) when none is specified.
pub const DEFAULT_SHARD_SIZE: usize = 1024;

/// Supported tensor element types. FTDS v1.x writes [`TensorDType::F64`] only;
/// the enum exists so future versions can add F16/BF16 without a major bump.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TensorDType {
    /// 32-bit float (reserved; not emitted by v1.x writers).
    F32 = 0,
    /// 64-bit float (the only dtype emitted by v1.x writers).
    F64 = 1,
}

impl TensorDType {
    /// Element size in bytes.
    pub const fn byte_size(self) -> usize {
        match self {
            TensorDType::F32 => 4,
            TensorDType::F64 => 8,
        }
    }

    /// Discriminant byte as written to the shard header.
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Parse the discriminant byte read from a shard header.
    pub fn from_byte(b: u8) -> std::result::Result<Self, TensorDatasetError> {
        match b {
            0 => Ok(TensorDType::F32),
            1 => Ok(TensorDType::F64),
            other => Err(TensorDatasetError::UnknownDType(other)),
        }
    }
}

/// All error conditions produced by this module.
#[derive(Debug, Error)]
pub enum TensorDatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON (de)serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bad FTDS magic bytes: expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 4], got: [u8; 4] },
    #[error("unsupported schema version: file={file}, reader={reader}")]
    UnsupportedVersion { file: String, reader: String },
    #[error("unsupported dtype tag in shard: {0}")]
    UnknownDType(u8),
    #[error("truncated shard: needed {needed} bytes, have {have} ({context})")]
    Truncated {
        needed: usize,
        have: usize,
        context: &'static str,
    },
    #[error("shape mismatch in {field}: declared={declared}, actual={actual}")]
    ShapeMismatch {
        field: &'static str,
        declared: usize,
        actual: usize,
    },
    #[error("dtype mismatch: manifest={manifest:?}, shard={shard:?}")]
    DTypeMismatch {
        manifest: TensorDType,
        shard: TensorDType,
    },
    #[error("non-finite value (NaN/Inf) at {location} index {index}")]
    NonFinite {
        location: &'static str,
        index: usize,
    },
    #[error("empty dataset: no samples written")]
    Empty,
    #[error("empty shard: declared n_samples == 0")]
    EmptyShard,
    #[error("sample arity mismatch: expected {expected} {kind}, got {got}")]
    Arity {
        kind: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("failed simulation cannot be formatted (sample_id={sample_id}): {reason}")]
    FailedSimulation { sample_id: usize, reason: String },
    #[error("sha256 mismatch for {path}: manifest={manifest}, actual={actual}")]
    Integrity {
        path: String,
        manifest: String,
        actual: String,
    },
    #[error("manifest references {declared} shards but {found} were found on disk")]
    MissingShards { declared: usize, found: usize },
}

/// Result alias used throughout the module.
pub type Result<T> = std::result::Result<T, TensorDatasetError>;

/// Specification of a single input feature column.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorFeatureSpec {
    /// Column name (matches `TensorDatasetManifest::input_feature_names`).
    pub name: String,
    /// Physical unit, e.g. `"W/m^2K"`. Empty string allowed for dimensionless.
    pub unit: String,
}

impl TensorFeatureSpec {
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        TensorFeatureSpec {
            name: name.into(),
            unit: unit.into(),
        }
    }

    /// Default input feature set used when converting a `SimulationOutput`
    /// produced by the standard parameter vector (window U-value, heating
    /// setpoint, cooling setpoint). Matches the indices documented in
    /// [`crate::ai::batch_runner::ParameterSpec`].
    pub fn defaults() -> Vec<Self> {
        vec![
            TensorFeatureSpec::new("window_u_value", "W/m^2K"),
            TensorFeatureSpec::new("heating_setpoint", "degC"),
            TensorFeatureSpec::new("cooling_setpoint", "degC"),
        ]
    }
}

/// Per-sample metadata captured alongside the tensors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub climate_zone: String,
    pub building_type: String,
}

impl SampleMetadata {
    pub fn new(climate_zone: impl Into<String>, building_type: impl Into<String>) -> Self {
        SampleMetadata {
            climate_zone: climate_zone.into(),
            building_type: building_type.into(),
        }
    }

    /// Conservative defaults matching `batch_runner::sampling::build_manifest`.
    pub fn defaults() -> Self {
        SampleMetadata::new("4A", "residential")
    }
}

/// Normalization statistics computed over the dataset (zero-mean/unit-variance
/// per column). Optional; present only when computed by the writer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub input_mean: Vec<f64>,
    pub input_std: Vec<f64>,
    pub target_mean: Vec<f64>,
    pub target_std: Vec<f64>,
}

/// Reference to a shard file inside a dataset directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardRef {
    /// File name (relative to the dataset directory).
    pub path: String,
    /// Number of samples in this shard.
    pub n_samples: usize,
    /// Lowercase hex SHA-256 of the shard file contents (excluding the
    /// trailing 32-byte checksum itself).
    pub sha256: String,
}

/// Top-level manifest describing a complete tensor dataset.
///
/// Serialized to `manifest.json` in the dataset directory and embedded (as a
/// JSON sidecar) in every shard header.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorDatasetManifest {
    pub schema_version: String,
    pub created_at_utc: String,
    pub dtype: TensorDType,
    pub n_samples_total: usize,
    pub n_input_features: usize,
    pub input_feature_names: Vec<String>,
    pub target_names: Vec<String>,
    pub has_timeseries: bool,
    pub timeseries_length: usize,
    pub normalization: Option<NormalizationStats>,
    pub shards: Vec<ShardRef>,
}

impl TensorDatasetManifest {
    /// Load a `manifest.json` from a dataset directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("manifest.json");
        let bytes = fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Write this manifest to `<dir>/manifest.json`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let path = dir.join("manifest.json");
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(path, bytes)?;
        Ok(())
    }
}

/// One in-memory training sample.
#[derive(Clone, Debug)]
pub struct TensorSample {
    pub sample_id: usize,
    /// Length must equal `TensorDatasetManifest::n_input_features`.
    pub inputs: Vec<f64>,
    /// Length must equal `TensorDatasetManifest::target_names.len()`.
    pub targets: Vec<f64>,
    /// Length must equal `TensorDatasetManifest::timeseries_length` when
    /// `has_timeseries` is true; `None` otherwise.
    pub timeseries: Option<Vec<f64>>,
}

impl TensorSample {
    /// Canonical scalar target column names produced by the default formatter.
    ///
    /// Order MUST stay stable across schema versions — append only.
    pub const DEFAULT_TARGET_NAMES: &'static [&'static str] = &[
        "total_energy_kwh",
        "peak_heating_load_w",
        "peak_cooling_load_w",
        "annual_heating_kwh",
        "annual_cooling_kwh",
        "eui_kwh_m2",
    ];

    /// Convenience accessor returning the default target names as owned vec.
    pub fn default_target_names() -> Vec<&'static str> {
        Self::DEFAULT_TARGET_NAMES.to_vec()
    }

    /// Convert a successful `SimulationOutput` plus its parameter vector into
    /// a `TensorSample`.
    ///
    /// `input_feature_names` must align positionally with `parameters` (i.e.
    /// `input_feature_names[i]` describes `parameters[i]`). The default feature
    /// set is [`TensorFeatureSpec::defaults`].
    ///
    /// Returns an error if the simulation failed or any arity is inconsistent.
    pub fn from_simulation_output(
        output: &SimulationOutput,
        parameters: &[f64],
        input_feature_names: &[String],
        timeseries_length: usize,
    ) -> Result<Self> {
        if !output.success {
            return Err(TensorDatasetError::FailedSimulation {
                sample_id: output.sample_id,
                reason: output
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "(no message)".to_string()),
            });
        }
        if parameters.len() != input_feature_names.len() {
            return Err(TensorDatasetError::Arity {
                kind: "input features",
                expected: input_feature_names.len(),
                got: parameters.len(),
            });
        }

        let targets = vec![
            output.total_energy_kwh,
            output.peak_heating_load_w,
            output.peak_cooling_load_w,
            output.annual_heating_kwh,
            output.annual_cooling_kwh,
            output.eui_kwh_m2,
        ];

        let timeseries = if timeseries_length == 0 {
            if !output.zone_temperatures.is_empty() {
                return Err(TensorDatasetError::Arity {
                    kind: "timeseries (writer declared length 0 but output has data)",
                    expected: 0,
                    got: output.zone_temperatures.len(),
                });
            }
            None
        } else {
            if output.zone_temperatures.len() != timeseries_length {
                return Err(TensorDatasetError::Arity {
                    kind: "timeseries",
                    expected: timeseries_length,
                    got: output.zone_temperatures.len(),
                });
            }
            Some(output.zone_temperatures.clone())
        };

        Ok(TensorSample {
            sample_id: output.sample_id,
            inputs: parameters.to_vec(),
            targets,
            timeseries,
        })
    }
}

/// Outcome of converting a `BatchResults` batch into tensor samples.
#[derive(Clone, Debug)]
pub struct DatasetExtraction {
    /// Successfully formatted samples (in input order).
    pub samples: Vec<TensorSample>,
    /// `sample_id`s that were skipped (failed sims or arity errors).
    pub skipped: Vec<(usize, String)>,
}

/// Convert an entire `BatchResults` batch into tensor samples.
///
/// Failed simulations are skipped (recorded in [`DatasetExtraction::skipped`])
/// rather than aborting the whole batch. `timeseries_length` should match the
/// value passed to the writer; pass `0` to drop zone-temperature traces.
pub fn batch_results_to_samples(
    results: &BatchResults,
    manifest: &ParameterManifest,
    input_feature_names: &[String],
    timeseries_length: usize,
) -> DatasetExtraction {
    let by_id: std::collections::HashMap<usize, &crate::ai::batch_runner::ParameterSample> =
        manifest.samples.iter().map(|s| (s.sample_id, s)).collect();

    let mut samples = Vec::with_capacity(results.outputs.len());
    let mut skipped = Vec::new();

    for output in &results.outputs {
        let parameters = match by_id.get(&output.sample_id) {
            Some(p) => &p.parameters,
            None => {
                skipped.push((
                    output.sample_id,
                    "no matching parameter sample in manifest".to_string(),
                ));
                continue;
            }
        };
        match TensorSample::from_simulation_output(
            output,
            parameters,
            input_feature_names,
            timeseries_length,
        ) {
            Ok(s) => samples.push(s),
            Err(e) => skipped.push((output.sample_id, e.to_string())),
        }
    }

    DatasetExtraction { samples, skipped }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Streaming writer that buffers samples and flushes them to numbered shard
/// files inside a dataset directory.
pub struct TensorDatasetWriter {
    out_dir: PathBuf,
    shard_size: usize,
    feature_spec: Vec<TensorFeatureSpec>,
    target_names: Vec<String>,
    timeseries_length: usize,
    buffer: Vec<TensorSample>,
    shard_index: usize,
    total_written: usize,
    shards: Vec<ShardRef>,
    created_at_utc: String,
}

impl TensorDatasetWriter {
    /// Create a new writer. Creates `out_dir` if it does not exist.
    ///
    /// `timeseries_length == 0` disables the timeseries tensor.
    /// `shard_size` is clamped to `>= 1`.
    pub fn new(
        out_dir: &Path,
        feature_spec: Vec<TensorFeatureSpec>,
        target_names: Vec<String>,
        timeseries_length: usize,
        shard_size: usize,
    ) -> Result<Self> {
        fs::create_dir_all(out_dir)?;
        let shard_size = if shard_size == 0 { 1 } else { shard_size };
        Ok(TensorDatasetWriter {
            out_dir: out_dir.to_path_buf(),
            shard_size,
            feature_spec,
            target_names,
            timeseries_length,
            buffer: Vec::with_capacity(shard_size),
            shard_index: 0,
            total_written: 0,
            shards: Vec::new(),
            created_at_utc: current_utc_iso8601(),
        })
    }

    /// Returns the input-feature column names.
    fn input_feature_names(&self) -> Vec<String> {
        self.feature_spec.iter().map(|s| s.name.clone()).collect()
    }

    /// Append a sample. Flushes a shard when the buffer fills.
    pub fn push(&mut self, sample: TensorSample) -> Result<()> {
        let expected_in = self.feature_spec.len();
        if sample.inputs.len() != expected_in {
            return Err(TensorDatasetError::Arity {
                kind: "input features",
                expected: expected_in,
                got: sample.inputs.len(),
            });
        }
        if sample.targets.len() != self.target_names.len() {
            return Err(TensorDatasetError::Arity {
                kind: "targets",
                expected: self.target_names.len(),
                got: sample.targets.len(),
            });
        }
        match (sample.timeseries.as_ref(), self.timeseries_length) {
            (None, 0) => {}
            (Some(v), n) if v.len() == n => {}
            (ts, n) => {
                return Err(TensorDatasetError::Arity {
                    kind: "timeseries",
                    expected: n,
                    got: ts.map(|v| v.len()).unwrap_or(0),
                });
            }
        }
        check_finite(&sample.inputs, "inputs")?;
        check_finite(&sample.targets, "targets")?;
        if let Some(ref ts) = sample.timeseries {
            check_finite(ts, "timeseries")?;
        }

        self.buffer.push(sample);
        if self.buffer.len() >= self.shard_size {
            self.flush_shard()?;
        }
        Ok(())
    }

    /// Flush all buffered samples as a final shard (no-op if buffer empty).
    pub fn finish(&mut self) -> Result<TensorDatasetManifest> {
        if !self.buffer.is_empty() {
            self.flush_shard()?;
        }
        if self.total_written == 0 {
            return Err(TensorDatasetError::Empty);
        }
        let manifest = TensorDatasetManifest {
            schema_version: TENSOR_DATASET_SCHEMA_VERSION.to_string(),
            created_at_utc: self.created_at_utc.clone(),
            dtype: TensorDType::F64,
            n_samples_total: self.total_written,
            n_input_features: self.feature_spec.len(),
            input_feature_names: self.input_feature_names(),
            target_names: self.target_names.clone(),
            has_timeseries: self.timeseries_length > 0,
            timeseries_length: self.timeseries_length,
            normalization: None,
            shards: std::mem::take(&mut self.shards),
        };
        manifest.save(&self.out_dir)?;
        Ok(manifest)
    }

    fn flush_shard(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let shard_name = format!("shard-{:06}.ftds", self.shard_index);
        let shard_path = self.out_dir.join(&shard_name);
        let n_samples = self.buffer.len();

        let header = TensorShardHeader {
            schema_version: TENSOR_DATASET_SCHEMA_VERSION.to_string(),
            dtype: TensorDType::F64,
            n_samples,
            n_input_features: self.feature_spec.len(),
            n_targets: self.target_names.len(),
            has_timeseries: self.timeseries_length > 0,
            timeseries_length: self.timeseries_length,
            input_feature_names: self.input_feature_names(),
            target_names: self.target_names.clone(),
        };

        let sample_ids: Vec<u64> = self.buffer.iter().map(|s| s.sample_id as u64).collect();

        write_shard_file(
            &shard_path,
            &header,
            self.feature_spec.len(),
            self.target_names.len(),
            self.timeseries_length,
            &self.buffer,
            &sample_ids,
        )?;

        let bytes = fs::read(&shard_path)?;
        let (checksum, _) = compute_body_checksum(&bytes);
        self.shards.push(ShardRef {
            path: shard_name,
            n_samples,
            sha256: checksum,
        });
        self.total_written += n_samples;
        self.shard_index += 1;
        self.buffer.clear();
        Ok(())
    }
}

/// Header embedded as a JSON sidecar inside each shard file (after the 48-byte
/// fixed header). Exposed so consumers can read feature/target column names and
/// declared shapes without re-deriving them from the manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorShardHeader {
    pub schema_version: String,
    pub dtype: TensorDType,
    pub n_samples: usize,
    pub n_input_features: usize,
    pub n_targets: usize,
    pub has_timeseries: bool,
    pub timeseries_length: usize,
    pub input_feature_names: Vec<String>,
    pub target_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Binary shard I/O
// ---------------------------------------------------------------------------

/// Parsed shard header + payload read back from disk.
#[derive(Clone, Debug)]
pub struct TensorShard {
    pub header: TensorShardHeader,
    pub sample_ids: Vec<u64>,
    /// Row-major `[N, F]`.
    pub inputs: Vec<f64>,
    /// Row-major `[N, T]`.
    pub targets: Vec<f64>,
    /// Row-major `[N, L]` (empty when `has_timeseries == false`).
    pub timeseries: Vec<f64>,
}

impl TensorShard {
    /// Borrow the inputs buffer as a 2D array view `[N, F]`.
    pub fn inputs_view(&self) -> ArrayView2<'_, f64> {
        let cols = self.header.n_input_features;
        let rows = self.header.n_samples;
        ArrayView2::from_shape((rows, cols), &self.inputs).unwrap_or_else(|e| {
            panic!("inputs shape ({rows},{cols}) incompatible with buffer: {e}")
        })
    }

    /// Borrow the targets buffer as a 2D array view `[N, T]`.
    pub fn targets_view(&self) -> ArrayView2<'_, f64> {
        let cols = self.header.n_targets;
        let rows = self.header.n_samples;
        ArrayView2::from_shape((rows, cols), &self.targets).unwrap_or_else(|e| {
            panic!("targets shape ({rows},{cols}) incompatible with buffer: {e}")
        })
    }
}

/// Write a single shard file in the FTDS v1 binary layout.
fn write_shard_file(
    path: &Path,
    header: &TensorShardHeader,
    n_features: usize,
    n_targets: usize,
    timeseries_length: usize,
    samples: &[TensorSample],
    sample_ids: &[u64],
) -> Result<()> {
    let sidecar_json = serde_json::to_vec(header)?;
    if sidecar_json.len() > u32::MAX as usize {
        return Err(TensorDatasetError::Truncated {
            needed: sidecar_json.len(),
            have: u32::MAX as usize,
            context: "sidecar JSON length exceeds u32",
        });
    }

    let mut file = fs::File::create(path)?;
    let mut hasher = Sha256::new();

    // --- Fixed header (48 bytes) ---
    let mut fixed = [0u8; FTDS_FIXED_HEADER_LEN];
    fixed[0..4].copy_from_slice(&FTDS_MAGIC);
    write_semver(&mut fixed[4..10], &header.schema_version);
    fixed[10] = header.dtype.as_byte();
    write_u64(&mut fixed[11..19], header.n_samples as u64);
    write_u64(&mut fixed[19..27], n_features as u64);
    write_u64(&mut fixed[27..35], n_targets as u64);
    fixed[35] = if header.has_timeseries { 1 } else { 0 };
    write_u64(&mut fixed[36..44], timeseries_length as u64);
    write_u32(&mut fixed[44..48], sidecar_json.len() as u32);

    file.write_all(&fixed)?;
    hasher.update(fixed);
    file.write_all(&sidecar_json)?;
    hasher.update(&sidecar_json);

    // --- Payload arrays (row-major little-endian f64) ---
    write_samples_array(&mut file, &mut hasher, samples, n_features, |s| &s.inputs)?;
    write_samples_array(&mut file, &mut hasher, samples, n_targets, |s| &s.targets)?;
    if header.has_timeseries {
        write_samples_array(&mut file, &mut hasher, samples, timeseries_length, |s| {
            s.timeseries.as_deref().unwrap_or(&[])
        })?;
    }

    // --- Per-sample IDs (u64 little-endian) ---
    let mut ids_bytes = Vec::with_capacity(sample_ids.len() * 8);
    for id in sample_ids {
        ids_bytes.extend_from_slice(&id.to_le_bytes());
    }
    file.write_all(&ids_bytes)?;
    hasher.update(&ids_bytes);

    // --- SHA-256 footer (32 bytes) ---
    let digest = hasher.finalize();
    file.write_all(&digest)?;
    file.flush()?;
    Ok(())
}

/// Write one row-major `[N, cols]` f64 payload array, updating the running hash.
fn write_samples_array<W, F>(
    file: &mut W,
    hasher: &mut Sha256,
    samples: &[TensorSample],
    cols: usize,
    select: F,
) -> Result<()>
where
    W: Write,
    F: Fn(&TensorSample) -> &[f64],
{
    let mut buf = Vec::with_capacity(samples.len() * cols);
    for s in samples {
        let row = select(s);
        // Arity already validated by `TensorDatasetWriter::push`; defensive
        // truncation/padding here would hide bugs, so assert length matches.
        assert_eq!(
            row.len(),
            cols,
            "internal: tensor row arity mismatch (expected {cols}, got {})",
            row.len()
        );
        for &v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    file.write_all(&buf)?;
    hasher.update(&buf);
    Ok(())
}

/// Parse an FTDS shard file from raw bytes.
pub fn parse_shard(bytes: &[u8]) -> Result<TensorShard> {
    if bytes.len() < FTDS_FIXED_HEADER_LEN {
        return Err(TensorDatasetError::Truncated {
            needed: FTDS_FIXED_HEADER_LEN,
            have: bytes.len(),
            context: "fixed header",
        });
    }

    let mut magic = [0u8; 4];
    magic.copy_from_slice(&bytes[0..4]);
    if magic != FTDS_MAGIC {
        return Err(TensorDatasetError::BadMagic {
            expected: FTDS_MAGIC,
            got: magic,
        });
    }

    let file_version = read_semver(&bytes[4..10])?;
    if !is_compatible_version(&file_version) {
        return Err(TensorDatasetError::UnsupportedVersion {
            file: file_version,
            reader: TENSOR_DATASET_SCHEMA_VERSION.to_string(),
        });
    }

    let dtype = TensorDType::from_byte(bytes[10])?;
    if dtype != TensorDType::F64 {
        // v1.x readers only support F64.
        return Err(TensorDatasetError::UnsupportedVersion {
            file: format!("dtype={dtype:?}"),
            reader: TENSOR_DATASET_SCHEMA_VERSION.to_string(),
        });
    }
    let n_samples = read_u64(&bytes[11..19]) as usize;
    let n_features = read_u64(&bytes[19..27]) as usize;
    let n_targets = read_u64(&bytes[27..35]) as usize;
    let has_timeseries = bytes[35] != 0;
    let timeseries_length = read_u64(&bytes[36..44]) as usize;
    let sidecar_len = read_u32(&bytes[44..48]) as usize;

    if n_samples == 0 {
        return Err(TensorDatasetError::EmptyShard);
    }

    let sidecar_end =
        FTDS_FIXED_HEADER_LEN
            .checked_add(sidecar_len)
            .ok_or(TensorDatasetError::Truncated {
                needed: usize::MAX,
                have: bytes.len(),
                context: "sidecar length overflow",
            })?;
    if bytes.len() < sidecar_end {
        return Err(TensorDatasetError::Truncated {
            needed: sidecar_end,
            have: bytes.len(),
            context: "sidecar JSON",
        });
    }
    let header: TensorShardHeader =
        serde_json::from_slice(&bytes[FTDS_FIXED_HEADER_LEN..sidecar_end])?;

    // Cross-check fixed header vs sidecar.
    if header.n_samples != n_samples
        || header.n_input_features != n_features
        || header.n_targets != n_targets
        || header.has_timeseries != has_timeseries
        || header.timeseries_length != timeseries_length
    {
        return Err(TensorDatasetError::ShapeMismatch {
            field: "sidecar vs fixed header",
            declared: n_samples,
            actual: header.n_samples,
        });
    }

    let elem = dtype.byte_size();
    let inputs_len = n_samples
        .checked_mul(n_features)
        .and_then(|n| n.checked_mul(elem))
        .ok_or(TensorDatasetError::Truncated {
            needed: usize::MAX,
            have: bytes.len(),
            context: "inputs length overflow",
        })?;
    let targets_len = n_samples
        .checked_mul(n_targets)
        .and_then(|n| n.checked_mul(elem))
        .ok_or(TensorDatasetError::Truncated {
            needed: usize::MAX,
            have: bytes.len(),
            context: "targets length overflow",
        })?;
    let ts_len = if has_timeseries {
        n_samples
            .checked_mul(timeseries_length)
            .and_then(|n| n.checked_mul(elem))
            .ok_or(TensorDatasetError::Truncated {
                needed: usize::MAX,
                have: bytes.len(),
                context: "timeseries length overflow",
            })?
    } else {
        0
    };
    let ids_len = n_samples
        .checked_mul(8)
        .ok_or(TensorDatasetError::Truncated {
            needed: usize::MAX,
            have: bytes.len(),
            context: "ids length overflow",
        })?;
    let footer_len = 32;

    let payload_needed = inputs_len
        .checked_add(targets_len)
        .and_then(|x| x.checked_add(ts_len))
        .and_then(|x| x.checked_add(ids_len))
        .and_then(|x| x.checked_add(footer_len))
        .ok_or(TensorDatasetError::Truncated {
            needed: usize::MAX,
            have: bytes.len(),
            context: "payload length overflow",
        })?;
    let total_needed =
        sidecar_end
            .checked_add(payload_needed)
            .ok_or(TensorDatasetError::Truncated {
                needed: usize::MAX,
                have: bytes.len(),
                context: "total length overflow",
            })?;
    if bytes.len() < total_needed {
        return Err(TensorDatasetError::Truncated {
            needed: total_needed,
            have: bytes.len(),
            context: "payload",
        });
    }

    let mut off = sidecar_end;
    let inputs = read_f64_block(&bytes[off..off + inputs_len])?;
    off += inputs_len;
    let targets = read_f64_block(&bytes[off..off + targets_len])?;
    off += targets_len;
    let timeseries = if has_timeseries {
        let v = read_f64_block(&bytes[off..off + ts_len])?;
        off += ts_len;
        v
    } else {
        Vec::new()
    };
    let sample_ids = read_u64_block(&bytes[off..off + ids_len])?;
    off += ids_len;

    // Verify footer checksum (covers everything except the 32-byte digest).
    let (expected, body_len) = compute_body_checksum(bytes);
    let actual_hex = hex_encode(&bytes[body_len..body_len + 32]);
    if expected != actual_hex {
        return Err(TensorDatasetError::Integrity {
            path: String::new(),
            manifest: expected,
            actual: actual_hex,
        });
    }
    let _ = off; // off == body_len by construction

    Ok(TensorShard {
        header,
        sample_ids,
        inputs,
        targets,
        timeseries,
    })
}

/// Read an FTDS shard file from disk.
pub fn read_shard(path: &Path) -> Result<TensorShard> {
    let bytes = fs::read(path)?;
    parse_shard(&bytes)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Result of validating a single shard or full dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationReport {
    pub ok: bool,
    pub n_samples: usize,
    pub errors: Vec<String>,
}

impl ValidationReport {
    fn ok(n_samples: usize) -> Self {
        ValidationReport {
            ok: true,
            n_samples,
            errors: Vec::new(),
        }
    }

    fn fail(errors: Vec<String>) -> Self {
        ValidationReport {
            ok: false,
            n_samples: 0,
            errors,
        }
    }
}

/// Validate a single shard file: structure, checksum, finiteness, and
/// (optionally) consistency against a manifest.
pub fn validate_shard(path: &Path, manifest: Option<&TensorDatasetManifest>) -> ValidationReport {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => return ValidationReport::fail(vec![format!("read error: {e}")]),
    };
    let shard = match parse_shard(&bytes) {
        Ok(s) => s,
        Err(e) => return ValidationReport::fail(vec![e.to_string()]),
    };

    let mut errors: Vec<String> = Vec::new();

    // Integrity checksum already verified inside parse_shard; re-affirm length.
    let (expected, body_len) = compute_body_checksum(&bytes);
    let stored = hex_encode(&bytes[body_len..body_len + 32]);
    if expected != stored {
        errors.push(format!(
            "sha256 mismatch: expected {expected}, stored {stored}"
        ));
    }

    // Finiteness of every payload value.
    for (i, v) in shard.inputs.iter().enumerate() {
        if !v.is_finite() {
            errors.push(
                TensorDatasetError::NonFinite {
                    location: "inputs",
                    index: i,
                }
                .to_string(),
            );
        }
    }
    for (i, v) in shard.targets.iter().enumerate() {
        if !v.is_finite() {
            errors.push(
                TensorDatasetError::NonFinite {
                    location: "targets",
                    index: i,
                }
                .to_string(),
            );
        }
    }
    for (i, v) in shard.timeseries.iter().enumerate() {
        if !v.is_finite() {
            errors.push(
                TensorDatasetError::NonFinite {
                    location: "timeseries",
                    index: i,
                }
                .to_string(),
            );
        }
    }

    if let Some(m) = manifest {
        if shard.header.schema_version != m.schema_version {
            errors.push(format!(
                "schema_version mismatch: manifest={}, shard={}",
                m.schema_version, shard.header.schema_version
            ));
        }
        if shard.header.dtype != m.dtype {
            errors.push(
                TensorDatasetError::DTypeMismatch {
                    manifest: m.dtype,
                    shard: shard.header.dtype,
                }
                .to_string(),
            );
        }
        if shard.header.n_input_features != m.n_input_features {
            errors.push(format!(
                "n_input_features mismatch: manifest={}, shard={}",
                m.n_input_features, shard.header.n_input_features
            ));
        }
        if shard.header.n_targets != m.target_names.len() {
            errors.push(format!(
                "n_targets mismatch: manifest={}, shard={}",
                m.target_names.len(),
                shard.header.n_targets
            ));
        }
        if shard.header.has_timeseries != m.has_timeseries
            || shard.header.timeseries_length != m.timeseries_length
        {
            errors.push(format!(
                "timeseries config mismatch: manifest=(has={},len={}), shard=(has={},len={})",
                m.has_timeseries,
                m.timeseries_length,
                shard.header.has_timeseries,
                shard.header.timeseries_length,
            ));
        }
        if shard.header.input_feature_names != m.input_feature_names {
            errors.push("input_feature_names mismatch".to_string());
        }
        if shard.header.target_names != m.target_names {
            errors.push("target_names mismatch".to_string());
        }
    }

    if errors.is_empty() {
        ValidationReport::ok(shard.header.n_samples)
    } else {
        ValidationReport::fail(errors)
    }
}

/// Validate an entire dataset directory: load `manifest.json`, verify every
/// referenced shard exists and passes [`validate_shard`].
pub fn validate_dataset_dir(dir: &Path) -> ValidationReport {
    let manifest = match TensorDatasetManifest::load(dir) {
        Ok(m) => m,
        Err(e) => return ValidationReport::fail(vec![format!("manifest load: {e}")]),
    };

    let mut total_samples = 0usize;
    let mut errors: Vec<String> = Vec::new();
    let mut found = 0usize;

    for ref_s in &manifest.shards {
        let shard_path = dir.join(&ref_s.path);
        if !shard_path.exists() {
            errors.push(format!("missing shard file: {}", ref_s.path));
            continue;
        }
        found += 1;
        let report = validate_shard(&shard_path, Some(&manifest));
        if !report.ok {
            errors.extend(
                report
                    .errors
                    .into_iter()
                    .map(|e| format!("{}: {e}", ref_s.path)),
            );
            continue;
        }
        if report.n_samples != ref_s.n_samples {
            errors.push(format!(
                "{}: n_samples mismatch manifest={}, shard={}",
                ref_s.path, ref_s.n_samples, report.n_samples
            ));
        }
        // Verify stored SHA-256 in the manifest matches the file body digest.
        let bytes = match fs::read(&shard_path) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{}: re-read error: {e}", ref_s.path));
                continue;
            }
        };
        let (actual, _) = compute_body_checksum(&bytes);
        if actual != ref_s.sha256 {
            errors.push(
                TensorDatasetError::Integrity {
                    path: ref_s.path.clone(),
                    manifest: ref_s.sha256.clone(),
                    actual,
                }
                .to_string(),
            );
        }
        total_samples += report.n_samples;
    }

    if found != manifest.shards.len() {
        errors.push(
            TensorDatasetError::MissingShards {
                declared: manifest.shards.len(),
                found,
            }
            .to_string(),
        );
    }
    if total_samples != manifest.n_samples_total {
        errors.push(format!(
            "total samples mismatch: manifest={}, summed={}",
            manifest.n_samples_total, total_samples
        ));
    }

    if errors.is_empty() {
        ValidationReport::ok(total_samples)
    } else {
        ValidationReport::fail(errors)
    }
}

// ---------------------------------------------------------------------------
// Internal byte helpers
// ---------------------------------------------------------------------------

fn write_u64(dst: &mut [u8], v: u64) {
    dst.copy_from_slice(&v.to_le_bytes());
}
fn write_u32(dst: &mut [u8], v: u32) {
    dst.copy_from_slice(&v.to_le_bytes());
}

/// Encode a semver string `MAJOR.MINOR.PATCH` into 6 bytes (u16 each, LE).
/// Non-conforming strings are rejected.
fn write_semver(dst: &mut [u8], version: &str) {
    let parts: Vec<&str> = version.split('.').collect();
    let (maj, min, patch) = match parts.as_slice() {
        [maj, min, patch] => (maj, min, patch),
        _ => {
            // Fall back to zeros; the reader will reject the version anyway.
            dst.copy_from_slice(&[0u8; 6]);
            return;
        }
    };
    let maj: u16 = maj.parse().unwrap_or(0);
    let min: u16 = min.parse().unwrap_or(0);
    let patch: u16 = patch.parse().unwrap_or(0);
    dst[0..2].copy_from_slice(&maj.to_le_bytes());
    dst[2..4].copy_from_slice(&min.to_le_bytes());
    dst[4..6].copy_from_slice(&patch.to_le_bytes());
}

fn read_u64(src: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(src);
    u64::from_le_bytes(buf)
}
fn read_u32(src: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(src);
    u32::from_le_bytes(buf)
}

fn read_semver(src: &[u8]) -> Result<String> {
    let read_u16 = |s: &[u8]| {
        let mut b = [0u8; 2];
        b.copy_from_slice(s);
        u16::from_le_bytes(b)
    };
    Ok(format!(
        "{}.{}.{}",
        read_u16(&src[0..2]),
        read_u16(&src[2..4]),
        read_u16(&src[4..6])
    ))
}

/// FTDS v1.x: only major == 1 is readable. Minor/patch may differ.
fn is_compatible_version(v: &str) -> bool {
    v.split('.').next() == Some("1")
}

fn read_f64_block(bytes: &[u8]) -> Result<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(TensorDatasetError::Truncated {
            needed: bytes.len() + (8 - bytes.len() % 8) % 8,
            have: bytes.len(),
            context: "f64 block alignment",
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    let mut i = 0;
    while i < bytes.len() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i..i + 8]);
        out.push(f64::from_le_bytes(buf));
        i += 8;
    }
    Ok(out)
}

fn read_u64_block(bytes: &[u8]) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(TensorDatasetError::Truncated {
            needed: bytes.len() + (8 - bytes.len() % 8) % 8,
            have: bytes.len(),
            context: "u64 block alignment",
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    let mut i = 0;
    while i < bytes.len() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i..i + 8]);
        out.push(u64::from_le_bytes(buf));
        i += 8;
    }
    Ok(out)
}

/// SHA-256 over everything except the trailing 32-byte digest.
/// Returns `(hex_checksum, body_length)`.
fn compute_body_checksum(bytes: &[u8]) -> (String, usize) {
    if bytes.len() < 32 {
        // Degenerate; hash everything, no body to carve out.
        let mut h = Sha256::new();
        h.update(bytes);
        return (hex_encode(&h.finalize()), bytes.len());
    }
    let body_len = bytes.len() - 32;
    let mut h = Sha256::new();
    h.update(&bytes[..body_len]);
    (hex_encode(&h.finalize()), body_len)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn check_finite(vals: &[f64], location: &'static str) -> Result<()> {
    for (i, v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(TensorDatasetError::NonFinite { location, index: i });
        }
    }
    Ok(())
}

fn current_utc_iso8601() -> String {
    // Minimal UTC timestamp without pulling chrono into the public API path.
    // Format: YYYY-MM-DDTHH:MM:SSZ (best-effort via SystemTime → days/hms).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_iso8601_utc(now)
}

fn format_iso8601_utc(epoch_secs: u64) -> String {
    // Civil date algorithm (Howard Hinnant, public domain). Verified by tests.
    let secs_per_day = 86_400u64;
    let days = (epoch_secs / secs_per_day) as i64;
    let sod = epoch_secs % secs_per_day;
    let hour = sod / 3600;
    let minute = (sod % 3600) / 60;
    let second = sod % 60;

    // days since 1970-01-01 → civil (Y, M, D). 719468 = days from 0000-03-01 to 1970-01-01.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, m, d, hour, minute, second
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: usize, inputs: Vec<f64>, targets: Vec<f64>) -> TensorSample {
        TensorSample {
            sample_id: id,
            inputs,
            targets,
            timeseries: None,
        }
    }

    #[test]
    fn semver_round_trip() {
        let mut buf = [0u8; 6];
        write_semver(&mut buf, TENSOR_DATASET_SCHEMA_VERSION);
        let v = read_semver(&buf).unwrap();
        assert_eq!(v, TENSOR_DATASET_SCHEMA_VERSION);
    }

    #[test]
    fn is_compatible_v1_only() {
        assert!(is_compatible_version("1.0.0"));
        assert!(is_compatible_version("1.5.3"));
        assert!(!is_compatible_version("2.0.0"));
        assert!(!is_compatible_version("0.9.0"));
    }

    #[test]
    fn dtype_byte_size() {
        assert_eq!(TensorDType::F32.byte_size(), 4);
        assert_eq!(TensorDType::F64.byte_size(), 8);
        assert_eq!(TensorDType::F64.as_byte(), 1);
        assert_eq!(TensorDType::from_byte(1).unwrap(), TensorDType::F64);
        assert!(TensorDType::from_byte(9).is_err());
    }

    #[test]
    fn iso8601_known_epoch() {
        // 1970-01-01 00:00:00 UTC
        assert_eq!(format_iso8601_utc(0), "1970-01-01T00:00:00Z");
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(format_iso8601_utc(1704067200), "2024-01-01T00:00:00Z");
        // 2026-07-26 12:34:56 UTC = 1785069296
        assert_eq!(format_iso8601_utc(1785069296), "2026-07-26T12:34:56Z");
    }

    #[test]
    fn hex_encoding_lowercase() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0xab]), "00ffab");
    }

    #[test]
    fn writer_rejects_arity_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorDatasetWriter::new(
            dir.path(),
            TensorFeatureSpec::defaults(), // 3 features
            TensorSample::DEFAULT_TARGET_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            0,
            2,
        )
        .unwrap();

        let err = writer
            .push(sample(0, vec![1.0, 2.0], vec![1.0; 6]))
            .unwrap_err();
        assert!(matches!(
            err,
            TensorDatasetError::Arity {
                kind: "input features",
                expected: 3,
                got: 2
            }
        ));

        let err = writer
            .push(sample(0, vec![1.0, 2.0, 3.0], vec![1.0; 3]))
            .unwrap_err();
        assert!(matches!(
            err,
            TensorDatasetError::Arity {
                kind: "targets",
                expected: 6,
                got: 3
            }
        ));
    }

    #[test]
    fn writer_rejects_non_finite() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorDatasetWriter::new(
            dir.path(),
            TensorFeatureSpec::defaults(),
            TensorSample::DEFAULT_TARGET_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            0,
            4,
        )
        .unwrap();
        let err = writer
            .push(sample(0, vec![f64::NAN, 2.0, 3.0], vec![1.0; 6]))
            .unwrap_err();
        assert!(matches!(
            err,
            TensorDatasetError::NonFinite {
                location: "inputs",
                index: 0
            }
        ));
    }

    #[test]
    fn empty_writer_finish_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorDatasetWriter::new(
            dir.path(),
            TensorFeatureSpec::defaults(),
            TensorSample::DEFAULT_TARGET_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            0,
            4,
        )
        .unwrap();
        assert!(matches!(writer.finish(), Err(TensorDatasetError::Empty)));
    }

    #[test]
    fn formatter_rejects_failed_simulation() {
        let out = SimulationOutput::failure(7, "boom".to_string());
        let err = TensorSample::from_simulation_output(
            &out,
            &[1.0, 2.0, 3.0],
            &TensorFeatureSpec::defaults()
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            TensorDatasetError::FailedSimulation { sample_id: 7, .. }
        ));
    }
}
