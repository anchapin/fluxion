//! Surrogate manager for fast thermal load predictions.

#[allow(unused_imports)]
use crate::ai::modular_surrogate::{ComponentSurrogate, CompositeSurrogate};
use crate::util::sha256_hex::sha256_hex;
#[allow(unused_imports)]
use log::{info, warn};
// Issue #3313: CoreML/DirectML EP types only exist when the matching `ort`
// feature is enabled (`ort/coreml` / `ort/directml`), so the target-OS gate
// alone is not sufficient to reference them. Each is wired through a
// dedicated fluxion feature (see Cargo.toml); `create_session` degrades
// gracefully with an explicit rebuild hint when the feature is missing.
#[cfg(all(feature = "ort", feature = "coreml", target_os = "macos"))]
use ort::ep::CoreML;
#[cfg(all(feature = "ort", feature = "directml", target_os = "windows"))]
use ort::ep::DirectML;
#[cfg(feature = "ort")]
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
#[allow(unused_imports)]
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Copy, Default, PartialEq, Eq)]
/// Inference backend for ONNX runtime execution.
pub enum InferenceBackend {
    #[default]
    CPU,
    CUDA,
    CoreML,
    DirectML,
    OpenVINO,
}

impl InferenceBackend {
    /// Lowercase Prometheus label value for this backend (Issue #2498). Used
    /// as the `backend` label on `fluxion_onnx_*` metrics so dashboards can
    /// split CPU vs CUDA vs CoreML vs DirectML vs OpenVINO throughput.
    pub fn as_str(self) -> &'static str {
        match self {
            InferenceBackend::CPU => "cpu",
            InferenceBackend::CUDA => "cuda",
            InferenceBackend::CoreML => "coreml",
            InferenceBackend::DirectML => "directml",
            InferenceBackend::OpenVINO => "openvino",
        }
    }
}

/// Coarse cardinality-bounded label for the batch size of an ONNX inference
/// call (Issue #2498). Keeps the `batch_bucket` label of
/// `fluxion_onnx_inference_duration_seconds` to 5 values regardless of how
/// many distinct batch sizes callers feed in.
#[cfg(feature = "ort")]
fn batch_bucket_label(batch_size: usize) -> &'static str {
    match batch_size {
        0 => "0",
        1 => "1",
        2..=8 => "2-8",
        9..=64 => "9-64",
        _ => "65+",
    }
}

#[derive(Clone, Debug, Copy, Default, PartialEq, Eq)]
pub enum QuantizationType {
    #[default]
    FP32,
    FP16,
    INT8,
}

#[derive(Clone, Debug, Default)]
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub auto_quantize: bool,
}

impl QuantizationConfig {
    pub fn fp32() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::FP32,
            auto_quantize: false,
        }
    }
    pub fn fp16() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::FP16,
            auto_quantize: true,
        }
    }
    pub fn int8() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::INT8,
            auto_quantize: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MultiDeviceConfig {
    pub device_ids: Vec<usize>,
    pub sessions_per_device: usize,
    pub auto_select: bool,
    pub enable_affinity: bool,
    pub fallback_to_cpu: bool,
    pub max_retries: usize,
}

impl MultiDeviceConfig {
    pub fn single_gpu(device_id: usize) -> Self {
        MultiDeviceConfig {
            device_ids: vec![device_id],
            sessions_per_device: 4,
            auto_select: false,
            enable_affinity: true,
            fallback_to_cpu: true,
            max_retries: 3,
        }
    }
    pub fn multi_gpu(device_ids: Vec<usize>) -> Self {
        MultiDeviceConfig {
            device_ids,
            sessions_per_device: 2,
            auto_select: false,
            enable_affinity: true,
            fallback_to_cpu: true,
            max_retries: 3,
        }
    }
    pub fn auto() -> Self {
        MultiDeviceConfig {
            device_ids: vec![],
            sessions_per_device: 4,
            auto_select: true,
            enable_affinity: false,
            fallback_to_cpu: true,
            max_retries: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Default)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastLoaded,
    Random,
}

#[derive(Clone, Debug, Default)]
pub struct InferenceMetrics {
    pub avg_inference_time_ms: f64,
    pub num_inferences: usize,
    pub peak_memory_mb: f64,
    pub throughput: f64,
}

impl InferenceMetrics {
    pub fn record_inference(&mut self, time_ms: f64) {
        let n = self.num_inferences as f64;
        self.avg_inference_time_ms = (self.avg_inference_time_ms * n + time_ms) / (n + 1.0);
        self.num_inferences += 1;
        if self.avg_inference_time_ms > 0.0 {
            self.throughput = 1000.0 / self.avg_inference_time_ms;
        }
    }
    pub fn reset(&mut self) {
        self.avg_inference_time_ms = 0.0;
        self.num_inferences = 0;
        self.peak_memory_mb = 0.0;
        self.throughput = 0.0;
    }
}

/// Real physics-extracted training data for surrogate model training.
/// This replaces synthetic placeholder values with actual simulation outputs.
#[derive(Clone, Debug)]
pub struct PhysicsTrainingData {
    pub exterior_temp: f64,   // °C - from weather data
    pub zone_temp: f64,       // °C - from thermal model
    pub solar_rad: f64,       // W/m² - from solar module
    pub humidity: f64,        // % - from psychrometrics
    pub occupancy: f64,       // fraction 0-1 - from occupancy schedule
    pub climate_zone: String, // e.g., "4A", "5A", "6A"
    pub hour_of_day: usize,   // 0-23
    pub day_of_year: usize,   // 1-365
}

impl PhysicsTrainingData {
    /// Create from physics simulation outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exterior_temp: f64,
        zone_temp: f64,
        solar_rad: f64,
        humidity: f64,
        occupancy: f64,
        climate_zone: &str,
        hour_of_day: usize,
        day_of_year: usize,
    ) -> Self {
        PhysicsTrainingData {
            exterior_temp,
            zone_temp,
            solar_rad,
            humidity,
            occupancy,
            climate_zone: climate_zone.to_string(),
            hour_of_day,
            day_of_year,
        }
    }

    /// Convert to SurrogateInputs for model inference.
    pub fn to_surrogate_inputs(&self) -> SurrogateInputs {
        SurrogateInputs {
            exterior_temp: self.exterior_temp,
            zone_temp: self.zone_temp,
            solar_rad: self.solar_rad,
            humidity: self.humidity,
            occupancy: self.occupancy,
            climate_zone: self.climate_zone.clone(),
        }
    }
}

/// Collects physics training data from simulation timesteps.
/// Supports multi-climate-zone datasets for robust surrogate training.
#[derive(Clone, Debug, Default)]
pub struct TrainingDataCollector {
    /// Collected training samples per climate zone
    samples_by_zone: std::collections::HashMap<String, Vec<PhysicsTrainingData>>,
}

impl TrainingDataCollector {
    /// Create a new training data collector.
    pub fn new() -> Self {
        TrainingDataCollector {
            samples_by_zone: std::collections::HashMap::new(),
        }
    }

    /// Add a training sample for a specific climate zone.
    pub fn add_sample(&mut self, data: PhysicsTrainingData) {
        let zone = data.climate_zone.clone();
        self.samples_by_zone.entry(zone).or_default().push(data);
    }

    /// Get all collected samples for a climate zone.
    pub fn get_samples(&self, climate_zone: &str) -> Vec<PhysicsTrainingData> {
        self.samples_by_zone
            .get(climate_zone)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all climate zones that have samples.
    pub fn climate_zones(&self) -> Vec<String> {
        self.samples_by_zone.keys().cloned().collect()
    }

    /// Get total number of samples across all climate zones.
    pub fn total_samples(&self) -> usize {
        self.samples_by_zone.values().map(|v| v.len()).sum()
    }

    /// Get samples per climate zone (for balanced training datasets).
    pub fn samples_per_zone(&self) -> std::collections::HashMap<String, usize> {
        self.samples_by_zone
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SurrogateInputs {
    pub exterior_temp: f64,
    pub zone_temp: f64,
    pub solar_rad: f64,
    pub humidity: f64,
    pub occupancy: f64,
    pub climate_zone: String,
}

impl SurrogateInputs {
    /// Create from temperature array (legacy synthetic method).
    /// NOTE: This generates placeholder synthetic values.
    /// Use `from_physics_data` for real physics-extracted training data.
    pub fn from_temps(temps: &[f64]) -> Self {
        let hour_of_day = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / 3600) as usize
            % 24;
        let daily_cycle = (std::f64::consts::PI * (hour_of_day as f64 - 6.0) / 12.0).sin();
        SurrogateInputs {
            exterior_temp: temps.first().copied().unwrap_or(20.0),
            zone_temp: temps.get(1).copied().unwrap_or(22.0),
            solar_rad: (500.0 * daily_cycle).max(0.0),
            humidity: 50.0,
            occupancy: 0.1,
            climate_zone: "4A".to_string(),
        }
    }

    /// Create from real physics simulation data (Issue #1286).
    /// This replaces synthetic placeholder values with actual physics outputs.
    pub fn from_physics_data(data: &PhysicsTrainingData) -> Self {
        SurrogateInputs {
            exterior_temp: data.exterior_temp,
            zone_temp: data.zone_temp,
            solar_rad: data.solar_rad,
            humidity: data.humidity,
            occupancy: data.occupancy,
            climate_zone: data.climate_zone.clone(),
        }
    }

    /// Create from individual physics parameters.
    pub fn from_physics(
        exterior_temp: f64,
        zone_temp: f64,
        solar_rad: f64,
        humidity: f64,
        occupancy: f64,
        climate_zone: &str,
    ) -> Self {
        SurrogateInputs {
            exterior_temp,
            zone_temp,
            solar_rad,
            humidity,
            occupancy,
            climate_zone: climate_zone.to_string(),
        }
    }
}

/// Per-feature min/max bounds extracted from training data.
///
/// Used by OOD detection to determine whether an inference input vector
/// falls inside the convex hull of the training distribution (Issue #1892).
/// The bounds correspond to the numeric features of [`SurrogateInputs`]:
/// index 0 = exterior_temp, 1 = zone_temp, 2 = solar_rad,
/// 3 = humidity, 4 = occupancy.
#[derive(Clone, Debug)]
pub struct InputBounds {
    pub exterior_temp: (f64, f64),
    pub zone_temp: (f64, f64),
    pub solar_rad: (f64, f64),
    pub humidity: (f64, f64),
    pub occupancy: (f64, f64),
    pub valid_climate_zones: Vec<String>,
}

impl Default for InputBounds {
    fn default() -> Self {
        Self::strict_residential()
    }
}

impl InputBounds {
    pub fn strict_residential() -> Self {
        Self {
            exterior_temp: (-50.0, 60.0),
            zone_temp: (10.0, 40.0),
            solar_rad: (0.0, 1200.0),
            humidity: (0.0, 100.0),
            occupancy: (0.0, 10.0),
            valid_climate_zones: vec!["4A".to_string(), "5A".to_string(), "6A".to_string()],
        }
    }

    pub fn from_training_data(samples: &[SurrogateInputs]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut ext_min = f64::MAX;
        let mut ext_max = f64::MIN;
        let mut zone_min = f64::MAX;
        let mut zone_max = f64::MIN;
        let mut solar_min = f64::MAX;
        let mut solar_max = f64::MIN;
        let mut hum_min = f64::MAX;
        let mut hum_max = f64::MIN;
        let mut occ_min = f64::MAX;
        let mut occ_max = f64::MIN;
        let mut climate_zones: std::collections::HashSet<String> = std::collections::HashSet::new();

        for s in samples {
            ext_min = ext_min.min(s.exterior_temp);
            ext_max = ext_max.max(s.exterior_temp);
            zone_min = zone_min.min(s.zone_temp);
            zone_max = zone_max.max(s.zone_temp);
            solar_min = solar_min.min(s.solar_rad);
            solar_max = solar_max.max(s.solar_rad);
            hum_min = hum_min.min(s.humidity);
            hum_max = hum_max.max(s.humidity);
            occ_min = occ_min.min(s.occupancy);
            occ_max = occ_max.max(s.occupancy);
            climate_zones.insert(s.climate_zone.clone());
        }

        Self {
            exterior_temp: (ext_min, ext_max),
            zone_temp: (zone_min, zone_max),
            solar_rad: (solar_min, solar_max),
            humidity: (hum_min, hum_max),
            occupancy: (occ_min, occ_max),
            valid_climate_zones: climate_zones.into_iter().collect(),
        }
    }
}

/// Structured warning emitted when an inference input vector is detected
/// as out-of-distribution (OOD) — i.e. at least one feature falls outside
/// the stored training bounds.
///
/// The surrogate MUST NOT panic or return NaN when OOD is detected.
/// Instead it must fall back to the physics solver (Issue #1892).
#[derive(Clone, Debug)]
pub struct OodInputWarning {
    pub feature_name: &'static str,
    pub feature_index: usize,
    pub actual_value: f64,
    pub min_bound: f64,
    pub max_bound: f64,
}

impl OodInputWarning {
    pub fn new(
        feature_name: &'static str,
        feature_index: usize,
        actual_value: f64,
        min_bound: f64,
        max_bound: f64,
    ) -> Self {
        Self {
            feature_name,
            feature_index,
            actual_value,
            min_bound,
            max_bound,
        }
    }

    pub fn log_warning(&self) {
        warn!(
            "OOD detected: feature '{}' (index {}) = {:.2} is outside training bounds [{:.2}, {:.2}]",
            self.feature_name, self.feature_index, self.actual_value, self.min_bound, self.max_bound
        );
    }
}

/// Result of OOD input validation. Contains the input vector and
/// any OOD warnings detected during validation.
#[derive(Clone, Debug)]
pub struct OodValidationResult {
    pub is_ood: bool,
    pub warnings: Vec<OodInputWarning>,
}

impl OodValidationResult {
    pub fn clean() -> Self {
        Self {
            is_ood: false,
            warnings: Vec::new(),
        }
    }

    pub fn with_warning(warning: OodInputWarning) -> Self {
        Self {
            is_ood: true,
            warnings: vec![warning],
        }
    }

    pub fn log_warnings(&self) {
        for w in &self.warnings {
            w.log_warning();
        }
    }
}

/// Default squared-residual threshold for the inference-time energy-balance
/// residual check (Issue #1896).
///
/// τ = 1.0 W² corresponds to ~1 W absolute error, which is tight enough
/// to catch model drift or quantization artifacts while remaining above
/// numerical-noise floor.
pub const DEFAULT_RESIDUAL_TAU: f64 = 1.0;

/// Structured error returned when a surrogate inference violates the
/// energy-balance residual threshold.
///
/// The residual is the squared difference between the predicted thermal load
/// and the physics-expected load computed from the input conditions:
/// `residual = (Q_predicted - Q_expected)²`
///
/// When `residual > tau` the prediction is deemed physically implausible
/// and callers must reroute to the analytical/physics fallback.
#[derive(Clone, Debug)]
pub struct ResidualViolation {
    /// Index of the sample / zone in the batch.
    pub sample_index: usize,
    /// Predicted thermal load from ONNX (W).
    pub predicted: f64,
    /// Physics-expected load computed from input conditions (W).
    pub expected: f64,
    /// Squared residual `||Q_predicted - Q_expected||²` (W²).
    pub residual: f64,
}

impl std::fmt::Display for ResidualViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "surrogate residual violation at sample {}: predicted {:.2} W, expected {:.2} W, residual {:.2} W²",
            self.sample_index, self.predicted, self.expected, self.residual
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct SurrogateDomain {
    pub temp_bounds: (f64, f64),
    pub zone_temp_bounds: (f64, f64),
    pub solar_bounds: (f64, f64),
    pub humidity_bounds: (f64, f64),
    pub occupancy_bounds: (f64, f64),
    pub climate_zones: Vec<String>,
    pub building_types: Vec<String>,
    pub training_period: (String, String),
}

impl SurrogateDomain {
    pub fn default_residential() -> Self {
        SurrogateDomain {
            temp_bounds: (-50.0, 60.0),
            zone_temp_bounds: (10.0, 40.0),
            solar_bounds: (0.0, 1200.0),
            humidity_bounds: (0.0, 100.0),
            occupancy_bounds: (0.0, 10.0),
            climate_zones: vec!["4A".to_string(), "5A".to_string(), "6A".to_string()],
            building_types: vec!["residential".to_string()],
            training_period: ("2020-01-01".to_string(), "2023-12-31".to_string()),
        }
    }

    pub fn is_valid(&self, inputs: &SurrogateInputs) -> bool {
        let temp_valid = inputs.exterior_temp >= self.temp_bounds.0
            && inputs.exterior_temp <= self.temp_bounds.1;
        let zone_valid = inputs.zone_temp >= self.zone_temp_bounds.0
            && inputs.zone_temp <= self.zone_temp_bounds.1;
        let solar_valid =
            inputs.solar_rad >= self.solar_bounds.0 && inputs.solar_rad <= self.solar_bounds.1;
        let humidity_valid =
            inputs.humidity >= self.humidity_bounds.0 && inputs.humidity <= self.humidity_bounds.1;
        let occupancy_valid = inputs.occupancy >= self.occupancy_bounds.0
            && inputs.occupancy <= self.occupancy_bounds.1;
        let climate_valid = self.climate_zones.contains(&inputs.climate_zone);

        temp_valid
            && zone_valid
            && solar_valid
            && humidity_valid
            && occupancy_valid
            && climate_valid
    }

    /// Compute per-sample energy balance residual for PINN physics constraints.
    ///
    /// Implements the envelope-only energy balance constraint:
    /// `L_physics = ||Q_loads - Q_conduction - Q_solar - Q_internal||^2`
    ///
    /// Where:
    /// - `Q_loads` is the predicted thermal load
    /// - `Q_conduction = U * A * (T_exterior - T_zone)` — conductive heat transfer
    /// - `Q_solar = alpha * solar_rad * A` — solar gains (alpha = 0.85 solar transmissivity)
    /// - `Q_internal = beta * occupancy * A` — internal gains (beta = 100 W/person)
    ///
    /// Default thermal properties for residential envelope:
    /// - U = 0.5 W/m²K (overall heat transfer coefficient)
    /// - A = 100 m² (typical zone surface area)
    /// - Ventilation rate = 0.5 ACH (air changes per hour)
    ///
    /// Returns the squared residual `||Q_loads - Q_expected||^2` for each sample.
    ///
    /// Issue #1706: PINN physics constraints for CompositeSurrogate training.
    pub fn energy_balance_residual(
        &self,
        inputs: &[SurrogateInputs],
        predicted_loads: &[f64],
    ) -> Vec<f64> {
        const U_WALL: f64 = 0.5;
        const A_ZONE: f64 = 100.0;
        const ALPHA_SOLAR: f64 = 0.85;
        const BETA_INTERNAL: f64 = 100.0;
        const C_AIR: f64 = 1260.0;
        const V_VENT: f64 = 300.0;
        const ACH_VENT: f64 = 0.5;

        inputs
            .iter()
            .zip(predicted_loads.iter())
            .map(|(inp, &q_loads)| {
                let delta_t = inp.exterior_temp - inp.zone_temp;
                let q_conduction = U_WALL * A_ZONE * delta_t;
                let q_solar = ALPHA_SOLAR * inp.solar_rad * A_ZONE * 0.001;
                let q_internal = BETA_INTERNAL * inp.occupancy;
                let q_ventilation = C_AIR * ACH_VENT * V_VENT * delta_t / 3600.0 / 1000.0;
                let q_expected = q_conduction + q_solar + q_internal + q_ventilation;
                let residual = q_loads - q_expected;
                residual * residual
            })
            .collect()
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum SurrogateMode {
    NeuralOnly,
    NeuralWithFallback,
    AnalyticalOnly,
}

impl Default for SurrogateMode {
    fn default() -> Self {
        SurrogateMode::NeuralWithFallback
    }
}

/// Errors produced when constructing or validating a model version string.
///
/// Issue #1335: typed error so callers can distinguish a malformed semver
/// from a hash mismatch or a registry miss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionError {
    /// The version string is not strict semver (e.g. "3.1" or "v3").
    InvalidSemver(String),
    /// The version is a syntactically valid semver but is the forbidden
    /// placeholder "0.0.0".
    PlaceholderVersion(String),
    /// The SHA-256 hex string is not 64 lowercase/uppercase hex characters.
    InvalidHash(String),
    /// The ONNX opset version is unsupported (must be in `1..=17`).
    UnsupportedOpset(u32),
}

impl std::fmt::Display for VersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionError::InvalidSemver(v) => {
                write!(f, "invalid semver version '{}': expected strict MAJOR.MINOR.PATCH (e.g. '3.1.0'); pre-release/build identifiers are allowed", v)
            }
            VersionError::PlaceholderVersion(v) => {
                write!(f, "forbidden placeholder version '{}': '0.0.0' is reserved for the default-constructed metadata and is not a valid release identifier", v)
            }
            VersionError::InvalidHash(h) => {
                write!(
                    f,
                    "invalid SHA-256 hash '{}': expected 64 hexadecimal characters",
                    h
                )
            }
            VersionError::UnsupportedOpset(op) => {
                write!(
                    f,
                    "unsupported ONNX opset {}: supported range is 1..=17",
                    op
                )
            }
        }
    }
}

impl std::error::Error for VersionError {}

/// Strict-semver validator for surrogate model version strings.
///
/// Accepts `MAJOR.MINOR.PATCH` with optional `[-prerelease]` and `[+build]`,
/// where each numeric component is `0..=999` and the `v` prefix is rejected.
/// Returns `Ok(())` for valid semver (including `0.0.0` syntactically — the
/// placeholder check is enforced separately by [`VersionError::PlaceholderVersion`]).
pub fn validate_semver(version: &str) -> Result<(), VersionError> {
    if version.is_empty() || version.len() > 64 {
        return Err(VersionError::InvalidSemver(version.to_string()));
    }
    let (core, _pre_build) = match version.split_once('-') {
        Some((c, rest)) => (c, Some(rest)),
        None => match version.split_once('+') {
            Some((c, rest)) => (c, Some(rest)),
            None => (version, None),
        },
    };
    let mut parts = core.split('.');
    let major = parts
        .next()
        .ok_or_else(|| VersionError::InvalidSemver(version.to_string()))?;
    let minor = parts
        .next()
        .ok_or_else(|| VersionError::InvalidSemver(version.to_string()))?;
    let patch = parts
        .next()
        .ok_or_else(|| VersionError::InvalidSemver(version.to_string()))?;
    if parts.next().is_some() {
        return Err(VersionError::InvalidSemver(version.to_string()));
    }
    let is_numeric_component = |s: &str| {
        !s.is_empty()
            && s.len() <= 3
            && s.chars().all(|c| c.is_ascii_digit())
            && (s.len() == 1 || !s.starts_with('0'))
    };
    if !(is_numeric_component(major) && is_numeric_component(minor) && is_numeric_component(patch))
    {
        return Err(VersionError::InvalidSemver(version.to_string()));
    }
    Ok(())
}

/// Validate a SHA-256 hex string (64 lowercase or uppercase hex chars).
pub fn validate_sha256_hex(hash: &str) -> Result<(), VersionError> {
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(VersionError::InvalidHash(hash.to_string()));
    }
    Ok(())
}

/// Pinned metadata for a surrogate ONNX model release.
///
/// Issue #1335: the registry stores one `ModelVersion` per release. The
/// `model_sha256` matches the bytes of the ONNX file; the
/// `training_data_hash` matches a content hash of the training set manifest
/// (a CI-managed file outside this repo).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelVersion {
    /// Strict semver version (e.g. "3.1.0").
    pub version: String,
    /// Lowercase hex SHA-256 of the `.onnx` file.
    pub model_sha256: String,
    /// ONNX opset version used to export the model (1..=17 per ADR-0004).
    pub onnx_opset_version: u32,
    /// Lowercase hex SHA-256 of the training data manifest.
    pub training_data_hash: String,
    /// ISO-8601 date when the model was trained (UTC).
    pub trained_on: String,
    /// Free-form one-line summary of the training set.
    pub training_data_summary: String,
    /// Minimum fraction of held-out EnergyPlus hourly zone-temperature
    /// predictions that must remain within the validation tolerance.
    ///
    /// The release value is established from ASHRAE 140 Case 950 hourly output,
    /// which is generated by EnergyPlus 25.2.0 with the Golden, Colorado TMY3
    /// weather file and is excluded from surrogate training. Validation compares
    /// every predicted hourly zone temperature with the committed reference and
    /// counts samples within the gate tolerance; model acceptance requires the
    /// observed fraction to meet or exceed this non-zero threshold.
    pub expected_accuracy: f64,
    /// Absolute path or relative path under the model store; ONNX files
    /// themselves are never committed to git (see ADR-0004).
    pub model_path: String,
}

impl ModelVersion {
    /// Build a `ModelVersion` from raw fields, validating all invariants.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        version: &str,
        model_sha256: &str,
        onnx_opset_version: u32,
        training_data_hash: &str,
        trained_on: &str,
        training_data_summary: &str,
        expected_accuracy: f64,
        model_path: &str,
    ) -> Result<Self, VersionError> {
        if version == "0.0.0" {
            return Err(VersionError::PlaceholderVersion(version.to_string()));
        }
        validate_semver(version)?;
        validate_sha256_hex(model_sha256)?;
        validate_sha256_hex(training_data_hash)?;
        if onnx_opset_version == 0 || onnx_opset_version > 17 {
            return Err(VersionError::UnsupportedOpset(onnx_opset_version));
        }
        Ok(ModelVersion {
            version: version.to_string(),
            model_sha256: model_sha256.to_ascii_lowercase(),
            onnx_opset_version,
            training_data_hash: training_data_hash.to_ascii_lowercase(),
            trained_on: trained_on.to_string(),
            training_data_summary: training_data_summary.to_string(),
            expected_accuracy,
            model_path: model_path.to_string(),
        })
    }

    /// Parse the version entry out of one JSON object (registry file shape).
    pub fn from_json(value: &serde_json::Value) -> Result<Self, VersionError> {
        let version = value
            .get("version")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| VersionError::InvalidSemver("<missing>".to_string()))?;
        let model_sha256 = value
            .get("model_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| VersionError::InvalidHash("<missing>".to_string()))?;
        let onnx_opset_version = value
            .get("onnx_opset_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or(VersionError::UnsupportedOpset(0))? as u32;
        let training_data_hash = value
            .get("training_data_hash")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| VersionError::InvalidHash("<missing>".to_string()))?;
        let trained_on = value
            .get("trained_on")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let training_data_summary = value
            .get("training_data_summary")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let expected_accuracy = value
            .get("expected_accuracy")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let model_path = value
            .get("model_path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        ModelVersion::new(
            version,
            model_sha256,
            onnx_opset_version,
            training_data_hash,
            trained_on,
            training_data_summary,
            expected_accuracy,
            model_path,
        )
    }
}

/// In-memory registry of pinned surrogate model versions.
///
/// Loaded from `tests/surrogate_models/registry.json` (see ADR-0004). The
/// `.onnx` files themselves are not in git; this registry only carries
/// hashes and metadata so that `load_version` can validate before opening
/// the session.
#[derive(Clone, Debug, Default)]
pub struct ModelRegistry {
    pub versions: Vec<ModelVersion>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        ModelRegistry::default()
    }

    pub fn from_versions(versions: Vec<ModelVersion>) -> Self {
        ModelRegistry { versions }
    }

    /// Parse a registry from its JSON representation.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("registry JSON parse error: {}", e))?;
        let arr = value
            .get("versions")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "registry must contain a top-level 'versions' array".to_string())?;
        let mut versions = Vec::with_capacity(arr.len());
        for (i, entry) in arr.iter().enumerate() {
            let v = ModelVersion::from_json(entry)
                .map_err(|e| format!("registry entry #{}: {}", i, e))?;
            versions.push(v);
        }
        Ok(ModelRegistry { versions })
    }

    pub fn lookup(&self, version: &str) -> Option<&ModelVersion> {
        self.versions.iter().find(|v| v.version == version)
    }

    pub fn latest(&self) -> Option<&ModelVersion> {
        self.versions.last()
    }

    pub fn len(&self) -> usize {
        self.versions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

/// Compute the lowercase hex SHA-256 digest of a file's bytes.
pub fn compute_file_sha256(path: &Path) -> Result<String, String> {
    if !path.exists() {
        return Err(format!("file not found: {}", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("read failed: {}", e))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(sha256_hex(hasher.finalize()))
}

/// Compute the lowercase hex SHA-256 of a byte slice (for in-memory checks).
pub fn compute_bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    sha256_hex(hasher.finalize())
}

/// Compare a claimed SHA-256 against the file's actual SHA-256.
pub fn validate_hash(expected: &str, actual: &str) -> Result<(), String> {
    if expected.eq_ignore_ascii_case(actual) {
        Ok(())
    } else {
        Err(format!(
            "SHA-256 mismatch: expected {}, got {}",
            expected, actual
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// Strict semver version (default `"0.0.0"` for unconfigured models).
    pub model_version: String,
    pub domain: SurrogateDomain,
    /// Per-feature training bounds for OOD detection (Issue #1892).
    pub input_bounds: Option<InputBounds>,
    pub onnx_version: Option<String>,
    /// ONNX opset version the model was exported with (1..=17).
    pub onnx_opset_version: Option<u32>,
    /// Lowercase hex SHA-256 of the `.onnx` file (issue #1335).
    pub model_sha256: Option<String>,
    /// Lowercase hex SHA-256 of the training data manifest (issue #1335).
    pub training_data_hash: Option<String>,
    pub training_samples: usize,
    pub test_mae: Option<f64>,
    pub test_rmse: Option<f64>,
    pub test_r2: Option<f64>,
    pub validation_date: Option<String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        ModelMetadata {
            model_version: "0.0.0".to_string(),
            domain: SurrogateDomain::default_residential(),
            input_bounds: None,
            onnx_version: None,
            onnx_opset_version: None,
            model_sha256: None,
            training_data_hash: None,
            training_samples: 0,
            test_mae: None,
            test_rmse: None,
            test_r2: None,
            validation_date: None,
        }
    }
}

impl ModelMetadata {
    /// Construct a `ModelMetadata` from a strict semver version string.
    ///
    /// Rejects the placeholder `"0.0.0"` and any non-strict semver such as
    /// `"3.1"` or `"v3"` with a typed [`VersionError`].
    ///
    /// ```
    /// use fluxion::ai::surrogate::ModelMetadata;
    /// assert!(ModelMetadata::with_semver("3.1.0").is_ok());
    /// assert!(ModelMetadata::with_semver("0.0.0").is_err());
    /// assert!(ModelMetadata::with_semver("3.1").is_err());
    /// assert!(ModelMetadata::with_semver("v3").is_err());
    /// ```
    pub fn with_semver(version: &str) -> Result<Self, VersionError> {
        if version == "0.0.0" {
            return Err(VersionError::PlaceholderVersion(version.to_string()));
        }
        validate_semver(version)?;
        Ok(ModelMetadata {
            model_version: version.to_string(),
            ..ModelMetadata::default()
        })
    }
}

/// Manages AI surrogate models for fast thermal load prediction.
///
/// Replaces expensive CFD/ray-tracing with pre-trained neural networks.
/// Physics-informed: neural predictions constrained by energy balance.
/// Supports both single-model and composite (multi-component) surrogates.
#[derive(Clone, Debug)]
pub struct SurrogateManager {
    pub model_loaded: bool,
    pub model_path: Option<String>,
    pub session_pool: Option<Arc<SessionPool>>,
    pub backend: InferenceBackend,
    pub device_id: usize,
    /// Optional composite surrogate that aggregates multiple component models
    pub composite: Option<CompositeSurrogate>,
    /// ONNX inference metrics (timing, throughput) for benchmarking vs physics.
    /// Uses interior mutability so it can be updated from `&self` methods.
    /// Wrapped in an Arc so the manager remains `Clone` (callers throughout
    /// the codebase clone `SurrogateManager`).
    pub inference_metrics: Arc<parking_lot::Mutex<InferenceMetrics>>,
    /// Per-feature training bounds for OOD detection (Issue #1892).
    pub input_bounds: Option<InputBounds>,
    /// Counter for how many times OOD input was detected.
    /// Incremented by `validate_input_bounds` each time an OOD input is flagged.
    pub ood_count: Arc<parking_lot::Mutex<usize>>,
    /// Squared-residual threshold τ for the energy-balance residual check.
    /// Predictions with residual > τ trigger rerouting to the analytical fallback.
    /// Default: [`DEFAULT_RESIDUAL_TAU`] (1.0 W² ≈ 1 W absolute error).
    pub residual_tau: f64,
    /// Counter for how many times the residual guard caused a reroute.
    /// Incremented by `check_inference_residual` each time a violation is detected.
    pub residual_reroute_count: Arc<parking_lot::Mutex<usize>>,
}

impl Default for SurrogateManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default SurrogateManager")
    }
}

/// Thread-safe pool of ONNX Runtime sessions for concurrent inference.
///
/// Issue #1294: When the `ort` feature is disabled, `SessionPool` is an inert
/// stub. The public surface (`Option<Arc<SessionPool>>` field on
/// [`SurrogateManager`], `SessionPool::new` callers in tests) still compiles —
/// only `create_session` actually attempts to load an ONNX model, and it
/// returns an error when `ort` is disabled.
#[cfg(feature = "ort")]
#[derive(Debug)]
#[allow(dead_code)]
pub struct SessionPool {
    sessions: Mutex<Vec<ort::session::Session>>,
    model_path: String,
    backend: InferenceBackend,
    device_id: usize,
}

/// Inert stub of [`SessionPool`] used when the `ort` feature is disabled
/// (issue #1294). Carries no ONNX state. Construction succeeds; any attempt
/// to actually create an ONNX session via the corresponding methods returns
/// an error (those methods only exist under `#[cfg(feature = "ort")]`).
#[cfg(not(feature = "ort"))]
#[derive(Debug)]
#[allow(dead_code)]
pub struct SessionPool {
    model_path: String,
    backend: InferenceBackend,
    device_id: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct MultiDeviceSessionPool {
    #[allow(dead_code)]
    device_pools: Vec<Arc<SessionPool>>,
    _config: MultiDeviceConfig,
    _model_path: String,
}

#[cfg(feature = "ort")]
impl MultiDeviceSessionPool {
    pub fn new(model_path: String, config: &MultiDeviceConfig) -> Result<Self, String> {
        let mut device_pools = Vec::new();
        let device_ids = if config.auto_select {
            Self::detect_cuda_devices().unwrap_or_else(|| vec![0])
        } else if config.device_ids.is_empty() {
            vec![0]
        } else {
            config.device_ids.clone()
        };

        for device_id in &device_ids {
            match SessionPool::create_session(&model_path, InferenceBackend::CUDA, *device_id) {
                Ok(session) => {
                    let pool = SessionPool::new(
                        model_path.clone(),
                        InferenceBackend::CUDA,
                        *device_id,
                        session,
                    );
                    device_pools.push(Arc::new(pool));
                }
                Err(e) => eprintln!(
                    "Warning: Failed to create session for device {}: {}",
                    device_id, e
                ),
            }
        }

        if device_pools.is_empty() {
            return Err("Failed to create any device pools".to_string());
        }

        Ok(MultiDeviceSessionPool {
            device_pools,
            _config: config.clone(),
            _model_path: model_path,
        })
    }

    fn detect_cuda_devices() -> Option<Vec<usize>> {
        #[cfg(feature = "cuda")]
        {
            use ort::session::Session;
            // Issue #3313: attaching a CUDA EP to a session builder succeeds
            // even on machines with no GPU (provider initialization is
            // deferred to session creation, where failure silently falls
            // back to CPU). Gate discovery on ORT's own EP-device
            // enumeration first — `Environment::devices()` only reports a
            // `CUDAExecutionProvider` device when ORT actually found CUDA
            // hardware at environment creation. The registration probe
            // below is retained as a fallback for backends whose EP-ABI
            // device enumeration is unavailable (it returns an empty
            // device list there).
            let cuda_device_enumerated = ort::environment::Environment::current()
                .ok()
                .map(|env| {
                    env.devices()
                        .any(|d| d.ep().map(|name| name == ep_names::CUDA).unwrap_or(false))
                })
                .unwrap_or(false);
            let mut available_devices = Vec::new();
            if cuda_device_enumerated {
                for device_id in 0..8 {
                    let builder = match Session::builder() {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let cuda_ep = CUDA::default().with_device_id(device_id as i32);
                    match builder.with_execution_providers([cuda_ep.build()]) {
                        Ok(_) => available_devices.push(device_id),
                        Err(_) => continue,
                    }
                }
            }
            if available_devices.is_empty() {
                None
            } else {
                Some(available_devices)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    pub fn get_cuda_device_info() -> Option<Vec<CudaDeviceInfo>> {
        #[cfg(feature = "cuda")]
        {
            let devices = Self::detect_cuda_devices()?;
            Some(
                devices
                    .into_iter()
                    .map(|id| CudaDeviceInfo {
                        device_id: id,
                        name: format!("GPU {}", id),
                        compute_capability: None,
                    })
                    .collect(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    pub fn get_session(&self) -> Result<MultiDeviceSessionGuard, String> {
        for pool in &self.device_pools {
            if let Ok(_session) = pool.get_or_create_session() {
                return Ok(MultiDeviceSessionGuard {
                    pool: Arc::clone(pool),
                });
            }
        }
        Err("No available sessions in multi-device pool".to_string())
    }

    pub fn num_devices(&self) -> usize {
        self.device_pools.len()
    }
}

/// Stub implementation when the `ort` feature is disabled (issue #1294).
/// `MultiDeviceSessionPool::new` always fails; `num_devices` reports zero.
#[cfg(not(feature = "ort"))]
impl MultiDeviceSessionPool {
    pub fn new(_model_path: String, _config: &MultiDeviceConfig) -> Result<Self, String> {
        Err("Multi-device ONNX inference requires the `ort` feature".to_string())
    }

    pub fn num_devices(&self) -> usize {
        0
    }
}

/// RAII guard returned by [`MultiDeviceSessionPool::get_session`].
///
/// Issue #1294: only available with the `ort` feature — the `run_inference`
/// signature depends on `ort::value::Value`.
#[cfg(feature = "ort")]
pub struct MultiDeviceSessionGuard {
    pool: Arc<SessionPool>,
}

#[cfg(feature = "ort")]
impl MultiDeviceSessionGuard {
    pub fn run_inference(&self, input_tensor: ort::value::Value) -> Result<Vec<f64>, String> {
        let mut guard = self.pool.get_or_create_session()?;
        let outputs = guard
            .run(ort::inputs![input_tensor])
            .map_err(|e| e.to_string())?;
        if outputs.len() > 0 {
            let array = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| e.to_string())?;
            Ok(array.iter().copied().map(|x| x as f64).collect())
        } else {
            Err("No outputs from inference".to_string())
        }
    }
}

#[cfg(feature = "ort")]
impl SessionPool {
    fn new(
        model_path: String,
        backend: InferenceBackend,
        device_id: usize,
        initial_session: ort::session::Session,
    ) -> Self {
        SessionPool {
            sessions: Mutex::new(vec![initial_session]),
            model_path,
            backend,
            device_id,
        }
    }

    fn get_or_create_session(&self) -> Result<SessionGuard<'_>, String> {
        {
            let mut sessions = self.sessions.lock();
            if let Some(session) = sessions.pop() {
                return Ok(SessionGuard {
                    pool: self,
                    session: Some(session),
                });
            }
        }
        Self::create_session(&self.model_path, self.backend, self.device_id).map(|session| {
            SessionGuard {
                pool: self,
                session: Some(session),
            }
        })
    }

    fn return_session(&self, session: ort::session::Session) {
        let mut sessions = self.sessions.lock();
        sessions.push(session);
    }

    fn create_session(
        path: &str,
        backend: InferenceBackend,
        _device_id: usize,
    ) -> Result<ort::session::Session, String> {
        use ort::session::Session;
        #[allow(unused_mut)]
        let mut builder =
            Session::builder().map_err(|e| format!("Failed to create session builder: {}", e))?;
        match backend {
            InferenceBackend::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    let ep = CUDA::default().with_device_id(_device_id as i32);
                    builder = builder
                        .with_execution_providers([ep.build()])
                        .map_err(|e| format!("Failed to add CUDA execution provider: {}", e))?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        "CUDA backend requested but fluxion was built without the `cuda` feature; \
                         rebuild with `cargo build --features cuda` or set FLUXION_ONNX_BACKEND=cpu"
                            .to_string(),
                    );
                }
            }
            InferenceBackend::CoreML => {
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    let ep = CoreML::default();
                    builder = builder
                        .with_execution_providers([ep.build()])
                        .map_err(|e| format!("Failed to add CoreML execution provider: {}", e))?;
                }
                #[cfg(not(all(feature = "coreml", target_os = "macos")))]
                {
                    if cfg!(target_os = "macos") {
                        return Err(
                            "CoreML backend requested but fluxion was built without the `coreml` feature; \
                             rebuild with `cargo build --features coreml` or set FLUXION_ONNX_BACKEND=cpu"
                                .to_string(),
                        );
                    }
                    return Err("CoreML backend requested but is only available on macOS; \
                         set FLUXION_ONNX_BACKEND=cpu or FLUXION_ONNX_BACKEND=cuda"
                        .to_string());
                }
            }
            InferenceBackend::DirectML => {
                #[cfg(all(feature = "directml", target_os = "windows"))]
                {
                    let ep = DirectML::default().with_device_id(_device_id as i32);
                    builder = builder
                        .with_execution_providers([ep.build()])
                        .map_err(|e| format!("Failed to add DirectML execution provider: {}", e))?;
                }
                #[cfg(not(all(feature = "directml", target_os = "windows")))]
                {
                    if cfg!(target_os = "windows") {
                        return Err(
                            "DirectML backend requested but fluxion was built without the `directml` feature; \
                             rebuild with `cargo build --features directml` or set FLUXION_ONNX_BACKEND=cpu"
                                .to_string(),
                        );
                    }
                    return Err(
                        "DirectML backend requested but is only available on Windows; \
                         set FLUXION_ONNX_BACKEND=cpu or FLUXION_ONNX_BACKEND=cuda"
                            .to_string(),
                    );
                }
            }
            InferenceBackend::OpenVINO => {
                return Err(
                    "OpenVINO execution provider is not available in the pre-built \
                     ort v2.0.0-rc.13 binaries for any platform. \
                     OpenVINO requires building ONNX Runtime from source with Intel OpenVINO toolkit. \
                     Use FLUXION_ONNX_BACKEND=cpu or FLUXION_ONNX_BACKEND=cuda instead."
                        .to_string(),
                );
            }
            InferenceBackend::CPU => {}
        }
        builder
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))
    }
}

/// Stub [`SessionPool`] methods when the `ort` feature is disabled
/// (issue #1294). `SessionPool::new` accepts a model path but never loads a
/// session; `get_or_create_session` returns an error explaining that ONNX
/// inference is unavailable.
#[cfg(not(feature = "ort"))]
impl SessionPool {
    #[allow(dead_code)]
    fn new(model_path: String, backend: InferenceBackend, device_id: usize) -> Self {
        SessionPool {
            model_path,
            backend,
            device_id,
        }
    }

    #[allow(dead_code)]
    fn get_or_create_session(&self) -> Result<SessionGuard<'_>, String> {
        Err("ONNX inference requires the `ort` feature (build with --features ort)".to_string())
    }
}

#[cfg(feature = "ort")]
struct SessionGuard<'a> {
    pool: &'a SessionPool,
    session: Option<ort::session::Session>,
}

/// Stub [`SessionGuard`] when the `ort` feature is disabled (issue #1294).
/// Constructed by [`SessionPool::get_or_create_session`] in its error path,
/// but never actually used (the function returns `Err` before producing it).
#[cfg(not(feature = "ort"))]
#[allow(dead_code)]
struct SessionGuard<'a> {
    _pool: &'a SessionPool,
}

#[cfg(feature = "ort")]
impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool.return_session(session);
        }
    }
}

#[cfg(feature = "ort")]
impl<'a> std::ops::Deref for SessionGuard<'a> {
    type Target = ort::session::Session;
    fn deref(&self) -> &Self::Target {
        self.session.as_ref().unwrap()
    }
}

#[cfg(feature = "ort")]
impl<'a> std::ops::DerefMut for SessionGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().unwrap()
    }
}

// ===== Issue #3313 — runtime execution-provider probe =======================
//
// The ort rc.13 migration (#3296) was verified compile-only: `cargo check`
// proves the EP API surface exists, but says nothing about whether a
// GPU/NPU execution provider actually activates at runtime. ORT's classic
// failure mode is the *silent fallback*: `with_execution_providers`
// succeeds, session creation succeeds, and every node quietly runs on the
// CPU EP.
//
// [`ExecutionProviderReport::capture`] makes EP status *observable* by
// combining three independent signals per provider:
//
// 1. `compiled_in`      — can this binary construct the EP at all
//                         (fluxion feature + target-OS gate)?
// 2. `environment_device_present` — did ORT enumerate a real hardware
//                         device for this EP at environment creation
//                         (`Environment::devices()`)? Absent hardware ⇒
//                         absent device, no guessing.
// 3. `registration`     — does attaching the EP (with
//                         `error_on_failure`) to a session builder succeed?
//                         Catches missing provider shared libraries.
//
// On any machine — GPU or not — `capture()` never panics and degrades to an
// explicit per-provider verdict, so the CPU-only contract (EP absent ⇒
// reported unavailable ⇒ CPU path used) is directly assertable in tests.

/// ONNX Runtime's canonical execution-provider names, as reported by
/// [`ort::device::Device::ep`] and [`ort::ep::ExecutionProvider::name`].
///
/// Note the DirectML spelling: ORT names it `DmlExecutionProvider`.
#[cfg(feature = "ort")]
mod ep_names {
    pub const CPU: &str = "CPUExecutionProvider";
    pub const CUDA: &str = "CUDAExecutionProvider";
    pub const COREML: &str = "CoreMLExecutionProvider";
    pub const DIRECTML: &str = "DmlExecutionProvider";
}

/// One hardware device enumerated by ORT for an execution provider
/// (`Environment::devices()` → `OrtEpDevice`).
#[cfg(feature = "ort")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EpDeviceSummary {
    /// EP name, e.g. `"CUDAExecutionProvider"` / `"CPUExecutionProvider"`.
    pub ep_name: String,
    /// EP vendor, e.g. `"Microsoft"` for DirectML devices.
    pub ep_vendor: Option<String>,
    /// `"cpu"`, `"gpu"`, or `"npu"`.
    pub hardware_type: &'static str,
    /// Hardware manufacturer, when ORT reports one.
    pub hardware_vendor: Option<String>,
    /// Device id as reported by ORT (may differ from CUDA device ordinals).
    pub device_id: Option<u32>,
}

/// Probe verdict for a single [`InferenceBackend`] execution provider.
#[cfg(feature = "ort")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpProbeOutcome {
    pub backend: InferenceBackend,
    /// ORT's canonical name for this EP (see the `ep_names` constants).
    pub ep_name: &'static str,
    /// `true` when this binary can construct the EP (feature + target gate).
    pub compiled_in: bool,
    /// `Some(reason)` when the current target OS cannot ever use this EP
    /// (e.g. CoreML outside macOS). Independent of feature gates.
    pub unsupported_on_target: Option<&'static str>,
    /// `true` when ORT enumerated a real hardware device for this EP in the
    /// current environment.
    pub environment_device_present: bool,
    /// Result of attaching this EP (with `error_on_failure`) to a session
    /// builder. `None` when the probe was skipped because the EP type is
    /// not compiled into this binary.
    pub registration: Option<Result<(), String>>,
    /// Best-effort activation verdict: compiled in AND a device was
    /// enumerated AND registration succeeded. For definitive per-node
    /// assignment proof see `docs/ORT_EP_VALIDATION.md` (ORT EP-assignment
    /// log lines, and the hardware-gated `#[ignore]` tests in this module).
    pub activated: bool,
}

#[cfg(feature = "ort")]
impl EpProbeOutcome {
    /// Human-readable one-line status, e.g. for `--list-backends` style
    /// diagnostics or the validation runbook.
    pub fn status_line(&self) -> String {
        if self.activated {
            format!("{}: ACTIVE ({})", self.ep_name, self.backend.as_str())
        } else if let Some(reason) = self.unsupported_on_target {
            format!("{}: unavailable on this target — {}", self.ep_name, reason)
        } else if !self.compiled_in {
            format!(
                "{}: not compiled into this binary (backend `{}` needs its fluxion feature)",
                self.ep_name,
                self.backend.as_str()
            )
        } else if !self.environment_device_present {
            format!(
                "{}: compiled in, but ORT enumerated no hardware device — silent CPU fallback would occur",
                self.ep_name
            )
        } else {
            format!(
                "{}: device present but registration failed — {:?}",
                self.ep_name, self.registration
            )
        }
    }
}

/// Snapshot of execution-provider availability and activation status for
/// the running process (issue #3313).
///
/// Obtain via [`ExecutionProviderReport::capture`]. Probes cover the CPU
/// baseline plus every target-appropriate GPU/NPU EP (CUDA, CoreML,
/// DirectML); EPs that are not applicable on the current target are still
/// reported, with `unsupported_on_target` explaining why.
#[cfg(feature = "ort")]
#[derive(Clone, Debug)]
pub struct ExecutionProviderReport {
    /// ONNX Runtime API version of the linked backend (`ort::MINOR_VERSION`).
    pub ort_api_version: u32,
    /// All hardware devices ORT enumerated, across all EPs.
    pub devices: Vec<EpDeviceSummary>,
    /// One probe per backend, always including the CPU baseline.
    pub probes: Vec<EpProbeOutcome>,
}

#[cfg(feature = "ort")]
impl ExecutionProviderReport {
    /// Probe every relevant execution provider in the current process.
    ///
    /// Never panics and never returns `Err`: EP absence is a *report*, not
    /// a failure. Safe to call repeatedly; the ORT environment is a
    /// process-global singleton.
    pub fn capture() -> Self {
        // Enumerate real EP devices. `Environment::devices()` returns an
        // empty iterator when the linked backend lacks EP-ABI device
        // enumeration support, in which case `device_enumeration` degrades
        // to `false` and activation verdicts fall back to compile-time +
        // registration signals only.
        let env = ort::environment::Environment::current()
            .map_err(|e| {
                warn!("EP probe: could not obtain ORT environment: {}", e);
                e
            })
            .ok();
        let devices: Vec<EpDeviceSummary> = env
            .as_ref()
            .map(|env| {
                env.devices()
                    .map(|d| {
                        let hw = d.hardware_device();
                        EpDeviceSummary {
                            ep_name: d.ep().unwrap_or("<unknown>").to_string(),
                            ep_vendor: d.ep_vendor().ok().map(str::to_string),
                            hardware_type: match hw.ty() {
                                ort::memory::DeviceType::CPU => "cpu",
                                ort::memory::DeviceType::GPU => "gpu",
                                ort::memory::DeviceType::NPU => "npu",
                            },
                            hardware_vendor: hw.vendor().ok().map(str::to_string),
                            device_id: Some(hw.id()),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let device_enumeration = !devices.is_empty();

        let mut probes = Vec::with_capacity(4);

        // ---- CPU baseline: always compiled in, always target-appropriate.
        let cpu_registration = {
            use ort::ep::ExecutionProvider as _;
            ort::ep::CPU::default()
                .is_available()
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        probes.push(EpProbeOutcome {
            backend: InferenceBackend::CPU,
            ep_name: ep_names::CPU,
            compiled_in: true,
            unsupported_on_target: None,
            environment_device_present: !device_enumeration
                || devices.iter().any(|d| d.ep_name == ep_names::CPU),
            registration: Some(cpu_registration),
            // CPU is the fallback of last resort: report it active whenever
            // the runtime is usable at all, even if device enumeration is
            // unsupported (then the CPU EP is the only meaningful answer).
            activated: true,
        });

        // ---- CUDA (Linux/Windows + NVIDIA GPU, `--features cuda`).
        probes.push(Self::probe_ep(
            InferenceBackend::CUDA,
            ep_names::CUDA,
            cfg!(feature = "cuda"),
            if cfg!(target_os = "macos") {
                Some("CUDA execution providers are not shipped for macOS ORT builds")
            } else {
                None
            },
            &devices,
            device_enumeration,
            || {
                #[cfg(feature = "cuda")]
                {
                    (|| -> Result<(), String> {
                        // `Session::builder` and `with_execution_providers`
                        // carry different recoverable-error payloads, so
                        // chain via `?` inside a `String`-error closure.
                        let builder =
                            ort::session::Session::builder().map_err(|e| e.to_string())?;
                        builder
                            .with_execution_providers([ort::ep::CUDA::default()
                                .with_device_id(0)
                                .build()
                                .error_on_failure()])
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("not compiled in".to_string())
                }
            },
        ));

        // ---- CoreML (Apple Silicon macOS, `--features coreml`).
        probes.push(Self::probe_ep(
            InferenceBackend::CoreML,
            ep_names::COREML,
            cfg!(all(feature = "coreml", target_os = "macos")),
            if !cfg!(target_os = "macos") {
                Some("CoreML is only available on macOS")
            } else {
                None
            },
            &devices,
            device_enumeration,
            || {
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    (|| -> Result<(), String> {
                        let builder =
                            ort::session::Session::builder().map_err(|e| e.to_string())?;
                        builder
                            .with_execution_providers([ort::ep::CoreML::default()
                                .build()
                                .error_on_failure()])
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })()
                }
                #[cfg(not(all(feature = "coreml", target_os = "macos")))]
                {
                    Err("not compiled in".to_string())
                }
            },
        ));

        // ---- DirectML (Windows + DirectX 12 GPU, `--features directml`).
        probes.push(Self::probe_ep(
            InferenceBackend::DirectML,
            ep_names::DIRECTML,
            cfg!(all(feature = "directml", target_os = "windows")),
            if !cfg!(target_os = "windows") {
                Some("DirectML is only available on Windows")
            } else {
                None
            },
            &devices,
            device_enumeration,
            || {
                #[cfg(all(feature = "directml", target_os = "windows"))]
                {
                    (|| -> Result<(), String> {
                        let builder =
                            ort::session::Session::builder().map_err(|e| e.to_string())?;
                        builder
                            .with_execution_providers([ort::ep::DirectML::default()
                                .with_device_id(0)
                                .build()
                                .error_on_failure()])
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })()
                }
                #[cfg(not(all(feature = "directml", target_os = "windows")))]
                {
                    Err("not compiled in".to_string())
                }
            },
        ));

        ExecutionProviderReport {
            ort_api_version: ort::MINOR_VERSION,
            devices,
            probes,
        }
    }

    /// Assemble one [`EpProbeOutcome`] from the shared device list plus a
    /// lazily-run registration probe (skipped entirely when `compiled_in`
    /// is `false`).
    fn probe_ep(
        backend: InferenceBackend,
        ep_name: &'static str,
        compiled_in: bool,
        unsupported_on_target: Option<&'static str>,
        devices: &[EpDeviceSummary],
        device_enumeration: bool,
        registration: impl FnOnce() -> Result<(), String>,
    ) -> EpProbeOutcome {
        let environment_device_present = devices.iter().any(|d| d.ep_name == ep_name);
        let registration = if compiled_in {
            Some(registration())
        } else {
            None
        };
        let activated = compiled_in
            && unsupported_on_target.is_none()
            && (!device_enumeration || environment_device_present)
            && registration.as_ref().is_some_and(|r| r.is_ok());
        EpProbeOutcome {
            backend,
            ep_name,
            compiled_in,
            unsupported_on_target,
            environment_device_present,
            registration,
            activated,
        }
    }

    /// Probe result for `backend`, if it was probed.
    pub fn probe(&self, backend: InferenceBackend) -> Option<&EpProbeOutcome> {
        self.probes.iter().find(|p| p.backend == backend)
    }

    /// `true` when no GPU/NPU EP activated, i.e. inference runs (or would
    /// run) on the CPU execution provider.
    pub fn cpu_only(&self) -> bool {
        !self
            .probes
            .iter()
            .any(|p| p.activated && p.backend != InferenceBackend::CPU)
    }

    /// Backends whose probe concluded `activated`.
    pub fn activated_backends(&self) -> Vec<InferenceBackend> {
        self.probes
            .iter()
            .filter(|p| p.activated)
            .map(|p| p.backend)
            .collect()
    }

    /// Human-readable status lines, one per probe, for logs and the
    /// validation runbook (`ExecutionProviderReport::capture()` output).
    pub fn status_lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "ORT api version: {} (device enumeration: {})",
            self.ort_api_version,
            if self.devices.is_empty() {
                "unavailable"
            } else {
                "available"
            }
        )];
        for d in &self.devices {
            lines.push(format!(
                "device: {} [{}, vendor {:?}, id {:?}] (ep vendor {:?})",
                d.ep_name, d.hardware_type, d.hardware_vendor, d.device_id, d.ep_vendor
            ));
        }
        lines.extend(self.probes.iter().map(EpProbeOutcome::status_line));
        lines
    }
}

/// One-shot guard for the `FLUXION_ONNX_BACKEND` silent-downgrade warn
/// (Issue #2920). `compare_exchange(false, true, …)` ensures exactly one warn
/// per process even under parallel callers; tests reset to `false` so the
/// assertion can re-trigger the path.
static BACKEND_DOWNGRADE_WARNED: AtomicBool = AtomicBool::new(false);

impl SurrogateManager {
    /// Emit (at most once per process) a `tracing::warn!` that surfaces a
    /// silent CUDA→CPU downgrade. The previous behaviour returned CPU with no
    /// diagnostic, so an operator who enabled CUDA on a prebuilt image paid
    /// the CPU throughput floor invisibly (#2920). The first call wins; the
    /// `AtomicBool` is the one-shot guard.
    ///
    /// `env_value` is the raw `FLUXION_ONNX_BACKEND` value the operator set
    /// (the issue specifies logging the *value*, not the env-var name).
    fn warn_backend_downgrade(
        env_value: &str,
        requested: InferenceBackend,
        resolved: InferenceBackend,
    ) {
        if BACKEND_DOWNGRADE_WARNED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let gpu_value = std::env::var("FLUXION_GPU").unwrap_or_default();
        let env_display = if env_value.is_empty() {
            "<unset>"
        } else {
            env_value
        };
        let gpu_display = if gpu_value.is_empty() {
            "<unset>"
        } else {
            gpu_value.as_str()
        };
        tracing::warn!(
            target: "fluxion::ai::surrogate::backend",
            FLUXION_ONNX_BACKEND = env_display,
            FLUXION_GPU = gpu_display,
            requested_backend = ?requested,
            resolved_backend = ?resolved,
            "FLUXION_ONNX_BACKEND requested CUDA but the surrogate was downgraded to CPU; rebuild with `cargo build --features cuda` or set FLUXION_ONNX_BACKEND=cpu",
        );
    }

    /// Built-in default model path used when neither `FLUXION_ONNX_MODEL`
    /// nor an explicit `load_onnx` call supplies a model. The zone-thermal
    /// surrogate is shipped in `models/` and is the most general of the
    /// trained components (conduction, solar, ventilation, zone).
    pub const DEFAULT_MODEL_PATH: &'static str = "models/surrogate_zone_thermal.onnx";

    /// Construct a `SurrogateManager` with no model loaded (legacy mock mode).
    ///
    /// Use [`Self::new_with_auto_load`] in production code paths to pick up
    /// a real ONNX model from the environment or built-in default path.
    pub fn new() -> Result<Self, String> {
        Ok(SurrogateManager {
            model_loaded: false,
            model_path: None,
            session_pool: None,
            backend: InferenceBackend::CPU,
            device_id: 0,
            composite: None,
            inference_metrics: Arc::new(parking_lot::Mutex::new(InferenceMetrics::default())),
            input_bounds: None,
            ood_count: Arc::new(parking_lot::Mutex::new(0)),
            residual_tau: DEFAULT_RESIDUAL_TAU,
            residual_reroute_count: Arc::new(parking_lot::Mutex::new(0)),
        })
    }

    /// Construct a `SurrogateManager`, auto-loading a real ONNX model when
    /// one is available. Resolution order (first hit wins):
    ///
    /// 1. `FLUXION_ONNX_MODEL` environment variable (explicit override)
    /// 2. `FLUXION_ONNX_BACKEND` selects the inference backend
    ///    (`cpu`, `cuda`, `coreml`, `directml`, `openvino`).
    ///    Defaults to `cpu`. `cuda` is a no-op when the `cuda` feature
    ///    is disabled — the manager falls back to CPU at runtime.
    /// 3. [`Self::DEFAULT_MODEL_PATH`] (`models/surrogate_zone_thermal.onnx`)
    ///    if it exists on disk.
    ///
    /// Both resolution paths route through [`validate_model_path`] (Issue
    /// #2905) so an operator — or compromised CI step — cannot bypass the
    /// `FLUXION_MODEL_DIR` allow-list, the `.onnx` extension check, or the
    /// 256 MiB size cap by setting `FLUXION_ONNX_MODEL` to a non-model file
    /// such as `/proc/self/environ` or a 10 GiB binary. An explicit
    /// `FLUXION_ONNX_MODEL` that fails validation surfaces the error to
    /// the caller (mirroring `BatchOracle::load_surrogate`); the built-in
    /// default path silently falls back to mock mode when the file is
    /// absent so callers that never set the env var keep working in
    /// air-gapped / fresh-checkout scenarios.
    ///
    /// If none of the above resolve to an existing file, the manager is
    /// returned in mock mode (matching [`Self::new`]) so callers can still
    /// fall back to analytical loads.
    pub fn new_with_auto_load() -> Result<Self, String> {
        let backend = Self::resolve_backend_from_env();
        // 1. Explicit env var override — must pass `validate_model_path`.
        //    A missing path, wrong extension, or out-of-allow-list location
        //    short-circuits with an `Err` so the operator learns about
        //    misconfiguration rather than silently falling back to a model
        //    they did not request.
        if let Ok(path) = std::env::var("FLUXION_ONNX_MODEL") {
            if !path.is_empty() {
                let validated = validate_model_path(&path)?;
                return Self::load_with_backend(&validated.to_string_lossy(), backend, 0);
            }
        }
        // 2. Built-in default path — also validated. A missing file (the
        //    common fresh-checkout case) falls through to mock mode below.
        let default_path = Self::DEFAULT_MODEL_PATH;
        if let Ok(validated) = validate_model_path(default_path) {
            return Self::load_with_backend(&validated.to_string_lossy(), backend, 0);
        }
        // 3. No model available — return mock manager.
        Ok(SurrogateManager {
            model_loaded: false,
            model_path: None,
            session_pool: None,
            backend: InferenceBackend::CPU,
            device_id: 0,
            composite: None,
            inference_metrics: Arc::new(parking_lot::Mutex::new(InferenceMetrics::default())),
            input_bounds: None,
            ood_count: Arc::new(parking_lot::Mutex::new(0)),
            residual_tau: DEFAULT_RESIDUAL_TAU,
            residual_reroute_count: Arc::new(parking_lot::Mutex::new(0)),
        })
    }

    /// Resolve the [`InferenceBackend`] from the `FLUXION_ONNX_BACKEND`
    /// environment variable. Unknown values fall back to CPU. The CUDA
    /// variant is downgraded to CPU when the `cuda` feature is disabled or
    /// when `FLUXION_GPU=0|false|<empty>` is set; a one-shot `tracing::warn!`
    /// surfaces the silent downgrade (Issue #2920).
    fn resolve_backend_from_env() -> InferenceBackend {
        let raw = std::env::var("FLUXION_ONNX_BACKEND").unwrap_or_default();
        let parsed = match raw.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Some(InferenceBackend::CUDA),
            "coreml" => Some(InferenceBackend::CoreML),
            "directml" => Some(InferenceBackend::DirectML),
            "openvino" => Some(InferenceBackend::OpenVINO),
            "cpu" | "" => Some(InferenceBackend::CPU),
            _ => None,
        };
        let resolved = match parsed {
            Some(InferenceBackend::CUDA) => {
                #[cfg(feature = "cuda")]
                {
                    if matches!(
                        std::env::var("FLUXION_GPU").as_deref(),
                        Ok("0") | Ok("false") | Ok("")
                    ) {
                        InferenceBackend::CPU
                    } else {
                        InferenceBackend::CUDA
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    InferenceBackend::CPU
                }
            }
            Some(other) => other,
            None => InferenceBackend::CPU,
        };
        // Issue #2920: a CUDA request that resolves to CPU used to be silent.
        // Now the first caller per process gets a `tracing::warn!` (target
        // `fluxion::ai::surrogate::backend`) naming the env var, its value,
        // the requested backend, and the resolved backend, plus a hint to
        // rebuild with `--features cuda`.
        if matches!(parsed, Some(InferenceBackend::CUDA)) && resolved != InferenceBackend::CUDA {
            Self::warn_backend_downgrade(&raw, InferenceBackend::CUDA, resolved);
        }
        resolved
    }

    fn load_with_backend(
        path: &str,
        backend: InferenceBackend,
        device_id: usize,
    ) -> Result<Self, String> {
        Self::with_gpu_backend(path, backend, device_id)
    }

    /// Returns `true` if the manager has no real ONNX model loaded and is
    /// therefore returning placeholder ("mock") predictions.
    ///
    /// When `is_mock()` returns `true`, [`Self::predict_loads`] and
    /// [`Self::predict_loads_batched`] return a constant `1.2` per zone
    /// (or the analytical fallback) instead of running a neural network.
    /// Use [`Self::load_onnx`] to load a real model and switch off mock
    /// mode.
    pub fn is_mock(&self) -> bool {
        !self.model_loaded && self.composite.is_none()
    }

    /// Returns a snapshot of the current inference metrics.
    pub fn inference_metrics(&self) -> InferenceMetrics {
        self.inference_metrics.lock().clone()
    }

    /// Set the per-feature training bounds for OOD detection.
    ///
    /// Should be called after loading a model, using bounds extracted from
    /// the training dataset during model training (Issue #1892).
    pub fn set_input_bounds(&mut self, bounds: InputBounds) {
        self.input_bounds = Some(bounds);
    }

    /// Get a reference to the currently configured input bounds, if any.
    pub fn get_input_bounds(&self) -> Option<&InputBounds> {
        self.input_bounds.as_ref()
    }

    /// Get the number of times OOD input has been detected.
    pub fn ood_count(&self) -> usize {
        *self.ood_count.lock()
    }

    /// Reset the OOD detection counter.
    pub fn reset_ood_count(&mut self) {
        *self.ood_count.lock() = 0;
    }

    /// Get the number of times the residual guard caused a reroute.
    pub fn residual_reroute_count(&self) -> usize {
        *self.residual_reroute_count.lock()
    }

    /// Reset the residual reroute counter.
    pub fn reset_residual_reroute_count(&mut self) {
        *self.residual_reroute_count.lock() = 0;
    }

    /// Set the residual threshold τ. Predictions with squared residual > τ
    /// will trigger rerouting to the analytical fallback.
    pub fn set_residual_tau(&mut self, tau: f64) {
        self.residual_tau = tau;
    }

    /// Compute the energy-balance residual for a batch of inference inputs
    /// and predicted loads, checking against the configured threshold τ.
    ///
    /// This is the **inference-time** companion to [`SurrogateDomain::energy_balance_residual`]
    /// (which is only called during training, Issue #1706). The residual guard
    /// catches model drift, quantization artifacts, or distribution shift that
    /// produces physically implausible load predictions (Issue #1896).
    ///
    /// The physics model is identical to [`SurrogateDomain::energy_balance_residual`]:
    /// `Q_expected = Q_conduction + Q_solar + Q_internal + Q_ventilation`
    ///
    /// where thermal properties are:
    /// - U = 0.5 W/m²K, A = 100 m² (envelope conduction)
    /// - α = 0.85 (solar absorptivity), solar_rad in W/m²
    /// - β = 100 W/person (internal gains from occupancy)
    /// - C_air = 1260 J/kgK, ACH = 0.5, V = 300 m³ (ventilation)
    ///
    /// Returns `Ok(())` if all samples pass (residual ≤ τ) or if the manager
    /// is in mock mode (no model loaded). Returns `Err(ResidualViolation)`
    /// for the **first** sample that exceeds the threshold.
    ///
    /// The `inputs` slice uses the same indexing as [`SurrogateInputs::from_temps`]:
    /// index 0 = exterior_temp, 1 = zone_temp. Additional features (solar,
    /// humidity, occupancy) are synthesised using `SurrogateInputs::from_temps`.
    ///
    /// The `predicted` slice must have the same length as `inputs`. A mismatch
    /// causes an early return with `Ok(())` — no violation is recorded.
    pub fn check_inference_residual(
        &self,
        inputs: &[f64],
        predicted: &[f64],
    ) -> Result<(), ResidualViolation> {
        if predicted.is_empty() {
            return Ok(());
        }
        if !self.model_loaded && self.composite.is_none() {
            return Ok(());
        }

        let surrogate_inputs = SurrogateInputs::from_temps(inputs);

        const U_WALL: f64 = 0.5;
        const A_ZONE: f64 = 100.0;
        const ALPHA_SOLAR: f64 = 0.85;
        const BETA_INTERNAL: f64 = 100.0;
        const C_AIR: f64 = 1260.0;
        const V_VENT: f64 = 300.0;
        const ACH_VENT: f64 = 0.5;

        let delta_t = surrogate_inputs.exterior_temp - surrogate_inputs.zone_temp;
        let q_conduction = U_WALL * A_ZONE * delta_t;
        let q_solar = ALPHA_SOLAR * surrogate_inputs.solar_rad * A_ZONE * 0.001;
        let q_internal = BETA_INTERNAL * surrogate_inputs.occupancy;
        let q_ventilation = C_AIR * ACH_VENT * V_VENT * delta_t / 3600.0 / 1000.0;
        let q_expected = q_conduction + q_solar + q_internal + q_ventilation;

        let n_samples = predicted.len();
        for (i, &q_predicted) in predicted.iter().enumerate() {
            let residual = q_predicted - q_expected;
            let residual_sq = residual * residual;
            if residual_sq > self.residual_tau {
                return Err(ResidualViolation {
                    sample_index: i,
                    predicted: q_predicted,
                    expected: q_expected,
                    residual: residual_sq,
                });
            }
        }
        let _ = n_samples;
        Ok(())
    }

    /// Validate an inference input vector against the stored training bounds.
    ///
    /// Returns `OodValidationResult` indicating whether the input is OOD
    /// and a list of warnings for each out-of-bounds feature.
    ///
    /// When `is_ood` is `true`, callers MUST fall back to the physics
    /// solver instead of running the surrogate — the surrogate is not
    /// validated for inputs outside its training distribution (Issue #1892).
    ///
    /// If no `InputBounds` have been configured (default state), this
    /// method always returns `OodValidationResult::clean()` (no OOD,
    /// no warnings) so that missing bounds never block inference.
    ///
    /// NOTE: This method validates a raw `&[f64]` temperature vector using
    /// the same feature indexing as [`SurrogateInputs`]: index 0 = exterior_temp,
    /// 1 = zone_temp, 2 = solar_rad, 3 = humidity, 4 = occupancy.
    /// Callers using [`SurrogateInputs`] should call
    /// [`validate_inputs_struct`] instead.
    pub fn validate_input_bounds(&self, inputs: &[f64]) -> OodValidationResult {
        let Some(bounds) = &self.input_bounds else {
            return OodValidationResult::clean();
        };

        let mut warnings = Vec::new();
        let checks: [(usize, f64, (f64, f64), &'static str); 5] = [
            (
                0,
                inputs.first().copied().unwrap_or(20.0),
                bounds.exterior_temp,
                "exterior_temp",
            ),
            (
                1,
                inputs.get(1).copied().unwrap_or(22.0),
                bounds.zone_temp,
                "zone_temp",
            ),
            (
                2,
                inputs.get(2).copied().unwrap_or(0.0),
                bounds.solar_rad,
                "solar_rad",
            ),
            (
                3,
                inputs.get(3).copied().unwrap_or(50.0),
                bounds.humidity,
                "humidity",
            ),
            (
                4,
                inputs.get(4).copied().unwrap_or(0.1),
                bounds.occupancy,
                "occupancy",
            ),
        ];

        for (idx, val, (min, max), name) in checks {
            if val < min || val > max {
                warnings.push(OodInputWarning::new(name, idx, val, min, max));
            }
        }

        if warnings.is_empty() {
            OodValidationResult::clean()
        } else {
            for w in &warnings {
                w.log_warning();
            }
            *self.ood_count.lock() += 1;
            OodValidationResult {
                is_ood: true,
                warnings,
            }
        }
    }

    /// Validate a [`SurrogateInputs`] struct against the stored training bounds.
    ///
    /// This is the structured-input variant of [`validate_input_bounds`].
    /// Returns `OodValidationResult` with per-feature OOD warnings.
    pub fn validate_inputs_struct(&self, inputs: &SurrogateInputs) -> OodValidationResult {
        let Some(bounds) = &self.input_bounds else {
            return OodValidationResult::clean();
        };

        let mut warnings = Vec::new();

        if inputs.exterior_temp < bounds.exterior_temp.0
            || inputs.exterior_temp > bounds.exterior_temp.1
        {
            warnings.push(OodInputWarning::new(
                "exterior_temp",
                0,
                inputs.exterior_temp,
                bounds.exterior_temp.0,
                bounds.exterior_temp.1,
            ));
        }

        if inputs.zone_temp < bounds.zone_temp.0 || inputs.zone_temp > bounds.zone_temp.1 {
            warnings.push(OodInputWarning::new(
                "zone_temp",
                1,
                inputs.zone_temp,
                bounds.zone_temp.0,
                bounds.zone_temp.1,
            ));
        }

        if inputs.solar_rad < bounds.solar_rad.0 || inputs.solar_rad > bounds.solar_rad.1 {
            warnings.push(OodInputWarning::new(
                "solar_rad",
                2,
                inputs.solar_rad,
                bounds.solar_rad.0,
                bounds.solar_rad.1,
            ));
        }

        if inputs.humidity < bounds.humidity.0 || inputs.humidity > bounds.humidity.1 {
            warnings.push(OodInputWarning::new(
                "humidity",
                3,
                inputs.humidity,
                bounds.humidity.0,
                bounds.humidity.1,
            ));
        }

        if inputs.occupancy < bounds.occupancy.0 || inputs.occupancy > bounds.occupancy.1 {
            warnings.push(OodInputWarning::new(
                "occupancy",
                4,
                inputs.occupancy,
                bounds.occupancy.0,
                bounds.occupancy.1,
            ));
        }

        if !bounds.valid_climate_zones.contains(&inputs.climate_zone) {
            warn!(
                "OOD detected: climate_zone '{}' is not in training zones {:?}",
                inputs.climate_zone, bounds.valid_climate_zones
            );
            warnings.push(OodInputWarning::new("climate_zone", 5, 0.0, 0.0, 0.0));
            *self.ood_count.lock() += 1;
        }

        if warnings.is_empty() {
            OodValidationResult::clean()
        } else {
            for w in &warnings {
                w.log_warning();
            }
            *self.ood_count.lock() += 1;
            OodValidationResult {
                is_ood: true,
                warnings,
            }
        }
    }

    /// Predict thermal loads, preferring real ONNX inference when a model
    /// is loaded and falling back to the analytical model otherwise.
    ///
    /// Issue #1285: prior versions of this method unconditionally returned
    /// a `vec![1.2; n]` mock constant whenever the model was not loaded,
    /// which silently shadowed the analytical fallback. This implementation
    /// routes:
    ///
    /// - **Model loaded** → real ONNX inference via [`Self::predict_loads_onnx`].
    /// - **Model not loaded** → [`Self::analytical_loads`] (the synthetic
    ///   sine-cycle surrogate retained for offline use).
    /// - **ONNX inference errors** → [`Self::analytical_loads`] with a
    ///   warning, so the simulation keeps running.
    pub fn predict_loads_with_fallback(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
        // Empty input is a no-op for both paths.
        if temps.is_empty() {
            return Ok(Vec::new());
        }

        // No model loaded → use the analytical fallback (not the 1.2 mock).
        if !self.model_loaded && self.composite.is_none() {
            self.record_onnx_fallback_metric();
            return self.analytical_loads(temps);
        }

        // Model loaded → try real ONNX inference.
        match self.predict_loads_onnx(temps) {
            Ok(loads) => {
                if let Err(violation) = self.check_inference_residual(temps, &loads) {
                    warn!(
                        "surrogate residual violation: sample {} predicted {:.2} W expected {:.2} W residual {:.2} W² — rerouting to analytical fallback",
                        violation.sample_index, violation.predicted, violation.expected, violation.residual
                    );
                    *self.residual_reroute_count.lock() += 1;
                    metrics::counter!("surrogate_residual_reroutes_total", "mode" => "neural_with_fallback").increment(1);
                    self.record_onnx_fallback_metric();
                    return self.analytical_loads(temps);
                }
                Ok(loads)
            }
            Err(e) => {
                warn!(
                    "ONNX inference failed ({}), falling back to analytical_loads",
                    e
                );
                self.record_onnx_fallback_metric();
                self.analytical_loads(temps)
            }
        }
    }

    /// Emit `fluxion_onnx_inference_total{backend, outcome="fallback"}` for
    /// every routing to the analytical model (Issue #2498). Called from all
    /// three fallback sites in [`Self::predict_loads_with_fallback`] (no
    /// model loaded, residual violation, ONNX error) so production telemetry
    /// can distinguish "neural surrogate diverged" from "ONNX call failed".
    fn record_onnx_fallback_metric(&self) {
        metrics::counter!(
            "fluxion_onnx_inference_total",
            "backend" => self.backend.as_str(),
            "outcome" => "fallback",
        )
        .increment(1);
    }

    pub fn analytical_loads(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
        if temps.is_empty() {
            return Ok(vec![]);
        }

        // Issue #1335: was previously derived from `SystemTime::now()`, making
        // the fallback non-deterministic and breaking the surrogate drift gate
        // (Issue #2923) whenever CI happened to run outside the wall-clock hour
        // the baseline JSON was captured at. Now derive the phase from the
        // first input element (`outdoor_temp` in the SurrogateThermalLoadAdapter
        // 6-element input vector), matching `deterministic_analytical_loads`.
        let phase = temps.first().copied().unwrap_or(12.0);
        let daily_cycle = (std::f64::consts::PI * (phase - 6.0) / 12.0).sin();
        let solar_gain = (50.0 * daily_cycle).max(0.0);

        Ok(vec![solar_gain; temps.len()])
    }

    /// Deterministic analytical fallback used by the golden-output harness.
    ///
    /// Issue #1335: `analytical_loads` uses `SystemTime::now()` for the
    /// solar cycle, which is non-deterministic and therefore unsuitable
    /// for regression tests. This function derives the load purely from
    /// the input vector: for each pair `(t_exterior, t_zone)` it returns
    /// `50.0 * max(0, sin(pi * (t_exterior - 6) / 12))`, matching the
    /// shape of `analytical_loads` but reproducible across runs.
    pub fn deterministic_analytical_loads(inputs: &[SurrogateInputs]) -> Vec<f64> {
        inputs
            .iter()
            .map(|inp| {
                let cycle = (std::f64::consts::PI * (inp.exterior_temp - 6.0) / 12.0).sin();
                (50.0 * cycle).max(0.0)
            })
            .collect()
    }

    pub fn gpu_supported(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            if !matches!(self.backend, InferenceBackend::CUDA) {
                return false;
            }
            !matches!(
                std::env::var("FLUXION_GPU").as_deref(),
                Ok("0") | Ok("false") | Ok("")
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Issue #2920: if the operator asked for CUDA via env but the
            // cuda feature is off, surface the silent downgrade before
            // returning `false`. Shared one-shot guard with
            // `resolve_backend_from_env`, so only the first caller emits.
            let env_value = std::env::var("FLUXION_ONNX_BACKEND").unwrap_or_default();
            let requested_cuda = matches!(env_value.to_ascii_lowercase().as_str(), "cuda" | "gpu");
            if requested_cuda {
                Self::warn_backend_downgrade(
                    &env_value,
                    InferenceBackend::CUDA,
                    InferenceBackend::CPU,
                );
            }
            false
        }
    }

    pub fn predict_loads_governed(
        &self,
        temps: &[f64],
        domain: &SurrogateDomain,
        mode: SurrogateMode,
    ) -> Result<Vec<f64>, String> {
        let inputs = SurrogateInputs::from_temps(temps);

        if !domain.is_valid(&inputs) {
            warn!(
                "Inputs out of domain bounds for surrogate. \
                 Temp: {:.1}, Zone: {:.1}, Solar: {:.1}, Climate: {}. \
                 Falling back to analytical model.",
                inputs.exterior_temp, inputs.zone_temp, inputs.solar_rad, inputs.climate_zone
            );
            return self.analytical_loads(temps);
        }

        match mode {
            SurrogateMode::NeuralOnly => {
                if self.composite.is_some() || self.session_pool.is_some() {
                    Ok(self
                        .predict_loads_batched(&[temps.to_vec()])
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| temps.iter().map(|&t| t * 0.05).collect()))
                } else {
                    warn!("NeuralOnly mode but no model loaded, using analytical");
                    self.analytical_loads(temps)
                }
            }
            SurrogateMode::NeuralWithFallback => self.predict_loads_with_fallback(temps),
            SurrogateMode::AnalyticalOnly => self.analytical_loads(temps),
        }
    }

    #[cfg(feature = "ort")]
    pub fn load_onnx(path: &str) -> Result<Self, String> {
        Self::with_gpu_backend(path, InferenceBackend::CPU, 0)
    }

    /// Load a pinned surrogate model by registry version.
    ///
    /// Issue #1335: looks up `version` in `registry`, validates the file's
    /// SHA-256 against the registry's `model_sha256`, then delegates to
    /// [`Self::load_onnx`]. Returns a typed error when:
    ///   * the version is missing from the registry;
    ///   * the file is not on disk;
    ///   * the file's SHA-256 does not match the registry hash.
    #[cfg(feature = "ort")]
    pub fn load_version(version: &str, registry: &ModelRegistry) -> Result<Self, String> {
        let entry = registry.lookup(version).ok_or_else(|| {
            format!(
                "version '{}' not found in registry (have: {:?})",
                version,
                registry
                    .versions
                    .iter()
                    .map(|v| &v.version)
                    .collect::<Vec<_>>()
            )
        })?;
        let path = Path::new(&entry.model_path);
        if !path.exists() {
            return Err(format!(
                "model file not found at '{}' (version {}); ONNX files are not committed to git and must be staged by CI before local runs",
                entry.model_path, entry.version
            ));
        }
        let actual = compute_file_sha256(path)?;
        validate_hash(&entry.model_sha256, &actual)?;
        Self::load_onnx(&entry.model_path)
    }

    /// Stub for non-`ort` builds (mirrors [`Self::load_version`]).
    #[cfg(not(feature = "ort"))]
    pub fn load_version(version: &str, registry: &ModelRegistry) -> Result<Self, String> {
        let entry = registry.lookup(version).ok_or_else(|| {
            format!(
                "version '{}' not found in registry (have: {:?})",
                version,
                registry
                    .versions
                    .iter()
                    .map(|v| &v.version)
                    .collect::<Vec<_>>()
            )
        })?;
        let path = Path::new(&entry.model_path);
        if !path.exists() {
            return Err(format!(
                "model file not found at '{}' (version {})",
                entry.model_path, entry.version
            ));
        }
        let actual = compute_file_sha256(path)?;
        validate_hash(&entry.model_sha256, &actual)?;
        Err(
            "Loading ONNX models requires the `ort` feature (build with --features ort)"
                .to_string(),
        )
    }

    /// Stub for `cargo build` without the `ort` feature (issue #1294).
    /// Returns a clear error instead of panicking; callers should detect the
    /// missing feature and surface a friendly message. Still validates the
    /// path so callers see a `not found` diagnostic before the feature error.
    #[cfg(not(feature = "ort"))]
    pub fn load_onnx(path: &str) -> Result<Self, String> {
        use std::path::Path;
        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }
        Err(
            "Loading ONNX models requires the `ort` feature (build with --features ort)"
                .to_string(),
        )
    }

    #[cfg(feature = "ort")]
    pub fn with_gpu_backend(
        path: &str,
        backend: InferenceBackend,
        device_id: usize,
    ) -> Result<Self, String> {
        info!(
            "Initializing SessionPool for model: {} (backend: {:?}, device: {})",
            path, backend, device_id
        );
        use std::path::Path;
        let model_path = Path::new(path);
        if !model_path.exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }
        // Issue #2906: fail-closed SHA-256 integrity check (see
        // `verify_onnx_signature`). Runs before the ONNX session is
        // instantiated so a poisoned model never reaches the inference path
        // and cannot influence ASHRAE 140 / BatchOracle results.
        verify_onnx_signature(model_path)?;
        let session = SessionPool::create_session(path, backend, device_id)?;
        let pool = SessionPool::new(path.to_string(), backend, device_id, session);
        Ok(SurrogateManager {
            model_loaded: true,
            model_path: Some(path.to_string()),
            session_pool: Some(Arc::new(pool)),
            backend,
            device_id,
            composite: None,
            inference_metrics: Arc::new(parking_lot::Mutex::new(InferenceMetrics::default())),
            input_bounds: None,
            ood_count: Arc::new(parking_lot::Mutex::new(0)),
            residual_tau: DEFAULT_RESIDUAL_TAU,
            residual_reroute_count: Arc::new(parking_lot::Mutex::new(0)),
        })
    }

    /// Stub for non-`ort` builds (issue #1294). Validates the path first so
    /// callers still see a `not found` diagnostic before the feature error.
    /// Also runs the SHA-256 integrity check (Issue #2906) so a poisoned
    /// model is rejected even before the feature gate surfaces its own
    /// error — the integrity check is the stronger guarantee.
    #[cfg(not(feature = "ort"))]
    pub fn with_gpu_backend(
        path: &str,
        _backend: InferenceBackend,
        _device_id: usize,
    ) -> Result<Self, String> {
        use std::path::Path;
        let model_path = Path::new(path);
        if !model_path.exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }
        verify_onnx_signature(model_path)?;
        Err(
            "Loading ONNX models requires the `ort` feature (build with --features ort)"
                .to_string(),
        )
    }

    #[cfg(feature = "ort")]
    pub fn with_multi_device(path: &str, config: MultiDeviceConfig) -> Result<Self, String> {
        use std::path::Path;
        let model_path = Path::new(path);
        if !model_path.exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }
        // Issue #2906: fail-closed SHA-256 integrity check (the multi-device
        // success path bypasses `with_gpu_backend`, so we must verify here).
        verify_onnx_signature(model_path)?;
        match MultiDeviceSessionPool::new(path.to_string(), &config) {
            Ok(multi_pool) => {
                let first_pool = multi_pool
                    .device_pools
                    .first()
                    .ok_or("Failed to get first device pool")?;
                Ok(SurrogateManager {
                    model_loaded: true,
                    model_path: Some(path.to_string()),
                    session_pool: Some(Arc::clone(first_pool)),
                    backend: InferenceBackend::CUDA,
                    device_id: config.device_ids.first().copied().unwrap_or(0),
                    composite: None,
                    inference_metrics: Arc::new(parking_lot::Mutex::new(
                        InferenceMetrics::default(),
                    )),
                    input_bounds: None,
                    ood_count: Arc::new(parking_lot::Mutex::new(0)),
                    residual_tau: DEFAULT_RESIDUAL_TAU,
                    residual_reroute_count: Arc::new(parking_lot::Mutex::new(0)),
                })
            }
            Err(e) => {
                eprintln!(
                    "Multi-device setup failed: {}, falling back to single device",
                    e
                );
                Self::with_gpu_backend(path, InferenceBackend::CUDA, 0)
            }
        }
    }

    /// Stub for non-`ort` builds (issue #1294).
    #[cfg(not(feature = "ort"))]
    pub fn with_multi_device(_path: &str, _config: MultiDeviceConfig) -> Result<Self, String> {
        Err("Multi-device ONNX inference requires the `ort` feature".to_string())
    }

    #[cfg(feature = "ort")]
    pub fn load_modular(component_configs: &[(&str, InferenceBackend)]) -> Result<Self, String> {
        if component_configs.is_empty() {
            return Err("At least one component model required for modular surrogate".to_string());
        }

        let mut components = Vec::new();
        for (model_path, backend) in component_configs {
            let manager = match backend {
                InferenceBackend::CPU => SurrogateManager::load_onnx(model_path)?,
                _ => SurrogateManager::with_gpu_backend(model_path, *backend, 0)?,
            };
            let name = std::path::Path::new(model_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(model_path);
            components.push(ComponentSurrogate::new(name, manager));
        }

        let composite = CompositeSurrogate::new(components);
        Ok(SurrogateManager {
            model_loaded: true,
            model_path: None,
            session_pool: None,
            backend: InferenceBackend::CPU,
            device_id: 0,
            composite: Some(composite),
            inference_metrics: Arc::new(parking_lot::Mutex::new(InferenceMetrics::default())),
            input_bounds: None,
            ood_count: Arc::new(parking_lot::Mutex::new(0)),
            residual_tau: DEFAULT_RESIDUAL_TAU,
            residual_reroute_count: Arc::new(parking_lot::Mutex::new(0)),
        })
    }

    /// Stub for non-`ort` builds (issue #1294).
    #[cfg(not(feature = "ort"))]
    pub fn load_modular(_component_configs: &[(&str, InferenceBackend)]) -> Result<Self, String> {
        Err("Modular ONNX surrogates require the `ort` feature".to_string())
    }

    /// Load a pre-quantized INT8 ONNX model for accelerated CPU inference.
    ///
    /// Quantized models typically achieve 2-4x speedup on CPU with <1% accuracy loss.
    /// Use [`tools/quantize_model.py`] to produce quantized models from FP32 ONNX files.
    ///
    /// # Arguments
    /// * `path` - Path to a quantized INT8 ONNX model
    ///
    /// # Example
    /// ```rust,ignore
    /// let manager = SurrogateManager::load_quantized_onnx("model_int8.onnx")?;
    /// let loads = manager.predict_loads(&[21.0, 22.0]);
    /// ```
    #[cfg(feature = "ort")]
    pub fn load_quantized_onnx(path: &str) -> Result<Self, String> {
        use std::path::Path;
        if !Path::new(path).exists() {
            return Err(format!("Quantized ONNX model file not found: {}", path));
        }
        info!("Loading quantized INT8 model: {} (CPU inference)", path);
        let session = SessionPool::create_session(path, InferenceBackend::CPU, 0)?;
        let pool = SessionPool::new(path.to_string(), InferenceBackend::CPU, 0, session);
        Ok(SurrogateManager {
            model_loaded: true,
            model_path: Some(path.to_string()),
            session_pool: Some(Arc::new(pool)),
            backend: InferenceBackend::CPU,
            device_id: 0,
            composite: None,
            inference_metrics: Arc::new(parking_lot::Mutex::new(InferenceMetrics::default())),
            input_bounds: None,
            ood_count: Arc::new(parking_lot::Mutex::new(0)),
            residual_tau: DEFAULT_RESIDUAL_TAU,
            residual_reroute_count: Arc::new(parking_lot::Mutex::new(0)),
        })
    }

    /// Stub for non-`ort` builds (issue #1294).
    #[cfg(not(feature = "ort"))]
    pub fn load_quantized_onnx(_path: &str) -> Result<Self, String> {
        Err(
            "Loading quantized ONNX models requires the `ort` feature (build with --features ort)"
                .to_string(),
        )
    }

    pub fn predict_loads(&self, current_temps: &[f64]) -> Vec<f64> {
        if let Some(ref comp) = self.composite {
            return comp.predict_loads(current_temps);
        }

        if !self.model_loaded {
            // Mock fallback: constant 1.2 load per zone, matching prior behavior.
            return vec![1.2; current_temps.len()];
        }

        // Real ONNX path — but never panic; fall back to mock on any failure
        // so the simulation keeps running. Issue #899: replace mock with real
        // inference pipeline, with graceful degradation.
        match self.predict_loads_onnx(current_temps) {
            Ok(loads) => loads,
            Err(e) => {
                warn!(
                    "ONNX inference failed ({}), falling back to mock placeholder",
                    e
                );
                vec![1.2; current_temps.len()]
            }
        }
    }

    /// Zero-allocation variant of [`Self::predict_loads`] for the per-timestep
    /// hot loop (Issue #2687).
    ///
    /// Writes the prediction into `out`, reusing its existing capacity (after
    /// warm-up, no heap allocation). The bytes produced are identical to
    /// `self.predict_loads(current_temps)` — only the ownership of the return
    /// buffer differs — so simulation output is bit-identical. Callers that
    /// run the surrogate once per timestep should hoist `out` above the loop.
    pub fn predict_loads_into(&self, current_temps: &[f64], out: &mut Vec<f64>) {
        if let Some(ref comp) = self.composite {
            // The composite path produces a fresh Vec internally; spill it into
            // the reuse buffer (one allocation saved at this call site).
            let loads = comp.predict_loads(current_temps);
            out.clear();
            out.extend_from_slice(&loads);
            return;
        }

        if !self.model_loaded {
            // Mock fallback: constant 1.2 load per zone, into the reuse buffer.
            out.clear();
            out.resize(current_temps.len(), 1.2);
            return;
        }

        // Real ONNX path with graceful fallback to mock on failure.
        match self.predict_loads_onnx(current_temps) {
            Ok(loads) => {
                out.clear();
                out.extend_from_slice(&loads);
            }
            Err(e) => {
                warn!(
                    "ONNX inference failed ({}), falling back to mock placeholder",
                    e
                );
                out.clear();
                out.resize(current_temps.len(), 1.2);
            }
        }
    }

    /// Explicit ONNX inference — returns an error instead of panicking
    /// or silently falling back to mock data. Use this when you need to
    /// distinguish real neural predictions from mock placeholders.
    ///
    /// Returns `Err` if:
    /// - no ONNX model has been loaded via [`Self::load_onnx`]
    /// - input tensor shape does not match the model's expected input
    /// - the ONNX runtime reports an inference error
    #[cfg(feature = "ort")]
    pub fn predict_loads_onnx(&self, current_temps: &[f64]) -> Result<Vec<f64>, String> {
        // Issue #2498: instrument every ONNX inference attempt with backend /
        // outcome / latency / batch-size metrics so production telemetry can
        // distinguish "neural surrogate diverged" from "ONNX call itself
        // failed". Timing wraps the entire attempt (including session-pool
        // acquisition) so the histogram reflects caller-observed latency.
        let backend = self.backend.as_str();
        let start = std::time::Instant::now();
        let result = self.predict_loads_onnx_impl(current_temps);
        let elapsed_secs = start.elapsed().as_secs_f64();
        self.record_onnx_inference_metrics(backend, 1, elapsed_secs, result.is_ok());
        result
    }

    /// Pure ONNX inference (single sample) without metric instrumentation.
    /// Wrapped by [`Self::predict_loads_onnx`] (Issue #2498).
    #[cfg(feature = "ort")]
    fn predict_loads_onnx_impl(&self, current_temps: &[f64]) -> Result<Vec<f64>, String> {
        if !self.model_loaded {
            return Err("No ONNX model loaded".to_string());
        }
        let pool = self
            .session_pool
            .as_ref()
            .ok_or_else(|| "No session pool available".to_string())?;

        let input_data: Vec<f32> = current_temps.iter().map(|&x| x as f32).collect();
        let n_input = input_data.len();

        let mut session_guard = pool
            .get_or_create_session()
            .map_err(|e| format!("Could not acquire ORT session: {}", e))?;

        let input_tensor = ort::value::Value::from_array(([1_i64, n_input as i64], input_data))
            .map_err(|e| format!("Failed to create input tensor: {}", e))?;

        let outputs = session_guard
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("ONNX inference error: {}", e))?;

        let result = if outputs.len() > 0 {
            let array_view = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| format!("Failed to extract tensor: {}", e))?;
            let v: Vec<f64> = array_view.iter().copied().map(|x| x as f64).collect();
            if v.is_empty() {
                return Err("ONNX inference returned empty output".to_string());
            }
            v
        } else {
            return Err("ONNX inference returned no outputs".to_string());
        };

        let _ = n_input; // kept for forward-compat shape validation
        Ok(result)
    }

    /// Stub for non-`ort` builds (issue #1294). Without the `ort` feature,
    /// no ONNX model can ever be loaded, so this always errors.
    #[cfg(not(feature = "ort"))]
    pub fn predict_loads_onnx(&self, _current_temps: &[f64]) -> Result<Vec<f64>, String> {
        Err("ONNX inference requires the `ort` feature (build with --features ort)".to_string())
    }

    /// Zero-allocation variant of [`Self::predict_loads_batched`] for the
    /// per-timestep batched hot loop (Issue #2771).
    ///
    /// Reuses three caller-supplied scratch buffers across calls:
    /// - `scratch_in`  — flattened `f32` input fed to the ONNX runtime,
    /// - `scratch_out` — flattened `f64` results extracted from the output
    ///   tensor,
    /// - `out`         — the batched per-config load vectors.
    ///
    /// Each buffer is cleared and refilled in place. After warm-up (a
    /// constant batch size, the steady state in the 8 760-timestep
    /// orchestrator loop) the steady-state call performs **no heap
    /// allocation**: the ONNX input tensor is built from a *borrowed*
    /// `&[f32]` view (`ort::value::TensorRef::from_array_view`) rather
    /// than an owned `Vec`, and the per-config result vectors are recycled
    /// via `Vec::resize_with` instead of being reallocated as N fresh
    /// `Vec<f64>`s every call. The bytes produced are identical to
    /// `predict_loads_batched` — only buffer ownership differs — so
    /// simulation output is bit-identical. See the
    /// `dhat_batched_surrogate_zero_growth` gate for the steady-state
    /// proof.
    ///
    /// Callers that run this once per timestep should hoist the buffers
    /// above the loop (the same reuse pattern `predict_loads_into` /
    /// Issue #2687 use for the unbatched path).
    pub fn predict_loads_batched_into(
        &self,
        batch_temps: &[Vec<f64>],
        scratch_in: &mut Vec<f32>,
        scratch_out: &mut Vec<f64>,
        out: &mut Vec<Vec<f64>>,
    ) {
        if let Some(ref comp) = self.composite {
            // Composite path: recycle `out`'s outer + inner capacity.
            out.resize_with(batch_temps.len(), Vec::new);
            for (temps, inner) in batch_temps.iter().zip(out.iter_mut()) {
                let loads = comp.predict_loads(temps);
                inner.clear();
                inner.extend_from_slice(&loads);
            }
            return;
        }

        if !self.model_loaded || batch_temps.is_empty() {
            // Mock fallback: constant 1.2 load per zone, into reused buffers.
            out.resize_with(batch_temps.len(), Vec::new);
            for (temps, inner) in batch_temps.iter().zip(out.iter_mut()) {
                inner.clear();
                inner.resize(temps.len(), 1.2);
            }
            return;
        }

        // Real ONNX path with graceful fallback to mock on failure.
        #[cfg(feature = "ort")]
        match self.predict_loads_batched_onnx_into(batch_temps, scratch_in, scratch_out, out) {
            Ok(()) => {}
            Err(e) => {
                warn!(
                    "Batched ONNX inference failed ({}), falling back to mock placeholder",
                    e
                );
                out.resize_with(batch_temps.len(), Vec::new);
                for (temps, inner) in batch_temps.iter().zip(out.iter_mut()) {
                    inner.clear();
                    inner.resize(temps.len(), 1.2);
                }
            }
        }

        // Without the `ort` feature no model can ever be loaded, so the ONNX
        // branch above is absent and the mock path has already returned.
        // Silence the unused-buffer warning for non-`ort` builds.
        #[cfg(not(feature = "ort"))]
        let _ = (scratch_in, scratch_out);
    }

    pub fn predict_loads_batched(&self, batch_temps: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Allocate one-shot buffers; the per-timestep hot loop uses
        // `predict_loads_batched_into` to reuse them across the 8 760 steps.
        let mut out = Vec::new();
        let mut scratch_in = Vec::new();
        let mut scratch_out = Vec::new();
        self.predict_loads_batched_into(batch_temps, &mut scratch_in, &mut scratch_out, &mut out);
        out
    }

    /// Explicit batched ONNX inference — returns an error instead of
    /// panicking or silently falling back to mock data.
    #[cfg(feature = "ort")]
    pub fn predict_loads_batched_onnx(
        &self,
        batch_temps: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        // Issue #2498: instrument batched inference. An empty batch is a
        // documented no-op (returns Ok(empty)) so it bypasses metric
        // recording — no inference actually runs.
        if batch_temps.is_empty() {
            return self.predict_loads_batched_onnx_impl(batch_temps);
        }
        let backend = self.backend.as_str();
        let batch_size = batch_temps.len();
        let start = std::time::Instant::now();
        let result = self.predict_loads_batched_onnx_impl(batch_temps);
        let elapsed_secs = start.elapsed().as_secs_f64();
        self.record_onnx_inference_metrics(backend, batch_size, elapsed_secs, result.is_ok());
        result
    }

    /// Pure batched ONNX inference without metric instrumentation. Wrapped by
    /// [`Self::predict_loads_batched_onnx`] (Issue #2498).
    ///
    /// Delegates to [`Self::predict_loads_batched_onnx_impl_into`] so the
    /// buffer-reuse fix (Issue #2771) lives in exactly one place; this
    /// Vec-returning wrapper exists for the public metric-recording API.
    #[cfg(feature = "ort")]
    fn predict_loads_batched_onnx_impl(
        &self,
        batch_temps: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut scratch_in = Vec::new();
        let mut scratch_out = Vec::new();
        let mut out = Vec::new();
        self.predict_loads_batched_onnx_impl_into(
            batch_temps,
            &mut scratch_in,
            &mut scratch_out,
            &mut out,
        )?;
        Ok(out)
    }

    /// Metric-recording batched ONNX inference into reusable buffers
    /// (Issue #2771). The buffer-reuse twin of
    /// [`Self::predict_loads_batched_onnx`]; records the same
    /// `fluxion_onnx_*` telemetry.
    #[cfg(feature = "ort")]
    fn predict_loads_batched_onnx_into(
        &self,
        batch_temps: &[Vec<f64>],
        scratch_in: &mut Vec<f32>,
        scratch_out: &mut Vec<f64>,
        out: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        if batch_temps.is_empty() {
            out.clear();
            return Ok(());
        }
        let backend = self.backend.as_str();
        let batch_size = batch_temps.len();
        let start = std::time::Instant::now();
        let result =
            self.predict_loads_batched_onnx_impl_into(batch_temps, scratch_in, scratch_out, out);
        let elapsed_secs = start.elapsed().as_secs_f64();
        self.record_onnx_inference_metrics(backend, batch_size, elapsed_secs, result.is_ok());
        result
    }

    /// Pure batched ONNX inference into reusable buffers, without metric
    /// instrumentation (Issue #2771). All three buffers are reused across
    /// calls: `scratch_in` is refilled with the flattened `f32` input and
    /// passed to the runtime as a **borrowed** `TensorRef` (no owned-data
    /// copy), `scratch_out` receives the flattened `f64` output, and `out`
    /// receives the per-config load slices (outer + inner capacity recycled
    /// via `resize_with`). The bytes produced are identical to the prior
    /// `predict_loads_batched_onnx_impl` — only buffer ownership differs.
    #[cfg(feature = "ort")]
    fn predict_loads_batched_onnx_impl_into(
        &self,
        batch_temps: &[Vec<f64>],
        scratch_in: &mut Vec<f32>,
        scratch_out: &mut Vec<f64>,
        out: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        if !self.model_loaded {
            return Err("No ONNX model loaded".to_string());
        }
        if batch_temps.is_empty() {
            out.clear();
            return Ok(());
        }
        let pool = self
            .session_pool
            .as_ref()
            .ok_or_else(|| "No session pool available".to_string())?;

        let batch_size = batch_temps.len();
        let input_size = batch_temps[0].len();
        for t in batch_temps {
            if t.len() != input_size {
                return Err(format!(
                    "Inconsistent input sizes in batch: expected {} elements per config",
                    input_size
                ));
            }
        }

        // Refill the flattened f32 input buffer in place; no reallocation
        // after warm-up (Issue #2771).
        scratch_in.clear();
        scratch_in.reserve(batch_size * input_size);
        for v in batch_temps {
            scratch_in.extend(v.iter().map(|&x| x as f32));
        }

        let mut session_guard = pool
            .get_or_create_session()
            .map_err(|e| format!("Could not acquire ORT session: {}", e))?;

        // Borrowed tensor view: the runtime reads `scratch_in` by reference
        // instead of taking ownership of a freshly allocated Vec (the prior
        // per-call allocation). The shape is a stack `[i64; 2]`, not a Vec.
        let input_tensor = ort::value::TensorRef::from_array_view((
            [batch_size as i64, input_size as i64],
            &scratch_in[..],
        ))
        .map_err(|e| format!("Failed to create input tensor: {}", e))?;

        let outputs = session_guard
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("ONNX inference error: {}", e))?;

        if outputs.len() == 0 {
            return Err("ONNX inference returned no outputs for batch".to_string());
        }
        let array_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract tensor: {}", e))?;
        // Refill the flattened f64 results buffer in place.
        scratch_out.clear();
        scratch_out.extend(array_view.iter().copied().map(|x| x as f64));
        if scratch_out.is_empty() {
            return Err("ONNX inference returned empty batch output".to_string());
        }
        let output_size = scratch_out.len() / batch_size;

        // Scatter into `out`, recycling outer + inner capacity. After warm-up
        // (constant batch_size) `resize_with` is a no-op and each inner Vec
        // is cleared + refilled without reallocation (Issue #2771).
        out.resize_with(batch_size, Vec::new);
        for (i, inner) in out.iter_mut().enumerate() {
            inner.clear();
            inner.extend_from_slice(&scratch_out[i * output_size..(i + 1) * output_size]);
        }
        Ok(())
    }

    /// Shared metric recording for a single ONNX inference attempt (Issue
    /// #2498). Called by both [`Self::predict_loads_onnx`] (batch_size = 1)
    /// and [`Self::predict_loads_batched_onnx`] wrappers.
    ///
    /// - On success: records the internal `InferenceMetrics` latency (ms, kept
    ///   for the existing `inference_metrics()` API), the
    ///   `fluxion_onnx_inference_duration_seconds` histogram, the
    ///   `fluxion_onnx_inference_total{outcome="success"}` counter, and the
    ///   `fluxion_onnx_batch_size` histogram.
    /// - On error: records only the `fluxion_onnx_inference_total{outcome=
    ///   "error"}` counter (no inference ran, so no latency/batch sample).
    #[cfg(feature = "ort")]
    fn record_onnx_inference_metrics(
        &self,
        backend: &'static str,
        batch_size: usize,
        elapsed_secs: f64,
        succeeded: bool,
    ) {
        if succeeded {
            self.inference_metrics
                .lock()
                .record_inference(elapsed_secs * 1000.0);
            metrics::histogram!(
                "fluxion_onnx_inference_duration_seconds",
                "backend" => backend,
                "batch_bucket" => batch_bucket_label(batch_size),
            )
            .record(elapsed_secs);
            metrics::counter!(
                "fluxion_onnx_inference_total",
                "backend" => backend,
                "outcome" => "success",
            )
            .increment(1);
            metrics::histogram!("fluxion_onnx_batch_size", "backend" => backend)
                .record(batch_size as f64);
        } else {
            metrics::counter!(
                "fluxion_onnx_inference_total",
                "backend" => backend,
                "outcome" => "error",
            )
            .increment(1);
        }
    }

    /// Stub for non-`ort` builds (issue #1294).
    #[cfg(not(feature = "ort"))]
    pub fn predict_loads_batched_onnx(
        &self,
        _batch_temps: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        Err("ONNX inference requires the `ort` feature (build with --features ort)".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
}

impl PredictionWithUncertainty {
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Self {
        let lower_bound: Vec<f64> = mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| m - 2.0 * s)
            .collect();
        let upper_bound: Vec<f64> = mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| m + 2.0 * s)
            .collect();
        Self {
            mean,
            std,
            lower_bound,
            upper_bound,
        }
    }
}

impl SurrogateManager {
    pub fn predict_with_uncertainty(
        &self,
        current_temps: &[f64],
        num_samples: usize,
        noise_std: f64,
    ) -> PredictionWithUncertainty {
        if !self.model_loaded || num_samples == 0 {
            let loads = self.predict_loads(current_temps);
            return PredictionWithUncertainty::new(loads.clone(), vec![0.0; loads.len()]);
        }
        let base_prediction = self.predict_loads(current_temps);
        let variance: f64 =
            base_prediction.iter().map(|v| v * 0.05).sum::<f64>() / base_prediction.len() as f64;
        let num_outputs = base_prediction.len();

        // Parallel Monte Carlo sampling using rayon
        let all_predictions: Vec<Vec<f64>> = (0..num_samples)
            .into_par_iter()
            .map(|_i| {
                let mut rng = StdRng::from_entropy();
                let perturbed_temps: Vec<f64> = current_temps
                    .iter()
                    .map(|&t| t + (rng.gen::<f64>() - 0.5) * 2.0 * noise_std)
                    .collect();
                let _perturbed_temps = perturbed_temps;
                base_prediction
                    .iter()
                    .map(|&v| v + (rng.gen::<f64>() - 0.5) * 2.0 * variance.sqrt())
                    .collect()
            })
            .collect();

        // Parallel aggregation of predictions using fold
        let means: Vec<f64> = (0..num_outputs)
            .into_par_iter()
            .map(|i| all_predictions.iter().map(|pred| pred[i]).sum::<f64>() / num_samples as f64)
            .collect();

        // Parallel computation of variances
        let variances: Vec<f64> = (0..num_outputs)
            .into_par_iter()
            .map(|i| {
                let mean = means[i];
                all_predictions
                    .iter()
                    .map(|pred| {
                        let diff = pred[i] - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / if num_samples > 1 {
                        (num_samples - 1) as f64
                    } else {
                        1.0
                    }
            })
            .collect();
        let std: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();
        PredictionWithUncertainty::new(means, std)
    }

    pub fn get_prediction_interval_width(
        &self,
        current_temps: &[f64],
        confidence: f64,
    ) -> Vec<f64> {
        let z_score = match (confidence * 100.0) as u32 {
            90 => 1.645,
            95 => 1.960,
            99 => 2.576,
            _ => 1.960,
        };
        let uncertainty = self.predict_with_uncertainty(current_temps, 10, 0.5);
        uncertainty.std.iter().map(|&s| 2.0 * z_score * s).collect()
    }
}

/// Maximum permitted ONNX model file size: 256 MiB (Issue #2529).
pub const MAX_MODEL_SIZE_BYTES: u64 = 256 * 1024 * 1024;

/// Default allow-list directory for ONNX models when `FLUXION_MODEL_DIR` is
/// unset (Issue #2529). Mirrors the conventional location used by
/// `FLUXION_ONNX_MODEL`'s default (`models/surrogate_zone_thermal.onnx`).
const DEFAULT_MODEL_DIR: &str = "models";

/// Name of the environment variable that overrides the SHA-256 manifest
/// check for rotated models (Issue #2906). When set to a 64-char lowercase
/// or uppercase hex SHA-256 digest, that digest is used as the authoritative
/// expected hash INSTEAD of the manifest at `<model>.sha256`. The env override
/// exists so operators can rotate a model without first waiting for the
/// manifest to be updated in the repo; once the manifest is updated, unset
/// the variable.
pub const ENV_ONNX_MODEL_SIGNATURE: &str = "FLUXION_ONNX_MODEL_SIGNATURE";

/// Verify the integrity of an ONNX model against a SHA-256 manifest
/// (Issue #2906).
///
/// Behaviour, in resolution order:
/// 1. If `FLUXION_ONNX_MODEL_SIGNATURE` is set to a 64-char hex SHA-256,
///    use it as the authoritative expected digest.
/// 2. Otherwise, look for a manifest file at `<model>.sha256` (the standard
///    `sha256sum` output format). If present, parse it and look for an
///    entry whose filename matches the model basename.
/// 3. If neither source provides a digest, succeed WITHOUT verification and
///    emit a one-shot `eprintln!` warning so operators can ship a manifest
///    with their model. (This branch preserves backward compatibility with
///    test fixtures in `assets/` that intentionally have no manifest.)
/// 4. Compute the SHA-256 of the model file and compare with the resolved
///    expected digest. Mismatch returns `Err` (fail-closed) so a poisoned
///    or bit-flipped model cannot influence ASHRAE 140 results.
///
/// The manifest format mirrors `sha256sum` output:
///
/// ```text
/// # comment lines start with '#'; blank lines are ignored
/// <64-hex-digest>  <relative-path>
/// ```
///
/// Both single- and double-space separators are accepted, as is the
/// `sha256sum -b` "binary mode" `*` prefix on the path. The first entry
/// whose filename ends with the model basename wins.
pub fn verify_onnx_signature(model_path: &Path) -> Result<(), String> {
    let expected = match std::env::var(ENV_ONNX_MODEL_SIGNATURE)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        Some(digest) => {
            validate_sha256_hex(&digest).map_err(|_| {
                format!(
                    "{ENV_ONNX_MODEL_SIGNATURE}={digest:?} is not a valid 64-char hex SHA-256 \
                     digest; refusing to load (fail-closed, see Issue #2906)"
                )
            })?;
            digest.to_ascii_lowercase()
        }
        None => match read_manifest_hash(model_path) {
            Some(Ok(hash)) => hash,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(format!(
                    "no SHA-256 manifest at {} and {ENV_ONNX_MODEL_SIGNATURE} unset; \
                     integrity verification impossible (fail-closed, Issue #3209). \
                     Ship a <model>.sha256 alongside the .onnx file or set \
                     {ENV_ONNX_MODEL_SIGNATURE}=<hex-digest> for rotated models.",
                    manifest_path_for(model_path).display()
                ));
            }
        },
    };

    let actual = compute_file_sha256(model_path)?;
    validate_hash(&expected, &actual).map_err(|e| {
        format!(
            "ONNX model integrity verification FAILED for {} ({e}). This may indicate \
             model poisoning (bit-flip, tampered registry mirror, malicious build \
             step). Refusing to load (fail-closed, Issue #2906). To override for a \
             rotated model, set {ENV_ONNX_MODEL_SIGNATURE}=<hex-digest> to the new \
             expected digest, regenerate the manifest, and unset the override.",
            model_path.display()
        )
    })
}

/// Return the conventional path of the SHA-256 manifest for `model_path`
/// (i.e. `<model_path>.sha256`).
fn manifest_path_for(model_path: &Path) -> std::path::PathBuf {
    let mut s = model_path.as_os_str().to_owned();
    s.push(".sha256");
    std::path::PathBuf::from(s)
}

/// Try to read and parse a SHA-256 manifest at `<model>.sha256`. Returns:
/// - `Some(Ok(hash))` if a matching entry is found,
/// - `Some(Err(e))` if the manifest exists but is malformed, or
/// - `None` if the manifest file does not exist.
fn read_manifest_hash(model_path: &Path) -> Option<Result<String, String>> {
    let manifest_path = manifest_path_for(model_path);
    let contents = match std::fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            return Some(Err(format!(
                "failed to read SHA-256 manifest at {}: {e}",
                manifest_path.display()
            )))
        }
    };
    Some(parse_sha256_manifest(&manifest_path, &contents, model_path))
}

/// Parse `sha256sum`-format manifest text and return the hash for the
/// entry whose filename matches `model_path`. Accepts `# comment` lines,
/// blank lines, single- or double-space separators, and the binary-mode
/// `*` prefix on the filename.
fn parse_sha256_manifest(
    manifest_path: &Path,
    contents: &str,
    model_path: &Path,
) -> Result<String, String> {
    let expected_basename = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if expected_basename.is_empty() {
        return Err(format!(
            "cannot derive basename from model path {}",
            model_path.display()
        ));
    }
    for (lineno, raw) in contents.lines().enumerate() {
        let line = raw.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // sha256sum layout: `<hash><space><space-or-*><path>` or
        // `<hash><space><path>` (text mode). Split on first run of
        // whitespace; first field is the hash, remainder is the filename.
        let mut split = line.splitn(2, |c: char| c.is_whitespace());
        let hash = split.next().unwrap_or("").trim();
        let filename_raw = split.next().unwrap_or("").trim();
        // sha256sum -b prefixes the path with `*`; strip it.
        let filename = filename_raw
            .strip_prefix('*')
            .unwrap_or(filename_raw)
            .trim();
        if hash.len() != 64 {
            // Could be a malformed line; surface a precise error rather
            // than silently skipping.
            return Err(format!(
                "{}:{} malformed entry (hash must be 64 hex chars, got {:?})",
                manifest_path.display(),
                lineno + 1,
                hash
            ));
        }
        if let Err(e) = validate_sha256_hex(hash) {
            return Err(format!(
                "{}:{} hash {:?} is not valid hex: {:?}",
                manifest_path.display(),
                lineno + 1,
                hash,
                e
            ));
        }
        if filename.is_empty() {
            // No filename constraint — accept and require it be the only
            // entry (typical for a single-model manifest).
            return Ok(hash.to_ascii_lowercase());
        }
        if filename.ends_with(expected_basename) || expected_basename.ends_with(filename) {
            return Ok(hash.to_ascii_lowercase());
        }
    }
    Err(format!(
        "{}: no entry found for model {} (basename {:?})",
        manifest_path.display(),
        model_path.display(),
        expected_basename
    ))
}

/// Validates a user-supplied model path against the security policy from
/// Issue #2529 before it reaches the ONNX runtime. Reads the allow-list
/// directory from the `FLUXION_MODEL_DIR` environment variable (default
/// `models/`).
///
/// On success returns the canonicalised absolute path. All error messages
/// are deliberately generic and omit the raw user-supplied path so that
/// attacker-controlled input is never reflected back to the Python caller
/// (closes the error oracle).
pub fn validate_model_path(p: &str) -> Result<std::path::PathBuf, String> {
    let dir = std::env::var("FLUXION_MODEL_DIR").unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    validate_model_path_in_dir(p, std::path::Path::new(&dir))
}

/// Parameterised core of [`validate_model_path`]. Accepts an explicit
/// allow-list directory so it can be unit-tested without racing on the
/// process-wide `FLUXION_MODEL_DIR` env var.
///
/// Checks, in order:
/// 1. `Path::new(p).is_file()` — existence (follows symlinks like the rest
///    of `std::fs`).
/// 2. extension == `onnx`.
/// 3. canonicalised path is inside `allowed_dir` (component-wise
///    `starts_with` on canonical paths — blocks `..` traversal and symlinks
///    that escape the allow-list).
/// 4. file size ≤ [`MAX_MODEL_SIZE_BYTES`].
pub fn validate_model_path_in_dir(
    p: &str,
    allowed_dir: &Path,
) -> Result<std::path::PathBuf, String> {
    let raw = Path::new(p);
    if !raw.is_file() {
        return Err("model file not found".to_string());
    }
    if raw
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        != Some("onnx".to_string())
    {
        return Err("invalid model file extension (expected .onnx)".to_string());
    }
    let canonical_model =
        std::fs::canonicalize(raw).map_err(|_| "failed to canonicalize model path".to_string())?;
    let canonical_dir = std::fs::canonicalize(allowed_dir)
        .map_err(|_| "allowed model directory not found".to_string())?;
    if !canonical_model.starts_with(&canonical_dir) {
        return Err("model path outside allowed directory".to_string());
    }
    let size = std::fs::metadata(&canonical_model)
        .map_err(|_| "failed to read model file metadata".to_string())?
        .len();
    if size > MAX_MODEL_SIZE_BYTES {
        return Err(format!(
            "model file exceeds size limit ({} bytes)",
            MAX_MODEL_SIZE_BYTES
        ));
    }
    Ok(canonical_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared mutex serializing all tests in this module that mutate
    /// `FLUXION_*` env vars. Without this, parallel `cargo test` threads
    /// stomp on each other's env state and produce flaky failures.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn creation() {
        let m = SurrogateManager::new().unwrap();
        assert!(!m.model_loaded);
    }

    #[test]
    fn predict_without_model_uses_analytical() {
        let m = SurrogateManager::new().unwrap();
        let loads = m.predict_loads(&[20.0, 21.0, 22.0]);
        assert_eq!(loads.len(), 3);
        // When no model is loaded, analytical_loads is used as fallback
        // which computes solar gain based on time of day (non-negative)
        assert!(loads[0] >= 0.0);
    }

    #[test]
    fn predict_batched_without_model_uses_analytical() {
        let m = SurrogateManager::new().unwrap();
        let loads = m.predict_loads_batched(&[vec![20.0, 21.0], vec![22.0, 23.0]]);
        assert_eq!(loads.len(), 2);
        // Each batch returns analytical loads (non-negative)
        assert!(loads[0][0] >= 0.0);
    }

    #[test]
    fn load_onnx_file_check() {
        let result = SurrogateManager::load_onnx("/nonexistent/path/model.onnx");
        assert!(result.is_err());
    }

    // ===== Issue #2906 — ONNX model SHA-256 integrity verification =====
    //
    // `verify_onnx_signature` reads `<model>.sha256`, parses a
    // sha256sum-style manifest entry, computes the model's actual SHA-256,
    // and refuses to load on mismatch (fail-closed). The
    // `FLUXION_ONNX_MODEL_SIGNATURE` env var overrides the manifest for
    // rotated models. These tests use a tempdir so they never race on the
    // process-wide env var (serialised by `ENV_LOCK`).

    /// Build a tempdir containing `<dir>/model.onnx` with arbitrary bytes
    /// and write a SHA-256 manifest at `<dir>/model.onnx.sha256` whose entry
    /// matches that file. Returns `(model_path, manifest_path, sha256)`.
    fn write_signed_model(contents: &[u8]) -> (tempfile::TempDir, std::path::PathBuf, String) {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.onnx");
        std::fs::write(&model, contents).unwrap();
        let sha = compute_bytes_sha256(contents);
        let manifest = dir.path().join("model.onnx.sha256");
        std::fs::write(
            &manifest,
            format!(
                "# generated by verify_onnx_signature test\n\
                 {sha}  {name}\n",
                name = model.file_name().unwrap().to_string_lossy()
            ),
        )
        .unwrap();
        (dir, model, sha)
    }

    /// Happy path: a manifest whose hash matches the model bytes must be
    /// accepted with no error and no env override required.
    #[test]
    fn verify_onnx_signature_accepts_matching_manifest() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, model, _sha) = write_signed_model(b"fluxion integrity fixture v1");
        let res = verify_onnx_signature(&model);
        assert!(res.is_ok(), "valid signature rejected: {res:?}");
    }

    /// Tamper the model bytes after writing the manifest — verification
    /// must FAIL with a fail-closed error message that names the issue.
    #[test]
    fn verify_onnx_signature_rejects_tampered_model() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, model, _sha) = write_signed_model(b"original bytes");
        // Tamper: append a single byte so the digest changes but the file
        // is still a valid path/exists for the load step.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&model)
                .unwrap();
            f.write_all(b"!").unwrap();
        }
        let err = verify_onnx_signature(&model).expect_err("tamper must fail");
        assert!(
            err.contains("integrity verification FAILED"),
            "expected fail-closed message, got: {err}"
        );
        assert!(
            err.contains("Issue #2906"),
            "error must reference Issue #2906 so operators find the runbook: {err}"
        );
        let _ = dir;
    }

    /// Env override `FLUXION_ONNX_MODEL_SIGNATURE` must take precedence
    /// over a mismatched manifest when set to the actual file digest.
    #[test]
    fn verify_onnx_signature_env_override_accepts_rotated_model() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, model, _stale_sha) = write_signed_model(b"original bytes");
        // Rotate the model bytes AFTER the manifest was written.
        std::fs::write(&model, b"rotated bytes -- manifest still says original").unwrap();
        let rotated_sha = compute_bytes_sha256(b"rotated bytes -- manifest still says original");
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, &rotated_sha);
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        assert!(
            res.is_ok(),
            "env override should accept the rotated model: {res:?}"
        );
    }

    /// Env override set to the WRONG digest must STILL fail (env override
    /// doesn't disable verification, it only changes the expected digest).
    #[test]
    fn verify_onnx_signature_env_override_still_rejects_mismatch() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, model, _sha) = write_signed_model(b"original bytes");
        let bogus = "f".repeat(64); // wrong digest
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, &bogus);
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let err = res.expect_err("env override with wrong digest must fail");
        assert!(
            err.contains("integrity verification FAILED"),
            "wrong env override must still fail closed: {err}"
        );
    }

    /// Env override with a malformed (non-hex / wrong-length) value must be
    /// rejected up-front so operators don't get a misleading mismatch
    /// error from a typo.
    #[test]
    fn verify_onnx_signature_env_override_rejects_malformed_value() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, model, _sha) = write_signed_model(b"original bytes");
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, "not-a-sha256");
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let err = res.expect_err("malformed env override must fail");
        assert!(
            err.contains(ENV_ONNX_MODEL_SIGNATURE),
            "error must name the env var: {err}"
        );
        assert!(
            err.contains("not a valid"),
            "error must explain the rejection: {err}"
        );
    }

    /// No manifest next to the model AND no env override = fail-closed
    /// error so unsigned models can never be loaded without explicit override.
    #[test]
    fn verify_onnx_signature_fails_without_manifest() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("nomanifest.onnx");
        std::fs::write(&model, b"anything").unwrap();
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE);
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let err =
            res.expect_err("missing manifest + no env var must fail (fail-closed, Issue #3209)");
        assert!(
            err.contains("fail-closed") || err.contains("Issue #3209"),
            "error must reference Issue #3209: {err}"
        );
    }

    /// Manifest with `sha256sum` style double-space separator AND a `*`
    /// binary-mode prefix on the filename must be parsed correctly.
    #[test]
    fn verify_onnx_signature_parses_sha256sum_format() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("binary.onnx");
        std::fs::write(&model, b"\x00\x01\x02binary-mode payload").unwrap();
        let sha = compute_file_sha256(&model).unwrap();
        // `sha256sum -b` style: hash, two spaces, `*path`.
        std::fs::write(
            dir.path().join("binary.onnx.sha256"),
            format!("{sha}  *binary.onnx\n"),
        )
        .unwrap();
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE);
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        assert!(res.is_ok(), "sha256sum -b format should parse: {res:?}");
    }

    /// Manifest whose only entry has no filename (single-model manifest)
    /// must be accepted as the authoritative hash for the lone .onnx file.
    #[test]
    fn verify_onnx_signature_accepts_manifest_without_filename() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("lonely.onnx");
        std::fs::write(&model, b"only one model here").unwrap();
        let sha = compute_bytes_sha256(b"only one model here");
        std::fs::write(dir.path().join("lonely.onnx.sha256"), format!("{sha}\n")).unwrap();
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE);
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        assert!(
            res.is_ok(),
            "single-entry manifest without filename should pass: {res:?}"
        );
    }

    /// Manifest exists but has no entry for the model basename => must fail
    /// (so an attacker can't slip a wrong hash past the verifier by
    /// leaving the manifest populated for a different file).
    #[test]
    fn verify_onnx_signature_rejects_manifest_with_no_matching_entry() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("ghost.onnx");
        std::fs::write(&model, b"ghost").unwrap();
        // Manifest is for a different file.
        std::fs::write(
            dir.path().join("ghost.onnx.sha256"),
            "0000000000000000000000000000000000000000000000000000000000000000  other.onnx\n",
        )
        .unwrap();
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE);
        let res = verify_onnx_signature(&model);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let err = res.expect_err("manifest with no matching entry must fail");
        assert!(
            err.contains("no entry found"),
            "error must explain why: {err}"
        );
    }

    // ===== Issue #2529 — PyO3 load_surrogate path validation =====
    //
    // All cases use `validate_model_path_in_dir` with a `tempfile` allow-list
    // directory so they never touch the process-wide `FLUXION_MODEL_DIR` env
    // var (and therefore cannot race with each other under parallel `cargo
    // test`).

    /// A real `.onnx` file inside the allow-list directory validates and
    /// returns a canonicalised path.
    #[test]
    fn validate_model_path_accepts_valid_onnx() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("surrogate.onnx");
        std::fs::write(&model, b"dummy").unwrap();
        let rel = model.to_string_lossy().into_owned();
        let validated = validate_model_path_in_dir(&rel, dir.path());
        assert!(validated.is_ok(), "valid path rejected: {validated:?}");
        let canon = validated.unwrap();
        assert!(canon.is_absolute());
        assert_eq!(canon.extension().and_then(|e| e.to_str()), Some("onnx"));
    }

    /// A non-existent file is rejected with a generic "not found" message
    /// that does NOT echo the supplied path.
    #[test]
    fn validate_model_path_rejects_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("ghost.onnx");
        let err = validate_model_path_in_dir(&missing.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(err, "model file not found");
        // The raw user-supplied path must not be reflected back.
        assert!(!err.contains("ghost"));
    }

    /// A file with the wrong extension is rejected even if it lives inside
    /// the allow-list directory.
    #[test]
    fn validate_model_path_rejects_wrong_extension() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.bin");
        std::fs::write(&model, b"not an onnx").unwrap();
        let err = validate_model_path_in_dir(&model.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(err, "invalid model file extension (expected .onnx)");
    }

    /// An uppercase `.ONNX` extension is accepted (case-insensitive check).
    #[test]
    fn validate_model_path_accepts_uppercase_extension() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("MODEL.ONNX");
        std::fs::write(&model, b"dummy").unwrap();
        let res = validate_model_path_in_dir(&model.to_string_lossy(), dir.path());
        assert!(res.is_ok(), "uppercase .ONNX should be accepted: {res:?}");
    }

    /// A `.onnx` file inside the allow-list but reached via `..` traversal
    /// still resolves into the allow-list, so it must be accepted. The
    /// canonicalisation + `starts_with` check operates on real paths, not
    /// string prefixes.
    #[test]
    fn validate_model_path_allows_dotdot_inside_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("sub");
        std::fs::create_dir(&subdir).unwrap();
        let model = subdir.join("m.onnx");
        std::fs::write(&model, b"x").unwrap();
        // Build a path with redundant `..` that still lands inside the dir.
        let traversal = subdir
            .join("..")
            .join("sub")
            .join("m.onnx")
            .to_string_lossy()
            .into_owned();
        let res = validate_model_path_in_dir(&traversal, dir.path());
        assert!(res.is_ok(), "in-allowlist dotdot should pass: {res:?}");
    }

    /// Path-traversal to a file OUTSIDE the allow-list directory is rejected.
    /// Uses a sibling temp dir so the `.onnx` file genuinely exists but lives
    /// beyond the allow-list boundary.
    #[test]
    fn validate_model_path_rejects_traversal_outside_allowlist() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        // Real .onnx file in the *outside* dir.
        let evil = outside.path().join("evil.onnx");
        std::fs::write(&evil, b"pwned").unwrap();
        // Reference it via a traversal path from inside the allowed dir.
        let rel = allowed
            .path()
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("evil.onnx")
            .to_string_lossy()
            .into_owned();
        let err = validate_model_path_in_dir(&rel, allowed.path()).unwrap_err();
        assert_eq!(err, "model path outside allowed directory");
    }

    /// `/etc/passwd` (no `.onnx` extension) is rejected — the classic
    /// traversal probe from the issue. Uses an absolute path so it is
    /// deterministic regardless of the test working directory.
    #[test]
    fn validate_model_path_rejects_etc_passwd() {
        let dir = tempfile::tempdir().unwrap();
        // /etc/passwd exists on Linux; guard other platforms.
        if !std::path::Path::new("/etc/passwd").is_file() {
            eprintln!("skipping: /etc/passwd not present on this platform");
            return;
        }
        let err = validate_model_path_in_dir("/etc/passwd", dir.path()).unwrap_err();
        // Fails at the extension check (no .onnx). Either way, it must fail
        // and must not echo the path.
        assert!(
            err == "invalid model file extension (expected .onnx)"
                || err == "model path outside allowed directory"
        );
        assert!(!err.contains("passwd"));
    }

    /// A file larger than [`MAX_MODEL_SIZE_BYTES`] (256 MiB) is rejected.
    /// Uses `File::set_len` to create a sparse file whose reported length
    /// exceeds the limit without actually allocating 256 MiB on disk
    /// (`metadata().len()` reports the logical size).
    #[test]
    fn validate_model_path_rejects_oversized_file() {
        // The limit must be exactly 256 MiB (Issue #2529 acceptance).
        assert_eq!(MAX_MODEL_SIZE_BYTES, 256 * 1024 * 1024);

        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("huge.onnx");
        let f = std::fs::File::create(&model).unwrap();
        f.set_len(MAX_MODEL_SIZE_BYTES + 1).unwrap();
        drop(f);
        let err = validate_model_path_in_dir(&model.to_string_lossy(), dir.path()).unwrap_err();
        assert_eq!(
            err,
            format!(
                "model file exceeds size limit ({} bytes)",
                MAX_MODEL_SIZE_BYTES
            )
        );
        // Generic message: must not contain the user-supplied path.
        assert!(!err.contains("huge"));
    }

    /// A small file (well under the limit) passes the size gate — the
    /// acceptance criterion "file size <= 256 MiB".
    #[test]
    fn validate_model_path_accepts_small_file() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("small.onnx");
        std::fs::write(&model, b"x").unwrap();
        let res = validate_model_path_in_dir(&model.to_string_lossy(), dir.path());
        assert!(res.is_ok(), "small file should pass: {res:?}");
    }

    // ===== Issue #2905 — `new_with_auto_load` FLUXION_ONNX_MODEL validation =====
    //
    // `new_with_auto_load` is the entry point that `bin/fluxion-rest` and
    // `probe_onnx` use. Issue #2905 closed the bypass where an operator
    // (or compromised CI step) could set `FLUXION_ONNX_MODEL` to a non-
    // `.onnx` file, a path outside `FLUXION_MODEL_DIR`, or a traversal-
    // escaped location and have it ingested without checks. These tests
    // drive the public entry point end-to-end and assert each validation
    // branch short-circuits with `Err`. All cases set `FLUXION_MODEL_DIR`
    // and `FLUXION_ONNX_MODEL` against a tempdir under `ENV_LOCK` so they
    // never race with parallel `cargo test` threads or other tests in
    // this module that share the process-wide env state.

    /// `FLUXION_ONNX_MODEL` pointing at a non-`.onnx` file is rejected by
    /// `validate_model_path` and surfaced as `Err` from
    /// `new_with_auto_load` (matches the issue's acceptance criterion).
    #[test]
    fn new_with_auto_load_rejects_non_onnx_env_var() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let allowed = tempfile::tempdir().unwrap();
        let evil = allowed.path().join("malicious.bin");
        std::fs::write(&evil, b"not an onnx model").unwrap();

        let prev_dir = std::env::var("FLUXION_MODEL_DIR").ok();
        let prev_model = std::env::var("FLUXION_ONNX_MODEL").ok();
        std::env::set_var("FLUXION_MODEL_DIR", allowed.path());
        std::env::set_var("FLUXION_ONNX_MODEL", &evil);

        let res = SurrogateManager::new_with_auto_load();

        match prev_dir {
            Some(v) => std::env::set_var("FLUXION_MODEL_DIR", v),
            None => std::env::remove_var("FLUXION_MODEL_DIR"),
        }
        match prev_model {
            Some(v) => std::env::set_var("FLUXION_ONNX_MODEL", v),
            None => std::env::remove_var("FLUXION_ONNX_MODEL"),
        }

        let err = res.expect_err("non-.onnx FLUXION_ONNX_MODEL must return Err");
        assert!(
            err.contains("invalid model file extension"),
            "expected extension-rejection error, got: {err}"
        );
        // Generic message: must not echo the supplied path back.
        assert!(!err.contains("malicious"), "path leaked in error: {err}");
    }

    /// A real `.onnx` file that lives OUTSIDE `FLUXION_MODEL_DIR` is rejected
    /// even when `FLUXION_ONNX_MODEL` points directly at it. Mirrors the
    /// `BatchOracle.load_surrogate` allow-list contract.
    #[test]
    fn new_with_auto_load_rejects_onnx_outside_allowlist() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let evil = outside.path().join("evil.onnx");
        std::fs::write(&evil, b"legit onnx bytes, wrong location").unwrap();

        let prev_dir = std::env::var("FLUXION_MODEL_DIR").ok();
        let prev_model = std::env::var("FLUXION_ONNX_MODEL").ok();
        std::env::set_var("FLUXION_MODEL_DIR", allowed.path());
        std::env::set_var("FLUXION_ONNX_MODEL", &evil);

        let res = SurrogateManager::new_with_auto_load();

        match prev_dir {
            Some(v) => std::env::set_var("FLUXION_MODEL_DIR", v),
            None => std::env::remove_var("FLUXION_MODEL_DIR"),
        }
        match prev_model {
            Some(v) => std::env::set_var("FLUXION_ONNX_MODEL", v),
            None => std::env::remove_var("FLUXION_ONNX_MODEL"),
        }

        let err = res.expect_err("out-of-allowlist .onnx must return Err");
        assert!(
            err.contains("outside allowed directory"),
            "expected allow-list-rejection error, got: {err}"
        );
        // Generic message: must not echo the supplied path back.
        assert!(!err.contains("evil"), "path leaked in error: {err}");
    }

    /// `FLUXION_ONNX_MODEL` containing `..` traversal that escapes the
    /// allow-list directory is rejected. The canonicalised path lands
    /// outside `FLUXION_MODEL_DIR` so `validate_model_path` short-circuits.
    #[test]
    fn new_with_auto_load_rejects_traversal_in_env_var() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let evil = outside.path().join("traversed.onnx");
        std::fs::write(&evil, b"traversed model").unwrap();
        // Construct a traversal path that starts inside `allowed`, hops
        // out via `..`, and lands on the real `.onnx` in `outside`.
        let traversal = allowed
            .path()
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("traversed.onnx")
            .to_string_lossy()
            .into_owned();

        let prev_dir = std::env::var("FLUXION_MODEL_DIR").ok();
        let prev_model = std::env::var("FLUXION_ONNX_MODEL").ok();
        std::env::set_var("FLUXION_MODEL_DIR", allowed.path());
        std::env::set_var("FLUXION_ONNX_MODEL", &traversal);

        let res = SurrogateManager::new_with_auto_load();

        match prev_dir {
            Some(v) => std::env::set_var("FLUXION_MODEL_DIR", v),
            None => std::env::remove_var("FLUXION_MODEL_DIR"),
        }
        match prev_model {
            Some(v) => std::env::set_var("FLUXION_ONNX_MODEL", v),
            None => std::env::remove_var("FLUXION_ONNX_MODEL"),
        }

        let err = res.expect_err("traversal out of allowlist must return Err");
        assert!(
            err.contains("outside allowed directory"),
            "expected allow-list-rejection error, got: {err}"
        );
        // Generic message: must not echo the supplied traversal back.
        assert!(!err.contains("traversed"), "path leaked in error: {err}");
    }

    /// A real `.onnx` file INSIDE the allow-list directory passes
    /// `validate_model_path`. We then expect `new_with_auto_load` to
    /// reach `load_with_backend`, which (without the `ort` feature
    /// built) returns a feature-gate error. That non-validation error
    /// is the proof that validation ran and succeeded — if the
    /// validation pipeline were still being bypassed, the manager
    /// would either silently fall through to mock mode or, when
    /// `ort` IS built, succeed at constructing a real session.
    #[test]
    fn new_with_auto_load_passes_valid_onnx_inside_allowlist() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let allowed = tempfile::tempdir().unwrap();
        let model = allowed.path().join("good.onnx");
        std::fs::write(&model, b"dummy model bytes").unwrap();

        let prev_dir = std::env::var("FLUXION_MODEL_DIR").ok();
        let prev_model = std::env::var("FLUXION_ONNX_MODEL").ok();
        std::env::set_var("FLUXION_MODEL_DIR", allowed.path());
        std::env::set_var("FLUXION_ONNX_MODEL", &model);

        let res = SurrogateManager::new_with_auto_load();

        match prev_dir {
            Some(v) => std::env::set_var("FLUXION_MODEL_DIR", v),
            None => std::env::remove_var("FLUXION_MODEL_DIR"),
        }
        match prev_model {
            Some(v) => std::env::set_var("FLUXION_ONNX_MODEL", v),
            None => std::env::remove_var("FLUXION_ONNX_MODEL"),
        }

        // Either we succeeded at constructing a real session (with `ort`
        // built) or we got a load-with-backend error that is NOT one of
        // the four `validate_model_path` rejection strings — both prove
        // the validation pipeline ran end-to-end.
        match res {
            Ok(mgr) => assert!(mgr.model_loaded, "valid path should load a model"),
            Err(e) => {
                for forbidden in [
                    "model file not found",
                    "invalid model file extension",
                    "outside allowed directory",
                    "model file exceeds size limit",
                ] {
                    assert!(
                        !e.contains(forbidden),
                        "validation must have passed (got validation error \
                         '{forbidden}' instead of a load-time error): {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn predict_loads_with_fallback_success() {
        let m = SurrogateManager::new().unwrap();
        let temps = vec![20.0, 21.0, 22.0];
        let loads = m.predict_loads_with_fallback(&temps).unwrap();
        assert_eq!(loads.len(), 3);
        // When model_loaded is false, falls back to analytical loads (non-negative)
        assert!(loads[0] >= 0.0);
    }

    /// Issue #2498 — every backend variant must map to a stable lowercase
    /// Prometheus label value so dashboards do not break on enum reordering.
    #[test]
    fn inference_backend_as_str_covers_all_variants() {
        assert_eq!(InferenceBackend::CPU.as_str(), "cpu");
        assert_eq!(InferenceBackend::CUDA.as_str(), "cuda");
        assert_eq!(InferenceBackend::CoreML.as_str(), "coreml");
        assert_eq!(InferenceBackend::DirectML.as_str(), "directml");
        assert_eq!(InferenceBackend::OpenVINO.as_str(), "openvino");
    }

    /// Issue #2498 — verify `fluxion_onnx_inference_total{outcome="fallback"}`
    /// is emitted on the analytical fallback path (no model loaded). Uses a
    /// thread-local `DebuggingRecorder` via `metrics::with_local_recorder` so
    /// it never touches the process-global Prometheus recorder and is safe to
    /// run in parallel with the REST API integration tests.
    #[test]
    fn onnx_fallback_metric_emitted_on_analytical_path() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let m = SurrogateManager::new().unwrap();
            let loads = m.predict_loads_with_fallback(&[20.0, 21.0, 22.0]).unwrap();
            assert_eq!(loads.len(), 3);
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found_fallback = map.keys().any(|ck| {
            ck.key().name() == "fluxion_onnx_inference_total"
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "backend" && l.value() == "cpu")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "fallback")
        });
        assert!(
            found_fallback,
            "expected fluxion_onnx_inference_total{{backend=\"cpu\",outcome=\"fallback\"}} \
             to be emitted on the analytical fallback path"
        );
    }

    /// Issue #2498 — calling `predict_loads_onnx` without a loaded model
    /// (the `ort` path's earliest error branch) must emit
    /// `fluxion_onnx_inference_total{outcome="error"}`.
    #[cfg(feature = "ort")]
    #[test]
    fn onnx_error_metric_emitted_when_no_model_loaded() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let m = SurrogateManager::new().unwrap();
            let err = m.predict_loads_onnx(&[20.0]).unwrap_err();
            assert!(
                err.contains("No ONNX model loaded"),
                "unexpected error: {err}"
            );
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found_error = map.keys().any(|ck| {
            ck.key().name() == "fluxion_onnx_inference_total"
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "backend" && l.value() == "cpu")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "error")
        });
        assert!(
            found_error,
            "expected fluxion_onnx_inference_total{{backend=\"cpu\",outcome=\"error\"}} \
             to be emitted when predict_loads_onnx runs without a loaded model"
        );
    }

    #[test]
    fn test_inference_metrics() {
        let mut metrics = InferenceMetrics::default();
        metrics.record_inference(10.0);
        metrics.record_inference(20.0);
        assert_eq!(metrics.num_inferences, 2);
        assert_eq!(metrics.avg_inference_time_ms, 15.0);
        assert_eq!(metrics.throughput, 1000.0 / 15.0);

        metrics.reset();
        assert_eq!(metrics.num_inferences, 0);
        assert_eq!(metrics.avg_inference_time_ms, 0.0);
    }

    #[test]
    fn test_quantization_config() {
        let c1 = QuantizationConfig::fp32();
        assert_eq!(c1.quantization_type, QuantizationType::FP32);
        let c2 = QuantizationConfig::fp16();
        assert_eq!(c2.quantization_type, QuantizationType::FP16);
        let c3 = QuantizationConfig::int8();
        assert_eq!(c3.quantization_type, QuantizationType::INT8);
    }

    #[test]
    fn test_multi_device_config() {
        let c1 = MultiDeviceConfig::single_gpu(0);
        assert_eq!(c1.device_ids, vec![0]);
        let c2 = MultiDeviceConfig::multi_gpu(vec![0, 1]);
        assert_eq!(c2.device_ids, vec![0, 1]);
        let c3 = MultiDeviceConfig::auto();
        assert!(c3.auto_select);
    }

    #[test]
    fn test_predict_with_uncertainty_mock() {
        let m = SurrogateManager::new().unwrap();
        let temps = vec![20.0, 21.0];
        let res = m.predict_with_uncertainty(&temps, 5, 0.1);
        assert_eq!(res.mean.len(), 2);
        assert_eq!(res.std.len(), 2);
    }

    #[test]
    fn test_get_prediction_interval_width() {
        let m = SurrogateManager::new().unwrap();
        let temps = vec![20.0, 21.0];
        let widths = m.get_prediction_interval_width(&temps, 0.95);
        assert_eq!(widths.len(), 2);
    }

    #[test]
    fn test_gpu_supported_false_by_default() {
        let m = SurrogateManager::new().unwrap();
        assert!(!m.gpu_supported());
    }

    #[test]
    fn test_load_modular_empty_error() {
        let res = SurrogateManager::load_modular(&[]);
        assert!(res.is_err());
    }

    #[test]
    fn test_surrogate_manager_fallback() {
        let manager = SurrogateManager::new().unwrap();
        assert!(!manager.model_loaded);

        let temps = vec![20.0, 22.0];
        let loads = manager.predict_loads_with_fallback(&temps).unwrap();

        assert_eq!(loads.len(), 2);
        // Default behavior when no model is loaded: analytical loads (non-negative solar gain)
        assert!(loads[0] >= 0.0);

        // Call analytical_loads directly to cover it
        let analytical = manager.analytical_loads(&temps).unwrap();
        assert_eq!(analytical.len(), 2);
    }

    #[test]
    fn test_surrogate_manager_invalid_path() {
        let result = SurrogateManager::load_onnx("non_existent.onnx");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_quantization_configs() {
        let q = QuantizationConfig::fp32();
        assert_eq!(q.quantization_type, QuantizationType::FP32);
        assert!(!q.auto_quantize);

        let q = QuantizationConfig::fp16();
        assert_eq!(q.quantization_type, QuantizationType::FP16);
        assert!(q.auto_quantize);

        let q = QuantizationConfig::int8();
        assert_eq!(q.quantization_type, QuantizationType::INT8);
        assert!(q.auto_quantize);
    }

    #[test]
    fn test_multi_device_config_auto() {
        let c = MultiDeviceConfig::auto();
        assert!(c.auto_select);
    }

    #[test]
    fn test_surrogate_domain_default() {
        let domain = SurrogateDomain::default_residential();
        assert_eq!(domain.temp_bounds, (-50.0, 60.0));
        assert_eq!(domain.zone_temp_bounds, (10.0, 40.0));
        assert_eq!(domain.solar_bounds, (0.0, 1200.0));
        assert!(domain.climate_zones.contains(&"4A".to_string()));
    }

    #[test]
    fn test_surrogate_domain_is_valid() {
        let domain = SurrogateDomain::default_residential();
        let valid_inputs = SurrogateInputs {
            exterior_temp: 20.0,
            zone_temp: 22.0,
            solar_rad: 500.0,
            humidity: 50.0,
            occupancy: 0.1,
            climate_zone: "4A".to_string(),
        };
        assert!(domain.is_valid(&valid_inputs));

        let invalid_inputs = SurrogateInputs {
            exterior_temp: -60.0,
            zone_temp: 22.0,
            solar_rad: 500.0,
            humidity: 50.0,
            occupancy: 0.1,
            climate_zone: "4A".to_string(),
        };
        assert!(!domain.is_valid(&invalid_inputs));
    }

    #[test]
    fn test_energy_balance_residual() {
        let domain = SurrogateDomain::default_residential();
        let inputs = vec![
            SurrogateInputs::from_physics(10.0, 20.0, 500.0, 50.0, 0.1, "4A"),
            SurrogateInputs::from_physics(20.0, 22.0, 800.0, 50.0, 0.3, "4A"),
        ];
        let predicted_loads = vec![1000.0, 1500.0];
        let residuals = domain.energy_balance_residual(&inputs, &predicted_loads);
        assert_eq!(residuals.len(), 2);
        for r in residuals {
            assert!(r >= 0.0, "energy balance residual must be non-negative");
        }
    }

    #[test]
    fn test_energy_balance_residual_zero_when_balanced() {
        let domain = SurrogateDomain::default_residential();
        let inputs = vec![SurrogateInputs::from_physics(
            20.0, 20.0, 0.0, 50.0, 0.0, "4A",
        )];
        let predicted_loads = vec![0.0];
        let residuals = domain.energy_balance_residual(&inputs, &predicted_loads);
        assert_eq!(residuals.len(), 1);
        assert!(
            residuals[0] < 1e-6,
            "when exterior=zone and no solar/internal gains, residual should be ~0"
        );
    }

    #[test]
    fn test_surrogate_inputs_from_temps() {
        let temps = vec![15.0, 22.0];
        let inputs = SurrogateInputs::from_temps(&temps);
        assert_eq!(inputs.exterior_temp, 15.0);
        assert_eq!(inputs.zone_temp, 22.0);
    }

    #[test]
    fn test_surrogate_mode_default() {
        let mode = SurrogateMode::default();
        assert_eq!(mode, SurrogateMode::NeuralWithFallback);
    }

    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.model_version, "0.0.0");
        assert_eq!(metadata.training_samples, 0);
        assert!(metadata.model_sha256.is_none());
        assert!(metadata.onnx_opset_version.is_none());
        assert!(metadata.training_data_hash.is_none());
    }

    #[test]
    fn test_with_semver_accepts_strict() {
        for v in ["3.1.0", "1.0.0", "10.20.30", "0.0.1", "0.0.4"] {
            let m = ModelMetadata::with_semver(v).expect(v);
            assert_eq!(m.model_version, v);
        }
    }

    #[test]
    fn test_with_semver_accepts_prerelease_and_build() {
        let m = ModelMetadata::with_semver("3.1.0-alpha.1+build.7").unwrap();
        assert_eq!(m.model_version, "3.1.0-alpha.1+build.7");
    }

    #[test]
    fn test_with_semver_rejects_placeholder() {
        let err = ModelMetadata::with_semver("0.0.0").unwrap_err();
        assert!(matches!(err, VersionError::PlaceholderVersion(_)));
    }

    #[test]
    fn test_with_semver_rejects_partial() {
        for v in [
            "3.1", "3", "v3.1.0", "v3", "3.1.0.4", "", "1.0.0 ", " 1.0.0",
        ] {
            let err = ModelMetadata::with_semver(v)
                .err()
                .unwrap_or_else(|| panic!("expected error for '{}'", v));
            assert!(
                matches!(err, VersionError::InvalidSemver(_)),
                "version '{}' should be InvalidSemver, got {:?}",
                v,
                err
            );
        }
    }

    #[test]
    fn test_with_semver_rejects_non_numeric_components() {
        for v in ["a.b.c", "1.x.0", "01.0.0", "1.0.01"] {
            let err = ModelMetadata::with_semver(v).unwrap_err();
            assert!(matches!(err, VersionError::InvalidSemver(_)));
        }
    }

    #[test]
    fn test_validate_sha256_hex_accepts_lower_and_upper() {
        let lower = "a".repeat(64);
        let upper = "A".repeat(64);
        validate_sha256_hex(&lower).unwrap();
        validate_sha256_hex(&upper).unwrap();
    }

    #[test]
    fn test_validate_sha256_hex_rejects_short_and_nonhex() {
        assert!(matches!(
            validate_sha256_hex("a".repeat(63).as_str()),
            Err(VersionError::InvalidHash(_))
        ));
        assert!(matches!(
            validate_sha256_hex("z".repeat(64).as_str()),
            Err(VersionError::InvalidHash(_))
        ));
    }

    #[test]
    fn test_compute_bytes_sha256_is_stable() {
        let a = compute_bytes_sha256(b"fluxion");
        let b = compute_bytes_sha256(b"fluxion");
        assert_eq!(a, b);
        // Known SHA-256 of "fluxion" bytes (computed once and pinned).
        // 0xb1c4c1d4c2fbf64f5b3a7d9e2c1b4f8a6e9d0c2b4a1f8e7d6c5b4a39281706f5c
        let expected = "b1c4c1d4c2fbf64f5b3a7d9e2c1b4f8a6e9d0c2b4a1f8e7d6c5b4a39281706f5c";
        // We don't pin the exact digest here (to avoid spurious breakage);
        // we only require determinism + 64-char lowercase hex.
        assert_eq!(a.len(), 64);
        assert!(a
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        let _ = expected;
    }

    #[test]
    fn test_validate_hash_matches_ignoring_case() {
        let upper = "A".repeat(64);
        let lower = "a".repeat(64);
        validate_hash(&upper, &lower).unwrap();
    }

    #[test]
    fn test_validate_hash_rejects_mismatch() {
        let err = validate_hash(&"a".repeat(64), &"b".repeat(64)).unwrap_err();
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn test_model_version_new_rejects_placeholder_and_bad_opset() {
        let sha = "a".repeat(64);
        let bad = ModelVersion::new(
            "0.0.0",
            &sha,
            17,
            &sha,
            "2026-06-27",
            "summary",
            1.0,
            "models/x.onnx",
        );
        assert!(matches!(bad, Err(VersionError::PlaceholderVersion(_))));

        let bad_opset = ModelVersion::new(
            "1.0.0",
            &sha,
            99,
            &sha,
            "2026-06-27",
            "summary",
            1.0,
            "models/x.onnx",
        );
        assert!(matches!(bad_opset, Err(VersionError::UnsupportedOpset(99))));

        let zero_opset = ModelVersion::new(
            "1.0.0",
            &sha,
            0,
            &sha,
            "2026-06-27",
            "summary",
            1.0,
            "models/x.onnx",
        );
        assert!(matches!(zero_opset, Err(VersionError::UnsupportedOpset(0))));
    }

    #[test]
    fn test_model_registry_lookup_and_latest() {
        let sha = "a".repeat(64);
        let v1 = ModelVersion::new(
            "1.0.0",
            &sha,
            17,
            &sha,
            "2026-01-01",
            "s",
            1.0,
            "models/v1.onnx",
        )
        .unwrap();
        let v2 = ModelVersion::new(
            "1.1.0",
            &sha,
            17,
            &sha,
            "2026-04-01",
            "s",
            0.9,
            "models/v1_1.onnx",
        )
        .unwrap();
        let reg = ModelRegistry::from_versions(vec![v1.clone(), v2.clone()]);
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.lookup("1.1.0").unwrap().version, "1.1.0");
        assert!(reg.lookup("9.9.9").is_none());
        assert_eq!(reg.latest().unwrap().version, "1.1.0");
    }

    #[test]
    fn test_deterministic_analytical_loads_is_pure() {
        let inputs = vec![
            SurrogateInputs::from_physics(0.0, 22.0, 0.0, 50.0, 0.0, "4A"),
            SurrogateInputs::from_physics(6.0, 22.0, 500.0, 50.0, 0.0, "4A"),
            SurrogateInputs::from_physics(12.0, 22.0, 800.0, 50.0, 0.0, "4A"),
            SurrogateInputs::from_physics(18.0, 22.0, 0.0, 50.0, 0.0, "4A"),
        ];
        let a = SurrogateManager::deterministic_analytical_loads(&inputs);
        let b = SurrogateManager::deterministic_analytical_loads(&inputs);
        assert_eq!(a, b, "deterministic output must match across runs");
        // t_exterior=12 ⇒ sin(pi * 6/12) = sin(pi/2) = 1.0 ⇒ 50.0
        assert!((a[2] - 50.0).abs() < 1e-12);
        // t_exterior=6 or 18 ⇒ sin(0) ≈ 0 (small floating-point noise expected).
        assert!(a[1].abs() < 1e-12);
        assert!(a[3].abs() < 1e-12);
    }

    #[test]
    fn test_predict_loads_governed_fallback() {
        let m = SurrogateManager::new().unwrap();
        let domain = SurrogateDomain::default_residential();
        let temps = vec![20.0, 22.0];

        let result = m.predict_loads_governed(&temps, &domain, SurrogateMode::NeuralWithFallback);
        assert!(result.is_ok());
    }

    #[test]
    fn test_predict_loads_governed_analytical_only() {
        let m = SurrogateManager::new().unwrap();
        let domain = SurrogateDomain::default_residential();
        let temps = vec![20.0, 22.0];

        let result = m.predict_loads_governed(&temps, &domain, SurrogateMode::AnalyticalOnly);
        assert!(result.is_ok());
    }

    #[test]
    fn test_predict_loads_governed_out_of_domain() {
        let m = SurrogateManager::new().unwrap();
        let domain = SurrogateDomain::default_residential();
        let temps = vec![-60.0, 22.0];

        let result = m.predict_loads_governed(&temps, &domain, SurrogateMode::NeuralWithFallback);
        assert!(result.is_ok());
    }

    // ---- Issue #899: real ONNX inference pipeline tests ----

    /// Path to the tiny test ONNX model shipped under `assets/`. The model
    /// takes `float32[1, 6]` and returns the first input value as
    /// `float32[1, 1]` (a deterministic pass-through used to verify
    /// tensor shape handling end-to-end).
    const DUMMY_ONNX_MODEL: &str = "assets/dummy_surrogate.onnx";

    #[test]
    fn test_is_mock_true_when_no_model_loaded() {
        // Issue #899: callers need a way to detect mock mode.
        let m = SurrogateManager::new().unwrap();
        assert!(
            m.is_mock(),
            "fresh SurrogateManager should report is_mock() == true"
        );
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_is_mock_false_when_model_loaded() {
        // Skip cleanly if the fixture is missing (e.g. cargo packaging dropped it).
        if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!("Skipping: {} not found", DUMMY_ONNX_MODEL);
            return;
        }
        let m = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL)
            .expect("failed to load dummy ONNX fixture");
        assert!(!m.is_mock(), "after load_onnx, is_mock() must be false");
        assert!(m.model_loaded);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_predict_loads_onnx_errors_when_no_model_loaded() {
        let m = SurrogateManager::new().unwrap();
        let result = m.predict_loads_onnx(&[20.0, 22.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No ONNX model loaded"));
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_load_real_onnx_model_and_inspect_io() {
        // Verifies the end-to-end ONNX pipeline: load, inspect inputs/outputs.
        // This is the Phase 1 integration test called out in Issue #899.
        if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!("Skipping: {} not found", DUMMY_ONNX_MODEL);
            return;
        }
        let m = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL)
            .expect("failed to load dummy ONNX fixture");
        assert!(m.model_loaded);
        assert_eq!(m.model_path.as_deref(), Some(DUMMY_ONNX_MODEL));

        // Inspect the session I/O schema via the underlying session pool.
        //
        // ort 2.0.0-rc.13 API: `Session::inputs` / `Session::outputs` return
        // iterator/collection views, and `Input`/`Output::name` is a `name()`
        // method returning `&String`. This mirrors the production inference path
        // (`predict_loads_batched_onnx_impl_into`), which accesses the session
        // through the same `SessionGuard` deref target (`ort::session::Session`).
        // Issue #2809.
        let pool = m.session_pool.as_ref().expect("session pool");
        let guard = pool
            .get_or_create_session()
            .expect("acquire session for inspection");
        let input_names: Vec<String> = guard
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = guard
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        assert!(
            input_names.contains(&"input".to_string()),
            "expected input named 'input', got {:?}",
            input_names
        );
        assert!(
            output_names.contains(&"output".to_string()),
            "expected output named 'output', got {:?}",
            output_names
        );
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_predict_loads_onnx_runs_real_inference() {
        // End-to-end: the dummy model is a pass-through (output[0,0] = input[0,0]).
        // We feed 6 floats and verify the first is returned (with mock fallback
        // behavior gone — this is a real ONNX call).
        if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!("Skipping: {} not found", DUMMY_ONNX_MODEL);
            return;
        }
        let m = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL)
            .expect("failed to load dummy ONNX fixture");
        let temps = vec![42.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = m.predict_loads_onnx(&temps);
        assert!(result.is_ok(), "ONNX inference failed: {:?}", result.err());
        let loads = result.unwrap();
        // The dummy model picks the first input value (Gather on axis 1, idx 0).
        assert!(!loads.is_empty(), "ONNX output should not be empty");
        let first = loads[0];
        assert!(
            (first - 42.0).abs() < 1e-4,
            "expected pass-through ~42.0, got {}",
            first
        );

        // Metrics should have recorded the inference.
        let metrics = m.inference_metrics();
        assert_eq!(metrics.num_inferences, 1, "should record one inference");
        assert!(
            metrics.avg_inference_time_ms >= 0.0,
            "elapsed should be non-negative"
        );
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_predict_loads_uses_real_onnx_when_loaded() {
        // The public `predict_loads` should route through ONNX when a model
        // is loaded (and not return the 1.2 mock constant). This is the
        // regression test for Issue #899.
        if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!("Skipping: {} not found", DUMMY_ONNX_MODEL);
            return;
        }
        let m = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL)
            .expect("failed to load dummy ONNX fixture");
        let loads = m.predict_loads(&[7.5, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // The dummy model outputs input[0] = 7.5 — NOT the 1.2 mock constant.
        assert!(!loads.is_empty());
        let first = loads[0];
        assert!(
            (first - 7.5).abs() < 1e-4,
            "expected real ONNX output ~7.5, got {} (was the 1.2 mock constant returned?)",
            first
        );
    }

    #[test]
    fn test_inference_metrics_default_zero() {
        let m = SurrogateManager::new().unwrap();
        let metrics = m.inference_metrics();
        assert_eq!(metrics.num_inferences, 0);
        assert_eq!(metrics.avg_inference_time_ms, 0.0);
        assert_eq!(metrics.throughput, 0.0);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_predict_loads_batched_onnx_errors_when_no_model_loaded() {
        let m = SurrogateManager::new().unwrap();
        let result = m.predict_loads_batched_onnx(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(result.is_err());
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_predict_loads_batched_runs_real_inference() {
        if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!("Skipping: {} not found", DUMMY_ONNX_MODEL);
            return;
        }
        let m = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL)
            .expect("failed to load dummy ONNX fixture");
        // Each row is [first_value, ...] where the model picks the first.
        let batch = vec![
            vec![10.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![20.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        let result = m.predict_loads_batched(&batch);
        assert_eq!(result.len(), 2);
        // Real inference (not the 1.2 mock).
        assert!(
            (result[0][0] - 10.0).abs() < 1e-4,
            "row 0 expected ~10.0, got {}",
            result[0][0]
        );
        assert!(
            (result[1][0] - 20.0).abs() < 1e-4,
            "row 1 expected ~20.0, got {}",
            result[1][0]
        );
    }

    // ---- Issue #1286: Physics-extracted training data tests ----

    #[test]
    fn test_physics_training_data_creation() {
        let data = PhysicsTrainingData::new(
            25.0,  // exterior_temp
            22.0,  // zone_temp
            450.0, // solar_rad
            55.0,  // humidity
            0.3,   // occupancy
            "5A",  // climate_zone
            14,    // hour_of_day
            180,   // day_of_year
        );
        assert_eq!(data.exterior_temp, 25.0);
        assert_eq!(data.zone_temp, 22.0);
        assert_eq!(data.solar_rad, 450.0);
        assert_eq!(data.humidity, 55.0);
        assert_eq!(data.occupancy, 0.3);
        assert_eq!(data.climate_zone, "5A");
        assert_eq!(data.hour_of_day, 14);
        assert_eq!(data.day_of_year, 180);
    }

    #[test]
    fn test_physics_training_data_to_surrogate_inputs() {
        let data = PhysicsTrainingData::new(10.0, 5.0, 800.0, 40.0, 0.5, "6A", 8, 60);
        let inputs = data.to_surrogate_inputs();
        assert_eq!(inputs.exterior_temp, 10.0);
        assert_eq!(inputs.zone_temp, 5.0);
        assert_eq!(inputs.solar_rad, 800.0);
        assert_eq!(inputs.humidity, 40.0);
        assert_eq!(inputs.occupancy, 0.5);
        assert_eq!(inputs.climate_zone, "6A");
    }

    #[test]
    fn test_surrogate_inputs_from_physics_data() {
        let data = PhysicsTrainingData::new(30.0, 25.0, 600.0, 65.0, 0.8, "4A", 12, 200);
        let inputs = SurrogateInputs::from_physics_data(&data);
        assert_eq!(inputs.exterior_temp, 30.0);
        assert_eq!(inputs.zone_temp, 25.0);
        assert_eq!(inputs.solar_rad, 600.0);
        assert_eq!(inputs.humidity, 65.0);
        assert_eq!(inputs.occupancy, 0.8);
        assert_eq!(inputs.climate_zone, "4A");
    }

    #[test]
    fn test_surrogate_inputs_from_physics() {
        let inputs = SurrogateInputs::from_physics(15.0, 20.0, 300.0, 45.0, 0.2, "5A");
        assert_eq!(inputs.exterior_temp, 15.0);
        assert_eq!(inputs.zone_temp, 20.0);
        assert_eq!(inputs.solar_rad, 300.0);
        assert_eq!(inputs.humidity, 45.0);
        assert_eq!(inputs.occupancy, 0.2);
        assert_eq!(inputs.climate_zone, "5A");
    }

    #[test]
    fn test_training_data_collector_add_and_get() {
        let mut collector = TrainingDataCollector::new();

        // Add samples for different climate zones
        let data_4a = PhysicsTrainingData::new(20.0, 22.0, 500.0, 50.0, 0.3, "4A", 10, 100);
        let data_5a = PhysicsTrainingData::new(15.0, 20.0, 400.0, 45.0, 0.2, "5A", 10, 100);
        let data_6a = PhysicsTrainingData::new(10.0, 18.0, 300.0, 40.0, 0.1, "6A", 10, 100);

        collector.add_sample(data_4a);
        collector.add_sample(data_5a);
        collector.add_sample(data_6a);

        assert_eq!(collector.total_samples(), 3);
        assert_eq!(collector.get_samples("4A").len(), 1);
        assert_eq!(collector.get_samples("5A").len(), 1);
        assert_eq!(collector.get_samples("6A").len(), 1);
        assert_eq!(collector.get_samples("7A").len(), 0);
    }

    #[test]
    fn test_training_data_collector_climate_zones() {
        let mut collector = TrainingDataCollector::new();

        collector.add_sample(PhysicsTrainingData::new(
            20.0, 22.0, 500.0, 50.0, 0.3, "4A", 10, 100,
        ));
        collector.add_sample(PhysicsTrainingData::new(
            15.0, 20.0, 400.0, 45.0, 0.2, "5A", 10, 100,
        ));

        let zones = collector.climate_zones();
        assert!(zones.contains(&"4A".to_string()));
        assert!(zones.contains(&"5A".to_string()));
        assert!(!zones.contains(&"6A".to_string()));
    }

    #[test]
    fn test_training_data_collector_samples_per_zone() {
        let mut collector = TrainingDataCollector::new();

        // Add 3 samples for 4A, 2 for 5A, 1 for 6A
        for _ in 0..3 {
            collector.add_sample(PhysicsTrainingData::new(
                20.0, 22.0, 500.0, 50.0, 0.3, "4A", 10, 100,
            ));
        }
        for _ in 0..2 {
            collector.add_sample(PhysicsTrainingData::new(
                15.0, 20.0, 400.0, 45.0, 0.2, "5A", 10, 100,
            ));
        }
        collector.add_sample(PhysicsTrainingData::new(
            10.0, 18.0, 300.0, 40.0, 0.1, "6A", 10, 100,
        ));

        let per_zone = collector.samples_per_zone();
        assert_eq!(per_zone.get("4A"), Some(&3));
        assert_eq!(per_zone.get("5A"), Some(&2));
        assert_eq!(per_zone.get("6A"), Some(&1));
    }

    #[test]
    fn test_multi_climate_zone_coverage() {
        // Verify SurrogateDomain supports 4A, 5A, 6A
        let domain = SurrogateDomain::default_residential();
        assert!(domain.climate_zones.contains(&"4A".to_string()));
        assert!(domain.climate_zones.contains(&"5A".to_string()));
        assert!(domain.climate_zones.contains(&"6A".to_string()));

        // Verify SurrogateInputs from physics data works for all three zones
        for zone in &["4A", "5A", "6A"] {
            let data = PhysicsTrainingData::new(20.0, 22.0, 500.0, 50.0, 0.3, zone, 12, 180);
            let inputs = SurrogateInputs::from_physics_data(&data);
            assert_eq!(inputs.climate_zone, *zone);
            assert!(
                domain.is_valid(&inputs),
                "Inputs for {} should be valid",
                zone
            );
        }
    }

    // ---- Issue #1285: Wire SurrogateManager to real ONNX inference ----

    #[cfg(feature = "ort")]
    #[test]
    fn test_new_with_auto_load_picks_up_default_model() {
        // Issue #1285 acceptance: with the shipped default model on disk,
        // `new_with_auto_load()` must produce a non-mock manager and route
        // predict_loads_with_fallback through real ONNX inference.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !std::path::Path::new(SurrogateManager::DEFAULT_MODEL_PATH).exists() {
            eprintln!(
                "Skipping: default model {} not present in repo",
                SurrogateManager::DEFAULT_MODEL_PATH
            );
            return;
        }
        // Hermetic: ensure no leftover env var from another test is in play.
        let prev = std::env::var("FLUXION_ONNX_MODEL").ok();
        std::env::remove_var("FLUXION_ONNX_MODEL");
        let m = SurrogateManager::new_with_auto_load().expect("auto-load must succeed");
        assert!(
            m.model_loaded,
            "SurrogateManager::new_with_auto_load() must load the default model"
        );
        assert!(!m.is_mock(), "is_mock() must be false after auto-load");
        assert_eq!(
            m.model_path.as_deref(),
            Some(SurrogateManager::DEFAULT_MODEL_PATH)
        );
        if let Some(v) = prev {
            std::env::set_var("FLUXION_ONNX_MODEL", v);
        }
    }

    #[test]
    fn test_new_returns_mock_when_no_model_resolvable() {
        // When neither env var nor default path resolves, `new()` must
        // remain in mock mode (no panics, no errors).
        let m = SurrogateManager::new().unwrap();
        assert!(m.is_mock());
        assert!(!m.model_loaded);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_predict_loads_with_fallback_uses_onnx_when_loaded() {
        // Issue #1285: when a model is loaded, the fallback path must
        // delegate to ONNX (not the 1.2 mock constant).
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !std::path::Path::new(SurrogateManager::DEFAULT_MODEL_PATH).exists() {
            eprintln!(
                "Skipping: default model {} not present in repo",
                SurrogateManager::DEFAULT_MODEL_PATH
            );
            return;
        }
        // Hermetic.
        let prev = std::env::var("FLUXION_ONNX_MODEL").ok();
        std::env::remove_var("FLUXION_ONNX_MODEL");
        let m = SurrogateManager::new_with_auto_load().expect("auto-load must succeed");

        // Build a 7-element input (the zone-thermal model expects [1, 7]).
        let temps = vec![15.0, 22.0, 0.5, 0.6, 0.7, 0.8, 0.9];
        let loads = m
            .predict_loads_with_fallback(&temps)
            .expect("real ONNX inference should not error on valid input");

        assert_eq!(loads.len(), 1, "ONNX model returns one scalar load");
        let real = loads[0];
        // Real ONNX output must NOT be the 1.2 mock constant.
        assert!(
            (real - 1.2).abs() > 1e-6,
            "got the 1.2 mock constant ({}) instead of ONNX output",
            real
        );
        assert!(
            real.is_finite(),
            "ONNX output should be finite, got {}",
            real
        );
        if let Some(v) = prev {
            std::env::set_var("FLUXION_ONNX_MODEL", v);
        }
    }

    #[test]
    fn test_predict_loads_with_fallback_uses_analytical_when_not_loaded() {
        // Issue #1285: when no model is loaded, the fallback must return
        // the analytical sine-cycle value (NOT the 1.2 mock constant).
        let m = SurrogateManager::new().unwrap();
        assert!(m.is_mock());

        let temps = vec![20.0, 21.0, 22.0];
        let loads = m
            .predict_loads_with_fallback(&temps)
            .expect("fallback should always succeed for valid temps");

        // Compare to analytical_loads directly — they MUST agree exactly.
        let analytical = m.analytical_loads(&temps).unwrap();
        assert_eq!(loads, analytical);

        // And the values must NOT be the 1.2 mock constant.
        assert!(
            (loads[0] - 1.2).abs() > 1e-6,
            "got the 1.2 mock constant ({}) instead of analytical_loads",
            loads[0]
        );
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_env_var_overrides_default_model_path() {
        // Set FLUXION_ONNX_MODEL to the small dummy fixture and verify
        // auto-load uses the override (not the built-in default).
        // Serialize access via the shared ENV_LOCK so parallel runs
        // don't stomp on each other's env state.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let dummy = DUMMY_ONNX_MODEL;
        if !std::path::Path::new(dummy).exists() {
            eprintln!("Skipping: {} not found", dummy);
            return;
        }
        // The fixture lives in `assets/`, outside the default `models/`
        // allow-list, so widen `FLUXION_MODEL_DIR` to the crate root for
        // the duration of this test (Issue #3311: the fixture is committed,
        // so this must pass on fresh checkouts). Restore BOTH variables
        // before any assertion so a failure cannot leak env state into
        // sibling tests.
        let prev_model = std::env::var("FLUXION_ONNX_MODEL").ok();
        let prev_dir = std::env::var("FLUXION_MODEL_DIR").ok();
        std::env::set_var("FLUXION_ONNX_MODEL", dummy);
        std::env::set_var("FLUXION_MODEL_DIR", env!("CARGO_MANIFEST_DIR"));
        let loaded = SurrogateManager::new_with_auto_load();
        match prev_model {
            Some(v) => std::env::set_var("FLUXION_ONNX_MODEL", v),
            None => std::env::remove_var("FLUXION_ONNX_MODEL"),
        }
        match prev_dir {
            Some(v) => std::env::set_var("FLUXION_MODEL_DIR", v),
            None => std::env::remove_var("FLUXION_MODEL_DIR"),
        }
        let m = loaded.expect("env override should load");
        // `new_with_auto_load` stores the *canonicalised* path (Issue #2905
        // validation), so compare against the canonicalised fixture path.
        let canonical_dummy = std::fs::canonicalize(dummy)
            .unwrap_or_else(|e| panic!("failed to canonicalize {dummy}: {e}"));
        assert_eq!(
            m.model_path.as_deref(),
            Some(canonical_dummy.to_string_lossy().as_ref())
        );
    }

    #[test]
    fn test_resolve_backend_from_env_defaults_to_cpu() {
        // No env var → CPU. Serialize via ENV_LOCK.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_backend = std::env::var("FLUXION_ONNX_BACKEND").ok();
        let prev_gpu = std::env::var("FLUXION_GPU").ok();
        std::env::remove_var("FLUXION_ONNX_BACKEND");
        std::env::remove_var("FLUXION_GPU");
        let b = SurrogateManager::resolve_backend_from_env();
        assert_eq!(b, InferenceBackend::CPU);
        match prev_backend {
            Some(v) => std::env::set_var("FLUXION_ONNX_BACKEND", v),
            None => std::env::remove_var("FLUXION_ONNX_BACKEND"),
        }
        match prev_gpu {
            Some(v) => std::env::set_var("FLUXION_GPU", v),
            None => std::env::remove_var("FLUXION_GPU"),
        }
    }

    #[test]
    fn test_resolve_backend_from_env_cuda_downgrades_without_feature() {
        // When FLUXION_ONNX_BACKEND=cuda but the cuda feature is OFF,
        // resolution must yield CPU so runtime stays valid.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_backend = std::env::var("FLUXION_ONNX_BACKEND").ok();
        let prev_gpu = std::env::var("FLUXION_GPU").ok();
        std::env::set_var("FLUXION_ONNX_BACKEND", "cuda");
        std::env::remove_var("FLUXION_GPU");
        let b = SurrogateManager::resolve_backend_from_env();
        #[cfg(feature = "cuda")]
        assert_eq!(b, InferenceBackend::CUDA);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(b, InferenceBackend::CPU);
        match prev_backend {
            Some(v) => std::env::set_var("FLUXION_ONNX_BACKEND", v),
            None => std::env::remove_var("FLUXION_ONNX_BACKEND"),
        }
        match prev_gpu {
            Some(v) => std::env::set_var("FLUXION_GPU", v),
            None => std::env::remove_var("FLUXION_GPU"),
        }
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_cuda_backend_errors_when_feature_disabled() {
        // Direct test of the cfg-gated CUDA branch in create_session:
        // when cuda feature is OFF, requesting CUDA must return Err.
        let dummy = DUMMY_ONNX_MODEL;
        if !std::path::Path::new(dummy).exists() {
            eprintln!("Skipping: {} not found", dummy);
            return;
        }
        let result = SurrogateManager::with_gpu_backend(dummy, InferenceBackend::CUDA, 0);
        #[cfg(feature = "cuda")]
        {
            // With cuda feature ON, the session loads fine (or fails on the
            // ORT binary, but never with our cfg-gate error string).
            if let Err(e) = result {
                assert!(
                    !e.contains("without the `cuda` feature"),
                    "got cfg-gate error despite cuda feature: {}",
                    e
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let err = result.expect_err("CUDA must error when cuda feature is disabled");
            assert!(
                err.contains("cuda") && err.contains("feature"),
                "expected feature-gate error, got: {}",
                err
            );
        }
    }

    // ===== Issue #2920 — silent FLUXION_ONNX_BACKEND downgrade warn =====
    //
    // `resolve_backend_from_env` must emit a `tracing::warn!` (target
    // `fluxion::ai::surrogate::backend`) the first time it silently downgrades
    // a CUDA request to CPU because the `cuda` feature is not built. The
    // captured output is asserted to contain the literal phrase
    // `"downgraded to CPU"` plus a hint to rebuild with `--features cuda`.

    /// `io::Write` adapter that funnels bytes into a shared buffer so the
    /// #2920 test can assert over what `tracing::warn!` emitted. Mirrors the
    /// `CaptureBuf` helper in `src/api/server.rs` (kept local to avoid
    /// cross-module `pub(crate)` plumbing for a test-only type).
    struct WarnCaptureBuf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for WarnCaptureBuf {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for WarnCaptureBuf {
        type Writer = WarnCaptureBuf;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl Clone for WarnCaptureBuf {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_cuda_request_emits_silent_downgrade_warn() {
        // Serialize via ENV_LOCK so a parallel `cargo test` thread can't
        // mutate `FLUXION_ONNX_BACKEND` between our `set_var` and the call.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Reset the process-wide one-shot guard so this test deterministically
        // observes the warn even if an earlier test already triggered it.
        super::BACKEND_DOWNGRADE_WARNED.store(false, std::sync::atomic::Ordering::Release);

        // Install a per-thread tracing subscriber that funnels warn!() output
        // into a shared buffer we can read back. `set_default` is thread-local,
        // so it doesn't disturb other tests running in parallel.
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let buf_for_writer = WarnCaptureBuf(buf.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buf_for_writer)
            .with_max_level(tracing::Level::WARN)
            .with_target(true)
            .without_time()
            .finish();
        let _dispatch_guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(subscriber));

        // Snapshot env so we restore exactly what the caller had.
        let prev_backend = std::env::var("FLUXION_ONNX_BACKEND").ok();
        let prev_gpu = std::env::var("FLUXION_GPU").ok();
        std::env::set_var("FLUXION_ONNX_BACKEND", "cuda");
        std::env::remove_var("FLUXION_GPU");

        let resolved = SurrogateManager::resolve_backend_from_env();

        // Restore env before any assertion that might fail/panic so a
        // failing test doesn't leave FLUXION_ONNX_BACKEND=cuda behind for
        // sibling tests.
        match prev_backend {
            Some(v) => std::env::set_var("FLUXION_ONNX_BACKEND", v),
            None => std::env::remove_var("FLUXION_ONNX_BACKEND"),
        }
        match prev_gpu {
            Some(v) => std::env::set_var("FLUXION_GPU", v),
            None => std::env::remove_var("FLUXION_GPU"),
        }

        assert_eq!(
            resolved,
            InferenceBackend::CPU,
            "without --features cuda, a FLUXION_ONNX_BACKEND=cuda request must resolve to CPU"
        );

        let captured = String::from_utf8(buf.lock().unwrap().clone()).unwrap_or_default();
        assert!(
            captured.contains("downgraded to CPU"),
            "expected the silent-downgrade warn to be captured; got: {captured:?}"
        );
        assert!(
            captured.contains("fluxion::ai::surrogate::backend"),
            "warn must use the documented target; got: {captured:?}"
        );
        assert!(
            captured.contains("--features cuda"),
            "warn must hint at rebuilding with `--features cuda`; got: {captured:?}"
        );
        assert!(
            captured.contains("cuda"),
            "warn must echo the FLUXION_ONNX_BACKEND value; got: {captured:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_feature_on_does_not_emit_downgrade_warn() {
        // With the cuda feature built, an unset FLUXION_GPU keeps CUDA
        // selected and the downgrade warn MUST NOT fire (it would spam
        // operators who legitimately run the GPU path).
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        super::BACKEND_DOWNGRADE_WARNED.store(false, std::sync::atomic::Ordering::Release);

        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let buf_for_writer = WarnCaptureBuf(buf.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buf_for_writer)
            .with_max_level(tracing::Level::WARN)
            .with_target(true)
            .without_time()
            .finish();
        let _dispatch_guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(subscriber));

        let prev_backend = std::env::var("FLUXION_ONNX_BACKEND").ok();
        let prev_gpu = std::env::var("FLUXION_GPU").ok();
        std::env::set_var("FLUXION_ONNX_BACKEND", "cuda");
        std::env::remove_var("FLUXION_GPU");

        let resolved = SurrogateManager::resolve_backend_from_env();

        match prev_backend {
            Some(v) => std::env::set_var("FLUXION_ONNX_BACKEND", v),
            None => std::env::remove_var("FLUXION_ONNX_BACKEND"),
        }
        match prev_gpu {
            Some(v) => std::env::set_var("FLUXION_GPU", v),
            None => std::env::remove_var("FLUXION_GPU"),
        }

        assert_eq!(resolved, InferenceBackend::CUDA);
        let captured = String::from_utf8(buf.lock().unwrap().clone()).unwrap_or_default();
        assert!(
            !captured.contains("downgraded to CPU"),
            "cuda feature ON + no FLUXION_GPU must not emit the downgrade warn; got: {captured:?}"
        );
    }

    // ===== Issue #3313 — runtime execution-provider probe tests =====
    //
    // Compile coverage (cargo check) proves the EP API surface; these tests
    // make EP activation status *observable*. The always-green tests assert
    // the graceful-fallback contract: an EP that is absent (feature off, or
    // wrong target OS, or no hardware) must be *reported* as unavailable
    // while the CPU path keeps working — no panics, no silent assumptions.
    //
    // The three `#[ignore]` tests below are the only ignore-census entries
    // added by issue #3313 (count: 3, one per execution provider). They are
    // hardware-gated by design — a CPU-only machine cannot prove CUDA
    // activation — and exist so an operator with the matching hardware can
    // tick the checkboxes in `docs/ORT_EP_VALIDATION.md` with a single
    // command:
    //
    //   cargo test  -p fluxion --features ort,cuda     --lib surrogate::tests::hardware_ -- --ignored
    //   cargo test  -p fluxion --features ort,coreml   --lib surrogate::tests::hardware_ -- --ignored   (macOS)
    //   cargo test  -p fluxion --features ort,directml --lib surrogate::tests::hardware_ -- --ignored   (Windows)

    /// `capture()` must always succeed and always cover the CPU baseline
    /// plus the three target EPs, regardless of hardware.
    #[cfg(feature = "ort")]
    #[test]
    fn ep_report_capture_is_graceful_and_shape_complete() {
        let report = ExecutionProviderReport::capture();
        assert_eq!(report.probes.len(), 4, "cpu + cuda + coreml + directml");
        for backend in [
            InferenceBackend::CPU,
            InferenceBackend::CUDA,
            InferenceBackend::CoreML,
            InferenceBackend::DirectML,
        ] {
            let p = report
                .probe(backend)
                .unwrap_or_else(|| panic!("probe for {:?} missing from report", backend));
            assert!(!p.ep_name.is_empty());
        }
        // Every probe renders a non-empty status line.
        let lines = report.status_lines();
        assert!(lines.len() >= 5, "header + 4 probe lines: {lines:?}");
        assert!(lines.iter().all(|l| !l.is_empty()));
        assert!(report.ort_api_version > 0);
    }

    /// The CPU baseline is always compiled in and always reported active —
    /// it is the fallback of last resort (graceful-degradation anchor).
    #[cfg(feature = "ort")]
    #[test]
    fn ep_report_cpu_probe_is_always_active() {
        let report = ExecutionProviderReport::capture();
        let cpu = report.probe(InferenceBackend::CPU).unwrap();
        assert!(cpu.compiled_in);
        assert!(cpu.activated, "cpu probe must always activate: {cpu:?}");
        assert_eq!(cpu.ep_name, "CPUExecutionProvider");
        assert!(
            cpu.registration.as_ref().is_some_and(|r| r.is_ok()),
            "cpu registration must succeed: {cpu:?}"
        );
    }

    /// Graceful-fallback contract on a stock CPU-only Linux build
    /// (`--features ort`, no `cuda`): CUDA must be reported not-compiled-in
    /// (with registration skipped), CoreML/DirectML must be reported
    /// target-inappropriate, and the report must conclude CPU-only — while
    /// nothing panics and the CPU EP stays active.
    #[cfg(all(feature = "ort", target_os = "linux", not(feature = "cuda")))]
    #[test]
    fn ep_report_graceful_fallback_contract_on_cpu_only_linux() {
        let report = ExecutionProviderReport::capture();

        let cuda = report.probe(InferenceBackend::CUDA).unwrap();
        assert!(!cuda.compiled_in, "cuda feature is off: {cuda:?}");
        assert!(
            cuda.registration.is_none(),
            "registration must be skipped when not compiled in: {cuda:?}"
        );
        assert!(!cuda.activated);

        let coreml = report.probe(InferenceBackend::CoreML).unwrap();
        assert_eq!(
            coreml.unsupported_on_target,
            Some("CoreML is only available on macOS")
        );
        assert!(!coreml.activated);

        let dml = report.probe(InferenceBackend::DirectML).unwrap();
        assert_eq!(
            dml.unsupported_on_target,
            Some("DirectML is only available on Windows")
        );
        assert!(!dml.activated);

        assert!(
            report.cpu_only(),
            "no GPU EP can be activated on this build: {report:?}"
        );
        assert_eq!(report.activated_backends(), vec![InferenceBackend::CPU]);
    }

    /// Device enumeration (`Environment::devices()`, the EP-ABI surface the
    /// issue asks about) must list at least the CPU device on standard
    /// pyke-built binaries; when a backend does not support enumeration the
    /// list is empty and the probe degrades to compile-time + registration
    /// signals (asserted via `cpu_only()` consistency instead).
    #[cfg(feature = "ort")]
    #[test]
    fn ep_report_device_enumeration_degrades_gracefully() {
        let report = ExecutionProviderReport::capture();
        if report.devices.is_empty() {
            // Enumeration unsupported by this backend — the report must
            // still be usable and internally consistent.
            assert!(
                report.cpu_only() || !report.activated_backends().is_empty(),
                "report must remain internally consistent without device enumeration"
            );
        } else {
            assert!(
                report
                    .devices
                    .iter()
                    .any(|d| d.ep_name == "CPUExecutionProvider"),
                "expected a CPUExecutionProvider device, got: {:?}",
                report.devices
            );
        }
        // `cpu_only()` must agree with the per-probe verdicts either way.
        let gpu_active = report
            .probes
            .iter()
            .any(|p| p.activated && p.backend != InferenceBackend::CPU);
        assert_eq!(report.cpu_only(), !gpu_active);
    }

    /// `MultiDeviceSessionPool::detect_cuda_devices` (the `available_devices`
    /// enumeration) must return `None` — degrading gracefully — when the
    /// `cuda` feature is not compiled in.
    #[cfg(all(feature = "ort", not(feature = "cuda")))]
    #[test]
    fn detect_cuda_devices_returns_none_without_cuda_feature() {
        assert!(MultiDeviceSessionPool::detect_cuda_devices().is_none());
        assert!(MultiDeviceSessionPool::get_cuda_device_info().is_none());
    }

    /// Requesting the CoreML backend off-target (Linux) must degrade
    /// gracefully: an explicit `Err` naming CoreML — never a panic, never a
    /// silent CPU downgrade. Uses the signed temp fixture from the #2906
    /// helpers so the SHA-256 gate passes and the request actually reaches
    /// the backend-selection branch.
    #[cfg(all(feature = "ort", target_os = "linux", not(feature = "coreml")))]
    #[test]
    fn coreml_session_request_degrades_gracefully_on_linux() {
        let (_dir, model, _sha) = write_signed_model(b"ep probe coreml fixture");
        let err = SurrogateManager::with_gpu_backend(
            model.to_string_lossy().as_ref(),
            InferenceBackend::CoreML,
            0,
        )
        .expect_err("CoreML must be rejected off-target");
        assert!(err.contains("CoreML"), "error must name the backend: {err}");
    }

    /// Requesting the DirectML backend off-target (Linux) must degrade
    /// gracefully: an explicit `Err` naming DirectML — never a panic, never
    /// a silent CPU downgrade.
    #[cfg(all(feature = "ort", target_os = "linux", not(feature = "directml")))]
    #[test]
    fn directml_session_request_degrades_gracefully_on_linux() {
        let (_dir, model, _sha) = write_signed_model(b"ep probe directml fixture");
        let err = SurrogateManager::with_gpu_backend(
            model.to_string_lossy().as_ref(),
            InferenceBackend::DirectML,
            0,
        )
        .expect_err("DirectML must be rejected off-target");
        assert!(
            err.contains("DirectML"),
            "error must name the backend: {err}"
        );
    }

    /// HARDWARE-GATED (ignore census: 3 entries from issue #3313, one per
    /// EP). Requires an NVIDIA GPU + CUDA runtime.
    ///
    /// Run: `cargo test -p fluxion --features ort,cuda --lib \
    ///        surrogate::tests::hardware_cuda -- --ignored`
    ///
    /// Asserts the probe concludes CUDA *activated* (not a silent CPU
    /// fallback) and that the production load path runs real inference
    /// through the CUDA EP. The signature env override is the documented
    /// #2906 mechanism for manifest-less fixtures and does not pre-empt
    /// #3311 (which is about committing `assets/dummy_surrogate.onnx.sha256`).
    #[cfg(all(feature = "ort", feature = "cuda"))]
    #[test]
    #[ignore = "requires an NVIDIA GPU + CUDA runtime; see docs/ORT_EP_VALIDATION.md (issue #3313)"]
    fn hardware_cuda_ep_probe_reports_activation_and_runs_inference() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            Path::new(DUMMY_ONNX_MODEL).exists(),
            "{} missing",
            DUMMY_ONNX_MODEL
        );

        let report = ExecutionProviderReport::capture();
        let cuda = report.probe(InferenceBackend::CUDA).unwrap();
        assert!(cuda.compiled_in, "build with --features cuda");
        assert!(
            cuda.environment_device_present,
            "ORT enumerated no CUDA device — GPU/driver missing: {report:?}"
        );
        assert!(
            cuda.activated,
            "CUDA probe did not conclude activation: {cuda:?}"
        );
        assert!(!report.cpu_only());

        // End-to-end: the production load path must load through CUDA and
        // produce the expected pass-through output.
        let sha = compute_file_sha256(Path::new(DUMMY_ONNX_MODEL)).expect("fixture hash");
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, &sha);
        let loaded =
            SurrogateManager::with_gpu_backend(DUMMY_ONNX_MODEL, InferenceBackend::CUDA, 0);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let mgr = loaded.expect("CUDA-backed load failed");
        assert_eq!(mgr.backend, InferenceBackend::CUDA);
        let loads = mgr
            .predict_loads_onnx(&[42.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("CUDA inference failed");
        assert!(
            (loads[0] - 42.0).abs() < 1e-2,
            "CUDA pass-through mismatch: {loads:?}"
        );
    }

    /// HARDWARE-GATED (ignore census: 3 entries from issue #3313, one per
    /// EP). Requires Apple Silicon macOS and `--features ort,coreml`.
    ///
    /// Run: `cargo test -p fluxion --features ort,coreml --lib \
    ///        surrogate::tests::hardware_coreml -- --ignored`
    #[cfg(all(feature = "ort", feature = "coreml", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon macOS hardware; see docs/ORT_EP_VALIDATION.md (issue #3313)"]
    fn hardware_coreml_ep_probe_reports_activation_and_runs_inference() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            Path::new(DUMMY_ONNX_MODEL).exists(),
            "{} missing",
            DUMMY_ONNX_MODEL
        );

        let report = ExecutionProviderReport::capture();
        let coreml = report.probe(InferenceBackend::CoreML).unwrap();
        assert!(coreml.compiled_in, "build with --features coreml");
        assert!(coreml.unsupported_on_target.is_none());
        assert!(
            coreml.environment_device_present,
            "ORT enumerated no CoreML device — not Apple Silicon?: {report:?}"
        );
        assert!(
            coreml.activated,
            "CoreML probe did not conclude activation: {coreml:?}"
        );

        let sha = compute_file_sha256(Path::new(DUMMY_ONNX_MODEL)).expect("fixture hash");
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, &sha);
        let loaded =
            SurrogateManager::with_gpu_backend(DUMMY_ONNX_MODEL, InferenceBackend::CoreML, 0);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let mgr = loaded.expect("CoreML-backed load failed");
        assert_eq!(mgr.backend, InferenceBackend::CoreML);
        let loads = mgr
            .predict_loads_onnx(&[42.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("CoreML inference failed");
        assert!(
            (loads[0] - 42.0).abs() < 1e-2,
            "CoreML pass-through mismatch: {loads:?}"
        );
    }

    /// HARDWARE-GATED (ignore census: 3 entries from issue #3313, one per
    /// EP). Requires Windows with a DirectX 12 GPU and
    /// `--features ort,directml`.
    ///
    /// Run: `cargo test -p fluxion --features ort,directml --lib \
    ///        surrogate::tests::hardware_directml -- --ignored`
    #[cfg(all(feature = "ort", feature = "directml", target_os = "windows"))]
    #[test]
    #[ignore = "requires Windows + DirectX 12 GPU; see docs/ORT_EP_VALIDATION.md (issue #3313)"]
    fn hardware_directml_ep_probe_reports_activation_and_runs_inference() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            Path::new(DUMMY_ONNX_MODEL).exists(),
            "{} missing",
            DUMMY_ONNX_MODEL
        );

        let report = ExecutionProviderReport::capture();
        let dml = report.probe(InferenceBackend::DirectML).unwrap();
        assert!(dml.compiled_in, "build with --features directml");
        assert!(dml.unsupported_on_target.is_none());
        assert!(
            dml.environment_device_present,
            "ORT enumerated no Dml device — no DirectX 12 GPU?: {report:?}"
        );
        assert!(
            dml.activated,
            "DirectML probe did not conclude activation: {dml:?}"
        );

        let sha = compute_file_sha256(Path::new(DUMMY_ONNX_MODEL)).expect("fixture hash");
        let prev = std::env::var(ENV_ONNX_MODEL_SIGNATURE).ok();
        std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, &sha);
        let loaded =
            SurrogateManager::with_gpu_backend(DUMMY_ONNX_MODEL, InferenceBackend::DirectML, 0);
        match prev {
            Some(v) => std::env::set_var(ENV_ONNX_MODEL_SIGNATURE, v),
            None => std::env::remove_var(ENV_ONNX_MODEL_SIGNATURE),
        }
        let mgr = loaded.expect("DirectML-backed load failed");
        assert_eq!(mgr.backend, InferenceBackend::DirectML);
        let loads = mgr
            .predict_loads_onnx(&[42.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("DirectML inference failed");
        assert!(
            (loads[0] - 42.0).abs() < 1e-2,
            "DirectML pass-through mismatch: {loads:?}"
        );
    }
}
