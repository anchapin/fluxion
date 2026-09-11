//! Surrogate orchestrator: [`SurrogateManager`](crate::ai::surrogate::SurrogateManager).
//!
//! Owns model lifecycle (load / auto-load / versioned load), the analytical
//! fallback, OOD validation, the energy-balance residual guard, training-data
//! collection, and the domain/input model types.

#[allow(unused_imports)]
use crate::ai::modular_surrogate::{ComponentSurrogate, CompositeSurrogate};
#[allow(unused_imports)]
use log::{info, warn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(feature = "ort")]
use super::integrity::open_and_verify_onnx;
use super::integrity::{
    compute_file_sha256, validate_hash, validate_model_path, validate_semver,
    verify_onnx_signature, ModelRegistry, VersionError,
};
use super::metrics::InferenceMetrics;
#[cfg(feature = "ort")]
use super::session_pool::batch_bucket_label;
#[cfg(feature = "ort")]
use super::session_pool::MultiDeviceSessionPool;
use super::session_pool::{InferenceBackend, MultiDeviceConfig, SessionPool};

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
        // Issue #3636: phase from input (matches analytical_loads #1335).
        let phase = temps.first().copied().unwrap_or(12.0);
        let daily_cycle = (std::f64::consts::PI * (phase - 6.0) / 12.0).sin();
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
/// One-shot guard for the `FLUXION_ONNX_BACKEND` silent-downgrade warn
/// (Issue #2920). `compare_exchange(false, true, …)` ensures exactly one warn
/// per process even under parallel callers; tests reset to `false` so the
/// assertion can re-trigger the path.
pub(crate) static BACKEND_DOWNGRADE_WARNED: AtomicBool = AtomicBool::new(false);

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
    pub(crate) fn resolve_backend_from_env() -> InferenceBackend {
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
        // Issue #3573: open the model exactly once with O_NOFOLLOW,
        // hash the bytes, and pass the SAME bytes (already in memory)
        // to `commit_from_memory`. The previous flow —
        // `verify_onnx_signature` followed by
        // `SessionPool::create_session(path, …)` — performed two
        // independent filesystem reads of the same path, giving a
        // process-local attacker a TOCTOU window to swap the file
        // between the integrity check and the ort session open. The
        // single-handle + commit_from_memory handoff eliminates the
        // second read entirely.
        let bytes = open_and_verify_onnx(model_path)?;
        let session = SessionPool::create_session_from_bytes(&bytes, backend, device_id)?;
        let pool = SessionPool::new(
            path.to_string(),
            Arc::new(bytes),
            backend,
            device_id,
            session,
        );
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
        // Issue #3573: open the model once with O_NOFOLLOW, hash the
        // bytes, and pass the same `Arc<Vec<u8>>` to every per-device
        // `SessionPool` via `MultiDeviceSessionPool::from_bytes`. Each
        // pool's `commit_from_memory` call materialises its session
        // from the shared verified buffer — no per-device code path
        // re-opens the model at `path`, so a directory swap between
        // verify and load cannot reach any of them.
        let bytes = open_and_verify_onnx(model_path)?;
        let bytes_arc = Arc::new(bytes);
        match MultiDeviceSessionPool::from_bytes(bytes_arc, path.to_string(), &config) {
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
        // Issue #3573: open once with O_NOFOLLOW, hash, hand the bytes
        // to both the verifier (already done inside
        // `open_and_verify_onnx`) and `commit_from_memory`.
        // `SessionPool::create_session(path)` would re-resolve `path`
        // — we avoid that here by going straight through the
        // bytes-based API.
        let bytes = open_and_verify_onnx(Path::new(path))?;
        let session = SessionPool::create_session_from_bytes(&bytes, InferenceBackend::CPU, 0)?;
        let pool = SessionPool::new(
            path.to_string(),
            Arc::new(bytes),
            InferenceBackend::CPU,
            0,
            session,
        );
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
                let mut rng = StdRng::from_os_rng();
                let perturbed_temps: Vec<f64> = current_temps
                    .iter()
                    .map(|&t| t + (rng.random::<f64>() - 0.5) * 2.0 * noise_std)
                    .collect();
                let _perturbed_temps = perturbed_temps;
                base_prediction
                    .iter()
                    .map(|&v| v + (rng.random::<f64>() - 0.5) * 2.0 * variance.sqrt())
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
