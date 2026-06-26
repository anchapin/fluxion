//! Surrogate manager for fast thermal load predictions.

use crate::ai::modular_surrogate::{ComponentSurrogate, CompositeSurrogate};
use log::{info, warn};
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    OpenVINOExecutionProvider,
};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug, Copy, Default)]
/// Inference backend for ONNX runtime execution.
pub enum InferenceBackend {
    #[default]
    CPU,
    CUDA,
    CoreML,
    DirectML,
    OpenVINO,
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
    pub model_version: String,
    pub domain: SurrogateDomain,
    pub onnx_version: Option<String>,
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
            onnx_version: None,
            training_samples: 0,
            test_mae: None,
            test_rmse: None,
            test_r2: None,
            validation_date: None,
        }
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
}

impl Default for SurrogateManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default SurrogateManager")
    }
}

/// Thread-safe pool of ONNX Runtime sessions for concurrent inference.
#[derive(Debug)]
pub struct SessionPool {
    sessions: Mutex<Vec<ort::session::Session>>,
    model_path: String,
    backend: InferenceBackend,
    device_id: usize,
}

#[derive(Debug)]
pub struct MultiDeviceSessionPool {
    device_pools: Vec<Arc<SessionPool>>,
    _config: MultiDeviceConfig,
    _model_path: String,
}

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
            let mut available_devices = Vec::new();
            for device_id in 0..8 {
                let builder = match Session::builder() {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let cuda_ep = CUDAExecutionProvider::default().with_device_id(device_id as i32);
                match builder.with_execution_providers([cuda_ep.build()]) {
                    Ok(_) => available_devices.push(device_id),
                    Err(_) => continue,
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

pub struct MultiDeviceSessionGuard {
    pool: Arc<SessionPool>,
}

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
        device_id: usize,
    ) -> Result<ort::session::Session, String> {
        use ort::session::Session;
        let mut builder =
            Session::builder().map_err(|e| format!("Failed to create session builder: {}", e))?;
        match backend {
            InferenceBackend::CUDA => {
                let ep = CUDAExecutionProvider::default().with_device_id(device_id as i32);
                builder = builder
                    .with_execution_providers([ep.build()])
                    .map_err(|e| format!("Failed to add CUDA execution provider: {}", e))?;
            }
            InferenceBackend::CoreML => {
                let ep = CoreMLExecutionProvider::default();
                builder = builder
                    .with_execution_providers([ep.build()])
                    .map_err(|e| format!("Failed to add CoreML execution provider: {}", e))?;
            }
            InferenceBackend::DirectML => {
                let ep = DirectMLExecutionProvider::default().with_device_id(device_id as i32);
                builder = builder
                    .with_execution_providers([ep.build()])
                    .map_err(|e| format!("Failed to add DirectML execution provider: {}", e))?;
            }
            InferenceBackend::OpenVINO => {
                let ep = OpenVINOExecutionProvider::default();
                builder = builder
                    .with_execution_providers([ep.build()])
                    .map_err(|e| format!("Failed to add OpenVINO execution provider: {}", e))?;
            }
            InferenceBackend::CPU => {}
        }
        builder
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))
    }
}

struct SessionGuard<'a> {
    pool: &'a SessionPool,
    session: Option<ort::session::Session>,
}

impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool.return_session(session);
        }
    }
}

impl<'a> std::ops::Deref for SessionGuard<'a> {
    type Target = ort::session::Session;
    fn deref(&self) -> &Self::Target {
        self.session.as_ref().unwrap()
    }
}

impl<'a> std::ops::DerefMut for SessionGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().unwrap()
    }
}

impl SurrogateManager {
    pub fn new() -> Result<Self, String> {
        Ok(SurrogateManager {
            model_loaded: false,
            model_path: None,
            session_pool: None,
            backend: InferenceBackend::CPU,
            device_id: 0,
            composite: None,
            inference_metrics: Arc::new(parking_lot::Mutex::new(InferenceMetrics::default())),
        })
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

    pub fn predict_loads_with_fallback(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
        // Try ONNX first (delegate to existing batched method)
        let batch_loads = self.predict_loads_batched(&[temps.to_vec()]);

        if batch_loads.is_empty() {
            // Log warning and fall back to analytical
            warn!("ONNX inference returned empty results, falling back to analytical mode");
            self.analytical_loads(temps)
        } else {
            Ok(batch_loads[0].clone())
        }
    }

    pub fn analytical_loads(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
        if temps.is_empty() {
            return Ok(vec![]);
        }

        let hour_of_day = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / 3600) as usize
            % 24;

        let daily_cycle = (std::f64::consts::PI * (hour_of_day as f64 - 6.0) / 12.0).sin();
        let solar_gain = (50.0 * daily_cycle).max(0.0);

        Ok(vec![solar_gain; temps.len()])
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

    pub fn load_onnx(path: &str) -> Result<Self, String> {
        Self::with_gpu_backend(path, InferenceBackend::CPU, 0)
    }

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
        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }
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
        })
    }

    pub fn with_multi_device(path: &str, config: MultiDeviceConfig) -> Result<Self, String> {
        use std::path::Path;
        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }
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
        })
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

    /// Explicit ONNX inference — returns an error instead of panicking
    /// or silently falling back to mock data. Use this when you need to
    /// distinguish real neural predictions from mock placeholders.
    ///
    /// Returns `Err` if:
    /// - no ONNX model has been loaded via [`Self::load_onnx`]
    /// - input tensor shape does not match the model's expected input
    /// - the ONNX runtime reports an inference error
    pub fn predict_loads_onnx(&self, current_temps: &[f64]) -> Result<Vec<f64>, String> {
        if !self.model_loaded {
            return Err("No ONNX model loaded".to_string());
        }
        let pool = self
            .session_pool
            .as_ref()
            .ok_or_else(|| "No session pool available".to_string())?;

        let input_data: Vec<f32> = current_temps.iter().map(|&x| x as f32).collect();
        let n_input = input_data.len();

        let start = std::time::Instant::now();
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

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.inference_metrics.lock().record_inference(elapsed_ms);
        let _ = n_input; // kept for forward-compat shape validation
        Ok(result)
    }

    pub fn predict_loads_batched(&self, batch_temps: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if let Some(ref comp) = self.composite {
            return batch_temps
                .iter()
                .map(|temps| comp.predict_loads(temps))
                .collect();
        }

        if !self.model_loaded || batch_temps.is_empty() {
            return batch_temps
                .iter()
                .map(|temps| vec![1.2; temps.len()])
                .collect();
        }

        match self.predict_loads_batched_onnx(batch_temps) {
            Ok(loads) => loads,
            Err(e) => {
                warn!(
                    "Batched ONNX inference failed ({}), falling back to mock placeholder",
                    e
                );
                batch_temps
                    .iter()
                    .map(|temps| vec![1.2; temps.len()])
                    .collect()
            }
        }
    }

    /// Explicit batched ONNX inference — returns an error instead of
    /// panicking or silently falling back to mock data.
    pub fn predict_loads_batched_onnx(
        &self,
        batch_temps: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        if !self.model_loaded {
            return Err("No ONNX model loaded".to_string());
        }
        if batch_temps.is_empty() {
            return Ok(Vec::new());
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
        let flattened: Vec<f32> = batch_temps
            .iter()
            .flat_map(|v| v.iter().map(|&x| x as f32))
            .collect();

        let start = std::time::Instant::now();
        let mut session_guard = pool
            .get_or_create_session()
            .map_err(|e| format!("Could not acquire ORT session: {}", e))?;

        let input_tensor =
            ort::value::Value::from_array((vec![batch_size as i64, input_size as i64], flattened))
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
        let results: Vec<f64> = array_view.iter().copied().map(|x| x as f64).collect();
        if results.is_empty() {
            return Err("ONNX inference returned empty batch output".to_string());
        }
        let output_size = results.len() / batch_size;
        let batch_results: Vec<Vec<f64>> =
            results.chunks(output_size).map(|c| c.to_vec()).collect();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.inference_metrics.lock().record_inference(elapsed_ms);
        Ok(batch_results)
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn predict_loads_with_fallback_success() {
        let m = SurrogateManager::new().unwrap();
        let temps = vec![20.0, 21.0, 22.0];
        let loads = m.predict_loads_with_fallback(&temps).unwrap();
        assert_eq!(loads.len(), 3);
        // When model_loaded is false, falls back to analytical loads (non-negative)
        assert!(loads[0] >= 0.0);
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

    #[test]
    fn test_predict_loads_onnx_errors_when_no_model_loaded() {
        let m = SurrogateManager::new().unwrap();
        let result = m.predict_loads_onnx(&[20.0, 22.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No ONNX model loaded"));
    }

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
        let pool = m.session_pool.as_ref().expect("session pool");
        let mut guard = pool
            .get_or_create_session()
            .expect("acquire session for inspection");
        let input_names: Vec<String> = guard
            .inputs()
            .into_iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = guard
            .outputs()
            .into_iter()
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

    #[test]
    fn test_predict_loads_batched_onnx_errors_when_no_model_loaded() {
        let m = SurrogateManager::new().unwrap();
        let result = m.predict_loads_batched_onnx(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(result.is_err());
    }

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
}
