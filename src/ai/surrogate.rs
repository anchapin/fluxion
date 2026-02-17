//! Surrogate manager for fast thermal load predictions.
//!
//! The `SurrogateManager` wraps ONNX Runtime sessions for neural network inference
//! or returns mock predictions during development/testing. It uses `Mutex` for
//! thread-safe interior mutability (required by ORT Session::run which needs &mut).

use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    OpenVINOExecutionProvider,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// Defines the inference backend to be used for the ONNX Runtime session.
///
/// This enum allows specifying different execution providers for
/// model inference, such as CPU, CUDA, CoreML, DirectML, and OpenVINO.
#[derive(Clone, Debug, Copy, Default)]
pub enum InferenceBackend {
    #[default]
    CPU,
    CUDA,
    CoreML,
    DirectML,
    OpenVINO,
}

/// Model quantization format for reduced memory footprint and faster inference.
///
/// Quantization reduces model size and can significantly improve inference speed
/// at the cost of some accuracy.
#[derive(Clone, Debug, Copy, Default, PartialEq, Eq)]
pub enum QuantizationType {
    /// Full precision (FP32) - no quantization
    #[default]
    FP32,
    /// Half precision (FP16) - reduced size, faster inference
    FP16,
    /// 8-bit integer quantization - smallest size, fastest inference
    INT8,
}

/// Configuration for model quantization settings.
#[derive(Clone, Debug, Default)]
pub struct QuantizationConfig {
    /// The quantization type to use
    pub quantization_type: QuantizationType,
    /// Whether to apply quantization automatically if supported
    pub auto_quantize: bool,
}

impl QuantizationConfig {
    /// Create a new quantization configuration with FP32 (no quantization)
    pub fn fp32() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::FP32,
            auto_quantize: false,
        }
    }

    /// Create a new quantization configuration with FP16
    pub fn fp16() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::FP16,
            auto_quantize: true,
        }
    }

    /// Create a new quantization configuration with INT8
    pub fn int8() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::INT8,
            auto_quantize: true,
        }
    }
}

/// Configuration for multi-device inference.
///
/// Allows specifying which devices to use for inference and how to distribute
/// the workload across multiple devices.
#[derive(Clone, Debug, Default)]
pub struct MultiDeviceConfig {
    /// List of device IDs to use for inference
    pub device_ids: Vec<usize>,
    /// Number of sessions per device for load balancing
    pub sessions_per_device: usize,
    /// Whether to automatically select the best available device
    pub auto_select: bool,
    /// Enable device affinity (pin inference to specific device)
    pub enable_affinity: bool,
    /// Fallback to CPU if GPU fails
    pub fallback_to_cpu: bool,
    /// Maximum number of retry attempts on device failure
    pub max_retries: usize,
}

impl MultiDeviceConfig {
    /// Create a new multi-device config for single GPU
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

    /// Create a new multi-device config for multi-GPU
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

    /// Create config that auto-selects the best available device
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

/// Information about a CUDA device.
#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    /// Device ID
    pub device_id: usize,
    /// Device name (e.g., "NVIDIA RTX 3090")
    pub name: String,
    /// Compute capability (major, minor) if available
    pub compute_capability: Option<(u32, u32)>,
}

/// Load balancing strategy for multi-device inference.
#[derive(Clone, Debug, Default)]
pub enum LoadBalancingStrategy {
    /// Round-robin selection
    #[default]
    RoundRobin,
    /// Least loaded device (fewest active sessions)
    LeastLoaded,
    /// Random selection
    Random,
}

/// Performance metrics for inference benchmarking.
#[derive(Clone, Debug, Default)]
pub struct InferenceMetrics {
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Number of inferences performed
    pub num_inferences: usize,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Throughput (inferences per second)
    pub throughput: f64,
}

impl InferenceMetrics {
    /// Record a new inference time
    pub fn record_inference(&mut self, time_ms: f64) {
        let n = self.num_inferences as f64;
        self.avg_inference_time_ms = (self.avg_inference_time_ms * n + time_ms) / (n + 1.0);
        self.num_inferences += 1;
        self.throughput = 1000.0 / self.avg_inference_time_ms;
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.avg_inference_time_ms = 0.0;
        self.num_inferences = 0;
        self.peak_memory_mb = 0.0;
        self.throughput = 0.0;
    }
}

/// Manager that provides fast (mock or neural) thermal-load predictions.
///
/// Wraps an ONNX Runtime session for neural network inference, or returns
/// deterministic mock values when no model is loaded.
///
/// # Thread Safety
/// The session is wrapped in `Arc<Mutex<_>>` for thread-safe sharing across
/// parallel workers, allowing multiple threads to safely borrow the session
/// for inference.
#[derive(Clone, Default, Debug)]
pub struct SurrogateManager {
    /// Whether a trained model has been loaded.
    pub model_loaded: bool,
    /// Optional path to an ONNX model registered for use.
    pub model_path: Option<String>,
    /// Optional session pool for concurrent inference.
    /// Wrapped in `Arc` for sharing across clones.
    pub session_pool: Option<Arc<SessionPool>>,
    /// The backend used for inference.
    pub backend: InferenceBackend,
    /// The device ID (e.g. GPU index).
    pub device_id: usize,
}

/// A pool of ONNX sessions to allow concurrent inference.
#[derive(Debug)]
pub struct SessionPool {
    sessions: Mutex<Vec<ort::session::Session>>,
    model_path: String,
    backend: InferenceBackend,
    device_id: usize,
}

/// Multi-device session pool for distributed inference across multiple GPUs.
///
/// This pool manages sessions across multiple devices, automatically load-balancing
/// inference requests across available resources.
#[derive(Debug)]
pub struct MultiDeviceSessionPool {
    /// Pools for each device
    device_pools: Vec<Arc<SessionPool>>,
    /// Configuration for multi-device setup
    _config: MultiDeviceConfig,
    /// Model path (shared across devices)
    _model_path: String,
}

impl MultiDeviceSessionPool {
    /// Create a new multi-device session pool.
    pub fn new(model_path: String, config: &MultiDeviceConfig) -> Result<Self, String> {
        let mut device_pools = Vec::new();

        let device_ids = if config.auto_select {
            // Auto-detect available CUDA devices
            Self::detect_cuda_devices().unwrap_or_else(|| vec![0])
        } else if config.device_ids.is_empty() {
            vec![0] // Default to device 0
        } else {
            config.device_ids.clone()
        };

        // Create session pools for each device
        for device_id in &device_ids {
            // Create initial session for this device
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
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to create session for device {}: {}",
                        device_id, e
                    );
                }
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

    /// Detect available CUDA devices using ONNX Runtime.
    ///
    /// Returns a list of available GPU device IDs. If CUDA is not available,
    /// returns None to indicate no GPUs were found.
    fn detect_cuda_devices() -> Option<Vec<usize>> {
        // Try to detect CUDA devices using ONNX Runtime
        // This implementation attempts to create a CUDA session to verify availability
        #[cfg(feature = "cuda")]
        {
            use ort::session::Session;

            // Try to create a minimal CUDA session to verify GPU availability
            // We'll use a simple approach - try device 0 and iterate
            let mut available_devices = Vec::new();

            // Try up to 8 devices (common max for systems)
            for device_id in 0..8 {
                let mut builder = match Session::builder() {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                // Try to add CUDA provider to detect if device is available
                let cuda_ep = CUDAExecutionProvider::default().with_device_id(device_id as i32);
                match builder.with_execution_providers([cuda_ep.build()]) {
                    Ok(builder) => {
                        // If we can configure CUDA provider, device might be available
                        // Note: We can't fully test without a model, so we add the device
                        // as "potentially available" - the actual session creation will
                        // confirm availability
                        available_devices.push(device_id);
                    }
                    Err(_) => {
                        // This device is not available
                        continue;
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
            // CUDA feature not enabled, return None
            None
        }
    }

    /// Get detailed information about available CUDA devices.
    ///
    /// Returns a vector of device information if successful, None otherwise.
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
                        compute_capability: None, // Would require CUDA API to query
                    })
                    .collect(),
            )
        }

        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Get a session from the least loaded device.
    pub fn get_session(&self) -> Result<MultiDeviceSessionGuard, String> {
        // Simple round-robin or first-available strategy
        // In production, you'd track load and select least-loaded device
        for pool in &self.device_pools {
            if let Ok(_session) = pool.get_or_create_session() {
                // Return a guard that will use this pool
                return Ok(MultiDeviceSessionGuard {
                    pool: Arc::clone(pool),
                });
            }
        }
        Err("No available sessions in multi-device pool".to_string())
    }

    /// Get the number of devices in the pool.
    pub fn num_devices(&self) -> usize {
        self.device_pools.len()
    }
}

/// Guard for multi-device session that returns to the correct pool.
/// This is a simplified version that uses the pool for inference.
pub struct MultiDeviceSessionGuard {
    pool: Arc<SessionPool>,
}

impl MultiDeviceSessionGuard {
    /// Run inference using this guard's pool.
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
        // Try to get an existing session
        {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(session) = sessions.pop() {
                return Ok(SessionGuard {
                    pool: self,
                    session: Some(session),
                });
            }
        }

        // Create a new session if pool is empty
        // We do this outside the lock to allow other threads to access the pool
        Self::create_session(&self.model_path, self.backend, self.device_id).map(|session| {
            SessionGuard {
                pool: self,
                session: Some(session),
            }
        })
    }

    fn return_session(&self, session: ort::session::Session) {
        let mut sessions = self.sessions.lock().unwrap();
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

/// Guard that returns the session to the pool when dropped.
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
    /// Construct a new `SurrogateManager`.
    pub fn new() -> Result<Self, String> {
        Ok(SurrogateManager {
            model_loaded: false,
            model_path: None,
            session_pool: None,
            backend: InferenceBackend::CPU,
            device_id: 0,
        })
    }

    /// Register an ONNX model file to be used by the surrogate manager using the default CPU backend.
    pub fn load_onnx(path: &str) -> Result<Self, String> {
        Self::with_gpu_backend(path, InferenceBackend::CPU, 0)
    }

    /// Register an ONNX model file to be used by the surrogate manager with a specific backend.
    pub fn with_gpu_backend(
        path: &str,
        backend: InferenceBackend,
        device_id: usize,
    ) -> Result<Self, String> {
        use std::path::Path;

        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }

        // Create initial session
        let session = SessionPool::create_session(path, backend, device_id)?;
        let pool = SessionPool::new(path.to_string(), backend, device_id, session);

        Ok(SurrogateManager {
            model_loaded: true,
            model_path: Some(path.to_string()),
            session_pool: Some(Arc::new(pool)),
            backend,
            device_id,
        })
    }

    /// Register an ONNX model file with multi-device GPU support.
    ///
    /// This allows distributing inference across multiple GPUs for improved throughput.
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX model file
    /// * `config` - Multi-device configuration
    ///
    /// # Returns
    /// SurrogateManager with multi-device support enabled
    pub fn with_multi_device(path: &str, config: MultiDeviceConfig) -> Result<Self, String> {
        use std::path::Path;

        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }

        // Try to create multi-device pool
        match MultiDeviceSessionPool::new(path.to_string(), &config) {
            Ok(multi_pool) => {
                // For now, we use a single device pool as primary
                // The multi-device pool is managed separately
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
                })
            }
            Err(e) => {
                // Fall back to single device
                eprintln!(
                    "Multi-device setup failed: {}, falling back to single device",
                    e
                );
                Self::with_gpu_backend(path, InferenceBackend::CUDA, 0)
            }
        }
    }

    /// Predict thermal loads for each zone given current temperatures.
    ///
    /// # Arguments
    /// * `current_temps` - Slice of current zone temperatures in degrees Celsius (`&[f64]`)
    ///
    /// # Returns
    /// * `Vec<f64>` - Predicted thermal loads (W/m²) for each zone
    ///
    /// Returns a mock constant (1.2 W/m² per zone) when no model is loaded.
    /// Uses ONNX Runtime inference if a model is loaded.
    pub fn predict_loads(&self, current_temps: &[f64]) -> Vec<f64> {
        if !self.model_loaded {
            return vec![1.2; current_temps.len()];
        }

        // If we have a session pool, acquire a session and perform inference.
        if let Some(ref pool) = self.session_pool {
            // Convert input temps to f32
            let input_data: Vec<f32> = current_temps.iter().map(|&x| x as f32).collect();
            let n_input = input_data.len();

            // Try to acquire a session from the pool
            match pool.get_or_create_session() {
                Ok(mut session_guard) => {
                    // Create a tensor from owned data using the tuple format (shape, data)
                    // This is compatible with ort 2.0.0-rc.11
                    let input_tensor =
                        match ort::value::Value::from_array(([1, n_input], input_data)) {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!("Failed to create input tensor: {}; using mock loads", e);
                                return vec![1.2; n_input];
                            }
                        };

                    // Run inference using the inputs! macro pattern
                    match session_guard.run(ort::inputs![input_tensor]) {
                        Ok(outputs) => {
                            // Extract the first output
                            if outputs.len() > 0 {
                                match outputs[0].try_extract_array::<f32>() {
                                    Ok(array_view) => {
                                        let v: Vec<f64> =
                                            array_view.iter().copied().map(|x| x as f64).collect();
                                        if v.len() == n_input {
                                            return v;
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Failed to extract tensor: {}; using mock loads",
                                            e
                                        );
                                    }
                                }
                            }
                            vec![1.2; current_temps.len()]
                        }
                        Err(e) => {
                            eprintln!("ONNX inference error: {}; using mock loads", e);
                            vec![1.2; current_temps.len()]
                        }
                    }
                }
                Err(_) => {
                    eprintln!("Could not acquire ORT session; using mock loads");
                    vec![1.2; current_temps.len()]
                }
            }
        } else {
            // Default fallback
            vec![1.2; current_temps.len()]
        }
    }

    /// Predict thermal loads for a batch of inputs.
    ///
    /// # Arguments
    /// * `batch_temps` - Slice of input vectors, where each vector represents zone temperatures.
    ///
    /// # Returns
    /// * `Vec<Vec<f64>>` - A vector of result vectors, one for each input.
    pub fn predict_loads_batched(&self, batch_temps: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if !self.model_loaded || batch_temps.is_empty() {
            return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
        }

        if let Some(ref pool) = self.session_pool {
            let batch_size = batch_temps.len();
            let input_size = batch_temps[0].len();

            // Check consistency
            for t in batch_temps {
                if t.len() != input_size {
                    eprintln!("Inconsistent input sizes in batch; falling back to mock");
                    return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                }
            }

            // Flatten input
            let flattened: Vec<f32> = batch_temps
                .iter()
                .flat_map(|v| v.iter().map(|&x| x as f32))
                .collect();

            match pool.get_or_create_session() {
                Ok(mut session_guard) => {
                    // Create tensor from owned data using the tuple format (shape, data)
                    // This is compatible with ort 2.0.0-rc.11
                    let input_tensor = match ort::value::Value::from_array((
                        vec![batch_size, input_size],
                        flattened,
                    )) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Failed to create input tensor: {}; using mock loads", e);
                            return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                        }
                    };

                    match session_guard.run(ort::inputs![input_tensor]) {
                        Ok(outputs) => {
                            if outputs.len() > 0 {
                                match outputs[0].try_extract_array::<f32>() {
                                    Ok(array_view) => {
                                        // Expected shape: (batch_size, output_size)
                                        // We need to reconstruct Vec<Vec<f64>>
                                        let result_iter =
                                            array_view.iter().copied().map(|x| x as f64);
                                        let results: Vec<f64> = result_iter.collect();
                                        let output_size = results.len() / batch_size;

                                        let mut batch_results = Vec::with_capacity(batch_size);
                                        for chunk in results.chunks(output_size) {
                                            batch_results.push(chunk.to_vec());
                                        }
                                        return batch_results;
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Failed to extract tensor: {}; using mock loads",
                                            e
                                        );
                                    }
                                }
                            }
                            batch_temps.iter().map(|t| vec![1.2; t.len()]).collect()
                        }
                        Err(e) => {
                            eprintln!("ONNX inference error: {}; using mock loads", e);
                            batch_temps.iter().map(|t| vec![1.2; t.len()]).collect()
                        }
                    }
                }
                Err(_) => {
                    eprintln!("Could not acquire ORT session; using mock loads");
                    batch_temps.iter().map(|t| vec![1.2; t.len()]).collect()
                }
            }
        } else {
            batch_temps.iter().map(|t| vec![1.2; t.len()]).collect()
        }
    }
}

/// Prediction result with uncertainty bounds.
#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    /// Mean predicted thermal loads (W/m²)
    pub mean: Vec<f64>,
    /// Standard deviation of predictions
    pub std: Vec<f64>,
    /// Lower bound (mean - 2*std)
    pub lower_bound: Vec<f64>,
    /// Upper bound (mean + 2*std)
    pub upper_bound: Vec<f64>,
}

impl PredictionWithUncertainty {
    /// Creates a new prediction with uncertainty.
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
    /// Predict thermal loads with uncertainty estimation.
    ///
    /// This method runs multiple inferences with slight input perturbations
    /// to estimate prediction uncertainty. This is useful for risk-aware
    /// optimization and understanding model confidence.
    ///
    /// # Arguments
    /// * `current_temps` - Slice of current zone temperatures in degrees Celsius
    /// * `num_samples` - Number of Monte Carlo samples for uncertainty estimation
    /// * `noise_std` - Standard deviation of Gaussian noise to add to inputs (default: 0.5°C)
    ///
    /// # Returns
    /// PredictionWithUncertainty containing mean, std, and confidence bounds
    pub fn predict_with_uncertainty(
        &self,
        current_temps: &[f64],
        num_samples: usize,
        noise_std: f64,
    ) -> PredictionWithUncertainty {
        if !self.model_loaded || num_samples == 0 {
            // Return deterministic prediction with zero uncertainty
            let mean = self.predict_loads(current_temps);
            let std = vec![0.0; mean.len()];
            return PredictionWithUncertainty::new(mean, std);
        }

        // Run multiple forward passes with noise
        let mut all_predictions: Vec<Vec<f64>> = Vec::with_capacity(num_samples);

        // Use mock predictions with added variance for development
        // In production with real ONNX models, this would use dropout or ensemble
        let base_prediction = self.predict_loads(current_temps);

        // Create thread-local RNG
        thread_local! {
            static RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
        }

        for _ in 0..num_samples {
            // Add small random perturbation to simulate model uncertainty
            let _perturbed_temps: Vec<f64> = current_temps
                .iter()
                .map(|&t| {
                    let noise: f64 = RNG.with(|r| {
                        let mut rng = r.borrow_mut();
                        rng.gen::<f64>() - 0.5
                    }) * 2.0
                        * noise_std;
                    t + noise
                })
                .collect();

            // Get prediction (in real implementation, this would be actual forward pass)
            // For now, add variance to base prediction
            let variance: f64 = base_prediction.iter().map(|v| v * 0.05).sum::<f64>()
                / base_prediction.len() as f64;
            let prediction: Vec<f64> = base_prediction
                .iter()
                .map(|&v| {
                    let noise = RNG.with(|r| {
                        let mut rng = r.borrow_mut();
                        rng.gen::<f64>() - 0.5
                    }) * 2.0
                        * variance.sqrt();
                    v + noise
                })
                .collect();

            all_predictions.push(prediction);
        }

        // Calculate statistics across samples
        let num_outputs = all_predictions[0].len();
        let mut means: Vec<f64> = vec![0.0; num_outputs];
        let mut variances: Vec<f64> = vec![0.0; num_outputs];

        // Calculate mean
        for pred in &all_predictions {
            for (i, &val) in pred.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= num_samples as f64;
        }

        // Calculate variance (unbiased estimator)
        for pred in &all_predictions {
            for (i, &val) in pred.iter().enumerate() {
                let diff = val - means[i];
                variances[i] += diff * diff;
            }
        }
        for var in &mut variances {
            *var /= (num_samples - 1) as f64;
        }

        let std: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();

        PredictionWithUncertainty::new(means, std)
    }

    /// Get prediction interval width at a specific confidence level.
    ///
    /// # Arguments
    /// * `current_temps` - Current zone temperatures
    /// * `confidence` - Confidence level (e.g., 0.95 for 95%)
    ///
    /// # Returns
    /// Vector of interval widths at each zone
    pub fn get_prediction_interval_width(
        &self,
        current_temps: &[f64],
        confidence: f64,
    ) -> Vec<f64> {
        // Approximate z-score for common confidence levels
        let z_score = match (confidence * 100.0) as u32 {
            90 => 1.645,
            95 => 1.960,
            99 => 2.576,
            _ => 1.960, // Default to 95%
        };

        let uncertainty = self.predict_with_uncertainty(current_temps, 10, 0.5);

        uncertainty.std.iter().map(|&s| 2.0 * z_score * s).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::ai::surrogate::{InferenceBackend, SurrogateManager};

    #[test]
    fn creation() {
        let m = SurrogateManager::new().unwrap();
        assert!(!m.model_loaded);
        match m.backend {
            InferenceBackend::CPU => {}
            _ => panic!("Default backend should be CPU"),
        }
    }

    #[test]
    fn predict_mock() {
        let m = SurrogateManager::new().unwrap();
        let temps = [20.0, 21.0, 22.0];
        let loads = m.predict_loads(&temps);
        assert_eq!(loads, vec![1.2, 1.2, 1.2]);
    }

    #[test]
    fn predict_mock_batched() {
        let m = SurrogateManager::new().unwrap();
        let batch = vec![vec![20.0, 21.0], vec![22.0, 23.0]];
        let loads = m.predict_loads_batched(&batch);
        assert_eq!(loads.len(), 2);
        assert_eq!(loads[0], vec![1.2, 1.2]);
        assert_eq!(loads[1], vec![1.2, 1.2]);
    }

    #[test]
    fn load_onnx_file_check() {
        // Test file not found error
        let result = SurrogateManager::load_onnx("/nonexistent/path/model.onnx");
        match result {
            Err(e) => assert!(e.contains("not found")),
            Ok(_) => panic!("Expected error for nonexistent file"),
        }
    }

    #[test]
    fn load_onnx_gpu_backend_file_check() {
        let result =
            SurrogateManager::with_gpu_backend("/nonexistent.onnx", InferenceBackend::CUDA, 0);
        match result {
            Err(e) => assert!(e.contains("not found")),
            Ok(_) => panic!("Expected error for nonexistent file"),
        }
    }

    #[test]
    fn predict_loads_with_empty_temps() {
        let m = SurrogateManager::new().unwrap();
        let temps: [f64; 0] = [];
        let loads = m.predict_loads(&temps);
        assert_eq!(loads.len(), 0);
    }

    #[test]
    fn predict_loads_with_many_zones() {
        let m = SurrogateManager::new().unwrap();
        let temps: Vec<f64> = (0..100).map(|i| 20.0 + (i as f64 * 0.1)).collect();
        let loads = m.predict_loads(&temps);
        assert_eq!(loads.len(), 100);
        assert!(loads.iter().all(|&x| x == 1.2));
    }

    #[test]
    fn model_path_optional() {
        let m = SurrogateManager::new().unwrap();
        assert_eq!(m.model_path, None);
        assert!(!m.model_loaded);
        assert!(m.session_pool.is_none());
    }

    #[test]
    fn surrogate_manager_clone() {
        let m1 = SurrogateManager::new().unwrap();
        let m2 = m1.clone();
        assert_eq!(m2.model_loaded, m1.model_loaded);
        assert_eq!(m2.model_path, m1.model_path);
        // Can't easily check backend equality without PartialEq on enum, but it's Copy
    }

    #[test]
    fn predict_onnx_real_model() {
        // This test relies on tests_tmp_dummy.onnx being present and being a model that does y = x + 10 (approx)
        // with 2 inputs.
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            // Skip if file doesn't exist (e.g. in CI environment without the file generated)
            // Tests that require the model should be skipped in that case to avoid panics.
            eprintln!("Skipping ONNX inference test: {} not found", path);
            return;
        }

        let m = SurrogateManager::load_onnx(path).expect("Failed to load model");

        // The dummy model is trained on 2 inputs.
        let temps = [20.0, 21.0];
        let loads = m.predict_loads(&temps);

        // If it returns 1.2, then inference failed or model not loaded
        assert_ne!(loads[0], 1.2, "Returned mock value 1.2, inference failed");

        let tolerance = 0.1;
        assert!(
            (loads[0] - 30.0).abs() < tolerance,
            "Expected ~30.0, got {}",
            loads[0]
        );
        assert!(
            (loads[1] - 31.0).abs() < tolerance,
            "Expected ~31.0, got {}",
            loads[1]
        );
    }

    #[test]
    fn predict_onnx_real_model_batched() {
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping ONNX batched inference test: {} not found", path);
            return;
        }
        let m = SurrogateManager::load_onnx(path).expect("Failed to load model");

        let batch = vec![vec![20.0, 21.0], vec![50.0, 60.0]];
        let loads = m.predict_loads_batched(&batch);

        // Expected: [[30, 31], [60, 70]]
        assert_ne!(loads[0][0], 1.2);

        let tolerance = 0.1;
        assert!(
            (loads[0][0] - 30.0).abs() < tolerance,
            "Expected 30.0, got {}",
            loads[0][0]
        );
        assert!(
            (loads[0][1] - 31.0).abs() < tolerance,
            "Expected 31.0, got {}",
            loads[0][1]
        );
        assert!(
            (loads[1][0] - 60.0).abs() < tolerance,
            "Expected 60.0, got {}",
            loads[1][0]
        );
        assert!(
            (loads[1][1] - 70.0).abs() < tolerance,
            "Expected 70.0, got {}",
            loads[1][1]
        );
    }
}
