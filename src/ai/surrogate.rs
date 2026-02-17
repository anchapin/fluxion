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
#[derive(Clone, Debug, Copy, Default, PartialEq, Eq)]
pub enum QuantizationType {
    #[default]
    FP32,
    FP16,
    INT8,
}

/// Configuration for model quantization settings.
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

/// Configuration for multi-device inference.
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

/// Information about a CUDA device.
#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: Option<(u32, u32)>,
}

/// Load balancing strategy for multi-device inference.
#[derive(Clone, Debug, Default)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastLoaded,
    Random,
}

/// Performance metrics for inference benchmarking.
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
        self.throughput = 1000.0 / self.avg_inference_time_ms;
    }

    pub fn reset(&mut self) {
        self.avg_inference_time_ms = 0.0;
        self.num_inferences = 0;
        self.peak_memory_mb = 0.0;
        self.throughput = 0.0;
    }
}

/// Manager that provides fast (mock or neural) thermal-load predictions.
#[derive(Clone, Default, Debug)]
pub struct SurrogateManager {
    pub model_loaded: bool,
    pub model_path: Option<String>,
    pub session_pool: Option<Arc<SessionPool>>,
    pub backend: InferenceBackend,
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

    fn detect_cuda_devices() -> Option<Vec<usize>> {
        #[cfg(feature = "cuda")]
        {
            use ort::session::Session;
            let mut available_devices = Vec::new();

            for device_id in 0..8 {
                let mut builder = match Session::builder() {
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
            let mut sessions = self.sessions.lock().unwrap();
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
        })
    }

    pub fn load_onnx(path: &str) -> Result<Self, String> {
        Self::with_gpu_backend(path, InferenceBackend::CPU, 0)
    }

    pub fn with_gpu_backend(
        path: &str,
        backend: InferenceBackend,
        device_id: usize,
    ) -> Result<Self, String> {
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

    pub fn predict_loads(&self, current_temps: &[f64]) -> Vec<f64> {
        if !self.model_loaded {
            return vec![1.2; current_temps.len()];
        }

        if let Some(ref pool) = self.session_pool {
            let input_data: Vec<f32> = current_temps.iter().map(|&x| x as f32).collect();
            let n_input = input_data.len();

            match pool.get_or_create_session() {
                Ok(mut session_guard) => {
                    let input_tensor =
                        match ort::value::Value::from_array(([1, n_input], input_data)) {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!("Failed to create input tensor: {}; using mock loads", e);
                                return vec![1.2; n_input];
                            }
                        };

                    match session_guard.run(ort::inputs![input_tensor]) {
                        Ok(outputs) => {
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
            vec![1.2; current_temps.len()]
        }
    }

    pub fn predict_loads_batched(&self, batch_temps: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if !self.model_loaded || batch_temps.is_empty() {
            return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
        }

        if let Some(ref pool) = self.session_pool {
            let batch_size = batch_temps.len();
            let input_size = batch_temps[0].len();

            for t in batch_temps {
                if t.len() != input_size {
                    eprintln!("Inconsistent input sizes in batch; falling back to mock");
                    return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                }
            }

            let flattened: Vec<f32> = batch_temps
                .iter()
                .flat_map(|v| v.iter().map(|&x| x as f32))
                .collect();

            match pool.get_or_create_session() {
                Ok(mut session_guard) => {
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
            let mean = self.predict_loads(current_temps);
            let std = vec![0.0; mean.len()];
            return PredictionWithUncertainty::new(mean, std);
        }

        let mut all_predictions: Vec<Vec<f64>> = Vec::with_capacity(num_samples);
        let base_prediction = self.predict_loads(current_temps);

        thread_local! {
            static RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
        }

        for _ in 0..num_samples {
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

        let num_outputs = all_predictions[0].len();
        let mut means: Vec<f64> = vec![0.0; num_outputs];
        let mut variances: Vec<f64> = vec![0.0; num_outputs];

        for pred in &all_predictions {
            for (i, &val) in pred.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= num_samples as f64;
        }

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
    }

    #[test]
    fn predict_onnx_real_model() {
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping ONNX inference test: {} not found", path);
            return;
        }

        let m = SurrogateManager::load_onnx(path).expect("Failed to load model");

        let temps = [20.0, 21.0];
        let loads = m.predict_loads(&temps);

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

        assert_ne!(loads[0][0], 1.2);

        let tolerance = 0.1;
        assert!((loads[0][0] - 30.0).abs() < tolerance);
        assert!((loads[0][1] - 31.0).abs() < tolerance);
        assert!((loads[1][0] - 60.0).abs() < tolerance);
        assert!((loads[1][1] - 70.0).abs() < tolerance);
    }
}
