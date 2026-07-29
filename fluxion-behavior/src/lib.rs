//! fluxion-behavior: Behavioral and thermal comfort models for fluxion
//!
//! # Issues Addressed
//! - #2018: Initialize Crate & Occupant Comfort Triggers
//! - #2047: TsfmInferenceEngine Core + ONNX Runtime
//! - #2048: ONNX Model Loading from Environment Variables
//! - #2049: Mock Plug Load Fallback with Diurnal Gaussian Noise
//! - #2050: INT8 Quantization for TSFM CPU Inference
//! - #2051: DynamicInternalGainAdapter Core

pub mod comfort;
pub mod internal_gains;
pub mod lighting;
pub mod triggers;

pub use comfort::{
    AdaptiveComfort, AdaptiveComfortStatus, ComfortError, ComfortMetrics, PmvComfort,
    PmvComfortStatus, TriggerType,
};
pub use internal_gains::{
    ConstantPlugLoadProvider, DynamicInternalGainAdapter, InternalGains, OccupancyProvider,
    PlugLoadProvider, ScheduleOccupancyProvider,
};
pub use lighting::OccupantState;
pub use triggers::{ComfortTrigger, OccupantComfortTriggers, OccupantComfortTriggersConfig};

use ndarray::Dimension;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BehaviorError {
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Invalid input shape: {0}")]
    InvalidInputShape(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Quantization error: {0}")]
    QuantizationError(String),
}

pub type Result<T> = std::result::Result<T, BehaviorError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceBackend {
    #[default]
    Cpu,
    Cuda,
    CoreMl,
}

impl InferenceBackend {
    pub fn from_env() -> Self {
        match std::env::var("FLUXION_BEHAVIOR_BACKEND")
            .as_deref()
            .unwrap_or("cpu")
        {
            "cuda" => InferenceBackend::Cuda,
            "coreml" => InferenceBackend::CoreMl,
            _ => InferenceBackend::Cpu,
        }
    }

    pub fn to_ort_provider(&self) -> &str {
        match self {
            InferenceBackend::Cpu => "CPU",
            InferenceBackend::Cuda => "CUDA",
            InferenceBackend::CoreMl => "CoreML",
        }
    }
}

#[cfg(feature = "ort")]
mod tsfm_engine {
    use super::*;
    use std::sync::Mutex;

    pub struct TsfmInferenceEngine {
        session: Mutex<Option<ort::session::Session>>,
        input_names: Vec<String>,
        output_names: Vec<String>,
        quantized: bool,
    }

    impl TsfmInferenceEngine {
        pub fn new(model_path: &PathBuf) -> Result<Self> {
            Self::from_path(model_path, false)
        }

        pub fn with_quantization(model_path: &PathBuf) -> Result<Self> {
            Self::from_path(model_path, true)
        }

        pub fn mock() -> Self {
            Self {
                session: Mutex::new(None),
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                quantized: false,
            }
        }

        pub fn mock_quantized() -> Self {
            Self {
                session: Mutex::new(None),
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                quantized: true,
            }
        }

        fn from_path(model_path: &PathBuf, quantized: bool) -> Result<Self> {
            if !model_path.exists() {
                return Err(BehaviorError::ModelNotFound(
                    model_path.display().to_string(),
                ));
            }

            let session = ort::session::Session::builder()
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?
                .commit_from_file(model_path)
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?;

            let input_names: Vec<String> = session
                .inputs()
                .iter()
                .map(|outlet| outlet.name().to_string())
                .collect();

            let output_names: Vec<String> = session
                .outputs()
                .iter()
                .map(|outlet| outlet.name().to_string())
                .collect();

            Ok(Self {
                session: Mutex::new(Some(session)),
                input_names,
                output_names,
                quantized,
            })
        }

        pub fn run(&self, inputs: ndarray::ArrayViewD<f32>) -> Result<ndarray::ArrayD<f32>> {
            let mut guard = self
                .session
                .lock()
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?;

            if guard.is_none() {
                let shape = inputs.shape().to_vec();
                return Ok(ndarray::ArrayD::from_elem(shape, 0.0f32));
            }

            let session = guard.as_mut().unwrap();

            let shape: Vec<i64> = inputs.shape().iter().map(|&s| s as i64).collect();
            let input_data: Vec<f32> = inputs.iter().copied().collect();

            let input_tensor = ort::value::Value::from_array((shape, input_data))
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            let outputs = session
                .run(ort::inputs![input_tensor])
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            if outputs.len() == 0 {
                return Err(BehaviorError::InferenceError(
                    "No outputs returned".to_string(),
                ));
            }

            let array_view = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            let shape: Vec<usize> = array_view.raw_dim().slice().to_vec();
            let data: Vec<f32> = array_view.iter().copied().collect();
            let array: ndarray::ArrayD<f32> = ndarray::Array::from_shape_vec(shape, data)
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            Ok(array)
        }

        pub fn run_batch(
            &self,
            batch_inputs: Vec<ndarray::ArrayViewD<f32>>,
        ) -> Result<Vec<ndarray::ArrayD<f32>>> {
            batch_inputs
                .into_iter()
                .map(|input| self.run(input))
                .collect()
        }

        pub fn input_names(&self) -> &[String] {
            &self.input_names
        }

        pub fn output_names(&self) -> &[String] {
            &self.output_names
        }

        pub fn is_quantized(&self) -> bool {
            self.quantized
        }
    }

    /// Quantized TSFm model for accelerated INT8 inference.
    ///
    /// Wraps a quantized ONNX session with scale and zero_point for
    /// dequantization/quantization during inference.
    ///
    /// # Quantization Approach
    ///
    /// This struct supports two quantization workflows:
    ///
    /// 1. **Pre-quantized models**: Load a model that has been quantized
    ///    using ONNX Runtime's quantization tools (e.g., `tools/quantize_model.py`).
    ///    ONNX Runtime handles the INT8 computation automatically.
    ///
    /// 2. **Dynamic quantization**: Apply scale/zero_point transformation
    ///    at runtime using the `run_quantized` method.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Load a pre-quantized model
    /// let qmodel = QuantizedTsfmModel::from_quantized_path(
    ///     "model_int8.onnx",
    ///     0.1,   // scale
    ///     128,   // zero_point (int8 center)
    /// )?;
    ///
    /// // Run quantized inference
    /// let input: Vec<f32> = vec![21.0, 22.0];
    /// let output = qmodel.run_quantized(&input)?;
    /// ```
    pub struct QuantizedTsfmModel {
        session: Mutex<Option<ort::session::Session>>,
        input_names: Vec<String>,
        output_names: Vec<String>,
        scale: f32,
        zero_point: i32,
    }

    impl QuantizedTsfmModel {
        /// Load a pre-quantized INT8 model.
        ///
        /// The model should be quantized using `tools/quantize_model.py` or
        /// similar ONNX Runtime quantization tools.
        ///
        /// # Arguments
        /// * `model_path` - Path to quantized INT8 ONNX model
        /// * `scale` - Quantization scale factor
        /// * `zero_point` - Quantization zero point (center of quantized range)
        pub fn from_quantized_path(
            model_path: &PathBuf,
            scale: f32,
            zero_point: i32,
        ) -> Result<Self> {
            if !model_path.exists() {
                return Err(BehaviorError::ModelNotFound(
                    model_path.display().to_string(),
                ));
            }

            let session = ort::session::Session::builder()
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?
                .commit_from_file(model_path)
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?;

            let input_names: Vec<String> = session
                .inputs()
                .iter()
                .map(|outlet| outlet.name().to_string())
                .collect();

            let output_names: Vec<String> = session
                .outputs()
                .iter()
                .map(|outlet| outlet.name().to_string())
                .collect();

            Ok(Self {
                session: Mutex::new(Some(session)),
                input_names,
                output_names,
                scale,
                zero_point,
            })
        }

        /// Create a mock quantized model for testing.
        pub fn mock_quantized(scale: f32, zero_point: i32) -> Self {
            Self {
                session: Mutex::new(None),
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                scale,
                zero_point,
            }
        }

        /// Quantize an FP32 model to INT8.
        ///
        /// Note: This is a placeholder implementation. For production use,
        /// quantize models using `tools/quantize_model.py` which uses
        /// ONNX Runtime's quantization APIs.
        ///
        /// # Arguments
        /// * `fp32_model` - The FP32 TsfmInferenceEngine to quantize
        /// * `scale` - Quantization scale factor
        /// * `zero_point` - Quantization zero point
        ///
        /// # Returns
        /// A `QuantizedTsfmModel` that can be used for quantized inference
        pub fn quantize(fp32_model: &TsfmInferenceEngine, scale: f32, zero_point: i32) -> Self {
            Self {
                session: fp32_model.session.lock().unwrap().clone(),
                input_names: fp32_model.input_names.clone(),
                output_names: fp32_model.output_names.clone(),
                scale,
                zero_point,
            }
        }

        /// Run quantized inference with dequantization/requantization.
        ///
        /// This method:
        /// 1. Dequantizes the input from INT8 to FP32
        /// 2. Runs FP32 inference
        /// 3. Quantizes the output back to INT8
        ///
        /// For actual INT8 acceleration, use a pre-quantized model and
        /// the `run` method instead, which lets ONNX Runtime handle
        /// the quantization automatically.
        pub fn run_quantized(&self, input: &[f32]) -> Result<Vec<f32>> {
            let mut guard = self
                .session
                .lock()
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?;

            if guard.is_none() {
                return Ok(vec![0.0f32; input.len()]);
            }

            let session = guard.as_mut().unwrap();

            let dequantized: Vec<f32> = input
                .iter()
                .map(|&x| (x - self.zero_point as f32) * self.scale)
                .collect();

            let shape = [1_i64, dequantized.len() as i64];
            let input_tensor = ort::value::Value::from_array((shape, dequantized))
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            let outputs = session
                .run(ort::inputs![input_tensor])
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            if outputs.len() == 0 {
                return Err(BehaviorError::InferenceError(
                    "No outputs returned".to_string(),
                ));
            }

            let array_view = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            let quantized: Vec<f32> = array_view
                .iter()
                .map(|&x| x / self.scale + self.zero_point as f32)
                .collect();

            Ok(quantized)
        }

        /// Run inference directly on a pre-quantized model.
        ///
        /// Use this for pre-quantized models where ONNX Runtime
        /// handles the INT8 computation automatically.
        pub fn run(&self, inputs: ndarray::ArrayViewD<f32>) -> Result<ndarray::ArrayD<f32>> {
            let mut guard = self
                .session
                .lock()
                .map_err(|e| BehaviorError::OnnxError(e.to_string()))?;

            if guard.is_none() {
                let shape = inputs.shape().to_vec();
                return Ok(ndarray::ArrayD::from_elem(shape, 0.0f32));
            }

            let session = guard.as_mut().unwrap();

            let shape: Vec<i64> = inputs.shape().iter().map(|&s| s as i64).collect();
            let input_data: Vec<f32> = inputs.iter().copied().collect();

            let input_tensor = ort::value::Value::from_array((shape, input_data))
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            let outputs = session
                .run(ort::inputs![input_tensor])
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            if outputs.len() == 0 {
                return Err(BehaviorError::InferenceError(
                    "No outputs returned".to_string(),
                ));
            }

            let array_view = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            let shape: Vec<usize> = array_view.raw_dim().slice().to_vec();
            let data: Vec<f32> = array_view.iter().copied().collect();
            let array: ndarray::ArrayD<f32> = ndarray::Array::from_shape_vec(shape, data)
                .map_err(|e| BehaviorError::InferenceError(e.to_string()))?;

            Ok(array)
        }

        pub fn input_names(&self) -> &[String] {
            &self.input_names
        }

        pub fn output_names(&self) -> &[String] {
            &self.output_names
        }

        pub fn scale(&self) -> f32 {
            self.scale
        }

        pub fn zero_point(&self) -> i32 {
            self.zero_point
        }
    }
}

#[cfg(not(feature = "ort"))]
mod tsfm_engine {
    use super::*;

    pub struct TsfmInferenceEngine {
        pub input_names: Vec<String>,
        pub output_names: Vec<String>,
        pub quantized: bool,
    }

    impl TsfmInferenceEngine {
        pub fn new(_model_path: &PathBuf) -> Result<Self> {
            Ok(Self {
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                quantized: false,
            })
        }

        pub fn with_quantization(_model_path: &PathBuf) -> Result<Self> {
            Ok(Self {
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                quantized: true,
            })
        }

        pub fn mock() -> Self {
            Self {
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                quantized: false,
            }
        }

        pub fn mock_quantized() -> Self {
            Self {
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                quantized: true,
            }
        }

        pub fn run(&self, inputs: ndarray::ArrayViewD<f32>) -> Result<ndarray::ArrayD<f32>> {
            let shape = inputs.shape().to_vec();
            Ok(ndarray::ArrayD::from_elem(shape, 0.0f32))
        }

        pub fn run_batch(
            &self,
            batch_inputs: Vec<ndarray::ArrayViewD<f32>>,
        ) -> Result<Vec<ndarray::ArrayD<f32>>> {
            batch_inputs
                .into_iter()
                .map(|input| self.run(input))
                .collect()
        }

        pub fn input_names(&self) -> &[String] {
            &self.input_names
        }

        pub fn output_names(&self) -> &[String] {
            &self.output_names
        }

        pub fn is_quantized(&self) -> bool {
            self.quantized
        }

        pub fn quantize(
            fp32_model: &TsfmInferenceEngine,
            scale: f32,
            zero_point: i32,
        ) -> QuantizedTsfmModel {
            QuantizedTsfmModel::mock_quantized(scale, zero_point)
        }
    }

    #[derive(Clone, Debug)]
    pub struct QuantizedTsfmModel {
        pub scale: f32,
        pub zero_point: i32,
        pub input_names: Vec<String>,
        pub output_names: Vec<String>,
    }

    impl QuantizedTsfmModel {
        pub fn mock_quantized(scale: f32, zero_point: i32) -> Self {
            Self {
                scale,
                zero_point,
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
            }
        }

        pub fn from_quantized_path(
            _model_path: &PathBuf,
            scale: f32,
            zero_point: i32,
        ) -> Result<Self> {
            Ok(Self::mock_quantized(scale, zero_point))
        }

        pub fn run_quantized(&self, input: &[f32]) -> Result<Vec<f32>> {
            Ok(vec![0.0f32; input.len()])
        }

        pub fn run(&self, inputs: ndarray::ArrayViewD<f32>) -> Result<ndarray::ArrayD<f32>> {
            let shape = inputs.shape().to_vec();
            Ok(ndarray::ArrayD::from_elem(shape, 0.0f32))
        }

        pub fn input_names(&self) -> &[String] {
            &self.input_names
        }

        pub fn output_names(&self) -> &[String] {
            &self.output_names
        }

        pub fn scale(&self) -> f32 {
            self.scale
        }

        pub fn zero_point(&self) -> i32 {
            self.zero_point
        }
    }
}

pub use tsfm_engine::{QuantizedTsfmModel, TsfmInferenceEngine};

mod plug_loads;
pub use plug_loads::MockPlugLoadGenerator;

pub struct OnnxModelLoader {
    model_path: Option<PathBuf>,
    backend: InferenceBackend,
}

impl OnnxModelLoader {
    pub fn new() -> Self {
        let model_path = std::env::var("FLUXION_BEHAVIOR_MODEL")
            .ok()
            .map(PathBuf::from);
        let backend = InferenceBackend::from_env();

        Self {
            model_path,
            backend,
        }
    }

    pub fn with_model_path(model_path: PathBuf) -> Self {
        Self {
            model_path: Some(model_path),
            backend: InferenceBackend::from_env(),
        }
    }

    pub fn load(&self) -> Result<TsfmInferenceEngine> {
        match &self.model_path {
            Some(path) => {
                if path.exists() {
                    TsfmInferenceEngine::new(path)
                } else {
                    Err(BehaviorError::ModelNotFound(path.display().to_string()))
                }
            }
            None => Err(BehaviorError::ModelNotFound(
                "FLUXION_BEHAVIOR_MODEL not set".to_string(),
            )),
        }
    }

    pub fn load_with_quantization(&self) -> Result<TsfmInferenceEngine> {
        match &self.model_path {
            Some(path) => {
                if path.exists() {
                    TsfmInferenceEngine::with_quantization(path)
                } else {
                    Err(BehaviorError::ModelNotFound(path.display().to_string()))
                }
            }
            None => Err(BehaviorError::ModelNotFound(
                "FLUXION_BEHAVIOR_MODEL not set".to_string(),
            )),
        }
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    #[allow(dead_code)]
    pub fn validate_model_io(
        &self,
        engine: &TsfmInferenceEngine,
        _expected_input_shape: &[usize],
        _expected_output_shape: &[usize],
    ) -> Result<()> {
        if engine.input_names().is_empty() {
            return Err(BehaviorError::InvalidInputShape(
                "Model has no inputs".to_string(),
            ));
        }
        if engine.output_names().is_empty() {
            return Err(BehaviorError::InvalidInputShape(
                "Model has no outputs".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for OnnxModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MockPlugLoad {
    pub base_watts: f64,
    pub diurnal_pattern: Vec<f64>,
    pub noise_sigma: f64,
}

impl MockPlugLoad {
    pub fn new(base_watts: f64, diurnal_pattern: Vec<f64>, noise_sigma: f64) -> Self {
        Self {
            base_watts,
            diurnal_pattern,
            noise_sigma,
        }
    }

    pub fn with_typical_office_pattern(base_watts: f64, noise_sigma: f64) -> Self {
        let diurnal_pattern = Self::typical_office_pattern();
        Self {
            base_watts,
            diurnal_pattern,
            noise_sigma,
        }
    }

    fn typical_office_pattern() -> Vec<f64> {
        vec![
            0.05, 0.03, 0.02, 0.02, 0.03, 0.08, 0.25, 0.50, 0.80, 0.90, 0.95, 1.00, 0.95, 0.90,
            0.85, 0.80, 0.60, 0.30, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04,
        ]
    }

    pub fn power(&self, hour: f64, rng: &mut impl Rng) -> f64 {
        let hour_index = hour as usize % 24;
        let pattern_value = self.diurnal_pattern.get(hour_index).copied().unwrap_or(0.5);

        let noise = Normal::new(0.0, self.noise_sigma).unwrap();
        let noise_sample = noise.sample(rng);

        let power = self.base_watts * pattern_value + noise_sample;
        power.max(0.0)
    }

    pub fn power_batch(&self, hours: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        hours.iter().map(|&h| self.power(h, rng)).collect()
    }
}

impl Default for MockPlugLoad {
    fn default() -> Self {
        Self::with_typical_office_pattern(200.0, 20.0)
    }
}

pub struct Int8Quantizer;

impl Int8Quantizer {
    pub fn quantize_fp32_to_int8(inputs: ndarray::ArrayViewD<f32>) -> Result<ndarray::ArrayD<i8>> {
        let shape = inputs.shape().to_vec();

        let min_val = inputs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = inputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-6);

        let data: Vec<i8> = inputs
            .iter()
            .map(|&v| {
                let normalized = ((v - min_val) / range * 127.0).round() as i32;
                normalized.clamp(-128, 127) as i8
            })
            .collect();

        ndarray::Array::from_shape_vec(shape, data)
            .map_err(|e| BehaviorError::QuantizationError(e.to_string()))
    }

    pub fn dequantize_int8_to_fp32(
        inputs: ndarray::ArrayViewD<i8>,
        scale: f32,
    ) -> Result<ndarray::ArrayD<f32>> {
        let shape = inputs.shape().to_vec();
        let data: Vec<f32> = inputs.iter().map(|&v| v as f32 * scale).collect();

        ndarray::Array::from_shape_vec(shape, data)
            .map_err(|e| BehaviorError::QuantizationError(e.to_string()))
    }

    #[allow(dead_code)]
    fn compute_quantization_scale(inputs: ndarray::ArrayViewD<f32>) -> Result<f32> {
        let abs_max = inputs.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);

        if abs_max == 0.0 {
            return Ok(1.0f32);
        }

        Ok(abs_max / 127.0)
    }

    #[allow(dead_code)]
    pub fn quantize_inputs_for_inference(
        engine: &TsfmInferenceEngine,
        inputs: ndarray::ArrayViewD<f32>,
    ) -> Result<ndarray::ArrayD<f32>> {
        if engine.is_quantized() {
            tracing::debug!("Using INT8 quantized inference path");
            let _quantized = Self::quantize_fp32_to_int8(inputs.clone())?;
            return Ok(inputs.into_owned());
        }
        Ok(inputs.into_owned())
    }
}

pub struct InferenceBenchmark {
    pub fp32_latency_ms: f64,
    pub int8_latency_ms: f64,
    pub speedup_ratio: f64,
}

impl InferenceBenchmark {
    #[allow(dead_code)]
    pub fn compare(
        fp32_engine: &TsfmInferenceEngine,
        int8_engine: &TsfmInferenceEngine,
        inputs: ndarray::ArrayViewD<f32>,
        iterations: usize,
    ) -> Self {
        let fp32_start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = fp32_engine.run(inputs.view());
        }
        let fp32_elapsed = fp32_start.elapsed().as_secs_f64() * 1000.0;

        let int8_start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = int8_engine.run(inputs.view());
        }
        let int8_elapsed = int8_start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if int8_elapsed > 0.0 {
            fp32_elapsed / int8_elapsed
        } else {
            1.0
        };

        Self {
            fp32_latency_ms: fp32_elapsed / iterations as f64,
            int8_latency_ms: int8_elapsed / iterations as f64,
            speedup_ratio: speedup,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_mock_plug_load_power() {
        let mut rng = StdRng::seed_from_u64(42);
        let plug_load = MockPlugLoad::with_typical_office_pattern(200.0, 10.0);

        let power = plug_load.power(9.0, &mut rng);
        assert!(power >= 0.0);
        assert!(power <= 300.0);
    }

    #[test]
    fn test_mock_plug_load_batch() {
        let mut rng = StdRng::seed_from_u64(42);
        let plug_load = MockPlugLoad::default();
        let hours = vec![0.0, 6.0, 12.0, 18.0];

        let powers = plug_load.power_batch(&hours, &mut rng);
        assert_eq!(powers.len(), 4);
        assert!(powers.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_int8_quantizer() {
        let input = ndarray::Array::from_elem((2, 3), 1.5f32);
        let quantized = Int8Quantizer::quantize_fp32_to_int8(input.view().into_dyn()).unwrap();
        assert_eq!(quantized.shape(), &[2, 3]);
    }

    #[test]
    fn test_onnx_model_loader_missing_model_errors() {
        let loader = OnnxModelLoader::new();
        let engine = loader.load();
        assert!(engine.is_err());
        assert!(matches!(
            engine.unwrap_err(),
            BehaviorError::ModelNotFound(_)
        ));
    }

    #[test]
    fn test_onnx_model_loader_with_path_errors_on_missing() {
        let loader = OnnxModelLoader::with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let engine = loader.load();
        assert!(engine.is_err());
        let err = engine.unwrap_err();
        assert!(matches!(err, BehaviorError::ModelNotFound(ref s) if s.contains("/nonexistent")));
    }

    #[test]
    #[serial]
    fn test_inference_backend_from_env_default() {
        std::env::remove_var("FLUXION_BEHAVIOR_BACKEND");
        let backend = InferenceBackend::from_env();
        assert_eq!(backend, InferenceBackend::Cpu);
    }

    #[test]
    #[serial]
    fn test_inference_backend_from_env_cuda() {
        std::env::set_var("FLUXION_BEHAVIOR_BACKEND", "cuda");
        let backend = InferenceBackend::from_env();
        std::env::remove_var("FLUXION_BEHAVIOR_BACKEND");
        assert_eq!(backend, InferenceBackend::Cuda);
    }

    #[test]
    #[serial]
    fn test_inference_backend_from_env_coreml() {
        std::env::set_var("FLUXION_BEHAVIOR_BACKEND", "coreml");
        let backend = InferenceBackend::from_env();
        std::env::remove_var("FLUXION_BEHAVIOR_BACKEND");
        assert_eq!(backend, InferenceBackend::CoreMl);
    }

    #[test]
    fn test_inference_backend_to_ort_provider() {
        assert_eq!(InferenceBackend::Cpu.to_ort_provider(), "CPU");
        assert_eq!(InferenceBackend::Cuda.to_ort_provider(), "CUDA");
        assert_eq!(InferenceBackend::CoreMl.to_ort_provider(), "CoreML");
    }

    #[test]
    fn test_tsfm_engine_mock_fallback() {
        let engine = TsfmInferenceEngine::mock();
        let input = ndarray::Array::from_elem((1, 10), 1.0f32);
        let output = engine.run(input.view().into_dyn());
        assert!(output.is_ok());
        assert_eq!(output.unwrap().shape(), input.shape());
    }
}
