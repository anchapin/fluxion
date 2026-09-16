//! Surrogate configuration and utility tests for src/ai/surrogate.rs
//!
//! Tests pure logic functions that don't require ONNX models or GPU hardware.

use fluxion::ai::surrogate::{
    InferenceBackend, InferenceMetrics, MultiDeviceConfig, PredictionWithUncertainty,
    QuantizationConfig, QuantizationType, SurrogateManager,
};

// QuantizationConfig tests

#[test]
fn test_quantization_config_fp32() {
    let config = QuantizationConfig::fp32();
    assert!(matches!(config.quantization_type, QuantizationType::FP32));
    assert!(!config.auto_quantize);
}

#[test]
fn test_quantization_config_fp16() {
    let config = QuantizationConfig::fp16();
    assert!(matches!(config.quantization_type, QuantizationType::FP16));
    assert!(config.auto_quantize);
}

#[test]
fn test_quantization_config_int8() {
    let config = QuantizationConfig::int8();
    assert!(matches!(config.quantization_type, QuantizationType::INT8));
    assert!(config.auto_quantize);
}

#[test]
fn test_quantization_config_default() {
    let config = QuantizationConfig::default();
    assert!(matches!(config.quantization_type, QuantizationType::FP32));
    assert!(!config.auto_quantize);
}

// MultiDeviceConfig tests

#[test]
fn test_multi_device_config_single_gpu() {
    let config = MultiDeviceConfig::single_gpu(0);
    assert_eq!(config.device_ids, vec![0]);
    assert_eq!(config.sessions_per_device, 4);
    assert!(!config.auto_select);
    assert!(config.enable_affinity);
    assert!(config.fallback_to_cpu);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_multi_device_config_single_gpu_device_1() {
    let config = MultiDeviceConfig::single_gpu(1);
    assert_eq!(config.device_ids, vec![1]);
}

#[test]
fn test_multi_device_config_multi_gpu() {
    let config = MultiDeviceConfig::multi_gpu(vec![0, 1, 2]);
    assert_eq!(config.device_ids, vec![0, 1, 2]);
    assert_eq!(config.sessions_per_device, 2);
    assert!(!config.auto_select);
    assert!(config.enable_affinity);
    assert!(config.fallback_to_cpu);
}

#[test]
fn test_multi_device_config_auto() {
    let config = MultiDeviceConfig::auto();
    assert!(config.device_ids.is_empty());
    assert_eq!(config.sessions_per_device, 4);
    assert!(config.auto_select);
    assert!(!config.enable_affinity);
    assert!(config.fallback_to_cpu);
}

#[test]
fn test_multi_device_config_default() {
    let config = MultiDeviceConfig::default();
    assert!(config.device_ids.is_empty());
    assert_eq!(config.sessions_per_device, 0);
    assert!(!config.auto_select);
    assert!(!config.enable_affinity);
    assert!(!config.fallback_to_cpu);
}

// InferenceMetrics tests

#[test]
fn test_inference_metrics_default() {
    let metrics = InferenceMetrics::default();
    assert_eq!(metrics.avg_inference_time_ms, 0.0);
    assert_eq!(metrics.num_inferences, 0);
    assert_eq!(metrics.peak_memory_mb, 0.0);
    assert_eq!(metrics.throughput, 0.0);
}

#[test]
fn test_inference_metrics_record_single_inference() {
    let mut metrics = InferenceMetrics::default();
    metrics.record_inference(100.0);

    assert_eq!(metrics.avg_inference_time_ms, 100.0);
    assert_eq!(metrics.num_inferences, 1);
    assert!((metrics.throughput - 10.0).abs() < 0.01); // 1000/100 = 10
}

#[test]
fn test_inference_metrics_record_multiple_inferences() {
    let mut metrics = InferenceMetrics::default();
    metrics.record_inference(100.0);
    metrics.record_inference(200.0);

    assert_eq!(metrics.avg_inference_time_ms, 150.0);
    assert_eq!(metrics.num_inferences, 2);
    assert!((metrics.throughput - 1000.0 / 150.0).abs() < 0.01);
}

#[test]
fn test_inference_metrics_record_many_inferences() {
    let mut metrics = InferenceMetrics::default();
    for i in 0..10 {
        metrics.record_inference((i + 1) as f64 * 10.0);
    }

    assert_eq!(metrics.num_inferences, 10);
    // Average of 10, 20, 30, ..., 100 = 55
    assert!((metrics.avg_inference_time_ms - 55.0).abs() < 0.01);
}

#[test]
fn test_inference_metrics_reset() {
    let mut metrics = InferenceMetrics::default();
    metrics.record_inference(100.0);
    metrics.reset();

    assert_eq!(metrics.avg_inference_time_ms, 0.0);
    assert_eq!(metrics.num_inferences, 0);
    assert_eq!(metrics.peak_memory_mb, 0.0);
    assert_eq!(metrics.throughput, 0.0);
}

#[test]
fn test_inference_metrics_throughput_calculation() {
    let mut metrics = InferenceMetrics::default();
    metrics.record_inference(50.0); // 50ms per inference

    // Throughput = 1000 / avg_time_ms = 1000 / 50 = 20 inferences/sec
    assert!((metrics.throughput - 20.0).abs() < 0.01);
}

#[test]
fn test_inference_metrics_fast_inference() {
    let mut metrics = InferenceMetrics::default();
    metrics.record_inference(1.0); // 1ms per inference

    assert!((metrics.throughput - 1000.0).abs() < 0.01);
}

#[test]
fn test_inference_metrics_slow_inference() {
    let mut metrics = InferenceMetrics::default();
    metrics.record_inference(1000.0); // 1s per inference

    assert!((metrics.throughput - 1.0).abs() < 0.01);
}

// PredictionWithUncertainty tests

#[test]
fn test_prediction_with_uncertainty_new() {
    let mean = vec![10.0, 20.0, 30.0];
    let std = vec![1.0, 2.0, 3.0];
    let pred = PredictionWithUncertainty::new(mean.clone(), std.clone());

    assert_eq!(pred.mean, mean);
    assert_eq!(pred.std, std);

    // Bounds should be mean ± 2*std
    assert_eq!(pred.lower_bound, vec![8.0, 16.0, 24.0]);
    assert_eq!(pred.upper_bound, vec![12.0, 24.0, 36.0]);
}

#[test]
fn test_prediction_with_uncertainty_single_value() {
    let mean = vec![50.0];
    let std = vec![5.0];
    let pred = PredictionWithUncertainty::new(mean, std);

    assert_eq!(pred.lower_bound, vec![40.0]);
    assert_eq!(pred.upper_bound, vec![60.0]);
}

#[test]
fn test_prediction_with_uncertainty_zero_std() {
    let mean = vec![10.0, 20.0];
    let std = vec![0.0, 0.0];
    let pred = PredictionWithUncertainty::new(mean.clone(), std);

    assert_eq!(pred.lower_bound, mean);
    assert_eq!(pred.upper_bound, mean);
}

#[test]
fn test_prediction_with_uncertainty_large_std() {
    let mean = vec![100.0];
    let std = vec![50.0];
    let pred = PredictionWithUncertainty::new(mean, std);

    assert_eq!(pred.lower_bound, vec![0.0]);
    assert_eq!(pred.upper_bound, vec![200.0]);
}

#[test]
fn test_prediction_with_uncertainty_negative_mean() {
    let mean = vec![-10.0];
    let std = vec![2.0];
    let pred = PredictionWithUncertainty::new(mean, std);

    assert_eq!(pred.lower_bound, vec![-14.0]);
    assert_eq!(pred.upper_bound, vec![-6.0]);
}

#[test]
fn test_prediction_with_uncertainty_empty() {
    let pred = PredictionWithUncertainty::new(vec![], vec![]);

    assert!(pred.mean.is_empty());
    assert!(pred.std.is_empty());
    assert!(pred.lower_bound.is_empty());
    assert!(pred.upper_bound.is_empty());
}

#[test]
fn test_prediction_with_uncertainty_bounds_width() {
    let mean = vec![0.0];
    let std = vec![10.0];
    let pred = PredictionWithUncertainty::new(mean, std);

    // Width should be 4*std (from -2σ to +2σ)
    let width = pred.upper_bound[0] - pred.lower_bound[0];
    assert!((width - 40.0).abs() < 0.01);
}

// SurrogateManager basic tests

#[test]
fn test_surrogate_manager_new() {
    let manager = SurrogateManager::new().unwrap();
    assert!(!manager.model_loaded);
    assert!(manager.model_path.is_none());
    assert!(manager.session_pool.is_none());
    assert!(matches!(manager.backend, InferenceBackend::CPU));
    assert_eq!(manager.device_id, 0);
}

#[test]
fn test_surrogate_manager_clone() {
    let manager = SurrogateManager::new().unwrap();
    let cloned = manager.clone();

    assert_eq!(cloned.model_loaded, manager.model_loaded);
    assert_eq!(cloned.model_path, manager.model_path);
    assert_eq!(cloned.device_id, manager.device_id);
}

#[test]
fn test_surrogate_manager_load_onnx_nonexistent_file() {
    let result = SurrogateManager::load_onnx("/nonexistent/path/model.onnx");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("not found"));
}

#[test]
fn test_surrogate_manager_with_gpu_backend_nonexistent_file() {
    let result = SurrogateManager::with_gpu_backend("/nonexistent.onnx", InferenceBackend::CPU, 0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("not found"));
}

#[test]
fn test_surrogate_manager_with_gpu_backend_cuda_nonexistent() {
    let result = SurrogateManager::with_gpu_backend("/nonexistent.onnx", InferenceBackend::CUDA, 0);
    assert!(result.is_err());
}

#[test]
fn test_surrogate_manager_with_gpu_backend_coreml_nonexistent() {
    let result =
        SurrogateManager::with_gpu_backend("/nonexistent.onnx", InferenceBackend::CoreML, 0);
    assert!(result.is_err());
}

#[test]
fn test_surrogate_manager_with_gpu_backend_directml_nonexistent() {
    let result =
        SurrogateManager::with_gpu_backend("/nonexistent.onnx", InferenceBackend::DirectML, 0);
    assert!(result.is_err());
}

#[test]
fn test_surrogate_manager_with_gpu_backend_openvino_nonexistent() {
    let result =
        SurrogateManager::with_gpu_backend("/nonexistent.onnx", InferenceBackend::OpenVINO, 0);
    assert!(result.is_err());
}

#[test]
fn test_inference_backend_default() {
    let backend = InferenceBackend::default();
    assert!(matches!(backend, InferenceBackend::CPU));
}

/// Issue #1336: explicitly pin the safe default to detect any future drift
/// in the `#[default]` enum tag. CPU is the only universally-available
/// execution provider, so any switch away from it requires intentional
/// review and a corresponding ARCHITECTURE.md update.
#[test]
fn test_inference_backend_default_is_cpu() {
    let backend = InferenceBackend::default();
    assert!(
        matches!(backend, InferenceBackend::CPU),
        "InferenceBackend::default() must be CPU; got {:?}",
        backend
    );
    // Companion assertion: the backend must equal itself via PartialEq
    // (catches accidental Eq-semantics changes).
    assert_eq!(backend, InferenceBackend::CPU);
}

#[test]
fn test_inference_backend_variants() {
    // Just verify all variants exist and can be created
    let _cpu = InferenceBackend::CPU;
    let _cuda = InferenceBackend::CUDA;
    let _coreml = InferenceBackend::CoreML;
    let _directml = InferenceBackend::DirectML;
    let _openvino = InferenceBackend::OpenVINO;
}

#[test]
fn test_quantization_type_variants() {
    let _fp32 = QuantizationType::FP32;
    let _fp16 = QuantizationType::FP16;
    let _int8 = QuantizationType::INT8;
}
