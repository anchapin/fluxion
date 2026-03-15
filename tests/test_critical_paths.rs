//! Critical path tests for error handling and edge cases.
//!
//! This test file targets low-coverage critical paths identified in coverage analysis:
//! - ASHRAE 140 validator error paths (19.4% → target 60%)
//! - Thermal model builder edge cases (37.1% → target 70%)
//! - Surrogate manager error handling (31.2% → target 60%)
//! - Weather data parsing errors (already 84%+, but add edge cases)

use fluxion::ai::surrogate::{
    InferenceMetrics, MultiDeviceConfig, QuantizationConfig, SurrogateManager,
};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_model::PhysicsThermalModel;
use fluxion::validation::ashrae_140_validator::ASHRAE140Validator;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::ThermalModelTrait;

#[test]
fn test_validator_handles_missing_reference_file() {
    // Test that validator gracefully handles missing multi-reference database
    let temp_dir = tempfile::tempdir().unwrap();
    let fake_path = temp_dir.path().join("nonexistent.json");

    let validator = ASHRAE140Validator::new().with_multi_reference(&fake_path);

    // Validator should still be functional even without multi-reference data
    // (it falls back to default validation)
    // We just test it doesn't panic - the internal state is private
    let _validator = validator;
}

#[test]
fn test_validator_invalid_reference_file() {
    // Test handling of invalid reference file format
    let temp_dir = tempfile::tempdir().unwrap();
    let invalid_path = temp_dir.path().join("invalid.json");

    // Write invalid JSON
    std::fs::write(&invalid_path, "{ invalid json }").unwrap();

    let validator = ASHRAE140Validator::new().with_multi_reference(&invalid_path);

    // Should gracefully handle parse errors
    // We just test it doesn't panic - the internal state is private
    let _validator = validator;
}

#[test]
fn test_thermal_model_apply_parameters_invalid_u_value() {
    // Test thermal model handles invalid window U-value
    let mut model = ThermalModel::new(1);

    // Test U-value below minimum (0.1 W/m²K)
    let invalid_params_low = vec![0.05, 21.0];
    model.apply_parameters(&invalid_params_low);
    // Model should handle gracefully (clamped or error handling)

    // Test U-value above maximum (5.0 W/m²K)
    let invalid_params_high = vec![10.0, 21.0];
    model.apply_parameters(&invalid_params_high);
    // Model should handle gracefully
}

#[test]
fn test_thermal_model_apply_parameters_invalid_setpoint() {
    // Test thermal model handles invalid HVAC setpoints
    let mut model = ThermalModel::new(1);

    // Test setpoint below minimum (15°C)
    let invalid_params_low = vec![1.5, 10.0];
    model.apply_parameters(&invalid_params_low);

    // Test setpoint above maximum (30°C)
    let invalid_params_high = vec![1.5, 40.0];
    model.apply_parameters(&invalid_params_high);
}

#[test]
fn test_thermal_model_apply_parameters_empty_vector() {
    // Test thermal model handles empty parameter vector
    let mut model = ThermalModel::new(1);
    let empty_params: Vec<f64> = vec![];

    // Should handle gracefully without panic
    model.apply_parameters(&empty_params);
}

#[test]
fn test_thermal_model_apply_parameters_truncated_vector() {
    // Test thermal model handles truncated parameter vector
    let mut model = ThermalModel::new(1);
    let truncated_params = vec![1.5]; // Missing second parameter

    // Should handle gracefully (use defaults for missing)
    model.apply_parameters(&truncated_params);
}

#[test]
fn test_thermal_model_solve_timesteps_zero_steps() {
    // Test thermal model handles zero timesteps
    let mut model = ThermalModel::new(1);
    let surrogates = SurrogateManager::new().unwrap();

    // Should handle zero steps gracefully
    let result = model.solve_timesteps(0, &surrogates, false, None, None, None);

    // Zero steps should result in zero energy
    assert_eq!(result, 0.0);
}

#[test]
fn test_thermal_model_solve_timesteps_single_step() {
    // Test thermal model handles single timestep
    let mut model = ThermalModel::new(1);
    let surrogates = SurrogateManager::new().unwrap();

    // Should handle single step
    let result = model.solve_timesteps(1, &surrogates, false, None, None, None);

    // Should produce some result (not NaN or Inf)
    assert!(result.is_finite());
}

#[test]
fn test_thermal_model_solve_timesteps_negative_steps() {
    // Test thermal model handles negative timestep count
    // Since usize is unsigned, we test with zero instead
    let mut model = ThermalModel::new(1);
    let surrogates = SurrogateManager::new().unwrap();

    // Should handle zero steps gracefully
    let result = model.solve_timesteps(0, &surrogates, false, None, None, None);

    // Should handle gracefully (returns 0)
    assert_eq!(result, 0.0);
}

#[test]
fn test_surrogate_manager_predict_loads_no_model_loaded() {
    // Test surrogate manager handles prediction without loaded model
    let manager = SurrogateManager::new().unwrap();

    // Should return mock predictions when no model loaded
    let temps = vec![20.0, 21.0, 22.0];
    let result = manager.predict_loads(&temps);

    // Should return some prediction (mock values)
    assert!(!result.is_empty());
    assert!(result.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_surrogate_manager_predict_loads_empty_temps() {
    // Test surrogate manager handles empty temperature vector
    let manager = SurrogateManager::new().unwrap();
    let empty_temps: Vec<f64> = vec![];

    // Should handle empty input gracefully
    let result = manager.predict_loads(&empty_temps);

    // Should return empty result or handle gracefully
    assert_eq!(result.len(), 0);
}

#[test]
fn test_surrogate_manager_predict_loads_invalid_temps() {
    // Test surrogate manager handles invalid temperature values
    let manager = SurrogateManager::new().unwrap();

    // Test with NaN temperatures
    let nan_temps = vec![20.0, f64::NAN, 22.0];
    let result = manager.predict_loads(&nan_temps);

    // Should handle NaN values gracefully
    assert!(!result.is_empty());

    // Test with Inf temperatures
    let inf_temps = vec![20.0, f64::INFINITY, 22.0];
    let result = manager.predict_loads(&inf_temps);

    // Should handle Inf values gracefully
    assert!(!result.is_empty());
}

#[test]
fn test_quantization_config_types() {
    // Test quantization configuration
    let fp32 = QuantizationConfig::fp32();
    assert_eq!(
        fp32.quantization_type,
        fluxion::ai::surrogate::QuantizationType::FP32
    );

    let fp16 = QuantizationConfig::fp16();
    assert_eq!(
        fp16.quantization_type,
        fluxion::ai::surrogate::QuantizationType::FP16
    );
    assert!(fp16.auto_quantize);

    let int8 = QuantizationConfig::int8();
    assert_eq!(
        int8.quantization_type,
        fluxion::ai::surrogate::QuantizationType::INT8
    );
    assert!(int8.auto_quantize);
}

#[test]
fn test_inference_metrics_record() {
    // Test inference metrics recording
    let mut metrics = InferenceMetrics::default();

    assert_eq!(metrics.num_inferences, 0);
    assert_eq!(metrics.avg_inference_time_ms, 0.0);

    metrics.record_inference(10.0);
    assert_eq!(metrics.num_inferences, 1);
    assert_eq!(metrics.avg_inference_time_ms, 10.0);

    metrics.record_inference(20.0);
    assert_eq!(metrics.num_inferences, 2);
    assert_eq!(metrics.avg_inference_time_ms, 15.0);
}

#[test]
fn test_inference_metrics_reset() {
    // Test inference metrics reset
    let mut metrics = InferenceMetrics::default();

    metrics.record_inference(10.0);
    metrics.record_inference(20.0);

    assert_eq!(metrics.num_inferences, 2);

    metrics.reset();

    assert_eq!(metrics.num_inferences, 0);
    assert_eq!(metrics.avg_inference_time_ms, 0.0);
    assert_eq!(metrics.peak_memory_mb, 0.0);
    assert_eq!(metrics.throughput, 0.0);
}

#[test]
fn test_multi_device_config_single_gpu() {
    // Test single GPU configuration
    let config = MultiDeviceConfig::single_gpu(0);
    assert_eq!(config.device_ids, vec![0]);
    assert_eq!(config.sessions_per_device, 4);
    assert!(!config.auto_select);
    assert!(config.enable_affinity);
    assert!(config.fallback_to_cpu);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_multi_device_config_multi_gpu() {
    // Test multi-GPU configuration
    let config = MultiDeviceConfig::multi_gpu(vec![0, 1, 2]);
    assert_eq!(config.device_ids, vec![0, 1, 2]);
    assert_eq!(config.sessions_per_device, 2);
    assert!(!config.auto_select);
    assert!(config.enable_affinity);
    assert!(config.fallback_to_cpu);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_multi_device_config_auto() {
    // Test auto-select configuration
    let config = MultiDeviceConfig::auto();
    assert!(config.device_ids.is_empty());
    assert_eq!(config.sessions_per_device, 4);
    assert!(config.auto_select);
    assert!(!config.enable_affinity);
    assert!(config.fallback_to_cpu);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_weather_epw_missing_file() {
    // Test EPW weather source handles missing file
    let nonexistent_path = "path/to/nonexistent/file.epw";
    let result = EpwWeatherSource::from_file(nonexistent_path);

    // Should return error for missing file
    assert!(result.is_err());
}

#[test]
fn test_weather_epw_empty_file() {
    // Test EPW weather source handles empty file
    let temp_dir = tempfile::tempdir().unwrap();
    let empty_epw = temp_dir.path().join("empty.epw");

    std::fs::write(&empty_epw, "").unwrap();

    let result = EpwWeatherSource::from_file(empty_epw.to_str().unwrap());

    // Should return error for empty file
    assert!(result.is_err());
}

#[test]
fn test_weather_epw_invalid_format() {
    // Test EPW weather source handles invalid format
    let temp_dir = tempfile::tempdir().unwrap();
    let invalid_epw = temp_dir.path().join("invalid.epw");

    // Write invalid EPW header
    std::fs::write(&invalid_epw, "INVALID EPW FORMAT").unwrap();

    let result = EpwWeatherSource::from_file(invalid_epw.to_str().unwrap());

    // Should return error for invalid format
    assert!(result.is_err());
}

#[test]
fn test_validator_diagnostics_config() {
    // Test validator with diagnostics configuration
    use fluxion::validation::diagnostic::DiagnosticConfig;

    let config = DiagnosticConfig::full();
    let validator = ASHRAE140Validator::with_diagnostics(config.clone());

    // Should create validator with diagnostics enabled
    // (we can't access internal state, just test it doesn't panic)
    let _validator = validator;
}

#[test]
fn test_validator_full_diagnostics() {
    // Test validator creation with full diagnostics
    let validator = ASHRAE140Validator::with_full_diagnostics();

    // Should create validator with full diagnostics enabled
    // (we can't access internal state, just test it doesn't panic)
    let _validator = validator;
}

#[test]
fn test_thermal_model_builder_creation() {
    // Test thermal model builder pattern
    let model = PhysicsThermalModel::new(1);

    // Should create a valid model
    assert_eq!(model.num_zones(), 1);
    assert!(model.is_valid());
}
