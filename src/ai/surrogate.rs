//! Surrogate manager for fast thermal load predictions.
//!
//! The `SurrogateManager` wraps ONNX Runtime sessions for neural network inference
//! or returns mock predictions during development/testing. It uses `Mutex` for
//! thread-safe interior mutability (required by ORT Session::run which needs &mut).

use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    OpenVINOExecutionProvider,
};
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
    /// Optional ONNX Runtime session for inference using the `ort` crate.
    /// Wrapped in `Mutex` to allow thread-safe interior mutability (Session::run requires &mut).
    pub session: Option<Arc<Mutex<ort::session::Session>>>,
    /// The backend used for inference.
    pub backend: InferenceBackend,
    /// The device ID (e.g. GPU index).
    pub device_id: usize,
}

impl SurrogateManager {
    /// Construct a new `SurrogateManager`.
    pub fn new() -> Result<Self, String> {
        Ok(SurrogateManager {
            model_loaded: false,
            model_path: None,
            session: None,
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
        use ort::session::Session;
        use std::path::Path;

        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }

        // Initialize and build ONNX session
        let mut builder =
            Session::builder().map_err(|e| format!("Failed to create session builder: {}", e))?;

        // Configure execution provider based on backend
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
            InferenceBackend::CPU => {
                // Default behavior, no specific provider needed as CPU is fallback/default
            }
        }

        let session = builder
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

        Ok(SurrogateManager {
            model_loaded: true,
            model_path: Some(path.to_string()),
            session: Some(Arc::new(Mutex::new(session))),
            backend,
            device_id,
        })
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

        // If we have a session, perform inference.
        if let Some(ref session_cell) = self.session {
            use ndarray::Array2;
            use ort::value::TensorRef;

            // Convert input temps into an ndarray 2D float32 array (1, N)
            let n_input = current_temps.len();
            let input_arr = match Array2::from_shape_vec(
                (1, n_input),
                current_temps.iter().map(|&x| x as f32).collect(),
            ) {
                Ok(arr) => arr,
                Err(e) => {
                    eprintln!("Failed to reshape array: {}; using mock loads", e);
                    return vec![1.2; n_input];
                }
            };

            // Try to lock the session for inference
            match session_cell.lock() {
                Ok(mut session) => {
                    // Create a tensor reference from the ndarray
                    let tensor_ref = match TensorRef::from_array_view(&input_arr) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Failed to create tensor ref: {}; using mock loads", e);
                            return vec![1.2; n_input];
                        }
                    };

                    // Run inference using the inputs! macro pattern
                    match session.run(ort::inputs![tensor_ref]) {
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
                    eprintln!("Could not lock ORT session; using mock loads");
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

        if let Some(ref session_cell) = self.session {
            use ndarray::Array2;
            use ort::value::TensorRef;

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
            let input_arr = match Array2::from_shape_vec((batch_size, input_size), flattened) {
                Ok(arr) => arr,
                Err(e) => {
                    eprintln!("Failed to reshape array: {}; using mock loads", e);
                    return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                }
            };

            match session_cell.lock() {
                Ok(mut session) => {
                    let tensor_ref = match TensorRef::from_array_view(&input_arr) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Failed to create tensor ref: {}; using mock loads", e);
                            return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                        }
                    };

                    match session.run(ort::inputs![tensor_ref]) {
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
                            return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                        }
                        Err(e) => {
                            eprintln!("ONNX inference error: {}; using mock loads", e);
                            return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                        }
                    }
                }
                Err(_) => {
                    eprintln!("Could not lock ORT session; using mock loads");
                    return batch_temps.iter().map(|t| vec![1.2; t.len()]).collect();
                }
            }
        }
        // Fallback default
        batch_temps.iter().map(|t| vec![1.2; t.len()]).collect()
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
        assert!(m.session.is_none());
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
            // But for this task, we expect it to exist.
            panic!("tests_tmp_dummy.onnx not found. Run generate_dummy_model.py first.");
        }

        let m = SurrogateManager::load_onnx(path).expect("Failed to load model");

        // The dummy model is trained on 2 inputs.
        let temps = [20.0, 21.0];
        let loads = m.predict_loads(&temps);

        // If it returns 1.2, then inference failed or model not loaded
        assert_ne!(loads[0], 1.2, "Returned mock value 1.2, inference failed");

        let tolerance = 0.1;
        assert!((loads[0] - 30.0).abs() < tolerance, "Expected ~30.0, got {}", loads[0]);
        assert!((loads[1] - 31.0).abs() < tolerance, "Expected ~31.0, got {}", loads[1]);
    }

    #[test]
    fn predict_onnx_real_model_batched() {
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
             panic!("tests_tmp_dummy.onnx not found.");
        }
        let m = SurrogateManager::load_onnx(path).expect("Failed to load model");

        let batch = vec![vec![20.0, 21.0], vec![50.0, 60.0]];
        let loads = m.predict_loads_batched(&batch);

        // Expected: [[30, 31], [60, 70]]
        assert_ne!(loads[0][0], 1.2);

        let tolerance = 0.1;
        assert!((loads[0][0] - 30.0).abs() < tolerance, "Expected 30.0, got {}", loads[0][0]);
        assert!((loads[0][1] - 31.0).abs() < tolerance, "Expected 31.0, got {}", loads[0][1]);
        assert!((loads[1][0] - 60.0).abs() < tolerance, "Expected 60.0, got {}", loads[1][0]);
        assert!((loads[1][1] - 70.0).abs() < tolerance, "Expected 70.0, got {}", loads[1][1]);
    }
}
