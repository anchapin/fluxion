//! Surrogate manager for fast thermal load predictions.
//!
//! The `SurrogateManager` wraps ONNX Runtime sessions for neural network inference
//! or returns mock predictions during development/testing. It uses `Mutex` for
//! thread-safe interior mutability (required by ORT Session::run which needs &mut).

use std::sync::{Arc, Mutex};

/// Manager that provides fast (mock or neural) thermal-load predictions.
///
/// Wraps an ONNX Runtime session for neural network inference, or returns
/// deterministic mock values when no model is loaded.
///
/// # Thread Safety
/// The session is wrapped in `Arc<Mutex<_>>` for thread-safe sharing across
/// parallel workers, allowing multiple threads to safely borrow the session
/// for inference.
#[derive(Clone, Default)]
pub struct SurrogateManager {
    /// Whether a trained model has been loaded.
    pub model_loaded: bool,
    /// Optional path to an ONNX model registered for use.
    pub model_path: Option<String>,
    /// Optional ONNX Runtime session for inference using the `ort` crate.
    /// Wrapped in `Mutex` to allow thread-safe interior mutability (Session::run requires &mut).
    pub session: Option<Arc<Mutex<ort::session::Session>>>,
}

impl SurrogateManager {
    /// Construct a new `SurrogateManager`.
    pub fn new() -> Result<Self, String> {
        Ok(SurrogateManager {
            model_loaded: false,
            model_path: None,
            session: None,
        })
    }

    /// Register an ONNX model file to be used by the surrogate manager.
    pub fn load_onnx(path: &str) -> Result<Self, String> {
        use ort::session::Session;
        use std::path::Path;

        if !Path::new(path).exists() {
            return Err(format!("ONNX model file not found: {}", path));
        }

        // Initialize and build ONNX session
        let session = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

        Ok(SurrogateManager {
            model_loaded: true,
            model_path: Some(path.to_string()),
            session: Some(Arc::new(Mutex::new(session))),
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
            use ndarray::Array1;
            use ort::value::TensorRef;

            // Convert input temps into an ndarray 1D float32 array
            let input_arr: Array1<f32> =
                Array1::from(current_temps.iter().map(|&x| x as f32).collect::<Vec<_>>());

            // Try to lock the session for inference
            match session_cell.lock() {
                Ok(mut session) => {
                    // Create a tensor reference from the ndarray
                    let tensor_ref = match TensorRef::from_array_view(&input_arr) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Failed to create tensor ref: {}; using mock loads", e);
                            return vec![1.2; current_temps.len()];
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
                                        if v.len() == current_temps.len() {
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
}

#[cfg(test)]
mod tests {
    use crate::ai::surrogate::SurrogateManager;

    #[test]
    fn creation() {
        let m = SurrogateManager::new().unwrap();
        assert!(!m.model_loaded);
    }

    #[test]
    fn predict_mock() {
        let m = SurrogateManager::new().unwrap();
        let temps = [20.0, 21.0, 22.0];
        let loads = m.predict_loads(&temps);
        assert_eq!(loads, vec![1.2, 1.2, 1.2]);
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
}
