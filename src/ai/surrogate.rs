//! Surrogate manager for fast thermal load predictions.
//!
//! Placeholder `SurrogateManager` used by the physics engine for tests and
//! early development. Returns mock predictions when no model is loaded.

/// Manager that provides fast (mock or neural) thermal-load predictions.
///
/// In production this wraps an ONNX runtime session; in tests it returns
/// deterministic mock values when no model is loaded.
/// This struct is cloneable for parallel evaluation across population vectors.
#[derive(Clone, Default)]
/// Surrogate manager for thermal load predictions.
pub struct SurrogateManager {
    /// Whether a trained model has been loaded.
    pub model_loaded: bool,
}

impl SurrogateManager {
    /// Construct a new `SurrogateManager`.
    pub fn new() -> Result<Self, String> {
        Ok(SurrogateManager {
            model_loaded: false,
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
    ///
    /// # Panics
    /// Panics if a model is marked as loaded but ONNX inference is not implemented.
    pub fn predict_loads(&self, current_temps: &[f64]) -> Vec<f64> {
        if !self.model_loaded {
            return vec![1.2; current_temps.len()];
        }

        panic!("ONNX inference path is not implemented. SurrogateManager.model_loaded is true, but inference cannot be performed. Please implement ONNX inference before using this feature.");
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
}
