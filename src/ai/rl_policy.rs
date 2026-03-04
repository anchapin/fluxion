//! RL Policy inference module for ONNX-exported reinforcement learning policies.
//!
//! This module provides functionality to load and run ONNX-exported RL policies
//! (trained with Stable Baselines3 or similar frameworks) directly in the Rust engine.
//!
//! The ONNX policy takes observations (building state) as input and outputs
//! actions (HVAC setpoints) for autonomous control without Python.

use ort::session::Session;
use std::path::Path;
use std::sync::Arc;

/// RL Policy action types
#[derive(Clone, Debug)]
pub enum PolicyAction {
    /// Continuous action (HVAC setpoints, etc.)
    Continuous(Vec<f64>),
    /// Discrete action (mode selection, etc.)
    Discrete(Vec<i64>),
}

/// RL Policy inference result
#[derive(Clone, Debug)]
pub struct PolicyInference {
    pub action: PolicyAction,
    pub action_mean: Vec<f64>,
    pub action_std: Option<Vec<f64>>,
}

/// Configuration for RL Policy inference
#[derive(Clone, Debug)]
pub struct PolicyConfig {
    /// Path to ONNX policy model
    pub model_path: String,
    /// Inference backend to use
    pub backend: crate::ai::surrogate::InferenceBackend,
    /// Device ID for GPU inference
    pub device_id: usize,
    /// Action space bounds (min, max)
    pub action_bounds: Option<(Vec<f64>, Vec<f64>)>,
    /// Whether to use deterministic actions
    pub deterministic: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        PolicyConfig {
            model_path: String::new(),
            backend: crate::ai::surrogate::InferenceBackend::CPU,
            device_id: 0,
            action_bounds: None,
            deterministic: true,
        }
    }
}

/// RL Policy manager for ONNX-exported policies.
///
/// Loads and runs ONNX-exported RL policies for autonomous HVAC control.
#[derive(Clone)]
pub struct RLPolicyManager {
    session: Option<Arc<Session>>,
    config: PolicyConfig,
    /// Observation dimension
    pub obs_dim: usize,
    /// Action dimension
    pub action_dim: usize,
}

impl std::fmt::Debug for RLPolicyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RLPolicyManager")
            .field("config", &self.config)
            .field("obs_dim", &self.obs_dim)
            .field("action_dim", &self.action_dim)
            .field("session_loaded", &self.session.is_some())
            .finish()
    }
}

impl RLPolicyManager {
    /// Create a new RL policy manager without loading a model.
    ///
    /// This will return mock actions until a model is loaded.
    pub fn new() -> Result<Self, String> {
        Ok(RLPolicyManager {
            session: None,
            config: PolicyConfig::default(),
            obs_dim: 8,  // Default: [outdoor_temp, zone_temp, solar_rad, hour, day_of_week, month, heating_setpoint, cooling_setpoint]
            action_dim: 2,  // Default: [heating_setpoint, cooling_setpoint]
        })
    }

    /// Load an ONNX policy model.
    ///
    /// # Arguments
    /// * `path` - Path to ONNX model file
    /// * `backend` - Inference backend (CPU, CUDA, etc.)
    /// * `device_id` - Device ID for GPU inference
    ///
    /// # Returns
    /// * `RLPolicyManager` - Manager with loaded model
    pub fn load_policy(path: &str, backend: crate::ai::surrogate::InferenceBackend, device_id: usize) -> Result<Self, String> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(format!("ONNX policy not found: {}", path.display()));
        }

        // Build session based on backend
        let session: Session = match backend {
            crate::ai::surrogate::InferenceBackend::CPU => {
                Session::builder()
                    .map_err(|e| format!("Failed to create ONNX session: {}", e))?
                    .commit_from_file(path)
                    .map_err(|e| format!("Failed to load ONNX model: {}", e))?
            }
            #[cfg(feature = "cuda")]
            crate::ai::surrogate::InferenceBackend::CUDA => {
                use ort::execution_providers::CUDAExecutionProvider;
                let cuda_ep = CUDAExecutionProvider::default().with_device_id(device_id as i32);
                Session::builder()
                    .map_err(|e| format!("Failed to create ONNX session: {}", e))?
                    .with_execution_providers([cuda_ep.build()])
                    .map_err(|e| format!("Failed to configure CUDA provider: {}", e))?
                    .commit_from_file(path)
                    .map_err(|e| format!("Failed to load ONNX model: {}", e))?
            }
            #[cfg(not(feature = "cuda"))]
            crate::ai::surrogate::InferenceBackend::CUDA => {
                eprintln!("CUDA not available, falling back to CPU");
                Session::builder()
                    .map_err(|e| format!("Failed to create ONNX session: {}", e))?
                    .commit_from_file(path)
                    .map_err(|e| format!("Failed to load ONNX model: {}", e))?
            }
            _ => {
                Session::builder()
                    .map_err(|e| format!("Failed to create ONNX session: {}", e))?
                    .commit_from_file(path)
                    .map_err(|e| format!("Failed to load ONNX model: {}", e))?
            }
        };

        let config = PolicyConfig {
            model_path: path.to_string_lossy().to_string(),
            backend,
            device_id,
            action_bounds: None,
            deterministic: true,
        };

        Ok(RLPolicyManager {
            session: Some(Arc::new(session)),
            config,
            obs_dim: 8,  // Default from training
            action_dim: 2,
        })
    }

    /// Predict action from observation.
    ///
    /// # Arguments
    /// * `observation` - Current state observation vector
    ///
    /// # Returns
    /// * `PolicyInference` - Action and related info
    pub fn predict(&self, observation: &[f64]) -> PolicyInference {
        if let Some(ref session) = self.session {
            // Run inference
            let obs_len = observation.len();
            
            // Create input tensor (f32)
            let input_values: Vec<f32> = observation.iter().map(|&x| x as f32).collect();
            
            match ort::value::Tensor::from_array(([1, obs_len], input_values)) {
                Ok(input_tensor) => {
                    match session.run(ort::inputs!["observation" => input_tensor].into()) {
                        Ok(outputs) => {
                            if let Some(output) = outputs.get("action") {
                                match output.try_extract_array::<f32>() {
                                    Ok(action_array) => {
                                        let action: Vec<f64> = action_array.iter().copied().map(|x| x as f64).collect();
                                        return PolicyInference {
                                            action: PolicyAction::Continuous(action.clone()),
                                            action_mean: action,
                                            action_std: None,
                                        };
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to extract action: {}", e);
                                    }
                                }
                            }
                            // Try first output if "action" not found
                            if !outputs.is_empty() {
                                match outputs[0].try_extract_array::<f32>() {
                                    Ok(action_array) => {
                                        let action: Vec<f64> = action_array.iter().copied().map(|x| x as f64).collect();
                                        return PolicyInference {
                                            action: PolicyAction::Continuous(action.clone()),
                                            action_mean: action,
                                            action_std: None,
                                        };
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to extract action from output 0: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("ONNX inference error: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to create input tensor: {}", e);
                }
            }
        }

        // Fallback: return default action
        // Default: heating=20°C, cooling=24°C
        PolicyInference {
            action: PolicyAction::Continuous(vec![20.0, 24.0]),
            action_mean: vec![20.0, 24.0],
            action_std: None,
        }
    }

    /// Predict action with batched observations.
    ///
    /// # Arguments
    /// * `observations` - Batch of observations
    ///
    /// # Returns
    /// * `Vec<PolicyInference>` - Actions for each observation
    pub fn predict_batch(&self, observations: &[Vec<f64>]) -> Vec<PolicyInference> {
        if let Some(ref session) = self.session {
            // Check if model supports batching
            if observations.is_empty() {
                return vec![];
            }

            let batch_size = observations.len();
            let obs_len = observations[0].len();

            // Flatten observations
            let input_values: Vec<f32> = observations
                .iter()
                .flat_map(|obs| obs.iter().map(|&x| x as f32))
                .collect();

            match ort::value::Tensor::from_array((vec![batch_size, obs_len], input_values)) {
                Ok(input_tensor) => {
                    match session.run(ort::inputs!["observation" => input_tensor].into()) {
                        Ok(outputs) => {
                            // Try "action" output first, then first output
                            let output_tensor = outputs.get("action")
                                .or_else(|| outputs.get(0));
                            
                            if let Some(output) = output_tensor {
                                match output.try_extract_array::<f32>() {
                                    Ok(action_array) => {
                                        // Reshape to [batch_size, action_dim]
                                        let action_dim = action_array.len() / batch_size;
                                        let mut results = Vec::with_capacity(batch_size);
                                        
                                        for chunk in action_array.chunks(action_dim) {
                                            let action: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
                                            results.push(PolicyInference {
                                                action: PolicyAction::Continuous(action.clone()),
                                                action_mean: action,
                                                action_std: None,
                                            });
                                        }
                                        return results;
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to extract batch action: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("ONNX batch inference error: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to create batch input tensor: {}", e);
                }
            }
        }

        // Fallback: return default actions
        vec![
            PolicyInference {
                action: PolicyAction::Continuous(vec![20.0, 24.0]),
                action_mean: vec![20.0, 24.0],
                action_std: None,
            };
            observations.len()
        ]
    }

    /// Check if a policy model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// Get the model path.
    pub fn model_path(&self) -> &str {
        &self.config.model_path
    }
}

impl Default for RLPolicyManager {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let manager = RLPolicyManager::new().unwrap();
        assert!(!manager.is_loaded());
    }

    #[test]
    fn test_predict_without_model() {
        let manager = RLPolicyManager::new().unwrap();
        let obs = vec![20.0, 22.0, 100.0, 12.0, 1.0, 6.0, 20.0, 24.0];  // Sample observation
        let inference = manager.predict(&obs);
        
        // Should return default action when no model loaded
        assert_eq!(inference.action_mean.len(), 2);
    }

    #[test]
    fn test_predict_batch() {
        let manager = RLPolicyManager::new().unwrap();
        let obs_batch = vec![
            vec![20.0, 22.0, 100.0, 12.0, 1.0, 6.0, 20.0, 24.0],
            vec![25.0, 24.0, 200.0, 14.0, 2.0, 7.0, 21.0, 25.0],
        ];
        let results = manager.predict_batch(&obs_batch);
        
        assert_eq!(results.len(), 2);
        for result in &results {
            assert_eq!(result.action_mean.len(), 2);
        }
    }
}
