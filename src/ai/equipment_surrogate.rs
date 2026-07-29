//! ONNX surrogate integration for complex HVAC equipment.
//!
//! This module provides [`SurrogateEquipmentNode`] which wraps either physics equations
//! or an ONNX surrogate model for fast inference of highly non-linear HVAC components
//! (wet cooling coils, centrifugal chillers with part-load curves, etc.).
//!
//! # Architecture
//!
//! - [`EquipmentNode<M>`] — enum that holds either physics or surrogate variant
//! - [`SurrogateEquipmentNode<M>`] — ONNX-powered surrogate with OOD detection
//! - [`GaussianMixtureModel`] — GMM-based out-of-distribution detector
//!
//! # Feature Gating
//!
//! ONNX support is behind the `ort` feature flag. When disabled, only the
//! physics path is available. The `ort` crate does NOT support WASM targets,
//! so this module is unconditionally unavailable on `wasm32`.

use std::marker::PhantomData;

#[derive(Debug, Clone, thiserror::Error)]
pub enum EquipmentError {
    #[error("ONNX inference failed: {0}")]
    InferenceError(String),
    #[error("Out-of-distribution input detected; falling back to physics")]
    OodDetected,
    #[error("Physics fallback failed: {0}")]
    PhysicsError(String),
    #[error("Invalid medium: expected {expected}, got {actual}")]
    MediumMismatch {
        expected: &'static str,
        actual: &'static str,
    },
}

/// Input vector for equipment evaluation.
/// Contains the operating point used for both surrogate inference and OOD detection.
#[derive(Debug, Clone, PartialEq)]
pub struct EquipmentInput {
    pub inlet_temperature: f64,    // K
    pub inlet_mass_flow_rate: f64, // kg/s
    pub inlet_pressure: f64,       // Pa
    pub timestep: usize,
}

impl EquipmentInput {
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.inlet_temperature,
            self.inlet_mass_flow_rate,
            self.inlet_pressure,
            self.timestep as f64,
        ]
    }
}

/// Output vector from equipment evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct EquipmentOutput {
    pub outlet_temperature: f64,   // K
    pub outlet_enthalpy_rate: f64, // W
    pub efficiency: f64,           // dimensionless
    pub capacity: f64,             // W
}

/// GMM-based out-of-distribution detector.
///
/// Uses a Gaussian Mixture Model trained on nominal operating conditions.
/// When the Mahalanobis distance of an input exceeds the threshold,
/// the equipment falls back to physics equations.
#[derive(Debug, Clone)]
pub struct GaussianMixtureModel {
    /// Means of each Gaussian component [n_components x n_features]
    means: Vec<Vec<f64>>,
    /// Covariance matrices of each component [n_components x n_features x n_features]
    covariances: Vec<Vec<Vec<f64>>>,
    /// Mixture weights [n_components]
    weights: Vec<f64>,
    /// Number of components
    n_components: usize,
    /// Feature dimensionality
    n_features: usize,
    /// Mahalanobis distance threshold for OOD detection
    threshold: f64,
}

impl GaussianMixtureModel {
    /// Create a new GMM with the given number of components and feature dimensionality.
    pub fn new(n_components: usize, n_features: usize, threshold: f64) -> Self {
        Self {
            means: vec![vec![0.0; n_features]; n_components],
            covariances: vec![vec![vec![0.0; n_features]; n_features]; n_components],
            weights: vec![1.0 / n_components as f64; n_components],
            n_components,
            n_features,
            threshold,
        }
    }

    /// Train the GMM on nominal operating data.
    ///
    /// # Arguments
    /// * `data` - Training samples [n_samples x n_features]
    /// * `n_iter` - Maximum EM iterations
    /// * `tol` - Convergence tolerance
    #[allow(clippy::needless_range_loop)] // GMM EM uses explicit 2D indexing; enumerate refactor would obscure algorithm
    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        n_iter: usize,
        _tol: f64,
    ) -> Result<(), EquipmentError> {
        if data.is_empty() {
            return Err(EquipmentError::InferenceError(
                "Training data is empty".into(),
            ));
        }
        let n_samples = data.len();
        self.n_features = data[0].len();

        // Initialize means by k-means clustering (simplified: use first n_components samples)
        for (i, mean) in self.means.iter_mut().enumerate() {
            if i < n_samples {
                mean.copy_from_slice(&data[i.min(n_samples - 1)]);
            }
        }

        // Initialize covariances as identity matrix
        for cov in &mut self.covariances {
            for (j, row) in cov.iter_mut().enumerate() {
                for k in 0..self.n_features {
                    row[k] = if j == k { 1.0 } else { 0.0 };
                }
            }
        }

        // Initialize weights uniformly
        let w = 1.0 / self.n_components as f64;
        self.weights.fill(w);

        // EM iterations (simplified implementation)
        let mut responsibilities = vec![vec![0.0; self.n_components]; n_samples];

        for _iter in 0..n_iter {
            // E-step: compute responsibilities
            let mut total_resp: f64 = 0.0;
            for (i, sample) in data.iter().enumerate() {
                for (k, weight) in self.weights.iter().enumerate() {
                    let mahal = self.mahalanobis_distance(sample, k);
                    responsibilities[i][k] = weight * (-mahal / 2.0).exp();
                    total_resp += responsibilities[i][k];
                }
            }

            // Normalize
            for i in 0..n_samples {
                for k in 0..self.n_components {
                    responsibilities[i][k] /= total_resp.max(f64::EPSILON);
                }
            }

            // M-step: update parameters
            for k in 0..self.n_components {
                let mut Nk: f64 = 0.0;
                let mut new_mean = vec![0.0; self.n_features];
                let mut new_cov = vec![vec![0.0; self.n_features]; self.n_features];

                for (i, sample) in data.iter().enumerate() {
                    Nk += responsibilities[i][k];
                    for j in 0..self.n_features {
                        new_mean[j] += responsibilities[i][k] * sample[j];
                    }
                }

                // Normalize mean
                if Nk > f64::EPSILON {
                    for j in 0..self.n_features {
                        new_mean[j] /= Nk;
                    }
                }

                // Update covariance
                for (i, sample) in data.iter().enumerate() {
                    for j in 0..self.n_features {
                        for l in 0..self.n_features {
                            let diff_j = sample[j] - new_mean[j];
                            let diff_l = sample[l] - new_mean[l];
                            new_cov[j][l] += responsibilities[i][k] * diff_j * diff_l;
                        }
                    }
                }

                if Nk > f64::EPSILON {
                    for j in 0..self.n_features {
                        for l in 0..self.n_features {
                            new_cov[j][l] /= Nk;
                        }
                    }
                }

                self.means[k] = new_mean;
                self.covariances[k] = new_cov;
                self.weights[k] = Nk / n_samples as f64;
            }
        }

        Ok(())
    }

    /// Compute Mahalanobis distance from a sample to a Gaussian component.
    fn mahalanobis_distance(&self, sample: &[f64], component: usize) -> f64 {
        let mean = &self.means[component];
        let cov = &self.covariances[component];

        // Compute (x - mu)^T * Sigma^-1 * (x - mu)
        let mut diff = vec![0.0; self.n_features];
        for i in 0..self.n_features {
            diff[i] = sample[i] - mean[i];
        }

        // Simplified: use diagonal covariance approximation
        let mut mahal_sq = 0.0;
        for i in 0..self.n_features {
            let var = cov[i][i].max(f64::EPSILON);
            mahal_sq += diff[i] * diff[i] / var;
        }

        mahal_sq
    }

    /// Check if an input is out-of-distribution.
    ///
    /// Returns true if the minimum Mahalanobis distance across all components
    /// exceeds the configured threshold.
    pub fn is_ood(&self, input: &EquipmentInput) -> bool {
        let features = input.to_feature_vector();
        self.is_ood_vector(&features)
    }

    /// Check if a feature vector is out-of-distribution.
    pub fn is_ood_vector(&self, features: &[f64]) -> bool {
        if self.n_components == 0 {
            return false;
        }

        let min_mahal = (0..self.n_components)
            .map(|k| self.mahalanobis_distance(features, k))
            .fold(f64::INFINITY, f64::min);

        min_mahal > self.threshold
    }

    /// Get the Mahalanobis distance for a sample.
    pub fn mahalanobis(&self, input: &EquipmentInput) -> f64 {
        let features = input.to_feature_vector();
        (0..self.n_components)
            .map(|k| self.mahalanobis_distance(&features, k))
            .fold(f64::INFINITY, f64::min)
    }
}

/// Physics fallback trait for equipment that can be used when OOD is detected.
pub trait PhysicsEquipment {
    fn evaluate(&self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError>;
}

/// Simple physics model for a chiller.
#[derive(Debug, Clone)]
pub struct ChillerPhysics {
    pub nominal_capacity: f64, // W
    pub nominal_power: f64,    // W
    pub nominal_cop: f64,
    pub water_flow_rate: f64, // kg/s
}

impl ChillerPhysics {
    pub fn new(nominal_capacity: f64, nominal_power: f64) -> Self {
        let nominal_cop = nominal_capacity / nominal_power.max(f64::EPSILON);
        Self {
            nominal_capacity,
            nominal_power,
            nominal_cop,
            water_flow_rate: 0.01, // kg/s
        }
    }

    fn cop_at_conditions(&self, entering_water_temp: f64, _chilled_water_temp: f64) -> f64 {
        // Simplified chiller part-load curve
        // COP decreases at high entering water temperatures
        let temp_factor = 1.0 - 0.003 * (entering_water_temp - 273.15 - 25.0).max(0.0);
        (self.nominal_cop * temp_factor).max(1.0)
    }
}

impl PhysicsEquipment for ChillerPhysics {
    fn evaluate(&self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError> {
        let cp_water = 4184.0; // J/(kg·K) for water
        let delta_t = 6.0; // K - typical chilled water temperature rise

        let capacity = self.nominal_capacity;
        let cop =
            self.cop_at_conditions(input.inlet_temperature, input.inlet_temperature - delta_t);
        let power = capacity / cop.max(f64::EPSILON);

        let outlet_temp = input.inlet_temperature - delta_t;
        let enthalpy_rate = input.inlet_mass_flow_rate * cp_water * delta_t;

        let _ = power; // suppress unused warning

        Ok(EquipmentOutput {
            outlet_temperature: outlet_temp,
            outlet_enthalpy_rate: enthalpy_rate,
            efficiency: cop / self.nominal_cop,
            capacity,
        })
    }
}

/// Simple physics model for a cooling coil.
#[derive(Debug, Clone)]
pub struct CoolingCoilPhysics {
    pub effectiveness: f64,
    pub nominal_capacity: f64, // W
    pub air_flow_rate: f64,    // kg/s
    pub water_flow_rate: f64,  // kg/s
}

impl CoolingCoilPhysics {
    pub fn new(effectiveness: f64, nominal_capacity: f64) -> Self {
        Self {
            effectiveness,
            nominal_capacity,
            air_flow_rate: 1.0,
            water_flow_rate: 0.1,
        }
    }
}

impl PhysicsEquipment for CoolingCoilPhysics {
    fn evaluate(&self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError> {
        let cp_air = 1006.0; // J/(kg·K) for air
        let h_water = 4184.0;

        // Maximum possible heat transfer
        let q_max =
            self.air_flow_rate * cp_air * (input.inlet_temperature - 273.15 - 10.0).max(0.0);

        // Actual heat transfer based on effectiveness
        let q_actual = q_max * self.effectiveness;

        // Outlet air temperature
        let delta_t_air = q_actual / (self.air_flow_rate * cp_air);
        let outlet_temp = input.inlet_temperature - delta_t_air;

        // Water side heat gain
        let enthalpy_rate = self.water_flow_rate * h_water * delta_t_air.abs();

        Ok(EquipmentOutput {
            outlet_temperature: outlet_temp,
            outlet_enthalpy_rate: enthalpy_rate,
            efficiency: self.effectiveness,
            capacity: q_actual,
        })
    }
}

/// Equipment node that wraps either physics equations or an ONNX surrogate.
///
/// This enum allows transparent switching between physics and surrogate modes
/// based on OOD detection.
pub enum EquipmentNode<M> {
    Physics(Box<dyn PhysicsEquipment>),
    Surrogate(SurrogateEquipmentNode<M>),
}

impl<M> EquipmentNode<M> {
    /// Execute one evaluation step.
    ///
    /// For surrogate mode, this first checks OOD detection. If OOD is detected,
    /// falls back to physics. Otherwise, runs ONNX inference.
    pub fn evaluate(&mut self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError> {
        match self {
            EquipmentNode::Physics(physics) => physics.evaluate(input),
            EquipmentNode::Surrogate(surrogate) => surrogate.evaluate(input),
        }
    }
}

/// Surrogate equipment node with ONNX inference and OOD detection.
///
/// This struct wraps an ONNX model for fast inference of HVAC equipment,
/// combined with a GMM-based OOD detector that triggers physics fallback
/// when inputs are outside the training distribution.
#[cfg(feature = "ort")]
pub struct SurrogateEquipmentNode<M> {
    #[allow(dead_code)]
    inference_session: ort::session::Session,
    ood_detector: GaussianMixtureModel,
    physics_fallback: Box<dyn PhysicsEquipment>,
    _phantom: PhantomData<M>,
}

/// Stub of [`SurrogateEquipmentNode`] when `ort` feature is disabled.
#[cfg(not(feature = "ort"))]
pub struct SurrogateEquipmentNode<M> {
    ood_detector: GaussianMixtureModel,
    physics_fallback: Box<dyn PhysicsEquipment>,
    _phantom: PhantomData<M>,
}

#[cfg(feature = "ort")]
impl<M> SurrogateEquipmentNode<M> {
    /// Create a new SurrogateEquipmentNode.
    ///
    /// # Arguments
    /// * `model_path` - Path to ONNX model file
    /// * `ood_detector` - Pre-trained GMM for OOD detection
    /// * `physics_fallback` - Physics model used when OOD is detected
    pub fn new(
        model_path: &str,
        ood_detector: GaussianMixtureModel,
        physics_fallback: Box<dyn PhysicsEquipment>,
    ) -> Result<Self, EquipmentError> {
        use ort::session::Session;
        let session = Session::builder()
            .map_err(|e| {
                EquipmentError::InferenceError(format!("Failed to create session builder: {}", e))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                EquipmentError::InferenceError(format!("Failed to load ONNX model: {}", e))
            })?;

        Ok(Self {
            inference_session: session,
            ood_detector,
            physics_fallback,
            _phantom: PhantomData,
        })
    }

    /// Evaluate using ONNX inference, falling back to physics if OOD.
    pub fn evaluate(&mut self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError> {
        if self.ood_detector.is_ood(input) {
            log::debug!("OOD detected for input: {:?}", input);
            return self.physics_fallback.evaluate(input);
        }

        self.infer_onnx(input)
    }

    fn infer_onnx(&mut self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError> {
        use ort::value::Value;

        let features: Vec<f32> = input
            .to_feature_vector()
            .iter()
            .map(|&x| x as f32)
            .collect();
        let n_features = features.len();

        // Create input tensor [1 x n_features]
        let input_tensor =
            Value::from_array(([1_i64, n_features as i64], features)).map_err(|e| {
                EquipmentError::InferenceError(format!("Failed to create input tensor: {}", e))
            })?;

        // Run inference
        let outputs = self
            .inference_session
            .run(ort::inputs![input_tensor])
            .map_err(|e| EquipmentError::InferenceError(format!("ONNX inference failed: {}", e)))?;

        // Extract output tensor - outputs[0] is indexed by position, not by name
        let array_view = outputs[0].try_extract_array::<f32>().map_err(|e| {
            EquipmentError::InferenceError(format!("Failed to extract tensor: {}", e))
        })?;

        // Parse output [outlet_temp, enthalpy_rate, efficiency, capacity]
        if array_view.len() < 4 {
            return Err(EquipmentError::InferenceError(format!(
                "Expected 4 outputs, got {}",
                array_view.len()
            )));
        }

        Ok(EquipmentOutput {
            outlet_temperature: array_view[0] as f64,
            outlet_enthalpy_rate: array_view[1] as f64,
            efficiency: array_view[2] as f64,
            capacity: array_view[3] as f64,
        })
    }
}

/// Stub implementation when `ort` feature is disabled.
#[cfg(not(feature = "ort"))]
impl<M> SurrogateEquipmentNode<M> {
    /// Create a new SurrogateEquipmentNode (stub when ort disabled).
    pub fn new(
        _model_path: &str,
        ood_detector: GaussianMixtureModel,
        physics_fallback: Box<dyn PhysicsEquipment>,
    ) -> Result<Self, EquipmentError> {
        let _ = ood_detector;
        Ok(Self {
            ood_detector: GaussianMixtureModel::new(1, 4, f64::INFINITY),
            physics_fallback,
            _phantom: PhantomData,
        })
    }

    /// Evaluate using physics fallback (ONNX not available without `ort` feature).
    pub fn evaluate(&mut self, input: &EquipmentInput) -> Result<EquipmentOutput, EquipmentError> {
        self.physics_fallback.evaluate(input)
    }
}

impl<M> SurrogateEquipmentNode<M> {
    /// Get the OOD detector.
    pub fn ood_detector(&self) -> &GaussianMixtureModel {
        &self.ood_detector
    }

    /// Check if input would trigger OOD fallback.
    pub fn is_ood(&self, input: &EquipmentInput) -> bool {
        self.ood_detector.is_ood(input)
    }
}

/// Create a surrogate equipment node from an ONNX model for a chiller.
///
/// This is a convenience constructor that sets up a chiller physics fallback
/// and initializes the GMM with default parameters.
#[cfg(feature = "ort")]
pub fn chiller_surrogate_node<M>(
    model_path: &str,
    nominal_capacity: f64,
    nominal_power: f64,
) -> Result<SurrogateEquipmentNode<M>, EquipmentError> {
    let physics = Box::new(ChillerPhysics::new(nominal_capacity, nominal_power));

    // Train GMM on nominal conditions (T_inlet around 280-300K, flow 0.01-0.1 kg/s)
    let mut gmm = GaussianMixtureModel::new(2, 4, 15.0);

    // Generate nominal training data
    let mut training_data = Vec::new();
    for t_inlet in [280.0, 285.0, 290.0, 295.0, 300.0].iter() {
        for mdot in [0.01, 0.05, 0.1].iter() {
            for p_inlet in [101325.0, 200000.0, 300000.0].iter() {
                training_data.push(vec![*t_inlet, *mdot, *p_inlet, 0.0]);
            }
        }
    }

    gmm.fit(&training_data, 50, 1e-4)
        .map_err(|e| EquipmentError::InferenceError(format!("GMM training failed: {}", e)))?;

    SurrogateEquipmentNode::new(model_path, gmm, physics)
}

/// Create a surrogate equipment node from an ONNX model for a cooling coil.
///
/// This is a convenience constructor that sets up a cooling coil physics fallback
/// and initializes the GMM with default parameters.
#[cfg(feature = "ort")]
pub fn cooling_coil_surrogate_node<M>(
    model_path: &str,
    effectiveness: f64,
    nominal_capacity: f64,
) -> Result<SurrogateEquipmentNode<M>, EquipmentError> {
    let physics = Box::new(CoolingCoilPhysics::new(effectiveness, nominal_capacity));

    let mut gmm = GaussianMixtureModel::new(2, 4, 15.0);

    let mut training_data = Vec::new();
    for t_inlet in [280.0, 290.0, 300.0, 310.0].iter() {
        for mdot in [0.5, 1.0, 1.5].iter() {
            for p_inlet in [101325.0].iter() {
                training_data.push(vec![*t_inlet, *mdot, *p_inlet, 0.0]);
            }
        }
    }

    gmm.fit(&training_data, 50, 1e-4)
        .map_err(|e| EquipmentError::InferenceError(format!("GMM training failed: {}", e)))?;

    SurrogateEquipmentNode::new(model_path, gmm, physics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmm_is_ood() {
        let mut gmm = GaussianMixtureModel::new(2, 4, 15.0);

        // Training data: nominal conditions around 290K
        let training_data = vec![
            vec![290.0, 0.05, 101325.0, 0.0],
            vec![291.0, 0.05, 101325.0, 0.0],
            vec![289.0, 0.05, 101325.0, 0.0],
            vec![290.0, 0.06, 101325.0, 0.0],
            vec![290.0, 0.04, 101325.0, 0.0],
        ];

        gmm.fit(&training_data, 10, 1e-4).unwrap();

        // Test nominal input - should NOT be OOD
        let nominal = EquipmentInput {
            inlet_temperature: 290.0,
            inlet_mass_flow_rate: 0.05,
            inlet_pressure: 101325.0,
            timestep: 0,
        };
        assert!(!gmm.is_ood(&nominal), "Nominal input should not be OOD");

        // Test extreme input - SHOULD be OOD
        let extreme = EquipmentInput {
            inlet_temperature: 400.0, // Very high temp
            inlet_mass_flow_rate: 0.05,
            inlet_pressure: 101325.0,
            timestep: 0,
        };
        assert!(gmm.is_ood(&extreme), "Extreme input should be OOD");
    }

    #[test]
    fn test_chiller_physics() {
        let chiller = ChillerPhysics::new(100_000.0, 25_000.0); // 100kW cap, 25kW power

        let input = EquipmentInput {
            inlet_temperature: 295.0,
            inlet_mass_flow_rate: 0.1,
            inlet_pressure: 200000.0,
            timestep: 0,
        };

        let output = chiller.evaluate(&input).unwrap();

        assert!(output.outlet_temperature < input.inlet_temperature);
        assert!(output.capacity > 0.0);
        assert!(output.efficiency > 0.0 && output.efficiency <= 1.0);
    }

    #[test]
    fn test_cooling_coil_physics() {
        let coil = CoolingCoilPhysics::new(0.8, 50_000.0);

        let input = EquipmentInput {
            inlet_temperature: 300.0,
            inlet_mass_flow_rate: 1.0,
            inlet_pressure: 101325.0,
            timestep: 0,
        };

        let output = coil.evaluate(&input).unwrap();

        assert!(output.outlet_temperature < input.inlet_temperature);
        assert!(output.capacity > 0.0);
        assert!((output.efficiency - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_equipment_node_physics() {
        let chiller = ChillerPhysics::new(100_000.0, 25_000.0);
        let mut node: EquipmentNode<()> = EquipmentNode::Physics(Box::new(chiller));

        let input = EquipmentInput {
            inlet_temperature: 295.0,
            inlet_mass_flow_rate: 0.1,
            inlet_pressure: 200000.0,
            timestep: 0,
        };

        let output = node.evaluate(&input).unwrap();
        assert!(output.capacity > 0.0);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn test_surrogate_node_without_model_file() {
        // This test verifies the stub path when ort is enabled but no model exists
        let physics = Box::new(ChillerPhysics::new(100_000.0, 25_000.0));
        let gmm = GaussianMixtureModel::new(1, 4, f64::INFINITY);

        // Using a non-existent model path - should fail gracefully
        let result = SurrogateEquipmentNode::<()>::new("/nonexistent/model.onnx", gmm, physics);
        assert!(result.is_err());
    }

    #[test]
    fn test_ood_mahalanobis() {
        let mut gmm = GaussianMixtureModel::new(1, 2, 10.0);

        let training_data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        gmm.fit(&training_data, 20, 1e-6).unwrap();

        // Point at origin should have low Mahalanobis distance
        let input = EquipmentInput {
            inlet_temperature: 273.15,
            inlet_mass_flow_rate: 0.0,
            inlet_pressure: 0.0,
            timestep: 0,
        };

        // Just verify it doesn't panic
        let mahal = gmm.mahalanobis(&input);
        assert!(mahal.is_finite());
    }
}
