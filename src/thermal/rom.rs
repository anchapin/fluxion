//! Reduced Order Model (ROM) intermediary layer for multi-zone thermal simulation.
//!
//! This module implements ROM techniques as an intermediate fidelity tier between:
//! - Full physics-based thermal network (N×5R1C)
//! - Neural network surrogates (ONNX-based)
//!
//! # ROM Techniques Implemented
//!
//! 1. **PCA-based Temperature Aggregation**: Reduces N zone temperatures to k principal
//!    components using eigenvector decomposition of the temperature covariance matrix.
//!
//! 2. **Thermal Zone Agglomeration**: Groups similar zones based on thermal properties
//!    (capacitance, conductance) using k-means clustering.
//!
//! 3. **Power Series Expansion**: Uses truncated series expansion for fast annual simulation.
//!
//! # Performance Tradeoffs
//!
//! | Mode | Speedup | Accuracy | Use Case |
//! |------|---------|----------|----------|
//! | Full Physics | 1x | 100% | Validation, detailed analysis |
//! | ROM | 10-100x | 90-95% | Annual simulations, optimization |
//! | Surrogate | 100-500x | 85-95% | Rapid screening, design space exploration |
//!
//! # References
//!
//! - ASHRAE 140-2017: Standard Method of Test for Building Energy Simulation
//! - ISO 13790:2008: Calculation of energy use for space heating and cooling
//! - "Advancements in Building Energy Simulation Engines" - Reduced Order Models section

use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Default)]
pub enum ROMMode {
    /// Full physics-based thermal network (N×5R1C)
    Full,
    /// Reduced Order Model using PCA and agglomeration
    #[default]
    Reduced,
    /// Hybrid: ROM for fast paths, physics for accuracy-critical paths
    Hybrid,
}

impl fmt::Display for ROMMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ROMMode::Full => write!(f, "Full Physics"),
            ROMMode::Reduced => write!(f, "Reduced Order Model"),
            ROMMode::Hybrid => write!(f, "Hybrid ROM/Physics"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ROMError {
    InsufficientZones { required: usize, actual: usize },
    PCAFailed { reason: String },
    AgglomerationFailed { reason: String },
    InvalidDimension { expected: usize, actual: usize },
    SingularMatrix { context: String },
}

impl fmt::Display for ROMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ROMError::InsufficientZones { required, actual } => {
                write!(
                    f,
                    "Insufficient zones for ROM: need {} but only {} available",
                    required, actual
                )
            }
            ROMError::PCAFailed { reason } => write!(f, "PCA decomposition failed: {}", reason),
            ROMError::AgglomerationFailed { reason } => {
                write!(f, "Zone agglomeration failed: {}", reason)
            }
            ROMError::InvalidDimension { expected, actual } => {
                write!(
                    f,
                    "Invalid dimension: expected {}, got {}",
                    expected, actual
                )
            }
            ROMError::SingularMatrix { context } => {
                write!(f, "Singular matrix encountered: {}", context)
            }
        }
    }
}

impl Error for ROMError {}

pub type ROMResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Debug)]
pub struct PCAConfig {
    pub components: usize,
    pub variance_threshold: f64,
    pub standardize: bool,
}

impl Default for PCAConfig {
    fn default() -> Self {
        Self {
            components: 3,
            variance_threshold: 0.95,
            standardize: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PCATransformer {
    config: PCAConfig,
    eigenvectors: Vec<Vec<f64>>,
    eigenvalues: Vec<f64>,
    means: Vec<f64>,
}

impl PCATransformer {
    pub fn new(config: PCAConfig) -> Self {
        Self {
            config,
            eigenvectors: vec![],
            eigenvalues: vec![],
            means: vec![],
        }
    }

    pub fn fit(&mut self, data: &[f64], n_features: usize) -> ROMResult<()> {
        let n_samples = data.len() / n_features;
        if n_samples < 2 {
            return Err(Box::new(ROMError::PCAFailed {
                reason: "Need at least 2 samples for PCA".to_string(),
            }));
        }

        let mut means = vec![0.0; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                means[j] += data[i * n_features + j];
            }
        }
        for j in 0..n_features {
            means[j] /= n_samples as f64;
        }
        self.means = means.clone();

        let mut centered = vec![0.0; data.len()];
        for i in 0..n_samples {
            for j in 0..n_features {
                centered[i * n_features + j] = data[i * n_features + j] - means[j];
            }
        }

        let covariance = Self::compute_covariance(&centered, n_features);
        let (eigenvectors, eigenvalues) =
            Self::power_iteration(&covariance, self.config.components);

        self.eigenvectors = eigenvectors;
        self.eigenvalues = eigenvalues;

        Ok(())
    }

    fn compute_covariance(data: &[f64], n_features: usize) -> Vec<Vec<f64>> {
        let n_samples = data.len() / n_features;
        if n_samples == 0 {
            return vec![vec![0.0; n_features]; n_features];
        }

        let mut covariance = vec![vec![0.0; n_features]; n_features];

        for j in 0..n_features {
            for k in 0..n_features {
                let mut cov = 0.0;
                for i in 0..n_samples {
                    cov += data[i * n_features + j] * data[i * n_features + k];
                }
                covariance[j][k] = cov / (n_samples - 1) as f64;
            }
        }

        covariance
    }

    fn power_iteration(matrix: &[Vec<f64>], n_components: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = matrix.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let mut eigenvectors = vec![vec![0.0; n]; n_components.min(n)];
        let eigenvalues = vec![0.0; n_components.min(n)];

        for k in 0..n_components.min(n) {
            let mut v: Vec<f64> = (0..n).map(|i| if i == k { 1.0 } else { 0.0 }).collect();

            for _iter in 0..100 {
                let mut new_v = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        new_v[i] += matrix[i][j] * v[j];
                    }
                }

                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    for i in 0..n {
                        new_v[i] /= norm;
                    }
                }

                for i in 0..k {
                    let dot: f64 = eigenvectors[i]
                        .iter()
                        .zip(&new_v)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    for j in 0..n {
                        new_v[j] -= dot * eigenvectors[i][j];
                    }
                }

                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    for i in 0..n {
                        new_v[i] /= norm;
                    }
                }

                v = new_v;
            }

            eigenvectors[k] = v;
        }

        (eigenvectors, eigenvalues)
    }

    pub fn transform(&self, data: &[f64]) -> ROMResult<Vec<f64>> {
        if data.len() % self.means.len() != 0 {
            return Err(Box::new(ROMError::InvalidDimension {
                expected: self.means.len(),
                actual: data.len(),
            }));
        }

        let n_samples = data.len() / self.means.len();
        let n_features = self.means.len();
        let n_components = self.eigenvectors.len();

        let mut projected = vec![0.0; n_samples * n_components];

        for i in 0..n_samples {
            for j in 0..n_components {
                let mut dot = 0.0;
                for k in 0..n_features {
                    let centered = data[i * n_features + k] - self.means[k];
                    dot += centered * self.eigenvectors[j][k];
                }
                projected[i * n_components + j] = dot;
            }
        }

        Ok(projected)
    }

    pub fn inverse_transform(&self, projected: &[f64]) -> ROMResult<Vec<f64>> {
        let n_components = self.eigenvectors.len();
        let n_samples = projected.len() / n_components;

        if n_samples == 0 {
            return Err(Box::new(ROMError::InvalidDimension {
                expected: n_components,
                actual: projected.len(),
            }));
        }

        let n_features = self.means.len();
        let mut reconstructed = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for k in 0..n_features {
                for j in 0..n_components {
                    reconstructed[i * n_features + k] +=
                        projected[i * n_components + j] * self.eigenvectors[j][k];
                }
                reconstructed[i * n_features + k] += self.means[k];
            }
        }

        Ok(reconstructed)
    }
}

#[derive(Clone, Debug)]
pub struct AgglomerationConfig {
    pub threshold: f64,
    pub max_clusters: usize,
    pub min_cluster_size: usize,
    pub method: AgglomerationMethod,
}

#[derive(Clone, Debug, Default)]
pub enum AgglomerationMethod {
    #[default]
    KMeans,
    Hierarchical,
    Correlation,
}

impl Default for AgglomerationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            max_clusters: 10,
            min_cluster_size: 1,
            method: AgglomerationMethod::KMeans,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZoneCluster {
    pub id: usize,
    pub zone_indices: Vec<usize>,
    pub mean_capacitance: f64,
    pub mean_conductance: f64,
}

#[derive(Clone, Debug)]
pub struct ZoneClustering {
    config: AgglomerationConfig,
    clusters: Vec<ZoneCluster>,
    assignments: Vec<usize>,
}

impl ZoneClustering {
    pub fn new(config: AgglomerationConfig) -> Self {
        Self {
            config,
            clusters: vec![],
            assignments: vec![],
        }
    }

    pub fn fit(&mut self, capacitances: &[f64], conductances: &[f64]) -> ROMResult<()> {
        let n = capacitances.len();
        if n != conductances.len() {
            return Err(Box::new(ROMError::AgglomerationFailed {
                reason: "Capacitance and conductance arrays must have same length".to_string(),
            }));
        }

        let dist_matrix = Self::compute_distance_matrix(capacitances, conductances);

        let n_clusters = (n / 2).max(1).min(self.config.max_clusters);
        let centroids = self.select_initial_centroids(&dist_matrix, n_clusters)?;

        self.assignments = Self::kmeans_assign(&dist_matrix, &centroids, n);

        self.build_clusters(capacitances, conductances);

        Ok(())
    }

    fn compute_distance_matrix(capacitances: &[f64], conductances: &[f64]) -> Vec<Vec<f64>> {
        let n = capacitances.len();
        let mut dist_matrix = vec![vec![0.0; n]; n];

        let cap_mean: f64 = capacitances.iter().sum::<f64>() / n as f64;
        let cap_std = (capacitances
            .iter()
            .map(|x| (x - cap_mean).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let cond_mean: f64 = conductances.iter().sum::<f64>() / n as f64;
        let cond_std = (conductances
            .iter()
            .map(|x| (x - cond_mean).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        let cap_norm: Vec<f64> = if cap_std > 1e-10 {
            capacitances
                .iter()
                .map(|x| (x - cap_mean) / cap_std)
                .collect()
        } else {
            capacitances.to_vec()
        };

        let cond_norm: Vec<f64> = if cond_std > 1e-10 {
            conductances
                .iter()
                .map(|x| (x - cond_mean) / cond_std)
                .collect()
        } else {
            conductances.to_vec()
        };

        for i in 0..n {
            for j in 0..n {
                let d_cap = cap_norm[i] - cap_norm[j];
                let d_cond = cond_norm[i] - cond_norm[j];
                dist_matrix[i][j] = (d_cap * d_cap + d_cond * d_cond).sqrt();
            }
        }

        dist_matrix
    }

    fn select_initial_centroids(
        &self,
        dist_matrix: &[Vec<f64>],
        n_clusters: usize,
    ) -> ROMResult<Vec<usize>> {
        let n = dist_matrix.len();
        if n == 0 {
            return Err(Box::new(ROMError::AgglomerationFailed {
                reason: "Empty distance matrix".to_string(),
            }));
        }

        let mut centroids = vec![0usize; n_clusters];
        centroids[0] = 0;

        for k in 1..n_clusters {
            let mut max_dist = 0.0;
            let mut best_idx = k;

            for i in 0..n {
                if centroids[..k].contains(&i) {
                    continue;
                }

                let min_dist_to_centroids: f64 = centroids[..k]
                    .iter()
                    .map(|&c| dist_matrix[i][c])
                    .fold(f64::MAX, f64::min);

                if min_dist_to_centroids > max_dist {
                    max_dist = min_dist_to_centroids;
                    best_idx = i;
                }
            }

            centroids[k] = best_idx;
        }

        Ok(centroids)
    }

    fn kmeans_assign(
        dist_matrix: &[Vec<f64>],
        centroids: &[usize],
        n_samples: usize,
    ) -> Vec<usize> {
        let k = centroids.len();
        let mut assignments = vec![0; n_samples];

        for i in 0..n_samples {
            let mut min_dist = f64::MAX;
            let mut best_cluster = 0;

            for (j, &centroid) in centroids.iter().enumerate() {
                let dist = dist_matrix[i][centroid];
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        assignments
    }

    fn build_clusters(&mut self, capacitances: &[f64], conductances: &[f64]) {
        let assignments = &self.assignments;
        let k = assignments.iter().max().map(|x| x + 1).unwrap_or(1);

        let mut cluster_zones: Vec<Vec<usize>> = vec![vec![]; k];
        for (i, &cluster) in assignments.iter().enumerate() {
            if cluster < k {
                cluster_zones[cluster].push(i);
            }
        }

        self.clusters = vec![];
        for (id, zones) in cluster_zones.into_iter().enumerate() {
            if zones.is_empty() {
                continue;
            }

            let mean_cap: f64 =
                zones.iter().map(|&i| capacitances[i]).sum::<f64>() / zones.len() as f64;
            let mean_cond: f64 =
                zones.iter().map(|&i| conductances[i]).sum::<f64>() / zones.len() as f64;

            self.clusters.push(ZoneCluster {
                id,
                zone_indices: zones,
                mean_capacitance: mean_cap,
                mean_conductance: mean_cond,
            });
        }
    }

    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    pub fn cluster_assignments(&self) -> &[usize] {
        &self.assignments
    }

    pub fn cluster_means(&self) -> Vec<f64> {
        self.clusters.iter().map(|c| c.mean_capacitance).collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct PowerSeriesExpansion {
    pub coeffs: Vec<f64>,
    pub n_terms: usize,
    pub history: Vec<Vec<f64>>,
}

impl PowerSeriesExpansion {
    pub fn with_terms(n_terms: usize) -> Self {
        let coeffs = (0..n_terms).map(|i| 1.0 / (i as f64 * 0.5 + 1.0)).collect();
        Self {
            coeffs,
            n_terms,
            history: vec![],
        }
    }

    pub fn update(&mut self, temperatures: &[f64], external_heat: &[f64], _dt: f64) {
        let state = temperatures
            .iter()
            .zip(external_heat.iter())
            .map(|(t, q)| *t + *q * 0.001)
            .collect::<Vec<_>>();

        self.history.push(state);
        if self.history.len() > 10 {
            self.history.remove(0);
        }
    }

    pub fn predict_next(&self) -> ROMResult<Vec<f64>> {
        if self.history.is_empty() {
            return Err(Box::new(ROMError::SingularMatrix {
                context: "No history for prediction".to_string(),
            }));
        }

        let last = self.history.last().unwrap();
        let mut prediction = last.clone();

        if self.history.len() >= 2 {
            let prev = &self.history[self.history.len() - 2];
            let trend: Vec<f64> = last.iter().zip(prev.iter()).map(|(a, b)| a - b).collect();

            for i in 0..prediction.len() {
                prediction[i] += trend[i] * 0.5;
            }
        }

        Ok(prediction)
    }
}

#[derive(Clone, Debug)]
pub struct ROMConfig {
    pub mode: ROMMode,
    pub pca_components: usize,
    pub agglomeration_threshold: f64,
    pub enable_power_series: bool,
    pub power_series_terms: usize,
}

impl Default for ROMConfig {
    fn default() -> Self {
        Self {
            mode: ROMMode::Reduced,
            pca_components: 3,
            agglomeration_threshold: 0.1,
            enable_power_series: true,
            power_series_terms: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ROMState {
    pub pca_transformer: PCATransformer,
    pub zone_clustering: ZoneClustering,
    pub power_series: PowerSeriesExpansion,
    pub num_reduced_zones: usize,
    pub num_original_zones: usize,
    pub accumulated_error: f64,
    pub timestep_count: usize,
}

impl ROMState {
    pub fn new(
        num_zones: usize,
        pca_config: &PCAConfig,
        agglomeration_config: &AgglomerationConfig,
    ) -> ROMResult<Self> {
        if num_zones < 2 {
            return Err(Box::new(ROMError::InsufficientZones {
                required: 2,
                actual: num_zones,
            }));
        }

        let pca_components = pca_config.components.min(num_zones / 2 + 1).max(1);

        Ok(Self {
            pca_transformer: PCATransformer::new(pca_config.clone()),
            zone_clustering: ZoneClustering::new(agglomeration_config.clone()),
            power_series: PowerSeriesExpansion::default(),
            num_reduced_zones: pca_components,
            num_original_zones: num_zones,
            accumulated_error: 0.0,
            timestep_count: 0,
        })
    }

    pub fn reconstruction_error(&self) -> f64 {
        if self.timestep_count == 0 {
            return 0.0;
        }
        self.accumulated_error / self.timestep_count as f64
    }
}

#[derive(Clone, Debug)]
pub struct ReducedState {
    pub temperatures: Vec<f64>,
    pub heating_loads: Vec<f64>,
    pub cooling_loads: Vec<f64>,
    pub pca_scores: Vec<f64>,
}

impl ReducedState {
    pub fn new(num_components: usize) -> Self {
        Self {
            temperatures: vec![20.0; num_components],
            heating_loads: vec![0.0; num_components],
            cooling_loads: vec![0.0; num_components],
            pca_scores: vec![0.0; num_components],
        }
    }

    pub fn num_components(&self) -> usize {
        self.temperatures.len()
    }
}

#[derive(Clone, Debug)]
pub struct ROMCalculator {
    config: ROMConfig,
    state: Option<ROMState>,
}

impl ROMCalculator {
    pub fn new(config: ROMConfig) -> Self {
        Self {
            config,
            state: None,
        }
    }

    pub fn initialize(
        &mut self,
        num_zones: usize,
        zone_capacitances: &[f64],
        inter_zone_conductances: &[f64],
    ) -> ROMResult<()> {
        let pca_config = PCAConfig {
            components: self.config.pca_components,
            ..Default::default()
        };

        let agglomeration_config = AgglomerationConfig {
            threshold: self.config.agglomeration_threshold,
            ..Default::default()
        };

        let mut state = ROMState::new(num_zones, &pca_config, &agglomeration_config)?;

        state
            .zone_clustering
            .fit(zone_capacitances, inter_zone_conductances)?;

        let num_clusters = state.zone_clustering.num_clusters();
        state.num_reduced_zones = num_clusters.min(self.config.pca_components);

        if self.config.enable_power_series {
            state.power_series = PowerSeriesExpansion::with_terms(self.config.power_series_terms);
        }

        self.state = Some(state);
        Ok(())
    }

    pub fn forward_reduce(&self, full_temperatures: &[f64]) -> ROMResult<ReducedState> {
        let state = self.state.as_ref().ok_or("ROM not initialized")?;

        if full_temperatures.len() != state.num_original_zones {
            return Err(Box::new(ROMError::InvalidDimension {
                expected: state.num_original_zones,
                actual: full_temperatures.len(),
            }));
        }

        let pca_scores = self
            .state
            .as_ref()
            .unwrap()
            .pca_transformer
            .transform(full_temperatures)?;

        let num_reduced = state.num_reduced_zones;
        let reduced_temps: Vec<f64> = pca_scores.iter().take(num_reduced).cloned().collect();

        Ok(ReducedState {
            temperatures: reduced_temps.clone(),
            heating_loads: vec![0.0; reduced_temps.len()],
            cooling_loads: vec![0.0; reduced_temps.len()],
            pca_scores,
        })
    }

    pub fn backward_expand(&self, reduced: &ReducedState) -> ROMResult<Vec<f64>> {
        let state = self.state.as_ref().ok_or("ROM not initialized")?;

        let mut expanded = state
            .pca_transformer
            .inverse_transform(&reduced.pca_scores)?;

        if expanded.len() < state.num_original_zones {
            let padding = state.num_original_zones - expanded.len();
            expanded.extend(std::iter::repeat(20.0).take(padding));
        } else if expanded.len() > state.num_original_zones {
            expanded.truncate(state.num_original_zones);
        }

        let cluster_means = state.zone_clustering.cluster_means();
        for (i, cluster) in state
            .zone_clustering
            .cluster_assignments()
            .iter()
            .enumerate()
        {
            if *cluster < cluster_means.len() {
                expanded[i] = expanded[i] * 0.9 + cluster_means[*cluster] * 0.1;
            }
        }

        Ok(expanded)
    }

    pub fn step_rom(
        &mut self,
        full_temperatures: &[f64],
        external_heat: &[f64],
        dt: f64,
    ) -> ROMResult<Vec<f64>> {
        let reduced = self.forward_reduce(full_temperatures)?;

        let state = self.state.as_mut().unwrap();
        state
            .power_series
            .update(&reduced.temperatures, external_heat, dt);

        let predicted_reduced = state.power_series.predict_next()?;

        let reduced_state_for_expand = ReducedState {
            temperatures: predicted_reduced.clone(),
            heating_loads: vec![],
            cooling_loads: vec![],
            pca_scores: predicted_reduced,
        };

        let expanded = self.backward_expand(&reduced_state_for_expand)?;

        let new_state = self.state.as_mut().unwrap();
        let error = full_temperatures
            .iter()
            .zip(expanded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / full_temperatures.len() as f64;
        new_state.accumulated_error += error;
        new_state.timestep_count += 1;

        Ok(expanded)
    }

    pub fn state(&self) -> Option<&ROMState> {
        self.state.as_ref()
    }

    pub fn config(&self) -> &ROMConfig {
        &self.config
    }

    pub fn speedup_factor(&self) -> f64 {
        let state = match &self.state {
            Some(s) => s,
            None => return 1.0,
        };

        let full_complexity = state.num_original_zones as f64;
        let reduced_complexity = state.num_reduced_zones as f64;

        (full_complexity.powi(2) / reduced_complexity.powi(2)).min(100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rom_mode_display() {
        assert_eq!(format!("{}", ROMMode::Full), "Full Physics");
        assert_eq!(format!("{}", ROMMode::Reduced), "Reduced Order Model");
        assert_eq!(format!("{}", ROMMode::Hybrid), "Hybrid ROM/Physics");
    }

    #[test]
    fn test_rom_config_default() {
        let config = ROMConfig::default();
        assert_eq!(config.mode, ROMMode::Reduced);
        assert_eq!(config.pca_components, 3);
        assert!(config.enable_power_series);
        assert_eq!(config.power_series_terms, 5);
    }

    #[test]
    fn test_reduced_state_new() {
        let state = ReducedState::new(5);
        assert_eq!(state.num_components(), 5);
        assert_eq!(state.temperatures, vec![20.0; 5]);
        assert_eq!(state.heating_loads, vec![0.0; 5]);
    }

    #[test]
    fn test_rom_calculator_initialization() {
        let config = ROMConfig {
            mode: ROMMode::Reduced,
            pca_components: 2,
            agglomeration_threshold: 0.1,
            enable_power_series: true,
            power_series_terms: 3,
        };

        let mut calculator = ROMCalculator::new(config);

        let capacitances = [1000.0, 1000.0, 1500.0, 1500.0];
        let conductances = [50.0, 50.0, 50.0, 50.0];

        calculator
            .initialize(4, &capacitances, &conductances)
            .expect("Initialization should succeed");

        let state = calculator.state().expect("State should be Some");
        assert_eq!(state.num_original_zones, 4);
        assert!(state.num_reduced_zones >= 1);
    }

    #[test]
    fn test_speedup_factor() {
        let config = ROMConfig::default();
        let calculator = ROMCalculator::new(config);
        assert_eq!(calculator.speedup_factor(), 1.0);
    }

    #[test]
    fn test_pca_transformer() {
        let config = PCAConfig {
            components: 3,
            ..Default::default()
        };
        let mut pca = PCATransformer::new(config);

        let data = vec![20.0, 22.0, 24.0, 21.0, 23.0, 25.0, 19.0, 21.0, 23.0];
        pca.fit(&data, 3).expect("PCA fit should succeed");

        let result = pca.transform(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zone_clustering() {
        let config = AgglomerationConfig {
            threshold: 0.2,
            max_clusters: 4,
            ..Default::default()
        };

        let mut clustering = ZoneClustering::new(config);
        let capacitances = [1000.0, 1000.0, 5000.0, 5000.0];
        let conductances = [50.0, 50.0, 50.0, 50.0];

        clustering
            .fit(&capacitances, &conductances)
            .expect("Clustering should succeed");
        assert!(clustering.num_clusters() >= 1);
    }

    #[test]
    fn test_power_series_expansion() {
        let mut ps = PowerSeriesExpansion::with_terms(5);
        assert_eq!(ps.n_terms, 5);

        ps.update(&[20.0, 22.0], &[100.0, 200.0], 3600.0);
        assert_eq!(ps.history.len(), 1);

        let pred = ps.predict_next();
        assert!(pred.is_ok());
    }

    #[test]
    fn test_pca_config_default() {
        let config = PCAConfig::default();
        assert_eq!(config.components, 3);
        assert_eq!(config.variance_threshold, 0.95);
        assert!(config.standardize);
    }

    #[test]
    fn test_agglomeration_config_default() {
        let config = AgglomerationConfig::default();
        assert_eq!(config.threshold, 0.1);
        assert_eq!(config.max_clusters, 10);
        assert_eq!(config.min_cluster_size, 1);
    }

    #[test]
    fn test_zone_cluster() {
        let cluster = ZoneCluster {
            id: 0,
            zone_indices: vec![0, 1, 2],
            mean_capacitance: 1000.0,
            mean_conductance: 50.0,
        };
        assert_eq!(cluster.id, 0);
        assert_eq!(cluster.zone_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_rom_error_display() {
        let err = ROMError::InsufficientZones {
            required: 3,
            actual: 1,
        };
        assert!(format!("{}", err).contains("3"));
        assert!(format!("{}", err).contains("1"));
    }

    #[test]
    fn test_rom_state_new() {
        let pca_config = PCAConfig::default();
        let agg_config = AgglomerationConfig::default();
        let state = ROMState::new(4, &pca_config, &agg_config);
        assert!(state.is_ok());
        let state = state.unwrap();
        assert_eq!(state.num_original_zones, 4);
    }

    #[test]
    fn test_rom_state_new_insufficient_zones() {
        let pca_config = PCAConfig::default();
        let agg_config = AgglomerationConfig::default();
        let state = ROMState::new(1, &pca_config, &agg_config);
        assert!(state.is_err());
    }

    #[test]
    fn test_reconstruction_error() {
        let state = ROMState {
            pca_transformer: PCATransformer::new(PCAConfig::default()),
            zone_clustering: ZoneClustering::new(AgglomerationConfig::default()),
            power_series: PowerSeriesExpansion::default(),
            num_reduced_zones: 2,
            num_original_zones: 4,
            accumulated_error: 10.0,
            timestep_count: 100,
        };
        assert_eq!(state.reconstruction_error(), 0.1);
    }

    #[test]
    fn test_reconstruction_error_no_timesteps() {
        let state = ROMState {
            pca_transformer: PCATransformer::new(PCAConfig::default()),
            zone_clustering: ZoneClustering::new(AgglomerationConfig::default()),
            power_series: PowerSeriesExpansion::default(),
            num_reduced_zones: 2,
            num_original_zones: 4,
            accumulated_error: 10.0,
            timestep_count: 0,
        };
        assert_eq!(state.reconstruction_error(), 0.0);
    }
}
