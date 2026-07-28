//! Unscented Kalman Filter (UKF) implementation for state estimation.
//!
//! Issue #2022 (UKF core) — Phase 1 of the Twin project.
//!
//! This module provides a simplified UKF implementation specifically designed for
//! thermal State estimation with 3D state vectors (t_air, t_mass, t_surface).

use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::SeedableRng;

const UKF_ALPHA: f64 = 0.001;
const UKF_BETA: f64 = 2.0;
const UKF_KAPPA: f64 = 0.0;

const STATE_DIM: usize = 3;

#[derive(Clone, Debug)]
pub struct ThermalStateVector(Vector3<f64>);

impl ThermalStateVector {
    pub fn new(t_air: f64, t_mass: f64, t_surface: f64) -> Self {
        ThermalStateVector(Vector3::new(t_air, t_mass, t_surface))
    }

    pub fn t_air(&self) -> f64 {
        self.0[0]
    }

    pub fn t_mass(&self) -> f64 {
        self.0[1]
    }

    pub fn t_surface(&self) -> f64 {
        self.0[2]
    }
}

impl From<ThermalStateVector> for Vector3<f64> {
    fn from(s: ThermalStateVector) -> Self {
        s.0
    }
}

impl From<Vector3<f64>> for ThermalStateVector {
    fn from(v: Vector3<f64>) -> Self {
        ThermalStateVector(v)
    }
}

#[derive(Clone, Debug)]
pub struct ThermalMeasurementVector(Vector3<f64>);

impl ThermalMeasurementVector {
    pub fn new(t_air_measured: f64) -> Self {
        ThermalMeasurementVector(Vector3::new(t_air_measured, 0.0, 0.0))
    }

    pub fn t_air_measured(&self) -> f64 {
        self.0[0]
    }
}

impl From<ThermalMeasurementVector> for Vector3<f64> {
    fn from(m: ThermalMeasurementVector) -> Self {
        m.0
    }
}

impl From<Vector3<f64>> for ThermalMeasurementVector {
    fn from(v: Vector3<f64>) -> Self {
        ThermalMeasurementVector(v)
    }
}

pub struct UnscentedKalmanFilter {
    state: ThermalStateVector,
    covariance: Matrix3<f64>,
    process_noise: f64,
    measurement_noise: f64,
    lambda: f64,
    weights_mean: Vec<f64>,
    weights_cov: Vec<f64>,
    sigma_points: Vec<Vector3<f64>>,
    _rng: StdRng,
}

impl UnscentedKalmanFilter {
    pub fn new(
        initial_state: ThermalStateVector,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        let n = STATE_DIM;
        let lambda = UKF_ALPHA.powi(2) * (n as f64 + UKF_KAPPA) - n as f64;

        let weights_mean = Self::compute_weights_mean(n, lambda);
        let weights_cov = Self::compute_weights_cov(n, lambda);

        let covariance = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));

        Self {
            state: initial_state,
            covariance,
            process_noise,
            measurement_noise,
            lambda,
            weights_mean,
            weights_cov,
            sigma_points: Vec::with_capacity(2 * n + 1),
            _rng: StdRng::seed_from_u64(42),
        }
    }

    fn compute_weights_mean(n: usize, lambda: f64) -> Vec<f64> {
        let mut weights = Vec::with_capacity(2 * n + 1);
        weights.push(lambda / (n as f64 + lambda));
        for _ in 0..2 * n {
            weights.push(1.0 / (2.0 * (n as f64 + lambda)));
        }
        weights
    }

    fn compute_weights_cov(n: usize, lambda: f64) -> Vec<f64> {
        let mut weights = Vec::with_capacity(2 * n + 1);
        weights.push(lambda / (n as f64 + lambda) + (1.0 - UKF_ALPHA.powi(2) + UKF_BETA));
        for _ in 0..2 * n {
            weights.push(1.0 / (2.0 * (n as f64 + lambda)));
        }
        weights
    }

    fn compute_sigma_points(&mut self) {
        let state_vec: Vector3<f64> = self.state.clone().into();
        let sqrt_cov = self.covariance.clone().cholesky().unwrap().l();
        let gamma = (STATE_DIM as f64 + self.lambda).sqrt();

        self.sigma_points.clear();
        self.sigma_points.push(state_vec.clone());

        for i in 0..STATE_DIM {
            let col = sqrt_cov.column(i).into_owned();
            let plus = state_vec.clone() + col * gamma;
            let minus = state_vec.clone() - col * gamma;
            self.sigma_points.push(plus);
            self.sigma_points.push(minus);
        }
    }

    pub fn predict<F>(&mut self, state_transition: &F, _control: &())
    where
        F: Fn(&ThermalStateVector, f64) -> ThermalStateVector,
    {
        self.compute_sigma_points();

        let mut predicted_state_sum = Vector3::zeros();
        for (i, sigma) in self.sigma_points.iter().enumerate() {
            let sigma_state: ThermalStateVector = sigma.clone().into();
            let predicted = state_transition(&sigma_state, 1.0);
            let predicted_vec: Vector3<f64> = predicted.into();
            predicted_state_sum += self.weights_mean[i] * predicted_vec;
        }

        let mut predicted_cov = Matrix3::zeros();
        for (i, sigma) in self.sigma_points.iter().enumerate() {
            let sigma_state: ThermalStateVector = sigma.clone().into();
            let predicted = state_transition(&sigma_state, 1.0);
            let predicted_vec: Vector3<f64> = predicted.into();
            let diff = predicted_vec.clone() - predicted_state_sum.clone();
            predicted_cov += self.weights_cov[i] * diff.clone() * diff.transpose();
        }

        let process_noise_matrix = Matrix3::from_diagonal(&Vector3::new(
            self.process_noise,
            self.process_noise,
            self.process_noise,
        ));
        predicted_cov += process_noise_matrix;

        self.state = predicted_state_sum.into();
        self.covariance = predicted_cov;
    }

    pub fn update<G>(&mut self, measurement: &ThermalMeasurementVector, measurement_fn: &G)
    where
        G: Fn(&ThermalStateVector) -> ThermalMeasurementVector,
    {
        let measurement_vec: Vector3<f64> = measurement.clone().into();

        self.compute_sigma_points();

        let mut predicted_meas_sum = Vector3::zeros();
        for (i, sigma) in self.sigma_points.iter().enumerate() {
            let sigma_state: ThermalStateVector = sigma.clone().into();
            let predicted_meas = measurement_fn(&sigma_state);
            let meas_vec: Vector3<f64> = predicted_meas.into();
            predicted_meas_sum += self.weights_mean[i] * meas_vec;
        }

        let mut innovation_cov = Matrix3::zeros();
        for (i, sigma) in self.sigma_points.iter().enumerate() {
            let sigma_state: ThermalStateVector = sigma.clone().into();
            let predicted_meas = measurement_fn(&sigma_state);
            let meas_vec: Vector3<f64> = predicted_meas.into();
            let diff = meas_vec.clone() - predicted_meas_sum.clone();
            innovation_cov += self.weights_cov[i] * diff.clone() * diff.transpose();
        }

        let noise_matrix = Matrix3::from_diagonal(&Vector3::new(self.measurement_noise, 0.0, 0.0));
        innovation_cov += noise_matrix;

        let mut cross_cov = Matrix3::zeros();
        for (i, sigma) in self.sigma_points.iter().enumerate() {
            let sigma_state: ThermalStateVector = sigma.clone().into();
            let sigma_state_vec: Vector3<f64> = sigma_state.clone().into();
            let current_state_vec: Vector3<f64> = self.state.clone().into();
            let predicted_meas = measurement_fn(&sigma_state);
            let meas_vec: Vector3<f64> = predicted_meas.into();
            let state_diff = sigma_state_vec.clone() - current_state_vec.clone();
            let meas_diff = meas_vec.clone() - predicted_meas_sum.clone();
            cross_cov += self.weights_cov[i] * state_diff * meas_diff.transpose();
        }

        let innovation_cov_clone = innovation_cov.clone();
        let kalman_gain = if let Some(inv) = innovation_cov_clone.try_inverse() {
            cross_cov * inv
        } else {
            Matrix3::zeros()
        };

        let innovation = measurement_vec.clone() - predicted_meas_sum.clone();
        let state_update = kalman_gain.clone() * innovation;

        let state_vec: Vector3<f64> = self.state.clone().into();
        self.state = (state_vec + state_update).into();

        let gain_cross = kalman_gain.clone() * innovation_cov;
        self.covariance = self.covariance.clone() - gain_cross * kalman_gain.transpose();
    }

    pub fn state(&self) -> &ThermalStateVector {
        &self.state
    }

    pub fn covariance(&self) -> &Matrix3<f64> {
        &self.covariance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ukf_creation() {
        let initial = ThermalStateVector::new(20.0, 18.0, 19.0);
        let ukf = UnscentedKalmanFilter::new(initial, 0.1, 0.1);
        assert_eq!(ukf.state.0.len(), 3);
    }

    #[test]
    fn test_ukf_sigma_points() {
        let initial = ThermalStateVector::new(20.0, 18.0, 19.0);
        let mut ukf = UnscentedKalmanFilter::new(initial, 0.1, 0.1);
        ukf.compute_sigma_points();
        assert_eq!(ukf.sigma_points.len(), 7);
    }

    #[test]
    fn test_thermal_state_vector() {
        let state = ThermalStateVector::new(20.0, 18.0, 19.0);
        assert!((state.t_air() - 20.0).abs() < 1e-10);
        assert!((state.t_mass() - 18.0).abs() < 1e-10);
        assert!((state.t_surface() - 19.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_vector() {
        let meas = ThermalMeasurementVector::new(22.0);
        assert!((meas.t_air_measured() - 22.0).abs() < 1e-10);
    }
}
