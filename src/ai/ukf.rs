//! Unscented Kalman Filter (UKF) for nonlinear state estimation
//!
//! Implements the van der Merwe scaled sigma point UKF algorithm for
//! nonlinear state estimation. Used for digital twin state estimation
//! in building energy models.
//!
//! # References
//! - Van der Merwe, R. (2004). Sigma-Point Kalman Filters for Probabilistic
//!   Inference in Dynamic State-Space Models
//! - Julier, S.J. & Uhlmann, J.K. (1997). A New Extension of the Kalman
//!   Filter to Nonlinear Systems

use nalgebra::{DMatrix, DVector};
use rand_distr::Distribution;

pub struct UnscentedKalmanFilter {
    state: DVector<f64>,
    P: DMatrix<f64>,
    Q: DMatrix<f64>,
    R: DMatrix<f64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
    n: usize,
}

impl UnscentedKalmanFilter {
    pub fn new(initial_state: DVector<f64>, process_noise: f64, measurement_noise: f64) -> Self {
        let n = initial_state.len();
        let Q = DMatrix::from_diagonal(&DVector::from_element(n, process_noise));
        let R = DMatrix::from_diagonal(&DVector::from_element(1, measurement_noise));
        let P = DMatrix::from_diagonal(&DVector::from_element(n, 1.0));

        Self {
            state: initial_state,
            P,
            Q,
            R,
            alpha: 0.001,
            beta: 2.0,
            kappa: 0.0,
            n,
        }
    }

    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.P
    }

    fn sigma_points(&self) -> (Vec<DVector<f64>>, Vec<f64>, Vec<f64>) {
        let lambda_param = self.alpha.powi(2) * (self.n as f64 + self.kappa) - self.n as f64;
        let c = (self.n as f64 + lambda_param).sqrt();

        let mut sigma: Vec<DVector<f64>> = Vec::with_capacity(2 * self.n + 1);
        sigma.push(self.state.clone());

        for i in 0..self.n {
            let col = self.P.column(i).into_owned();
            let scaled_col = col * c;
            sigma.push(&self.state + &scaled_col);
            sigma.push(&self.state - &scaled_col);
        }

        let w0_mean = lambda_param / (self.n as f64 + lambda_param);
        let w0_cov = w0_mean + (1.0 - self.alpha.powi(2) + self.beta);
        let wi = 1.0 / (2.0 * (self.n as f64 + lambda_param));

        let mut weights_mean = vec![w0_mean; 2 * self.n + 1];
        let mut weights_cov = vec![w0_cov; 2 * self.n + 1];
        weights_mean[1..].fill(wi);
        weights_cov[1..].fill(wi);

        (sigma, weights_mean, weights_cov)
    }

    pub fn predict<F>(&mut self, f: F, _dt: &())
    where
        F: Fn(&DVector<f64>, &()) -> DVector<f64>,
    {
        let (sigma, weights_mean, weights_cov) = self.sigma_points();

        let mut x_pred = DVector::zeros(self.n);
        for (i, sigma_i) in sigma.iter().enumerate() {
            x_pred += weights_mean[i] * f(sigma_i, _dt);
        }

        let mut P_pred = DMatrix::zeros(self.n, self.n);
        for (i, sigma_i) in sigma.iter().enumerate() {
            let diff = f(sigma_i, _dt) - &x_pred;
            P_pred += weights_cov[i] * diff.clone() * diff.transpose();
        }
        P_pred += &self.Q;

        self.state = x_pred;
        self.P = P_pred;
    }

    pub fn update<G>(&mut self, z: f64, g: G)
    where
        G: Fn(&DVector<f64>) -> f64,
    {
        let (sigma, weights_mean, weights_cov) = self.sigma_points();

        let mut z_pred = 0.0;
        for (i, sigma_i) in sigma.iter().enumerate() {
            z_pred += weights_mean[i] * g(sigma_i);
        }

        let mut P_zz = 0.0;
        for (i, sigma_i) in sigma.iter().enumerate() {
            let diff = g(sigma_i) - z_pred;
            P_zz += weights_cov[i] * diff.powi(2);
        }
        P_zz += self.R[(0, 0)];

        let mut P_xz = DVector::zeros(self.n);
        for (i, sigma_i) in sigma.iter().enumerate() {
            let diff_x = &sigma[i] - &self.state;
            let diff_z = g(sigma_i) - z_pred;
            P_xz += weights_cov[i] * diff_x * diff_z;
        }

        let K = P_xz.clone() / P_zz;

        let innovation = z - z_pred;
        self.state = self.state.clone() + K.clone() * innovation;
        self.P = &self.P - (K.clone() * K.transpose()) * P_zz;
    }
}

pub fn rand_normal() -> f64 {
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();
    normal.sample(&mut rng)
}

pub fn van_der_pol_state(x0: f64, y0: f64) -> DVector<f64> {
    DVector::from_vec(vec![x0, y0])
}

pub fn van_der_pol_step(state: &DVector<f64>, dt: f64) -> DVector<f64> {
    let mu = 1.0;
    let x = state[0];
    let y = state[1];

    let k1_dx = y;
    let k1_dy = mu * (1.0 - x * x) * y - x;

    let x2 = x + 0.5 * dt * k1_dx;
    let y2 = y + 0.5 * dt * k1_dy;
    let k2_dx = y2;
    let k2_dy = mu * (1.0 - x2 * x2) * y2 - x2;

    let x3 = x + 0.5 * dt * k2_dx;
    let y3 = y + 0.5 * dt * k2_dy;
    let k3_dx = y3;
    let k3_dy = mu * (1.0 - x3 * x3) * y3 - x3;

    let x4 = x + dt * k3_dx;
    let y4 = y + dt * k3_dy;
    let k4_dx = y4;
    let k4_dy = mu * (1.0 - x4 * x4) * y4 - x4;

    let x_new = x + (dt / 6.0) * (k1_dx + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx);
    let y_new = y + (dt / 6.0) * (k1_dy + 2.0 * k2_dy + 2.0 * k3_dy + k4_dy);

    DVector::from_vec(vec![x_new, y_new])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ukf_van_der_pol_convergence() {
        let mut ukf = UnscentedKalmanFilter::new(van_der_pol_state(0.0, 1.0), 0.01, 0.1);

        let dt = 0.01;
        let mut estimates = Vec::new();

        for step in 0..100 {
            let true_state = van_der_pol_step(ukf.state(), dt);

            let measurement = true_state[0] + 0.1 * rand_normal();

            ukf.predict(|s, _| van_der_pol_step(s, dt), &());
            ukf.update(measurement, |s| s[0]);

            estimates.push(ukf.state().clone());

            if step >= 10 {
                let error = (estimates[step][0] - true_state[0]).abs();
                assert!(
                    error < 0.3,
                    "UKF did not converge at step {}, error: {}",
                    step,
                    error
                );
            }
        }
    }
}
