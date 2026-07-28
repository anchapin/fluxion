//! # fluxion-twin
//!
//! Digital twin core — Unscented Kalman Filter for non-linear state estimation.
//!
//! The UKF uses the sigma-point approach to capture mean and covariance propagation
//! through non-linear transformations without linearization errors inherent in the
//! Extended Kalman Filter (EKF).
//!
//! # Architecture
//!
//! - [`UnscentedKalmanFilter`] — main filter struct with generic state/measurement types
//! - [`StateTransitionFn`] — function pointer: `x_{k+1} = f(x_k, u_k)`
//! - [`MeasurementFn`] — function pointer: `y_k = h(x_k)`
//! - [`KalmanError`] — error enum for all filter operations
//!
//! # References
//!
//! - Wan, E.A. & van der Merwe, R. (2000). The Unscented Kalman Filter for Nonlinear Estimation.

use nalgebra::DMatrix;
#[cfg(test)]
use nalgebra::DVector;
use std::vec::Vec;

pub mod error;
pub mod telemetry;
pub use error::KalmanError;
pub use telemetry::{MqttTelemetryConsumer, TelemetryError, TelemetryMessage};

pub trait StateVector: Clone {
    fn zeros() -> Self;
    fn as_slice(&self) -> &[f64];
    fn from_slice(slice: &[f64]) -> Self;
    fn dim(&self) -> usize;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn scale(&self, s: f64) -> Self;
}

pub trait MeasurementVector: Clone {
    fn zeros() -> Self;
    fn as_slice(&self) -> &[f64];
    fn from_slice(slice: &[f64]) -> Self;
    fn dim(&self) -> usize;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn scale(&self, s: f64) -> Self;
}

impl StateVector for Vec<f64> {
    fn zeros() -> Self {
        vec![0.0]
    }
    fn as_slice(&self) -> &[f64] {
        self
    }
    fn from_slice(slice: &[f64]) -> Self {
        slice.to_vec()
    }
    fn dim(&self) -> usize {
        self.len()
    }
    fn add(&self, other: &Self) -> Self {
        self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }
    fn sub(&self, other: &Self) -> Self {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }
    fn scale(&self, s: f64) -> Self {
        self.iter().map(|x| x * s).collect()
    }
}

impl MeasurementVector for Vec<f64> {
    fn zeros() -> Self {
        vec![0.0]
    }
    fn as_slice(&self) -> &[f64] {
        self
    }
    fn from_slice(slice: &[f64]) -> Self {
        slice.to_vec()
    }
    fn dim(&self) -> usize {
        self.len()
    }
    fn add(&self, other: &Self) -> Self {
        self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }
    fn sub(&self, other: &Self) -> Self {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }
    fn scale(&self, s: f64) -> Self {
        self.iter().map(|x| x * s).collect()
    }
}

pub struct UnscentedKalmanFilter<S, M>
where
    S: StateVector + Send + Sync,
    M: MeasurementVector + Send + Sync,
{
    pub state: S,
    pub p_covariance: DMatrix<f64>,
    pub process_noise: DMatrix<f64>,
    pub measurement_noise: DMatrix<f64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
    n: usize,
    m: usize,
    state_transition: Box<dyn Fn(&S, &[f64]) -> S + Send + Sync>,
    measurement_fn: Box<dyn Fn(&S) -> M + Send + Sync>,
}

impl<S, M> UnscentedKalmanFilter<S, M>
where
    S: StateVector + Send + Sync,
    M: MeasurementVector + Send + Sync,
{
    pub fn new(
        initial_state: S,
        initial_covariance: DMatrix<f64>,
        process_noise: DMatrix<f64>,
        measurement_noise: DMatrix<f64>,
        state_transition: impl Fn(&S, &[f64]) -> S + Send + Sync + 'static,
        measurement_fn: impl Fn(&S) -> M + Send + Sync + 'static,
    ) -> Self {
        let n = initial_state.dim();
        let m = measurement_fn(&initial_state).dim();
        assert_eq!(initial_covariance.nrows(), n);
        assert_eq!(initial_covariance.ncols(), n);
        assert_eq!(process_noise.nrows(), n);
        assert_eq!(process_noise.ncols(), n);
        assert_eq!(measurement_noise.nrows(), m);
        assert_eq!(measurement_noise.ncols(), m);
        Self {
            state: initial_state,
            p_covariance: initial_covariance,
            process_noise,
            measurement_noise,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 3.0 - (1.0 * n as f64),
            n,
            m,
            state_transition: Box::new(state_transition),
            measurement_fn: Box::new(measurement_fn),
        }
    }

    pub fn state_dim(&self) -> usize {
        self.n
    }

    pub fn measurement_dim(&self) -> usize {
        self.m
    }

    fn lambda(&self) -> f64 {
        self.alpha.powi(2) * (self.n as f64 + self.kappa) - self.n as f64
    }

    fn sigma_point_weights_mean(&self) -> Vec<f64> {
        let n = self.n as f64;
        let lambda = self.lambda();
        let mut w = vec![lambda / (n + lambda)];
        let w_leaf = 0.5 / (n + lambda);
        for _ in 0..2 * self.n {
            w.push(w_leaf);
        }
        w
    }

    fn sigma_point_weights_cov(&self) -> Vec<f64> {
        let n = self.n as f64;
        let lambda = self.lambda();
        let mut w = vec![lambda / (n + lambda) + (1.0 - self.alpha.powi(2) + self.beta)];
        let w_leaf = 0.5 / (n + lambda);
        for _ in 0..2 * self.n {
            w.push(w_leaf);
        }
        w
    }

    fn matrix_sqrt(&self, mat: &DMatrix<f64>) -> Result<DMatrix<f64>, KalmanError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(KalmanError::DimensionMismatch {
                expected: n,
                got: mat.ncols(),
            });
        }

        let mut l = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                if i == j {
                    let diag_sq = mat[(i, i)] - sum;
                    if diag_sq <= 0.0 {
                        l[(i, j)] = 0.0;
                    } else {
                        l[(i, j)] = diag_sq.sqrt();
                    }
                } else {
                    let denom = l[(j, j)];
                    if denom == 0.0 {
                        l[(i, j)] = 0.0;
                    } else {
                        l[(i, j)] = (mat[(i, j)] - sum) / denom;
                    }
                }
            }
        }

        let mut is_nonzero = false;
        for i in 0..n {
            for j in 0..=i {
                if l[(i, j)] != 0.0 {
                    is_nonzero = true;
                    break;
                }
            }
        }

        if !is_nonzero {
            return Err(KalmanError::NonPositiveDefiniteMatrix);
        }

        Ok(l)
    }

    fn generate_sigma_points(&self) -> Result<(Vec<S>, Vec<f64>, Vec<f64>), KalmanError> {
        let n = self.n as f64;
        let lambda = self.lambda();
        let sqrt_cov = self.matrix_sqrt(&(self.p_covariance.clone() * (n + lambda)))?;
        let w_m = self.sigma_point_weights_mean();
        let w_c = self.sigma_point_weights_cov();

        let mut sigma_points = Vec::with_capacity(2 * self.n + 1);
        sigma_points.push(self.state.clone());

        for i in 0..self.n {
            let col_i = (0..self.n).map(|r| sqrt_cov[(r, i)]).collect::<Vec<_>>();
            sigma_points.push(self.state.add(&S::from_slice(&col_i)));
            sigma_points.push(self.state.sub(&S::from_slice(&col_i)));
        }

        Ok((sigma_points, w_m, w_c))
    }

    pub fn predict(&mut self, u: &[f64]) -> Result<(), KalmanError> {
        let (sigma_points, w_m, w_c) = self.generate_sigma_points()?;

        let mut x_pred_vec = vec![0.0; self.n];
        for (i, sp) in sigma_points.iter().enumerate() {
            let propagated = (self.state_transition)(sp, u);
            let pv = propagated.as_slice();
            for j in 0..self.n {
                x_pred_vec[j] += w_m[i] * pv[j];
            }
        }
        let x_pred = S::from_slice(&x_pred_vec);

        let mut p_pred = DMatrix::zeros(self.n, self.n);
        for (i, sp) in sigma_points.iter().enumerate() {
            let diff = sp.sub(&x_pred);
            let diff_vec = diff.as_slice();
            for a in 0..self.n {
                for b in 0..self.n {
                    p_pred[(a, b)] += w_c[i] * diff_vec[a] * diff_vec[b];
                }
            }
        }
        p_pred = &p_pred + &self.process_noise;

        self.state = x_pred;
        self.p_covariance = p_pred;

        Ok(())
    }

    pub fn update(&mut self, z: &M) -> Result<(), KalmanError> {
        let (sigma_points, w_m, w_c) = self.generate_sigma_points()?;

        let z_sigma: Vec<M> = sigma_points
            .iter()
            .map(|sp| (self.measurement_fn)(sp))
            .collect();

        let mut z_pred_vec = vec![0.0; self.m];
        for (i, zs) in z_sigma.iter().enumerate() {
            let zv = zs.as_slice();
            for j in 0..self.m {
                z_pred_vec[j] += w_m[i] * zv[j];
            }
        }
        let z_pred = M::from_slice(&z_pred_vec);

        let mut s_cov = DMatrix::zeros(self.m, self.m);
        for (i, zs) in z_sigma.iter().enumerate() {
            let diff_z = zs.sub(&z_pred);
            let dz = diff_z.as_slice();
            for a in 0..self.m {
                for b in 0..self.m {
                    s_cov[(a, b)] += w_c[i] * dz[a] * dz[b];
                }
            }
        }
        s_cov = &s_cov + &self.measurement_noise;

        let s_inv = s_cov
            .clone()
            .try_inverse()
            .ok_or(KalmanError::SingularMatrix)?;

        let mut p_xz = DMatrix::zeros(self.n, self.m);
        for (i, sp) in sigma_points.iter().enumerate() {
            let diff_x = sp.sub(&self.state);
            let dx = diff_x.as_slice();
            let diff_z = z_sigma[i].sub(&z_pred);
            let dz = diff_z.as_slice();
            for a in 0..self.n {
                for b in 0..self.m {
                    p_xz[(a, b)] += w_c[i] * dx[a] * dz[b];
                }
            }
        }

        let k_gain = &p_xz * &s_inv;

        let innovation = z.sub(&z_pred);
        let innov_vec = innovation.as_slice();

        let kx_vec: Vec<f64> = (0..self.n)
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..self.m {
                    sum += k_gain[(i, j)] * innov_vec[j];
                }
                sum
            })
            .collect();

        self.state = self.state.add(&S::from_slice(&kx_vec));

        let kh = &k_gain * &s_cov;
        let kh_kt = &kh * &k_gain.transpose();
        let p_updated = &self.p_covariance - &kh_kt;

        self.p_covariance = p_updated;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_van_der_pol_oscillator_convergence() {
        let dt = 0.01;
        let n_steps = 200;

        let true_x0 = 2.0;
        let true_x1 = 0.0;
        let mu = 1.0;

        let initial_state = vec![true_x0 + 0.5, true_x1 + 0.3];
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![0.1, 0.1]));
        let r = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01]));

        let mut rng = StdRng::seed_from_u64(42);

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state.clone(),
            p0,
            q,
            r,
            |x: &Vec<f64>, u: &[f64]| -> Vec<f64> {
                let dt = u[0];
                let mu = u[1];
                let x0 = x[0];
                let x1 = x[1];
                vec![x0 + dt * x1, x1 + dt * (mu * (1.0 - x0 * x0) * x1 - x0)]
            },
            |x: &Vec<f64>| -> Vec<f64> { vec![x[0]] },
        );

        let mut true_state = vec![true_x0, true_x1];
        let mut estimated_states = Vec::new();
        let mut true_states = Vec::new();

        for _step in 0..n_steps {
            let u = vec![dt, mu];

            let noisy_measurement = true_state[0] + rng.gen::<f64>() * 0.1;

            ukf.predict(&u).unwrap();
            ukf.update(&vec![noisy_measurement]).unwrap();

            estimated_states.push(ukf.state.as_slice().to_vec());
            true_states.push(true_state.clone());

            let x0_next = true_state[0] + dt * true_state[1];
            let x1_next = true_state[1]
                + dt * (mu * (1.0 - true_state[0] * true_state[0]) * true_state[1] - true_state[0]);
            true_state = vec![x0_next, x1_next];
        }

        let converged_estimate = &estimated_states[9];
        let true_at_9 = &true_states[9];
        let tracking_error = ((converged_estimate[0] - true_at_9[0]).powi(2)
            + (converged_estimate[1] - true_at_9[1]).powi(2))
        .sqrt();

        assert!(
            tracking_error < 1.0,
            "UKF should converge within 10 steps, tracking error = {}",
            tracking_error
        );

        let p_final = &ukf.p_covariance;
        assert!(
            p_final[(0, 0)] >= 0.0,
            "Covariance diagonal should be non-negative"
        );
        assert!(
            p_final[(1, 1)] >= 0.0,
            "Covariance diagonal should be non-negative"
        );

        let det = p_final[(0, 0)] * p_final[(1, 1)] - p_final[(0, 1)] * p_final[(1, 0)];
        assert!(
            det >= 0.0,
            "Covariance determinant should be non-negative (positive semi-definite)"
        );
    }

    #[test]
    fn test_covariance_positive_semi_definite() {
        let initial_state = vec![1.0, 0.0];
        let mut p0 = DMatrix::zeros(2, 2);
        p0[(0, 0)] = 1.0;
        p0[(0, 1)] = 0.5;
        p0[(1, 0)] = 0.5;
        p0[(1, 1)] = 1.0;
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01]));
        let r = DMatrix::from_diagonal(&DVector::from_vec(vec![0.1]));

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            p0,
            q,
            r,
            |x: &Vec<f64>, _: &[f64]| -> Vec<f64> { vec![x[0] * 0.9, x[1] * 0.8] },
            |x: &Vec<f64>| -> Vec<f64> { vec![x[0]] },
        );

        for _ in 0..50 {
            ukf.predict(&vec![0.0]).unwrap();
            ukf.update(&vec![1.0]).unwrap();

            let p = &ukf.p_covariance;
            assert!(
                p[(0, 0)] >= 0.0 && p[(1, 1)] >= 0.0,
                "Covariance diagonal must be non-negative"
            );

            let det = p[(0, 0)] * p[(1, 1)] - p[(0, 1)] * p[(1, 0)];
            assert!(
                det >= 0.0,
                "Covariance determinant must be >= 0 (positive semi-definite). Got det = {}",
                det
            );

            let trace = p[(0, 0)] + p[(1, 1)];
            assert!(trace >= 0.0, "Covariance trace must be non-negative");
        }
    }

    #[test]
    fn test_matrix_sqrt_positive_definite() {
        let mut p = DMatrix::zeros(2, 2);
        p[(0, 0)] = 4.0;
        p[(0, 1)] = 2.0;
        p[(1, 0)] = 2.0;
        p[(1, 1)] = 5.0;

        let ukf: UnscentedKalmanFilter<Vec<f64>, Vec<f64>> = UnscentedKalmanFilter::new(
            vec![1.0, 0.0],
            DMatrix::identity(2, 2),
            DMatrix::identity(2, 2),
            DMatrix::identity(1, 1),
            |_, _| vec![0.0, 0.0],
            |_| vec![0.0],
        );

        let sqrt_p = ukf.matrix_sqrt(&p).unwrap();

        let sqrt_p_sq = &sqrt_p * &sqrt_p.transpose();

        for i in 0..2 {
            for j in 0..2 {
                let err = (sqrt_p_sq[(i, j)] - p[(i, j)]).abs();
                assert!(
                    err < 1e-10,
                    "sqrt(P)*sqrt(P)[{},{}] = {} should equal P[{},{}] = {}",
                    i,
                    j,
                    sqrt_p_sq[(i, j)],
                    i,
                    j,
                    p[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_inverse_against_identity() {
        let _ukf: UnscentedKalmanFilter<Vec<f64>, Vec<f64>> = UnscentedKalmanFilter::new(
            vec![1.0, 0.0],
            DMatrix::identity(2, 2),
            DMatrix::identity(2, 2),
            DMatrix::identity(1, 1),
            |_, _| vec![0.0, 0.0],
            |_| vec![0.0],
        );

        let identity = DMatrix::identity(2, 2);
        let inv_identity = identity
            .try_inverse()
            .ok_or(KalmanError::SingularMatrix)
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                let actual: f64 = inv_identity[(i, j)];
                let err = (actual - expected).abs();
                assert!(
                    err < 1e-10,
                    "inv(I)[{},{}] = {} should be {}",
                    i,
                    j,
                    actual,
                    expected
                );
            }
        }
    }
}
