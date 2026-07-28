use faer::Mat;

use crate::error::{KalmanError, KalmanResult};

#[derive(Debug, Clone)]
pub struct UnscentedKalmanFilter {
    state: Vec<f64>,
    p_covariance: Vec<Vec<f64>>,
    n: usize,
    m: usize,
    process_noise: Vec<Vec<f64>>,
    measurement_noise: Vec<Vec<f64>>,
    alpha: f64,
    beta: f64,
    kappa: f64,
}

impl UnscentedKalmanFilter {
    pub fn new(state: Vec<f64>, process_noise: f64, measurement_noise: f64) -> Self {
        let n = state.len();
        let m = state.len();

        let p_covariance = identity(n);
        let process_noise = scalar_mul(&identity(n), process_noise);
        let measurement_noise = scalar_mul(&identity(m), measurement_noise);

        Self {
            state,
            p_covariance,
            n,
            m,
            process_noise,
            measurement_noise,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    pub fn with_covariances(
        state: Vec<f64>,
        p_covariance: Vec<Vec<f64>>,
        process_noise: Vec<Vec<f64>>,
        measurement_noise: Vec<Vec<f64>>,
    ) -> Self {
        let n = state.len();
        let m = measurement_noise.len();
        Self {
            state,
            p_covariance,
            n,
            m,
            process_noise,
            measurement_noise,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    pub fn covariance(&self) -> &Vec<Vec<f64>> {
        &self.p_covariance
    }

    pub fn predict<F>(&mut self, state_transition: F, control_input: &[f64])
    where
        F: Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync,
    {
        let lambda_ = self.alpha.powi(2) * (self.n as f64 + self.kappa) - self.n as f64;
        let c = (self.n as f64 + lambda_).sqrt();

        let sigma_points = self.generate_sigma_points(c);

        let predicted_sigma_points: Vec<Vec<f64>> = sigma_points
            .iter()
            .map(|sp| state_transition(sp, control_input))
            .collect();

        let predicted_state = Self::compute_mean_impl(&predicted_sigma_points, lambda_, self.n);
        let predicted_cov = self.compute_covariance_impl(
            &predicted_sigma_points,
            &predicted_state,
            lambda_,
            &self.process_noise,
        );

        self.state = predicted_state;
        self.p_covariance = predicted_cov;
    }

    pub fn update<F>(&mut self, measurement: &[f64], measurement_function: F) -> KalmanResult<()>
    where
        F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    {
        let lambda_ = self.alpha.powi(2) * (self.n as f64 + self.kappa) - self.n as f64;
        let c = (self.n as f64 + lambda_).sqrt();

        let sigma_points = self.generate_sigma_points(c);

        let predicted_measurements: Vec<Vec<f64>> = sigma_points
            .iter()
            .map(|sp| measurement_function(sp))
            .collect();

        let mean_measurement = Self::compute_mean_impl(&predicted_measurements, lambda_, self.m);

        let cross_cov = Self::compute_cross_cov_impl(
            &sigma_points,
            &self.state,
            &predicted_measurements,
            &mean_measurement,
            lambda_,
            self.n,
            self.m,
        );

        let meas_cov = Self::compute_meas_cov_impl(
            &predicted_measurements,
            &mean_measurement,
            lambda_,
            self.m,
        );

        let meas_cov_with_noise = mat_add(&meas_cov, &self.measurement_noise);

        let kalman_gain =
            Self::compute_kalman_gain_impl(&cross_cov, &meas_cov_with_noise, self.n, self.m)?;

        let mut innovation = measurement.to_vec();
        for i in 0..self.m {
            innovation[i] -= mean_measurement[i];
        }

        let state_update = mat_vec_mul(&kalman_gain, &innovation);

        self.state = vec_add(&self.state, &state_update);

        let p_cov_update =
            Self::compute_cov_update_impl(&kalman_gain, &meas_cov_with_noise, self.n)?;

        self.p_covariance = p_cov_update;

        Ok(())
    }

    fn generate_sigma_points(&self, c: f64) -> Vec<Vec<f64>> {
        let mut sigma_points = Vec::with_capacity(2 * self.n + 1);
        sigma_points.push(self.state.clone());

        let p_cov = Mat::from_fn(self.n, self.n, |i, j| self.p_covariance[i][j]);
        let p_cov_copy = p_cov.clone();

        for i in 0..self.n {
            let col: Vec<f64> = (0..self.n).map(|row| p_cov_copy[(row, i)]).collect();
            let col_scaled: Vec<f64> = col.iter().map(|&val| val * c).collect();

            sigma_points.push(vec_add(&self.state, &col_scaled));
            sigma_points.push(vec_sub(&self.state, &col_scaled));
        }

        sigma_points
    }

    fn compute_mean_impl(points: &[Vec<f64>], lambda_: f64, dim: usize) -> Vec<f64> {
        let wm_0 = lambda_ / (dim as f64 + lambda_);
        let wm_i = 1.0 / (2.0 * (dim as f64 + lambda_));

        let mut mean = vec![0.0; dim];
        for i in 0..dim {
            mean[i] = wm_0 * points[0][i];
        }

        for point in points.iter().skip(1) {
            for i in 0..dim {
                mean[i] += wm_i * point[i];
            }
        }

        mean
    }

    fn compute_covariance_impl(
        &self,
        points: &[Vec<f64>],
        mean: &[f64],
        lambda_: f64,
        noise: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let wc_0 = lambda_ / (self.n as f64 + lambda_) + (1.0 - self.alpha.powi(2) + self.beta);
        let wc_i = 1.0 / (2.0 * (self.n as f64 + lambda_));

        let mut cov = noise.to_vec();

        let diff_0 = vec_sub(&points[0], mean);
        for i in 0..self.n {
            for j in 0..self.n {
                cov[i][j] += wc_0 * diff_0[i] * diff_0[j];
            }
        }

        for point in points.iter().skip(1) {
            let diff = vec_sub(point, mean);
            for i in 0..self.n {
                for j in 0..self.n {
                    cov[i][j] += wc_i * diff[i] * diff[j];
                }
            }
        }

        cov
    }

    fn compute_cross_cov_impl(
        sigma_points: &[Vec<f64>],
        state_mean: &[f64],
        predicted_measurements: &[Vec<f64>],
        mean_measurement: &[f64],
        lambda_: f64,
        n: usize,
        m: usize,
    ) -> Vec<Vec<f64>> {
        let wc_0 = lambda_ / (n as f64 + lambda_) + (1.0 - 1e-3_f64.powi(2) + 2.0);
        let wc_i = 1.0 / (2.0 * (n as f64 + lambda_));

        let mut cross_cov = vec![vec![0.0; m]; n];

        let state_diff_0 = vec_sub(&sigma_points[0], state_mean);
        let meas_diff_0 = vec_sub(&predicted_measurements[0], mean_measurement);
        for i in 0..n {
            for j in 0..m {
                cross_cov[i][j] += wc_0 * state_diff_0[i] * meas_diff_0[j];
            }
        }

        for (sp, mp) in sigma_points
            .iter()
            .skip(1)
            .zip(predicted_measurements.iter().skip(1))
        {
            let state_diff = vec_sub(sp, state_mean);
            let meas_diff = vec_sub(mp, mean_measurement);
            for i in 0..n {
                for j in 0..m {
                    cross_cov[i][j] += wc_i * state_diff[i] * meas_diff[j];
                }
            }
        }

        cross_cov
    }

    fn compute_meas_cov_impl(
        predicted_measurements: &[Vec<f64>],
        mean_measurement: &[f64],
        lambda_: f64,
        m: usize,
    ) -> Vec<Vec<f64>> {
        let n = predicted_measurements[0].len();
        let wc_0 = lambda_ / (n as f64 + lambda_) + (1.0 - 1e-3_f64.powi(2) + 2.0);
        let wc_i = 1.0 / (2.0 * (n as f64 + lambda_));

        let mut meas_cov = vec![vec![0.0; m]; m];

        let meas_diff_0 = vec_sub(&predicted_measurements[0], mean_measurement);
        for i in 0..m {
            for j in 0..m {
                meas_cov[i][j] += wc_0 * meas_diff_0[i] * meas_diff_0[j];
            }
        }

        for mp in predicted_measurements.iter().skip(1) {
            let meas_diff = vec_sub(mp, mean_measurement);
            for i in 0..m {
                for j in 0..m {
                    meas_cov[i][j] += wc_i * meas_diff[i] * meas_diff[j];
                }
            }
        }

        meas_cov
    }

    fn compute_kalman_gain_impl(
        cross_cov: &[Vec<f64>],
        meas_cov: &[Vec<f64>],
        n: usize,
        m: usize,
    ) -> KalmanResult<Vec<Vec<f64>>> {
        let inv_meas_cov = invert_matrix(meas_cov)?;

        let mut kalman_gain = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                for k in 0..m {
                    kalman_gain[i][j] += cross_cov[i][k] * inv_meas_cov[k][j];
                }
            }
        }

        Ok(kalman_gain)
    }

    fn compute_cov_update_impl(
        kalman_gain: &[Vec<f64>],
        meas_cov: &[Vec<f64>],
        n: usize,
    ) -> KalmanResult<Vec<Vec<f64>>> {
        let k_times_s = mat_mul(kalman_gain, meas_cov);
        let s_times_k_t = mat_mul(&k_times_s, &transpose(kalman_gain));

        let p_cov_update = mat_sub(&identity(n), &s_times_k_t);

        let symmetry_fix: Vec<Vec<f64>> = {
            let mut s = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    s[i][j] = (p_cov_update[i][j] + p_cov_update[j][i]) / 2.0;
                }
            }
            s
        };

        Ok(symmetry_fix)
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.n {
            for j in 0..self.n {
                if (self.p_covariance[i][j] - self.p_covariance[j][i]).abs() > 1e-8 {
                    return false;
                }
            }
        }
        true
    }

    pub fn trace(&self) -> f64 {
        let mut tr = 0.0;
        for i in 0..self.n {
            tr += self.p_covariance[i][i];
        }
        tr
    }

    pub fn has_nan_or_inf(&self) -> bool {
        for i in 0..self.n {
            for j in 0..self.n {
                let val = self.p_covariance[i][j];
                if val.is_nan() || val.is_infinite() {
                    return true;
                }
            }
        }
        false
    }

    #[allow(dead_code)]
    pub fn eigenvalues(&self) -> KalmanResult<Vec<f64>> {
        let mut eigenvals = Vec::with_capacity(self.n);

        if self.n == 2 {
            let a = self.p_covariance[0][0];
            let b = self.p_covariance[0][1];
            let c = self.p_covariance[1][0];
            let d = self.p_covariance[1][1];

            let trace = a + d;
            let det = a * d - b * c;
            let disc = (trace * trace - 4.0 * det).sqrt();

            eigenvals.push((trace + disc) / 2.0);
            eigenvals.push((trace - disc) / 2.0);
        } else {
            for i in 0..self.n {
                eigenvals.push(self.p_covariance[i][i]);
            }
        }

        Ok(eigenvals)
    }

    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }
}

fn identity(size: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; size]; size];
    for (i, row) in m.iter_mut().enumerate().take(size) {
        row[i] = 1.0;
    }
    m
}

fn scalar_mul(a: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|&val| val * scalar).collect())
        .collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| row_a.iter().zip(row_b.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| row_a.iter().zip(row_b.iter()).map(|(x, y)| x - y).collect())
        .collect()
}

fn mat_vec_mul(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
        .collect()
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let p = b[0].len();
    let m = b.len();

    let mut result = vec![vec![0.0; p]; n];
    for i in 0..n {
        for j in 0..p {
            for k in 0..m {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            result[j][i] = a[i][j];
        }
    }
    result
}

fn invert_matrix(a: &[Vec<f64>]) -> KalmanResult<Vec<Vec<f64>>> {
    let n = a.len();

    if n == 2 {
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if det.abs() < 1e-10 {
            return Err(KalmanError::SingularMatrix);
        }
        let inv_det = 1.0 / det;
        return Ok(vec![
            vec![a[1][1] * inv_det, -a[0][1] * inv_det],
            vec![-a[1][0] * inv_det, a[0][0] * inv_det],
        ]);
    }

    let mut aug = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.resize(2 * n, 0.0);
            new_row[n + i] = 1.0;
            new_row
        })
        .collect::<Vec<_>>();

    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }

        if aug[max_row][col].abs() < 1e-10 {
            return Err(KalmanError::SingularMatrix);
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        #[allow(clippy::needless_range_loop)]
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for i in 0..n {
            if i != col {
                let factor = aug[i][col];
                #[allow(clippy::needless_range_loop)]
                for j in 0..(2 * n) {
                    aug[i][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let inv = aug.iter().map(|row| row[n..].to_vec()).collect();

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_test(size: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; size]; size];
        for i in 0..size {
            m[i][i] = 1.0;
        }
        m
    }

    fn scalar_mul_test(a: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
        a.iter()
            .map(|row| row.iter().map(|&val| val * scalar).collect())
            .collect()
    }

    #[test]
    fn test_ukf_van_der_pol_oscillator() {
        const DT: f64 = 0.1;
        const NUM_STEPS: usize = 100;

        let initial_state = vec![2.0, 0.0];
        let initial_covariance = identity_test(2);
        let process_noise = scalar_mul_test(&identity_test(2), 0.01);
        let measurement_noise = scalar_mul_test(&identity_test(2), 0.1);

        let mut ukf = UnscentedKalmanFilter::with_covariances(
            initial_state.clone(),
            initial_covariance,
            process_noise,
            measurement_noise,
        );

        let state_transition = |x: &[f64], _u: &[f64]| -> Vec<f64> {
            let x1 = x[0];
            let x2 = x[1];
            let dx1 = x1 + DT * x2;
            let dx2 = x2 + DT * (0.1 * (1.0 - x1 * x1) * x2 - x1);
            vec![dx1, dx2]
        };

        let measurement_function = |x: &[f64]| -> Vec<f64> { vec![x[0], x[1]] };

        let mut converged = false;
        for step in 0..NUM_STEPS {
            ukf.predict(state_transition, &[]);

            let true_state = {
                let x1 = initial_state[0] * (1.0 + 0.001 * (step as f64) * (step as f64).cos());
                let x2 = -initial_state[0] * 0.001 * 2.0 * (step as f64) * (step as f64).sin()
                    + 0.1 * (1.0 - initial_state[0] * initial_state[0]) * initial_state[1] * DT
                    - initial_state[0] * DT;
                vec![x1, x2]
            };

            let measurement = vec![true_state[0] + 0.05, true_state[1] + 0.05];

            let result = ukf.update(&measurement, measurement_function);
            assert!(result.is_ok(), "Update failed at step {}", step);

            let state_error = ((ukf.state()[0] - true_state[0]).powi(2)
                + (ukf.state()[1] - true_state[1]).powi(2))
            .sqrt();

            if step > 50 && state_error < 0.5 {
                converged = true;
                break;
            }
        }

        assert!(converged, "UKF did not converge within {} steps", NUM_STEPS);
    }

    #[test]
    fn test_ukf_covariance_symmetry() {
        let initial_state = vec![1.0, 0.0];
        let process_noise = 0.01;
        let measurement_noise = 0.1;

        let mut ukf = UnscentedKalmanFilter::new(initial_state, process_noise, measurement_noise);

        let state_transition = |x: &[f64], _u: &[f64]| -> Vec<f64> {
            vec![x[0] + 0.1 * x[1], x[1] + 0.1 * (-x[0] + 0.1 * x[1])]
        };

        let measurement_function = |x: &[f64]| -> Vec<f64> { x.to_vec() };

        for _ in 0..20 {
            ukf.predict(state_transition, &[]);
            let measurement = vec![ukf.state()[0] + 0.1, ukf.state()[1] + 0.1];
            let _ = ukf.update(&measurement, measurement_function);

            assert!(ukf.is_symmetric(), "Covariance not symmetric");
        }
    }

    #[test]
    fn test_ukf_covariance_stability_1000_steps() {
        let initial_state = vec![1.0, 0.0];
        let initial_covariance = identity_test(2);
        let process_noise = scalar_mul_test(&identity_test(2), 0.01);
        let measurement_noise = scalar_mul_test(&identity_test(2), 0.1);

        let mut ukf = UnscentedKalmanFilter::with_covariances(
            initial_state,
            initial_covariance,
            process_noise,
            measurement_noise,
        );

        let state_transition = |x: &[f64], _u: &[f64]| -> Vec<f64> {
            vec![x[0] + 0.1 * x[1], x[1] + 0.1 * (-x[0] + 0.1 * x[1])]
        };

        let measurement_function = |x: &[f64]| -> Vec<f64> { x.to_vec() };

        let initial_trace = ukf.trace();

        for step in 0..1000 {
            ukf.predict(state_transition, &[]);

            let z = vec![ukf.state()[0] + 0.1, ukf.state()[1] + 0.1];
            let result = ukf.update(&z, measurement_function);
            assert!(result.is_ok(), "Update failed at step {}", step);

            let p = ukf.covariance();

            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        p[i][j].is_finite(),
                        "Covariance has NaN/Inf at step {}: [{}][{}] = {}",
                        step,
                        i,
                        j,
                        p[i][j]
                    );
                }
            }

            assert!(
                ukf.is_symmetric(),
                "Covariance not symmetric at step {}",
                step
            );

            if let Ok(eigenvalues) = ukf.eigenvalues() {
                for (idx, &eigenvalue) in eigenvalues.iter().enumerate() {
                    assert!(
                        eigenvalue >= -1e-6,
                        "Covariance not PSD at step {}: eigenvalue[{}] = {}",
                        step,
                        idx,
                        eigenvalue
                    );
                }
            }

            let trace = ukf.trace();
            assert!(
                trace < initial_trace * 10.0,
                "Covariance diverged at step {}: trace={}, initial_trace={}",
                step,
                trace,
                initial_trace
            );
        }
    }

    #[test]
    fn test_ukf_no_nan_inf_values() {
        let initial_state = vec![1.0, 0.0, 0.0];
        let initial_covariance = identity_test(3);
        let process_noise = scalar_mul_test(&identity_test(3), 0.01);
        let measurement_noise = scalar_mul_test(&identity_test(3), 0.1);

        let mut ukf = UnscentedKalmanFilter::with_covariances(
            initial_state,
            initial_covariance,
            process_noise,
            measurement_noise,
        );

        let state_transition = |x: &[f64], _u: &[f64]| -> Vec<f64> {
            vec![
                x[0] + 0.1 * x[1],
                x[1] + 0.1 * (-x[0] + 0.1 * x[1]),
                x[2] + 0.1 * x[2] * 0.01,
            ]
        };

        let measurement_function = |x: &[f64]| -> Vec<f64> { x.to_vec() };

        for _ in 0..500 {
            ukf.predict(state_transition, &[]);
            let z = vec![
                ukf.state()[0] + 0.1,
                ukf.state()[1] + 0.1,
                ukf.state()[2] + 0.1,
            ];
            let _ = ukf.update(&z, measurement_function);

            assert!(
                !ukf.has_nan_or_inf(),
                "Covariance contains NaN or Inf values"
            );
            assert!(
                !ukf.state().iter().any(|&x| x.is_nan() || x.is_infinite()),
                "State contains NaN or Inf values"
            );
        }
    }
}
