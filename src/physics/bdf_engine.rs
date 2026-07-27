//! BDF (Backward Differentiation Formula) Time-Stepping Engine for DAE Systems
//!
//! This module implements an implicit BDF solver suitable for stiff differential-algebraic
//! equations (DAEs) arising in building energy modeling, particularly for thermal networks
//! with algebraic constraints (e.g., zone energy balance).

use thiserror::Error;

#[cfg(feature = "std")]
use std::fmt;

#[derive(Debug, Clone, Error)]
pub enum BdfError {
    #[error("Invalid order: BDF order must be 1-6, got {0}")]
    InvalidOrder(usize),

    #[error("Convergence failed after {0} iterations: residual = {1:.2e}")]
    ConvergenceFailed(usize, f64),

    #[error("Singular matrix in Newton-Raphson solve")]
    SingularMatrix,

    #[error("Step size underflow: dt = {0:.2e} below minimum {1:.2e}")]
    StepSizeUnderflow(f64, f64),

    #[error("Step size overflow: dt = {0:.2e} above maximum {1:.2e}")]
    StepSizeOverflow(f64, f64),

    #[error("Invalid dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Numerical overflow in residual computation")]
    NumericalOverflow,
}

pub type BdfResult<T> = Result<T, BdfError>;

pub const MAX_BDF_ORDER: usize = 6;
pub const MIN_BDF_ORDER: usize = 1;

pub mod coefficients {
    //! BDF Coefficients for orders 1-6

    use crate::physics::bdf_engine::{BdfError, BdfResult};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BdfCoefficients {
        pub k: usize,
        pub alpha: &'static [f64],
        pub beta: f64,
        pub gamma: f64,
    }

    impl BdfCoefficients {
        pub fn new(k: usize) -> BdfResult<Self> {
            match k {
                1 => Ok(BDF1),
                2 => Ok(BDF2),
                3 => Ok(BDF3),
                4 => Ok(BDF4),
                5 => Ok(BDF5),
                6 => Ok(BDF6),
                _ => Err(BdfError::InvalidOrder(k)),
            }
        }
    }

    pub static BDF1: BdfCoefficients = BdfCoefficients {
        k: 1,
        alpha: &[1.0, -1.0],
        beta: 1.0,
        gamma: 1.0,
    };

    pub static BDF2: BdfCoefficients = BdfCoefficients {
        k: 2,
        alpha: &[1.0 / 2.0, -4.0 / 2.0, 3.0 / 2.0],
        beta: 2.0 / 3.0,
        gamma: 2.0 / 3.0,
    };

    pub static BDF3: BdfCoefficients = BdfCoefficients {
        k: 3,
        alpha: &[
            1.0 / 6.0,
            -18.0 / 6.0,
            36.0 / 6.0,
            -11.0 / 6.0,
        ],
        beta: 3.0 / 11.0,
        gamma: 6.0 / 11.0,
    };

    pub static BDF4: BdfCoefficients = BdfCoefficients {
        k: 4,
        alpha: &[
            1.0 / 12.0,
            -48.0 / 12.0,
            144.0 / 12.0,
            -80.0 / 12.0,
            25.0 / 12.0,
        ],
        beta: 12.0 / 25.0,
        gamma: 12.0 / 25.0,
    };

    pub static BDF5: BdfCoefficients = BdfCoefficients {
        k: 5,
        alpha: &[
            1.0 / 60.0,
            -300.0 / 60.0,
            1200.0 / 60.0,
            -900.0 / 60.0,
            400.0 / 60.0,
            -137.0 / 60.0,
        ],
        beta: 60.0 / 137.0,
        gamma: 60.0 / 137.0,
    };

    pub static BDF6: BdfCoefficients = BdfCoefficients {
        k: 6,
        alpha: &[
            1.0 / 60.0,
            -360.0 / 60.0,
            1800.0 / 60.0,
            -1680.0 / 60.0,
            1050.0 / 60.0,
            -480.0 / 60.0,
            147.0 / 60.0,
        ],
        beta: 60.0 / 147.0,
        gamma: 60.0 / 147.0,
    };

    pub static ALL_BDF_COEFFICIENTS: [&BdfCoefficients; 6] = [
        &BDF1, &BDF2, &BDF3, &BDF4, &BDF5, &BDF6,
    ];

    pub fn get_coefficients(order: usize) -> BdfResult<&'static BdfCoefficients> {
        if order < 1 || order > 6 {
            return Err(BdfError::InvalidOrder(order));
        }
        Ok(ALL_BDF_COEFFICIENTS[order - 1])
    }

    pub fn compute_bdf_coefficients(order: usize) -> BdfResult<(Vec<f64>, f64)> {
        let coeff = get_coefficients(order)?;
        Ok((coeff.alpha.to_vec(), coeff.beta))
    }
}

pub mod newton_raphson {
    //! Newton-Raphson Nonlinear Solver

    use crate::physics::bdf_engine::{BdfError, BdfResult};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NewtonRaphsonConfig {
        pub max_iterations: usize,
        pub residual_tolerance: f64,
        pub update_tolerance: f64,
        pub damping_factor: f64,
    }

    impl Default for NewtonRaphsonConfig {
        fn default() -> Self {
            Self {
                max_iterations: 50,
                residual_tolerance: 1e-8,
                update_tolerance: 1e-10,
                damping_factor: 1.0,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NewtonRaphsonStats {
        pub iterations: usize,
        pub final_residual: f64,
        pub converged: bool,
    }

    pub trait ResidualFunction<T> {
        fn eval(&self, x: &[T], residual: &mut [T]);
    }

    pub struct NewtonRaphsonSolver {
        config: NewtonRaphsonConfig,
    }

    impl NewtonRaphsonSolver {
        pub fn new(config: NewtonRaphsonConfig) -> Self {
            Self { config }
        }

        pub fn with_default_config() -> Self {
            Self::new(NewtonRaphsonConfig::default())
        }

        pub fn solve<R>(
            &self,
            x0: &[f64],
            func: &R,
        ) -> BdfResult<(Vec<f64>, NewtonRaphsonStats)>
        where
            R: ResidualFunction<f64>,
        {
            let n = x0.len();
            let mut x = x0.to_vec();
            let mut residual = vec![0.0; n];
            let mut update = vec![0.0; n];
            let mut jacobian = vec![0.0; n * n];

            for iter in 0..self.config.max_iterations {
                func.eval(&x, &mut residual);

                let residual_norm = norm(&residual);

                if residual_norm < self.config.residual_tolerance {
                    return Ok((
                        x.clone(),
                        NewtonRaphsonStats {
                            iterations: iter + 1,
                            final_residual: residual_norm,
                            converged: true,
                        },
                    ));
                }

                compute_numerical_jacobian(&x, &residual, func, &mut jacobian);
                solve_linear_system(&jacobian, &residual, &mut update);

                let update_norm = norm(&update);

                if update_norm < self.config.update_tolerance {
                    return Ok((
                        x.clone(),
                        NewtonRaphsonStats {
                            iterations: iter + 1,
                            final_residual: residual_norm,
                            converged: true,
                        },
                    ));
                }

                for i in 0..n {
                    x[i] -= self.config.damping_factor * update[i];
                }
            }

            func.eval(&x, &mut residual);
            let final_residual = norm(&residual);

            Err(BdfError::ConvergenceFailed(
                self.config.max_iterations,
                final_residual,
            ))
        }
    }

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn compute_numerical_jacobian<R>(
        x: &[f64],
        residual: &[f64],
        func: &R,
        jac: &mut [f64],
    ) where
        R: ResidualFunction<f64>,
    {
        let n = x.len();
        let eps = 1e-8;

        for j in 0..n {
            let mut x_perturbed = x.to_vec();
            x_perturbed[j] += eps;
            let mut residual_perturbed = vec![0.0; n];
            func.eval(&x_perturbed, &mut residual_perturbed);

            let dx = eps;
            for i in 0..n {
                jac[i * n + j] = (residual_perturbed[i] - residual[i]) / dx;
            }
        }
    }

    fn solve_linear_system(jac: &[f64], rhs: &[f64], solution: &mut [f64]) {
        let n = rhs.len();
        solution.copy_from_slice(rhs);

        let mut aug = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = jac[i * n + j];
            }
            aug[i * (n + 1) + n] = rhs[i];
        }

        for k in 0..n {
            let mut max_val = aug[k * (n + 1) + k].abs();
            let mut max_idx = k;
            for i in (k + 1)..n {
                let val = aug[i * (n + 1) + k].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            if max_val < 1e-14 {
                continue;
            }

            if max_idx != k {
                for j in 0..=n {
                    aug.swap(k * (n + 1) + j, max_idx * (n + 1) + j);
                }
            }

            let pivot = aug[k * (n + 1) + k];
            for j in k..=n {
                aug[k * (n + 1) + j] /= pivot;
            }

            for i in 0..n {
                if i != k {
                    let factor = aug[i * (n + 1) + k];
                    for j in k..=n {
                        aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
                    }
                }
            }
        }

        for i in 0..n {
            solution[i] = aug[i * (n + 1) + n];
        }
    }

    pub struct SimpleResidual {
        pub n: usize,
    }

    impl ResidualFunction<f64> for SimpleResidual {
        fn eval(&self, x: &[f64], residual: &mut [f64]) {
            for i in 0..self.n {
                residual[i] = x[i] * x[i] - 1.0;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_newton_raphson_simple() {
            let solver = NewtonRaphsonSolver::with_default_config();
            let func = SimpleResidual { n: 1 };

            let result = solver.solve(&[2.0], &func);
            assert!(result.is_ok());

            let (solution, stats) = result.unwrap();
            assert!(stats.converged);
            assert!((solution[0] - 1.0).abs() < 1e-6);
        }

        #[test]
        fn test_newton_raphson_multi_dim() {
            let solver = NewtonRaphsonSolver::with_default_config();
            let func = SimpleResidual { n: 2 };

            let result = solver.solve(&[2.0, 2.0], &func);
            assert!(result.is_ok());

            let (solution, stats) = result.unwrap();
            assert!(stats.converged);
            assert!((solution[0] - 1.0).abs() < 1e-6);
            assert!((solution[1] - 1.0).abs() < 1e-6);
        }
    }
}

pub mod adaptive_step {
    //! Adaptive Step Size Controller for BDF

    use crate::physics::bdf_engine::{BdfError, BdfResult};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct AdaptiveStepConfig {
        pub min_dt: f64,
        pub max_dt: f64,
        pub initial_dt: f64,
        pub safety_factor: f64,
        pub min_order: usize,
        pub max_order: usize,
    }

    impl Default for AdaptiveStepConfig {
        fn default() -> Self {
            Self {
                min_dt: 1e-6,
                max_dt: 3600.0,
                initial_dt: 60.0,
                safety_factor: 0.9,
                min_order: 1,
                max_order: 6,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct StepSize {
        pub dt: f64,
        pub order: usize,
    }

    pub struct AdaptiveStepController {
        config: AdaptiveStepConfig,
    }

    impl AdaptiveStepController {
        pub fn new(config: AdaptiveStepConfig) -> Self {
            Self { config }
        }

        pub fn with_default_config() -> Self {
            Self::new(AdaptiveStepConfig::default())
        }

        pub fn compute_next_step(
            &self,
            current: StepSize,
            error_estimate: f64,
            tolerance: f64,
        ) -> BdfResult<StepSize> {
            let new_dt =
                self.compute_step_size(current.dt, error_estimate, tolerance, current.order);

            let new_dt = new_dt.clamp(self.config.min_dt, self.config.max_dt);

            if new_dt < self.config.min_dt {
                return Err(BdfError::StepSizeUnderflow(new_dt, self.config.min_dt));
            }
            if new_dt > self.config.max_dt {
                return Err(BdfError::StepSizeOverflow(new_dt, self.config.max_dt));
            }

            Ok(StepSize {
                dt: new_dt,
                order: current.order,
            })
        }

        fn compute_step_size(&self, dt: f64, error: f64, tolerance: f64, order: usize) -> f64 {
            if error < tolerance {
                return dt * 2.0;
            }

            let exponent = 1.0 / (order as f64 + 1.0);
            let factor = (tolerance / error).powf(exponent) * self.config.safety_factor;

            dt * factor.clamp(0.1, 2.0)
        }

        pub fn suggest_order(&self, current: StepSize, errors: &[f64]) -> usize {
            if errors.is_empty() {
                return current.order;
            }

            let min_err_idx = errors
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let suggested_order =
                (self.config.min_order + min_err_idx).min(self.config.max_order);
            suggested_order.max(self.config.min_order)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_adaptive_step_initial() {
            let controller = AdaptiveStepController::with_default_config();
            let step = StepSize { dt: 60.0, order: 3 };

            let next = controller.compute_next_step(step, 1e-4, 1e-6);
            assert!(next.is_ok());
        }

        #[test]
        fn test_step_size_growth_on_success() {
            let controller = AdaptiveStepController::with_default_config();
            let step = StepSize { dt: 60.0, order: 1 };

            let next = controller.compute_next_step(step, 1e-10, 1e-6);
            assert!(next.is_ok());
            let next_step = next.unwrap();
            assert!(next_step.dt > step.dt);
        }
    }
}

pub mod time_stepping {
    //! BDF Time-Stepping Engine

    use crate::physics::bdf_engine::{
        adaptive_step::AdaptiveStepConfig,
        coefficients::{get_coefficients, BdfCoefficients},
        newton_raphson::{NewtonRaphsonConfig, NewtonRaphsonSolver, ResidualFunction},
        BdfError, BdfResult, MAX_BDF_ORDER,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TimeSteppingConfig {
        pub bdf_config: NewtonRaphsonConfig,
        pub step_config: AdaptiveStepConfig,
        pub max_steps: usize,
        pub tolerance: f64,
    }

    impl Default for TimeSteppingConfig {
        fn default() -> Self {
            Self {
                bdf_config: NewtonRaphsonConfig::default(),
                step_config: AdaptiveStepConfig::default(),
                max_steps: 100000,
                tolerance: 1e-6,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TimeSteppingStats {
        pub steps_taken: usize,
        pub steps_rejected: usize,
        pub final_time: f64,
        pub converged: bool,
    }

    pub trait DaeSystem<T> {
        fn residual(&self, t: T, y: &[T], yp: &[T], r: &mut [T]);
        fn dimension(&self) -> usize;
    }

    #[allow(dead_code)]
    pub struct BdfTimeStepper {
        config: TimeSteppingConfig,
        history: Vec<Vec<f64>>,
        time_history: Vec<f64>,
        current_order: usize,
    }

    impl BdfTimeStepper {
        pub fn new(config: TimeSteppingConfig) -> Self {
            Self {
                config,
                history: Vec::new(),
                time_history: Vec::new(),
                current_order: 1,
            }
        }

        pub fn with_default_config() -> Self {
            Self::new(TimeSteppingConfig::default())
        }

        pub fn initialize(&mut self, t0: f64, y0: &[f64]) -> BdfResult<()> {
            let n = y0.len();
            self.history.clear();
            self.time_history.clear();

            for _ in 0..MAX_BDF_ORDER {
                self.history.push(vec![0.0; n]);
                self.time_history.push(0.0);
            }

            self.history[0].copy_from_slice(y0);
            self.time_history[0] = t0;

            self.current_order = 1;
            Ok(())
        }

        pub fn step<S>(&mut self, dt: f64, system: &S) -> BdfResult<(Vec<f64>, TimeSteppingStats)>
        where
            S: DaeSystem<f64>,
        {
            let n = system.dimension();
            let t_new = self.time_history[0] + dt;

            let coeff = get_coefficients(self.current_order)?;

            let mut y_new = vec![0.0; n];
            self.predict(&mut y_new);

            let solver = NewtonRaphsonSolver::with_default_config();
            let bdf_residual = BdfResidual {
                system,
                dt,
                coeff,
                t_new,
                history: &self.history,
                n,
            };

            let (solution, nr_stats) = solver.solve(&y_new, &bdf_residual)?;

            if !nr_stats.converged {
                return Err(BdfError::ConvergenceFailed(
                    nr_stats.iterations,
                    nr_stats.final_residual,
                ));
            }

            self.correct(&solution, t_new);

            y_new[..n].copy_from_slice(&solution[..n]);

            Ok((
                y_new,
                TimeSteppingStats {
                    steps_taken: 1,
                    steps_rejected: 0,
                    final_time: t_new,
                    converged: true,
                },
            ))
        }

        fn predict(&self, y_pred: &mut [f64]) {
            if self.history.is_empty() {
                return;
            }
            if self.history.len() < 2 {
                if let Some(h) = self.history.first() {
                    y_pred.copy_from_slice(h);
                }
                return;
            }

            let k = self.current_order;
            let coeff = get_coefficients(k).expect("Invalid BDF order");

            for (i, pred_item) in y_pred.iter_mut().enumerate() {
                let mut pred = 0.0;
                for j in 0..k {
                    pred += coeff.alpha[j] * self.history[j][i];
                }
                *pred_item = -pred;
            }
        }

        fn correct(&mut self, y_new: &[f64], t_new: f64) {
            let n = self.history.len();
            for j in (1..n).rev() {
                let src = self.history[j - 1].clone();
                self.history[j].copy_from_slice(&src);
            }
            self.history[0].copy_from_slice(y_new);

            let m = self.time_history.len();
            for j in (1..m).rev() {
                self.time_history[j] = self.time_history[j - 1];
            }
            self.time_history[0] = t_new;
        }
    }

    struct BdfResidual<'a, S> {
        system: &'a S,
        dt: f64,
        coeff: &'static BdfCoefficients,
        t_new: f64,
        history: &'a [Vec<f64>],
        n: usize,
    }

    impl<'a, S> ResidualFunction<f64> for BdfResidual<'a, S>
    where
        S: DaeSystem<f64>,
    {
        fn eval(&self, y: &[f64], residual: &mut [f64]) {
            let mut yp = vec![0.0; self.n];

            let mut sum = 0.0;
            for j in 0..self.coeff.k {
                for (i, yp_item) in yp.iter_mut().enumerate().take(self.n) {
                    *yp_item += self.coeff.alpha[j] * self.history[j][i];
                }
                sum += self.coeff.alpha[j];
            }

            for (i, yp_item) in yp.iter_mut().enumerate().take(self.n) {
                *yp_item += self.coeff.alpha[self.coeff.k] * y[i];
                *yp_item /= self.dt * self.coeff.beta;
            }

            self.system
                .residual(self.t_new, y, &yp, residual);

            for (i, res_item) in residual.iter_mut().enumerate().take(self.n) {
                *res_item = sum * y[i] / (self.dt * self.coeff.beta);
            }
        }
    }

    #[cfg(test)]
    struct SimpleOscillator;

    #[cfg(test)]
    impl DaeSystem<f64> for SimpleOscillator {
        fn residual(&self, _t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
            r[0] = yp[0] - y[1];
            r[1] = yp[1] + y[0];
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bdf_time_stepper_init() {
            let mut stepper = BdfTimeStepper::with_default_config();
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();
        }

        #[test]
        fn test_bdf_first_order() {
            let mut stepper = BdfTimeStepper::with_default_config();
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();

            let system = SimpleOscillator;
            let result = stepper.step(0.01, &system);
            assert!(result.is_ok());
        }
    }
}

pub use coefficients::{get_coefficients, BdfCoefficients, BDF1, BDF2, BDF3, BDF4, BDF5, BDF6};
pub use newton_raphson::{NewtonRaphsonConfig, NewtonRaphsonSolver, NewtonRaphsonStats, ResidualFunction};
pub use adaptive_step::{AdaptiveStepConfig, AdaptiveStepController, StepSize};
pub use time_stepping::{DaeSystem, BdfTimeStepper, TimeSteppingConfig, TimeSteppingStats};
