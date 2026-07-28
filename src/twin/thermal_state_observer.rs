//! ThermalStateObserver: UKF-based thermal state estimation with 9R4C model.
//!
//! Issue #2059: [Twin] ThermalStateObserver Core + 9R4C Integration
//!
//! The observer estimates unmeasurable wall temperatures using only air temperature
//! measurements, using an Unscented Kalman Filter (UKF) with the 9R4C thermal model.

use crate::physics::multi_node_solver::MultiNodeSolver;
use crate::twin::ukf::{ThermalMeasurementVector, ThermalStateVector, UnscentedKalmanFilter};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ThermalState {
    pub t_air: f64,
    pub t_mass: f64,
    pub t_surface: f64,
}

impl From<&ThermalStateVector> for ThermalState {
    fn from(s: &ThermalStateVector) -> Self {
        ThermalState {
            t_air: s.t_air(),
            t_mass: s.t_mass(),
            t_surface: s.t_surface(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThermalMeasurement {
    pub t_air_measured: f64,
}

impl From<&ThermalMeasurement> for ThermalMeasurementVector {
    fn from(m: &ThermalMeasurement) -> Self {
        ThermalMeasurementVector::new(m.t_air_measured)
    }
}

pub struct ThermalStateObserver {
    ukf: UnscentedKalmanFilter,
    thermal_model: MultiNodeSolver,
}

impl ThermalStateObserver {
    pub fn new(thermal_model: MultiNodeSolver, process_noise: f64, measurement_noise: f64) -> Self {
        let initial_state = ThermalStateVector::new(20.0, 18.0, 19.0);
        let ukf = UnscentedKalmanFilter::new(initial_state, process_noise, measurement_noise);
        Self { ukf, thermal_model }
    }

    pub fn with_initial_state(
        thermal_model: MultiNodeSolver,
        initial_state: ThermalState,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        let initial = ThermalStateVector::new(
            initial_state.t_air,
            initial_state.t_mass,
            initial_state.t_surface,
        );
        let ukf = UnscentedKalmanFilter::new(initial, process_noise, measurement_noise);
        Self { ukf, thermal_model }
    }

    pub fn predict(&mut self, dt: Duration) {
        let dt_secs = dt.as_secs_f64();
        let model = self.thermal_model.clone();

        let state_transition = |state: &ThermalStateVector, _dt: f64| -> ThermalStateVector {
            let s = ThermalState::from(state);
            let mut m = model.clone();
            m.zone_temperature = s.t_air;
            m.surface_temperature = s.t_surface;
            m.mass.wall.temperature = s.t_mass;
            m.mass.roof.temperature = s.t_mass;
            m.mass.floor.temperature = s.t_mass;
            m.mass.internal.temperature = s.t_mass;
            m.step(dt_secs);
            let t_air = m.zone_temperature;
            let t_surface = m.surface_temperature;
            let t_mass_avg = (m.mass.wall.temperature
                + m.mass.roof.temperature
                + m.mass.floor.temperature
                + m.mass.internal.temperature)
                / 4.0;
            ThermalStateVector::new(t_air, t_mass_avg, t_surface)
        };

        self.ukf.predict(&state_transition, &());
    }

    pub fn update(&mut self, measurement: &ThermalMeasurement) {
        let measurement_fn = |state: &ThermalStateVector| -> ThermalMeasurementVector {
            ThermalMeasurementVector::new(state.t_air())
        };
        let meas_vec: ThermalMeasurementVector = measurement.into();
        self.ukf.update(&meas_vec, &measurement_fn);
    }

    pub fn estimate(&self) -> ThermalState {
        ThermalState::from(self.ukf.state())
    }

    pub fn covariance(&self) -> &nalgebra::Matrix3<f64> {
        self.ukf.covariance()
    }

    pub fn thermal_model(&self) -> &MultiNodeSolver {
        &self.thermal_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxion_core::multi_node::{MultiNodeThermalMass, ThermalMassNode};

    fn create_test_solver() -> MultiNodeSolver {
        let wall = ThermalMassNode::new(15.0, 50000.0, 5.0, 2.0);
        let roof = ThermalMassNode::new(15.0, 50000.0, 5.0, 2.0);
        let floor = ThermalMassNode::new(15.0, 50000.0, 5.0, 2.0);
        let internal = ThermalMassNode::new(15.0, 50000.0, 5.0, 2.0);
        MultiNodeSolver::new(10.0, wall, roof, floor, internal)
    }

    #[test]
    fn test_observer_creation() {
        let solver = create_test_solver();
        let observer = ThermalStateObserver::new(solver, 0.1, 0.1);
        let estimate = observer.estimate();
        assert!((estimate.t_air - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_observer_predict() {
        let solver = create_test_solver();
        let mut observer = ThermalStateObserver::new(solver, 0.1, 0.1);
        observer.predict(Duration::from_secs(3600));
        let estimate = observer.estimate();
        assert!(estimate.t_air.is_finite());
        assert!(estimate.t_mass.is_finite());
        assert!(estimate.t_surface.is_finite());
    }

    #[test]
    fn test_observer_update() {
        let solver = create_test_solver();
        let mut observer = ThermalStateObserver::new(solver, 0.1, 0.1);
        let measurement = ThermalMeasurement {
            t_air_measured: 22.0,
        };
        observer.update(&measurement);
        let estimate = observer.estimate();
        assert!(
            estimate.t_air.is_finite(),
            "t_air should be finite after update"
        );
        assert!(
            estimate.t_mass.is_finite(),
            "t_mass should be finite after update"
        );
        assert!(
            estimate.t_surface.is_finite(),
            "t_surface should be finite after update"
        );
    }

    #[test]
    fn test_observer_predict_then_update() {
        let solver = create_test_solver();
        let mut observer = ThermalStateObserver::new(solver, 0.1, 0.1);
        observer.predict(Duration::from_secs(3600));
        let measurement = ThermalMeasurement {
            t_air_measured: 23.0,
        };
        observer.update(&measurement);
        let estimate = observer.estimate();
        assert!(estimate.t_air.is_finite());
        assert!(estimate.t_mass.is_finite());
        assert!(estimate.t_surface.is_finite());
    }

    #[test]
    fn test_observer_with_custom_initial_state() {
        let solver = create_test_solver();
        let initial = ThermalState {
            t_air: 25.0,
            t_mass: 20.0,
            t_surface: 22.0,
        };
        let observer = ThermalStateObserver::with_initial_state(solver, initial, 0.1, 0.1);
        let estimate = observer.estimate();
        assert!((estimate.t_air - 25.0).abs() < 0.1);
        assert!((estimate.t_mass - 20.0).abs() < 0.1);
        assert!((estimate.t_surface - 22.0).abs() < 0.1);
    }
}
