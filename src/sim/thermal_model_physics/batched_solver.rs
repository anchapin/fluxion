//! Batched coordinator–worker timestep solver.
//!
//! Hosts [`ThermalModel::solve_timesteps_batched`], the worker side of
//! the coordinator–worker pattern used for parallel annual energy
//! simulation. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.

use crossbeam::channel::{Receiver, Sender};

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::{get_daily_cycle, ThermalModel};

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Core physics simulation loop for annual building energy performance.
    ///
    /// Simulates hourly thermal dynamics using batched inference with a coordinator.
    ///
    /// This method implements the worker side of the coordinator-worker pattern.
    /// At each timestep, it sends its current temperature state to the coordinator,
    /// waits for the predicted loads, and then completes the physics calculation.
    pub fn solve_timesteps_batched(
        &mut self,
        steps: usize,
        tx: Sender<Vec<f64>>,
        rx: Receiver<Vec<f64>>,
    ) -> f64 {
        let cycle = get_daily_cycle();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                let hour_of_day = t % 24;
                let daily_cycle = cycle[hour_of_day];
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;

                // 1. Send current state to coordinator
                let temps = self.get_temperatures();
                tx.send(temps).expect("Failed to send state to coordinator");

                // 2. Receive predicted loads from coordinator
                let loads = rx.recv().expect("Failed to receive loads from coordinator");
                self.set_loads(&loads);

                // 3. Solve physics for this timestep
                self.step_physics(t, outdoor_temp, 3600.0)
            })
            .sum();

        // Normalize by total floor area to get EUI
        let total_area = self.0.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
    }
}
