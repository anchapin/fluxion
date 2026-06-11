//! Physics-step dispatcher for `ThermalModel`.
//!
//! Hosts [`ThermalModel::step_physics`], the dispatcher that routes to
//! the correct 5R1C/6R2C/8R3C/9R4C implementation based on the model's
//! configured network type. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Solve physics for one timestep (assumes loads already set).
    ///
    /// This method performs only the physics calculation portion of solve_single_step,
    /// assuming that loads have already been set via set_loads() or calculated externally.
    /// This enables batched inference: collect all temperatures, run one batched prediction,
    /// distribute loads, then call this method in parallel.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (used for ground temperature)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `dt_seconds` - Timestep duration in seconds (default: 3600.0 for 1-hour timestep)
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    ///
    /// Issue #351: Calculate solar gains internally if weather data is available
    pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        // Record call for wiring validation (Plan 21-10)
        #[cfg(feature = "wiring-tracing")]
        if let Some(ref tracer) = self.0.tracer {
            tracer.record_call("step_physics");
        }

        // Issue #351: Calculate loads from weather data if not already set
        // This is needed for ASHRAE 140 validation where step_physics is called directly
        if self.0.weather.is_some() {
            self.calc_analytical_loads(timestep, true);
        }

        // Branch based on thermal model type
        if self.is_nine_r4c_model() {
            self.step_physics_9r4c(timestep, outdoor_temp, dt_seconds)
        } else if self.is_8r3c_model() {
            self.step_physics_8r3c(timestep, outdoor_temp, dt_seconds)
        } else if self.is_6r2c_model() {
            self.step_physics_6r2c(timestep, outdoor_temp, dt_seconds)
        } else {
            self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds)
        }
    }
}
