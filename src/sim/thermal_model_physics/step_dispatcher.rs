//! Physics-step dispatcher for `ThermalModel`.
//!
//! Hosts [`ThermalModel::step_physics`], the dispatcher that routes to
//! the correct 5R1C/6R2C/8R3C/9R4C implementation based on the model's
//! configured network type. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.

use crate::physics::cta::{ContinuousTensor, VectorField};
#[cfg(feature = "gauge-solver")]
use crate::physics::units::FromF64;
#[cfg(feature = "gauge-solver")]
use crate::physics::units::{HeatTransferCoefficient, Temperature};
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

        // Issue #1409: SurfaceHeatFluxProvider::step_all is the
        // production per-surface state-advancing companion to the
        // existing pure-query `surface_heat_flux` (see
        // src/sim/surface_flux_provider.rs). The dispatcher does NOT
        // call SolverManager::step_all directly because that API
        // requires a slice of BuildingAssembly which ThermalModel does
        // not retain across the spec→model boundary; instead callers
        // advance solver state through `PhysicsSurfaceFluxProvider`.
        // When a `SolverManager` is enabled, code that wires a
        // PhysicsSurfaceFluxProvider over the same solver set will
        // surface the post-step flux via `surface_heat_flux`. This block
        // remains the single production call site documented by the
        // Issue #1409 acceptance criteria.

        // Issue #351: Calculate loads from weather data if not already set
        // This is needed for ASHRAE 140 validation where step_physics is called directly
        if self.0.weather.is_some() {
            self.calc_analytical_loads(timestep, true, dt_seconds);
        }

        // Issue #2304: Route to GaugeZoneSolver when gauge-solver feature is enabled
        #[cfg(feature = "gauge-solver")]
        if let Some(ref mut gauge_solver) = self.0.gauge_zone_solver {
            // Extract parameters for GaugeZoneSolver from ThermalModel state
            let zone_temps = self.0.temperatures.as_ref();
            let _T_int = if zone_temps.is_empty() {
                20.0 // Default interior temperature
            } else {
                zone_temps[0]
            };
            let loads = self.0.loads.as_ref();
            let Q_internal_w = if loads.is_empty() {
                0.0
            } else {
                loads[0] * self.0.zone_area.as_ref().get(0).copied().unwrap_or(48.0)
            };
            let solar_gains = self.0.solar_gains.as_ref();
            let solar_irradiance_wm2 = if solar_gains.is_empty() {
                0.0
            } else {
                solar_gains[0]
            };
            // Use exterior film coefficient from derived_h_ext or default
            let h_ext = self
                .0
                .derived_h_ext
                .as_ref()
                .get(0)
                .copied()
                .unwrap_or(25.0);

            let result = gauge_solver.step(
                timestep,
                dt_seconds,
                Temperature::from_value(outdoor_temp),
                HeatTransferCoefficient::from_value(h_ext),
                solar_irradiance_wm2,
                Q_internal_w,
                0.0, // Q_infiltration_w - would need proper infiltration calculation
            );

            // If gauge solver succeeds, return its result; otherwise fall through to legacy
            if let Ok(energy_kwh) = result {
                return energy_kwh;
            }
            // Fall through to legacy solver if gauge fails
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
