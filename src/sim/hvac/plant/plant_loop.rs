//! Plant-loop sequential-iterative solver.
//!
//! Implements a simplified version of the EnergyPlus plant loop algorithm:
//!
//! 1. **Demand side** — sequential component evaluation (pumps → use
//!    devices).
//! 2. **Supply side** — sequential component evaluation (plant equipment).
//! 3. **Convergence check** — repeat until supply outlet temperatures
//!    stabilize within [`PlantLoop::tolerance_c`].
//!
//! This approach correctly handles the algebraic coupling between supply
//! and demand temperatures without requiring a full nonlinear solver.

use serde::{Deserialize, Serialize};

use super::plant_component::{FluidState, PlantComponent};

/// Result of a single [`PlantLoop::solve`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantLoopResult {
    /// Temperature at the loop supply header [°C].
    pub supply_header_temperature: f64,
    /// Temperature at the loop demand header [°C].
    pub demand_header_temperature: f64,
    /// Net electrical power consumed by all loop components [W].
    pub total_electrical_power_w: f64,
    /// Net heat transfer rate into the working fluid [W].
    /// Positive for heating loops, negative for cooling loops.
    pub total_heat_transfer_w: f64,
    /// Number of iterations the solver required.
    pub iterations: u32,
    /// Whether the solver converged within the iteration limit.
    pub converged: bool,
}

/// A closed-loop plant system (chilled-water, hot-water, or
/// condenser-water loop).
///
/// # Topology
///
/// ```text
///  [Supply Header] ← supply-side equipment (chiller/boiler) ←
///       ↕ (pipe)
///  [Demand Header] → demand-side equipment (pumps, coils, HXs) →
/// ```
///
/// The loop does not own its components; it borrows them via trait objects.
/// The caller is responsible for keeping them alive and updating mutable
/// state between timesteps.
#[derive(Debug)]
pub struct PlantLoop {
    /// Human-readable loop name.
    pub id: String,
    /// Loop supply temperature setpoint [°C].
    pub supply_setpoint_c: f64,
    /// Convergence tolerance on header temperatures [°C].
    pub tolerance_c: f64,
    /// Maximum iterations per solve.
    pub max_iterations: u32,
}

impl PlantLoop {
    /// Create a new plant loop.
    pub fn new(id: String, supply_setpoint_c: f64) -> Self {
        Self {
            id,
            supply_setpoint_c,
            tolerance_c: 0.05, // 50 mK convergence
            max_iterations: 50,
        }
    }

    /// Solve the plant loop for one timestep.
    ///
    /// # Arguments
    /// * `supply_components` — equipment on the supply (plant) side,
    ///   ordered from the return header toward the supply header (i.e.
    ///   chillers / boilers).
    /// * `demand_components` — equipment on the demand (load) side,
    ///   ordered from the supply header toward the return header (i.e.
    ///   pumps, coils, heat exchangers).
    /// * `supply_return_inlet` — fluid state returning from the demand
    ///   side to the supply side.
    /// * `outdoor_temp` — outdoor dry-bulb temperature [°C] for cooling
    ///   towers / air-cooled equipment.
    /// * `dt` — timestep length [s].
    pub fn solve(
        &self,
        supply_components: &[&dyn PlantComponent],
        demand_components: &[&dyn PlantComponent],
        supply_return_inlet: FluidState,
        outdoor_temp: f64,
        dt: f64,
    ) -> PlantLoopResult {
        let mut total_electrical_w = 0.0;
        let mut total_heat_w = 0.0;
        let mut supply_header_temp = self.supply_setpoint_c;
        let mut demand_header_temp = supply_return_inlet.temperature;

        let mut converged = false;
        let mut iterations = 0u32;

        for iter in 0..self.max_iterations {
            iterations = iter + 1;

            // --- Demand side ---
            // Demand side flows from supply_header → demand_header
            let mut demand_inlet = FluidState {
                temperature: supply_header_temp,
                flow_rate: supply_return_inlet.flow_rate,
            };
            for comp in demand_components {
                let result = comp.evaluate(demand_inlet, outdoor_temp, dt);
                demand_inlet = result.outlet;
                total_electrical_w += result.electrical_power_w;
                total_heat_w += result.heat_transfer_w;
            }
            let new_demand_header_temp = demand_inlet.temperature;

            // --- Supply side ---
            // Supply side flows from demand_header → supply_header
            let mut supply_inlet = FluidState {
                temperature: new_demand_header_temp,
                flow_rate: supply_return_inlet.flow_rate,
            };
            for comp in supply_components {
                let result = comp.evaluate(supply_inlet, outdoor_temp, dt);
                supply_inlet = result.outlet;
                total_electrical_w += result.electrical_power_w;
                total_heat_w += result.heat_transfer_w;
            }
            let new_supply_header_temp = supply_inlet.temperature;

            // --- Convergence check ---
            let supply_delta = (new_supply_header_temp - supply_header_temp).abs();
            let demand_delta = (new_demand_header_temp - demand_header_temp).abs();

            supply_header_temp = new_supply_header_temp;
            demand_header_temp = new_demand_header_temp;

            if supply_delta < self.tolerance_c && demand_delta < self.tolerance_c {
                converged = true;
                break;
            }
        }

        PlantLoopResult {
            supply_header_temperature: supply_header_temp,
            demand_header_temperature: demand_header_temp,
            total_electrical_power_w: total_electrical_w,
            total_heat_transfer_w: total_heat_w,
            iterations,
            converged,
        }
    }
}

/// Check energy balance for a loop result.
///
/// Returns `Ok(())` if the net heat transfer plus electrical power is
/// within the specified tolerance of the expected balance, or
/// `Err(message)` otherwise.
pub fn check_energy_balance(
    result: &PlantLoopResult,
    expected_balance_w: f64,
    tolerance_w: f64,
) -> Result<(), String> {
    let actual = result.total_heat_transfer_w + result.total_electrical_power_w;
    let delta = (actual - expected_balance_w).abs();
    if delta > tolerance_w {
        Err(format!(
            "Energy balance violated: |{actual:.1} - {expected_balance_w:.1}| = {delta:.1} W > {tolerance_w:.1} W tolerance"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::plant_component::PlantComponentResult;
    use super::super::pump::PumpConstantSpeed;
    use super::*;

    /// A simple heat source for testing.
    struct ConstantHeatSource {
        heat_w: f64,
    }

    impl PlantComponent for ConstantHeatSource {
        fn id(&self) -> &str {
            "HeatSource"
        }
        fn evaluate(
            &self,
            inlet: FluidState,
            _outdoor_temp: f64,
            _dt: f64,
        ) -> PlantComponentResult {
            let rho = super::super::fluid_properties::water_density(inlet.temperature);
            let cp = super::super::fluid_properties::water_cp(inlet.temperature);
            let mass_flow = inlet.flow_rate * rho;
            let dt_rise = if mass_flow > 0.0 {
                self.heat_w / (mass_flow * cp)
            } else {
                0.0
            };
            PlantComponentResult {
                outlet: FluidState {
                    temperature: inlet.temperature + dt_rise,
                    flow_rate: inlet.flow_rate,
                },
                electrical_power_w: 0.0,
                heat_transfer_w: self.heat_w,
            }
        }
    }

    /// A simple heat sink for testing (e.g., cooling coil).
    struct ConstantHeatSink {
        heat_w: f64,
    }

    impl PlantComponent for ConstantHeatSink {
        fn id(&self) -> &str {
            "HeatSink"
        }
        fn evaluate(
            &self,
            inlet: FluidState,
            _outdoor_temp: f64,
            _dt: f64,
        ) -> PlantComponentResult {
            let rho = super::super::fluid_properties::water_density(inlet.temperature);
            let cp = super::super::fluid_properties::water_cp(inlet.temperature);
            let mass_flow = inlet.flow_rate * rho;
            let dt_drop = if mass_flow > 0.0 {
                self.heat_w / (mass_flow * cp)
            } else {
                0.0
            };
            PlantComponentResult {
                outlet: FluidState {
                    temperature: (inlet.temperature - dt_drop).max(5.0),
                    flow_rate: inlet.flow_rate,
                },
                electrical_power_w: 0.0,
                heat_transfer_w: -self.heat_w,
            }
        }
    }

    #[test]
    fn test_empty_loop_converges() {
        // Initial supply_header = setpoint (7.0), demand_header = inlet (12.0).
        // On first iter the demand side has no components, so new_demand = 7.0.
        // Convergence check sees |7-12| > tol → iter 2 where both are 7.0.
        let loop_ = PlantLoop::new("TestLoop".to_string(), 7.0);
        let inlet = FluidState {
            temperature: 12.0,
            flow_rate: 0.01,
        };
        let result = loop_.solve(&[], &[], inlet, 25.0, 3600.0);
        assert!(result.converged);
        assert!(result.iterations <= 2);
    }

    #[test]
    fn test_boiler_loop_converges() {
        let loop_ = PlantLoop::new("HotWaterLoop".to_string(), 70.0);
        let boiler = ConstantHeatSource { heat_w: 50_000.0 };
        let pump = PumpConstantSpeed::new(
            "Pump-1".to_string(),
            0.002, // 2 L/s
            15.0,
            0.70,
            0.90,
        );
        // Balanced load: heat sink on demand side removes exactly what
        // the boiler adds, allowing the solver to find equilibrium.
        let sink = ConstantHeatSink { heat_w: 50_000.0 };

        let inlet = FluidState {
            temperature: 50.0,
            flow_rate: 0.002,
        };
        let result = loop_.solve(&[&boiler], &[&pump, &sink], inlet, 25.0, 3600.0);
        assert!(
            result.converged,
            "solver did not converge in {} iterations",
            result.iterations
        );
        assert!(result.total_electrical_power_w > 0.0);
        // Balanced source/sink means net heat transfer is zero, but water
        // was heated above inlet by the boiler.
        assert!(
            result.supply_header_temperature > inlet.temperature,
            "supply temp {} should be above inlet {}",
            result.supply_header_temperature,
            inlet.temperature
        );
    }

    #[test]
    fn test_chiller_loop_converges() {
        // Simulates a chilled-water loop with a chiller (heat sink) on
        // the supply side.  The chiller removes heat from the loop,
        // lowering the supply header temperature.
        let loop_ = PlantLoop::new("ChilledWaterLoop".to_string(), 7.0);
        // Chiller: removes 100 kW from the water
        let chiller = ConstantHeatSink { heat_w: 100_000.0 };
        let pump = PumpConstantSpeed::new("Pump-1".to_string(), 0.015, 18.0, 0.72, 0.90);

        let inlet = FluidState {
            temperature: 12.0,
            flow_rate: 0.015,
        };
        // Supply: chiller (removes heat), Demand: pump (circulates)
        let result = loop_.solve(&[&chiller], &[&pump], inlet, 25.0, 3600.0);
        assert!(
            result.converged,
            "solver did not converge in {} iterations",
            result.iterations
        );
        // Chiller should cool the water below inlet temperature
        assert!(
            result.supply_header_temperature < inlet.temperature,
            "supply temp {} not cooled below inlet {}",
            result.supply_header_temperature,
            inlet.temperature
        );
    }

    #[test]
    fn test_energy_balance_check_passes() {
        let result = PlantLoopResult {
            supply_header_temperature: 70.0,
            demand_header_temperature: 50.0,
            total_electrical_power_w: 1_000.0,
            total_heat_transfer_w: 49_000.0,
            iterations: 3,
            converged: true,
        };
        assert!(check_energy_balance(&result, 50_000.0, 500.0).is_ok());
    }

    #[test]
    fn test_energy_balance_check_fails() {
        let result = PlantLoopResult {
            supply_header_temperature: 70.0,
            demand_header_temperature: 50.0,
            total_electrical_power_w: 1_000.0,
            total_heat_transfer_w: 49_000.0,
            iterations: 3,
            converged: true,
        };
        assert!(check_energy_balance(&result, 100_000.0, 500.0).is_err());
    }
}
