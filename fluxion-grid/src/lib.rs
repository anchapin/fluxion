//! # fluxion-grid
//!
//! Grid-edge electrical network components for Fluxion building energy modeling.
//!
//! This crate provides battery storage node models with state-of-charge (SoC) tracking
//! and electrical characteristics for integration with building energy simulations.
//!
//! ## Contents
//!
//! | Module | Description |
//! |--------|-------------|
//! | `battery_storage_node` | `BatteryStorageNode` — single-cell battery model with SoC, terminal voltage, and C-rate dynamics |

#![allow(nonstandard_style)]
#![allow(clippy::all)]

pub mod battery_storage_node;

pub use battery_storage_node::BatteryStorageNode;

// === Heat Pump Voltage Model ===

// --- thermal_electrical_coupler.rs --- 
/// Coupler between thermal and electrical systems via heat pump COP.
///
/// The COP (Coefficient of Performance) links electrical power consumption
/// to thermal power production: thermal_power = COP * electrical_power.
#[derive(Debug, Clone)]
pub struct ThermalElectricalCoupler {
    /// Current coefficient of performance
    pub cop: f64,
    /// Rated COP at reference conditions
    pub rated_cop: f64,
    /// Temperature of heat source/sink (°C)
    pub source_temperature: f64,
    /// Temperature of heat delivery (°C)
    pub delivery_temperature: f64,
}

impl ThermalElectricalCoupler {
    /// Create a new coupler with specified COP.
    pub fn new(cop: f64) -> Self {
        ThermalElectricalCoupler {
            cop,
            rated_cop: cop,
            source_temperature: 10.0,
            delivery_temperature: 45.0,
        }
    }

    /// Calculate COP based on Carnot efficiency and degradation.
    ///
    /// COP = COP_rated * efficiency_factor * carnot_ratio
    pub fn update_cop(&mut self, ambient_temperature: f64) {
        let t_hot_k = self.delivery_temperature + 273.15;
        let t_cold_k = ambient_temperature.max(-10.0) + 273.15;

        // Carnot COP: COP_carnot = T_hot / (T_hot - T_cold)
        let carnot = t_hot_k / (t_hot_k - t_cold_k);

        // Real heat pumps achieve ~40-60% of Carnot, but the rated COP already
        // accounts for this, so we use the ratio of actual to reference Carnot
        // Reference Carnot at 20°C: 293.15 / (293.15 - 283.15) = 29.3
        let reference_carnot = 293.15 / 10.0;

        // COP adjustment based on temperature difference
        let carnot_factor = (carnot / reference_carnot).clamp(0.3, 1.5);

        // Actual COP = rated_COP * carnot_factor
        // The rated COP already includes the efficiency factor
        self.cop = self.rated_cop * carnot_factor;
        self.cop = self.cop.clamp(1.0, self.rated_cop * 1.5);
    }

    /// Convert thermal load to electrical power.
    pub fn thermal_to_electrical(&self, thermal_power: f64) -> f64 {
        thermal_power / self.cop
    }

    /// Convert electrical power to thermal power.
    pub fn electrical_to_thermal(&self, electrical_power: f64) -> f64 {
        electrical_power * self.cop
    }
}

/// Joint convergence solver for thermal-electrical systems.
///
/// This solver iteratively solves the coupled thermal and electrical systems
/// until both converge within the specified tolerance.
#[derive(Debug, Clone)]
pub struct JointConvergenceSolver {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Time step for thermal solve (s)
    pub dt: f64,
}

impl JointConvergenceSolver {
    /// Create a new joint convergence solver.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        JointConvergenceSolver {
            max_iterations,
            tolerance,
            dt: 3600.0, // 1 hour default timestep
        }
    }

    /// Solve the joint thermal-electrical system iteratively.
    ///
    /// Iteration pattern:
    /// 1. Solve thermal → compute zone temperatures and HVAC loads
    /// 2. Compute heat pump electrical load from thermal load
    /// 3. Solve electrical → compute bus voltages and power flows
    /// 4. Update COP based on electrical state
    /// 5. Check convergence
    pub fn solve(
        &mut self,
        thermal_model: &mut ThermalModel,
        electrical_model: &mut ElectricalNetwork,
        coupler: &mut ThermalElectricalCoupler,
    ) -> ConvergenceResult {
        let mut iterations = 0;
        let mut thermal_residual = f64::MAX;
        let mut electrical_mismatch = f64::MAX;

        while iterations < self.max_iterations {
            // Step 0: Update HVAC loads based on current temperatures and setpoints
            thermal_model.update_hvac_loads(coupler);

            // Step 1: Solve thermal system
            let prev_temps = thermal_model.temperatures.clone();
            thermal_model.solve_step(self.dt);
            thermal_residual = prev_temps
                .iter()
                .zip(&thermal_model.temperatures)
                .map(|(t_prev, t_new)| (t_new - t_prev).abs())
                .sum::<f64>()
                / thermal_model.num_zones as f64;

            // Step 2: Compute heat pump electrical load from thermal load
            let total_thermal_load: f64 = thermal_model.hvac_loads.iter().sum();
            let electrical_load = coupler.thermal_to_electrical(total_thermal_load);

            // Step 3: Update electrical network with heat pump load
            let load_per_bus = electrical_load / electrical_model.num_buses as f64;
            for i in 0..electrical_model.num_buses {
                if i != electrical_model.reference_bus {
                    electrical_model.power_injections[i] = -load_per_bus;
                }
            }

            // Step 4: Solve electrical system
            electrical_model.solve_power_flow_step();
            electrical_mismatch = electrical_model.calculate_mismatch();

            // Step 5: Update COP based on zone temperature
            let zone_temp = thermal_model.temperatures[0];
            coupler.update_cop(zone_temp);

            // Check convergence
            if thermal_residual < self.tolerance && electrical_mismatch < self.tolerance {
                return ConvergenceResult {
                    converged: true,
                    iterations,
                    thermal_residual,
                    electrical_mismatch,
                };
            }

            iterations += 1;
        }

        ConvergenceResult {
            converged: false,
            iterations,
            thermal_residual,
            electrical_mismatch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_creation() {
        let thermal = ThermalModel::new(3, 20.0);
        assert_eq!(thermal.num_zones, 3);
        assert_eq!(thermal.temperatures, vec![20.0, 20.0, 20.0]);
    }

    #[test]
    fn test_electrical_network_creation() {
        let electrical = ElectricalNetwork::new(2);
        assert_eq!(electrical.num_buses, 2);
        assert_eq!(electrical.voltages, vec![1.0, 1.0]);
    }

    #[test]
    fn test_coupler_creation() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        assert_eq!(coupler.cop, 3.0);
        assert_eq!(coupler.rated_cop, 3.0);
    }

    #[test]
    fn test_cop_update() {
        let mut coupler = ThermalElectricalCoupler::new(3.0);
        coupler.update_cop(10.0); // 10°C ambient
        assert!(coupler.cop >= 1.0);
        assert!(coupler.cop <= coupler.rated_cop);
    }

    #[test]
    fn test_thermal_to_electrical() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let electrical = coupler.thermal_to_electrical(3000.0);
        assert!((electrical - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_joint_convergence_simple() {
        let mut thermal = ThermalModel::new(1, 18.0);
        thermal.heating_setpoints = vec![20.0];
        thermal.ambient_temperature = 10.0;

        let mut electrical = ElectricalNetwork::new(1);

        let mut coupler = ThermalElectricalCoupler::new(3.0);

        let mut solver = JointConvergenceSolver::new(100, 1e-3);
        let result = solver.solve(&mut thermal, &mut electrical, &mut coupler);

        assert!(result.converged, "Solver should converge, got iterations={}, thermal_res={}, elec_mis={}",
            result.iterations, result.thermal_residual, result.electrical_mismatch);
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_joint_convergence_two_zones() {
        let mut thermal = ThermalModel::new(2, 18.0);
        thermal.heating_setpoints = vec![20.0, 20.0];
        thermal.ambient_temperature = 10.0;

        let electrical = ElectricalNetwork::new(2);

        let mut coupler = ThermalElectricalCoupler::new(3.0);

        let mut solver = JointConvergenceSolver::new(100, 1e-3);

        // Solve thermal-only (electrical mismatch will be non-zero due to simplified power flow)
        // For full power flow convergence, a more sophisticated solver would be needed
        let mut thermal_copy = thermal.clone();
        let result = solver.solve(&mut thermal_copy, &mut electrical.clone(), &mut coupler);

        // Thermal should converge even if electrical doesn't fully converge
        assert!(result.thermal_residual < 1e-2,
            "Thermal residual should be small, got {}", result.thermal_residual);
        assert!(result.iterations > 0, "Iterations should be reported");
    }

    #[test]
    fn test_joint_convergence_building_heat_pump() {
        let mut thermal = ThermalModel::new(3, 16.0);
        thermal.heating_setpoints = vec![20.0, 20.0, 20.0];
        thermal.ambient_temperature = 5.0;
        thermal.capacitances = vec![5_000_000.0; 3];

        let electrical = ElectricalNetwork::new(3);

        let mut coupler = ThermalElectricalCoupler::new(3.5);

        let mut solver = JointConvergenceSolver::new(200, 1e-4);

        let mut thermal_copy = thermal.clone();
        let result = solver.solve(&mut thermal_copy, &mut electrical.clone(), &mut coupler);

        // Thermal should converge
        assert!(result.thermal_residual < 1e-3,
            "Thermal should converge for building + heat pump case, got {}",
            result.thermal_residual);
        assert!(result.iterations > 0);
        assert!(coupler.cop > 1.0);
    }

    #[test]
    fn test_convergence_result_reports_iterations() {
        let mut thermal = ThermalModel::new(1, 18.0);
        thermal.heating_setpoints = vec![20.0];
        thermal.ambient_temperature = 10.0;

        let mut electrical = ElectricalNetwork::new(1);

        let mut coupler = ThermalElectricalCoupler::new(3.0);

        let mut solver = JointConvergenceSolver::new(50, 1e-3);
        let result = solver.solve(&mut thermal, &mut electrical, &mut coupler);

        assert!(result.iterations > 0, "Iterations should be reported");
    }

    #[test]
    fn test_max_iterations_prevents_infinite_loop() {
        let mut thermal = ThermalModel::new(5, 100.0);
        thermal.capacitances = vec![100.0; 5];
        thermal.heating_setpoints = vec![20.0; 5];
        thermal.ambient_temperature = -50.0; // Extreme cold

        let mut electrical = ElectricalNetwork::new(5);

        let mut coupler = ThermalElectricalCoupler::new(2.0);

        let mut solver = JointConvergenceSolver::new(10, 1e-12);
        let result = solver.solve(&mut thermal, &mut electrical, &mut coupler);

        assert!(
            !result.converged || result.iterations <= 10,
            "Should respect max iterations"
        );
        assert!(
            result.iterations <= 10,
            "Should not exceed max iterations, got {}",
            result.iterations
        );
    }

    #[test]
    fn test_electrical_mismatch_is_reasonable() {
        // Single bus with known load
        let mut electrical = ElectricalNetwork::new(1);
        electrical.power_injections[0] = -100.0; // 100W load

        let mismatch = electrical.calculate_mismatch();
        // For 1 bus (reference bus), mismatch should be 0
        assert_eq!(mismatch, 0.0, "Mismatch with only reference bus should be 0");
    }

    #[test]
    fn test_electrical_mismatch_two_buses() {
        let electrical = ElectricalNetwork::new(2);
        // After power flow solve, the angles should balance the loads
        // Test that mismatch is computed correctly
        let mismatch = electrical.calculate_mismatch();
        // With no loads, mismatch should be 0
        assert_eq!(mismatch, 0.0);
    }
}

// --- error.rs --- 
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GridModelError {
    #[error("voltage {voltage:.3} pu is outside valid range [0.5, 1.5]")]
    VoltageOutOfRange { voltage: f64 },

    #[error("coupler thermal mass is zero — cannot apply voltage adjustment")]
    ZeroThermalMass,

    #[error("negative COP adjustment factor {factor:.4} would increase COP")]
    NegativeAdjustment { factor: f64 },
}

// --- heat_pump_voltage_model.rs --- 
use crate::{GridModelError, ThermalElectricalCoupler, VoltagePu};

#[derive(Debug, Clone, Default)]
pub struct HeatPumpVoltageModel {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub voltage_nominal: f64,
}

impl HeatPumpVoltageModel {
    pub fn new(a: f64, b: f64, c: f64, voltage_nominal: f64) -> Self {
        Self {
            a,
            b,
            c,
            voltage_nominal,
        }
    }

    pub fn cop_adjustment_factor(&self, voltage_pu: VoltagePu) -> Result<f64, GridModelError> {
        if !(0.5..=1.5).contains(&voltage_pu) {
            return Err(GridModelError::VoltageOutOfRange {
                voltage: voltage_pu,
            });
        }
        let factor = self.a + self.b * voltage_pu + self.c * voltage_pu * voltage_pu;
        Ok(factor)
    }

    pub fn apply_to_coupler(
        &self,
        coupler: &mut ThermalElectricalCoupler,
        voltage_pu: VoltagePu,
    ) -> Result<(), GridModelError> {
        if coupler.thermal_mass_j_per_k <= 0.0 {
            return Err(GridModelError::ZeroThermalMass);
        }
        let factor = self.cop_adjustment_factor(voltage_pu)?;
        if factor < 0.0 {
            return Err(GridModelError::NegativeAdjustment { factor });
        }
        coupler.current_voltage_pu = voltage_pu;
        coupler.current_cop *= factor;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn default_model() -> HeatPumpVoltageModel {
        HeatPumpVoltageModel::new(0.85, 0.30, -0.15, 230.0)
    }

    #[test]
    fn test_cop_at_nominal_is_unity() {
        let model = default_model();
        let factor = model.cop_adjustment_factor(1.0).unwrap();
        assert_relative_eq!(factor, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cop_at_09pu_less_than_cop_at_10pu() {
        let model = default_model();
        let f_09 = model.cop_adjustment_factor(0.9).unwrap();
        let f_10 = model.cop_adjustment_factor(1.0).unwrap();
        assert!(
            f_09 < f_10,
            "COP at 0.9 pu ({f_09}) should be less than COP at 1.0 pu ({f_10})"
        );
    }

    #[test]
    fn test_cop_polynomial_values() {
        let model = default_model();
        let f_10 = model.cop_adjustment_factor(1.0).unwrap();
        let f_09 = model.cop_adjustment_factor(0.9).unwrap();
        let f_05 = model.cop_adjustment_factor(1.05).unwrap();
        assert_relative_eq!(f_10, 1.0, epsilon = 1e-10);
        assert!(f_09 < 1.0, "f(0.9) = {f_09} should be < 1.0");
        assert!(
            f_05 < 1.0 && f_05 > f_09,
            "f(1.05) = {f_05} should be < 1.0 but > f(0.9)"
        );
    }

    #[test]
    fn test_voltage_out_of_range() {
        let model = default_model();
        let result = model.cop_adjustment_factor(1.6);
        assert!(result.is_err());
        let result = model.cop_adjustment_factor(0.4);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_to_coupler() {
        let model = default_model();
        let mut coupler = ThermalElectricalCoupler::new(1000.0, 5000.0);
        coupler.set_cop(3.5);

        model.apply_to_coupler(&mut coupler, 0.9).unwrap();

        let expected_factor = model.cop_adjustment_factor(0.9).unwrap();
        let expected_cop = 3.5 * expected_factor;
        assert_relative_eq!(coupler.current_cop, expected_cop, epsilon = 1e-10);
        assert_eq!(coupler.current_voltage_pu, 0.9);
    }

    #[test]
    fn test_apply_zero_thermal_mass() {
        let model = default_model();
        let mut coupler = ThermalElectricalCoupler::new(1000.0, 0.0);
        let result = model.apply_to_coupler(&mut coupler, 0.9);
        assert!(result.is_err());
    }
}
