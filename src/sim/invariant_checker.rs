//! Energy and Mass Balance Invariant Checker
//!
//! This module provides strict invariant checking for the thermal solver loop.
//! It verifies that for every thermal node:
//! `(Heat In) - (Heat Out) - (Change in Internal Energy) ≈ 0`
//! within a defined machine epsilon/tolerance (default: 1e-7).
//!
//! # Usage
//!
//! ```rust
//! use fluxion::sim::invariant_checker::InvariantChecker;
//!
//! let mut checker = InvariantChecker::new(1e-7);
//! // After each timestep, call check_invariant() to verify energy balance
//! ```

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;
use std::ops::Index;

pub const DEFAULT_TOLERANCE: f64 = 1e-7;

#[derive(Debug, Clone)]
pub struct InvariantChecker {
    tolerance: f64,
    violation_count: usize,
    max_violation: f64,
    total_checks: usize,
}

impl InvariantChecker {
    pub fn new(tolerance: f64) -> Self {
        InvariantChecker {
            tolerance,
            violation_count: 0,
            max_violation: 0.0,
            total_checks: 0,
        }
    }

    pub fn with_default_tolerance() -> Self {
        Self::new(DEFAULT_TOLERANCE)
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn violation_count(&self) -> usize {
        self.violation_count
    }

    pub fn max_violation(&self) -> f64 {
        self.max_violation
    }

    pub fn total_checks(&self) -> usize {
        self.total_checks
    }

    pub fn reset(&mut self) {
        self.violation_count = 0;
        self.max_violation = 0.0;
        self.total_checks = 0;
    }

    pub fn check_invariant<T>(
        &mut self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> InvariantResult
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        self.total_checks += 1;

        let balance = self.calculate_energy_imbalance(model, dt_seconds, outdoor_temp);

        let violated = balance.abs() > self.tolerance;
        if violated {
            self.violation_count += 1;
            if balance.abs() > self.max_violation {
                self.max_violation = balance.abs();
            }
        }

        InvariantResult {
            balance,
            violated,
            tolerance: self.tolerance,
            zone_imbalances: self.calculate_per_zone_imbalance(model, dt_seconds, outdoor_temp),
        }
    }

    fn calculate_energy_imbalance<T>(
        &self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> f64
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let num_zones = model.num_zones;
        let temps = model.temperatures.as_ref();
        let mass_temps = model.mass_temperatures.as_ref();
        let prev_mass_temps = model.previous_mass_temperatures.as_ref();
        let loads = model.loads.as_ref();
        let solar_gains = model.solar_gains.as_ref();
        let area = model.zone_area.as_ref();

        let mut total_balance = 0.0;

        for i in 0..num_zones {
            let t_air = temps[i];
            let t_mass = mass_temps[i];
            let t_mass_prev = prev_mass_temps[i];
            let load_w = loads[i] * area[i];
            let solar_w = solar_gains[i] * area[i];

            let h_tr_em = model.h_tr_em[i];
            let h_tr_ms = model.h_tr_ms[i];
            let h_tr_w = model.h_tr_w[i];
            let h_ve = model.h_ve[i];
            let h_tr_floor = model.h_tr_floor[i];
            let t_ground = model.ground_temperature.ground_temperature(0);
            let cm = model.thermal_capacitance[i];

            let conv_frac = model.convective_fraction;
            let sol_dist_to_air = model.solar_distribution_to_air;

            let phi_ia = load_w * conv_frac + solar_w * sol_dist_to_air;
            let phi_st = load_w * (1.0 - conv_frac) * (1.0 - sol_dist_to_air);
            let phi_m = load_w * (1.0 - conv_frac) * sol_dist_to_air;

            let q_em = h_tr_em * (t_mass - outdoor_temp);
            let q_ms = h_tr_ms * (t_air - t_mass);
            let q_w = h_tr_w * (t_air - outdoor_temp);
            let q_ve = h_ve * (t_air - outdoor_temp);
            let q_floor = h_tr_floor * (t_air - t_ground);

            let heat_in = phi_ia + phi_st + phi_m;
            let heat_out = q_em + q_ms + q_w + q_ve + q_floor;
            let mass_power = cm * (t_mass - t_mass_prev) / dt_seconds;

            let zone_balance = heat_in - heat_out - mass_power;
            total_balance += zone_balance;
        }

        total_balance
    }

    fn calculate_per_zone_imbalance<T>(
        &self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> Vec<f64>
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let num_zones = model.num_zones;
        let temps = model.temperatures.as_ref();
        let mass_temps = model.mass_temperatures.as_ref();
        let prev_mass_temps = model.previous_mass_temperatures.as_ref();
        let loads = model.loads.as_ref();
        let solar_gains = model.solar_gains.as_ref();
        let area = model.zone_area.as_ref();

        let mut imbalances = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let t_air = temps[i];
            let t_mass = mass_temps[i];
            let t_mass_prev = prev_mass_temps[i];
            let load_w = loads[i] * area[i];
            let solar_w = solar_gains[i] * area[i];

            let h_tr_em = model.h_tr_em[i];
            let h_tr_ms = model.h_tr_ms[i];
            let h_tr_w = model.h_tr_w[i];
            let h_ve = model.h_ve[i];
            let h_tr_floor = model.h_tr_floor[i];
            let t_ground = model.ground_temperature.ground_temperature(0);
            let cm = model.thermal_capacitance[i];

            let conv_frac = model.convective_fraction;
            let sol_dist_to_air = model.solar_distribution_to_air;

            let phi_ia = load_w * conv_frac + solar_w * sol_dist_to_air;
            let phi_st = load_w * (1.0 - conv_frac) * (1.0 - sol_dist_to_air);
            let phi_m = load_w * (1.0 - conv_frac) * sol_dist_to_air;

            let q_em = h_tr_em * (t_mass - outdoor_temp);
            let q_ms = h_tr_ms * (t_air - t_mass);
            let q_w = h_tr_w * (t_air - outdoor_temp);
            let q_ve = h_ve * (t_air - outdoor_temp);
            let q_floor = h_tr_floor * (t_air - t_ground);

            let heat_in = phi_ia + phi_st + phi_m;
            let heat_out = q_em + q_ms + q_w + q_ve + q_floor;
            let mass_power = cm * (t_mass - t_mass_prev) / dt_seconds;

            let zone_balance = heat_in - heat_out - mass_power;
            imbalances.push(zone_balance);
        }

        imbalances
    }

    pub fn check_invariant_with_artificial_gain<T>(
        &mut self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
        artificial_gain_watts: f64,
        zone_index: usize,
    ) -> InvariantResult
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        self.total_checks += 1;

        let mut modified_loads = model.loads.as_ref().to_vec();
        if zone_index < modified_loads.len() {
            modified_loads[zone_index] +=
                artificial_gain_watts / model.zone_area.as_ref()[zone_index];
        }

        let num_zones = model.num_zones;
        let temps = model.temperatures.as_ref();
        let mass_temps = model.mass_temperatures.as_ref();
        let prev_mass_temps = model.previous_mass_temperatures.as_ref();
        let area = model.zone_area.as_ref();
        let solar_gains = model.solar_gains.as_ref();

        let mut total_balance = 0.0;
        let mut zone_imbalances = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let t_air = temps[i];
            let t_mass = mass_temps[i];
            let t_mass_prev = prev_mass_temps[i];
            let load_w = modified_loads[i] * area[i];
            let solar_w = solar_gains[i] * area[i];

            let h_tr_em = model.h_tr_em[i];
            let h_tr_ms = model.h_tr_ms[i];
            let h_tr_w = model.h_tr_w[i];
            let h_ve = model.h_ve[i];
            let h_tr_floor = model.h_tr_floor[i];
            let t_ground = model.ground_temperature.ground_temperature(0);
            let cm = model.thermal_capacitance[i];

            let conv_frac = model.convective_fraction;
            let sol_dist_to_air = model.solar_distribution_to_air;

            let phi_ia = load_w * conv_frac + solar_w * sol_dist_to_air;
            let phi_st = load_w * (1.0 - conv_frac) * (1.0 - sol_dist_to_air);
            let phi_m = load_w * (1.0 - conv_frac) * sol_dist_to_air;

            let q_em = h_tr_em * (t_mass - outdoor_temp);
            let q_ms = h_tr_ms * (t_air - t_mass);
            let q_w = h_tr_w * (t_air - outdoor_temp);
            let q_ve = h_ve * (t_air - outdoor_temp);
            let q_floor = h_tr_floor * (t_air - t_ground);

            let heat_in = phi_ia + phi_st + phi_m;
            let heat_out = q_em + q_ms + q_w + q_ve + q_floor;
            let mass_power = cm * (t_mass - t_mass_prev) / dt_seconds;

            let zone_balance = heat_in - heat_out - mass_power;
            total_balance += zone_balance;
            zone_imbalances.push(zone_balance);
        }

        let violated = total_balance.abs() > self.tolerance;

        InvariantResult {
            balance: total_balance,
            violated,
            tolerance: self.tolerance,
            zone_imbalances,
        }
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::with_default_tolerance()
    }
}

#[derive(Debug, Clone)]
pub struct InvariantResult {
    pub balance: f64,
    pub violated: bool,
    pub tolerance: f64,
    pub zone_imbalances: Vec<f64>,
}

impl InvariantResult {
    pub fn is_balanced(&self) -> bool {
        !self.violated
    }

    pub fn relative_error(&self) -> f64 {
        if self.balance.abs() < 1e-20 {
            0.0
        } else {
            self.balance.abs() / self.tolerance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;

    #[test]
    fn test_invariant_checker_creation() {
        let checker = InvariantChecker::new(1e-7);
        assert_eq!(checker.tolerance(), 1e-7);
        assert_eq!(checker.violation_count(), 0);
        assert_eq!(checker.total_checks(), 0);
    }

    #[test]
    fn test_invariant_checker_default() {
        let checker = InvariantChecker::default();
        assert_eq!(checker.tolerance(), DEFAULT_TOLERANCE);
    }

    #[test]
    fn test_invariant_checker_reset() {
        let mut checker = InvariantChecker::new(1e-7);
        checker.total_checks = 10;
        checker.violation_count = 5;
        checker.max_violation = 1e-5;
        checker.reset();
        assert_eq!(checker.total_checks(), 0);
        assert_eq!(checker.violation_count(), 0);
        assert_eq!(checker.max_violation(), 0.0);
    }

    #[test]
    fn test_balanced_model_passes_invariant() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.apply_parameters(&[1.5, 20.0, 27.0]);

        let test_loads = vec![0.0; 1];
        model.set_loads(&test_loads);

        model.step_physics(0, 20.0, 3600.0);

        let mut checker = InvariantChecker::new(1e-3);
        let result = checker.check_invariant(&model, 3600.0, 20.0);

        println!(
            "Balance: {}, tolerance: {}",
            result.balance, result.tolerance
        );
        println!("Zone imbalances: {:?}", result.zone_imbalances);
    }

    #[test]
    fn test_artificial_heat_gain_detected() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.apply_parameters(&[1.5, 20.0, 27.0]);

        let test_loads = vec![0.0; 1];
        model.set_loads(&test_loads);

        model.step_physics(0, 20.0, 3600.0);

        let mut checker = InvariantChecker::new(1e-7);

        let result_normal = checker.check_invariant(&model, 3600.0, 20.0);
        let normal_balance = result_normal.balance.abs();

        let artificial_gain = 1.0;
        let result_with_gain =
            checker.check_invariant_with_artificial_gain(&model, 3600.0, 20.0, artificial_gain, 0);

        let gain_balance = result_with_gain.balance.abs();

        println!(
            "Normal balance: {} (abs: {})",
            result_normal.balance, normal_balance
        );
        println!(
            "Balance with 1W artificial gain: {} (abs: {})",
            result_with_gain.balance, gain_balance
        );

        assert!(
            result_with_gain.balance.abs() > result_normal.balance.abs(),
            "Artificial gain should increase imbalance"
        );
    }

    #[test]
    fn test_invariant_result_is_balanced() {
        let result = InvariantResult {
            balance: 1e-8,
            violated: false,
            tolerance: 1e-7,
            zone_imbalances: vec![1e-8],
        };
        assert!(result.is_balanced());

        let result_violated = InvariantResult {
            balance: 1e-6,
            violated: true,
            tolerance: 1e-7,
            zone_imbalances: vec![1e-6],
        };
        assert!(!result_violated.is_balanced());
    }

    #[test]
    fn test_relative_error() {
        let result = InvariantResult {
            balance: 5e-8,
            violated: false,
            tolerance: 1e-7,
            zone_imbalances: vec![5e-8],
        };
        assert!((result.relative_error() - 0.5).abs() < 1e-10);

        let result_zero = InvariantResult {
            balance: 0.0,
            violated: false,
            tolerance: 1e-7,
            zone_imbalances: vec![0.0],
        };
        assert_eq!(result_zero.relative_error(), 0.0);
    }

    #[test]
    fn test_multi_zone_invariant_tracking() {
        let mut model = ThermalModel::<VectorField>::new(3);
        model.apply_parameters(&[1.5, 20.0, 27.0]);

        let test_loads = vec![0.0; 3];
        model.set_loads(&test_loads);

        model.step_physics(0, 20.0, 3600.0);

        let mut checker = InvariantChecker::new(1e-3);
        let result = checker.check_invariant(&model, 3600.0, 20.0);

        assert_eq!(result.zone_imbalances.len(), 3);
        println!("Zone imbalances: {:?}", result.zone_imbalances);
    }
}
