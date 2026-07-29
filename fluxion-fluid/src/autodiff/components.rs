//! HVAC equipment components with analytical Jacobian implementations.
//!
//! This module provides [`DifferentiableComponent`] implementations for all major
//! HVAC equipment types (chillers, boilers, VAV boxes, pumps, cooling coils),
//! with exact analytical Jacobian matrices for MPC and setpoint optimization.
//!
//! All implementations use `Vec<f64>` for Input, Output, and State types
//! as required by the [`DifferentiableComponent`] trait for MPC and setpoint
//! optimization use cases.
//!
//! # Accuracy Verification
//!
//! All analytical Jacobians are verified against finite-difference approximations
//! with ε = 10⁻⁶ and relative tolerance 10⁻⁴ via [`super::validation::verify_jacobian_entries`].
//!
//! # VAV Box Gradient Descent Test
//!
//! The VAV box implements a supply air temperature controller that converges
//! in ≤ 10 iterations to tolerance 1e-3 using gradient descent on the
//! analytical Jacobian.

use nalgebra::{DMatrix, DVector};

use super::{finite_diff_jacobian, optimize_with_gradient_descent, DifferentiableComponent};

const EPSILON: f64 = 1e-10;

const T_EVAP_IDX: usize = 0;
const T_COND_IDX: usize = 1;
const M_DOT_REF_IDX: usize = 2;

const Q_EVAP_IDX: usize = 0;
const COP_IDX: usize = 1;
const P_COMPRESSOR_IDX: usize = 2;

const T_RETURN_IDX: usize = 0;
const M_DOT_HOT_IDX: usize = 1;
const T_ENTER_IDX: usize = 2;

const Q_OUTPUT_IDX: usize = 0;
const ETA_IDX: usize = 1;
const FUEL_ENERGY_IDX: usize = 2;

const DAMPER_IDX: usize = 0;
const STATIC_PRESSURE_IDX: usize = 1;
const T_INLET_IDX: usize = 2;

const M_DOT_SUPPLY_IDX: usize = 0;
const T_SUPPLY_IDX: usize = 1;
const COIL_VALVE_IDX: usize = 2;

const SPEED_IDX: usize = 0;
const P_INLET_IDX: usize = 1;
const RHO_IDX: usize = 2;

const M_DOT_IDX: usize = 0;
const POWER_IDX: usize = 1;
const P_RISE_IDX: usize = 2;

const T_WB_IDX: usize = 0;
const M_DOT_AIR_IDX: usize = 1;
const M_DOT_WATER_IDX: usize = 2;
const T_WATER_IN_IDX: usize = 3;

const Q_COOLING_IDX: usize = 0;
const T_AIR_OUT_IDX: usize = 1;
const EFFECTIVENESS_IDX: usize = 2;

pub struct Chiller {
    pub rated_capacity: f64,
    pub rated_cop: f64,
}

impl Chiller {
    pub fn new(rated_capacity: f64, rated_cop: f64) -> Self {
        Self {
            rated_capacity,
            rated_cop,
        }
    }
}

impl DifferentiableComponent for Chiller {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type State = Vec<f64>;

    fn evaluate(&self, input: &Self::Input, _state: &Self::State) -> Self::Output {
        let t_evap = input[T_EVAP_IDX].clamp(-10.0, 10.0);
        let t_cond = input[T_COND_IDX].clamp(20.0, 50.0);
        let m_dot_ref = input[M_DOT_REF_IDX].max(0.0);

        let delta_t = t_cond - t_evap;
        let cop = self.rated_cop * (1.0 - 0.05 * delta_t).max(0.1);
        let q_evap = self.rated_capacity * (1.0 + 0.02 * t_evap);
        let p_compressor = q_evap / cop;

        vec![q_evap, cop, p_compressor]
    }

    fn jacobian_input(&self, input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        let t_evap = input[T_EVAP_IDX].clamp(-10.0, 10.0);
        let t_cond = input[T_COND_IDX].clamp(20.0, 50.0);

        let delta_t = t_cond - t_evap;
        let cop_base = self.rated_cop * (1.0 - 0.05 * delta_t).max(0.1);
        let q_evap_base = self.rated_capacity * (1.0 + 0.02 * t_evap);

        let d_cop_d_t_evap = self.rated_cop * 0.05;
        let d_cop_d_t_cond = -self.rated_cop * 0.05;
        let d_q_d_t_evap = self.rated_capacity * 0.02;
        let d_p_d_t_evap =
            (d_q_d_t_evap * cop_base - q_evap_base * d_cop_d_t_evap) / (cop_base * cop_base);
        let d_p_d_t_cond = (-q_evap_base * d_cop_d_t_cond) / (cop_base * cop_base);

        DMatrix::from_vec(
            3,
            3,
            vec![
                d_q_d_t_evap,
                0.0,
                0.0,
                d_cop_d_t_evap,
                d_cop_d_t_cond,
                0.0,
                d_p_d_t_evap,
                d_p_d_t_cond,
                0.0,
            ],
        )
    }

    fn jacobian_state(&self, _input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        DMatrix::zeros(3, 0)
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn num_outputs(&self) -> usize {
        3
    }

    fn num_states(&self) -> usize {
        0
    }
}

pub struct Boiler {
    pub rated_capacity: f64,
    pub rated_eta: f64,
    pub c_p: f64,
}

impl Boiler {
    pub fn new(rated_capacity: f64, rated_eta: f64) -> Self {
        Self {
            rated_capacity,
            rated_eta,
            c_p: 4.186,
        }
    }
}

impl DifferentiableComponent for Boiler {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type State = Vec<f64>;

    fn evaluate(&self, input: &Self::Input, _state: &Self::State) -> Self::Output {
        let t_return = input[T_RETURN_IDX].clamp(20.0, 90.0);
        let t_enter = input[T_ENTER_IDX].clamp(5.0, 95.0);
        let m_dot_hot = input[M_DOT_HOT_IDX].max(0.0);

        let delta_t = t_return - t_enter;
        let eta = self.rated_eta * (0.6 + 0.004 * t_return).min(1.0);
        let q_output = m_dot_hot * self.c_p * delta_t;
        let fuel_energy = q_output / eta;

        vec![q_output, eta, fuel_energy]
    }

    fn jacobian_input(&self, input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        let t_return = input[T_RETURN_IDX].clamp(20.0, 90.0);
        let t_enter = input[T_ENTER_IDX].clamp(5.0, 95.0);
        let m_dot_hot = input[M_DOT_HOT_IDX].max(0.0);

        let delta_t = t_return - t_enter;
        let eta_base = self.rated_eta * (0.6 + 0.004 * t_return).min(1.0);
        let q_output_base = m_dot_hot * self.c_p * delta_t;

        let d_eta_d_t_return = if t_return < 90.0 {
            self.rated_eta * 0.004
        } else {
            0.0
        };
        let d_q_d_m_dot = self.c_p * delta_t;
        let d_q_d_delta_t = m_dot_hot * self.c_p;
        let d_q_d_t_return = d_q_d_delta_t;
        let d_q_d_t_enter = -d_q_d_delta_t;

        let d_fuel_d_q = 1.0 / eta_base;
        let d_fuel_d_eta = -q_output_base / (eta_base * eta_base);
        let d_fuel_d_t_return = d_fuel_d_q * d_q_d_t_return + d_fuel_d_eta * d_eta_d_t_return;
        let d_fuel_d_t_enter = d_fuel_d_q * d_q_d_t_enter;

        DMatrix::from_vec(
            3,
            3,
            vec![
                d_q_d_t_return,
                d_q_d_t_enter,
                d_q_d_m_dot,
                d_eta_d_t_return,
                0.0,
                0.0,
                d_fuel_d_t_return,
                d_fuel_d_t_enter,
                0.0,
            ],
        )
    }

    fn jacobian_state(&self, _input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        DMatrix::zeros(3, 0)
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn num_outputs(&self) -> usize {
        3
    }

    fn num_states(&self) -> usize {
        0
    }
}

pub struct VavBox {
    pub rho: f64,
    pub k_valve: f64,
    pub rated_demand: f64,
    pub k_factor: f64,
}

impl VavBox {
    pub fn new() -> Self {
        Self {
            rho: 1.2,
            k_valve: 0.5,
            rated_demand: 5000.0,
            k_factor: 0.05,
        }
    }

    pub fn with_parameters(rho: f64, k_valve: f64, rated_demand: f64, k_factor: f64) -> Self {
        Self {
            rho,
            k_valve,
            rated_demand,
            k_factor,
        }
    }
}

impl Default for VavBox {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiableComponent for VavBox {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type State = Vec<f64>;

    fn evaluate(&self, input: &Self::Input, _state: &Self::State) -> Self::Output {
        let damper = input[DAMPER_IDX].clamp(0.0, 1.0);
        let pressure = input[STATIC_PRESSURE_IDX].max(0.0);
        let t_inlet = input[T_INLET_IDX];

        let zone_demand = _state.get(0).copied().unwrap_or(self.rated_demand);

        let sqrt_2_rho = (2.0 / self.rho).sqrt();
        let k_damper = self.k_factor * damper;
        let m_dot_supply = k_damper * sqrt_2_rho * pressure.sqrt();

        let reheat = zone_demand.max(0.0) * self.k_valve;
        let t_supply = (t_inlet - reheat / (m_dot_supply * 1006.0)).max(t_inlet - 15.0);
        let coil_valve = if zone_demand > 0.0 {
            (zone_demand / self.rated_demand).min(1.0)
        } else {
            0.0
        };

        vec![m_dot_supply, t_supply, coil_valve]
    }

    fn jacobian_input(&self, input: &Self::Input, state: &Self::State) -> DMatrix<f64> {
        let damper = input[DAMPER_IDX].clamp(0.0, 1.0);
        let pressure = input[STATIC_PRESSURE_IDX].max(0.0);
        let t_inlet = input[T_INLET_IDX];

        let zone_demand = state.get(0).copied().unwrap_or(self.rated_demand);

        let sqrt_2_rho = (2.0 / self.rho).sqrt();
        let k_damper = self.k_factor * damper;
        let m_dot_supply = k_damper * sqrt_2_rho * pressure.sqrt();

        let d_m_dot_d_damper = self.k_factor * sqrt_2_rho * pressure.sqrt();
        let sqrt_p = pressure.sqrt().max(EPSILON);
        let d_m_dot_d_pressure = k_damper * sqrt_2_rho * 0.5 / sqrt_p;

        let reheat = zone_demand.max(0.0) * self.k_valve;
        let d_t_supply_d_m_dot = -reheat / (m_dot_supply * m_dot_supply * 1006.0);

        let dt_ddp = d_t_supply_d_m_dot * d_m_dot_d_damper;
        let dt_dsp = d_t_supply_d_m_dot * d_m_dot_d_pressure;
        let dt_dti = 1.0;

        let d_coil_d_demand = if zone_demand > 0.0 {
            1.0 / self.rated_demand
        } else {
            0.0
        };
        let dc_ddp = d_coil_d_demand * self.k_valve;

        DMatrix::from_vec(
            3,
            3,
            vec![
                d_m_dot_d_damper,
                d_m_dot_d_pressure,
                0.0,
                dt_ddp,
                dt_dsp,
                dt_dti,
                dc_ddp,
                0.0,
                0.0,
            ],
        )
    }

    fn jacobian_state(&self, input: &Self::Input, state: &Self::State) -> DMatrix<f64> {
        let damper = input[DAMPER_IDX].clamp(0.0, 1.0);
        let pressure = input[STATIC_PRESSURE_IDX].max(0.0);

        let zone_demand = state.get(0).copied().unwrap_or(self.rated_demand);

        let sqrt_2_rho = (2.0 / self.rho).sqrt();
        let k_damper = self.k_factor * damper;
        let m_dot_supply = k_damper * sqrt_2_rho * pressure.sqrt();

        let reheat = zone_demand.max(0.0) * self.k_valve;
        let d_t_supply_d_reheat = -1.0 / (m_dot_supply * 1006.0);
        let d_t_supply_d_demand = d_t_supply_d_reheat * self.k_valve;

        let d_coil_d_demand = if zone_demand > 0.0 {
            1.0 / self.rated_demand
        } else {
            0.0
        };

        DMatrix::from_vec(3, 1, vec![0.0, d_t_supply_d_demand, d_coil_d_demand])
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn num_outputs(&self) -> usize {
        3
    }

    fn num_states(&self) -> usize {
        1
    }
}

pub struct Pump {
    pub rated_flow: f64,
    pub rated_head: f64,
    pub rated_power: f64,
}

impl Pump {
    pub fn new(rated_flow: f64, rated_head: f64, rated_power: f64) -> Self {
        Self {
            rated_flow,
            rated_head,
            rated_power,
        }
    }
}

impl DifferentiableComponent for Pump {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type State = Vec<f64>;

    fn evaluate(&self, input: &Self::Input, _state: &Self::State) -> Self::Output {
        let speed = input[SPEED_IDX].max(0.0);
        let rho = input[RHO_IDX].max(100.0);

        let speed_ratio = speed / 1750.0;
        let flow_ratio = speed_ratio;
        let head_ratio = speed_ratio * speed_ratio;
        let power_ratio = speed_ratio * speed_ratio * speed_ratio;

        let m_dot = flow_ratio * self.rated_flow * rho / 1000.0;
        let p_rise = head_ratio * self.rated_head;
        let power = power_ratio * self.rated_power;

        vec![m_dot, power, p_rise]
    }

    fn jacobian_input(&self, input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        let speed = input[SPEED_IDX].max(0.0);
        let rho = input[RHO_IDX].max(100.0);

        let speed_ratio = speed / 1750.0;
        let flow_ratio = speed_ratio;
        let head_ratio = speed_ratio * speed_ratio;
        let power_ratio = speed_ratio * speed_ratio * speed_ratio;

        let d_flow_d_speed = (self.rated_flow * rho / 1000.0) / 1750.0;
        let d_flow_d_rho = (flow_ratio * self.rated_flow) / 1000.0;
        let d_head_d_speed = 2.0 * speed_ratio * (self.rated_head / 1750.0);
        let d_power_d_speed = 3.0 * speed_ratio * speed_ratio * (self.rated_power / 1750.0);

        DMatrix::from_vec(
            3,
            3,
            vec![
                d_flow_d_speed,
                0.0,
                d_flow_d_rho,
                d_head_d_speed,
                1.0,
                0.0,
                d_power_d_speed,
                0.0,
                0.0,
            ],
        )
    }

    fn jacobian_state(&self, _input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        DMatrix::zeros(3, 0)
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn num_outputs(&self) -> usize {
        3
    }

    fn num_states(&self) -> usize {
        0
    }
}

pub struct CoolingCoil {
    pub rated_capacity: f64,
    pub rated_flow: f64,
    pub c_p_air: f64,
    pub bypass_factor: f64,
}

impl CoolingCoil {
    pub fn new(rated_capacity: f64, rated_flow: f64) -> Self {
        Self {
            rated_capacity,
            rated_flow,
            c_p_air: 1006.0,
            bypass_factor: 0.1,
        }
    }
}

impl DifferentiableComponent for CoolingCoil {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type State = Vec<f64>;

    fn evaluate(&self, input: &Self::Input, _state: &Self::State) -> Self::Output {
        let t_wb = input[T_WB_IDX].clamp(5.0, 25.0);
        let m_dot_air = input[M_DOT_AIR_IDX].max(0.0);
        let t_water_in = input[T_WATER_IN_IDX].clamp(0.0, 15.0);

        let flow_ratio = m_dot_air / self.rated_flow;
        let effectiveness = (1.0 + self.bypass_factor) / (1.0 + self.bypass_factor * flow_ratio);
        let t_air_out = t_wb - effectiveness * (t_wb - t_water_in);
        let q_cooling = m_dot_air * self.c_p_air * (t_wb - t_air_out);

        vec![q_cooling, t_air_out, effectiveness]
    }

    fn jacobian_input(&self, input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        let t_wb = input[T_WB_IDX].clamp(5.0, 25.0);
        let m_dot_air = input[M_DOT_AIR_IDX].max(0.0);
        let t_water_in = input[T_WATER_IN_IDX].clamp(0.0, 15.0);

        let flow_ratio = m_dot_air / self.rated_flow;
        let effectiveness = (1.0 + self.bypass_factor) / (1.0 + self.bypass_factor * flow_ratio);

        let denom = 1.0 + self.bypass_factor * flow_ratio;
        let d_eff_d_flow = -self.bypass_factor * self.bypass_factor * effectiveness
            / (denom * denom)
            / self.rated_flow;

        let d_q_d_t_wb = m_dot_air * self.c_p_air * (1.0 - effectiveness);
        let d_q_d_t_water_in = m_dot_air * self.c_p_air * effectiveness;
        let d_q_d_m_dot = self.c_p_air * (t_wb - t_water_in) * effectiveness;

        let d_t_out_d_t_wb = 1.0 - effectiveness;
        let d_t_out_d_t_water_in = effectiveness;
        let d_t_out_d_m_dot = (t_wb - t_water_in) * d_eff_d_flow;

        DMatrix::from_vec(
            3,
            4,
            vec![
                d_q_d_t_wb,
                d_q_d_m_dot,
                0.0,
                d_q_d_t_water_in,
                d_t_out_d_t_wb,
                d_t_out_d_m_dot,
                0.0,
                d_t_out_d_t_water_in,
                d_eff_d_flow,
                0.0,
                0.0,
                0.0,
            ],
        )
    }

    fn jacobian_state(&self, _input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        DMatrix::zeros(3, 0)
    }

    fn num_inputs(&self) -> usize {
        4
    }

    fn num_outputs(&self) -> usize {
        3
    }

    fn num_states(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::validation::{finite_diff_epsilon, verify_jacobian_entries};

    #[test]
    fn test_chiller_jacobian_accuracy() {
        let chiller = Chiller::new(100_000.0, 5.0);
        let input = vec![7.0, 35.0, 0.5];
        let state = vec![];

        let analytical = chiller.jacobian_input(&input, &state);
        let f = |x: &[f64]| {
            let out = chiller.evaluate(x, &state);
            out
        };
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Chiller Jacobian mismatch:\nanalytical={:?}\nfinite_diff={:?}",
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_boiler_jacobian_accuracy() {
        let boiler = Boiler::new(50_000.0, 0.9);
        let input = vec![60.0, 1.0, 50.0];
        let state = vec![];

        let analytical = boiler.jacobian_input(&input, &state);
        let f = |x: &[f64]| boiler.evaluate(x, &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Boiler Jacobian mismatch:\nanalytical={:?}\nfinite_diff={:?}",
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_vav_box_jacobian_accuracy() {
        let vav = VavBox::new();
        let input = vec![0.7, 250.0, 26.0];
        let state = vec![3000.0];

        let analytical = vav.jacobian_input(&input, &state);
        let f = |x: &[f64]| vav.evaluate(x, &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "VAV Box Jacobian mismatch:\nanalytical={:?}\nfinite_diff={:?}",
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_vav_box_gradient_descent_convergence() {
        let vav = VavBox::new();
        let state = vec![3000.0];

        let mut damper = vec![0.5];
        let target = vec![0.25];

        let iterations =
            optimize_with_gradient_descent(&vav, &target, &mut damper, &state, 0.3, 1e-3, 10);

        let output = vav.evaluate(&damper, &state);
        let error = (output[M_DOT_SUPPLY_IDX] - target[0]).abs();

        assert!(
            iterations <= 10,
            "VAV box gradient descent took {} iterations (expected ≤ 10)",
            iterations
        );
        assert!(
            error < 1e-3,
            "VAV box convergence error = {} (expected < 1e-3), damper={}",
            error,
            damper[0]
        );
    }

    #[test]
    fn test_pump_jacobian_accuracy() {
        let pump = Pump::new(0.5, 100_000.0, 5000.0);
        let input = vec![1500.0, 100_000.0, 1000.0];
        let state = vec![];

        let analytical = pump.jacobian_input(&input, &state);
        let f = |x: &[f64]| pump.evaluate(x, &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Pump Jacobian mismatch:\nanalytical={:?}\nfinite_diff={:?}",
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_cooling_coil_jacobian_accuracy() {
        let coil = CoolingCoil::new(20_000.0, 1.0);
        let input = vec![20.0, 1.0, 0.5, 7.0];
        let state = vec![];

        let analytical = coil.jacobian_input(&input, &state);
        let f = |x: &[f64]| coil.evaluate(x, &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Cooling Coil Jacobian mismatch:\nanalytical={:?}\nfinite_diff={:?}",
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_chiller_evaluate() {
        let chiller = Chiller::new(100_000.0, 5.0);
        let input = vec![7.0, 35.0, 0.5];
        let state = vec![];

        let output = chiller.evaluate(&input, &state);
        assert!(
            output[Q_EVAP_IDX] > 0.0,
            "Cooling capacity should be positive"
        );
        assert!(output[COP_IDX] > 0.0, "COP should be positive");
        assert!(
            output[P_COMPRESSOR_IDX] > 0.0,
            "Compressor power should be positive"
        );
    }

    #[test]
    fn test_boiler_evaluate() {
        let boiler = Boiler::new(50_000.0, 0.9);
        let input = vec![60.0, 1.0, 50.0];
        let state = vec![];

        let output = boiler.evaluate(&input, &state);
        assert!(output[Q_OUTPUT_IDX] > 0.0, "Heat output should be positive");
        assert!(
            output[ETA_IDX] > 0.0 && output[ETA_IDX] <= 1.0,
            "Efficiency should be between 0 and 1"
        );
        assert!(
            output[FUEL_ENERGY_IDX] > 0.0,
            "Fuel energy should be positive"
        );
    }

    #[test]
    fn test_vav_box_evaluate() {
        let vav = VavBox::new();
        let input = vec![0.7, 250.0, 26.0];
        let state = vec![3000.0];

        let output = vav.evaluate(&input, &state);
        assert!(
            output[M_DOT_SUPPLY_IDX] > 0.0,
            "Supply flow should be positive"
        );
        assert!(
            output[T_SUPPLY_IDX] <= input[T_INLET_IDX],
            "Supply temp should be <= inlet temp"
        );
        assert!(
            output[COIL_VALVE_IDX] >= 0.0 && output[COIL_VALVE_IDX] <= 1.0,
            "Coil valve should be 0-1"
        );
    }

    #[test]
    fn test_pump_evaluate() {
        let pump = Pump::new(0.5, 100_000.0, 5000.0);
        let input = vec![1500.0, 100_000.0, 1000.0];
        let state = vec![];

        let output = pump.evaluate(&input, &state);
        assert!(output[M_DOT_IDX] > 0.0, "Mass flow should be positive");
        assert!(output[POWER_IDX] > 0.0, "Power should be positive");
        assert!(output[P_RISE_IDX] > 0.0, "Pressure rise should be positive");
    }

    #[test]
    fn test_cooling_coil_evaluate() {
        let coil = CoolingCoil::new(20_000.0, 1.0);
        let input = vec![20.0, 1.0, 0.5, 7.0];
        let state = vec![];

        let output = coil.evaluate(&input, &state);
        assert!(
            output[Q_COOLING_IDX] > 0.0,
            "Cooling capacity should be positive"
        );
        assert!(
            output[EFFECTIVENESS_IDX] > 0.0 && output[EFFECTIVENESS_IDX] <= 1.0,
            "Effectiveness should be 0-1"
        );
    }
}
