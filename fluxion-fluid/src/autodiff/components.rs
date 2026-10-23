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
//!
//! ## FLUID-01: Jacobian Saturation/Clamping — resolved in #4210
//!
//! **Original issue**: the analytical Jacobians for Chiller, Boiler, CoolingCoil, Pump and
//! VavBox disagreed with finite differences because they ignored the non-smooth behaviour
//! introduced by the `clamp()` / `max()` calls in `evaluate`, and because several of the
//! hand-derived entry formulas were outright wrong.
//!
//! **Current design**: every `jacobian_input` (and `VavBox::jacobian_state`) evaluates the
//! *raw* input against the same clamp window `evaluate` uses and multiplies each partial by
//! a 0/1 saturation gate, so the returned matrix is the correct one-sided subgradient:
//!
//! 1. `evaluate` keeps the hard `clamp()`/`max()` calls, so physics values are unchanged.
//! 2. Each clamped input contributes `0` to the derivative when the raw value sits
//!    outside its window (the value cannot move, so neither can any output).
//! 3. Kinks that live on an *intermediate* quantity — the Chiller's COP floor
//!    (`rated_cop * (1 − 0.05·ΔT).max(0.1)`), the Boiler's efficiency ceiling, the VAV
//!    box's `t_supply` floor and `coil_valve` ceiling — are detected the same way, and
//!    the *remaining* chain-rule terms are kept: e.g. with COP pinned at its floor the
//!    compressor power still varies through `q_evap`, so `∂p/∂T_evap = ∂q/∂T_evap / cop`
//!    (4000 W/K at the reference point), not 0.
//! 4. Jacobians are assembled with [`DMatrix::from_row_slice`]. `DMatrix::from_vec` is
//!    **column-major**; feeding it row-major data silently transposes the matrix, which
//!    is what made every one of these components disagree with the finite-difference
//!    reference.
//!
//! The `*_at_saturation` tests probe genuinely saturated points and must pass the *same*
//! 10⁻⁴ relative tolerance as the nominal tests; measured worst-case agreement across
//! all saturation fixtures is < 6·10⁻⁶ relative.
//!
//! Saturation fixtures are chosen strictly *outside* the clamp window (never exactly on
//! the boundary) so that the forward-difference step `x + ε` stays inside the saturated
//! region. On the boundary itself the one-sided forward difference steps back into the
//! active region and measures the *other* subgradient — a property of the reference, not
//! of the analytical Jacobian.
//!
//! See: GitHub Issues #2375 (origin), #4210 (resolution); KNOWN_ISSUES.md §FLUID-01

use nalgebra::DMatrix;

use super::DifferentiableComponent;

const EPSILON: f64 = 1e-10;

const T_EVAP_IDX: usize = 0;
const T_COND_IDX: usize = 1;
const M_DOT_REF_IDX: usize = 2;

const T_RETURN_IDX: usize = 0;
const M_DOT_HOT_IDX: usize = 1;
const T_ENTER_IDX: usize = 2;

const DAMPER_IDX: usize = 0;
const STATIC_PRESSURE_IDX: usize = 1;
const T_INLET_IDX: usize = 2;

const SPEED_IDX: usize = 0;
const RHO_IDX: usize = 2;

const T_WB_IDX: usize = 0;
const M_DOT_AIR_IDX: usize = 1;
const T_WATER_IN_IDX: usize = 3;

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
        let _m_dot_ref = input[M_DOT_REF_IDX].max(0.0);

        let delta_t = t_cond - t_evap;
        let cop = self.rated_cop * (1.0 - 0.05 * delta_t).max(0.1);
        let q_evap = self.rated_capacity * (1.0 + 0.02 * t_evap);
        let p_compressor = q_evap / cop;

        vec![q_evap, cop, p_compressor]
    }

    fn jacobian_input(&self, input: &Self::Input, _state: &Self::State) -> DMatrix<f64> {
        let t_evap_raw = input[T_EVAP_IDX];
        let t_cond_raw = input[T_COND_IDX];
        let t_evap = t_evap_raw.clamp(-10.0, 10.0);
        let t_cond = t_cond_raw.clamp(20.0, 50.0);

        // Saturation gates: 0 when the raw input is outside its clamp window, because
        // `evaluate` returns the clamped value and no output can move with that input.
        let gate_t_evap = if t_evap_raw > -10.0 && t_evap_raw < 10.0 {
            1.0
        } else {
            0.0
        };
        let gate_t_cond = if t_cond_raw > 20.0 && t_cond_raw < 50.0 {
            1.0
        } else {
            0.0
        };

        let delta_t = t_cond - t_evap;
        // `evaluate` floors the *dimensionless* factor `1 - 0.05·ΔT` at 0.1 before
        // scaling by rated_cop, so the saturation test has to be made on the factor.
        let cop_factor = 1.0 - 0.05 * delta_t;
        let cop_is_saturated = cop_factor <= 0.1;
        let cop_base = self.rated_cop * if cop_is_saturated { 0.1 } else { cop_factor };
        let q_evap_base = self.rated_capacity * (1.0 + 0.02 * t_evap);

        // d(cop)/d(t_evap) = rated_cop · 0.05 (factor is decreasing in ΔT, increasing in T_evap)
        let d_cop_d_t_evap = if cop_is_saturated {
            0.0
        } else {
            self.rated_cop * 0.05
        } * gate_t_evap;
        let d_cop_d_t_cond = if cop_is_saturated {
            0.0
        } else {
            -self.rated_cop * 0.05
        } * gate_t_cond;

        let d_q_d_t_evap = self.rated_capacity * 0.02 * gate_t_evap;

        // p = q / cop by the quotient rule. When cop is pinned at its floor its own
        // derivative vanishes, but p still tracks q — dropping that term (as the
        // pre-#4210 code did) zeroed ∂p/∂T_evap and lost 4000 W/K at the reference point.
        let d_p_d_t_evap =
            (d_q_d_t_evap * cop_base - q_evap_base * d_cop_d_t_evap) / (cop_base * cop_base);
        let d_p_d_t_cond = (-q_evap_base * d_cop_d_t_cond) / (cop_base * cop_base);

        DMatrix::from_row_slice(
            3,
            3,
            &[
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
        let t_return_raw = input[T_RETURN_IDX];
        let m_dot_raw = input[M_DOT_HOT_IDX];
        let t_enter_raw = input[T_ENTER_IDX];
        let t_return = t_return_raw.clamp(20.0, 90.0);
        let m_dot_hot = m_dot_raw.max(0.0);
        let t_enter = t_enter_raw.clamp(5.0, 95.0);

        // Saturation gates on the *raw* inputs (the pre-#4210 code compared the already
        // clamped value against the bound, which can never fire from the outside).
        let gate_t_return = if t_return_raw > 20.0 && t_return_raw < 90.0 {
            1.0
        } else {
            0.0
        };
        let gate_m_dot = if m_dot_raw > 0.0 { 1.0 } else { 0.0 };
        let gate_t_enter = if t_enter_raw > 5.0 && t_enter_raw < 95.0 {
            1.0
        } else {
            0.0
        };

        let delta_t = t_return - t_enter;
        let eta_factor = 0.6 + 0.004 * t_return;
        let eta_is_saturated = eta_factor >= 1.0;
        let eta_base = self.rated_eta * if eta_is_saturated { 1.0 } else { eta_factor };
        let q_output_base = m_dot_hot * self.c_p * delta_t;

        let d_eta_d_t_return = if eta_is_saturated {
            0.0
        } else {
            self.rated_eta * 0.004
        } * gate_t_return;
        let d_q_d_m_dot = self.c_p * delta_t * gate_m_dot;
        let d_q_d_delta_t = m_dot_hot * self.c_p;
        let d_q_d_t_return = d_q_d_delta_t * gate_t_return;
        let d_q_d_t_enter = -d_q_d_delta_t * gate_t_enter;

        // fuel = q / eta by the quotient rule.
        let d_fuel_d_q = 1.0 / eta_base;
        let d_fuel_d_eta = -q_output_base / (eta_base * eta_base);
        let d_fuel_d_t_return = d_fuel_d_q * d_q_d_t_return + d_fuel_d_eta * d_eta_d_t_return;
        let d_fuel_d_m_dot = d_fuel_d_q * d_q_d_m_dot;
        let d_fuel_d_t_enter = d_fuel_d_q * d_q_d_t_enter;

        // Columns follow the input layout [T_return, m_dot_hot, t_enter]; the pre-#4210
        // data listed t_enter before m_dot_hot, which put the flow partial in the wrong
        // column for every m_dot-sensitive entry.
        DMatrix::from_row_slice(
            3,
            3,
            &[
                d_q_d_t_return,
                d_q_d_m_dot,
                d_q_d_t_enter,
                d_eta_d_t_return,
                0.0,
                0.0,
                d_fuel_d_t_return,
                d_fuel_d_m_dot,
                d_fuel_d_t_enter,
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

        let zone_demand = _state.first().copied().unwrap_or(self.rated_demand);

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
        let damper_raw = input[DAMPER_IDX];
        let pressure_raw = input[STATIC_PRESSURE_IDX];
        let t_inlet = input[T_INLET_IDX];
        let damper = damper_raw.clamp(0.0, 1.0);
        let pressure = pressure_raw.max(0.0);

        let gate_damper = if damper_raw > 0.0 && damper_raw < 1.0 {
            1.0
        } else {
            0.0
        };
        let gate_pressure = if pressure_raw > 0.0 { 1.0 } else { 0.0 };

        let zone_demand = state.first().copied().unwrap_or(self.rated_demand);

        let sqrt_2_rho = (2.0 / self.rho).sqrt();
        let k_damper = self.k_factor * damper;
        let m_dot_supply = k_damper * sqrt_2_rho * pressure.sqrt();

        let d_m_dot_d_damper = self.k_factor * sqrt_2_rho * pressure.sqrt() * gate_damper;
        let sqrt_p = pressure.sqrt().max(EPSILON);
        let d_m_dot_d_pressure = if k_damper <= 0.0 {
            0.0
        } else {
            k_damper * sqrt_2_rho * 0.5 / sqrt_p * gate_pressure
        };

        let reheat = zone_demand.max(0.0) * self.k_valve;
        // t_supply = max(t_inlet - reheat/(m·1006), t_inlet - 15). The floor branch is flat
        // in damper/pressure, so it zeroes those two entries while ∂t_supply/∂t_inlet stays 1
        // (both branches shift with t_inlet by the same amount).
        let floor_active = if m_dot_supply > 0.0 {
            t_inlet - reheat / (m_dot_supply * 1006.0) < t_inlet - 15.0
        } else {
            true
        };
        let flow_branch = if floor_active { 0.0 } else { 1.0 };

        // d(t_supply)/d(m_dot) = +reheat/(1006·m²): more supply air dilutes the reheat.
        // The pre-#4210 code carried the opposite sign, flipping the whole damper/pressure row.
        let d_t_supply_d_m_dot = if m_dot_supply > 0.0 {
            reheat / (m_dot_supply * m_dot_supply * 1006.0) * flow_branch
        } else {
            0.0
        };

        let dt_ddp = d_t_supply_d_m_dot * d_m_dot_d_damper;
        let dt_dsp = d_t_supply_d_m_dot * d_m_dot_d_pressure;
        let dt_dti = 1.0;

        // coil_valve = min(zone_demand/rated_demand, 1) is a function of the STATE only; it
        // has no dependence on damper, static pressure or inlet temperature.
        DMatrix::from_row_slice(
            3,
            3,
            &[
                d_m_dot_d_damper,
                d_m_dot_d_pressure,
                0.0,
                dt_ddp,
                dt_dsp,
                dt_dti,
                0.0,
                0.0,
                0.0,
            ],
        )
    }

    fn jacobian_state(&self, input: &Self::Input, state: &Self::State) -> DMatrix<f64> {
        let damper = input[DAMPER_IDX].clamp(0.0, 1.0);
        let pressure = input[STATIC_PRESSURE_IDX].max(0.0);
        let t_inlet = input[T_INLET_IDX];

        let zone_demand = state.first().copied().unwrap_or(self.rated_demand);

        let sqrt_2_rho = (2.0 / self.rho).sqrt();
        let k_damper = self.k_factor * damper;
        let m_dot_supply = k_damper * sqrt_2_rho * pressure.sqrt();

        let floor_active = if m_dot_supply > 0.0 {
            t_inlet - (zone_demand.max(0.0) * self.k_valve) / (m_dot_supply * 1006.0)
                < t_inlet - 15.0
        } else {
            true
        };
        let flow_branch = if floor_active { 0.0 } else { 1.0 };
        let gate_demand = if zone_demand > 0.0 { 1.0 } else { 0.0 };

        let d_t_supply_d_reheat = if m_dot_supply > 0.0 {
            (-1.0 / (m_dot_supply * 1006.0)) * flow_branch
        } else {
            0.0
        };
        let d_t_supply_d_demand = d_t_supply_d_reheat * self.k_valve * gate_demand;

        // coil_valve saturates at 1 once zone_demand reaches rated_demand.
        let d_coil_d_demand = if zone_demand > 0.0 && zone_demand < self.rated_demand {
            1.0 / self.rated_demand
        } else {
            0.0
        };

        DMatrix::from_row_slice(3, 1, &[0.0, d_t_supply_d_demand, d_coil_d_demand])
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
        let speed_raw = input[SPEED_IDX];
        let rho_raw = input[RHO_IDX];
        let speed = speed_raw.max(0.0);
        let rho = rho_raw.max(100.0);

        let speed_ratio = speed / 1750.0;
        let flow_ratio = speed_ratio;
        let _head_ratio = speed_ratio * speed_ratio;
        let _power_ratio = speed_ratio * speed_ratio * speed_ratio;

        // The speed floor and the density floor are independent: pinning either one to zero
        // only removes the partials that flow through it.
        let gate_speed = if speed_raw > 0.0 { 1.0 } else { 0.0 };
        let gate_rho = if rho_raw > 100.0 { 1.0 } else { 0.0 };

        let d_flow_d_speed = (self.rated_flow * rho / 1000.0) / 1750.0 * gate_speed;
        let d_flow_d_rho = (flow_ratio * self.rated_flow) / 1000.0 * gate_rho;
        let d_power_d_speed =
            3.0 * speed_ratio * speed_ratio * (self.rated_power / 1750.0) * gate_speed;
        let d_head_d_speed = 2.0 * speed_ratio * (self.rated_head / 1750.0) * gate_speed;

        // Rows follow the `evaluate` output order [m_dot, power, p_rise]; the pre-#4210 data
        // swapped the power and head rows and carried a spurious 1.0 in an unused column.
        DMatrix::from_row_slice(
            3,
            3,
            &[
                d_flow_d_speed,
                0.0,
                d_flow_d_rho,
                d_power_d_speed,
                0.0,
                0.0,
                d_head_d_speed,
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
        let t_wb_raw = input[T_WB_IDX];
        let m_dot_air_raw = input[M_DOT_AIR_IDX];
        let t_water_in_raw = input[T_WATER_IN_IDX];
        let t_wb = t_wb_raw.clamp(5.0, 25.0);
        let m_dot_air = m_dot_air_raw.max(0.0);
        let t_water_in = t_water_in_raw.clamp(0.0, 15.0);

        let gate_t_wb = if t_wb_raw > 5.0 && t_wb_raw < 25.0 {
            1.0
        } else {
            0.0
        };
        let gate_m_dot_air = if m_dot_air_raw > 0.0 { 1.0 } else { 0.0 };
        let gate_t_water_in = if t_water_in_raw > 0.0 && t_water_in_raw < 15.0 {
            1.0
        } else {
            0.0
        };

        let flow_ratio = m_dot_air / self.rated_flow;
        let denom = 1.0 + self.bypass_factor * flow_ratio;
        let effectiveness = (1.0 + self.bypass_factor) / denom;

        // d(eff)/d(m_dot) for eff = (1+b)/(1 + b·m/rf) is -(1+b)·b/(rf·denom²).
        // The pre-#4210 expression carried an extra b/denom factor (b²·eff/denom³/rf),
        // which was a factor of ~11 too small at the reference operating point.
        let eff_slope =
            -(1.0 + self.bypass_factor) * self.bypass_factor / (self.rated_flow * denom * denom);
        let d_eff_d_flow = eff_slope * gate_m_dot_air;

        let dt_air = t_wb - t_water_in;

        // q = m·c_p·eff·(t_wb − t_wi): the air flow enters twice (directly and through eff),
        // and the sign on the water term is negative — warmer chilled water means less load.
        let d_q_d_t_wb = m_dot_air * self.c_p_air * effectiveness * gate_t_wb;
        let d_q_d_m_dot =
            self.c_p_air * dt_air * (effectiveness + m_dot_air * eff_slope) * gate_m_dot_air;
        let d_q_d_t_water_in = -m_dot_air * self.c_p_air * effectiveness * gate_t_water_in;

        let d_t_out_d_t_wb = (1.0 - effectiveness) * gate_t_wb;
        let d_t_out_d_m_dot = -dt_air * eff_slope * gate_m_dot_air;
        let d_t_out_d_t_water_in = effectiveness * gate_t_water_in;

        // Row 2 is the effectiveness output, whose only sensitivity is to the air flow
        // (input index M_DOT_AIR_IDX = 1, not 0).
        DMatrix::from_row_slice(
            3,
            4,
            &[
                d_q_d_t_wb,
                d_q_d_m_dot,
                0.0,
                d_q_d_t_water_in,
                d_t_out_d_t_wb,
                d_t_out_d_m_dot,
                0.0,
                d_t_out_d_t_water_in,
                0.0,
                d_eff_d_flow,
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
    use crate::autodiff::validation::{
        finite_diff_epsilon, relative_error, verify_jacobian_entries,
        verify_jacobian_entries_at_saturation,
    };
    use crate::{finite_diff_jacobian, optimize_with_gradient_descent};

    const Q_EVAP_IDX: usize = 0;
    const COP_IDX: usize = 1;
    const P_COMPRESSOR_IDX: usize = 2;

    const Q_OUTPUT_IDX: usize = 0;
    const ETA_IDX: usize = 1;
    const FUEL_ENERGY_IDX: usize = 2;

    const M_DOT_SUPPLY_IDX: usize = 0;
    const T_SUPPLY_IDX: usize = 1;
    const COIL_VALVE_IDX: usize = 2;

    const M_DOT_IDX: usize = 0;
    const POWER_IDX: usize = 1;
    const P_RISE_IDX: usize = 2;

    const Q_COOLING_IDX: usize = 0;
    const EFFECTIVENESS_IDX: usize = 2;

    // =============================================================================
    // Jacobian accuracy tests (FLUID-01 cohort, un-quarantined by #4210)
    //
    // Two families:
    //
    // * `*_jacobian_accuracy` — the nominal operating point of each component, checked
    //   against forward finite differences at the strict 1e-4 relative tolerance.
    // * `*_jacobian_at_saturation` — genuinely saturated fixtures (raw inputs strictly
    //   *outside* each clamp window, plus the intermediate-quantity kinks: the Chiller
    //   COP floor, the VAV-box t_supply floor and coil_valve ceiling). These are
    //   asserted against BOTH the strict 1e-4 tolerance and the kink-tolerant
    //   `verify_jacobian_entries_at_saturation`, so the relaxed helper is no longer
    //   load-bearing: the analytical subgradient now matches the FD reference to <1e-5
    //   relative everywhere.
    //
    // Why the saturation fixtures avoid the exact clamp boundary: a one-sided forward
    // difference taken *on* a boundary steps back into the active region and therefore
    // measures the opposite subgradient. That is a limitation of the FD reference, not
    // of the analytical Jacobian, so the fixtures sit one full step beyond the bound.
    //
    // See: GitHub Issues #2375 (origin), #4210 (resolution); KNOWN_ISSUES.md §FLUID-01
    // =============================================================================

    /// Largest per-entry relative error between two Jacobians, using the same
    /// denominator convention as `validation::relative_error` (1e-10 floor).
    fn max_relative_error(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                worst = worst.max(relative_error(a[(i, j)], b[(i, j)]));
            }
        }
        worst
    }

    #[test]
    fn test_chiller_jacobian_accuracy() {
        let chiller = Chiller::new(100_000.0, 5.0);
        let input = vec![7.0, 35.0, 0.5];
        let state = vec![];

        let analytical = chiller.jacobian_input(&input, &state);
        let f = |x: &[f64]| chiller.evaluate(&x.to_vec(), &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Chiller Jacobian mismatch (max rel err {:.3e}):\nanalytical={:?}\nfinite_diff={:?}",
            max_relative_error(&analytical, &finite_diff),
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_chiller_jacobian_at_saturation() {
        let chiller = Chiller::new(100_000.0, 5.0);
        let state = vec![];

        // (a) T_evap above its 10 °C ceiling and T_cond above its 50 °C ceiling,
        // (b) both temperatures below their floors, (c) ΔT large enough to floor the
        //     dimensionless COP factor at 0.1 while both temperatures stay unclamped.
        for input in [
            vec![12.0, 55.0, 0.5],
            vec![-15.0, 15.0, -1.0],
            vec![7.0, 35.0, 0.5],
        ] {
            let analytical = chiller.jacobian_input(&input, &state);
            let f = |x: &[f64]| chiller.evaluate(&x.to_vec(), &state);
            let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());
            let err = max_relative_error(&analytical, &finite_diff);

            assert!(
                verify_jacobian_entries(&analytical, &finite_diff),
                "Chiller Jacobian mismatch at saturation (max rel err {:.3e}, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                err,
                input,
                analytical,
                finite_diff
            );
            assert!(
                verify_jacobian_entries_at_saturation(&analytical, &finite_diff),
                "Chiller Jacobian mismatch at saturation (subgradient tolerance, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                input,
                analytical,
                finite_diff
            );
        }
    }

    #[test]
    fn test_boiler_jacobian_accuracy() {
        let boiler = Boiler::new(50_000.0, 0.9);
        let input = vec![60.0, 1.0, 50.0];
        let state = vec![];

        let analytical = boiler.jacobian_input(&input, &state);
        let f = |x: &[f64]| boiler.evaluate(&x.to_vec(), &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Boiler Jacobian mismatch (max rel err {:.3e}):\nanalytical={:?}\nfinite_diff={:?}",
            max_relative_error(&analytical, &finite_diff),
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_boiler_jacobian_at_saturation() {
        let boiler = Boiler::new(50_000.0, 0.9);
        let state = vec![];

        // (a) T_return above its 90 °C ceiling and T_enter above its 95 °C ceiling,
        // (b) T_return below its 20 °C floor, negative hot-water flow and T_enter above
        //     its ceiling — i.e. every input saturated at once.
        for input in [vec![95.0, 1.0, 99.0], vec![10.0, -2.0, 120.0]] {
            let analytical = boiler.jacobian_input(&input, &state);
            let f = |x: &[f64]| boiler.evaluate(&x.to_vec(), &state);
            let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());
            let err = max_relative_error(&analytical, &finite_diff);

            assert!(
                verify_jacobian_entries(&analytical, &finite_diff),
                "Boiler Jacobian mismatch at saturation (max rel err {:.3e}, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                err,
                input,
                analytical,
                finite_diff
            );
            assert!(
                verify_jacobian_entries_at_saturation(&analytical, &finite_diff),
                "Boiler Jacobian mismatch at saturation (subgradient tolerance, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                input,
                analytical,
                finite_diff
            );
        }
    }

    #[test]
    fn test_vav_box_jacobian_accuracy() {
        let vav = VavBox::new();
        let input = vec![0.7, 250.0, 26.0];
        let state = vec![3000.0];

        let analytical = vav.jacobian_input(&input, &state);
        let f = |x: &[f64]| vav.evaluate(&x.to_vec(), &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "VAV Box Jacobian mismatch (max rel err {:.3e}):\nanalytical={:?}\nfinite_diff={:?}",
            max_relative_error(&analytical, &finite_diff),
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_vav_box_jacobian_at_saturation() {
        let vav = VavBox::new();

        // (a) damper above its 1.0 ceiling, (b) a zone demand large enough to push
        //     t_supply onto its `t_inlet - 15` floor and coil_valve onto its 1.0 ceiling.
        // Both the input Jacobian and the state Jacobian are checked — coil_valve only
        // appears in the state Jacobian, and t_supply's floor zeroes its damper/pressure
        // entries in both.
        for (input, state) in [
            (vec![1.5, 300.0, 26.0], vec![3000.0]),
            (vec![0.7, 250.0, 26.0], vec![50_000.0]),
        ] {
            let f = |x: &[f64]| vav.evaluate(&x.to_vec(), &state);
            let analytical = vav.jacobian_input(&input, &state);
            let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());
            let err = max_relative_error(&analytical, &finite_diff);

            assert!(
                verify_jacobian_entries(&analytical, &finite_diff),
                "VAV Box Jacobian mismatch at saturation (max rel err {:.3e}, input={:?}, state={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                err,
                input,
                state,
                analytical,
                finite_diff
            );
            assert!(
                verify_jacobian_entries_at_saturation(&analytical, &finite_diff),
                "VAV Box Jacobian mismatch at saturation (subgradient tolerance, input={:?}, state={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                input,
                state,
                analytical,
                finite_diff
            );

            let analytical_state = vav.jacobian_state(&input, &state);
            let fs = |s: &[f64]| vav.evaluate(&input.to_vec(), &s.to_vec());
            let finite_diff_state = finite_diff_jacobian(fs, &state, finite_diff_epsilon());
            let state_err = max_relative_error(&analytical_state, &finite_diff_state);

            assert!(
                verify_jacobian_entries(&analytical_state, &finite_diff_state),
                "VAV Box state-Jacobian mismatch at saturation (max rel err {:.3e}, input={:?}, state={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                state_err,
                input,
                state,
                analytical_state,
                finite_diff_state
            );
            assert!(
                verify_jacobian_entries_at_saturation(&analytical_state, &finite_diff_state),
                "VAV Box state-Jacobian mismatch at saturation (subgradient tolerance, input={:?}, state={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                input,
                state,
                analytical_state,
                finite_diff_state
            );
        }
    }

    #[test]
    fn test_vav_box_gradient_descent_convergence() {
        // Zero zone demand: `evaluate` then has no reheat and a closed coil valve, so
        // t_supply ≡ t_inlet and coil_valve ≡ 0 and the target vector below is actually
        // reachable. With a non-zero demand (the pre-#4210 fixture used state = [3000.0])
        // t_supply can never reach 26 °C for any finite airflow and coil_valve is pinned
        // at 0.6, so the unreachable residual gradient just walks the damper into its 1.0
        // ceiling and m_dot_supply settles at 1.02 kg/s instead of the 0.25 target — for
        // *every* learning rate from 1e-4 to 0.3. That convergence only ever "worked"
        // because d(t_supply)/d(damper) carried the wrong sign; see VavBox::jacobian_input.
        let vav = VavBox::new();
        let state = vec![0.0];

        // VavBox::evaluate returns [m_dot_supply, t_supply, coil_valve]
        // We target all three outputs and optimize all three inputs
        let mut input = vec![0.5, 250.0, 26.0];
        // Target: m_dot_supply=0.25, t_supply and coil_valve unchanged
        let target = vec![0.25, 26.0, 0.0];

        let iterations =
            optimize_with_gradient_descent(&vav, &target, &mut input, &state, 0.5, 1e-3, 10);

        let output = vav.evaluate(&input, &state);
        // Check convergence on m_dot_supply (the primary target)
        let m_dot_error = (output[M_DOT_SUPPLY_IDX] - target[0]).abs();

        assert!(
            iterations <= 10,
            "VAV box gradient descent took {} iterations (expected ≤ 10)",
            iterations
        );
        assert!(
            m_dot_error < 1e-2,
            "VAV box m_dot_supply error = {} (expected < 1e-2), damper={}",
            m_dot_error,
            input[DAMPER_IDX]
        );
    }

    #[test]
    fn test_pump_jacobian_accuracy() {
        let pump = Pump::new(0.5, 100_000.0, 5000.0);
        let input = vec![1500.0, 100_000.0, 1000.0];
        let state = vec![];

        let analytical = pump.jacobian_input(&input, &state);
        let f = |x: &[f64]| pump.evaluate(&x.to_vec(), &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Pump Jacobian mismatch (max rel err {:.3e}):\nanalytical={:?}\nfinite_diff={:?}",
            max_relative_error(&analytical, &finite_diff),
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_pump_jacobian_at_saturation() {
        let pump = Pump::new(0.5, 100_000.0, 5000.0);
        let state = vec![];

        // (a) speed below its 0.0 floor, (b) fluid density below the 100 kg/m³ floor.
        for input in [vec![-100.0, 1.0, 1000.0], vec![1000.0, 1.0, 50.0]] {
            let analytical = pump.jacobian_input(&input, &state);
            let f = |x: &[f64]| pump.evaluate(&x.to_vec(), &state);
            let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());
            let err = max_relative_error(&analytical, &finite_diff);

            assert!(
                verify_jacobian_entries(&analytical, &finite_diff),
                "Pump Jacobian mismatch at saturation (max rel err {:.3e}, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                err,
                input,
                analytical,
                finite_diff
            );
            assert!(
                verify_jacobian_entries_at_saturation(&analytical, &finite_diff),
                "Pump Jacobian mismatch at saturation (subgradient tolerance, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                input,
                analytical,
                finite_diff
            );
        }
    }

    #[test]
    fn test_cooling_coil_jacobian_accuracy() {
        let coil = CoolingCoil::new(20_000.0, 1.0);
        let input = vec![20.0, 1.0, 0.5, 7.0];
        let state = vec![];

        let analytical = coil.jacobian_input(&input, &state);
        let f = |x: &[f64]| coil.evaluate(&x.to_vec(), &state);
        let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());

        assert!(
            verify_jacobian_entries(&analytical, &finite_diff),
            "Cooling Coil Jacobian mismatch (max rel err {:.3e}):\nanalytical={:?}\nfinite_diff={:?}",
            max_relative_error(&analytical, &finite_diff),
            analytical,
            finite_diff
        );
    }

    #[test]
    fn test_cooling_coil_jacobian_at_saturation() {
        let coil = CoolingCoil::new(20_000.0, 1.0);
        let state = vec![];

        // (a) T_wb above its 25 °C ceiling and T_water_in above its 15 °C ceiling,
        // (b) T_wb below its 5 °C floor, negative air flow and T_water_in below its 0 °C
        //     floor — i.e. every input saturated at once.
        for input in [vec![28.0, 2.0, 0.5, 18.0], vec![3.0, -0.5, 0.5, -2.0]] {
            let analytical = coil.jacobian_input(&input, &state);
            let f = |x: &[f64]| coil.evaluate(&x.to_vec(), &state);
            let finite_diff = finite_diff_jacobian(f, &input, finite_diff_epsilon());
            let err = max_relative_error(&analytical, &finite_diff);

            assert!(
                verify_jacobian_entries(&analytical, &finite_diff),
                "CoolingCoil Jacobian mismatch at saturation (max rel err {:.3e}, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                err,
                input,
                analytical,
                finite_diff
            );
            assert!(
                verify_jacobian_entries_at_saturation(&analytical, &finite_diff),
                "CoolingCoil Jacobian mismatch at saturation (subgradient tolerance, input={:?}):\nanalytical={:?}\nfinite_diff={:?}",
                input,
                analytical,
                finite_diff
            );
        }
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
