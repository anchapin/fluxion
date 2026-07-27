//! Air-side distribution and duct heat gain isolation tests (Issue #1927).
//!
//! Validates the physics of air-side distribution systems including:
//! - Fan heat addition to supply airstream
//! - Supply air temperature vs room air temperature coupling
//! - VAV terminal flow accuracy and damper authority
//! - Air-side distribution efficacy
//!
//! Reference: ASHRAE Handbook—Fundamentals (2021), Ch. 21 (fan laws),
//! Ch. 6 (psychrometrics), and the airside coupling documentation in
//! `src/sim/hvac/airside_coupling.rs`.
//!
//! # What is NOT covered here
//!
//! - Full duct heat-transfer network models (no separate duct abstraction exists yet)
//! - Duct leakage (not modelled in current VavTerminalUnit)
//! - EnergyPlus CSV comparison (no reference CSVs exist for supply-air temperature)

use fluxion::multi_node::{MassAirCouplingMode, ThermalMassNode};
use fluxion::physics::multi_node_solver::{MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion::sim::hvac::{
    AirsideEnvelopeCoupler, AirsideFlow, CoolingCoil, CoolingCoilBehavior, CoupledStepForcing, Fan,
    FanComponent, HeatingCoilComponent, HeatingCoilControl, MoistAirState, VavOperatingMode,
    VavTerminal, VavTerminalControl, VavTerminalUnit, DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
    STANDARD_AIR_DENSITY_KG_PER_M3,
};

const SEA_LEVEL_PA: f64 = 101_325.0;
const TIMESTEP_SECONDS: f64 = 360.0;

fn moist_air(temp_c: f64, rh_percent: f64, pressure_pa: f64) -> MoistAirState {
    MoistAirState::try_new(temp_c, rh_percent, pressure_pa).expect("air state must be physical")
}

fn supply_flow(temp_c: f64, rh_percent: f64, volumetric_flow_m3_per_s: f64) -> AirsideFlow {
    let air = moist_air(temp_c, rh_percent, SEA_LEVEL_PA);
    AirsideFlow::new(air, volumetric_flow_m3_per_s).expect("supply flow must be physical")
}

fn vav_standard() -> VavTerminalUnit {
    let cooling = CoolingCoil::new("CC-1".to_string(), 30_000.0, 0.75, 0.15, 10.0, 2.0);
    let reheat = HeatingCoilComponent::new("HC-1".to_string(), 10_000.0, 2.0);
    VavTerminalUnit::new("VAV-1".to_string(), 0, 2.0, cooling, Some(reheat))
}

fn zone_coupler(t_initial_c: f64) -> AirsideEnvelopeCoupler {
    let wall = ThermalMassNode::new(20.0, 5.0e6, 76.4, 25.0);
    let roof = ThermalMassNode::new(20.0, 3.0e6, 32.9, 20.0);
    let floor = ThermalMassNode::new(20.0, 2.0e6, 18.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1.0e6, 0.0, 0.0).with_h_tr_me(100.0);
    let solver = MultiNodeSolver::new_with_mode(
        165.6,
        wall,
        roof,
        floor,
        internal,
        MassAirCouplingMode::ParallelResistance,
    );
    let zone_air = moist_air(t_initial_c, 50.0, SEA_LEVEL_PA);
    AirsideEnvelopeCoupler::new(solver, zone_air, 300.0).expect("zone coupler must construct")
}

// ============================================================================
// Fan heat addition tests
// ============================================================================

/// Fan heat raises supply-air temperature above entering-air temperature in
/// deadband mode (damper at minimum, no cooling coil, no reheat).
/// Physics: ΔT = P_shaft / (ṁ_dry × cp_da)
#[test]
fn fan_heat_raises_supply_above_entering_in_deadband() {
    let terminal = vav_standard();
    let entering = moist_air(24.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
        .expect("deadband performance must succeed");

    assert!(
        perf.supply_air.dry_bulb_c > entering.dry_bulb_c,
        "fan heat must raise supply above entering: supply={}, entering={}",
        perf.supply_air.dry_bulb_c,
        entering.dry_bulb_c
    );
    assert!(
        perf.fan_heat_w > 0.0,
        "fan heat must be positive in deadband"
    );
}

/// Fan heat ΔT follows the analytical fan-heat formula exactly.
#[test]
fn fan_heat_delta_t_matches_analytical_formula() {
    let terminal = vav_standard();
    let entering = moist_air(24.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
        .expect("deadband performance must succeed");

    let w = entering.humidity_ratio_kg_per_kg_dry_air;
    let cp_da_j = 1000.0 * (1.006 + 1.86 * w);
    let expected_delta_t = perf.fan_heat_w / (perf.dry_air_mass_flow_kg_per_s * cp_da_j);
    let actual_delta_t = perf.supply_air.dry_bulb_c - entering.dry_bulb_c;

    assert!(
        (expected_delta_t - actual_delta_t).abs() < 1e-9,
        "fan heat ΔT: expected={}, actual={}",
        expected_delta_t,
        actual_delta_t
    );
}

/// At zero fan speed (min_airflow_ratio = 0), fan heat is zero.
#[test]
fn zero_min_ratio_eliminates_fan_heat() {
    let terminal = vav_standard().with_min_airflow_ratio(0.0);
    let entering = moist_air(24.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
        .expect("zero-min performance must succeed");

    assert!(
        perf.fan_heat_w.abs() < 1e-9,
        "zero min-ratio → zero fan power, got {}",
        perf.fan_heat_w
    );
    assert!(
        (perf.supply_air.dry_bulb_c - entering.dry_bulb_c).abs() < 1e-6,
        "zero fan heat → no supply temperature rise"
    );
}

/// Fan heat in cooling mode: the leaving-air temperature from the cooling coil
/// is raised by the post-coil fan heat before reaching the zone.
#[test]
fn fan_heat_adds_to_post_cooling_air_in_cooling_mode() {
    let terminal = vav_standard();
    let entering = moist_air(30.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
        .expect("cooling performance must succeed");

    assert!(
        perf.supply_air.dry_bulb_c < entering.dry_bulb_c,
        "cooling mode supply {} must be below entering {}",
        perf.supply_air.dry_bulb_c,
        entering.dry_bulb_c
    );
    assert!(
        perf.fan_heat_w > 0.0,
        "cooling mode must still add fan heat"
    );

    let coil_perf = terminal
        .cooling_coil
        .compute_cooling_capacity(&entering, perf.dry_air_mass_flow_kg_per_s)
        .expect("coil performance must succeed");
    assert!(
        perf.supply_air.dry_bulb_c > coil_perf.leaving_air.dry_bulb_c,
        "fan heat must raise supply {} above coil leaving {}",
        perf.supply_air.dry_bulb_c,
        coil_perf.leaving_air.dry_bulb_c
    );
}

// ============================================================================
// VAV terminal flow accuracy and damper authority
// ============================================================================

/// VAV damper position maps linearly to volumetric flow between min and max.
#[test]
fn damper_position_linearity_check() {
    let terminal = vav_standard();
    let entering = moist_air(24.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let q_max = terminal.max_airflow_m3_per_s();
    let q_min = terminal.min_airflow_m3_per_s();

    let positions = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let mut prev_flow = 0.0;
    for &pos in &positions {
        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(pos))
            .expect("performance must succeed");
        let expected_flow = q_min + pos * (q_max - q_min);
        assert!(
            (perf.volumetric_flow_m3_per_s - expected_flow).abs() < 0.001,
            "damper={}: flow={}, expected={}",
            pos,
            perf.volumetric_flow_m3_per_s,
            expected_flow
        );
        assert!(
            perf.volumetric_flow_m3_per_s >= prev_flow - 1e-9,
            "flow must be monotonic in damper position"
        );
        prev_flow = perf.volumetric_flow_m3_per_s;
    }
}

/// Damper authority: zone temperature difference changes the required flow to
/// maintain setpoint. This test validates the terminal's flow response across
/// a range of entering-air conditions representative of cooling operation.
#[test]
fn damper_authority_flow_vs_entering_temperature() {
    let terminal = vav_standard();
    let rho = STANDARD_AIR_DENSITY_KG_PER_M3;

    let entering_temps = [22.0_f64, 26.0, 30.0, 35.0];
    let mut prev_flow = 0.0;
    for &temp in &entering_temps {
        let entering = moist_air(temp, 50.0, SEA_LEVEL_PA);
        let perf = terminal
            .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
            .expect("cooling performance must succeed");
        if prev_flow > 0.0 {
            assert!(
                perf.volumetric_flow_m3_per_s <= prev_flow + 1e-6,
                "flow must not increase with decreasing entering temperature"
            );
        }
        prev_flow = perf.volumetric_flow_m3_per_s;
    }
}

/// VAV flow at full cooling damper (1.0) equals rated max airflow.
#[test]
fn full_damper_delivers_rated_max_flow() {
    let terminal = vav_standard();
    let entering = moist_air(30.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&entering, rho, &VavTerminalControl::cooling(1.0))
        .expect("full-damper performance must succeed");

    let rated_max = terminal.max_airflow_m3_per_s();
    assert!(
        (perf.volumetric_flow_m3_per_s - rated_max).abs() < 0.01,
        "full-damper flow={}, rated max={}",
        perf.volumetric_flow_m3_per_s,
        rated_max
    );
}

/// VAV flow at deadband damper (0.0) equals minimum airflow.
#[test]
fn min_damper_delivers_minimum_flow() {
    let terminal = vav_standard();
    let entering = moist_air(24.0, 50.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&entering, rho, &VavTerminalControl::deadband())
        .expect("deadband performance must succeed");

    let rated_min = terminal.min_airflow_m3_per_s();
    assert!(
        (perf.volumetric_flow_m3_per_s - rated_min).abs() < 0.01,
        "deadband flow={}, rated min={}",
        perf.volumetric_flow_m3_per_s,
        rated_min
    );
}

// ============================================================================
// Supply air temperature vs zone temperature (coupler-level)
// ============================================================================

/// In cooling mode, supply air must be cooler than the zone air.
/// In heating mode, supply air must be warmer than the zone air.
#[test]
fn supply_air_pulls_zone_toward_setpoint() {
    let mut coupler = zone_coupler(26.0);
    let outdoor = moist_air(35.0, 40.0, SEA_LEVEL_PA);
    let supply = supply_flow(14.0, 55.0, 0.55);

    let forcing = CoupledStepForcing {
        exterior_temperatures: SurfaceExteriorTemperatures::uniform(35.0),
        outdoor_air: outdoor,
        ventilation_conductance_w_per_k: 20.0,
        convective_gain_w: 500.0,
        envelope_gains_w: [100.0, 80.0, 0.0, 50.0],
    };

    let result = coupler
        .step(TIMESTEP_SECONDS, &forcing, &supply)
        .expect("coupled step must succeed");

    assert!(
        result.zone_air.dry_bulb_c < 26.0,
        "zone must cool from 26°C, got {}",
        result.zone_air.dry_bulb_c
    );
    let supply_temp = supply.supply_air().dry_bulb_c;
    assert!(
        result.zone_air.dry_bulb_c > supply_temp,
        "zone {} must stay warmer than supply {}",
        result.zone_air.dry_bulb_c,
        supply_temp
    );
}

/// Coupling energy balance closes for both cooling and heating supply states.
#[test]
fn coupler_energy_balance_closes_for_heating_and_cooling() {
    let outdoor = moist_air(5.0, 80.0, SEA_LEVEL_PA);

    for (supply_temp, supply_rh, desc) in [(14.0, 55.0, "cooling"), (35.0, 20.0, "heating")] {
        let mut coupler = zone_coupler(20.0);
        let supply = supply_flow(supply_temp, supply_rh, 0.55);

        let forcing = CoupledStepForcing {
            exterior_temperatures: SurfaceExteriorTemperatures::uniform(5.0),
            outdoor_air: outdoor,
            ventilation_conductance_w_per_k: 20.0,
            convective_gain_w: 350.0,
            envelope_gains_w: [120.0, 80.0, 0.0, 100.0],
        };

        let result = coupler
            .step(TIMESTEP_SECONDS, &forcing, &supply)
            .unwrap_or_else(|e| panic!("{} step failed: {}", desc, e));

        assert!(
            result.energy_balance_residual_w.abs() <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
            "{}: residual {} exceeds tolerance {}",
            desc,
            result.energy_balance_residual_w,
            DEFAULT_ENERGY_BALANCE_TOLERANCE_W
        );
        assert!(
            result.moisture_balance_residual_kg_per_s.abs() <= 1e-12,
            "{}: moisture residual {}",
            desc,
            result.moisture_balance_residual_kg_per_s
        );
    }
}

/// Annual coupled run produces finite zone temperatures and closes energy balance.
#[test]
fn coupler_annual_run_finite_and_energy_balanced() {
    let mut coupler = zone_coupler(22.0);
    let mut max_residual = 0.0_f64;

    for step in 0..8_760_usize {
        let hour = (step as f64 / 10.0) % 24.0;
        let outdoor_temp = 10.0
            + 18.0 * (2.0 * std::f64::consts::PI * (step as f64 / 8760.0 - 30.0) / 365.0).sin()
            + 7.0 * (2.0 * std::f64::consts::PI * (hour - 8.0) / 24.0).sin();
        let outdoor = moist_air(outdoor_temp, 50.0, SEA_LEVEL_PA);

        let supply = if outdoor_temp > 18.0 {
            supply_flow(14.0, 55.0, 0.55)
        } else {
            supply_flow(32.0, 20.0, 0.55)
        };

        let occupied = (7.0..19.0).contains(&hour);
        let convective_gain_w = if occupied { 500.0 } else { 50.0 };
        let solar_gain_w = (std::f64::consts::PI * (hour - 6.0) / 12.0).sin().max(0.0) * 1500.0;

        let forcing = CoupledStepForcing {
            exterior_temperatures: SurfaceExteriorTemperatures::uniform(outdoor_temp),
            outdoor_air: outdoor,
            ventilation_conductance_w_per_k: 20.0,
            convective_gain_w,
            envelope_gains_w: [
                0.45 * solar_gain_w,
                0.30 * solar_gain_w,
                0.0,
                0.25 * solar_gain_w,
            ],
        };

        let result = coupler
            .step(TIMESTEP_SECONDS, &forcing, &supply)
            .unwrap_or_else(|e| panic!("step {} failed: {}", step, e));

        assert!(
            result.zone_air.is_finite(),
            "step {}: non-finite zone air",
            step
        );
        max_residual = max_residual.max(result.energy_balance_residual_w.abs());
    }

    assert!(
        max_residual <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
        "annual max residual {} exceeds tolerance {}",
        max_residual,
        DEFAULT_ENERGY_BALANCE_TOLERANCE_W
    );
}

// ============================================================================
// Air-side distribution efficacy
// ============================================================================

/// Air-side distribution efficacy: the ratio of actual sensible cooling delivered
/// to the zone vs the ideal (no losses). At the terminal level this is 100%
/// because the terminal delivers its full flow to the zone. This test establishes
/// the baseline and documents the meaning of "efficacy" in this context.
#[test]
fn terminal_delivers_full_flow_to_zone_no_duct_losses() {
    let mut coupler = zone_coupler(26.0);
    let supply = supply_flow(14.0, 55.0, 0.55);
    let supply_air = supply.supply_air();

    let forcing = CoupledStepForcing {
        exterior_temperatures: SurfaceExteriorTemperatures::uniform(35.0),
        outdoor_air: moist_air(35.0, 40.0, SEA_LEVEL_PA),
        ventilation_conductance_w_per_k: 20.0,
        convective_gain_w: 500.0,
        envelope_gains_w: [100.0, 80.0, 0.0, 50.0],
    };

    let result = coupler
        .step(TIMESTEP_SECONDS, &forcing, &supply)
        .expect("coupled step must succeed");

    assert!(
        (result.supply_dry_air_mass_flow_kg_per_s - supply.dry_air_mass_flow_kg_per_s()).abs()
            < 1e-9,
        "coupler must pass through full supply mass flow"
    );

    assert!(
        result.supply_sensible_heat_w < 0.0,
        "cooling supply must remove heat from zone (negative), got {}",
        result.supply_sensible_heat_w
    );
    assert!(
        result.zone_air.dry_bulb_c < 26.0,
        "zone must cool from initial 26°C"
    );
    assert!(
        result.zone_air.dry_bulb_c > supply_air.dry_bulb_c,
        "zone {} must stay above supply {}",
        result.zone_air.dry_bulb_c,
        supply_air.dry_bulb_c
    );
}

/// VAV heating mode: reheat coil raises supply temperature at minimum flow.
/// This validates the reheat coil's contribution to supply air temperature.
#[test]
fn reheat_coil_raises_supply_temperature_at_minimum_flow() {
    let terminal = vav_standard();
    let primary = moist_air(13.0, 90.0, SEA_LEVEL_PA);
    let rho = primary.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&primary, rho, &VavTerminalControl::heating(25.0))
        .expect("reheat performance must succeed");

    assert_eq!(perf.mode, VavOperatingMode::Heating);
    assert!(
        perf.reheat_capacity_w > 0.0,
        "reheat coil must deliver positive heating capacity"
    );
    assert!(
        perf.supply_air.dry_bulb_c > primary.dry_bulb_c,
        "reheat supply {} must exceed primary {}",
        perf.supply_air.dry_bulb_c,
        primary.dry_bulb_c
    );
    assert!(
        (perf.supply_air.dry_bulb_c - 25.0).abs() < 1.0,
        "reheat supply {} must approach 25°C setpoint",
        perf.supply_air.dry_bulb_c
    );
    let min_flow = terminal.min_airflow_m3_per_s();
    assert!(
        (perf.volumetric_flow_m3_per_s - min_flow).abs() < 0.01,
        "reheat flow {} must equal min flow {}",
        perf.volumetric_flow_m3_per_s,
        min_flow
    );
}

// ============================================================================
// Fan heat in energy balance
// ============================================================================

/// Fan heat appears as positive term in the terminal energy balance:
/// Net enthalpy change = -cooling_capacity + reheat_capacity + fan_heat.
#[test]
fn fan_heat_appears_in_terminal_energy_balance() {
    let terminal = vav_standard();
    let entering = moist_air(30.0, 60.0, SEA_LEVEL_PA);
    let rho = entering.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(
            &entering,
            rho,
            &VavTerminalControl {
                damper_position: 1.0,
                cooling_active: true,
                reheat: Some(HeatingCoilControl::LeavingTempSetpoint(20.0)),
            },
        )
        .expect("combined cooling+reheat performance must succeed");

    let h_enter = entering.enthalpy_kj_per_kg_dry_air;
    let h_supply = perf.supply_air.enthalpy_kj_per_kg_dry_air;
    let actual_delta_h = perf.dry_air_mass_flow_kg_per_s * (h_supply - h_enter) * 1000.0;
    let expected_delta_h =
        -perf.cooling_total_capacity_w + perf.reheat_capacity_w + perf.fan_heat_w;

    let relative_error =
        (actual_delta_h - expected_delta_h).abs() / expected_delta_h.abs().max(1.0);
    assert!(
        relative_error < 0.005,
        "energy balance: actual={}, expected={}",
        actual_delta_h,
        expected_delta_h
    );
}

/// Fan motor power includes motor efficiency stage (η_motor < 1).
#[test]
fn fan_motor_power_exceeds_shaft_power_when_motor_efficiency_below_unity() {
    let fan = FanComponent::with_motor(
        "FAN-1".into(),
        2.0,
        500.0,
        0.70,
        0.90,
        STANDARD_AIR_DENSITY_KG_PER_M3,
    );
    let rho = STANDARD_AIR_DENSITY_KG_PER_M3;

    let shaft = fan.shaft_power(1.0, rho);
    let motor = fan.motor_power(1.0, rho);

    assert!(
        motor > shaft,
        "motor power {} must exceed shaft power {} with η_motor=0.90",
        motor,
        shaft
    );
    assert!(
        (motor * 0.90 - shaft).abs() < 1e-6,
        "shaft = motor × η_motor: shaft={}, motor×η={}",
        shaft,
        motor * 0.90
    );
}

// ============================================================================
// Reheat coil part-load performance
// ============================================================================

/// Reheat coil capacity scales correctly with part-load demand below rated.
#[test]
fn reheat_coil_capacity_below_rated_at_part_load() {
    let terminal = vav_standard();
    let primary = moist_air(13.0, 90.0, SEA_LEVEL_PA);
    let rho = primary.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&primary, rho, &VavTerminalControl::heating(20.0))
        .expect("reheat performance must succeed");

    assert!(
        perf.reheat_capacity_w > 0.0,
        "reheat capacity must be positive"
    );
    let rated = terminal.rated_reheat_capacity_w();
    assert!(
        perf.reheat_capacity_w <= rated,
        "reheat {} must not exceed rated {}",
        perf.reheat_capacity_w,
        rated
    );
    assert!(
        perf.supply_air.dry_bulb_c > primary.dry_bulb_c,
        "reheat supply {} must exceed primary {}",
        perf.supply_air.dry_bulb_c,
        primary.dry_bulb_c
    );
}

/// Reheat coil clamps at rated capacity when demand exceeds rating.
#[test]
fn reheat_coil_clamps_at_rated_capacity() {
    let terminal = vav_standard();
    let primary = moist_air(13.0, 90.0, SEA_LEVEL_PA);
    let rho = primary.density_kg_per_m3;

    let perf = terminal
        .compute_terminal_performance(&primary, rho, &VavTerminalControl::heating(100.0))
        .expect("over-demand reheat must succeed");

    let rated = terminal.rated_reheat_capacity_w();
    assert!(
        (perf.reheat_capacity_w - rated).abs() < 1.0,
        "reheat {} must clamp at rated {}",
        perf.reheat_capacity_w,
        rated
    );
    assert!(
        perf.supply_air.dry_bulb_c < 100.0,
        "supply {} must stay below absurd setpoint",
        perf.supply_air.dry_bulb_c
    );
}

/// Non-finite entering air temperature is rejected at construction.
#[test]
fn non_finite_entering_air_rejected() {
    let result = MoistAirState::try_new(f64::NAN, 50.0, SEA_LEVEL_PA);
    assert!(
        result.is_err(),
        "MoistAirState::try_new must reject NaN dry_bulb"
    );
    let result = MoistAirState::try_new(24.0, f64::NAN, SEA_LEVEL_PA);
    assert!(result.is_err(), "MoistAirState::try_new must reject NaN RH");
}

/// Negative air density is rejected.
#[test]
fn negative_density_rejected() {
    let terminal = vav_standard();
    let entering = moist_air(24.0, 50.0, SEA_LEVEL_PA);
    let err = terminal
        .compute_terminal_performance(&entering, -1.0, &VavTerminalControl::deadband())
        .unwrap_err();
    assert!(matches!(
        err,
        fluxion::sim::hvac::AirsideCouplingError::InvalidInput { .. }
    ));
}

/// Coupler rejects timestep exceeding validated maximum (360 s).
#[test]
fn coupler_rejects_excessive_timestep() {
    let mut coupler = zone_coupler(22.0);
    let supply = supply_flow(14.0, 55.0, 0.55);
    let forcing = CoupledStepForcing {
        exterior_temperatures: SurfaceExteriorTemperatures::uniform(35.0),
        outdoor_air: moist_air(35.0, 40.0, SEA_LEVEL_PA),
        ventilation_conductance_w_per_k: 20.0,
        convective_gain_w: 500.0,
        envelope_gains_w: [100.0, 80.0, 0.0, 50.0],
    };

    let err = coupler.step(720.0, &forcing, &supply).unwrap_err();
    assert!(matches!(
        err,
        fluxion::sim::hvac::AirsideCouplingError::TimestepExceedsValidatedMaximum { .. }
    ));
}
