//! Integration tests for the plant-loop equipment models.
//!
//! Tests cover:
//! - PlantComponent trait implementations
//! - CoolingTowerSingleSpeed performance
//! - PumpConstantSpeed and PumpVariableSpeed affinity laws
//! - PlantLoop sequential-iterative solver convergence
//! - Energy balance checks

use fluxion::sim::hvac::plant::cooling_tower::CoolingTowerSingleSpeed;
use fluxion::sim::hvac::plant::fluid_properties;
use fluxion::sim::hvac::plant::plant_component::{
    FluidState, PlantComponent, PlantComponentResult,
};
use fluxion::sim::hvac::plant::plant_loop::{check_energy_balance, PlantLoop};
use fluxion::sim::hvac::plant::pump::{PumpConstantSpeed, PumpVariableSpeed};

// ---------------------------------------------------------------------------
// Helper: constant heat source / sink for loop tests
// ---------------------------------------------------------------------------

struct ConstantHeatSource {
    heat_w: f64,
}

impl PlantComponent for ConstantHeatSource {
    fn id(&self) -> &str {
        "HeatSource"
    }
    fn evaluate(&self, inlet: FluidState, _outdoor_temp: f64, _dt: f64) -> PlantComponentResult {
        let rho = fluid_properties::water_density(inlet.temperature);
        let cp = fluid_properties::water_cp(inlet.temperature);
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

struct ConstantHeatSink {
    heat_w: f64,
}

impl PlantComponent for ConstantHeatSink {
    fn id(&self) -> &str {
        "HeatSink"
    }
    fn evaluate(&self, inlet: FluidState, _outdoor_temp: f64, _dt: f64) -> PlantComponentResult {
        let rho = fluid_properties::water_density(inlet.temperature);
        let cp = fluid_properties::water_cp(inlet.temperature);
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

// ===========================================================================
// Fluid property tests
// ===========================================================================

#[test]
fn water_density_varies_with_temperature() {
    let rho_0 = fluid_properties::water_density(0.0);
    let rho_25 = fluid_properties::water_density(25.0);
    let rho_60 = fluid_properties::water_density(60.0);
    assert!(rho_0 > rho_25, "density at 0 °C should exceed 25 °C");
    assert!(rho_25 > rho_60, "density at 25 °C should exceed 60 °C");
}

#[test]
fn water_cp_positive_finite() {
    for t in 0..=100 {
        let cp = fluid_properties::water_cp(t as f64);
        assert!(cp > 4000.0 && cp < 4300.0, "water_cp({t}) = {cp}");
    }
}

// ===========================================================================
// CoolingTower tests
// ===========================================================================

#[test]
fn cooling_tower_rejects_heat_at_design_conditions() {
    let tower = CoolingTowerSingleSpeed::new("CT-INT-1".to_string(), 500_000.0, 5.0, 10.0, 0.02);
    let inlet = FluidState {
        temperature: 35.0,
        flow_rate: 0.02,
    };
    let result = tower.evaluate(inlet, 25.0, 3600.0);
    assert!(
        result.outlet.temperature < inlet.temperature,
        "outlet {} not cooler than inlet {}",
        result.outlet.temperature,
        inlet.temperature
    );
    assert!(
        result.heat_transfer_w < 0.0,
        "heat should be negative (rejected)"
    );
    assert!(result.electrical_power_w > 0.0, "fan should consume power");
}

#[test]
fn cooling_tower_outlet_cannot_exceed_inlet() {
    let tower = CoolingTowerSingleSpeed::new("CT-INT-2".to_string(), 500_000.0, 3.0, 8.0, 0.015);
    let inlet = FluidState {
        temperature: 40.0,
        flow_rate: 0.015,
    };
    let result = tower.evaluate(inlet, 35.0, 3600.0);
    assert!(
        result.outlet.temperature <= inlet.temperature,
        "outlet {} warmer than inlet {}",
        result.outlet.temperature,
        inlet.temperature
    );
}

#[test]
fn cooling_tower_approach_temperature() {
    let tower = CoolingTowerSingleSpeed::new(
        "CT-INT-3".to_string(),
        300_000.0,
        4.0, // design approach
        6.0, // design range
        0.012,
    );
    let inlet = FluidState {
        temperature: 31.0,
        flow_rate: 0.012,
    };
    let result = tower.evaluate(inlet, 20.0, 3600.0);
    // At design conditions, outlet should approach 20 + 4 = 24 °C
    assert!(
        result.outlet.temperature < 28.0,
        "outlet {} too warm for 4 °C approach",
        result.outlet.temperature
    );
}

#[test]
fn cooling_tower_no_flow_zero_power() {
    let tower = CoolingTowerSingleSpeed::new("CT-INT-4".to_string(), 500_000.0, 5.0, 10.0, 0.02);
    let inlet = FluidState {
        temperature: 35.0,
        flow_rate: 0.0,
    };
    let result = tower.evaluate(inlet, 25.0, 3600.0);
    assert_eq!(result.electrical_power_w, 0.0);
    assert_eq!(result.heat_transfer_w, 0.0);
}

// ===========================================================================
// Pump tests
// ===========================================================================

#[test]
fn constant_speed_pump_power_scales_with_flow() {
    let pump = PumpConstantSpeed::new("PUMP-INT-1".to_string(), 0.01, 20.0, 0.75, 0.90);
    let inlet_full = FluidState {
        temperature: 20.0,
        flow_rate: 0.01,
    };
    let inlet_half = FluidState {
        temperature: 20.0,
        flow_rate: 0.005,
    };
    let r_full = pump.evaluate(inlet_full, 20.0, 60.0);
    let r_half = pump.evaluate(inlet_half, 20.0, 60.0);
    assert!(
        r_full.electrical_power_w > r_half.electrical_power_w,
        "full-flow power {} should exceed half-flow {}",
        r_full.electrical_power_w,
        r_half.electrical_power_w
    );
}

#[test]
fn variable_speed_pump_affinity_cubic_power() {
    let mut pump = PumpVariableSpeed::new("PUMP-INT-2".to_string(), 0.01, 20.0, 0.75, 0.90);
    let inlet = FluidState {
        temperature: 20.0,
        flow_rate: 0.01,
    };
    pump.set_speed(1.0);
    let r1 = pump.evaluate(inlet, 20.0, 60.0);

    pump.set_speed(0.5);
    let r2 = pump.evaluate(inlet, 20.0, 60.0);

    // Power at 50% speed should be less than 50% of full speed
    assert!(
        r2.electrical_power_w < r1.electrical_power_w * 0.5,
        "VSP power ratio {} not cubic enough",
        r2.electrical_power_w / r1.electrical_power_w
    );
}

#[test]
fn pump_outlet_temperature_unchanged() {
    let pump = PumpConstantSpeed::new("PUMP-INT-3".to_string(), 0.01, 20.0, 0.75, 0.90);
    let inlet = FluidState {
        temperature: 18.5,
        flow_rate: 0.01,
    };
    let result = pump.evaluate(inlet, 20.0, 60.0);
    assert!(
        (result.outlet.temperature - 18.5).abs() < 0.01,
        "pump changed temperature from 18.5 to {}",
        result.outlet.temperature
    );
}

// ===========================================================================
// PlantLoop solver tests
// ===========================================================================

#[test]
fn empty_loop_converges() {
    let loop_ = PlantLoop::new("EmptyLoop".to_string(), 7.0);
    let inlet = FluidState {
        temperature: 12.0,
        flow_rate: 0.01,
    };
    let result = loop_.solve(&[], &[], inlet, 25.0, 3600.0);
    assert!(result.converged);
    assert!(result.iterations <= 2);
}

#[test]
fn heating_loop_converges() {
    let loop_ = PlantLoop::new("HWLoop".to_string(), 70.0);
    let boiler = ConstantHeatSource { heat_w: 50_000.0 };
    let pump = PumpConstantSpeed::new("HW-Pump".to_string(), 0.002, 15.0, 0.70, 0.90);
    // Balanced heat sink on demand side so the solver can find equilibrium.
    let sink = ConstantHeatSink { heat_w: 50_000.0 };
    let inlet = FluidState {
        temperature: 50.0,
        flow_rate: 0.002,
    };
    let result = loop_.solve(&[&boiler], &[&pump, &sink], inlet, 25.0, 3600.0);
    assert!(
        result.converged,
        "heating loop did not converge in {} iters",
        result.iterations
    );
    // Water was heated above inlet temperature by the boiler.
    assert!(
        result.supply_header_temperature > inlet.temperature,
        "supply temp {} should be above inlet {}",
        result.supply_header_temperature,
        inlet.temperature
    );
    assert!(result.total_electrical_power_w > 0.0);
}

#[test]
fn cooling_loop_converges() {
    let loop_ = PlantLoop::new("CHWLoop".to_string(), 7.0);
    // Chiller: removes heat from the chilled-water supply side.
    let chiller = ConstantHeatSink { heat_w: 100_000.0 };
    let pump = PumpConstantSpeed::new("CHW-Pump".to_string(), 0.015, 18.0, 0.72, 0.90);
    let inlet = FluidState {
        temperature: 12.0,
        flow_rate: 0.015,
    };
    let result = loop_.solve(&[&chiller], &[&pump], inlet, 25.0, 3600.0);
    assert!(
        result.converged,
        "cooling loop did not converge in {} iters",
        result.iterations
    );
    assert!(
        result.supply_header_temperature < inlet.temperature,
        "supply temp {} not below inlet {}",
        result.supply_header_temperature,
        inlet.temperature
    );
}

#[test]
fn multi_component_loop_converges() {
    let loop_ = PlantLoop::new("MultiComp".to_string(), 70.0);
    let boiler1 = ConstantHeatSource { heat_w: 30_000.0 };
    let boiler2 = ConstantHeatSource { heat_w: 20_000.0 };
    let pump = PumpConstantSpeed::new("Pump-A".to_string(), 0.003, 15.0, 0.70, 0.90);
    // Match total boiler output (50 kW) so the loop reaches equilibrium.
    let sink = ConstantHeatSink { heat_w: 50_000.0 };

    let inlet = FluidState {
        temperature: 45.0,
        flow_rate: 0.003,
    };
    let result = loop_.solve(&[&boiler1, &boiler2], &[&pump, &sink], inlet, 25.0, 3600.0);
    assert!(
        result.converged,
        "multi-component loop did not converge in {} iters",
        result.iterations
    );
    // Both boilers contribute heat; water should be warmer than inlet.
    assert!(
        result.supply_header_temperature > inlet.temperature,
        "supply temp {} should be above inlet {}",
        result.supply_header_temperature,
        inlet.temperature
    );
}

#[test]
fn loop_energy_balance_check() {
    let result = fluxion::sim::hvac::plant::PlantLoopResult {
        supply_header_temperature: 70.0,
        demand_header_temperature: 50.0,
        total_electrical_power_w: 1_500.0,
        total_heat_transfer_w: 48_500.0,
        iterations: 5,
        converged: true,
    };
    // Total should be 50,000 W
    assert!(check_energy_balance(&result, 50_000.0, 500.0).is_ok());
    // Should fail if expected is way off
    assert!(check_energy_balance(&result, 100_000.0, 500.0).is_err());
}

#[test]
fn loop_does_not_panic_on_zero_flow() {
    let loop_ = PlantLoop::new("ZeroFlow".to_string(), 7.0);
    let tower = CoolingTowerSingleSpeed::new("CT-Zero".to_string(), 500_000.0, 5.0, 10.0, 0.02);
    let inlet = FluidState {
        temperature: 35.0,
        flow_rate: 0.0,
    };
    let result = loop_.solve(&[&tower], &[], inlet, 25.0, 3600.0);
    assert!(result.converged);
}
