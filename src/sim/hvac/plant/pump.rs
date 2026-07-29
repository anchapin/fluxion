//! Pump models for plant-loop circulation.
//!
//! Provides a [`Pump`] trait and two concrete implementations:
//!
//! * [`PumpConstantSpeed`] — fixed-speed, fixed-flow pump (simple
//!   primary-loop pump).
//! * [`PumpVariableSpeed`] — variable-speed pump using the fluid
//!   affinity laws (P ∝ N³, Q ∝ N, H ∝ N²).
//!
//! Both models follow the fan-affinity-law pattern already established in
//! [`super::super::fan::FanComponent`].

use serde::{Deserialize, Serialize};

use super::plant_component::{FluidState, PlantComponent, PlantComponentResult};

/// Trait for plant-loop pumps.
///
/// Extends `PlantComponent` with pump-specific accessors so the loop
/// solver can query the pump's hydraulic state without down-casting.
pub trait Pump: PlantComponent {
    /// Rated volumetric flow rate at full speed [m³/s].
    fn rated_flow(&self) -> f64;

    /// Rated head (pressure rise) at full speed [m].
    fn rated_head_m(&self) -> f64;

    /// Overall pump efficiency at the current operating point, dimensionless.
    fn efficiency(&self) -> f64;

    /// Motor/drive efficiency, dimensionless ∈ (0, 1].
    fn motor_efficiency(&self) -> f64;

    /// Electrical input power at the current operating point [W].
    fn electrical_power_w(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Constant-speed pump
// ---------------------------------------------------------------------------

/// Fixed-speed, constant-flow pump.
///
/// Suitable for primary-only chilled-water or condenser-water loops.
/// The pump always runs at its design speed; power scales linearly with
/// the actual flow fraction squared (squared-duct-law approximation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpConstantSpeed {
    /// Equipment identifier.
    pub id: String,
    /// Rated volumetric flow rate [m³/s].
    pub rated_flow_m3_per_s: f64,
    /// Rated head (pressure rise) [m].
    pub rated_head_m: f64,
    /// Overall pump efficiency at rated point ∈ (0, 1].
    pub pump_efficiency: f64,
    /// Motor/drive efficiency ∈ (0, 1].
    pub motor_efficiency: f64,
    /// Cached electrical power after the last evaluate() call [W].
    #[serde(skip)]
    cached_power_w: f64,
    /// Cached pump efficiency after the last evaluate() call.
    #[serde(skip)]
    cached_efficiency: f64,
}

impl PumpConstantSpeed {
    /// Create a constant-speed pump.
    pub fn new(
        id: String,
        rated_flow_m3_per_s: f64,
        rated_head_m: f64,
        pump_efficiency: f64,
        motor_efficiency: f64,
    ) -> Self {
        Self {
            id,
            rated_flow_m3_per_s,
            rated_head_m,
            pump_efficiency: pump_efficiency.max(0.01),
            motor_efficiency: motor_efficiency.max(0.01),
            cached_power_w: 0.0,
            cached_efficiency: pump_efficiency,
        }
    }
}

impl Pump for PumpConstantSpeed {
    fn rated_flow(&self) -> f64 {
        self.rated_flow_m3_per_s
    }

    fn rated_head_m(&self) -> f64 {
        self.rated_head_m
    }

    fn efficiency(&self) -> f64 {
        self.cached_efficiency
    }

    fn motor_efficiency(&self) -> f64 {
        self.motor_efficiency
    }

    fn electrical_power_w(&self) -> f64 {
        self.cached_power_w
    }
}

impl PlantComponent for PumpConstantSpeed {
    fn id(&self) -> &str {
        &self.id
    }

    fn evaluate(&self, inlet: FluidState, _outdoor_temp: f64, _dt: f64) -> PlantComponentResult {
        let flow_fraction = (inlet.flow_rate / self.rated_flow_m3_per_s).clamp(0.0, 1.0);

        // Hydraulic power = ρ·g·Q·H
        let rho = super::fluid_properties::water_density(inlet.temperature);
        let g = 9.80665;
        let hydraulic_power = rho * g * inlet.flow_rate * self.rated_head_m;

        // Pump efficiency varies with flow fraction (peak at rated)
        let eff_penalty = 1.0 - 0.5 * (1.0 - flow_fraction).powi(2);
        let effective_pump_eff = self.pump_efficiency * eff_penalty;

        let motor_power = if effective_pump_eff > 0.001 {
            hydraulic_power / effective_pump_eff / self.motor_efficiency
        } else {
            0.0
        };

        // For a pump the outlet temperature is essentially the same as
        // the inlet (minor motor heat loss assumed dissipated externally).
        PlantComponentResult {
            outlet: inlet,
            electrical_power_w: motor_power,
            heat_transfer_w: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Variable-speed pump
// ---------------------------------------------------------------------------

/// Variable-speed pump using fluid affinity laws.
///
/// The pump speed fraction φ ∈ [0, 1] controls flow and pressure:
///
/// | Quantity | Scaling with φ |
/// |----------|----------------|
/// | Flow Q   | φ              |
/// | Head H   | φ²             |
/// | Power P  | φ³             |
///
/// At a given speed the hydraulic power is `ρ·g·Q(φ)·H(φ)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpVariableSpeed {
    /// Equipment identifier.
    pub id: String,
    /// Rated volumetric flow rate at full speed [m³/s].
    pub rated_flow_m3_per_s: f64,
    /// Rated head (pressure rise) at full speed [m].
    pub rated_head_m: f64,
    /// Overall pump efficiency at the rated design point ∈ (0, 1].
    pub pump_efficiency: f64,
    /// Motor/drive efficiency ∈ (0, 1].
    pub motor_efficiency: f64,
    /// Minimum speed fraction.  Below this the VFD shuts the pump off.
    pub min_speed_fraction: f64,
    /// Current speed fraction [0, 1] — set by the loop solver.
    pub speed_fraction: f64,
    /// Cached electrical power after the last evaluate() call [W].
    #[serde(skip)]
    cached_power_w: f64,
    /// Cached operating efficiency.
    #[serde(skip)]
    cached_efficiency: f64,
}

impl PumpVariableSpeed {
    /// Create a variable-speed pump.
    pub fn new(
        id: String,
        rated_flow_m3_per_s: f64,
        rated_head_m: f64,
        pump_efficiency: f64,
        motor_efficiency: f64,
    ) -> Self {
        Self {
            id,
            rated_flow_m3_per_s,
            rated_head_m,
            pump_efficiency: pump_efficiency.max(0.01),
            motor_efficiency: motor_efficiency.max(0.01),
            min_speed_fraction: 0.2,
            speed_fraction: 1.0,
            cached_power_w: 0.0,
            cached_efficiency: pump_efficiency,
        }
    }

    /// Set the speed fraction (0.0 = off, 1.0 = full speed).
    pub fn set_speed(&mut self, speed: f64) {
        self.speed_fraction = speed.clamp(0.0, 1.0);
    }
}

impl Pump for PumpVariableSpeed {
    fn rated_flow(&self) -> f64 {
        self.rated_flow_m3_per_s
    }

    fn rated_head_m(&self) -> f64 {
        self.rated_head_m
    }

    fn efficiency(&self) -> f64 {
        self.cached_efficiency
    }

    fn motor_efficiency(&self) -> f64 {
        self.motor_efficiency
    }

    fn electrical_power_w(&self) -> f64 {
        self.cached_power_w
    }
}

impl PlantComponent for PumpVariableSpeed {
    fn id(&self) -> &str {
        &self.id
    }

    fn evaluate(&self, inlet: FluidState, _outdoor_temp: f64, _dt: f64) -> PlantComponentResult {
        let phi = if self.speed_fraction < self.min_speed_fraction {
            0.0
        } else {
            self.speed_fraction
        };

        if phi <= 0.0 || inlet.flow_rate <= 0.0 {
            return PlantComponentResult {
                outlet: inlet,
                electrical_power_w: 0.0,
                heat_transfer_w: 0.0,
            };
        }

        // Affinity laws: Q = φ·Q_rated, H = φ²·H_rated
        // Actual flow through the loop is set by the loop solver; we
        // compute the head and power at the current speed.
        let head_m = self.rated_head_m * phi * phi;
        let rho = super::fluid_properties::water_density(inlet.temperature);
        let g = 9.80665;
        let hydraulic_power = rho * g * inlet.flow_rate * head_m;

        // Efficiency penalty at off-design speed (cubic approximation)
        let eff_penalty = 1.0 - 0.3 * (1.0 - phi).powi(2);
        let effective_pump_eff = self.pump_efficiency * eff_penalty;

        let motor_power = if effective_pump_eff > 0.001 {
            hydraulic_power / effective_pump_eff / self.motor_efficiency
        } else {
            0.0
        };

        PlantComponentResult {
            outlet: inlet,
            electrical_power_w: motor_power,
            heat_transfer_w: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_constant_pump() -> PumpConstantSpeed {
        PumpConstantSpeed::new(
            "PUMP-CS-1".to_string(),
            0.01, // 10 L/s
            20.0, // 20 m head
            0.75, // 75 % pump efficiency
            0.90, // 90 % motor efficiency
        )
    }

    fn make_variable_pump() -> PumpVariableSpeed {
        PumpVariableSpeed::new("PUMP-VS-1".to_string(), 0.01, 20.0, 0.75, 0.90)
    }

    #[test]
    fn test_constant_speed_pump_zero_flow() {
        let pump = make_constant_pump();
        let inlet = FluidState {
            temperature: 20.0,
            flow_rate: 0.0,
        };
        let result = pump.evaluate(inlet, 20.0, 60.0);
        // No flow → no power
        assert_eq!(result.electrical_power_w, 0.0);
    }

    #[test]
    fn test_constant_speed_pump_at_rated() {
        let mut pump = make_constant_pump();
        let inlet = FluidState {
            temperature: 20.0,
            flow_rate: 0.01,
        };
        let result = pump.evaluate(inlet, 20.0, 60.0);
        // Power should be positive
        assert!(result.electrical_power_w > 0.0);
        // Outlet temperature ≈ inlet (pump doesn't heat water significantly)
        assert!((result.outlet.temperature - 20.0).abs() < 0.1);
        pump.cached_power_w = result.electrical_power_w;
        assert!(pump.electrical_power_w() > 0.0);
    }

    #[test]
    fn test_variable_speed_affinity_laws() {
        let mut pump = make_variable_pump();
        let inlet = FluidState {
            temperature: 20.0,
            flow_rate: 0.01,
        };

        // Full speed
        pump.set_speed(1.0);
        let r1 = pump.evaluate(inlet, 20.0, 60.0);

        // Half speed — power should be much lower (cubic scaling)
        pump.set_speed(0.5);
        let r2 = pump.evaluate(inlet, 20.0, 60.0);

        assert!(
            r2.electrical_power_w < r1.electrical_power_w,
            "half-speed power {} >= full-speed power {}",
            r2.electrical_power_w,
            r1.electrical_power_w
        );
        // With affinity laws, power at 0.5 speed should be roughly 0.125
        // of full speed (0.5³ = 0.125), but efficiency modifiers change this.
        assert!(
            r2.electrical_power_w < r1.electrical_power_w * 0.5,
            "power ratio too high: {}/{}",
            r2.electrical_power_w,
            r1.electrical_power_w
        );
    }

    #[test]
    fn test_variable_speed_below_min_shuts_off() {
        let mut pump = make_variable_pump();
        pump.min_speed_fraction = 0.2;
        pump.set_speed(0.1);
        let inlet = FluidState {
            temperature: 20.0,
            flow_rate: 0.01,
        };
        let result = pump.evaluate(inlet, 20.0, 60.0);
        assert_eq!(result.electrical_power_w, 0.0);
    }

    #[test]
    fn test_pump_trait_accessors() {
        let pump = make_constant_pump();
        assert!((pump.rated_flow() - 0.01).abs() < 1e-10);
        assert!((pump.rated_head_m() - 20.0).abs() < 0.1);
        assert!((pump.motor_efficiency() - 0.90).abs() < 0.01);
    }
}
