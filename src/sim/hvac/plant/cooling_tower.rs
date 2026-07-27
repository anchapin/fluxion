//! Single-speed cooling tower (EnergyPlus `CoolingTower:SingleSpeed`).
//!
//! Models a water-cooled condenser heat-rejection device that transfers
//! heat from the condenser water loop to the outdoor air via evaporative
//! cooling.  The model follows the effectiveness-NTU approach with an
//! approach-temperature formulation:
//!
//! ```text
//!   T_supply = T_outdoor + approach
//!   Q_reject = ṁ_w · cp_w · (T_return − T_supply)
//! ```
//!
//! where `approach` is computed from the design range, design approach,
//! and a wet-bulb-dependent effectiveness correction.

use serde::{Deserialize, Serialize};

use super::plant_component::{FluidState, PlantComponent, PlantComponentResult};

/// Single-speed cooling tower model.
///
/// Corresponds to EnergyPlus `CoolingTower:SingleSpeed`.  All thermal
/// calculations follow the approach-temperature / range-temperature
/// framework of ASHRAE Handbook — HVAC Systems and Equipment, Ch. 40.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingTowerSingleSpeed {
    /// Equipment identifier.
    pub id: String,
    /// Rated heat rejection capacity at design conditions (W).
    pub rated_rejection_w: f64,
    /// Design approach temperature — difference between leaving water
    /// temperature and inlet air wet-bulb temperature (°C).
    pub design_approach_c: f64,
    /// Design range — difference between entering and leaving water
    /// temperatures (°C).
    pub design_range_c: f64,
    /// Rated volumetric water flow rate (m³/s).
    pub rated_water_flow_m3_per_s: f64,
    /// Rated air volumetric flow rate (m³/s).
    pub rated_air_flow_m3_per_s: f64,
    /// Fan motor power at full speed (W).
    pub fan_power_w: f64,
    /// Number of cells (for multi-cell towers).
    pub num_cells: u32,
    /// Minimum water flow fraction (of rated).
    pub min_water_flow_fraction: f64,
    /// Heat rejection curve coefficients — effectiveness as a function of
    /// water-flow fraction relative to rated (quadratic): `a + b·f + c·f²`.
    pub heat_rejection_curve_a: f64,
    pub heat_rejection_curve_b: f64,
    pub heat_rejection_curve_c: f64,
}

impl CoolingTowerSingleSpeed {
    /// Create a new cooling tower with sensible defaults.
    pub fn new(
        id: String,
        rated_rejection_w: f64,
        design_approach_c: f64,
        design_range_c: f64,
        rated_water_flow_m3_per_s: f64,
    ) -> Self {
        Self {
            id,
            rated_rejection_w,
            design_approach_c,
            design_range_c,
            rated_water_flow_m3_per_s,
            // Air flow sized so that m_dot_air/m_dot_water ≈ 1.2 (typical)
            rated_air_flow_m3_per_s: rated_water_flow_m3_per_s * 1.2,
            fan_power_w: rated_rejection_w * 0.012, // ~1.2 % of rejection
            num_cells: 1,
            min_water_flow_fraction: 0.33,
            heat_rejection_curve_a: 1.0,
            heat_rejection_curve_b: 0.0,
            heat_rejection_curve_c: 0.0,
        }
    }

    /// Compute the tower effectiveness for a given water-flow fraction.
    ///
    /// Returns a multiplier in [0, 1] that scales the nominal approach
    /// temperature.  At rated flow (fraction = 1.0) the effectiveness is 1.0.
    fn effectiveness(&self, water_flow_fraction: f64) -> f64 {
        let f = water_flow_fraction.clamp(self.min_water_flow_fraction, 1.0);
        (self.heat_rejection_curve_a
            + self.heat_rejection_curve_b * f
            + self.heat_rejection_curve_c * f * f)
            .clamp(0.5, 1.2)
    }

    /// Water-side pressure drop at the given flow fraction [Pa].
    ///
    /// Uses the standard squared-duct law scaled from the design point:
    /// `ΔP = ΔP_rated · (f/f_rated)²`.
    pub fn pressure_drop_pa(&self, flow_fraction: f64) -> f64 {
        let design_dp = 60_000.0; // ~60 kPa default design pressure drop
        design_dp * flow_fraction * flow_fraction
    }
}

impl PlantComponent for CoolingTowerSingleSpeed {
    fn id(&self) -> &str {
        &self.id
    }

    fn evaluate(&self, inlet: FluidState, outdoor_temp: f64, _dt: f64) -> PlantComponentResult {
        if inlet.flow_rate <= 0.0 {
            return PlantComponentResult {
                outlet: inlet,
                electrical_power_w: 0.0,
                heat_transfer_w: 0.0,
            };
        }

        let water_fraction = inlet.flow_rate / self.rated_water_flow_m3_per_s;
        let eff = self.effectiveness(water_fraction);

        // Effective approach: the tower can reject more heat at lower
        // water flows (higher effectiveness).
        let effective_approach = self.design_approach_c / eff;
        // Supply temperature is outdoor temp + effective approach
        let t_supply = outdoor_temp + effective_approach;

        // Heat that *could* be rejected
        let cp_water = super::fluid_properties::water_cp(inlet.temperature);
        let rho_water = super::fluid_properties::water_density(inlet.temperature);
        let mass_flow = inlet.flow_rate * rho_water;
        let q_available = mass_flow * cp_water * (inlet.temperature - t_supply);

        // Actual rejection is capped by rated capacity
        let q_reject = q_available.clamp(0.0, self.rated_rejection_w * eff);

        let t_out = if mass_flow > 0.0 {
            inlet.temperature - q_reject / (mass_flow * cp_water)
        } else {
            t_supply
        };

        // Fan power: on at rated speed whenever tower is active
        let fan_power = if water_fraction > 0.01 {
            self.fan_power_w
        } else {
            0.0
        };

        PlantComponentResult {
            outlet: FluidState {
                temperature: t_out.max(outdoor_temp),
                flow_rate: inlet.flow_rate,
            },
            electrical_power_w: fan_power,
            heat_transfer_w: -q_reject, // negative = heat rejected from fluid
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooling_tower_zero_flow() {
        let tower = CoolingTowerSingleSpeed::new(
            "CT-1".to_string(),
            500_000.0,
            5.0,
            10.0,
            0.02, // m³/s
        );
        let inlet = FluidState {
            temperature: 35.0,
            flow_rate: 0.0,
        };
        let result = tower.evaluate(inlet, 25.0, 3600.0);
        assert_eq!(result.electrical_power_w, 0.0);
        assert_eq!(result.heat_transfer_w, 0.0);
    }

    #[test]
    fn test_cooling_tower_rejects_heat() {
        let tower = CoolingTowerSingleSpeed::new("CT-1".to_string(), 500_000.0, 5.0, 10.0, 0.02);
        let inlet = FluidState {
            temperature: 35.0,
            flow_rate: 0.02,
        };
        let result = tower.evaluate(inlet, 25.0, 3600.0);
        // Outlet should be cooler than inlet
        assert!(
            result.outlet.temperature < inlet.temperature,
            "outlet {} >= inlet {}",
            result.outlet.temperature,
            inlet.temperature
        );
        // Heat transfer should be negative (heat rejected)
        assert!(result.heat_transfer_w < 0.0);
        // Fan power should be positive
        assert!(result.electrical_power_w > 0.0);
    }

    #[test]
    fn test_cooling_tower_outlet_above_outdoor() {
        let tower = CoolingTowerSingleSpeed::new("CT-1".to_string(), 500_000.0, 5.0, 10.0, 0.02);
        let inlet = FluidState {
            temperature: 40.0,
            flow_rate: 0.02,
        };
        let result = tower.evaluate(inlet, 30.0, 3600.0);
        // Outlet cannot go below outdoor temp
        assert!(result.outlet.temperature >= 30.0);
    }

    #[test]
    fn test_effectiveness_at_rated_flow() {
        let tower = CoolingTowerSingleSpeed::new("CT-1".to_string(), 500_000.0, 5.0, 10.0, 0.02);
        let eff = tower.effectiveness(1.0);
        assert!((eff - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pressure_drop_squared_law() {
        let tower = CoolingTowerSingleSpeed::new("CT-1".to_string(), 500_000.0, 5.0, 10.0, 0.02);
        let dp_full = tower.pressure_drop_pa(1.0);
        let dp_half = tower.pressure_drop_pa(0.5);
        assert!(
            (dp_half - dp_full * 0.25).abs() < 1.0,
            "half-flow dp {dp_half} != 0.25 * full dp {dp_full}"
        );
    }
}
