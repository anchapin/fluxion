//! Commercial Refrigeration System Models
//!
//! This module provides models for commercial refrigeration systems including
//! walk-in coolers, walk-in freezers, compressor racks, and air-cooled condensers.
//!
//! These components model the heat extraction from refrigerated spaces and the
//! power consumption of the refrigeration system.
//!
//! ## EnergyPlus Mapping
//!
//! | Component | EnergyPlus Object |
//! |-----------|-------------------|
//! | WalkInCooler | Refrigeration:Case |
//! | WalkInFreezer | Refrigeration:ZoneCase |
//! | CompressorRack | Refrigeration:CompressorRack |
//! | AirCooledCondenser | Refrigeration:CaseAndWalkIn |

use serde::{Deserialize, Serialize};

/// Standard refrigeration cycle constants
pub mod constants {
    /// Ratio of compressor power to refrigeration load at rated conditions (W/W)
    /// Typical values: 0.3-0.5 for commercial refrigeration
    pub const DEFAULT_COMPRESSOR_COP: f64 = 2.5;

    /// Heat of extraction ratio for walk-in coolers
    /// Fraction of heat removed that is due to latent cooling (moisture removal)
    pub const DEFAULT_LATENT_HEAT_RATIO: f64 = 0.25;

    /// Design evaporator temperature for coolers (°C)
    pub const DEFAULT_COOLER_EVAPORATOR_TEMP: f64 = 0.0;

    /// Design evaporator temperature for freezers (°C)
    pub const DEFAULT_FREEZER_EVAPORATOR_TEMP: f64 = -25.0;

    /// Design condenser temperature for air-cooled condensers (°C)
    pub const DEFAULT_CONDENSER_AIR_TEMP: f64 = 35.0;

    /// Minimum ambient temperature for condenser operation (°C)
    pub const MIN_CONDENSER_AMBIENT_TEMP: f64 = -30.0;

    /// Maximum ambient temperature for condenser operation (°C)
    pub const MAX_CONDENSER_AMBIENT_TEMP: f64 = 55.0;
}

/// Operating mode for refrigeration systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefrigerationMode {
    /// System is actively cooling
    Cooling,
    /// System is off
    Off,
    /// System is in defrost mode
    Defrost,
}

/// Walk-in cooler (refrigerated display case).
///
/// Corresponds to EnergyPlus `Refrigeration:Case`.
/// Models a refrigerated case that maintains product temperature typically
/// between 0°C and 5°C.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkInCooler {
    /// Equipment identifier
    pub id: String,
    /// Designed cooling capacity at reference conditions (W)
    pub rated_capacity_w: f64,
    /// Operating temperature setpoint (°C)
    pub temperature_setpoint_c: f64,
    /// Temperature differential for control (K)
    pub temperature_differential_k: f64,
    /// Rated ambient temperature for capacity rating (°C)
    pub rated_ambient_temp_c: f64,
    /// Rated internal temperature for capacity rating (°C)
    pub rated_internal_temp_c: f64,
    /// Current operating mode
    pub mode: RefrigerationMode,
    /// Latent heat removal fraction (0-1)
    pub latent_heat_ratio: f64,
    /// Fan power consumption (W)
    pub fan_power_w: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Current refrigeration load (W)
    pub current_load_w: f64,
    /// Total heat removed this timestep (J)
    pub total_heat_removed_j: f64,
    /// Total energy consumed this timestep (J)
    pub total_energy_consumed_j: f64,
}

impl WalkInCooler {
    /// Create a new walk-in cooler with default parameters.
    pub fn new(id: String, rated_capacity_w: f64) -> Self {
        Self {
            id,
            rated_capacity_w,
            temperature_setpoint_c: 3.0, // Typical cooler temperature
            temperature_differential_k: 2.0,
            rated_ambient_temp_c: 25.0,
            rated_internal_temp_c: 3.0,
            mode: RefrigerationMode::Off,
            latent_heat_ratio: constants::DEFAULT_LATENT_HEAT_RATIO,
            fan_power_w: rated_capacity_w * 0.02, // ~2% of capacity
            current_plr: 0.0,
            current_load_w: 0.0,
            total_heat_removed_j: 0.0,
            total_energy_consumed_j: 0.0,
        }
    }

    /// Calculate the sensible cooling capacity at current conditions.
    ///
    /// Capacity decreases with higher ambient temperature due to increased
    /// thermal gains through the cabinet insulation.
    pub fn sensible_capacity_at_conditions(
        &self,
        internal_temp_c: f64,
        ambient_temp_c: f64,
    ) -> f64 {
        // Temperature lift effect: capacity decreases as ambient-to-internal
        // temperature difference increases
        let temp_diff = (ambient_temp_c - internal_temp_c).max(0.0);
        let rated_temp_diff = self.rated_ambient_temp_c - self.rated_internal_temp_c;
        let capacity_factor = 1.0 - (temp_diff - rated_temp_diff) * 0.01;
        self.rated_capacity_w * capacity_factor.max(0.3)
    }

    /// Calculate the total cooling load including latent heat.
    ///
    /// The total load includes sensible cooling plus latent moisture removal.
    pub fn total_cooling_load(&self, sensible_load_w: f64) -> f64 {
        // Latent heat adds to the sensible load
        sensible_load_w * (1.0 + self.latent_heat_ratio)
    }

    /// Update the cooler state based on current conditions.
    ///
    /// Returns the heat removed (W) and determines if the system should be running.
    pub fn update(
        &mut self,
        internal_temp_c: f64,
        ambient_temp_c: f64,
        sensible_load_w: f64,
        _dt: f64,
    ) -> f64 {
        let total_load = self.total_cooling_load(sensible_load_w);
        let capacity = self.sensible_capacity_at_conditions(internal_temp_c, ambient_temp_c);
        self.current_load_w = total_load;

        // Determine operating mode based on temperature
        if internal_temp_c > self.temperature_setpoint_c + self.temperature_differential_k {
            self.mode = RefrigerationMode::Cooling;
        } else if internal_temp_c < self.temperature_setpoint_c - self.temperature_differential_k {
            self.mode = RefrigerationMode::Off;
        } else {
            // In deadband - maintain current state but don't increase load
            if self.mode == RefrigerationMode::Cooling && self.current_plr > 0.0 {
                // Continue running at reduced capacity
            } else {
                self.mode = RefrigerationMode::Off;
            }
        }

        // Calculate part-load ratio
        if self.mode == RefrigerationMode::Cooling && capacity > 0.0 {
            self.current_plr = (total_load / capacity).clamp(0.0, 1.0);
        } else {
            self.current_plr = 0.0;
        }

        // Heat removed is the minimum of load and available capacity
        let heat_removed = if self.mode == RefrigerationMode::Cooling {
            total_load.min(capacity)
        } else {
            0.0
        };

        self.total_heat_removed_j += heat_removed * _dt;
        self.total_energy_consumed_j += self.fan_power_w * _dt;

        heat_removed
    }

    /// Reset cumulative energy counters.
    pub fn reset_counters(&mut self) {
        self.total_heat_removed_j = 0.0;
        self.total_energy_consumed_j = 0.0;
    }
}

/// Walk-in freezer (refrigerated zone case).
///
/// Corresponds to EnergyPlus `Refrigeration:ZoneCase`.
/// Models a freezer case that maintains product temperature typically
/// between -25°C and -18°C.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkInFreezer {
    /// Equipment identifier
    pub id: String,
    /// Designed cooling capacity at reference conditions (W)
    pub rated_capacity_w: f64,
    /// Operating temperature setpoint (°C)
    pub temperature_setpoint_c: f64,
    /// Temperature differential for control (K)
    pub temperature_differential_k: f64,
    /// Rated ambient temperature for capacity rating (°C)
    pub rated_ambient_temp_c: f64,
    /// Rated internal temperature for capacity rating (°C)
    pub rated_internal_temp_c: f64,
    /// Current operating mode
    pub mode: RefrigerationMode,
    /// Latent heat removal fraction (0-1) - lower for freezers
    pub latent_heat_ratio: f64,
    /// Fan power consumption (W)
    pub fan_power_w: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Current refrigeration load (W)
    pub current_load_w: f64,
    /// Total heat removed this timestep (J)
    pub total_heat_removed_j: f64,
    /// Total energy consumed this timestep (J)
    pub total_energy_consumed_j: f64,
    /// Frost accumulation factor (affects capacity)
    pub frost_factor: f64,
}

impl WalkInFreezer {
    /// Create a new walk-in freezer with default parameters.
    pub fn new(id: String, rated_capacity_w: f64) -> Self {
        Self {
            id,
            rated_capacity_w,
            temperature_setpoint_c: -20.0, // Typical freezer temperature
            temperature_differential_k: 2.0,
            rated_ambient_temp_c: 25.0,
            rated_internal_temp_c: -20.0,
            mode: RefrigerationMode::Off,
            latent_heat_ratio: 0.1, // Freezers have lower latent loads
            fan_power_w: rated_capacity_w * 0.015, // ~1.5% of capacity
            current_plr: 0.0,
            current_load_w: 0.0,
            total_heat_removed_j: 0.0,
            total_energy_consumed_j: 0.0,
            frost_factor: 1.0,
        }
    }

    /// Calculate the sensible cooling capacity at current conditions.
    ///
    /// Freezer capacity is affected by frost accumulation and temperature lift.
    pub fn sensible_capacity_at_conditions(
        &self,
        _internal_temp_c: f64,
        ambient_temp_c: f64,
    ) -> f64 {
        // Temperature lift effect is more pronounced for freezers
        let temp_diff = (ambient_temp_c - self.rated_internal_temp_c).max(0.0);
        let rated_temp_diff = self.rated_ambient_temp_c - self.rated_internal_temp_c;
        let capacity_factor = 1.0 - (temp_diff - rated_temp_diff) * 0.015;
        // Frost factor reduces capacity over time
        self.rated_capacity_w * capacity_factor.max(0.2) * self.frost_factor
    }

    /// Calculate the total cooling load including latent heat.
    pub fn total_cooling_load(&self, sensible_load_w: f64) -> f64 {
        sensible_load_w * (1.0 + self.latent_heat_ratio)
    }

    /// Update the freezer state based on current conditions.
    pub fn update(
        &mut self,
        internal_temp_c: f64,
        ambient_temp_c: f64,
        sensible_load_w: f64,
        _dt: f64,
    ) -> f64 {
        let total_load = self.total_cooling_load(sensible_load_w);
        let capacity = self.sensible_capacity_at_conditions(internal_temp_c, ambient_temp_c);
        self.current_load_w = total_load;

        // Determine operating mode
        if internal_temp_c > self.temperature_setpoint_c + self.temperature_differential_k {
            self.mode = RefrigerationMode::Cooling;
        } else {
            self.mode = RefrigerationMode::Off;
        }

        // Calculate part-load ratio
        if self.mode == RefrigerationMode::Cooling && capacity > 0.0 {
            self.current_plr = (total_load / capacity).clamp(0.0, 1.0);
        } else {
            self.current_plr = 0.0;
        }

        // Heat removed
        let heat_removed = if self.mode == RefrigerationMode::Cooling {
            total_load.min(capacity)
        } else {
            0.0
        };

        // Update frost factor (increases with time if doors are open)
        // Simplified model: frost builds up slowly when running
        if self.mode == RefrigerationMode::Cooling {
            self.frost_factor = (self.frost_factor - 0.0001).max(0.85);
        } else {
            // Defrost slowly when off
            self.frost_factor = (self.frost_factor + 0.001).min(1.0);
        }

        self.total_heat_removed_j += heat_removed * _dt;
        self.total_energy_consumed_j += self.fan_power_w * _dt;

        heat_removed
    }

    /// Reset cumulative energy counters.
    pub fn reset_counters(&mut self) {
        self.total_heat_removed_j = 0.0;
        self.total_energy_consumed_j = 0.0;
    }

    /// Perform a defrost cycle.
    ///
    /// Returns the energy consumed during defrost.
    pub fn defrost(&mut self, duration_s: f64, defrost_power_w: f64) -> f64 {
        self.mode = RefrigerationMode::Defrost;
        let energy = defrost_power_w * duration_s;
        self.total_energy_consumed_j += energy;
        // After defrost, frost factor resets
        self.frost_factor = 1.0;
        energy
    }
}

/// Compressor rack for refrigeration systems.
///
/// Corresponds to EnergyPlus `Refrigeration:CompressorRack`.
/// Groups multiple compressors and calculates total power consumption
/// based on the total refrigeration load and condensing temperature.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorRack {
    /// Equipment identifier
    pub id: String,
    /// Total rated compressor power at reference conditions (W)
    pub rated_power_w: f64,
    /// Rated condensing temperature (°C)
    pub rated_condensing_temp_c: f64,
    /// Rated evaporating temperature (°C)
    pub rated_evaporating_temp_c: f64,
    /// COP at rated conditions (W/W)
    pub rated_cop: f64,
    /// Current total refrigeration load (W)
    pub current_load_w: f64,
    /// Current condensing temperature (°C)
    pub condensing_temp_c: f64,
    /// Current evaporating temperature (°C)
    pub evaporating_temp_c: f64,
    /// Compressor power curve coefficients [a, b, c] for
    /// `power = a + b*T_cond + c*T_cond^2` at constant load
    pub power_curve_a: f64,
    pub power_curve_b: f64,
    pub power_curve_c: f64,
    /// Load vs power curve coefficients for part-load ratio
    pub part_load_curve_a: f64,
    pub part_load_curve_b: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Total power consumed this timestep (J)
    pub total_power_consumed_j: f64,
    /// Total heat rejected this timestep (J)
    pub total_heat_rejected_j: f64,
}

impl CompressorRack {
    /// Create a new compressor rack with default parameters.
    pub fn new(id: String, rated_power_w: f64) -> Self {
        Self {
            id,
            rated_power_w,
            rated_condensing_temp_c: 35.0,
            rated_evaporating_temp_c: -25.0,
            rated_cop: constants::DEFAULT_COMPRESSOR_COP,
            current_load_w: 0.0,
            condensing_temp_c: 35.0,
            evaporating_temp_c: -25.0,
            // Default polynomial coefficients for power curve
            // Temperature factor = a + b*T_cond + c*T_cond^2
            // At 35°C design temp: 0.9 + 0.003*35 ≈ 1.0 (rated capacity)
            power_curve_a: 0.9,
            power_curve_b: 0.003,
            power_curve_c: 0.0,
            // Part-load curve coefficients (typically from manufacturer data)
            part_load_curve_a: 1.0,
            part_load_curve_b: -0.2,
            current_plr: 0.0,
            total_power_consumed_j: 0.0,
            total_heat_rejected_j: 0.0,
        }
    }

    /// Calculate the temperature lift (difference between condensing and evaporating temps).
    pub fn temperature_lift(&self) -> f64 {
        self.condensing_temp_c - self.evaporating_temp_c
    }

    /// Calculate the COP at current conditions.
    ///
    /// COP decreases with higher condensing temperature and lower evaporating temperature.
    pub fn cop_at_conditions(&self) -> f64 {
        let lift = self.temperature_lift();
        // Simplified COP model: COP = rated_COP * (1 - k * lift_factor)
        // where lift_factor is normalized temperature lift
        let rated_lift = self.rated_condensing_temp_c - self.rated_evaporating_temp_c;
        let lift_factor = (lift - rated_lift) / rated_lift;
        // COP degrades approximately 4% per degree of additional lift
        let cop_factor = 1.0 - 0.04 * lift_factor;
        self.rated_cop * cop_factor.max(0.3)
    }

    /// Calculate compressor power using the performance curve.
    ///
    /// Power = (a + b*T_cond + c*T_cond^2) * (part_load_factor)
    pub fn power_at_conditions(&mut self, load_w: f64) -> f64 {
        if load_w <= 0.0 {
            return 0.0;
        }

        // Calculate PLR
        let plr = (load_w / self.rated_power_w).clamp(0.0, 1.0);
        self.current_plr = plr;

        // Temperature-dependent power factor
        let temp_factor = self.power_curve_a
            + self.power_curve_b * self.condensing_temp_c
            + self.power_curve_c * self.condensing_temp_c * self.condensing_temp_c;

        // Part-load power factor (typically from manufacturer data)
        // Power at part-load is not linear - use quadratic curve
        let plr_factor = self.part_load_curve_a + self.part_load_curve_b * plr;

        // Total power = rated_power * temp_factor * plr_factor
        let power = self.rated_power_w * temp_factor * plr_factor;
        power.max(0.0)
    }

    /// Update the compressor rack state and calculate power consumption.
    ///
    /// Returns the power consumed (W) and heat rejected (W).
    pub fn update(
        &mut self,
        load_w: f64,
        condensing_temp_c: f64,
        evaporating_temp_c: f64,
        dt: f64,
    ) -> (f64, f64) {
        self.current_load_w = load_w;
        self.condensing_temp_c = condensing_temp_c;
        self.evaporating_temp_c = evaporating_temp_c;

        let power = self.power_at_conditions(load_w);
        // Heat rejected = refrigeration load + compressor power (energy conservation)
        let heat_rejected = load_w + power;

        self.total_power_consumed_j += power * dt;
        self.total_heat_rejected_j += heat_rejected * dt;

        (power, heat_rejected)
    }

    /// Reset cumulative energy counters.
    pub fn reset_counters(&mut self) {
        self.total_power_consumed_j = 0.0;
        self.total_heat_rejected_j = 0.0;
    }
}

/// Air-cooled condenser for refrigeration systems.
///
/// Corresponds to EnergyPlus `Refrigeration:CaseAndWalkIn`.
/// Rejects heat from the refrigeration system to the outdoor air.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirCooledCondenser {
    /// Equipment identifier
    pub id: String,
    /// Rated heat rejection capacity at design conditions (W)
    pub rated_heat_rejection_w: f64,
    /// Design air temperature (°C)
    pub design_air_temp_c: f64,
    /// Design subcooling temperature rise (K)
    pub design_subcooling_k: f64,
    /// Rated air flow rate (m³/s)
    pub rated_air_flow_m3_per_s: f64,
    /// Fan power at full speed (W)
    pub fan_power_w: f64,
    /// Minimum ambient temperature for operation (°C)
    pub min_ambient_temp_c: f64,
    /// Maximum ambient temperature for operation (°C)
    pub max_ambient_temp_c: f64,
    /// Heat rejection curve coefficients [a, b, c] for
    /// `capacity = a + b*T_air + c*T_air^2`
    pub heat_rejection_curve_a: f64,
    pub heat_rejection_curve_b: f64,
    pub heat_rejection_curve_c: f64,
    /// Current air flow fraction (0.0 to 1.0)
    pub current_air_flow_fraction: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Total heat rejected this timestep (J)
    pub total_heat_rejected_j: f64,
    /// Total fan energy consumed this timestep (J)
    pub total_fan_energy_j: f64,
}

impl AirCooledCondenser {
    /// Create a new air-cooled condenser with default parameters.
    pub fn new(id: String, rated_heat_rejection_w: f64) -> Self {
        Self {
            id,
            rated_heat_rejection_w,
            design_air_temp_c: constants::DEFAULT_CONDENSER_AIR_TEMP,
            design_subcooling_k: 5.0,
            rated_air_flow_m3_per_s: rated_heat_rejection_w / 30000.0, // ~30 W per m³/s
            fan_power_w: rated_heat_rejection_w * 0.015,               // ~1.5% of capacity
            min_ambient_temp_c: constants::MIN_CONDENSER_AMBIENT_TEMP,
            max_ambient_temp_c: constants::MAX_CONDENSER_AMBIENT_TEMP,
            // Default quadratic heat rejection curve
            heat_rejection_curve_a: 1.2,
            heat_rejection_curve_b: -0.015,
            heat_rejection_curve_c: 0.0002,
            current_air_flow_fraction: 1.0,
            current_plr: 0.0,
            total_heat_rejected_j: 0.0,
            total_fan_energy_j: 0.0,
        }
    }

    /// Calculate heat rejection capacity at current conditions.
    ///
    /// Capacity decreases at higher ambient temperatures due to reduced
    /// temperature differential for heat transfer.
    pub fn heat_rejection_capacity(&self, ambient_temp_c: f64) -> f64 {
        if ambient_temp_c < self.min_ambient_temp_c || ambient_temp_c > self.max_ambient_temp_c {
            return 0.0;
        }

        let temp_diff = ambient_temp_c - self.design_air_temp_c;
        let capacity_factor = self.heat_rejection_curve_a
            + self.heat_rejection_curve_b * temp_diff
            + self.heat_rejection_curve_c * temp_diff * temp_diff;
        self.rated_heat_rejection_w * capacity_factor.max(0.1)
    }

    /// Calculate fan power at current conditions.
    ///
    /// Fan power varies with air flow fraction cubed (affinity laws).
    pub fn fan_power_at_fraction(&self, air_flow_fraction: f64) -> f64 {
        let fraction = air_flow_fraction.clamp(0.0, 1.0);
        self.fan_power_w * fraction * fraction * fraction
    }

    /// Update the condenser state and calculate heat rejection.
    ///
    /// Returns the heat rejected (W) and fan power (W).
    pub fn update(&mut self, heat_to_reject_w: f64, ambient_temp_c: f64, dt: f64) -> (f64, f64) {
        let capacity = self.heat_rejection_capacity(ambient_temp_c);

        // Part-load ratio based on capacity
        if capacity > 0.0 {
            self.current_plr = (heat_to_reject_w / capacity).clamp(0.0, 1.0);
        } else {
            self.current_plr = 0.0;
        }

        // Air flow fraction can modulate to match load
        // At part-load, we can reduce fan speed to save energy
        self.current_air_flow_fraction = self.current_plr.sqrt();

        // Heat rejected is limited by capacity and load
        let heat_rejected = heat_to_reject_w.min(capacity);

        // Fan power depends on air flow
        let fan_power = self.fan_power_at_fraction(self.current_air_flow_fraction);

        self.total_heat_rejected_j += heat_rejected * dt;
        self.total_fan_energy_j += fan_power * dt;

        (heat_rejected, fan_power)
    }

    /// Reset cumulative energy counters.
    pub fn reset_counters(&mut self) {
        self.total_heat_rejected_j = 0.0;
        self.total_fan_energy_j = 0.0;
    }

    /// Check if the condenser can reject the required heat.
    pub fn can_reject(&self, heat_w: f64, ambient_temp_c: f64) -> bool {
        let capacity = self.heat_rejection_capacity(ambient_temp_c);
        capacity >= heat_w
    }
}

/// Complete refrigeration system combining all components.
///
/// This struct provides a unified interface for modeling a complete
/// refrigeration system including cases, compressors, and condensers.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefrigerationSystem {
    /// System identifier
    pub id: String,
    /// Walk-in coolers in the system
    pub coolers: Vec<WalkInCooler>,
    /// Walk-in freezers in the system
    pub freezers: Vec<WalkInFreezer>,
    /// Compressor rack serving the system
    pub compressor_rack: CompressorRack,
    /// Air-cooled condenser
    pub condenser: AirCooledCondenser,
    /// Ambient temperature (°C)
    pub ambient_temp_c: f64,
}

impl RefrigerationSystem {
    /// Create a new refrigeration system with the given components.
    pub fn new(
        id: String,
        coolers: Vec<WalkInCooler>,
        freezers: Vec<WalkInFreezer>,
        compressor_rack: CompressorRack,
        condenser: AirCooledCondenser,
    ) -> Self {
        Self {
            id,
            coolers,
            freezers,
            compressor_rack,
            condenser,
            ambient_temp_c: 25.0,
        }
    }

    /// Calculate total refrigeration load from all cases.
    pub fn total_case_load(&self) -> f64 {
        let cooler_load: f64 = self.coolers.iter().map(|c| c.current_load_w).sum();
        let freezer_load: f64 = self.freezers.iter().map(|f| f.current_load_w).sum();
        cooler_load + freezer_load
    }

    /// Calculate total power consumption of the system.
    pub fn total_power(&self) -> f64 {
        let case_power: f64 = self
            .coolers
            .iter()
            .map(|c| c.fan_power_w * c.current_plr)
            .sum();
        let freezer_power: f64 = self
            .freezers
            .iter()
            .map(|f| f.fan_power_w * f.current_plr)
            .sum();
        let compressor_power =
            self.compressor_rack.current_load_w / self.compressor_rack.cop_at_conditions();
        let condenser_fan = self
            .condenser
            .fan_power_at_fraction(self.condenser.current_air_flow_fraction);
        case_power + freezer_power + compressor_power + condenser_fan
    }

    /// Calculate the system COP (cooling capacity / power input).
    pub fn system_cop(&self) -> f64 {
        let total_load = self.total_case_load();
        let power = self.total_power();
        if power > 0.0 {
            total_load / power
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_in_cooler_creation() {
        let cooler = WalkInCooler::new("COOL-1".to_string(), 5000.0);
        assert_eq!(cooler.id, "COOL-1");
        assert_eq!(cooler.rated_capacity_w, 5000.0);
        assert_eq!(cooler.temperature_setpoint_c, 3.0);
        assert_eq!(cooler.mode, RefrigerationMode::Off);
    }

    #[test]
    fn test_walk_in_cooler_cooling() {
        let mut cooler = WalkInCooler::new("COOL-1".to_string(), 5000.0);
        // Internal temp above setpoint (3+2=5) should trigger cooling
        let heat_removed = cooler.update(6.0, 25.0, 2000.0, 3600.0);
        assert_eq!(cooler.mode, RefrigerationMode::Cooling);
        assert!(heat_removed > 0.0);
    }

    #[test]
    fn test_walk_in_cooler_at_setpoint() {
        let mut cooler = WalkInCooler::new("COOL-1".to_string(), 5000.0);
        // At setpoint temperature
        let _heat_removed = cooler.update(3.0, 25.0, 2000.0, 3600.0);
        // Should be in deadband or off
        assert!(cooler.mode == RefrigerationMode::Off || cooler.current_plr < 1.0);
    }

    #[test]
    fn test_walk_in_freezer_creation() {
        let freezer = WalkInFreezer::new("FREEZE-1".to_string(), 3000.0);
        assert_eq!(freezer.id, "FREEZE-1");
        assert_eq!(freezer.rated_capacity_w, 3000.0);
        assert_eq!(freezer.temperature_setpoint_c, -20.0);
    }

    #[test]
    fn test_walk_in_freezer_cooling() {
        let mut freezer = WalkInFreezer::new("FREEZE-1".to_string(), 3000.0);
        // Internal temp above setpoint should trigger cooling
        let heat_removed = freezer.update(-15.0, 25.0, 1500.0, 3600.0);
        assert_eq!(freezer.mode, RefrigerationMode::Cooling);
        assert!(heat_removed > 0.0);
    }

    #[test]
    fn test_walk_in_freezer_defrost() {
        let mut freezer = WalkInFreezer::new("FREEZE-1".to_string(), 3000.0);
        let energy = freezer.defrost(1800.0, 2000.0);
        assert_eq!(freezer.mode, RefrigerationMode::Defrost);
        assert!(energy > 0.0);
        // Frost factor should reset after defrost
        assert_eq!(freezer.frost_factor, 1.0);
    }

    #[test]
    fn test_compressor_rack_creation() {
        let rack = CompressorRack::new("COMP-1".to_string(), 10000.0);
        assert_eq!(rack.id, "COMP-1");
        assert_eq!(rack.rated_power_w, 10000.0);
        assert_eq!(rack.rated_cop, constants::DEFAULT_COMPRESSOR_COP);
    }

    #[test]
    fn test_compressor_rack_power_calculation() {
        let mut rack = CompressorRack::new("COMP-1".to_string(), 10000.0);
        // At rated conditions with new coefficients
        // temp_factor ≈ 1.0, plr = 0.5, plr_factor = 1.0 + (-0.2)*0.5 = 0.9
        // power = 10000 * 1.0 * 0.9 = 9000 W
        let power = rack.power_at_conditions(5000.0);
        assert!(power > 0.0);
        assert!(power < 10000.0);
        assert!((power - 9000.0).abs() < 100.0);
    }

    #[test]
    fn test_compressor_rack_temperature_lift() {
        let mut rack = CompressorRack::new("COMP-1".to_string(), 10000.0);
        rack.update(5000.0, 40.0, -20.0, 3600.0);
        // Temperature lift should be 60°C
        assert_eq!(rack.temperature_lift(), 60.0);
    }

    #[test]
    fn test_compressor_rack_cop_decreases_with_lift() {
        let mut rack = CompressorRack::new("COMP-1".to_string(), 10000.0);
        // At rated conditions: lift = 35 - (-25) = 60°C
        let cop_rated = rack.cop_at_conditions();
        // Increase condensing temp (higher lift)
        rack.evaporating_temp_c = -30.0; // Lower evaporating temp
        let cop_higher_lift = rack.cop_at_conditions();
        // COP should decrease with higher lift
        assert!(cop_higher_lift < cop_rated);
    }

    #[test]
    fn test_compressor_rack_update() {
        let mut rack = CompressorRack::new("COMP-1".to_string(), 10000.0);
        let (power, heat_rejected) = rack.update(5000.0, 35.0, -25.0, 3600.0);
        assert!(power > 0.0);
        assert!(heat_rejected > power); // heat_rejected = load + power
        assert!(rack.total_power_consumed_j > 0.0);
    }

    #[test]
    fn test_air_cooled_condenser_creation() {
        let condenser = AirCooledCondenser::new("COND-1".to_string(), 15000.0);
        assert_eq!(condenser.id, "COND-1");
        assert_eq!(condenser.rated_heat_rejection_w, 15000.0);
    }

    #[test]
    fn test_air_cooled_condenser_capacity_at_temperature() {
        let condenser = AirCooledCondenser::new("COND-1".to_string(), 15000.0);
        // At design temperature (35°C), capacity factor is 1.2
        let capacity_design = condenser.heat_rejection_capacity(35.0);
        // Default curve gives 1.2x at design temp
        assert!((capacity_design - 18000.0).abs() < 100.0); // 15000 * 1.2
                                                            // At higher temperature - capacity should decrease
        let capacity_hot = condenser.heat_rejection_capacity(45.0);
        assert!(capacity_hot < capacity_design);
    }

    #[test]
    fn test_air_cooled_condenser_update() {
        let mut condenser = AirCooledCondenser::new("COND-1".to_string(), 15000.0);
        let (heat_rejected, fan_power) = condenser.update(10000.0, 35.0, 3600.0);
        assert!(heat_rejected > 0.0);
        assert!(fan_power > 0.0);
        assert!(condenser.total_heat_rejected_j > 0.0);
    }

    #[test]
    fn test_air_cooled_condenser_fan_law() {
        let condenser = AirCooledCondenser::new("COND-1".to_string(), 15000.0);
        // Fan power follows cubic law
        let power_full = condenser.fan_power_at_fraction(1.0);
        let power_half = condenser.fan_power_at_fraction(0.5);
        assert!((power_half - power_full * 0.125).abs() < 1.0); // 0.5^3 = 0.125
    }

    #[test]
    fn test_air_cooled_condenser_can_reject() {
        let condenser = AirCooledCondenser::new("COND-1".to_string(), 15000.0);
        assert!(condenser.can_reject(10000.0, 35.0));
        assert!(!condenser.can_reject(20000.0, 35.0)); // Exceeds capacity
        assert!(!condenser.can_reject(10000.0, 60.0)); // Too hot
    }

    #[test]
    fn test_refrigeration_system_integration() {
        let coolers = vec![WalkInCooler::new("COOL-1".to_string(), 5000.0)];
        let freezers = vec![WalkInFreezer::new("FREEZE-1".to_string(), 3000.0)];
        let compressor = CompressorRack::new("COMP-1".to_string(), 10000.0);
        let condenser = AirCooledCondenser::new("COND-1".to_string(), 15000.0);

        let system = RefrigerationSystem::new(
            "REF-SYS-1".to_string(),
            coolers,
            freezers,
            compressor,
            condenser,
        );

        assert_eq!(system.coolers.len(), 1);
        assert_eq!(system.freezers.len(), 1);
    }

    #[test]
    fn test_coolers_freezers_separate_loads() {
        let mut cooler = WalkInCooler::new("COOL-1".to_string(), 5000.0);
        let mut freezer = WalkInFreezer::new("FREEZE-1".to_string(), 3000.0);

        // Cooler at 6°C (above 3+2=5 threshold), freezer at -15°C (above -20+2=-18 threshold)
        cooler.update(6.0, 25.0, 2000.0, 3600.0);
        freezer.update(-15.0, 25.0, 1500.0, 3600.0);

        assert_eq!(cooler.mode, RefrigerationMode::Cooling);
        assert_eq!(freezer.mode, RefrigerationMode::Cooling);
        assert!(cooler.current_plr > 0.0);
        assert!(freezer.current_plr > 0.0);
    }

    #[test]
    fn test_reset_counters() {
        let mut cooler = WalkInCooler::new("COOL-1".to_string(), 5000.0);
        cooler.update(6.0, 25.0, 2000.0, 3600.0);
        assert!(cooler.total_heat_removed_j > 0.0);

        cooler.reset_counters();
        assert_eq!(cooler.total_heat_removed_j, 0.0);
        assert_eq!(cooler.total_energy_consumed_j, 0.0);
    }
}
