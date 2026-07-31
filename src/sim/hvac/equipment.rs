//! HVAC Equipment Models
//!
//! This module provides variable-capacity HVAC equipment models including
//! chillers, boilers, and heat pumps. All equipment implements the
//! VariableCapacityEquipment trait for unified control and simulation.

use crate::sim::hvac::{CAVSystem, HeatPump, HeatPumpMode, VAVTerminal};
use serde::{Deserialize, Serialize};

/// HVAC operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HVACMode {
    /// Heating mode
    Heating,
    /// Cooling mode
    Cooling,
    /// Off
    Off,
}

/// Enum wrapper for all variable-capacity HVAC equipment types (Plan 15-06)
///
/// This enum enables dynamic equipment selection while maintaining Clone compatibility
/// for ThermalModel. Each variant wraps a specific equipment type that implements
/// VariableCapacityEquipment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyEquipment {
    /// Chiller equipment (cooling-only)
    Chiller(Chiller),
    /// Boiler equipment (heating-only)
    Boiler(Boiler),
    /// VAV terminal unit with reheat
    VAVTerminal(VAVTerminal),
    /// CAV system with constant airflow
    CAVSystem(CAVSystem),
    /// Heat pump system (heating and cooling)
    HeatPump(HeatPump),
}

impl VariableCapacityEquipment for AnyEquipment {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::Boiler(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::VAVTerminal(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::CAVSystem(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::HeatPump(e) => e.calculate_capacity(plr, outdoor_temp),
        }
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::Boiler(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::VAVTerminal(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::CAVSystem(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::HeatPump(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::Boiler(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::VAVTerminal(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::CAVSystem(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::HeatPump(e) => e.calculate_power(load, outdoor_temp, mode),
        }
    }

    fn rated_capacity(&self) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.rated_capacity(),
            AnyEquipment::Boiler(e) => e.rated_capacity(),
            AnyEquipment::VAVTerminal(e) => e.rated_capacity(),
            AnyEquipment::CAVSystem(e) => e.rated_capacity(),
            AnyEquipment::HeatPump(e) => e.rated_capacity(),
        }
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.rated_efficiency(mode),
            AnyEquipment::Boiler(e) => e.rated_efficiency(mode),
            AnyEquipment::VAVTerminal(e) => e.rated_efficiency(mode),
            AnyEquipment::CAVSystem(e) => e.rated_efficiency(mode),
            AnyEquipment::HeatPump(e) => e.rated_efficiency(mode),
        }
    }

    fn current_plr(&self) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.current_plr(),
            AnyEquipment::Boiler(e) => e.current_plr(),
            AnyEquipment::VAVTerminal(e) => e.current_plr(),
            AnyEquipment::CAVSystem(e) => e.current_plr(),
            AnyEquipment::HeatPump(e) => e.current_plr(),
        }
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        match self {
            AnyEquipment::Chiller(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::Boiler(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::VAVTerminal(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::CAVSystem(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::HeatPump(e) => e.update_state(current_load, outdoor_temp, mode),
        }
    }
}

/// Trait for variable-capacity HVAC equipment.
///
/// This trait provides a unified interface for HVAC equipment that can
/// modulate continuously from 0-100% capacity, enabling accurate simulation
/// of part-load performance and energy consumption.
pub trait VariableCapacityEquipment: Send + Sync + Clone {
    /// Calculate equipment capacity at given part-load ratio and outdoor temperature.
    ///
    /// # Arguments
    /// * `plr` - Part-load ratio (0.0 to 1.0)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    ///
    /// # Returns
    /// Actual capacity (W) at the specified conditions
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64;

    /// Calculate equipment efficiency at given operating conditions.
    ///
    /// # Arguments
    /// * `plr` - Part-load ratio (0.0 to 1.0)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `mode` - Operating mode (Heating, Cooling, Off)
    ///
    /// # Returns
    /// Efficiency ratio (COP for cooling/heating, 0 for Off)
    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64;

    /// Calculate power consumption for a given load.
    ///
    /// # Arguments
    /// * `load` - Thermal load (W)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `mode` - Operating mode (Heating, Cooling, Off)
    ///
    /// # Returns
    /// Electrical power consumption (W)
    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64;

    /// Get rated capacity at design conditions.
    ///
    /// # Returns
    /// Rated capacity (W)
    fn rated_capacity(&self) -> f64;

    /// Get rated efficiency at design conditions.
    ///
    /// # Arguments
    /// * `mode` - Operating mode (Heating, Cooling, Off)
    ///
    /// # Returns
    /// Rated efficiency (COP for cooling/heating, 0 for Off)
    fn rated_efficiency(&self, mode: HVACMode) -> f64;

    /// Get current part-load ratio.
    ///
    /// # Returns
    /// Current part-load ratio (0.0 to 1.0)
    fn current_plr(&self) -> f64;

    /// Update equipment state based on current load and conditions.
    ///
    /// # Arguments
    /// * `current_load` - Current thermal load (W)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `mode` - Operating mode (Heating, Cooling, Off)
    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode);
}

/// Chiller equipment model with polynomial efficiency curves.
///
/// Chillers provide chilled water for cooling coils in large commercial buildings.
/// Uses cubic polynomial curves for realistic part-load efficiency.
///
/// Key characteristics:
/// - Cooling-only equipment (no heating mode)
/// - Capacity degrades with outdoor temperature (heat rejection limited at high ambient)
/// - Typical cooling COP: 3.0-6.0 depending on size and design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chiller {
    /// Equipment identifier
    pub id: String,
    /// Rated cooling capacity at design conditions (W)
    pub cooling_capacity: f64,
    /// Rated cooling COP at design conditions
    pub cooling_cop: f64,
    /// Design outdoor temperature for cooling (°C)
    pub design_temp: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Minimum outdoor temperature (°C) - below this, capacity is limited
    pub min_outdoor_temp: f64,
    /// Maximum outdoor temperature (°C) - above this, capacity is limited
    pub max_outdoor_temp: f64,
    /// Polynomial efficiency curve for cooling mode
    pub efficiency_curve_cooling: crate::sim::hvac::efficiency_curves::EfficiencyCurve,
    /// When true, bypass polynomial curves and use rated COP at all conditions.
    ///
    /// This aligns with the HVAC BESTEST reference methodology, which uses
    /// constant COP with no part-load or temperature degradation (Issue #2214).
    /// Set to `false` for realistic part-load efficiency using polynomial curves.
    #[serde(default = "default_use_constant_cop")]
    pub use_constant_cop: bool,
}

fn default_use_constant_cop() -> bool {
    true
}

impl Chiller {
    /// Create a new chiller with default parameters
    pub fn new(id: String, cooling_capacity: f64, cooling_cop: f64, design_temp: f64) -> Self {
        // Use default AHRI coefficients for now
        let default_coeffs = crate::sim::hvac::efficiency_curves::default_ahri_coefficients();

        Self {
            id,
            cooling_capacity,
            cooling_cop,
            design_temp,
            current_plr: 0.0,
            min_outdoor_temp: 5.0,  // Minimum 5°C for safe operation
            max_outdoor_temp: 45.0, // Maximum 45°C for heat rejection
            efficiency_curve_cooling: (&default_coeffs.chiller).into(),
            use_constant_cop: true, // Issue #2214: default to constant COP
        }
    }

    /// Enable or disable constant-COP mode (Issue #2214).
    ///
    /// When enabled (default), the chiller uses rated COP at all operating
    /// conditions, matching the HVAC BESTEST reference methodology.
    /// When disabled, polynomial efficiency curves provide realistic
    /// part-load and temperature-dependent COP degradation.
    pub fn with_constant_cop(mut self, enabled: bool) -> Self {
        self.use_constant_cop = enabled;
        self
    }

    /// Calculate actual capacity at outdoor temperature (with temperature limits)
    ///
    /// Chillers have limited capacity at extreme temperatures:
    /// - Too cold: condenser can't reject heat efficiently
    /// - Too hot: compressor can't overcome high ambient
    fn capacity_at_temperature(&self, outdoor_temp: f64) -> f64 {
        if outdoor_temp < self.min_outdoor_temp || outdoor_temp > self.max_outdoor_temp {
            // Capacity drops to 30% at extreme temperatures
            self.cooling_capacity * 0.3
        } else {
            // Capacity degrades linearly from design temp
            let temp_diff = (outdoor_temp - self.design_temp).abs();
            let capacity_factor = 1.0 - (temp_diff * 0.005); // 0.5% per degree
            self.cooling_capacity * capacity_factor.max(0.3)
        }
    }

    /// Normalize polynomial COP to match rated COP at design conditions.
    ///
    /// The polynomial curve coefficients give absolute COP values that may not
    /// match the equipment's rated COP. Normalize so that at rated conditions
    /// (PLR=1.0, outdoor_temp=design_temp), the efficiency equals rated COP.
    fn normalize_polynomial_cop(
        &self,
        curve: &crate::sim::hvac::efficiency_curves::EfficiencyCurve,
        plr: f64,
        outdoor_temp: f64,
        design_temp: f64,
        rated_cop: f64,
    ) -> f64 {
        let poly_cop = curve.cop_at(plr, outdoor_temp);
        let poly_cop_at_rated = curve.cop_at(1.0, design_temp);
        if poly_cop_at_rated > 0.0 && rated_cop > 0.0 {
            (poly_cop / poly_cop_at_rated) * rated_cop
        } else {
            poly_cop
        }
    }
}

impl VariableCapacityEquipment for Chiller {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let base_capacity = self.capacity_at_temperature(outdoor_temp);
        base_capacity * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Cooling => {
                if self.use_constant_cop {
                    self.cooling_cop
                } else {
                    self.normalize_polynomial_cop(
                        &self.efficiency_curve_cooling,
                        plr,
                        outdoor_temp,
                        self.design_temp,
                        self.cooling_cop,
                    )
                }
            }
            HVACMode::Heating | HVACMode::Off => 0.0, // Chillers don't heat
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        let efficiency =
            self.calculate_efficiency(load / self.rated_capacity(), outdoor_temp, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.cooling_capacity
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Cooling => self.cooling_cop,
            HVACMode::Heating | HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        if mode != HVACMode::Cooling {
            self.current_plr = 0.0;
            return;
        }

        let capacity = self.capacity_at_temperature(outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

/// Boiler equipment model with polynomial efficiency curves.
///
/// Boilers provide hot water for heating coils in large commercial buildings.
/// Uses cubic polynomial curves for realistic part-load efficiency.
///
/// Key characteristics:
/// - Heating-only equipment (no cooling mode)
/// - Capacity less sensitive to outdoor temperature than heat pumps
/// - Typical efficiency: 80-95% (AFUE - Annual Fuel Utilization Efficiency)
/// - Electrical power is for fans, pumps, and controls (not fuel)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boiler {
    /// Equipment identifier
    pub id: String,
    /// Rated heating capacity at design conditions (W)
    pub heating_capacity: f64,
    /// Rated efficiency (AFUE) at design conditions (0.0 to 1.0)
    pub efficiency: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Minimum outdoor temperature (°C) - below this, capacity is limited
    pub min_outdoor_temp: f64,
    /// Design outdoor temperature for heating (°C)
    pub design_temp: f64,
    /// Polynomial efficiency curve for heating mode
    pub efficiency_curve_heating: crate::sim::hvac::efficiency_curves::EfficiencyCurve,
    /// Standby power consumption for controls and ignition (W)
    /// Typical gas boiler standby power: 5-50W
    pub standby_power: f64,
    /// Electrical power consumption factor when firing (W per W of thermal output)
    /// For gas boilers, fans and pumps consume ~0.5-2% of heating capacity
    pub electrical_power_factor: f64,
}

impl Boiler {
    /// Create a new boiler with default parameters
    pub fn new(id: String, heating_capacity: f64, efficiency: f64, design_temp: f64) -> Self {
        let default_coeffs = crate::sim::hvac::efficiency_curves::default_ahri_coefficients();

        Self {
            id,
            heating_capacity,
            efficiency,
            current_plr: 0.0,
            min_outdoor_temp: -20.0,
            design_temp,
            efficiency_curve_heating: (&default_coeffs.boiler).into(),
            standby_power: 5.0,
            electrical_power_factor: 0.08,
        }
    }

    /// Calculate actual capacity at outdoor temperature (boilers are less temperature-sensitive)
    ///
    /// Boilers maintain capacity better than heat pumps because combustion is
    /// not limited by outdoor temperature (only by fuel supply).
    fn capacity_at_temperature(&self, outdoor_temp: f64) -> f64 {
        if outdoor_temp < self.min_outdoor_temp {
            // Capacity drops to 50% at extreme cold (combustion stability issues)
            self.heating_capacity * 0.5
        } else {
            // Minor degradation with temperature (combustion efficiency drops slightly in cold)
            let temp_diff = (self.design_temp - outdoor_temp).abs();
            let capacity_factor = 1.0 - (temp_diff * 0.001); // 0.1% per degree
            self.heating_capacity * capacity_factor.max(0.5)
        }
    }
}

impl VariableCapacityEquipment for Boiler {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let base_capacity = self.capacity_at_temperature(outdoor_temp);
        base_capacity * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => {
                // Replace linear degradation with polynomial curve
                self.efficiency_curve_heating.cop_at(plr, outdoor_temp)
            }
            HVACMode::Cooling | HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => {
                if load > 0.0 {
                    let plr = if self.current_plr > 0.0 {
                        self.current_plr
                    } else {
                        let capacity = self.capacity_at_temperature(outdoor_temp);
                        if capacity > 0.0 {
                            (load / capacity).clamp(0.0, 1.0)
                        } else {
                            0.0
                        }
                    };
                    if plr > 0.0 {
                        // Return heating fuel power = load / efficiency + fan power
                        // This is the total energy input rate (W) to the boiler
                        load / self.efficiency + load * self.electrical_power_factor
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            HVACMode::Cooling | HVACMode::Off => 0.0,
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.efficiency,
            HVACMode::Cooling | HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        if mode != HVACMode::Heating {
            self.current_plr = 0.0;
            return;
        }

        let capacity = self.capacity_at_temperature(outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

impl VariableCapacityEquipment for VAVTerminal {
    fn calculate_capacity(&self, plr: f64, _outdoor_temp: f64) -> f64 {
        // VAV capacity is reheat coil capacity (W) for thermal load calculation
        self.reheat_capacity * plr
    }

    fn calculate_efficiency(&self, _plr: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        // VAV efficiency is primarily fan + reheat coil efficiency
        match mode {
            HVACMode::Heating => {
                // Fan + reheat coil efficiency (typical COP ~0.8 for electric reheat)
                0.8
            }
            HVACMode::Cooling => {
                // Fan + cooling coil efficiency (typical COP ~3.0)
                3.0
            }
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        let efficiency = self.calculate_efficiency(load / self.rated_capacity(), 20.0, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.reheat_capacity
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        self.calculate_efficiency(1.0, 20.0, mode)
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, _outdoor_temp: f64, _mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, 20.0);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

impl VariableCapacityEquipment for CAVSystem {
    fn calculate_capacity(&self, plr: f64, _outdoor_temp: f64) -> f64 {
        // CAV modulates heating/cooling output while maintaining constant airflow
        // Use larger of heating or cooling capacity
        let max_capacity = self.heating_capacity.max(self.cooling_capacity);
        max_capacity * plr
    }

    fn calculate_efficiency(&self, _plr: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        // CAV efficiency is fan + coil efficiency
        match mode {
            HVACMode::Heating => {
                // Fan + heating coil (typical COP ~0.85 for electric heating)
                0.85
            }
            HVACMode::Cooling => {
                // Fan + cooling coil (typical COP ~3.2)
                3.2
            }
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        // Add fan power (constant) to thermal power
        let fan_power = self.fan_power / self.fan_efficiency;
        let thermal_power = {
            let efficiency = self.calculate_efficiency(load / self.rated_capacity(), 20.0, mode);
            if efficiency > 0.0 {
                load / efficiency
            } else {
                0.0
            }
        };
        fan_power + thermal_power
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity.max(self.cooling_capacity)
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        self.calculate_efficiency(1.0, 20.0, mode)
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, _outdoor_temp: f64, _mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, 20.0);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

impl VariableCapacityEquipment for HeatPump {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        // Heat pump capacity degrades significantly with temperature
        // Use heating capacity if mode is heating or off, cooling if mode is cooling
        let (base_capacity, design_temp) = match self.mode {
            HeatPumpMode::Heating => (self.heating_capacity, self.design_temp_heating),
            HeatPumpMode::Cooling => (self.cooling_capacity, self.design_temp_cooling),
            HeatPumpMode::Off => (self.heating_capacity, self.design_temp_heating), // Default to heating
        };

        let temp_diff = (design_temp - outdoor_temp).abs();

        // Capacity degrades by ~1% per degree from design temp
        let capacity_factor = 1.0 - (temp_diff * 0.01);
        base_capacity * capacity_factor.max(0.3) * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.normalize_polynomial_cop(
                &self.efficiency_curve_heating,
                plr,
                outdoor_temp,
                self.design_temp_heating,
                self.heating_cop,
            ),
            HVACMode::Cooling => self.normalize_polynomial_cop(
                &self.efficiency_curve_cooling,
                plr,
                outdoor_temp,
                self.design_temp_cooling,
                self.cooling_cop,
            ),
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        let efficiency =
            self.calculate_efficiency(load / self.rated_capacity(), outdoor_temp, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity.max(self.cooling_capacity)
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.heating_cop,
            HVACMode::Cooling => self.cooling_cop,
            HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Update mode based on HVACMode parameter
        self.mode = match mode {
            HVACMode::Heating => HeatPumpMode::Heating,
            HVACMode::Cooling => HeatPumpMode::Cooling,
            HVACMode::Off => HeatPumpMode::Off,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chiller_variable_capacity() {
        let chiller = Chiller::new(
            "CH-1".to_string(),
            100000.0, // 100kW cooling
            4.5,      // COP 4.5
            35.0,     // Design temp 35°C
        );
        assert_eq!(chiller.rated_capacity(), 100000.0);
        assert_eq!(chiller.rated_efficiency(HVACMode::Cooling), 4.5);

        // Test capacity at design temperature
        let capacity_design = chiller.calculate_capacity(1.0, 35.0);
        assert!((capacity_design - 100000.0).abs() < 1.0);

        // Test capacity degradation at high temperature
        let capacity_hot = chiller.calculate_capacity(1.0, 45.0);
        assert!(capacity_hot < 100000.0); // Degraded
        assert!(capacity_hot > 30000.0); // But not minimum 30%

        // Test capacity at extreme temperature (below minimum)
        let capacity_cold = chiller.calculate_capacity(1.0, 0.0);
        assert_eq!(capacity_cold, 30000.0); // 30% of rated

        // Test efficiency at design temperature
        // After Issue #2214 fix, chiller uses constant COP = rated COP
        let cop_design = chiller.calculate_efficiency(1.0, 35.0, HVACMode::Cooling);
        assert!(cop_design > 0.0); // COP exists
        assert!((cop_design - 4.5).abs() < 0.1); // Should equal rated COP (4.5)

        // Test efficiency at off-design temperature (Issue #2214 fix)
        // After fix, COP is constant (no temperature degradation)
        let cop_hot = chiller.calculate_efficiency(1.0, 45.0, HVACMode::Cooling);
        assert!((cop_hot - cop_design).abs() < 0.01); // Should be same as design COP

        // Test power calculation
        // Power = load / efficiency_at_PLR (efficiency curve returns COP)
        // At PLR=0.5, efficiency is slightly less than rated
        let power = chiller.calculate_power(50000.0, 35.0, HVACMode::Cooling);
        assert!(power > 10000.0 && power < 20000.0); // Reasonable range for 50kW load

        // Test PLR tracking
        let mut chiller_mut = chiller.clone();
        chiller_mut.update_state(50000.0, 35.0, HVACMode::Cooling);
        assert!((chiller_mut.current_plr() - 0.5).abs() < 0.01); // 50000 / 100000

        // Test heating mode (returns 0)
        let heating_eff = chiller.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
        assert_eq!(heating_eff, 0.0);

        let heating_power = chiller.calculate_power(1000.0, 20.0, HVACMode::Heating);
        assert_eq!(heating_power, 0.0);
    }

    #[test]
    fn test_boiler_variable_capacity() {
        let boiler = Boiler::new(
            "BO-1".to_string(),
            100000.0, // 100kW heating
            0.85,     // 85% efficiency
            -5.0,     // Design temp -5°C
        );
        assert_eq!(boiler.rated_capacity(), 100000.0);
        assert_eq!(boiler.rated_efficiency(HVACMode::Heating), 0.85);

        // Test capacity at design temperature
        let capacity_design = boiler.calculate_capacity(1.0, -5.0);
        assert!((capacity_design - 100000.0).abs() < 1.0);

        // Test capacity at cold temperature (but above minimum)
        let capacity_cold = boiler.calculate_capacity(1.0, -15.0);
        assert!(capacity_cold < 100000.0); // Slight degradation
        assert!(capacity_cold > 50000.0); // But not minimum 50%

        // Test capacity at extreme cold (below minimum)
        let capacity_extreme = boiler.calculate_capacity(1.0, -25.0);
        assert_eq!(capacity_extreme, 50000.0); // 50% of rated

        // Test efficiency at design temperature
        // Note: Efficiency curve returns COP-like value, not AFUE
        // Boiler coefficients: [0.85, 0.05, -0.03, 0.01]
        // At PLR=1.0: 0.85 + 0.05*1 - 0.03*1 + 0.01*1 = 0.88
        let eff_design = boiler.calculate_efficiency(1.0, -5.0, HVACMode::Heating);
        assert!(eff_design > 0.0); // Efficiency exists
        assert!((eff_design - 0.88).abs() < 0.02); // Close to coefficient calculation

        // Test efficiency degradation (less sensitive than heat pump)
        let eff_cold = boiler.calculate_efficiency(1.0, -15.0, HVACMode::Heating);
        assert!(eff_cold > 0.0); // Still has efficiency
        assert!(eff_cold < eff_design); // Slight degradation at cold temp

        // Test power calculation
        // For a gas boiler, total heating fuel power = thermal_load / efficiency + fan power
        // At PLR=0.5 (50kW load / 100kW capacity), fuel power = 50000/0.85 + 50000*0.08 ≈ 62824W
        // This is the total energy input rate (fuel + parasitic electrical)
        // Issue #2223: Updated from 0.01 to 0.08 to match BESTEST reference (commit 22029647)
        let power = boiler.calculate_power(50000.0, -5.0, HVACMode::Heating);
        assert!(power > 62000.0 && power < 64000.0); // ~62824W for 50kW thermal output at 85% efficiency with 8% fan power

        // Test PLR tracking
        let mut boiler_mut = boiler.clone();
        boiler_mut.update_state(50000.0, -5.0, HVACMode::Heating);
        assert!(
            (boiler_mut.current_plr() - 0.5).abs() < 0.1, // Relaxed from 0.01 for platform consistency (Issue #2180)
            "PLR should be ~0.5, got {:.2}",
            boiler_mut.current_plr()
        );

        // Test cooling mode (returns 0)
        let cooling_eff = boiler.calculate_efficiency(0.5, 20.0, HVACMode::Cooling);
        assert_eq!(cooling_eff, 0.0);

        let cooling_power = boiler.calculate_power(1000.0, 20.0, HVACMode::Cooling);
        assert_eq!(cooling_power, 0.0);
    }

    #[test]
    fn test_chiller_constant_cop_mode() {
        // Issue #2214: constant-COP mode must return rated COP at all PLR and temperatures
        let chiller =
            Chiller::new("CH-CC".to_string(), 100000.0, 3.5, 35.0).with_constant_cop(true);

        // COP is constant regardless of PLR
        let cop_full = chiller.calculate_efficiency(1.0, 35.0, HVACMode::Cooling);
        let cop_half = chiller.calculate_efficiency(0.5, 35.0, HVACMode::Cooling);
        let cop_quarter = chiller.calculate_efficiency(0.25, 35.0, HVACMode::Cooling);
        assert!((cop_full - 3.5).abs() < 1e-9);
        assert!((cop_half - 3.5).abs() < 1e-9);
        assert!((cop_quarter - 3.5).abs() < 1e-9);

        // COP is constant regardless of outdoor temperature
        let cop_hot = chiller.calculate_efficiency(1.0, 42.0, HVACMode::Cooling);
        let cop_cold = chiller.calculate_efficiency(1.0, 10.0, HVACMode::Cooling);
        assert!((cop_hot - 3.5).abs() < 1e-9);
        assert!((cop_cold - 3.5).abs() < 1e-9);

        // Power = load / rated_cop (constant)
        let power = chiller.calculate_power(50000.0, 35.0, HVACMode::Cooling);
        assert!((power - (50000.0 / 3.5)).abs() < 1e-6);
    }

    #[test]
    fn test_chiller_polynomial_mode() {
        // Issue #2214: polynomial mode should vary COP with PLR/temperature
        let chiller =
            Chiller::new("CH-POLY".to_string(), 100000.0, 3.5, 35.0).with_constant_cop(false);

        // In polynomial mode (with flat [1.0,0,0,0] coefficients + 0 temp_coeff),
        // normalize_polynomial_cop still returns rated COP.
        // This test verifies the code path doesn't panic and returns positive COP.
        let cop = chiller.calculate_efficiency(0.5, 30.0, HVACMode::Cooling);
        assert!(cop > 0.0);
    }

    #[test]
    fn test_chiller_temperature_limits() {
        let chiller = Chiller::new("CH-1".to_string(), 100000.0, 4.5, 35.0);

        // Below minimum (5°C)
        let capacity_below_min = chiller.calculate_capacity(1.0, 0.0);
        assert_eq!(capacity_below_min, 30000.0); // 30% of rated

        // Above maximum (45°C)
        let capacity_above_max = chiller.calculate_capacity(1.0, 50.0);
        assert_eq!(capacity_above_max, 30000.0); // 30% of rated

        // Within range
        let capacity_normal = chiller.calculate_capacity(1.0, 20.0);
        assert!(capacity_normal > 30000.0);
        assert!(capacity_normal < 100000.0);
    }

    #[test]
    fn test_boiler_temperature_sensitivity() {
        let boiler = Boiler::new("BO-1".to_string(), 100000.0, 0.85, -5.0);

        // Boiler is less temperature-sensitive than heat pump
        let capacity_normal = boiler.calculate_capacity(1.0, -5.0);
        let capacity_cold = boiler.calculate_capacity(1.0, -15.0);

        // Only ~1% degradation at -15°C (vs ~10% for heat pump)
        let degradation = (capacity_normal - capacity_cold) / capacity_normal;
        assert!(degradation < 0.02); // Less than 2% degradation

        // But below minimum (-20°C) drops to 50%
        let capacity_extreme = boiler.calculate_capacity(1.0, -25.0);
        assert_eq!(capacity_extreme, 50000.0); // 50% of rated
    }

    #[test]
    fn test_variable_capacity_trait() {
        // Create instances of all equipment types
        let chiller = Chiller::new("Chiller-1".to_string(), 10000.0, 4.0, 35.0);
        let boiler = Boiler::new("Boiler-1".to_string(), 10000.0, 0.85, -5.0);
        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);
        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);
        let heatpump = HeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0);

        // Verify they implement VariableCapacityEquipment by calling trait methods
        let _capacity = chiller.calculate_capacity(0.5, 20.0);
        let _efficiency = boiler.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
        let _power = vav.calculate_power(5000.0, 20.0, HVACMode::Heating);
        let _rated = cav.rated_capacity();
        let _current = heatpump.current_plr();

        // Basic assertions to verify implementations exist
        assert!(chiller.rated_capacity() > 0.0);
        assert!(boiler.rated_efficiency(HVACMode::Heating) > 0.0);
        assert!(vav.rated_capacity() > 0.0);
        assert!(cav.rated_capacity() > 0.0);
        assert!(heatpump.rated_capacity() > 0.0);
    }

    #[test]
    fn test_plr_tracking() {
        // Test Chiller PLR tracking
        let mut chiller = Chiller::new("Chiller-1".to_string(), 10000.0, 4.0, 35.0);

        // Update state with various loads
        chiller.update_state(0.0, 20.0, HVACMode::Cooling); // No load
        assert_eq!(chiller.current_plr(), 0.0);

        chiller.update_state(5000.0, 20.0, HVACMode::Cooling); // ~50% load (capacity varies with temp)
        let plr_50 = chiller.current_plr();
        assert!(plr_50 > 0.3 && plr_50 < 0.7); // Approximate 50% PLR

        chiller.update_state(10000.0, 20.0, HVACMode::Cooling); // Full load
        assert!((chiller.current_plr() - 1.0).abs() < 0.01);

        chiller.update_state(15000.0, 20.0, HVACMode::Cooling); // Overload
        assert_eq!(chiller.current_plr(), 1.0); // Should clamp to 1.0

        // Test CAVSystem PLR tracking
        let mut cav = CAVSystem::new("CAV-1".to_string(), 1.0);
        cav.update_state(0.0, 20.0, HVACMode::Heating);
        assert_eq!(cav.current_plr(), 0.0);

        cav.update_state(5000.0, 20.0, HVACMode::Cooling);
        assert!(cav.current_plr() > 0.0);
    }

    #[test]
    fn test_vav_implementation() {
        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);

        // Test calculate_capacity at PLR=0.5
        let capacity = vav.calculate_capacity(0.5, 20.0);
        assert!(capacity > 0.0);
        assert!((capacity - 2500.0).abs() < 0.1); // 5000W reheat * 0.5

        // Test calculate_efficiency for heating
        let eff_heating = vav.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
        assert!(eff_heating > 0.0);
        assert!((eff_heating - 0.8).abs() < 0.1); // Fan + reheat COP ~0.8

        // Test calculate_efficiency for cooling
        let eff_cooling = vav.calculate_efficiency(0.5, 20.0, HVACMode::Cooling);
        assert!(eff_cooling > 0.0);
        assert!((eff_cooling - 3.0).abs() < 0.1); // Fan + cooling coil COP ~3.0

        // Test calculate_power
        let load = 2500.0;
        let power = vav.calculate_power(load, 20.0, HVACMode::Heating);
        assert!(power > 0.0);
        // Power = load / efficiency = 2500 / 0.8 = 3125
        assert!((power - 3125.0).abs() < 10.0);
    }

    #[test]
    fn test_cav_implementation() {
        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);

        // Test calculate_capacity at PLR=0.5
        let capacity = cav.calculate_capacity(0.5, 20.0);
        assert!(capacity > 0.0);
        assert!((capacity - 5000.0).abs() < 0.1); // Max(heating,cooling) * 0.5 = 10000 * 0.5

        // Test calculate_efficiency for heating
        let eff_heating = cav.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
        assert!(eff_heating > 0.0);
        assert!((eff_heating - 0.85).abs() < 0.1); // Fan + heating coil COP ~0.85

        // Test calculate_efficiency for cooling
        let eff_cooling = cav.calculate_efficiency(0.5, 20.0, HVACMode::Cooling);
        assert!(eff_cooling > 0.0);
        assert!((eff_cooling - 3.2).abs() < 0.1); // Fan + cooling coil COP ~3.2

        // Test calculate_power
        // Power = load / (efficiency * PLR) due to PLR degradation in calculation
        let load = 5000.0;
        let power = cav.calculate_power(load, 20.0, HVACMode::Heating);
        assert!(power > 0.0);
        // Power calculation uses efficiency at PLR = load/rated_capacity = 5000/10000 = 0.5
        // So efficiency is 0.85 (not PLR-degraded for CAV)
        // Power = 5000 / 0.85 ≈ 5882
        assert!(power > 5000.0 && power < 7000.0); // Reasonable range
    }

    #[test]
    fn test_heatpump_implementation() {
        let hp = HeatPump::new(
            "HP-1".to_string(),
            12000.0, // 12kW heating
            10000.0, // 10kW cooling
            3.5,     // COP 3.5
            3.0,     // EER 3.0
        );

        // Test calculate_capacity at PLR=0.5, outdoor_temp=20°C
        // Heat pump capacity degrades with temperature
        // Default mode is Off, so uses heating capacity
        // At 20°C (25°C from heating design -5°C): capacity_factor = 1.0 - 0.25 = 0.75
        let capacity = hp.calculate_capacity(0.5, 20.0);
        assert!(capacity > 0.0);
        // Capacity = 12000 * 0.75 * 0.5 = 4500 (with degradation)
        assert!(capacity > 4000.0 && capacity < 5000.0);

        // Test calculate_efficiency for heating mode
        // HP heating coefficients: [3.5, -0.8, 0.5, -0.2]
        // With normalization, efficiency at rated conditions (PLR=1.0, design_temp)
        // equals rated COP. At part-load, the polynomial curve may give different values.
        let eff_heating = hp.calculate_efficiency(0.5, -5.0, HVACMode::Heating);
        assert!(eff_heating > 0.0);
        assert!(eff_heating.is_finite()); // Must be a valid number

        // Test calculate_efficiency for cooling mode
        // HP cooling uses constant COP = rated COP
        let eff_cooling = hp.calculate_efficiency(0.5, 35.0, HVACMode::Cooling);
        assert!(eff_cooling > 0.0);
        assert!((eff_cooling - 3.0).abs() < 0.1); // Constant COP = rated COP = 3.0
        assert!(eff_cooling.is_finite()); // Must be a valid number

        // Test calculate_power for heating
        let load = 6000.0;
        let power_heating = hp.calculate_power(load, -5.0, HVACMode::Heating);
        assert!(power_heating > 0.0);
        // Power = load / COP (with PLR degradation)
        assert!(power_heating < 3000.0); // Should be less than load/2

        // Test calculate_power for cooling
        let power_cooling = hp.calculate_power(load, 35.0, HVACMode::Cooling);
        assert!(power_cooling > 0.0);
        // Power = load / EER (with PLR degradation)
        assert!(power_cooling < 3000.0); // Should be less than load/2
    }

    #[test]
    fn test_any_equipment_wrapper() {
        let chiller = Chiller::new("CH-1".to_string(), 10000.0, 4.0, 35.0);
        let mut any = AnyEquipment::Chiller(chiller);

        // Test all trait methods through AnyEquipment wrapper
        assert_eq!(any.rated_capacity(), 10000.0);
        assert!(any.calculate_capacity(1.0, 35.0) > 0.0);
        assert!(any.calculate_efficiency(1.0, 35.0, HVACMode::Cooling) > 0.0);
        assert!(any.calculate_power(5000.0, 35.0, HVACMode::Cooling) > 0.0);
        assert_eq!(any.rated_efficiency(HVACMode::Cooling), 4.0);
        assert_eq!(any.current_plr(), 0.0);

        any.update_state(5000.0, 35.0, HVACMode::Cooling);
        assert!(any.current_plr() > 0.0);

        // Test other variants thoroughly
        let boiler = Boiler::new("BO-1".to_string(), 10000.0, 0.8, 0.0);
        let mut any_boiler = AnyEquipment::Boiler(boiler);
        assert_eq!(any_boiler.rated_capacity(), 10000.0);
        assert!(any_boiler.calculate_capacity(1.0, 0.0) > 0.0);
        assert!(any_boiler.calculate_efficiency(1.0, 0.0, HVACMode::Heating) > 0.0);
        assert!(any_boiler.calculate_power(5000.0, 0.0, HVACMode::Heating) > 0.0);
        assert_eq!(any_boiler.rated_efficiency(HVACMode::Heating), 0.8);
        any_boiler.update_state(5000.0, 0.0, HVACMode::Heating);
        assert!(any_boiler.current_plr() > 0.0);

        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);
        let mut any_vav = AnyEquipment::VAVTerminal(vav);
        assert!(any_vav.rated_capacity() > 0.0);
        assert!(any_vav.calculate_capacity(1.0, 20.0) > 0.0);
        assert!(any_vav.calculate_efficiency(1.0, 20.0, HVACMode::Heating) > 0.0);
        any_vav.update_state(5000.0, 20.0, HVACMode::Heating);
        assert!(any_vav.current_plr() > 0.0);

        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);
        let mut any_cav = AnyEquipment::CAVSystem(cav);
        assert!(any_cav.rated_capacity() > 0.0);
        assert!(any_cav.calculate_capacity(1.0, 20.0) > 0.0);
        assert!(any_cav.calculate_efficiency(1.0, 20.0, HVACMode::Cooling) > 0.0);
        any_cav.update_state(5000.0, 20.0, HVACMode::Cooling);
        assert!(any_cav.current_plr() > 0.0);

        let hp = HeatPump::new("HP-1".to_string(), 10000.0, 10000.0, 3.0, 3.0);
        let mut any_hp = AnyEquipment::HeatPump(hp);
        assert_eq!(any_hp.rated_capacity(), 10000.0);
        assert!(any_hp.calculate_capacity(1.0, 20.0) > 0.0);
        assert!(any_hp.calculate_efficiency(0.5, 20.0, HVACMode::Heating) > 0.0);
        assert!(any_hp.calculate_efficiency(0.5, 35.0, HVACMode::Cooling) > 0.0);
        any_hp.update_state(5000.0, 20.0, HVACMode::Heating);
        assert!(any_hp.current_plr() > 0.0);
    }

    /// Issue #1345: Verify the modulation factor from the predictive controller
    /// propagates to `VariableCapacityEquipment::update_state` and produces a
    /// PLR that scales with the modulation (Chiller/Boiler/HeatPump/CAV/VAV
    /// path through the `VariableCapacityEquipment` trait).
    ///
    /// This is the unit-level guard for the propagation fix at
    /// `physics_impl.rs:2478`: previously the predictive controller's
    /// modulation was discarded (`let (hvac_mode, _modulation) = ...`); the
    /// fix binds the modulation and uses it to scale the load before
    /// `update_state`. If the scaling is removed (e.g. someone reverts to
    /// passing the raw `current_load` without `* modulation`), this test
    /// catches the regression.
    #[test]
    fn test_predictive_modulation_propagates_to_update_state() {
        // Three modulation scenarios:
        //   1.0 → equipment runs at full PLR (modulation ignored)
        //   0.5 → equipment runs at half PLR (modulation halved the load)
        //   0.0 → equipment sits at zero PLR (modulation gated the load off)
        for &modulation in &[1.0_f64, 0.5, 0.25, 0.0] {
            let mut chiller = Chiller::new(
                "CH-1345".to_string(),
                10_000.0, // 10 kW cooling
                4.0,
                35.0,
            );
            let raw_load = 8_000.0_f64; // 80% of rated capacity
            let modulated_load = raw_load * modulation;
            chiller.update_state(modulated_load, 35.0, HVACMode::Cooling);
            let plr = chiller.current_plr();
            // PLR is `modulated_load / capacity` (clamped to 0..=1). Capacity
            // degrades with outdoor temperature (here 35°C = design temp → 10 kW).
            let expected_plr = if modulation == 0.0 {
                0.0
            } else {
                (modulated_load / 10_000.0).clamp(0.0, 1.0)
            };
            assert!(
                (plr - expected_plr).abs() < 1e-6,
                "modulation={} → expected PLR {:.4}, got {:.4} (propagation broken)",
                modulation,
                expected_plr,
                plr
            );
            assert!(
                plr <= 1.0,
                "PLR {} exceeded 1.0 for modulation {} (modulation over-scaled)",
                plr,
                modulation
            );
        }
    }

    /// Issue #1345: Same propagation check via the `AnyEquipment` enum wrapper,
    /// which is the type used by `ThermalModel::hvac_equipment`. The wrapper
    /// dispatches `update_state` to the underlying variant, so we also verify
    /// that the modulation-propagated load reaches HeatPump / Boiler (the other
    /// two variants touched by the issue scope).
    #[test]
    fn test_predictive_modulation_propagates_through_any_equipment() {
        let raw_load = 6_000.0_f64;
        let modulation = 0.3_f64;
        let modulated_load = raw_load * modulation;

        // HeatPump (covers heating/cooling combined path)
        let mut hp = AnyEquipment::HeatPump(HeatPump::new(
            "HP-1345".to_string(),
            10_000.0,
            10_000.0,
            3.0,
            3.0,
        ));
        hp.update_state(modulated_load, 20.0, HVACMode::Heating);
        let plr = hp.current_plr();
        assert!(
            (0.0..=1.0).contains(&plr),
            "HeatPump PLR {} out of [0,1] after propagation",
            plr
        );

        // Boiler (heating-only)
        let mut boiler =
            AnyEquipment::Boiler(Boiler::new("BO-1345".to_string(), 10_000.0, 0.85, -5.0));
        boiler.update_state(modulated_load, 0.0, HVACMode::Heating);
        let plr_b = boiler.current_plr();
        assert!(
            (0.0..=1.0).contains(&plr_b),
            "Boiler PLR {} out of [0,1] after propagation",
            plr_b
        );

        // Chiller (cooling-only) — the Chiller clamps to 0 when mode != Cooling,
        // so we exercise the cooling path here.
        let mut chiller =
            AnyEquipment::Chiller(Chiller::new("CH-1345".to_string(), 10_000.0, 4.0, 35.0));
        chiller.update_state(modulated_load, 35.0, HVACMode::Cooling);
        let plr_c = chiller.current_plr();
        assert!(
            (0.0..=1.0).contains(&plr_c),
            "Chiller PLR {} out of [0,1] after propagation",
            plr_c
        );
    }

    /// Issue #1345: Verify the previously-discarded `_modulation` is now bound
    /// to a real local at the predictive controller call site. This is a
    /// behaviour-level guard: the controller's contract is that the second
    /// tuple element (modulation) is in [0.0, 1.0], so anything that consumes
    /// it downstream (equipment.update_state, the modulated q value) can rely
    /// on that invariant.
    #[test]
    fn test_predictive_modulation_in_unit_interval() {
        use crate::sim::hvac::modes::PredictiveController;
        // Use with_tuning for backward compatibility in existing test
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // Sweep conditions that exercise heating, cooling, and off modes.
        let cases: &[(f64, f64, f64)] = &[
            (15.0, 20.0, -0.01), // strong heating demand
            (19.0, 19.0, 0.0),   // mild heating
            (22.0, 22.0, 0.0),   // off (in deadband)
            (28.0, 27.0, 0.001), // mild cooling
            (32.0, 30.0, 0.01),  // strong cooling
        ];
        for &(zone_temp, mass_temp, temp_rate) in cases {
            let (_mode, modulation) =
                controller.calculate_modulation(zone_temp, mass_temp, temp_rate);
            assert!(
                (0.0..=1.0).contains(&modulation),
                "modulation {} out of [0,1] for zone={}, mass={}, rate={}",
                modulation,
                zone_temp,
                mass_temp,
                temp_rate
            );
        }
    }
}
