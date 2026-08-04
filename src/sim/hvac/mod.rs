//! HVAC System Models
//!
//! This module provides advanced HVAC system modeling capabilities including
//! Variable Air Volume (VAV), Constant Air Volume (CAV), and heat pump systems.

pub mod airside_coupling;
pub mod airside_state;
pub mod cav_terminal;
pub mod cooling_coil;
pub mod cycling;
pub mod doas;
pub mod economizer;
pub mod efficiency_curves;
pub mod equipment;
pub mod fan;
pub mod heating_coil;
pub mod ideal_loads;
pub mod modes;
pub mod part_load_curves;
pub mod plant;
pub mod refrigeration;
pub mod vav_terminal;
pub mod zone_equipment;
pub mod zones;

// Re-export common types for convenience
pub use airside_coupling::{AirsideEnvelopeCoupler, CoupledStepForcing, CoupledStepResult};
pub use airside_state::{
    AirsideCouplingError, AirsideFlow, MoistAirState, DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
    MAX_VALIDATED_TIMESTEP_SECONDS,
};
pub use cav_terminal::{
    CavOperatingMode, CavTerminal, CavTerminalControl, CavTerminalPerformance, CavTerminalUnit,
};
pub use cooling_coil::{CoilPerformance, CoolingCoil, CoolingCoilBehavior};
pub use cycling::CyclingTracker;
pub use doas::{Doas, DoasControl, DoasMode, DoasPerformance, DoasUnit};
pub use economizer::{calculate_free_cooling_capacity, is_economizer_active, EconomizerMode};
pub use efficiency_curves::{
    default_ahri_coefficients, CurveCoefficients, EfficiencyCurve, EfficiencyCurveConfig,
};
pub use equipment::{AnyEquipment, Boiler, Chiller, HVACMode, VariableCapacityEquipment};
pub use fan::{Fan, FanComponent, STANDARD_AIR_DENSITY_KG_PER_M3};
pub use heating_coil::{HeatingCoil, HeatingCoilComponent, HeatingCoilControl, HeatingCoilResult};
pub use ideal_loads::IdealLoadsSystem;
pub use modes::PredictiveController;
pub use part_load_curves::{
    boiler_part_load_coeffs, chiller_part_load_coeffs, vav_fan_power_coeffs,
    vav_fan_power_with_spr_coeffs, AshrStdCoeffs, BiquadraticCoeffs, BoilerPartLoadCurve,
    ChillerPartLoadCurve, CurveType, FanPowerCurve, PartLoadCurve, QuadraticCoeffs,
};
pub use plant::{
    check_energy_balance, water_cp, water_density, CoolingTowerSingleSpeed, FluidState,
    PlantComponent, PlantComponentResult, PlantLoop, PlantLoopResult, PlantMode, Pump,
    PumpConstantSpeed, PumpVariableSpeed, WATER_CP_J_PER_KG_K, WATER_DENSITY_KG_PER_M3,
};
pub use refrigeration::{
    AirCooledCondenser, CompressorRack, RefrigerationMode, RefrigerationSystem, WalkInCooler,
    WalkInFreezer,
};
pub use vav_terminal::{
    VavOperatingMode, VavTerminal, VavTerminalControl, VavTerminalPerformance, VavTerminalUnit,
};
pub use zone_equipment::{
    AnyZoneEquipment, BaseboardHeater, FourPipeFanCoil, HotWaterBaseboard,
    LowTemperatureRadiantSurface, PackagedTerminalAC, PackagedTerminalHeatPump, RadiantSurfaceType,
    ZoneEquipment, ZoneEquipmentMode, ZoneEquipmentSetpoints, ZoneHeatInjection,
};

use serde::{Deserialize, Serialize};

/// HVAC system types supported by the simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HVACSystemType {
    /// Simple on/off HVAC with fixed capacity
    Simple,
    /// Variable Air Volume system with terminal reheat
    VAV,
    /// Constant Air Volume system with fixed airflow
    CAV,
    /// Heat pump system with COP curves
    HeatPump,
    /// Ideal air loads system with infinite capacity (for ASHRAE 140 validation)
    Ideal,
}

/// Represents a VAV (Variable Air Volume) terminal unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAVTerminal {
    /// Terminal unit identifier
    pub id: String,
    /// Zone served by this terminal
    pub zone_id: usize,
    /// Maximum air flow rate (m³/s)
    pub max_airflow: f64,
    /// Minimum air flow rate (m³/s)
    pub min_airflow: f64,
    /// Reheat coil capacity (W)
    pub reheat_capacity: f64,
    /// Current airflow setpoint (m³/s)
    pub airflow_setpoint: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
}

impl VAVTerminal {
    /// Create a new VAV terminal unit
    pub fn new(id: String, zone_id: usize, max_airflow: f64) -> Self {
        Self {
            id,
            zone_id,
            max_airflow,
            min_airflow: max_airflow * 0.3, // Minimum 30% of max
            reheat_capacity: 5000.0,        // Default 5kW reheat
            airflow_setpoint: max_airflow,
            current_plr: 0.0,
        }
    }

    /// Calculate heating demand from reheat coil
    pub fn reheat_demand(&self, supply_temp: f64, zone_temp: f64) -> f64 {
        if zone_temp < 20.0 {
            // Need reheat to maintain minimum supply temp
            let temp_diff = (supply_temp - 18.0).max(0.0);
            // Q = ρ * cp * V̇ * ΔT
            let rho = 1.2; // kg/m³
            let cp = 1005.0; // J/kg·K
            let mass_flow = self.airflow_setpoint * rho; // kg/s
            mass_flow * cp * temp_diff
        } else {
            0.0
        }
    }
}

/// Outdoor temperature bin with hours (°C, hours)
pub type TempBin = (f64, f64);

/// Represents a CAV (Constant Air Volume) system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CAVSystem {
    /// System identifier
    pub id: String,
    /// Design air flow rate (m³/s)
    pub design_airflow: f64,
    /// Fan power consumption (W)
    pub fan_power: f64,
    /// Fan efficiency (0-1)
    pub fan_efficiency: f64,
    /// Heating coil capacity (W)
    pub heating_capacity: f64,
    /// Cooling coil capacity (W)
    pub cooling_capacity: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Terminal unit for psychrometric calculations
    pub terminal: CavTerminalUnit,
}

impl CAVSystem {
    /// Create a new CAV system with a terminal unit.
    ///
    /// The terminal is created with cooling and optional heating coils sized
    /// to the system's heating/cooling capacities.
    pub fn new(id: String, design_airflow: f64) -> Self {
        let cooling = CoolingCoil::new(
            format!("{id}-CC"),
            10000.0, // rated capacity W
            0.75,    // rated SHR
            0.15,    // bypass factor
            10.0,    // ADP (°C)
            2.0,     // design mass flow kg/s
        );

        let heating = HeatingCoilComponent::new(
            format!("{id}-HC"),
            10000.0, // rated capacity W
            2.0,     // design mass flow kg/s
        );

        let terminal = CavTerminalUnit::new(id.clone(), 0, design_airflow, cooling, Some(heating));

        Self {
            id,
            design_airflow,
            fan_power: design_airflow * 500.0, // Default 500 W per m³/s
            fan_efficiency: 0.7,
            heating_capacity: 10000.0, // Default 10kW
            cooling_capacity: 10000.0, // Default 10kW
            current_plr: 0.0,
            terminal,
        }
    }

    /// Create a CAV system with custom coil capacities.
    pub fn with_coils(
        id: String,
        design_airflow: f64,
        cooling_capacity: f64,
        heating_capacity: f64,
    ) -> Self {
        let cooling = CoolingCoil::new(
            format!("{id}-CC"),
            cooling_capacity,
            0.75, // rated SHR
            0.15, // bypass factor
            10.0, // ADP (°C)
            2.0,  // design mass flow kg/s
        );

        let heating = HeatingCoilComponent::new(
            format!("{id}-HC"),
            heating_capacity,
            2.0, // design mass flow kg/s
        );

        let terminal = CavTerminalUnit::new(id.clone(), 0, design_airflow, cooling, Some(heating));

        Self {
            id,
            design_airflow,
            fan_power: design_airflow * 500.0,
            fan_efficiency: 0.7,
            heating_capacity,
            cooling_capacity,
            current_plr: 0.0,
            terminal,
        }
    }

    /// Calculate fan power consumption
    pub fn fan_power_consumption(&self) -> f64 {
        self.fan_power / self.fan_efficiency
    }

    /// Simulate annual energy consumption using the psychrometric model.
    ///
    /// Uses proper mode transitions (Cooling/Heating/Deadband) based on
    /// zone setpoints and outdoor temperatures.
    ///
    /// # Arguments
    /// * `outdoor_temps_h` - Outdoor temperature bins with hours: `[(temp_c, hours), ...]`
    /// * `cooling_setpoint` - Zone cooling setpoint (°C)
    /// * `heating_setpoint` - Zone heating setpoint (°C)
    /// * `_deadband` - Deadband width (°C) [reserved for future use]
    ///
    /// # Returns
    /// * `(annual_energy_kwh, peak_demand_w)`
    #[allow(dead_code)]
    pub fn simulate_annual(
        &self,
        outdoor_temps_h: &[(f64, f64)],
        cooling_setpoint: f64,
        heating_setpoint: f64,
        _deadband: f64,
    ) -> (f64, f64) {
        let start = std::time::Instant::now();
        let standard_pressure_pa = 101325.0_f64;

        let mut total_energy_kwh: f64 = 0.0;
        let mut peak_demand_w: f64 = 0.0;

        for &(outdoor_temp, hours) in outdoor_temps_h {
            if hours == 0.0 {
                continue;
            }

            // Determine operating mode based on outdoor temperature and setpoints
            // For CAV serving a conditioned zone:
            // - If outdoor temp > cooling setpoint → Cooling mode
            // - If outdoor temp < heating setpoint → Heating mode
            // - Otherwise → Deadband (ventilation only)
            let control = if outdoor_temp > cooling_setpoint {
                // Cooling mode: zone needs cooling, terminal cools
                CavTerminalControl::cooling()
            } else if outdoor_temp < heating_setpoint {
                // Heating mode: zone needs heating, terminal heats to maintain
                // supply air at heating setpoint
                let supply_setpoint = heating_setpoint + 5.0; // Slightly above zone heating setpoint
                CavTerminalControl::heating(supply_setpoint)
            } else {
                // Deadband: no cooling/heating needed, ventilation only
                CavTerminalControl::deadband()
            };

            // Derive entering air conditions from outdoor temperature.
            // For a CAV terminal serving a zone, the entering air at the coil
            // is a mix of outdoor air and return air. At higher outdoor temps,
            // the entering dry-bulb is warmer.
            let entering_dry_bulb_c = 20.0 + (outdoor_temp - 5.0).clamp(0.0, 10.0);
            let entering_rh_percent = 50.0;

            let entering = match MoistAirState::try_new(
                entering_dry_bulb_c,
                entering_rh_percent,
                standard_pressure_pa,
            ) {
                Ok(state) => state,
                Err(_) => continue,
            };

            let air_density = entering.density_kg_per_m3;

            let perf =
                match self
                    .terminal
                    .compute_terminal_performance(&entering, air_density, &control)
                {
                    Ok(p) => p,
                    Err(_) => continue,
                };

            // Total power = fan motor power + any coil power overhead
            // For cooling mode: fan motor power
            // For heating mode: fan motor power + heating coil uses heating plant energy (not counted here)
            // For deadband: minimum fan power
            let power = perf.fan_motor_power_w;

            total_energy_kwh += power * hours / 1000.0;
            peak_demand_w = peak_demand_w.max(power);
        }

        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            eprintln!(
                "  CAVSystem::simulate_annual: {:.2}s",
                elapsed.as_secs_f64()
            );
        }

        (total_energy_kwh, peak_demand_w)
    }
}

/// Heat pump operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeatPumpMode {
    /// Heating mode
    Heating,
    /// Cooling mode
    Cooling,
    /// Off
    Off,
}

/// Represents a heat pump system with COP curves
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatPump {
    /// System identifier
    pub id: String,
    /// Rated heating capacity at design conditions (W)
    pub heating_capacity: f64,
    /// Rated cooling capacity at design conditions (W)
    pub cooling_capacity: f64,
    /// Rated heating COP at design conditions
    pub heating_cop: f64,
    /// Rated cooling COP (EER) at design conditions
    pub cooling_cop: f64,
    /// Design outdoor temperature for heating (°C)
    pub design_temp_heating: f64,
    /// Design outdoor temperature for cooling (°C)
    pub design_temp_cooling: f64,
    /// Current operating mode
    pub mode: HeatPumpMode,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Polynomial efficiency curve for heating mode
    pub efficiency_curve_heating: efficiency_curves::EfficiencyCurve,
    /// Polynomial efficiency curve for cooling mode
    pub efficiency_curve_cooling: efficiency_curves::EfficiencyCurve,
}

impl HeatPump {
    /// Create a new heat pump
    pub fn new(
        id: String,
        heating_capacity: f64,
        cooling_capacity: f64,
        heating_cop: f64,
        cooling_cop: f64,
    ) -> Self {
        // Use default AHRI coefficients for now
        let default_coeffs = efficiency_curves::default_ahri_coefficients();

        Self {
            id,
            heating_capacity,
            cooling_capacity,
            heating_cop,
            cooling_cop,
            design_temp_heating: -5.0, // Design heating temp
            design_temp_cooling: 35.0, // Design cooling temp
            mode: HeatPumpMode::Off,
            current_plr: 0.0,
            efficiency_curve_heating: (&default_coeffs.heatpump_heating).into(),
            efficiency_curve_cooling: (&default_coeffs.heatpump_cooling).into(),
        }
    }

    /// Calculate actual COP based on outdoor temperature
    /// Uses constant rated COP (no temperature degradation) to match reference behavior
    pub fn heating_cop_at_temperature(&self, _outdoor_temp: f64) -> f64 {
        self.heating_cop
    }

    /// Calculate actual COP based on outdoor temperature for cooling
    pub fn cooling_cop_at_temperature(&self, _outdoor_temp: f64) -> f64 {
        self.cooling_cop
    }

    /// Calculate heating power consumption
    pub fn heating_power(&self, outdoor_temp: f64) -> f64 {
        if self.mode != HeatPumpMode::Heating {
            return 0.0;
        }
        // Capacity also degrades with temperature
        let temp_diff = (self.design_temp_heating - outdoor_temp).abs();
        let capacity_factor = 1.0 - (temp_diff * 0.01);
        let actual_capacity = self.heating_capacity * capacity_factor.max(0.3);

        let cop = self.heating_cop_at_temperature(outdoor_temp);
        actual_capacity / cop
    }

    /// Calculate cooling power consumption
    pub fn cooling_power(&self, outdoor_temp: f64) -> f64 {
        if self.mode != HeatPumpMode::Cooling {
            return 0.0;
        }
        let temp_diff = (outdoor_temp - self.design_temp_cooling).abs();
        let capacity_factor = 1.0 - (temp_diff * 0.015);
        let actual_capacity = self.cooling_capacity * capacity_factor.max(0.3);

        let cop = self.cooling_cop_at_temperature(outdoor_temp);
        actual_capacity / cop
    }

    /// Set the operating mode based on zone temperature and setpoints
    pub fn set_mode(&mut self, zone_temp: f64, heating_sp: f64, cooling_sp: f64) {
        self.mode = if zone_temp < heating_sp {
            HeatPumpMode::Heating
        } else if zone_temp > cooling_sp {
            HeatPumpMode::Cooling
        } else {
            HeatPumpMode::Off
        };
    }

    fn normalize_polynomial_cop(
        &self,
        curve: &efficiency_curves::EfficiencyCurve,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vav_terminal() {
        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);
        assert_eq!(vav.max_airflow, 0.5);
        assert_eq!(vav.min_airflow, 0.15);

        // Test reheat demand: supply_temp (20°C) > zone_temp (18°C), needs reheat
        let demand = vav.reheat_demand(20.0, 18.0);
        assert!(demand > 0.0);

        // No reheat needed when zone temp is comfortable
        let no_demand = vav.reheat_demand(20.0, 22.0);
        assert!(no_demand == 0.0);
    }

    #[test]
    fn test_cav_system() {
        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);
        assert_eq!(cav.design_airflow, 1.0);
        assert!(cav.fan_power_consumption() > 0.0);
    }

    #[test]
    fn test_heat_pump_cop() {
        let hp = HeatPump::new(
            "HP-1".to_string(),
            12000.0, // 12kW heating
            10000.0, // 10kW cooling
            3.5,     // COP 3.5
            3.0,     // EER 3.0
        );

        // At design temperature, COP should be rated COP
        let cop_at_design = hp.heating_cop_at_temperature(-5.0);
        assert!((cop_at_design - 3.5).abs() < 0.1);

        // At any temperature, COP should be constant (no temperature degradation)
        let cop_cold = hp.heating_cop_at_temperature(-15.0);
        assert!((cop_cold - 3.5).abs() < 0.1);
    }

    #[test]
    fn test_heat_pump_mode() {
        let mut hp = HeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0);

        hp.set_mode(18.0, 20.0, 27.0);
        assert_eq!(hp.mode, HeatPumpMode::Heating);

        hp.set_mode(28.0, 20.0, 27.0);
        assert_eq!(hp.mode, HeatPumpMode::Cooling);

        hp.set_mode(22.0, 20.0, 27.0);
        assert_eq!(hp.mode, HeatPumpMode::Off);
    }

    #[test]
    fn test_heat_pump_power_consumption() {
        let mut hp = HeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0);

        // Test heating power
        hp.mode = HeatPumpMode::Heating;
        let power_heat = hp.heating_power(0.0);
        assert!(power_heat > 0.0);

        // Power should be 0 when mode is off
        hp.mode = HeatPumpMode::Off;
        assert_eq!(hp.heating_power(0.0), 0.0);

        // Test cooling power
        hp.mode = HeatPumpMode::Cooling;
        let power_cool = hp.cooling_power(30.0);
        assert!(power_cool > 0.0);

        // Power should be 0 when mode is off
        hp.mode = HeatPumpMode::Off;
        assert_eq!(hp.cooling_power(30.0), 0.0);
    }

    #[test]
    fn test_heat_pump_cop_cooling() {
        let hp = HeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0);

        // At design cooling temp (35.0), COP should be rated
        let cop_at_design = hp.cooling_cop_at_temperature(35.0);
        assert!((cop_at_design - 3.0).abs() < 0.1);

        // At any temperature, COP should be constant (no temperature degradation)
        let cop_hot = hp.cooling_cop_at_temperature(45.0);
        assert!((cop_hot - 3.0).abs() < 0.1);
    }
}
