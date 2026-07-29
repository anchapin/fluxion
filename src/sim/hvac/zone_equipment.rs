//! Zone Equipment Models
//!
//! This module provides zone-level HVAC equipment models including baseboard heaters,
//! radiant surfaces, PTAC/PTHP units, and fan coil units. Each equipment type
//! implements the `ZoneEquipment` trait and contributes to the zone heat balance
//! through the `ZoneHeatInjection` result struct.
//!
//! # Zone Heat Balance Integration
//!
//! Zone equipment injects heat into the zone through two paths:
//! - **Air node** (`q_air`): Convective heat that directly affects zone air temperature
//! - **Surface node** (`q_surface_radiant`): Radiant heat that is absorbed by surfaces
//!   and subsequently released to the zone air through convection
//!
//! The split between these paths is characterized by the radiant fraction (`fr_rad`).
//!
//! # Equipment Types
//!
//! | Equipment | Heat Path | Notes |
//! |-----------|-----------|-------|
//! | Electric Baseboard | 100% q_air | Pure convective, instant response |
//! | Hot Water Baseboard | 100% q_air | Water-side dynamics add lag |
//! | Radiant Floor | q_surface_radiant | Low-temperature surface heating |
//! | Radiant Ceiling | q_surface_radiant | Panel radiant heating |
//! | PTAC | q_air (sensible + latent) | Package terminal AC |
//! | PTHP | q_air (sensible + latent) | Package terminal heat pump |
//! | Fan Coil Unit | q_air (sensible + latent) | 4-pipe FCU |

use serde::{Deserialize, Serialize};

/// Result of a zone equipment step, representing heat injection into the zone.
///
/// This struct captures the heat contribution from zone equipment to the zone
/// heat balance. The total heat is split between air-node and surface-node
/// paths based on the equipment's radiant fraction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZoneHeatInjection {
    /// Heat injection into the air node (W). Positive = heating, negative = cooling.
    pub q_air: f64,
    /// Heat injection into the surface node via radiation (W).
    /// This heat is absorbed by surfaces and later released to the zone air.
    pub q_surface_radiant: f64,
    /// Latent heat gain to the zone (W). Positive = humidification, negative = dehumidification.
    pub q_latent: f64,
    /// Electrical power consumption (W). Includes fan, controls, etc.
    pub electrical_power: f64,
    /// Water-side heat transfer (W). Positive = heat delivered to water, negative = heat extracted.
    /// Only meaningful for water-based equipment (HW baseboard, FCU).
    pub q_water_side: f64,
    /// Part-load ratio of the equipment (0.0 to 1.0).
    pub part_load_ratio: f64,
    /// Operating mode of the equipment.
    pub mode: ZoneEquipmentMode,
}

impl Default for ZoneHeatInjection {
    fn default() -> Self {
        Self {
            q_air: 0.0,
            q_surface_radiant: 0.0,
            q_latent: 0.0,
            electrical_power: 0.0,
            q_water_side: 0.0,
            part_load_ratio: 0.0,
            mode: ZoneEquipmentMode::Off,
        }
    }
}

impl ZoneHeatInjection {
    /// Create a new zone heat injection result.
    pub fn new(
        q_air: f64,
        q_surface_radiant: f64,
        q_latent: f64,
        electrical_power: f64,
        q_water_side: f64,
        part_load_ratio: f64,
        mode: ZoneEquipmentMode,
    ) -> Self {
        Self {
            q_air,
            q_surface_radiant,
            q_latent,
            electrical_power,
            q_water_side,
            part_load_ratio,
            mode,
        }
    }

    /// Create a zero (off) heat injection.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Total convective heat to the zone (air node + latent equivalent).
    pub fn total_convective(&self) -> f64 {
        self.q_air + self.q_latent
    }

    /// Total heat injection to the zone (all paths).
    pub fn total(&self) -> f64 {
        self.q_air + self.q_surface_radiant + self.q_latent
    }
}

/// Operating mode of zone equipment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneEquipmentMode {
    /// Equipment is off.
    Off,
    /// Equipment is heating.
    Heating,
    /// Equipment is cooling.
    Cooling,
    /// Equipment is in deadband (neither heating nor cooling).
    Deadband,
}

/// Zone equipment operating setpoints for control.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZoneEquipmentSetpoints {
    /// Zone heating setpoint (°C).
    pub heating_setpoint: f64,
    /// Zone cooling setpoint (°C).
    pub cooling_setpoint: f64,
    /// Zone air temperature (°C).
    pub zone_temp: f64,
    /// Outdoor air temperature (°C).
    pub outdoor_temp: f64,
    /// Supply water temperature for water-based equipment (°C).
    pub supply_water_temp: Option<f64>,
    /// Return water temperature for water-based equipment (°C).
    pub return_water_temp: Option<f64>,
    /// Zone humidity ratio (kg_water/kg_dry_air).
    pub humidity_ratio: Option<f64>,
    /// Supply air humidity ratio for air-based equipment.
    pub supply_humidity_ratio: Option<f64>,
}

impl ZoneEquipmentSetpoints {
    /// Create new setpoints for air-based equipment.
    pub fn new(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        zone_temp: f64,
        outdoor_temp: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            zone_temp,
            outdoor_temp,
            supply_water_temp: None,
            return_water_temp: None,
            humidity_ratio: None,
            supply_humidity_ratio: None,
        }
    }

    /// Create new setpoints for water-based equipment.
    pub fn with_water(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        zone_temp: f64,
        outdoor_temp: f64,
        supply_water_temp: f64,
        return_water_temp: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            zone_temp,
            outdoor_temp,
            supply_water_temp: Some(supply_water_temp),
            return_water_temp: Some(return_water_temp),
            humidity_ratio: None,
            supply_humidity_ratio: None,
        }
    }

    /// Create new setpoints with humidity for equipment with latent capacity.
    pub fn with_humidity(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        zone_temp: f64,
        outdoor_temp: f64,
        humidity_ratio: f64,
        supply_humidity_ratio: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            zone_temp,
            outdoor_temp,
            supply_water_temp: None,
            return_water_temp: None,
            humidity_ratio: Some(humidity_ratio),
            supply_humidity_ratio: Some(supply_humidity_ratio),
        }
    }
}

/// Trait for zone-level HVAC equipment.
///
/// This trait provides a unified interface for zone equipment that injects heat
/// into the zone heat balance. Equipment implements `step()` to compute heat
/// injection based on zone conditions and setpoints.
pub trait ZoneEquipment: Send + Sync + Clone {
    /// Execute one simulation timestep for the zone equipment.
    ///
    /// # Arguments
    /// * `setpoints` - Current zone and outdoor conditions
    /// * `dt` - Timestep duration (s)
    ///
    /// # Returns
    /// Heat injection into the zone heat balance.
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, dt: f64) -> ZoneHeatInjection;

    /// Get the nominal capacity of the equipment (W).
    fn nominal_capacity(&self) -> f64;

    /// Get the equipment type identifier.
    fn equipment_type(&self) -> &'static str;

    /// Reset the equipment to its initial state.
    fn reset(&mut self);
}

/// Electric baseboard heater.
///
/// Electric resistance heating with 100% convective heat transfer to the zone air.
/// This is the simplest zone equipment type - it provides instantaneous, 100% efficient
/// electric heating with no thermal lag or water-side complexity.
///
/// # EnergyPlus Model
///
/// EnergyPlus `ZoneHVAC:Baseboard:RadiantConvective:Electric` models this as:
/// - Nominal capacity (W)
/// - Efficiency (fraction of electrical input converted to thermal output)
/// - Fraction radiant (typically 0 for electric baseboard)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseboardHeater {
    /// Equipment identifier.
    pub id: String,
    /// Rated heating capacity (W).
    pub capacity: f64,
    /// Heating efficiency (0.0 to 1.0). For electric baseboard, typically 0.95-1.0.
    pub efficiency: f64,
    /// Control deadband (K). Prevents rapid cycling.
    pub deadband: f64,
    /// Current part-load ratio from last step (0.0 to 1.0).
    pub current_plr: f64,
}

impl BaseboardHeater {
    /// Create a new electric baseboard heater.
    pub fn new(id: String, capacity: f64) -> Self {
        Self {
            id,
            capacity,
            efficiency: 0.95, // Typical electric baseboard efficiency
            deadband: 0.5,    // 0.5 K deadband to prevent short-cycling
            current_plr: 0.0,
        }
    }

    /// Create a new electric baseboard heater with custom efficiency.
    pub fn with_efficiency(id: String, capacity: f64, efficiency: f64) -> Self {
        Self {
            id,
            capacity,
            efficiency,
            deadband: 0.5,
            current_plr: 0.0,
        }
    }
}

impl ZoneEquipment for BaseboardHeater {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, _dt: f64) -> ZoneHeatInjection {
        let t_zone = setpoints.zone_temp;
        let t_heating = setpoints.heating_setpoint;
        let t_cooling = setpoints.cooling_setpoint;

        // Determine heating requirement with deadband
        let target_temp = if t_zone < (t_heating - self.deadband / 2.0) {
            // Need heating: aim for heating setpoint
            t_heating
        } else if t_zone > (t_cooling + self.deadband / 2.0) {
            // Zone too hot - baseboard can't cool
            return ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: 0.0,
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            };
        } else {
            // In deadband - no heating needed
            return ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: 0.0,
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            };
        };

        // Calculate required heating
        let temp_deficit = target_temp - t_zone;
        let required_heating = self.capacity * (temp_deficit / 10.0).clamp(0.0, 1.0);
        let plr = (required_heating / self.capacity).clamp(0.0, 1.0);

        self.current_plr = plr;

        // 100% convective (electric resistance heating)
        let q_air = required_heating * self.efficiency;
        let electrical_power = required_heating / self.efficiency;

        ZoneHeatInjection {
            q_air,
            q_surface_radiant: 0.0,
            q_latent: 0.0,
            electrical_power,
            q_water_side: 0.0,
            part_load_ratio: plr,
            mode: ZoneEquipmentMode::Heating,
        }
    }

    fn nominal_capacity(&self) -> f64 {
        self.capacity
    }

    fn equipment_type(&self) -> &'static str {
        "ElectricBaseboard"
    }

    fn reset(&mut self) {
        self.current_plr = 0.0;
    }
}

/// Hot water baseboard heater.
///
/// Hot water baseboard with convective fins. Heat transfer is purely convective
/// to the zone air, but the water-side dynamics introduce a thermal lag.
///
/// # EnergyPlus Model
///
/// EnergyPlus `ZoneHVAC:Baseboard:RadiantConvective:Water` models this as:
/// - Rated capacity (W)
/// - Water flow rate (kg/s)
/// - Water inlet temperature (°C)
/// - Fraction radiant (typically 0 for baseboard)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotWaterBaseboard {
    /// Equipment identifier.
    pub id: String,
    /// Rated heating capacity (W).
    pub capacity: f64,
    /// Fraction of heat that is radiant (0.0 to 1.0). Typically 0.0 for baseboard.
    pub fraction_radiant: f64,
    /// Control deadband (K).
    pub deadband: f64,
    /// Water mass flow rate (kg/s).
    pub water_flow_rate: f64,
    /// Specific heat of water (J/kg·K).
    pub water_cp: f64,
    /// Previous water inlet temperature for lag calculation (°C).
    pub prev_inlet_temp: f64,
    /// Current part-load ratio.
    pub current_plr: f64,
    /// Water mass for dynamic calculation (kg).
    pub water_mass: f64,
}

impl HotWaterBaseboard {
    /// Create a new hot water baseboard heater.
    pub fn new(id: String, capacity: f64, water_flow_rate: f64) -> Self {
        Self {
            id,
            capacity,
            fraction_radiant: 0.0, // Baseboard is typically fully convective
            deadband: 0.5,
            water_flow_rate,
            water_cp: 4186.0,      // J/kg·K at 60°C
            prev_inlet_temp: 60.0, // DefaultHW supply temp
            current_plr: 0.0,
            water_mass: 1.0, // 1 kg of water in the system
        }
    }

    /// Create with custom radiant fraction.
    pub fn with_fraction_radiant(
        id: String,
        capacity: f64,
        water_flow_rate: f64,
        fr_rad: f64,
    ) -> Self {
        let mut bb = Self::new(id, capacity, water_flow_rate);
        bb.fraction_radiant = fr_rad;
        bb
    }
}

impl ZoneEquipment for HotWaterBaseboard {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, dt: f64) -> ZoneHeatInjection {
        let t_zone = setpoints.zone_temp;
        let t_heating = setpoints.heating_setpoint;
        let t_cooling = setpoints.cooling_setpoint;

        let supply_temp = setpoints.supply_water_temp.unwrap_or(60.0);
        let return_temp = setpoints.return_water_temp.unwrap_or(40.0);

        // Determine heating requirement
        if t_zone >= t_heating - self.deadband / 2.0 || t_zone > t_cooling + self.deadband / 2.0 {
            // No heating needed or in cooling mode
            // Decay water temperature toward ambient
            let temp_decay = (self.prev_inlet_temp - t_zone) * 0.01 * dt / 60.0;
            self.prev_inlet_temp -= temp_decay;

            return ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: 0.0,
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            };
        }

        // Calculate water-side heat transfer
        let delta_t_water = (supply_temp - return_temp).max(1.0);
        let q_water = self.water_flow_rate * self.water_cp * delta_t_water;
        let plr = (q_water / self.capacity).clamp(0.0, 1.0);

        // Radiant fraction goes to surface, convective to air
        let q_convective = q_water * (1.0 - self.fraction_radiant);
        let q_radiant = q_water * self.fraction_radiant;

        // Water-side dynamics with lag
        let _water_heat_capacity = self.water_mass * self.water_cp;
        let lag_factor = 1.0 - (-dt / 30.0_f64).exp(); // 30s time constant
        let effective_inlet =
            supply_temp - (supply_temp - self.prev_inlet_temp) * (1.0 - lag_factor);
        self.prev_inlet_temp = effective_inlet;

        self.current_plr = plr;

        ZoneHeatInjection {
            q_air: q_convective,
            q_surface_radiant: q_radiant,
            q_latent: 0.0,
            electrical_power: 0.0, // HW baseboard has no electrical power
            q_water_side: q_water,
            part_load_ratio: plr,
            mode: ZoneEquipmentMode::Heating,
        }
    }

    fn nominal_capacity(&self) -> f64 {
        self.capacity
    }

    fn equipment_type(&self) -> &'static str {
        "HotWaterBaseboard"
    }

    fn reset(&mut self) {
        self.current_plr = 0.0;
        self.prev_inlet_temp = 60.0;
    }
}

/// Low-temperature radiant surface (floor or ceiling).
///
/// Radiant heating surfaces heat the zone through thermal radiation absorbed by
/// building surfaces (floors, walls, ceiling). The radiant fraction is absorbed
/// directly by surfaces, while the convective fraction goes to the zone air.
///
/// # EnergyPlus Model
///
/// EnergyPlus `ZoneHVAC:LowTemperatureRadiant` models this as:
/// - Surface temperature limits (°C)
/// - Maximum water flow rate (kg/s)
/// - Heating capacity (W/m²)
/// - Fraction radiant vs convective
///
/// # Thermal Model
///
/// The radiant heat is distributed to surfaces and then convected to the zone air.
/// The surface temperature history affects subsequent heat transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowTemperatureRadiantSurface {
    /// Equipment identifier.
    pub id: String,
    /// Surface type (Floor or Ceiling).
    pub surface_type: RadiantSurfaceType,
    /// Rated heating capacity (W).
    pub capacity: f64,
    /// Fraction of heat that is radiant (0.0 to 1.0).
    /// Typical values: floor = 0.5-0.6, ceiling = 0.4-0.5.
    pub fraction_radiant: f64,
    /// Surface temperature (°C) - used for radiant heat calculation.
    pub surface_temp: f64,
    /// Mean radiant temperature of surrounding surfaces (°C).
    pub mean_radiant_temp: f64,
    /// Control deadband (K).
    pub deadband: f64,
    /// Maximum surface temperature (°C) - safety limit.
    pub max_surface_temp: f64,
    /// Current part-load ratio.
    pub current_plr: f64,
    /// Stefan-Boltzmann constant (W/m²·K⁴).
    pub stefan_boltzmann: f64,
    /// Emissivity of the radiant surface.
    pub emissivity: f64,
    /// Surface area (m²).
    pub area: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadiantSurfaceType {
    Floor,
    Ceiling,
}

impl LowTemperatureRadiantSurface {
    /// Create a new radiant floor surface.
    pub fn new_floor(id: String, capacity: f64, area: f64) -> Self {
        Self {
            id,
            surface_type: RadiantSurfaceType::Floor,
            capacity,
            fraction_radiant: 0.55, // Floor radiant typically 50-60%
            surface_temp: 20.0,
            mean_radiant_temp: 20.0,
            deadband: 0.5,
            max_surface_temp: 29.0, // Floor temp limit per ASHRAE 140
            current_plr: 0.0,
            stefan_boltzmann: 5.67e-8,
            emissivity: 0.9,
            area,
        }
    }

    /// Create a new radiant ceiling surface.
    pub fn new_ceiling(id: String, capacity: f64, area: f64) -> Self {
        Self {
            id,
            surface_type: RadiantSurfaceType::Ceiling,
            capacity,
            fraction_radiant: 0.45, // Ceiling radiant typically 40-50%
            surface_temp: 20.0,
            mean_radiant_temp: 20.0,
            deadband: 0.5,
            max_surface_temp: 40.0, // Higher limit for ceiling
            current_plr: 0.0,
            stefan_boltzmann: 5.67e-8,
            emissivity: 0.9,
            area,
        }
    }
}

impl ZoneEquipment for LowTemperatureRadiantSurface {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, dt: f64) -> ZoneHeatInjection {
        let t_zone = setpoints.zone_temp;
        let t_heating = setpoints.heating_setpoint;
        let t_cooling = setpoints.cooling_setpoint;

        // Determine if heating is needed
        let effective_temp = 0.5 * t_zone + 0.5 * self.mean_radiant_temp;

        if effective_temp >= t_heating - self.deadband / 2.0
            || effective_temp > t_cooling + self.deadband / 2.0
        {
            // No heating needed
            // Surface temperature decays toward zone air temperature
            let decay_rate = 1.0 - (-dt / 300.0_f64).exp(); // 5 min time constant
            self.surface_temp += (t_zone - self.surface_temp) * decay_rate;

            return ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: 0.0,
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            };
        }

        // Calculate required heating
        let temp_deficit = t_heating - effective_temp;
        let plr = (temp_deficit / 5.0).clamp(0.0, 1.0); // Assume 5K deficit = full capacity

        let q_total = self.capacity * plr;
        let q_radiant = q_total * self.fraction_radiant;
        let q_convective = q_total * (1.0 - self.fraction_radiant);

        // Update surface temperature (energy balance on the surface node)
        // Q_in = emissivity * sigma * A * (T_surf^4 - T_mrt^4)
        let t_surf_k = self.surface_temp + 273.15;
        let t_mrt_k = self.mean_radiant_temp + 273.15;
        let radiant_loss = self.emissivity
            * self.stefan_boltzmann
            * self.area
            * (t_surf_k.powi(4) - t_mrt_k.powi(4));

        // Update surface temp based on energy balance
        let surface_capacity = 5000.0; // J/K for typical radiant slab
        let dT_surf = (q_radiant - radiant_loss) * dt / surface_capacity;
        self.surface_temp = (self.surface_temp + dT_surf).clamp(t_zone, self.max_surface_temp);

        self.current_plr = plr;

        ZoneHeatInjection {
            q_air: q_convective,
            q_surface_radiant: q_radiant,
            q_latent: 0.0,
            electrical_power: 0.0,
            q_water_side: q_total, // Water-based system
            part_load_ratio: plr,
            mode: ZoneEquipmentMode::Heating,
        }
    }

    fn nominal_capacity(&self) -> f64 {
        self.capacity
    }

    fn equipment_type(&self) -> &'static str {
        match self.surface_type {
            RadiantSurfaceType::Floor => "RadiantFloor",
            RadiantSurfaceType::Ceiling => "RadiantCeiling",
        }
    }

    fn reset(&mut self) {
        self.current_plr = 0.0;
        self.surface_temp = 20.0;
    }
}

/// Packaged Terminal Air Conditioner (PTAC).
///
/// A PTAC is a single-package unit that provides cooling (and optionally heating)
/// to a single zone. It includes a cooling coil, supply fan, and exhaust fan.
///
/// # EnergyPlus Model
///
/// EnergyPlus `ZoneHVAC:PackagedTerminalAirConditioner` models this as:
/// - Rated sensible cooling capacity (W)
/// - Rated SHR (sensible heat ratio)
/// - Supply air flow rate (m³/s)
/// - Fan power (W)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackagedTerminalAC {
    /// Equipment identifier.
    pub id: String,
    /// Rated sensible cooling capacity (W).
    pub cooling_capacity: f64,
    /// Sensible Heat Ratio (SHR). Fraction of total capacity that is sensible.
    pub shr: f64,
    /// Supply air temperature setpoint for cooling (°C).
    pub cooling_supply_temp: f64,
    /// Supply air flow rate (m³/s).
    pub airflow_rate: f64,
    /// Fan power consumption (W).
    pub fan_power: f64,
    /// Fan efficiency (0.0 to 1.0).
    pub fan_efficiency: f64,
    /// Standard air density (kg/m³).
    pub air_density: f64,
    /// Specific heat of air (J/kg·K).
    pub air_cp: f64,
    /// Control deadband (K).
    pub deadband: f64,
    /// Current part-load ratio.
    pub current_plr: f64,
    /// Current operating mode.
    pub current_mode: ZoneEquipmentMode,
}

impl PackagedTerminalAC {
    /// Create a new PTAC.
    pub fn new(id: String, cooling_capacity: f64, airflow_rate: f64) -> Self {
        Self {
            id,
            cooling_capacity,
            shr: 0.75,                 // Typical PTAC SHR
            cooling_supply_temp: 13.0, // °C
            airflow_rate,
            fan_power: 100.0, // W
            fan_efficiency: 0.7,
            air_density: 1.2, // kg/m³
            air_cp: 1005.0,   // J/kg·K
            deadband: 0.5,
            current_plr: 0.0,
            current_mode: ZoneEquipmentMode::Off,
        }
    }
}

impl ZoneEquipment for PackagedTerminalAC {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, _dt: f64) -> ZoneHeatInjection {
        let t_zone = setpoints.zone_temp;
        let t_heating = setpoints.heating_setpoint;
        let t_cooling = setpoints.cooling_setpoint;

        let zone_humidity = setpoints.humidity_ratio.unwrap_or(0.010);
        let supply_humidity = setpoints.supply_humidity_ratio.unwrap_or(0.008);

        // Determine cooling requirement
        if t_zone <= t_cooling + self.deadband / 2.0 && t_zone >= t_heating - self.deadband / 2.0 {
            // In deadband - no cooling needed
            let fan_power = self.fan_power * 0.1; // Standby fan power

            return ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: fan_power,
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            };
        }

        if t_zone <= t_cooling + self.deadband / 2.0 {
            // Zone at or below cooling setpoint - no cooling
            return ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: self.fan_power * 0.1,
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Off,
            };
        }

        // Calculate cooling load
        let temp_deficit = t_zone - t_cooling;
        let plr = (temp_deficit / 10.0).clamp(0.2, 1.0); // Min 20% to avoid cycling

        let q_sensible = self.cooling_capacity * self.shr * plr;
        let _q_latent = self.cooling_capacity * (1.0 - self.shr) * plr;

        // Supply air heat removal from zone
        let supply_air_temp = self.cooling_supply_temp;
        let mass_flow = self.airflow_rate * self.air_density;
        let q_air_cooling = mass_flow * self.air_cp * (t_zone - supply_air_temp);

        // Adjust for latent (moisture removal)
        let latent_factor = if zone_humidity > supply_humidity {
            let h_fg = 2501000.0; // J/kg
            mass_flow * (zone_humidity - supply_humidity) * h_fg
        } else {
            0.0
        };

        self.current_plr = plr;
        self.current_mode = ZoneEquipmentMode::Cooling;

        ZoneHeatInjection {
            q_air: -(q_air_cooling - latent_factor), // Negative = heat removed from zone
            q_surface_radiant: 0.0,
            q_latent: latent_factor, // Positive = heat added to zone from condensation
            electrical_power: self.fan_power + q_sensible / 3.0, // EER-based
            q_water_side: 0.0,
            part_load_ratio: plr,
            mode: ZoneEquipmentMode::Cooling,
        }
    }

    fn nominal_capacity(&self) -> f64 {
        self.cooling_capacity
    }

    fn equipment_type(&self) -> &'static str {
        "PTAC"
    }

    fn reset(&mut self) {
        self.current_plr = 0.0;
        self.current_mode = ZoneEquipmentMode::Off;
    }
}

/// Packaged Terminal Heat Pump (PTHP).
///
/// A PTHP is a single-package unit that provides both cooling and heating
/// to a single zone. It includes a compressor, reversing valve, cooling coil,
/// heating coil (reverse cycle), supply fan, and exhaust fan.
///
/// # EnergyPlus Model
///
/// EnergyPlus `ZoneHVAC:PackagedTerminalHeatPump` models this as:
/// - Rated cooling capacity (W)
/// - Rated heating capacity (W)
/// - Rated COP (heating and cooling)
/// - Supply air flow rate (m³/s)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackagedTerminalHeatPump {
    /// Equipment identifier.
    pub id: String,
    /// Rated sensible cooling capacity (W).
    pub cooling_capacity: f64,
    /// Rated heating capacity (W).
    pub heating_capacity: f64,
    /// Cooling COP.
    pub cooling_cop: f64,
    /// Heating COP.
    pub heating_cop: f64,
    /// Sensible Heat Ratio for cooling.
    pub shr: f64,
    /// Supply air temperature for cooling (°C).
    pub cooling_supply_temp: f64,
    /// Supply air temperature for heating (°C).
    pub heating_supply_temp: f64,
    /// Supply air flow rate (m³/s).
    pub airflow_rate: f64,
    /// Fan power consumption (W).
    pub fan_power: f64,
    /// Standard air density (kg/m³).
    pub air_density: f64,
    /// Specific heat of air (J/kg·K).
    pub air_cp: f64,
    /// Control deadband (K).
    pub deadband: f64,
    /// Minimum outdoor temperature for heat pump operation (°C).
    pub min_heat_pump_temp: f64,
    /// Current part-load ratio.
    pub current_plr: f64,
    /// Current operating mode.
    pub current_mode: ZoneEquipmentMode,
}

impl PackagedTerminalHeatPump {
    /// Create a new PTHP.
    pub fn new(
        id: String,
        cooling_capacity: f64,
        heating_capacity: f64,
        airflow_rate: f64,
    ) -> Self {
        Self {
            id,
            cooling_capacity,
            heating_capacity,
            cooling_cop: 3.0,
            heating_cop: 3.0,
            shr: 0.75,
            cooling_supply_temp: 13.0,
            heating_supply_temp: 40.0,
            airflow_rate,
            fan_power: 100.0,
            air_density: 1.2,
            air_cp: 1005.0,
            deadband: 0.5,
            min_heat_pump_temp: -5.0, // Below this, use electric resistance
            current_plr: 0.0,
            current_mode: ZoneEquipmentMode::Off,
        }
    }
}

impl ZoneEquipment for PackagedTerminalHeatPump {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, _dt: f64) -> ZoneHeatInjection {
        let t_zone = setpoints.zone_temp;
        let t_heating = setpoints.heating_setpoint;
        let t_cooling = setpoints.cooling_setpoint;
        let t_outdoor = setpoints.outdoor_temp;

        let zone_humidity = setpoints.humidity_ratio.unwrap_or(0.010);
        let supply_humidity = setpoints.supply_humidity_ratio.unwrap_or(0.008);

        // Determine operating mode
        if t_zone < t_heating - self.deadband / 2.0 {
            // HEATING MODE
            let temp_deficit = t_heating - t_zone;

            // Check if heat pump can operate (temperature above minimum)
            let use_heat_pump = t_outdoor > self.min_heat_pump_temp;

            let plr = (temp_deficit / 10.0).clamp(0.2, 1.0);
            self.current_plr = plr;

            if use_heat_pump {
                // Heat pump heating
                let q_heating = self.heating_capacity * plr;
                let supply_air_temp = self.heating_supply_temp;

                // Air heat gain
                let mass_flow = self.airflow_rate * self.air_density;
                let q_air_heating = mass_flow * self.air_cp * (supply_air_temp - t_zone);

                let electrical_power = q_heating / self.heating_cop + self.fan_power;

                self.current_mode = ZoneEquipmentMode::Heating;

                ZoneHeatInjection {
                    q_air: q_air_heating,
                    q_surface_radiant: 0.0,
                    q_latent: 0.0,
                    electrical_power,
                    q_water_side: 0.0,
                    part_load_ratio: plr,
                    mode: ZoneEquipmentMode::Heating,
                }
            } else {
                // Electric resistance heating (frost protection mode)
                let q_resistance = self.heating_capacity * 0.5 * plr; // 50% of heat pump capacity
                let electrical_power = q_resistance + self.fan_power;

                self.current_mode = ZoneEquipmentMode::Heating;

                ZoneHeatInjection {
                    q_air: q_resistance,
                    q_surface_radiant: 0.0,
                    q_latent: 0.0,
                    electrical_power,
                    q_water_side: 0.0,
                    part_load_ratio: plr,
                    mode: ZoneEquipmentMode::Heating,
                }
            }
        } else if t_zone > t_cooling + self.deadband / 2.0 {
            // COOLING MODE
            let temp_deficit = t_zone - t_cooling;
            let plr = (temp_deficit / 10.0).clamp(0.2, 1.0);

            let q_cooling = self.cooling_capacity * plr;
            let supply_air_temp = self.cooling_supply_temp;

            // Air heat removal
            let mass_flow = self.airflow_rate * self.air_density;
            let q_air_cooling = mass_flow * self.air_cp * (t_zone - supply_air_temp);

            // Latent heat removal
            let latent_factor = if zone_humidity > supply_humidity {
                let h_fg = 2501000.0;
                mass_flow * (zone_humidity - supply_humidity) * h_fg
            } else {
                0.0
            };

            self.current_plr = plr;
            self.current_mode = ZoneEquipmentMode::Cooling;

            ZoneHeatInjection {
                q_air: -(q_air_cooling - latent_factor), // Negative = heat removed from zone
                q_surface_radiant: 0.0,
                q_latent: latent_factor, // Positive = heat added to zone from condensation
                electrical_power: q_cooling / self.cooling_cop + self.fan_power,
                q_water_side: 0.0,
                part_load_ratio: plr,
                mode: ZoneEquipmentMode::Cooling,
            }
        } else {
            // DEADBAND - no heating or cooling
            ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: self.fan_power * 0.1, // Standby
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            }
        }
    }

    fn nominal_capacity(&self) -> f64 {
        self.cooling_capacity.max(self.heating_capacity)
    }

    fn equipment_type(&self) -> &'static str {
        "PTHP"
    }

    fn reset(&mut self) {
        self.current_plr = 0.0;
        self.current_mode = ZoneEquipmentMode::Off;
    }
}

/// Four-pipe fan coil unit.
///
/// A fan coil unit uses hot/chilled water from a central plant to provide
/// zone-level heating and cooling. It has separate coils for heating and cooling
/// (four pipes: 2 supply, 2 return).
///
/// # EnergyPlus Model
///
/// EnergyPlus `ZoneHVAC:FourPipeFanCoil` models this as:
/// - Rated heating capacity (W)
/// - Rated cooling capacity (W)
/// - Supply air flow rate (m³/s)
/// - Outdoor air flow rate (m³/s) for ventilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourPipeFanCoil {
    /// Equipment identifier.
    pub id: String,
    /// Rated heating capacity (W).
    pub heating_capacity: f64,
    /// Rated cooling capacity (W).
    pub cooling_capacity: f64,
    /// Supply air flow rate (m³/s).
    pub airflow_rate: f64,
    /// Fan power consumption (W).
    pub fan_power: f64,
    /// Water mass flow rate (kg/s).
    pub water_flow_rate: f64,
    /// Standard air density (kg/m³).
    pub air_density: f64,
    /// Specific heat of air (J/kg·K).
    pub air_cp: f64,
    /// Specific heat of water (J/kg·K).
    pub water_cp: f64,
    /// Control deadband (K).
    pub deadband: f64,
    /// Current part-load ratio.
    pub current_plr: f64,
    /// Current operating mode.
    pub current_mode: ZoneEquipmentMode,
}

impl FourPipeFanCoil {
    /// Create a new four-pipe fan coil unit.
    pub fn new(
        id: String,
        heating_capacity: f64,
        cooling_capacity: f64,
        airflow_rate: f64,
    ) -> Self {
        Self {
            id,
            heating_capacity,
            cooling_capacity,
            airflow_rate,
            fan_power: 50.0,
            water_flow_rate: 0.1,
            air_density: 1.2,
            air_cp: 1005.0,
            water_cp: 4186.0,
            deadband: 0.5,
            current_plr: 0.0,
            current_mode: ZoneEquipmentMode::Off,
        }
    }
}

impl ZoneEquipment for FourPipeFanCoil {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, _dt: f64) -> ZoneHeatInjection {
        let t_zone = setpoints.zone_temp;
        let t_heating = setpoints.heating_setpoint;
        let t_cooling = setpoints.cooling_setpoint;

        let supply_temp = setpoints.supply_water_temp.unwrap_or(45.0);
        let return_temp = setpoints.return_water_temp.unwrap_or(40.0);

        // Determine operating mode
        if t_zone < t_heating - self.deadband / 2.0 {
            // HEATING MODE
            let temp_deficit = t_heating - t_zone;
            let plr = (temp_deficit / 10.0).clamp(0.2, 1.0);

            // Water-side heat transfer
            let q_water = self.water_flow_rate * self.water_cp * (supply_temp - return_temp);
            let q_heating = q_water.min(self.heating_capacity * plr);

            // Air-side heating
            let mass_flow = self.airflow_rate * self.air_density;
            let supply_air_temp = return_temp; // Approximate
            let q_air_heating = mass_flow * self.air_cp * (supply_air_temp - t_zone).max(0.0);

            self.current_plr = plr;
            self.current_mode = ZoneEquipmentMode::Heating;

            ZoneHeatInjection {
                q_air: q_air_heating,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: self.fan_power * plr,
                q_water_side: q_heating,
                part_load_ratio: plr,
                mode: ZoneEquipmentMode::Heating,
            }
        } else if t_zone > t_cooling + self.deadband / 2.0 {
            // COOLING MODE
            let temp_deficit = t_zone - t_cooling;
            let plr = (temp_deficit / 10.0).clamp(0.2, 1.0);

            // Water-side heat transfer
            let q_water = self.water_flow_rate * self.water_cp * (return_temp - supply_temp);
            let q_cooling = q_water.min(self.cooling_capacity * plr);

            // Air-side cooling
            let mass_flow = self.airflow_rate * self.air_density;
            let supply_air_temp = return_temp; // Approximate
            let q_air_cooling = mass_flow * self.air_cp * (supply_air_temp - t_zone);

            self.current_plr = plr;
            self.current_mode = ZoneEquipmentMode::Cooling;

            ZoneHeatInjection {
                q_air: q_air_cooling,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: self.fan_power * plr,
                q_water_side: q_cooling,
                part_load_ratio: plr,
                mode: ZoneEquipmentMode::Cooling,
            }
        } else {
            // DEADBAND
            ZoneHeatInjection {
                q_air: 0.0,
                q_surface_radiant: 0.0,
                q_latent: 0.0,
                electrical_power: self.fan_power * 0.05, // Standby
                q_water_side: 0.0,
                part_load_ratio: 0.0,
                mode: ZoneEquipmentMode::Deadband,
            }
        }
    }

    fn nominal_capacity(&self) -> f64 {
        self.heating_capacity.max(self.cooling_capacity)
    }

    fn equipment_type(&self) -> &'static str {
        "FourPipeFanCoil"
    }

    fn reset(&mut self) {
        self.current_plr = 0.0;
        self.current_mode = ZoneEquipmentMode::Off;
    }
}

/// Zone equipment enum for dynamic dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyZoneEquipment {
    Baseboard(BaseboardHeater),
    HotWaterBaseboard(HotWaterBaseboard),
    RadiantSurface(LowTemperatureRadiantSurface),
    PTAC(PackagedTerminalAC),
    PTHP(PackagedTerminalHeatPump),
    FanCoil(FourPipeFanCoil),
}

impl ZoneEquipment for AnyZoneEquipment {
    fn step(&mut self, setpoints: &ZoneEquipmentSetpoints, dt: f64) -> ZoneHeatInjection {
        match self {
            AnyZoneEquipment::Baseboard(e) => e.step(setpoints, dt),
            AnyZoneEquipment::HotWaterBaseboard(e) => e.step(setpoints, dt),
            AnyZoneEquipment::RadiantSurface(e) => e.step(setpoints, dt),
            AnyZoneEquipment::PTAC(e) => e.step(setpoints, dt),
            AnyZoneEquipment::PTHP(e) => e.step(setpoints, dt),
            AnyZoneEquipment::FanCoil(e) => e.step(setpoints, dt),
        }
    }

    fn nominal_capacity(&self) -> f64 {
        match self {
            AnyZoneEquipment::Baseboard(e) => e.nominal_capacity(),
            AnyZoneEquipment::HotWaterBaseboard(e) => e.nominal_capacity(),
            AnyZoneEquipment::RadiantSurface(e) => e.nominal_capacity(),
            AnyZoneEquipment::PTAC(e) => e.nominal_capacity(),
            AnyZoneEquipment::PTHP(e) => e.nominal_capacity(),
            AnyZoneEquipment::FanCoil(e) => e.nominal_capacity(),
        }
    }

    fn equipment_type(&self) -> &'static str {
        match self {
            AnyZoneEquipment::Baseboard(e) => e.equipment_type(),
            AnyZoneEquipment::HotWaterBaseboard(e) => e.equipment_type(),
            AnyZoneEquipment::RadiantSurface(e) => e.equipment_type(),
            AnyZoneEquipment::PTAC(e) => e.equipment_type(),
            AnyZoneEquipment::PTHP(e) => e.equipment_type(),
            AnyZoneEquipment::FanCoil(e) => e.equipment_type(),
        }
    }

    fn reset(&mut self) {
        match self {
            AnyZoneEquipment::Baseboard(e) => e.reset(),
            AnyZoneEquipment::HotWaterBaseboard(e) => e.reset(),
            AnyZoneEquipment::RadiantSurface(e) => e.reset(),
            AnyZoneEquipment::PTAC(e) => e.reset(),
            AnyZoneEquipment::PTHP(e) => e.reset(),
            AnyZoneEquipment::FanCoil(e) => e.reset(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseboard_heating() {
        let mut bb = BaseboardHeater::new("BB-1".to_string(), 5000.0);

        let setpoints = ZoneEquipmentSetpoints::new(20.0, 27.0, 18.0, 10.0);
        let result = bb.step(&setpoints, 3600.0);

        assert!(
            result.q_air > 0.0,
            "Baseboard should heat when zone is cold"
        );
        assert_eq!(result.mode, ZoneEquipmentMode::Heating);
        assert!(result.part_load_ratio > 0.0);
    }

    #[test]
    fn test_baseboard_deadband() {
        let mut bb = BaseboardHeater::new("BB-1".to_string(), 5000.0);

        // Zone at setpoint - should be in deadband
        let setpoints = ZoneEquipmentSetpoints::new(20.0, 27.0, 20.0, 10.0);
        let result = bb.step(&setpoints, 3600.0);

        assert_eq!(result.q_air, 0.0);
        assert_eq!(result.mode, ZoneEquipmentMode::Deadband);
    }

    #[test]
    fn test_ptac_cooling() {
        let mut ptac = PackagedTerminalAC::new("PTAC-1".to_string(), 5000.0, 0.3);

        let setpoints = ZoneEquipmentSetpoints::with_humidity(20.0, 27.0, 30.0, 35.0, 0.012, 0.008);
        let result = ptac.step(&setpoints, 3600.0);

        assert!(result.q_air < 0.0, "PTAC should cool when zone is hot");
        assert!(result.electrical_power > 0.0);
        assert_eq!(result.mode, ZoneEquipmentMode::Cooling);
    }

    #[test]
    fn test_pthp_heating_and_cooling() {
        let mut pthp = PackagedTerminalHeatPump::new("PTHP-1".to_string(), 5000.0, 4500.0, 0.3);

        // Test heating
        let heating_setpoints = ZoneEquipmentSetpoints::new(20.0, 27.0, 18.0, 10.0);
        let heating_result = pthp.step(&heating_setpoints, 3600.0);
        assert!(heating_result.q_air > 0.0);
        assert_eq!(heating_result.mode, ZoneEquipmentMode::Heating);

        // Test cooling
        let cooling_setpoints =
            ZoneEquipmentSetpoints::with_humidity(20.0, 27.0, 30.0, 35.0, 0.012, 0.008);
        let cooling_result = pthp.step(&cooling_setpoints, 3600.0);
        assert!(cooling_result.q_air < 0.0);
        assert_eq!(cooling_result.mode, ZoneEquipmentMode::Cooling);
    }

    #[test]
    fn test_pthp_low_temp_resistance() {
        let mut pthp = PackagedTerminalHeatPump::new("PTHP-1".to_string(), 5000.0, 4500.0, 0.3);

        // Very cold outdoor - should switch to resistance heating
        let setpoints = ZoneEquipmentSetpoints::new(20.0, 27.0, 18.0, -10.0);
        let result = pthp.step(&setpoints, 3600.0);

        assert!(result.q_air > 0.0);
        assert_eq!(result.mode, ZoneEquipmentMode::Heating);
    }

    #[test]
    fn test_radiant_floor() {
        let mut radiant = LowTemperatureRadiantSurface::new_floor("RF-1".to_string(), 2000.0, 20.0);

        let setpoints = ZoneEquipmentSetpoints::new(20.0, 27.0, 18.0, 10.0);
        let result = radiant.step(&setpoints, 3600.0);

        assert!(result.q_air > 0.0 || result.q_surface_radiant > 0.0);
        assert_eq!(result.mode, ZoneEquipmentMode::Heating);
    }

    #[test]
    fn test_fan_coil_heating_cooling() {
        let mut fcu = FourPipeFanCoil::new("FCU-1".to_string(), 4000.0, 3500.0, 0.2);

        // Test heating
        let heating_setpoints =
            ZoneEquipmentSetpoints::with_water(20.0, 27.0, 18.0, 10.0, 50.0, 45.0);
        let heating_result = fcu.step(&heating_setpoints, 3600.0);
        assert!(heating_result.q_air > 0.0);
        assert_eq!(heating_result.mode, ZoneEquipmentMode::Heating);

        // Reset and test cooling
        fcu.reset();
        let cooling_setpoints =
            ZoneEquipmentSetpoints::with_water(20.0, 27.0, 30.0, 35.0, 7.0, 12.0);
        let cooling_result = fcu.step(&cooling_setpoints, 3600.0);
        assert!(cooling_result.q_air < 0.0);
        assert_eq!(cooling_result.mode, ZoneEquipmentMode::Cooling);
    }

    #[test]
    fn test_zone_heat_injection_total() {
        let injection = ZoneHeatInjection::new(
            1000.0,
            500.0,
            0.0,
            100.0,
            0.0,
            0.8,
            ZoneEquipmentMode::Heating,
        );
        assert_eq!(injection.total(), 1500.0);
        assert_eq!(injection.total_convective(), 1000.0);
    }

    #[test]
    fn test_any_zone_equipment_dispatch() {
        let mut equipment: AnyZoneEquipment =
            AnyZoneEquipment::Baseboard(BaseboardHeater::new("BB-1".to_string(), 5000.0));

        let setpoints = ZoneEquipmentSetpoints::new(20.0, 27.0, 18.0, 10.0);
        let result = equipment.step(&setpoints, 3600.0);

        assert_eq!(equipment.equipment_type(), "ElectricBaseboard");
        assert!(result.q_air > 0.0);
    }
}
