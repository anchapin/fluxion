// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for deep HVAC configuration (Issue #1798, Plan T9.4).
//!
//! Exposes equipment types (VAV, CAV, heat pump, chiller, boiler), zone
//! setpoints, daily/HVAC schedules, and zone-level control strategies to
//! JavaScript/TypeScript consumers, providing Node.js parity with the PyO3
//! exposure in T9.3 (`src/python/hvac_bindings.rs`).
//!
//! Enums are exchanged as strings (e.g. `"heating"`, `"ideal_loads"`) to match
//! the established NAPI convention used by `NineR4CConfig::coupling_mode` and
//! the Python `HVACStatus` mapping.

use std::sync::{Arc, Mutex};

use crate::physics::cta::VectorField;
use crate::sim::hvac::equipment::{Boiler, Chiller, HVACMode, VariableCapacityEquipment};
use crate::sim::hvac::zones::zone_control::{ControlStrategy, HVACStatus, ZoneControl};
use crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints as CoreZoneSetpoints;
use crate::sim::hvac::{CAVSystem, HeatPump, HeatPumpMode, VAVTerminal};
use crate::sim::schedule::{DailySchedule, HVACSchedule, ScheduleType};
use crate::thermal::thermal_model::ThermalModel;

// ────────────────────────────────────────────────────────────────────────────
// Equipment: VAV terminal
// ────────────────────────────────────────────────────────────────────────────

/// Variable Air Volume (VAV) terminal unit configuration.
///
/// Composes a modulating damper with optional reheat. Exposes the design
/// airflow limits and reheat coil capacity to Node.js so a full VAV system
/// can be assembled and round-tripped from JavaScript.
#[napi_derive::napi]
pub struct HvacVavTerminal {
    inner: VAVTerminal,
}

#[napi_derive::napi]
impl HvacVavTerminal {
    /// Create a new VAV terminal unit.
    ///
    /// The minimum airflow defaults to 30% of `maxAirflow` and the reheat
    /// coil defaults to 5 kW, matching the Rust `VAVTerminal::new` defaults.
    ///
    /// # Arguments
    /// * `id` - Terminal unit identifier
    /// * `zoneId` - Index of the zone served by this terminal
    /// * `maxAirflow` - Maximum (design cooling) airflow [m³/s]
    #[napi(constructor)]
    pub fn new(id: String, zone_id: u32, max_airflow: f64) -> Self {
        Self {
            inner: VAVTerminal::new(id, zone_id as usize, max_airflow),
        }
    }

    /// Terminal unit identifier.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Index of the zone served by this terminal.
    #[napi(getter)]
    pub fn zone_id(&self) -> u32 {
        self.inner.zone_id as u32
    }

    /// Maximum (design cooling) airflow [m³/s].
    #[napi(getter)]
    pub fn max_airflow(&self) -> f64 {
        self.inner.max_airflow
    }

    /// Minimum (ventilation / turndown) airflow [m³/s].
    #[napi(getter)]
    pub fn min_airflow(&self) -> f64 {
        self.inner.min_airflow
    }

    /// Set the minimum airflow [m³/s].
    #[napi(setter)]
    pub fn set_min_airflow(&mut self, value: f64) {
        self.inner.min_airflow = value;
    }

    /// Reheat coil rated capacity [W].
    #[napi(getter)]
    pub fn reheat_capacity(&self) -> f64 {
        self.inner.reheat_capacity
    }

    /// Set the reheat coil rated capacity [W].
    #[napi(setter)]
    pub fn set_reheat_capacity(&mut self, value: f64) {
        self.inner.reheat_capacity = value;
    }

    /// Current airflow setpoint [m³/s].
    #[napi(getter)]
    pub fn airflow_setpoint(&self) -> f64 {
        self.inner.airflow_setpoint
    }

    /// Set the current airflow setpoint [m³/s].
    #[napi(setter)]
    pub fn set_airflow_setpoint(&mut self, value: f64) {
        self.inner.airflow_setpoint = value;
    }

    /// Reheat coil sensible demand [W] for the given supply and zone
    /// temperatures.
    ///
    /// Uses the psychrometric relation Q = ρ·cp·V̇·ΔT with ρ = 1.2 kg/m³ and
    /// cp = 1005 J/kg·K. Returns 0 when the zone is already at or above the
    /// comfort threshold.
    ///
    /// # Arguments
    /// * `supplyTemp` - Supply air dry-bulb temperature leaving the terminal [°C]
    /// * `zoneTemp` - Current zone air temperature [°C]
    #[napi]
    pub fn reheat_demand(&self, supply_temp: f64, zone_temp: f64) -> f64 {
        self.inner.reheat_demand(supply_temp, zone_temp)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Equipment: CAV system
// ────────────────────────────────────────────────────────────────────────────

/// Constant Air Volume (CAV) system configuration.
#[napi_derive::napi]
pub struct HvacCavSystem {
    inner: CAVSystem,
}

#[napi_derive::napi]
impl HvacCavSystem {
    /// Create a new CAV system.
    ///
    /// Fan power defaults to `500 W per (m³/s)` of design airflow and fan
    /// efficiency defaults to 0.7, matching the Rust `CAVSystem::new`
    /// defaults.
    ///
    /// # Arguments
    /// * `id` - System identifier
    /// * `designAirflow` - Design (constant) air flow rate [m³/s]
    #[napi(constructor)]
    pub fn new(id: String, design_airflow: f64) -> Self {
        Self {
            inner: CAVSystem::new(id, design_airflow),
        }
    }

    /// System identifier.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Design air flow rate [m³/s].
    #[napi(getter)]
    pub fn design_airflow(&self) -> f64 {
        self.inner.design_airflow
    }

    /// Fan rated power consumption [W].
    #[napi(getter)]
    pub fn fan_power(&self) -> f64 {
        self.inner.fan_power
    }

    /// Set the fan rated power consumption [W].
    #[napi(setter)]
    pub fn set_fan_power(&mut self, value: f64) {
        self.inner.fan_power = value;
    }

    /// Fan efficiency (0–1).
    #[napi(getter)]
    pub fn fan_efficiency(&self) -> f64 {
        self.inner.fan_efficiency
    }

    /// Set the fan efficiency (0–1).
    #[napi(setter)]
    pub fn set_fan_efficiency(&mut self, value: f64) {
        self.inner.fan_efficiency = value;
    }

    /// Heating coil capacity [W].
    #[napi(getter)]
    pub fn heating_capacity(&self) -> f64 {
        self.inner.heating_capacity
    }

    /// Set the heating coil capacity [W].
    #[napi(setter)]
    pub fn set_heating_capacity(&mut self, value: f64) {
        self.inner.heating_capacity = value;
    }

    /// Cooling coil capacity [W].
    #[napi(getter)]
    pub fn cooling_capacity(&self) -> f64 {
        self.inner.cooling_capacity
    }

    /// Set the cooling coil capacity [W].
    #[napi(setter)]
    pub fn set_cooling_capacity(&mut self, value: f64) {
        self.inner.cooling_capacity = value;
    }

    /// Actual electrical fan power consumption [W] = fanPower / fanEfficiency.
    #[napi]
    pub fn fan_power_consumption(&self) -> f64 {
        self.inner.fan_power_consumption()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Equipment: Heat pump
// ────────────────────────────────────────────────────────────────────────────

/// Heat pump system with temperature-dependent COP curves.
#[napi_derive::napi]
pub struct HvacHeatPump {
    inner: HeatPump,
}

#[napi_derive::napi]
impl HvacHeatPump {
    /// Create a new heat pump.
    ///
    /// # Arguments
    /// * `id` - System identifier
    /// * `heatingCapacity` - Rated heating capacity at design conditions [W]
    /// * `coolingCapacity` - Rated cooling capacity at design conditions [W]
    /// * `heatingCop` - Rated heating COP at design conditions
    /// * `coolingCop` - Rated cooling COP (EER) at design conditions
    #[napi(constructor)]
    pub fn new(
        id: String,
        heating_capacity: f64,
        cooling_capacity: f64,
        heating_cop: f64,
        cooling_cop: f64,
    ) -> Self {
        Self {
            inner: HeatPump::new(
                id,
                heating_capacity,
                cooling_capacity,
                heating_cop,
                cooling_cop,
            ),
        }
    }

    /// System identifier.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Rated heating capacity [W].
    #[napi(getter)]
    pub fn heating_capacity(&self) -> f64 {
        self.inner.heating_capacity
    }

    /// Rated cooling capacity [W].
    #[napi(getter)]
    pub fn cooling_capacity(&self) -> f64 {
        self.inner.cooling_capacity
    }

    /// Rated heating COP.
    #[napi(getter)]
    pub fn heating_cop(&self) -> f64 {
        self.inner.heating_cop
    }

    /// Rated cooling COP (EER).
    #[napi(getter)]
    pub fn cooling_cop(&self) -> f64 {
        self.inner.cooling_cop
    }

    /// Current operating mode: `"heating"`, `"cooling"`, or `"off"`.
    #[napi(getter)]
    pub fn mode(&self) -> String {
        heat_pump_mode_str(&self.inner.mode)
    }

    /// Heating COP degraded for the given outdoor temperature.
    ///
    /// COP degrades ~2% per °C away from the design heating temperature,
    /// floored at 50% of rated COP.
    #[napi]
    pub fn heating_cop_at_temperature(&self, outdoor_temp: f64) -> f64 {
        self.inner.heating_cop_at_temperature(outdoor_temp)
    }

    /// Cooling COP (EER) degraded for the given outdoor temperature.
    ///
    /// EER degrades ~3% per °C away from the design cooling temperature,
    /// floored at 50% of rated COP.
    #[napi]
    pub fn cooling_cop_at_temperature(&self, outdoor_temp: f64) -> f64 {
        self.inner.cooling_cop_at_temperature(outdoor_temp)
    }

    /// Electrical heating power consumption [W] at the given outdoor
    /// temperature. Returns 0 unless the unit is in heating mode.
    #[napi]
    pub fn heating_power(&self, outdoor_temp: f64) -> f64 {
        self.inner.heating_power(outdoor_temp)
    }

    /// Electrical cooling power consumption [W] at the given outdoor
    /// temperature. Returns 0 unless the unit is in cooling mode.
    #[napi]
    pub fn cooling_power(&self, outdoor_temp: f64) -> f64 {
        self.inner.cooling_power(outdoor_temp)
    }

    /// Select the operating mode from the zone temperature and deadband
    /// setpoints.
    ///
    /// # Arguments
    /// * `zoneTemp` - Current zone air temperature [°C]
    /// * `heatingSetpoint` - Heating deadband lower bound [°C]
    /// * `coolingSetpoint` - Cooling deadband upper bound [°C]
    #[napi]
    pub fn set_mode(&mut self, zone_temp: f64, heating_sp: f64, cooling_sp: f64) {
        self.inner.set_mode(zone_temp, heating_sp, cooling_sp);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Equipment: Chiller
// ────────────────────────────────────────────────────────────────────────────

/// Variable-capacity chiller with polynomial part-load efficiency curves.
#[napi_derive::napi]
pub struct HvacChiller {
    inner: Chiller,
}

#[napi_derive::napi]
impl HvacChiller {
    /// Create a new chiller with default AHRI efficiency curves.
    ///
    /// # Arguments
    /// * `id` - Equipment identifier
    /// * `coolingCapacity` - Rated cooling capacity at design conditions [W]
    /// * `coolingCop` - Rated cooling COP at design conditions
    /// * `designTemp` - Design outdoor temperature for cooling [°C]
    #[napi(constructor)]
    pub fn new(id: String, cooling_capacity: f64, cooling_cop: f64, design_temp: f64) -> Self {
        Self {
            inner: Chiller::new(id, cooling_capacity, cooling_cop, design_temp),
        }
    }

    /// Equipment identifier.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Rated cooling capacity [W].
    #[napi(getter)]
    pub fn cooling_capacity(&self) -> f64 {
        self.inner.cooling_capacity
    }

    /// Rated cooling COP.
    #[napi(getter)]
    pub fn cooling_cop(&self) -> f64 {
        self.inner.cooling_cop
    }

    /// Design outdoor temperature for cooling [°C].
    #[napi(getter)]
    pub fn design_temp(&self) -> f64 {
        self.inner.design_temp
    }

    /// Rated capacity [W] (alias for the rated cooling capacity).
    #[napi]
    pub fn rated_capacity(&self) -> f64 {
        self.inner.rated_capacity()
    }

    /// Actual available capacity [W] at the given part-load ratio and outdoor
    /// temperature, after temperature-based degradation.
    ///
    /// # Arguments
    /// * `plr` - Part-load ratio (0.0–1.0)
    /// * `outdoorTemp` - Outdoor air temperature [°C]
    #[napi]
    pub fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.inner.calculate_capacity(plr, outdoor_temp)
    }

    /// Electrical power consumption [W] for the given load and conditions.
    ///
    /// # Arguments
    /// * `load` - Thermal load [W]
    /// * `outdoorTemp` - Outdoor air temperature [°C]
    /// * `mode` - Operating mode: `"cooling"` (chillers only cool)
    #[napi]
    pub fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: String) -> f64 {
        self.inner
            .calculate_power(load, outdoor_temp, parse_hvac_mode(&mode))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Equipment: Boiler
// ────────────────────────────────────────────────────────────────────────────

/// Variable-capacity boiler with polynomial part-load efficiency curves.
#[napi_derive::napi]
pub struct HvacBoiler {
    inner: Boiler,
}

#[napi_derive::napi]
impl HvacBoiler {
    /// Create a new boiler with default AHRI efficiency curves.
    ///
    /// # Arguments
    /// * `id` - Equipment identifier
    /// * `heatingCapacity` - Rated heating capacity at design conditions [W]
    /// * `efficiency` - Rated thermal efficiency (AFUE, 0–1)
    /// * `designTemp` - Design outdoor temperature for heating [°C]
    #[napi(constructor)]
    pub fn new(id: String, heating_capacity: f64, efficiency: f64, design_temp: f64) -> Self {
        Self {
            inner: Boiler::new(id, heating_capacity, efficiency, design_temp),
        }
    }

    /// Equipment identifier.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Rated heating capacity [W].
    #[napi(getter)]
    pub fn heating_capacity(&self) -> f64 {
        self.inner.heating_capacity
    }

    /// Rated thermal efficiency (AFUE, 0–1).
    #[napi(getter)]
    pub fn efficiency(&self) -> f64 {
        self.inner.efficiency
    }

    /// Design outdoor temperature for heating [°C].
    #[napi(getter)]
    pub fn design_temp(&self) -> f64 {
        self.inner.design_temp
    }

    /// Rated capacity [W] (alias for the rated heating capacity).
    #[napi]
    pub fn rated_capacity(&self) -> f64 {
        self.inner.rated_capacity()
    }

    /// Actual available capacity [W] at the given part-load ratio and outdoor
    /// temperature.
    ///
    /// # Arguments
    /// * `plr` - Part-load ratio (0.0–1.0)
    /// * `outdoorTemp` - Outdoor air temperature [°C]
    #[napi]
    pub fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.inner.calculate_capacity(plr, outdoor_temp)
    }

    /// Electrical power consumption [W] for the given load and conditions.
    ///
    /// # Arguments
    /// * `load` - Thermal load [W]
    /// * `outdoorTemp` - Outdoor air temperature [°C]
    /// * `mode` - Operating mode: `"heating"` (boilers only heat)
    #[napi]
    pub fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: String) -> f64 {
        self.inner
            .calculate_power(load, outdoor_temp, parse_hvac_mode(&mode))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Zone setpoints
// ────────────────────────────────────────────────────────────────────────────

/// Per-zone heating/cooling setpoints and deadband configuration.
///
/// Mirrors the Python `ZoneSetpoints` class (T9.3). Defaults: 20 °C heating,
/// 24 °C cooling, 2 °C deadband. Temperatures must be in [10, 40] °C and
/// deadband in (0, 5] °C; heating must remain below cooling.
#[napi_derive::napi]
pub struct ZoneSetpoints {
    inner: CoreZoneSetpoints,
}

#[napi_derive::napi]
impl ZoneSetpoints {
    /// Create a new setpoints container with default values for `numZones`
    /// zones.
    #[napi(constructor)]
    pub fn new(num_zones: u32) -> napi::bindgen_prelude::Result<Self> {
        if num_zones < 1 {
            return Err(napi::bindgen_prelude::Error::from_reason(
                "Number of zones must be at least 1",
            ));
        }
        Ok(Self {
            inner: CoreZoneSetpoints::new(num_zones as usize),
        })
    }

    /// Set the heating setpoint [°C] for a zone.
    #[napi]
    pub fn set_heating_setpoint(
        &mut self,
        zone_id: u32,
        temp: f64,
    ) -> napi::bindgen_prelude::Result<()> {
        self.inner
            .set_heating_setpoint(zone_id as usize, temp)
            .map_err(napi_err)
    }

    /// Set the cooling setpoint [°C] for a zone.
    #[napi]
    pub fn set_cooling_setpoint(
        &mut self,
        zone_id: u32,
        temp: f64,
    ) -> napi::bindgen_prelude::Result<()> {
        self.inner
            .set_cooling_setpoint(zone_id as usize, temp)
            .map_err(napi_err)
    }

    /// Set the deadband [°C] for a zone.
    #[napi]
    pub fn set_deadband(
        &mut self,
        zone_id: u32,
        deadband: f64,
    ) -> napi::bindgen_prelude::Result<()> {
        self.inner
            .set_deadband(zone_id as usize, deadband)
            .map_err(napi_err)
    }

    /// Heating setpoint [°C] for a zone.
    #[napi]
    pub fn get_heating_setpoint(&self, zone_id: u32) -> napi::bindgen_prelude::Result<f64> {
        check_zone(zone_id as usize, self.inner.num_zones())?;
        Ok(self.inner.get_heating_setpoint(zone_id as usize))
    }

    /// Cooling setpoint [°C] for a zone.
    #[napi]
    pub fn get_cooling_setpoint(&self, zone_id: u32) -> napi::bindgen_prelude::Result<f64> {
        check_zone(zone_id as usize, self.inner.num_zones())?;
        Ok(self.inner.get_cooling_setpoint(zone_id as usize))
    }

    /// Deadband [°C] for a zone.
    #[napi]
    pub fn get_deadband(&self, zone_id: u32) -> napi::bindgen_prelude::Result<f64> {
        check_zone(zone_id as usize, self.inner.num_zones())?;
        Ok(self.inner.get_deadband(zone_id as usize))
    }

    /// Validate every zone's setpoints and deadbands.
    #[napi]
    pub fn validate(&self) -> napi::bindgen_prelude::Result<()> {
        self.inner.validate_setpoints().map_err(napi_err)
    }

    /// Number of zones.
    #[napi(getter)]
    pub fn num_zones(&self) -> u32 {
        self.inner.num_zones() as u32
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Schedules
// ────────────────────────────────────────────────────────────────────────────

/// Hourly daily schedule (24 values) used to drive setpoint profiles.
///
/// Mirrors the Python `DailySchedule` class (T9.3). Schedule type is one of
/// `"Constant"`, `"DailyCycle"`, `"Weekly"`, or `"Custom"`.
#[napi_derive::napi]
pub struct HvacDailySchedule {
    inner: DailySchedule,
}

#[napi_derive::napi]
impl HvacDailySchedule {
    /// Create a new schedule with all 24 hours initialized to zero.
    ///
    /// # Arguments
    /// * `name` - Schedule name / identifier
    /// * `scheduleType` - One of `"Constant"`, `"DailyCycle"`, `"Weekly"`, `"Custom"`
    #[napi(constructor)]
    pub fn new(name: String, schedule_type: String) -> napi::bindgen_prelude::Result<Self> {
        let st = parse_schedule_type(&schedule_type)?;
        let mut inner = if matches!(st, ScheduleType::Weekly) {
            DailySchedule::weekly(name.clone())
        } else {
            DailySchedule::new()
        };
        inner.name = name;
        inner.schedule_type = st;
        Ok(Self { inner })
    }

    /// Set the value for a single hour (0–23).
    #[napi]
    pub fn set_hour(&mut self, hour: u32, value: f64) {
        self.inner.set_hour(hour as usize, value);
    }

    /// Fill every hour in `[startHour, endHour)` with `value`.
    #[napi]
    pub fn fill_range(&mut self, start_hour: u32, end_hour: u32, value: f64) {
        self.inner
            .fill_range(start_hour as usize, end_hour as usize, value);
    }

    /// Value for the given hour (0–23).
    #[napi]
    pub fn value(&self, hour: u32) -> f64 {
        self.inner.value(hour as usize)
    }

    /// Schedule name / identifier.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Schedule type as a string (`"Constant"`, `"DailyCycle"`, `"Weekly"`, `"Custom"`).
    #[napi(getter)]
    pub fn schedule_type(&self) -> String {
        schedule_type_str(&self.inner.schedule_type)
    }

    /// Build a constant schedule where every hour equals `value`.
    #[napi(factory)]
    pub fn constant(value: f64) -> Self {
        Self {
            inner: DailySchedule::constant(value),
        }
    }
}

/// Composite heating + cooling schedule pair driving time-varying setpoints.
///
/// Mirrors the Python `HVACSchedule` class (T9.3). Supports constant,
/// night-setback, operating-hours, and free-floating profiles.
#[napi_derive::napi]
pub struct HvacSchedule {
    inner: HVACSchedule,
}

#[napi_derive::napi]
impl HvacSchedule {
    /// Create a new schedule with all setpoints initialized to zero.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: HVACSchedule::new(),
        }
    }

    /// Constant heating/cooling setpoints for all 24 hours.
    ///
    /// # Arguments
    /// * `heatingSp` - Heating setpoint [°C]
    /// * `coolingSp` - Cooling setpoint [°C]
    #[napi(factory)]
    pub fn constant_schedule(heating_sp: f64, cooling_sp: f64) -> Self {
        Self {
            inner: HVACSchedule::constant_schedule(heating_sp, cooling_sp),
        }
    }

    /// Night-setback heating profile with constant cooling setpoint.
    ///
    /// # Arguments
    /// * `dayHeat` - Daytime heating setpoint [°C]
    /// * `nightHeat` - Nighttime (setback) heating setpoint [°C]
    /// * `coolSp` - Cooling setpoint [°C]
    /// * `nightStart` - Start hour of the setback window (0–23)
    /// * `nightEnd` - End hour of the setback window (0–23)
    #[napi(factory)]
    pub fn setback_schedule(
        day_heat: f64,
        night_heat: f64,
        cool_sp: f64,
        night_start: u32,
        night_end: u32,
    ) -> Self {
        Self {
            inner: HVACSchedule::setback_schedule(
                day_heat,
                night_heat,
                cool_sp,
                night_start as usize,
                night_end as usize,
            ),
        }
    }

    /// Operating-hours profile: HVAC active only between `startHour` and
    /// `endHour`; disabled (heating = −100 °C, cooling = +100 °C) outside.
    ///
    /// # Arguments
    /// * `heatingSp` - In-hours heating setpoint [°C]
    /// * `coolingSp` - In-hours cooling setpoint [°C]
    /// * `startHour` - Operating window start hour (0–23)
    /// * `endHour` - Operating window end hour (0–23)
    #[napi(factory)]
    pub fn with_operating_hours(
        heating_sp: f64,
        cooling_sp: f64,
        start_hour: u32,
        end_hour: u32,
    ) -> Self {
        Self {
            inner: HVACSchedule::with_operating_hours(
                heating_sp,
                cooling_sp,
                start_hour as usize,
                end_hour as usize,
            ),
        }
    }

    /// Free-floating profile: no HVAC control at any hour.
    #[napi(factory)]
    pub fn free_floating() -> Self {
        Self {
            inner: HVACSchedule::free_floating(),
        }
    }

    /// True when both heating and cooling are disabled for every hour.
    #[napi]
    pub fn is_free_floating(&self) -> bool {
        self.inner.is_free_floating()
    }

    /// Heating setpoint [°C] for the given hour (0–23).
    #[napi]
    pub fn heating_setpoint(&self, hour: u32) -> f64 {
        self.inner.heating_setpoint(hour as usize)
    }

    /// Cooling setpoint [°C] for the given hour (0–23).
    #[napi]
    pub fn cooling_setpoint(&self, hour: u32) -> f64 {
        self.inner.cooling_setpoint(hour as usize)
    }

    /// Clone of the underlying heating daily schedule.
    #[napi]
    pub fn get_heating_schedule(&self) -> HvacDailySchedule {
        HvacDailySchedule {
            inner: self.inner.heating.clone(),
        }
    }

    /// Clone of the underlying cooling daily schedule.
    #[napi]
    pub fn get_cooling_schedule(&self) -> HvacDailySchedule {
        HvacDailySchedule {
            inner: self.inner.cooling.clone(),
        }
    }
}

impl Default for HvacSchedule {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Zone control
// ────────────────────────────────────────────────────────────────────────────

/// Zone-level HVAC controller with selectable control strategies.
///
/// Wraps [`ZoneControl`] with an internally-managed thermal model so the
/// controller is self-contained from Node.js. Strategies (Issue #1798):
/// - `"ideal_loads"` — thermodynamic ideal-loads response (default)
/// - `"staged_equipment"` — cycling losses + part-load degradation
/// - `"schedule_aware"` — predictive control with time-varying setpoints
#[napi_derive::napi]
pub struct ZoneController {
    inner: Arc<Mutex<ZoneControl>>,
}

#[napi_derive::napi]
impl ZoneController {
    /// Create a new controller for `numZones` zones with default (ideal-loads)
    /// strategy and default setpoints (20/24 °C, 2 °C deadband).
    #[napi(constructor)]
    pub fn new(num_zones: u32) -> napi::bindgen_prelude::Result<Self> {
        if num_zones < 1 {
            return Err(napi::bindgen_prelude::Error::from_reason(
                "Number of zones must be at least 1",
            ));
        }
        let n = num_zones as usize;
        let thermal_model = Arc::new(ThermalModel::new(n, 20.0));
        let setpoints = CoreZoneSetpoints::new(n);
        let control = ZoneControl::new(thermal_model, setpoints);
        Ok(Self {
            inner: Arc::new(Mutex::new(control)),
        })
    }

    /// Set the control strategy for a zone.
    ///
    /// # Arguments
    /// * `zoneId` - Zone index (0-based)
    /// * `strategy` - `"ideal_loads"`, `"staged_equipment"`, or `"schedule_aware"`
    #[napi]
    pub fn set_zone_strategy(
        &self,
        zone_id: u32,
        strategy: String,
    ) -> napi::bindgen_prelude::Result<()> {
        let s = parse_control_strategy(&strategy)?;
        let mut guard = self.lock()?;
        guard.set_zone_strategy(zone_id as usize, s);
        Ok(())
    }

    /// Current control strategy for a zone, or `null` if the zone does not
    /// exist.
    #[napi]
    pub fn get_zone_strategy(&self, zone_id: u32) -> napi::bindgen_prelude::Result<Option<String>> {
        let guard = self.lock()?;
        Ok(guard
            .get_zone_strategy(zone_id as usize)
            .map(|s| control_strategy_str(s).to_string()))
    }

    /// Update HVAC controls for all zones from the current zone temperatures.
    ///
    /// Returns the energy input [W] computed for each zone.
    ///
    /// # Arguments
    /// * `temperatures` - Current zone air temperatures [°C] (one per zone)
    #[napi]
    pub fn update_controls(
        &self,
        temperatures: Vec<f64>,
    ) -> napi::bindgen_prelude::Result<Vec<f64>> {
        let mut guard = self.lock()?;
        let temps = VectorField::new(temperatures);
        let energy = guard.update_zone_controls(&temps);
        Ok(energy.as_slice().to_vec())
    }

    /// HVAC status for a zone: `"heating"`, `"cooling"`, or `"off"`.
    #[napi]
    pub fn get_zone_status(&self, zone_id: u32) -> napi::bindgen_prelude::Result<String> {
        let guard = self.lock()?;
        let status = guard.get_zone_hvac_status(zone_id as usize);
        Ok(hvac_status_str(&status))
    }
}

impl ZoneController {
    fn lock(&self) -> napi::bindgen_prelude::Result<std::sync::MutexGuard<'_, ZoneControl>> {
        self.inner.lock().map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("Failed to lock ZoneControl: {}", e))
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Enum string helpers
// ────────────────────────────────────────────────────────────────────────────

fn heat_pump_mode_str(m: &HeatPumpMode) -> String {
    match m {
        HeatPumpMode::Heating => "heating",
        HeatPumpMode::Cooling => "cooling",
        HeatPumpMode::Off => "off",
    }
    .to_string()
}

fn parse_hvac_mode(s: &str) -> HVACMode {
    match s {
        "heating" => HVACMode::Heating,
        "cooling" => HVACMode::Cooling,
        _ => HVACMode::Off,
    }
}

fn hvac_status_str(s: &HVACStatus) -> String {
    match s {
        HVACStatus::Heating => "heating",
        HVACStatus::Cooling => "cooling",
        HVACStatus::Off => "off",
    }
    .to_string()
}

fn control_strategy_str(s: ControlStrategy) -> &'static str {
    match s {
        ControlStrategy::IdealLoads => "ideal_loads",
        ControlStrategy::StagedEquipment => "staged_equipment",
        ControlStrategy::ScheduleAware => "schedule_aware",
    }
}

fn parse_control_strategy(s: &str) -> napi::bindgen_prelude::Result<ControlStrategy> {
    Ok(match s {
        "ideal_loads" => ControlStrategy::IdealLoads,
        "staged_equipment" => ControlStrategy::StagedEquipment,
        "schedule_aware" => ControlStrategy::ScheduleAware,
        other => {
            return Err(napi::bindgen_prelude::Error::from_reason(format!(
                "Unknown control strategy '{other}'. Expected one of: ideal_loads, staged_equipment, schedule_aware"
            )))
        }
    })
}

fn schedule_type_str(s: &ScheduleType) -> String {
    match s {
        ScheduleType::Constant => "Constant",
        ScheduleType::DailyCycle => "DailyCycle",
        ScheduleType::Weekly => "Weekly",
        ScheduleType::Custom => "Custom",
    }
    .to_string()
}

fn parse_schedule_type(s: &str) -> napi::bindgen_prelude::Result<ScheduleType> {
    Ok(match s {
        "Constant" => ScheduleType::Constant,
        "DailyCycle" => ScheduleType::DailyCycle,
        "Weekly" => ScheduleType::Weekly,
        "Custom" => ScheduleType::Custom,
        other => {
            return Err(napi::bindgen_prelude::Error::from_reason(format!(
                "Invalid schedule type '{other}'. Use one of: Constant, DailyCycle, Weekly, Custom"
            )))
        }
    })
}

fn check_zone(zone_id: usize, num_zones: usize) -> napi::bindgen_prelude::Result<()> {
    if zone_id >= num_zones {
        return Err(napi::bindgen_prelude::Error::from_reason(format!(
            "Zone ID {zone_id} is out of range (0-{})",
            num_zones.saturating_sub(1)
        )));
    }
    Ok(())
}

fn napi_err(e: String) -> napi::bindgen_prelude::Error {
    napi::bindgen_prelude::Error::from_reason(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vav_terminal_round_trip() {
        let mut vav = HvacVavTerminal::new("VAV-1".to_string(), 0, 0.5);
        assert_eq!(vav.id(), "VAV-1");
        assert_eq!(vav.zone_id(), 0);
        assert!((vav.max_airflow() - 0.5).abs() < 1e-9);
        // min defaults to 30% of max
        assert!((vav.min_airflow() - 0.15).abs() < 1e-9);
        assert!((vav.reheat_capacity() - 5000.0).abs() < 1e-9);

        vav.set_reheat_capacity(7500.0);
        vav.set_airflow_setpoint(0.4);
        assert!((vav.reheat_capacity() - 7500.0).abs() < 1e-9);
        assert!((vav.airflow_setpoint() - 0.4).abs() < 1e-9);

        // Reheat is delivered when the zone is below comfort threshold.
        let demand = vav.reheat_demand(20.0, 18.0);
        assert!(demand > 0.0);
        assert!(vav.reheat_demand(20.0, 22.0) == 0.0);
    }

    #[test]
    fn cav_system_round_trip() {
        let mut cav = HvacCavSystem::new("CAV-1".to_string(), 1.0);
        assert_eq!(cav.id(), "CAV-1");
        assert!((cav.design_airflow() - 1.0).abs() < 1e-9);
        // default fan power = 500 W per (m^3/s)
        assert!((cav.fan_power() - 500.0).abs() < 1e-9);
        cav.set_fan_efficiency(0.8);
        assert!((cav.fan_efficiency() - 0.8).abs() < 1e-9);
        // consumption = fan_power / efficiency
        assert!((cav.fan_power_consumption() - (500.0 / 0.8)).abs() < 1e-6);
    }

    #[test]
    fn heat_pump_round_trip_and_mode() {
        let mut hp = HvacHeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0);
        assert_eq!(hp.id(), "HP-1");
        assert_eq!(hp.mode(), "off");

        // COP at design heating temp ~ rated
        let cop_at_design = hp.heating_cop_at_temperature(-5.0);
        assert!((cop_at_design - 3.5).abs() < 0.1);
        // COP is constant (no temperature degradation)
        assert!((hp.heating_cop_at_temperature(-15.0) - 3.5).abs() < 0.1);

        hp.set_mode(18.0, 20.0, 27.0);
        assert_eq!(hp.mode(), "heating");
        assert!(hp.heating_power(-5.0) > 0.0);
        assert!(hp.cooling_power(35.0) == 0.0);

        hp.set_mode(28.0, 20.0, 27.0);
        assert_eq!(hp.mode(), "cooling");
        assert!(hp.cooling_power(35.0) > 0.0);
    }

    #[test]
    fn chiller_and_boiler_capacity() {
        let chiller = HvacChiller::new("CH-1".to_string(), 50000.0, 4.0, 35.0);
        assert!((chiller.rated_capacity() - 50000.0).abs() < 1e-9);
        // Full PLR at design temp delivers ~ rated capacity
        let cap = chiller.calculate_capacity(1.0, 35.0);
        assert!(cap > 0.0);
        // Cooling power = load / cop
        let power = chiller.calculate_power(cap, 35.0, "cooling".to_string());
        assert!(power > 0.0);

        let boiler = HvacBoiler::new("BL-1".to_string(), 40000.0, 0.9, -5.0);
        assert!((boiler.rated_capacity() - 40000.0).abs() < 1e-9);
        let bcap = boiler.calculate_capacity(1.0, -5.0);
        assert!(bcap > 0.0);
        // Non-heating mode => no power
        assert!(boiler.calculate_power(bcap, -5.0, "cooling".to_string()) == 0.0);
    }

    #[test]
    fn zone_setpoints_round_trip() {
        let mut sp = ZoneSetpoints::new(2).unwrap();
        assert_eq!(sp.num_zones(), 2);
        assert!((sp.get_heating_setpoint(0).unwrap() - 20.0).abs() < 1e-9);
        sp.set_heating_setpoint(0, 21.0).unwrap();
        sp.set_cooling_setpoint(0, 25.0).unwrap();
        assert!((sp.get_heating_setpoint(0).unwrap() - 21.0).abs() < 1e-9);
        assert!((sp.get_cooling_setpoint(0).unwrap() - 25.0).abs() < 1e-9);
        assert!(sp.validate().is_ok());
        // out of range
        assert!(sp.set_heating_setpoint(0, 5.0).is_err());
        // bad zone
        assert!(sp.get_deadband(9).is_err());
        // invalid zone count
        assert!(ZoneSetpoints::new(0).is_err());
    }

    #[test]
    fn daily_schedule_round_trip() {
        let mut ds = HvacDailySchedule::new("occ".to_string(), "DailyCycle".to_string()).unwrap();
        ds.fill_range(8, 18, 21.0);
        assert!((ds.value(12) - 21.0).abs() < 1e-9);
        assert!((ds.value(2) - 0.0).abs() < 1e-9);
        assert_eq!(ds.name(), "occ");
        assert_eq!(ds.schedule_type(), "DailyCycle");

        let constant = HvacDailySchedule::constant(24.0);
        assert!((constant.value(0) - 24.0).abs() < 1e-9);
        assert!((constant.value(23) - 24.0).abs() < 1e-9);

        assert!(HvacDailySchedule::new("x".to_string(), "Bogus".to_string()).is_err());
    }

    #[test]
    fn hvac_schedule_profiles() {
        let constant = HvacSchedule::constant_schedule(20.0, 24.0);
        assert!(!constant.is_free_floating());
        assert!((constant.heating_setpoint(5) - 20.0).abs() < 1e-9);
        assert!((constant.cooling_setpoint(5) - 24.0).abs() < 1e-9);

        let setback = HvacSchedule::setback_schedule(20.0, 15.0, 25.0, 22, 6);
        assert!((setback.heating_setpoint(2) - 15.0).abs() < 1e-9);
        assert!((setback.heating_setpoint(10) - 20.0).abs() < 1e-9);

        let occ = HvacSchedule::with_operating_hours(20.0, 24.0, 8, 18);
        assert!((occ.heating_setpoint(12) - 20.0).abs() < 1e-9);
        // Outside operating hours heating is disabled (-100)
        assert!((occ.heating_setpoint(2) - (-100.0)).abs() < 1e-9);

        let ff = HvacSchedule::free_floating();
        assert!(ff.is_free_floating());

        // Sub-schedules round-trip
        let heat = constant.get_heating_schedule();
        assert!((heat.value(0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn zone_controller_strategies_and_status() {
        let ctrl = ZoneController::new(2).unwrap();
        assert_eq!(
            ctrl.get_zone_strategy(0).unwrap(),
            Some("ideal_loads".to_string())
        );
        ctrl.set_zone_strategy(0, "staged_equipment".to_string())
            .unwrap();
        assert_eq!(
            ctrl.get_zone_strategy(0).unwrap(),
            Some("staged_equipment".to_string())
        );
        // Unknown strategy rejected
        assert!(ctrl.set_zone_strategy(0, "bogus".to_string()).is_err());

        // Cold zone => heating; hot zone => cooling
        let energy = ctrl.update_controls(vec![15.0, 30.0]).unwrap();
        assert_eq!(energy.len(), 2);
        assert_eq!(ctrl.get_zone_status(0).unwrap(), "heating");
        assert_eq!(ctrl.get_zone_status(1).unwrap(), "cooling");

        assert!(ZoneController::new(0).is_err());
    }
}
