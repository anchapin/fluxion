//! Setpoint state — per-zone primary state, setpoints, schedules, zone geometry.
//!
//! Extracted from `ThermalModelData` (Issue #2878). Owns the most-frequently-
//! accessed per-zone tensors (`temperatures`, `loads`, the geometry/material
//! scalars) plus the heating/cooling setpoints, daily schedules, and the
//! ventilation airflow used by the economizer / night-ventilation paths.

use crate::sim::schedule::DailySchedule;
use super::ContinuousTensor;

pub struct SetpointState<T: ContinuousTensor<f64>> {
    // Per-zone primary state (the zone air temperature and the per-zone load).
    pub temperatures: T,
    pub loads: T,

    // Setpoints + schedules.
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub heating_setpoints: T,
    pub cooling_setpoints: T,
    pub heating_schedule: DailySchedule,
    pub cooling_schedule: DailySchedule,

    // Per-zone zone geometry / material scalars (W/K or m² or kg/m³).
    pub zone_area: T,
    pub wall_area: T,
    pub roof_area: T,
    pub floor_area: T,
    pub ceiling_height: T,
    pub air_density: T,
    pub heat_capacity: T,
    pub window_ratio: T,
    pub aspect_ratio: T,
    pub infiltration_rate: T,

    // Opaque surface U-values from construction (Issue #375).
    pub wall_u_value: f64,
    pub roof_u_value: f64,
    pub floor_u_value: f64,

    // Per-zone zone volume (m³) and inter-zone common-wall area (m²).
    pub zone_volume: T,
    pub common_wall_area: f64,

    // Ventilation / envelope bridge.
    pub thermal_bridge_coefficient: f64,
    pub ventilation_airflow_m3_per_s: f64,
    pub h_vent_mass: f64,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for SetpointState<T> {
    fn clone(&self) -> Self {
        Self {
            temperatures: self.temperatures.clone(),
            loads: self.loads.clone(),

            heating_setpoint: self.heating_setpoint,
            cooling_setpoint: self.cooling_setpoint,
            heating_setpoints: self.heating_setpoints.clone(),
            cooling_setpoints: self.cooling_setpoints.clone(),
            heating_schedule: self.heating_schedule.clone(),
            cooling_schedule: self.cooling_schedule.clone(),

            zone_area: self.zone_area.clone(),
            wall_area: self.wall_area.clone(),
            roof_area: self.roof_area.clone(),
            floor_area: self.floor_area.clone(),
            ceiling_height: self.ceiling_height.clone(),
            air_density: self.air_density.clone(),
            heat_capacity: self.heat_capacity.clone(),
            window_ratio: self.window_ratio.clone(),
            aspect_ratio: self.aspect_ratio.clone(),
            infiltration_rate: self.infiltration_rate.clone(),

            wall_u_value: self.wall_u_value,
            roof_u_value: self.roof_u_value,
            floor_u_value: self.floor_u_value,

            zone_volume: self.zone_volume.clone(),
            common_wall_area: self.common_wall_area,

            thermal_bridge_coefficient: self.thermal_bridge_coefficient,
            ventilation_airflow_m3_per_s: self.ventilation_airflow_m3_per_s,
            h_vent_mass: self.h_vent_mass,
        }
    }
}
