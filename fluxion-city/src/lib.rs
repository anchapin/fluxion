//! fluxion-city: Urban-scale building energy modeling
//!
//! Provides parallel dispatch for multiple buildings with deterministic
//! execution guarantees and performance benchmarking.
//!
//! # Architecture
//!
//! - [`BuildingGroup`] - A single building with thermal state
//! - [`UrbanRadiationSystem`] - Urban heat island / solar shading model
//! - [`UrbanStepDispatcher`] - Parallel timestep coordinator using Rayon
//! - [`UrbanStepResult`] - Aggregated results from parallel evaluation

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ============================================================================
// UrbanRadiationSystem (Issue #2029 / #2031 — urban heat island + solar shading)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrbanRadiationSystem {
    pub direct_normal: f64,
    pub diffuse_horizontal: f64,
    pub ground_reflectance: f64,
    pub urban_sky_view_factor: f64,
    pub neighboring_shading_factor: f64,
    pub heat_island_offset: f64,
}

impl UrbanRadiationSystem {
    pub fn new(
        direct_normal: f64,
        diffuse_horizontal: f64,
        ground_reflectance: f64,
        urban_sky_view_factor: f64,
        neighboring_shading_factor: f64,
        heat_island_offset: f64,
    ) -> Self {
        Self {
            direct_normal,
            diffuse_horizontal,
            ground_reflectance,
            urban_sky_view_factor,
            neighboring_shading_factor,
            heat_island_offset,
        }
    }

    pub fn effective_irradiance(&self, surface_tilt: f64, _surface_azimuth: f64) -> f64 {
        let sky_factor = self.urban_sky_view_factor * (1.0 - self.neighboring_shading_factor);
        let tilt_factor = (surface_tilt.to_radians()).cos();
        let direct_component = self.direct_normal * tilt_factor * sky_factor;
        let diffuse_component = self.diffuse_horizontal * sky_factor * tilt_factor;
        let ground_reflected = self.diffuse_horizontal * self.ground_reflectance * (1.0 - tilt_factor);
        direct_component + diffuse_component + ground_reflected
    }

    pub fn ambient_temperature(&self, outdoor_drybulb: f64) -> f64 {
        outdoor_drybulb + self.heat_island_offset
    }
}

// ============================================================================
// BuildingGroup — individual building thermal model
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingGroup {
    pub id: u32,
    pub zone_temperature: f64,
    pub wall_temperature: f64,
    pub roof_temperature: f64,
    pub floor_mass_temperature: f64,
    pub hvac_setpoint_cooling: f64,
    pub hvac_setpoint_heating: f64,
    pub floor_area: f64,
    pub coefficient_of_performance: f64,
    pub lighting_load: f64,
    pub occupancy_gain: f64,
    pub equipment_gain: f64,
    pub wall_u_value: f64,
    pub roof_u_value: f64,
    pub window_u_value: f64,
    pub window_shading_coefficient: f64,
}

impl BuildingGroup {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            zone_temperature: 22.0,
            wall_temperature: 20.0,
            roof_temperature: 18.0,
            floor_mass_temperature: 20.0,
            hvac_setpoint_cooling: 26.0,
            hvac_setpoint_heating: 18.0,
            floor_area: 100.0,
            coefficient_of_performance: 3.0,
            lighting_load: 10.0,
            occupancy_gain: 100.0,
            equipment_gain: 200.0,
            wall_u_value: 0.5,
            roof_u_value: 0.3,
            window_u_value: 2.0,
            window_shading_coefficient: 0.7,
        }
    }

    pub fn with_area(mut self, area: f64) -> Self {
        self.floor_area = area;
        self
    }

    pub fn with_u_values(mut self, wall: f64, roof: f64, window: f64) -> Self {
        self.wall_u_value = wall;
        self.roof_u_value = roof;
        self.window_u_value = window;
        self
    }

    pub fn step(&mut self, dt: &Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        let dt_hours = dt.as_secs_f64() / 3600.0;

        let _sky_irradiance = radiation.effective_irradiance(90.0, 0.0);
        let wall_irradiance = radiation.effective_irradiance(90.0, 0.0);
        let roof_irradiance = radiation.effective_irradiance(0.0, 0.0);

        let wall_solar_gain = wall_irradiance * 0.15 * self.window_shading_coefficient;
        let roof_solar_gain = roof_irradiance * 0.12 * self.window_shading_coefficient;

        let wall_loss = self.wall_u_value * (self.wall_temperature - outdoor_temp) * dt_hours;
        let roof_loss = self.roof_u_value * (self.roof_temperature - outdoor_temp) * dt_hours;
        let window_loss = self.window_u_value * (self.zone_temperature - outdoor_temp) * dt_hours;

        let wall_mass_coupling = 0.05 * (self.wall_temperature - self.floor_mass_temperature) * dt_hours;
        let roof_mass_coupling = 0.04 * (self.roof_temperature - self.floor_mass_temperature) * dt_hours;

        let heating_setpoint = self.hvac_setpoint_heating;
        let cooling_setpoint = self.hvac_setpoint_cooling;
        let zone_error = if self.zone_temperature < heating_setpoint {
            heating_setpoint - self.zone_temperature
        } else if self.zone_temperature > cooling_setpoint {
            self.zone_temperature - cooling_setpoint
        } else {
            0.0
        };

        let internal_gains = self.occupancy_gain + self.equipment_gain + self.lighting_load;
        let hvac_impact = (zone_error * 50.0 / self.coefficient_of_performance) * dt_hours;

        self.wall_temperature += wall_solar_gain - wall_loss + wall_mass_coupling;
        self.roof_temperature += roof_solar_gain - roof_loss + roof_mass_coupling;
        self.floor_mass_temperature += (wall_mass_coupling + roof_mass_coupling) * 0.5;
        self.zone_temperature += (internal_gains - window_loss - hvac_impact) / (self.floor_area * 0.5);

        self.wall_temperature = self.wall_temperature.clamp(-30.0, 60.0);
        self.roof_temperature = self.roof_temperature.clamp(-30.0, 60.0);
        self.floor_mass_temperature = self.floor_mass_temperature.clamp(-30.0, 60.0);
        self.zone_temperature = self.zone_temperature.clamp(10.0, 40.0);
    }

    pub fn thermal_load(&self) -> f64 {
        let load = (self.hvac_setpoint_heating - self.zone_temperature).abs()
            + (self.zone_temperature - self.hvac_setpoint_cooling).abs();
        load * self.floor_area * 0.1
    }

    pub fn energy_consumption(&self) -> f64 {
        let cooling_load = (self.zone_temperature - self.hvac_setpoint_cooling).max(0.0);
        let heating_load = (self.hvac_setpoint_heating - self.zone_temperature).max(0.0);
        (cooling_load + heating_load) * self.floor_area * 0.05 / self.coefficient_of_performance
    }
}

// ============================================================================
// UrbanStepResult — aggregation from parallel building evaluation
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UrbanStepResult {
    pub total_thermal_load: f64,
    pub total_energy_consumption: f64,
    pub avg_zone_temperature: f64,
    pub max_zone_temperature: f64,
    pub min_zone_temperature: f64,
    pub buildings_processed: usize,
}

impl UrbanStepResult {
    pub fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, building: &BuildingGroup) {
        self.total_thermal_load += building.thermal_load();
        self.total_energy_consumption += building.energy_consumption();
        self.max_zone_temperature = self.max_zone_temperature.max(building.zone_temperature);
        self.min_zone_temperature = self.min_zone_temperature.min(building.zone_temperature);
    }

    fn finalize(&mut self, count: usize) {
        self.buildings_processed = count;
        if count > 0 {
            self.avg_zone_temperature = self.total_thermal_load / count as f64;
        }
    }
}

// ============================================================================
// UrbanStepDispatcher — Rayon-based parallel building evaluation (Issue #2032)
// ============================================================================

#[derive(Debug, Clone)]
pub struct UrbanStepDispatcher {
    pub building_groups: Vec<BuildingGroup>,
}

impl UrbanStepDispatcher {
    pub fn new() -> Self {
        Self {
            building_groups: Vec::new(),
        }
    }

    pub fn add_building(&mut self, building: BuildingGroup) {
        self.building_groups.push(building);
    }

    pub fn with_buildings(mut self, buildings: Vec<BuildingGroup>) -> Self {
        self.building_groups = buildings;
        self
    }

    pub fn step_all(&mut self, dt: Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) -> UrbanStepResult {
        if self.building_groups.is_empty() {
            return UrbanStepResult::new();
        }

        let count = self.building_groups.len();

        self.building_groups
            .par_iter_mut()
            .with_max_len(1)
            .for_each(|building| {
                building.step(&dt, radiation, outdoor_temp);
            });

        let mut result = UrbanStepResult::new();
        for building in &self.building_groups {
            result.update(building);
        }
        result.finalize(count);

        result
    }

    pub fn len(&self) -> usize {
        self.building_groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.building_groups.is_empty()
    }
}

impl Default for UrbanStepDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Deterministic utilities (Issue #2033)
// ============================================================================

pub fn verify_deterministic_results<T: Clone + PartialEq>(results: &[T], _labels: &[&str]) -> bool {
    if results.is_empty() {
        return true;
    }
    let first = &results[0];
    results.iter().all(|r| r == first)
}

pub fn deterministic_reduction<T: Clone + Send + Sync + std::ops::Add<Output = T> + Default>(
    values: &[T],
) -> T {
    if values.is_empty() {
        return T::default();
    }
    let mut sorted: Vec<_> = values.iter().enumerate().collect();
    sorted.sort_by_key(|(idx, _)| *idx);
    sorted.into_iter().map(|(_, v)| v.clone()).reduce(|a, b| a + b).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_radiation() -> UrbanRadiationSystem {
        UrbanRadiationSystem::new(
            800.0,
            120.0,
            0.2,
            0.85,
            0.1,
            2.0,
        )
    }

    fn make_test_buildings(n: usize) -> Vec<BuildingGroup> {
        (0..n).map(|i| BuildingGroup::new(i as u32)).collect()
    }

    #[test]
    fn test_building_group_step() {
        let radiation = make_test_radiation();
        let mut building = BuildingGroup::new(1);
        let initial_temp = building.zone_temperature;

        building.step(&Duration::from_secs(3600), &radiation, 30.0);
        assert_ne!(building.zone_temperature, initial_temp);
    }

    #[test]
    fn test_building_energy_consumption() {
        let mut building = BuildingGroup::new(1);
        building.zone_temperature = 28.0;
        let energy = building.energy_consumption();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_thermal_load() {
        let building = BuildingGroup::new(1);
        let load = building.thermal_load();
        assert!(load >= 0.0);
    }

    #[test]
    fn test_dispatcher_empty() {
        let mut dispatcher = UrbanStepDispatcher::new();
        let radiation = make_test_radiation();
        let result = dispatcher.step_all(Duration::from_secs(3600), &radiation, 30.0);
        assert_eq!(result.buildings_processed, 0);
    }

    #[test]
    fn test_dispatcher_single_building() {
        let radiation = make_test_radiation();
        let buildings = make_test_buildings(1);
        let mut dispatcher = UrbanStepDispatcher::new().with_buildings(buildings);
        let result = dispatcher.step_all(Duration::from_secs(3600), &radiation, 30.0);
        assert_eq!(result.buildings_processed, 1);
    }

    #[test]
    fn test_deterministic_reduction() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = deterministic_reduction(&values);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_deterministic_empty() {
        let values: Vec<f64> = vec![];
        let result = deterministic_reduction(&values);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_verify_deterministic_results() {
        let results = vec![1.0, 1.0, 1.0];
        let labels = vec!["a", "b", "c"];
        assert!(verify_deterministic_results(&results, &labels));
    }

    #[test]
    fn test_urban_radiation_effective_irradiance() {
        let radiation = make_test_radiation();
        let irradiance = radiation.effective_irradiance(90.0, 0.0);
        assert!(irradiance >= 0.0);
    }

    #[test]
    fn test_urban_radiation_ambient_temperature() {
        let radiation = make_test_radiation();
        let ambient = radiation.ambient_temperature(25.0);
        assert_eq!(ambient, 27.0);
    }

    #[test]
    fn test_dispatcher_multiple_runs_deterministic() {
        let radiation = make_test_radiation();
        let buildings = make_test_buildings(10);

        let mut dispatcher1 = UrbanStepDispatcher::new().with_buildings(buildings.clone());
        let result1 = dispatcher1.step_all(Duration::from_secs(3600), &radiation, 30.0);

        let mut dispatcher2 = UrbanStepDispatcher::new().with_buildings(buildings);
        let result2 = dispatcher2.step_all(Duration::from_secs(3600), &radiation, 30.0);

        assert_eq!(result1.total_thermal_load, result2.total_thermal_load);
        assert_eq!(result1.total_energy_consumption, result2.total_energy_consumption);
        assert_eq!(result1.avg_zone_temperature, result2.avg_zone_temperature);
    }
}
