//! Parallel Harness for Urban Building Simulation (Issue #2034)
//!
//! Thread-safe parallel execution of urban radiation/thermal simulations
//! with configurable worker threads and memory-efficient building management.

use std::time::Duration;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const STEFAN_BOLTZMANN: f64 = 5.67e-8;

#[derive(Debug, Clone)]
pub struct BuildingGroup {
    pub id: u32,
    pub area: f64,
    pub u_wall: f64,
    pub u_roof: f64,
    pub u_floor: f64,
    pub temperature: f64,
    pub outdoor_temperature: f64,
    pub absorbed_solar: f64,
    pub emitted_longwave: f64,
}

impl BuildingGroup {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            area: 100.0,
            u_wall: 0.5,
            u_roof: 0.3,
            u_floor: 2.0,
            temperature: 293.15,
            outdoor_temperature: 293.15,
            absorbed_solar: 0.0,
            emitted_longwave: 0.0,
        }
    }

    pub fn with_area(mut self, area: f64) -> Self {
        self.area = area;
        self
    }

    pub fn with_u_values(mut self, u_wall: f64, u_roof: f64, u_floor: f64) -> Self {
        self.u_wall = u_wall;
        self.u_roof = u_roof;
        self.u_floor = u_floor;
        self
    }

    pub fn step(&mut self, dt: &Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        let dt_hours = dt.as_secs_f64() / 3600.0;
        let surface_area = self.area * 4.0;
        let wall_area = surface_area * 0.8;
        let roof_area = surface_area * 0.2;

        let q_solar = radiation.solar_irradiance * self.area * radiation.absorptivity;
        let q_sky = radiation.sky_temperature.powi(4) - self.temperature.powi(4);
        let q_longwave = radiation.emissivity * STEFAN_BOLTZMANN * wall_area * q_sky;
        let q_conduction_wall = self.u_wall * wall_area * (outdoor_temp - self.temperature);
        let q_conduction_roof = self.u_roof * roof_area * (outdoor_temp - self.temperature);
        let q_conduction_floor =
            self.u_floor * (self.area * 0.1) * (self.outdoor_temperature - self.temperature);

        let net_gain =
            q_solar + q_longwave + q_conduction_wall + q_conduction_roof + q_conduction_floor;
        let heat_capacity = self.area * 500.0 * 1005.0;

        self.temperature += (net_gain / heat_capacity) * dt_hours;
        self.temperature = self.temperature.max(200.0).min(400.0);
        self.absorbed_solar = q_solar;
        self.emitted_longwave = q_longwave;
    }
}

#[derive(Debug, Clone)]
pub struct UrbanRadiationSystem {
    pub solar_irradiance: f64,
    pub sky_temperature: f64,
    pub emissivity: f64,
    pub absorptivity: f64,
    pub latitude: f64,
    pub longitude: f64,
}

impl UrbanRadiationSystem {
    pub fn new(
        solar_irradiance: f64,
        sky_temperature: f64,
        emissivity: f64,
        absorptivity: f64,
        latitude: f64,
        longitude: f64,
    ) -> Self {
        Self {
            solar_irradiance,
            sky_temperature,
            emissivity,
            absorptivity,
            latitude,
            longitude,
        }
    }
}

#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct UrbanStepDispatcher {
    buildings: Vec<BuildingGroup>,
    num_threads: usize,
}

#[cfg(feature = "parallel")]
impl UrbanStepDispatcher {
    pub fn with_buildings(buildings: Vec<BuildingGroup>) -> Self {
        Self {
            buildings,
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn num_buildings(&self) -> usize {
        self.buildings.len()
    }

    pub fn step_all(&mut self, dt: Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        let num_threads = self.num_threads;
        let chunk_size = (self.buildings.len() / num_threads).max(1);

        self.buildings.par_chunks_mut(chunk_size).for_each(|chunk| {
            for building in chunk.iter_mut() {
                building.step(&dt, radiation, outdoor_temp);
            }
        });
    }

    pub fn step_sequential(
        &mut self,
        dt: &Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) {
        for building in self.buildings.iter_mut() {
            building.step(dt, radiation, outdoor_temp);
        }
    }

    pub fn get_buildings(&self) -> &[BuildingGroup] {
        &self.buildings
    }

    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
    }
}

#[cfg(not(feature = "parallel"))]
#[derive(Debug, Clone)]
pub struct UrbanStepDispatcher {
    buildings: Vec<BuildingGroup>,
}

#[cfg(not(feature = "parallel"))]
impl UrbanStepDispatcher {
    pub fn with_buildings(buildings: Vec<BuildingGroup>) -> Self {
        Self { buildings }
    }

    pub fn with_threads(mut self, _num_threads: usize) -> Self {
        self
    }

    pub fn num_buildings(&self) -> usize {
        self.buildings.len()
    }

    pub fn step_all(&mut self, dt: Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        for building in self.buildings.iter_mut() {
            building.step(&dt, radiation, outdoor_temp);
        }
    }

    pub fn step_sequential(
        &mut self,
        dt: &Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) {
        self.step_all(dt, radiation, outdoor_temp);
    }

    pub fn get_buildings(&self) -> &[BuildingGroup] {
        &self.buildings
    }

    pub fn set_num_threads(&mut self, _num_threads: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_building_group_creation() {
        let building = BuildingGroup::new(1);
        assert_eq!(building.id, 1);
        assert_eq!(building.area, 100.0);
    }

    #[test]
    fn test_building_group_with_area() {
        let building = BuildingGroup::new(1).with_area(200.0);
        assert_eq!(building.area, 200.0);
    }

    #[test]
    fn test_building_group_with_u_values() {
        let building = BuildingGroup::new(1).with_u_values(0.3, 0.2, 1.5);
        assert_eq!(building.u_wall, 0.3);
        assert_eq!(building.u_roof, 0.2);
        assert_eq!(building.u_floor, 1.5);
    }

    #[test]
    fn test_building_step() {
        let mut building = BuildingGroup::new(1);
        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);
        let initial_temp = building.temperature;

        building.step(&dt, &radiation, 300.0);
        assert_ne!(building.temperature, initial_temp);
    }

    #[test]
    fn test_urban_radiation_system_creation() {
        let radiation = UrbanRadiationSystem::new(800.0, 120.0, 0.2, 0.85, 0.1, 2.0);
        assert_eq!(radiation.solar_irradiance, 800.0);
        assert_eq!(radiation.sky_temperature, 120.0);
    }

    #[test]
    fn test_urban_step_dispatcher_creation() {
        let buildings: Vec<BuildingGroup> = (0..5).map(|i| BuildingGroup::new(i as u32)).collect();
        let dispatcher = UrbanStepDispatcher::with_buildings(buildings);
        assert_eq!(dispatcher.num_buildings(), 5);
    }

    #[test]
    fn test_urban_step_dispatcher_step() {
        let buildings: Vec<BuildingGroup> = (0..10).map(|i| BuildingGroup::new(i as u32)).collect();
        let mut dispatcher = UrbanStepDispatcher::with_buildings(buildings);
        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        dispatcher.step_all(dt, &radiation, 300.0);
        assert_eq!(dispatcher.num_buildings(), 10);
    }

    #[test]
    fn test_sequential_vs_parallel_consistency() {
        let buildings: Vec<BuildingGroup> = (0..20)
            .map(|i| BuildingGroup::new(i as u32).with_area(100.0 + i as f64))
            .collect();
        let mut dispatcher_seq = UrbanStepDispatcher::with_buildings(buildings.clone());
        let mut dispatcher_par = UrbanStepDispatcher::with_buildings(buildings.clone());

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        dispatcher_seq.step_sequential(&dt, &radiation, 300.0);
        dispatcher_par.step_all(dt, &radiation, 300.0);

        let seq_buildings = dispatcher_seq.get_buildings();
        let par_buildings = dispatcher_par.get_buildings();

        for (seq, par) in seq_buildings.iter().zip(par_buildings.iter()) {
            approx::assert_abs_diff_eq!(seq.temperature, par.temperature, epsilon = 1e-10);
        }
    }
}
