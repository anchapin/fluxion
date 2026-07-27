//! Urban energy modeling and city-scale radiation tests for Fluxion.
//!
//! This crate provides:
//! - 5-building test configuration for energy conservation verification
//! - Longwave radiative equilibrium tests
//! - Surface radiation balance tests
//!
//! # Energy Conservation Principle
//!
//! For any enclosed surface in radiative equilibrium:
//! `Q_absorbed = Q_emitted + Q_transmitted`
//!
//! The net radiation `Q_net = Q_absorbed - Q_emitted - Q_transmitted` must be zero
//! (within numerical tolerance of 1e-6 W).

use serde::{Deserialize, Serialize};

const STEFAN_BOLTZMANN: f64 = 5.67e-8;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BuildingConfig {
    pub length: f64,
    pub width: f64,
    pub height: f64,
    pub emissivity: f64,
    pub absorptivity: f64,
    pub thermal_conductance: f64,
}

impl BuildingConfig {
    pub fn new(
        length: f64,
        width: f64,
        height: f64,
        emissivity: f64,
        absorptivity: f64,
        thermal_conductance: f64,
    ) -> Self {
        Self {
            length,
            width,
            height,
            emissivity,
            absorptivity,
            thermal_conductance,
        }
    }

    pub fn surface_area(&self) -> f64 {
        2.0 * (self.length * self.width + self.length * self.height + self.width * self.height)
    }

    pub fn wall_area(&self) -> f64 {
        2.0 * (self.length * self.height + self.width * self.height)
    }

    pub fn roof_area(&self) -> f64 {
        self.length * self.width
    }

    pub fn floor_area(&self) -> f64 {
        self.length * self.width
    }
}

impl Default for BuildingConfig {
    fn default() -> Self {
        Self {
            length: 10.0,
            width: 10.0,
            height: 3.0,
            emissivity: 0.9,
            absorptivity: 0.7,
            thermal_conductance: 0.45,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceRadiation {
    pub absorbed: f64,
    pub emitted: f64,
    pub transmitted: f64,
    pub reflected: f64,
}

impl SurfaceRadiation {
    pub fn new(absorbed: f64, emitted: f64, transmitted: f64, reflected: f64) -> Self {
        Self {
            absorbed,
            emitted,
            transmitted,
            reflected,
        }
    }

    pub fn net_radiation(&self) -> f64 {
        self.absorbed - self.emitted - self.transmitted - self.reflected
    }

    pub fn is_conserved(&self, tolerance: f64) -> bool {
        self.net_radiation().abs() < tolerance
    }
}

pub struct EnergyConservationTest {
    pub buildings: Vec<BuildingConfig>,
    pub ambient_temperature: f64,
    pub solar_irradiance: f64,
    pub sky_temperature: f64,
}

impl Default for EnergyConservationTest {
    fn default() -> Self {
        Self {
            buildings: Vec::new(),
            ambient_temperature: 293.15,
            solar_irradiance: 0.0,
            sky_temperature: 270.0,
        }
    }
}

impl EnergyConservationTest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_buildings(mut self, buildings: Vec<BuildingConfig>) -> Self {
        self.buildings = buildings;
        self
    }

    pub fn with_ambient_temperature(mut self, temperature: f64) -> Self {
        self.ambient_temperature = temperature;
        self
    }

    pub fn with_solar_irradiance(mut self, irradiance: f64) -> Self {
        self.solar_irradiance = irradiance;
        self
    }

    pub fn with_sky_temperature(mut self, temperature: f64) -> Self {
        self.sky_temperature = temperature;
        self
    }

    pub fn create_5_building_config() -> Self {
        let building1 = BuildingConfig {
            length: 10.0,
            width: 10.0,
            height: 3.0,
            emissivity: 0.9,
            absorptivity: 0.7,
            thermal_conductance: 0.45,
        };
        let building2 = BuildingConfig {
            length: 8.0,
            width: 8.0,
            height: 4.0,
            emissivity: 0.85,
            absorptivity: 0.6,
            thermal_conductance: 0.40,
        };
        let building3 = BuildingConfig {
            length: 12.0,
            width: 6.0,
            height: 3.5,
            emissivity: 0.9,
            absorptivity: 0.75,
            thermal_conductance: 0.50,
        };
        let building4 = BuildingConfig {
            length: 7.0,
            width: 7.0,
            height: 5.0,
            emissivity: 0.88,
            absorptivity: 0.65,
            thermal_conductance: 0.42,
        };
        let building5 = BuildingConfig {
            length: 9.0,
            width: 11.0,
            height: 3.0,
            emissivity: 0.92,
            absorptivity: 0.70,
            thermal_conductance: 0.48,
        };

        Self {
            buildings: vec![building1, building2, building3, building4, building5],
            ambient_temperature: 293.15,
            solar_irradiance: 500.0,
            sky_temperature: 270.0,
        }
    }

    fn net_balance_at_temperature(&self, building: &BuildingConfig, t_surface: f64) -> f64 {
        let wall_area = building.wall_area();
        let roof_area = building.roof_area();
        let total_area = building.surface_area();

        let absorbed_solar = building.absorptivity * self.solar_irradiance * (wall_area + roof_area);

        let emitted = building.emissivity * STEFAN_BOLTZMANN * total_area * t_surface.powi(4);

        let transmitted = building.thermal_conductance * total_area * (t_surface - self.ambient_temperature);

        let sky_radiation = building.emissivity
            * STEFAN_BOLTZMANN
            * total_area
            * (t_surface.powi(4) - self.sky_temperature.powi(4));

        absorbed_solar - emitted - transmitted - sky_radiation
    }

    fn find_equilibrium_temperature(&self, building: &BuildingConfig) -> f64 {
        let t_min = 200.0;
        let t_max = 400.0;
        let balance_tolerance = 1e-12;
        let t_tolerance = 1e-14;
        let max_iterations = 500;

        let mut t_low = t_min;
        let mut t_high = t_max;
        let mut balance_low = self.net_balance_at_temperature(building, t_low);
        let balance_high = self.net_balance_at_temperature(building, t_high);

        if balance_low * balance_high > 0.0 {
            if balance_low > 0.0 {
                return t_min;
            } else {
                return t_max;
            }
        }

        for _ in 0..max_iterations {
            let t_mid = (t_low + t_high) / 2.0;
            let balance_mid = self.net_balance_at_temperature(building, t_mid);

            if balance_mid.abs() < balance_tolerance {
                return t_mid;
            }

            if (t_high - t_low) < t_tolerance * t_low {
                return t_mid;
            }

            if balance_low * balance_mid <= 0.0 {
                t_high = t_mid;
            } else {
                t_low = t_mid;
                balance_low = balance_mid;
            }
        }

        (t_low + t_high) / 2.0
    }

    pub fn surface_radiation_balance(&self, building_index: usize) -> Option<SurfaceRadiation> {
        let building = self.buildings.get(building_index)?;

        let wall_area = building.wall_area();
        let roof_area = building.roof_area();
        let total_area = building.surface_area();

        let absorbed_solar = building.absorptivity * self.solar_irradiance * (wall_area + roof_area);

        let t_eq = self.find_equilibrium_temperature(building);

        let emitted = building.emissivity * STEFAN_BOLTZMANN * total_area * t_eq.powi(4);

        let transmitted = building.thermal_conductance * total_area * (t_eq - self.ambient_temperature);

        let sky_radiation = building.emissivity
            * STEFAN_BOLTZMANN
            * total_area
            * (t_eq.powi(4) - self.sky_temperature.powi(4));

        Some(SurfaceRadiation::new(
            absorbed_solar,
            emitted + sky_radiation,
            transmitted,
            0.0,
        ))
    }

    pub fn verify_conservation(&self) -> bool {
        let imbalance = self.max_imbalance();
        imbalance < 1e-6
    }

    pub fn max_imbalance(&self) -> f64 {
        let mut max_imbalance = 0.0f64;

        for building in &self.buildings {
            if let Some(radiation) = self.surface_radiation_balance_for_building(building) {
                let imbalance = radiation.net_radiation().abs();
                if imbalance > max_imbalance {
                    max_imbalance = imbalance;
                }
            }
        }

        max_imbalance
    }

    pub fn all_surfaces_balanced(&self, tolerance: f64) -> bool {
        for (i, _) in self.buildings.iter().enumerate() {
            if let Some(radiation) = self.surface_radiation_balance(i) {
                if !radiation.is_conserved(tolerance) {
                    return false;
                }
            }
        }
        true
    }

    pub fn net_radiation_for_enclosed_surfaces(&self) -> f64 {
        let mut total_net = 0.0;

        for building in &self.buildings {
            if let Some(radiation) = self.surface_radiation_balance_for_building(building) {
                total_net += radiation.net_radiation();
            }
        }

        total_net
    }

    fn surface_radiation_balance_for_building(&self, building: &BuildingConfig) -> Option<SurfaceRadiation> {
        let wall_area = building.wall_area();
        let roof_area = building.roof_area();
        let total_area = building.surface_area();

        let absorbed_solar = building.absorptivity * self.solar_irradiance * (wall_area + roof_area);

        let t_eq = self.find_equilibrium_temperature(building);

        let emitted = building.emissivity * STEFAN_BOLTZMANN * total_area * t_eq.powi(4);

        let transmitted = building.thermal_conductance * total_area * (t_eq - self.ambient_temperature);

        let sky_radiation = building.emissivity
            * STEFAN_BOLTZMANN
            * total_area
            * (t_eq.powi(4) - self.sky_temperature.powi(4));

        Some(SurfaceRadiation::new(
            absorbed_solar,
            emitted + sky_radiation,
            transmitted,
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_5_building_energy_conservation() {
        let test = EnergyConservationTest::create_5_building_config();
        let imbalance = test.max_imbalance();
        assert!(
            imbalance < 1e-6,
            "Energy imbalance {} exceeds tolerance 1e-6 W",
            imbalance
        );
    }

    #[test]
    fn test_energy_conservation_verify() {
        let test = EnergyConservationTest::create_5_building_config();
        assert!(
            test.verify_conservation(),
            "Energy conservation verification failed"
        );
    }

    #[test]
    fn test_surface_radiation_balance() {
        let test = EnergyConservationTest::create_5_building_config();
        let balance = test.surface_radiation_balance(0);
        assert!(
            balance.is_some(),
            "Should get radiation balance for building 0"
        );
        let balance = balance.unwrap();
        assert!(
            balance.net_radiation().abs() < 1e-6,
            "Net radiation {} should be near zero",
            balance.net_radiation()
        );
    }

    #[test]
    fn test_all_surfaces_balanced() {
        let test = EnergyConservationTest::create_5_building_config();
        assert!(
            test.all_surfaces_balanced(1e-6),
            "All surfaces should be balanced within tolerance"
        );
    }

    #[test]
    fn test_building_surface_area() {
        let building = BuildingConfig::default();
        let expected_area = 2.0 * (100.0 + 30.0 + 30.0);
        assert!((building.surface_area() - expected_area).abs() < 1e-10);
    }

    #[test]
    fn test_building_wall_area() {
        let building = BuildingConfig::default();
        let expected_wall = 2.0 * (10.0 * 3.0 + 10.0 * 3.0);
        assert!((building.wall_area() - expected_wall).abs() < 1e-10);
    }

    #[test]
    fn test_building_roof_area() {
        let building = BuildingConfig::default();
        let expected_roof = 100.0;
        assert!((building.roof_area() - expected_roof).abs() < 1e-10);
    }

    #[test]
    fn test_single_building_conservation() {
        let building = BuildingConfig::default();
        let test = EnergyConservationTest::new()
            .with_buildings(vec![building])
            .with_ambient_temperature(293.15)
            .with_solar_irradiance(500.0)
            .with_sky_temperature(270.0);

        assert!(
            test.verify_conservation(),
            "Single building should satisfy energy conservation"
        );
    }

    #[test]
    fn test_zero_solar_irradiance() {
        let test = EnergyConservationTest::create_5_building_config()
            .with_solar_irradiance(0.0);

        assert!(
            test.verify_conservation(),
            "Should still conserve energy with zero solar"
        );
    }

    #[test]
    fn test_longwave_equilibrium() {
        let test = EnergyConservationTest::create_5_building_config();

        for (i, building) in test.buildings.iter().enumerate() {
            let radiation = test.surface_radiation_balance(i);
            assert!(
                radiation.is_some(),
                "Building {} should have radiation balance",
                i
            );

            let rad = radiation.unwrap();
            let imbalance = (rad.absorbed - rad.emitted - rad.transmitted).abs();
            assert!(
                imbalance < 1e-6,
                "Building {} longwave equilibrium imbalance: {}",
                i,
                imbalance
            );
        }
    }

    #[test]
    fn test_enclosed_surfaces_net_zero() {
        let test = EnergyConservationTest::create_5_building_config();

        for (i, _) in test.buildings.iter().enumerate() {
            let radiation = test.surface_radiation_balance(i);
            assert!(radiation.is_some());

            let rad = radiation.unwrap();
            let net = rad.net_radiation();
            assert!(
                net.abs() < 1e-6,
                "Building {} net radiation {} should be zero",
                i,
                net
            );
        }
    }

    #[test]
    fn test_net_radiation_for_enclosed_surfaces() {
        let test = EnergyConservationTest::create_5_building_config();
        let total_net = test.net_radiation_for_enclosed_surfaces();
        assert!(
            total_net.abs() < 1e-6,
            "Total net radiation {} should be zero for enclosed surfaces",
            total_net
        );
    }
}
