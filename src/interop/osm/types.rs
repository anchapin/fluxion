// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio OSM data types.
//!
//! These types represent the OpenStudio schema objects that are read from
//! or written to OSM files. They provide a mapping layer between the
//! OpenStudio XML representation and Fluxion's internal types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmBuilding {
    pub name: String,
    pub north_axis: f64,
    pub terrain: String,
    pub floorspaces_stories: Option<i32>,
    pub floor_area: Option<f64>,
    pub building_type: Option<String>,
}

impl Default for OsmBuilding {
    fn default() -> Self {
        OsmBuilding {
            name: "Building".to_string(),
            north_axis: 0.0,
            terrain: "Suburbs".to_string(),
            floorspaces_stories: None,
            floor_area: None,
            building_type: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmThermalZone {
    pub name: String,
    pub zone_name: Option<String>,
    pub multiplier: Option<i32>,
    pub volume: Option<f64>,
    pub floor_area: Option<f64>,
}

impl Default for OsmThermalZone {
    fn default() -> Self {
        OsmThermalZone {
            name: "Thermal Zone 1".to_string(),
            zone_name: None,
            multiplier: Some(1),
            volume: None,
            floor_area: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSpace {
    pub name: String,
    pub thermal_zone: Option<String>,
    pub building_story: Option<String>,
    pub x_position: Option<f64>,
    pub y_position: Option<f64>,
    pub z_position: Option<f64>,
    pub direction_of_relative_north: Option<f64>,
    pub building_unit: Option<String>,
}

impl Default for OsmSpace {
    fn default() -> Self {
        OsmSpace {
            name: "Space 1".to_string(),
            thermal_zone: None,
            building_story: None,
            x_position: Some(0.0),
            y_position: Some(0.0),
            z_position: Some(0.0),
            direction_of_relative_north: Some(0.0),
            building_unit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSurface {
    pub name: String,
    pub surface_type: String,
    pub construction: Option<String>,
    pub space: String,
    pub outside_boundary_condition: String,
    pub sun_exposure: Option<String>,
    pub wind_exposure: Option<String>,
    pub vertices: Vec<OsmVertex>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmVertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSubSurface {
    pub name: String,
    pub surface: String,
    pub surface_type: String,
    pub construction: Option<String>,
    pub vertices: Vec<OsmVertex>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmMaterial {
    pub name: String,
    pub material_type: Option<String>,
    pub thickness: Option<f64>,
    pub conductivity: Option<f64>,
    pub density: Option<f64>,
    pub specific_heat: Option<f64>,
    pub roughness: Option<String>,
    pub thermal_absorptance: Option<f64>,
    pub solar_absorptance: Option<f64>,
    pub visible_absorptance: Option<f64>,
}

impl Default for OsmMaterial {
    fn default() -> Self {
        OsmMaterial {
            name: "Material".to_string(),
            material_type: Some("StandardOpaqueMaterial".to_string()),
            thickness: Some(0.1),
            conductivity: Some(1.0),
            density: Some(1000.0),
            specific_heat: Some(1000.0),
            roughness: Some("MediumRough".to_string()),
            thermal_absorptance: Some(0.9),
            solar_absorptance: Some(0.7),
            visible_absorptance: Some(0.7),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmConstruction {
    pub name: String,
    pub layers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmLayer {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSchedule {
    pub name: String,
    pub schedule_type: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSite {
    pub name: String,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub time_zone: Option<f64>,
    pub elevation: Option<f64>,
    pub terrain: Option<String>,
}

impl Default for OsmSite {
    fn default() -> Self {
        OsmSite {
            name: "Site".to_string(),
            latitude: Some(39.7392),
            longitude: Some(-104.9903),
            time_zone: Some(-7.0),
            elevation: Some(1609.0),
            terrain: Some("Suburbs".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmWeatherFile {
    pub file_name: String,
    pub path_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmThermostat {
    pub name: String,
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
}

impl Default for OsmThermostat {
    fn default() -> Self {
        OsmThermostat {
            name: "Thermostat".to_string(),
            heating_setpoint: 20.0,
            cooling_setpoint: 24.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OsmModel {
    pub version: String,
    pub building: Option<OsmBuilding>,
    pub site: Option<OsmSite>,
    pub weather_file: Option<OsmWeatherFile>,
    pub thermal_zones: Vec<OsmThermalZone>,
    pub spaces: Vec<OsmSpace>,
    pub surfaces: Vec<OsmSurface>,
    pub sub_surfaces: Vec<OsmSubSurface>,
    pub materials: Vec<OsmMaterial>,
    pub constructions: Vec<OsmConstruction>,
    pub schedules: Vec<OsmSchedule>,
    pub thermostats: Vec<OsmThermostat>,
}
