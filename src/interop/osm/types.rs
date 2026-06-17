// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio Model (OSM) data structures for representing building geometry,
//! materials, constructions, and thermal zones.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OsmDocument {
    pub version: String,
    pub objects: Vec<OsmObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmObject {
    pub object_type: String,
    pub handle: Option<String>,
    pub fields: HashMap<String, String>,
    pub comments: Vec<String>,
}

impl OsmObject {
    pub fn new(object_type: impl Into<String>) -> Self {
        OsmObject {
            object_type: object_type.into(),
            handle: None,
            fields: HashMap::new(),
            comments: Vec::new(),
        }
    }

    pub fn get_field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    pub fn get_required_field(&self, key: &str) -> Result<&str, crate::interop::osm::error::OsmError> {
        self.fields.get(key).map(|s| s.as_str())
            .ok_or_else(|| crate::interop::osm::error::OsmError::missing_field(&self.object_type, key))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildingStory {
    pub handle: String,
    pub name: String,
    pub level: Option<f64>,
    pub height: Option<f64>,
    pub spaces: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Space {
    pub handle: String,
    pub name: String,
    pub zone_handle: Option<String>,
    pub story_handle: Option<String>,
    pub area: Option<f64>,
    pub volume: Option<f64>,
    pub surfaces: Vec<Surface>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Surface {
    pub handle: String,
    pub name: String,
    pub surface_type: String,
    pub construction_handle: Option<String>,
    pub building_boundary: Option<String>,
    pub outside_boundary_condition: Option<String>,
    pub sun_exposure: Option<String>,
    pub wind_exposure: Option<String>,
    pub area: Option<f64>,
    pub vertices: Vec<Vertex>,
    pub adjacent_space_handle: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubSurface {
    pub handle: String,
    pub name: String,
    pub surface_handle: String,
    pub construction_handle: Option<String>,
    pub window_type: Option<String>,
    pub area: Option<f64>,
    pub vertices: Vec<Vertex>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Construction {
    pub handle: String,
    pub name: String,
    pub layer_handles: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Layer {
    pub handle: String,
    pub name: String,
    pub material_handles: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Material {
    pub handle: String,
    pub name: String,
    pub roughness: Option<String>,
    pub thickness: Option<f64>,
    pub conductivity: Option<f64>,
    pub density: Option<f64>,
    pub specific_heat: Option<f64>,
    pub emissivity: Option<f64>,
    pub absorptance: Option<f64>,
    pub vapor_transmission: Option<f64>,
}

impl Material {
    pub fn to_construction_layer(&self) -> Option<crate::sim::construction::ConstructionLayer> {
        let conductivity = self.conductivity?;
        let density = self.density?;
        let specific_heat = self.specific_heat?;
        let thickness = self.thickness?;

        Some(crate::sim::construction::ConstructionLayer::new(
            self.name.clone(),
            conductivity,
            density,
            specific_heat,
            thickness,
        ))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermalZone {
    pub handle: String,
    pub name: String,
    pub thermostat_handle: Option<String>,
    pub multiplier: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildingUnit {
    pub handle: String,
    pub name: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Site {
    pub handle: String,
    pub name: String,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub elevation: Option<f64>,
    pub time_zone: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Building {
    pub handle: String,
    pub name: String,
    pub building_story_handles: Vec<String>,
    pub zone_handles: Vec<String>,
    pub area: Option<f64>,
    pub number_of_floors: Option<i32>,
    pub floor_height: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schedule {
    pub handle: String,
    pub name: String,
    pub schedule_type: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Thermostat {
    pub handle: String,
    pub name: String,
    pub heating_setpoint: Option<f64>,
    pub cooling_setpoint: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct People {
    pub handle: String,
    pub name: String,
    pub zone_handle: Option<String>,
    pub number_of_people: Option<f64>,
    pub people_per_area: Option<f64>,
    pub fraction_radiant: Option<f64>,
    pub schedule_handle: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lights {
    pub handle: String,
    pub name: String,
    pub zone_handle: Option<String>,
    pub watts_per_zone_floor_area: Option<f64>,
    pub fraction_radiant: Option<f64>,
    pub fraction_visible: Option<f64>,
    pub schedule_handle: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElectricEquipment {
    pub handle: String,
    pub name: String,
    pub zone_handle: Option<String>,
    pub watts_per_zone_floor_area: Option<f64>,
    pub fraction_radiant: Option<f64>,
    pub schedule_handle: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZoneHVACEquipment {
    pub handle: String,
    pub name: String,
    pub zone_handle: String,
    pub equipment: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Zone {
    pub handle: String,
    pub name: String,
    pub zone_handle: Option<String>,
    pub multiplier: Option<f64>,
    pub volume: Option<f64>,
}

pub struct OsmContext {
    pub materials: HashMap<String, Material>,
    pub layers: HashMap<String, Layer>,
    pub constructions: HashMap<String, Construction>,
    pub thermal_zones: HashMap<String, ThermalZone>,
    pub spaces: HashMap<String, Space>,
    pub building_stories: HashMap<String, BuildingStory>,
    pub surfaces: HashMap<String, Surface>,
    pub sub_surfaces: HashMap<String, SubSurface>,
    pub schedules: HashMap<String, Schedule>,
    pub thermostats: HashMap<String, Thermostat>,
    pub people: HashMap<String, People>,
    pub lights: HashMap<String, Lights>,
    pub electric_equipment: HashMap<String, ElectricEquipment>,
    pub site: Option<Site>,
    pub building: Option<Building>,
}

impl Default for OsmContext {
    fn default() -> Self {
        OsmContext {
            materials: HashMap::new(),
            layers: HashMap::new(),
            constructions: HashMap::new(),
            thermal_zones: HashMap::new(),
            spaces: HashMap::new(),
            building_stories: HashMap::new(),
            surfaces: HashMap::new(),
            sub_surfaces: HashMap::new(),
            schedules: HashMap::new(),
            thermostats: HashMap::new(),
            people: HashMap::new(),
            lights: HashMap::new(),
            electric_equipment: HashMap::new(),
            site: None,
            building: None,
        }
    }
}
