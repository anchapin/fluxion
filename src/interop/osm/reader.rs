// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio OSM file reader.
//!
//! This module provides functionality to parse OSM (OpenStudio Model) XML files
//! and convert them into Fluxion's SimulationSchema.

use std::path::Path;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use crate::api::schema::{
    ConstructionSet, ControlConfig, ControlSet, Geometry,
    SchemaMetadata, SchemaVersion, SimulationSchema, SimulationSchemaV1,
    SurfaceConstruction, WeatherData, WindowSpec, ZoneGeometry,
};
use crate::sim::construction::ConstructionLayer;
use crate::sim::schedule::{DailySchedule, HVACSchedule};

use super::error::OsmError;
use crate::interop::osm::types::{
    OsmBuilding, OsmConstruction, OsmMaterial, OsmModel, OsmSchedule, OsmSite,
    OsmSpace, OsmSubSurface, OsmSurface, OsmThermostat, OsmThermalZone,
    OsmVertex, OsmWeatherFile,
};

pub struct OsmReader {
    model: OsmModel,
}

impl OsmReader {
    pub fn new() -> Self {
        OsmReader {
            model: OsmModel::default(),
        }
    }

    pub fn from_path(&mut self, path: &Path) -> Result<SimulationSchema, OsmError> {
        let content = std::fs::read_to_string(path)?;
        self.from_str(&content)
    }

    pub fn from_str(&mut self, xml: &str) -> Result<SimulationSchema, OsmError> {
        self.model = OsmModel::default();
        self.parse_xml(xml)?;
        self.to_schema()
    }

    fn parse_xml(&mut self, xml: &str) -> Result<(), OsmError> {
        let mut reader = Reader::from_str(xml);
        reader.trim_text(true);

        let mut buf = Vec::new();
        let mut current_element = String::new();
        let mut element_stack: Vec<String> = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    current_element = name.clone();
                    element_stack.push(name.clone());

                    self.process_element_start(&name, &e)?;
                }
                Ok(Event::Empty(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    self.process_element_start(&name, &e)?;
                }
                Ok(Event::End(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    element_stack.pop();
                    current_element = element_stack
                        .last()
                        .cloned()
                        .unwrap_or_default();
                }
                Ok(Event::Text(e)) => {
                    let text = e.unescape().unwrap_or_default().to_string();
                    if !text.trim().is_empty() {
                        self.process_text(&current_element, &text);
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(OsmError::Parse(e.to_string())),
                _ => {}
            }
            buf.clear();
        }

        Ok(())
    }

    fn process_element_start(&mut self, name: &str, e: &BytesStart) -> Result<(), OsmError> {
        match name {
            "OS:Version" => {
                if let Some(v) = self.get_attr(e, "version") {
                    self.model.version = v;
                }
            }
            "OS:Building" => {
                let building = OsmBuilding {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Building".to_string()),
                    north_axis: self
                        .get_attr(e, "north_Axis")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    terrain: self.get_attr(e, "terrain").unwrap_or_else(|| "Suburbs".to_string()),
                    floorspaces_stories: self
                        .get_attr(e, "floorspaces_Story")
                        .and_then(|v| v.parse().ok()),
                    floor_area: self.get_attr(e, "floor_Area").and_then(|v| v.parse().ok()),
                    building_type: self.get_attr(e, "buildingType"),
                };
                self.model.building = Some(building);
            }
            "OS:Site" => {
                let site = OsmSite {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Site".to_string()),
                    latitude: self.get_attr(e, "latitude").and_then(|v| v.parse().ok()),
                    longitude: self.get_attr(e, "longitude").and_then(|v| v.parse().ok()),
                    time_zone: self.get_attr(e, "time_Zone").and_then(|v| v.parse().ok()),
                    elevation: self.get_attr(e, "elevation").and_then(|v| v.parse().ok()),
                    terrain: self.get_attr(e, "terrain"),
                };
                self.model.site = Some(site);
            }
            "OS:WeatherFile" => {
                let weather_file = OsmWeatherFile {
                    file_name: self
                        .get_attr(e, "file_Name")
                        .unwrap_or_else(|| String::new()),
                    path_type: self.get_attr(e, "path_Type"),
                };
                self.model.weather_file = Some(weather_file);
            }
            "OS:ThermalZone" => {
                let zone = OsmThermalZone {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Thermal Zone".to_string()),
                    zone_name: self.get_attr(e, "zone_Name"),
                    multiplier: self
                        .get_attr(e, "multiplier")
                        .and_then(|v| v.parse().ok()),
                    volume: self.get_attr(e, "volume").and_then(|v| v.parse().ok()),
                    floor_area: self
                        .get_attr(e, "floor_Area")
                        .and_then(|v| v.parse().ok()),
                };
                self.model.thermal_zones.push(zone);
            }
            "OS:Space" => {
                let space = OsmSpace {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Space".to_string()),
                    thermal_zone: self.get_attr(e, "thermal_Zone"),
                    building_story: self.get_attr(e, "building_Story"),
                    x_position: self.get_attr(e, "x_Position").and_then(|v| v.parse().ok()),
                    y_position: self.get_attr(e, "y_Position").and_then(|v| v.parse().ok()),
                    z_position: self.get_attr(e, "z_Position").and_then(|v| v.parse().ok()),
                    direction_of_relative_north: self
                        .get_attr(e, "direction_of_Relative_North")
                        .and_then(|v| v.parse().ok()),
                    building_unit: self.get_attr(e, "building_Unit"),
                };
                self.model.spaces.push(space);
            }
            "OS:Surface" => {
                let surface = OsmSurface {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Surface".to_string()),
                    surface_type: self
                        .get_attr(e, "surface_Type")
                        .unwrap_or_else(|| "Wall".to_string()),
                    construction: self.get_attr(e, "construction_Name"),
                    space: self.get_attr(e, "space_Name").unwrap_or_else(|| String::new()),
                    outside_boundary_condition: self
                        .get_attr(e, "outside_Boundary_Condition")
                        .unwrap_or_else(|| "Outdoors".to_string()),
                    sun_exposure: self.get_attr(e, "sun_Exposure"),
                    wind_exposure: self.get_attr(e, "wind_Exposure"),
                    vertices: Vec::new(),
                };
                self.model.surfaces.push(surface);
            }
            "OS:SubSurface" => {
                let sub_surface = OsmSubSurface {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "SubSurface".to_string()),
                    surface: self.get_attr(e, "surface_Name").unwrap_or_else(|| String::new()),
                    surface_type: self
                        .get_attr(e, "subSurface_Type")
                        .unwrap_or_else(|| "FixedWindow".to_string()),
                    construction: self.get_attr(e, "construction_Name"),
                    vertices: Vec::new(),
                };
                self.model.sub_surfaces.push(sub_surface);
            }
            "OS:Vertex" => {
                let vertex = OsmVertex {
                    x: self.get_attr(e, "x").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    y: self.get_attr(e, "y").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    z: self.get_attr(e, "z").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                };
                if let Some(last_surface) = self.model.surfaces.last_mut() {
                    last_surface.vertices.push(vertex);
                } else if let Some(last_sub) = self.model.sub_surfaces.last_mut() {
                    last_sub.vertices.push(vertex);
                }
            }
            "OS:Material" => {
                let material = OsmMaterial {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Material".to_string()),
                    material_type: self.get_attr(e, "material_Type"),
                    thickness: self.get_attr(e, "thickness").and_then(|v| v.parse().ok()),
                    conductivity: self
                        .get_attr(e, "conductivity")
                        .and_then(|v| v.parse().ok()),
                    density: self.get_attr(e, "density").and_then(|v| v.parse().ok()),
                    specific_heat: self
                        .get_attr(e, "specific_Heat")
                        .and_then(|v| v.parse().ok()),
                    roughness: self.get_attr(e, "roughness"),
                    thermal_absorptance: self
                        .get_attr(e, "thermal_Absorptance")
                        .and_then(|v| v.parse().ok()),
                    solar_absorptance: self
                        .get_attr(e, "solar_Absorptance")
                        .and_then(|v| v.parse().ok()),
                    visible_absorptance: self
                        .get_attr(e, "visible_Absorptance")
                        .and_then(|v| v.parse().ok()),
                };
                self.model.materials.push(material);
            }
            "OS:Construction" => {
                let construction = OsmConstruction {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Construction".to_string()),
                    layers: Vec::new(),
                };
                self.model.constructions.push(construction);
            }
            "OS:Layer" => {}
            "OS:Schedule:Constant" | "OS:Schedule:Ruleset" => {
                let schedule = OsmSchedule {
                    name: self.get_attr(e, "name").unwrap_or_else(|| "Schedule".to_string()),
                    schedule_type: name.replace("OS:", ""),
                    values: Vec::new(),
                };
                self.model.schedules.push(schedule);
            }
            "OS:ThermostatSetpointDualSetpoint" => {
                let thermostat = OsmThermostat {
                    name: self
                        .get_attr(e, "name")
                        .unwrap_or_else(|| "Thermostat".to_string()),
                    heating_setpoint: self
                        .get_attr(e, "heating_Setpoint_Temperature")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(20.0),
                    cooling_setpoint: self
                        .get_attr(e, "cooling_Setpoint_Temperature")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(24.0),
                };
                self.model.thermostats.push(thermostat);
            }
            _ => {}
        }
        Ok(())
    }

    fn get_attr<'a>(&self, e: &'a BytesStart, attr: &str) -> Option<String> {
        e.attributes()
            .filter_map(|a| a.ok())
            .find(|a| a.key.as_ref() == attr.as_bytes())
            .map(|a| String::from_utf8_lossy(&a.value).to_string())
    }

    fn process_text(&mut self, _element: &str, _text: &str) {
        // Text content processing is handled via attributes in OSM
    }

    fn to_schema(&self) -> Result<SimulationSchema, OsmError> {
        let zone_geometries = self.extract_zones()?;
        let geometry = self.extract_geometry(&zone_geometries)?;
        let constructions = self.extract_constructions()?;
        let schedules = self.extract_schedules()?;
        let weather = self.extract_weather()?;
        let controls = self.extract_controls()?;

        let schema = SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata {
                name: self
                    .model
                    .building
                    .as_ref()
                    .map(|b| b.name.clone())
                    .unwrap_or_else(|| "OpenStudio Model".to_string()),
                description: format!(
                    "Imported from OpenStudio OSM (version: {})",
                    self.model.version
                ),
                author: None,
                created_at: Some(chrono::Utc::now().format("%Y-%m-%d").to_string()),
                schema_version: SchemaVersion::V1,
            },
            geometry,
            constructions,
            schedules,
            weather,
            controls,
            output: Default::default(),
        };

        Ok(SimulationSchema::V1(schema))
    }

    fn extract_zones(&self) -> Result<Vec<ZoneGeometry>, OsmError> {
        let mut zones = Vec::new();

        for tz in &self.model.thermal_zones {
            let zone = ZoneGeometry {
                name: tz.name.clone(),
                floor_area: tz.floor_area.unwrap_or(48.0),
                volume: tz.volume.unwrap_or(129.6),
                height: 2.7,
            };
            zones.push(zone);
        }

        if zones.is_empty() {
            zones.push(ZoneGeometry::default());
        }

        Ok(zones)
    }

    fn extract_geometry(&self, zones: &[ZoneGeometry]) -> Result<Geometry, OsmError> {
        let total_floor_area: f64 = zones.iter().map(|z| z.floor_area).sum();
        let total_volume: f64 = zones.iter().map(|z| z.volume).sum();
        let number_of_floors = self
            .model
            .building
            .as_ref()
            .and_then(|b| b.floorspaces_stories)
            .unwrap_or(1) as usize;
        let floor_height = if !zones.is_empty() {
            zones[0].volume / zones[0].floor_area
        } else {
            2.7
        };

        Ok(Geometry {
            zones: zones.to_vec(),
            total_floor_area,
            total_volume,
            number_of_floors,
            floor_height,
        })
    }

    fn extract_constructions(&self) -> Result<ConstructionSet, OsmError> {
        let wall = self.build_default_wall_construction()?;
        let roof = self.build_default_roof_construction()?;
        let floor = self.build_default_floor_construction()?;

        Ok(ConstructionSet {
            wall,
            roof,
            floor,
            interzone: None,
        })
    }

    fn build_default_wall_construction(&self) -> Result<SurfaceConstruction, OsmError> {
        let mut layers = Vec::new();

        if let Some(ref mat) = self.model.materials.first() {
            let thickness = mat.thickness.unwrap_or(0.1);
            let conductivity = mat.conductivity.unwrap_or(1.0);
            let density = mat.density.unwrap_or(1000.0);
            let specific_heat = mat.specific_heat.unwrap_or(1000.0);

            layers.push(ConstructionLayer::new(
                &mat.name,
                thickness,
                density,
                specific_heat,
                1.0 / conductivity,
            ));
        }

        if layers.is_empty() {
            layers.push(ConstructionLayer::new("Plasterboard", 0.016, 950.0, 840.0, 0.012));
            layers.push(ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066));
            layers.push(ConstructionLayer::new("Wood siding", 0.14, 500.0, 1300.0, 0.009));
        }

        Ok(SurfaceConstruction {
            name: "Default Wall".to_string(),
            layers,
            window: Some(WindowSpec::default()),
        })
    }

    fn build_default_roof_construction(&self) -> Result<SurfaceConstruction, OsmError> {
        let layers = vec![ConstructionLayer::new("Concrete", 0.2, 2300.0, 880.0, 1.4)];

        Ok(SurfaceConstruction {
            name: "Default Roof".to_string(),
            layers,
            window: None,
        })
    }

    fn build_default_floor_construction(&self) -> Result<SurfaceConstruction, OsmError> {
        let layers = vec![ConstructionLayer::new(
            "Concrete slab",
            0.15,
            2300.0,
            880.0,
            1.4,
        )];

        Ok(SurfaceConstruction {
            name: "Default Floor".to_string(),
            layers,
            window: None,
        })
    }

    fn extract_schedules(&self) -> Result<crate::api::schema::ScheduleSet, OsmError> {
        let occupancy = DailySchedule::weekly("Occupancy".to_string());
        let lighting = DailySchedule::weekly("Lighting".to_string());
        let hvac = HVACSchedule::constant_schedule(20.0, 24.0);

        Ok(crate::api::schema::ScheduleSet {
            occupancy,
            lighting,
            hvac,
            infiltration: None,
        })
    }

    fn extract_weather(&self) -> Result<WeatherData, OsmError> {
        if let Some(ref weather_file) = self.model.weather_file {
            if !weather_file.file_name.is_empty() {
                return Ok(WeatherData::EpwFile {
                    path: std::path::PathBuf::from(&weather_file.file_name),
                });
            }
        }

        if let Some(ref site) = self.model.site {
            let location = format!(
                "{},{}",
                site.latitude.unwrap_or(39.7392),
                site.longitude.unwrap_or(-104.9903)
            );
            return Ok(WeatherData::TmyLocation { location });
        }

        Ok(WeatherData::default())
    }

    fn extract_controls(&self) -> Result<ControlSet, OsmError> {
        let zone_control = if let Some(ref thermostat) = self.model.thermostats.first() {
            ControlConfig {
                heating_setpoint: thermostat.heating_setpoint,
                cooling_setpoint: thermostat.cooling_setpoint,
                deadband_tolerance: 0.5,
                heating_capacity: 100_000.0,
                cooling_capacity: 100_000.0,
            }
        } else {
            ControlConfig::default()
        };

        Ok(ControlSet {
            zone_control,
            global_control: None,
        })
    }
}

impl Default for OsmReader {
    fn default() -> Self {
        Self::new()
    }
}
