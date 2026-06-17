// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio OSM file writer.
//!
//! This module provides functionality to serialize Fluxion's SimulationSchema
//! into OpenStudio Model (OSM) XML files.

use std::path::Path;

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::Writer;
use std::io::Cursor;

use crate::api::schema::{
    ConstructionSet, ControlSet, Geometry, SimulationSchema,
    WeatherData,
};
use crate::sim::schedule::HVACSchedule;

use super::error::OsmError;
use crate::interop::osm::types::{
    OsmBuilding, OsmConstruction, OsmMaterial, OsmModel,
    OsmSite, OsmSpace, OsmThermostat, OsmThermalZone,
    OsmWeatherFile,
};

pub struct OsmWriter {
    indent: bool,
}

impl OsmWriter {
    pub fn new() -> Self {
        OsmWriter { indent: true }
    }

    pub fn write(&self, schema: &SimulationSchema, path: &Path) -> Result<(), OsmError> {
        let xml = self.to_string(schema)?;
        std::fs::write(path, xml)?;
        Ok(())
    }

    pub fn to_string(&self, schema: &SimulationSchema) -> Result<String, OsmError> {
        let model = self.schema_to_osm_model(schema)?;
        self.model_to_xml(&model)
    }

    fn schema_to_osm_model(&self, schema: &SimulationSchema) -> Result<OsmModel, OsmError> {
        let (geometry, constructions, controls, weather) = match schema {
            SimulationSchema::V1(s) => (
                &s.geometry,
                &s.constructions,
                &s.controls,
                &s.weather,
            ),
        };

        let mut model = OsmModel::default();
        model.version = "1.0.0".to_string();

        model.building = Some(self.geometry_to_building(geometry));
        model.site = Some(self.geometry_to_site());
        model.weather_file = self.extract_weather_file(weather);
        model.thermal_zones = self.geometry_to_thermal_zones(geometry);
        model.spaces = self.geometry_to_spaces(geometry);
        model.materials = self.constructions_to_materials(constructions);
        model.constructions = self.constructions_to_constructions(constructions)?;
        model.thermostats = vec![self.controls_to_thermostat(controls)];

        Ok(model)
    }

    fn geometry_to_building(&self, geometry: &Geometry) -> OsmBuilding {
        OsmBuilding {
            name: "Building".to_string(),
            north_axis: 0.0,
            terrain: "Suburbs".to_string(),
            floorspaces_stories: Some(geometry.number_of_floors as i32),
            floor_area: Some(geometry.total_floor_area),
            building_type: Some("Office".to_string()),
        }
    }

    fn geometry_to_site(&self) -> OsmSite {
        OsmSite {
            name: "Site".to_string(),
            latitude: Some(39.7392),
            longitude: Some(-104.9903),
            time_zone: Some(-7.0),
            elevation: Some(1609.0),
            terrain: Some("Suburbs".to_string()),
        }
    }

    fn extract_weather_file(&self, weather: &WeatherData) -> Option<OsmWeatherFile> {
        match weather {
            WeatherData::EpwFile { path } => Some(OsmWeatherFile {
                file_name: path.to_string_lossy().to_string(),
                path_type: Some("absolute".to_string()),
            }),
            _ => None,
        }
    }

    fn geometry_to_thermal_zones(&self, geometry: &Geometry) -> Vec<OsmThermalZone> {
        geometry
            .zones
            .iter()
            .map(|z| OsmThermalZone {
                name: z.name.clone(),
                zone_name: Some(z.name.clone()),
                multiplier: Some(1),
                volume: Some(z.volume),
                floor_area: Some(z.floor_area),
            })
            .collect()
    }

    fn geometry_to_spaces(&self, geometry: &Geometry) -> Vec<OsmSpace> {
        geometry
            .zones
            .iter()
            .map(|z| OsmSpace {
                name: format!("{} Space", z.name),
                thermal_zone: Some(z.name.clone()),
                building_story: None,
                x_position: Some(0.0),
                y_position: Some(0.0),
                z_position: Some(0.0),
                direction_of_relative_north: Some(0.0),
                building_unit: None,
            })
            .collect()
    }

    fn constructions_to_materials(&self, constructions: &ConstructionSet) -> Vec<OsmMaterial> {
        let mut materials = Vec::new();

        for (idx, layer) in constructions.wall.layers.iter().enumerate() {
            materials.push(OsmMaterial {
                name: format!("Wall Material {}", idx),
                material_type: Some("StandardOpaqueMaterial".to_string()),
                thickness: Some(layer.thickness),
                conductivity: Some(layer.conductivity),
                density: Some(layer.density),
                specific_heat: Some(layer.specific_heat),
                roughness: Some("MediumRough".to_string()),
                thermal_absorptance: Some(0.9),
                solar_absorptance: Some(0.7),
                visible_absorptance: Some(0.7),
            });
        }

        if constructions.wall.window.is_some() {
            materials.push(OsmMaterial {
                name: "Window Material".to_string(),
                material_type: Some("StandardOpaqueMaterial".to_string()),
                thickness: Some(0.006),
                conductivity: Some(2.5 * 0.006),
                density: Some(2500.0),
                specific_heat: Some(840.0),
                roughness: None,
                thermal_absorptance: None,
                solar_absorptance: None,
                visible_absorptance: None,
            });
        }

        for (idx, layer) in constructions.roof.layers.iter().enumerate() {
            materials.push(OsmMaterial {
                name: format!("Roof Material {}", idx),
                material_type: Some("StandardOpaqueMaterial".to_string()),
                thickness: Some(layer.thickness),
                conductivity: Some(layer.conductivity),
                density: Some(layer.density),
                specific_heat: Some(layer.specific_heat),
                roughness: Some("MediumRough".to_string()),
                thermal_absorptance: Some(0.9),
                solar_absorptance: Some(0.7),
                visible_absorptance: Some(0.7),
            });
        }

        for (idx, layer) in constructions.floor.layers.iter().enumerate() {
            materials.push(OsmMaterial {
                name: format!("Floor Material {}", idx),
                material_type: Some("StandardOpaqueMaterial".to_string()),
                thickness: Some(layer.thickness),
                conductivity: Some(layer.conductivity),
                density: Some(layer.density),
                specific_heat: Some(layer.specific_heat),
                roughness: Some("MediumRough".to_string()),
                thermal_absorptance: Some(0.9),
                solar_absorptance: Some(0.7),
                visible_absorptance: Some(0.7),
            });
        }

        materials
    }

    fn constructions_to_constructions(
        &self,
        constructions: &ConstructionSet,
    ) -> Result<Vec<OsmConstruction>, OsmError> {
        let mut result = Vec::new();

        let wall_layer_names: Vec<String> = constructions
            .wall
            .layers
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("Wall Material {}", idx))
            .collect();

        result.push(OsmConstruction {
            name: constructions.wall.name.clone(),
            layers: wall_layer_names.clone(),
        });

        if constructions.wall.window.is_some() {
            let mut win_layers = wall_layer_names.clone();
            win_layers.push("Window Material".to_string());
            result.push(OsmConstruction {
                name: format!("{} with Windows", constructions.wall.name),
                layers: win_layers,
            });
        }

        let roof_layer_names: Vec<String> = constructions
            .roof
            .layers
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("Roof Material {}", idx))
            .collect();
        result.push(OsmConstruction {
            name: constructions.roof.name.clone(),
            layers: roof_layer_names,
        });

        let floor_layer_names: Vec<String> = constructions
            .floor
            .layers
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("Floor Material {}", idx))
            .collect();
        result.push(OsmConstruction {
            name: constructions.floor.name.clone(),
            layers: floor_layer_names,
        });

        Ok(result)
    }

    fn controls_to_thermostat(&self, controls: &ControlSet) -> OsmThermostat {
        OsmThermostat {
            name: "Thermostat".to_string(),
            heating_setpoint: controls.zone_control.heating_setpoint,
            cooling_setpoint: controls.zone_control.cooling_setpoint,
        }
    }

    fn model_to_xml(&self, model: &OsmModel) -> Result<String, OsmError> {
        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        writer.write_event(Event::Decl(quick_xml::events::BytesDecl::new(
            "1.0",
            Some("UTF-8"),
            None,
        )))?;

        let mut root = BytesStart::new("OpenStudioApplication");
        root.push_attribute(("version", "1.0.0"));
        root.push_attribute(("schemaVersion", "1.0.0"));
        writer.write_event(Event::Start(root.clone()))?;

        let version_elem = BytesStart::new("OS:Version");
        let mut version_elem = version_elem;
        version_elem.push_attribute(("version", model.version.as_str()));
        writer.write_event(Event::Empty(version_elem))?;

        if let Some(ref site) = model.site {
            self.write_site(&mut writer, site)?;
        }

        if let Some(ref weather) = model.weather_file {
            self.write_weather_file(&mut writer, weather)?;
        }

        if let Some(ref building) = model.building {
            self.write_building(&mut writer, building)?;
        }

        for zone in &model.thermal_zones {
            self.write_thermal_zone(&mut writer, zone)?;
        }

        for space in &model.spaces {
            self.write_space(&mut writer, space)?;
        }

        for material in &model.materials {
            self.write_material(&mut writer, material)?;
        }

        for construction in &model.constructions {
            self.write_construction(&mut writer, construction)?;
        }

        for thermostat in &model.thermostats {
            self.write_thermostat(&mut writer, thermostat)?;
        }

        writer.write_event(Event::End(BytesEnd::new("OpenStudioApplication")))?;

        let result = writer.into_inner().into_inner();
        let xml = String::from_utf8(result).map_err(|e| OsmError::Parse(e.to_string()))?;

        Ok(xml)
    }

    fn write_site(&self, writer: &mut Writer<Cursor<Vec<u8>>>, site: &OsmSite) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:Site");
        elem.push_attribute(("name", site.name.as_str()));
        if let Some(lat) = site.latitude {
            elem.push_attribute(("latitude", format!("{:.6}", lat).as_str()));
        }
        if let Some(lon) = site.longitude {
            elem.push_attribute(("longitude", format!("{:.6}", lon).as_str()));
        }
        if let Some(tz) = site.time_zone {
            elem.push_attribute(("time_Zone", format!("{:.1}", tz).as_str()));
        }
        if let Some(elev) = site.elevation {
            elem.push_attribute(("elevation", format!("{:.1}", elev).as_str()));
        }
        if let Some(ref terrain) = site.terrain {
            elem.push_attribute(("terrain", terrain.as_str()));
        }
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }

    fn write_weather_file(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        weather: &OsmWeatherFile,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:WeatherFile");
        elem.push_attribute(("file_Name", weather.file_name.as_str()));
        if let Some(ref path_type) = weather.path_type {
            elem.push_attribute(("path_Type", path_type.as_str()));
        }
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }

    fn write_building(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        building: &OsmBuilding,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:Building");
        elem.push_attribute(("name", building.name.as_str()));
        elem.push_attribute(("north_Axis", format!("{:.1}", building.north_axis).as_str()));
        elem.push_attribute(("terrain", building.terrain.as_str()));
        if let Some(stories) = building.floorspaces_stories {
            elem.push_attribute(("floorspaces_Story", stories.to_string().as_str()));
        }
        if let Some(area) = building.floor_area {
            elem.push_attribute(("floor_Area", format!("{:.2}", area).as_str()));
        }
        if let Some(ref btype) = building.building_type {
            elem.push_attribute(("buildingType", btype.as_str()));
        }
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }

    fn write_thermal_zone(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        zone: &OsmThermalZone,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:ThermalZone");
        elem.push_attribute(("name", zone.name.as_str()));
        if let Some(ref zone_name) = zone.zone_name {
            elem.push_attribute(("zone_Name", zone_name.as_str()));
        }
        if let Some(mult) = zone.multiplier {
            elem.push_attribute(("multiplier", mult.to_string().as_str()));
        }
        if let Some(vol) = zone.volume {
            elem.push_attribute(("volume", format!("{:.2}", vol).as_str()));
        }
        if let Some(area) = zone.floor_area {
            elem.push_attribute(("floor_Area", format!("{:.2}", area).as_str()));
        }
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }

    fn write_space(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        space: &OsmSpace,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:Space");
        elem.push_attribute(("name", space.name.as_str()));
        if let Some(ref tz) = space.thermal_zone {
            elem.push_attribute(("thermal_Zone", tz.as_str()));
        }
        if let Some(x) = space.x_position {
            elem.push_attribute(("x_Position", format!("{:.3}", x).as_str()));
        }
        if let Some(y) = space.y_position {
            elem.push_attribute(("y_Position", format!("{:.3}", y).as_str()));
        }
        if let Some(z) = space.z_position {
            elem.push_attribute(("z_Position", format!("{:.3}", z).as_str()));
        }
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }

    fn write_material(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        material: &OsmMaterial,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:Material");
        elem.push_attribute(("name", material.name.as_str()));
        if let Some(ref mtype) = material.material_type {
            elem.push_attribute(("material_Type", mtype.as_str()));
        }
        if let Some(thick) = material.thickness {
            elem.push_attribute(("thickness", format!("{:.4}", thick).as_str()));
        }
        if let Some(cond) = material.conductivity {
            elem.push_attribute(("conductivity", format!("{:.4}", cond).as_str()));
        }
        if let Some(dens) = material.density {
            elem.push_attribute(("density", format!("{:.2}", dens).as_str()));
        }
        if let Some(sh) = material.specific_heat {
            elem.push_attribute(("specific_Heat", format!("{:.2}", sh).as_str()));
        }
        if let Some(ref rough) = material.roughness {
            elem.push_attribute(("roughness", rough.as_str()));
        }
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }

    fn write_construction(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        construction: &OsmConstruction,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:Construction");
        elem.push_attribute(("name", construction.name.as_str()));
        writer.write_event(Event::Start(elem.clone()))?;

        for layer_name in &construction.layers {
            let mut layer_elem = BytesStart::new("OS:Layer");
            layer_elem.push_attribute(("name", layer_name.as_str()));
            writer.write_event(Event::Empty(layer_elem))?;
        }

        writer.write_event(Event::End(BytesEnd::new("OS:Construction")))?;
        Ok(())
    }

    fn write_thermostat(
        &self,
        writer: &mut Writer<Cursor<Vec<u8>>>,
        thermostat: &OsmThermostat,
    ) -> Result<(), OsmError> {
        let mut elem = BytesStart::new("OS:ThermostatSetpointDualSetpoint");
        elem.push_attribute(("name", thermostat.name.as_str()));
        elem.push_attribute((
            "heating_Setpoint_Temperature",
            format!("{:.1}", thermostat.heating_setpoint).as_str(),
        ));
        elem.push_attribute((
            "cooling_Setpoint_Temperature",
            format!("{:.1}", thermostat.cooling_setpoint).as_str(),
        ));
        writer.write_event(Event::Empty(elem))?;
        Ok(())
    }
}

impl Default for OsmWriter {
    fn default() -> Self {
        Self::new()
    }
}
