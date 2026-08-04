// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! gbXML writer - exports fluxion schema to gbXML files.
//!
//! This module provides functionality to convert fluxion's [`SimulationSchema`]
//! into gbXML format for export to BIM tools.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::gbxml::{export_gbxml, GbXmlWriter};
//!
//! let writer = GbXmlWriter::new();
//! writer.export_gbxml(&schema, "output.xml")?;
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

use crate::api::schema::{ConstructionSet, SimulationSchemaV1, WeatherData, ZoneGeometry};
use crate::interop::gbxml::error::GbXmlError;

/// Export a SimulationSchema to gbXML file.
pub fn export_gbxml(schema: &SimulationSchemaV1, path: impl AsRef<Path>) -> Result<(), GbXmlError> {
    let file = File::create(path.as_ref())
        .map_err(|e| GbXmlError::io_error(path.as_ref(), e.to_string()))?;
    let writer = BufWriter::new(file);

    let mut gbxml_writer = GbXmlWriter::new();
    gbxml_writer.write_schema(schema, writer)
}

/// GbXmlWriter for exporting to gbXML format.
#[allow(dead_code)]
pub struct GbXmlWriter {
    construction_counter: usize,
    layer_counter: usize,
    material_counter: usize,
    space_counter: usize,
    surface_counter: usize,
    schedule_counter: usize,
    zone_counter: usize,
}

impl GbXmlWriter {
    /// Create a new GbXmlWriter.
    pub fn new() -> Self {
        GbXmlWriter {
            construction_counter: 0,
            layer_counter: 0,
            material_counter: 0,
            space_counter: 0,
            surface_counter: 0,
            schedule_counter: 0,
            zone_counter: 0,
        }
    }

    /// Write a SimulationSchema to a gbXML writer.
    pub fn write_schema<W: std::io::Write>(
        &mut self,
        schema: &SimulationSchemaV1,
        output: W,
    ) -> Result<(), GbXmlError> {
        let mut writer = Writer::new_with_indent(output, b' ', 2);

        // Write gbXML header
        let mut root = BytesStart::new("gbXML");
        root.push_attribute(("xmlns", "http://www.gbxml.org/schema"));
        root.push_attribute(("version", "8.01"));
        writer.write_event(Event::Start(root))?;

        // Write Campus
        self.write_campus(schema, &mut writer)?;

        // Write Zones (thermal zone properties)
        self.write_zones(schema, &mut writer)?;

        // Write Schedules
        self.write_schedules(&schema.schedules, &mut writer)?;

        // Write Constructions, Layers, Materials
        self.write_constructions(&schema.constructions, &mut writer)?;

        writer.write_event(Event::End(BytesEnd::new("gbXML")))?;

        Ok(())
    }

    fn write_campus<W: std::io::Write>(
        &mut self,
        schema: &SimulationSchemaV1,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        // Campus element
        let mut campus = BytesStart::new("Campus");
        campus.push_attribute(("id", "campus1"));
        campus.push_attribute(("name", schema.metadata.name.as_str()));
        writer.write_event(Event::Start(campus))?;

        // Location
        self.write_location(schema, writer)?;

        // Building
        self.write_building(schema, writer)?;

        writer.write_event(Event::End(BytesEnd::new("Campus")))?;

        Ok(())
    }

    fn write_location<W: std::io::Write>(
        &mut self,
        schema: &SimulationSchemaV1,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        writer.write_event(Event::Start(BytesStart::new("Location")))?;

        // Get location name from weather or schema
        let location_name = match &schema.weather {
            WeatherData::TmyLocation { location } => location.clone(),
            WeatherData::EpwFile { path } => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string(),
            WeatherData::Inline { .. } => "Inline".to_string(),
        };

        write_text_element(writer, "Name", &location_name)?;
        write_text_element(writer, "Latitude", "39.739")?;
        write_text_element(writer, "Longitude", "-104.984")?;

        writer.write_event(Event::End(BytesEnd::new("Location")))?;

        Ok(())
    }

    fn write_building<W: std::io::Write>(
        &mut self,
        schema: &SimulationSchemaV1,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        let mut building = BytesStart::new("Building");
        building.push_attribute(("id", "building1"));
        building.push_attribute(("name", schema.metadata.name.as_str()));
        writer.write_event(Event::Start(building))?;

        // Write each zone as a BuildingStorey
        // Note: In gbXML, a Space is a thermal zone, but BuildingStorey is a floor
        // For simplicity, we put all zones on floor 0
        let mut storey = BytesStart::new("BuildingStorey");
        storey.push_attribute(("id", "storey1"));
        storey.push_attribute(("name", "Floor 1"));
        storey.push_attribute(("level", "0"));
        writer.write_event(Event::Start(storey))?;

        for (zone_idx, zone) in schema.geometry.zones.iter().enumerate() {
            self.write_space(zone, zone_idx, schema, writer)?;
        }

        writer.write_event(Event::End(BytesEnd::new("BuildingStorey")))?;

        writer.write_event(Event::End(BytesEnd::new("Building")))?;

        Ok(())
    }

    fn write_space<W: std::io::Write>(
        &mut self,
        zone: &ZoneGeometry,
        zone_idx: usize,
        _schema: &SimulationSchemaV1,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        self.space_counter += 1;
        let space_id = format!("space{}", self.space_counter);
        let zone_id = format!("zone{}", zone_idx + 1);

        let mut space = BytesStart::new("Space");
        space.push_attribute(("id", space_id.as_str()));
        space.push_attribute(("name", zone.name.as_str()));
        writer.write_event(Event::Start(space))?;

        write_text_element(writer, "Area", &zone.floor_area.to_string())?;
        write_text_element(writer, "Volume", &zone.volume.to_string())?;

        // ZoneIdRef - link to thermal zone properties
        let mut zone_ref = BytesStart::new("ZoneIdRef");
        zone_ref.push_attribute(("zoneIdRef", zone_id.as_str()));
        writer.write_event(Event::Empty(zone_ref))?;

        // Write surfaces with proper construction IDs based on surface type
        let surface_names = [
            "North Wall",
            "East Wall",
            "South Wall",
            "West Wall",
            "Roof",
            "Floor",
        ];
        let surface_types = [
            "ExteriorWall",
            "ExteriorWall",
            "ExteriorWall",
            "ExteriorWall",
            "Roof",
            "Floor",
        ];
        // Correct construction IDs based on surface type
        let construction_ids = [
            "construction_wall",
            "construction_wall",
            "construction_wall",
            "construction_wall",
            "construction_roof",
            "construction_floor",
        ];
        let areas = [
            zone.floor_area * 0.25,
            zone.floor_area * 0.25,
            zone.floor_area * 0.25,
            zone.floor_area * 0.25,
            zone.floor_area,
            zone.floor_area,
        ];

        for ((name, surf_type), (construction_id, area)) in surface_names
            .iter()
            .zip(surface_types.iter())
            .zip(construction_ids.iter().zip(areas.iter()))
        {
            self.surface_counter += 1;
            let surf_id = format!("surface{}", self.surface_counter);

            let mut surface = BytesStart::new("Surface");
            surface.push_attribute(("id", surf_id.as_str()));
            surface.push_attribute(("name", *name));
            surface.push_attribute(("surfaceType", *surf_type));
            surface.push_attribute(("constructionIdRef", *construction_id));
            writer.write_event(Event::Start(surface))?;

            write_text_element(writer, "Area", &area.to_string())?;

            // RectangularGeometry
            let mut geom = BytesStart::new("RectangularGeometry");
            let azimuth = if *surf_type == "Roof" { "0" } else { "180" };
            geom.push_attribute(("Azimuth", azimuth));
            geom.push_attribute(("Tilt", if *surf_type == "Floor" { "180" } else { "90" }));
            writer.write_event(Event::Start(geom))?;

            // CartesianPoint
            let point = BytesStart::new("CartesianPoint");
            writer.write_event(Event::Start(point))?;
            write_text_element(writer, "Coordinate", "0.0")?;
            write_text_element(writer, "Coordinate", "0.0")?;
            write_text_element(writer, "Coordinate", "0.0")?;
            writer.write_event(Event::End(BytesEnd::new("CartesianPoint")))?;
            writer.write_event(Event::End(BytesEnd::new("RectangularGeometry")))?;

            // AdjacentSpaceId
            let mut adj = BytesStart::new("AdjacentSpaceId");
            adj.push_attribute(("spaceIdRef", space_id.as_str()));
            writer.write_event(Event::Empty(adj))?;

            writer.write_event(Event::End(BytesEnd::new("Surface")))?;
        }

        writer.write_event(Event::End(BytesEnd::new("Space")))?;

        Ok(())
    }

    /// Write Zone elements for thermal zone properties (LoadClass, Schedules, InternalGains).
    fn write_zones<W: std::io::Write>(
        &mut self,
        schema: &SimulationSchemaV1,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        for (zone_idx, zone) in schema.geometry.zones.iter().enumerate() {
            self.zone_counter += 1;
            let zone_id = format!("zone{}", zone_idx + 1);

            let mut zone_elem = BytesStart::new("Zone");
            zone_elem.push_attribute(("id", zone_id.as_str()));
            zone_elem.push_attribute(("name", zone.name.as_str()));
            writer.write_event(Event::Start(zone_elem))?;

            // LoadClass - building load classification based on occupancy type
            // Default to "Commercial" as a reasonable assumption for ASHRAE cases
            write_text_element(writer, "LoadClass", "Commercial")?;

            // SchedulesIdRef - references to schedule elements
            let schedule_ids = ["schedule_occupancy", "schedule_lighting", "schedule_hvac"];
            for schedule_id in &schedule_ids {
                let mut sched_ref = BytesStart::new("SchedulesIdRef");
                sched_ref.push_attribute(("scheduleIdRef", *schedule_id));
                writer.write_event(Event::Empty(sched_ref))?;
            }

            // InternalGains - people, lights, equipment
            let gains = BytesStart::new("InternalGains");
            writer.write_event(Event::Start(gains))?;

            // People - occupant heat gain
            let mut people = BytesStart::new("People");
            people.push_attribute(("gainPerPerson", "100.0")); // W per person
            writer.write_event(Event::Start(people))?;
            let mut people_sched = BytesStart::new("SchedulesIdRef");
            people_sched.push_attribute(("scheduleIdRef", "schedule_occupancy"));
            writer.write_event(Event::Empty(people_sched))?;
            writer.write_event(Event::End(BytesEnd::new("People")))?;

            // Lights - lighting heat gain
            let mut lights = BytesStart::new("Lights");
            lights.push_attribute(("gainPerFloorArea", "10.0")); // W/m²
            writer.write_event(Event::Start(lights))?;
            let mut lights_sched = BytesStart::new("SchedulesIdRef");
            lights_sched.push_attribute(("scheduleIdRef", "schedule_lighting"));
            writer.write_event(Event::Empty(lights_sched))?;
            writer.write_event(Event::End(BytesEnd::new("Lights")))?;

            // Equipment - miscellaneous equipment gains
            let mut equip = BytesStart::new("Equipment");
            equip.push_attribute(("gainPerFloorArea", "5.0")); // W/m²
            writer.write_event(Event::Start(equip))?;
            let mut equip_sched = BytesStart::new("SchedulesIdRef");
            equip_sched.push_attribute(("scheduleIdRef", "schedule_occupancy"));
            writer.write_event(Event::Empty(equip_sched))?;
            writer.write_event(Event::End(BytesEnd::new("Equipment")))?;

            writer.write_event(Event::End(BytesEnd::new("InternalGains")))?;

            writer.write_event(Event::End(BytesEnd::new("Zone")))?;
        }

        Ok(())
    }

    /// Write Schedule elements for occupancy, lighting, and HVAC schedules.
    fn write_schedules<W: std::io::Write>(
        &mut self,
        schedules: &crate::api::schema::ScheduleSet,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        // Occupancy schedule
        self.write_schedule(
            "schedule_occupancy",
            "Occupancy",
            &schedules.occupancy,
            writer,
        )?;

        // Lighting schedule
        self.write_schedule("schedule_lighting", "Lighting", &schedules.lighting, writer)?;

        // HVAC schedule (heating/cooling setpoints)
        self.write_hvac_schedule("schedule_hvac", "HVAC", &schedules.hvac, writer)?;

        Ok(())
    }

    /// Write a daily/weekly schedule element.
    fn write_schedule<W: std::io::Write>(
        &mut self,
        schedule_id: &str,
        schedule_name: &str,
        schedule: &crate::sim::schedule::DailySchedule,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        use crate::sim::schedule::{ScheduleType, ScheduleValues};

        let schedule_type_str = match schedule.schedule_type {
            ScheduleType::Constant => "Constant",
            ScheduleType::DailyCycle => "Daily",
            ScheduleType::Weekly => "Weekly",
            ScheduleType::Custom => "Custom",
        };

        let mut sched_elem = BytesStart::new("Schedule");
        sched_elem.push_attribute(("id", schedule_id));
        sched_elem.push_attribute(("name", schedule_name));
        sched_elem.push_attribute(("scheduleType", "Temperature"));
        writer.write_event(Event::Start(sched_elem))?;

        // ScheduleType
        let mut type_elem = BytesStart::new("ScheduleType");
        type_elem.push_attribute(("timeType", "Hourly"));
        writer.write_event(Event::Start(type_elem))?;
        write_text_element(writer, "ScheduleType", schedule_type_str)?;
        writer.write_event(Event::End(BytesEnd::new("ScheduleType")))?;

        // ScheduleValues - export first day values as representative
        let mut values_elem = BytesStart::new("ScheduleValues");
        values_elem.push_attribute(("dayType", "Monday"));
        writer.write_event(Event::Start(values_elem))?;

        match &schedule.values {
            ScheduleValues::Daily(hours) => {
                for (i, value) in hours.iter().enumerate() {
                    let mut hourly = BytesStart::new("HourlyValue");
                    hourly.push_attribute(("hour", i.to_string().as_str()));
                    writer.write_event(Event::Start(hourly))?;
                    write_text_element(writer, "Value", &value.to_string())?;
                    writer.write_event(Event::End(BytesEnd::new("HourlyValue")))?;
                }
            }
            ScheduleValues::Weekly(weeks) => {
                // Export Monday (first day) as representative
                for (i, value) in weeks[0].iter().enumerate() {
                    let mut hourly = BytesStart::new("HourlyValue");
                    hourly.push_attribute(("hour", i.to_string().as_str()));
                    writer.write_event(Event::Start(hourly))?;
                    write_text_element(writer, "Value", &value.to_string())?;
                    writer.write_event(Event::End(BytesEnd::new("HourlyValue")))?;
                }
            }
        }

        writer.write_event(Event::End(BytesEnd::new("ScheduleValues")))?;
        writer.write_event(Event::End(BytesEnd::new("Schedule")))?;

        Ok(())
    }

    /// Write HVAC schedule element (heating and cooling setpoints).
    fn write_hvac_schedule<W: std::io::Write>(
        &mut self,
        schedule_id: &str,
        schedule_name: &str,
        hvac: &crate::sim::schedule::HVACSchedule,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        let mut sched_elem = BytesStart::new("Schedule");
        sched_elem.push_attribute(("id", schedule_id));
        sched_elem.push_attribute(("name", schedule_name));
        sched_elem.push_attribute(("scheduleType", "Temperature"));
        writer.write_event(Event::Start(sched_elem))?;

        // ScheduleType
        let mut type_elem = BytesStart::new("ScheduleType");
        type_elem.push_attribute(("timeType", "Hourly"));
        writer.write_event(Event::Start(type_elem))?;
        write_text_element(writer, "ScheduleType", "Weekly")?;
        writer.write_event(Event::End(BytesEnd::new("ScheduleType")))?;

        // ScheduleValues - export Monday as representative
        let mut values_elem = BytesStart::new("ScheduleValues");
        values_elem.push_attribute(("dayType", "Monday"));
        writer.write_event(Event::Start(values_elem))?;

        // Export cooling schedule values for Monday (first day)
        use crate::sim::schedule::ScheduleValues;
        if let ScheduleValues::Daily(hours) = &hvac.cooling.values {
            for (i, value) in hours.iter().enumerate() {
                let mut hourly = BytesStart::new("HourlyValue");
                hourly.push_attribute(("hour", i.to_string().as_str()));
                writer.write_event(Event::Start(hourly))?;
                write_text_element(writer, "Value", &value.to_string())?;
                writer.write_event(Event::End(BytesEnd::new("HourlyValue")))?;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("ScheduleValues")))?;
        writer.write_event(Event::End(BytesEnd::new("Schedule")))?;

        Ok(())
    }

    fn write_constructions<W: std::io::Write>(
        &mut self,
        _constructions: &ConstructionSet,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        // Write wall construction
        self.write_simple_construction(
            "construction_wall",
            "Standard Wall",
            vec![
                ("layer_wall_1", "Concrete", 0.1, 1.4, 2300.0, 840.0),
                ("layer_wall_2", "Insulation", 0.05, 0.04, 50.0, 840.0),
            ],
            writer,
        )?;

        // Write roof construction
        self.write_simple_construction(
            "construction_roof",
            "Standard Roof",
            vec![
                ("layer_roof_1", "RoofMaterial", 0.1, 1.4, 2300.0, 840.0),
                ("layer_roof_2", "RoofInsulation", 0.1, 0.04, 50.0, 840.0),
            ],
            writer,
        )?;

        // Write floor construction
        self.write_simple_construction(
            "construction_floor",
            "Standard Floor",
            vec![("layer_floor_1", "Concrete", 0.15, 1.4, 2300.0, 840.0)],
            writer,
        )?;

        Ok(())
    }

    fn write_simple_construction<W: std::io::Write>(
        &mut self,
        construction_id: &str,
        construction_name: &str,
        layers: Vec<(&str, &str, f64, f64, f64, f64)>,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        // Write layers and materials first
        let mut layer_ids: Vec<String> = Vec::new();
        for (_layer_name, mat_name, thickness, conductivity, density, specific_heat) in &layers {
            self.layer_counter += 1;
            let layer_id = format!("layer_{}_{}", construction_id, self.layer_counter);
            layer_ids.push(layer_id.clone());

            // Write layer
            let mut layer_elem = BytesStart::new("Layer");
            layer_elem.push_attribute(("id", layer_id.as_str()));
            writer.write_event(Event::Start(layer_elem.clone()))?;

            // Material
            self.material_counter += 1;
            let mat_id = format!("material_{}", self.material_counter);

            let mut mat_elem = BytesStart::new("Material");
            mat_elem.push_attribute(("id", mat_id.as_str()));
            mat_elem.push_attribute(("name", *mat_name));
            writer.write_event(Event::Start(mat_elem))?;

            write_text_element(writer, "Thickness", &thickness.to_string())?;
            write_text_element(writer, "Conductivity", &conductivity.to_string())?;
            write_text_element(writer, "Density", &density.to_string())?;
            write_text_element(writer, "SpecificHeat", &specific_heat.to_string())?;

            writer.write_event(Event::End(BytesEnd::new("Material")))?;

            // MaterialIdRef in layer
            let mut mat_ref = BytesStart::new("MaterialIdRef");
            mat_ref.push_attribute(("materialIdRef", mat_id.as_str()));
            writer.write_event(Event::Empty(mat_ref))?;

            writer.write_event(Event::End(BytesEnd::new("Layer")))?;
        }

        // Write construction element
        let layer_count = layer_ids.len().to_string();
        let mut const_elem = BytesStart::new("Construction");
        const_elem.push_attribute(("id", construction_id));
        const_elem.push_attribute(("name", construction_name));
        const_elem.push_attribute(("layerCount", layer_count.as_str()));
        writer.write_event(Event::Start(const_elem))?;

        for layer_id in &layer_ids {
            let mut layer_ref = BytesStart::new("LayerIdRef");
            layer_ref.push_attribute(("layerIdRef", layer_id.as_str()));
            writer.write_event(Event::Empty(layer_ref))?;
        }

        writer.write_event(Event::End(BytesEnd::new("Construction")))?;

        Ok(())
    }
}

impl Default for GbXmlWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn write_text_element<W: std::io::Write>(
    writer: &mut Writer<W>,
    name: &str,
    value: &str,
) -> Result<(), GbXmlError> {
    writer.write_event(Event::Start(BytesStart::new(name)))?;
    writer.write_event(Event::Text(BytesText::new(value)))?;
    writer.write_event(Event::End(BytesEnd::new(name)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::schema::{
        ConstructionSet, ControlSet, Geometry, SchemaMetadata, SchemaVersion, SimulationOutput,
        SimulationSchemaV1, WeatherData, ZoneGeometry,
    };

    fn create_test_schema() -> SimulationSchemaV1 {
        SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata {
                name: "Test Building".to_string(),
                description: "Test building for gbXML export".to_string(),
                author: Some("Test".to_string()),
                created_at: Some("2026-01-01".to_string()),
                schema_version: SchemaVersion::V1,
            },
            geometry: Geometry {
                zones: vec![ZoneGeometry {
                    name: "Zone 1".to_string(),
                    floor_area: 48.0,
                    volume: 129.6,
                    height: 2.7,
                }],
                total_floor_area: 48.0,
                total_volume: 129.6,
                number_of_floors: 1,
                floor_height: 2.7,
            },
            constructions: ConstructionSet::default(),
            schedules: crate::api::schema::ScheduleSet::default(),
            weather: WeatherData::TmyLocation {
                location: "Denver, CO".to_string(),
            },
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        }
    }

    #[test]
    fn test_export_gbxml() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = GbXmlWriter::new();
        writer
            .write_schema(&schema, &mut output)
            .expect("Should export");

        let xml_str = String::from_utf8(output).expect("Should be valid UTF-8");
        assert!(xml_str.contains("gbXML"));
        assert!(xml_str.contains("Test Building"));
        assert!(xml_str.contains("Zone 1"));
    }

    #[test]
    fn test_roundtrip() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = GbXmlWriter::new();
        writer
            .write_schema(&schema, &mut output)
            .expect("Should export");

        let xml_str = String::from_utf8(output).expect("Should be valid UTF-8");

        // Parse it back
        let reader = crate::interop::gbxml::reader::GbXmlReader::new();
        let parsed = reader.parse(&xml_str).expect("Should parse exported gbXML");
        assert_eq!(parsed.geometry.zones.len(), 1);
        assert_eq!(parsed.geometry.zones[0].name, "Zone 1");
    }

    #[test]
    fn test_thermal_zone_properties_export() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = GbXmlWriter::new();
        writer
            .write_schema(&schema, &mut output)
            .expect("Should export");

        let xml_str = String::from_utf8(output).expect("Should be valid UTF-8");

        // Verify Zone elements with thermal properties
        assert!(xml_str.contains("<Zone"));
        assert!(xml_str.contains("LoadClass"));
        assert!(xml_str.contains("Commercial"));

        // Verify InternalGains elements
        assert!(xml_str.contains("InternalGains"));
        assert!(xml_str.contains("People"));
        assert!(xml_str.contains("Lights"));
        assert!(xml_str.contains("Equipment"));

        // Verify SchedulesIdRef elements
        assert!(xml_str.contains("SchedulesIdRef"));
        assert!(xml_str.contains("schedule_occupancy"));
        assert!(xml_str.contains("schedule_lighting"));
        assert!(xml_str.contains("schedule_hvac"));

        // Verify Schedule elements
        assert!(xml_str.contains("<Schedule"));
        assert!(xml_str.contains("Occupancy"));
        assert!(xml_str.contains("Lighting"));
        assert!(xml_str.contains("HVAC"));
    }

    #[test]
    fn test_surface_construction_id_ref() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = GbXmlWriter::new();
        writer
            .write_schema(&schema, &mut output)
            .expect("Should export");

        let xml_str = String::from_utf8(output).expect("Should be valid UTF-8");

        // Verify surfaces reference correct constructions
        assert!(xml_str.contains("constructionIdRef=\"construction_wall\""));
        assert!(xml_str.contains("constructionIdRef=\"construction_roof\""));
        assert!(xml_str.contains("constructionIdRef=\"construction_floor\""));
    }
}
