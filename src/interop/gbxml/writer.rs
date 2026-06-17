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

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use std::io::Cursor;

use crate::api::schema::{
    ConstructionSet, ControlSet, Geometry, SchemaMetadata, SimulationOutput,
    SimulationSchemaV1, SurfaceConstruction, WeatherData, WindowSpec, ZoneGeometry,
};
use crate::interop::gbxml::error::GbXmlError;
use crate::interop::gbxml::types::*;
use crate::sim::construction::ConstructionLayer;

/// Export a SimulationSchema to gbXML file.
pub fn export_gbxml(schema: &SimulationSchemaV1, path: impl AsRef<Path>) -> Result<(), GbXmlError> {
    let file = File::create(path.as_ref())
        .map_err(|e| GbXmlError::io_error(path.as_ref(), e.to_string()))?;
    let writer = BufWriter::new(file);

    let mut gbxml_writer = GbXmlWriter::new();
    gbxml_writer.write_schema(schema, writer)
}

/// GbXmlWriter for exporting to gbXML format.
pub struct GbXmlWriter {
    construction_counter: usize,
    layer_counter: usize,
    material_counter: usize,
    space_counter: usize,
    surface_counter: usize,
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
            WeatherData::EpwFile { path } => {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Unknown")
                    .to_string()
            }
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
        schema: &SimulationSchemaV1,
        writer: &mut Writer<W>,
    ) -> Result<(), GbXmlError> {
        self.space_counter += 1;
        let space_id = format!("space{}", self.space_counter);

        let mut space = BytesStart::new("Space");
        space.push_attribute(("id", space_id.as_str()));
        space.push_attribute(("name", zone.name.as_str()));
        writer.write_event(Event::Start(space))?;

        write_text_element(writer, "Area", &zone.floor_area.to_string())?;
        write_text_element(writer, "Volume", &zone.volume.to_string())?;

        // Write default wall construction for each surface
        // In a full implementation, we'd look up actual constructions
        let surface_names = ["North Wall", "East Wall", "South Wall", "West Wall", "Roof", "Floor"];
        let surface_types = ["ExteriorWall", "ExteriorWall", "ExteriorWall", "ExteriorWall", "Roof", "Floor"];
        let areas = [
            zone.floor_area * 0.25,
            zone.floor_area * 0.25,
            zone.floor_area * 0.25,
            zone.floor_area * 0.25,
            zone.floor_area,
            zone.floor_area,
        ];

        for (surf_idx, ((name, surf_type), area)) in surface_names.iter()
            .zip(surface_types.iter())
            .zip(areas.iter())
            .enumerate()
        {
            self.surface_counter += 1;
            let surf_id = format!("surface{}", self.surface_counter);

            let mut surface = BytesStart::new("Surface");
            surface.push_attribute(("id", surf_id.as_str()));
            surface.push_attribute(("name", *name));
            surface.push_attribute(("surfaceType", *surf_type));
            surface.push_attribute(("constructionIdRef", "construction_wall"));
            writer.write_event(Event::Start(surface))?;

            write_text_element(writer, "Area", &area.to_string())?;

            // RectangularGeometry
            let mut geom = BytesStart::new("RectangularGeometry");
            let azimuth = if *surf_type == "Roof" { "0" } else { "180" };
            geom.push_attribute(("Azimuth", azimuth));
            geom.push_attribute(("Tilt", if *surf_type == "Floor" { "180" } else { "90" }));
            writer.write_event(Event::Start(geom))?;

            // CartesianPoint
            let mut point = BytesStart::new("CartesianPoint");
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

    fn write_constructions<W: std::io::Write>(
        &mut self,
        constructions: &ConstructionSet,
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
            vec![
                ("layer_floor_1", "Concrete", 0.15, 1.4, 2300.0, 840.0),
            ],
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
        for (layer_name, mat_name, thickness, conductivity, density, specific_heat) in &layers {
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
    use crate::api::schema::{SimulationSchemaV1, SchemaMetadata, SchemaVersion};

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
        writer.write_schema(&schema, &mut output).expect("Should export");

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
        writer.write_schema(&schema, &mut output).expect("Should export");

        let xml_str = String::from_utf8(output).expect("Should be valid UTF-8");

        // Parse it back
        let reader = crate::interop::gbxml::reader::GbXmlReader::new();
        let parsed = reader.parse(&xml_str).expect("Should parse exported gbXML");
        assert_eq!(parsed.geometry.zones.len(), 1);
        assert_eq!(parsed.geometry.zones[0].name, "Zone 1");
    }
}
