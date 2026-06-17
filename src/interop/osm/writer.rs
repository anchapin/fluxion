// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OSM file writer - exports fluxion schema to OpenStudio Model files.
//!
//! This module provides functionality to serialize fluxion's [`SimulationSchemaV1`]
//! format into OSM (OpenStudio Model) files for interoperability with the
//! OpenStudio SDK ecosystem.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::osm::{export_osm, OsmWriter};
//!
//! let writer = OsmWriter::new();
//! writer.export_osm(&schema, "output.osm")?;
//! ```
//!
//! # Limitations
//!
//! This is an initial implementation with the following known limitations:
//! - Limited HVAC system export
//! - Basic schedule representation
//! - Simplified construction export

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::api::schema::SimulationSchemaV1;
use crate::interop::osm::error::OsmError;
use crate::sim::construction::ConstructionLayer;

pub fn export_osm(schema: &SimulationSchemaV1, path: impl AsRef<Path>) -> Result<(), OsmError> {
    let mut writer = OsmWriter::new();
    writer.export_osm(schema, path)
}

pub struct OsmWriter {
    #[allow(dead_code)]
    indent: usize,
    handle_counter: usize,
}

impl OsmWriter {
    pub fn new() -> Self {
        OsmWriter {
            indent: 0,
            handle_counter: 0,
        }
    }

    pub fn export_osm(
        &mut self,
        schema: &SimulationSchemaV1,
        path: impl AsRef<Path>,
    ) -> Result<(), OsmError> {
        let file = File::create(path.as_ref()).map_err(|e| OsmError::ExportError(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        self.write_header(&mut writer)?;
        self.write_version(&mut writer)?;
        self.write_site(&mut writer, schema)?;
        self.write_building(&mut writer, schema)?;
        self.write_materials(&mut writer, schema)?;
        self.write_constructions(&mut writer, schema)?;
        self.write_thermal_zones(&mut writer, schema)?;
        self.write_spaces(&mut writer, schema)?;
        self.write_surfaces(&mut writer, schema)?;

        writer
            .flush()
            .map_err(|e| OsmError::ExportError(e.to_string()))?;

        Ok(())
    }

    fn write_header(&mut self, writer: &mut dyn Write) -> Result<(), OsmError> {
        writeln!(
            writer,
            "================================================================================"
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, " FLUXION MODEL - Generated OpenStudio Model File")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(
            writer,
            "================================================================================"
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        Ok(())
    }

    fn write_version(&mut self, writer: &mut dyn Write) -> Result<(), OsmError> {
        writeln!(writer, "OS:Version,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{version}}, !- Handle")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  3.6.0; !- Version Identifier")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        Ok(())
    }

    fn write_site(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        let lat = match &schema.weather {
            crate::api::schema::WeatherData::TmyLocation { location } => location
                .split(',')
                .next()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(39.739),
            _ => 39.739,
        };

        let lon = match &schema.weather {
            crate::api::schema::WeatherData::TmyLocation { location } => location
                .split(',')
                .nth(1)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(-104.984),
            _ => -104.984,
        };

        writeln!(writer, "OS:Site,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{site-{}}}, !- Handle", self.handle_counter())
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Name", schema.metadata.name)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Latitude", lat)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Longitude", lon)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  1609; !- Elevation {{m}}")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        Ok(())
    }

    fn write_building(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        writeln!(writer, "OS:Building,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{bldg-{}}}, !- Handle", self.handle_counter())
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Name", schema.metadata.name)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  , !- Building Story Names")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  , !- Thermal Zone Names")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(
            writer,
            "  {}, !- Floor Area {{m2}}",
            schema.geometry.total_floor_area
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(
            writer,
            "  {}, !- Number of Floors",
            schema.geometry.number_of_floors
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(
            writer,
            "  {}, !- Floor Height {{m}}",
            schema.geometry.floor_height
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  ;").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        Ok(())
    }

    fn write_materials(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        let mut material_handles: Vec<String> = Vec::new();

        for (i, layer) in schema.constructions.wall.layers.iter().enumerate() {
            let handle = format!("{{mat-w{}}}", i);
            material_handles.push(handle.clone());
            self.write_material(writer, &handle, layer)?;
        }

        for (i, layer) in schema.constructions.roof.layers.iter().enumerate() {
            let handle = format!("{{mat-r{}}}", i);
            self.write_material(writer, &handle, layer)?;
        }

        for (i, layer) in schema.constructions.floor.layers.iter().enumerate() {
            let handle = format!("{{mat-f{}}}", i);
            self.write_material(writer, &handle, layer)?;
        }

        Ok(())
    }

    fn write_material(
        &mut self,
        writer: &mut dyn Write,
        handle: &str,
        layer: &ConstructionLayer,
    ) -> Result<(), OsmError> {
        writeln!(writer, "OS:Material,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Handle", handle)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Name", layer.name)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  MediumRough, !- Roughness")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Thickness {{m}}", layer.thickness)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(
            writer,
            "  {}, !- Conductivity {{W/m-K}}",
            layer.conductivity
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Density {{kg/m3}}", layer.density)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(
            writer,
            "  {}, !- Specific Heat {{J/kg-K}}",
            layer.specific_heat
        )
        .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}; !- Emissivity", layer.emissivity)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        Ok(())
    }

    fn write_constructions(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        self.write_construction(writer, "ExtWall", &schema.constructions.wall)?;

        if schema.constructions.wall.layers.len() != schema.constructions.roof.layers.len()
            || schema
                .constructions
                .wall
                .layers
                .iter()
                .zip(schema.constructions.roof.layers.iter())
                .any(|(a, b)| a.name != b.name)
        {
            self.write_construction(writer, "Roof", &schema.constructions.roof)?;
        }

        if schema.constructions.wall.layers.len() != schema.constructions.floor.layers.len()
            || schema
                .constructions
                .wall
                .layers
                .iter()
                .zip(schema.constructions.floor.layers.iter())
                .any(|(a, b)| a.name != b.name)
        {
            self.write_construction(writer, "Floor", &schema.constructions.floor)?;
        }

        Ok(())
    }

    fn write_construction(
        &mut self,
        writer: &mut dyn Write,
        name: &str,
        surface: &crate::api::schema::SurfaceConstruction,
    ) -> Result<(), OsmError> {
        writeln!(writer, "OS:Construction,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{cons-{}}}, !- Handle", self.handle_counter())
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Name", name)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;

        for (i, layer) in surface.layers.iter().enumerate() {
            let mat_handle = format!("{{mat-{}{}}}", &name[..1].to_lowercase(), i);
            if i == surface.layers.len() - 1 {
                writeln!(writer, "  {}; !- Layer {}", mat_handle, i + 1)
                    .map_err(|e| OsmError::ExportError(e.to_string()))?;
            } else {
                writeln!(writer, "  {}, !- Layer {}", mat_handle, i + 1)
                    .map_err(|e| OsmError::ExportError(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn write_thermal_zones(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        for (i, zone) in schema.geometry.zones.iter().enumerate() {
            writeln!(writer, "OS:ThermalZone,")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {{zone-{}}}, !- Handle", i)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {}, !- Name", zone.name)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  , !- Thermostat Handle")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  1; !- Multiplier")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        }
        Ok(())
    }

    fn write_spaces(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        for (i, zone) in schema.geometry.zones.iter().enumerate() {
            writeln!(writer, "OS:Space,").map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {{space-{}}}, !- Handle", i)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {}, !- Name", zone.name)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {{zone-{}}}, !- Zone Handle", i)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  , !- Building Story Handle")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  0, !- X Origin {{m}}")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  0, !- Y Origin {{m}}")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  0; !- Z Origin {{m}}")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        }
        Ok(())
    }

    fn write_surfaces(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        let total_area = schema.geometry.total_floor_area;
        let perimeter = (total_area * 4.0).sqrt() * 4.0;
        let wall_height = schema.geometry.floor_height;

        let wall_area = perimeter * wall_height / 4.0;
        let wall_types = ["West Wall", "North Wall", "East Wall", "South Wall"];

        for (i, wall_type) in wall_types.iter().enumerate() {
            writeln!(writer, "OS:Surface,").map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {{surf-w{}}}, !- Handle", i)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {}, !- Name", wall_type)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  Wall, !- Surface Type")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {{cons-w0}}, !- Construction Handle",)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  , !- Building Boundary Type")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  Outdoors, !- Outside Boundary Condition")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  , !- Sun Exposure")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  ; !- Wind Exposure")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        }

        let roof_area = total_area;
        writeln!(writer, "OS:Surface,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{surf-r0}}, !- Handle")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  Roof, !- Name").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  RoofCeiling, !- Surface Type")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{cons-r0}}, !- Construction Handle")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  , !- Building Boundary Type")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  Outdoors, !- Outside Boundary Condition")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  , !- Sun Exposure")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  ; !- Wind Exposure")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;

        writeln!(writer, "OS:Surface,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{surf-f0}}, !- Handle")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  Floor, !- Name").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  Floor, !- Surface Type")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{cons-f0}}, !- Construction Handle")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  , !- Building Boundary Type")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  Ground, !- Outside Boundary Condition")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  NoSun, !- Sun Exposure")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  NoWind; !- Wind Exposure")
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;

        Ok(())
    }

    fn handle_counter(&mut self) -> usize {
        self.handle_counter += 1;
        self.handle_counter
    }
}

impl Default for OsmWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::schema::{
        ConstructionSet, ControlSet, Geometry, SchemaMetadata, SchemaVersion, SimulationOutput,
        SimulationSchemaV1, SurfaceConstruction, WeatherData, ZoneGeometry,
    };

    fn create_test_schema() -> SimulationSchemaV1 {
        SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata {
                name: "Test Building".to_string(),
                description: "Test OSM export".to_string(),
                author: None,
                created_at: Some("2026-01-01".to_string()),
                schema_version: SchemaVersion::V1,
            },
            geometry: Geometry {
                zones: vec![ZoneGeometry {
                    name: "Zone 1".to_string(),
                    floor_area: 100.0,
                    volume: 270.0,
                    height: 2.7,
                }],
                total_floor_area: 100.0,
                total_volume: 270.0,
                number_of_floors: 1,
                floor_height: 2.7,
            },
            constructions: ConstructionSet::default(),
            schedules: crate::api::schema::ScheduleSet::default(),
            weather: WeatherData::TmyLocation {
                location: "40.0, -105.0".to_string(),
            },
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        }
    }

    #[test]
    fn test_export_osm() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut writer = OsmWriter::new();
        let schema = create_test_schema();

        let path = temp_dir.path().join("test_export.osm");
        writer.export_osm(&schema, &path).expect("Should export");

        let content = std::fs::read_to_string(&path).expect("Should read");
        assert!(content.contains("Test Building"));
        assert!(content.contains("OS:Material"));
        assert!(content.contains("OS:Construction"));
        assert!(content.contains("OS:ThermalZone"));
    }

    #[test]
    fn test_export_osm_to_file() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut writer = OsmWriter::new();
        let schema = create_test_schema();

        let path = temp_dir.path().join("test_export.osm");
        writer.export_osm(&schema, &path).expect("Should export");

        let content = std::fs::read_to_string(&path).expect("Should read");
        assert!(content.contains("Test Building"));
    }
}
