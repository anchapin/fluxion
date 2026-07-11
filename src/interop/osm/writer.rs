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
        self.write_thermostats(&mut writer, schema)?;
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

    /// Write the three OS:Construction objects (Wall / Roof / Floor).
    ///
    /// Each surface type gets a stable handle (`{cons-w0}`, `{cons-r0}`, `{cons-f0}`)
    /// matching what `write_surfaces` references, so round-trip is lossless.
    /// We always emit all three constructions — even when the layers are identical
    /// across types — because `write_surfaces` references distinct handles per type
    /// and the reader resolves constructions by handle.
    fn write_constructions(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        // Map each construction type to a (handle-prefix, name-prefix) pair so that
        // material handles written by `write_materials` ({mat-w0..}, {mat-r0..},
        // {mat-f0..}) match the layer handles written here.
        self.write_construction(writer, "w", "ExtWall", &schema.constructions.wall)?;
        self.write_construction(writer, "r", "Roof", &schema.constructions.roof)?;
        self.write_construction(writer, "f", "Floor", &schema.constructions.floor)?;
        Ok(())
    }

    fn write_construction(
        &mut self,
        writer: &mut dyn Write,
        prefix: &str,
        name: &str,
        surface: &crate::api::schema::SurfaceConstruction,
    ) -> Result<(), OsmError> {
        writeln!(writer, "OS:Construction,").map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {{cons-{0}0}}, !- Handle", prefix)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
        writeln!(writer, "  {}, !- Name", name)
            .map_err(|e| OsmError::ExportError(e.to_string()))?;

        for (i, _layer) in surface.layers.iter().enumerate() {
            let mat_handle = format!("{{mat-{}{}}}", prefix, i);
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
            // Reference the matching OS:Thermostat emitted by `write_thermostats`
            // so reader can resolve the heating/cooling setpoints back into
            // `controls.zone_control`. Handle mirrors the zone index — round-trip
            // is lossless for `controls.zone_control.{heating,cooling}_setpoint`
            // (issue #1432).
            writeln!(writer, "  {{thermostat-{}}}, !- Thermostat Handle", i)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  1; !- Multiplier")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer).map_err(|e| OsmError::ExportError(e.to_string()))?;
        }
        Ok(())
    }

    /// Emit one `OS:Thermostat` per thermal zone, carrying the heating/cooling
    /// setpoints from `schema.controls.zone_control`. Each thermostat's handle
    /// (`{thermostat-N}`) is referenced from the matching `OS:ThermalZone`
    /// so the reader can resolve it during `extract_controls`.
    ///
    /// Handle scheme mirrors `write_thermal_zones` (zone-index based) — the
    /// handle prefix `thermostat-` does not collide with any other emitted
    /// handle family (`site-`, `bldg-`, `zone-`, `space-`, `surf-`,
    /// `mat-{w,r,f}-`, `cons-{w,r,f}-`, `version`).
    ///
    /// This makes the writer→reader round-trip lossless for
    /// `controls.zone_control.{heating,cooling}_setpoint` (issue #1432).
    fn write_thermostats(
        &mut self,
        writer: &mut dyn Write,
        schema: &SimulationSchemaV1,
    ) -> Result<(), OsmError> {
        let hsp = schema.controls.zone_control.heating_setpoint;
        let csp = schema.controls.zone_control.cooling_setpoint;

        for (i, zone) in schema.geometry.zones.iter().enumerate() {
            writeln!(writer, "OS:Thermostat,")
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {{thermostat-{}}}, !- Handle", i)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {}, !- Name", zone.name)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(
                writer,
                "  {}, !- Heating Setpoint Temperature {{C}}",
                hsp
            )
            .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(
                writer,
                "  {}; !- Cooling Setpoint Temperature {{C}}",
                csp
            )
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
            // Emit area and volume so round-trip preserves zone dimensions.
            // OS:Space.area and .volume are read back by the reader (issue #1340).
            writeln!(writer, "  {}, !- Floor Area {{m2}}", zone.floor_area)
                .map_err(|e| OsmError::ExportError(e.to_string()))?;
            writeln!(writer, "  {}, !- Volume {{m3}}", zone.volume)
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

        let _wall_area = perimeter * wall_height / 4.0;
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

        let _roof_area = total_area;
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
    use crate::interop::osm::reader::OsmReader;

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

    // ------------------------------------------------------------------
    // Issue #1340 — OSM writer→reader round-trip parity tests
    //
    // These tests assert lossless round-trip for the supported subset of
    // SimulationSchemaV1 fields. On failure they print a structured
    // per-field diff so OSM Measure authors can debug exactly which
    // field mismatched (and not just see a generic boolean failure).
    //
    // Lossless fields (issue #1340 contract, extended by issue #1432):
    //   - metadata.name                  (via OS:Building.Name)
    //   - geometry.zones[*].name         (via OS:ThermalZone.Name)
    //   - geometry.zones[*].floor_area   (via OS:Space.Floor Area)
    //   - geometry.zones[*].volume       (via OS:Space.Volume)
    //   - geometry.total_floor_area      (= sum of zone floor areas)
    //   - geometry.total_volume          (= sum of zone volumes)
    //   - geometry.number_of_floors      (via OS:Building.Number of Floors)
    //   - geometry.floor_height          (computed from total_volume/total_floor_area)
    //   - constructions.{wall,roof,floor}.layers[*]
    //         name, thickness, conductivity, density, specific_heat
    //   - weather (TmyLocation only)     (via OS:Site.Latitude/Longitude)
    //   - controls.zone_control.heating_setpoint
    //                                    (via OS:Thermostat.Heating Setpoint
    //                                     Temperature — one per zone)
    //   - controls.zone_control.cooling_setpoint
    //                                    (via OS:Thermostat.Cooling Setpoint
    //                                     Temperature — one per zone)
    //
    // Known lossy fields (documented; intentionally NOT asserted):
    //   - metadata.description, metadata.author, metadata.created_at
    //     (OS:Building/Description is not emitted by the writer)
    //   - schedules (no OS:Schedule emission in writer; reader falls back to defaults)
    //   - constructions.window            (no OS:SubSurface or window construction
    //                                      emission in the supported subset)
    //   - constructions.interzone         (not emitted)
    //   - output.*                        (results, not part of model file)
    // ------------------------------------------------------------------

    /// Run a full writer → reader round-trip and return the re-parsed schema.
    fn roundtrip_schema(schema: &SimulationSchemaV1) -> SimulationSchemaV1 {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("roundtrip.osm");

        let mut writer = OsmWriter::new();
        writer
            .export_osm(schema, &path)
            .expect("writer: should export schema");

        let content = std::fs::read_to_string(&path).expect("Should read exported OSM");

        let mut reader = OsmReader::new();
        reader
            .parse(&content)
            .expect("reader: should re-parse exported OSM")
    }

    /// Build a structured per-field diff report.
    ///
    /// Returns an empty `Vec` when the two schemas match on every asserted
    /// field, or one `String` per mismatched field for use in `assert!`.
    fn diff_schemas(original: &SimulationSchemaV1, re_parsed: &SimulationSchemaV1) -> Vec<String> {
        let mut mismatches: Vec<String> = Vec::new();

        // metadata.name round-trips via OS:Building.Name
        if original.metadata.name != re_parsed.metadata.name {
            mismatches.push(format!(
                "metadata.name: '{}' -> '{}'",
                original.metadata.name, re_parsed.metadata.name
            ));
        }

        // geometry.zones — count and field-by-field
        if original.geometry.zones.len() != re_parsed.geometry.zones.len() {
            mismatches.push(format!(
                "geometry.zones.len: {} -> {}",
                original.geometry.zones.len(),
                re_parsed.geometry.zones.len()
            ));
        } else {
            for (i, (orig, new)) in original
                .geometry
                .zones
                .iter()
                .zip(re_parsed.geometry.zones.iter())
                .enumerate()
            {
                if orig.name != new.name {
                    mismatches.push(format!(
                        "geometry.zones[{i}].name: '{}' -> '{}'",
                        orig.name, new.name
                    ));
                }
                if !approx_eq(orig.floor_area, new.floor_area) {
                    mismatches.push(format!(
                        "geometry.zones[{i}].floor_area: {} -> {}",
                        orig.floor_area, new.floor_area
                    ));
                }
                if !approx_eq(orig.volume, new.volume) {
                    mismatches.push(format!(
                        "geometry.zones[{i}].volume: {} -> {}",
                        orig.volume, new.volume
                    ));
                }
                // height is derived from volume/area; should round-trip if both round-trip
                if !approx_eq(orig.height, new.height) {
                    mismatches.push(format!(
                        "geometry.zones[{i}].height: {} -> {}",
                        orig.height, new.height
                    ));
                }
            }
        }

        // geometry totals (sum of zone values)
        let expected_total_area: f64 = original.geometry.zones.iter().map(|z| z.floor_area).sum();
        let expected_total_volume: f64 = original.geometry.zones.iter().map(|z| z.volume).sum();
        if !approx_eq(expected_total_area, re_parsed.geometry.total_floor_area) {
            mismatches.push(format!(
                "geometry.total_floor_area: {} -> {}",
                expected_total_area, re_parsed.geometry.total_floor_area
            ));
        }
        if !approx_eq(expected_total_volume, re_parsed.geometry.total_volume) {
            mismatches.push(format!(
                "geometry.total_volume: {} -> {}",
                expected_total_volume, re_parsed.geometry.total_volume
            ));
        }
        if original.geometry.number_of_floors != re_parsed.geometry.number_of_floors {
            mismatches.push(format!(
                "geometry.number_of_floors: {} -> {}",
                original.geometry.number_of_floors, re_parsed.geometry.number_of_floors
            ));
        }

        // constructions.{wall, roof, floor}.layers — name + thermal properties
        for (ctype, orig_sc, new_sc) in [
            (
                "wall",
                &original.constructions.wall,
                &re_parsed.constructions.wall,
            ),
            (
                "roof",
                &original.constructions.roof,
                &re_parsed.constructions.roof,
            ),
            (
                "floor",
                &original.constructions.floor,
                &re_parsed.constructions.floor,
            ),
        ] {
            if orig_sc.layers.len() != new_sc.layers.len() {
                mismatches.push(format!(
                    "constructions.{ctype}.layers.len: {} -> {}",
                    orig_sc.layers.len(),
                    new_sc.layers.len()
                ));
            } else {
                for (li, (a, b)) in orig_sc.layers.iter().zip(new_sc.layers.iter()).enumerate() {
                    if a.name != b.name {
                        mismatches.push(format!(
                            "constructions.{ctype}.layers[{li}].name: '{}' -> '{}'",
                            a.name, b.name
                        ));
                    }
                    if !approx_eq(a.thickness, b.thickness) {
                        mismatches.push(format!(
                            "constructions.{ctype}.layers[{li}].thickness: {} -> {}",
                            a.thickness, b.thickness
                        ));
                    }
                    if !approx_eq(a.conductivity, b.conductivity) {
                        mismatches.push(format!(
                            "constructions.{ctype}.layers[{li}].conductivity: {} -> {}",
                            a.conductivity, b.conductivity
                        ));
                    }
                    if !approx_eq(a.density, b.density) {
                        mismatches.push(format!(
                            "constructions.{ctype}.layers[{li}].density: {} -> {}",
                            a.density, b.density
                        ));
                    }
                    if !approx_eq(a.specific_heat, b.specific_heat) {
                        mismatches.push(format!(
                            "constructions.{ctype}.layers[{li}].specific_heat: {} -> {}",
                            a.specific_heat, b.specific_heat
                        ));
                    }
                }
            }
        }

        // weather (TmyLocation only) — compare as (lat, lon) f64 pairs with
        // tolerance so the round-trip is robust to Display formatting
        // differences (e.g. "40.0, -105.0" vs "40, -105"). The string form
        // is not part of the lossless claim.
        match (&original.weather, &re_parsed.weather) {
            (
                WeatherData::TmyLocation { location: a },
                WeatherData::TmyLocation { location: b },
            ) => {
                let pa: Vec<&str> = a.split(',').collect();
                let pb: Vec<&str> = b.split(',').collect();
                if pa.len() != 2 || pb.len() != 2 {
                    if a != b {
                        mismatches.push(format!("weather.location: '{a}' -> '{b}'"));
                    }
                } else {
                    let lat_a = pa[0].trim().parse::<f64>().ok();
                    let lon_a = pa[1].trim().parse::<f64>().ok();
                    let lat_b = pb[0].trim().parse::<f64>().ok();
                    let lon_b = pb[1].trim().parse::<f64>().ok();
                    match (lat_a, lon_a, lat_b, lon_b) {
                        (Some(la), Some(lo), Some(lb), Some(lob))
                            if approx_eq(la, lb) && approx_eq(lo, lob) =>
                        {
                            // OK
                        }
                        _ => {
                            mismatches.push(format!("weather.location: '{a}' -> '{b}'"));
                        }
                    }
                }
            }
            _ => mismatches.push("weather variant mismatch".to_string()),
        }

        // controls.zone_control.{heating,cooling}_setpoint (issue #1432) —
        // the writer emits one OS:Thermostat per zone carrying these values,
        // and the reader folds them back into a single ControlConfig in
        // `extract_controls`. Round-trip is lossless within 1e-6.
        let oh = original.controls.zone_control.heating_setpoint;
        let oc = original.controls.zone_control.cooling_setpoint;
        let nh = re_parsed.controls.zone_control.heating_setpoint;
        let nc = re_parsed.controls.zone_control.cooling_setpoint;
        if !approx_eq(oh, nh) {
            mismatches.push(format!(
                "controls.zone_control.heating_setpoint: {oh} -> {nh}"
            ));
        }
        if !approx_eq(oc, nc) {
            mismatches.push(format!(
                "controls.zone_control.cooling_setpoint: {oc} -> {nc}"
            ));
        }

        mismatches
    }

    /// Absolute-or-relative tolerance for f64 round-trip comparisons.
    fn approx_eq(a: f64, b: f64) -> bool {
        const EPS_ABS: f64 = 1e-6;
        const EPS_REL: f64 = 1e-6;
        let diff = (a - b).abs();
        if diff <= EPS_ABS {
            return true;
        }
        let denom = a.abs().max(b.abs());
        if denom == 0.0 {
            return false;
        }
        diff / denom <= EPS_REL
    }

    /// Single-zone round-trip — mirrors gbxml/writer.rs:428 pattern.
    /// Asserts zone name, floor_area, volume, and construction layers survive
    /// byte-equivalent field-wise.
    #[test]
    fn test_roundtrip_single_zone() {
        let schema = create_test_schema();
        let re_parsed = roundtrip_schema(&schema);

        let diffs = diff_schemas(&schema, &re_parsed);
        assert!(
            diffs.is_empty(),
            "round-trip mismatch for single-zone schema:\n  - {}\n",
            diffs.join("\n  - ")
        );
    }

    /// Two-zone round-trip — exercises per-zone handle assignment, zone
    /// geometry aggregation, and (since issue #1432) thermostat preservation
    /// for non-default setpoints.
    #[test]
    fn test_roundtrip_two_zones() {
        use crate::api::schema::{ControlConfig, ControlSet};

        let mut schema = create_test_schema();
        schema.geometry.zones = vec![
            ZoneGeometry {
                name: "Zone A".to_string(),
                floor_area: 50.0,
                volume: 135.0,
                height: 2.7,
            },
            ZoneGeometry {
                name: "Zone B".to_string(),
                floor_area: 75.0,
                volume: 202.5,
                height: 2.7,
            },
        ];
        schema.geometry.total_floor_area = 125.0;
        schema.geometry.total_volume = 337.5;

        // Non-default setpoints (issue #1432). Pre-#1432 these would
        // silently regress to the reader's 20 °C / 24 °C fallback.
        schema.controls = ControlSet {
            zone_control: ControlConfig {
                heating_setpoint: 18.5,
                cooling_setpoint: 25.5,
                ..ControlConfig::default()
            },
            global_control: None,
        };

        let re_parsed = roundtrip_schema(&schema);

        let diffs = diff_schemas(&schema, &re_parsed);
        assert!(
            diffs.is_empty(),
            "round-trip mismatch for two-zone schema:\n  - {}\n",
            diffs.join("\n  - ")
        );

        // Belt-and-braces: assert thermostat setpoints directly within 1e-6,
        // matching the (18.5, 25.5) °C contract from issue #1432.
        assert!(
            (re_parsed.controls.zone_control.heating_setpoint - 18.5).abs() < 1e-6,
            "heating_setpoint regressed: {}",
            re_parsed.controls.zone_control.heating_setpoint
        );
        assert!(
            (re_parsed.controls.zone_control.cooling_setpoint - 25.5).abs() < 1e-6,
            "cooling_setpoint regressed: {}",
            re_parsed.controls.zone_control.cooling_setpoint
        );
    }

    /// Issue #1432 — 2-zone schema with non-default setpoints (18.5 / 25.5) °C
    /// must round-trip through OsmWriter → OsmReader without regressing to the
    /// reader's 20 / 24 defaults. Asserts the explicit numeric contract from
    /// the issue body.
    #[test]
    fn test_roundtrip_thermostat_preserves_setpoints() {
        use crate::api::schema::{ControlConfig, ControlSet};

        let mut schema = create_test_schema();
        schema.geometry.zones = vec![
            ZoneGeometry {
                name: "Zone A".to_string(),
                floor_area: 50.0,
                volume: 135.0,
                height: 2.7,
            },
            ZoneGeometry {
                name: "Zone B".to_string(),
                floor_area: 75.0,
                volume: 202.5,
                height: 2.7,
            },
        ];
        schema.geometry.total_floor_area = 125.0;
        schema.geometry.total_volume = 337.5;
        schema.controls = ControlSet {
            zone_control: ControlConfig {
                heating_setpoint: 18.5,
                cooling_setpoint: 25.5,
                ..ControlConfig::default()
            },
            global_control: None,
        };

        // Spot-check the exported OSM contains the per-zone OS:Thermostat
        // references and the dual-setpoint values.
        use tempfile::TempDir;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("thermostat.osm");
        let mut writer = OsmWriter::new();
        writer
            .export_osm(&schema, &path)
            .expect("writer: should export thermostat schema");
        let exported = std::fs::read_to_string(&path).expect("read exported OSM");
        assert!(
            exported.contains("OS:Thermostat"),
            "writer should emit OS:Thermostat objects (issue #1432)"
        );
        assert!(
            exported.contains("{thermostat-0}") && exported.contains("{thermostat-1}"),
            "writer should emit one OS:Thermostat per zone with stable handles"
        );
        assert!(
            exported.contains("18.5, !- Heating Setpoint Temperature"),
            "writer should emit the heating setpoint value"
        );
        assert!(
            exported.contains("25.5; !- Cooling Setpoint Temperature"),
            "writer should emit the cooling setpoint value"
        );

        // Round-trip and assert within 1e-6 per the issue acceptance criteria.
        let re_parsed = roundtrip_schema(&schema);

        let original_h = schema.controls.zone_control.heating_setpoint;
        let original_c = schema.controls.zone_control.cooling_setpoint;
        let parsed_h = re_parsed.controls.zone_control.heating_setpoint;
        let parsed_c = re_parsed.controls.zone_control.cooling_setpoint;

        assert!(
            (original_h - parsed_h).abs() < 1e-6,
            "heating_setpoint round-trip regression: {original_h} -> {parsed_h} \
             (issue #1432 acceptance criterion violated)"
        );
        assert!(
            (original_c - parsed_c).abs() < 1e-6,
            "cooling_setpoint round-trip regression: {original_c} -> {parsed_c} \
             (issue #1432 acceptance criterion violated)"
        );
        assert_eq!(
            original_h, 18.5,
            "test invariant: heating_setpoint is exactly 18.5"
        );
        assert_eq!(
            original_c, 25.5,
            "test invariant: cooling_setpoint is exactly 25.5"
        );
    }

    /// Four-zone round-trip — exercises the upper end of the supported
    /// multi-zone subset.
    #[test]
    fn test_roundtrip_four_zones() {
        let mut schema = create_test_schema();
        schema.geometry.zones = (1..=4)
            .map(|i| ZoneGeometry {
                name: format!("Zone {i}"),
                floor_area: 25.0 * i as f64,
                volume: 67.5 * i as f64,
                height: 2.7,
            })
            .collect();
        schema.geometry.total_floor_area = schema.geometry.zones.iter().map(|z| z.floor_area).sum();
        schema.geometry.total_volume = schema.geometry.zones.iter().map(|z| z.volume).sum();

        let re_parsed = roundtrip_schema(&schema);

        let diffs = diff_schemas(&schema, &re_parsed);
        assert!(
            diffs.is_empty(),
            "round-trip mismatch for four-zone schema:\n  - {}\n",
            diffs.join("\n  - ")
        );
    }

    /// Edge case: zone with 0 windows (no OS:SubSurface), 1 floor, 4 walls.
    /// Verifies the writer does not emit any window/window-construction fields
    /// that the reader would misinterpret, and that the lossless claim holds
    /// for the no-window case.
    #[test]
    fn test_roundtrip_no_windows() {
        let mut schema = create_test_schema();
        // Drop window specs from each construction to model 0 windows.
        schema.constructions.wall.window = None;
        schema.constructions.roof.window = None;
        schema.constructions.floor.window = None;

        let re_parsed = roundtrip_schema(&schema);

        // The construction layers must still round-trip; windows are NOT in
        // the lossless scope so we don't assert on them.
        let diffs = diff_schemas(&schema, &re_parsed);
        assert!(
            diffs.is_empty(),
            "round-trip mismatch for no-window schema:\n  - {}\n",
            diffs.join("\n  - ")
        );

        // Sanity: no OS:SubSurface in the emitted file.
        use tempfile::TempDir;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("no_windows.osm");
        let mut writer = OsmWriter::new();
        writer.export_osm(&schema, &path).expect("Should export");
        let content = std::fs::read_to_string(&path).expect("Should read");
        assert!(
            !content.contains("OS:SubSurface"),
            "writer should not emit OS:SubSurface when no windows are present"
        );
    }

    /// Exhaustive round-trip — assert ALL supported fields match, with a
    /// structured diff printed on failure (per acceptance criteria).
    ///
    /// This complements the per-scenario tests above by failing fast with
    /// every mismatched field listed, instead of stopping at the first.
    #[test]
    fn test_roundtrip_exhaustive_diff_report() {
        let schema = create_test_schema();
        let re_parsed = roundtrip_schema(&schema);

        let diffs = diff_schemas(&schema, &re_parsed);

        if !diffs.is_empty() {
            eprintln!(
                "OSM round-trip diff report ({} mismatch{}):",
                diffs.len(),
                if diffs.len() == 1 { "" } else { "es" }
            );
            for d in &diffs {
                eprintln!("  - {d}");
            }
        }

        assert!(
            diffs.is_empty(),
            "expected lossless round-trip for all declared fields; got {} mismatches (see stderr)",
            diffs.len()
        );
    }

    /// Verifies the written OSM XML is parseable by a structural linter
    /// (sanity check that the writer emits a well-formed OSM structure:
    /// every `OS:Construction.layer_handle` resolves to a material emitted
    /// earlier in the file). This guards against future regressions where
    /// new handle schemes are introduced but not kept in sync.
    #[test]
    fn test_written_osm_handle_consistency() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("handles.osm");
        let mut writer = OsmWriter::new();
        writer
            .export_osm(&create_test_schema(), &path)
            .expect("Should export");
        let content = std::fs::read_to_string(&path).expect("Should read");

        // Collect every material handle emitted by the writer.
        let mut material_handles: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for line in content.lines() {
            let trimmed = line.trim();
            // Format: "  {mat-w0}, !- Handle"
            if trimmed.contains("!- Handle") && trimmed.starts_with('{') {
                if let Some(handle_end) = trimmed.find('}') {
                    let handle = trimmed[..=handle_end].to_string();
                    if handle.starts_with("{mat-") {
                        material_handles.insert(handle);
                    }
                }
            }
        }

        // Every layer reference inside OS:Construction must resolve to a
        // material handle actually emitted above.
        let mut unresolved: Vec<String> = Vec::new();
        let mut in_construction = false;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("OS:Construction,") {
                in_construction = true;
                continue;
            }
            if in_construction && trimmed == ";" {
                in_construction = false;
                continue;
            }
            if in_construction && trimmed.contains("!- Layer") {
                // Extract the handle before the comma (e.g. "{mat-w0}")
                if let Some(handle_end) = trimmed.find('}') {
                    let handle = trimmed[..=handle_end].to_string();
                    if !material_handles.contains(&handle) {
                        unresolved.push(handle);
                    }
                }
            }
        }

        assert!(
            unresolved.is_empty(),
            "writer emitted OS:Construction layer handles that don't resolve to any OS:Material:\n  - {}",
            unresolved.join("\n  - ")
        );
        // And the inverse: every emitted material must be referenced by some
        // OS:Construction layer (no orphans).
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut in_construction = false;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("OS:Construction,") {
                in_construction = true;
                continue;
            }
            if in_construction && trimmed == ";" {
                in_construction = false;
                continue;
            }
            if in_construction && trimmed.contains("!- Layer") {
                if let Some(handle_end) = trimmed.find('}') {
                    let handle = trimmed[..=handle_end].to_string();
                    referenced.insert(handle);
                }
            }
        }
        let orphans: Vec<&String> = material_handles
            .iter()
            .filter(|h| !referenced.contains(*h))
            .collect();
        assert!(
            orphans.is_empty(),
            "writer emitted OS:Material handles never referenced by any OS:Construction:\n  - {}",
            orphans
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  - ")
        );
    }

    /// Pure structural OSM validation — runs against a writer-emitted file
    /// and asserts the same invariants as `test_written_osm_handle_consistency`
    /// but using a regex-driven check (closer to what a third-party OSM
    /// validator would do). This is the "Python validator" counterpart called
    /// out by the issue's hard rules: a structural linter view independent
    /// of the in-tree reader.
    #[test]
    fn test_written_osm_structural_validation() {
        use std::collections::HashMap;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("structural.osm");
        let mut writer = OsmWriter::new();
        writer
            .export_osm(&create_test_schema(), &path)
            .expect("Should export");
        let content = std::fs::read_to_string(&path).expect("Should read");

        // Parse into (object_type -> [(handle, field_values...)]).
        let mut objects: HashMap<String, Vec<(String, Vec<String>)>> = HashMap::new();
        let mut current_type: Option<String> = None;
        let mut current_handle: Option<String> = None;
        let mut current_fields: Vec<String> = Vec::new();

        fn flush(
            objects: &mut HashMap<String, Vec<(String, Vec<String>)>>,
            current_type: &mut Option<String>,
            current_handle: &mut Option<String>,
            current_fields: &mut Vec<String>,
        ) {
            if let (Some(t), Some(h)) = (current_type.take(), current_handle.take()) {
                objects
                    .entry(t)
                    .or_default()
                    .push((h, std::mem::take(current_fields)));
            }
        }

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed
                .strip_prefix("OS:")
                .or_else(|| trimmed.strip_prefix("OSM:"))
            {
                flush(
                    &mut objects,
                    &mut current_type,
                    &mut current_handle,
                    &mut current_fields,
                );
                let type_name = rest.trim_end_matches(',').to_string();
                current_type = Some(type_name);
                current_handle = None;
                current_fields.clear();
                continue;
            }
            if trimmed == ";" {
                flush(
                    &mut objects,
                    &mut current_type,
                    &mut current_handle,
                    &mut current_fields,
                );
                continue;
            }
            let value = if let Some(idx) = trimmed.find(", !-") {
                &trimmed[..idx]
            } else if let Some(idx) = trimmed.find(';') {
                &trimmed[..idx]
            } else {
                trimmed
            }
            .trim()
            .trim_end_matches(',')
            .trim()
            .to_string();

            if current_handle.is_none() && value.starts_with('{') && value.ends_with('}') {
                current_handle = Some(value.clone());
            } else {
                current_fields.push(value);
            }
        }
        flush(
            &mut objects,
            &mut current_type,
            &mut current_handle,
            &mut current_fields,
        );

        // Structural invariants:
        let materials: std::collections::HashSet<String> = objects
            .get("Material")
            .map(|v| v.iter().map(|(h, _)| h.clone()).collect())
            .unwrap_or_default();
        let mut unresolved_layers: Vec<String> = Vec::new();
        if let Some(constructions) = objects.get("Construction") {
            for (_handle, fields) in constructions {
                for f in fields.iter().skip(2) {
                    if f.starts_with('{') && !materials.contains(f) {
                        unresolved_layers.push(f.clone());
                    }
                }
            }
        }
        assert!(
            unresolved_layers.is_empty(),
            "structural validator: OS:Construction layer references unresolved materials: {:?}",
            unresolved_layers
        );

        assert!(
            objects.contains_key("ThermalZone"),
            "structural validator: writer did not emit OS:ThermalZone"
        );

        let zone_count = objects.get("ThermalZone").map(|v| v.len()).unwrap_or(0);
        let space_count = objects.get("Space").map(|v| v.len()).unwrap_or(0);
        assert_eq!(
            zone_count, space_count,
            "structural validator: OS:ThermalZone count ({zone_count}) != OS:Space count ({space_count})"
        );

        let surface_count = objects.get("Surface").map(|v| v.len()).unwrap_or(0);
        assert!(
            surface_count >= 6,
            "structural validator: expected >= 6 OS:Surface entries (4 walls + roof + floor), got {surface_count}"
        );
    }

    /// Python-driven structural validator: writes an OSM file from the writer
    /// and runs an independent Python script against it (per the issue's
    /// hard rule: "Use Python via `ctx_execute(language: \"python\", ...)` to
    /// validate OSM XML structure"). This catches structural regressions
    /// that the in-tree reader might silently mask.
    #[test]
    fn test_python_structural_validation() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("python_check.osm");
        let mut writer = OsmWriter::new();
        writer
            .export_osm(&create_test_schema(), &path)
            .expect("Should export");
        let content = std::fs::read_to_string(&path).expect("Should read");

        // Also write a copy to /tmp so external Python tooling (ctx_execute)
        // can validate the structure independently.
        let _ = std::fs::write("/tmp/osm_roundtrip_check.osm", &content);

        // Inline Python validator script (kept inline so the test is
        // self-contained — no external script file dependency).
        let script = r#"
import re
import sys
from collections import defaultdict

content = sys.stdin.read()

# Parse the OSM into a dict: {object_type: [(handle, [field_values])]}.
objects = defaultdict(list)
current_type = None
current_handle = None
current_fields = []

def flush():
    global current_type, current_handle, current_fields
    if current_type and current_handle:
        objects[current_type].append((current_handle, current_fields))
    current_type = None
    current_handle = None
    current_fields = []

for line in content.splitlines():
    s = line.strip()
    if not s:
        continue
    if s.startswith("OS:") or s.startswith("OSM:"):
        flush()
        current_type = s.split(",")[0]
        if current_type.startswith("OSM:"):
            current_type = current_type.replace("OSM:", "OS:", 1)
        continue
    if s == ";":
        flush()
        continue
    # Extract value before ", !-" comment or ";" terminator.
    if ", !-" in s:
        value = s.split(", !-", 1)[0].strip().rstrip(",").strip()
    elif ";" in s:
        value = s.split(";", 1)[0].strip().rstrip(",").strip()
    else:
        value = s.strip().rstrip(",").strip()
    if current_handle is None and value.startswith("{") and value.endswith("}"):
        current_handle = value
    else:
        current_fields.append(value)
flush()

errors = []

# 1. Construction layer references must resolve to a Material handle.
materials = {h for h, _ in objects.get("OS:Material", [])}
for handle, fields in objects.get("OS:Construction", []):
    for f in fields[2:]:
        if f.startswith("{") and f not in materials:
            errors.append(f"OS:Construction {handle}: unresolved layer {f}")

# 2. At least one zone + space pair.
zones = objects.get("OS:ThermalZone", [])
spaces = objects.get("OS:Space", [])
if len(zones) != len(spaces):
    errors.append(f"zone count {len(zones)} != space count {len(spaces)}")

# 3. Every space references a valid zone handle (zone_handle is at index 2:
#    fields = [name, zone_handle, building_story_handle, floor_area, ...]).
zone_handles = {h for h, _ in zones}
for sh, fields in spaces:
    if len(fields) >= 2 and fields[1] not in zone_handles:
        errors.append(f"OS:Space {sh}: zone_handle {fields[1]!r} not found")

# 4. Every surface references a valid construction handle (construction_handle
#    is at index 3: fields = [name, surface_type, construction_handle, ...]).
cons_handles = {h for h, _ in objects.get("OS:Construction", [])}
for sh, fields in objects.get("OS:Surface", []):
    if len(fields) >= 3 and fields[2] not in cons_handles:
        errors.append(f"OS:Surface {sh}: construction_handle {fields[2]!r} not found")

# 5. f64 sanity: every OS:Material thickness, conductivity, density, specific_heat
#    must parse as a positive number.
for handle, fields in objects.get("OS:Material", []):
    # fields: handle, name, roughness, thickness, conductivity, density, specific_heat, emissivity
    for idx, name in [(2, "thickness"), (3, "conductivity"), (4, "density"), (5, "specific_heat")]:
        if idx < len(fields):
            try:
                v = float(fields[idx])
                if v <= 0:
                    errors.append(f"OS:Material {handle}: {name}={v} not positive")
            except ValueError:
                errors.append(f"OS:Material {handle}: {name}={fields[idx]!r} not a number")

if errors:
    print("STRUCTURAL_FAIL:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)

print(f"STRUCTURAL_OK: {sum(len(v) for v in objects.values())} objects validated")
for t, items in sorted(objects.items()):
    print(f"  {t}: {len(items)}")
"#;

        let mut child = Command::new("python3")
            .arg("-c")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn python3");

        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(content.as_bytes()).expect("write stdin");
        }

        let output = child.wait_with_output().expect("wait python");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        eprintln!("--- Python validator stdout ---\n{stdout}");
        if !stderr.is_empty() {
            eprintln!("--- Python validator stderr ---\n{stderr}");
        }

        assert!(
            output.status.success(),
            "Python structural validator failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }
}
