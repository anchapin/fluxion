// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio OSM (OpenStudio Model) file import/export support.
//!
//! This module provides functionality to read and write OpenStudio Model files
//! (.osm format), enabling interoperability between Fluxion and the broader
//! building energy modeling ecosystem.
//!
//! # OSM Format
//!
//! OpenStudio Model files are XML documents that describe building geometry,
//! constructions, schedules, HVAC systems, and other model components.
//! See [OpenStudio documentation](https://nrel.github.io/OpenStudio-user-documentation/)
//! for the full schema reference.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::osm::{OsmReader, OsmWriter};
//! use fluxion::api::schema::SimulationSchema;
//!
//! // Read an OSM file
//! let reader = OsmReader::new();
//! let schema = reader.from_path("model.osm").unwrap();
//!
//! // Write a schema to OSM
//! let writer = OsmWriter::new();
//! writer.write(&schema, "output.osm").unwrap();
//! ```

pub mod error;
pub mod reader;
pub mod types;
pub mod writer;

pub use error::OsmError;
pub use reader::OsmReader;
pub use types::{
    OsmBuilding, OsmConstruction, OsmMaterial, OsmModel, OsmSchedule, OsmSite,
    OsmSpace, OsmSubSurface, OsmSurface, OsmThermostat, OsmThermalZone, OsmVertex,
    OsmWeatherFile,
};
pub use writer::OsmWriter;

use crate::api::schema::SimulationSchema;
use std::path::Path;

impl SimulationSchema {
    pub fn from_osm(path: &Path) -> Result<Self, OsmError> {
        let mut reader = OsmReader::new();
        reader.from_path(path)
    }

    pub fn to_osm(&self, path: &Path) -> Result<(), OsmError> {
        let writer = OsmWriter::new();
        writer.write(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_OSM: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<OpenStudioApplication version="1.0.0">
  <OS:Version version="1.0.0"/>
  <OS:Site name="Site" latitude="39.739200" longitude="-104.990300" time_Zone="-7.0" elevation="1609.0" terrain="Suburbs"/>
  <OS:Building name="Test Building" north_Axis="0.0" terrain="Suburbs" floorspaces_Story="1" floor_Area="100.00" buildingType="Office"/>
  <OS:Material name="Concrete" material_Type="StandardOpaqueMaterial" thickness="0.1500" conductivity="1.4000" density="2300.00" specific_Heat="880.00" roughness="MediumRough"/>
  <OS:Material name="Insulation" material_Type="StandardOpaqueMaterial" thickness="0.0500" conductivity="0.0400" density="20.00" specific_Heat="840.00" roughness="MediumRough"/>
  <OS:Construction name="Wall Construction">
    <OS:Layer name="Concrete"/>
    <OS:Layer name="Insulation"/>
  </OS:Construction>
  <OS:ThermalZone name="Zone 1" zone_Name="Zone 1" multiplier="1" volume="270.00" floor_Area="100.00"/>
  <OS:ThermostatSetpointDualSetpoint name="Thermostat" heating_Setpoint_Temperature="20.0" cooling_Setpoint_Temperature="24.0"/>
</OpenStudioApplication>"#;

    #[test]
    fn test_osm_reader_default() {
        let mut reader = OsmReader::new();
        assert!(reader.from_str("<OpenStudioApplication/>").is_ok());
    }

    #[test]
    fn test_osm_writer_default() {
        let writer = OsmWriter::new();
        assert!(writer.to_string(&SimulationSchema::default()).is_ok());
    }

    #[test]
    fn test_osm_error_display() {
        let err = OsmError::MissingRequired("OS:Building".to_string());
        assert_eq!(
            format!("{}", err),
            "Missing required OSM object: OS:Building"
        );
    }

    #[test]
    fn test_round_trip() {
        let schema = SimulationSchema::default();
        let writer = OsmWriter::new();
        let xml = writer.to_string(&schema).unwrap();

        let mut reader = OsmReader::new();
        let result = reader.from_str(&xml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_minimal_osm() {
        let mut reader = OsmReader::new();
        let result = reader.from_str(MINIMAL_OSM);
        assert!(result.is_ok());

        let schema = result.unwrap();
        match schema {
            SimulationSchema::V1(v1) => {
                assert_eq!(v1.metadata.name, "Test Building");
                assert!(v1.geometry.total_floor_area > 0.0);
            }
        }
    }

    #[test]
    fn test_round_trip_preserves_materials() {
        // Write default schema to XML
        let schema = SimulationSchema::default();
        let writer = OsmWriter::new();
        let xml = writer.to_string(&schema).unwrap();

        // Read it back
        let mut reader = OsmReader::new();
        let result = reader.from_str(&xml);
        assert!(result.is_ok());

        // Verify we can write it again (round-trip)
        let schema2 = result.unwrap();
        let xml2 = writer.to_string(&schema2).unwrap();
        assert!(!xml2.is_empty());

        // Both XMLs should contain OS:Material elements
        assert!(xml.contains("OS:Material") && xml2.contains("OS:Material"));
    }

    #[test]
    fn test_parse_osm_with_site() {
        let osm_with_site = r#"<?xml version="1.0" encoding="UTF-8"?>
<OpenStudioApplication version="1.0.0">
  <OS:Version version="1.0.0"/>
  <OS:Site name="Denver Site" latitude="39.739200" longitude="-104.990300"/>
  <OS:Building name="Office Building" north_Axis="15.0" terrain="City"/>
</OpenStudioApplication>"#;

        let mut reader = OsmReader::new();
        let result = reader.from_str(osm_with_site);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_site_data() {
        let osm = r#"<?xml version="1.0" encoding="UTF-8"?>
<OpenStudioApplication version="1.0.0">
  <OS:Site name="Test Site" latitude="40.0" longitude="-105.0"/>
</OpenStudioApplication>"#;

        let mut reader = OsmReader::new();
        let result = reader.from_str(osm);
        assert!(result.is_ok());
    }
}
