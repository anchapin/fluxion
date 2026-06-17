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

    #[test]
    fn test_osm_reader_default() {
        let reader = OsmReader::new();
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
}
