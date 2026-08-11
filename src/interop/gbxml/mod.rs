// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! gbXML import/export support for Fluxion BIM integration.
//!
//! This module provides functionality to read and write gbXML (Green Building XML)
//! files for integration with BIM tools like Revit and AutoCAD.
//!
//! # Overview
//!
//! gbXML is an industry-standard schema for exchanging building geometry and
//! construction data between BIM and energy analysis software. This module
//! enables fluxion to:
//!
//! - **Import** building models from BIM tools via gbXML files
//! - **Export** fluxion simulation models as gbXML for other tools
//!
//! # gbXML Structure
//!
//! gbXML files organize building data hierarchically:
//!
//! ```text
//! Campus
//! └── Building
//!      └── BuildingStorey (floor)
//!           └── Space (thermal zone)
//!                └── Surface (wall/roof/floor)
//!                     ├── Construction (layer assembly)
//!                     ├── Layer
//!                     └── Material
//! ```
//!
//! # Example
//!
//! ## Import
//!
//! ```ignore
//! use fluxion::interop::gbxml::import_gbxml;
//!
//! // Read a gbXML file from a BIM tool
//! let schema = import_gbxml("building_from_revit.xml")?;
//! println!("Imported {} zones", schema.geometry.zones.len());
//! ```
//!
//! ## Export
//!
//! ```ignore
//! use fluxion::interop::gbxml::export_gbxml;
//!
//! // Export a fluxion model to gbXML
//! export_gbxml(&schema, "fluxion_model.xml")?;
//! ```
//!
//! # Limitations
//!
//! - Single building only (Campus with multiple buildings not supported)
//! - Rectangular geometry only (complex CAD surfaces simplified)
//! - No HVAC systems (only building envelope exported)
//! - No schedules (default schedules assumed)
//!
//! # References
//!
//! - [gbXML Schema Documentation](https://www.gbxml.org/schema_doc/6.01/GreenBuildingXML_Ver6.01.html)
//! - [gbXML Official Site](https://www.gbxml.org/)
//!
//! # Module Structure
//!
//! - [`types`] - gbXML schema type definitions
//! - [`error`] - Error types for gbXML operations
//! - [`reader`] - Import gbXML files into fluxion schema
//! - [`writer`] - Export fluxion schema to gbXML files

pub mod error;
pub mod reader;
pub mod types;
pub mod writer;

pub use error::GbXmlError;
pub use reader::{
    import_gbxml, import_gbxml_with_limits, parse_gbxml, parse_gbxml_with_limits, GbXmlReader,
};
pub use writer::{export_gbxml, GbXmlWriter};
