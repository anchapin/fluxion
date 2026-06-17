// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio Model (OSM) interoperability for Fluxion.
//!
//! This module provides functionality to import and export OpenStudio Model files
//! (.osm) for direct interoperability with the OpenStudio SDK ecosystem.
//!
//! # Scope
//!
//! This module supports:
//!
//! ## Import (OSM -> Fluxion)
//! - Parse OSM file format (OpenStudio IDD schema)
//! - Map OS:Space, OS:Surface, OS:SubSurface to fluxion model objects
//! - Map OS:Construction, OS:Material to fluxion material library
//! - Map OS:ThermalZone to fluxion Zone
//! - Extract schedule and load data for occupancy-driven simulations
//!
//! ## Export (Fluxion -> OSM)
//! - Serialize fluxion model back to OSM for round-trip or hand-off to OpenStudio Measures
//!
//! # OSM Format
//!
//! OSM files use a line-oriented key-value format similar to IDF but with
//! OpenStudio-specific IDD schema. Objects are defined with their type name
//! followed by comma-separated fields.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::osm::{import_osm, export_osm, OsmReader, OsmWriter};
//!
//! // Import an OSM file
//! let schema = import_osm("building.osm")?;
//!
//! // Export to OSM
//! let writer = OsmWriter::new();
//! writer.export_osm(&schema, "output.osm")?;
//! ```
//!
//! # Limitations
//!
//! This is an initial implementation with the following known limitations:
//!
//! ## Import
//! - Limited HVAC system extraction
//! - Simplified zone mapping
//! - Basic schedule extraction
//!
//! ## Export
//! - Limited HVAC system export
//! - Basic schedule representation
//! - Simplified construction export
//! - Window geometry not fully exported

pub mod error;
pub mod types;
pub mod reader;
pub mod writer;

pub use error::OsmError;
pub use reader::{OsmReader, import_osm};
pub use writer::{OsmWriter, export_osm};
