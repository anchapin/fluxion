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
//! # Round-Trip Stability (issue #1340)
//!
//! Writer→reader round-trip is **stable** for single- and multi-zone schemas
//! within the supported subset. Tests live in
//! `src/interop/osm/writer.rs::tests` (`test_roundtrip_single_zone`,
//! `test_roundtrip_two_zones`, `test_roundtrip_four_zones`,
//! `test_roundtrip_no_windows`, `test_roundtrip_exhaustive_diff_report`).
//!
//! ## Lossless fields
//!
//! The following `SimulationSchemaV1` fields round-trip byte-equivalent
//! field-wise (f64 comparisons within `1e-6` absolute or relative tolerance):
//!
//! - `metadata.name` (via `OS:Building.Name`)
//! - `geometry.zones[*].name`, `.floor_area`, `.volume`, `.height`
//!   (via `OS:ThermalZone.Name` and `OS:Space.Floor Area` / `Volume`)
//! - `geometry.total_floor_area`, `.total_volume` (sum of zone values)
//! - `geometry.number_of_floors` (via `OS:Building.Number of Floors`)
//! - `geometry.floor_height` (computed from `total_volume / total_floor_area`)
//! - `constructions.{wall,roof,floor}.layers[*]`
//!   (`.name`, `.thickness`, `.conductivity`, `.density`, `.specific_heat`)
//! - `controls.zone_control.heating_setpoint`
//!   (via `OS:Thermostat.Heating Setpoint Temperature`, one per zone)
//! - `controls.zone_control.cooling_setpoint`
//!   (via `OS:Thermostat.Cooling Setpoint Temperature`, one per zone)
//! - `weather` for the `TmyLocation` variant (lat/lon pair, within tolerance)
//!
//! ## Known lossy fields
//!
//! The following fields are NOT guaranteed to round-trip; they fall back to
//! `Default` during read:
//!
//! - `metadata.description`, `.author`, `.created_at`
//! - `schedules.*` (writer does not yet emit `OS:Schedule:*`)
//! - `constructions.{wall,roof,floor}.window`
//!   (no `OS:SubSurface` emission in the supported subset)
//! - `constructions.interzone` (not emitted)
//! - `weather` for the `EpwFile` and `Inline` variants
//! - `output.*` (simulation results, not part of model file)
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
pub mod parser;
pub mod reader;
pub mod types;
pub mod writer;

pub use error::OsmError;
pub use parser::OsmParser;
pub use reader::{import_osm, import_osm_with_limits, OsmReader};
pub use writer::{export_osm, OsmWriter};
