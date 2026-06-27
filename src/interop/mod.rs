// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FMI (Functional Mock-up Interface) interoperability for Fluxion.
//!
//! This module provides FMI 2.0 Co-Simulation export capabilities for
//! the Fluxion building energy modeling engine.
//!
//! # Scope
//!
//! The current implementation generates valid FMI 2.0 `.fmu` archives
//! for single- or multi-zone thermal networks.
//!
//! ## Export Mode (Fluxion → FMU)
//! - **Multi-zone** support — per-zone (outdoor temperature, solar gains,
//!   internal gains) inputs and (zone temperature, heating/cooling load)
//!   outputs.  N zones => 7 × N ScalarVariables (#1339).
//! - **Configurable communication timestep** — set on `FmiConfig`
//!   (default 3600 s; accepts 60 / 300 / 600 / 3600 s).
//! - **Standalone Co-Simulation FMU** — declared with
//!   `needsExecutionTool="true"` so the master drives the simulation.
//!
//! ## Known limitations
//! - **FMI 3.0** features (Hybrid Co-Simulation, terminals) are not
//!   implemented — out of scope until upstream tooling stabilizes.
//! - **FMU import** (`FmiMode::Import`) is reserved for a future issue.
//!
//! # FMI Standard
//!
//! Implements FMI 2.0 for Co-Simulation (export).  See
//! <https://fmi-standard.org/>.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::fmi::{FmiExporter, FmiConfig, ZoneVariables};
//!
//! let exporter = FmiExporter::new()
//!     .with_zones(vec![
//!         ZoneVariables::new("zone"),
//!         ZoneVariables::new("bedroom"),
//!         ZoneVariables::new("kitchen"),
//!     ]);
//! exporter.export_fmu("fluxion_three_zone.fmu")?;
//! ```

pub mod fmi;
pub mod gbxml;
pub mod ifc;
pub mod osm;

pub use fmi::{FmiConfig, FmiExporter, FmiMode, ZoneVariables};
pub use gbxml::{export_gbxml, import_gbxml, GbXmlError, GbXmlReader, GbXmlWriter};
pub use ifc::{import_ifc, IfcError, IfcModel, IfcParser, IfcToSchema};
pub use osm::{export_osm, import_osm, OsmError, OsmReader, OsmWriter};
