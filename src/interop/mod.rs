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
//! for single- or multi-zone thermal networks, and can re-import them
//! (`FmiMode::Import`) for co-simulation mastered by Fluxion (#1708).
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
//! ## Import Mode (FMU → Fluxion, `FmiMode::Import`)
//! - [`FmiImporter`] parses a `.fmu` archive's `modelDescription.xml`
//!   (via `quick-xml`) and rebuilds a [`ThermalModel`] with the correct
//!   zone count and communication timestep (#1708).
//! - [`FmuCoSimulationMaster`] drives the re-imported model one
//!   `doStep` at a time, forwarding per-timestep weather inputs to
//!   `ThermalModel::step_physics` and reporting zone temperature +
//!   heating/cooling loads.
//!
//! ## Known limitations
//! - **FMI 3.0** features (Hybrid Co-Simulation, terminals) are not
//!   implemented — out of scope until upstream tooling stabilizes.
//!
//! # FMI Standard
//!
//! Implements FMI 2.0 for Co-Simulation (export and import).  See
//! <https://fmi-standard.org/>.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::fmi::{FmiConfig, FmiExporter, FmiImporter, ZoneVariables};
//!
//! // Export a 3-zone FMU …
//! let exporter = FmiExporter::new()
//!     .with_zones(vec![
//!         ZoneVariables::new("zone"),
//!         ZoneVariables::new("bedroom"),
//!         ZoneVariables::new("kitchen"),
//!     ]);
//! exporter.export_fmu("fluxion_three_zone.fmu")?;
//!
//! // … then re-import it (#1708).
//! let fmu = FmiImporter::new().import("fluxion_three_zone.fmu".as_ref())?;
//! assert_eq!(fmu.zone_count(), 3);
//! ```

pub mod fmi;
pub mod gbxml;
pub mod ifc;
pub mod osm;

pub use fmi::{
    import_fmu, FmiConfig, FmiExporter, FmiImporter, FmiMode, FmuCoSimulationMaster, FmuInputs,
    FmuOutputs, ImportedFmu, ImportedModelDescription, ImportedScalarVariable, ZoneVariables,
};
pub use gbxml::{export_gbxml, import_gbxml, GbXmlError, GbXmlReader, GbXmlWriter};
pub use ifc::{export_ifc, import_ifc, IfcError, IfcGeometryParser, IfcModel, IfcParser, IfcToSchema, IfcWriter};
pub use osm::{export_osm, import_osm, OsmError, OsmReader, OsmWriter};
