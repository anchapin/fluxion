// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! IFC4 STEP geometry import scaffold (issue #1343 + #1612).
//!
//! Parses the IFC4 entity types from a STEP physical file (ISO 10303-21)
//! and maps them onto Fluxion's [`SimulationSchemaV1`].
//!
//! # Scope (issue #1343 + #1612)
//!
//! - IFC4 only — IFC2X3 is **not** supported (deferred).
//! - Entities typed: [`IfcBuilding`], [`IfcBuildingStorey`],
//!   [`IfcSpace`], [`IfcWall`], [`IfcSlab`], [`IfcRoof`]
//!   (issue #1612 extended the initial #1343 scaffold with building/storey).
//!   Everything else is captured generically into [`GenericEntity`] so callers
//!   can inspect or forward it.
//! - Material handling: `IfcMaterialLayerSetUsage` → list of
//!   `(material, thickness)` pairs via `IfcRelAssociatesMaterial`.
//! - Zone geometry is extracted via [`IfcGeometryParser`] in [`geometry`]
//!   using `IfcRelContainedInSpatialStructure` for zone element assignment.
//!
//! # Module structure
//!
//! - [`error`] — IFC-specific error type, follows the `thiserror` pattern
//!   used by the rest of `crate::interop`.
//! - [`step_lexer`] — Character-level tokenizer for ISO 10303-21 STEP
//!   physical files. Yields [`RawEntity`] records (id + name + raw arg
//!   body) suitable for the typed parser in [`parser`].
//! - [`parser`] — Builds an [`IfcModel`] from the lexer's stream and
//!   extracts building/storey/wall/slab/roof/space entities plus the
//!   supporting material and relationship records.
//! - [`geometry`] — Extracts spatial hierarchy and zone geometry
//!   (`IfcGeometryParser`).
//! - [`mapping`] — Converts an [`IfcModel`] into a
//!   [`SimulationSchemaV1`].
//!
//! # References
//!
//! - IFC4 ADD2 schema (TC1 release):
//!   <https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/>
//! - ISO 10303-21 (STEP physical file format):
//!   <https://en.wikipedia.org/wiki/ISO_10303-21>
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::ifc::import_ifc;
//!
//! let schema = import_ifc("tests/fixtures/ifc/sample.ifc")?;
//! assert_eq!(schema.geometry.zones.len(), 1);
//! ```

pub mod error;
pub mod geometry;
pub mod mapping;
pub mod parser;
pub mod step_lexer;

pub use error::IfcError;
pub use geometry::IfcGeometryParser;
pub use mapping::{import_ifc, IfcToSchema};
pub use parser::{IfcBuilding, IfcBuildingStorey, IfcModel, IfcParser, IfcRoof, IfcSlab, IfcSpace, IfcWall, MaterialLayerSpec};
pub use step_lexer::{tokenize, RawEntity};
