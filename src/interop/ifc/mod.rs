// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! IFC4 STEP geometry import and export scaffold (issue #1343 + #1612 + #1908 + #2309).
//!
//! Parses IFC4 entity types from a STEP physical file (ISO 10303-21)
//! and maps them onto Fluxion's [`SimulationSchemaV1`]. Also exports
//! [`SimulationSchemaV1`] back to IFC4 STEP format.
//!
//! # Scope (issue #1343 + #1612 + #1908 + #2309)
//!
//! - IFC4 only — IFC2X3 is **not** supported (deferred).
//! - Entities typed: [`IfcBuilding`], [`IfcBuildingStorey`],
//!   [`IfcSpace`], [`IfcWall`], [`IfcSlab`], [`IfcRoof`],
//!   [`IfcWindow`], [`IfcDoor`]
//!   (issue #1612 extended the initial #1343 scaffold with building/storey,
//!   issue #2309 added window/door support).
//!   Everything else is captured generically into [`GenericEntity`] so callers
//!   can inspect or forward it.
//! - Material handling: `IfcMaterialLayerSetUsage` → list of
//!   `(material, thickness)` pairs via `IfcRelAssociatesMaterial`.
//! - Zone geometry is extracted via [`IfcGeometryParser`] in [`geometry`]
//!   using `IfcRelContainedInSpatialStructure` for zone element assignment.
//!
//! # Lossless-field contract (issue #2309)
//!
//! The IFC import-export round-trip preserves the following fields:
//!
//! | Field | Preservation guarantee |
//! |-------|----------------------|
//! | Zone count | Exact — one [`ZoneGeometry`] per [`IfcSpace`] |
//! | Zone names | Exact — `IfcSpace.Name` → `ZoneGeometry.name` |
//! | Floor area | Within 0.5 % — falls back to 24 m² default when footprint cannot be decoded |
//! | Material layers | Exact — [`ConstructionLayer`] per [`IfcMaterialLayer`] (thickness, material name) |
//! | Wall construction | Exact — layers from `IfcMaterialLayerSet` associated with walls |
//! | Roof construction | Exact — layers from `IfcMaterialLayerSet` associated with roofs |
//! | Floor construction | Exact — layers from `IfcMaterialLayerSet` associated with slabs of type `.FLOOR.` |
//!
//! Round-trip test: `import_ifc` → `export_ifc` → `import_ifc` must preserve
//! zone count, floor area (within 0.5 %), and material layer counts.
//!
//! # Module structure
//!
//! - [`error`] — IFC-specific error type, follows the `thiserror` pattern
//!   used by the rest of `crate::interop`.
//! - [`step_lexer`] — Character-level tokenizer for ISO 10303-21 STEP
//!   physical files. Yields [`RawEntity`] records (id + name + raw arg
//!   body) suitable for the typed parser in [`parser`].
//! - [`parser`] — Builds an [`IfcModel`] from the lexer's stream and
//!   extracts building/storey/wall/slab/roof/window/door/space entities plus the
//!   supporting material and relationship records.
//! - [`geometry`] — Extracts spatial hierarchy and zone geometry
//!   (`IfcGeometryParser`).
//! - [`mapping`] — Converts an [`IfcModel`] into a
//!   [`SimulationSchemaV1`].
//! - [`writer`] — Exports a [`SimulationSchemaV1`] to an IFC4 STEP
//!   physical file, including [`IfcBuilding`], [`IfcBuildingStorey`],
//!   [`IfcSpace`], [`IfcBuildingElementProxy`], [`IfcWindow`], [`IfcDoor`],
//!   and [`IfcMaterialLayer`] entities (issue #1908 + #2309).
//!
//! # References
//!
//! - IFC4 ADD2 schema (TC1 release):
//!   <https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/>
//! - ISO 10303-21 (STEP physical file format):
//!   <https://en.wikipedia.org/wiki/ISO_10303-21>
//!
//! # Example (import)
//!
//! ```ignore
//! use fluxion::interop::ifc::import_ifc;
//!
//! let schema = import_ifc("tests/fixtures/ifc/sample.ifc")?;
//! assert_eq!(schema.geometry.zones.len(), 1);
//! ```
//!
//! # Example (export)
//!
//! ```ignore
//! use fluxion::interop::ifc::export_ifc;
//!
//! export_ifc(&schema, "output.ifc")?;
//! ```

pub mod error;
pub mod geometry;
pub mod mapping;
pub mod parser;
pub mod step_lexer;
pub mod writer;

pub use error::IfcError;
pub use geometry::IfcGeometryParser;
pub use mapping::{import_ifc, import_ifc_with_limits, IfcToSchema};
pub use parser::{
    IfcBuilding, IfcBuildingStorey, IfcDoor, IfcModel, IfcParser, IfcRoof, IfcSlab, IfcSpace,
    IfcWall, IfcWindow, MaterialLayerSpec,
};
pub use step_lexer::{tokenize, tokenize_with_schema_and_limits, RawEntity};
pub use writer::{export_ifc, write_ifc_file, IfcWriter};
