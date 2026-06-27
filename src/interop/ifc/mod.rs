// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! IFC4 STEP geometry import scaffold (issue #1343).
//!
//! Parses the four IFC4 entity types required by the issue
//! ([`IfcWall`], [`IfcSlab`], [`IfcRoof`], [`IfcSpace`]) from a STEP
//! physical file (ISO 10303-21) and maps them onto Fluxion's
//! [`SimulationSchemaV1`].
//!
//! # Scope (issue #1343 — MVP scaffold)
//!
//! - IFC4 only — IFC2X3 is **not** supported (deferred).
//! - Only the four `IfcSharedBldgElements` (wall/slab/roof) and the
//!   `IfcSpace` (zone) entities are typed; everything else is captured
//!   generically into [`EntityRecord`] so callers can inspect or forward it.
//! - Material handling is intentionally minimal: a single
//!   `IfcMaterialLayerSetUsage` → list of `(material, thickness)` pairs via
//!   the matching `IfcRelAssociatesMaterial`. No property sets, no
//!   `IfcMaterialList`, no conditional `IfcMaterialLayerSet` rewrites.
//! - Geometry is reduced to entity counts and (for `IfcSpace`) footprint
//!   area; per-wall vertices are not consumed. The per-surface conduction
//!   solver reads its own geometry, so the scaffold only needs to populate
//!   the right number of `SurfaceConstruction`s with the right thicknesses.
//!
//! # Module structure
//!
//! - [`error`] — IFC-specific error type, follows the `thiserror` pattern
//!   used by the rest of `crate::interop`.
//! - [`step_lexer`] — Character-level tokenizer for ISO 10303-21 STEP
//!   physical files. Yields [`RawEntity`] records (id + name + raw arg
//!   body) suitable for the typed parser in [`parser`].
//! - [`parser`] — Builds an [`IfcModel`] from the lexer's stream and
//!   extracts the four wall/slab/roof/space entities plus the supporting
//!   material and relationship records.
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
pub mod mapping;
pub mod parser;
pub mod step_lexer;

pub use error::IfcError;
pub use mapping::{import_ifc, IfcToSchema};
pub use parser::{IfcModel, IfcParser, IfcRoof, IfcSlab, IfcSpace, IfcWall, MaterialLayerSpec};
pub use step_lexer::{tokenize, RawEntity};