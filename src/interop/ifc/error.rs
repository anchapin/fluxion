// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Error types for the IFC4 STEP import scaffold.
//!
//! Mirrors the `thiserror` pattern used by
//! [`crate::interop::osm::error::OsmError`] so the rest of the crate can
//! treat interop errors uniformly.

use thiserror::Error;

/// Errors produced by the IFC4 STEP lexer, parser, and mapper.
#[derive(Error, Debug)]
pub enum IfcError {
    /// Wrapped I/O failure (e.g. file not found, permission denied).
    #[error("Failed to read IFC file: {0}")]
    Io(#[from] std::io::Error),

    /// Lexer or parser failure with the offending line number.
    ///
    /// `line` is 1-indexed to match what STEP processors themselves
    /// print in their own diagnostics.
    #[error("IFC parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    /// The input file does not start with `ISO-10303-21;` or is missing
    /// the required `FILE_SCHEMA(('IFC4'))` declaration.
    #[error("Unsupported IFC schema: expected 'IFC4', got '{0}'")]
    UnsupportedSchema(String),

    /// The model contains a reference (`#N`) that has no corresponding
    /// entity definition. STEP files must define every id they reference.
    #[error("Unresolved IFC reference: #{0}")]
    UnresolvedReference(u64),

    /// The `IfcParser` could not classify an entity into one of the four
    /// MVP types (wall/slab/roof/space). The entity is still captured into
    /// [`crate::interop::ifc::parser::IfcModel::entities`] so callers can
    /// inspect it; this variant is reserved for entities we *explicitly*
    /// decide to reject (e.g. `IfcWindow` until #1121 follow-up lands).
    #[error("Unsupported IFC entity: {0}")]
    UnsupportedEntity(String),

    /// Conversion from [`crate::interop::ifc::parser::IfcModel`] to
    /// [`crate::api::schema::SimulationSchemaV1`] failed (e.g. a wall
    /// references a material layer set that has no layers).
    #[error("IFC conversion error: {0}")]
    Conversion(String),
}

impl IfcError {
    /// Convenience constructor for parse errors.
    pub fn parse_error(line: usize, message: impl Into<String>) -> Self {
        IfcError::Parse {
            line,
            message: message.into(),
        }
    }

    /// Convenience constructor for unsupported-entity errors.
    pub fn unsupported_entity(entity: impl Into<String>) -> Self {
        IfcError::UnsupportedEntity(entity.into())
    }

    /// Convenience constructor for conversion errors.
    pub fn conversion_error(message: impl Into<String>) -> Self {
        IfcError::Conversion(message.into())
    }
}