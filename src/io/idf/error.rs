// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Error types for EnergyPlus IDF parsing.
//!
//! Follows the same `thiserror` pattern used by [`crate::interop::osm::error::OsmError`]
//! so the rest of the crate can treat interop errors uniformly.

use thiserror::Error;

/// Errors produced by the IDF lexer, parser, and (in future phases)
/// the `IdfFile → SimulationSchema` converter.
#[derive(Error, Debug)]
pub enum IdfError {
    /// Wrapped I/O failure (e.g. file not found, permission denied).
    #[error("Failed to read IDF file: {0}")]
    Io(#[from] std::io::Error),

    /// Lexer or parser failure with the offending line number.
    ///
    /// `line` is 1-indexed to match what EnergyPlus itself prints in its
    /// own error output, which makes cross-referencing user-facing
    /// diagnostics easier.
    #[error("IDF parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    /// IDF → SimulationSchema conversion failure.
    #[error("IDF conversion error: {0}")]
    Conversion(String),

    /// Reserved for object types we explicitly decide not to support even
    /// after a future feature extension. For the MVP, unknown object types
    /// are silently captured into [`crate::io::idf::parser::IdfObject`] so
    /// `UnsupportedObject` is currently only constructed by tests.
    #[error("Unsupported IDF object type: {0}")]
    UnsupportedObject(String),

    /// Reserved for EnergyPlus `Version` values outside the allow-list
    /// (`24-2`, `25-1`, `25-2`) per `docs/idf-import-design.md` §4.3.
    /// Introduced in issue #1435.
    #[error("Unsupported EnergyPlus version: {0} (allowed: 24-2, 25-1, 25-2)")]
    UnsupportedVersion(String),
}

impl IdfError {
    /// Convenience constructor for parse errors — avoids the verbosity of
    /// struct-literal `IdfError::Parse { line, message }` at every call site.
    pub fn parse_error(line: usize, message: impl Into<String>) -> Self {
        IdfError::Parse {
            line,
            message: message.into(),
        }
    }

    /// Convenience constructor for conversion errors.
    pub fn conversion_error(message: impl Into<String>) -> Self {
        IdfError::Conversion(message.into())
    }

    /// Convenience constructor for unsupported-object errors.
    pub fn unsupported_object(object_type: impl Into<String>) -> Self {
        IdfError::UnsupportedObject(object_type.into())
    }

    /// Convenience constructor for unsupported-version errors.
    pub fn unsupported_version(version: impl Into<String>) -> Self {
        IdfError::UnsupportedVersion(version.into())
    }
}
