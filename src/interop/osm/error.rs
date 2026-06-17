// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Error types for OSM parsing and writing.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OsmError {
    #[error("Failed to read file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Invalid object: {0}")]
    InvalidObject(String),

    #[error("Missing required field '{field}' in {object}")]
    MissingField { object: String, field: String },

    #[error("Unknown object type: {0}")]
    UnknownObjectType(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("Export error: {0}")]
    ExportError(String),
}

impl OsmError {
    pub fn parse_error(line: usize, message: impl Into<String>) -> Self {
        OsmError::ParseError {
            line,
            message: message.into(),
        }
    }

    pub fn missing_field(object: impl Into<String>, field: impl Into<String>) -> Self {
        OsmError::MissingField {
            object: object.into(),
            field: field.into(),
        }
    }

    pub fn invalid_object(msg: impl Into<String>) -> Self {
        OsmError::InvalidObject(msg.into())
    }
}
