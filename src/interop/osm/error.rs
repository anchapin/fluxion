// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenStudio OSM file error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OsmError {
    #[error("Failed to parse OSM XML: {0}")]
    Parse(String),

    #[error("Missing required OSM object: {0}")]
    MissingRequired(String),

    #[error("Unsupported OSM version: {0}")]
    UnsupportedVersion(String),

    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    #[error("Invalid material layer at index {index}: {message}")]
    InvalidMaterialLayer { index: usize, message: String },

    #[error("Missing required field '{field}' in {object}")]
    MissingField { field: String, object: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<serde_json::Error> for OsmError {
    fn from(err: serde_json::Error) -> Self {
        OsmError::Serialization(err.to_string())
    }
}
