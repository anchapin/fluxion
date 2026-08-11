// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! gbXML import/export error types.
//!
//! This module defines errors that can occur during gbXML parsing,
//! validation, and serialization.

use std::fmt;
use std::io;
use std::path::PathBuf;

use quick_xml::Error as XmlError;

/// Errors that can occur during gbXML import/export operations.
#[derive(Debug)]
pub enum GbXmlError {
    /// XML parsing error
    XmlParseError(String),
    /// XML serialization error
    XmlSerializeError(String),
    /// IO error (file not found, permission denied, etc.)
    IoError { path: PathBuf, message: String },
    /// Invalid gbXML structure (missing required elements, etc.)
    InvalidStructure(String),
    /// Unsupported gbXML feature
    UnsupportedFeature(String),
    /// Validation error
    ValidationError(String),
    /// Missing required element
    MissingElement(String),
    /// Invalid coordinate system
    InvalidCoordinate(String),
    /// Material property error
    InvalidMaterialProperty(String),
    /// Parser size limit exceeded (issue #2527 DoS hardening).
    SizeLimitExceeded(String),
}

impl fmt::Display for GbXmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GbXmlError::XmlParseError(msg) => {
                write!(f, "gbXML XML parsing error: {}", msg)
            }
            GbXmlError::XmlSerializeError(msg) => {
                write!(f, "gbXML XML serialization error: {}", msg)
            }
            GbXmlError::IoError { path, message } => {
                write!(f, "IO error accessing '{}': {}", path.display(), message)
            }
            GbXmlError::InvalidStructure(msg) => {
                write!(f, "Invalid gbXML structure: {}", msg)
            }
            GbXmlError::UnsupportedFeature(msg) => {
                write!(f, "Unsupported gbXML feature: {}", msg)
            }
            GbXmlError::ValidationError(msg) => {
                write!(f, "gbXML validation error: {}", msg)
            }
            GbXmlError::MissingElement(elem) => {
                write!(f, "Missing required gbXML element: {}", elem)
            }
            GbXmlError::InvalidCoordinate(msg) => {
                write!(f, "Invalid coordinate in gbXML: {}", msg)
            }
            GbXmlError::InvalidMaterialProperty(msg) => {
                write!(f, "Invalid material property in gbXML: {}", msg)
            }
            GbXmlError::SizeLimitExceeded(msg) => {
                write!(f, "gbXML parser size limit exceeded: {}", msg)
            }
        }
    }
}

impl std::error::Error for GbXmlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<io::Error> for GbXmlError {
    fn from(err: io::Error) -> Self {
        GbXmlError::IoError {
            path: PathBuf::new(),
            message: err.to_string(),
        }
    }
}

impl From<XmlError> for GbXmlError {
    fn from(err: XmlError) -> Self {
        GbXmlError::XmlParseError(err.to_string())
    }
}

impl From<fluxion_core::parser_limits::ParseLimitError> for GbXmlError {
    fn from(e: fluxion_core::parser_limits::ParseLimitError) -> Self {
        GbXmlError::SizeLimitExceeded(e.to_string())
    }
}

impl GbXmlError {
    /// Create an IO error with a specific path.
    pub fn io_error(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        GbXmlError::IoError {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Create a missing element error.
    pub fn missing_element(element: impl Into<String>) -> Self {
        GbXmlError::MissingElement(element.into())
    }

    /// Create an invalid structure error.
    pub fn invalid_structure(msg: impl Into<String>) -> Self {
        GbXmlError::InvalidStructure(msg.into())
    }

    /// Create an unsupported feature error.
    pub fn unsupported_feature(feature: impl Into<String>) -> Self {
        GbXmlError::UnsupportedFeature(feature.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GbXmlError::missing_element("Construction");
        assert!(err.to_string().contains("Construction"));

        let err = GbXmlError::invalid_structure("Invalid surface type");
        assert!(err.to_string().contains("Invalid surface type"));
    }

    #[test]
    fn test_io_error_with_path() {
        let err = GbXmlError::io_error("/path/to/file.xml", "File not found");
        assert!(err.to_string().contains("/path/to/file.xml"));
        assert!(err.to_string().contains("File not found"));
    }
}
