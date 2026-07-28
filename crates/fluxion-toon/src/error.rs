//! Error types for TOON serialization/deserialization.

use thiserror::Error;

/// Errors that can occur during TOON serialization or deserialization.
#[derive(Debug, Error)]
pub enum ToonError {
    /// Unexpected end of input.
    #[error("unexpected end of input")]
    Eof,

    /// Invalid header format.
    #[error("invalid header: expected 'toon:v1', got '{0}'")]
    InvalidHeader(String),

    /// Malformed patch string.
    #[error("malformed patch: {0}")]
    MalformedPatch(String),

    /// Custom error message.
    #[error("{0}")]
    Custom(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type alias for TOON operations.
pub type Result<T> = std::result::Result<T, ToonError>;
