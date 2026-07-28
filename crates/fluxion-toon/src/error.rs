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

    /// Length mismatch: declared count differs from actual row counts.
    #[error("length mismatch: declared {declared} rows, found {found}")]
    LengthMismatch { declared: usize, found: usize },

    /// Invalid syntax on a specific line.
    #[error("invalid syntax at line {line}: {message}")]
    InvalidSyntax { line: usize, message: String },

    /// Custom error message.
    #[error("{0}")]
    Custom(String),

    /// Deserialization error.
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Patch error.
    #[error("patch error: {0}")]
    PatchError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type alias for TOON operations.
pub type Result<T> = std::result::Result<T, ToonError>;
