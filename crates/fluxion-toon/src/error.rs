//! Error types for TOON serialization/deserialization

use thiserror::Error;

/// Errors that can occur during TOON processing
#[derive(Debug, Error)]
pub enum ToonError {
    /// Serialization failed
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Deserialization failed
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Length mismatch between header count and actual values
    #[error("length mismatch: header declares {expected} values, found {actual}")]
    LengthMismatch {
        /// Expected number of values from header
        expected: usize,
        /// Actual number of values found
        actual: usize,
    },

    /// Invalid TOON syntax
    #[error("invalid syntax at line {line}: {message}")]
    InvalidSyntax {
        /// Line number where error occurred
        line: usize,
        /// Error message
        message: String,
    },

    /// Patch parsing error (LLM response)
    #[error("patch parsing error: {0}")]
    PatchError(String),
}
