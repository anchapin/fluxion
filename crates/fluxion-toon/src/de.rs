//! TOON Deserializer implementation
//!
//! Uses `winnow` for efficient streaming parsing with explicit length guardrails.

use crate::error::ToonError;

/// Deserialize a TOON string to a value
pub fn deserialize_from_str<T: serde::de::DeserializeOwned>(_s: &str) -> Result<T, ToonError> {
    // TODO(#2068): Implement full deserializer with length validation
    // Uses winnow for streaming parsing
    // Validates length headers match actual value counts
    Err(ToonError::Deserialization(
        "deserializer not yet implemented (issue #2068)".to_string(),
    ))
}

// TODO(#2068): Implement winnow-based parser with:
// - Length header parsing: @field_name[count]
// - CSV value parsing with proper escaping
// - LengthMismatch guardrail that compares declared vs actual counts
