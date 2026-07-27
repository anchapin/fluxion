//! TOON Serializer implementation
//!
//! Collapses uniform flat-struct arrays into CSV-style blocks with explicit
//! count headers to reduce token usage.

use crate::error::ToonError;
use serde::Serialize;

/// Serialize a value to a TOON string
pub fn serialize_to_string<T: Serialize>(value: &T) -> Result<String, ToonError> {
    // Scaffold: delegates to JSON until issue #2067 implements CSV collapse
    let json = serde_json::to_string(value).map_err(|e| ToonError::Serialization(e.to_string()))?;

    // TODO(#2067): Implement CSV collapse for uniform arrays
    // The full implementation will:
    // 1. Detect uniform arrays (Vec<f64>, Vec<i32>, etc.)
    // 2. Collapse them to CSV-style: @field_name[count] = val1, val2, ...
    // 3. Preserve structure for mixed types

    Ok(json)
}

// TODO(#2067): Implement full Serde Serializer with CSV collapse for uniform arrays
//
// The serializer will detect uniform arrays (e.g., Vec<f64>) and collapse them:
// [22.5, 23.1, 21.8] -> @temp_c[3]\n22.5, 23.1, 21.8
//
// Example transformation:
// {
//   "zone": "Office Floor 3",        -> @zone = "Office Floor 3"
//   "temperatures": [22.5, 23.1],   -> @temperatures[2] = 22.5, 23.1
//   "setpoint": 22.0                 -> @setpoint = 22.0
// }
