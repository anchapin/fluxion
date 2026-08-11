//! TOON Deserializer implementation
//!
//! Implements `serde::de::Deserialize` for `ToonSlice` by converting
//! the parsed `ToonDocument` to `serde_json::Value` and delegating deserialization.

use crate::error::{Result, ToonError};
use crate::parse::ToonDocument;
use serde::de::DeserializeOwned;

pub use crate::parse::ToonDocument as ToonSlice;

/// Deserialize a TOON string to a value using the new parser with length guardrails.
pub fn deserialize_from_str<T: DeserializeOwned>(input: &str) -> Result<T> {
    let doc = ToonDocument::parse(input)?;
    let json = doc.to_json();
    T::deserialize(json).map_err(|e| ToonError::Deserialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Deserialize)]
    struct Zone {
        name: String,
        temperature: f64,
    }

    #[derive(Debug, serde::Deserialize)]
    struct Zones {
        zones: Vec<Zone>,
    }

    #[test]
    fn test_deserialize_toon_document() {
        let input = r#"toon:v1
count=2
zones:2
name,temperature
Zone1,22.5
Zone2,23.0
"#;
        let result: Zones = deserialize_from_str(input).unwrap();
        assert_eq!(result.zones.len(), 2);
    }

    #[test]
    fn test_length_mismatch_error() {
        let input = r#"toon:v1
count=3
zones:3
name,temperature
Zone1,Zone2
"#;
        let err = deserialize_from_str::<serde_json::Value>(input).unwrap_err();
        match err {
            ToonError::LengthMismatch { declared, found } => {
                assert_eq!(declared, 3);
                assert_eq!(found, 1);
            }
            _ => panic!("expected LengthMismatch, got {:?}", err),
        }
    }

    #[test]
    fn test_invalid_syntax_error() {
        let input = r#"toon:v1
zones:2
"#;
        let err = deserialize_from_str::<serde_json::Value>(input).unwrap_err();
        match err {
            ToonError::InvalidSyntax { .. } => {}
            _ => panic!("expected InvalidSyntax, got {:?}", err),
        }
    }

    // ----- Issue #2527: array-element cap -----------------------------------

    #[test]
    fn rejects_huge_declared_array() {
        // Declare an array larger than MAX_ARRAY_ELEMENTS (1M) without
        // supplying any data. The cap must fire before the parser tries
        // to allocate a billion-entry Vec via `(0..len).map(...)`.
        let input = format!(
            "toon:v1\nzones:{}\nname,temperature\n",
            crate::parse::MAX_ARRAY_ELEMENTS + 1
        );
        let err = deserialize_from_str::<serde_json::Value>(&input).unwrap_err();
        match err {
            ToonError::TooLarge(msg) => {
                assert!(
                    msg.contains("zones"),
                    "message should name the array: {}",
                    msg
                );
            }
            _ => panic!("expected TooLarge, got {:?}", err),
        }
    }

    #[test]
    fn normal_array_parses_within_cap() {
        let input = r#"toon:v1
count=2
zones:2
name,temperature
Zone1,22.5
Zone2,23.0
"#;
        let result: Zones = deserialize_from_str(input).unwrap();
        assert_eq!(result.zones.len(), 2);
    }
}
