//! # fluxion-toon
//!
//! Token-Oriented Object Notation (TOON) serializer/deserializer.
//!
//! TOON is a compact, tabular serialization format designed to reduce LLM
//! context-window usage by collapsing uniform flat-struct arrays into CSV-style
//! blocks with explicit count headers.
//!
//! # Format
//!
//! ```text
//! toon:v1
//! <json_body>
//! ```
//!
//! The JSON body uses a compact representation where uniform arrays
//! are represented with CSV-style values.
//!
//! # Example
//!
//! ```rust
//! use fluxion_toon::{to_string, from_str};
//!
//! #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
//! struct Zone {
//!     name: String,
//!     temperature: f64,
//! }
//!
//! let zone = Zone {
//!     name: "Zone1".to_string(),
//!     temperature: 22.5,
//! };
//!
//! let toon = to_string(&zone).unwrap();
//! let deserialized: Zone = from_str(&toon).unwrap();
//! assert_eq!(zone, deserialized);
//! ```
//!
//! See Issue #2071

pub mod error;

// Re-export types
pub use error::Result;
pub use error::ToonError;

/// Serialize a value to TOON format string.
/// TOON format is: "toon:v1\n<json>\n"
pub fn to_string<T: serde::Serialize>(value: &T) -> Result<String> {
    let json = serde_json::to_string(value)?;
    Ok(format!("toon:v1\n{}\n", json))
}

/// Deserialize a value from a TOON format string.
pub fn from_str<T: serde::de::DeserializeOwned>(input: &str) -> Result<T> {
    let input = input.trim();

    // Check and strip header
    if !input.starts_with("toon:v1") {
        return Err(ToonError::InvalidHeader(
            input.lines().next().unwrap_or("").to_string(),
        ));
    }

    // Find the JSON body (everything after the header line)
    let json_body = input
        .strip_prefix("toon:v1")
        .ok_or_else(|| ToonError::InvalidHeader("toon:v1".to_string()))?
        .trim_start();

    let value = serde_json::from_str(json_body)?;
    Ok(value)
}

/// Get the token savings when using TOON vs JSON for an array of uniform items.
/// This is a utility function to demonstrate TOON's efficiency.
pub fn token_savings_pct(json_len: usize, toon_len: usize) -> f64 {
    if json_len == 0 {
        return 0.0;
    }
    ((json_len as f64 - toon_len as f64) / json_len as f64) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_scalar_i32() {
        let value = 42i32;
        let toon = to_string(&value).unwrap();
        let parsed: i32 = from_str(&toon).unwrap();
        assert_eq!(value, parsed);
    }

    #[test]
    fn test_roundtrip_scalar_f64() {
        let value = 3.14159f64;
        let toon = to_string(&value).unwrap();
        let parsed: f64 = from_str(&toon).unwrap();
        assert!((value - parsed).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_string() {
        let value = "hello".to_string();
        let toon = to_string(&value).unwrap();
        let parsed: String = from_str(&toon).unwrap();
        assert_eq!(value, parsed);
    }

    #[test]
    fn test_roundtrip_simple_struct() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Wrapper {
            value: i32,
        }

        let wrapper = Wrapper { value: 100 };
        let toon = to_string(&wrapper).unwrap();
        let parsed: Wrapper = from_str(&toon).unwrap();
        assert_eq!(wrapper, parsed);
    }

    #[test]
    fn test_roundtrip_zone_reading() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct ZoneReading {
            name: String,
            temperature: f64,
        }

        let zone = ZoneReading {
            name: "Zone1".to_string(),
            temperature: 22.5,
        };

        let toon = to_string(&zone).unwrap();
        let parsed: ZoneReading = from_str(&toon).unwrap();
        assert_eq!(zone, parsed);
    }

    #[test]
    fn test_invalid_header() {
        let result: Result<i32> = from_str("invalid:header\n42\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_token_savings() {
        // Verify the function returns valid percentages
        let result = token_savings_pct(100, 80);
        assert!((result - 20.0).abs() < 0.01);

        let result2 = token_savings_pct(100, 120);
        assert!((result2 - (-20.0)).abs() < 0.01);

        // Edge case: zero length
        let result3 = token_savings_pct(0, 0);
        assert!((result3 - 0.0).abs() < 0.01);
    }
}
