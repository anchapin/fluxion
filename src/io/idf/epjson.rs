// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Parser for EnergyPlus epJSON (JSON representation of IDF).
//!
//! epJSON maps each IDF object type to a dictionary of named instances.
//! Per design §2.2, object-type keys are matched case-insensitively.
//!
//! ```json
//! {
//!   "Version": {
//!     "Version 1": { "version_identifier": "25.2" }
//!   },
//!   "Building": {
//!     "Main Building": {
//!       "name": "RefBox",
//!       "north_axis": 0.0
//!     }
//!   }
//! }
//! ```
//!
//! The top-level keys are object types; their values are dictionaries of
//! named objects; each named object contains field-value pairs. Field values
//! may be strings, numbers, or arrays (for extensible/vector fields).

use serde_json::Value;

use fluxion_core::parser_limits::ParserLimits;

use super::error::IdfError;
use super::parser::{IdfFile, IdfObject, IdfParser, IdfValue};

impl IdfParser {
    /// Parse an in-memory epJSON document into an [`IdfFile`] with the
    /// strict default parser limits (64 MiB / 1M lines — issue #2527).
    ///
    /// The JSON is expected to follow the EnergyPlus epJSON schema:
    /// top-level keys are object types; values are dictionaries of
    /// named instances; instances contain field-value pairs.
    ///
    /// # Case sensitivity
    ///
    /// Object-type keys (e.g. `"Version"`, `"Building"`) are matched
    /// case-insensitively, consistent with EnergyPlus conventions.
    ///
    /// # Errors
    ///
    /// Returns [`IdfError::Parse`] if the JSON is malformed or cannot be
    /// interpreted as epJSON, or [`IdfError::SizeLimitExceeded`] if the
    /// input exceeds the configured [`ParserLimits`].
    pub fn from_epjson_str(input: &str) -> Result<IdfFile, IdfError> {
        Self::from_epjson_str_with_limits(input, &ParserLimits::default())
    }

    /// Parse an in-memory epJSON document with explicit [`ParserLimits`]
    /// (issue #2527). The byte/line caps are enforced **before**
    /// `serde_json::from_str` allocates the DOM.
    pub fn from_epjson_str_with_limits(
        input: &str,
        limits: &ParserLimits,
    ) -> Result<IdfFile, IdfError> {
        limits.check_file_bytes(input.len())?;
        limits.check_lines(input.lines().count())?;

        let json: Value = serde_json::from_str(input).map_err(|e| IdfError::Parse {
            line: 1,
            message: format!("invalid JSON: {e}"),
        })?;

        let Value::Object(top) = &json else {
            return Err(IdfError::Parse {
                line: 1,
                message: "epJSON root must be a JSON object".to_string(),
            });
        };

        let mut idf = IdfFile::default();

        for (object_type, instances) in top.iter() {
            let Value::Object(instances_map) = instances else {
                continue;
            };

            for (instance_name, fields) in instances_map.iter() {
                let Value::Object(field_map) = fields else {
                    continue;
                };

                let obj = parse_epjson_object(object_type, instance_name, field_map)?;

                if obj.object_type.eq_ignore_ascii_case("Version") && idf.version.is_none() {
                    idf.version = obj.fields.first().and_then(|v| v.to_display_string());
                }
                idf.objects.push(obj);
            }
        }

        Ok(idf)
    }

    /// Parse an epJSON document from a filesystem path.
    pub fn from_epjson_path(path: &std::path::Path) -> Result<IdfFile, IdfError> {
        Self::from_epjson_path_with_limits(path, &ParserLimits::default())
    }

    /// Parse an epJSON document from a filesystem path with explicit
    /// [`ParserLimits`] (issue #2527). The on-disk size is checked
    /// before the file is read.
    pub fn from_epjson_path_with_limits(
        path: &std::path::Path,
        limits: &ParserLimits,
    ) -> Result<IdfFile, IdfError> {
        let file_len = std::fs::metadata(path).map_err(IdfError::from)?.len() as usize;
        limits.check_file_bytes(file_len)?;
        let content = std::fs::read_to_string(path)?;
        Self::from_epjson_str_with_limits(&content, limits)
    }
}

/// Convert a single epJSON instance (field-value map) into an [`IdfObject`].
fn parse_epjson_object(
    object_type: &str,
    instance_name: &str,
    field_map: &serde_json::Map<String, Value>,
) -> Result<IdfObject, IdfError> {
    let mut fields = Vec::new();

    for (_field_name, field_value) in field_map.iter() {
        let idf_value = parse_epjson_value(field_value);
        fields.push(idf_value);
    }

    let name = if instance_name.is_empty() || object_type.eq_ignore_ascii_case("Version") {
        None
    } else {
        Some(instance_name.to_string())
    };

    Ok(IdfObject {
        object_type: object_type.to_string(),
        name,
        fields,
        line: 1,
    })
}

/// Convert a serde_json [`Value`] into an [`IdfValue`].
fn parse_epjson_value(value: &Value) -> IdfValue {
    match value {
        Value::String(s) => {
            if s.is_empty() {
                IdfValue::Empty
            } else {
                IdfValue::String(s.clone())
            }
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                IdfValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                IdfValue::Real(f)
            } else {
                IdfValue::String(n.to_string())
            }
        }
        Value::Null => IdfValue::Empty,
        Value::Bool(b) => IdfValue::String(b.to_string()),
        Value::Array(arr) => {
            let strings: Vec<String> = arr
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Null => String::new(),
                    _ => v.to_string(),
                })
                .collect();
            IdfValue::String(strings.join(","))
        }
        Value::Object(_) => IdfValue::String(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_epjson_version() {
        let src = r#"{
          "Version": {
            "Version 1": {
              "version_identifier": "25.2"
            }
          }
        }"#;
        let idf = IdfParser::from_epjson_str(src).unwrap();
        assert_eq!(idf.version.as_deref(), Some("25.2"));
        assert_eq!(idf.objects.len(), 1);
        let obj = &idf.objects[0];
        assert_eq!(obj.object_type, "Version");
        assert!(obj.name.is_none());
    }

    #[test]
    fn parses_building_object() {
        let src = r#"{
          "Building": {
            "MainBuilding": {
              "name": "RefBox",
              "north_axis": 0.0,
              "terrain": "Suburbs",
              "loads_convergence_tolerance_value": 0.04,
              "temperature_convergence_tolerance_value": 0.4,
              "solar_distribution": "FullExterior",
              "maximum_number_of_warmup_days": 25
            }
          }
        }"#;
        let idf = IdfParser::from_epjson_str(src).unwrap();
        assert_eq!(idf.objects.len(), 1);
        let obj = &idf.objects[0];
        assert_eq!(obj.object_type, "Building");
        assert_eq!(obj.name.as_deref(), Some("MainBuilding"));
    }

    #[test]
    fn case_insensitive_object_types() {
        let src_lower = r#"{
          "version": {
            "Version 1": { "version_identifier": "25.2" }
          }
        }"#;
        let idf_lower = IdfParser::from_epjson_str(src_lower).unwrap();
        assert!(idf_lower.versions().next().is_some());

        let src_upper = r#"{
          "VERSION": {
            "Version 1": { "version_identifier": "25.2" }
          }
        }"#;
        let idf_upper = IdfParser::from_epjson_str(src_upper).unwrap();
        assert!(idf_upper.versions().next().is_some());
    }

    #[test]
    fn empty_fields_become_empty_variant() {
        let src = r#"{
          "Zone": {
            "Zone 1": {
              "name": "Zone1",
              "direction_of_relative_north": 0.0,
              "x_origin": 0.0,
              "y_origin": null,
              "z_origin": 0.0
            }
          }
        }"#;
        let idf = IdfParser::from_epjson_str(src).unwrap();
        assert_eq!(idf.objects.len(), 1);
        let obj = &idf.objects[0];
        let empty_count = obj.fields.iter().filter(|f| *f == &IdfValue::Empty).count();
        assert_eq!(empty_count, 1);
    }

    #[test]
    fn unknown_object_types_are_captured() {
        let src = r#"{
          "TotallyMadeUpObject": {
            "Instance 1": {
              "field_a": "hello"
            }
          }
        }"#;
        let idf = IdfParser::from_epjson_str(src).unwrap();
        assert_eq!(idf.objects.len(), 1);
        assert_eq!(idf.objects[0].object_type, "TotallyMadeUpObject");
    }

    #[test]
    fn parses_multi_instance_object_type() {
        let src = r#"{
          "Material": {
            "GypsumBoard": {
              "name": "GypsumBoard",
              "roughness": "MediumSmooth",
              "thickness": 0.0127,
              "conductivity": 0.16
            },
            "Insulation": {
              "name": "Insulation",
              "roughness": "MediumRough",
              "thickness": 0.05,
              "conductivity": 0.04
            }
          }
        }"#;
        let idf = IdfParser::from_epjson_str(src).unwrap();
        assert_eq!(idf.objects.len(), 2);
        assert_eq!(idf.materials().count(), 2);
    }

    #[test]
    fn invalid_json_returns_error() {
        let src = "not valid json at all";
        let result = IdfParser::from_epjson_str(src);
        assert!(result.is_err());
    }

    /// Issue #1707 acceptance criterion #2 (unit-level): parse epJSON,
    /// serialize the resulting [`IdfFile`] back to IDF text, re-parse the
    /// IDF text, and verify object-level consistency.
    #[test]
    fn epjson_idf_round_trip_preserves_object_count() {
        let src = r#"{
          "Version": {
            "Version 1": { "version_identifier": "25.2" }
          },
          "Building": {
            "MainBuilding": {
              "name": "RefBox",
              "north_axis": 0.0,
              "terrain": "Suburbs"
            }
          },
          "Zone": {
            "Zone1": {
              "name": "Zone1",
              "direction_of_relative_north": 0.0,
              "x_origin": 0.0,
              "y_origin": 0.0,
              "z_origin": 0.0
            }
          },
          "Material": {
            "GypsumBoard": {
              "name": "GypsumBoard",
              "roughness": "MediumSmooth",
              "thickness": 0.0127,
              "conductivity": 0.16,
              "density": 800,
              "specific_heat": 1090
            }
          }
        }"#;

        let idf_a = IdfParser::from_epjson_str(src).expect("parses epJSON");

        // Serialize IdfFile back to IDF text.
        let mut idf_text = String::new();
        for obj in &idf_a.objects {
            idf_text.push_str(&obj.object_type);
            for field in &obj.fields {
                idf_text.push_str(", ");
                match field {
                    IdfValue::String(s) => {
                        idf_text.push('"');
                        idf_text.push_str(s);
                        idf_text.push('"');
                    }
                    IdfValue::Real(f) => idf_text.push_str(&f.to_string()),
                    IdfValue::Integer(i) => idf_text.push_str(&i.to_string()),
                    IdfValue::Empty => {}
                }
            }
            idf_text.push_str(";\n");
        }

        // Re-parse the IDF text.
        let idf_b = IdfParser::from_str(&idf_text).expect("re-parses IDF text");

        // Object count and types must match.
        assert_eq!(
            idf_a.objects.len(),
            idf_b.objects.len(),
            "object count should match after round-trip"
        );
        for (a, b) in idf_a.objects.iter().zip(idf_b.objects.iter()) {
            assert_eq!(
                a.object_type, b.object_type,
                "object_type mismatch after round-trip"
            );
            assert_eq!(
                a.fields.len(),
                b.fields.len(),
                "field count mismatch for {} after round-trip",
                a.object_type
            );
        }

        // Version must survive the round-trip.
        assert_eq!(idf_a.version, idf_b.version);
    }
}
