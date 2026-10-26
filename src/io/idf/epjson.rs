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

// ---------------------------------------------------------------------------
// Export — the inverse of `parse_epjson_object`
// ---------------------------------------------------------------------------
//
// The parser is deliberately lossy in one respect: `parse_epjson_object`
// keeps the *values* of an epJSON instance but not the field names (the
// `IdfObject` model has no slot for them), and it flattens JSON arrays
// into a single comma-joined string (`parse_epjson_value`). The exporter
// below is the exact inverse of those two rules:
//
// * instance keys are the object names (empty for the object types that
//   legitimately have no name field),
// * field keys are positional (`field_00`, `field_01`, ...) so that the
//   original field order survives the round trip — the parser reads
//   epJSON objects in `serde_json::Map` key order, which is *not* the
//   order the fields were written in,
// * numeric values stay JSON numbers, empty fields become `null`, and
//   comma-joined strings are expanded back into JSON arrays.
//
// Field *names* cannot be recovered from an `IdfFile` (the parser drops
// them), so the exporter emits positional keys. That keeps the export a
// faithful, loss-free inverse of this parser; a future IDD-backed
// exporter can upgrade the positional keys to canonical EnergyPlus field
// names without changing the document structure.

/// Object types whose objects legitimately carry no name field.
///
/// The IDF parser documents these as the object types without a name
/// field (`Version`, `Timestep`, `GlobalGeometryRules` and
/// `Site:GroundTemperature:BuildingSurface`); their epJSON instance key
/// is the empty string, which the parser maps back to `name: None`.
///
/// Any *other* object without a name is a malformed internal state: the
/// epJSON instance map is keyed by name, so such an object cannot be
/// represented. The exporter reports that as an error instead of
/// silently dropping the object.
const NAMELESS_OBJECT_TYPES: &[&str] = &[
    "version",
    "timestep",
    "globalgeometryrules",
    "site:groundtemperature:buildingsurface",
];

impl IdfFile {
    /// Serialize this file as an epJSON document.
    ///
    /// This is the inverse of [`IdfParser::from_epjson_str`]: the
    /// resulting document parses back into an `IdfFile` with the same
    /// objects, names and field values.
    ///
    /// # Errors
    ///
    /// Returns [`IdfError::Conversion`] when the file cannot be
    /// represented in epJSON — an object without a name (and without a
    /// name-bearing object type), an object with an empty object type, or
    /// a non-finite floating point field (JSON has no `NaN`/`Infinity`).
    pub fn to_epjson_string(&self) -> Result<String, IdfError> {
        let value = self.to_epjson_value()?;
        serde_json::to_string_pretty(&value).map_err(|e| {
            IdfError::conversion_error(format!("failed to serialize epJSON document: {e}"))
        })
    }

    /// Serialize this file as an epJSON document into `writer`.
    ///
    /// Equivalent to [`IdfFile::to_epjson_string`] but streams the
    /// document straight into the writer.
    ///
    /// # Errors
    ///
    /// Same as [`IdfFile::to_epjson_string`], plus any I/O failure
    /// reported by the writer.
    pub fn to_epjson_writer<W: std::io::Write>(&self, writer: &mut W) -> Result<(), IdfError> {
        let document = self.to_epjson_string()?;
        writer.write_all(document.as_bytes())?;
        Ok(())
    }

    /// Build the epJSON representation of this file as a
    /// [`serde_json::Value`].
    ///
    /// The document mirrors the epJSON layout described in the module
    /// docs: top-level keys are object types, each maps to a dictionary
    /// of named instances, and each instance maps field keys to values.
    pub fn to_epjson_value(&self) -> Result<Value, IdfError> {
        let mut root: serde_json::Map<String, Value> = serde_json::Map::new();

        for (index, object) in self.objects.iter().enumerate() {
            let object_type = object.object_type.trim();
            if object_type.is_empty() {
                return Err(IdfError::conversion_error(format!(
                    "object #{index} has an empty object type and cannot be represented in epJSON"
                )));
            }

            let base_key = epjson_instance_key(object, index)?;
            let fields = epjson_field_map(object)?;

            let instances = root
                .entry(object_type.to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            let instance_map = instances.as_object_mut().ok_or_else(|| {
                IdfError::conversion_error(format!(
                    "object #{index} of type '{object_type}' conflicts with an existing epJSON entry"
                ))
            })?;

            // epJSON instance keys are unique within an object type. Two
            // objects that share a name would otherwise collapse into one
            // and the second object would be silently dropped, so the
            // colliding key is disambiguated with a numeric suffix.
            let mut key = base_key.clone();
            let mut occurrence = 1usize;
            while instance_map.contains_key(&key) {
                occurrence += 1;
                key = format!("{base_key} ({occurrence})");
            }
            instance_map.insert(key, Value::Object(fields));
        }

        Ok(Value::Object(root))
    }
}

/// Compute the epJSON instance key for an object.
///
/// Named objects are keyed by their name; objects of a type that has no
/// name field use the empty key (which the parser maps back to
/// `name: None`). Every other nameless object is an error.
fn epjson_instance_key(object: &IdfObject, index: usize) -> Result<String, IdfError> {
    if let Some(name) = object.name.as_deref() {
        if !name.is_empty() {
            return Ok(name.to_string());
        }
    }

    let object_type = object.object_type.trim().to_ascii_lowercase();
    if NAMELESS_OBJECT_TYPES.contains(&object_type.as_str()) {
        return Ok(String::new());
    }

    Err(IdfError::conversion_error(format!(
        "object #{index} of type '{}' has no name; epJSON instances are keyed by name, so this object cannot be exported",
        object.object_type
    )))
}

/// Build the field map of a single epJSON instance.
///
/// Keys are positional (`field_00`, `field_01`, ...) because the parser
/// does not retain epJSON field names. The zero padding keeps the keys
/// in field order when the document is read back (the parser iterates
/// epJSON objects in map key order).
fn epjson_field_map(object: &IdfObject) -> Result<serde_json::Map<String, Value>, IdfError> {
    let width = field_key_width(object.fields.len());
    let mut map = serde_json::Map::new();

    for (index, field) in object.fields.iter().enumerate() {
        let key = format!("field_{index:0width$}");
        let value = epjson_field_value(field, &object.object_type, index)?;
        map.insert(key, value);
    }

    Ok(map)
}

/// Number of decimal digits used for the positional field keys.
fn field_key_width(field_count: usize) -> usize {
    let digits = field_count.to_string().len();
    digits.max(2)
}

/// Convert a single [`IdfValue`] into its epJSON representation.
///
/// * [`IdfValue::Integer`] / [`IdfValue::Real`] become JSON numbers,
/// * [`IdfValue::Empty`] becomes `null`,
/// * [`IdfValue::String`] becomes a JSON string, or — when it is the
///   comma-joined form of an epJSON array — a JSON array again.
fn epjson_field_value(
    field: &IdfValue,
    object_type: &str,
    index: usize,
) -> Result<Value, IdfError> {
    match field {
        IdfValue::Integer(i) => Ok(Value::Number((*i).into())),
        IdfValue::Real(f) => serde_json::Number::from_f64(*f).map(Value::Number).ok_or_else(|| {
            IdfError::conversion_error(format!(
                "object of type '{object_type}' has a non-finite numeric field at index {index} ({f}); JSON cannot represent NaN or Infinity"
            ))
        }),
        IdfValue::Empty => Ok(Value::Null),
        IdfValue::String(s) if s.is_empty() => Ok(Value::Null),
        IdfValue::String(s) => Ok(match split_epjson_list(s) {
            Some(parts) => Value::Array(parts.iter().map(|part| epjson_scalar(part)).collect()),
            None => Value::String(s.clone()),
        }),
    }
}

/// Split a comma-joined string back into its epJSON array elements.
///
/// This reverses the array flattening performed by
/// [`parse_epjson_value`]. Only list-shaped values are expanded: the
/// elements must either all be numeric (a numeric vector such as a
/// vertex coordinate list) or all be whitespace-free tokens. Text values
/// that merely happen to contain a comma (e.g. `Hello, World!`) stay
/// strings.
fn split_epjson_list(value: &str) -> Option<Vec<String>> {
    if !value.contains(',') {
        return None;
    }

    let parts: Vec<String> = value.split(',').map(|part| part.to_string()).collect();
    let all_numeric = parts
        .iter()
        .all(|part| !part.trim().is_empty() && part.trim().parse::<f64>().is_ok());
    let all_tokens = parts
        .iter()
        .all(|part| !part.is_empty() && !part.chars().any(char::is_whitespace));

    if all_numeric || all_tokens {
        Some(parts)
    } else {
        None
    }
}

/// Convert one array element into its epJSON representation.
///
/// Numeric elements are emitted as JSON numbers, but only when the
/// canonical rendering of the number is identical to the source text —
/// otherwise the element is kept as a string so that the array can be
/// re-joined into the exact original value.
fn epjson_scalar(text: &str) -> Value {
    if let Ok(i) = text.parse::<i64>() {
        if i.to_string() == text {
            return Value::Number(i.into());
        }
    }
    if let Ok(f) = text.parse::<f64>() {
        // Emit the number whenever the text parses as a finite float. The
        // parser's array flattening joins elements with serde_json's own
        // Number rendering, so a serde number here re-joins to the exact
        // original string on re-parse. (Comparing against Rust's f64
        // Display instead would reject "0.0" — Display renders it as "0" —
        // and silently demote numeric array elements to strings.)
        if f.is_finite() {
            if let Some(number) = serde_json::Number::from_f64(f) {
                return Value::Number(number);
            }
        }
    }
    Value::String(text.to_string())
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

    // -----------------------------------------------------------------------
    // Export — helpers
    // -----------------------------------------------------------------------

    /// Strip the disambiguation suffix the exporter appends to colliding
    /// epJSON instance keys (`"ZONE1 (2)"` -> `"ZONE1"`).
    fn strip_disambiguation(name: &str) -> &str {
        if let Some(open) = name.rfind(" (") {
            if name.ends_with(')') {
                let digits = &name[open + 2..name.len() - 1];
                if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                    return &name[..open];
                }
            }
        }
        name
    }

    /// Canonical, order-insensitive view of an [`IdfFile`]'s objects.
    ///
    /// The epJSON document groups objects by type and keys instances by
    /// name, so neither the object order nor the *names* of colliding
    /// instances can be preserved by a round trip. Everything else —
    /// object type, name and the ordered field values — must survive.
    fn canonical_objects(idf: &IdfFile) -> Vec<(String, Option<String>, Vec<IdfValue>)> {
        let mut objects: Vec<(String, Option<String>, Vec<IdfValue>)> = idf
            .objects
            .iter()
            .map(|object| {
                (
                    object.object_type.clone(),
                    // The epJSON parser normalizes Version instances to
                    // nameless (parse_epjson_object), while the text-IDF
                    // parser stores the version string as the name. The
                    // version value itself is still compared via fields.
                    if object.object_type.eq_ignore_ascii_case("Version") {
                        None
                    } else {
                        object
                            .name
                            .as_deref()
                            .map(strip_disambiguation)
                            .map(str::to_string)
                    },
                    object.fields.clone(),
                )
            })
            .collect();
        objects.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| format!("{:?}", a.2).cmp(&format!("{:?}", b.2)))
        });
        objects
    }

    fn reference_models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("reference_data")
            .join("energyplus_models")
    }

    // -----------------------------------------------------------------------
    // Export — behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn exports_epjson_document_structure() {
        let src = r#"{
          "Version": { "Version 1": { "version_identifier": "25.2" } },
          "Building": {
            "MainBuilding": {
              "name": "RefBox",
              "north_axis": 0.0,
              "terrain": "Suburbs"
            }
          }
        }"#;

        let idf = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let document = idf.to_epjson_value().expect("exports epJSON");

        // Top level keys are object types, values are instance maps.
        let building = &document["Building"]["MainBuilding"];
        assert!(building.is_object(), "instance map expected: {document}");
        // The instance is keyed by the object name.
        assert_eq!(
            building["field_00"].as_str(),
            Some("RefBox"),
            "first field is the name: {building}"
        );
        // Numeric fields stay JSON numbers.
        assert!(
            building["field_01"].is_number(),
            "numeric field must stay a JSON number: {building}"
        );
        // Text fields stay JSON strings.
        assert_eq!(building["field_02"].as_str(), Some("Suburbs"));
    }

    #[test]
    fn export_round_trips_object_for_object() {
        let src = r#"{
          "Version": { "Version 1": { "version_identifier": "25.2" } },
          "Building": {
            "MainBuilding": {
              "name": "RefBox",
              "north_axis": 0.0,
              "terrain": "Suburbs",
              "loads_convergence_tolerance_value": 0.04
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
          }
        }"#;

        let original = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let exported = original.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&exported).expect("re-parses exported epJSON");

        assert_eq!(
            original.objects.len(),
            reparsed.objects.len(),
            "no object may be dropped by the round trip"
        );
        assert_eq!(
            canonical_objects(&original),
            canonical_objects(&reparsed),
            "objects must survive the epJSON round trip unchanged"
        );
        assert_eq!(original.version, reparsed.version);
    }

    #[test]
    fn export_preserves_field_order() {
        // The parser keeps epJSON fields in map key order; the exporter
        // therefore has to use positional keys so that the field order
        // survives a round trip.
        let src = r#"{
          "Building": {
            "MainBuilding": {
              "name": "RefBox",
              "north_axis": 0.0,
              "terrain": "Suburbs"
            }
          }
        }"#;

        let original = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let exported = original.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&exported).expect("re-parses exported epJSON");

        let before = &original.objects[0].fields;
        let after = &reparsed.objects[0].fields;
        assert_eq!(before, after, "field order must be preserved");
    }

    #[test]
    fn export_expands_joined_arrays_into_json_arrays() {
        // `parse_epjson_value` flattens JSON arrays into a comma joined
        // string; the exporter has to expand them again.
        let src = r#"{
          "BuildingSurface:Detailed": {
            "Wall 1": {
              "name": "Wall 1",
              "vertices": [0.0, 0.0, 2.7]
            }
          }
        }"#;

        let idf = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let document = idf.to_epjson_value().expect("exports epJSON");

        let vertices = &document["BuildingSurface:Detailed"]["Wall 1"]["field_01"];
        assert!(
            vertices.is_array(),
            "array field must be exported as a JSON array: {vertices}"
        );
        let elements = vertices.as_array().expect("array");
        assert_eq!(elements.len(), 3);
        assert!(elements.iter().all(|element| element.is_number()));

        // ... and the array must round-trip to the same field value.
        let exported = idf.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&exported).expect("re-parses exported epJSON");
        assert_eq!(
            idf.objects[0].fields, reparsed.objects[0].fields,
            "array field must round-trip unchanged"
        );
    }

    #[test]
    fn export_keeps_text_with_commas_as_string() {
        let src = r#"{
          "Material": { "Hello": { "name": "Hello, World!" } }
        }"#;

        let idf = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let document = idf.to_epjson_value().expect("exports epJSON");
        let name = &document["Material"]["Hello"]["field_00"];
        assert!(name.is_string(), "free text must stay a string: {name}");
        assert_eq!(name.as_str(), Some("Hello, World!"));
    }

    #[test]
    fn export_is_idempotent() {
        let src = r#"{
          "Version": { "Version 1": { "version_identifier": "25.2" } },
          "Building": {
            "MainBuilding": { "name": "RefBox", "north_axis": 0.0 }
          }
        }"#;

        let idf = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let first = idf.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&first).expect("re-parses exported epJSON");
        let second = reparsed.to_epjson_string().expect("exports epJSON");

        assert_eq!(first, second, "epJSON export must be stable");
    }

    #[test]
    fn export_keeps_objects_that_share_a_name() {
        // Two objects of the same type may share a name (EnergyPlus
        // `Output:Variable` is keyed by the variable key value). Both
        // must survive the export — neither may be silently dropped.
        let src = r#"{
          "Output:Variable": {
            "ZONE1": { "name": "ZONE1", "variable_name": "Zone Mean Air Temperature" },
            "ZONE1 ": { "name": "ZONE1", "variable_name": "Zone Air System Sensible Heating Energy" }
          }
        }"#;

        let mut idf = IdfParser::from_epjson_str(src).expect("parses epJSON");
        // Force the collision the exporter has to cope with.
        idf.objects[1].name = Some("ZONE1".to_string());

        let exported = idf.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&exported).expect("re-parses exported epJSON");
        assert_eq!(
            idf.objects.len(),
            reparsed.objects.len(),
            "no object may be dropped by the export"
        );
    }

    #[test]
    fn export_errors_on_object_without_name() {
        let idf = IdfFile {
            version: None,
            objects: vec![IdfObject {
                object_type: "Building".to_string(),
                name: None,
                fields: vec![IdfValue::Real(0.0)],
                line: 1,
            }],
        };

        let error = idf
            .to_epjson_string()
            .expect_err("nameless object must fail");
        let message = error.to_string();
        assert!(
            message.contains("Building") && message.contains("no name"),
            "error must name the offending object: {message}"
        );
    }

    #[test]
    fn export_errors_on_empty_object_type() {
        let idf = IdfFile {
            version: None,
            objects: vec![IdfObject {
                object_type: "  ".to_string(),
                name: Some("Thing".to_string()),
                fields: vec![IdfValue::Empty],
                line: 1,
            }],
        };

        let error = idf
            .to_epjson_string()
            .expect_err("empty object type must fail");
        assert!(
            error.to_string().contains("empty object type"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn export_errors_on_non_finite_number() {
        let idf = IdfFile {
            version: None,
            objects: vec![IdfObject {
                object_type: "Building".to_string(),
                name: Some("Main".to_string()),
                fields: vec![IdfValue::Real(f64::NAN)],
                line: 1,
            }],
        };

        let error = idf
            .to_epjson_string()
            .expect_err("NaN must fail to serialize");
        assert!(
            error.to_string().contains("non-finite"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn export_writer_matches_string() {
        let src = r#"{
          "Building": { "MainBuilding": { "name": "RefBox", "north_axis": 0.0 } }
        }"#;

        let idf = IdfParser::from_epjson_str(src).expect("parses epJSON");
        let expected = idf.to_epjson_string().expect("exports epJSON");

        let mut buffer: Vec<u8> = Vec::new();
        idf.to_epjson_writer(&mut buffer).expect("writes epJSON");
        assert_eq!(String::from_utf8(buffer).expect("utf8"), expected);
    }

    // -----------------------------------------------------------------------
    // Export — real reference models
    // -----------------------------------------------------------------------

    #[test]
    fn export_round_trips_ashrae_140_case_600() {
        let path = reference_models_dir().join("ashrae_140_case_600.idf");
        let original = IdfParser::from_path(&path).expect("parses ASHRAE 140 case 600");

        let exported = original.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&exported).expect("re-parses exported epJSON");

        assert_eq!(
            original.objects.len(),
            reparsed.objects.len(),
            "no object may be dropped by the round trip"
        );
        assert_eq!(
            canonical_objects(&original),
            canonical_objects(&reparsed),
            "objects must survive the epJSON round trip unchanged"
        );
        assert_eq!(original.version, reparsed.version);
    }

    #[test]
    fn export_round_trips_every_reference_idf_fixture() {
        let dir = reference_models_dir();
        let mut fixtures: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("reference model directory")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "idf"))
            .collect();
        fixtures.sort();

        assert!(
            !fixtures.is_empty(),
            "expected IDF fixtures in {}",
            dir.display()
        );

        for fixture in fixtures {
            let original = IdfParser::from_path(&fixture)
                .unwrap_or_else(|e| panic!("parses {}: {e}", fixture.display()));
            let exported = original
                .to_epjson_string()
                .unwrap_or_else(|e| panic!("exports {}: {e}", fixture.display()));
            let reparsed = IdfParser::from_epjson_str(&exported)
                .unwrap_or_else(|e| panic!("re-parses exported {}: {e}", fixture.display()));

            assert_eq!(
                original.objects.len(),
                reparsed.objects.len(),
                "no object may be dropped for {}",
                fixture.display()
            );
            assert_eq!(
                canonical_objects(&original),
                canonical_objects(&reparsed),
                "objects must survive the round trip for {}",
                fixture.display()
            );
        }
    }

    #[test]
    fn export_round_trips_reference_epjson_fixture() {
        let path = reference_models_dir().join("case900.epJSON");
        let original = IdfParser::from_epjson_path(&path).expect("parses case900.epJSON");

        let exported = original.to_epjson_string().expect("exports epJSON");
        let reparsed = IdfParser::from_epjson_str(&exported).expect("re-parses exported epJSON");

        assert_eq!(
            original.objects.len(),
            reparsed.objects.len(),
            "no object may be dropped by the round trip"
        );
        assert_eq!(
            canonical_objects(&original),
            canonical_objects(&reparsed),
            "objects must survive the epJSON round trip unchanged"
        );
    }
}
