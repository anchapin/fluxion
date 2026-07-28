//! TOON Serializer implementation
//!
//! Collapses uniform flat-struct arrays into CSV-style blocks with explicit
//! count headers to reduce token usage.
//!
//! # Format
//!
//! - Uniform arrays: `name[N]{field1,field2,...}: val1,val2,...`
//! - Non-uniform arrays: standard JSON-like format
//! - Scalars: `name: value`

use std::fmt::Write;

pub trait ToonSerializable: serde::Serialize + Clone {
    fn toon_field_names() -> Vec<&'static str>;
    fn is_uniform_slice(_items: &[Self]) -> bool {
        true
    }
}

pub fn serialize_to_string<T: serde::Serialize>(value: &T) -> Result<String, std::fmt::Error> {
    let json = serde_json::to_string(value).map_err(|_| std::fmt::Error)?;
    let toon = collapse_uniform_arrays(&json);
    Ok(toon)
}

fn collapse_uniform_arrays(json: &str) -> String {
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return json.to_string(),
    };
    transform_value(&value, 0, None)
}

fn transform_value(value: &serde_json::Value, indent: usize, field_name: Option<&str>) -> String {
    match value {
        serde_json::Value::Object(obj) => {
            let mut output = String::new();
            let mut sorted_keys: Vec<_> = obj.keys().collect();
            sorted_keys.sort();
            for (i, key) in sorted_keys.iter().enumerate() {
                let comma = if i < sorted_keys.len() - 1 { "," } else { "" };
                let val = obj.get(key.as_str()).unwrap();
                if let serde_json::Value::Array(arr) = val {
                    if arr.len() > 0 && arr.iter().all(|v| v.is_object()) {
                        let first_obj = arr[0].as_object().unwrap();
                        let mut fields: Vec<&str> = first_obj.keys().map(|s| s.as_str()).collect();
                        fields.sort();
                        let all_same = arr.iter().all(|v| {
                            if let Some(o) = v.as_object() {
                                let mut other_fields: Vec<_> =
                                    o.keys().map(|k| k.as_str()).collect();
                                other_fields.sort();
                                fields.iter().zip(other_fields.iter()).all(|(a, b)| a == b)
                            } else {
                                false
                            }
                        });
                        if all_same && !fields.is_empty() {
                            writeln!(output, "{}[{}]{{{}}}:", key, arr.len(), fields.join(","))
                                .unwrap();
                            for item in arr {
                                if let Some(o) = item.as_object() {
                                    let row: Vec<String> = fields
                                        .iter()
                                        .map(|f| {
                                            o.get(*f)
                                                .map(|v| match v {
                                                    serde_json::Value::String(s) => s.clone(),
                                                    serde_json::Value::Number(n) => n.to_string(),
                                                    serde_json::Value::Bool(b) => b.to_string(),
                                                    serde_json::Value::Null => String::new(),
                                                    _ => v.to_string(),
                                                })
                                                .unwrap_or_default()
                                        })
                                        .collect();
                                    writeln!(output, "{}", row.join(",")).unwrap();
                                }
                            }
                            continue;
                        }
                    }
                }
                let val_str = transform_value(val, indent, Some(key));
                let key_escaped = if key.contains(' ') || key.contains(':') {
                    format!("\"{}\"", key)
                } else {
                    key.to_string()
                };
                if val_str.contains('\n') {
                    writeln!(output, "{}: {}", key_escaped, val_str).unwrap();
                } else {
                    writeln!(output, "{}: {}{}", key_escaped, val_str, comma).unwrap();
                }
            }
            output
        }
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                if let Some(name) = field_name {
                    return format!("{}[]:", name);
                }
                return "[]".to_string();
            }
            if arr.len() > 0 && arr.iter().all(|v| v.is_object()) {
                let first_obj = arr[0].as_object().unwrap();
                let mut fields: Vec<&str> = first_obj.keys().map(|s| s.as_str()).collect();
                fields.sort();
                let all_same = arr.iter().all(|v| {
                    if let Some(o) = v.as_object() {
                        let mut other_fields: Vec<_> = o.keys().map(|k| k.as_str()).collect();
                        other_fields.sort();
                        fields.iter().zip(other_fields.iter()).all(|(a, b)| a == b)
                    } else {
                        false
                    }
                });
                if all_same && !fields.is_empty() {
                    let mut output = String::new();
                    let name = field_name.unwrap_or("Array");
                    writeln!(output, "{}[{}]{{{}}}:", name, arr.len(), fields.join(",")).unwrap();
                    for item in arr {
                        if let Some(o) = item.as_object() {
                            let row: Vec<String> = fields
                                .iter()
                                .map(|f| {
                                    o.get(*f)
                                        .map(|v| match v {
                                            serde_json::Value::String(s) => s.clone(),
                                            serde_json::Value::Number(n) => n.to_string(),
                                            serde_json::Value::Bool(b) => b.to_string(),
                                            serde_json::Value::Null => String::new(),
                                            _ => v.to_string(),
                                        })
                                        .unwrap_or_default()
                                })
                                .collect();
                            writeln!(output, "{}", row.join(",")).unwrap();
                        }
                    }
                    return output;
                }
            }
            let mut output = String::new();
            for (i, item) in arr.iter().enumerate() {
                let comma = if i < arr.len() - 1 { "," } else { "" };
                let item_str = transform_value(item, indent, None);
                if item_str.contains('\n') {
                    writeln!(output, "{}{}", item_str, comma).unwrap();
                } else {
                    write!(output, "{}{}", item_str, comma).unwrap();
                }
            }
            format!("[{}]", output.trim_end_matches(','))
        }
        serde_json::Value::String(s) => format!("\"{}\"", s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
    struct temperatureZone {
        id: String,
        temp_c: f64,
        humidity_rh: f64,
    }

    impl ToonSerializable for temperatureZone {
        fn toon_field_names() -> Vec<&'static str> {
            vec!["id", "temp_c", "humidity_rh"]
        }
    }

    #[test]
    fn test_serialize_uniform_struct_array() {
        let zones = vec![
            temperatureZone {
                id: "Z1".to_string(),
                temp_c: 22.5,
                humidity_rh: 45.0,
            },
            temperatureZone {
                id: "Z2".to_string(),
                temp_c: 23.1,
                humidity_rh: 50.0,
            },
            temperatureZone {
                id: "Z3".to_string(),
                temp_c: 21.8,
                humidity_rh: 48.0,
            },
        ];

        let result = serialize_to_string(&zones).unwrap();
        let lines: Vec<&str> = result.lines().collect();

        assert!(lines[0].starts_with("Array[3]{"), "Got: {}", lines[0]);
        assert!(
            lines[0].contains("humidity_rh")
                && lines[0].contains("id")
                && lines[0].contains("temp_c"),
            "Got: {}",
            lines[0]
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("Z1") && l.contains("22.5") && l.contains("45")),
            "Got: {:?}",
            lines
        );
    }

    #[test]
    fn test_serialize_primitive_array() {
        let temps = vec![22.5, 23.1, 21.8, 24.0, 20.5];
        let result = serialize_to_string(&temps).unwrap();
        assert!(result.contains("22.5"));
    }

    #[test]
    fn test_serialize_single_struct() {
        let zone = temperatureZone {
            id: "Z1".to_string(),
            temp_c: 22.5,
            humidity_rh: 45.0,
        };
        let result = serialize_to_string(&zone).unwrap();
        assert!(result.contains("id"));
        assert!(result.contains("22.5"));
    }

    #[test]
    fn test_serialize_primitive() {
        let value = 42.0;
        let result = serialize_to_string(&value).unwrap();
        assert!(result.contains("42"));
    }

    #[test]
    fn test_serialize_empty_array() {
        let zones: Vec<temperatureZone> = vec![];
        let result = serialize_to_string(&zones).unwrap();
        assert!(
            result.contains("Array[0]") || result.contains("[]"),
            "Got: {}",
            result
        );
    }

    #[test]
    fn test_serialize_struct_with_array_field() {
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        struct Building {
            name: String,
            temperatures: Vec<f64>,
        }

        let building = Building {
            name: "Building A".to_string(),
            temperatures: vec![22.5, 23.1, 21.8],
        };

        let result = serialize_to_string(&building).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        assert!(
            lines.iter().any(|l| l.contains("Building A")),
            "Got: {:?}",
            lines
        );
    }
}
