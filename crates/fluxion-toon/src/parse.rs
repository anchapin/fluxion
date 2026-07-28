//! TOON parser using `winnow` for efficient streaming tokenization.
//!
//! The parser tokenizes input into:
//! - Header lines: `toon:v1`, `count=<N>`, `array_name:<N>`
//! - Data rows: CSV-formatted values (parsed using winnow)
//!
//! Length guardrails compare actual row counts against declared counts.

use crate::error::{Result, ToonError};
use serde_json::{Map, Value};

#[derive(Debug)]
pub struct ToonDocument {
    pub count: Option<usize>,
    pub arrays: Map<String, Value>,
}

impl ToonDocument {
    pub fn parse(s: &str) -> Result<Self> {
        let mut lines: Vec<&str> = s.lines().collect();

        if lines.is_empty() {
            return Err(ToonError::Eof);
        }

        let first = lines.remove(0).trim();
        if !first.starts_with("toon:v1") {
            return Err(ToonError::InvalidHeader(first.to_string()));
        }

        let mut doc = ToonDocument {
            count: None,
            arrays: Map::new(),
        };

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            i += 1;

            if line.is_empty() {
                continue;
            }

            if let Some((key, val)) = parse_key_val(line) {
                if key == "count" {
                    doc.count = Some(val.parse().map_err(|_| ToonError::InvalidSyntax {
                        line: i,
                        message: format!("invalid count value: {}", val),
                    })?);
                }
            } else if let Some((array_name, len_str)) = parse_array_header(line) {
                let array_name = array_name.to_string();
                let len: usize = len_str.parse().map_err(|_| ToonError::InvalidSyntax {
                    line: i,
                    message: format!("invalid array length: {}", len_str),
                })?;

                if i >= lines.len() {
                    return Err(ToonError::InvalidSyntax {
                        line: i,
                        message: "expected field names row after array header".to_string(),
                    });
                }

                let field_line = lines[i].trim();
                i += 1;

                if field_line.is_empty() {
                    return Err(ToonError::InvalidSyntax {
                        line: i - 1,
                        message: "expected field names row after array header".to_string(),
                    });
                }

                let field_names: Vec<String> = parse_csv_row(field_line);

                let mut data_rows: Vec<Vec<String>> = Vec::new();

                for _row_idx in 0..len {
                    if i >= lines.len() {
                        return Err(ToonError::LengthMismatch {
                            declared: len,
                            found: data_rows.len(),
                        });
                    }

                    let data_line = lines[i].trim();
                    i += 1;

                    if data_line.is_empty() {
                        continue;
                    }

                    let values: Vec<String> = parse_csv_row(data_line);

                    if values.len() != field_names.len() {
                        return Err(ToonError::InvalidSyntax {
                            line: i - 1,
                            message: format!(
                                "row has {} values but expected {} fields",
                                values.len(),
                                field_names.len()
                            ),
                        });
                    }

                    data_rows.push(values);
                }

                let objects: Vec<Value> = (0..len)
                    .map(|obj_idx| {
                        let mut obj = Map::new();
                        for (field_idx, fname) in field_names.iter().enumerate() {
                            if let Some(row) = data_rows.get(obj_idx) {
                                if let Some(val) = row.get(field_idx) {
                                    let v = coerce_value(val);
                                    obj.insert(fname.clone(), v);
                                }
                            }
                        }
                        Value::Object(obj)
                    })
                    .collect();

                doc.arrays.insert(array_name, Value::Array(objects));
            }
        }

        Ok(doc)
    }

    pub fn to_json(&self) -> Value {
        let mut root = Map::new();
        if let Some(count) = self.count {
            root.insert("count".to_string(), Value::Number(count.into()));
        }
        for (key, val) in &self.arrays {
            root.insert(key.clone(), val.clone());
        }
        Value::Object(root)
    }
}

fn parse_key_val(s: &str) -> Option<(&str, &str)> {
    let eq_pos = s.find('=')?;
    let key = s[..eq_pos].trim();
    let val = s[eq_pos + 1..].trim();
    Some((key, val))
}

fn parse_array_header(s: &str) -> Option<(&str, &str)> {
    let colon_pos = s.find(':')?;
    let name = s[..colon_pos].trim();
    let len = s[colon_pos + 1..].trim();
    if name.is_empty() || len.is_empty() {
        return None;
    }
    Some((name, len))
}

fn parse_csv_row(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let chars: Vec<char> = s.chars().collect();
    let mut j = 0;

    while j < chars.len() {
        let c = chars[j];
        if c == '"' {
            if in_quotes && j + 1 < chars.len() && chars[j + 1] == '"' {
                current.push('"');
                j += 2;
                continue;
            }
            in_quotes = !in_quotes;
            j += 1;
        } else if c == ',' && !in_quotes {
            result.push(current.trim().to_string());
            current = String::new();
            j += 1;
        } else {
            current.push(c);
            j += 1;
        }
    }
    result.push(current.trim().to_string());
    result
}

fn coerce_value(s: &str) -> Value {
    let s = s.trim();
    if s.is_empty() {
        return Value::Null;
    }
    if let Ok(v) = s.parse::<i64>() {
        return Value::Number(v.into());
    }
    if let Ok(v) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(v) {
            return Value::Number(n);
        }
    }
    if s.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if s.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    Value::String(s.to_string())
}

// --- PR #2155 additions: alternative TOON format spec (name[N]{fields}: vals) ---

/// Represents a parsed scalar line: `name: value`
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedScalar {
    pub name: String,
    pub value: String,
}

/// Represents a parsed array header: `name[N]{field1,field2,...}:`
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedArrayHeader {
    pub name: String,
    pub count: usize,
    pub fields: Vec<String>,
}

/// Represents a parsed array row: `val1,val2,...`
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedArrayRow {
    pub values: Vec<String>,
}

/// Parse a scalar line: `name: value`
pub fn parse_line(input: &str) -> Result<ParsedScalar> {
    let input = input.trim();
    let parts: Vec<&str> = input.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(ToonError::InvalidSyntax(format!(
            "expected 'name: value' format, got '{}'",
            input
        )));
    }
    let name = parts[0].trim().to_string();
    let value = parts[1].trim().to_string();
    Ok(ParsedScalar { name, value })
}

/// Parse a uniform array header: `name[N]{field1,field2,...}:`
pub fn parse_uniform_array_header(input: &str) -> Result<ParsedArrayHeader> {
    let input = input.trim().trim_end_matches(':');
    let (name_part, rest) = input.split_once('[').ok_or_else(|| {
        ToonError::InvalidSyntax(format!("missing '[' in array header: {}", input))
    })?;

    let name = name_part.trim().to_string();

    let (count_part, fields_part) = rest.split_once("]{").ok_or_else(|| {
        ToonError::InvalidSyntax(format!("missing ']{{' in array header: {}", input))
    })?;

    let count = count_part
        .parse::<usize>()
        .map_err(|_| ToonError::InvalidSyntax(format!("invalid count: {}", count_part)))?;

    let fields: Vec<String> = fields_part
        .trim_end_matches('}')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    Ok(ParsedArrayHeader {
        name,
        count,
        fields,
    })
}

/// Parse a uniform array row: `val1,val2,...`
pub fn parse_array_row(input: &str, expected_fields: usize) -> Result<ParsedArrayRow> {
    let input = input.trim();
    let values: Vec<String> = input.split(',').map(|s| s.trim().to_string()).collect();

    if values.len() != expected_fields {
        return Err(ToonError::MalformedRow {
            line: 0,
            expected: expected_fields,
            found: values.len(),
        });
    }

    Ok(ParsedArrayRow { values })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_toon() {
        let input = "toon:v1\ncount=2\nzones:2\nname,temperature\nZone1,22.5\nZone2,23.0\n";
        let doc = ToonDocument::parse(input).unwrap();
        assert_eq!(doc.count, Some(2));
        assert!(doc.arrays.contains_key("zones"));
        let json = doc.to_json();
        assert_eq!(json["count"], 2);
        assert!(json["zones"].is_array());
        assert_eq!(json["zones"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_length_mismatch_missing_rows() {
        let input = "toon:v1\ncount=2\nzones:2\nname,temperature\nZone1,22.5\n";
        let err = ToonDocument::parse(input).unwrap_err();
        match err {
            ToonError::LengthMismatch { declared, found } => {
                assert_eq!(declared, 2);
                assert_eq!(found, 1);
            }
            _ => panic!("expected LengthMismatch, got {:?}", err),
        }
    }

    #[test]
    fn test_invalid_syntax_missing_field_row() {
        let input = "toon:v1\nzones:2\n";
        let err = ToonDocument::parse(input).unwrap_err();
        match err {
            ToonError::InvalidSyntax { line, .. } => {
                assert!(line > 0);
            }
            _ => panic!("expected InvalidSyntax, got {:?}", err),
        }
    }

    #[test]
    fn test_invalid_header() {
        let input = "not-toon:v1\n42\n";
        let err = ToonDocument::parse(input).unwrap_err();
        match err {
            ToonError::InvalidHeader(_) => {}
            _ => panic!("expected InvalidHeader, got {:?}", err),
        }
    }

    #[test]
    fn test_coerce_values() {
        assert!(matches!(coerce_value("42"), Value::Number(n) if n.as_i64() == Some(42)));
        assert!(matches!(coerce_value("3.14"), Value::Number(n) if n.as_f64().is_some()));
        assert_eq!(coerce_value("true"), Value::Bool(true));
        assert_eq!(coerce_value("false"), Value::Bool(false));
        assert_eq!(coerce_value(""), Value::Null);
    }

    #[test]
    fn test_roundtrip_temperature_zones() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct TemperatureZone {
            name: String,
            temperature: f64,
        }

        let zones = vec![
            TemperatureZone {
                name: "Zone1".to_string(),
                temperature: 22.5,
            },
            TemperatureZone {
                name: "Zone2".to_string(),
                temperature: 23.0,
            },
            TemperatureZone {
                name: "Zone3".to_string(),
                temperature: 21.8,
            },
        ];

        let toon = crate::to_string(&zones).unwrap();
        let parsed: Vec<TemperatureZone> = crate::from_str(&toon).unwrap();
        assert_eq!(zones.len(), parsed.len());
        for (orig, deser) in zones.iter().zip(parsed.iter()) {
            assert_eq!(orig.name, deser.name);
            assert!((orig.temperature - deser.temperature).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parse_csv_row() {
        assert_eq!(parse_csv_row("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(parse_csv_row("a, b , c"), vec!["a", "b", "c"]);
        assert_eq!(parse_csv_row("\"a,b\",c"), vec!["a,b", "c"]);
    }

    #[test]
    fn test_parse_scalar() -> Result<()> {
        let input = "setpoint: 22.0";
        let parsed = parse_line(input)?;
        assert_eq!(parsed.name, "setpoint");
        assert_eq!(parsed.value, "22.0");
        Ok(())
    }

    #[test]
    fn test_parse_uniform_array_header() -> Result<()> {
        let input = "zone_temps[3]{id,temp_c,humidity_rh}:";
        let parsed = parse_uniform_array_header(input)?;
        assert_eq!(parsed.name, "zone_temps");
        assert_eq!(parsed.count, 3);
        assert_eq!(parsed.fields, vec!["id", "temp_c", "humidity_rh"]);
        Ok(())
    }

    #[test]
    fn test_parse_array_row() -> Result<()> {
        let fields = &["id", "temp_c", "humidity_rh"];
        let input = "z0, 21.4, 45.0";
        let row = parse_array_row(input, fields.len())?;
        assert_eq!(row.values.len(), 3);
        assert_eq!(row.values[0], "z0");
        assert_eq!(row.values[1], "21.4");
        assert_eq!(row.values[2], "45.0");
        Ok(())
    }

    #[test]
    fn test_length_mismatch_error() {
        let fields = &["id", "temp_c"];
        let input = "z0, 21.4, 45.0";
        let result = parse_array_row(input, fields.len());
        assert!(matches!(
            result,
            Err(ToonError::MalformedRow {
                line: _,
                expected: 2,
                found: 3
            })
        ));
    }
}
