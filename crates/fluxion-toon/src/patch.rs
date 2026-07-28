//! LLM Response Patch Parser
//!
//! Parses TOON-formatted parameter patches from LLM responses,
//! handling markdown codeblock wrapping automatically.

use crate::error::{ToonError, Result};

/// A parsed parameter patch from an LLM response.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPatch {
    /// Parameter name (e.g., "wall_r_value", "heating_setpoint").
    pub param: String,
    /// Numeric value to set.
    pub value: f64,
    /// Optional target identifier (e.g., zone ID, surface name).
    pub target: Option<String>,
}

/// Parse a TOON patch from an LLM response string.
///
/// Strips markdown codeblock fences (` ```toon`, ` ```ton`, etc.) and
/// parses the extracted content into a `ModelPatch`.
///
/// # Arguments
///
/// * `input` - Raw LLM response that may contain TOON in codeblocks
///
/// # Returns
///
/// A `ModelPatch` on success, or a `ToonError::MalformedPatch` on failure.
///
/// # Example
///
/// ```
/// use fluxion_toon::patch::parse_toon_patch;
///
/// let patch = parse_toon_patch("wall_r_value[1]{name,r_value}: heavy_wall, 3.5").unwrap();
/// assert_eq!(patch.param, "wall_r_value");
/// assert_eq!(patch.value, 3.5);
/// assert_eq!(patch.target.as_deref(), Some("heavy_wall"));
/// ```
pub fn parse_toon_patch(input: &str) -> Result<ModelPatch> {
    let content = strip_codeblock_fences(input.trim());
    parse_model_patch(content)
}

/// Strip markdown codeblock fences from a string.
///
/// Handles:
/// - ` ```toon ... ``` `
/// - ` ```ton ... ``` `
/// - ` ```json ... ``` `
/// - Single backtick variants
/// - Trailing fences without opening
fn strip_codeblock_fences(s: &str) -> &str {
    let s = s.trim();

    // Strip opening fence: ```toon, ```ton, ```json, ``` etc.
    for fence in [
        "```toon\n",
        "```toon\r\n",
        "```ton\n",
        "```ton\r\n",
        "```json\n",
        "```json\r\n",
        "```\n",
        "```\r\n",
        "```toon",
        "```ton",
        "```json",
        "```",
    ] {
        if let Some(rest) = s.strip_prefix(fence) {
            let rest = rest.trim();
            // Strip trailing fence ```
            if let Some(stripped) = rest.strip_suffix("```") {
                return stripped.trim();
            }
            if let Some(stripped) = rest.strip_suffix("```\n") {
                return stripped.trim();
            }
            if let Some(stripped) = rest.strip_suffix("```\r\n") {
                return stripped.trim();
            }
            return rest;
        }
    }

    s
}

/// Parse a model patch from a stripped TOON string.
///
/// Format: `param[index]{fields}: target, value`
/// or: `param[index]{fields}: value`
fn parse_model_patch(s: &str) -> Result<ModelPatch> {
    let s = s.trim();

    if s.is_empty() {
        return Err(ToonError::MalformedPatch("empty input".to_string()));
    }

    // Find the colon separator (required)
    let colon_pos = s.find(':').ok_or_else(|| {
        ToonError::MalformedPatch(format!("missing ':' separator in patch: {}", s))
    })?;

    let before_colon = &s[..colon_pos];
    let after_colon = s[colon_pos + 1..].trim();

    if after_colon.is_empty() {
        return Err(ToonError::MalformedPatch(format!(
            "missing value after ':' in patch: {}",
            s
        )));
    }

    // Parse before colon: param[index]{fields}
    // Find the [ index ] and { fields } if present
    let (param, _index, _fields) = parse_param_header(before_colon)?;

    // Parse after colon: [target,] value
    let (target, value) = parse_patch_value(after_colon)?;

    Ok(ModelPatch {
        param,
        value,
        target,
    })
}

fn parse_param_header(s: &str) -> Result<(String, Option<String>, Option<String>)> {
    let s = s.trim();

    if s.is_empty() {
        return Err(ToonError::MalformedPatch("empty parameter header".to_string()));
    }

    let mut param_end = s.len();

    // Check for [index] suffix
    let index = if let Some(start) = s.find('[') {
        if start > 0 {
            param_end = start;
        } else {
            return Err(ToonError::MalformedPatch(format!(
                "unexpected '[' at start of param header: {}",
                s
            )));
        }
        let end = s.find(']').ok_or_else(|| {
            ToonError::MalformedPatch(format!("unclosed '[' in param header: {}", s))
        })?;
        if end <= start + 1 {
            return Err(ToonError::MalformedPatch(format!(
                "empty index in param header: {}",
                s
            )));
        }
        let idx = s[start + 1..end].trim();
        if idx.is_empty() {
            return Err(ToonError::MalformedPatch(format!(
                "empty index in param header: {}",
                s
            )));
        }
        param_end = start;
        Some(idx.to_string())
    } else {
        None
    };

    // Check for {fields} suffix
    let fields = if let Some(start) = s[param_end..].find('{') {
        let abs_start = param_end + start;
        if start > 0 && param_end == s.len() {
            // {fields} after [index]
        } else if start > 0 {
            param_end = abs_start;
        } else {
            return Err(ToonError::MalformedPatch(format!(
                "unexpected '{{' in param header: {}",
                s
            )));
        }
        let end = s[param_end..].find('}').ok_or_else(|| {
            ToonError::MalformedPatch(format!("unclosed '{{' in param header: {}", s))
        })?;
        let abs_end = param_end + end;
        if end <= start + 1 {
            return Err(ToonError::MalformedPatch(format!(
                "empty fields in param header: {}",
                s
            )));
        }
        let flds = s[param_end + 1..abs_end].trim();
        if flds.is_empty() {
            return Err(ToonError::MalformedPatch(format!(
                "empty fields in param header: {}",
                s
            )));
        }
        param_end = abs_start;
        Some(flds.to_string())
    } else {
        None
    };

    let param = s[..param_end].trim().to_string();
    if param.is_empty() {
        return Err(ToonError::MalformedPatch(format!(
            "empty parameter name in header: {}",
            s
        )));
    }

    Ok((param, index, fields))
}

fn parse_patch_value(s: &str) -> Result<(Option<String>, f64)> {
    let s = s.trim();

    // Split on last comma to separate target from value
    // This handles cases like "heavy_wall, 3.5" where target has no commas
    // but also "component, path, to, value" style if needed

    let comma_pos = s.rfind(',').ok_or_else(|| {
        // No comma means the whole string is just the value
        let value = parse_number(s)?;
        return Ok((None, value));
    })?;

    if comma_pos == 0 {
        return Err(ToonError::MalformedPatch(format!(
            "missing value before ',' in patch value: {}",
            s
        )));
    }

    let target_part = s[..comma_pos].trim();
    let value_part = s[comma_pos + 1..].trim();

    if value_part.is_empty() {
        return Err(ToonError::MalformedPatch(format!(
            "missing value after ',' in patch value: {}",
            s
        )));
    }

    let value = parse_number(value_part)?;
    let target = if target_part.is_empty() {
        None
    } else {
        Some(target_part.to_string())
    };

    Ok((target, value))
}

fn parse_number(s: &str) -> Result<f64> {
    let s = s.trim();

    // Try parsing as f64 directly first
    s.parse::<f64>().map_err(|_| {
        ToonError::MalformedPatch(format!("cannot parse '{}' as a number", s))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_patch() {
        let patch = parse_toon_patch("wall_r_value[1]{name,r_value}: heavy_wall, 3.5").unwrap();
        assert_eq!(patch.param, "wall_r_value");
        assert_eq!(patch.value, 3.5);
        assert_eq!(patch.target.as_deref(), Some("heavy_wall"));
    }

    #[test]
    fn test_parse_patch_with_zone_target() {
        let patch = parse_toon_patch("heating_setpoint[1]{zone,value}: z0, 21.0").unwrap();
        assert_eq!(patch.param, "heating_setpoint");
        assert_eq!(patch.value, 21.0);
        assert_eq!(patch.target.as_deref(), Some("z0"));
    }

    #[test]
    fn test_parse_patch_no_target() {
        let patch = parse_toon_patch("temperature[0]: 25.0").unwrap();
        assert_eq!(patch.param, "temperature");
        assert_eq!(patch.value, 25.0);
        assert_eq!(patch.target, None);
    }

    #[test]
    fn test_parse_patch_integer_value() {
        let patch = parse_toon_patch("count[1]: items, 42").unwrap();
        assert_eq!(patch.param, "count");
        assert_eq!(patch.value, 42.0);
        assert_eq!(patch.target.as_deref(), Some("items"));
    }

    #[test]
    fn test_parse_patch_no_index_no_fields() {
        let patch = parse_toon_patch("setpoint: office, 22.5").unwrap();
        assert_eq!(patch.param, "setpoint");
        assert_eq!(patch.value, 22.5);
        assert_eq!(patch.target.as_deref(), Some("office"));
    }

    #[test]
    fn test_strip_codeblock_toon() {
        let input = "```toon\nwall_r_value[1]{name,r_value}: heavy_wall, 3.5\n```";
        let patch = parse_toon_patch(input).unwrap();
        assert_eq!(patch.param, "wall_r_value");
        assert_eq!(patch.value, 3.5);
        assert_eq!(patch.target.as_deref(), Some("heavy_wall"));
    }

    #[test]
    fn test_strip_codeblock_ton() {
        let input = "```ton\nheating_setpoint[1]{zone,value}: z0, 21.0\n```";
        let patch = parse_toon_patch(input).unwrap();
        assert_eq!(patch.param, "heating_setpoint");
        assert_eq!(patch.value, 21.0);
    }

    #[test]
    fn test_strip_codeblock_json() {
        let input = "```json\nzone_temps[3]{id,temp_c}: zone3, 18.5\n```";
        let patch = parse_toon_patch(input).unwrap();
        assert_eq!(patch.param, "zone_temps");
        assert_eq!(patch.value, 18.5);
    }

    #[test]
    fn test_strip_codeblock_crlf() {
        let input = "```toon\r\nsetpoint: office, 22.5\r\n```";
        let patch = parse_toon_patch(input).unwrap();
        assert_eq!(patch.param, "setpoint");
        assert_eq!(patch.value, 22.5);
    }

    #[test]
    fn test_strip_codeblock_with_leading_text() {
        let input = "Here is the patch:\n```toon\nwall_r_value[1]: heavy_wall, 3.5\n```\nPlease apply this.";
        let patch = parse_toon_patch(input).unwrap();
        assert_eq!(patch.param, "wall_r_value");
        assert_eq!(patch.value, 3.5);
    }

    #[test]
    fn test_malformed_missing_colon() {
        let result = parse_toon_patch("wall_r_value[1]{name,r_value} heavy_wall, 3.5");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToonError::MalformedPatch(_)));
    }

    #[test]
    fn test_malformed_missing_value() {
        let result = parse_toon_patch("wall_r_value[1]{name,r_value}:");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToonError::MalformedPatch(_)));
    }

    #[test]
    fn test_malformed_invalid_value() {
        let result = parse_toon_patch("wall_r_value[1]{name,r_value}: heavy_wall, not_a_number");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToonError::MalformedPatch(_)));
    }

    #[test]
    fn test_malformed_empty_after_colon_comma() {
        let result = parse_toon_patch("wall_r_value[1]{name,r_value}: heavy_wall, ");
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_empty_input() {
        let result = parse_toon_patch("");
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_whitespace_only() {
        let result = parse_toon_patch("   \n  ");
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_codeblock_no_content() {
        // ```toon with nothing after the opening fence before closing fence
        let result = parse_toon_patch("```toon\nzone_temps[3]{id,temp_c}:```");
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_value() {
        let patch = parse_toon_patch("offset[1]: sensor, -5.5").unwrap();
        assert_eq!(patch.param, "offset");
        assert_eq!(patch.value, -5.5);
        assert_eq!(patch.target.as_deref(), Some("sensor"));
    }

    #[test]
    fn test_target_with_commas() {
        // Target "a, b" contains comma — value is "1.5", target is "a, b"
        let patch = parse_toon_patch("param[1]: a, b, 1.5").unwrap();
        assert_eq!(patch.param, "param");
        assert_eq!(patch.value, 1.5);
        assert_eq!(patch.target.as_deref(), Some("a, b"));
    }

    #[test]
    fn test_strip_fences_trailing_only() {
        // Just a trailing fence without opening
        let patch = parse_toon_patch("setpoint: office, 22.5\n```").unwrap();
        assert_eq!(patch.param, "setpoint");
        assert_eq!(patch.value, 22.5);
    }
}
