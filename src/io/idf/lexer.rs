// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Hand-written lexer for the EnergyPlus IDF (Input Data File) format.
//!
//! # Format Recap (see `docs/idf-import-design.md` §2.1 and §5.1)
//!
//! An IDF file is a sequence of objects, each terminated by a semicolon.
//! Fields within an object are separated by commas. Two subtleties make a
//! naïve `split(',')` / `split(';')` parser incorrect:
//!
//! 1. **Quoted strings** may contain commas, semicolons, and even other
//!    quotes (escaped via doubled `""`). The lexer tracks quote state so
//!    separators inside `"…"` are not field/object delimiters.
//! 2. **Comments** begin with `!` and run to end-of-line. They may appear
//!    at the start of a line (whole-line comment) or after a field on the
//!    same line (trailing comment). A `!` inside a quoted string is *not*
//!    a comment introducer.
//!
//! In addition, EnergyPlus object names are **case-insensitive**, so the
//! lexer returns the object name in its original case (the parser does
//! case-insensitive comparison when classifying objects).
//!
//! # Output
//!
//! The lexer yields a `Vec<RawObject>` where each `RawObject` is the
//! text span between consecutive object terminators, with comments
//! stripped. The parser then turns each `RawObject` into an
//! [`IdfObject`](super::parser::IdfObject) by splitting on field commas
//! (still respecting quoted strings).

use super::error::IdfError;

/// A single object as emitted by the lexer: the type name (preserving case),
/// the body text after the type name, and the line on which the object
/// starts in the source file (1-indexed).
///
/// `body` is the **raw** text between the type name and the terminating
/// semicolon, with comments stripped. It still contains the field-level
/// commas and quoted strings — the parser is responsible for splitting
/// fields while respecting quoted strings.
#[derive(Debug, Clone, PartialEq)]
pub struct RawObject {
    pub object_type: String,
    pub body: String,
    pub line: usize,
}

/// Tokenize an IDF document into a sequence of [`RawObject`]s.
///
/// # Errors
///
/// Returns [`IdfError::Parse`] if:
/// - A quoted string is started but never closed before EOF.
/// - An object terminator (`;`) appears inside a quoted string — this is
///   technically allowed by some legacy parsers but EnergyPlus itself
///   rejects it, so we do too.
pub fn tokenize(source: &str) -> Result<Vec<RawObject>, IdfError> {
    let mut objects: Vec<RawObject> = Vec::new();
    let mut current_type: Option<String> = None;
    let mut current_body = String::new();
    let mut current_start_line: usize = 1;

    // Per-object state.
    let mut in_object = false;

    // Per-document state.
    let mut in_quotes = false;
    // Inside an open quoted string, set to `true` when we just emitted
    // the FIRST `"` of a `""` escape pair so the next iteration can
    // detect that its `"` is the matching SECOND half. Reset on every
    // non-quote character (and on a real closing quote).
    let mut prev_quote_open: bool = false;
    let mut line: usize = 1;

    // Walk the source character by character. Using a byte iterator would
    // be faster but EnergyPlus files are 7-bit ASCII in practice; we keep
    // the readable char-based form for clarity.
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Track line numbers for error messages and RawObject.line.
        if c == '\n' {
            line += 1;
            // Don't increment i here — we still want to process the
            // newline as part of whitespace stripping below.
        }

        // --- Inside an open quoted string -------------------------------
        if in_quotes {
            match c {
                '"' => {
                    // Inside a quoted string, a `"` either closes the
                    // string or is the second half of a `""` escape
                    // pair. We can tell them apart by checking whether
                    // the LAST character we pushed onto `current_body`
                    // is also a `"` that did NOT close the string —
                    // if so, this `"` is the matching half of the
                    // escape and the string stays open. Both `"`s are
                    // pushed so the parser can detect the pair later.
                    let prev_was_unescaped_quote = current_body.ends_with('"') && prev_quote_open;
                    current_body.push('"');
                    if prev_was_unescaped_quote {
                        // Second half of an escape pair — keep the
                        // string open.
                        prev_quote_open = false;
                    } else if i + 1 >= chars.len() || chars[i + 1] != '"' {
                        // Lone closing `"`.
                        in_quotes = false;
                        prev_quote_open = false;
                    } else {
                        // First half of an escape pair — the next char
                        // is also `"`. Remember so we know the *next*
                        // iteration's `"` is part of the same pair.
                        prev_quote_open = true;
                    }
                }
                ';' if !in_object => {
                    // Shouldn't happen — semicolons only matter as
                    // object terminators when not inside a string and
                    // when not already inside an object. Treat as a
                    // parse error to surface malformed input early.
                    return Err(IdfError::parse_error(
                        line,
                        "Stray ';' encountered inside a quoted string",
                    ));
                }
                _ => {
                    current_body.push(c);
                    prev_quote_open = false;
                }
            }
            i += 1;
            continue;
        }

        // --- Outside quotes ---------------------------------------------
        match c {
            '!' => {
                // Comment: skip to end of line. `!` inside a quoted
                // string is handled by the in_quotes branch above.
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                // Loop will re-evaluate; the '\n' branch increments line.
                continue;
            }
            '"' => {
                in_quotes = true;
                current_body.push('"');
            }
            ',' if in_object => {
                // Field separator inside an object body.
                current_body.push(',');
            }
            ';' => {
                // Object terminator. Close out the current object.
                if in_object {
                    let body = current_body.trim().to_string();
                    let object_type = current_type
                        .take()
                        .expect("in_object guarantees current_type is Some");
                    objects.push(RawObject {
                        object_type,
                        body,
                        line: current_start_line,
                    });
                    current_body.clear();
                    in_object = false;
                }
                // ';' outside any object (e.g. lone terminator at end of
                // file) is silently ignored, matching EnergyPlus.
            }
            '\n' => {
                // Collapse runs of newlines (and the spaces around them)
                // into a single space within an object body, but only if
                // we're collecting fields. Field values that are pure
                // numbers (e.g. "0.04") don't contain newlines, and
                // quoted strings preserve their own whitespace.
                if in_object && !current_body.ends_with(' ') && !current_body.is_empty() {
                    current_body.push(' ');
                }
            }
            ' ' | '\t' | '\r' => {
                // Collapse other whitespace the same way.
                if in_object && !current_body.ends_with(' ') && !current_body.is_empty() {
                    current_body.push(' ');
                }
            }
            _ => {
                if !in_object {
                    // Starting a new object — capture the type name
                    // character-by-character up to the next comma,
                    // newline, or quote.
                    let mut type_name = String::new();
                    while i < chars.len() {
                        let nc = chars[i];
                        if nc == ',' || nc == '\n' || nc == '\r' || nc == ';' {
                            break;
                        }
                        // Skip leading inline comments that begin before
                        // the object name (rare but legal in IDF).
                        if nc == '!' {
                            while i < chars.len() && chars[i] != '\n' {
                                i += 1;
                            }
                            break;
                        }
                        type_name.push(nc);
                        i += 1;
                    }
                    let trimmed = type_name.trim().to_string();
                    if !trimmed.is_empty() {
                        current_type = Some(trimmed);
                        current_start_line = line;
                        current_body.clear();
                        in_object = true;
                        // Skip the delimiter (comma, newline, etc.)
                        // that ended the type name — it is NOT part of
                        // the first field. Do NOT advance past the
                        // terminating `;` though, since some legacy
                        // parsers allow `ObjectType;` (no fields).
                        if i < chars.len() && chars[i] != ';' {
                            i += 1;
                            // Skip any run of whitespace and additional
                            // blank lines between the type name and the
                            // first field.
                            while i < chars.len()
                                && (chars[i] == ' ' || chars[i] == '\t' || chars[i] == '\r')
                            {
                                i += 1;
                            }
                            // Don't consume newlines here — they
                            // re-enter the outer loop and update `line`.
                            // The body builder will insert single spaces
                            // between fields, which is what we want.
                            if i < chars.len() && chars[i] == '\n' {
                                // Skip leading blank lines so the body
                                // doesn't start with a space.
                            }
                        }
                        continue;
                    }
                } else {
                    current_body.push(c);
                }
            }
        }
        i += 1;
    }

    if in_quotes {
        return Err(IdfError::parse_error(
            line,
            "Unterminated quoted string at end of file",
        ));
    }

    if in_object {
        // Trailing object with no terminating ';'.
        let body = current_body.trim().to_string();
        if let Some(object_type) = current_type.take() {
            objects.push(RawObject {
                object_type,
                body,
                line: current_start_line,
            });
        }
    }

    Ok(objects)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_objects() {
        assert!(tokenize("").unwrap().is_empty());
        assert!(tokenize("\n\n   \n").unwrap().is_empty());
    }

    #[test]
    fn single_version_object() {
        let src = "Version, 25.2;";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].object_type, "Version");
        assert_eq!(objs[0].body, "25.2");
        assert_eq!(objs[0].line, 1);
    }

    #[test]
    fn quoted_comma_is_not_a_field_separator() {
        // The comma inside "Hello, World!" must NOT split the field.
        let src = r#"Material, "Hello, World!", OtherField;"#;
        let objs = tokenize(src).unwrap();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].object_type, "Material");
        // Body should still contain both commas so the parser can split
        // them correctly while respecting the quote.
        assert_eq!(objs[0].body, r#""Hello, World!", OtherField"#);
    }

    #[test]
    fn multiline_object_collects_fields() {
        let src = "Material,\n  OUTR_WOOD,\n  MediumSmooth,\n  0.010, 0.115, 540, 1210;";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].object_type, "Material");
        assert_eq!(
            objs[0].body,
            "OUTR_WOOD, MediumSmooth, 0.010, 0.115, 540, 1210"
        );
        assert_eq!(objs[0].line, 1);
    }

    #[test]
    fn trailing_line_comment_after_last_field_is_stripped() {
        let src = "Material, OUTR_WOOD, 0.010;  ! trailing comment\n";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].body, "OUTR_WOOD, 0.010");
    }

    #[test]
    fn whole_line_comment_is_skipped() {
        let src = "! This is a comment\nVersion, 25.2;\n! another comment\n";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].object_type, "Version");
        assert_eq!(objs[0].body, "25.2");
    }

    #[test]
    fn unterminated_quote_returns_error() {
        let src = "Material, \"unterminated, 0.010;";
        let err = tokenize(src).unwrap_err();
        match err {
            IdfError::Parse { line, .. } => assert_eq!(line, 1),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn case_is_preserved_in_object_name() {
        // The lexer preserves case; the parser does case-insensitive
        // matching when classifying objects.
        let src = "VERSION, 25.2;";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs[0].object_type, "VERSION");
    }

    #[test]
    fn line_number_tracks_object_start() {
        let src = "\n\nVersion, 25.2;";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs[0].line, 3);
    }

    #[test]
    fn multiple_objects_in_one_file() {
        let src = "Version, 25.2;\nTimestep, 1;\n";
        let objs = tokenize(src).unwrap();
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].object_type, "Version");
        assert_eq!(objs[1].object_type, "Timestep");
        assert_eq!(objs[1].line, 2);
    }
}
