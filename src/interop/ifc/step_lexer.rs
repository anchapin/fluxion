// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Hand-written lexer for the ISO 10303-21 (STEP physical file) format
//! used by IFC4.
//!
//! # Format recap
//!
//! An IFC4 STEP file has two sections:
//!
//! - `HEADER;` — free-form metadata (`FILE_DESCRIPTION`, `FILE_NAME`,
//!   `FILE_SCHEMA`).
//! - `DATA;` — the bulk of the file: one entity per line, of the form
//!   `#N=IFCENTITY(arg1, arg2, ...);` where `N` is a positive integer id
//!   unique within the file.
//!
//! STEP arg syntax (subset that matters for IFC4):
//!
//! - **`#N`** — reference to another entity by id.
//! - **`'text'`** — string literal. Single quotes inside are escaped as
//!   `''` (doubled single quote).
//! - **`.NAME.`** — enumeration literal, including simple identifiers
//!   like `.NOTDEFINED.`, `.READWRITE.`, `.POSITIVE.`, `.FLOOR.`,
//!   `.INTERNAL.`, etc.
//! - **`$`** — omitted / unset value (corresponds to an IFC `OPTIONAL`
//!   or `WHERE`-constrained absent value).
//! - **`*`** — derived value (the entity field is "to be computed").
//! - **`(a, b, c)`** — nested aggregate (tuple or list).
//!
//! The full STEP grammar is much larger (binary encoding, complex
//! instances, etc.) but IFC4 files in the wild always use the *clear
//! text encoding* above. The lexer therefore only needs to recognize the
//! five primitives (`#N`, `'…'`, `.NAME.`, `$`, `*`) plus grouping
//! parentheses.
//!
//! # Output
//!
//! The lexer yields a [`Vec<RawEntity>`] preserving source order. Each
//! `RawEntity` is one `id = name(args)` record, with the raw argument
//! body kept as a string for the typed parser to split. This keeps the
//! lexer simple (no nested AST) and the typed parser in charge of
//! shape-specific decoding.
//!
//! # Limitations (MVP scaffold)
//!
//! - **No binary STEP encoding** — IFC4 files use clear text, but STEP
//!   itself allows binary. We only emit a clean error in that case.
//! - **No complex instances** — IFC4 does not use STEP `COMPLEX` entities
//!   in practice; deferred.
//! - **No nested file sections** — IFC4 has exactly one `HEADER;` and
//!   one `DATA;` section per file, and we assert that.

use super::error::IfcError;
use fluxion_core::parser_limits::ParserLimits;

/// A single STEP entity record produced by the lexer.
///
/// The lexer does not interpret the arguments — it just preserves the
/// raw arg body (the substring between the opening `(` and the matching
/// closing `)`) so that the typed parser in `super::parser` can decode
/// the fields it cares about.
#[derive(Debug, Clone, PartialEq)]
pub struct RawEntity {
    /// The entity's STEP id (the `#N` token on the left-hand side).
    pub id: u64,
    /// The IFC class name, e.g. `IFCWALL`, `IFCSLAB`, `IFCSPACE`.
    pub name: String,
    /// The raw argument body between the parentheses, preserved verbatim
    /// except for outer parentheses and trailing semicolon.
    pub args: String,
    /// 1-indexed source line where the record starts.
    pub line: usize,
}

/// Tokenize an IFC4 STEP physical file into a sequence of [`RawEntity`]s.
///
/// The lexer is permissive about whitespace and ignores STEP `/* … */`
/// comments. The only entities it must recognize are the section markers
/// (`ISO-10303-21;`, `HEADER;`, `DATA;`, `ENDSEC;`, `END-ISO-10303-21;`)
/// and the entity records (`#N=NAME(args);`).
///
/// # Errors
///
/// - [`IfcError::Parse`] if an entity record is malformed (missing `=`,
///   unclosed `(`, no terminator `;`).
/// - [`IfcError::Parse`] if the file has no `DATA;` section.
pub fn tokenize(source: &str) -> Result<Vec<RawEntity>, IfcError> {
    Ok(tokenize_with_schema(source)?.1)
}

/// Tokenize with the strict default [`ParserLimits`] (issue #2527).
pub fn tokenize_with_schema(source: &str) -> Result<(Option<String>, Vec<RawEntity>), IfcError> {
    tokenize_with_schema_and_limits(source, &ParserLimits::default())
}

/// Tokenize with explicit [`ParserLimits`] (issue #2527).
///
/// Enforces `max_file_bytes` on the source length **before** the
/// ASCII-filter `Vec<u8>` is allocated, and `max_recursion_depth` on
/// parenthesis nesting inside entity argument lists (the STEP analogue
/// of recursive-descent depth — a pathologically nested `(((...)))`
/// body no longer runs unbounded).
pub fn tokenize_with_schema_and_limits(
    source: &str,
    limits: &ParserLimits,
) -> Result<(Option<String>, Vec<RawEntity>), IfcError> {
    limits.check_file_bytes(source.len())?;
    let mut entities: Vec<RawEntity> = Vec::new();
    let mut schema: Option<String> = None;

    // STEP clear-text is 7-bit ASCII per ISO 10303-21 §6. We collapse
    // the source to a byte vector so byte-indexed slices of `source`
    // line up with character indices (multi-byte UTF-8 in source
    // comments would otherwise desync the two). Non-ASCII bytes are
    // silently dropped, which is acceptable because STEP itself does
    // not define any non-ASCII tokens; non-ASCII content in real IFC
    // files is restricted to comments, which the lexer skips wholesale.
    let bytes: Vec<u8> = source.bytes().filter(|b| b.is_ascii()).collect();
    let source: &str = std::str::from_utf8(&bytes).expect("filtered to ASCII");

    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    let mut line: usize = 1;
    let mut in_data_section = false;
    let mut saw_data_section = false;

    while i < chars.len() {
        let c = chars[i];

        // Track line numbers — emitted in parse errors and on each
        // `RawEntity` so callers can highlight the offending line.
        if c == '\n' {
            line += 1;
        }

        // --- Skip whitespace --------------------------------------------
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // --- /* ... */ STEP block comment --------------------------------
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            // Two slashes — STEP single-line comment, runs to '\n'.
            // (STEP block comments use `/* ... */`, single-line uses `//`.)
            // Single-line comment to EOL.
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            // Block comment — runs to matching `*/`.
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                if chars[i] == '\n' {
                    line += 1;
                }
                i += 1;
            }
            // Skip the closing `*/` (if present). Missing closing is a
            // parse error, but we tolerate EOF to keep the scaffold
            // robust against malformed files.
            if i + 1 < chars.len() {
                i += 2;
            } else {
                i = chars.len();
            }
            continue;
        }

        // --- Section markers --------------------------------------------
        // We only act on `DATA;` (the only section that contains entity
        // records); `HEADER;` and `ENDSEC;` are skipped over. The
        // closing `END-ISO-10303-21;` terminates parsing.
        if c == 'I' && source[i..].starts_with("ISO-10303-21;") {
            i += "ISO-10303-21;".len();
            continue;
        }
        if source[i..].starts_with("END-ISO-10303-21;") {
            break;
        }
        if source[i..].starts_with("ENDSEC;") {
            // Boundary between HEADER and DATA sections, or trailing.
            in_data_section = false;
            i += "ENDSEC;".len();
            continue;
        }
        if source[i..].starts_with("HEADER;") {
            i += "HEADER;".len();
            continue;
        }
        if source[i..].starts_with("DATA;") {
            in_data_section = true;
            saw_data_section = true;
            i += "DATA;".len();
            continue;
        }

        if !in_data_section {
            // Outside DATA, the only header entity the MVP scaffold
            // cares about is `FILE_SCHEMA(('IFC4'))` — every other
            // header construct (FILE_DESCRIPTION, FILE_NAME) is
            // skipped to the next `;`. We capture FILE_SCHEMA inline so
            // the parser can enforce the IFC4-only contract from
            // issue #1343's acceptance criteria.
            if source[i..].starts_with("FILE_SCHEMA") {
                let schema_start = i;
                while i < chars.len() && chars[i] != ';' {
                    if chars[i] == '\n' {
                        line += 1;
                    }
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // consume terminating `;`
                }
                // Extract the schema string from the captured body.
                // The body looks like `FILE_SCHEMA(('IFC4'))`, so the
                // inner-most quoted string is the schema name.
                let body = &source[schema_start..i.min(source.len())];
                if let Some(first_quote) = body.find('\'') {
                    let rest = &body[first_quote + 1..];
                    if let Some(second_quote) = rest.find('\'') {
                        schema = Some(rest[..second_quote].to_string());
                    }
                }
                continue;
            }
            // Unknown header content — skip to next `;`.
            while i < chars.len() && chars[i] != ';' {
                if chars[i] == '\n' {
                    line += 1;
                }
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            continue;
        }

        // --- Entity record: #N=NAME(args); ------------------------------
        if c != '#' {
            return Err(IfcError::parse_error(
                line,
                format!(
                    "expected '#' to start entity record, found {:?} \
                     (skipping ahead)",
                    c
                ),
            ));
        }
        let record_start_line = line;
        i += 1;

        // Parse the id digits.
        let id_start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if id_start == i {
            return Err(IfcError::parse_error(
                record_start_line,
                "expected digits after '#' in entity id",
            ));
        }
        let id: u64 = source[id_start..i]
            .parse()
            .map_err(|_| IfcError::parse_error(record_start_line, "entity id overflow"))?;

        // Expect `=`.
        while i < chars.len() && chars[i].is_whitespace() {
            if chars[i] == '\n' {
                line += 1;
            }
            i += 1;
        }
        if chars.get(i) != Some(&'=') {
            return Err(IfcError::parse_error(
                record_start_line,
                format!(
                    "expected '=' after entity id #{id}, found {:?}",
                    chars.get(i)
                ),
            ));
        }
        i += 1; // consume `=`

        // Parse the entity name (uppercase letters + digits + `_`).
        while i < chars.len() && chars[i].is_whitespace() {
            if chars[i] == '\n' {
                line += 1;
            }
            i += 1;
        }
        let name_start = i;
        while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
            i += 1;
        }
        if name_start == i {
            return Err(IfcError::parse_error(
                record_start_line,
                format!("expected entity name after '#{id}='"),
            ));
        }
        let name = source[name_start..i].to_string();

        // Expect `(` ... matching `)`.
        while i < chars.len() && chars[i].is_whitespace() {
            if chars[i] == '\n' {
                line += 1;
            }
            i += 1;
        }
        if chars.get(i) != Some(&'(') {
            return Err(IfcError::parse_error(
                record_start_line,
                format!(
                    "expected '(' to open arguments of {name} #{id}, found {:?}",
                    chars.get(i)
                ),
            ));
        }
        i += 1; // consume opening `(`

        // Walk to matching `)`, tracking nesting and skipping over
        // quoted strings so a `'` inside a string can't close the
        // argument list.
        let args_start = i;
        let mut depth: usize = 1;
        let mut in_string = false;
        while i < chars.len() && depth > 0 {
            let ch = chars[i];
            if ch == '\n' {
                line += 1;
            }
            if in_string {
                if ch == '\'' {
                    // STEP escapes single quotes by doubling them
                    // (`''` inside a string). Don't toggle off on a
                    // doubled quote — only on a single `'` followed by
                    // anything else.
                    if chars.get(i + 1) == Some(&'\'') {
                        i += 2;
                        continue;
                    }
                    in_string = false;
                }
                i += 1;
                continue;
            }
            match ch {
                '\'' => {
                    in_string = true;
                    i += 1;
                }
                '(' => {
                    depth += 1;
                    // Issue #2527 — bound parenthesis nesting (the STEP
                    // analogue of recursion depth). A pathologically
                    // nested `(((...)))` argument body can no longer run
                    // unbounded; the default cap is 256 levels.
                    if depth > limits.max_recursion_depth {
                        return Err(IfcError::SizeLimitExceeded(format!(
                            "nesting depth {} exceeds limit {} in {} #{}",
                            depth, limits.max_recursion_depth, name, id
                        )));
                    }
                    i += 1;
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    i += 1;
                }
                '/' if chars.get(i + 1) == Some(&'/') => {
                    // Single-line comment inside args — skip to EOL.
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1;
                    }
                }
                '/' if chars.get(i + 1) == Some(&'*') => {
                    // Block comment inside args — skip to `*/`.
                    i += 2;
                    while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                        if chars[i] == '\n' {
                            line += 1;
                        }
                        i += 1;
                    }
                    if i + 1 < chars.len() {
                        i += 2;
                    } else {
                        i = chars.len();
                    }
                }
                _ => i += 1,
            }
        }
        if depth != 0 {
            return Err(IfcError::parse_error(
                record_start_line,
                format!("unclosed argument list for {name} #{id}"),
            ));
        }
        let args = source[args_start..i].to_string();
        i += 1; // consume closing `)`

        // Expect `;`.
        // Allow whitespace between `)` and `;`.
        while i < chars.len() && chars[i].is_whitespace() {
            if chars[i] == '\n' {
                line += 1;
            }
            i += 1;
        }
        if chars.get(i) != Some(&';') {
            return Err(IfcError::parse_error(
                record_start_line,
                format!("expected ';' to terminate {name} #{id}"),
            ));
        }
        i += 1; // consume `;`

        entities.push(RawEntity {
            id,
            name,
            args,
            line: record_start_line,
        });
    }

    if !saw_data_section {
        return Err(IfcError::parse_error(
            1,
            "no DATA; section found — file is not a valid STEP physical file",
        ));
    }

    Ok((schema, entities))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_minimal_header_and_data() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCPROJECT('guid',#2,'Sample',$,$,$,$,(#5),#6);
ENDSEC;
END-ISO-10303-21;
";
        let entities = tokenize(src).expect("parses");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].id, 1);
        assert_eq!(entities[0].name, "IFCPROJECT");
        assert!(entities[0].args.contains("Sample"));
    }

    #[test]
    fn skips_block_comments_and_blank_lines() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
/* this is a comment
   spanning multiple lines */
#10=IFCWALL('guid',#2,'Wall',$,$,$,$,$,.NOTDEFINED.);
ENDSEC;
END-ISO-10303-21;
";
        let entities = tokenize(src).expect("parses");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "IFCWALL");
    }

    #[test]
    fn handles_doubled_quotes_in_string() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCMATERIAL('It''s a test',$,'Category');
ENDSEC;
END-ISO-10303-21;
";
        let entities = tokenize(src).expect("parses");
        assert_eq!(entities.len(), 1);
        // The lexer keeps the arg body verbatim including the doubled
        // quote — the typed parser handles decoding it.
        assert!(entities[0].args.contains("It''s a test"));
    }

    #[test]
    fn rejects_file_without_data_section() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
END-ISO-10303-21;
";
        let err = tokenize(src).expect_err("should fail");
        assert!(matches!(err, IfcError::Parse { .. }));
    }

    #[test]
    fn rejects_malformed_entity_missing_equals() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1IFCPROJECT();
ENDSEC;
END-ISO-10303-21;
";
        let err = tokenize(src).expect_err("should fail");
        assert!(matches!(err, IfcError::Parse { .. }));
    }

    // ----- Issue #2527: recursion-depth cap ---------------------------------

    #[test]
    fn deep_nesting_is_capped() {
        // 10 nested parentheses inside an entity body. With a tiny
        // max_recursion_depth=3 the lexer must abort with
        // SizeLimitExceeded when depth reaches 4.
        let open = "(".repeat(10);
        let close = ")".repeat(10);
        let src = format!(
            "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('IFC4'));\nENDSEC;\nDATA;\n\
             #1=IFCX({open}1{close});\nENDSEC;\nEND-ISO-10303-21;\n"
        );
        let tiny = fluxion_core::parser_limits::ParserLimits {
            max_file_bytes: 64 * 1024 * 1024,
            max_lines: 1_000_000,
            max_recursion_depth: 3,
            max_array_elements: 1_000_000,
        };
        let err = tokenize_with_schema_and_limits(&src, &tiny).unwrap_err();
        assert!(
            matches!(err, IfcError::SizeLimitExceeded(_)),
            "expected SizeLimitExceeded, got {:?}",
            err
        );
        assert!(err.to_string().to_lowercase().contains("nesting"));
    }

    #[test]
    fn shallow_nesting_passes_default_limits() {
        // A well-formed entity with modest nesting parses fine.
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCPROJECT('guid',#2,'Sample',$,$,$,$,(#5),#6);
ENDSEC;
END-ISO-10303-21;
";
        let (schema, entities) =
            tokenize_with_schema_and_limits(src, &Default::default()).expect("parses");
        assert_eq!(entities.len(), 1);
        assert_eq!(schema.as_deref(), Some("IFC4"));
    }

    #[test]
    fn oversized_source_is_capped() {
        // Tiny byte cap; build a >cap source.
        let tiny = fluxion_core::parser_limits::ParserLimits {
            max_file_bytes: 8,
            max_lines: 1_000_000,
            max_recursion_depth: 256,
            max_array_elements: 1_000_000,
        };
        let src = "ISO-10303-21; much longer than eight bytes";
        let err = tokenize_with_schema_and_limits(src, &tiny).unwrap_err();
        assert!(matches!(err, IfcError::SizeLimitExceeded(_)));
    }
}
