//! Regression test for issue #2502 — physics `eprintln!` must be gated.
//!
//! Issue #1967 introduced the `debug-physics` Cargo feature to gate unconditional
//! `eprintln!` calls in physics hot loops so that default builds emit nothing.
//! Issue #2502 found that `src/physics/state_space_ctf.rs` had 100+ ungated
//! `eprintln!` calls (across `matrix_exponential_faer`, `expm_higham_padé13`, and
//! diagnostic paths) that fired on every CTF coefficient computation in release
//! builds — a single 8760-step ASHRAE 600 run could emit thousands of discarded
//! stderr lines.
//!
//! This test enforces the gate at the source level so the drift cannot reappear.
//! It scans every `.rs` file under `src/physics/` and asserts that each
//! `eprintln!`, `eprint!`, `println!`, `print!`, and `dbg!` macro call is
//! unreachable in a default (no `debug-physics`) build — i.e. it is either inside
//! a `#[cfg(test)]` module or within a `#[cfg(feature = "debug-physics")]` region
//! (block, statement, or direct attribute).
//!
//! The scanner is a small scope tracker: it strips Rust strings/chars/comments
//! (so format-string `{}` placeholders and `;`/`{}` inside string literals do not
//! corrupt brace/semicolon counting), then walks the source tracking a scope
//! stack. A `#[cfg(feature = "debug-physics")]` attribute marks the next block
//! `{`, statement (`;`-terminated), or macro call as debug-gated; a
//! `#[cfg(test)]` attribute marks the following `mod` block as test-only.
//!
//! Reference: GitHub issues #1967 and #2502.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn physics_print_macros_are_gated() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let physics_dir = manifest_dir.join("src").join("physics");

    let mut offenders: Vec<String> = Vec::new();
    let mut files_scanned = 0usize;
    let mut macros_checked = 0usize;
    scan_dir(&physics_dir, &mut offenders, &mut files_scanned, &mut macros_checked);

    assert!(
        files_scanned > 0,
        "expected to scan at least one .rs file under src/physics/"
    );

    if !offenders.is_empty() {
        panic!(
            "Found {} ungated print macro(s) (eprintln!/println!/dbg!) in src/physics/ \
             that are reachable without the `debug-physics` feature (issues #1967, #2502). \
             Each must be inside a #[cfg(test)] module or a \
             #[cfg(feature = \"debug-physics\")] region:\n  - {}\n\
             (scanned {} files, {} macro call sites)",
            offenders.len(),
            offenders.join("\n  - "),
            files_scanned,
            macros_checked
        );
    }
    // Sanity: the CTF state-space file must be scanned and its (gated) macros counted.
    assert!(
        macros_checked >= 28,
        "expected to find at least the 28 library-path print macros that were gated for #2502, \
         but only checked {} total. The scanner may be broken.",
        macros_checked
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Normal,
    Debug,
    Test,
}

fn scan_dir(
    dir: &Path,
    offenders: &mut Vec<String>,
    files: &mut usize,
    macros: &mut usize,
) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, offenders, files, macros);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            if let Ok(content) = fs::read_to_string(&path) {
                *files += 1;
                scan_file(&path, &content, offenders, macros);
            }
        }
    }
}

fn scan_file(path: &Path, content: &str, offenders: &mut Vec<String>, macros: &mut usize) {
    let stripped = strip_strings_and_comments(content);

    let mut scopes: Vec<Kind> = Vec::new();
    let mut pending_debug = false;
    let mut pending_test = false;

    for (idx, line) in stripped.lines().enumerate() {
        // Attribute detection from the ORIGINAL (unstripped) line so the
        // `"debug-physics"` string literal inside #[cfg(...)] is preserved.
        if let Some(orig) = content.lines().nth(idx) {
            if orig.contains("debug-physics") {
                pending_debug = true;
            }
            if orig.contains("cfg(test)") {
                pending_test = true;
            }
        }

        let bytes = line.as_bytes();
        let mut j = 0;
        while j < bytes.len() {
            // Macro call detection (strings are stripped, so only real calls remain).
            // Operate on raw bytes to avoid UTF-8 char-boundary panics from
            // non-ASCII codepoints (e.g. `≠`, `Φ`) elsewhere on the line.
            if j == 0 || !is_ident_byte(bytes[j - 1]) {
                if let Some((consumed, name)) = match_macro(&bytes[j..]) {
                    *macros += 1;
                    let gated = pending_debug
                        || scopes.iter().any(|s| *s == Kind::Debug || *s == Kind::Test);
                    if !gated {
                        offenders.push(format!(
                            "{}:{} ungated `{}`",
                            path.display(),
                            idx + 1,
                            name
                        ));
                    }
                    pending_debug = false;
                    pending_test = false;
                    j += consumed;
                    continue;
                }
            }

            let c = bytes[j];
            match c {
                b'{' => {
                    let kind = if pending_debug {
                        Kind::Debug
                    } else if pending_test {
                        Kind::Test
                    } else {
                        Kind::Normal
                    };
                    scopes.push(kind);
                    pending_debug = false;
                    pending_test = false;
                }
                b'}' => {
                    scopes.pop();
                }
                b';' => {
                    // A statement terminator consumes any pending attribute.
                    pending_debug = false;
                    pending_test = false;
                }
                _ => {}
            }
            j += 1;
        }
    }
}

/// Returns `(byte_length_including_bang, display_name)` if `s` begins with one of
/// the target print/debug macros immediately followed by `!`.
fn match_macro(s: &[u8]) -> Option<(usize, &'static str)> {
    for (kw, name) in [
        ("eprintln!", "eprintln!"),
        ("eprint!", "eprint!"),
        ("println!", "println!"),
        ("print!", "print!"),
        ("dbg!", "dbg!"),
    ] {
        if s.starts_with(kw.as_bytes()) {
            return Some((kw.len(), name));
        }
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

/// Replace the contents of Rust string literals, char literals, line comments,
/// and block comments with spaces (preserving newlines and total length) so that
/// `{`, `}`, `;`, and identifier tokens inside strings/comments do not corrupt
/// the brace/scope tracking. Positions and line numbers are preserved exactly.
fn strip_strings_and_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut i = 0;

    enum St {
        Normal,
        Line,
        Block,
        Str,
        Char,
    }
    let mut st = St::Normal;

    while i < n {
        let c = bytes[i];
        match st {
            St::Normal => match c {
                b'/' if i + 1 < n && bytes[i + 1] == b'/' => {
                    st = St::Line;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                }
                b'/' if i + 1 < n && bytes[i + 1] == b'*' => {
                    st = St::Block;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                }
                b'"' => {
                    st = St::Str;
                    out.push(b' ');
                    i += 1;
                }
                b'\'' => {
                    if is_char_literal(bytes, i, n) {
                        st = St::Char;
                        out.push(b' ');
                        i += 1;
                    } else {
                        // Lifetime label — keep as-is (contains no braces/semis).
                        out.push(c);
                        i += 1;
                    }
                }
                _ => {
                    out.push(c);
                    i += 1;
                }
            },
            St::Line => {
                if c == b'\n' {
                    out.push(b'\n');
                    st = St::Normal;
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            St::Block => {
                if c == b'*' && i + 1 < n && bytes[i + 1] == b'/' {
                    out.push(b' ');
                    out.push(b' ');
                    st = St::Normal;
                    i += 2;
                } else {
                    out.push(if c == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
            St::Str => {
                if c == b'\\' && i + 1 < n {
                    // Escape sequence — blank the backslash and the escaped byte,
                    // but preserve a line feed so that string line-continuations
                    // (`\` + LF) do not drop newlines and desynchronize the
                    // stripped text's line numbers from the original.
                    out.push(b' ');
                    out.push(if bytes[i + 1] == b'\n' { b'\n' } else { b' ' });
                    i += 2;
                } else if c == b'"' {
                    out.push(b' ');
                    st = St::Normal;
                    i += 1;
                } else {
                    out.push(if c == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
            St::Char => {
                if c == b'\\' && i + 1 < n {
                    out.push(b' ');
                    out.push(if bytes[i + 1] == b'\n' { b'\n' } else { b' ' });
                    i += 2;
                } else if c == b'\'' {
                    out.push(b' ');
                    st = St::Normal;
                    i += 1;
                } else {
                    out.push(if c == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
        }
    }

    String::from_utf8(out).unwrap_or_default()
}

/// Heuristic: is the `'` at `bytes[i]` the start of a char literal (vs a lifetime)?
/// - `'\` … → escaped char literal
/// - `'x'` → two-byte char literal
fn is_char_literal(bytes: &[u8], i: usize, n: usize) -> bool {
    if i + 1 >= n {
        return false;
    }
    if bytes[i + 1] == b'\\' {
        return true;
    }
    i + 2 < n && bytes[i + 1] != b'\\' && bytes[i + 2] == b'\''
}
