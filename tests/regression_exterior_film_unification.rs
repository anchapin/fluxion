//! Regression test guard for the v2023 exterior-film-coefficient unification
//! (Issue #1419 / #1504).
//!
//! Prior to #1419, ASHRAE 140 surface heat-transfer paths referenced the legacy
//! 6.7 m/s design-wind value `29.3 W/m²K` either directly or via the reciprocal
//! `1.0 / 29.3`. PR #1419 unified the production default to the v2023 constant
//! `EXTERIOR_FILM_COEFF = 18.3 W/m²K`, but PR #1420/#1490 re-introduced the
//! legacy `1.0 / 29.3` literal inside an ASHRAE 140 Case 900 test assertion.
//! That single line failed silently in `cargo test --lib` locally and produced
//! a cascade of identical CI failures across 8+ concurrent PRs rebased onto
//! the post-#1419 main.
//!
//! This test enforces the unification at the source level so the drift cannot
//! re-appear:
//!
//! 1. `EXTERIOR_FILM_COEFF == 18.3` (compile-time public-API pin).
//! 2. No `.rs` file under `src/` contains the bare-arithmetic literal
//!    `1.0 / 29.3` (or whitespace-equivalent forms). The constant definition
//!    `pub const ASHRAE140_H_EXT: f64 = 29.3;` in
//!    `src/physics/constants/thermal/ashrae_140/materials.rs` is the **only**
//!    sanctioned location for the legacy 6.7 m/s value, and even there it is
//!    the named constant, not a bare arithmetic expression. The reciprocal
//!    `1.0 / 29.3` is always an error — derive `1.0 / ASHRAE140_H_EXT` or use
//!    `EXTERIOR_FILM_COEFF` (18.3) instead.
//!
//! Reference: GitHub issue #1504.

use fluxion::physics::constants::EXTERIOR_FILM_COEFF;

/// Pin the canonical exterior-film coefficient to the v2023 ASHRAE 140 value.
///
/// This is the single source of truth for `h_exterior` in production physics.
/// Any change to `EXTERIOR_FILM_COEFF` requires re-validating ASHRAE 140
/// Case 600–950 reference envelopes and updating `ARCHITECTURE.md` §
/// "Module 3 Conduction / h_exterior canonical constant".
#[test]
fn test_exterior_film_coeff_is_18_3_w_per_m2k() {
    assert_eq!(
        EXTERIOR_FILM_COEFF, 18.3,
        "EXTERIOR_FILM_COEFF must equal 18.3 W/m²K (ASHRAE 140 v2023, ~3.4 m/s wind). \
         Changing this constant requires re-validating the ASHRAE 140 Case 600–950 \
         envelope tests and updating ARCHITECTURE.md §Module 3 Conduction. \
         See issue #1504."
    );
}

/// Reject any re-introduction of the legacy `1.0 / 29.3` literal in `src/`.
///
/// The literal `1.0 / 29.3` is the reciprocal of the legacy ASHRAE 140
/// 6.7 m/s design-wind exterior-film coefficient. After PR #1419 unified
/// `h_exterior` to the v2023 constant (`EXTERIOR_FILM_COEFF = 18.3 W/m²K`),
/// any expression of the form `1.0 / 29.3` in production code is a drift
/// signal — usually a copy-paste from older ASHRAE 140 reference material
/// that quietly re-introduces the wrong coefficient.
///
/// The legacy 6.7 m/s `h_ext = 29.3` constant itself is intentionally
/// preserved at `src/physics/constants/thermal/ashrae_140/materials.rs`
/// as the named constant `ASHRAE140_H_EXT` for backward-compatibility with
/// legacy ASHRAE 140 design-wind scenarios. Code that needs the legacy
/// resistance must derive it from the named constant:
/// `1.0 / ASHRAE140_H_EXT` — never the bare literal.
#[test]
fn test_no_legacy_one_over_29_3_literal_in_src() {
    let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations: Vec<String> = Vec::new();

    walk_rs_files(&src_root, &mut |path: &std::path::Path| {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };
        for (idx, line) in content.lines().enumerate() {
            if contains_legacy_one_over_29_3(line) {
                let line_no = idx + 1;
                let rel = path
                    .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                    .unwrap_or(path);
                violations.push(format!(
                    "regression (issue #1504): {}:{} contains forbidden literal '1.0 / 29.3' \
                     (legacy ASHRAE 140 6.7 m/s h_exterior reciprocal). \
                     Replace with `1.0 / EXTERIOR_FILM_COEFF` (18.3 W/m²K, v2023) or \
                     `1.0 / ASHRAE140_H_EXT` if the legacy 6.7 m/s value is explicitly required. \
                     Line: {}",
                    rel.display(),
                    line_no,
                    line.trim()
                ));
            }
        }
    });

    assert!(
        violations.is_empty(),
        "Found {} forbidden legacy literal(s) under src/:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

/// Reject any bare `29.3` literal in a *computation* context under `src/`.
///
/// Issue #2679: the FD-solver path in `src/sim/thermal_model_core.rs` called
/// `SurfaceBC::new_exterior(29.3, ...)` directly. That is not the reciprocal
/// form `1.0 / 29.3` caught by [`test_no_legacy_one_over_29_3_literal_in_src`],
/// so the existing guard passed while the violation shipped in production
/// physics code.
///
/// This test scans every `.rs` file under `src/` for a `29.3` token that is
/// **not** a comment, **not** a doc-string, and **not** the sanctioned named
/// constant definition `pub const ASHRAE140_H_EXT: f64 = 29.3;` in
/// `src/physics/constants/thermal/ashrae_140/materials.rs` (the single
/// intentional storage of the legacy 6.7 m/s value, kept for backward
/// compatibility with legacy ASHRAE 140 design-wind scenarios).
///
/// Any other bare `29.3` in source is almost certainly a re-introduction of
/// the forbidden coefficient into an active computation path. Replace it with
/// the imported canonical constant `EXTERIOR_FILM_COEFF` (18.3 W/m²K, v2023)
/// or, if the legacy 6.7 m/s value is genuinely required, reference it via the
/// named constant `ASHRAE140_H_EXT`.
///
/// Reference: GitHub issue #2679.
#[test]
fn test_no_bare_29_3_literal_in_src_computation_paths() {
    let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    // The single sanctioned location for the legacy 29.3 value as a named
    // constant. Use forward slashes so this is portable across platforms.
    let sanctioned_const_file =
        std::path::Path::new("src/physics/constants/thermal/ashrae_140/materials.rs");

    let mut violations: Vec<String> = Vec::new();

    walk_rs_files(&src_root, &mut |path: &std::path::Path| {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let rel = path
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(path);
        let is_sanctioned = rel == sanctioned_const_file;

        for (idx, raw_line) in content.lines().enumerate() {
            let line_no = idx + 1;
            // Skip comments and doc-strings entirely — we only police
            // computation paths, not documentation of the legacy value.
            let trimmed = raw_line.trim_start();
            if trimmed.starts_with("//")
                || trimmed.starts_with("///")
                || trimmed.starts_with("//!")
                || trimmed.starts_with('*')
            {
                continue;
            }

            for occurrence in find_bare_29_3_offsets(raw_line) {
                // In the sanctioned constant file, the named constant
                // definition line `pub const ASHRAE140_H_EXT: f64 = 29.3;`
                // is allowed (it is the storage of the legacy value, not a
                // computation). Any *other* bare 29.3 in that file would
                // still be flagged.
                if is_sanctioned && line_is_ashrae140_h_ext_definition(raw_line) {
                    continue;
                }

                violations.push(format!(
                    "regression (issue #2679): {}:{} contains a bare `29.3` literal \
                     outside a comment. The legacy 29.3 W/m²K (6.7 m/s wind) exterior \
                     film coefficient MUST NOT appear in any computation path \
                     (AGENTS.md §Critical Physics Constants). Replace with the imported \
                     canonical `EXTERIOR_FILM_COEFF` (18.3 W/m²K, v2023) from \
                     `src/physics/constants/thermal/ashrae_140/v2023.rs`, or — only if \
                     the legacy 6.7 m/s value is explicitly required — reference the \
                     named constant `ASHRAE140_H_EXT`. Column {}: {}",
                    rel.display(),
                    line_no,
                    occurrence + 1,
                    raw_line.trim()
                ));
            }
        }
    });

    assert!(
        violations.is_empty(),
        "Found {} forbidden bare 29.3 literal(s) in computation paths under src/:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

/// Return the byte offsets of every `29.3` token in `line` that is followed by
/// a non-identifier boundary (so `29.33` or `29.3_f64` are not matched).
fn find_bare_29_3_offsets(line: &str) -> Vec<usize> {
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let pattern = b"29.3";
    let mut i = 0;
    while i + 4 <= n {
        if &bytes[i..i + 4] == pattern {
            let after = i + 4;
            let ok_boundary = after == n
                || !(bytes[after].is_ascii_alphanumeric()
                    || bytes[after] == b'_'
                    || bytes[after] == b'.');
            // Also ensure the preceding char is not an identifier continuation
            // or a digit/`.` (so `129.3`, `x29.3`, `.29.3` are not matched).
            let prev_ok = i == 0
                || !(bytes[i - 1].is_ascii_alphanumeric()
                    || bytes[i - 1] == b'_'
                    || bytes[i - 1] == b'.');
            if ok_boundary && prev_ok {
                out.push(i);
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    out
}

/// Recognise the sanctioned named-constant definition line for the legacy
/// exterior film coefficient. Allows whitespace variations but anchors on the
/// `pub const ASHRAE140_H_EXT` identifier and a `29.3` literal on the same line.
fn line_is_ashrae140_h_ext_definition(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("pub const ASHRAE140_H_EXT") && t.contains("29.3")
}

/// Recursively walk a directory and invoke `f` on every `.rs` file.
fn walk_rs_files(dir: &std::path::Path, f: &mut dyn FnMut(&std::path::Path)) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, f);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            f(&path);
        }
    }
}

/// Detect any whitespace-equivalent form of the bare arithmetic `1.0 / 29.3`.
///
/// The leading `1.0` (or `1.`) anchors the pattern to the reciprocal form,
/// avoiding false positives on documentation text such as
/// `/// Value: 1/29.3 ≈ 0.03413 m²K/W` (no `1.0`) or standalone `29.3`
/// constants like `pub const ASHRAE140_H_EXT: f64 = 29.3;`.
///
/// Forms caught:
///   * `1.0 / 29.3`
///   * `1.0/29.3`
///   * `1. / 29.3`
///   * `1./29.3`
///
/// After matching `29.3`, the next character must not be an identifier
/// continuation (`[A-Za-z0-9_]`) so that `29.33` is not confused with `29.3`
/// followed by `3`.
fn contains_legacy_one_over_29_3(line: &str) -> bool {
    let bytes = line.as_bytes();
    let n = bytes.len();

    // Walk the line and try to match `1.0` or `1.` followed by `/ 29.3`.
    let mut i = 0;
    while i < n {
        if bytes[i] != b'1' {
            i += 1;
            continue;
        }

        // Try to match a numeric literal starting at `i`.
        let after_num = match consume_numeric_literal(bytes, i) {
            Some(end) => end,
            None => {
                i += 1;
                continue;
            }
        };

        // Skip whitespace between the number and the operator.
        let mut j = after_num;
        while j < n && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        if j >= n || bytes[j] != b'/' {
            i = after_num;
            continue;
        }
        j += 1;
        while j < n && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }

        // Expect `29.3` followed by a non-identifier character.
        if j + 4 <= n
            && &bytes[j..j + 4] == b"29.3"
            && (j + 4 == n || !(bytes[j + 4].is_ascii_alphanumeric() || bytes[j + 4] == b'_'))
        {
            return true;
        }

        i = after_num;
    }
    false
}

/// Try to consume a numeric literal starting at `start`. Returns the index
/// just past the last digit if the literal is of the form `1.0...` or
/// `1.digits` (the only forms that can participate in the legacy reciprocal
/// `1.0 / 29.3`).
fn consume_numeric_literal(bytes: &[u8], start: usize) -> Option<usize> {
    let n = bytes.len();
    if start >= n || bytes[start] != b'1' {
        return None;
    }
    let mut j = start + 1;
    if j >= n || bytes[j] != b'.' {
        return None;
    }
    j += 1;
    if j >= n || !bytes[j].is_ascii_digit() {
        return None;
    }
    while j < n && bytes[j].is_ascii_digit() {
        j += 1;
    }
    Some(j)
}

#[cfg(test)]
mod self_tests {
    use super::contains_legacy_one_over_29_3;

    #[test]
    fn detects_space_form() {
        assert!(contains_legacy_one_over_29_3("let r = 1.0 / 29.3;"));
    }

    #[test]
    fn detects_no_space_form() {
        assert!(contains_legacy_one_over_29_3("let r = 1.0/29.3;"));
    }

    #[test]
    fn detects_with_trailing_comma() {
        assert!(contains_legacy_one_over_29_3(
            "            (props.surface_resistance_outside - 1.0 / 29.3).abs() < 1e-10,"
        ));
    }

    #[test]
    fn detects_with_extra_leading_zero() {
        assert!(contains_legacy_one_over_29_3("let r = 1.00 / 29.3;"));
    }

    #[test]
    fn allows_bare_29_3_constant_definition() {
        assert!(!contains_legacy_one_over_29_3(
            "pub const ASHRAE140_H_EXT: f64 = 29.3;"
        ));
    }

    #[test]
    fn allows_named_constant_reciprocal() {
        assert!(!contains_legacy_one_over_29_3(
            "pub const ASHRAE140_R_EXT: f64 = 1.0 / ASHRAE140_H_EXT;"
        ));
    }

    #[test]
    fn allows_18_3_form() {
        assert!(!contains_legacy_one_over_29_3(
            "let r = 1.0 / EXTERIOR_FILM_COEFF;"
        ));
    }

    #[test]
    fn allows_documentation_with_one_over_29_3_text() {
        // `1/29.3` (no decimal on the 1) is a doc comment — allowed.
        assert!(!contains_legacy_one_over_29_3(
            "/// Exterior surface thermal resistance. Value: 1/29.3 ≈ 0.03413 m²K/W"
        ));
    }

    #[test]
    fn allows_legacy_comment_referencing_29_3() {
        assert!(!contains_legacy_one_over_29_3(
            "// Previous draft used legacy 29.3 which broke after the v2023 unification."
        ));
    }

    #[test]
    fn does_not_match_long_decimal_continuation() {
        // `29.33` should NOT trigger — we anchor to `29.3` followed by a
        // non-identifier boundary.
        assert!(!contains_legacy_one_over_29_3("let x = 1.0 / 29.33;"));
    }

    #[test]
    fn does_not_match_unrelated_29_3_followed_by_underscore() {
        // `29.3_X` should NOT trigger — the trailing `_` makes it an identifier.
        assert!(!contains_legacy_one_over_29_3("let x = 1.0 / 29.3_X;"));
    }
}

#[cfg(test)]
mod bare_29_3_detector_tests {
    use super::{find_bare_29_3_offsets, line_is_ashrae140_h_ext_definition};

    #[test]
    fn detects_bare_literal_in_function_call() {
        // The exact #2679 regression form.
        let line = "                let exterior_bc = SurfaceBC::new_exterior(29.3, t_ext, 0.0);";
        assert!(!find_bare_29_3_offsets(line).is_empty());
    }

    #[test]
    fn detects_bare_literal_assignment() {
        assert!(!find_bare_29_3_offsets("let h = 29.3;").is_empty());
    }

    #[test]
    fn does_not_detect_29_33() {
        assert!(find_bare_29_3_offsets("let h = 29.33;").is_empty());
    }

    #[test]
    fn does_not_detect_preceded_by_digit() {
        // `129.3` must not trigger — preceding digit is an identifier-ish boundary.
        assert!(find_bare_29_3_offsets("let h = 129.3;").is_empty());
    }

    #[test]
    fn does_not_detect_preceded_by_dot() {
        // `.29.3` is not a real token but should not match anyway.
        assert!(find_bare_29_3_offsets("let h = .29.3;").is_empty());
    }

    #[test]
    fn does_not_detect_inside_identifier_suffix() {
        assert!(find_bare_29_3_offsets("let h = foo_29_3;").is_empty());
    }

    #[test]
    fn sanctioned_const_line_is_recognised() {
        assert!(line_is_ashrae140_h_ext_definition(
            "pub const ASHRAE140_H_EXT: f64 = 29.3;"
        ));
    }

    #[test]
    fn non_sanctioned_const_line_is_not_recognised() {
        assert!(!line_is_ashrae140_h_ext_definition(
            "pub const SOMETHING_ELSE: f64 = 29.3;"
        ));
    }
}
