//! Contract pin for `fluxion-evaluator` dynamic-loading security
//! acceptance criteria (Issue #3554).
//!
//! The CI drift gate
//! (`scripts/check_evaluator_dynamic_secure.py`) is the authoritative
//! enforcement path; this Rust integration test is a redundant,
//! machine-verifiable pin that runs in the cargo test matrix. A future
//! PR that regresses the SECURITY_ACCEPTANCE.md or the dynamic.rs stub
//! contract will fail both checks — the Python gate in CI and the
//! Rust test in `cargo test -p fluxion-evaluator`.
//!
//! Scope guard: this test does NOT exercise the loader (that is the
//! follow-up PR's job). It only reads the source files and asserts
//! the static contract described in SECURITY_ACCEPTANCE.md.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn crate_dir() -> PathBuf {
    workspace_root()
}

fn dynamic_rs_path() -> PathBuf {
    crate_dir().join("src").join("dynamic.rs")
}

fn sandbox_rs_path() -> PathBuf {
    crate_dir().join("src").join("sandbox.rs")
}

fn acceptance_md_path() -> PathBuf {
    crate_dir().join("SECURITY_ACCEPTANCE.md")
}

fn cargo_toml_path() -> PathBuf {
    crate_dir().join("Cargo.toml")
}

fn read_required(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "required file missing or unreadable: {} ({})",
            path.display(),
            e
        )
    })
}

/// (1) SECURITY_ACCEPTANCE.md exists and documents every criterion (a)
/// through (d) with a `### (x)` heading.
#[test]
fn security_acceptance_md_documents_all_criteria() {
    let text = read_required(&acceptance_md_path());
    for letter in ["a", "b", "c", "d"] {
        let pattern = format!("### ({}) ", letter);
        assert!(
            text.contains(&pattern),
            "SECURITY_ACCEPTANCE.md must contain a `### ({})` heading for criterion ({}); \
             this is the canonical acceptance contract for the follow-up `libloading` PR (Issue #3554).",
            letter,
            letter,
        );
    }
}

/// (2) The dynamic-load sandbox-side anchoring comment block in
/// sandbox.rs lists every criterion (a)-(d) as a SECURITY-ACCEPTANCE
/// TODO. The follow-up PR MUST convert each TODO into a
/// `// SECURITY-ACCEPTANCE: satisfied` line in dynamic.rs.
#[test]
fn sandbox_rs_anchors_each_acceptance_criterion() {
    let text = read_required(&sandbox_rs_path());
    for letter in ["a", "b", "c", "d"] {
        let token = format!("SECURITY-ACCEPTANCE: TODO ({})", letter);
        assert!(
            text.contains(&token),
            "sandbox.rs must carry a `{}` anchor so the follow-up PR has a \
             single-file checklist per criterion (Issue #3554).",
            token,
        );
    }
}

/// (3) dynamic.rs is currently in stub state — every public return
/// path returns `DynamicLoadError::NotImplementedInThisBuild` (or, for
/// the feature-off branch, `FeatureNotEnabled`). The follow-up PR is
/// allowed to swap the stub for a real loader, but ONLY if it adds
/// all four `// SECURITY-ACCEPTANCE: satisfied` markers.
#[test]
fn dynamic_rs_stub_contract_holds() {
    let text = read_required(&dynamic_rs_path());

    // Strip line + block comments so a doc comment that references
    // `libloading::Library` (the follow-up crate name) doesn't
    // masquerade as a real implementation site.
    let code_only = strip_rust_comments(&text);

    let has_implemented = code_only.contains("Ok(DynamicKernel")
        || code_only.contains("Ok(DynamicKernel::")
        || code_only.contains("Library::new")
        || code_only.contains("libloading::");
    if has_implemented {
        // Implemented branch: must have all four markers.
        for letter in ["a", "b", "c", "d"] {
            let marker = format!("SECURITY-ACCEPTANCE: satisfied — ({})", letter);
            assert!(
                text.contains(&marker),
                "dynamic.rs appears to implement the loader but is missing the \
                 `{}` marker. The drift gate will fail without it (Issue #3554).",
                marker,
            );
        }
    } else {
        // Stub branch: at least one canonical stub variant must appear.
        assert!(
            code_only.contains("DynamicLoadError::NotImplementedInThisBuild")
                || code_only.contains("DynamicLoadError::FeatureNotEnabled"),
            "dynamic.rs is neither a stub nor a marked-implemented loader; \
             contract violated (Issue #3554)."
        );
    }
}

/// Remove `//` line comments and `/* ... */` block comments from a
/// Rust source string. String literals are NOT scrubbed (good enough
/// for this gate; the stub has no string literals that contain the
/// tokens we care about).
fn strip_rust_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Block comment.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            out.push(' ');
            continue;
        }
        // Line comment.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Doc comment (`///` / `//!`) — also a line comment.
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// (4) The Cargo.toml `[features] dynamic = []` stub MUST stay empty
/// of `libloading` (and other banned FFI crates) until issue #3310's
/// duplicate-version budget is raised.
#[test]
fn cargo_toml_does_not_silently_pull_libloading() {
    let text = read_required(&cargo_toml_path());
    for dep in ["libloading", "dlopen-ffi", "sharedlib"] {
        let banned_line = format!("{} =", dep);
        for raw in text.lines() {
            let trimmed = raw.trim_start();
            if trimmed.starts_with('#') {
                continue;
            }
            assert!(
                !raw.contains(&banned_line),
                "Cargo.toml must not silently pull `{}` into fluxion-evaluator; \
                 that gate is the whole point of Issue #3554 + #3310.",
                dep,
            );
        }
    }
}

/// (5) dynamic.rs defines every `DynamicLoadError` variant that the
/// issue's acceptance criteria require to exist as a target variant for
/// the loader's hardening paths:
/// - `NotImplementedInThisBuild` (stub)
/// - `CandidateError` (catch_unwind / signal handler sink)
/// - `FeatureNotEnabled` (default-build path)
/// - `UnsupportedAbiVersion` (signature-verification secondary check)
/// - `NotFound` (load-time path error)
#[test]
fn dynamic_load_error_variants_are_present() {
    let text = read_required(&dynamic_rs_path());
    for variant in [
        "NotImplementedInThisBuild",
        "CandidateError",
        "FeatureNotEnabled",
        "UnsupportedAbiVersion",
        "NotFound",
    ] {
        assert!(
            text.contains(variant),
            "DynamicLoadError::{} must be defined; the SECURITY_ACCEPTANCE \
             criteria reference it (Issue #3554).",
            variant,
        );
    }
}
