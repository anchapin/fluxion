// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Exporter write-path confinement (Issue #3728).
//!
//! Every exporter on the FFI surfaces (`src/napi/{osm,gbxml,fmi}_exporter.rs`
//! and `src/python/{bindings,osm_bindings}.rs`) accepted a raw, unvalidated
//! write path. Without confinement, an untrusted embedding context (Tauri
//! IPC, a Python service, a Node server) could overwrite the ONNX model
//! itself, breaking the sha256 signature gate's assumptions — or any
//! other file the process can reach. The read side already had a
//! confinement pattern (`validate_model_path` #2529; `validate_epw_path`
//! #2915) using canonicalize + `starts_with` + symlink refusal + generic
//! errors. This module ports that same pattern to the *write* side.
//!
//! # Policy
//!
//! - The destination must live inside the operator-configured
//!   `FLUXION_EXPORT_DIR` allow-list (default: `exports/`, relative to
//!   the process working directory).
//! - The destination's parent directory must exist (no creating
//!   attacker-controlled directories).
//! - The destination's parent must NOT be a symlink (mirrors the read
//!   side: a symlinked parent could be swapped to escape the allow-list
//!   between the check and the write).
//! - The destination's extension must match the per-exporter pin
//!   (`.osm`, `.xml` for gbXML, `.fmu`). Case-insensitive.
//! - Errors are deliberately generic and omit the raw user-supplied path
//!   so attacker-controlled input is never reflected back to the caller
//!   (closes the error oracle — same posture as `validate_model_path`).
//!
//! # Opt-out
//!
//! Set `FLUXION_EXPORT_ALLOW_UNRESTRICTED=1` to bypass the confinement
//! (returns the absolute path unchanged). Mirrors the
//! `FLUXION_REST_ALLOW_INSECURE=1` posture for the REST surface so the
//! escape hatch is documented in `docs/FEATURES.md` and discoverable.
//! Production deploys MUST leave this unset; it exists for local dev and
//! for callers that need to target an external mount the allow-list
//! cannot express.

use std::path::{Path, PathBuf};

/// Environment variable name for the exporter allow-list directory.
/// Matches the `FLUXION_MODEL_DIR` / `FLUXION_EPW_DIR` pattern used by
/// the read-side validators (#2529, #2915).
pub const FLUXION_EXPORT_DIR_ENV: &str = "FLUXION_EXPORT_DIR";

/// Default exporter allow-list directory when `FLUXION_EXPORT_DIR` is
/// unset. Relative to the process working directory. Mirrors the
/// `DEFAULT_MODEL_DIR = "models/"` default used by
/// `validate_model_path` (`src/ai/surrogate/integrity.rs`).
pub const DEFAULT_EXPORT_DIR: &str = "exports/";

/// Environment variable name for the explicit opt-out. Mirrors the
/// `FLUXION_REST_ALLOW_INSECURE=1` posture for the REST surface — when
/// set, [`validate_export_path`] / [`validate_export_path_in_dir`]
/// return the absolute path unchanged (still pinned to the requested
/// extension so a caller cannot turn the bypass into a total bypass).
pub const FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV: &str = "FLUXION_EXPORT_ALLOW_UNRESTRICTED";

/// Validate `p` as a write destination for an exporter on the FFI
/// surfaces (Issue #3728).
///
/// Reads the allow-list directory from [`FLUXION_EXPORT_DIR_ENV`]
/// (default: [`DEFAULT_EXPORT_DIR`]). When
/// [`FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV`] is set to a truthy value
/// (`1|true|yes|on`, case-insensitive — same vocabulary as the
/// `FLUXION_REST_ALLOW_INSECURE` escape hatch documented in
/// `docs/FEATURES.md`), the allow-list *containment* check is skipped,
/// but the extension pin and the parent-existence / parent-symlink
/// gates still fire. The bypass therefore lets the caller target an
/// external mount the allow-list cannot express (e.g. an S3 FUSE mount,
/// a CI artifact dir outside the workspace) without widening the
/// surface to "write to any extension".
///
/// `required_ext` is the per-exporter extension pin (`.osm` / `.xml` /
/// `.fmu`). Case-insensitive match. The destination must end with this
/// extension (with a dot, case-insensitive) — pinning stops the FFI
/// surface from being used as a "write to any extension" primitive
/// just because the allow-list was dropped.
///
/// On success returns the canonicalised absolute path. All error
/// messages are deliberately generic and omit the raw user-supplied
/// path so attacker-controlled input is never reflected back to the
/// caller (closes the error oracle).
pub fn validate_export_path(p: &str, required_ext: &str) -> Result<PathBuf, String> {
    let dir =
        std::env::var(FLUXION_EXPORT_DIR_ENV).unwrap_or_else(|_| DEFAULT_EXPORT_DIR.to_string());
    let allow_unrestricted =
        crate::util::env_bool::env_bool(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, false);
    validate_export_path_in_dir(p, Path::new(&dir), required_ext, allow_unrestricted)
}

/// Parameterised core of [`validate_export_path`]. Accepts an explicit
/// allow-list directory so it can be unit-tested without racing on the
/// process-wide [`FLUXION_EXPORT_DIR_ENV`] env var. `allow_unrestricted`
/// mirrors [`FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV`] — when `true`, the
/// containment check is skipped (the other gates still fire).
///
/// Checks, in order:
/// 1. `required_ext` is non-empty (programmer error guard).
/// 2. The destination's extension matches `required_ext`
///    (case-insensitive, dot-pinned).
/// 3. The destination's parent directory exists.
/// 4. The destination's parent is NOT a symlink (mirrors the read side's
///    symlink policy from `validate_model_path_in_dir` #3651 — a
///    symlinked parent could be swapped to escape the allow-list
///    between the check and the write).
/// 5. **Unless `allow_unrestricted` is set**, the canonicalised parent
///    is inside the canonicalised `allowed_dir` (component-wise
///    `starts_with` — blocks `..` traversal and symlinks that escape
///    the allow-list).
///
/// On success returns the canonicalised absolute destination path. The
/// destination itself need not exist (we are *writing* it) so no
/// `is_file()` / size check applies — that is the asymmetric counterpart
/// of `validate_model_path_in_dir`'s read-side contract.
pub fn validate_export_path_in_dir(
    p: &str,
    allowed_dir: &Path,
    required_ext: &str,
    allow_unrestricted: bool,
) -> Result<PathBuf, String> {
    if required_ext.is_empty() {
        return Err("required extension pin is empty".to_string());
    }

    // Extension pin. Strip a leading dot so callers can pass either
    // `".osm"` or `"osm"`. Match is ASCII case-insensitive.
    let required_lower = required_ext.trim_start_matches('.').to_ascii_lowercase();
    let raw = Path::new(p);
    let actual_ext = raw
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if actual_ext.as_deref() != Some(required_lower.as_str()) {
        return Err(format!(
            "invalid exporter file extension (expected .{required_lower})"
        ));
    }

    // The destination must have a resolvable parent — a bare filename
    // like `"output.osm"` would otherwise pin to the process CWD, which
    // is operator-dependent and not under the allow-list's control
    // (the allow-list is itself relative to CWD, so a basename-only path
    // could land anywhere). Reject so the caller passes an explicit
    // allow-list-relative path.
    let parent = raw
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| {
            "exporter path must include a parent directory inside the allow-list".to_string()
        })?;

    // Parent must exist (no creating attacker-controlled directories).
    // We use `symlink_metadata` so a symlinked parent is observed as a
    // symlink (and rejected below) instead of being followed. The
    // symlink check is FIRST because `FileType` for a symlink reports
    // `is_dir() == false` — checking the directory bit first would
    // surface a misleading "not a directory" error for a clearly
    // symlinked path (the symlink-to-directory case).
    let parent_meta = std::fs::symlink_metadata(parent)
        .map_err(|_| "exporter parent directory not found".to_string())?;
    if parent_meta.file_type().is_symlink() {
        return Err("exporter parent path may not be a symbolic link".to_string());
    }
    if !parent_meta.file_type().is_dir() {
        return Err("exporter parent path is not a directory".to_string());
    }

    // Canonicalise the (existing) parent. canonicalize() would fail on
    // the destination itself (it does not exist yet — we are *writing*
    // it), so we canonicalise only the parent and append the
    // destination's filename. This is equivalent to canonicalising the
    // full path once it exists because the parent is the only resolvable
    // component on the write side.
    let canonical_parent = std::fs::canonicalize(parent)
        .map_err(|_| "failed to canonicalize exporter parent path".to_string())?;

    // Containment check — skipped only when the operator has explicitly
    // opted out via `FLUXION_EXPORT_ALLOW_UNRESTRICTED=1`. The opt-out
    // is fail-closed on typos (`env_bool` returns `false` for unknown
    // tokens, mirroring the `FLUXION_REST_ALLOW_INSECURE` posture).
    if !allow_unrestricted {
        let canonical_dir = std::fs::canonicalize(allowed_dir)
            .map_err(|_| "allowed export directory not found".to_string())?;
        if !canonical_parent.starts_with(&canonical_dir) {
            return Err("exporter path outside allowed directory".to_string());
        }
    }

    let file_name = raw
        .file_name()
        .ok_or_else(|| "exporter path must include a file name".to_string())?;
    Ok(canonical_parent.join(file_name))
}

#[cfg(test)]
mod tests {
    //! Issue #3728 unit tests for the exporter write-path confinement.
    //!
    //! Mirrors the inline-tests layout of `validate_model_path_in_dir`
    //! (`src/ai/surrogate/integrity.rs`) — every test constructs a
    //! `tempfile::tempdir()` to act as the allow-list so parallel
    //! `cargo test` threads do not race on a process-wide env var, and
    //! restores the env on scope exit through `EnvGuard`.

    use super::*;
    use std::sync::{Mutex, MutexGuard};

    // ---- Env isolation ----

    const ENV_KEYS: &[&str] = &[
        FLUXION_EXPORT_DIR_ENV,
        FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV,
    ];

    /// Serialises every env-mutating test so parallel `cargo test`
    /// threads cannot clobber each other's variables. `EnvGuard` holds
    /// this lock from construction until drop (when the saved
    /// environment is restored).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
        _lock: MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn new() -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let saved = ENV_KEYS
                .iter()
                .map(|k| (*k, std::env::var(k).ok()))
                .collect();
            for k in ENV_KEYS {
                std::env::remove_var(k);
            }
            Self { saved, _lock: lock }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    // ---- required_ext guard ----

    #[test]
    fn rejects_empty_required_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = validate_export_path_in_dir(
            dir.path().join("out").to_str().unwrap(),
            dir.path(),
            "",
            false,
        )
        .unwrap_err();
        assert!(
            err.contains("required extension pin is empty"),
            "got: {err}"
        );
    }

    // ---- Extension pin ----

    #[test]
    fn rejects_wrong_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("not_osm.txt");
        let err = validate_export_path_in_dir(target.to_str().unwrap(), dir.path(), "osm", false)
            .unwrap_err();
        assert!(
            err.contains("invalid exporter file extension"),
            "got: {err}"
        );
        assert!(err.contains(".osm"), "got: {err}");
    }

    #[test]
    fn accepts_uppercase_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("OUT.OSM");
        let res = validate_export_path_in_dir(target.to_str().unwrap(), dir.path(), "osm", false);
        assert!(
            res.is_ok(),
            "uppercase extension should be accepted: {:?}",
            res
        );
    }

    #[test]
    fn accepts_extension_with_or_without_leading_dot() {
        let dir = tempfile::tempdir().expect("tempdir");
        // ".osm" and "osm" both work — the validator strips a leading dot.
        let a = validate_export_path_in_dir(
            dir.path().join("a.osm").to_str().unwrap(),
            dir.path(),
            ".osm",
            false,
        );
        let b = validate_export_path_in_dir(
            dir.path().join("b.osm").to_str().unwrap(),
            dir.path(),
            "osm",
            false,
        );
        assert!(a.is_ok() && b.is_ok(), "a={:?} b={:?}", a, b);
    }

    // ---- Parent existence ----

    #[test]
    fn rejects_missing_parent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("nope_dir").join("a.osm");
        let err = validate_export_path_in_dir(target.to_str().unwrap(), dir.path(), "osm", false)
            .unwrap_err();
        assert!(err.contains("parent"), "got: {err}");
    }

    #[test]
    fn rejects_bare_basename_no_parent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = validate_export_path_in_dir("a.osm", dir.path(), "osm", false).unwrap_err();
        assert!(
            err.contains("parent directory"),
            "basename-only path must be rejected, got: {err}"
        );
    }

    // ---- Symlink policy (Issue #3728 sibling of #3651) ----

    #[test]
    fn rejects_symlinked_parent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("real_dir");
        std::fs::create_dir(&real).expect("mkdir real");
        let link = dir.path().join("link_dir");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let target = link.join("a.osm");
        let err = validate_export_path_in_dir(target.to_str().unwrap(), dir.path(), "osm", false)
            .unwrap_err();
        assert!(err.contains("symbolic link"), "got: {err}");
    }

    // ---- Containment ----

    #[test]
    fn rejects_path_outside_allowlist_etc_passwd() {
        // The headline test: `/etc/passwd` (or anything outside the
        // allow-list) must NEVER be writable from the FFI surface.
        let dir = tempfile::tempdir().expect("tempdir");
        let err = validate_export_path_in_dir("/etc/passwd", dir.path(), "osm", false).unwrap_err();
        // `/etc/passwd` has no extension → extension pin fires first.
        assert!(
            err.contains("invalid exporter file extension"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_path_outside_allowlist_with_correct_extension() {
        // `/tmp/foo.osm` is real but OUTSIDE the allow-list → containment
        // fires. We must NOT rely on canonicalize to also catch this —
        // a tempdir that *is* `/tmp` could pass `starts_with` checks on
        // some filesystems, so the containment check is the gate.
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = std::env::temp_dir().join("definitely_not_in_allow_list_3728.osm");
        // Make sure parent exists for the symlink/exists branch.
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"x").unwrap();
        let err = validate_export_path_in_dir(outside.to_str().unwrap(), dir.path(), "osm", false)
            .unwrap_err();
        assert!(err.contains("outside allowed directory"), "got: {err}");
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn rejects_traversal_outside_allowlist() {
        let dir = tempfile::tempdir().expect("tempdir");
        let traversal = dir.path().join("..").join("escaped.osm");
        let err =
            validate_export_path_in_dir(traversal.to_str().unwrap(), dir.path(), "osm", false)
                .unwrap_err();
        // The `..` parent collapses to the parent of the tempdir (likely
        // `/tmp`), which is outside the allow-list → containment fires.
        // OR the parent of `..` does not exist on the tempfs and the
        // missing-parent branch fires first. Both are acceptable
        // rejections; we just need the validator to refuse.
        assert!(
            err.contains("outside allowed directory")
                || err.contains("parent directory not found")
                || err.contains("exporter parent"),
            "got: {err}"
        );
    }

    #[test]
    fn allows_dotdot_inside_allowlist() {
        // `dir/sub/../x.osm` resolves to `dir/x.osm` — inside the
        // allow-list. The validator canonicalises the parent before the
        // containment check, so this should succeed.
        let dir = tempfile::tempdir().expect("tempdir");
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).expect("mkdir sub");
        let target = sub.join("..").join("x.osm");
        let res = validate_export_path_in_dir(target.to_str().unwrap(), dir.path(), "osm", false);
        assert!(
            res.is_ok(),
            "dotdot collapsing inside allow-list should pass: {:?}",
            res
        );
    }

    // ---- Happy path ----

    #[test]
    fn accepts_valid_path_inside_allowlist() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("sub").join("out.osm");
        std::fs::create_dir(target.parent().unwrap()).expect("mkdir");
        let res = validate_export_path_in_dir(target.to_str().unwrap(), dir.path(), "osm", false);
        assert!(
            res.is_ok(),
            "valid in-allow-list path should pass: {:?}",
            res
        );
        let canonical = res.unwrap();
        assert!(
            canonical.is_absolute(),
            "result must be absolute, got: {}",
            canonical.display()
        );
        assert!(canonical.starts_with(dir.path().canonicalize().unwrap()));
    }

    // ---- Env-driven entry point ----

    #[test]
    fn validate_export_path_reads_fluxion_export_dir_env() {
        let _env = EnvGuard::new();
        let dir = tempfile::tempdir().expect("tempdir");
        std::env::set_var(FLUXION_EXPORT_DIR_ENV, dir.path().to_str().unwrap());
        let target = dir.path().join("out.osm");
        let res = validate_export_path(target.to_str().unwrap(), "osm");
        assert!(
            res.is_ok(),
            "env-driven allow-list should accept in-list path: {:?}",
            res
        );
    }

    #[test]
    fn validate_export_path_rejects_when_env_misses() {
        let _env = EnvGuard::new();
        std::env::set_var(FLUXION_EXPORT_DIR_ENV, "/nonexistent/allow/list/3728");
        let err = validate_export_path("/tmp/x.osm", "osm").unwrap_err();
        // Either the allow-list doesn't exist (canonicalize fails) OR
        // the temp path is outside — both are rejections, neither leaks
        // the user-supplied path.
        assert!(
            !err.contains("/tmp/x.osm"),
            "error must not leak user path: {err}"
        );
    }

    // ---- Opt-out ----

    #[test]
    fn unrestricted_bypass_still_pins_extension() {
        let _env = EnvGuard::new();
        std::env::set_var(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, "1");
        // Wrong extension → still rejected even under bypass.
        let err = validate_export_path("/tmp/x.txt", "osm").unwrap_err();
        assert!(
            err.contains("invalid exporter file extension"),
            "got: {err}"
        );
    }

    #[test]
    fn unrestricted_bypass_accepts_path_outside_allowlist() {
        // Issue #3728 acceptance: the opt-out exists *specifically* so a
        // caller can target an external mount the allow-list cannot
        // express. `/tmp/...` is outside any sane allow-list, but the
        // bypass should let it through (extension pin still fires, of
        // course).
        let _env = EnvGuard::new();
        std::env::set_var(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, "1");
        let target = std::env::temp_dir().join("fluxion_3728_bypass_target.osm");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        let res = validate_export_path(target.to_str().unwrap(), "osm");
        assert!(
            res.is_ok(),
            "bypass should let /tmp/*.osm through: {:?}",
            res
        );
    }

    #[test]
    fn unrestricted_bypass_still_rejects_symlinked_parent() {
        // Bypass only drops the *containment* gate. The parent-symlink
        // gate (sibling of Issue #3651) must still fire.
        let _env = EnvGuard::new();
        std::env::set_var(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, "1");
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("real_dir");
        std::fs::create_dir(&real).expect("mkdir real");
        let link = dir.path().join("link_dir");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let target = link.join("a.osm");
        let err = validate_export_path(target.to_str().unwrap(), "osm").unwrap_err();
        assert!(err.contains("symbolic link"), "got: {err}");
    }

    #[test]
    fn unrestricted_bypass_accepts_any_in_allowlist_path() {
        let _env = EnvGuard::new();
        std::env::set_var(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, "1");
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("a.osm");
        let res = validate_export_path(target.to_str().unwrap(), "osm");
        assert!(
            res.is_ok(),
            "bypass should accept in-allow-list path: {:?}",
            res
        );
    }

    #[test]
    fn unrestricted_bypass_still_canonicalises_absolute_path() {
        let _env = EnvGuard::new();
        std::env::set_var(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, "1");
        // Path that doesn't exist yet — the canonical path should
        // canonicalise the parent (which exists) and append the
        // filename, returning an absolute path.
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("new_dir").join("out.osm");
        std::fs::create_dir(target.parent().unwrap()).unwrap();
        let res = validate_export_path(target.to_str().unwrap(), "osm");
        assert!(res.is_ok(), "got: {:?}", res);
        assert!(res.unwrap().is_absolute());
    }

    #[test]
    fn unrestricted_bypass_defaults_closed_on_typo() {
        // `env_bool` rejects unknown tokens with a warn-and-false.
        // Mirrors the `FLUXION_REST_ALLOW_INSECURE` posture so the
        // bypass cannot be activated by a typo like
        // `FLUXION_EXPORT_ALLOW_UNRESTRICTED=yez`. We point the
        // allow-list at a real tempdir so the only failure path that
        // can surface is the containment verdict itself, not the
        // allow-list-doesn't-exist check.
        let _env = EnvGuard::new();
        let dir = tempfile::tempdir().expect("tempdir");
        std::env::set_var(FLUXION_EXPORT_DIR_ENV, dir.path().to_str().unwrap());
        std::env::set_var(FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV, "yez");
        let outside = std::env::temp_dir().join("fluxion_3728_typo_target.osm");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        let err = validate_export_path(outside.to_str().unwrap(), "osm").unwrap_err();
        assert!(
            err.contains("outside allowed directory"),
            "typo must NOT activate the bypass, got: {err}"
        );
    }
}
