// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Issue #3728 — exporter write-path confinement integration tests.
//!
//! These tests exercise the public `validate_export_path` API surface
//! that every NAPI and PyO3 binding delegates to
//! (`src/napi/{osm,gbxml,fmi}_exporter.rs` and
//! `src/python/{bindings,osm_bindings}.rs`). They also smoke-test the
//! inner `validate_export_path_in_dir` helper for completeness.
//!
//! The tests run under `cargo test --test all_tests exporter_path_` and
//! must not need the `napi-bindings` / `python-bindings` cargo features
//! — the underlying validator is reachable from the root crate's lib
//! without either feature gate.

use std::sync::{Mutex, MutexGuard};

/// Serialises every env-mutating test so parallel `cargo test` threads
/// do not clobber each other's variables (Issue #3453 convention:
/// `static ENV_LOCK: Mutex<()>` around any `std::env::set_var` / `remove_var`).
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Tracks the env keys this test mutates so the guard can restore them
/// on drop (success or panic — `Drop` runs from both paths).
const ENV_KEYS: &[&str] = &[
    fluxion::api::security::FLUXION_EXPORT_DIR_ENV,
    fluxion::api::security::FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV,
];

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

#[test]
fn integration_validate_export_path_rejects_etc_passwd() {
    // Headline test for Issue #3728: `/etc/passwd` must NEVER be
    // writable from the FFI exporter surface. We assert the validator
    // refuses it (the extension pin fires first because `/etc/passwd`
    // has no `.osm` / `.xml` / `.fmu` suffix; either rejection is
    // acceptable so long as the path is refused).
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var(
        fluxion::api::security::FLUXION_EXPORT_DIR_ENV,
        dir.path().to_str().unwrap(),
    );
    let err = fluxion::api::security::validate_export_path("/etc/passwd", "osm")
        .err()
        .expect("/etc/passwd must be rejected");
    assert!(
        !err.contains("/etc/passwd"),
        "error must not leak user path: {err}"
    );
}

#[test]
fn integration_validate_export_path_rejects_traversal_outside() {
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var(
        fluxion::api::security::FLUXION_EXPORT_DIR_ENV,
        dir.path().to_str().unwrap(),
    );
    // `../escaped.osm` collapses to the parent of the tempdir, which
    // is outside the allow-list. The validator must refuse.
    let traversal = dir.path().join("..").join("escaped_3728.osm");
    let err = fluxion::api::security::validate_export_path(traversal.to_str().unwrap(), "osm")
        .err()
        .expect("traversal must be rejected");
    assert!(
        !err.contains("escaped_3728.osm"),
        "error must not leak user-supplied filename: {err}"
    );
}

#[test]
fn integration_validate_export_path_pins_extension_per_exporter() {
    // The validator must enforce the per-exporter extension pin:
    // passing `.osm` to the gbXML exporter or `.fmu` to the OSM
    // exporter must be rejected even when the path is otherwise inside
    // the allow-list. This pins the FFI surface from being used to
    // write to the wrong file type.
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var(
        fluxion::api::security::FLUXION_EXPORT_DIR_ENV,
        dir.path().to_str().unwrap(),
    );
    let target = dir.path().join("wrong_ext.osm");

    // Pass .osm file with the wrong extension pin (.fmu) — must reject.
    let err = fluxion::api::security::validate_export_path(target.to_str().unwrap(), "fmu")
        .err()
        .expect(".osm file with .fmu pin must be rejected");
    assert!(
        err.contains("invalid exporter file extension"),
        "got: {err}"
    );

    // Pass .osm file with the correct .osm pin — must accept.
    let ok = fluxion::api::security::validate_export_path(target.to_str().unwrap(), "osm")
        .expect(".osm file with .osm pin must be accepted");
    assert!(ok.is_absolute());
}

#[cfg(unix)]
#[test]
fn integration_validate_export_path_rejects_symlinked_parent() {
    // Sibling of `validate_model_path_in_dir`'s Issue #3651 policy:
    // a symlinked parent is refused even when the symlink target sits
    // inside the allow-list — a symlink could be swapped to escape the
    // allow-list between the check and the write.
    //
    // Gated to Unix-only: parent-directory symlinks require privileges
    // on Windows that are not available in the CI runner. Coverage of
    // the symlink-parent rejection path on Windows is therefore
    // intentionally skipped; the other 6 tests in this file still
    // cover the non-symlinked rejection paths on all three CI lanes
    // (see Issue #3866).
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().expect("tempdir");
    let real = dir.path().join("real");
    std::fs::create_dir(&real).expect("mkdir real");
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");
    let target = link.join("out.osm");

    let err = fluxion::api::security::validate_export_path(target.to_str().unwrap(), "osm")
        .err()
        .expect("symlinked parent must be rejected");
    assert!(err.contains("symbolic link"), "got: {err}");
}

#[test]
fn integration_unrestricted_bypass_lets_external_mount_through() {
    // Issue #3728 acceptance: `FLUXION_EXPORT_ALLOW_UNRESTRICTED=1`
    // exists so a caller can target an external mount the allow-list
    // cannot express (S3 FUSE, CI artifact dir outside the workspace).
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var(
        fluxion::api::security::FLUXION_EXPORT_DIR_ENV,
        dir.path().to_str().unwrap(),
    );
    std::env::set_var(
        fluxion::api::security::FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV,
        "1",
    );

    // /tmp/foo.osm is outside the allow-list but the bypass lets it
    // through. (The extension pin still applies; `.txt` would be
    // refused.)
    let target = std::env::temp_dir().join("fluxion_3728_int_bypass.osm");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    let ok = fluxion::api::security::validate_export_path(target.to_str().unwrap(), "osm")
        .expect("bypass should let /tmp/foo.osm through");
    assert!(ok.is_absolute());
}

#[test]
fn integration_unrestricted_bypass_still_pins_extension() {
    // The opt-out only drops the *containment* gate. The extension pin
    // must still fire — otherwise the bypass is a total bypass that
    // lets the FFI surface overwrite any file type.
    let _env = EnvGuard::new();
    std::env::set_var(
        fluxion::api::security::FLUXION_EXPORT_ALLOW_UNRESTRICTED_ENV,
        "yes",
    );
    let err = fluxion::api::security::validate_export_path("/tmp/not_an_osm.txt", "osm")
        .err()
        .expect("extension pin must still fire under bypass");
    assert!(
        err.contains("invalid exporter file extension"),
        "got: {err}"
    );
}

#[test]
fn integration_validate_export_path_default_dir_is_exports() {
    // Sanity: the `DEFAULT_EXPORT_DIR` constant is `exports/`, matching
    // the `DEFAULT_MODEL_DIR = "models/"` posture on the read side.
    // Documented in `docs/SECURITY.md` and `docs/FEATURES.md`.
    assert_eq!(
        fluxion::api::security::DEFAULT_EXPORT_DIR,
        "exports/",
        "default exporter allow-list is relative `exports/` (mirrors models/)"
    );
}
