// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Fail-closed open of the operator-controlled `FLUXION_AUDIT_LOG` path
//! (Issue #3639, CWE-732 — incorrect default permissions).
//!
//! The legacy open in `src/bin/fluxion_rest.rs` used
//! `OpenOptions::new().create(true).append(true).open(p)`, which
//! (a) followed symlinks at every path component, (b) created the file
//! with the process umask (typically 0644 in containers, exposing the
//! audit trail — which carries client-identifier fingerprints since
//! Issue #3652 — to co-tenants), and (c) accepted a pre-existing path
//! that was not a regular file (FIFO, socket, device node).
//!
//! [`open_audit_log`] closes all three gaps with a strict, two-attempt
//! protocol that mirrors the `O_NOFOLLOW` single-handle pattern from
//! `open_and_verify_onnx` (Issue #3573):
//!
//! 1. **Create-first** — `O_CREAT | O_EXCL | O_APPEND` with an explicit
//!    `mode 0o600`. `O_EXCL` fails loudly when the path already exists
//!    (no clobbering of pre-existing audit trails, no symlink swap
//!    window), and the mode is applied atomically with creation, so
//!    there is no interval in which the file exists with lax
//!    permissions.
//! 2. **Strict re-open on `EEXIST`** — the existing path is opened
//!    `O_APPEND` with `create(false)` and `O_NOFOLLOW`, so a symlink at
//!    the final component is refused (`ELOOP`) even when an attacker
//!    swaps the directory entry between the two attempts.
//! 3. **Regular-file pin** — the handle's own `fstat` (not a path stat)
//!    must report a regular file; FIFOs, sockets, device nodes, and
//!    procfs pseudo-files are refused. `O_NONBLOCK` is included in the
//!    open flags so a FIFO target fails fast instead of blocking boot
//!    until an unrelated reader appears.
//! 4. **Owner-only tightening** — `fchmod 0o600` is applied to the held
//!    handle on both the create and re-open paths, so the audit log is
//!    owner-read/write regardless of the on-disk mode or umask.
//!
//! The protocol is fail-closed: every failure propagates as `Err` and
//! the caller must NOT fall back to a lax open — audit events stay on
//! stdout only (announced loudly), matching the MQTT-twin
//! "always TLS, no runtime bypass" posture in `AGENTS.md`.

use std::io;
use std::path::Path;

/// Open the audit log at `path` with the Issue #3639 strict protocol.
///
/// Semantics: the file is opened append-only and is **never truncated**
/// — pre-existing audit content is preserved (strict-append, per the
/// issue's `O_EXCL`-only-when-creating prescription). On success the
/// returned handle is append-mode and the on-disk mode is owner-only
/// (`0o600`). On any failure the caller must treat the audit file as
/// unavailable; there is deliberately no permissive fallback.
#[cfg(unix)]
pub fn open_audit_log(path: &Path) -> io::Result<std::fs::File> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    // O_NOFOLLOW / O_NONBLOCK values per fcntl(2). Hard-coded to avoid
    // pulling in `libc` as a direct dependency just for two constants
    // (same rationale as `open_and_verify_onnx`, Issue #3573).
    #[cfg(target_os = "linux")]
    const O_NOFOLLOW: i32 = 0o400_000;
    #[cfg(any(
        target_os = "macos",
        target_os = "freebsd",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
    ))]
    const O_NOFOLLOW: i32 = 0x0100;
    #[cfg(any(target_os = "openbsd", target_os = "netbsd"))]
    const O_NOFOLLOW: i32 = 0x0200;
    // Conservative fallback: no symlink rejection on unknown Unix
    // targets; the regular-file pin still refuses FIFO/socket/device.
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
    )))]
    const O_NOFOLLOW: i32 = 0;

    #[cfg(target_os = "linux")]
    const O_NONBLOCK: i32 = 0o4000;
    #[cfg(any(
        target_os = "macos",
        target_os = "freebsd",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "openbsd",
        target_os = "netbsd",
    ))]
    const O_NONBLOCK: i32 = 0x0004;
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
    )))]
    const O_NONBLOCK: i32 = 0;

    // ----- (1) create-first: O_CREAT | O_EXCL | O_APPEND, mode 0o600 ----
    match std::fs::OpenOptions::new()
        .append(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(path)
    {
        Ok(file) => {
            // Belt-and-braces fchmod so the mode is owner-only even under
            // a pathological umask that masks owner bits.
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            Ok(file)
        }
        // ----- (2) strict re-open of the pre-existing path --------------
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            let file = std::fs::OpenOptions::new()
                .append(true)
                .create(false)
                .custom_flags(O_NOFOLLOW | O_NONBLOCK)
                .open(path)
                .map_err(|e| {
                    if e.raw_os_error() == Some(40 /* ELOOP on Linux */)
                        || e.raw_os_error() == Some(62 /* ELOOP on macOS/BSD */)
                    {
                        io::Error::new(
                            e.kind(),
                            format!(
                                "FLUXION_AUDIT_LOG path {} is a symlink; O_NOFOLLOW refused \
                                 the open (Issue #3639, fail-closed)",
                                path.display()
                            ),
                        )
                    } else {
                        e
                    }
                })?;

            // ----- (3) regular-file pin via the held handle's fstat ------
            let ft = file.metadata()?.file_type();
            if !ft.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "FLUXION_AUDIT_LOG path {} is not a regular file (file_type={ft:?}); \
                         refusing (Issue #3639, fail-closed)",
                        path.display()
                    ),
                ));
            }

            // ----- (4) owner-only tightening via fchmod ------------------
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            Ok(file)
        }
        Err(e) => Err(e),
    }
}

/// Non-Unix variant. Windows has no `O_NOFOLLOW`/`mode` equivalents in
/// `std`, so this is a documented best-effort port of the strict
/// protocol: the symlink check is performed with the non-following
/// `symlink_metadata` before the open (a narrow TOCTOU window remains —
/// deployments requiring byte-exact symlink protection must run the
/// audit log on a Unix host, matching the Unix-targeted release
/// posture), creation uses `create_new` (the `O_CREAT | O_EXCL`
/// equivalent), and existing files are opened append-only without
/// create. Owner-only permissions (mode 0600) have no direct
/// Windows-std equivalent; the operator must apply an owner-only ACL to
/// the audit directory.
#[cfg(not(unix))]
pub fn open_audit_log(path: &Path) -> io::Result<std::fs::File> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => std::fs::OpenOptions::new()
            .append(true)
            .create_new(true)
            .open(path),
        Err(e) => Err(e),
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "FLUXION_AUDIT_LOG path {} is a symlink; refusing the open \
                         (Issue #3639, fail-closed)",
                        path.display()
                    ),
                ));
            }
            if !meta.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "FLUXION_AUDIT_LOG path {} is not a regular file; refusing the open \
                         (Issue #3639, fail-closed)",
                        path.display()
                    ),
                ));
            }
            std::fs::OpenOptions::new()
                .append(true)
                .create(false)
                .open(path)
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Create a fresh tempdir and return it with a unique audit-log path
    /// inside. The `TempDir` must be kept alive for the path to remain.
    fn temp_audit_path(tag: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("audit-{tag}.log"));
        (dir, path)
    }

    /// Issue #3639 acceptance: a freshly created audit log has owner-only
    /// permissions regardless of umask, and the handle is usable.
    #[test]
    fn new_file_is_created_with_owner_only_mode() {
        let (_dir, path) = temp_audit_path("mode0600");
        let file = open_audit_log(&path).expect("strict open must create the file");
        let mode = path.metadata().expect("stat").permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "created audit log must be 0o600 (CWE-732), got {:o}",
            mode & 0o777
        );
        drop(file);
        let contents = std::fs::read_to_string(&path).expect("read back");
        assert_eq!(contents, "", "a fresh audit log starts empty");
    }

    /// Issue #3639 acceptance: a pre-created symlink at the audit path is
    /// rejected (O_NOFOLLOW), and the symlink's target is never touched.
    #[test]
    fn symlinked_audit_target_is_refused() {
        let (dir, path) = temp_audit_path("symlink");
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "do-not-touch\n").expect("seed victim");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&victim, &path).expect("create symlink");

        let err = open_audit_log(&path).expect_err("symlink must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("symlink") || msg.contains("symbolic"),
            "refusal must name the symlink defence, got: {msg}"
        );
        assert_eq!(
            std::fs::read_to_string(&victim).expect("victim readable"),
            "do-not-touch\n",
            "symlink target must be untouched"
        );
    }

    /// Issue #3639 acceptance: an existing regular file is opened
    /// strict-append — pre-existing content is preserved (never
    /// clobbered/truncated) and the mode is tightened to 0o600.
    #[test]
    fn existing_file_is_appended_not_clobbered_and_tightened() {
        let (_dir, path) = temp_audit_path("append");
        std::fs::write(&path, "existing audit line\n").expect("seed existing log");
        // Start from a lax mode to prove the tightening fires.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("seed 0644");

        let mut file = open_audit_log(&path).expect("strict re-open must succeed");
        use std::io::Write;
        file.write_all(b"appended audit line\n")
            .expect("append write");
        drop(file);

        let contents = std::fs::read_to_string(&path).expect("read back");
        assert_eq!(
            contents, "existing audit line\nappended audit line\n",
            "strict-append must preserve pre-existing audit content"
        );
        let mode = path.metadata().expect("stat").permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "existing log must be tightened to 0o600"
        );
    }

    /// Issue #3639 acceptance: a FIFO at the audit path is refused instead
    /// of being opened (which would either block on a missing reader or tee
    /// audit records into an anonymous pipe).
    #[test]
    fn fifo_target_is_refused() {
        let (_dir, path) = temp_audit_path("fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .expect("spawn mkfifo");
        if !status.success() {
            // Environment lacks a working mkfifo; the FIFO branch cannot be
            // exercised here — the regular-file pin is still covered by the
            // re-open path on non-FIFO platforms.
            return;
        }
        let err = open_audit_log(&path).expect_err("FIFO must be refused");
        let msg = format!("{err}");
        // Either defence may fire, both are valid refusals:
        // - the regular-file pin (a reader was present, or the platform
        //   opened the FIFO successfully with O_NONBLOCK);
        // - ENXIO from the open itself: `O_NONBLOCK` on a FIFO with no
        //   reader fails fast instead of blocking boot on a missing
        //   consumer (Linux/macOS errno 6).
        assert!(
            msg.contains("not a regular file") || err.raw_os_error() == Some(6),
            "refusal must come from the regular-file pin or ENXIO, got: {msg}"
        );
    }

    /// `O_EXCL` semantics on the create attempt: the second open must never
    /// truncate the first one's content even when called back-to-back.
    #[test]
    fn repeated_opens_accumulate_instead_of_clobbering() {
        let (_dir, path) = temp_audit_path("excl");
        use std::io::Write;
        {
            let mut f1 = open_audit_log(&path).expect("first open creates");
            f1.write_all(b"first\n").expect("write first");
        }
        {
            let mut f2 = open_audit_log(&path).expect("second open re-opens existing");
            f2.write_all(b"second\n").expect("write second");
        }
        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "first\nsecond\n",
            "reopen must append, not truncate (no silent clobber)"
        );
    }
}
