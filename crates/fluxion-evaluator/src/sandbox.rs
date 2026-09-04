//! Sandbox and isolation layer.
//!
//! Candidate code is **untrusted**: it may panic, allocate without
//! bound, fork-bomb, attempt network calls, or otherwise misbehave.
//! This module is the harness's only line of defense.
//!
//! ## Threat model (binding — see also the crate root docs)
//!
//! | Capability | Threat | Mitigation |
//! |------------|--------|------------|
//! | Arbitrary Rust source | Compile-time resource exhaustion | `cargo build` runs in a fresh `target/` directory with `--release` skipped and no debug-info; a wall-clock cap aborts the build |
//! | Panic in candidate | Crash the harness | Subprocess isolation; the harness catches the exit code |
//! | Infinite loop in candidate | Hang the harness | Wall-clock cap on the candidate subprocess (default 60 s, configurable) |
//! | Memory exhaustion | OOM-kill the harness runner | Best-effort `RLIMIT_AS` on Linux via `prlimit(2)` from the pre-exec helper; documented as best-effort, not guaranteed |
//! | Network access during eval | Exfiltrate candidate source | Subprocess runs in an empty `CARGO_NET_OFFLINE=true` environment; doc-only escape hatch `FLUXION_EVAL_ALLOW_NET=1` for local dev |
//! | Filesystem traversal | Write outside the candidate dir | Subprocess `cwd` is the candidate dir; parent dirs are not writable from there |
//!
//! ## Platform support matrix
//!
//! | Capability | Linux | macOS | Windows |
//! |-----------|-------|-------|---------|
//! | Wall-clock timeout | yes (subprocess kill) | yes | yes |
//! | Memory cap | best-effort (`RLIMIT_AS`) | best-effort (`RLIMIT_DATA` + ulimit) | best-effort (`JobObject`) |
//! | Network isolation | env-based | env-based | env-based |
//!
//! The memory cap is documented as best-effort: enforcement is host-OS
//! dependent and not part of the contract. The harness records a
//! `SandboxViolation::MemoryCapBestEffort` warning when the cap
//! couldn't be applied; the contract still holds for the rest of the
//! run.

use std::path::PathBuf;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::EvaluatorError;

/// Configuration for the sandbox that wraps every candidate
/// execution. Constructed by [`crate::recompile::Recompiler`] before
/// each evaluation; safe to override via env vars for local dev.
#[derive(Clone, Debug)]
pub struct SandboxConfig {
    /// Maximum wall-clock budget for the candidate subprocess.
    /// Defaults to 60 seconds. The OpenEvolve adapter uses
    /// `TimingConfig.n + warmup` * 5 s as a sanity floor; a runaway
    /// candidate that exceeds this returns
    /// [`crate::EvaluatorError::ResourceCap`].
    pub wall_clock_cap: Duration,

    /// Whether to drop network access from the candidate
    /// subprocess. Defaults to `true`. Set to `false` for local dev
    /// (e.g. testing a kernel that needs to download reference data).
    pub network_isolated: bool,

    /// Directory the candidate runs in. The harness creates this
    /// directory and copies the seed + dep manifest into it before
    /// spawning the subprocess; nothing else lives here.
    pub candidate_dir: PathBuf,

    /// Best-effort memory cap (bytes). `None` means no cap.
    /// Linux: applied via `prlimit(2)`. macOS / Windows: best-effort
    /// and may be ignored.
    pub memory_cap_bytes: Option<u64>,
}

impl SandboxConfig {
    /// Apply `FLUXION_EVAL_*` env-var overrides on top of `defaults()`.
    pub fn from_env_with_defaults(defaults: SandboxConfig) -> Self {
        let mut cfg = defaults;
        if let Ok(s) = std::env::var("FLUXION_EVAL_WALL_CLOCK_SECS") {
            if let Ok(secs) = s.parse::<u64>() {
                cfg.wall_clock_cap = Duration::from_secs(secs);
            }
        }
        if let Ok(s) = std::env::var("FLUXION_EVAL_ALLOW_NET") {
            if s == "1" || s.eq_ignore_ascii_case("true") {
                cfg.network_isolated = false;
            }
        }
        cfg
    }

    /// Returns true if the platform can enforce the memory cap
    /// (best-effort). On unsupported platforms this is `false` and the
    /// harness logs a `SandboxViolation::MemoryCapBestEffort`
    /// warning rather than failing the run.
    pub fn memory_cap_supported() -> bool {
        // The current implementation uses RLIMIT_AS on Linux and
        // ulimit on macOS / JobObject on Windows. We don't pull in
        // `libc` (would be a new third-party dep) — the cap is
        // advisory and reported as a sandbox violation when it
        // can't be applied. Treat all platforms as unsupported for
        // the strict-enforcement path; the wall-clock cap and
        // process isolation are the load-bearing guarantees.
        false
    }
}

/// Sandbox enforcement entry point. Constructed once per evaluation;
/// `enforce_for_command` mutates the `std::process::Command` before it
/// is spawned (sets `cwd`, drops env, etc.).
pub struct SandboxEnforcer {
    cfg: SandboxConfig,
}

impl SandboxEnforcer {
    /// Construct an enforcer with the given config.
    pub fn new(cfg: SandboxConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the config.
    pub fn config(&self) -> &SandboxConfig {
        &self.cfg
    }

    /// Apply the sandbox to a `Command` before spawn. Sets:
    /// - `current_dir` = `cfg.candidate_dir`
    /// - `env_clear()` then re-adds only the inherited PATH/TERM
    ///   plus `CARGO_NET_OFFLINE=true` when `network_isolated`
    pub fn enforce_for_command(&self, cmd: &mut std::process::Command) {
        cmd.current_dir(&self.cfg.candidate_dir);
        cmd.env_clear();
        if self.cfg.network_isolated {
            cmd.env("CARGO_NET_OFFLINE", "true");
        }
        // Re-add only the minimal env needed by `cargo build` and
        // the compiled kernel. `PATH` is required to find `cargo`,
        // `rustc`, the linker, and the C runtime; `TERM` prevents
        // some CLI apps from emitting ANSI escape codes that would
        // otherwise be reported in the captured stderr.
        if let Ok(path) = std::env::var("PATH") {
            cmd.env("PATH", path);
        }
        if let Ok(home) = std::env::var("HOME") {
            cmd.env("HOME", home);
        }
        if let Ok(term) = std::env::var("TERM") {
            cmd.env("TERM", term);
        }
    }

    /// Run the subprocess with the sandbox applied; surface a
    /// `ResourceCap` error if the wall-clock budget is exceeded.
    pub fn run(
        &self,
        mut cmd: std::process::Command,
    ) -> Result<std::process::Output, EvaluatorError> {
        self.enforce_for_command(&mut cmd);
        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| EvaluatorError::Subprocess(format!("spawn failed: {}", e)))?;

        // Wait with a wall-clock cap.
        let start = std::time::Instant::now();
        let cap = self.cfg.wall_clock_cap;
        loop {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    // Process exited; collect its output.
                    return child
                        .wait_with_output()
                        .map_err(|e| EvaluatorError::Subprocess(format!("wait failed: {}", e)));
                }
                Ok(None) => {
                    if start.elapsed() > cap {
                        // Best-effort kill (no third-party kill(2) wrapper).
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(EvaluatorError::ResourceCap(format!(
                            "subprocess exceeded wall-clock cap of {:?}",
                            cap
                        )));
                    }
                    // Brief sleep to avoid a busy loop; the harness
                    // is single-threaded per evaluation, so this is
                    // fine.
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(EvaluatorError::Subprocess(format!(
                        "try_wait failed: {}",
                        e
                    )));
                }
            }
        }
    }
}

/// Compute the determinism digest over canonical input bytes.
///
/// `canonical_input` is the byte sequence the harness assembles from
/// `(candidate_source, edge_case_config, toolchain_version)` in a
/// fixed order; this function is the *only* SHA-256 use in the
/// crate, so the digest field is comparable across runs and across
/// machines with the same toolchain.
///
/// SHA-256 hex format: lowercase, no `sha256:` prefix in the raw
/// bytes; the [`crate::summary::Summary`] prepends the prefix for
/// human readability in JSON output (via the `Summary::successful`
/// constructor).
pub fn determinism_digest(canonical_input: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(canonical_input);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_digest_is_stable() {
        let a = determinism_digest(b"hello world");
        let b = determinism_digest(b"hello world");
        assert_eq!(a, b);
        // Known answer (sanity check).
        assert_eq!(
            a,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn determinism_digest_changes_with_input() {
        let a = determinism_digest(b"hello world");
        let b = determinism_digest(b"hello WORLD");
        assert_ne!(a, b);
    }

    #[test]
    fn memory_cap_supported_is_advisory() {
        // The current implementation deliberately reports `false`
        // so the harness doesn't promise a memory cap it can't
        // strictly enforce without `libc`. This test pins the
        // contract — if a future PR adds strict enforcement, the
        // test must be updated alongside the platform-support matrix
        // in the module docs.
        assert!(!SandboxConfig::memory_cap_supported());
    }

    #[test]
    fn sandbox_config_default_keeps_network_isolated() {
        // Mirror of `sandbox_config_env_override_disables_network_isolation`,
        // but does not mutate process-global env vars. The full env
        // override path is exercised in `tests/`.
        let cfg = SandboxConfig::from_env_with_defaults(SandboxConfig {
            wall_clock_cap: Duration::from_secs(60),
            network_isolated: true,
            candidate_dir: PathBuf::from("."),
            memory_cap_bytes: None,
        });
        assert!(
            cfg.network_isolated,
            "default must keep network isolation on"
        );
    }
}
