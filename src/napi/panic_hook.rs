// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI panic-safety hook + `catch_unwind` boundary (Issue #3734).
//!
//! ## Background
//!
//! A Rust panic inside a `#[napi]` function must never abort the host Node
//! process. Unlike PyO3, `napi-rs` (the Node binding layer used here) does
//! **not** wrap `#[napi]` invocations in `catch_unwind` automatically —
//! a reachable panic on the NAPI boundary tears down the entire Node
//! process rather than rejecting the call with a JavaScript `Error`.
//!
//! Phase A8 makes the gauge dispatcher panic loudly if a gauge backend is
//! missing, and `BatchOracle` is the population-parallel entry point that
//! steps physics from JS — exactly the hot path that must not abort the
//! host. To prevent a single reachable panic from killing the process we
//! add two layers of defence, mirroring [`crate::python::panic_hook`]
//! (Issue #2528):
//!
//! * [`install`] registers an idempotent `std::panic::set_hook` that emits
//!   a sanitised message (no absolute paths, no `file:line:col`, no
//!   env-like `KEY=value` tokens) to stderr. The sanitiser is exposed as
//!   [`sanitise_panic_message`] so callers / tests can verify it directly.
//!   This catches panics raised on worker threads (e.g. Rayon workers
//!   spawned by `BatchOracle::evaluate_population`) before they print
//!   internal layout to stderr.
//!
//! * [`catch_unwind_napi`] wraps a closure at the NAPI call boundary and
//!   converts any panic into a `napi::bindgen_prelude::Error` carrying the
//!   sanitised message, so a panic on the entry thread surfaces as a
//!   rejected JS promise / thrown `Error` instead of aborting the host.
//!
//! The `.expect` previously at `src/napi/batch_oracle.rs:83` (a known
//! panic vector named in issue #3734) is replaced with a `?` + `map_err`
//! path so the constructor returns a `NapiError` instead of panicking.
//!
//! ## Why two layers?
//!
//! A pure `std::panic::set_hook` alone is *not* sufficient on the NAPI
//! boundary: even with sanitised stderr output, the default unwind path
//! would still abort the Node process when the panic crosses the FFI
//! boundary unless something has caught it. Conversely, `catch_unwind`
//! alone would lose any panic that fires on a Rayon worker thread before
//! the calling thread joins back (because the panic propagates to the
//! worker, not the NAPI caller). The two layers cover both surfaces.
//!
//! ## Why not share the sanitiser with `crate::python::panic_hook`?
//!
//! The algorithms are identical but the modules are deliberately kept
//! separate: the Python hook is gated on `feature = "python-bindings"`
//! and depends on `pyo3` for `validate_population_array_shape`; the NAPI
//! hook is gated on `feature = "napi-bindings"` and depends on
//! `napi-rs` for `catch_unwind_napi`. Sharing the sanitiser would force
//! every consumer to take both feature flags. Each module duplicates the
//! ~30 lines of sanitisation logic so the dependency-light contract
//! holds.

use std::panic::AssertUnwindSafe;
use std::sync::Once;

static INSTALL_ONCE: Once = Once::new();

/// Install the NAPI-aware panic hook.
///
/// Idempotent: safe to call from `register()` on every native module
/// load (the `Once` ensures the hook is installed exactly once per
/// process). `std::panic::set_hook` itself is process-global, so
/// installing once covers every thread, including Rayon worker threads
/// spawned by `BatchOracle::evaluate_population`.
pub fn install() {
    INSTALL_ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            // `info.payload()` is `&(dyn Any + Send)`; extract the
            // human-readable message the same way the Python hook does,
            // so the JS/Node side sees the same sanitised text that
            // Python callers would.
            let raw = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                String::from("panic from Rust code")
            };
            let sanitised = sanitise_panic_message(&raw);
            // Location (file:line:col) is deliberately *not* forwarded —
            // it is the primary internal-layout leak vector. We emit only
            // a generic marker so operators can correlate with structured
            // logs without exposing source paths to JS callers or crash
            // dumps.
            eprintln!("fluxion: Rust panic intercepted (NAPI sanitised): {sanitised}");
        }));
    });
}

/// Run a NAPI-bound closure under `catch_unwind`, converting any panic
/// into a [`napi::bindgen_prelude::Error`] carrying the sanitised panic
/// message.
///
/// Use this at the body of every `#[napi]` function that runs
/// potentially-panicking Rust code — especially the population-parallel
/// hot loops in [`crate::napi::BatchOracle`] (which spawn Rayon workers
/// that may panic on malformed input) and the `ThermalSelector` /
/// gauge-dispatcher code paths named in Phase A8.
///
/// Without this wrapper, a panic inside the closure would unwind across
/// the FFI boundary and abort the host Node process. With it, the panic
/// is caught on the same thread that entered the NAPI function, the
/// stderr hook fires (sanitised message), and the JS caller observes a
/// rejected promise / thrown `Error` they can handle normally.
///
/// `AssertUnwindSafe` is used to lift the `UnwindSafe` bound on `F`'s
/// closure output — the panic-catching boundary on FFI is exactly the
/// case `AssertUnwindSafe` exists for, and the alternative (forcing every
/// caller to wrap in `AssertUnwindSafe` themselves) would leak the
/// `catch_unwind` detail into the call sites the issue is trying to fix.
pub fn catch_unwind_napi<F, T>(f: F) -> napi::bindgen_prelude::Result<T>
where
    F: FnOnce() -> napi::bindgen_prelude::Result<T>,
{
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let raw = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                String::from("panic from Rust code")
            };
            let sanitised = sanitise_panic_message(&raw);
            Err(napi::bindgen_prelude::Error::from_reason(format!(
                "fluxion: Rust panic intercepted at NAPI boundary: {sanitised}"
            )))
        }
    }
}

/// Sanitise a panic message for cross-FFI emission.
///
/// Strips, in priority order:
/// 1. Absolute source paths (`/home/…/src/foo.rs`, `C:\dev\foo.rs`) and the
///    `file.rs:line:col` / `file.rs:line` suffixes the default hook appends.
/// 2. Bare `something.rs` basenames (relative panics from `assert!` etc.).
/// 3. `KEY=value` / `TOKEN=…` env-like leaks that sometimes land in panic
///    payloads from `expect("…api_key=…")` style messages.
/// 4. Collapses runs of whitespace and caps overall length so a pathological
///    payload cannot flood stderr or a JS error chain.
///
/// The panic *topic* (e.g. "index out of bounds", "called Option::unwrap()
/// on a None") is preserved — it is genuinely useful for debugging and
/// carries no host-identifying information by itself.
///
/// Mirrors [`crate::python::panic_hook::sanitise_panic_message`] so both
/// binding surfaces emit byte-identical sanitised messages for the same
/// raw payload.
pub fn sanitise_panic_message(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    for chunk in msg.split_whitespace() {
        let trimmed = chunk.trim_matches(|c: char| c.is_ascii_punctuation());
        // Skip anything that looks like a path with a `.rs` segment or
        // a `file:line:col` location stack.
        if is_path_like(chunk) || is_path_like(trimmed) {
            continue;
        }
        // Skip env-like `KEY=value` leaks.
        if is_env_like(chunk) {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(chunk);
    }

    let mut out = if out.is_empty() {
        // The whole payload was a path/location — fall back to the generic
        // sentinel rather than emitting an empty (useless) message.
        String::from("panic from Rust code")
    } else {
        out
    };

    // Hard cap: never emit more than this many characters, so a malicious or
    // pathological panic payload cannot DOS stderr / crash aggregation.
    const MAX_LEN: usize = 512;
    if out.len() > MAX_LEN {
        out.truncate(MAX_LEN);
        out.push('…');
    }
    out
}

/// Heuristic: does this whitespace-delimited chunk look like a source path
/// or a `file:line:col` location token?
fn is_path_like(chunk: &str) -> bool {
    // `foo.rs`, `/abs/foo.rs`, `C:\foo.rs`, `foo.rs:42`, `foo.rs:42:13`
    let looks_like_rust_file = |s: &str| {
        let lower = s.to_ascii_lowercase();
        lower.contains(".rs") || lower.contains("\\") || s.starts_with('/')
    };
    let looks_like_loc = |s: &str| {
        // ends in `:digits` possibly chained (`:digits:digits`)
        let mut rest = s;
        let mut saw_colon_digit = false;
        while let Some(idx) = rest.rfind(':') {
            let tail = &rest[idx + 1..];
            if tail.is_empty() {
                break;
            }
            if !tail.bytes().all(|b| b.is_ascii_digit()) {
                break;
            }
            saw_colon_digit = true;
            rest = &rest[..idx];
        }
        saw_colon_digit && !rest.is_empty()
    };
    looks_like_rust_file(chunk) || looks_like_loc(chunk)
}

/// Heuristic: does this chunk look like a leaked `KEY=value` env binding?
fn is_env_like(chunk: &str) -> bool {
    let Some((key, _val)) = chunk.split_once('=') else {
        return false;
    };
    // Non-empty UPPER_SNAKE_CASE key prefix before `=`.
    !key.is_empty()
        && key
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b == b'_' || b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitiser_strips_absolute_paths() {
        let msg = "index out of bounds: the len is 0 but the index is 0 at /home/alex/Projects/fluxion/src/lib.rs:1645:18";
        let got = sanitise_panic_message(msg);
        assert!(!got.contains("/home/"), "absolute path leaked: {got}");
        assert!(!got.contains("lib.rs"), "basename leaked: {got}");
        assert!(!got.contains(":1645"), "line number leaked: {got}");
        assert!(
            got.contains("index out of bounds"),
            "panic topic dropped: {got}"
        );
    }

    #[test]
    fn sanitiser_strips_env_like_tokens() {
        let msg = "called Result::unwrap() on an Err value: API_KEY=sk-live-12345";
        let got = sanitise_panic_message(msg);
        assert!(!got.contains("API_KEY="), "env token leaked: {got}");
        assert!(got.contains("Result::unwrap"), "topic dropped: {got}");
    }

    #[test]
    fn sanitiser_preserves_plain_messages() {
        let msg = "called Option::unwrap() on a None value";
        assert_eq!(sanitise_panic_message(msg), msg);
    }

    #[test]
    fn sanitiser_falls_back_when_only_path_remains() {
        let got = sanitise_panic_message("src/foo.rs:10:5");
        assert_eq!(got, "panic from Rust code");
    }

    #[test]
    fn install_is_idempotent() {
        // Installing twice must not panic / must not stack hooks.
        install();
        install();
        // Re-acquire the chain to leave the process in a clean state for
        // other tests: `take_hook` + `set_hook(no-op)` then re-install.
        let _prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        // Because INSTALL_ONCE has already fired, this is a no-op — the
        // no-op hook we just installed stays. That is acceptable for a unit
        // test; production installs run exactly once at module init.
    }

    #[test]
    fn env_like_heuristic_is_conservative() {
        // Must NOT treat normal `=`-bearing tokens (URLs, math) as env leaks.
        assert!(!is_env_like("a=1"));
        assert!(!is_env_like("https://x"));
        assert!(is_env_like("SECRET_TOKEN=abc"));
        assert!(is_env_like("API_KEY="));
    }
}
