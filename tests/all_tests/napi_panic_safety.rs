//! Regression tests for Issue #3734:
//! "Add panic containment to Node bindings parity with the Python panic hook".
//!
//! Two layers of defence are verified here, mirroring the Python hook
//! regression suite at `tests/all_tests/unsafe_pyo3_panic_safety.rs`:
//!
//! 1. **Panic-hook sanitiser** (`sanitise_panic_message`): pure-Rust tests,
//!    no Node.js runtime required. These run on every
//!    `cargo test --features napi-bindings` invocation and guard the
//!    security-relevant guarantee — internal source paths and env-like
//!    tokens must never leak across the FFI boundary.
//!
//! 2. **`catch_unwind_napi` boundary** (`catch_unwind_napi`): the helper
//!    that wraps `#[napi]` function bodies. A panic inside the wrapped
//!    closure MUST surface as a `napi::bindgen_prelude::Error` carrying
//!    the sanitised message and MUST NOT abort the host process.
//!
//! The panic-hook *installation* itself (`panic_hook::install`) and its
//! idempotency are unit-tested inside `src/napi/panic_hook.rs` so they do
//! not need the Node runtime either.
//!
//! Acceptance from the issue body: "A regression test asserts a panic
//! inside a stepped model surfaces as a JS-level error, not a process
//! abort." This file is the Rust-side equivalent — the actual JS-level
//! verification is exercised by `npm/test.js`'s `BatchOracle` constructor
//! round-trip, which would abort the Node process if the boundary were
//! not in place.

#![cfg(feature = "napi-bindings")]

use fluxion::napi::panic_hook::{catch_unwind_napi, sanitise_panic_message};

// ---------------------------------------------------------------------------
// Layer 1: pure-Rust sanitiser regression (no Node.js runtime required).
// ---------------------------------------------------------------------------

#[test]
fn sanitiser_strips_source_path_and_location() {
    let raw = "index out of bounds: the len is 0 but the index is 0 \
               at /home/alex/Projects/fluxion/src/lib.rs:1645:18";
    let got = sanitise_panic_message(raw);
    assert!(!got.contains("/home/"), "absolute path leaked: {got}");
    assert!(!got.contains("lib.rs"), "source basename leaked: {got}");
    assert!(!got.contains(":1645"), "line number leaked: {got}");
    assert!(
        got.contains("index out of bounds"),
        "panic topic must survive sanitising: {got}"
    );
}

#[test]
fn sanitiser_strips_env_like_secret() {
    let raw = "called Result::unwrap() on an Err value: DWAVE_API_TOKEN=deadbeef";
    let got = sanitise_panic_message(raw);
    assert!(
        !got.contains("DWAVE_API_TOKEN"),
        "secret-like token leaked: {got}"
    );
    assert!(
        got.contains("Result::unwrap"),
        "topic dropped during secret scrub: {got}"
    );
}

#[test]
fn sanitiser_preserves_innocuous_message() {
    let raw = "called Option::unwrap() on a None value";
    assert_eq!(sanitise_panic_message(raw), raw);
}

// ---------------------------------------------------------------------------
// Layer 2: `catch_unwind_napi` boundary.
//
// The acceptance criterion from issue #3734: "A panic inside a stepped
// model surfaces as a JS-level error, not a process abort." Exercising the
// boundary directly is the closest Rust-side proxy for that guarantee
// without spinning up a Node child process in CI.
// ---------------------------------------------------------------------------

/// `catch_unwind_napi` MUST convert a panic into a `napi::Error` carrying
/// the sanitised message — the host process must keep running.
#[test]
fn catch_unwind_napi_converts_panic_to_error() {
    // If the panic were not caught the test process would abort here.
    let result: napi::bindgen_prelude::Result<()> =
        catch_unwind_napi(|| panic!("synthetic NAPI panic for issue #3734"));

    let err = result.expect_err("panic must surface as Err, not Ok(())");
    let msg = format!("{err}");
    assert!(
        msg.contains("synthetic NAPI panic"),
        "raw panic payload was lost: {msg}"
    );
    assert!(
        !msg.contains("/home/"),
        "absolute path leaked through catch_unwind boundary: {msg}"
    );
}

/// `catch_unwind_napi` MUST NOT add a `file:line:col` location stack
/// (which the default hook would emit) — the issue body explicitly names
/// this as the primary internal-layout leak vector.
#[test]
fn catch_unwind_napi_strips_location_stack() {
    // Construct the panic at a known site so the default hook would emit
    // `src/tests/all_tests/napi_panic_safety.rs:LINE:COL` if our sanitiser
    // weren't in place.
    let line_marker = std::panic::Location::caller();
    let result: napi::bindgen_prelude::Result<()> = catch_unwind_napi(|| {
        let _ = line_marker;
        panic!("oops")
    });

    let err = result.expect_err("expected panic to be caught");
    let msg = format!("{err}");
    assert!(
        !msg.contains("napi_panic_safety.rs"),
        "test source path leaked: {msg}"
    );
    assert!(msg.contains("oops"), "panic payload lost: {msg}");
}

/// `catch_unwind_napi` MUST pass through a successful return value
/// unchanged — `Ok(42)` flows through without being treated as a panic.
#[test]
fn catch_unwind_napi_passes_through_ok() {
    let result: napi::bindgen_prelude::Result<i32> = catch_unwind_napi(|| Ok(42));
    assert_eq!(result.expect("Ok closure must pass through"), 42);
}

/// `catch_unwind_napi` MUST propagate an `Err` returned by the closure
/// without wrapping or mutating it (the panic-catching path is distinct
/// from the normal-error path).
#[test]
fn catch_unwind_napi_propagates_inner_err() {
    let result: napi::bindgen_prelude::Result<i32> = catch_unwind_napi(|| {
        Err(napi::bindgen_prelude::Error::from_reason(
            "plain napi error",
        ))
    });
    let err = result.expect_err("inner Err must surface");
    // `napi::Error::Display` prefixes the status enum name; the inner
    // `reason` is still in the message. We assert on the reason substring
    // so the test is not coupled to napi-rs's exact Display formatting.
    let msg = format!("{err}");
    assert!(
        msg.contains("plain napi error"),
        "inner Err message must propagate unchanged, got: {msg}"
    );
}

/// `catch_unwind_napi` MUST sanitise `KEY=value` env-like tokens that
/// sometimes land in panic payloads from `expect("…api_key=…")` style
/// messages, even when the panic is raised at the FFI boundary.
#[test]
fn catch_unwind_napi_strips_env_like_secret_in_payload() {
    let result: napi::bindgen_prelude::Result<()> =
        catch_unwind_napi(|| panic!("inner failure: API_KEY=sk-live-deadbeef at /tmp/x.rs:1:1"));

    let err = result.expect_err("expected panic to be caught");
    let msg = format!("{err}");
    assert!(
        !msg.contains("API_KEY="),
        "env token leaked through boundary: {msg}"
    );
    assert!(
        !msg.contains("/tmp/x.rs"),
        "path leaked through boundary: {msg}"
    );
    assert!(msg.contains("inner failure"), "topic dropped: {msg}");
}
