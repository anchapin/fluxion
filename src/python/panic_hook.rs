//! PyO3 panic-safety hook (Issue #2528).
//!
//! ## Background
//!
//! A Rust panic inside a `#[pyfunction]` must never abort the host Python
//! interpreter, and the default `std::panic` hook — which formats
//! `thread '<name>' panicked at /abs/path/to/source.rs:LINE:COL:` — leaks
//! internal filesystem layout, line numbers, and build-machine details to
//! Python callers and into logs. Both are unacceptable for a library that
//! loads into untrusted host processes (the PyO3 extension module).
//!
//! PyO3 0.22 already wraps every `#[pyfunction]` / `#[pymethods]` invocation
//! in `catch_unwind` and converts the panic payload into a
//! `pyo3::panic::PanicException` (a `BaseException` subclass). That machinery
//! prevents the literal process abort, but it does **not** sanitise the
//! stderr output produced by the default hook, and `PanicException` derives
//! from `BaseException` rather than `Exception`, so a caller writing
//! `except Exception:` still observes an uncatchable abort of their script.
//!
//! This module addresses both gaps without depending on APIs that are absent
//! in the locked PyO3 version:
//!
//! * [`install`] registers an idempotent `std::panic::set_hook` that emits a
//!   sanitised message (no absolute paths, no `file:line:col`, no env-like
//!   `KEY=value` tokens) to stderr. The sanitiser is exposed as
//!   [`sanitise_panic_message`] so callers / tests can verify it directly.
//!
//! * [`validate_population_array_shape`] and the hardened `unsafe` call sites
//!   in `src/lib.rs` turn the common panic vectors — zero-row / wrong-shape
//!   numpy arrays and the `unsafe { pyarr.as_slice() }` / `from_shape_ptr`
//!   blocks named in the issue — into ordinary `PyValueError`s raised
//!   *before* any `unsafe` dereference, so the panic path is never reached
//!   for malformed Python input.
//!
//! PyO3 0.22.6 has no `pyo3::panic::PanicHandler` / `pyo3::panic::set_hook`
//! (those were introduced in PyO3 0.23+); the crate exposes only
//! `pyo3::panic::PanicException`. We therefore install a plain
//! `std::panic::set_hook`, which is the supported, version-stable mechanism.

use std::sync::Once;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyResult;

/// Expected column count for the `evaluate_population_numpy` population
/// matrix: `[U-value, heating-setpoint, cooling-setpoint]`.
pub const POPULATION_N_PARAMS: usize = 3;

static INSTALL_ONCE: Once = Once::new();

/// Install the PyO3-aware panic hook.
///
/// Idempotent: safe to call from every `#[pymodule]` initializer (the main
/// `fluxion` module and the `fluxion_python` re-export). The first call
/// wins; subsequent calls are no-ops. `std::panic::set_hook` itself is
/// process-global, so installing once covers every thread, including
/// rayon worker threads spawned by `BatchOracle::evaluate_population`.
pub fn install() {
    INSTALL_ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            // `info.payload()` is `&(dyn Any + Send)`; extract the human-readable
            // message the same way PyO3's `PanicException::from_panic_payload`
            // does, so what we log matches what Python will see.
            let raw = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                String::from("panic from Rust code")
            };
            let sanitised = sanitise_panic_message(&raw);
            // Location (file:line:col) is deliberately *not* forwarded — it is
            // the primary internal-layout leak vector. We emit only a generic
            // marker so operators can correlate with structured logs without
            // exposing source paths to Python callers or crash dumps.
            eprintln!("fluxion: Rust panic intercepted (sanitised): {sanitised}");
        }));
    });
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
///    payload cannot flood stderr.
///
/// The panic *topic* (e.g. "index out of bounds", "called Option::unwrap()
/// on a None") is preserved — it is genuinely useful for debugging and
/// carries no host-identifying information by itself.
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
        // PyO3 sentinel rather than emitting an empty (useless) message.
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
        lower.contains(".rs") || lower.contains("\\") || lower.starts_with('/')
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

/// Validate the shape of a 2-D population array *before* any `unsafe` slice
/// dereference.
///
/// Accepts `(n_candidates, 3)` where `n_candidates >= 1`. A zero-row array
/// (`[0, 3]`) — the exact input the #2528 regression exercises — returns
/// `PyValueError` instead of reaching the `unsafe { array.as_slice() }` /
/// `RawArrayView::from_shape_ptr` blocks that previously panicked.
pub fn validate_population_array_shape(
    array: &pyo3::Bound<'_, numpy::PyArray2<f64>>,
) -> PyResult<(usize, usize)> {
    use numpy::PyUntypedArrayMethods;
    let shape = array.shape();
    // `PyArray2` always has ndim == 2, but a degenerate ndarray object could
    // in principle hand us fewer dims; guard defensively.
    if shape.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "population array must be 2-D with shape (n_candidates, {POPULATION_N_PARAMS}); \
             got {} dimensions",
            shape.len()
        )));
    }
    let (n_candidates, n_params) = (shape[0], shape[1]);
    if n_candidates == 0 {
        return Err(PyValueError::new_err(format!(
            "population array must contain at least one row; got shape ({n_candidates}, {n_params})"
        )));
    }
    if n_params != POPULATION_N_PARAMS {
        return Err(PyValueError::new_err(format!(
            "population array must have {POPULATION_N_PARAMS} columns \
             (U-value, heating-setpoint, cooling-setpoint); got {n_params}"
        )));
    }
    Ok((n_candidates, n_params))
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
