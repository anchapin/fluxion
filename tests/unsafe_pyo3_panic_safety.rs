//! Regression tests for Issue #2528:
//! "PyO3 bindings lack a panic hook — a Rust panic inside any #[pyfunction]
//! aborts the host interpreter".
//!
//! Two layers of defence are verified here:
//!
//! 1. **Panic-hook sanitiser** (`sanitise_panic_message`): pure-Rust tests,
//!    no Python interpreter required. These run on every
//!    `cargo test --features python-bindings` invocation and guard the
//!    security-relevant guarantee — internal source paths and env-like
//!    tokens must never leak across the FFI boundary.
//!
//! 2. **Unsafe-site shape hardening** (`validate_population_array_shape`):
//!    the exact input named in the issue —
//!    `numpy::PyArray2::<f64>::new_bound(py, [0, 3], false)` (zero rows) —
//!    must return a catchable `PyValueError` rather than reaching the
//!    `unsafe { array.as_slice() }` / `RawArrayView::from_shape_ptr` blocks
//!    that previously panicked and aborted the host interpreter. This needs
//!    an embedded CPython interpreter, so it is gated on the
//!    `python-bindings` feature (which links libpython via `auto-initialize`).
//!
//! The panic-hook *installation* itself (`panic_hook::install`) and its
//! idempotency are unit-tested inside `src/python/panic_hook.rs` so they do
//! not need a live interpreter either.

#![cfg(feature = "python-bindings")]

use fluxion::numpy::PyArray2;
use fluxion::pyo3::{exceptions::PyValueError, Python};
use fluxion::python::panic_hook::{
    sanitise_panic_message, validate_population_array_shape, POPULATION_N_PARAMS,
};

// ---------------------------------------------------------------------------
// Layer 1: pure-Rust sanitiser regression (no interpreter required).
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
// Layer 2: shape hardening at the FFI boundary.
// ---------------------------------------------------------------------------

/// The exact input from the issue body: a `(0, 3)` population array.
///
/// Pre-fix this reached `unsafe { array.as_slice() }` and, on a non-contiguous
/// or degenerate shape, panicked and aborted the host interpreter. Post-fix
/// `validate_population_array_shape` rejects it with `PyValueError` *before*
/// any `unsafe` dereference.
#[test]
fn zero_row_population_array_returns_pyvalueerror() {
    Python::with_gil(|py| {
        // `(0, 3)`: zero candidates, three param columns — well-formed but
        // empty. `false` = not Fortran-ordered (C-contiguous).
        //
        // SAFETY: `PyArray2::new_bound` allocates an *uninitialised* buffer;
        // for a zero-element (`[0, 3]`) array there is no element storage to
        // initialise, so reading through `validate_population_array_shape`
        // (which only inspects `.shape()`, never the data pointer) is sound.
        let empty = unsafe { PyArray2::<f64>::new_bound(py, [0, POPULATION_N_PARAMS], false) };
        let err = validate_population_array_shape(&empty).expect_err(
            "a zero-row population array must be rejected before any unsafe slice access",
        );
        // Must surface as a catchable Exception subclass (PyValueError),
        // never a PanicException / BaseException / process abort.
        assert!(
            err.is_instance_of::<PyValueError>(py),
            "expected PyValueError, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("at least one row") || msg.contains("0"),
            "error message should explain the zero-row cause: {msg}"
        );
    });
}

#[test]
fn wrong_column_count_returns_pyvalueerror() {
    Python::with_gil(|py| {
        // Two candidates, two columns — wrong arity for `[U-value, heating, cooling]`.
        //
        // SAFETY: same as above — `validate_population_array_shape` inspects
        // only the shape, not the uninitialised element storage, so the
        // `new_bound` allocation is safe to hand it.
        let bad = unsafe { PyArray2::<f64>::new_bound(py, [2, 2], false) };
        let err =
            validate_population_array_shape(&bad).expect_err("wrong column count must be rejected");
        assert!(
            err.is_instance_of::<PyValueError>(py),
            "expected PyValueError, got {err:?}"
        );
    });
}

#[test]
fn well_formed_population_array_is_accepted() {
    Python::with_gil(|py| {
        // Shape (2, 3) with dummy data — must pass and report the right dims.
        // Initialise non-zero so a future strict-mode validator cannot reject
        // on contents rather than shape.
        let good =
            PyArray2::<f64>::from_vec2_bound(py, &[vec![0.5, 20.0, 24.0], vec![0.7, 18.0, 26.0]])
                .unwrap();
        let (n_candidates, n_params) =
            validate_population_array_shape(&good).expect("a (2, 3) population array is valid");
        assert_eq!((n_candidates, n_params), (2, POPULATION_N_PARAMS));
    });
}
