//! PyO3 Python bindings integration tests
//!
//! NOTE: Rust-side PyO3 integration tests are intentionally not implemented.
//!
//! ## Rationale
//!
//! INTEG-04 requirement ("Python-side integration tests validate PyO3 bindings
//! with real NumPy arrays") is fully satisfied by Python-side tests in
//! tests/integration/test_numpy_arrays.py.
//!
//! Python-side tests provide comprehensive FFI boundary coverage:
//! - test_array_shape_validation: Validates 1D, 2D, 3D array shapes preserved
//! - test_array_dtype_conversion: Validates f32/f64 dtype conversion to f64 internally
//! - test_large_numpy_array_handling: Validates 10,000+ element arrays work
//! - test_empty_array_handling: Validates empty arrays handled gracefully
//! - test_nan_array_handling: Validates NaN/Inf values propagated correctly
//!
//! ## Why Not Rust-Side Tests?
//!
//! Implementing Rust-side PyO3 integration tests requires resolving Python
//! symbol linking issues (undefined symbols: PyBytes_AsString, PyBytes_Size, etc.)
//! when running with `--features python-bindings`. This adds complexity without
//! additional benefit, since Python-side tests already comprehensively validate the
//! FFI boundary from the perspective that matters most: actual Python usage.
//!
//! ## Decision Summary
//!
//! **Decision:** Accept Python-side tests as sufficient for INTEG-04 requirement
//!
//! **Rationale:**
//! 1. Python-side tests fully validate the observable FFI contract
//! 2. Rust-side conversion logic in src/physics/cta.rs already has its own tests
//! 3. Proposed Rust-side PyO3 tests would have low value (testing PyO3 boilerplate)
//! 4. Python symbol linking blocker makes Rust-side tests high-effort/low-value
//! 5. INTEG-04 requirement is satisfied: "Python-side integration tests validate PyO3 bindings with real NumPy arrays"
//!
//! **Implementation:** Python-side tests in tests/integration/test_numpy_arrays.py provide 5 comprehensive tests:
//! - Shape validation (1D, 2D, 3D arrays)
//! - Dtype conversion (f32, f64 → f64 internal)
//! - Large arrays (10,000+ elements)
//! - Empty arrays (graceful handling)
//! - NaN/Inf handling (correct propagation)
//!
//! **Future Considerations:** Rust-side tests can be added in the future if specific edge cases are discovered that Python-side tests don't catch.
//!
//! See Phase 21 Plan 07 (21-07-PLAN.md) for full decision context and 21-07-SUMMARY.md for documentation of the decision outcome.
