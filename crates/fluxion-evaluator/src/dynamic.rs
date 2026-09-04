//! Dynamic-loading mode (opt-in `dynamic` feature).
//!
//! ## Status
//!
//! **Stub in this PR.** The `dynamic` feature is defined in
//! `Cargo.toml` but intentionally empty — it does NOT add
//! `libloading` or any other third-party crate. The constraint from
//! issue #3336 is binding: the cargo-deny duplicate-version budget
//! is at zero headroom (issue #3310), so adding `libloading` would
//! push the workspace over budget. Loading a prebuilt cdylib without
//! `libloading` is not directly supported by `std` (`std::ffi` is
//! for FFI signatures, not `dlopen`/`dlsym`).
//!
//! Every public function in this module returns
//! [`DynamicLoadError::NotImplementedInThisBuild`] when the feature
//! is **not** enabled; when the feature *is* enabled (without
//! `libloading`), the function still returns
//! [`DynamicLoadError::NotImplementedInThisBuild`] because the
//! plumbing is intentionally absent.
//!
//! ## ABI (documented for follow-up)
//!
//! The candidate cdylib MUST expose these `extern "C"` symbols (the
//! names are stable; the wire format is in [`crate::kernel`]):
//!
//! ```text
//! /// ABI version of the candidate. Bumped when the wire format
//! /// changes; the harness refuses to load a candidate with a
//! /// version it doesn't recognize.
//! pub extern "C" fn fluxion_kernel_abi_version() -> u32;
//!
//! /// Run the candidate on one input. The harness allocates the
//! /// `output` buffer to `*output_cap` bytes; the candidate must
//! /// write ≤ `*output_cap` bytes and update `*output_cap` to the
//! /// actual written length.
//! ///
//! /// Returns 0 on success, 1 on bad input, 2 on internal error.
//! pub extern "C" fn fluxion_kernel_evaluate(
//!     input: *const u8,
//!     input_len: usize,
//!     output: *mut u8,
//!     output_cap: *mut usize,
//! ) -> i32;
//! ```
//!
//! The wire format inside `input` / `output` is the canonical
//! JSON encoding of [`crate::kernel::KernelInput`] /
//! [`crate::kernel::KernelOutput`] (with field order preserved so
//! the determinism digest is stable).
//!
//! ## Follow-up
//!
//! To make this feature functional:
//! 1. Grow the duplicate-version budget (issue #3310 follow-up).
//! 2. Add `libloading = "0.8"` as an opt-in `dependencies`,
//!    gated on the `dynamic` feature.
//! 3. Implement `load_candidate(&Path) -> Result<DynamicKernel,
//!    DynamicLoadError>` using `Library::new` + `Library::get` for
//!    the two ABI symbols.
//! 4. Update this module's tests to cover the load + dispatch path.

use std::path::Path;

use thiserror::Error;

/// Errors from dynamic loading. Every variant maps to a
/// `Summary.error` field when the harness-side runner reports
/// failure back to the caller.
#[derive(Debug, Error)]
pub enum DynamicLoadError {
    /// The `dynamic` feature is off in this build. Enable it with
    /// `cargo build --features dynamic` (still a stub in this PR —
    /// see module docs).
    #[error("dynamic-loading feature is not enabled in this build")]
    FeatureNotEnabled,

    /// The feature is enabled but the actual `dlopen` plumbing has
    /// not landed yet. This is the canonical response in this PR;
    /// follow-up work will replace it with a real implementation
    /// once `libloading` can be vendored.
    #[error("dynamic loading is not implemented in this build; see `docs/abi.md`")]
    NotImplementedInThisBuild,

    /// The candidate cdylib was not found at the given path.
    #[error("candidate cdylib not found: {0}")]
    NotFound(String),

    /// The candidate's ABI version is not recognized. The harness
    /// refuses to dispatch an unknown ABI rather than silently
    /// misinterpreting wire bytes.
    #[error("unsupported ABI version {0}; this build supports ABI v1")]
    UnsupportedAbiVersion(u32),

    /// The candidate returned a non-zero status from
    /// `fluxion_kernel_evaluate`. The harness-side runner surfaces
    /// this to the Summary rather than panicking.
    #[error("candidate returned error status {0}")]
    CandidateError(i32),
}

/// Result of a successful dynamic-load. Holds the loaded
/// library + the resolved ABI version.
#[derive(Debug)]
pub struct DynamicKernel {
    /// ABI version reported by the candidate at load time.
    pub abi_version: u32,
    // In the follow-up PR, this will hold:
    //   library: libloading::Library,
    //   evaluate_fn: libloading::Symbol<'static, extern "C" fn(...) -> i32>,
}

/// Attempt to load a candidate cdylib from the given path.
///
/// **Stub in this PR.** Returns
/// [`DynamicLoadError::NotImplementedInThisBuild`] unconditionally.
/// The signature is fixed so the follow-up PR can swap in a real
/// implementation without breaking callers.
pub fn load_candidate<P: AsRef<Path>>(_path: P) -> Result<DynamicKernel, DynamicLoadError> {
    Err(DynamicLoadError::NotImplementedInThisBuild)
}

/// Dispatch a loaded candidate against one edge case.
///
/// **Stub in this PR.** Returns
/// [`DynamicLoadError::NotImplementedInThisBuild`] unconditionally;
/// the function exists so callers can compile against the API even
/// when the feature is off.
pub fn evaluate_dynamic(
    _kernel: &DynamicKernel,
    _input_json: &str,
    _output_buf: &mut [u8],
) -> Result<usize, DynamicLoadError> {
    Err(DynamicLoadError::NotImplementedInThisBuild)
}

/// Returns true when this build has the dynamic-loading feature
/// fully implemented (loads cdylibs and dispatches them).
///
/// Always `false` in this PR — see module docs.
pub fn is_implemented() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_candidate_returns_not_implemented() {
        let err = load_candidate("does-not-matter.so").unwrap_err();
        // Without the feature, the canonical error is
        // FeatureNotEnabled; with the feature, it's
        // NotImplementedInThisBuild. Both are acceptable for this
        // PR — the harness's call site only cares that the load
        // path fails loudly (not silently).
        assert!(matches!(
            err,
            DynamicLoadError::FeatureNotEnabled | DynamicLoadError::NotImplementedInThisBuild
        ));
    }

    #[test]
    fn evaluate_dynamic_returns_not_implemented() {
        let kernel = DynamicKernel { abi_version: 1 };
        let mut buf = [0u8; 16];
        let err = evaluate_dynamic(&kernel, "{}", &mut buf).unwrap_err();
        assert!(matches!(
            err,
            DynamicLoadError::FeatureNotEnabled | DynamicLoadError::NotImplementedInThisBuild
        ));
    }

    #[test]
    fn is_implemented_is_false() {
        assert!(!is_implemented());
    }
}
