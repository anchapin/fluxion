//! # fluxion-evaluator
//!
//! Deterministic headless evaluator harness for evolutionary kernel search.
//!
//! This crate is the **in-tree foundation** that any evolver (OpenEvolve,
//! AlphaEvolve, FunSearch, …) programs against. The evolver *proposes*
//! candidate Rust implementations; this crate is the only thing that
//! *scores* them. The contract:
//!
//! 1. **Candidate intake** — two modes:
//!    - **Recompilation** (default, hermetic): the evaluator copies the
//!      candidate source into a fresh crate, runs `cargo build` against the
//!      workspace dep set inside an isolated subprocess, then runs the
//!      compiled kernel against a fixed battery of edge cases.
//!    - **Dynamic loading** (opt-in `dynamic` feature, never used in CI):
//!      load a prebuilt cdylib that implements the documented C ABI
//!      (`docs/abi.md` inside the candidate crate). The feature is
//!      intentionally a stub in this PR — enabling it does NOT add
//!      `libloading`, because that would require a new third-party crate
//!      and the project is at zero headroom on the duplicate-version
//!      budget (issue #3310).
//!
//! 2. **Standardized JSON summary** (schema v1) — every evaluation
//!    emits exactly one [`Summary`] value serialized as JSON. The schema
//!    is **versioned** (`schema_version: 1`); future revisions bump the
//!    integer and document the migration in `CHANGELOG.md`.
//!
//! 3. **Determinism** — the harness itself is pure with respect to
//!    (candidate_id, config) — identical inputs produce byte-identical
//!    JSON (sha256 digest stored in `determinism_digest`). Wall-clock is
//!    never part of the score.
//!
//! 4. **Noise-robust latency** — measured as median-of-N with reported
//!    IQR spread, never a single shot.
//!
//! 5. **Sandbox** — recompilation runs in an isolated subprocess with a
//!    fresh `target/` directory; the threat model and platform-specific
//!    resource caps are documented in [`crate::sandbox`].
//!
//! ## Threat model (summary — see [`crate::sandbox`] for full text)
//!
//! Candidate code is **untrusted**: it runs in a subprocess with no
//! network access, with bounded wall-clock and (where the host supports
//! it) bounded memory. The harness never `unwrap()`s candidate errors
//! — every candidate panic / OOM / invariant violation is funneled into
//! the summary and an exit code.
//!
//! ## Exit codes
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0    | Evaluation succeeded (may still report low fitness) |
//! | 2    | Compile failure (recompilation mode) |
//! | 3    | Invariant hard-fail (fitness forced to 0.0) |
//! | 4    | Timeout or resource cap hit |
//! | 1    | Internal harness error (should never happen) |
//!
//! ## Reference evolver
//!
//! OpenEvolve is the recommended out-of-tree evolver
//! ([`algorithmicsuperintelligence/openevolve`](https://github.com/algorithmicsuperintelligence/openevolve)),
//! driving this harness via the JSON schema documented at
//! [`Summary`]. The campaign driver itself stays out-of-tree (issue
//! #3336: "the evolver itself stays out-of-tree and pluggable").

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![warn(missing_docs)]

pub mod invariant;
pub mod kernel;
pub mod latency;
pub mod recompile;
pub mod samples;
pub mod sandbox;
pub mod summary;

// `dynamic` is gated behind the `dynamic` feature, but the module is always
// present so callers can `match` on it without a cfg gate at every call site.
// When the feature is OFF (default) every public function returns
// `DynamicLoadError::FeatureNotEnabled` — the candidate load is a no-op error
// path, not a silent success.
pub mod dynamic;

#[cfg(test)]
mod tests;

pub use invariant::{InvariantCheck, InvariantResult, InvariantViolation};
pub use kernel::{CandidateId, EdgeCase, Kernel, KernelInput, KernelOutput, ReferenceOutput};
pub use latency::{LatencyAggregate, LatencyMeasurement, TimingConfig};
pub use recompile::{RecompileConfig, RecompileOutcome, Recompiler};
pub use samples::IdentityKernel;
pub use sandbox::{SandboxConfig, SandboxEnforcer};
pub use summary::{EvaluationOutcome, SchemaVersion, Summary, CURRENT_SCHEMA_VERSION};

/// Re-exported for ergonomic `use fluxion_evaluator::Result`.
pub type Result<T> = std::result::Result<T, EvaluatorError>;

/// Top-level error type. Every public entry point returns
/// `Result<_, EvaluatorError>`; the `#[from]` impls let `?` convert
/// between the variants without manual mapping.
#[derive(Debug, thiserror::Error)]
pub enum EvaluatorError {
    /// The candidate source could not be compiled.
    #[error("compile failure: {0}")]
    CompileFailure(String),
    /// The candidate process exceeded a resource cap.
    #[error("resource cap exceeded: {0}")]
    ResourceCap(String),
    /// The dynamic-loading feature was requested but is not enabled
    /// (or not implemented in this build — see `dynamic` module docs).
    #[error("dynamic load: {0}")]
    DynamicLoad(#[from] dynamic::DynamicLoadError),
    /// A subprocess that the harness spawned failed to launch or exited
    /// unexpectedly.
    #[error("subprocess failure: {0}")]
    Subprocess(String),
    /// I/O failure reading candidate source / writing summary.
    #[error("i/o failure: {0}")]
    Io(#[from] std::io::Error),
    /// A user-supplied config was malformed (e.g. zero latency samples).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}
