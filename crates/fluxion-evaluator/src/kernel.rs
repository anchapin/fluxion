//! The `Kernel` trait that candidates implement.
//!
//! A *candidate* is one proposed rewrite of a fixed numerical kernel
//! (state-space CTF discretization, solar/irradiance reductions, BDF DAE
//! step, …). The harness exercises it through a single trait so the
//! recompiled binary can be loaded and dispatched without coupling the
//! harness to any particular physics domain.
//!
//! ## Default-execution semantics
//!
//! `evaluate(&self, input)` returns the candidate's [`KernelOutput`]
//! **without** performing any invariant or accuracy check — those are
//! the harness's responsibility (see [`crate::invariant`]). This
//! separation keeps the trait minimal and lets candidates stay focused
//! on producing correct numerical output.
//!
//! ## Reference vs candidate
//!
//! The harness pairs each [`KernelInput`] with a [`ReferenceOutput`]
//! produced by the *current* (known-good) implementation. The
//! harness-side accuracy aggregator compares the candidate's
//! [`KernelOutput`] against the reference and feeds the residual into
//! the fitness score; the candidate itself never sees the reference
//! values (this prevents the evolver from overfitting the training set
//! by reading them out of the harness).
//!
//! ## Recompilation contract
//!
//! When the harness recompiles a candidate, the seed file must contain
//! a concrete type `pub struct Candidate` that implements
//! [`Kernel`]. The harness-generated wrapper (see
//! [`crate::recompile`]) instantiates `Candidate::default()` and calls
//! `evaluate(&Candidate::default(), input)` once per edge case.
//!
//! ## ABI (dynamic-loading mode)
//!
//! The dynamic-loading ABI is documented in `docs/abi.md` of the
//! candidate crate. In short: the candidate exposes a `extern "C" fn
//! fluxion_kernel_evaluate(input: *const u8, input_len: usize, output:
//! *mut u8, output_cap: usize) -> i32` symbol; see
//! [`crate::dynamic::DynamicLoadError::NotImplementedInThisBuild`] for
//! the current status of that mode.

use serde::{Deserialize, Serialize};

/// Stable identifier for a candidate within a campaign.
///
/// Convention (enforced by the OpenEvolve adapter, not the harness):
/// `"{seed-name}-{generation-index}"` (e.g. `"ctf-seed-0042"`). The
/// harness itself is identifier-agnostic — it preserves whatever
/// string the evolver passes in.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CandidateId(pub String);

impl std::fmt::Display for CandidateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<T: Into<String>> From<T> for CandidateId {
    fn from(value: T) -> Self {
        CandidateId(value.into())
    }
}

/// The fixed trait that every candidate implements.
///
/// Default-execution semantics: `evaluate` returns the candidate's
/// [`KernelOutput`] for the given [`KernelInput`] without performing
/// invariant or accuracy checks — those are the harness's job. The
/// `default()` impl is required so the recompiled binary can be
/// instantiated without a constructor signature.
pub trait Kernel: Default + Send + Sync {
    /// Run the candidate on one edge case.
    ///
    /// Contract:
    /// - **No I/O**, no allocations beyond what the kernel naturally
    ///   requires (the harness pins a memory cap — see
    ///   [`crate::sandbox`]).
    /// - **No panics** on bad input — return an `Err(KernelError::BadInput)`
    ///   instead. The harness reports the error in the summary rather
    ///   than tearing down the campaign on a single edge case.
    /// - **Deterministic** with respect to `input`: identical inputs
    ///   ⇒ identical outputs. No hidden state, no time-based RNG.
    /// - **IEEE 754** semantics. The harness relies on the
    ///   determinism-check job (issue #1297) to spot candidate code
    ///   that flips the sign of zero or relies on `+0.0 != -0.0`.
    fn evaluate(&self, input: &KernelInput) -> Result<KernelOutput, KernelError>;
}

/// Input for one edge case. The harness builds this from a TOML/JSON
/// config that the evolver passes; the candidate never sees the
/// reference values (see module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelInput {
    /// Human-readable name of the edge case (e.g. `"denver-jan"`,
    /// `"step-response-200mm"`). Carried into the summary's per-edge
    /// error map so evolvers can grep for which case regressed.
    pub case_name: String,

    /// Free-form numeric inputs to the kernel. The candidate reads the
    /// fields it needs and ignores the rest — different kernels have
    /// different schemas (CTF uses time-step vectors; solar uses
    /// sun-position tuples; BDF uses ODE state).
    ///
    /// Stable JSON encoding: `serde_json` with field order preserved,
    /// so two identical configs serialize to byte-identical JSON and
    /// contribute identically to the determinism digest.
    pub params: serde_json::Value,
}

/// Output from one edge case.
///
/// `payload` is the candidate's computed value for the given
/// `case_name`. Encoding is kernel-specific; the harness deserializes
/// it back to a `serde_json::Value` for the per-edge accuracy map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelOutput {
    /// Computed value. JSON-encodable so the harness can diff it
    /// against the reference without coupling to any kernel-specific
    /// Rust type.
    pub payload: serde_json::Value,
}

/// The reference output for one edge case — produced by the
/// known-good (pre-evolution) implementation, never seen by the
/// candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceOutput {
    /// Reference payload for the same `case_name` as the
    /// candidate's input.
    pub payload: serde_json::Value,
}

/// A single edge case in the harness battery.
///
/// The harness fixes the battery per campaign; the evolver never
/// modifies it (modifying the battery is the harness-side equivalent
/// of moving the goalposts).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeCase {
    /// Human-readable case name (e.g. `"denver-jan"`,
    /// `"step-response-200mm"`).
    pub name: String,

    /// Input handed to the candidate.
    pub input: KernelInput,

    /// Reference output produced by the known-good implementation.
    pub reference: ReferenceOutput,
}

/// Error returned by a candidate from [`Kernel::evaluate`].
///
/// Every variant becomes a `Summary.error` field rather than a panic;
/// the harness prefers partial-success summaries over campaign kills.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[allow(dead_code)]
pub enum KernelError {
    /// The candidate rejected the input as malformed (e.g. NaN
    /// parameter, dimension mismatch). The harness records this per
    /// edge case but continues with the rest.
    #[error("bad input: {0}")]
    BadInput(String),
    /// The candidate produced an output that the harness cannot
    /// decode (e.g. schema mismatch — the candidate was evolved for
    /// a different kernel version than the harness expects).
    #[error("output decode failure: {0}")]
    OutputDecode(String),
    /// The candidate hit an internal error that was neither bad
    /// input nor output decode (e.g. an unrecoverable numerical
    /// failure such as a singular matrix).
    #[error("internal kernel error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_input_round_trips_through_json() {
        let input = KernelInput {
            case_name: "demo".to_string(),
            params: serde_json::json!({"x": 1.0, "y": [1.0, 2.0, 3.0]}),
        };
        let json = serde_json::to_string(&input).unwrap();
        let parsed: KernelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, input);
    }

    #[test]
    fn candidate_id_display_is_inner_string() {
        let id: CandidateId = "ctf-seed-0042".into();
        assert_eq!(format!("{}", id), "ctf-seed-0042");
    }
}
